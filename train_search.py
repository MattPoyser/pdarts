import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import copy
from model_search import Network
from genotypes import PRIMITIVES
from genotypes import Genotype
import csv


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
parser.add_argument('--dataset', type=str, default="mnist", help='dataset to use')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=25, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='./tmp/checkpoints/', help='experiment path')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--tmp_data_dir', type=str, default='/data/cifar-10/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--dropout_rate', action='append', default=[], help='dropout rate of skip connect')
parser.add_argument('--add_width', action='append', default=['0'], help='add channels')
parser.add_argument('--add_layers', action='append', default=['0'], help='add layers')
parser.add_argument('--steps', type=int, default=4, help='no. nodes in cell')
parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')
parser.add_argument('--hardness', type=float, default=0.9, help='hardness parameter')
parser.add_argument('--mastery', type=float, default=0.1, help='mastery parameter')
parser.add_argument('--subset_size', type=int, default=100, help='subset parameter determining size of subdataset in dynamic loader')
parser.add_argument('--issave', type=bool, default=False, help='do we save indices for naswot')
parser.add_argument('--dynamic', type=bool, default=False, help='are we doing dynamic dataset')
parser.add_argument('--vanilla', type=bool, default=False, help='are we doing vanilla')
parser.add_argument('--isbad', type=bool, default=False, help='are we using bad autoencoder (ablation)')
parser.add_argument('--isTree', type=bool, default=False, help='do we use tree in dynamic dataloader (probably true)')
parser.add_argument('--init_train_epochs', type=int, default=5, help='minimum no. epochs to train before updating dynamic subset')
parser.add_argument('--is_csv', type=bool, default=False, help='saving with csv?')
parser.add_argument('--is_detection', type=bool, default=False, help='object detection?')
parser.add_argument('--ncc', type=bool, default=False, help='are we on ncc?')
parser.add_argument('--visualize', type=bool, default=False, help='are we visualizing results')

args = parser.parse_args()

# args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
# utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
# fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)

if args.cifar100:
    CIFAR_CLASSES = 100
    data_folder = 'cifar-100-python'
else:
    CIFAR_CLASSES = 10
    data_folder = 'cifar-10-batches-py'

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("args = %s", args.dataset)
    #  prepare dataset
    # if args.cifar100:
    #     train_transform, valid_transform = utils._data_transforms_cifar100(args)
    # else:
    #     train_transform, valid_transform = utils._data_transforms_cifar10(args)
    # if args.cifar100:
    #     train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=False, transform=train_transform)
    # else:
    #     train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=False, transform=train_transform)

    train_data, val_data = utils.get_data(args)
    # num_train = len(train_data)
    # indices = list(range(num_train))
    # split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.workers)
    
    # build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    switches = []
    switches_steps = sum([i for i in range(2, args.steps+2)])
    for i in range(switches_steps):
    # for i in range(14):
        switches.append([True for j in range(len(PRIMITIVES))])
    switches_normal = copy.deepcopy(switches)
    switches_reduce = copy.deepcopy(switches)

    # To be moved to args
    num_to_keep = [5, 3, 1]
    num_to_drop = [3, 2, 2]
    if len(args.add_width) == 3:
        add_width = args.add_width
    else:
        add_width = [0, 0, 0]
    if len(args.add_layers) == 3:
        add_layers = args.add_layers
    else:
        add_layers = [0, 6, 12]
    if len(args.dropout_rate) ==3:
        drop_rate = args.dropout_rate
    else:
        drop_rate = [0.0, 0.0, 0.0]
    eps_no_archs = [10, 10, 10]
    for sp in range(len(num_to_keep)):
        model = Network(args.init_channels + int(add_width[sp]), CIFAR_CLASSES, args.layers + int(add_layers[sp]), criterion, steps=args.steps, switches_normal=switches_normal, switches_reduce=switches_reduce, p=float(drop_rate[sp]), dataset=args.dataset)
        model = nn.DataParallel(model)
        model = model.cuda()
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        network_params = []
        for k, v in model.named_parameters():
            if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce')):
                network_params.append(v)       
        optimizer = torch.optim.SGD(
                network_params,
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
                    lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs), eta_min=args.learning_rate_min)
        sm_dim = -1
        epochs = args.epochs
        eps_no_arch = eps_no_archs[sp]
        scale_factor = 0.2
        hardness = None
        just_updated = True

        valid_acc = 0
        if args.issave:
            save_indices(train_queue.dataset.get_printable(), 0)
        for epoch in range(epochs):
            scheduler.step()

            epoch_type = get_epoch_type(epoch, hardness, valid_acc)
            if epoch_type or just_updated or not args.dynamic:  # 1 is train, as normal (0 is dataset update)
                just_updated = False

                lr = scheduler.get_lr()[0]
                logging.info('Epoch: %d lr: %e', epoch, lr)
                epoch_start = time.time()
                # training
                if epoch < eps_no_arch:
                    model.module.p = float(drop_rate[sp]) * (epochs - epoch - 1) / epochs
                    model.module.update_p()
                    train_acc, train_obj, _, _ = train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=False)
                else:
                    model.module.p = float(drop_rate[sp]) * np.exp(-(epoch - eps_no_arch) * scale_factor)
                    model.module.update_p()
                    train_acc, train_obj, hardness, correct = train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=True)
                    if args.dynamic:
                        train_queue.dataset.update_correct(correct)
                logging.info('Train_acc %f', train_acc)
                epoch_duration = time.time() - epoch_start
                logging.info('Epoch time: %ds', epoch_duration)
                # validation
                if epochs - epoch < 5:
                    valid_acc, valid_obj = infer(valid_queue, model, criterion)
                    logging.info('Valid_acc %f', valid_acc)

            else:
                print("updating subset")
                train_queue.dataset.update_subset(hardness, epoch)
                save_indices(train_queue.dataset.get_printable(), epoch, [train_queue.dataset.full_set.__getitem__(idx) for idx in train_queue.dataset.idx])
                just_updated = True
                if args.ncc and args.visualize:
                    train_queue.dataset.visualize(framework="pdarts")

        # utils.save(model, os.path.join(args.save, 'weights.pt'))
        print('------Dropping %d paths------' % num_to_drop[sp])
        # Save switches info for s-c refinement. 
        if sp == len(num_to_keep) - 1:
            switches_normal_2 = copy.deepcopy(switches_normal)
            switches_reduce_2 = copy.deepcopy(switches_reduce)
        # drop operations with low architecture weights
        arch_param = model.module.arch_parameters()
        normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()        
        # for i in range(14):
        for i in range(switches_steps):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_normal[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                # for the last stage, drop all Zero operations
                drop = get_min_k_no_zero(normal_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(normal_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_normal[i][idxs[idx]] = False
        reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()

        # for i in range(14):
        for i in range(switches_steps):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_reduce[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                drop = get_min_k_no_zero(reduce_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(reduce_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_reduce[i][idxs[idx]] = False
        logging.info('switches_normal = %s', switches_normal)
        logging_switches(switches_normal)
        logging.info('switches_reduce = %s', switches_reduce)
        logging_switches(switches_reduce)
        
        if sp == len(num_to_keep) - 1:
            arch_param = model.module.arch_parameters()
            normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
            reduce_prob = F.softmax(arch_param[1], dim=sm_dim).data.cpu().numpy()
            # normal_final = [0 for idx in range(14)]
            normal_final = [0 for idx in range(switches_steps)]
            # reduce_final = [0 for idx in range(14)]
            reduce_final = [0 for idx in range(switches_steps)]
            # remove all Zero operations
            # for i in range(14):
            for i in range(switches_steps):
                if switches_normal_2[i][0] == True:
                    normal_prob[i][0] = 0
                normal_final[i] = max(normal_prob[i])
                if switches_reduce_2[i][0] == True:
                    reduce_prob[i][0] = 0
                reduce_final[i] = max(reduce_prob[i])                
            # Generate Architecture, similar to DARTS
            keep_normal = [0, 1]
            keep_reduce = [0, 1]
            n = 3
            start = 2
            # for i in range(3): # (original) 3 gives keep_normal/reduce length of 8 (i.e. steps=4 *2 )
            for i in range(args.steps-1): # 8 gives keep_normal/reduce length of 16 (i.e. steps=8 * 2 ). general rule: steps-1
                end = start + n
                tbsn = normal_final[start:end]
                tbsr = reduce_final[start:end]
                edge_n = sorted(range(n), key=lambda x: tbsn[x])
                keep_normal.append(edge_n[-1] + start)
                keep_normal.append(edge_n[-2] + start)
                edge_r = sorted(range(n), key=lambda x: tbsr[x])
                keep_reduce.append(edge_r[-1] + start)
                keep_reduce.append(edge_r[-2] + start)
                start = end
                n = n + 1

            # set switches according the ranking of arch parameters
            # for i in range(14):
            for i in range(switches_steps):
                if not i in keep_normal:
                    for j in range(len(PRIMITIVES)):
                        switches_normal[i][j] = False
                if not i in keep_reduce:
                    for j in range(len(PRIMITIVES)):
                        switches_reduce[i][j] = False
            # translate switches into genotype
            genotype = parse_network(switches_normal, switches_reduce)
            logging.info(genotype)
            ## restrict skipconnect (normal cell only)
            logging.info('Restricting skipconnect...')

            top_max_sk = switches_steps
            # generating genotypes with different numbers of skip-connect operations
            for sks in range(0, top_max_sk+1):
                max_sk = top_max_sk - sks
                num_sk = check_sk_number(switches_normal)
                if not num_sk > max_sk:
                    continue
                while num_sk > max_sk:
                    normal_prob = delete_min_sk_prob(switches_normal, switches_normal_2, normal_prob)
                    switches_normal = keep_1_on(switches_normal_2, normal_prob)
                    switches_normal = keep_2_branches(switches_normal, normal_prob)
                    num_sk = check_sk_number(switches_normal)
                logging.info('Number of skip-connect: %d', max_sk)
                genotype = parse_network(switches_normal, switches_reduce)
                logging.info(genotype)

def train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=True):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    hardness = [None for i in range(len(train_queue))]
    correct = [None for i in range(len(train_queue))]
    batch_size = args.batch_size

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        if train_arch:
            # In the original implementation of DARTS, it is input_search, target_search = next(iter(valid_queue), which slows down
            # the training when using PyTorch 0.4 and above. 
            try:
                input_search, target_search = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(valid_queue)
                input_search, target_search = next(valid_queue_iter)
            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)
            optimizer_a.zero_grad()
            logits = model(input_search)
            loss_a = criterion(logits, target_search)

            # raise AttributeError(logits.shape, target.shape, loss_a.shape)
            new_hardness, new_correct = get_hardness(logits.cpu(), target.cpu(), False)
            hardness[(step * batch_size):(step * batch_size) + batch_size] = new_hardness  # assumes batch 1 takes idx 0-8, batch 2 takes 9-16, etc.
            correct[(step * batch_size):(step * batch_size) + batch_size] = new_correct

            loss_a.backward()
            nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            optimizer_a.step()

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(network_params, args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)
    # if train_arch:
    #     raise AttributeError(len(train_queue), batch_size, len(correct), len(new_correct), len(new_hardness))

    return top1.avg, objs.avg, hardness, correct


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def parse_network(switches_normal, switches_reduce):

    def _parse_switches(switches):
        n = 2
        start = 0
        gene = []
        step = args.steps
        for i in range(step):
            end = start + n
            for j in range(start, end):
                for k in range(len(switches[j])):
                    if switches[j][k]:
                        gene.append((PRIMITIVES[k], j - start))
            start = end
            n = n + 1
        return gene
    gene_normal = _parse_switches(switches_normal)
    gene_reduce = _parse_switches(switches_reduce)
    
    concat = range(2, 6)
    
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat, 
        reduce=gene_reduce, reduce_concat=concat
    )
    
    return genotype

def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmin(input)
        index.append(idx)
        input[idx] = 1
    
    return index
def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    if 0 in idxs:
        zf = True 
    else:
        zf = False
    if zf:
        w = w[1:]
        index.append(0)
        k = k - 1
    for i in range(k):
        idx = np.argmin(w)
        w[idx] = 1
        if zf:
            idx = idx + 1
        index.append(idx)
    return index
        
def logging_switches(switches):
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES[j])
        logging.info(ops)
        
def check_sk_number(switches):
    count = 0
    for i in range(len(switches)):
        if switches[i][3]:
            count = count + 1
    
    return count

def delete_min_sk_prob(switches_in, switches_bk, probs_in):
    def _get_sk_idx(switches_in, switches_bk, k):
        if not switches_in[k][3]:
            idx = -1
        else:
            idx = 0
            for i in range(3):
                if switches_bk[k][i]:
                    idx = idx + 1
        return idx
    probs_out = copy.deepcopy(probs_in)
    sk_prob = [1.0 for i in range(len(switches_bk))]
    for i in range(len(switches_in)):
        idx = _get_sk_idx(switches_in, switches_bk, i)
        if not idx == -1:
            sk_prob[i] = probs_out[i][idx]
    d_idx = np.argmin(sk_prob)
    idx = _get_sk_idx(switches_in, switches_bk, d_idx)
    probs_out[d_idx][idx] = 0.0
    
    return probs_out

def keep_1_on(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    for i in range(len(switches)):
        idxs = []
        for j in range(len(PRIMITIVES)):
            if switches[i][j]:
                idxs.append(j)
        drop = get_min_k_no_zero(probs[i, :], idxs, 2)
        for idx in drop:
            switches[i][idxs[idx]] = False            
    return switches

def keep_2_branches(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    final_prob = [0.0 for i in range(len(switches))]
    for i in range(len(switches)):
        final_prob[i] = max(probs[i])
    keep = [0, 1]
    n = 3
    start = 2
    for i in range(args.steps - 1):
        end = start + n
        tb = final_prob[start:end]
        edge = sorted(range(n), key=lambda x: tb[x])
        keep.append(edge[-1] + start)
        keep.append(edge[-2] + start)
        start = end
        n = n + 1
    for i in range(len(switches)):
        if not i in keep:
            for j in range(len(PRIMITIVES)):
                switches[i][j] = False  
    return switches  



################################# dynamic functions #####################################
def save_indices(data, epoch, images=None):
    if args.issave:
        if args.ncc:
            with open(f'/home2/lgfm95/nas/pdarts/tempSave/curriculums/{args.dataset}/indices_{args.dataset}_{epoch}.csv', 'w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=' ')
                csv_writer.writerow(data)
            if images is not None:
                image_dir = f'/home2/lgfm95/nas/pdarts/tempSave/curriculums/{args.dataset}/indices_{args.dataset}_{epoch}'
                os.makedirs(image_dir)
                for q, image in enumerate(images):
                    image.save(image_dir + f"{q}.png")

        else:
            with open(f'/hdd/PhD/nas/pdarts/tempSave/curriculums/{args.dataset}/indices_{args.dataset}_{epoch}.csv', 'w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=' ')
                csv_writer.writerow(data)


# low value for hardness means harder.
def get_hardness(output, target, is_multi):
    if not is_multi:
        # currently a binary association between correct classication => 0.8
        # we want it to be a softmax representation. if we instead take crossentropy loss of each individual cf target
        _, predicted = torch.max(output.data, 1)
        confidence = F.softmax(output, dim=1)
        try:
            hardness_scaler = np.where((predicted == target), 1, 0.1) # if correct, simply use confidence as measure of hardness
        except RuntimeError:
            raise AttributeError(output.shape, target.shape, predicted.shape, confidence.shape)
        # therefore if model can easily say yep this is object X, then confidence will be high. if it only just manages to identify
        # object X, confidence if lower
        # if object X is misclassified, hardness needs to be lower still.
        # assumes that it does not confidently misclassify.
        # raise AttributeError(output.shape, target.shape, predicted.shape, confidence.shape, hardness_scaler)
        hardness = [(confidence[i][predicted[i]] * hardness_scaler[i]).item() for i in range(output.size(0))]
    else:
        output = torch.sigmoid(output.float()).detach()
        output[output>0.5] = 1
        output[output<=0.5] = 0
        confidence = F.softmax(output, dim=1)

        hardness_scaler = []
        hardness = []
        assert len(output) == len(target) # should both be equal to batch size
        for q in range(len(output)):
            assert len(output[q]) == len(target[q]) # should both be equal to num_classes eg 184
            correct_avg = (np.array(output[q]) == np.array(target[q])).sum() / len(output[q])
            if correct_avg > 0.5: # this could be another threshold we change, or have it == hardness threshold
                hardness_scaler.append(1)
            else:
                hardness_scaler.append(0.1)

            correct = np.where(np.array(output[q]) == np.array(target[q]))[0]
            hardness_value = [confidence[q][i] * 1 if i in correct else confidence[q][i] * 0.1 for i in range(len(output[q]))]
            hardness.append(sum(hardness_value) / len(output[q]))


        hardness_scaler = np.asarray(hardness_scaler)
        hardness = np.array(hardness)
        # raise AttributeError(output, target, hardness_scaler, hardness)

    return hardness, hardness_scaler


def get_epoch_type(epoch, hardness, top1):
    # naive alternate, starting with normal training
    if not args.dynamic or epoch < args.init_train_epochs:
        return 1
    is_mastered = get_mastered(hardness, top1)
    if is_mastered:
        print("mastered, therefore epoch type 0")
        return 0
    print("not mastered, therefore epoch type 1")
    return 1


def get_mastered(hardness, top1):
    # if fraction of times where image is unconfidently/mis-classified is less than mastery threshold
    # TODO use hardness across history eg mean hardness over last 5
    # print("ahard", "\n")
    # for aHard in hardness:
        # print("ahard", aHard)
    # print("len hardness", len(hardness))
    # print("len hard ones", np.where(np.array(hardness) > 0.5))
    # print("len hard ones", len(np.where(np.array(hardness) > 0.5)[0]))
    # print("hardness calculations: ", (len(np.where(np.array(hardness) > g_config.hardness)[0]) / len(hardness)), g_config.mastery)

    #if percentage of items considered hard exceeds a mastery threshold, update the subset.
    if top1 is None:
        if (len(np.where(np.array(hardness) > args.hardness)[0]) / len(hardness)) < args.mastery:
            print("therefore not mastered")
            return 0
    else:
        # print("grep working", top1)
        if top1 < args.mastery:
            return 0
    # if len(np.where(np.array(hardness) < g_config.mastery)) > len(hardness)-2:
        # a lot of images still being misclassified
        # return 0
    print("therefore mastered")
    return 1


if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)
