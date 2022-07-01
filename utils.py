import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dset
import sys
sys.path.insert(0, "/home/matt/Documents/hem/perceptual")
sys.path.insert(0, "/home2/lgfm95/hem/perceptual")
sys.path.insert(0, "C:\\Users\\Matt\\Documents\\PhD\\x11\\HEM\\perceptual")
sys.path.insert(0, "/hdd/PhD/hem/perceptual")
from dataloader_classification import DynamicDataset
from subloader import SubDataset


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR_STD = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def data_transform_general(name):
    if name == 'imagenet':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif name == 'mnist':
        mean = [0.13066051707548254]
        std = [0.30810780244715075]
    elif name == 'fashion':
        mean = [0.28604063146254594]
        std = [0.35302426207299326]
    elif name == "cifar10":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    else:
        raise TypeError("Unknown dataset : {:}".format(name))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, valid_transform


def get_data(args):
    train_transform, test_transform = data_transform_general(args.dataset)
    pretrain_resume = "/home2/lgfm95/hem/perceptual/good.pth.tar"
    grayscale = False
    is_detection = False
    convert_to_paths = False
    convert_to_lbl_paths = False
    isize = 64
    nz = 8
    aisize = 3
    is_concat=False
    if args.dataset == "mnist":
        dset_cls = dset.MNIST
        dynamic_name = "mnist"
        grayscale = True
        auto_resume = "/home2/lgfm95/hem/perceptual/ganPercMnistGood.pth.tar"
        aisize = 1
    elif args.dataset == "fashion":
        dset_cls = dset.FashionMNIST
        dynamic_name = "fashion"
        grayscale = True
        auto_resume = "/home2/lgfm95/hem/perceptual/ganPercFashionGood.pth.tar"
        aisize = 1
    elif args.dataset == "cifar10":
        dset_cls = dset.CIFAR10
        dynamic_name = "cifar10"
        auto_resume = "/home2/lgfm95/hem/perceptual/tripletCifar10MseKGood.pth.tar"
        nz = 32
    elif args.dataset == "imagenet":
        dynamic_name = "imagenet"
        isize = 256
        auto_resume = "/home2/lgfm95/hem/perceptual/ganPercImagenetGood.pth.tar"
        convert_to_paths = True
    else:
        raise TypeError("Unknown dataset : {:}".format(args.name))

    if args.isbad:
        auto_resume = "badpath"

    normalize = transforms.Normalize(
        mean=[0.13066051707548254],
        std=[0.30810780244715075])
    perc_transforms = transforms.Compose([
        transforms.RandomResizedCrop(isize),
        transforms.ToTensor(),
        normalize,
    ])
    if args.dynamic:
        # print(perc_transforms)
        train_data = DynamicDataset(
            perc_transforms=perc_transforms,
            pretrain_resume=pretrain_resume,
            image_transforms=train_transform,
            val_transforms=test_transform,
            val=False,
            dataset_name=dynamic_name,
            auto_resume=auto_resume,
            hardness=args.hardness,
            isize=isize,
            nz=nz,
            aisize=aisize,
            grayscale=grayscale,
            isTsne=True,
            tree=args.isTree,
            subset_size=args.subset_size,
            is_csv=args.is_csv,
            is_detection=is_detection,
            is_concat=is_concat,
            seed=args.seed)
            # is_csv=False)
        # is_csv=False)
        if args.dataset == "imagenet":
            test_data = SubDataset(transforms=test_transform, val=True, dataset_name=dynamic_name,
                                   subset_size=10000)
        else:
            test_data = dset_cls(root=args.tmp_data_dir, train=False, download=False, transform=test_transform)
    else:
        if args.vanilla:
            if args.dataset == "imagenet":
                subset_size = 10000
                dynamic_name = "imagenet"
                train_data = SubDataset(transforms=train_transform, val_transforms=test_transform, val=False,
                                        dataset_name=dynamic_name, subset_size=subset_size)
                test_data = SubDataset(transforms=test_transform, val=True, dataset_name=dynamic_name, subset_size=subset_size)
            else:
                train_data = dset_cls(root=args.tmp_data_dir, train=True, download=False, transform=train_transform)
                test_data = dset_cls(root=args.tmp_data_dir, train=False, download=False, transform=test_transform)
        else: #abl
            if args.dataset == "imagenet":
                train_data = SubDataset(transforms=train_transform, val_transforms=test_transform, val=False,
                                        dataset_name=dynamic_name, subset_size=args.subset_size)
                test_data = SubDataset(transforms=test_transform, val=True, dataset_name=dynamic_name, subset_size=args.subset_size)
            else:
                subset_size = args.subset_size
                train_data = SubDataset(transforms=train_transform, val_transforms=test_transform, val=False, dataset_name=dynamic_name, subset_size=subset_size)
                test_data = SubDataset(transforms=test_transform, val=True, dataset_name=dynamic_name, subset_size=subset_size)

    return train_data, test_data


def get_validation_data(dataset, dir):
    if dataset == "mnist":
        dset_cls = dset.MNIST
        mean = [0.13066051707548254]
        std = [0.30810780244715075]
    elif dataset == "fashion":
        dset_cls = dset.FashionMNIST
        mean = [0.28604063146254594]
        std = [0.35302426207299326]
    elif dataset == "cifar10":
        dset_cls = dset.CIFAR10
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif dataset == "imagenet":
        subset_size = 10000
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return SubDataset(transforms=train_transform, val_transforms=valid_transform, val=False,
                                        dataset_name="imagenet", subset_size=subset_size), SubDataset(transforms=valid_transform, val=True, dataset_name="imagenet",
                               subset_size=subset_size)
    else:
        raise TypeError("Unknown dataset : {:}".format(dataset))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return dset_cls(root=dir, train=True, download=False, transform=train_transform), dset_cls(root=dir, train=False, download=False, transform=valid_transform)


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
