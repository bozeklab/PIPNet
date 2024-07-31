import numpy as np
import argparse
import torch
import os
import torch.optim
import torch.utils.data
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict
import torchvision.transforms.functional as F
from torch import Tensor
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import random
from sklearn.model_selection import train_test_split


def get_data(args: argparse.Namespace):
    """
    Load the proper dataset based on the parsed arguments
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.dataset == 'CUB-200-2011':
        return get_birds(True, './data/CUB_200_2011/dataset/train_crop', './data/CUB_200_2011/dataset/train',
                         './data/CUB_200_2011/dataset/test_crop', args.image_size, args.seed, args.validation_size,
                         './data/CUB_200_2011/dataset/train', './data/CUB_200_2011/dataset/test_full')
    if args.dataset == 'pets':
        return get_pets(True, './data/PETS/dataset/train', './data/PETS/dataset/train', './data/PETS/dataset/test',
                        args.image_size, args.seed, args.validation_size)
    if args.dataset == 'partimagenet':  # use --validation_size of 0.2
        return get_partimagenet(True, './data/partimagenet/dataset/all', './data/partimagenet/dataset/all', None,
                                args.image_size, args.seed, args.validation_size)
    if args.dataset == 'CARS':
        return get_cars(True, './data/cars/dataset/train', './data/cars/dataset/train', './data/cars/dataset/test',
                        args.image_size, args.seed, args.validation_size)
    if args.dataset == 'grayscale_example':
        return get_grayscale(True, './data/train', './data/train', './data/test', args.image_size, args.seed,
                             args.validation_size)
    if args.dataset == 'grayscale_mito':
        return get_grayscale(True, '/data/pwojcik/mito_work/dataset/train',
                             '/data/pwojcik/mito_work/dataset/train',
                             '/data/pwojcik/mito_work/dataset/test', args.image_size, args.image_size_ds,
                             args.seed,
                             args.validation_size)
    raise Exception(f'Could not load data set, data set "{args.dataset}" not found!')


def get_dataloaders(args: argparse.Namespace, device):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, targets = get_data(
        args)

    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    to_shuffle = True
    sampler = None

    num_workers = args.num_workers

    if args.weighted_loss:
        if targets is None:
            raise ValueError("Weighted loss not implemented for this dataset. Targets should be restructured")
        # https://discuss.pytorch.org/t/dataloader-using-subsetrandomsampler-and-weightedrandomsampler-at-the-same-time/29907
        class_sample_count = torch.tensor(
            [(targets[train_indices] == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.float()
        print("Weights for weighted sampler: ", weight, flush=True)
        samples_weight = torch.tensor([weight[t] for t in targets[train_indices]])
        # Create sampler, dataset, loader
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        to_shuffle = False

    pretrain_batchsize = args.batch_size_pretrain

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=to_shuffle,
                                              sampler=sampler,
                                              pin_memory=cuda,
                                              num_workers=num_workers,
                                              worker_init_fn=np.random.seed(args.seed),
                                              drop_last=True
                                              )
    if trainset_pretraining is not None:
        trainloader_pretraining = torch.utils.data.DataLoader(trainset_pretraining,
                                                              batch_size=pretrain_batchsize,
                                                              shuffle=to_shuffle,
                                                              sampler=sampler,
                                                              pin_memory=cuda,
                                                              num_workers=num_workers,
                                                              worker_init_fn=np.random.seed(args.seed),
                                                              drop_last=True
                                                              )

    else:
        trainloader_pretraining = torch.utils.data.DataLoader(trainset,
                                                              batch_size=pretrain_batchsize,
                                                              shuffle=to_shuffle,
                                                              sampler=sampler,
                                                              pin_memory=cuda,
                                                              num_workers=num_workers,
                                                              worker_init_fn=np.random.seed(args.seed),
                                                              drop_last=True
                                                              )

    trainloader_normal = torch.utils.data.DataLoader(trainset_normal,
                                                     batch_size=args.batch_size,
                                                     shuffle=to_shuffle,
                                                     sampler=sampler,
                                                     pin_memory=cuda,
                                                     num_workers=num_workers,
                                                     worker_init_fn=np.random.seed(args.seed),
                                                     drop_last=True
                                                     )
    trainloader_normal_augment = torch.utils.data.DataLoader(trainset_normal_augment,
                                                             batch_size=args.batch_size,
                                                             shuffle=to_shuffle,
                                                             sampler=sampler,
                                                             pin_memory=cuda,
                                                             num_workers=num_workers,
                                                             worker_init_fn=np.random.seed(args.seed),
                                                             drop_last=True
                                                             )

    projectloader = torch.utils.data.DataLoader(projectset,
                                                batch_size=1,
                                                shuffle=False,
                                                pin_memory=cuda,
                                                num_workers=num_workers,
                                                worker_init_fn=np.random.seed(args.seed),
                                                drop_last=False
                                                )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             pin_memory=cuda,
                                             num_workers=num_workers,
                                             worker_init_fn=np.random.seed(args.seed),
                                             drop_last=False
                                             )
    test_projectloader = torch.utils.data.DataLoader(testset_projection,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     pin_memory=cuda,
                                                     num_workers=num_workers,
                                                     worker_init_fn=np.random.seed(args.seed),
                                                     drop_last=False
                                                     )
    print("Num classes (k) = ", len(classes), classes[:5], "etc.", flush=True)
    return trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes


def create_datasets(transform1, transform2, transform1_ds, transform2_ds, transform_no_augment, transform_no_augment_ds,
                    num_channels: int, train_dir: str, project_dir: str, test_dir: str, seed: int,
                    validation_size: float, train_dir_pretrain=None, test_dir_projection=None, transform1p=None):
    trainvalset = ImageFolder(train_dir, is_valid_file=is_valid_file)
    classes = trainvalset.classes
    targets = trainvalset.targets
    indices = list(range(len(trainvalset)))

    train_indices = indices

    if test_dir is None:
        if validation_size <= 0.:
            raise ValueError(
                "There is no test set directory, so validation size should be > 0 such that training set can be split.")
        subset_targets = list(np.array(targets)[train_indices])
        train_indices, test_indices = train_test_split(train_indices, test_size=validation_size,
                                                       stratify=subset_targets, random_state=seed)
        testset = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transform_no_augment),
                                          indices=test_indices)
        print("Samples in trainset:", len(indices), "of which", len(train_indices), "for training and ",
              len(test_indices), "for testing.", flush=True)
    else:
        testset = DualTransformImageFolder(test_dir, transform1=transform_no_augment, transform2=transform_no_augment_ds)

    trainset = torch.utils.data.Subset(FourAugSupervisedDataset(trainvalset, transform1=transform1,
                                                                transform2=transform2,
                                                                transform3=transform1_ds,
                                                                transform4=transform2_ds), indices=train_indices)
    trainset_normal = torch.utils.data.Subset(
        DualTransformImageFolder(train_dir, transform1=transform_no_augment, transform2=transform_no_augment_ds), indices=train_indices)
    trainset_normal_augment = torch.utils.data.Subset(
        DualTransformImageFolder(train_dir, transform1=transforms.Compose([transform1, transform2]),
                                 transform2=transforms.Compose([transform1_ds, transform2_ds])),
        indices=train_indices)
    projectset = DualTransformImageFolder(project_dir, transform1=transform_no_augment, transform2=transform_no_augment_ds)

    if test_dir_projection is not None:
        testset_projection = DualTransformImageFolder(test_dir_projection, transform1=transform_no_augment, transform2=transform_no_augment_ds)
    else:
        testset_projection = testset
    if train_dir_pretrain is not None:
        trainvalset_pr = torchvision.datasets.ImageFolder(train_dir_pretrain)
        targets_pr = trainvalset_pr.targets
        indices_pr = list(range(len(trainvalset_pr)))
        train_indices_pr = indices_pr
        if test_dir is None:
            subset_targets_pr = list(np.array(targets_pr)[indices_pr])
            train_indices_pr, test_indices_pr = train_test_split(indices_pr, test_size=validation_size,
                                                                 stratify=subset_targets_pr, random_state=seed)

        trainset_pretraining = torch.utils.data.Subset(FourAugSupervisedDataset(trainvalset_pr,
                                                                                transform1=transform1,
                                                                                transform2=transform2,
                                                                                transform3=transform1_ds,
                                                                                transform4=transform2_ds),
                                                       indices=train_indices_pr)
    else:
        trainset_pretraining = None

    return trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, torch.LongTensor(
        targets)


def get_pets(augment: bool, train_dir: str, project_dir: str, test_dir: str, img_size: int, seed: int,
             validation_size: float):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size + 48, img_size + 48)),
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size + 8, scale=(0.95, 1.))
        ])

        transform2 = transforms.Compose([
            TrivialAugmentWideNoShape(),
            transforms.RandomCrop(size=(img_size, img_size)),  # includes crop
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform1 = transform_no_augment
        transform2 = transform_no_augment

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed,
                           validation_size)


def get_partimagenet(augment: bool, train_dir: str, project_dir: str, test_dir: str, img_size: int, seed: int,
                     validation_size: float):
    # Validation size was set to 0.2, such that 80% of the data is used for training
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size + 48, img_size + 48)),
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size + 8, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
            TrivialAugmentWideNoShape(),
            transforms.RandomCrop(size=(img_size, img_size)),  # includes crop
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform1 = transform_no_augment
        transform2 = transform_no_augment

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed,
                           validation_size)


def get_birds(augment: bool, train_dir: str, project_dir: str, test_dir: str, img_size: int, seed: int,
              validation_size: float, train_dir_pretrain=None, test_dir_projection=None):
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    transform1p = None
    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size + 8, img_size + 8)),
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size + 4, scale=(0.95, 1.))
        ])
        transform1p = transforms.Compose([
            transforms.Resize(size=(img_size + 32, img_size + 32)),
            # for pretraining, crop can be bigger since it doesn't matter when bird is not fully visible
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size + 4, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
            TrivialAugmentWideNoShape(),
            transforms.RandomCrop(size=(img_size, img_size)),  # includes crop
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform1 = transform_no_augment
        transform2 = transform_no_augment

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed,
                           validation_size, train_dir_pretrain, test_dir_projection, transform1p)


def get_cars(augment: bool, train_dir: str, project_dir: str, test_dir: str, img_size: int, seed: int,
             validation_size: float):
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size + 32, img_size + 32)),
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size + 4, scale=(0.95, 1.))
        ])

        transform2 = transforms.Compose([
            TrivialAugmentWideNoShapeWithColor(),
            transforms.RandomCrop(size=(img_size, img_size)),  # includes crop
            transforms.ToTensor(),
            normalize
        ])

    else:
        transform1 = transform_no_augment
        transform2 = transform_no_augment

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed,
                           validation_size)


def get_grayscale(augment: bool, train_dir: str, project_dir: str, test_dir: str, img_size: int, img_size_ds: int,
                  seed: int, validation_size: float, train_dir_pretrain=None):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.Grayscale(3),  # convert to grayscale with three channels
        transforms.ToTensor(),
        normalize
    ])

    transform_no_augment_ds = transforms.Compose([
        transforms.Resize(size=(img_size_ds, img_size_ds)),
        transforms.Grayscale(3),  # convert to grayscale with three channels
        transforms.ToTensor(),
        normalize
    ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size + 64, img_size + 64)),
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size + 16, scale=(0.95, 1.))
        ])
        transform1_ds = transforms.Compose([
            transforms.Resize(size=(img_size_ds + 64, img_size_ds + 64)),
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size_ds + 16, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
            TrivialAugmentWideNoShape(),
            transforms.RandomCrop(size=(img_size, img_size)),  # includes crop
            transforms.Grayscale(3),  # convert to grayscale with three channels
            transforms.ToTensor(),
            normalize
        ])
        transform2_ds = transforms.Compose([
            TrivialAugmentWideNoShape(),
            transforms.RandomCrop(size=(img_size_ds, img_size_ds)),  # includes crop
            transforms.Grayscale(3),  # convert to grayscale with three channels
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform1 = transform_no_augment
        transform2 = transform_no_augment

        transform1_ds = transform_no_augment_ds
        transform2_ds = transform_no_augment_ds

    return create_datasets(transform1, transform2, transform1_ds, transform2_ds,
                           transform_no_augment, transform_no_augment_ds, 3,
                           train_dir, project_dir, test_dir, seed, validation_size)


def is_valid_file(path: str) -> bool:
    return path.lower().endswith(IMG_EXTENSIONS) and 'mask' not in os.path.basename(path).lower()


def create_boolean_mask(mask_img):
    mask_img = F.rgb_to_grayscale(mask_img)
    m_shape = mask_img.shape
    if len(m_shape) == 3:
        mask_img=mask_img.squeeze()
    else:
        mask_img=mask_img[:, 0, :, :]
    mask = mask_img.numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = (mask * 255).astype(np.uint8)
    max_val = np.max(mask)
    min_val = np.min(mask)
    max_mask = (mask == max_val)
    min_mask = (mask == min_val)
    bool_mask = (max_mask | ~min_mask)
    return torch.tensor(bool_mask)


class DualTransformImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform1, transform2,  loader=Image.open,
                 is_valid_file=None):
        super(DualTransformImageFolder, self).__init__(root, transform1, transform2, loader, is_valid_file)
        self.transform1 = transform1
        self.transform2 = transform2
        self.imgs = [img for img in self.imgs if "mask" not in img[0]]

    def __getitem__(self, index):
        path, target = self.imgs[index]

        mask_image_path = os.path.join(os.path.dirname(path), 'mask_' + os.path.basename(path))
        mask = Image.open(mask_image_path).convert('RGB')

        mask_array = np.array(mask)
        mask_array[mask_array != 0] = 255
        mask = Image.fromarray(mask_array)

        sample = self.loader(path)
        st1 = torch.get_rng_state()
        sample1 = self.transform1(sample)
        torch.set_rng_state(st1)
        mask1 = self.transform1(mask)
        st2 = torch.get_rng_state()
        sample2 = self.transform2(sample)
        torch.set_rng_state(st2)
        mask2 = self.transform2(mask)

        masks = []
        for mask in [mask1, mask2]:
            bool_mask = create_boolean_mask(mask)
            masks.append(bool_mask)


        return sample1, sample2, masks[0], masks[1], target

    def __len__(self):
        return len(self.imgs)


class FourAugSupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""

    def __init__(self, dataset, transform1, transform2, transform3, transform4):
        self.dataset = dataset
        self.classes = dataset.classes
        self.flip = RandomHorizontalFlip()

        # Filter image files to exclude those with 'mask' in their names

        if type(dataset) == torchvision.datasets.folder.ImageFolder:
            self.imgs = dataset.imgs
            self.targets = dataset.targets
        else:
            self.targets = dataset._labels
            self.imgs = list(zip(dataset._image_files, dataset._labels))
        self.imgs = [img for img in dataset.samples if "mask" not in img[0]]
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.transform4 = transform4

    def __getitem__(self, index):
        image_path, target = self.imgs[index]
        image = Image.open(image_path).convert('RGB')
        mask_image_path = os.path.join(os.path.dirname(image_path), 'mask_' + os.path.basename(image_path))
        mask = Image.open(mask_image_path).convert('RGB')

        mask_array = np.array(mask)
        mask_array[mask_array != 0] = 255
        mask = Image.fromarray(mask_array)

        image_ = image.copy()
        mask_ = mask.copy()
        st1 = torch.get_rng_state()
        image = self.transform1(image)
        im1 = self.transform2(image)
        im2 = self.transform2(image)
        im2, hflip1 = self.flip(im2)

        torch.set_rng_state(st1)
        mask = self.transform1(mask)
        m1 = self.transform2(mask)
        m2 = self.transform2(mask)
        m2, _ = self.flip(m2)

        st2 = torch.get_rng_state()
        image_ds = self.transform3(image_)
        im1_ds = self.transform4(image_ds)
        im2_ds = self.transform4(image_ds)
        im2_ds, hflip2 = self.flip(im2_ds)

        torch.set_rng_state(st2)
        mask_ds = self.transform3(mask_)
        m1_ds = self.transform4(mask_ds)
        m2_ds = self.transform4(mask_ds)
        m2_ds, _ = self.flip(m2_ds)

        masks = []
        for mask in [m2, m2_ds]:
            bool_mask = create_boolean_mask(mask)
            masks.append(bool_mask)

        return im1, im2, masks[0], im1_ds, im2_ds, masks[1], hflip1, hflip2, target

    def __len__(self):
        return len(self.imgs)


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, img):
        if torch.rand(1) < 0.3:
            return F.hflip(img), True
        return img, False

# function copied from https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide (v0.12) and adapted
class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True),
        }


class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.5, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }


class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {

            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

