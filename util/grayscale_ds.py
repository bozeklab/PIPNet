import numpy as np
import argparse
import torch
import os
import torch.optim
import torch.utils.data
import torchvision
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from typing import Tuple, Dict
from torch import Tensor
import random
from sklearn.model_selection import train_test_split

from data import TrivialAugmentWideNoColor, TrivialAugmentWideNoShape, create_datasets


def get_grayscale(augment:bool, train_dir: str, project_dir: str, test_dir:str, img_size: int, img_size_ds: int,
                  seed:int, validation_size: float, train_dir_pretrain = None):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                           transforms.Resize(size=(img_size, img_size)),
                           transforms.Grayscale(3), #convert to grayscale with three channels
                           transforms.ToTensor(),
                           normalize
                          ])

    transform_no_augment_ds = transforms.Compose([
                              transforms.Resize(size=(img_size_ds, img_size_ds)),
                              transforms.Grayscale(3), #convert to grayscale with three channels
                              transforms.ToTensor(),
                              normalize
                            ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+64, img_size+64)),
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+16, scale=(0.95, 1.))
        ])
        transform1_ds = transforms.Compose([
                        transforms.Resize(size=(img_size_ds+64, img_size_ds+64)),
                        TrivialAugmentWideNoColor(),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomResizedCrop(img_size_ds+16, scale=(0.95, 1.))
                        ])
        transform2 = transforms.Compose([
                            TrivialAugmentWideNoShape(),
                            transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                            transforms.Grayscale(3),#convert to grayscale with three channels
                            transforms.ToTensor(),
                            normalize
                            ])
        transform2_ds = transforms.Compose([
                            TrivialAugmentWideNoShape(),
                            transforms.RandomCrop(size=(img_size_ds, img_size_ds)), #includes crop
                            transforms.Grayscale(3),#convert to grayscale with three channels
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
                           train_dir, project_dir, test_dir, seed, validation_size, train_dir_pretrain=train_dir_pretrain)


def display_tensor_images(xs1, xs2, m2, xs1_ds, xs2_ds, m2_ds,
                          h_coor_min, h_coor_max, w_coor_min, w_coor_max):
    # Ensure the tensors are on the CPU and convert them to NumPy arrays
    xs1 = xs1.cpu().numpy()
    xs2 = xs2.cpu().numpy()
    m2 = m2.cpu().numpy()
    m2_ds = m2_ds.cpu().numpy()
    xs1_ds = xs1_ds.cpu().numpy()
    xs2_ds = xs2_ds.cpu().numpy()

    # Transpose the arrays from [C, H, W] to [H, W, C]
    image1 = np.transpose(xs1, (1, 2, 0))
    image2 = np.transpose(xs2, (1, 2, 0))
    image1_ds = np.transpose(xs1_ds, (1, 2, 0))
    image2_ds = np.transpose(xs2_ds, (1, 2, 0))

    # Denormalize if necessary (assuming the images were normalized to [-1, 1])
    image1 = (image1 - image1.min()) / (image1.max() - image1.min())
    image1 = (image1 * 255).astype(np.uint8)

    m2 = m2 * 255
    m2_ds = m2_ds * 255

    image1_ds = (image1_ds - image1_ds.min()) / (image1_ds.max() - image1_ds.min())
    image1_ds = (image1_ds * 255).astype(np.uint8)

    image2 = (image2 - image2.min()) / (image2.max() - image2.min())
    image2 = (image2 * 255).astype(np.uint8)

    image2_ds = (image2_ds - image2_ds.min()) / (image2_ds.max() - image2_ds.min())
    image2_ds = (image2_ds * 255).astype(np.uint8)

    # Create a subplot to display both images
    fig, axes = plt.subplots(1, 6)
    axes[0].imshow(image1, cmap='gray')
    axes[0].axis('off')  # Turn off axis labels
    axes[0].set_title('xs1')

    rect = patches.Rectangle((w_coor_min, h_coor_min), w_coor_max - w_coor_min, h_coor_max - h_coor_min,
                             linewidth=1, edgecolor='r', facecolor='none')
    axes[0].add_patch(rect)

    axes[1].imshow(image2, cmap='gray')
    axes[1].axis('off')  # Turn off axis labels
    axes[1].set_title('xs2')

    axes[2].imshow(image1_ds, cmap='gray')
    axes[2].axis('off')  # Turn off axis labels
    axes[2].set_title('xs1_ds')

    axes[3].imshow(image2_ds, cmap='gray')
    axes[3].axis('off')  # Turn off axis labels
    axes[3].set_title('xs2_ds')

    axes[4].imshow(m2, cmap='gray')
    axes[4].axis('off')  # Turn off axis labels
    axes[4].set_title('mask')

    axes[5].imshow(m2_ds, cmap='gray')
    axes[5].axis('off')  # Turn off axis labels
    axes[5].set_title('mask_ds')

    plt.show()


def main():
    root_dir = '/Users/piotrwojcik/Downloads/mito_work/dataset'
    image_size = 448
    image_size_ds = 224
    seed = 1
    validation_size = 0.0
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, targets = get_grayscale(True,
                             os.path.join(root_dir, 'train'),
                             os.path.join(root_dir, 'train'),
                             os.path.join(root_dir, 'test'), image_size, image_size_ds, seed, validation_size)

    imgs = trainset.dataset.imgs
    for idx in range(len(trainset)):
        xs1, xs2, m2, xs1_ds, xs2_ds, m2_ds, hflip1, hflip2, ys = trainset[idx]
        print(idx)
        h_coor_min = 141
        w_coor_min = 130
        delta = 32
        msk_tensor_patch = m2[h_coor_min:(h_coor_min + delta), w_coor_min:(w_coor_min + delta)]
        num_white_pixels = torch.sum(msk_tensor_patch).item()
        #print(num_white_pixels)
        #print(msk_tensor_patch)
        #print(msk_tensor_patch.shape)
        #display_tensor_images(xs1, xs2, m2, xs1_ds, xs2_ds, m2_ds, h_coor_min, (h_coor_min + delta),
        #                      w_coor_min, (w_coor_min + delta))
        #if idx >= 15:
        #    break


if __name__ == "__main__":
    main()