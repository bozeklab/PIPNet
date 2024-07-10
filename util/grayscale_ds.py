import numpy as np
import argparse
import torch
import os
import torch.optim
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from typing import Tuple, Dict
from torch import Tensor
import random
from sklearn.model_selection import train_test_split

from data import TrivialAugmentWideNoColor, TrivialAugmentWideNoShape, create_datasets


def get_grayscale(augment:bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float, train_dir_pretrain = None):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.Grayscale(3), #convert to grayscale with three channels
                            transforms.ToTensor(),
                            normalize
                        ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)),
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+8, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
                            TrivialAugmentWideNoShape(),
                            transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                            transforms.Grayscale(3),#convert to grayscale with three channels
                            transforms.ToTensor(),
                            normalize
                            ])
    else:
        transform1 = transform_no_augment
        transform2 = transform_no_augment

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size)


def display_tensor_images(xs1, xs2):
    # Ensure the tensors are on the CPU and convert them to NumPy arrays
    xs1 = xs1.cpu().numpy()
    xs2 = xs2.cpu().numpy()

    # Transpose the arrays from [C, H, W] to [H, W, C]
    image1 = np.transpose(xs1, (1, 2, 0))
    image2 = np.transpose(xs2, (1, 2, 0))

    # Denormalize if necessary (assuming the images were normalized to [-1, 1])
    image1 = (image1 - image1.min()) / (image1.max() - image1.min())
    image1 = (image1 * 255).astype(np.uint8)

    image2 = (image2 - image2.min()) / (image2.max() - image2.min())
    image2 = (image2 * 255).astype(np.uint8)

    # Create a subplot to display both images
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image1, cmap='gray')
    axes[0].axis('off')  # Turn off axis labels
    axes[0].set_title('xs1')

    axes[1].imshow(image2, cmap='gray')
    axes[1].axis('off')  # Turn off axis labels
    axes[1].set_title('xs2')

    plt.show()


def main():
    root_dir = '/Users/piotrwojcik/Downloads/mito_scale_resized_512_split/'
    image_size = 512
    seed = 1
    validation_size = 0.0
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, targets = get_grayscale(True,
                             os.path.join(root_dir, 'train'),
                             os.path.join(root_dir, 'train'),
                             os.path.join(root_dir, 'test'), image_size, seed, validation_size)

    for idx in range(len(trainset)):
        xs1, xs2, ys = trainset[idx]

        display_tensor_images(xs1, xs2)

        if idx >= 15:
            break


if __name__ == "__main__":
    main()