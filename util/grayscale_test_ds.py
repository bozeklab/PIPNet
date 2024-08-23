import numpy as np
import argparse
import torch
import os
import torch.optim
import torch.utils.data
import torchvision
from PIL import Image, ImageDraw as D
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from typing import Tuple, Dict
import matplotlib.patches as patches
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
                           train_dir, project_dir, test_dir, seed, validation_size)


def display_tensor_images(xs, xs_ds, m, m_ds, h_coor_min, h_coor_max, w_coor_min, w_coor_max):
    # Ensure the tensors are on the CPU and convert them to NumPy arrays
    xs = xs.cpu().numpy()
    xs_ds = xs_ds.cpu().numpy()
    m = m.cpu().numpy()
    m_ds = m_ds.cpu().numpy()

    m = m * 255
    m_ds = m_ds * 255

    # Transpose the arrays from [C, H, W] to [H, W, C]
    image = np.transpose(xs, (1, 2, 0))
    image_ds = np.transpose(xs_ds, (1, 2, 0))

    # Denormalize if necessary (assuming the images were normalized to [-1, 1])
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)

    image_ds = (image_ds - image_ds.min()) / (image_ds.max() - image_ds.min())
    image_ds = (image_ds * 255).astype(np.uint8)

    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)

    image_ds = (image_ds - image_ds.min()) / (image_ds.max() - image_ds.min())
    image_ds = (image_ds * 255).astype(np.uint8)

    # Create a subplot to display both images
    fig, axes = plt.subplots(1, 4)
    axes[0].imshow(image, cmap='gray')
    axes[0].axis('off')  # Turn off axis labels
    axes[0].set_title('xs1')

    rect = patches.Rectangle((w_coor_min, h_coor_min), w_coor_max - w_coor_min, h_coor_max - h_coor_min,
                             linewidth=1, edgecolor='r', facecolor='none')
    axes[0].add_patch(rect)

    axes[1].imshow(image_ds, cmap='gray')
    axes[1].axis('off')  # Turn off axis labels
    axes[1].set_title('xs2')

    axes[2].imshow(m, cmap='gray')
    axes[2].axis('off')  # Turn off axis labels
    axes[2].set_title('mask')
    rect = patches.Rectangle((w_coor_min, h_coor_min), w_coor_max - w_coor_min, h_coor_max - h_coor_min,
                             linewidth=1, edgecolor='r', facecolor='none')
    axes[2].add_patch(rect)

    axes[3].imshow(m_ds, cmap='gray')
    axes[3].axis('off')  # Turn off axis labels
    axes[3].set_title('mask_ds')

    plt.show()


def main():
    root_dir = '/Users/piotrwojcik/Downloads/mito_work/dataset_512'

    example_img = '/Users/piotrwojcik/Downloads/mito_work/dataset_512/train/0_fl_fl/10kX_919_wt__0067.png'
    example_mask = '/Users/piotrwojcik/Downloads/mito_work/dataset_512/train/0_fl_fl/mask_10kX_919_wt__0067.png'

    image = transforms.Resize(size=(448, 448))(Image.open(example_img).convert("RGB"))
    mask = transforms.Resize(size=(448, 448))(Image.open(example_mask).convert("RGB"))
    msk_tensor = transforms.ToTensor()(mask)
    img_tensor = (transforms.ToTensor()(image) * 255).int()  # shape (1, 3, h, w)

    output = (img_tensor.numpy() * (0.6 * msk_tensor.numpy() + 0.4)).astype(np.uint8)
    output_image = Image.fromarray(np.squeeze(output).transpose(1, 2, 0))
    output_image.show()


    image_size = 448
    image_size_ds = 224
    seed = 1
    validation_size = 0.0
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, targets = get_grayscale(True,
                             os.path.join(root_dir, 'train'),
                             os.path.join(root_dir, 'train'),
                             os.path.join(root_dir, 'test'), image_size, image_size_ds, seed, validation_size)

    print(len(projectset))
    for idx in range(len(testset)):
        xs, xs_ds, m, m_ds, ys = testset[idx]

        h_coor_min = 141
        w_coor_min = 130
        delta = 32
        msk_tensor_patch = m[h_coor_min:(h_coor_min + delta), w_coor_min:(w_coor_min + delta)]
        num_white_pixels = torch.sum(msk_tensor_patch).item()
        print(num_white_pixels)
        print(msk_tensor_patch)
        print(msk_tensor_patch.shape)

        #display_tensor_images(xs, xs_ds, m, m_ds, h_coor_min, (h_coor_min + delta),
        #                      w_coor_min, (w_coor_min + delta))
        #if idx >= 15:
        #    break


if __name__ == "__main__":
    main()