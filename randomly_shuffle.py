import torch
import torchvision.transforms.functional as F
import numpy as np
import random
from PIL import Image


def shuffle_image_patches(img, patch_size=32):
    """
    Randomly shuffles patches of the given image.

    Args:
        img (PIL Image): Input image.
        patch_size (int): Size of each square patch.

    Returns:
        PIL Image: Augmented image with shuffled patches.
    """
    # Convert image to grayscale
    img = F.rgb_to_grayscale(img)
    #img = F.resize(img, (224, 224))

    # Convert PIL image to tensor
    img_tensor = F.to_tensor(img)

    # Get image dimensions
    c, h, w = img_tensor.shape

    # Ensure the patch size is valid
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError("Image dimensions should be divisible by patch_size")

    # Divide image into patches
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img_tensor[:, i:i + patch_size, j:j + patch_size]
            patches.append(patch)

    # Shuffle patches randomly
    random.shuffle(patches)

    # Reconstruct the shuffled image
    shuffled_img = torch.zeros_like(img_tensor)
    idx = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            shuffled_img[:, i:i + patch_size, j:j + patch_size] = patches[idx]
            idx += 1

    # Convert back to PIL image
    return F.to_pil_image(shuffled_img)


# Load an example image
img = Image.open("/Users/piotrwojcik/Downloads/mito_work/dataset/train/1_cl1/mask_10kX_919cl1__0019.png")

# Apply patch shuffling
shuffled_img = shuffle_image_patches(img, patch_size=32)

# Save or display the shuffled image
#shuffled_img.save("shuffled_image.png")
shuffled_img.show()

