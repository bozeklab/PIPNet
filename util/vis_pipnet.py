from tqdm import tqdm
import argparse
import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.data
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw as D
import torchvision.transforms as transforms
import torchvision
import random

from util.data import create_boolean_mask
# from util.func import get_patch_size  # DS-aware; we avoid it in single-scale


def get_patch_size_single(args: argparse.Namespace, proto_features_hw, img_size: int):
    """
    Single-scale patch geometry for ViT-like grids.

    proto_features_hw: (Hf, Wf) of proto map for one prototype (e.g. 16x16)
    img_size: image resolution used in visualization (args.image_size)

    Returns:
      patchsize_px, skip_px
    """
    Hf, Wf = proto_features_hw
    # assume square grid and square image
    # "cell" in pixels
    cell_h = img_size // Hf
    cell_w = img_size // Wf
    # for ViT patch embeddings, cell_h==cell_w==patch_size, but this works generally.
    patchsize = min(cell_h, cell_w)
    skip = patchsize
    return patchsize, skip


def build_masks_list_from_imgs(imgs):
    """
    Your dataset stores image paths in projectloader.dataset.imgs, but you build mask paths by
    prefixing 'mask_' to the filename.

    imgs: list of (img_path, target) or img_path
    returns: list of (mask_path, target) tuples aligned with imgs indices
    """
    masks = []
    for item in imgs:
        mask_path, target = item
        directory, filename = os.path.split(mask_path)
        name, extension = os.path.splitext(filename)
        new_maskname = "mask_" + name + extension
        new_mask_path = os.path.join(directory, new_maskname)
        masks.append((new_mask_path, target))
    return masks


@torch.no_grad()
def visualize_topk(
    net,
    projectloader,
    num_classes,
    device,
    foldername,
    args: argparse.Namespace,
    k=5,
    compute_jaccard=True
):
    """
    Single-scale top-k visualization:
    - uses proto_features only
    - pooled is [num_prototypes] (single scale)
    - no DS prototypes and no half-split
    """
    print("Visualizing prototypes for topk (single-scale)...", flush=True)
    out_dir = os.path.join(args.log_dir, foldername)
    os.makedirs(out_dir, exist_ok=True)

    P = net.module._num_prototypes
    saved = {p: 0 for p in range(P)}
    tensors_per_prototype = {p: [] for p in range(P)}
    m_jaccard = []

    imgs = projectloader.dataset.imgs
    masks = build_masks_list_from_imgs(imgs)

    net.eval()
    classification_weights = net.module._classification.weight  # [num_classes, P] (single scale)

    # Collect top-k indices per prototype by pooled score
    img_iter = tqdm(
        enumerate(projectloader),
        total=len(projectloader),
        mininterval=50.0,
        desc="Collecting topk",
        ncols=0,
    )

    topks = {}  # p -> list[(dataset_index, score)]
    for i, (xs, xs_ds, m, m_ds, ys) in img_iter:
        xs = xs.to(device)
        # xs_ds ignored

        proto_features, _, pooled, _out = net(xs=xs, xs_ds=xs_ds, inference=True)
        pooled = pooled.squeeze(0)          # [P]
        # proto_features = proto_features.squeeze(0)  # [P,Hf,Wf] (not needed here)

        for p in range(P):
            c_weight = torch.max(classification_weights[:, p])
            if c_weight > 1e-3:  # ignore irrelevant prototypes
                if p not in topks:
                    topks[p] = []
                score = pooled[p].item()
                if len(topks[p]) < k:
                    topks[p].append((i, score))
                else:
                    topks[p] = sorted(topks[p], key=lambda tup: tup[1], reverse=True)
                    if topks[p][-1][1] < score:
                        topks[p][-1] = (i, score)
                    elif topks[p][-1][1] == score:
                        if random.choice([0, 1]) == 1:
                            topks[p][-1] = (i, score)

    # Determine which dataset indices we need to revisit
    alli = []
    prototypes_not_used = []
    for p, lst in topks.items():
        found = any(score > 0.1 for _, score in lst)
        alli.extend([idx for idx, _ in lst])
        if not found:
            prototypes_not_used.append(p)

    print(len(prototypes_not_used), "prototypes do not have any similarity score > 0.1. Will be ignored.")

    abstained = 0
    img_iter = tqdm(
        enumerate(projectloader),
        total=len(projectloader),
        mininterval=50.0,
        desc="Visualizing topk",
        ncols=0,
    )

    for i, (xs, xs_ds, m, m_ds, ys) in img_iter:
        if i not in alli:
            continue

        xs = xs.to(device)
        ys = ys.to(device)

        proto_features, _, clamped_pooled, out = net(xs=xs, xs_ds=xs_ds, inference=True)
        outmax = torch.amax(out, dim=1)[0].item()
        if outmax == 0.0:
            abstained += 1

        # single scale tensors
        softmaxes = proto_features.squeeze(0)  # [P,Hf,Wf]
        img_size = args.image_size

        # precompute maxima maps for all prototypes
        max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)   # [Hf,Wf], [Hf,Wf] over P? careful
        # The above is wrong dimensionally because dim=0 collapses P.
        # We need per-prototype max location:
        # softmaxes[p] is [Hf,Wf], so do per-prototype:
        max_per_prototype = softmaxes.view(P, -1).max(dim=1).values  # [P]
        max_idx_flat = softmaxes.view(P, -1).max(dim=1).indices      # [P]
        Hf, Wf = softmaxes.shape[1], softmaxes.shape[2]
        max_h = (max_idx_flat // Wf).long()
        max_w = (max_idx_flat % Wf).long()

        patchsize, skip = get_patch_size_single(args, (Hf, Wf), img_size)

        for p, lst in topks.items():
            if p in prototypes_not_used:
                continue
            for idx, score in lst:
                if idx != i:
                    continue

                c_weight = torch.max(classification_weights[:, p])
                if (c_weight <= 1e-10) and ("pretrain" not in foldername):
                    continue

                h_idx = int(max_h[p].item())
                w_idx = int(max_w[p].item())

                img_to_open = imgs[i]
                mask_to_open = masks[i]
                if isinstance(img_to_open, (tuple, list)):
                    img_to_open = img_to_open[0]
                if isinstance(mask_to_open, (tuple, list)):
                    mask_to_open = mask_to_open[0]

                image = transforms.Resize(size=(img_size, img_size))(Image.open(img_to_open))
                mask = transforms.Resize(size=(img_size, img_size))(Image.open(mask_to_open))
                image = transforms.Grayscale(3)(image)

                img_tensor = transforms.ToTensor()(image).unsqueeze(0)  # (1,3,H,W)
                msk_tensor = transforms.ToTensor()(mask)
                msk_tensor = create_boolean_mask(msk_tensor)

                h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(
                    img_size, (P, Hf, Wf), patchsize, skip, h_idx, w_idx
                )
                img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                msk_tensor_patch = msk_tensor[h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                num_white_pixels = torch.sum(msk_tensor_patch).item()

                m_jaccard.append(num_white_pixels / (msk_tensor_patch.numel()))
                saved[p] += 1
                tensors_per_prototype[p].append(img_tensor_patch)

    print("Abstained:", abstained, flush=True)
    if compute_jaccard and len(m_jaccard) > 0:
        import statistics
        print("Jaccard:", statistics.mean(m_jaccard))

    all_tensors = []
    for p in range(P):
        if saved[p] > 0:
            # add text tile
            text = "P " + str(p)
            # Use last patch shape to size text image
            patch = tensors_per_prototype[p][-1]
            txtimage = Image.new("RGB", (patch.shape[2], patch.shape[1]), (0, 0, 0))
            draw = D.Draw(txtimage)
            draw.text((patch.shape[2] // 2, patch.shape[1] // 2), text, anchor="mm", fill="white")
            txttensor = transforms.ToTensor()(txtimage)
            tensors_per_prototype[p].append(txttensor)

            try:
                grid = torchvision.utils.make_grid(tensors_per_prototype[p], nrow=k + 1, padding=1)
                torchvision.utils.save_image(grid, os.path.join(out_dir, f"grid_topk_{p}.png"))
                if saved[p] >= k:
                    all_tensors += tensors_per_prototype[p]
            except Exception as e:
                print(f"Something is wrong for prototype {p}: {e}")

    if len(all_tensors) > 0:
        grid = torchvision.utils.make_grid(all_tensors, nrow=k + 1, padding=1)
        torchvision.utils.save_image(grid, os.path.join(out_dir, "grid_topk_all.png"))
    else:
        print("No top-k grids saved. Try longer pretraining.", flush=True)

    return topks


def remove_background(net, projectloader, num_classes, device, args: argparse.Namespace):
    """
    Single-scale background estimation.
    Returns fraction of FG patches per prototype.
    """
    print("Removing background prototypes (single-scale)...", flush=True)

    P = net.module._num_prototypes
    seen_max = {p: 0.0 for p in range(P)}
    fg_patches_per_prototype = {p: [] for p in range(P)}

    imgs = projectloader.dataset.imgs
    masks = build_masks_list_from_imgs(imgs)

    net.eval()
    classification_weights = net.module._classification.weight

    img_iter = tqdm(
        enumerate(projectloader),
        total=len(projectloader),
        mininterval=100.0,
        desc="Removing background prototypes",
        ncols=0,
    )

    images_seen_before = 0
    for i, (xs, xs_ds, m, m_ds, ys) in img_iter:
        xs = xs.to(device)
        ys = ys.to(device)

        proto_features, _, clamped_pooled, out = net(xs=xs, xs_ds=xs_ds, inference=True)
        softmaxes = proto_features.squeeze(0)  # [P,Hf,Wf]
        Hf, Wf = softmaxes.shape[1], softmaxes.shape[2]

        img_size = args.image_size
        patchsize, skip = get_patch_size_single(args, (Hf, Wf), img_size)

        # per-prototype peak
        max_vals = softmaxes.view(P, -1).max(dim=1).values
        max_idx_flat = softmaxes.view(P, -1).max(dim=1).indices
        max_h = (max_idx_flat // Wf).long()
        max_w = (max_idx_flat % Wf).long()

        for p in range(P):
            c_weight = torch.max(classification_weights[:, p])
            if c_weight <= 0:
                continue

            found_max = float(max_vals[p].item())
            seen_max[p] = max(seen_max[p], found_max)

            if found_max <= 0.5:
                continue

            h_idx = int(max_h[p].item())
            w_idx = int(max_w[p].item())

            img_to_open = imgs[images_seen_before]
            mask_to_open = masks[images_seen_before]
            if isinstance(img_to_open, (tuple, list)):
                img_to_open = img_to_open[0]
            if isinstance(mask_to_open, (tuple, list)):
                mask_to_open = mask_to_open[0]

            image = transforms.Resize(size=(img_size, img_size))(Image.open(img_to_open).convert("RGB"))
            mask = transforms.Resize(size=(img_size, img_size))(Image.open(mask_to_open).convert("RGB"))
            bool_mask = create_boolean_mask(transforms.ToTensor()(mask))

            h0, h1, w0, w1 = get_img_coordinates(img_size, (P, Hf, Wf), patchsize, skip, h_idx, w_idx)
            msk_patch = bool_mask[h0:h1, w0:w1]
            num_white_pixels = int(torch.sum(msk_patch).item())
            fg_patches_per_prototype[p].append(num_white_pixels >= 100)

        images_seen_before += len(ys)

    fractions = {}
    for p, bool_list in fg_patches_per_prototype.items():
        true_count = sum(bool_list)
        total_count = len(bool_list)
        fractions[p] = true_count / total_count if total_count > 0 else 0.0
    return fractions


def visualize(net, projectloader, num_classes, device, foldername, args: argparse.Namespace):
    """
    Single-scale visualization of prototypes.
    Saves patch rectangles + heatmap overlays for prototypes.
    """
    print("Visualizing prototypes (single-scale)...", flush=True)
    out_dir = os.path.join(args.log_dir, foldername)
    os.makedirs(out_dir, exist_ok=True)

    P = net.module._num_prototypes
    seen_max = {p: 0.0 for p in range(P)}
    saved = {p: 0 for p in range(P)}
    tensors_per_prototype = {p: [] for p in range(P)}
    abstainedimgs = set()
    notabstainedimgs = set()

    imgs = projectloader.dataset.imgs
    masks = build_masks_list_from_imgs(imgs)

    # skipping policy (unchanged)
    if len(imgs) / num_classes < 10:
        skip_img = 10
    elif len(imgs) / num_classes < 50:
        skip_img = 5
    else:
        skip_img = 2
    print("Every", skip_img, "is skipped to speed up visualization", flush=True)

    net.eval()
    classification_weights = net.module._classification.weight

    img_iter = tqdm(
        enumerate(projectloader),
        total=len(projectloader),
        mininterval=100.0,
        desc="Visualizing",
        ncols=0,
    )

    images_seen_before = 0
    for i, (xs, xs_ds, m, m_ds, ys) in img_iter:
        if i % skip_img == 0:
            images_seen_before += xs.shape[0]
            continue

        xs = xs.to(device)
        ys = ys.to(device)

        proto_features, _, clamped_pooled, out = net(xs=xs, xs_ds=xs_ds, inference=True)
        softmaxes = proto_features.squeeze(0)  # [P,Hf,Wf]
        Hf, Wf = softmaxes.shape[1], softmaxes.shape[2]
        img_size = args.image_size

        patchsize, skip = get_patch_size_single(args, (Hf, Wf), img_size)

        # peaks per prototype
        max_vals = softmaxes.view(P, -1).max(dim=1).values
        max_idx_flat = softmaxes.view(P, -1).max(dim=1).indices
        max_h = (max_idx_flat // Wf).long()
        max_w = (max_idx_flat % Wf).long()

        for p in range(P):
            c_weight = torch.max(classification_weights[:, p])
            if c_weight <= 0:
                continue

            found_max = float(max_vals[p].item())
            seen_max[p] = max(seen_max[p], found_max)

            imgname = imgs[images_seen_before]
            if out.max().item() < 1e-8:
                abstainedimgs.add(imgname)
            else:
                notabstainedimgs.add(imgname)

            if found_max <= 0.5:
                continue

            h_idx = int(max_h[p].item())
            w_idx = int(max_w[p].item())

            img_to_open = imgs[images_seen_before]
            mask_to_open = masks[images_seen_before]

            imglabel = None
            if isinstance(img_to_open, (tuple, list)):
                imglabel = img_to_open[1]
                img_to_open = img_to_open[0]
            if isinstance(mask_to_open, (tuple, list)):
                mask_to_open = mask_to_open[0]

            image = transforms.Resize(size=(img_size, img_size))(Image.open(img_to_open).convert("RGB"))
            mask = transforms.Resize(size=(img_size, img_size))(Image.open(mask_to_open).convert("RGB"))

            bool_mask = create_boolean_mask(transforms.ToTensor()(mask))
            img_tensor = transforms.ToTensor()(image).unsqueeze(0)  # (1,3,H,W)

            h0, h1, w0, w1 = get_img_coordinates(img_size, (P, Hf, Wf), patchsize, skip, h_idx, w_idx)
            img_patch = img_tensor[0, :, h0:h1, w0:w1]
            msk_patch = bool_mask[h0:h1, w0:w1]
            num_white_pixels = int(torch.sum(msk_patch).item())

            saved[p] += 1
            tensors_per_prototype[p].append((img_patch, found_max))

            # heatmap overlay on full image
            hm = softmaxes[p].detach().cpu().numpy()  # [Hf,Wf]
            hm_img = Image.fromarray(np.uint8(255 * hm)).resize((img_size, img_size), Image.BICUBIC)
            hm_np = np.array(hm_img).astype(np.float32) / 255.0

            heatmap = cv2.applyColorMap(np.uint8(255 * hm_np), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255.0
            heatmap = heatmap[..., ::-1]  # BGR->RGB

            base = img_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            heatmap_img = (0.2 * heatmap + 0.6 * base).clip(0, 1)

            boundary_color = "red" if num_white_pixels >= 100 else "yellow"

            save_path = os.path.join(out_dir, f"prototype_{p}")
            os.makedirs(save_path, exist_ok=True)

            # rectangle overlay on masked image
            msk_tensor = transforms.ToTensor()(mask)
            img_tensor_u8 = (transforms.ToTensor()(image) * 255).int()
            output = (img_tensor_u8.numpy() * (0.6 * msk_tensor.numpy() + 0.4)).astype(np.uint8)
            output = Image.fromarray(np.squeeze(output).transpose(1, 2, 0))
            draw = D.Draw(output)
            draw.rectangle([(w0, h0), (w1, h1)], outline=boundary_color, width=2)

            tag = f"p{p}_{imglabel}_{round(found_max,2)}_{os.path.basename(img_to_open).split('.')[0]}"
            output.save(os.path.join(save_path, f"{tag}_rect.png"))
            plt.imsave(os.path.join(save_path, f"{tag}_heat.png"), heatmap_img, vmin=0.0, vmax=1.0)

        images_seen_before += len(ys)

    print("num images abstained:", len(abstainedimgs), flush=True)
    print("num images not abstained:", len(notabstainedimgs), flush=True)

    # grids per prototype
    for p in range(P):
        if saved[p] > 0:
            try:
                sorted_by_score = sorted(tensors_per_prototype[p], key=lambda tup: tup[1], reverse=True)
                patches = [t[0] for t in sorted_by_score]
                grid = torchvision.utils.make_grid(patches, nrow=16, padding=1)
                torchvision.utils.save_image(grid, os.path.join(out_dir, f"grid_{p}.png"))
            except RuntimeError:
                pass


# convert latent location to coordinates of image patch
def get_img_coordinates(img_size, softmaxes_shape, patchsize, skip, h_idx, w_idx):
    """
    softmaxes_shape is expected to be (P,Hf,Wf) or similar
    """
    Hf = softmaxes_shape[1]
    Wf = softmaxes_shape[2]

    # Standard grid mapping: each cell is "skip" pixels, patch is patchsize
    h_coor_min = h_idx * skip
    h_coor_max = min(img_size, h_idx * skip + patchsize)
    w_coor_min = w_idx * skip
    w_coor_max = min(img_size, w_idx * skip + patchsize)

    # Clamp last cell to image boundary
    if h_idx == Hf - 1:
        h_coor_max = img_size
    if w_idx == Wf - 1:
        w_coor_max = img_size
    if h_coor_max == img_size:
        h_coor_min = img_size - patchsize
    if w_coor_max == img_size:
        w_coor_min = img_size - patchsize

    return h_coor_min, h_coor_max, w_coor_min, w_coor_max