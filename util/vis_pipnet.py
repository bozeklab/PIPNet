from tqdm import tqdm
import argparse
import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.data
import os
from PIL import Image, ImageDraw as D
import torchvision.transforms as transforms
import torchvision

from util.data import create_boolean_mask
from util.func import get_patch_size
import random

@torch.no_grad()                    
def visualize_topk(net, projectloader, num_classes, device, foldername, args: argparse.Namespace, k=5, compute_jaccard=False):
    print("Visualizing prototypes for topk...", flush=True)
    dir = os.path.join(args.log_dir, foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)

    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    tensors_per_prototype = dict()

    m_jaccard = []
    
    for p in range(net.module._num_prototypes):
        near_imgs_dir = os.path.join(dir, str(p))
        near_imgs_dirs[p]=near_imgs_dir
        seen_max[p]=0.
        saved[p]=0
        saved_ys[p]=[]
        tensors_per_prototype[p]=[]

    imgs = projectloader.dataset.imgs
    masks = []
    for i in imgs:
        mask_path, target = i
        directory, filename = os.path.split(mask_path)
        name, extension = os.path.splitext(filename)
        new_maskname = 'mask_' + name + extension
        new_mask_path = os.path.join(directory, new_maskname)
        masks.append((new_mask_path, target))

    
    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight

    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=50.,
                    desc='Collecting topk',
                    ncols=0)

    # Iterate through the data
    images_seen = 0
    topks = dict()
    # Iterate through the training set
    for i, (xs, xs_ds, m, m_ds, ys) in img_iter:
        images_seen+=1
        xs, xs_ds, m, m_ds, ys = xs.to(device), xs_ds.to(device), m.to(device), m_ds.to(device),  ys.to(device)

        with torch.no_grad():
            # Use the model to classify this batch of input data
            pfs, _, pooled, _ = net(xs=xs, xs_ds=xs_ds, inference=True)
            pooled = pooled.squeeze(0) 
            pfs = pfs.squeeze(0)

            # pfs shape [256, 16, 16]
            # pooled [256]

            for p in range(pooled.shape[0]):
                c_weight = torch.max(classification_weights[:,p]) 
                if c_weight > 1e-3:#ignore prototypes that are not relevant to any class
                    if p not in topks.keys():
                        topks[p] = []
                        
                    if len(topks[p]) < k:
                        topks[p].append((i, pooled[p].item()))
                    else:
                        topks[p] = sorted(topks[p], key=lambda tup: tup[1], reverse=True)
                        if topks[p][-1][1] < pooled[p].item():
                            topks[p][-1] = (i, pooled[p].item())
                        if topks[p][-1][1] == pooled[p].item():
                            # equal scores. randomly chose one (since dataset is not shuffled so latter images with same scores can now also get in topk).
                            replace_choice = random.choice([0, 1])
                            if replace_choice > 0:
                                topks[p][-1] = (i, pooled[p].item())

    alli = []
    prototypes_not_used = []
    for p in topks.keys():
        found = False
        for idx, score in topks[p]:
            alli.append(idx)
            if score > 0.1:  #in case prototypes have fewer than k well-related patches
                found = True
        if not found:
            prototypes_not_used.append(p)

    print(len(prototypes_not_used), "prototypes do not have any similarity score > 0.1. Will be ignored in visualisation.")
    abstained = 0
    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=50.,
                    desc='Visualizing topk',
                    ncols=0)
    for i, (xs, xs_ds, m, m_ds, ys) in img_iter: #shuffle is false so should lead to same order as in imgs
        if i in alli:
            xs, ys = xs.to(device), ys.to(device)
            for p in topks.keys():
                if p not in prototypes_not_used:
                    for idx, score in topks[p]:
                        if idx == i:
                            # Use the model to classify this batch of input data
                            with torch.no_grad():
                                proto_features, proto_features_ds, clamped_pooled, out = net(xs=xs, xs_ds=xs_ds, inference=True) #softmaxes has shape (1, num_prototypes, W, H)
                                outmax = torch.amax(out,dim=1)[0] #shape ([1]) because batch size of projectloader is 1
                                if outmax.item() == 0.:
                                    abstained+=1

                            if p >= net.module._num_prototypes // 2:
                                img_size = args.image_size_ds
                                softmaxes = proto_features_ds
                                pidx = p - net.module._num_prototypes // 2
                            else:
                                img_size = args.image_size
                                softmaxes = proto_features
                                pidx = p

                            patchsize, skip = get_patch_size(args, p, net.module._num_prototypes)

                            # Take the max per prototype.                             
                            max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
                            max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
                            max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1) #shape (num_prototypes)
                            
                            c_weight = torch.max(classification_weights[:, pidx]) #ignore prototypes that are not relevant to any class
                            if (c_weight > 1e-10) or ('pretrain' in foldername):
                                
                                h_idx = max_idx_per_prototype_h[pidx, max_idx_per_prototype_w[pidx]]
                                w_idx = max_idx_per_prototype_w[pidx]
                                
                                img_to_open = imgs[i]
                                mask_to_open = masks[i]
                                if isinstance(img_to_open, tuple) or isinstance(img_to_open, list): #dataset contains tuples of (img,label)
                                    img_to_open = img_to_open[0]

                                if isinstance(mask_to_open, tuple) or isinstance(mask_to_open, list): #dataset contains tuples of (img,label)
                                    mask_to_open = mask_to_open[0]
                                
                                image = transforms.Resize(size=(img_size, img_size))(Image.open(img_to_open))
                                mask = transforms.Resize(size=(img_size, img_size))(Image.open(mask_to_open))
                                image = transforms.Grayscale(3)(image)

                                img_tensor = transforms.ToTensor()(image).unsqueeze_(0) #shape (1, 3, h, w)
                                msk_tensor = transforms.ToTensor()(mask)
                                msk_tensor = create_boolean_mask(msk_tensor)
                                h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(img_size, softmaxes.shape, patchsize, skip, h_idx, w_idx)
                                img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                                msk_tensor_patch = msk_tensor[h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                                num_white_pixels = torch.sum(msk_tensor_patch).item()

                                m_jaccard.append(num_white_pixels/((h_coor_max-h_coor_min)*(w_coor_max-w_coor_min)))

                                # Count the number of white pixels
                                #num_white_pixels = torch.sum(mask).item()

                                saved[p]+=1
                                tensors_per_prototype[p].append(img_tensor_patch)

    print("Abstained: ", abstained, flush=True)
    if compute_jaccard:
        import statistics
        print('Jaccard: ', statistics.mean(m_jaccard))
    all_tensors = []
    for p in range(net.module._num_prototypes):
        print(p, saved[p])
        if saved[p]>0:
            # add text next to each topk-grid, to easily see which prototype it is
            text = "P "+str(p)
            txtimage = Image.new("RGB", (img_tensor_patch.shape[1],img_tensor_patch.shape[2]), (0, 0, 0))
            draw = D.Draw(txtimage)
            draw.text((img_tensor_patch.shape[0]//2, img_tensor_patch.shape[1]//2), text, anchor='mm', fill="white")
            txttensor = transforms.ToTensor()(txtimage)
            tensors_per_prototype[p].append(txttensor)
            # save top-k image patches in grid
            try:
                for i in range(len(tensors_per_prototype[p])):
                    print(tensors_per_prototype[p][i].shape)
                print()
                grid = torchvision.utils.make_grid(tensors_per_prototype[p], nrow=k+1, padding=1)
                torchvision.utils.save_image(grid,os.path.join(dir,"grid_topk_%s.png"%(str(p))))
                if saved[p]>=k:
                    all_tensors+=tensors_per_prototype[p]
                print('yes saved')
            except Exception as e:
                print(f"Something is wrong: {e}")
    if len(all_tensors)>0:
        grid = torchvision.utils.make_grid(all_tensors, nrow=k+1, padding=1)
        torchvision.utils.save_image(grid,os.path.join(dir,"grid_topk_all.png"))
    else:
        print("Pretrained prototypes not visualized. Try to pretrain longer.", flush=True)
    return topks


def remove_background(net, projectloader, num_classes, device, args: argparse.Namespace):
    print("Removing background prototypes...", flush=True)

    seen_max = dict()
    fg_patches_per_prototype = dict()

    for p in range(net.module._num_prototypes):
        seen_max[p] = 0.
        fg_patches_per_prototype[p] = []

    imgs = projectloader.dataset.imgs
    masks = []
    for i in imgs:
        mask_path, target = i
        directory, filename = os.path.split(mask_path)
        name, extension = os.path.splitext(filename)
        new_maskname = 'mask_' + name + extension
        new_mask_path = os.path.join(directory, new_maskname)
        masks.append((new_mask_path, target))

    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight
    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=100.,
                    desc='Removing background prototypes',
                    ncols=0)

    # Iterate through the data
    images_seen_before = 0
    for i, (xs, xs_ds, m, m_ds, ys) in img_iter:  # shuffle is false so should lead to same order as in imgs
        xs, xs_ds, m, m_ds, ys = xs.to(device), xs_ds.to(device), m.to(device), m_ds.to(device), ys.to(device)
        # Use the model to classify this batch of input data
        with torch.no_grad():
            proto_features, proto_features_ds, clamped_pooled, out = net(xs=xs, xs_ds=xs_ds, inference=True)

        for p in range(0, net.module._num_prototypes):
            patchsize, skip = get_patch_size(args, p, net.module._num_prototypes)
            if p >= net.module._num_prototypes // 2:
                img_size = args.image_size_ds
                softmaxes = proto_features_ds
                pidx = p - net.module._num_prototypes // 2
            else:
                img_size = args.image_size
                softmaxes = proto_features
                pidx = p

            max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
            # In PyTorch, images are represented as [channels, height, width]
            max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
            max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1)

            c_weight = torch.max(classification_weights[:, p])  # ignore prototypes that are not relevant to any class
            if c_weight > 0:
                h_idx = max_idx_per_prototype_h[pidx, max_idx_per_prototype_w[pidx]]
                w_idx = max_idx_per_prototype_w[pidx]
                idx_to_select = max_idx_per_prototype[pidx, h_idx, w_idx].item()
                found_max = max_per_prototype[pidx, h_idx, w_idx].item()

                if found_max > seen_max[p]:
                    seen_max[p] = found_max

                if found_max > 0.5:
                    img_to_open = imgs[images_seen_before + idx_to_select]
                    mask_to_open = masks[images_seen_before + idx_to_select]

                    if isinstance(img_to_open, tuple) or isinstance(img_to_open,
                                                                    list):  # dataset contains tuples of (img,label)
                        img_to_open = img_to_open[0]

                    if isinstance(mask_to_open, tuple) or isinstance(mask_to_open,
                                                                     list):  # dataset contains tuples of (img,label)
                        mask_to_open = mask_to_open[0]

                    image = transforms.Resize(size=(img_size, img_size))(Image.open(img_to_open).convert("RGB"))
                    mask = transforms.Resize(size=(img_size, img_size))(Image.open(mask_to_open).convert("RGB"))
                    msk_tensor = transforms.ToTensor()(mask)
                    bool_mask = create_boolean_mask(msk_tensor)
                    img_tensor = transforms.ToTensor()(image).unsqueeze_(0)  # shape (1, 3, h, w)
                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(img_size, softmaxes.shape,
                                                                                         patchsize, skip, h_idx, w_idx)
                    msk_tensor_patch = bool_mask[h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                    num_white_pixels = torch.sum(msk_tensor_patch).item()
                    if num_white_pixels >= 100:
                        fg_patches_per_prototype[p].append(True)
                    else:
                        fg_patches_per_prototype[p].append(False)

        images_seen_before += len(ys)
    fractions = dict()

    for k in fg_patches_per_prototype.keys():
        bool_list = fg_patches_per_prototype[k]
        true_count = sum(bool_list)  # Count the number of True values
        total_count = len(bool_list)  # Count the total number of items
        fraction = true_count / total_count if total_count > 0 else 0  # Calculate fraction
        fractions[k] = fraction
    return fractions


def visualize(net, projectloader, num_classes, device, foldername, args: argparse.Namespace):
    print("Visualizing prototypes...", flush=True)
    dir = os.path.join(args.log_dir, foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)

    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    tensors_per_prototype = dict()
    abstainedimgs = set()
    notabstainedimgs = set()
    
    for p in range(net.module._num_prototypes):
        near_imgs_dir = os.path.join(dir, str(p))
        near_imgs_dirs[p]=near_imgs_dir
        seen_max[p]=0.
        saved[p]=0
        saved_ys[p]=[]
        tensors_per_prototype[p]=[]

    imgs = projectloader.dataset.imgs
    masks = []
    for i in imgs:
        mask_path, target = i
        directory, filename = os.path.split(mask_path)
        name, extension = os.path.splitext(filename)
        new_maskname = 'mask_' + name + extension
        new_mask_path = os.path.join(directory, new_maskname)
        masks.append((new_mask_path, target))
    
    # skip some images for visualisation to speed up the process
    if len(imgs)/num_classes <10:
        skip_img=10
    elif len(imgs)/num_classes < 50:
        skip_img=5
    else:
        skip_img = 2
    print("Every", skip_img, "is skipped in order to speed up the visualisation process", flush=True)

    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight
    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=100.,
                    desc='Visualizing',
                    ncols=0)

    # Iterate through the data
    images_seen_before = 0
    for i, (xs, xs_ds, m, m_ds, ys) in img_iter: #shuffle is false so should lead to same order as in imgs
        if i % skip_img == 0:
            images_seen_before+=xs.shape[0]
            continue
        
        xs, xs_ds, m, m_ds, ys = xs.to(device), xs_ds.to(device), m.to(device), m_ds.to(device), ys.to(device)
        # Use the model to classify this batch of input data
        with torch.no_grad():
            proto_features, proto_features_ds, clamped_pooled, out = net(xs=xs, xs_ds=xs_ds, inference=True)

        for p in range(0, net.module._num_prototypes):
            patchsize, skip = get_patch_size(args, p, net.module._num_prototypes)
            if p >= net.module._num_prototypes // 2:
                img_size = args.image_size_ds
                softmaxes = proto_features_ds
                pidx = p - net.module._num_prototypes // 2
            else:
                img_size = args.image_size
                softmaxes = proto_features
                pidx = p

            max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
            # In PyTorch, images are represented as [channels, height, width]
            max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
            max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1)

            c_weight = torch.max(classification_weights[:,p]) #ignore prototypes that are not relevant to any class
            if c_weight>0:
                h_idx = max_idx_per_prototype_h[pidx, max_idx_per_prototype_w[pidx]]
                w_idx = max_idx_per_prototype_w[pidx]
                idx_to_select = max_idx_per_prototype[pidx,h_idx, w_idx].item()
                found_max = max_per_prototype[pidx,h_idx, w_idx].item()

                imgname = imgs[images_seen_before+idx_to_select]
                if out.max() < 1e-8:
                    abstainedimgs.add(imgname)
                else:
                    notabstainedimgs.add(imgname)
                
                if found_max > seen_max[p]:
                    seen_max[p]=found_max
               
                if found_max > 0.5:
                    img_to_open = imgs[images_seen_before+idx_to_select]
                    mask_to_open = masks[images_seen_before+idx_to_select]

                    if isinstance(img_to_open, tuple) or isinstance(img_to_open, list): #dataset contains tuples of (img,label)
                        imglabel = img_to_open[1]
                        img_to_open = img_to_open[0]

                    if isinstance(mask_to_open, tuple) or isinstance(mask_to_open, list):  # dataset contains tuples of (img,label)
                        mask_to_open = mask_to_open[0]

                    image = transforms.Resize(size=(img_size, img_size))(Image.open(img_to_open).convert("RGB"))
                    mask = transforms.Resize(size=(img_size, img_size))(Image.open(mask_to_open).convert("RGB"))
                    msk_tensor = transforms.ToTensor()(mask)
                    bool_mask = create_boolean_mask(msk_tensor)
                    img_tensor = transforms.ToTensor()(image).unsqueeze_(0) #shape (1, 3, h, w)
                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(img_size, softmaxes.shape, patchsize, skip, h_idx, w_idx)
                    img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                    msk_tensor_patch = bool_mask[h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                    saved[p]+=1
                    tensors_per_prototype[p].append((img_tensor_patch, found_max))

                    num_white_pixels = torch.sum(msk_tensor_patch).item()
                    if num_white_pixels >= 100:
                        boundary_color = "red"
                    else:
                        boundary_color = "yellow"

                    save_path = os.path.join(dir, "prototype_%s")%str(p)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    msk_tensor = transforms.ToTensor()(mask)
                    img_tensor = (transforms.ToTensor()(image) * 255).int()
                    output = (img_tensor.numpy() * (0.6 * msk_tensor.numpy() + 0.4)).astype(np.uint8)
                    output = Image.fromarray(np.squeeze(output).transpose(1, 2, 0))
                    draw = D.Draw(output)
                    draw.rectangle([(w_coor_min, h_coor_min), (w_coor_max, h_coor_max)], outline=boundary_color, width=2)
                    output.save(os.path.join(save_path, 'p%s_%s_%s_%s_rect.png'%(str(p),str(imglabel),str(round(found_max, 2)),str(img_to_open.split('/')[-1].split('.jpg')[0]))))

        images_seen_before+=len(ys)

    print("num images abstained: ", len(abstainedimgs), flush=True)
    print("num images not abstained: ", len(notabstainedimgs), flush=True)
    for p in range(net.module._num_prototypes):
        if saved[p]>0:
            try:
                sorted_by_second = sorted(tensors_per_prototype[p], key=lambda tup: tup[1], reverse=True)
                sorted_ps = [i[0] for i in sorted_by_second]
                grid = torchvision.utils.make_grid(sorted_ps, nrow=16, padding=1)
                torchvision.utils.save_image(grid,os.path.join(dir,"grid_%s.png"%(str(p))))
            except RuntimeError:
                pass

# convert latent location to coordinates of image patch
def get_img_coordinates(img_size, softmaxes_shape, patchsize, skip, h_idx, w_idx):
    # in case latent output size is 26x26. For convnext with smaller strides. 
    if softmaxes_shape[1] == 26 and softmaxes_shape[2] == 26:
        #Since the outer latent patches have a smaller receptive field, skip size is set to 4 for the first and last patch. 8 for rest.
        h_coor_min = max(0,(h_idx-1)*skip+4)
        if h_idx < softmaxes_shape[-1]-1:
            h_coor_max = h_coor_min + patchsize
        else:
            h_coor_min -= 4
            h_coor_max = h_coor_min + patchsize
        w_coor_min = max(0,(w_idx-1)*skip+4)
        if w_idx < softmaxes_shape[-1]-1:
            w_coor_max = w_coor_min + patchsize
        else:
            w_coor_min -= 4
            w_coor_max = w_coor_min + patchsize
    else:
        h_coor_min = h_idx*skip
        h_coor_max = min(img_size, h_idx*skip+patchsize)
        w_coor_min = w_idx*skip
        w_coor_max = min(img_size, w_idx*skip+patchsize)                                    
    
    if h_idx == softmaxes_shape[1]-1:
        h_coor_max = img_size
    if w_idx == softmaxes_shape[2] -1:
        w_coor_max = img_size
    if h_coor_max == img_size:
        h_coor_min = img_size-patchsize
    if w_coor_max == img_size:
        w_coor_min = img_size-patchsize

    return h_coor_min, h_coor_max, w_coor_min, w_coor_max
    