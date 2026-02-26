import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import wandb
import io
from PIL import Image


import torch


def sinkhorn_balance_probs_in_mask(
    Q: torch.Tensor,             # [B, D, H, W] probs (already softmax)
    mask: torch.Tensor,          # [B, H, W] bool
    *,
    n_iters: int = 5,
    eps: float = 1e-6,
    momentum: float = 0.0,       # 0 = replace by balanced; >0 = EMA mix with original
    k_active: int = None, # if None, choose automatically per mask
    min_k: int = 2,
    max_k: int = 10,    # if None, max_k = D
    pixels_per_proto: int = 64,  # auto K ≈ N_mask / pixels_per_proto
):
    """
    Rebalances prototype usage inside the mask.
    Works on probabilities (no logits). Output is still per-pixel probs.

    Returns:
      Q_out: [B, D, H, W]
    """
    assert Q.ndim == 4
    B, D, H, W = Q.shape
    assert mask.shape == (B, H, W) and mask.dtype == torch.bool

    N = H * W
    Qn = Q.permute(0, 2, 3, 1).reshape(B, N, D).clamp_min(eps)  # [B,N,D]
    M  = mask.reshape(B, N)                                      # [B,N]

    Q_out = Qn.clone()

    if max_k is None:
        max_k = D

    for b in range(B):
        mb = M[b]
        n_mask = int(mb.sum().item())
        if n_mask == 0:
            continue

        A0 = Qn[b, mb, :]  # [N_mask, D]
        A0 = A0 / (A0.sum(dim=1, keepdim=True) + eps)  # row-stochastic

        # choose K prototypes to balance inside this mask
        if k_active is None:
            K = max(min_k, min(max_k, n_mask // pixels_per_proto))
            K = min(K, D)
        else:
            K = max(min_k, min(max_k, k_active))
            K = min(K, D)

        if K < D:
            mass = A0.sum(dim=0)                      # [D]
            topk = torch.topk(mass, k=K, largest=True).indices
            A = A0[:, topk]                           # [N_mask, K]
        else:
            topk = None
            A = A0                                    # [N_mask, D]

        # Explicit column target mass (uniform over active prototypes)
        # Total mass per iteration is ~N_mask because rows sum to 1.
        target_col = A.sum() / A.shape[1]             # scalar

        for _ in range(n_iters):
            # row normalize (keep per-pixel distribution)
            A = A / (A.sum(dim=1, keepdim=True) + eps)

            # column scaling to hit uniform target
            col = A.sum(dim=0, keepdim=True) + eps    # [1,K] or [1,D]
            A = A * (target_col / col)

        # final row normalize for clean probs
        A = A / (A.sum(dim=1, keepdim=True) + eps)

        # put back into full D
        if topk is not None:
            A_full = torch.zeros_like(A0)
            A_full[:, topk] = A
            A_full = A_full / (A_full.sum(dim=1, keepdim=True) + eps)
        else:
            A_full = A

        if momentum > 0.0:
            A_full = momentum * A0 + (1.0 - momentum) * A_full
            A_full = A_full / (A_full.sum(dim=1, keepdim=True) + eps)

        Q_out[b, mb, :] = A_full

    return Q_out.view(B, H, W, D).permute(0, 3, 1, 2)


def create_proto_legend(colors, proto_ids=None, max_items=25):
    """
    colors: torch.Tensor (P,3) on CPU or GPU in [0,1]
    proto_ids: optional list of prototype indices to display
    """
    colors_cpu = colors.detach().cpu()
    P = colors_cpu.shape[0]

    if proto_ids is None:
        proto_ids = list(range(P))
    proto_ids = proto_ids[:max_items]

    fig_h = max(1.5, 0.28 * len(proto_ids))
    fig, ax = plt.subplots(figsize=(4.0, fig_h))

    for row, pid in enumerate(proto_ids):
        c = colors_cpu[pid].numpy()
        ax.add_patch(plt.Rectangle((0, row), 1, 1, color=c))
        ax.text(1.2, row + 0.5, f"Proto {pid}", va="center", fontsize=10)

    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, len(proto_ids))
    ax.invert_yaxis()
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def train_pipnet(net, train_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, epoch, nr_epochs, device, pretrain=False, finetune=False, global_step_base=0, progress_prefix: str = 'Train Epoch'):

    # Make sure the model is in train mode
    net.train()
    
    if pretrain:
        # Disable training of classification layer
        net.module._classification.requires_grad = False
        progress_prefix = 'Pretrain Epoch'
    else:
        # Enable training of classification layer (disabled in case of pretraining)
        net.module._classification.requires_grad = True
    
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.

    iters = len(train_loader)
    # Show progress on progress bar. 
    train_iter = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=progress_prefix+'%s'%epoch,
                    mininterval=2.,
                    ncols=0)
    
    count_param=0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count_param+=1           
    print("Number of parameters that require gradient: ", count_param, flush=True)

    if pretrain:
        align_pf_weight = (epoch/nr_epochs)*1.
        unif_weight = 0.5 #ignored
        t_weight = 5.
        cl_weight = 0.
    else:
        align_pf_weight = 5. 
        t_weight = 2.
        unif_weight = 0.
        cl_weight = 2.

    print("Align weight: ", align_pf_weight, ", U_tanh weight: ", t_weight, "Class weight:", cl_weight, flush=True)
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)
    
    lrs_net = []
    lrs_class = []
    global_step_offset = (epoch - 1) * len(train_loader)

    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, m2, xs1_ds, xs2_ds, m2_ds, hflip1, hflip2, ys) in train_iter:

        xs1, xs2, xs1_ds, xs2_ds, ys = xs1.to(device), xs2.to(device), xs1_ds.to(device), xs2_ds.to(device), ys.to(device)

        # Log example images occasionally
        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)

        # Forward pass
        proto_features, proto_features_ds, pooled, out = net(
            torch.cat([xs1, xs2]),
            torch.cat([xs1_ds, xs2_ds])
        )

        # ---- build visible mask for BIG scale (proto_features resolution) ----
        # proto_features: [2B, D, h, w]
        B2, D, h, w = proto_features.shape
        bs = xs1.shape[0]
        assert B2 == 2 * bs
        batch2, num_parts, grid_h, grid_w = proto_features.shape
        batch_size = xs1.shape[0]
        assert batch2 == 2 * batch_size

        def resize_mask_to_grid(mask_img, grid_h, grid_w, device):
            """
            mask_img: [B, H_img, W_img] or [B,1,H_img,W_img]
            returns:  [B, grid_h, grid_w] bool
            """
            if mask_img.dim() == 4 and mask_img.shape[1] == 1:
                mask_img = mask_img[:, 0]

            mask_img = mask_img.to(device=device, dtype=torch.float32)

            mask_grid = F.interpolate(
                mask_img.unsqueeze(1),  # [B,1,H,W]
                size=(grid_h, grid_w),
                mode="nearest"
            )[:, 0]  # [B,grid_h,grid_w]

            return mask_grid > 0.5

        batch2, D, grid_h, grid_w = proto_features.shape
        batch_size = xs1.shape[0]
        assert batch2 == 2 * batch_size

        # ---- BIG mask ----
        mask_view1_grid = torch.ones(batch_size, grid_h, grid_w, device=proto_features.device, dtype=torch.bool)
        mask_view2_grid = resize_mask_to_grid(m2, grid_h, grid_w, proto_features.device)  # [B,grid_h,grid_w]

        visible_mask_big = torch.cat([mask_view1_grid, mask_view2_grid], dim=0)  # [2B,grid_h,grid_w]

        # ---- SMALL/DS mask ----
        _, _, grid_h_ds, grid_w_ds = proto_features_ds.shape

        mask_view1_grid_ds = torch.ones(batch_size, grid_h_ds, grid_w_ds, device=proto_features.device,
                                        dtype=torch.bool)

        # Prefer m2_ds if it matches xs2_ds; otherwise you can reuse m2 and resize
        mask_view2_grid_ds = resize_mask_to_grid(m2_ds, grid_h_ds, grid_w_ds,
                                                 proto_features.device)  # [B,grid_h_ds,grid_w_ds]

        visible_mask_ds = torch.cat([mask_view1_grid_ds, mask_view2_grid_ds], dim=0)  # [2B,grid_h_ds,grid_w_ds]

        # ---- apply balancing on BOTH scales ----
        proto_features_bal = sinkhorn_balance_probs_in_mask(
            proto_features, visible_mask_big,
            n_iters=5,
            momentum=1.0,
        )

        proto_features_ds_bal = sinkhorn_balance_probs_in_mask(
            proto_features_ds, visible_mask_ds,
            n_iters=5,
            momentum=1.0,
        )

        # ---- recompute pooled/out so loss sees the balanced maps ----
        pooled_big = net.module._pool(proto_features_bal).flatten(1)  # [2B, D]
        pooled_ds = net.module._pool(proto_features_ds_bal).flatten(1)  # [2B, D]
        pooled = torch.cat([pooled_big, pooled_ds], dim=1)  # [2B, 2D]
        out = net.module._classification(pooled)
        log_every = 10
        if wandb.run is not None and (i % log_every == 0 or i == 0):
            bs = xs1.shape[0]
            max_images = min(2, bs)

            # proto_features assumed shape: (2*B, P, h, w) corresponding to cat([xs1, xs2])
            pf_xs2 = proto_features_bal[bs:]  # corresponds to xs2 (view 2)

            examples_original = []
            examples_overlay = []
            examples_mask_overlay = []

            num_prototypes = pf_xs2.shape[1]
            cmap = plt.get_cmap("hsv", num_prototypes)
            colors = torch.tensor([cmap(k)[:3] for k in range(num_prototypes)], dtype=torch.float32)
            for j in range(max_images):

                # ----- VIEW 2 IMAGE -----
                img = xs2[j].detach().cpu()
                img = torch.clamp(img, 0, 1)

                # ----- VIEW 2 MASK (already boolean + flipped correctly) -----
                mask = m2[j].detach().cpu()  # m1 is bm2 from dataset

                # Normalize mask shape to (H, W)
                if mask.dim() == 3:
                    mask = mask[0]  # (1,H,W) -> (H,W)

                mask = mask.float()  # already 0/1 from create_boolean_mask

                H_img, W_img = img.shape[-2], img.shape[-1]

                # Resize mask if needed (should normally already match)
                if mask.shape[-2:] != (H_img, W_img):
                    mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),
                        size=(H_img, W_img),
                        mode="nearest"
                    )[0, 0]

                # -----------------------------
                # Ground-truth mask visualization
                # -----------------------------

                mask3 = mask.unsqueeze(0).expand(3, -1, -1)  # (3,H,W)

                dim_factor = 0.25
                img_dimmed = img * (mask3 + (1 - mask3) * dim_factor)

                # red tint in masked region
                tint = torch.zeros_like(img)
                tint[0] = 1.0

                alpha = 0.35
                mask_overlay = img_dimmed * (1 - alpha * mask3) + tint * (alpha * mask3)
                mask_overlay = torch.clamp(mask_overlay, 0, 1)

                examples_mask_overlay.append(
                    wandb.Image(mask_overlay, caption=f"class: {ys[j].item()} (view2 GT mask)")
                )

                # -----------------------------
                # Prototype overlay
                # -----------------------------

                fmap = pf_xs2[j].detach().cpu()

                proto_idx = torch.argmax(fmap, dim=0)
                proto_conf = torch.max(fmap, dim=0).values

                proto_idx_up = F.interpolate(
                    proto_idx.unsqueeze(0).unsqueeze(0).float(),
                    size=(H_img, W_img),
                    mode="nearest"
                )[0, 0].long()

                proto_conf_up = F.interpolate(
                    proto_conf.unsqueeze(0).unsqueeze(0),
                    size=(H_img, W_img),
                    mode="bilinear",
                    align_corners=False
                )[0, 0].clamp(0, 1)

                colored = colors[proto_idx_up].permute(2, 0, 1).float()

                base_alpha = 0.15
                conf_alpha = 0.75
                alpha_map = (base_alpha + conf_alpha * proto_conf_up).clamp(0, 1)

                alpha_map = alpha_map * mask

                overlay = (
                        alpha_map.unsqueeze(0) * colored
                        + (1 - alpha_map).unsqueeze(0) * img_dimmed
                )

                overlay = torch.clamp(overlay, 0, 1)

                examples_original.append(
                    wandb.Image(img, caption=f"class: {ys[j].item()} (view2)")
                )
                examples_overlay.append(
                    wandb.Image(overlay, caption=f"class: {ys[j].item()} (view2 proto)")
                )
            # legend (only prototypes used in view2 examples)
            used = set()
            for j in range(max_images):
                fmap = pf_xs2[j].detach().cpu()
                used |= set(torch.unique(torch.argmax(fmap, dim=0)).tolist())
            used = sorted(list(used))

            legend_img = create_proto_legend(colors, proto_ids=used, max_items=25)

            global_step = global_step_base + (epoch - 1) * len(train_loader) + i
            phase = "pretrain" if pretrain else ("finetune" if finetune else "train")

            wandb.log(
                {
                    f"viz/original_{phase}": examples_original,
                    f"viz/prototype_overlay_{phase}": examples_overlay,
                    f"viz/mask_overlay_{phase}": examples_mask_overlay,
                    f"viz/prototype_legend_{phase}": wandb.Image(legend_img, caption="Legend: proto id → color"),
                },
                step=global_step,
            )

        def outside_soft_suppression(proto_features, visible_mask, power=2.0):
            """
            proto_features: [B, D, H, W]  (probabilities after softmax)
            visible_mask:   [B, H, W] bool (True = allowed region)
            power:          >1 increases focus on strong spikes
            """

            outside = (~visible_mask)  # [B,H,W] bool

            # Confidence per pixel = strongest prototype
            max_prob = proto_features.max(dim=1).values  # [B,H,W]

            # Keep only outside pixels
            max_prob_out = max_prob * outside.to(max_prob.dtype)

            # Normalize per image
            denom = outside.float().sum(dim=(1, 2)).clamp_min(1.0)
            per_img = (max_prob_out.pow(power).sum(dim=(1, 2)) / denom)

            return per_img.mean()

        loss, acc, loss_dict = calculate_loss(
            proto_features_bal, proto_features_ds_bal, pooled,
            hflip1, hflip2, out, ys,
            align_pf_weight, t_weight, unif_weight, cl_weight,
            net.module._classification.normalization_multiplier,
            pretrain, finetune, criterion, train_iter,
            print=True, EPS=1e-8
        )

        B = xs1.shape[0]  # batch size of one view

        # compute penalty on the SAME maps you train with
        pen_big = outside_soft_suppression(proto_features[B:], mask_view2_grid, power=2.0)
        pen_ds = outside_soft_suppression(proto_features_ds[B:], mask_view2_grid_ds, power=2.0)

        outside_pen = pen_big + pen_ds

        lambda_out = 1e-1  # start 1e-3..1e-2
        loss = loss + lambda_out * outside_pen
        # optional logging
        loss_dict = dict(loss_dict)  # ensures it's a plain mutable dict
        loss_dict[f"{phase}/loss_out_entropy"] = outside_pen.item()  # use item() for W&B safety
        loss_dict[f"{phase}/loss_total"] = loss.item()
        loss_dict["global_step"] = global_step_offset + i

        wandb.log(loss_dict)
        # Compute the gradient
        loss.backward()

        if not pretrain:
            optimizer_classifier.step()   
            scheduler_classifier.step(epoch - 1 + (i/iters))
            lrs_class.append(scheduler_classifier.get_last_lr()[0])
     
        if not finetune:
            optimizer_net.step()
            scheduler_net.step() 
            lrs_net.append(scheduler_net.get_last_lr()[0])
        else:
            lrs_net.append(0.)
            
        with torch.no_grad():
            total_acc+=acc
            total_loss+=loss.item()

        if not pretrain:
            with torch.no_grad():
                net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) #set weights in classification layer < 1e-3 to zero
                net.module._classification.normalization_multiplier.copy_(torch.clamp(net.module._classification.normalization_multiplier.data, min=1.0)) 
                if net.module._classification.bias is not None:
                    net.module._classification.bias.copy_(torch.clamp(net.module._classification.bias.data, min=0.))  
    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['loss'] = total_loss/float(i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class
    
    return train_info


def calculate_loss(proto_features, proto_features_ds, pooled, hflip, hflip_ds, out,
                   ys1, align_pf_weight, t_weight, unif_weight, cl_weight, net_normalization_multiplier, pretrain, finetune, criterion, train_iter, print=True, EPS=1e-10):
    ys = torch.cat([ys1,ys1])
    pooled1, pooled2 = pooled.chunk(2)
    pf1, pf2 = proto_features.chunk(2)
    N = hflip.shape[0]
    pf_parts = []
    for i in range(N):
        if hflip[i]:
            pf_parts.append(torch.flip(pf2[i], [2]).unsqueeze(0))
        else:
            pf_parts.append(pf2[i].unsqueeze(0))
    pf2 = torch.cat(pf_parts, dim=0)

    pf1_ds, pf2_ds = proto_features_ds.chunk(2)
    N = hflip_ds.shape[0]
    pf_parts_ds = []
    for i in range(N):
        if hflip_ds[i]:
            pf_parts_ds.append(torch.flip(pf2_ds[i], [2]).unsqueeze(0))
        else:
            pf_parts_ds.append(pf2_ds[i].unsqueeze(0))
    pf2_ds = torch.cat(pf_parts_ds, dim=0)

    embv2 = pf2.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    embv1 = pf1.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)

    embv2_ds = pf2_ds.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    embv1_ds = pf1_ds.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)

    # ---- compute losses ----
    a_loss_pf = (align_loss(embv1, embv2.detach()) + align_loss(embv2, embv1.detach())) / 2.
    a_loss_pf += (align_loss(embv1_ds, embv2_ds.detach()) + align_loss(embv2_ds, embv1_ds.detach())) / 2.

    tanh_loss = -(
            torch.log(torch.tanh(torch.sum(pooled1, dim=0)) + EPS).mean()
            + torch.log(torch.tanh(torch.sum(pooled2, dim=0)) + EPS).mean()
    ) / 2.

    class_loss = None  # <-- important

    # pretrain stage: only self-supervised losses
    if pretrain:
        loss = align_pf_weight * a_loss_pf + t_weight * tanh_loss

    # classification stage (optionally with/without finetune)
    else:
        softmax_inputs = torch.log1p(out ** net_normalization_multiplier)
        class_loss = criterion(F.log_softmax(softmax_inputs, dim=1), ys)

        if finetune:
            loss = cl_weight * class_loss
        else:
            loss = align_pf_weight * a_loss_pf + t_weight * tanh_loss + cl_weight * class_loss

    # ---- accuracy only when not pretrain ----
    acc = 0.0
    if not pretrain:
        ys_pred_max = torch.argmax(out, dim=1)
        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(ys))

    # ---- tqdm printing ----
    if print:
        with torch.no_grad():
            if pretrain:
                train_iter.set_postfix_str(
                    f"stage:pre L:{loss.item():.3f}, LA:{a_loss_pf.item():.2f}, "
                    f"LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled - 0.1), dim=1).float().mean().item():.1f}",
                    refresh=False
                )
            else:
                # finetune or joint, both have class_loss here
                train_iter.set_postfix_str(
                    f"stage:{'finetune' if finetune else 'joint'} "
                    f"L:{loss.item():.3f}, LC:{class_loss.item():.3f}, "
                    f"LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, "
                    f"num_scores>0.1:{torch.count_nonzero(torch.relu(pooled - 0.1), dim=1).float().mean().item():.1f}, "
                    f"Ac:{acc:.3f}",
                    refresh=False
                )

    # ---- wandb dict: log according to stage ----
    stage = "pretrain" if pretrain else ("finetune" if finetune else "joint")

    loss_dict = {
        "stage": stage,  # helps filtering in wandb
        f"{stage}/loss_total": float(loss.detach().item()),
        f"{stage}/loss_align_pf": float((align_pf_weight * a_loss_pf).detach().item()),
        f"{stage}/loss_tanh": float((t_weight * tanh_loss).detach().item()),
        "weights/align_pf": float(align_pf_weight),
        "weights/tanh": float(t_weight),
        "weights/class": float(cl_weight),
    }

    # only log class loss + acc when they exist (not pretrain)
    if class_loss is not None:
        loss_dict[f"{stage}/loss_class"] = float((cl_weight * class_loss).detach().item())
        loss_dict[f"{stage}/acc_step"] = float(acc)

    return loss, acc, loss_dict


# Extra uniform loss from https://www.tongzhouwang.info/hypersphere/. Currently not used but you could try adding it if you want. 
def uniform_loss(x, t=2):
    # print("sum elements: ", torch.sum(torch.pow(x,2), dim=1).shape, torch.sum(torch.pow(x,2), dim=1)) #--> should be ones
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + 1e-10).log()
    return loss

# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss