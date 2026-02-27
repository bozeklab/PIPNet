"""
pipnet_train_single_scale.py

Single-scale (no DS prototypes) training loop + losses + clustering regularization.

Assumptions / Notes:
- Your dataloader STILL yields: (xs1, xs2, m2, xs1_ds, xs2_ds, m2_ds, hflip1, hflip2, ys)
  We IGNORE xs*_ds and m2_ds completely (kept only to match the existing dataloader).
- Your model forward has been changed to single-scale, but you can keep the same signature:
    proto_features, proto_features_ds, pooled, out = net(xs, xs_ds)
  where proto_features_ds is None (or ignored).
- proto_features shape: [2B, D, h, w] corresponding to cat([xs1, xs2])
- pooled shape: [2B, D] (or [2B, 2D] if you kept a 2D classifier; see note in model forward).
  This file assumes pooled is [2B, D] (cleanest).
"""

import io
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from tqdm import tqdm


# -----------------------------
# Sinkhorn-like balancing in mask
# -----------------------------
def sinkhorn_balance_probs_in_mask(
    Q: torch.Tensor,             # [B, D, H, W] probs (already softmax)
    mask: torch.Tensor,          # [B, H, W] bool
    *,
    n_iters: int = 5,
    eps: float = 1e-6,
    momentum: float = 0.0,       # 0 = replace by balanced; >0 = EMA mix with original
    k_active: int = None,        # if None, choose automatically per mask
    min_k: int = 2,
    max_k: int = 10,             # if None, max_k = D
    pixels_per_proto: int = 16,  # NOTE: for ViT token grids, treat as TOKENS per proto
) -> torch.Tensor:
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
    M = mask.reshape(B, N)                                       # [B,N]

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
            mass = A0.sum(dim=0)  # [D]
            topk = torch.topk(mass, k=K, largest=True).indices
            A = A0[:, topk]       # [N_mask, K]
        else:
            topk = None
            A = A0                # [N_mask, D]

        # Column target mass (uniform over active prototypes)
        target_col = A.sum() / A.shape[1]  # scalar

        for _ in range(n_iters):
            A = A / (A.sum(dim=1, keepdim=True) + eps)            # row normalize
            col = A.sum(dim=0, keepdim=True) + eps                # [1,K]
            A = A * (target_col / col)                            # col scale

        A = A / (A.sum(dim=1, keepdim=True) + eps)                # final row norm

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


# -----------------------------
# Clustering loss via detached Sinkhorn target
# -----------------------------
def clustering_loss_from_sinkhorn_target(
    P: torch.Tensor,            # [B,D,H,W] probs (softmax)
    mask: torch.Tensor,         # [B,H,W] bool
    sinkhorn_fn,                # sinkhorn_balance_probs_in_mask
    *,
    n_iters: int = 5,
    eps: float = 1e-6,
    tau_pred: float = 1.0,      # optional sharpening on P (1.0 = none)
    lam: float = 1.0,
    **sinkhorn_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enforce clustering by matching network probs P to a balanced Sinkhorn target (detached).
    Applies loss only inside mask.

    Returns:
      loss_cluster, Q_tgt
    """
    assert P.ndim == 4
    B, D, H, W = P.shape
    assert mask.shape == (B, H, W) and mask.dtype == torch.bool

    # Optional sharpening/softening on P (still differentiable)
    if tau_pred != 1.0:
        logits = (P.clamp_min(eps)).log()
        P_use = F.softmax(logits / tau_pred, dim=1)
    else:
        P_use = P

    # Target assignment (no grad)
    with torch.no_grad():
        # prevent accidental double-passing of momentum
        sinkhorn_kwargs = {k: v for k, v in sinkhorn_kwargs.items() if k != "momentum"}
        Q_tgt = sinkhorn_fn(
            P_use, mask,
            n_iters=n_iters,
            momentum=0.0,   # IMPORTANT: real balanced target
            eps=eps,
            **sinkhorn_kwargs
        ).clamp_min(eps)

    # Cross-entropy with soft targets, only inside mask
    Pn = P_use.permute(0, 2, 3, 1).reshape(B, H * W, D)
    Qn = Q_tgt.permute(0, 2, 3, 1).reshape(B, H * W, D)
    Mn = mask.reshape(B, H * W).float()

    logP = (Pn.clamp_min(eps)).log()
    ce = -(Qn * logP).sum(dim=-1)  # [B, N]

    denom = Mn.sum(dim=1).clamp_min(1.0)
    per_img = (ce * Mn).sum(dim=1) / denom
    loss_cluster = per_img.mean()

    return lam * loss_cluster, Q_tgt


# -----------------------------
# Visualization helpers
# -----------------------------
def create_proto_legend(colors: torch.Tensor, proto_ids=None, max_items=25) -> Image.Image:
    """
    colors: torch.Tensor (P,3) in [0,1]
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


@torch.no_grad()
def overlay_topk_inside_outside_real(
    img_tensor: torch.Tensor,       # [3, H_img, W_img] in [0,1] or [0,255]
    proto_features: torch.Tensor,   # [D, H_tok, W_tok] probs
    visible_mask: torch.Tensor,     # [H_tok, W_tok] bool
    *,
    k: int = 3,
    alpha: float = 0.6
) -> np.ndarray:
    """
    Red  = top-k activation inside mask
    Blue = top-k activation outside mask
    Overlaid on real image.
    """
    topk_vals = torch.topk(proto_features, k=k, dim=0).values  # [k,H,W]
    conf = topk_vals.sum(dim=0)  # [H_tok,W_tok]

    inside = visible_mask
    outside = ~visible_mask

    conf_in = conf * inside
    conf_out = conf * outside

    max_val = conf.max().clamp_min(1e-6)
    conf_in = conf_in / max_val
    conf_out = conf_out / max_val

    conf_in_np = conf_in.cpu().numpy()
    conf_out_np = conf_out.cpu().numpy()

    H_img, W_img = img_tensor.shape[1:]
    conf_in_up = cv2.resize(conf_in_np, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
    conf_out_up = cv2.resize(conf_out_np, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    if img.max() <= 1.0:
        img = (img * 255.0)
    img = img.astype(np.uint8)

    heatmap = np.zeros_like(img, dtype=np.float32)
    heatmap[..., 0] = conf_in_up  # red
    heatmap[..., 2] = conf_out_up # blue
    heatmap = (heatmap * 255).astype(np.uint8)

    overlay = cv2.addWeighted(img, 1.0, heatmap, alpha, 0)
    return overlay


# -----------------------------
# Outside suppression loss (single scale)
# -----------------------------
def outside_soft_suppression(
    proto_features: torch.Tensor,   # [B,D,H,W] probs
    visible_mask: torch.Tensor,     # [B,H,W] bool (True = allowed region)
    *,
    power: float = 2.0,
    tau: float = 0.2,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Penalize confidence outside mask.
    Uses a smooth-max over prototypes.
    """
    outside = (~visible_mask).to(proto_features.dtype)  # [B,H,W]

    # smooth-max of probabilities: tau * logsumexp(log(p)/tau)
    conf = tau * torch.logsumexp((proto_features.clamp_min(eps)).log() / tau, dim=1)  # [B,H,W]
    conf_out = conf * outside

    denom = outside.sum(dim=(1, 2)).clamp_min(1.0)
    per_img = conf_out.pow(power).sum(dim=(1, 2)) / denom
    return per_img.mean()


# -----------------------------
# Alignment loss (unchanged)
# -----------------------------
def align_loss(inputs: torch.Tensor, targets: torch.Tensor, EPS=1e-12) -> torch.Tensor:
    assert inputs.shape == targets.shape
    assert targets.requires_grad is False
    loss = torch.einsum("nc,nc->n", [inputs, targets])
    return -torch.log(loss + EPS).mean()


# -----------------------------
# SINGLE-SCALE calculate_loss
# -----------------------------
def calculate_loss_single(
    proto_features: torch.Tensor,   # [2B,D,h,w]
    pooled: torch.Tensor,           # [2B,D]
    hflip_view2: torch.Tensor,      # [B] bool (flip applied to view2)
    out: torch.Tensor,              # [2B,num_classes]
    ys1: torch.Tensor,              # [B]
    *,
    align_pf_weight: float,
    t_weight: float,
    cl_weight: float,
    net_normalization_multiplier: torch.Tensor,
    pretrain: bool,
    finetune: bool,
    criterion,
    train_iter,
    print: bool = True,
    EPS: float = 1e-10
):
    ys = torch.cat([ys1, ys1], dim=0)
    pooled1, pooled2 = pooled.chunk(2)
    pf1, pf2 = proto_features.chunk(2)

    # align view2 proto maps with view1 using hflip flags
    N = hflip_view2.shape[0]
    pf2_parts = []
    for i in range(N):
        do_flip = bool(hflip_view2[i].item()) if torch.is_tensor(hflip_view2[i]) else bool(hflip_view2[i])
        pf2_parts.append(torch.flip(pf2[i], [2]).unsqueeze(0) if do_flip else pf2[i].unsqueeze(0))
    pf2 = torch.cat(pf2_parts, dim=0)

    embv1 = pf1.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
    embv2 = pf2.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)

    a_loss_pf = (align_loss(embv1, embv2.detach()) + align_loss(embv2, embv1.detach())) / 2.0

    tanh_loss = -(
        torch.log(torch.tanh(torch.sum(pooled1, dim=0)) + EPS).mean() +
        torch.log(torch.tanh(torch.sum(pooled2, dim=0)) + EPS).mean()
    ) / 2.0

    class_loss = None
    if pretrain:
        loss = align_pf_weight * a_loss_pf + t_weight * tanh_loss
    else:
        softmax_inputs = torch.log1p(out ** net_normalization_multiplier)
        class_loss = criterion(F.log_softmax(softmax_inputs, dim=1), ys)
        if finetune:
            loss = cl_weight * class_loss
        else:
            loss = align_pf_weight * a_loss_pf + t_weight * tanh_loss + cl_weight * class_loss

    acc = 0.0
    if not pretrain:
        ys_pred_max = torch.argmax(out, dim=1)
        acc = (ys_pred_max == ys).float().mean().item()

    if print:
        with torch.no_grad():
            if pretrain:
                train_iter.set_postfix_str(
                    f"stage:pre L:{loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}",
                    refresh=False
                )
            else:
                train_iter.set_postfix_str(
                    f"stage:{'finetune' if finetune else 'joint'} "
                    f"L:{loss.item():.3f}, LC:{class_loss.item():.3f}, "
                    f"LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, Ac:{acc:.3f}",
                    refresh=False
                )

    stage = "pretrain" if pretrain else ("finetune" if finetune else "joint")
    loss_dict = {
        "stage": stage,
        f"{stage}/loss_total": float(loss.detach().item()),
        f"{stage}/loss_align_pf": float((align_pf_weight * a_loss_pf).detach().item()),
        f"{stage}/loss_tanh": float((t_weight * tanh_loss).detach().item()),
        "weights/align_pf": float(align_pf_weight),
        "weights/tanh": float(t_weight),
        "weights/class": float(cl_weight),
    }
    if class_loss is not None:
        loss_dict[f"{stage}/loss_class"] = float((cl_weight * class_loss).detach().item())
        loss_dict[f"{stage}/acc_step"] = float(acc)

    return loss, acc, loss_dict


# -----------------------------
# Train loop (single scale, ignore DS inputs)
# -----------------------------
def train_pipnet(
    net,
    train_loader,
    optimizer_net,
    optimizer_classifier,
    scheduler_net,
    scheduler_classifier,
    criterion,
    epoch: int,
    nr_epochs: int,
    device,
    *,
    pretrain: bool = False,
    finetune: bool = False,
    global_step_base: int = 0,
    progress_prefix: str = "Train Epoch",
):
    net.train()

    if pretrain:
        net.module._classification.requires_grad = False
        progress_prefix = "Pretrain Epoch"
    else:
        net.module._classification.requires_grad = True

    train_info = dict()
    total_loss = 0.0
    total_acc = 0.0

    iters = len(train_loader)
    train_iter = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=progress_prefix + f"{epoch}",
        mininterval=2.0,
        ncols=0
    )

    count_param = sum(1 for _, p in net.named_parameters() if p.requires_grad)
    print("Number of parameters that require gradient:", count_param, flush=True)

    if pretrain:
        align_pf_weight = (epoch / nr_epochs) * 1.0
        t_weight = 5.0
        cl_weight = 0.0
    else:
        align_pf_weight = 5.0
        t_weight = 2.0
        cl_weight = 2.0

    print(
        f"Align weight: {align_pf_weight}, U_tanh weight: {t_weight}, Class weight: {cl_weight}",
        flush=True
    )
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)

    lrs_net = []
    lrs_class = []
    global_step_offset = (epoch - 1) * len(train_loader)

    def resize_mask_to_grid(mask_img, grid_h, grid_w, device_):
        """
        mask_img: [B,H,W] or [B,1,H,W]
        returns:  [B,grid_h,grid_w] bool
        """
        if mask_img.dim() == 4 and mask_img.shape[1] == 1:
            mask_img = mask_img[:, 0]
        mask_img = mask_img.to(device=device_, dtype=torch.float32)
        mask_grid = F.interpolate(mask_img.unsqueeze(1), size=(grid_h, grid_w), mode="nearest")[:, 0]
        return mask_grid > 0.5

    for i, (xs1, xs2, m2, xs1_ds, xs2_ds, m2_ds, hflip1, hflip2, ys) in train_iter:
        # Move only what we use
        xs1 = xs1.to(device)
        xs2 = xs2.to(device)
        ys = ys.to(device)

        # hflip2 might be CPU bool tensor; that's fine
        # m2 stays on CPU until resized, then moved inside resize_mask_to_grid

        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)

        # Forward (xs_ds still passed to satisfy signature; ignored by single-scale forward)
        proto_features, _, pooled, out = net(torch.cat([xs1, xs2]), torch.cat([xs1_ds, xs2_ds]))

        # Build view2 visible mask on proto grid
        B2, D, grid_h, grid_w = proto_features.shape
        bs = xs1.shape[0]
        assert B2 == 2 * bs

        mask_view2_grid = resize_mask_to_grid(m2, grid_h, grid_w, proto_features.device)  # [B,grid_h,grid_w]
        zeros_big = torch.zeros_like(mask_view2_grid)
        visible_mask_big_only = torch.cat([zeros_big, mask_view2_grid], dim=0)            # [2B,grid_h,grid_w]

        # Clustering loss (view2 only)
        loss_cl_big, _Q_big_tgt = clustering_loss_from_sinkhorn_target(
            proto_features,
            visible_mask_big_only,
            sinkhorn_balance_probs_in_mask,
            n_iters=5,
            lam=1.0,
            max_k=10,
            pixels_per_proto=16,
        )

        # Main SSL/class loss (single scale)
        loss, acc, loss_dict = calculate_loss_single(
            proto_features=proto_features,
            pooled=pooled,
            hflip_view2=hflip2,
            out=out,
            ys1=ys,
            align_pf_weight=align_pf_weight,
            t_weight=t_weight,
            cl_weight=cl_weight,
            net_normalization_multiplier=net.module._classification.normalization_multiplier,
            pretrain=pretrain,
            finetune=finetune,
            criterion=criterion,
            train_iter=train_iter,
            print=True,
            EPS=1e-8,
        )

        # Outside penalty (view2 only)
        pen_big = outside_soft_suppression(proto_features[bs:], mask_view2_grid, power=2.0)
        outside_pen = pen_big

        # Weights (tune carefully; start small and ramp if needed)
        lam_cluster = 0.1
        lambda_out = 1.0

        loss = loss + lambda_out * outside_pen + lam_cluster * loss_cl_big

        # Logging
        global_step = global_step_base + (epoch - 1) * len(train_loader) + i
        phase = "pretrain" if pretrain else ("finetune" if finetune else "train")

        loss_dict = dict(loss_dict)
        loss_dict[f"{phase}/outside_pen"] = float(outside_pen.detach())
        loss_dict[f"{phase}/cluster"] = float(loss_cl_big.detach())
        loss_dict[f"{phase}/loss_total_full"] = float(loss.detach())
        loss_dict["global_step"] = global_step_offset + i

        if wandb.run is not None:
            wandb.log(loss_dict, step=global_step)

        # Visualization occasionally (view2)
        log_every = 10
        if wandb.run is not None and (i % log_every == 0 or i == 0):
            max_images = min(2, bs)
            pf_xs2 = proto_features[bs:]  # view2 only

            num_prototypes = pf_xs2.shape[1]
            cmap = plt.get_cmap("hsv", num_prototypes)
            colors = torch.tensor([cmap(k)[:3] for k in range(num_prototypes)], dtype=torch.float32)

            examples_original = []
            examples_overlay = []
            examples_mask_overlay = []

            for j in range(max_images):
                img = xs2[j].detach().cpu().clamp(0, 1)

                # mask in image space for visualization
                mask_img = m2[j].detach().cpu()
                if mask_img.dim() == 3:
                    mask_img = mask_img[0]
                mask_img = mask_img.float()

                H_img, W_img = img.shape[-2], img.shape[-1]
                if mask_img.shape[-2:] != (H_img, W_img):
                    mask_img = F.interpolate(
                        mask_img.unsqueeze(0).unsqueeze(0),
                        size=(H_img, W_img),
                        mode="nearest"
                    )[0, 0]

                mask3 = mask_img.unsqueeze(0).expand(3, -1, -1)
                dim_factor = 0.25
                img_dimmed = img * (mask3 + (1 - mask3) * dim_factor)

                tint = torch.zeros_like(img)
                tint[0] = 1.0
                alpha = 0.35
                mask_overlay = (img_dimmed * (1 - alpha * mask3) + tint * (alpha * mask3)).clamp(0, 1)

                examples_mask_overlay.append(wandb.Image(mask_overlay, caption=f"class: {ys[j].item()} (view2 GT mask)"))

                # prototype overlay
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
                alpha_map = alpha_map * mask_img

                overlay = (
                    alpha_map.unsqueeze(0) * colored +
                    (1 - alpha_map).unsqueeze(0) * img_dimmed
                ).clamp(0, 1)

                examples_original.append(wandb.Image(img, caption=f"class: {ys[j].item()} (view2)"))
                examples_overlay.append(wandb.Image(overlay, caption=f"class: {ys[j].item()} (view2 proto)"))

            # legend
            used = set()
            for j in range(max_images):
                fmap = pf_xs2[j].detach().cpu()
                used |= set(torch.unique(torch.argmax(fmap, dim=0)).tolist())
            used = sorted(list(used))
            legend_img = create_proto_legend(colors, proto_ids=used, max_items=25)

            # topk inside/outside overlay for one example
            img0 = xs2[0].detach().cpu()
            proto0 = proto_features[bs + 0].detach().cpu()  # [D,h,w] on cpu ok
            mask0 = mask_view2_grid[0].detach().cpu()

            overlay_img = overlay_topk_inside_outside_real(
                img0, proto0, mask0, k=3, alpha=0.6
            )

            wandb.log(
                {
                    f"viz/original_{phase}": examples_original,
                    f"viz/prototype_overlay_{phase}": examples_overlay,
                    f"viz/mask_overlay_{phase}": examples_mask_overlay,
                    f"viz/prototype_legend_{phase}": wandb.Image(legend_img, caption="Legend: proto id → color"),
                    f"viz/topk_inside_outside_real_{phase}": wandb.Image(
                        overlay_img, caption="Red=inside, Blue=outside (top-k)"
                    ),
                },
                step=global_step
            )

        # Backprop + step
        loss.backward()

        if not pretrain:
            optimizer_classifier.step()
            scheduler_classifier.step(epoch - 1 + (i / iters))
            lrs_class.append(scheduler_classifier.get_last_lr()[0])

        if not finetune:
            optimizer_net.step()
            scheduler_net.step()
            lrs_net.append(scheduler_net.get_last_lr()[0])
        else:
            lrs_net.append(0.0)

        with torch.no_grad():
            total_acc += acc
            total_loss += float(loss.item())

        # clamp classifier params (kept from your original)
        if not pretrain:
            with torch.no_grad():
                net.module._classification.weight.copy_(
                    torch.clamp(net.module._classification.weight.data - 1e-3, min=0.0)
                )
                net.module._classification.normalization_multiplier.copy_(
                    torch.clamp(net.module._classification.normalization_multiplier.data, min=1.0)
                )
                if net.module._classification.bias is not None:
                    net.module._classification.bias.copy_(torch.clamp(net.module._classification.bias.data, min=0.0))

    train_info["train_accuracy"] = total_acc / float(i + 1)
    train_info["loss"] = total_loss / float(i + 1)
    train_info["lrs_net"] = lrs_net
    train_info["lrs_class"] = lrs_class
    return train_info