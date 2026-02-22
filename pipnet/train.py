import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import wandb
import io
from PIL import Image


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


def train_pipnet(net, train_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, epoch, nr_epochs, device, pretrain=False, finetune=False, progress_prefix: str = 'Train Epoch'):

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

        log_every = 10
        if wandb.run is not None and (i % log_every == 0 or i == 0):
            bs = xs1.shape[0]
            max_images = min(2, bs)

            pf_xs1 = proto_features[:bs]  # corresponds to xs1
            examples_original = []
            examples_overlay = []

            # If you have many prototypes, hsv gives more unique colors than tab20
            num_prototypes = pf_xs1.shape[1]  # proto_features shape assumed (B*2, P, H, W)
            cmap = plt.get_cmap("hsv", num_prototypes)
            colors = torch.tensor([cmap(k)[:3] for k in range(num_prototypes)], dtype=torch.float32)

            for j in range(max_images):
                img = xs1[j].detach().cpu()
                img = torch.clamp(img, 0, 1)

                fmap = pf_xs1[j].detach().cpu()  # (P, H, W) softmax probs

                # winner prototype per patch + confidence
                proto_idx = torch.argmax(fmap, dim=0)  # (H, W)
                proto_conf = torch.max(fmap, dim=0).values  # (H, W) in [0,1]

                # upsample both to image size
                H_img, W_img = img.shape[-2], img.shape[-1]
                proto_idx_up = F.interpolate(
                    proto_idx[None, None].float(), size=(H_img, W_img), mode="nearest"
                )[0, 0].long()

                proto_conf_up = F.interpolate(
                    proto_conf[None, None], size=(H_img, W_img), mode="bilinear", align_corners=False
                )[0, 0].clamp(0, 1)

                # colorize (H,W,3) -> (3,H,W)
                colored = colors[proto_idx_up]  # (H, W, 3)
                colored = colored.permute(2, 0, 1).float()  # (3, H, W)

                # confidence-weighted alpha (cleaner than constant alpha)
                # You can tune these:
                base_alpha = 0.15
                conf_alpha = 0.75
                alpha_map = (base_alpha + conf_alpha * proto_conf_up).clamp(0, 1)  # (H,W)
                alpha_map = alpha_map.unsqueeze(0)  # (1,H,W) for broadcasting

                overlay = alpha_map * colored + (1 - alpha_map) * img
                overlay = torch.clamp(overlay, 0, 1)

                # log
                examples_original.append(
                    wandb.Image(img, caption=f"class: {ys[j].item()}")
                )
                examples_overlay.append(
                    wandb.Image(overlay, caption=f"class: {ys[j].item()} (proto overlay)")
                )

            # (optional) show legend only for prototypes used in these images
            used = set()
            for j in range(max_images):
                fmap = pf_xs1[j].detach().cpu()
                used |= set(torch.unique(torch.argmax(fmap, dim=0)).tolist())
            used = sorted(list(used))

            legend_img = create_proto_legend(colors, proto_ids=used, max_items=25)

            global_step = (epoch - 1) * len(train_loader) + i
            phase = "pretrain" if pretrain else ("finetune" if finetune else "train")
            wandb.log(
                {
                    "viz/phase": phase,
                    "viz/original": examples_original,
                    "viz/prototype_overlay": examples_overlay,
                    "viz/prototype_legend": wandb.Image(legend_img, caption="Legend: proto id → color"),
                },
                step=global_step,
            )
        loss, acc, loss_dict = calculate_loss(proto_features, proto_features_ds, pooled, hflip1, hflip2, out, ys, align_pf_weight, t_weight, unif_weight, cl_weight,
                                   net.module._classification.normalization_multiplier, pretrain, finetune, criterion, train_iter, print=True, EPS=1e-8)
        global_step = global_step_offset + i
        loss_dict["global_step"] = global_step
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