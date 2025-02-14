import os

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import math

from pipnet.ccd_loss import CCD_loss
from pipnet.cka_loss import CKA_loss
from util.vis_pipnet import visualize_dist, build_image_grid, visualize_two_dists


def train_pipnet(net, train_loader, optimizer_net, scheduler_net, criterion, epoch, nr_epochs, class_mpc, device, pretrain=False, finetune=False, progress_prefix: str = 'Train Epoch', writer=None):

    # Make sure the model is in train mode
    net.train()
    
    if pretrain:
        # Disable training of classification layer
        progress_prefix = 'Pretrain Epoch'
    else:
        # Enable training of classification layer (disabled in case of pretraining)
        progress_prefix = 'Train Epoch'
    
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

    class_mpc_iter = tqdm(enumerate(train_loader),
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

    class_counts = torch.zeros(net.module._num_classes, device=device)

    if epoch == 1:
        with torch.no_grad():
            for _, (xs1, xs2, ys) in class_mpc_iter:
                xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)

                proto_features, pooled, out = net(xs1)
                argmax_indices = torch.argmax(proto_features, dim=1, keepdim=True)
                mask = torch.zeros_like(proto_features).scatter_(1, argmax_indices, 1)
                proto_features = proto_features * mask

                for c in range(net.module._num_classes):
                    #print('!!! c == ', c,  proto_features[ys == c, ...].shape, class_mpc[c, ...].shape, ys.shape)
                    #print(F.mse_loss(proto_features[ys == 0, ...].flatten(2).sum(0), proto_features[ys == 1, ...].flatten(2).sum(0)))
                    pf = proto_features[ys == c, ...].flatten(2).sum(0)
                    #pf_s, _ = torch.sort(pf, dim=1)
                    class_mpc[c, ...] += pf
                    #mse_loss = F.mse_loss(class_mpc[0], class_mpc[1])
    #
                    class_counts[c] += (ys == c).sum()
            class_mpc, _ = torch.sort(class_mpc, dim=2)
            class_mpc /= class_counts.view(-1, 1, 1)
            class_mpc *= 100.0

        for c1 in range(net.module._num_classes):
            for c2 in range(c1 + 1, net.module._num_classes):  # Avoid redundant calculations
                mse_loss = F.mse_loss(class_mpc[c1], class_mpc[c2])
                print(f"MSE Loss between class {c1} and class {c2}: {mse_loss.item():.6f}")

        dist_ps = []
        for p in range(32):
            dist_ps.append(visualize_two_dists(class_mpc[0, p, :].detach().cpu(), class_mpc[1, p, :].detach().cpu()))
        grid_image = build_image_grid(dist_ps)
        save_path = os.path.join('/data/pwojcik/PIPNet/', f"class_grid.png")
        grid_image.save(save_path)
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, ys) in train_iter:       
        
        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)
       
        # Reset the gradients
        #optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)
       
        # Perform a forward pass through the network
        proto_features, pooled, out = net(torch.cat([xs1, xs2]))
        #print('!!! ', proto_features.shape)
        loss, acc = calculate_loss(proto_features, pooled, out, ys, align_pf_weight, t_weight, unif_weight,  cl_weight, class_mpc, pretrain, finetune, criterion, train_iter, print_db=True, EPS=1e-8)
        
        # Compute the gradient
        print('backward')
        loss.backward()
        #optimizer_net.step()
        #if not pretrain:
            #optimizer_classifier.step()
            #scheduler_classifier.step(epoch - 1 + (i/iters))
            #lrs_class.append(scheduler_classifier.get_last_lr()[0])
     
        if not finetune:
            print('not finetune back')
            optimizer_net.step()
            scheduler_net.step() 
            lrs_net.append(scheduler_net.get_last_lr()[0])
        else:
            print('ft back')
            lrs_net.append(0.)
            
        with torch.no_grad():
            total_acc+=acc
            total_loss+=loss.item()

        # if not pretrain:
        #     with torch.no_grad():
        #         net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) #set weights in classification layer < 1e-3 to zero
        #         net.module._classification.normalization_multiplier.copy_(torch.clamp(net.module._classification.normalization_multiplier.data, min=1.0))
        #         if net.module._classification.bias is not None:
        #             net.module._classification.bias.copy_(torch.clamp(net.module._classification.bias.data, min=0.))
    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['loss'] = total_loss/float(i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class

    class_mpc_iter = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=progress_prefix+'%s'%epoch,
                    mininterval=2.,
                    ncols=0)

    with torch.no_grad():
        for _, (xs1, xs2, ys) in class_mpc_iter:
            xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)

            proto_features, pooled, out = net(xs1)
            argmax_indices = torch.argmax(proto_features, dim=1, keepdim=True)
            mask = torch.zeros_like(proto_features).scatter_(1, argmax_indices, 1)
            proto_features = proto_features * mask

            for c in range(net.module._num_classes):
                # print('!!! c == ', c,  proto_features[ys == c, ...].shape, class_mpc[c, ...].shape, ys.shape)
                # print(F.mse_loss(proto_features[ys == 0, ...].flatten(2).sum(0), proto_features[ys == 1, ...].flatten(2).sum(0)))
                pf = proto_features[ys == c, ...].flatten(2).sum(0)
                pf_s, _ = torch.sort(pf, dim=1)
                class_mpc[c, ...] += pf_s
                #mse_loss = F.mse_loss(class_mpc[0], class_mpc[1])
                #
                class_counts[c] += (ys == c).sum()
        class_mpc /= class_counts.view(-1, 1, 1)
        class_mpc *= 100.0

    return train_info


def calculate_loss(proto_features, pooled, out, ys1, align_pf_weight, t_weight, unif_weight, cl_weight, class_mpc, pretrain, finetune, criterion, train_iter, print_db=True, EPS=1e-10, writer=None):
    ys = torch.cat([ys1,ys1])
    pooled1, pooled2 = pooled.chunk(2)
    pf1, pf2 = proto_features.chunk(2)

    embv2 = pf2.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    embv1 = pf1.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    
    a_loss_pf = (align_loss(embv1, embv2.detach())+ align_loss(embv2, embv1.detach()))/2.
    #tanh_loss = -(torch.log(torch.tanh(torch.sum(pooled1,dim=0))+EPS).mean() + torch.log(torch.tanh(torch.sum(pooled2,dim=0))+EPS).mean())/2.

    cka = CKA_loss(concept_cha=1)
    ccd = CCD_loss()
    ck_loss = cka.forward(feature_map=pf1) + cka.forward(feature_map=pf2)
    print('CKA: ', ck_loss.item())
    #print('!!! ', (align_pf_weight*a_loss_pf).item(), (0.1 * ck_loss).item())

    if not finetune:
        loss = align_pf_weight*a_loss_pf
        #loss += t_weight * t_weight * tanh_loss
        loss += ck_loss
        #print(ck_loss, tanh_loss, t_weight * tanh_loss)
    if not pretrain:
        #softmax_inputs = torch.log1p(out**net_normalization_multiplier)
        #loss += ck_loss
        class_loss = ccd(pf1, ys1, class_mpc)
        loss += class_loss

        #if finetune:
        #    loss= cl_weight * class_loss
        #else:
        #    loss+= cl_weight * class_loss
    # Our tanh-loss optimizes for uniformity and was sufficient for our experiments. However, if pretraining of the prototypes is not working well for your dataset, you may try to add another uniformity loss from https://www.tongzhouwang.info/hypersphere/ Just uncomment the following three lines
    # else:
    #     uni_loss = (uniform_loss(F.normalize(pooled1+EPS,dim=1)) + uniform_loss(F.normalize(pooled2+EPS,dim=1)))/2.
    #     loss += unif_weight * uni_loss

    acc=0.
    if not pretrain:
        #ys_pred_max = torch.argmax(out, dim=1)
        #correct = torch.sum(torch.eq(ys_pred_max, ys))
        #acc = correct.item() / float(len(ys))
        acc = 0.0
    if print_db:
        with torch.no_grad():
            if pretrain:
                train_iter.set_postfix_str(
                f'L: {loss.item():.3f}, LA:{a_loss_pf.item():.2f}, CKA:{ck_loss.item():.5f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}',refresh=False)
            else:
                if finetune:
                    train_iter.set_postfix_str(
                    f'L:{loss.item():.3f},LC:{class_loss.item():.3f}, CKA:{ck_loss.item():.5f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac:{acc:.3f}',refresh=False)
                else:
                    train_iter.set_postfix_str(
                    f'L:{loss.item():.3f},CCD:{class_loss.item():.3f}, CKA:{ck_loss.item():.5f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}',refresh=False)
    return loss, acc

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