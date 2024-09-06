import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from features.convnext_features import convnext_tiny_26_features, convnext_tiny_13_features 
import torch
from torch import Tensor

class PIPNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_prototypes: int,
                 feature_net: nn.Module,
                 args: argparse.Namespace,
                 add_on_layers: nn.Module,
                 pool_layer: nn.Module,
                 classification_layer: nn.Module
                 ):
        super().__init__()
        assert num_classes > 0
        self._num_features = args.num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._pool = pool_layer
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier

    def forward(self, xs, xs_ds, inference=False):
        features = self._net(xs)
        features_ds = self._net(xs_ds)
        proto_features = self._add_on(features)
        proto_features_ds = self._add_on(features_ds)
        B, D, H, W = proto_features.shape
        _, _, H_ds, W_ds = proto_features_ds.shape
        proto_features_ds_ups = proto_features_ds(2, dim=1).repeat_interleave(2, dim=2)
        p_f = proto_features.view(B, D, -1).permute(0, 2, 1) #B, HW, D/2
        p_f_ds = proto_features_ds_ups.view(B, D, -1).permute(0, 2, 1) #B, H'W', D/2
        #p_f_ds_ups = p_f_ds.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2) #B, HW, D/2
        combined = torch.cat([p_f, p_f_ds], dim=0) #B, HW, D
        combined = combined.view(B, H, W, D).permite(0, 3, 1, 2)
        #softmax_combined = F.softmax(combined, dim=1)

        #p_f = softmax_combined[:p_f.size(0)]
        #p_f_ds_flat = softmax_combined[p_f.size(0):]
        #proto_features = p_f.view(B, H*W, D).permute(0, 2, 1).view(B, D, H, W)
        #proto_features_ds = p_f_ds_flat.view(B, H_ds*W_ds, D).permute(0, 2, 1).view(B, D, H_ds, W_ds)

        pooled = self._pool(combined)
        #pooled_ds = self._pool(proto_features_ds)
        #pooled = torch.cat([pooled, pooled_ds], dim=1)
        if inference:
            clamped_pooled = torch.where(pooled < 0.1, 0., pooled)  #during inference, ignore all prototypes that have 0.1 similarity or lower
            out = self._classification(clamped_pooled) #shape (bs*2, num_classes)
            return proto_features, proto_features_ds, clamped_pooled, out
        else:
            out = self._classification(pooled) #shape (bs*2, num_classes) 
            return proto_features, proto_features_ds, pooled, out


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 #'resnet50_inat': resnet50_features_inat,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'convnext_tiny_26': convnext_tiny_26_features,
                                 'convnext_tiny_13': convnext_tiny_13_features}

# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.normalization_multiplier = nn.Parameter(torch.ones((1,),requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input,torch.relu(self.weight), self.bias)


def get_network(num_classes: int, args: argparse.Namespace): 
    features = base_architecture_to_features[args.net](pretrained=not args.disable_pretrained)
    features_name = str(features).upper()
    if 'next' in args.net:
        features_name = str(args.net).upper()
    if features_name.startswith('RES') or features_name.startswith('CONVNEXT'):
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
    else:
        raise Exception('other base architecture NOT implemented')
    
    if args.num_features == 0:
        num_prototypes = 2 * first_add_on_layer_in_channels
        print("Number of prototypes: ", num_prototypes, flush=True)
        add_on_layers = nn.Identity() #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1jghlf
    else:
        num_prototypes = 2 * args.num_features
        print("Number of prototypes set from", first_add_on_layer_in_channels, "to", num_prototypes,". Extra 1x1 conv layer added. Not recommended.", flush=True)
        add_on_layers = nn.Sequential(
            nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=args.num_features, kernel_size=1, stride = 1, padding=0, bias=True)#,
            #nn.Softmax(dim=1), #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1
    )
    pool_layer = nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size=(1,1)), #outputs (bs, ps,1,1)
                nn.Flatten() #outputs (bs, ps)
                ) 
    
    if args.bias:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=True)
    else:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=False)
        
    return features, add_on_layers, pool_layer, classification_layer, num_prototypes


    