import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from features.dino_features import DinoV2Features
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

    def forward(self, xs, inference: bool = False):
        features = self._net(xs)

        # now add_on outputs [B, 2D, h, w]
        proto_features = self._add_on(features)
        proto_features = F.softmax(proto_features, dim=1)

        # pooled is already [B, 2D]
        pooled = self._pool(proto_features)

        if inference:
            pooled = torch.where(pooled < 0.1, 0.0, pooled)

        out = self._classification(pooled)
        return proto_features, None, pooled, out

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
    embed_sizes={"dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536}
    modelname = "dinov2_vitb14"

    # ---- backbone selection ----
    if args.net.startswith("dinov2_"):
        vit = torch.hub.load("facebookresearch/dinov2", args.net)
        pretrained = torch.load(args.vit_path, map_location=torch.device('cpu'))

        new_state_dict = {}
        for key, value in pretrained['teacher'].items():
            if 'dino_head' in key or "ibot_head" in key:
                continue
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value

        vit.pos_embed = torch.nn.Parameter(torch.zeros(1, 257, embed_sizes[modelname]))
        vit.load_state_dict(new_state_dict, strict=True)

        for p in vit.parameters():
            p.requires_grad = False

        N = 4
        for blk in vit.blocks[-N:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in vit.norm.parameters():
            p.requires_grad = True

        features = DinoV2Features(vit, which="x_norm_patchtokens")
        first_add_on_layer_in_channels = features.out_channels

    else:
        features = base_architecture_to_features[args.net](pretrained=not args.disable_pretrained)
        features_name = str(args.net).upper() if 'next' in args.net else str(features).upper()

        if features_name.startswith('RES') or features_name.startswith('CONVNEXT'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        else:
            raise Exception('other base architecture NOT implemented')

    # ---- prototypes: ALWAYS 2*D ----
    if args.num_features == 0:
        # D = backbone output channels
        D = first_add_on_layer_in_channels
        num_prototypes = 2 * D
        print("Number of prototypes:", num_prototypes, flush=True)

        # IMPORTANT: produce 2D prototype maps directly
        add_on_layers = nn.Conv2d(
            in_channels=D,
            out_channels=num_prototypes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    else:
        # D = args.num_features (your chosen bottleneck size)
        D = args.num_features
        num_prototypes = 2 * D
        print(
            "Number of prototypes set from", first_add_on_layer_in_channels,
            "to", num_prototypes, ". Extra 1x1 conv layer added.",
            flush=True
        )

        # backbone channels -> D -> 2D (so prototypes are 2D)
        add_on_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=first_add_on_layer_in_channels,
                out_channels=D,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.Conv2d(
                in_channels=D,
                out_channels=num_prototypes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
        )

    pool_layer = nn.Sequential(
        nn.AdaptiveMaxPool2d(output_size=(1,1)),
        nn.Flatten()
    )

    classification_layer = NonNegLinear(num_prototypes, num_classes, bias=bool(args.bias))
    return features, add_on_layers, pool_layer, classification_layer, num_prototypes
    