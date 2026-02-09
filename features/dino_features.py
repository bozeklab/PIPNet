import torch
import torch.nn as nn
import math

class DinoV2Features(nn.Module):
    """
    Makes DINOv2 ViT behave like a CNN backbone: returns [B, C, H, W].
    """
    def __init__(self, vit: nn.Module, which: str = "x_norm_patchtokens"):
        super().__init__()
        self.vit = vit
        self.which = which
        self.out_channels = getattr(vit, "embed_dim", vit.pos_embed.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vit.forward_features(x)
        patch_tokens = feats[self.which]           # [B, N, C]
        B, N, C = patch_tokens.shape
        side = int(math.sqrt(N))
        if side * side != N:
            raise ValueError(f"DINOv2 patch token count {N} is not square; cannot reshape.")
        fmap = patch_tokens.transpose(1, 2).contiguous().view(B, C, side, side)  # [B,C,H,W]
        return fmap
