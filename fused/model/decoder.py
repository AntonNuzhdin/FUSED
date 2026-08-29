"""Multi-scale aggregation of the SparseViT stage features.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusion(nn.Module):
    def __init__(self,
                 init_value=1e-6,
                 embed_dim=512,
                 predict_channels=1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        for i in range(1, 7):
            self.register_parameter(
                f"gamma{i}", nn.Parameter(init_value * torch.ones(embed_dim))
            )
        self.norm = norm_layer(embed_dim)
        for i in range(1, 5):
            setattr(self, f"conv_layer{i}", nn.Conv2d(320, embed_dim, kernel_size=1))
        self.conv_last = nn.Conv2d(embed_dim, predict_channels, kernel_size=1)

    def fuse_features(self, features):
        """Return the fused forensic map (B, C, H, W), before ``conv_last``."""
        c1, c2, c3, c4, c5, c6 = features

        c1 = self.conv_layer1(c1)
        c2 = self.conv_layer2(c2)
        c3 = self.conv_layer3(c3)
        c4 = self.conv_layer4(c4)
        b, c, h, w = c1.shape
        c5 = F.interpolate(c5, size=(h, w), mode='bilinear', align_corners=False)
        c6 = F.interpolate(c6, size=(h, w), mode='bilinear', align_corners=False)

        stages = [c.flatten(2).transpose(1, 2) for c in (c1, c2, c3, c4, c5, c6)]
        gammas = [self.gamma1, self.gamma2, self.gamma3,
                  self.gamma4, self.gamma5, self.gamma6]
        x = sum(g * s for g, s in zip(gammas, stages))
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

    def forward(self, features):
        return self.conv_last(self.fuse_features(features))
