from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ConvNormReLU


class EdgeDecoder(nn.Module):
    def __init__(self, z_channels: int = 256, base_channels: int = 64, norm: str = "instance"):
        super().__init__()
        c = base_channels * 8
        self.project = ConvNormReLU(z_channels, c, norm=norm)
        self.up1 = ConvNormReLU(c, c // 2, norm=norm)
        self.up2 = ConvNormReLU(c // 2, c // 4, norm=norm)
        self.up3 = ConvNormReLU(c // 4, c // 8, norm=norm)
        self.up4 = ConvNormReLU(c // 8, c // 16, norm=norm)
        self.out = nn.Conv2d(c // 16, 1, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up4(x)
        return torch.sigmoid(self.out(x))
