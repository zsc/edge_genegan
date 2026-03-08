from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvNormReLU


class RgbEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        z_channels: int = 256,
        a_channels: int = 256,
        base_channels: int = 64,
        norm: str = "instance",
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            ConvNormReLU(in_channels, base_channels, norm=norm, stride=2),
            ConvNormReLU(base_channels, base_channels * 2, norm=norm, stride=2),
            ConvNormReLU(base_channels * 2, base_channels * 4, norm=norm, stride=2),
            ConvNormReLU(base_channels * 4, base_channels * 8, norm=norm, stride=2),
        )
        self.z_head = nn.Conv2d(base_channels * 8, z_channels, kernel_size=1)
        self.a_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels * 8, a_channels),
        )

    def forward(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(frame)
        z = self.z_head(feat)
        a = self.a_head(feat)
        return z, a
