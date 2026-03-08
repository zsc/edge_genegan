from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvNormReLU


class PatchDiscriminator(nn.Module):
    """Simple PatchGAN-like discriminator for RGB frames."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_layers: int = 3, norm: str = "instance") -> None:
        super().__init__()
        channels = base_channels
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(max(1, int(num_layers))):
            next_channels = min(channels * 2, 512)
            layers.append(ConvNormReLU(channels, next_channels, norm=norm, kernel=4, stride=2))
            channels = next_channels
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")
        return self.head(self.backbone(x))
