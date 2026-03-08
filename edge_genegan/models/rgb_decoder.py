from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ConvNormReLU, AdaINBlock


class RgbDecoder(nn.Module):
    def __init__(
        self,
        z_channels: int = 256,
        a_channels: int = 256,
        base_channels: int = 64,
        norm: str = "instance",
        inject: str = "adain",
    ) -> None:
        super().__init__()
        self.inject = inject
        c = base_channels * 8
        self.project = ConvNormReLU(z_channels, c, norm=norm)
        self.adain1 = AdaINBlock(c // 1, a_channels)
        self.block1 = ConvNormReLU(c, c // 2, norm=norm)
        self.adain2 = AdaINBlock(c // 2, a_channels)
        self.block2 = ConvNormReLU(c // 2, c // 4, norm=norm)
        self.adain3 = AdaINBlock(c // 4, a_channels)
        self.block3 = ConvNormReLU(c // 4, c // 8, norm=norm)
        self.adain4 = AdaINBlock(c // 8, a_channels)
        self.block4 = ConvNormReLU(c // 8, c // 16, norm=norm)
        self.out = nn.Conv2d(c // 16, 3, kernel_size=1)

    def _apply_inject(self, x: torch.Tensor, a: torch.Tensor, block: AdaINBlock, channel: int) -> torch.Tensor:
        if self.inject == "adain":
            x = block(x, a)
        return x

    def forward(
        self,
        z: torch.Tensor,
        appearance: torch.Tensor,
        prev_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.project(z)
        x = self._apply_inject(x, appearance, self.adain1, x.shape[1])
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block1(x)

        x = self._apply_inject(x, appearance, self.adain2, x.shape[1])
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block2(x)

        x = self._apply_inject(x, appearance, self.adain3, x.shape[1])
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block3(x)

        x = self._apply_inject(x, appearance, self.adain4, x.shape[1])
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block4(x)
        x = torch.tanh(self.out(x))
        return x, None
