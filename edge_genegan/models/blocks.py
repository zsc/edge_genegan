from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _norm_cls(name: str):
    n = name.lower()
    if n == "batch":
        return nn.BatchNorm2d
    if n == "instance":
        return nn.InstanceNorm2d
    if n == "group":
        return lambda c: nn.GroupNorm(32, c)
    if n == "none":
        return None
    raise ValueError(f"Unsupported norm: {name}")


class ConvNormReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "batch", kernel: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
        norm_cls = _norm_cls(norm)
        self.norm = norm_cls(out_ch) if norm_cls is not None else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class AdaINBlock(nn.Module):
    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        self.style_to_gamma = nn.Linear(style_dim, channels)
        self.style_to_beta = nn.Linear(style_dim, channels)
        nn.init.zeros_(self.style_to_gamma.weight)
        nn.init.zeros_(self.style_to_gamma.bias)
        nn.init.zeros_(self.style_to_beta.weight)
        nn.init.zeros_(self.style_to_beta.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        std = torch.sqrt(var + 1e-6)
        x = (x - mean) / std

        gamma = self.style_to_gamma(style).view(B, C, 1, 1)
        beta = self.style_to_beta(style).view(B, C, 1, 1)
        return x * (1.0 + gamma) + beta
