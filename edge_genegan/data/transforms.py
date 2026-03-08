from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_tensor(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(x.shape)}")
    return F.interpolate(x.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)


def normalize_rgb(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] != 3:
        raise ValueError(f"Expected 3-channel rgb, got {tuple(x.shape)}")
    return x.mul(2.0).sub(1.0)


def denormalize_rgb(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] != 3:
        raise ValueError(f"Expected 3-channel rgb, got {tuple(x.shape)}")
    return x.add(1.0).mul(0.5)


def normalize_edge(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(x.shape)}")
    return x.clamp(0.0, 1.0)
