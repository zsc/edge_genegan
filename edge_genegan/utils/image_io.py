"""Image io and normalization helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def load_image_rgb(path: str | Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    arr = np.array(image, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1) / 255.0


def load_image_gray(path: str | Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    arr = np.array(image, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0) / 255.0


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float()
    if x.dim() == 4:
        x = x[0]
    if x.min() < 0.0:
        x = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
    else:
        x = x.clamp(0.0, 1.0)
    if x.shape[0] == 1:
        return (x[0].mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy())
    return (x.mul(255.0).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())


def write_image(path: str | Path, t: torch.Tensor, scale_minus_one_to_one: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    x = t.detach().float()
    if x.dim() == 4:
        x = x[0]
    if scale_minus_one_to_one:
        x = (x + 1.0) * 0.5
    if x.dim() == 3 and x.shape[0] == 1:
        out = (x[0].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
        cv2.imwrite(str(p), out)
        return
    out = (x.clamp(0.0, 1.0).permute(1, 2, 0).mul(255.0).round().to(torch.uint8).cpu().numpy())
    Image.fromarray(out).save(str(p))


def ensure_channels(image: torch.Tensor, target_ch: int) -> torch.Tensor:
    if image.dim() != 3:
        raise ValueError("Expected [C,H,W]")
    if image.shape[0] == target_ch:
        return image
    if target_ch == 1:
        if image.shape[0] == 3:
            gray = image.mean(dim=0, keepdim=True)
            return gray
        return image[:1]
    raise ValueError(f"Unsupported target channels: {target_ch}")
