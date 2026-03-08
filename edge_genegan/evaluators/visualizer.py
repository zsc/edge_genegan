from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


def _to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    if t.dim() != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(t.shape)}")
    if t.size(0) == 1:
        t = t.repeat(3, 1, 1)
    if t.min() < 0.0:
        x = t.detach().float().clamp(-1.0, 1.0)
        x = ((x + 1.0) * 127.5).round().clamp(0, 255)
    else:
        x = t.detach().float().clamp(0.0, 1.0) * 255.0
    return x.permute(1, 2, 0).to(torch.uint8).cpu().numpy()


def save_tensors_as_grid(
    tensors: list[torch.Tensor],
    path: str | Path,
    max_per_row: int = 4,
) -> None:
    if len(tensors) == 0:
        raise ValueError("No tensors to save")
    if not all(t.ndim == 3 for t in tensors):
        raise ValueError("All tensors must be [C,H,W]")
    h = max(t.shape[1] for t in tensors)
    w = max(t.shape[2] for t in tensors)
    rows = []
    idx = 0
    while idx < len(tensors):
        row_tensors = tensors[idx : idx + max_per_row]
        row_imgs = []
        for t in row_tensors:
            canvas = torch.zeros((t.shape[0], h, w), dtype=t.dtype, device=t.device)
            canvas[:, : t.shape[1], : t.shape[2]] = t
            row_imgs.append(_to_uint8_rgb(canvas))
        idx += max_per_row
        hpad = 4
        for _ in range(max_per_row - len(row_tensors)):
            row_imgs.append(np.zeros((h, w, 3), dtype=np.uint8))
        rows.append(np.concatenate(row_imgs, axis=1))
    canvas = np.concatenate(rows, axis=0)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if imageio is None:
        if canvas.ndim == 3:
            Image.fromarray(canvas).save(str(p))
        else:
            raise ValueError(f"Unsupported canvas shape for image save: {canvas.shape}")
    else:
        imageio.imwrite(p, canvas)


def save_pair_visualization(outputs: dict[str, torch.Tensor], path: str | Path) -> None:
    tensors = [
        outputs["frame_t"][0],
        outputs["frame_s"][0],
        outputs["edge_t"][0],
        outputs["edge_s"][0],
        outputs["frame_rec_t"][0],
        outputs["frame_rec_s"][0],
        outputs["frame_swap_t"][0],
        outputs["frame_swap_s"][0],
        outputs["edge_rec_t"][0],
        outputs["edge_rec_s"][0],
        outputs["edge_swap_t"][0],
        outputs["edge_swap_s"][0],
    ]
    save_tensors_as_grid(tensors, path, max_per_row=4)


def save_rollout_gif(frames: torch.Tensor, out_path: str | Path, fps: int = 8) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    imgs = [_to_uint8_rgb(f) for f in frames]
    if imageio is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w, _ = imgs[0].shape
        writer = cv2.VideoWriter(str(p), fourcc, float(fps), (w, h))
        for img in imgs:
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        writer.release()
    else:
        imageio.mimsave(str(p), imgs, fps=fps)
