from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class EdgeAdherenceExtractor(nn.Module):
    """Fixed edge extractor used for edge adherence loss."""

    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]])
        ky = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]])
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(frame.shape)}")
        if frame.size(1) == 3:
            gray = frame[:, 0:1, :, :] * 0.299 + frame[:, 1:2, :, :] * 0.587 + frame[:, 2:3, :, :] * 0.114
        elif frame.size(1) == 1:
            gray = frame
        else:
            raise ValueError(f"Expected 1 or 3 input channels, got {frame.size(1)}")
        gray = gray.clamp(-1.0, 1.0)
        gx = F.conv2d(gray, self.kx.expand(gray.size(1), -1, -1, -1), stride=1, padding=1, groups=gray.size(1))
        gy = F.conv2d(gray, self.ky.expand(gray.size(1), -1, -1, -1), stride=1, padding=1, groups=gray.size(1))
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        mag = mag / (mag.amax(dim=(-1, -2), keepdim=True) + 1e-6)
        return mag


def compute_edge_adherence_loss(
    frame_pred: torch.Tensor,
    edge_target: torch.Tensor,
    extractor: EdgeAdherenceExtractor,
    lambda_weight: float = 1.0,
) -> torch.Tensor:
    pred_edge = extractor(frame_pred)
    if pred_edge.shape != edge_target.shape:
        pred_edge = F.interpolate(pred_edge, size=edge_target.shape[-2:], mode="bilinear", align_corners=False)
    return F.l1_loss(pred_edge, edge_target) * float(lambda_weight)
