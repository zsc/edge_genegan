from __future__ import annotations

import torch

from .metrics import compute_edge_l1, compute_psnr


def evaluate_rollout(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_edges: torch.Tensor | None = None,
    target_edges: torch.Tensor | None = None,
) -> dict[str, float]:
    psnr = compute_psnr((pred + 1.0) / 2.0, (target + 1.0) / 2.0)
    result = {
        "psnr": float(psnr.mean().detach().cpu().item()),
        "psnr_min": float(psnr.min().detach().cpu().item()),
        "psnr_max": float(psnr.max().detach().cpu().item()),
    }
    if pred_edges is not None and target_edges is not None:
        edge_l1 = compute_edge_l1(pred_edges, target_edges).mean()
        result["edge_l1"] = float(edge_l1.detach().cpu().item())
    return result
