from __future__ import annotations

import torch


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_value: float = 1.0) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    psnr = 20.0 * torch.log10(torch.tensor(max_value, device=pred.device) + 1e-8) - 10.0 * torch.log10(mse + 1e-8)
    return psnr


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    x = pred
    y = target
    mux = x.mean(dim=[2, 3], keepdim=True)
    muy = y.mean(dim=[2, 3], keepdim=True)
    varx = ((x - mux) ** 2).mean(dim=[2, 3], keepdim=True)
    vary = ((y - muy) ** 2).mean(dim=[2, 3], keepdim=True)
    covxy = ((x - mux) * (y - muy)).mean(dim=[2, 3], keepdim=True)
    ssim = ((2 * mux * muy + c1) * (2 * covxy + c2)) / ((mux ** 2 + muy ** 2 + c1) * (varx + vary + c2))
    return ssim.mean(dim=[1, 2, 3])


def compute_edge_l1(pred_edge: torch.Tensor, target_edge: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred_edge - target_edge), dim=[1, 2, 3])


def edge_precision_recall_f1(
    pred_edge: torch.Tensor,
    target_edge: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = pred_edge > threshold
    t = target_edge > threshold
    tp = (p & t).sum(dim=[1, 2, 3]).float()
    fp = (p & ~t).sum(dim=[1, 2, 3]).float()
    fn = (~p & t).sum(dim=[1, 2, 3]).float()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return precision, recall, f1
