from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(a, b)


def compute_reconstruction_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    rec_t = l1_loss(outputs["frame_rec_t"], batch["frame_t"])
    rec_s = l1_loss(outputs["frame_rec_s"], batch["frame_s"])
    e_rec_t = l1_loss(outputs["edge_rec_t"], batch["edge_t"])
    e_rec_s = l1_loss(outputs["edge_rec_s"], batch["edge_s"])
    return {
        "L_rec_rgb_t": rec_t,
        "L_rec_rgb_s": rec_s,
        "L_rec_edge_t": e_rec_t,
        "L_rec_edge_s": e_rec_s,
        "L_rec_rgb": rec_t + rec_s,
        "L_rec_edge": e_rec_t + e_rec_s,
        "L_rec": rec_t + rec_s + e_rec_t + e_rec_s,
    }


def compute_swap_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    swap_t = l1_loss(outputs["frame_swap_t"], batch["frame_t"])
    swap_s = l1_loss(outputs["frame_swap_s"], batch["frame_s"])
    e_swap_t = l1_loss(outputs["edge_swap_t"], batch["edge_t"])
    e_swap_s = l1_loss(outputs["edge_swap_s"], batch["edge_s"])
    return {
        "L_swap_rgb_t": swap_t,
        "L_swap_rgb_s": swap_s,
        "L_swap_edge_t": e_swap_t,
        "L_swap_edge_s": e_swap_s,
        "L_swap_rgb": swap_t + swap_s,
        "L_swap_edge": e_swap_t + e_swap_s,
        "L_swap": swap_t + swap_s + e_swap_t + e_swap_s,
    }


def compute_shared_structure_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return F.l1_loss(outputs["z_e_t"], outputs["z_f_t"]) + F.l1_loss(outputs["z_e_s"], outputs["z_f_s"])


def compute_nulling_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    eps_t = outputs["eps_t"]
    eps_s = outputs["eps_s"]
    return eps_t.abs().mean() + eps_s.abs().mean()
