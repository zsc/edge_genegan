from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_cycle_losses(
    cycle_outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    frame_cyc_s = F.l1_loss(cycle_outputs["frame_swap_t"], batch["frame_s"])
    frame_cyc_t = F.l1_loss(cycle_outputs["frame_swap_s"], batch["frame_t"])
    edge_cyc_t = F.l1_loss(cycle_outputs["edge_swap_t"], batch["edge_t"])
    edge_cyc_s = F.l1_loss(cycle_outputs["edge_swap_s"], batch["edge_s"])
    return frame_cyc_s + frame_cyc_t + edge_cyc_t + edge_cyc_s
