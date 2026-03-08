from __future__ import annotations

import torch

from edge_genegan.models import EdgeRgbSwapSystem


def test_forward_pair_smoke() -> None:
    model = EdgeRgbSwapSystem()
    frame_t = torch.randn(1, 3, 256, 256)
    frame_s = torch.randn(1, 3, 256, 256)
    edge_t = torch.rand(1, 1, 256, 256)
    edge_s = torch.rand(1, 1, 256, 256)
    out = model.forward_pair(frame_t, frame_s, edge_t, edge_s)
    assert {"frame_rec_t", "frame_rec_s", "frame_swap_t", "frame_swap_s"} <= out.keys()
    assert out["frame_t"].shape == (1, 3, 256, 256)
