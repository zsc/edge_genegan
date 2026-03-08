from __future__ import annotations

import torch

from edge_genegan.models import EdgeRgbSwapSystem


def test_system_forward_pair_shapes() -> None:
    model = EdgeRgbSwapSystem(z_channels=64, a_channels=64, base_channels=16, norm="instance")
    frame_t = torch.rand(2, 3, 256, 256)
    frame_s = torch.rand(2, 3, 256, 256)
    edge_t = torch.rand(2, 1, 256, 256)
    edge_s = torch.rand(2, 1, 256, 256)

    out = model.forward_pair(frame_t, frame_s, edge_t, edge_s)
    assert out["frame_rec_t"].shape == (2, 3, 256, 256)
    assert out["frame_swap_s"].shape == (2, 3, 256, 256)
    assert out["edge_swap_t"].shape == (2, 1, 256, 256)


def test_system_rollout_shapes() -> None:
    model = EdgeRgbSwapSystem(z_channels=64, a_channels=64, base_channels=16, norm="instance")
    history = torch.rand(2, 3, 3, 256, 256)
    future_edges = torch.rand(2, 4, 1, 256, 256)
    out = model.rollout(history, future_edges)
    assert out["future_frames"].shape == (2, 4, 3, 256, 256)
    assert out["a_history"].shape == (2, 3, 64)
    assert out["a_star"].shape == (2, 64)
