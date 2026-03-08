from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from edge_genegan.data.vimeo_dataset import VimeoPairDataset, VimeoRolloutDataset


def _write_dummy_sequences(root: Path) -> None:
    rng = np.random.RandomState(0)
    clip_dir = root / "sequences" / "0001" / "0001"
    clip_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, 8):
        frame = (rng.rand(128, 128, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(clip_dir / f"im{i}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    split_dir = root
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "sep_trainlist.txt").write_text("0001/0001\n", encoding="utf-8")


def test_pair_dataset_shapes(tmp_path: Path) -> None:
    root = tmp_path / "vimeo_septuplet"
    _write_dummy_sequences(root)

    ds = VimeoPairDataset(
        root=root,
        split_file="sep_trainlist.txt",
        split="train",
        edge_root=None,
        image_size=64,
        max_gap=3,
    )
    sample = ds[0]
    assert {"frame_t", "frame_s", "edge_t", "edge_s", "t_index", "s_index", "gap", "clip_id"} <= sample.keys()
    assert sample["frame_t"].shape == (3, 64, 64)
    assert sample["edge_t"].shape == (1, 64, 64)
    assert sample["t_index"] != sample["s_index"]


def test_rollout_dataset_shapes(tmp_path: Path) -> None:
    root = tmp_path / "vimeo_septuplet"
    _write_dummy_sequences(root)

    ds = VimeoRolloutDataset(
        root=root,
        split_file="sep_trainlist.txt",
        split="train",
        edge_root=None,
        image_size=64,
        history_len=3,
        future_len=4,
    )
    sample = ds[0]
    assert sample["history_frames"].shape == (3, 3, 64, 64)
    assert sample["future_edges"].shape == (4, 1, 64, 64)
    assert sample["future_frames"].shape == (4, 3, 64, 64)
