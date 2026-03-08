from __future__ import annotations

from dataclasses import dataclass

from .vimeo_dataset import VimeoPairDataset, VimeoRolloutDataset


@dataclass
class EdgeCacheDatasetConfig:
    root: str
    edge_root: str | None
    split_file: str | None
    split: str
    image_size: int
    edge_mode: str
    max_gap: int
    clip_len: int
    history_len: int
    future_len: int


def build_pair_dataset(cfg: EdgeCacheDatasetConfig) -> VimeoPairDataset:
    return VimeoPairDataset(
        root=cfg.root,
        edge_root=cfg.edge_root,
        split_file=cfg.split_file,
        split=cfg.split,
        image_size=cfg.image_size,
        edge_mode=cfg.edge_mode,
        max_gap=cfg.max_gap,
        clip_len=cfg.clip_len,
        history_len=cfg.history_len,
        future_len=cfg.future_len,
    )


def build_rollout_dataset(cfg: EdgeCacheDatasetConfig) -> VimeoRolloutDataset:
    return VimeoRolloutDataset(
        root=cfg.root,
        edge_root=cfg.edge_root,
        split_file=cfg.split_file,
        split=cfg.split,
        image_size=cfg.image_size,
        edge_mode=cfg.edge_mode,
        max_gap=cfg.max_gap,
        clip_len=cfg.clip_len,
        history_len=cfg.history_len,
        future_len=cfg.future_len,
    )
