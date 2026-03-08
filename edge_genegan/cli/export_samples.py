from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from edge_genegan.cli.train import _build_model, _build_dataset
from edge_genegan.evaluators import save_pair_visualization
from edge_genegan.utils import load_config, resolve_device, dump_yaml


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("export-samples")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = load_config(str(args.config), default_path=Path(__file__).resolve().parents[2] / "configs" / "default.yaml")
    if args.output_dir is not None:
        cfg.setdefault("experiment", {})["output_dir"] = str(args.output_dir)

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(output_dir / "export_config_dump.yaml", cfg)

    device = resolve_device(args.device).device
    system, _ = _build_model(cfg)
    state = torch.load(str(args.checkpoint), map_location=device)
    system.load_state_dict(state["system"])
    system.to(device)
    system.eval()

    dataset_cfg = _build_dataset(cfg, args.split)
    from edge_genegan.data.edge_cache_dataset import build_pair_dataset

    dataset = build_pair_dataset(dataset_cfg)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 2),
        pin_memory=cfg["experiment"].get("pin_memory", True),
    )
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            out = system.forward_pair(batch["frame_t"], batch["frame_s"], batch["edge_t"], batch["edge_s"])
            save_pair_visualization(
                {
                    "frame_t": batch["frame_t"],
                    "frame_s": batch["frame_s"],
                    "edge_t": batch["edge_t"],
                    "edge_s": batch["edge_s"],
                    "frame_rec_t": out["frame_rec_t"],
                    "frame_rec_s": out["frame_rec_s"],
                    "frame_swap_t": out["frame_swap_t"],
                    "frame_swap_s": out["frame_swap_s"],
                    "edge_rec_t": out["edge_rec_t"],
                    "edge_rec_s": out["edge_rec_s"],
                    "edge_swap_t": out["edge_swap_t"],
                    "edge_swap_s": out["edge_swap_s"],
                },
                sample_dir / f"{i:05d}.png",
            )


if __name__ == "__main__":
    main()
