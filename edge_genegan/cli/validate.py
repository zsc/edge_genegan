from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from edge_genegan.cli.train import _build_dataset, _build_model
from edge_genegan.evaluators import save_pair_visualization
from edge_genegan.utils import load_config, resolve_device, setup_logger, dump_yaml


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("validate-edge-genegan")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    default_cfg = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    cfg = load_config(str(args.config), default_path=default_cfg)
    if args.output_dir is not None:
        cfg.setdefault("experiment", {})["output_dir"] = str(args.output_dir)

    output_dir = Path(cfg["experiment"].get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(output_dir / "validate_config_dump.yaml", cfg)

    logger = setup_logger(output_dir / "validate.log")
    device = resolve_device(args.device).device
    system, discriminator = _build_model(cfg)
    system.to(device)
    discriminator.to(device)
    system.eval()
    discriminator.eval()
    state = torch.load(str(args.checkpoint), map_location=device)
    system.load_state_dict(state["system"])
    discriminator.load_state_dict(state["discriminator"])

    # build dataset
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
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    metric_sum = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.max_batches:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
            out = system.forward_pair(batch["frame_t"], batch["frame_s"], batch["edge_t"], batch["edge_s"])
            # lightweight sanity visualization
            vis_path = sample_dir / f"val_{i:04d}.png"
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
                vis_path,
            )
            metric_sum += float((out["frame_swap_t"] - batch["frame_t"]).abs().mean().cpu())
            count += 1
    if count > 0:
        logger.info(f"Validation smoke metric(frame_swap_t vs frame_t l1): {metric_sum / count:.6f}")


if __name__ == "__main__":
    main()
