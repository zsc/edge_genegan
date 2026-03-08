from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from edge_genegan.data.edge_cache_dataset import EdgeCacheDatasetConfig, build_pair_dataset
from edge_genegan.evaluators import save_pair_visualization
from edge_genegan.models import EdgeRgbSwapSystem, PatchDiscriminator
from edge_genegan.trainers import Trainer
from edge_genegan.utils import load_config, resolve_device, set_deterministic, set_seed, setup_logger
from edge_genegan.utils import dump_yaml


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("train-edge-genegan")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stage", type=str, choices=["stage1", "stage2"], default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None, help="Optional hard cap on total train steps")
    return parser.parse_args(argv)


def _build_dataset(cfg: dict, split: str) -> EdgeCacheDatasetConfig:
    data = cfg["data"]
    stage = cfg["train"].get("stage", "stage1")
    max_gap = data.get(f"max_gap_{stage}", data.get("max_gap_stage1", 3))
    root_path = Path(data["root"])
    split_file = data.get(f"sep_{split}list_file", None)
    if split_file is None:
        if split == "val":
            split_file = "sep_vallist.txt"
        elif split == "test":
            split_file = "sep_testlist.txt"
        else:
            split_file = "sep_trainlist.txt"
    if split_file and not root_path.is_file():
        split_file = str(root_path / split_file)
    return EdgeCacheDatasetConfig(
        root=data["root"],
        edge_root=data.get("edge_root"),
        split_file=split_file,
        split=split,
        image_size=data.get("image_size", 256),
        edge_mode=data.get("edge_mode", "offline_soft_edge"),
        max_gap=max_gap,
        clip_len=data.get("clip_len", 7),
        history_len=data.get("history_len", 3),
        future_len=data.get("future_len", 4),
    )


def _build_loaders(cfg: dict) -> tuple[DataLoader, DataLoader | None]:
    data_cfg = cfg["data"]
    num_workers = int(data_cfg.get("num_workers", 2))
    val_every_epoch = int(cfg["train"].get("val_every_epoch", 1))
    common_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": cfg["experiment"].get("pin_memory", True),
    }
    if num_workers > 0:
        common_loader_kwargs["persistent_workers"] = True
    train_dataset = build_pair_dataset(_build_dataset(cfg, "train"))
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"].get("batch_size", 8),
        shuffle=True,
        **common_loader_kwargs,
        drop_last=True,
    )

    if val_every_epoch <= 0:
        return train_loader, None

    try:
        val_dataset = build_pair_dataset(_build_dataset(cfg, "val"))
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            **common_loader_kwargs,
        )
    except (FileNotFoundError, OSError, TypeError):
        try:
            val_dataset = build_pair_dataset(_build_dataset(cfg, "test"))
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                **common_loader_kwargs,
            )
        except (FileNotFoundError, OSError, TypeError):
            val_loader = None
    return train_loader, val_loader


def _build_model(cfg: dict) -> tuple[EdgeRgbSwapSystem, PatchDiscriminator]:
    model = cfg["model"]
    system = EdgeRgbSwapSystem(
        z_channels=model.get("z_channels", 256),
        a_channels=model.get("a_channels", 256),
        base_channels=model.get("base_channels", 64),
        norm=model.get("norm", "instance"),
        rgb_decoder_inject=model.get("rgb_decoder_inject", "adain"),
        use_temporal_state=model.get("use_temporal_state", False),
    )
    disc = PatchDiscriminator(
        in_channels=3,
        base_channels=model.get("discriminator_channels", 64),
        num_layers=model.get("discriminator_layers", 3),
        norm=model.get("discriminator_norm", model.get("norm", "instance")),
    )
    return system, disc


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    default_cfg = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    cfg = load_config(str(args.config), default_path=default_cfg)

    if args.stage is not None:
        cfg.setdefault("train", {})["stage"] = args.stage
    if args.output_dir is not None:
        cfg.setdefault("experiment", {})["output_dir"] = str(args.output_dir)
    if args.resume is not None:
        cfg.setdefault("train", {})["resume"] = str(args.resume)

    exp_cfg = cfg.setdefault("experiment", {})
    output_dir = Path(exp_cfg.get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(output_dir / "config_dump.yaml", cfg)

    device = resolve_device(args.device).device
    set_seed(cfg["experiment"].get("seed", 42))
    set_deterministic(cfg["experiment"].get("deterministic", False))

    logger = setup_logger(output_dir / "train.log")
    logger.info(f"Loading config from {args.config}")
    logger.info(f"Using device {device}")

    train_loader, val_loader = _build_loaders(cfg)
    system, discriminator = _build_model(cfg)
    trainer = Trainer(
        cfg=cfg,
        system=system,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        device=device,
        logger=logger,
    )
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    if val_loader is not None:
        logger.info(f"Val samples: {len(val_loader.dataset)}")

    resume = cfg["train"].get("resume", args.resume)
    if resume:
        logger.info(f"Resume from {resume}")
        trainer.load_checkpoint(resume)

    trainer.run(num_epochs=cfg["train"].get("epochs", 1), num_steps=args.steps)
    # save final checkpoint
    trainer.save_checkpoint(output_dir / "checkpoints" / "final.pt")

    # quick smoke export
    batch = next(iter(val_loader)) if val_loader is not None else None
    if batch is not None:
        out = trainer.system.forward_pair(
            batch["frame_t"].to(device),
            batch["frame_s"].to(device),
            batch["edge_t"].to(device),
            batch["edge_s"].to(device),
        )
        # legacy debug visual
        save_pair_visualization(out, output_dir / "samples" / "pair_example.png")


if __name__ == "__main__":
    main()
