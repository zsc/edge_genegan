"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    system: torch.nn.Module,
    discriminator: torch.nn.Module,
    opt_g: torch.optim.Optimizer | None,
    opt_d: torch.optim.Optimizer | None,
    config: dict[str, Any],
    epoch: int,
    step: int,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "system": system.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g": opt_g.state_dict() if opt_g is not None else None,
            "opt_d": opt_d.state_dict() if opt_d is not None else None,
            "epoch": int(epoch),
            "step": int(step),
            "config": config,
        },
        str(p),
    )


def load_checkpoint(
    path: str | Path,
    *,
    system: torch.nn.Module,
    discriminator: torch.nn.Module,
    opt_g: torch.optim.Optimizer | None = None,
    opt_d: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    ckpt = torch.load(str(path), map_location=map_location)
    system.load_state_dict(ckpt["system"], strict=False)
    if "discriminator" in ckpt:
        discriminator.load_state_dict(ckpt["discriminator"], strict=False)
    if opt_g is not None and ckpt.get("opt_g") is not None:
        opt_g.load_state_dict(ckpt["opt_g"])
    if opt_d is not None and ckpt.get("opt_d") is not None:
        opt_d.load_state_dict(ckpt["opt_d"])
    return ckpt


def point_latest(latest_path: str | Path, target_path: str | Path) -> None:
    latest = Path(latest_path)
    target = Path(target_path)
    latest.parent.mkdir(parents=True, exist_ok=True)
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(target)
    except OSError:
        import shutil
        shutil.copy2(target, latest)
