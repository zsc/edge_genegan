"""YAML config helpers."""

from __future__ import annotations

from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

ConfigDict = dict[str, Any]


def load_yaml(path: str | Path) -> ConfigDict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be dict, got {type(data)}")
    return data


def load_config(config_path: str | Path, default_path: str | Path | None = None) -> ConfigDict:
    cfg = load_yaml(default_path) if default_path is not None else {}
    override = load_yaml(config_path)
    return merge_configs(cfg, override)


def merge_configs(base: ConfigDict, update: ConfigDict) -> ConfigDict:
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_configs(out[k], v)
        else:
            out[k] = v
    return out


def dump_yaml(path: str | Path, cfg: ConfigDict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        data = asdict(cfg) if is_dataclass(cfg) else cfg
        yaml.safe_dump(data, f, sort_keys=False)


def ensure_absolute_dict(cfg: ConfigDict) -> ConfigDict:
    return dict(cfg)
