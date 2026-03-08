"""Optional training hooks for extensibility."""

from __future__ import annotations

from typing import Any


def before_step(state: dict[str, Any]) -> None:
    return None


def after_step(state: dict[str, Any]) -> None:
    return None
