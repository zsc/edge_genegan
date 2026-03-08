from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AppearanceAggregatorConfig:
    mode: str = "mean"
    ema_alpha: float = 0.9
    attn_hidden: int = 64


class AppearanceAggregator(nn.Module):
    """Aggregate a sequence of appearance vectors into one global vector."""

    def __init__(self, a_channels: int, mode: str = "mean", ema_alpha: float = 0.9, attn_hidden: int = 64) -> None:
        super().__init__()
        self.a_channels = int(a_channels)
        self.mode = mode
        self.ema_alpha = float(ema_alpha)

        if self.mode == "attention":
            hidden = max(1, int(attn_hidden))
            self._attention = nn.Sequential(
                nn.Linear(self.a_channels, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
            )
        else:
            self._attention = None

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        if a.ndim != 3:
            raise ValueError(f"Expected appearance shape [B, T, C], got {tuple(a.shape)}")
        if self.mode == "mean":
            return a.mean(dim=1)
        if self.mode == "max":
            return a.max(dim=1).values
        if self.mode == "attention":
            scores = self._attention(a).squeeze(-1)
            weights = torch.softmax(scores, dim=1)
            return (a * weights.unsqueeze(-1)).sum(dim=1)
        if self.mode == "ema":
            b, t, _ = a.shape
            device = a.device
            dtype = a.dtype
            offsets = torch.arange(t, 0, -1, dtype=dtype, device=device)
            weights = (self.ema_alpha ** offsets).clamp(min=1e-6)
            weights = weights / weights.sum()
            return (a * weights.view(1, t, 1)).sum(dim=1)
        raise ValueError(f"Unsupported appearance pooling mode: {self.mode}")

    @classmethod
    def from_config(cls, cfg: AppearanceAggregatorConfig | dict[str, object]) -> "AppearanceAggregator":
        if isinstance(cfg, AppearanceAggregatorConfig):
            return cls(cfg.a_channels, cfg.mode, cfg.ema_alpha, cfg.attn_hidden)
        return cls(
            a_channels=int(cfg["a_channels"]),
            mode=str(cfg.get("mode", "mean")),
            ema_alpha=float(cfg.get("ema_alpha", 0.9)),
            attn_hidden=int(cfg.get("attn_hidden", 64)),
        )
