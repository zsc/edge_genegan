from __future__ import annotations

import torch


class LPIPSLoss:
    """Optional LPIPS wrapper with graceful fallback."""

    def __init__(self, device: torch.device | str | None = None, net: str = "alex") -> None:
        self._model = None
        self.enabled = False
        try:
            import lpips  # type: ignore

            self._model = lpips.LPIPS(net=net).to(device if device is not None else "cpu")
            self.enabled = True
        except Exception:
            self._model = None
            self.enabled = False

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.tensor(0.0, device=x.device)
        return self._model(x, y).mean()
