"""Device utilities."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceSpec:
    device: torch.device
    type: str


def resolve_device(device: str | None) -> DeviceSpec:
    if device is None:
        if torch.cuda.is_available():
            return DeviceSpec(device=torch.device("cuda"), type="cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return DeviceSpec(device=torch.device("mps"), type="mps")
        return DeviceSpec(device=torch.device("cpu"), type="cpu")

    d = device.lower()
    if d in {"auto", "default"}:
        return resolve_device(None)
    if d in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            return DeviceSpec(device=torch.device("cpu"), type="cpu")
        return DeviceSpec(device=torch.device("cuda"), type="cuda")
    if d.startswith("cuda:"):
        if not torch.cuda.is_available():
            return DeviceSpec(device=torch.device("cpu"), type="cpu")
        return DeviceSpec(device=torch.device(d), type="cuda")
    if d == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return DeviceSpec(device=torch.device("mps"), type="mps")
        if torch.cuda.is_available():
            return DeviceSpec(device=torch.device("cuda"), type="cuda")
        return DeviceSpec(device=torch.device("cpu"), type="cpu")
    return DeviceSpec(device=torch.device(d), type=d)
