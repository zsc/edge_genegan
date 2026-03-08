from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from edge_genegan.evaluators import save_rollout_gif, save_tensors_as_grid
from edge_genegan.models import EdgeRgbSwapSystem
from edge_genegan.utils import load_config, resolve_device
from edge_genegan.data.transforms import resize_tensor


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("infer-rollout")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--history-dir", type=Path, default=None)
    parser.add_argument("--future-edge-dir", type=Path, default=None)
    parser.add_argument("--sample-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--future-length", type=int, default=4)
    parser.add_argument("--fps", type=int, default=8)
    return parser.parse_args(argv)


def _read_rgb_dir(path: Path, image_size: int) -> list[torch.Tensor]:
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not files:
        raise FileNotFoundError(f"No image files in {path}")
    out: list[torch.Tensor] = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Read failed: {f}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1) / 255.0
        img = resize_tensor(img, image_size)
        out.append(img.mul(2.0).sub(1.0))
    return out


def _read_edge_dir(path: Path, image_size: int) -> list[torch.Tensor]:
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not files:
        raise FileNotFoundError(f"No edge files in {path}")
    out: list[torch.Tensor] = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Read failed: {f}")
        img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        img = resize_tensor(img, image_size)
        out.append(img)
    return out


def _normalize_history_tensor(history: torch.Tensor) -> torch.Tensor:
    history = history.float()
    if history.max() > 1.5:
        history = history / 127.5 - 1.0
    elif history.min() >= 0.0:
        history = history * 2.0 - 1.0
    return history.clamp(-1.0, 1.0)


def _normalize_edge_tensor(edge: torch.Tensor) -> torch.Tensor:
    edge = edge.float()
    if edge.max() > 1.5:
        edge = edge / 255.0
    return edge.clamp(0.0, 1.0)


def _resize_sequence(x: torch.Tensor, image_size: int) -> torch.Tensor:
    if image_size <= 0:
        return x
    if x.dim() == 4:
        return F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    if x.dim() == 5:
        b, t, c, h, w = x.shape
        flat = x.reshape(b * t, c, h, w)
        flat = F.interpolate(flat, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return flat.reshape(b, t, c, image_size, image_size)
    raise ValueError(f"Expected 4D or 5D sequence tensor, got {tuple(x.shape)}")


def _tensor_to_uint8_rgb(frame: torch.Tensor) -> np.ndarray:
    frame = frame.detach().float()
    if frame.dim() != 3:
        raise ValueError(f"Expected frame [C,H,W], got {tuple(frame.shape)}")
    if frame.size(0) == 1:
        frame = frame.repeat(3, 1, 1)
    if frame.min() < 0.0:
        frame = (frame.clamp(-1.0, 1.0) + 1.0) * 127.5
    else:
        frame = frame.clamp(0.0, 1.0) * 255.0
    return frame.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def _load_sample(path: Path, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if path.suffix == ".npz":
        payload = np.load(path, allow_pickle=True)
    else:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) and not hasattr(payload, "keys"):
        raise TypeError("sample-path should contain dict-like data")
    if "history_frames" not in payload or "future_edges" not in payload:
        raise KeyError("sample-path must contain history_frames and future_edges")
    history = payload["history_frames"]
    future = payload["future_edges"]
    if not torch.is_tensor(history):
        history = torch.from_numpy(np.asarray(history))
    if not torch.is_tensor(future):
        future = torch.from_numpy(np.asarray(future))
    if history.ndim not in {4, 5}:
        raise ValueError("history_frames should be [T,3,H,W] or [B,T,3,H,W]")
    if future.ndim not in {4, 5}:
        raise ValueError("future_edges should be [K,1,H,W] or [B,K,1,H,W]")
    history = _normalize_history_tensor(_resize_sequence(history.float(), image_size))
    future = _normalize_edge_tensor(_resize_sequence(future.float(), image_size))
    return history, future


def _save_frames(frames: torch.Tensor, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames[0]):
        img = _tensor_to_uint8_rgb(frame)
        cv2.imwrite(str(output / f"frame_{i:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = load_config(str(args.config), default_path=Path(__file__).resolve().parents[2] / "configs" / "default.yaml")
    infer_cfg = cfg.setdefault("infer", {})
    device = resolve_device(args.device).device
    data_cfg = cfg.setdefault("data", {})
    image_size = int(data_cfg.get("image_size", 256))

    model = EdgeRgbSwapSystem(
        z_channels=cfg["model"].get("z_channels", 256),
        a_channels=cfg["model"].get("a_channels", 256),
        base_channels=cfg["model"].get("base_channels", 64),
        norm=cfg["model"].get("norm", "instance"),
        rgb_decoder_inject=cfg["model"].get("rgb_decoder_inject", "adain"),
        use_temporal_state=cfg["model"].get("use_temporal_state", False),
        appearance_pool=infer_cfg.get("appearance_pool", "mean"),
    ).to(device)
    ckpt = torch.load(str(args.checkpoint), map_location=device)
    model.load_state_dict(ckpt["system"])
    model.eval()

    if args.sample_path is not None:
        history, future_edges = _load_sample(args.sample_path, image_size)
        history = history.to(device)
        future_edges = future_edges.to(device)
    else:
        if args.history_dir is None or args.future_edge_dir is None:
            raise ValueError("--history-dir and --future-edge-dir required if --sample-path not provided")
        history_frames = _read_rgb_dir(args.history_dir, image_size)
        future_edges = _read_edge_dir(args.future_edge_dir, image_size)
        if len(history_frames) < args.history_length:
            raise ValueError(f"history-dir has fewer than {args.history_length} frames")
        if len(future_edges) < args.future_length:
            raise ValueError(f"future-edge-dir has fewer than {args.future_length} frames")
        history = torch.stack(history_frames[: args.history_length], dim=0)
        future_edges = torch.stack(future_edges[: args.future_length], dim=0)
        history = history.to(device)
        future_edges = future_edges.to(device)

    history = history.unsqueeze(0) if history.dim() == 4 else history
    future_edges = future_edges.unsqueeze(0) if future_edges.dim() == 4 else future_edges

    with torch.no_grad():
        out = model.rollout(history, future_edges)
    preds = out["future_frames"].detach().cpu()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_frames(preds, args.output_dir / "frames")
    if infer_cfg.get("save_png", True):
        preds = out["future_frames"][0].clamp(-1.0, 1.0)
        save_tensors_as_grid([f for f in preds], args.output_dir / "grid.png", max_per_row=4)
    if infer_cfg.get("save_mp4", True):
        save_rollout_gif(out["future_frames"][0], args.output_dir / "rollout.mp4", fps=args.fps)


if __name__ == "__main__":
    main()
