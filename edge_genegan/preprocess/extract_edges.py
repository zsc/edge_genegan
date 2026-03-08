from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


def _edge_from_rgb(path: Path, mode: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if mode in {"offline_canny", "canny"}:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, 80, 160)
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = mag / (mag.max() + 1e-6)
        edge = (mag * 255.0).astype(np.uint8)
    return edge


def _collect_inputs(root: Path) -> list[tuple[Path, Path]]:
    if not (root / "sequences").exists():
        raise FileNotFoundError(f"Expected directory with sequences/: {root}")
    out: list[tuple[Path, Path]] = []
    for p in sorted((root / "sequences").glob("*/*/im*.png")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        out.append((p, rel))
    if not out:
        raise RuntimeError(f"No images found under {root / 'sequences'}")
    return out


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("extract-edges")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--output-root", required=True, type=str)
    parser.add_argument("--mode", type=str, default="soft", choices=["soft", "canny", "offline_canny", "offline_soft_edge"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--processes", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_cli_parser().parse_args(argv)
    data_root = Path(args.data_root)
    out_root = Path(args.output_root)
    mode = args.mode
    if mode == "offline_canny":
        mode = "offline_canny"
    elif mode == "soft":
        mode = "offline_soft_edge"

    mode_key = "offline_soft_edge" if mode in {"soft", "offline_soft_edge"} else "offline_canny"
    outputs_root = out_root / mode_key
    inputs = _collect_inputs(data_root)
    pairs: list[tuple[Path, Path]] = []
    for src, rel in inputs:
        dst = outputs_root / rel
        pairs.append((src, dst))

    total = len(pairs)
    workers = max(1, int(args.processes))

    def _worker(item: tuple[Path, Path]) -> str:
        src, dst = item
        dst.parent.mkdir(parents=True, exist_ok=True)
        e = _edge_from_rgb(src, mode)
        if args.image_size > 0:
            e = cv2.resize(e, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(dst), e)
        return str(dst)

    if workers == 1:
        for item in pairs:
            _worker(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_worker, item) for item in pairs]
            for _ in as_completed(futures):
                pass

    print(f"[extract_edges] processed={total}, out_dir={outputs_root}")


if __name__ == "__main__":
    main()
