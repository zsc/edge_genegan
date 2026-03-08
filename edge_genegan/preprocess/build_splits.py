from __future__ import annotations

import argparse
import random
from pathlib import Path


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("build-splits")
    parser.add_argument("--vimeo-root", required=True, type=str)
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--val-ratio", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None)
    return parser


def _load_ids(vimeo_root: Path) -> list[str]:
    ids: list[str] = []
    for p in sorted((vimeo_root / "sequences").glob("*/*")):
        if p.is_dir():
            ids.append(f"{p.parent.name}/{p.name}")
    return ids


def _write(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in ids:
            f.write(c + "\n")


def main(argv: list[str] | None = None) -> None:
    args = build_cli_parser().parse_args(argv)
    root = Path(args.vimeo_root)
    ids = _load_ids(root)
    if not ids:
        raise RuntimeError(f"No clip ids found in {root / 'sequences'}")
    rng = random.Random(args.seed)
    rng.shuffle(ids)

    n_train = int(len(ids) * args.train_ratio)
    n_val = int(len(ids) * args.val_ratio)
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]

    out = Path(args.out_dir) if args.out_dir is not None else root
    _write(out / "sep_trainlist.txt", train_ids)
    _write(out / "sep_vallist.txt", val_ids)
    _write(out / "sep_testlist.txt", test_ids)
    print(
        f"[build_splits] total={len(ids)} train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} out_dir={out}"
    )


if __name__ == "__main__":
    main()
