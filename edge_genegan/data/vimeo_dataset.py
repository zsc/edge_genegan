from __future__ import annotations

from pathlib import Path
from typing import Any
import zipfile

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .samplers import UniformGapPairSampler
from .transforms import normalize_edge, normalize_rgb, resize_tensor


def _to_torch_rgb(path: Path | str) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1) / 255.0
    return img


def _to_torch_rgb_from_bytes(data: bytes) -> torch.Tensor:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Could not decode image bytes")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1) / 255.0
    return img


def _to_torch_gray(path: Path | str) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return torch.from_numpy(img.astype(np.float32)).unsqueeze(0) / 255.0


def _to_torch_gray_from_bytes(data: bytes) -> torch.Tensor:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Could not decode image bytes")
    return torch.from_numpy(img.astype(np.float32)).unsqueeze(0) / 255.0


def _zip_root_prefix(names: list[str]) -> str:
    for name in names:
        if name.endswith("/"):
            continue
        parts = [p for p in name.split("/") if p]
        if "sequences" not in parts:
            continue
        idx = parts.index("sequences")
        if idx > 0:
            return "/".join(parts[:idx])
        break
    return ""


def _build_split_ids(root: Path, split_file: str | None, split: str) -> list[str]:
    if split_file is None:
        base = root / "sequences"
        ids = []
        for c in sorted((root / "sequences").glob("*/*")):
            if c.is_dir():
                part1 = c.parent.name
                part2 = c.name
                ids.append(f"{part1}/{part2}")
        return ids

    if root.is_file() and root.suffix.lower() == ".zip":
        with zipfile.ZipFile(root, "r") as zf:
            names = zf.namelist()
        names_set = set(names)
        prefix = _zip_root_prefix(names)
        split_source = None
        if split_file is None:
            seq_prefix = f"{prefix}/sequences/" if prefix else "sequences/"
            ids = []
            for name in names:
                if not name.startswith(seq_prefix) or not name.endswith(".png"):
                    continue
                parts = [p for p in name.split("/") if p]
                try:
                    sid = parts.index("sequences")
                except ValueError:
                    continue
                if sid + 2 < len(parts):
                    ids.append(f"{parts[sid + 1]}/{parts[sid + 2]}")
            ids = sorted(set(ids))
            if not ids:
                raise RuntimeError(f"No clips found in zip split: {root}")
            return ids

        candidates = [split_file]
        split_name = Path(split_file).name
        if split_name not in candidates:
            candidates.append(split_name)
        if split_name and split_name != split_file:
            candidates.append(str(split_name))
            if prefix:
                candidates.append(f"{prefix}/{split_name}")
        if prefix:
            candidates.append(f"{prefix}/{split_file}")
        split_member = next((c for c in candidates if c in names_set), None)
        if split_member is None:
            raise FileNotFoundError(f"Split file not found in zip: {split_file}")
        with zipfile.ZipFile(root, "r") as zf:
            raw = zf.read(split_member).decode("utf-8", errors="ignore")
        split_source = split_member
        lines = raw.splitlines()
    else:
        split_path = root / split_file
        if not split_path.is_file():
            # support passing explicit path
            split_path = Path(split_file)
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        lines = split_path.read_text(encoding="utf-8").splitlines()
        split_source = str(split_path)

    ids: list[str] = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    if not ids:
        raise RuntimeError(f"No clips found in split file: {split_source}")
    return ids


def build_vimeo_split_ids(root: str, split_file: str | None = None, split: str = "train") -> list[str]:
    return _build_split_ids(Path(root), split_file, split)


class _BaseVimeoDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        edge_root: str | Path | None = None,
        split_file: str | None = None,
        split: str = "train",
        image_size: int = 256,
        edge_mode: str = "offline_soft_edge",
        max_gap: int = 3,
        clip_len: int = 7,
        history_len: int = 3,
        future_len: int = 4,
    ) -> None:
        self.root = Path(root)
        self.edge_root = Path(edge_root) if edge_root is not None else None
        self._zip_root = self.root.is_file() and self.root.suffix.lower() == ".zip"
        self._zip_prefix = ""
        self._zip_file: zipfile.ZipFile | None = None
        if self._zip_root:
            with zipfile.ZipFile(self.root, "r") as zf:
                self._zip_prefix = _zip_root_prefix(zf.namelist())
        self.image_size = int(image_size)
        self.edge_mode = edge_mode
        self.clip_len = int(clip_len)
        self.history_len = int(history_len)
        self.future_len = int(future_len)
        self.sampler = UniformGapPairSampler(clip_len=self.clip_len, max_gap=max_gap)
        self.clip_ids = _build_split_ids(self.root, split_file, split)

    def _zip_member(self, relative: str) -> str:
        rel = str(relative).strip("/")
        if self._zip_prefix:
            return f"{self._zip_prefix}/{rel}"
        return rel

    def _frame_member(self, clip_id: str, idx: int) -> str:
        return self._zip_member(f"sequences/{clip_id}/im{idx}.png")

    def _edge_member(self, clip_id: str, idx: int) -> str:
        return self._zip_member(f"sequences/{clip_id}/im{idx}.png")

    def _read_file_bytes(self, member: str) -> bytes:
        if not self._zip_root:
            raise RuntimeError("Not a zip dataset root")
        return self._get_zip_file().read(member)

    def _get_zip_file(self) -> zipfile.ZipFile:
        if not self._zip_root:
            raise RuntimeError("Not a zip dataset root")
        if self._zip_file is None:
            # Keep one handle per dataset instance/worker instead of reopening for every frame.
            self._zip_file = zipfile.ZipFile(self.root, "r")
        return self._zip_file

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_zip_file"] = None
        return state

    def __del__(self) -> None:
        if self._zip_file is not None:
            self._zip_file.close()

    def _load_rgb(self, path: Path | str) -> torch.Tensor:
        if self._zip_root:
            img = _to_torch_rgb_from_bytes(self._read_file_bytes(str(path)))
        else:
            img = _to_torch_rgb(Path(path))
        if self.image_size is not None:
            img = resize_tensor(img, self.image_size)
        return img

    def _load_gray(self, path: Path | str) -> torch.Tensor:
        if self._zip_root:
            edge = _to_torch_gray_from_bytes(self._read_file_bytes(str(path)))
        else:
            edge = _to_torch_gray(Path(path))
        if self.image_size is not None:
            edge = resize_tensor(edge, self.image_size)
        return edge

    def __len__(self) -> int:
        return len(self.clip_ids)

    def _load_frame(self, clip_id: str, idx: int) -> torch.Tensor:
        if self._zip_root:
            source: Path | str = self._frame_member(clip_id, idx)
        else:
            source = self.root / "sequences" / clip_id / f"im{idx}.png"
        frame = self._load_rgb(source)
        return normalize_rgb(frame)

    def _load_frame_raw(self, clip_id: str, idx: int) -> torch.Tensor:
        if self._zip_root:
            source: Path | str = self._frame_member(clip_id, idx)
        else:
            source = self.root / "sequences" / clip_id / f"im{idx}.png"
        if self._zip_root:
            frame = _to_torch_rgb_from_bytes(self._read_file_bytes(str(source)))
        else:
            frame = _to_torch_rgb(Path(source))
        if self.image_size is not None:
            frame = resize_tensor(frame, self.image_size)
        return frame

    def _load_edge_cached(self, clip_id: str, idx: int) -> torch.Tensor:
        if self.edge_root is None:
            return self._compute_soft_edge(clip_id, idx)
        if (self.edge_root / self.edge_mode).is_dir():
            base = self.edge_root / self.edge_mode
        else:
            base = self.edge_root
        epath = base / "sequences" / clip_id / f"im{idx}.png"
        if epath.is_file():
            edge = self._load_gray(epath)
            return normalize_edge(edge)
        return self._compute_soft_edge(clip_id, idx)

    def _compute_soft_edge(self, clip_id: str, idx: int) -> torch.Tensor:
        rgb = self._load_frame_raw(clip_id, idx)
        img = (rgb.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        if self.edge_mode == "offline_canny":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            e = cv2.Canny(gray, 80, 160).astype(np.float32) / 255.0
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            e = cv2.magnitude(gx, gy)
            e = e / (e.max() + 1e-6)
        return torch.from_numpy(e).unsqueeze(0)

    def _edge_for_frame(self, clip_id: str, idx: int) -> torch.Tensor:
        edge = self._load_edge_cached(clip_id, idx)
        edge = torch.clamp(edge, 0.0, 1.0)
        if edge.shape[0] != 1:
            edge = edge.mean(dim=0, keepdim=True)
        return edge

    def _get_indices(self) -> tuple[int, int]:
        t, s = self.sampler()
        return t + 1, s + 1


class VimeoPairDataset(_BaseVimeoDataset):
    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip_id = self.clip_ids[idx]
        t_index, s_index = self._get_indices()
        frame_t = self._load_frame(clip_id, t_index)
        frame_s = self._load_frame(clip_id, s_index)
        edge_t = self._edge_for_frame(clip_id, t_index)
        edge_s = self._edge_for_frame(clip_id, s_index)
        return {
            "frame_t": frame_t,
            "frame_s": frame_s,
            "edge_t": edge_t,
            "edge_s": edge_s,
            "t_index": t_index - 1,
            "s_index": s_index - 1,
            "gap": abs(s_index - t_index),
            "clip_id": clip_id,
        }


class VimeoRolloutDataset(_BaseVimeoDataset):
    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip_id = self.clip_ids[idx]
        history = []
        future = []
        future_gt = []
        for i in range(self.history_len):
            history.append(self._load_frame(clip_id, i + 1))
        for j in range(self.future_len):
            future_idx = self.history_len + j + 1
            future.append(self._edge_for_frame(clip_id, future_idx))
            future_gt.append(self._load_frame(clip_id, future_idx))

        return {
            "history_frames": torch.stack(history, dim=0),
            "future_edges": torch.stack(future, dim=0),
            "future_frames": torch.stack(future_gt, dim=0),
            "clip_id": clip_id,
            "history_len": self.history_len,
            "future_len": self.future_len,
        }
