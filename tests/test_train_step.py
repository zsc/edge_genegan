from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from edge_genegan.models import EdgeRgbSwapSystem, PatchDiscriminator
from edge_genegan.trainers import Trainer


class _MiniPairDataset(Dataset[dict[str, Any]]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict[str, Any]:
        torch.manual_seed(idx)
        return {
            "frame_t": torch.randn(3, 256, 256),
            "frame_s": torch.randn(3, 256, 256),
            "edge_t": torch.rand(1, 256, 256),
            "edge_s": torch.rand(1, 256, 256),
        }


def test_trainer_one_step(tmp_path) -> None:
    cfg = {
        "experiment": {"deterministic": False, "seed": 0},
        "train": {
            "lr_g": 1e-4,
            "lr_d": 1e-4,
            "betas": [0.5, 0.999],
            "grad_clip": 0.0,
            "amp": False,
        },
        "loss": {
            "use_gan": False,
            "use_lpips": False,
            "use_cycle": False,
            "lambda_rec": 10.0,
            "lambda_swap": 10.0,
            "lambda_e_rec": 5.0,
            "lambda_e_swap": 5.0,
            "lambda_shared": 2.0,
            "lambda_null": 0.1,
            "lambda_drift": 2.0,
            "lambda_edge": 5.0,
            "lambda_lpips": 1.0,
            "lambda_cyc": 0.5,
            "lambda_adv": 1.0,
        },
    }

    system = EdgeRgbSwapSystem(z_channels=32, a_channels=32, base_channels=8)
    disc = PatchDiscriminator(in_channels=3, base_channels=16, num_layers=2)
    ds = _MiniPairDataset()
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    trainer = Trainer(
        cfg=cfg,
        system=system,
        discriminator=disc,
        train_loader=loader,
        val_loader=None,
        output_dir=tmp_path,
        device="cpu",
    )
    batch = next(iter(loader))
    logs = trainer.train_step(batch)
    assert "L_total_g" in logs
    assert logs["L_total_g"] > 0
