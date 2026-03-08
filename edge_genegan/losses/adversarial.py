from __future__ import annotations

import torch
import torch.nn.functional as F


def discriminator_hinge_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()


def generator_hinge_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    return -fake_scores.mean()


def make_fake_scores(discriminator, real: torch.Tensor, fake: torch.Tensor, detach_fake: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    if detach_fake:
        fake_scores = discriminator(fake.detach())
        return discriminator(real), fake_scores
    return discriminator(real), discriminator(fake)
