"""Loss modules."""

from .adversarial import discriminator_hinge_loss, generator_hinge_loss, make_fake_scores
from .cycle import compute_cycle_losses
from .edge_adherence import EdgeAdherenceExtractor, compute_edge_adherence_loss
from .perceptual import LPIPSLoss
from .reconstruction import (
    compute_reconstruction_losses,
    compute_swap_losses,
    compute_shared_structure_loss,
    compute_nulling_loss,
)

__all__ = [
    "discriminator_hinge_loss",
    "generator_hinge_loss",
    "make_fake_scores",
    "compute_cycle_losses",
    "EdgeAdherenceExtractor",
    "compute_edge_adherence_loss",
    "LPIPSLoss",
    "compute_reconstruction_losses",
    "compute_swap_losses",
    "compute_shared_structure_loss",
    "compute_nulling_loss",
]
