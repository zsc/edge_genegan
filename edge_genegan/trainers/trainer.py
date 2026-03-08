from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from edge_genegan.evaluators import save_pair_visualization
from edge_genegan.losses import (
    EdgeAdherenceExtractor,
    LPIPSLoss,
    compute_cycle_losses,
    compute_edge_adherence_loss,
    compute_nulling_loss,
    compute_reconstruction_losses,
    compute_shared_structure_loss,
    compute_swap_losses,
    discriminator_hinge_loss,
    generator_hinge_loss,
)
from edge_genegan.utils import save_checkpoint as save_ckpt_utils, load_checkpoint as load_ckpt_utils


@dataclass
class TrainerState:
    epoch: int = 0
    step: int = 0


class Trainer:
    def __init__(
        self,
        cfg: dict[str, Any],
        system: nn.Module,
        discriminator: nn.Module,
        train_loader: DataLoader | None,
        val_loader: DataLoader | None = None,
        output_dir: str | Path = "./outputs",
        device: torch.device | str = "cpu",
        logger: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.system = system
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.device = torch.device(device)

        self.state = TrainerState()
        self.criterion_edge = EdgeAdherenceExtractor().to(self.device)
        self.lpips = LPIPSLoss(self.device, net="alex")

        self._loss_cfg = self.cfg.get("loss", {})
        self._train_cfg = self.cfg.get("train", {})

        self.use_gan = bool(self._loss_cfg.get("use_gan", False))
        self.use_cycle = bool(self._loss_cfg.get("use_cycle", False))
        self.use_lpips = bool(self._loss_cfg.get("use_lpips", False)) and self.lpips.enabled

        lr_g = float(self._train_cfg.get("lr_g", 2e-4))
        lr_d = float(self._train_cfg.get("lr_d", 2e-4))
        beta1, beta2 = self._train_cfg.get("betas", [0.5, 0.999])
        self.optim_g = torch.optim.Adam(self.system.parameters(), lr=lr_g, betas=(float(beta1), float(beta2)))
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(float(beta1), float(beta2)))

        self.grad_clip = float(self._train_cfg.get("grad_clip", 0.0))
        self.log_every = int(self._train_cfg.get("log_every", 100))
        self.vis_every = int(self._train_cfg.get("vis_every", 500))
        self.save_every_epoch = int(self._train_cfg.get("save_every_epoch", 1))

        exp_cfg = self.cfg.get("experiment", {})
        self.amp = bool(self._train_cfg.get("amp", exp_cfg.get("amp", False))) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        self.system.to(self.device)
        self.discriminator.to(self.device)

    def save_checkpoint(self, path: str | Path) -> None:
        save_ckpt_utils(
            path=path,
            system=self.system,
            discriminator=self.discriminator,
            opt_g=self.optim_g,
            opt_d=self.optim_d,
            config=self.cfg,
            epoch=self.state.epoch,
            step=self.state.step,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = load_ckpt_utils(
            path,
            system=self.system,
            discriminator=self.discriminator,
            opt_g=self.optim_g,
            opt_d=self.optim_d,
            map_location=self.device,
        )
        self.state.epoch = int(ckpt.get("epoch", 0))
        self.state.step = int(ckpt.get("step", 0))

    def _to_device(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                out[key] = value.to(self.device, non_blocking=True)
            else:
                out[key] = value
        return out

    def _loss_dict(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        losses = {}
        rec = compute_reconstruction_losses(outputs, batch)
        swap = compute_swap_losses(outputs, batch)
        losses.update(rec)
        losses.update(swap)
        losses["L_shared"] = compute_shared_structure_loss(outputs)
        losses["L_null"] = compute_nulling_loss(outputs)

        drift = torch.tensor(0.0, device=self.device)
        z_t = outputs["z_e_t"]
        z_s = outputs["z_e_s"]
        sw_t = self.system.encode_rgb(outputs["frame_swap_t"])
        sw_s = self.system.encode_rgb(outputs["frame_swap_s"])
        z_sw_t = sw_t["z"]
        a_sw_t = sw_t["a"]
        z_sw_s = sw_s["z"]
        a_sw_s = sw_s["a"]
        drift = drift + torch.nn.functional.l1_loss(z_sw_t, z_t) + torch.nn.functional.l1_loss(z_sw_s, z_s)
        drift = drift + torch.nn.functional.l1_loss(a_sw_t, outputs["a_s"]) + torch.nn.functional.l1_loss(a_sw_s, outputs["a_t"])
        losses["L_drift"] = drift

        losses["L_edge"] = compute_edge_adherence_loss(outputs["frame_swap_t"], batch["edge_t"], self.criterion_edge) + compute_edge_adherence_loss(
            outputs["frame_swap_s"],
            batch["edge_s"],
            self.criterion_edge,
        )

        if self.use_lpips:
            losses["L_lpips"] = self.lpips(outputs["frame_swap_t"], batch["frame_t"]) + self.lpips(outputs["frame_swap_s"], batch["frame_s"])
        else:
            losses["L_lpips"] = torch.tensor(0.0, device=self.device)

        if self.use_cycle:
            cycle_out = self.system.forward_pair(
                frame_t=outputs["frame_swap_t"],
                frame_s=outputs["frame_swap_s"],
                edge_t=outputs["edge_swap_t"],
                edge_s=outputs["edge_swap_s"],
            )
            losses["L_cycle"] = compute_cycle_losses(
                cycle_out,
                {
                    "frame_t": batch["frame_t"],
                    "frame_s": batch["frame_s"],
                    "edge_t": batch["edge_t"],
                    "edge_s": batch["edge_s"],
                },
            )
        else:
            losses["L_cycle"] = torch.tensor(0.0, device=self.device)

        total = (
            float(self._loss_cfg.get("lambda_rec", 10.0)) * (rec["L_rec_rgb"] + rec["L_rec_edge"] * float(self._loss_cfg.get("lambda_e_rec", 1.0)))
            + float(self._loss_cfg.get("lambda_swap", 10.0))
            * (swap["L_swap_rgb"] + swap["L_swap_edge"] * float(self._loss_cfg.get("lambda_e_swap", 1.0)))
            + float(self._loss_cfg.get("lambda_shared", 2.0)) * losses["L_shared"]
            + float(self._loss_cfg.get("lambda_null", 0.1)) * losses["L_null"]
            + float(self._loss_cfg.get("lambda_drift", 2.0)) * losses["L_drift"]
            + float(self._loss_cfg.get("lambda_edge", 5.0)) * losses["L_edge"]
            + float(self._loss_cfg.get("lambda_lpips", 1.0)) * losses["L_lpips"]
            + float(self._loss_cfg.get("lambda_cyc", 0.0)) * losses["L_cycle"]
        )
        losses["L_total"] = total
        return losses

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        batch = self._to_device(batch)
        self.system.train()
        if self.use_gan:
            self.discriminator.train()

        with autocast(enabled=self.amp):
            outputs = self.system.forward_pair(
                batch["frame_t"], batch["frame_s"], batch["edge_t"], batch["edge_s"]
            )
            losses = self._loss_dict(outputs, batch)
            fake_logits = torch.cat(
                [outputs["frame_rec_t"], outputs["frame_rec_s"], outputs["frame_swap_t"], outputs["frame_swap_s"]],
                dim=0,
            )
            real_logits = torch.cat([batch["frame_t"], batch["frame_s"]], dim=0)
            if self.use_gan:
                d_real = self.discriminator(real_logits)
                d_fake = self.discriminator(fake_logits.detach())
                losses["L_D"] = discriminator_hinge_loss(d_real, d_fake)
                losses["L_G_adv"] = generator_hinge_loss(self.discriminator(fake_logits))
                losses["L_total_g"] = losses["L_total"] + float(self._loss_cfg.get("lambda_adv", 1.0)) * losses["L_G_adv"]
            else:
                losses["L_D"] = torch.tensor(0.0, device=self.device)
                losses["L_G_adv"] = torch.tensor(0.0, device=self.device)
                losses["L_total_g"] = losses["L_total"]

        if self.use_gan:
            self.optim_d.zero_grad(set_to_none=True)
            self.scaler.scale(losses["L_D"]).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optim_d)
                clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
            self.scaler.step(self.optim_d)

        self.optim_g.zero_grad(set_to_none=True)
        self.scaler.scale(losses["L_total_g"]).backward()
        if self.grad_clip > 0:
            self.scaler.unscale_(self.optim_g)
            clip_grad_norm_(self.system.parameters(), self.grad_clip)
        self.scaler.step(self.optim_g)
        self.scaler.update()

        self.state.step += 1
        self._maybe_log(losses, self.state.step)
        self._maybe_visualize(batch, outputs, self.state.step)
        return {k: float(v.detach().cpu().item()) for k, v in losses.items()}

    def validate_step(self, batch: dict[str, Any]) -> dict[str, float]:
        batch = self._to_device(batch)
        self.system.eval()
        if self.use_gan:
            self.discriminator.eval()
        with torch.no_grad():
            with autocast(enabled=self.amp):
                outputs = self.system.forward_pair(
                    batch["frame_t"], batch["frame_s"], batch["edge_t"], batch["edge_s"]
                )
                losses = self._loss_dict(outputs, batch)
                fake_logits = torch.cat(
                    [outputs["frame_rec_t"], outputs["frame_rec_s"], outputs["frame_swap_t"], outputs["frame_swap_s"]],
                    dim=0,
                )
                real_logits = torch.cat([batch["frame_t"], batch["frame_s"]], dim=0)
                if self.use_gan:
                    d_real = self.discriminator(real_logits)
                    d_fake = self.discriminator(fake_logits)
                    losses["L_D"] = discriminator_hinge_loss(d_real, d_fake)
                    losses["L_G_adv"] = generator_hinge_loss(self.discriminator(fake_logits))
                    losses["L_total_g"] = losses["L_total"] + float(self._loss_cfg.get("lambda_adv", 1.0)) * losses["L_G_adv"]
                else:
                    losses["L_D"] = torch.tensor(0.0, device=self.device)
                    losses["L_G_adv"] = torch.tensor(0.0, device=self.device)
                    losses["L_total_g"] = losses["L_total"]
        return {k: float(v.detach().cpu().item()) for k, v in losses.items()}

    def run(self, num_epochs: int, num_steps: int | None = None) -> None:
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        val_every = int(self._train_cfg.get("val_every_epoch", 1))
        for epoch in range(self.state.epoch + 1, int(num_epochs) + 1):
            self.state.epoch = epoch
            if self.train_loader is None:
                break
            for batch_idx, batch in enumerate(self.train_loader):
                self.train_step(batch)
                if num_steps is not None and self.state.step >= num_steps:
                    break
            if num_steps is not None and self.state.step >= num_steps:
                if self.logger is not None:
                    self.logger.info(f"reached step cap {num_steps}")
                break
            if self.val_loader is not None and epoch % max(1, val_every) == 0:
                self._run_validate()
            if epoch % self.save_every_epoch == 0:
                epoch_ckpt = ckpt_dir / f"epoch_{epoch:04d}.pt"
                latest = ckpt_dir / "latest.pt"
                self.save_checkpoint(epoch_ckpt)
                self.save_checkpoint(latest)
                if self.logger is not None:
                    self.logger.info(f"saved checkpoint: {epoch_ckpt}")
                    self.logger.info(f"saved checkpoint: {latest}")

    def _run_validate(self) -> None:
        if self.val_loader is None:
            return
        total = {}
        count = 0
        for batch in self.val_loader:
            res = self.validate_step(batch)
            for key, value in res.items():
                total[key] = total.get(key, 0.0) + float(value)
            count += 1
        if count == 0:
            return
        avg = {k: v / count for k, v in total.items()}
        if self.logger is not None:
            msg = ", ".join(f"{k}={v:.4f}" for k, v in avg.items())
            self.logger.info(f"val: {msg}")

    def _maybe_log(self, losses: dict[str, torch.Tensor], step: int) -> None:
        if self.logger is None or self.log_every <= 0:
            return
        if step % self.log_every != 0:
            return
        msg = ", ".join(f"{k}={float(v.detach().cpu().item()):.4f}" for k, v in losses.items() if not torch.isnan(v))
        self.logger.info(f"train step={step}: {msg}")

    def _maybe_visualize(self, batch: dict[str, Any], outputs: dict[str, torch.Tensor], step: int) -> None:
        if self.vis_every <= 0 or step % self.vis_every != 0:
            return
        vis = {
            "frame_t": batch["frame_t"].detach(),
            "frame_s": batch["frame_s"].detach(),
            "edge_t": batch["edge_t"].detach(),
            "edge_s": batch["edge_s"].detach(),
            "frame_rec_t": outputs["frame_rec_t"].detach(),
            "frame_rec_s": outputs["frame_rec_s"].detach(),
            "frame_swap_t": outputs["frame_swap_t"].detach(),
            "frame_swap_s": outputs["frame_swap_s"].detach(),
            "edge_rec_t": outputs["edge_rec_t"].detach(),
            "edge_rec_s": outputs["edge_rec_s"].detach(),
            "edge_swap_t": outputs["edge_swap_t"].detach(),
            "edge_swap_s": outputs["edge_swap_s"].detach(),
        }
        sample_dir = self.output_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        vis_path = sample_dir / f"train_step_{step:06d}.png"
        save_pair_visualization(vis, vis_path)
        if self.logger is not None:
            self.logger.info(f"saved visualization: {vis_path}")
