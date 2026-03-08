from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .appearance_aggregator import AppearanceAggregator
from .edge_decoder import EdgeDecoder
from .edge_encoder import EdgeEncoder
from .rgb_decoder import RgbDecoder
from .rgb_encoder import RgbEncoder


@dataclass
class EdgeRgbSwapConfig:
    z_channels: int = 256
    a_channels: int = 256
    base_channels: int = 64
    norm: str = "instance"
    rgb_decoder_inject: str = "adain"
    use_temporal_state: bool = False
    appearance_pool: str = "mean"


class EdgeRgbSwapSystem(nn.Module):
    def __init__(
        self,
        z_channels: int = 256,
        a_channels: int = 256,
        base_channels: int = 64,
        norm: str = "instance",
        rgb_decoder_inject: str = "adain",
        use_temporal_state: bool = False,
        appearance_pool: str = "mean",
    ) -> None:
        super().__init__()
        self.z_channels = int(z_channels)
        self.a_channels = int(a_channels)
        self.base_channels = int(base_channels)
        self.norm = norm
        self.rgb_decoder_inject = rgb_decoder_inject
        self.use_temporal_state = bool(use_temporal_state)

        self.edge_encoder = EdgeEncoder(
            in_channels=1,
            z_channels=self.z_channels,
            base_channels=self.base_channels,
            norm=self.norm,
        )
        self.rgb_encoder = RgbEncoder(
            in_channels=3,
            z_channels=self.z_channels,
            a_channels=self.a_channels,
            base_channels=self.base_channels,
            norm=self.norm,
        )
        self.edge_decoder = EdgeDecoder(
            z_channels=self.z_channels,
            base_channels=self.base_channels,
            norm=self.norm,
        )
        self.rgb_decoder = RgbDecoder(
            z_channels=self.z_channels,
            a_channels=self.a_channels,
            base_channels=self.base_channels,
            norm=self.norm,
            inject=self.rgb_decoder_inject,
        )
        self.appearance_aggregator = AppearanceAggregator(a_channels=self.a_channels, mode=appearance_pool)

    def encode_edge(self, edge: torch.Tensor) -> dict[str, torch.Tensor]:
        z_e, eps = self.edge_encoder(edge)
        return {"z": z_e, "epsilon": eps}

    def encode_rgb(self, frame: torch.Tensor) -> dict[str, torch.Tensor]:
        z_f, a = self.rgb_encoder(frame)
        return {"z": z_f, "a": a}

    def decode_edge(self, z: torch.Tensor) -> torch.Tensor:
        return self.edge_decoder(z)

    def decode_rgb(
        self,
        z: torch.Tensor,
        appearance: torch.Tensor,
        prev_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.rgb_decoder(z, appearance, prev_state=prev_state)

    def forward_pair(
        self,
        frame_t: torch.Tensor,
        frame_s: torch.Tensor,
        edge_t: torch.Tensor,
        edge_s: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        e_t = self.encode_edge(edge_t)
        e_s = self.encode_edge(edge_s)
        f_t = self.encode_rgb(frame_t)
        f_s = self.encode_rgb(frame_s)

        frame_rec_t, rec_state_t = self.decode_rgb(f_t["z"], f_t["a"])
        frame_rec_s, rec_state_s = self.decode_rgb(f_s["z"], f_s["a"])
        edge_rec_t = self.decode_edge(e_t["z"])
        edge_rec_s = self.decode_edge(e_s["z"])

        frame_swap_t, swap_state_t = self.decode_rgb(e_t["z"], f_s["a"], prev_state=rec_state_t)
        frame_swap_s, swap_state_s = self.decode_rgb(e_s["z"], f_t["a"], prev_state=rec_state_s)
        edge_swap_t = self.decode_edge(f_t["z"])
        edge_swap_s = self.decode_edge(f_s["z"])

        return {
            "frame_t": frame_t,
            "frame_s": frame_s,
            "edge_t": edge_t,
            "edge_s": edge_s,
            "z_e_t": e_t["z"],
            "z_e_s": e_s["z"],
            "eps_t": e_t["epsilon"],
            "eps_s": e_s["epsilon"],
            "z_f_t": f_t["z"],
            "z_f_s": f_s["z"],
            "a_t": f_t["a"],
            "a_s": f_s["a"],
            "frame_rec_t": frame_rec_t,
            "frame_rec_s": frame_rec_s,
            "edge_rec_t": edge_rec_t,
            "edge_rec_s": edge_rec_s,
            "frame_swap_t": frame_swap_t,
            "frame_swap_s": frame_swap_s,
            "edge_swap_t": edge_swap_t,
            "edge_swap_s": edge_swap_s,
            "rec_state_t": rec_state_t,
            "rec_state_s": rec_state_s,
            "swap_state_t": swap_state_t,
            "swap_state_s": swap_state_s,
        }

    def rollout(
        self,
        history_frames: torch.Tensor,
        future_edges: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if history_frames.ndim == 4:
            history_frames = history_frames.unsqueeze(0)
        if future_edges.ndim == 4:
            future_edges = future_edges.unsqueeze(0)
        if history_frames.dim() != 5 or future_edges.dim() != 5:
            raise ValueError(
                "Expected history_frames [B, T, C, H, W], future_edges [B, K, C, H, W]"
            )
        if history_frames.size(2) != 3:
            raise ValueError(f"history_frames must have 3 channels, got {history_frames.shape[2]}")
        if future_edges.size(2) != 1:
            raise ValueError(f"future_edges must have 1 channel, got {future_edges.shape[2]}")

        b, t_hist, _, _, _ = history_frames.shape
        _, k, _, _, _ = future_edges.shape

        _, hist_a = self.rgb_encoder(history_frames.view(-1, 3, *history_frames.shape[-2:]))
        hist_a = hist_a.view(b, t_hist, self.a_channels)

        a_star = self.appearance_aggregator(hist_a)
        state: torch.Tensor | None = None
        outputs: list[torch.Tensor] = []
        for i in range(k):
            z_edge, _ = self.edge_encoder(future_edges[:, i])
            rgb, state = self.decode_rgb(z_edge, a_star, prev_state=state if self.use_temporal_state else None)
            outputs.append(rgb)

        return {
            "future_frames": torch.stack(outputs, dim=1),
            "a_history": hist_a,
            "a_star": a_star,
            "state": state,
        }
