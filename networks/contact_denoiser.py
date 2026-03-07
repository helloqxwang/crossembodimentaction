from __future__ import annotations

import math

import torch
import torch.nn as nn

from networks.mlp import MLP


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        timesteps = timesteps.float()
        device = timesteps.device
        half_dim = self.dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=device, dtype=torch.float32) / max(half_dim - 1, 1)
        )
        args = timesteps.unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((timesteps.shape[0], 1), device=device, dtype=emb.dtype)], dim=-1)
        return emb


class ContactConditionalDenoiser(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        global_cond_dim: int,
        time_embed_dim: int = 128,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = MLP(
            input_dim=time_embed_dim,
            output_dim=time_embed_dim,
            hidden_dims=(time_embed_dim * 2,),
            activation="silu",
            dropout=0.0,
            use_layer_norm=False,
        )
        self.net = MLP(
            input_dim=self.action_dim + int(global_cond_dim) + time_embed_dim,
            output_dim=self.action_dim,
            hidden_dims=tuple(int(x) for x in hidden_dims),
            activation="silu",
            dropout=0.0,
            use_layer_norm=False,
        )

    def forward(
        self,
        *,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if local_cond is not None:
            raise ValueError("local_cond is not supported for ContactConditionalDenoiser")
        if global_cond is None:
            raise ValueError("global_cond is required")
        if sample.ndim != 3 or int(sample.shape[1]) != 1 or int(sample.shape[2]) != self.action_dim:
            raise ValueError(f"Expected sample shape (B, 1, {self.action_dim}), got {tuple(sample.shape)}")

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        timestep = timestep.to(sample.device).view(-1)
        if int(timestep.shape[0]) == 1 and int(sample.shape[0]) > 1:
            timestep = timestep.expand(int(sample.shape[0]))
        elif int(timestep.shape[0]) != int(sample.shape[0]):
            raise ValueError(
                f"Expected timestep batch {sample.shape[0]}, got {tuple(timestep.shape)}"
            )

        time_feat = self.time_mlp(self.time_embed(timestep))
        sample_flat = sample[:, 0, :]
        fused = torch.cat([sample_flat, global_cond, time_feat], dim=-1)
        return self.net(fused).unsqueeze(1)
