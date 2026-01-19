from __future__ import annotations

from typing import Iterable, Sequence

import math
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """SIREN layer with sine activation and tailored initialization."""

    def __init__(self, in_features: int, out_features: int, *, is_first: bool, omega_0: float = 30.0) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)

        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SirenDecoder(nn.Module):
    """SIREN MLP decoder for SDF prediction from latent+xyz."""

    def __init__(
        self,
        *,
        latent_size: int,
        hidden_dims: Sequence[int] | Iterable[int],
        omega_0: float = 30.0,
        outermost_linear: bool = True,
    ) -> None:
        super().__init__()

        hidden_dims = tuple(hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty for SirenDecoder")

        dims = (latent_size + 3,) + hidden_dims
        layers: list[nn.Module] = []

        for idx in range(len(dims) - 1):
            layers.append(
                SineLayer(
                    dims[idx],
                    dims[idx + 1],
                    is_first=(idx == 0),
                    omega_0=omega_0,
                )
            )

        if outermost_linear:
            final = nn.Linear(dims[-1], 1)
            with torch.no_grad():
                bound = math.sqrt(6.0 / dims[-1]) / omega_0
                final.weight.uniform_(-bound, bound)
            layers.append(final)
        else:
            layers.append(SineLayer(dims[-1], 1, is_first=False, omega_0=omega_0))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError(f"Expected input with at least 2 dims (batch, features), got shape {tuple(x.shape)}")
        return self.net(x)


class SirenSplitDecoder(nn.Module):
    """SIREN decoder with separate query (xyz) and latent encoders."""

    def __init__(
        self,
        *,
        latent_size: int,
        hidden_features: int = 256,
        omega_0: float = 30.0,
        num_state_layers: int = 4,
        num_fusion_layers: int = 3,
    ) -> None:
        super().__init__()

        half_hidden = max(1, hidden_features // 2)

        self.query_layer = SineLayer(3, half_hidden, is_first=True, omega_0=omega_0)
        state_layers = [SineLayer(latent_size, half_hidden, is_first=True, omega_0=omega_0)]
        for _ in range(max(0, num_state_layers - 1)):
            state_layers.append(SineLayer(half_hidden, half_hidden, is_first=False, omega_0=omega_0))
        self.state_layers = nn.Sequential(*state_layers)

        fusion_layers = []
        in_dim = hidden_features
        for _ in range(max(1, num_fusion_layers)):
            fusion_layers.append(SineLayer(in_dim, hidden_features, is_first=False, omega_0=omega_0))
            in_dim = hidden_features
        self.fusion_layers = nn.Sequential(*fusion_layers)
        self.final = nn.Linear(hidden_features, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features) / omega_0
            self.final.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError(f"Expected input with at least 2 dims (batch, features), got shape {tuple(x.shape)}")
        xyz = x[:, -3:]
        latent = x[:, :-3]
        q_feat = self.query_layer(xyz)
        s_feat = self.state_layers(latent)
        fused = torch.cat([q_feat, s_feat], dim=-1)
        fused = self.fusion_layers(fused)
        return self.final(fused)
