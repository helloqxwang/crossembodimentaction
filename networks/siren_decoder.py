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
