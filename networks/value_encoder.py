from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from networks.mlp import MLP


def _geometric_frequencies(num_freqs: int, base: float = 2.0) -> torch.Tensor:
    if num_freqs <= 0:
        raise ValueError("num_freqs must be positive")
    return base ** torch.arange(num_freqs)


class ScalarValueEncoder(nn.Module):
    """Encode scalar joint values with fixed Fourier features then project to embed_dim."""

    def __init__(
        self,
        *,
        embed_dim: int = 128,
        use_positional: bool = True,
        mlp_hidden: Sequence[int] | Iterable[int] = (128,),
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        num_frequencies: int = 64,
        positional_head_hidden_dims: Sequence[int] | Iterable[int] | None = None,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")

        self.embed_dim = embed_dim
        self.use_positional = use_positional
        self.num_frequencies = num_frequencies
        head_hidden = tuple(positional_head_hidden_dims or ())

        if use_positional:
            # 2 * num_frequencies features (sin/cos), then a configurable head.
            self.register_buffer(
                "freqs",
                _geometric_frequencies(num_frequencies),
                persistent=True,
            )
            if len(head_hidden) == 0:
                self.proj = nn.Linear(2 * num_frequencies, embed_dim)
            else:
                self.proj = MLP(
                    input_dim=2 * num_frequencies,
                    output_dim=embed_dim,
                    hidden_dims=head_hidden,
                    activation=activation,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
        else:
            # Fallback to small MLP on raw scalar.
            self.mlp = MLP(
                input_dim=1,
                output_dim=embed_dim,
                hidden_dims=mlp_hidden,
                activation=activation,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )

    def _fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)
        x = x.unsqueeze(-1)  # (B, L, 1)
        scaled = x * (2 * math.pi * self.freqs.to(x))  # broadcast freqs
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected shape (batch, length), got {tuple(x.shape)}")

        if self.use_positional:
            base = self._fourier_features(x)
            B, L, D = base.shape
            out = self.proj(base.view(B * L, D))
            return out.view(B, L, -1)

        base = x.unsqueeze(-1)
        B, L, D = base.shape
        out = self.mlp(base.view(B * L, D))
        return out.view(B, L, -1)


class AxisFourierEncoder(nn.Module):
    """Encode joint rotation axes with random Fourier features then project to embed_dim."""

    def __init__(
        self,
        *,
        embed_dim: int = 256,
        num_frequencies: int = 64,
        sigma: float = 1.0,
        head_hidden_dims: Sequence[int] | Iterable[int] | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive")

        self.embed_dim = embed_dim
        self.num_frequencies = num_frequencies
        head_hidden = tuple(head_hidden_dims or ())

        # Fixed random projection matrix B ~ N(0, sigma^2)
        B = torch.randn(num_frequencies, 3) * sigma
        # Keep B in the state dict so saved checkpoints reload identical features.
        self.register_buffer("B", B, persistent=True)

        if len(head_hidden) == 0:
            self.proj = nn.Linear(2 * num_frequencies, embed_dim)
        else:
            self.proj = MLP(
                input_dim=2 * num_frequencies,
                output_dim=embed_dim,
                hidden_dims=head_hidden,
                activation=activation,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )

    def forward(self, axis: torch.Tensor) -> torch.Tensor:
        if axis.dim() != 3 or axis.size(-1) != 3:
            raise ValueError(
                f"Expected axis shape (batch, length, 3), got {tuple(axis.shape)}"
            )

        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # (B, L, K) where K=num_frequencies
        phases = 2 * math.pi * torch.matmul(axis, self.B.t())
        feats = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)

        B, L, D = feats.shape
        out = self.proj(feats.view(B * L, D))
        return out.view(B, L, -1)
