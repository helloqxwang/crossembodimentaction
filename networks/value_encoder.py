from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from networks.mlp import MLP


class ScalarValueEncoder(nn.Module):
    """Encode scalar joint values via either fixed Fourier features or legacy sinusoidal lift."""

    def __init__(
        self,
        *,
        embed_dim: int = 128,
        use_positional: bool = True,
        positional_type: str = "fourier",  # "fourier" or "legacy"
        mlp_hidden: Sequence[int] | Iterable[int] = (128,),
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        num_frequencies: int = 8,
        positional_head_hidden_dims: Sequence[int] | Iterable[int] | None = None,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")

        self.embed_dim = embed_dim
        self.use_positional = use_positional
        self.positional_type = positional_type.lower()
        self.num_frequencies = num_frequencies
        head_hidden = tuple(positional_head_hidden_dims or ())

        if use_positional:
            if self.positional_type not in {"fourier", "legacy"}:
                raise ValueError("positional_type must be 'fourier' or 'legacy'")

            if self.positional_type == "fourier":
                # 2 * num_frequencies features (sin/cos), then a configurable head.
                self.register_buffer(
                    "freqs",
                    2 ** torch.arange(num_frequencies),
                    persistent=True,
                )
                head_in = 2 * num_frequencies
            else:
                if embed_dim % 2 != 0:
                    raise ValueError("embed_dim must be even for legacy positional encoding")
                half = embed_dim // 2
                freq = torch.exp(
                    torch.arange(half) * (-math.log(10000.0) / max(half - 1, 1))
                )
                self.register_buffer("legacy_freq", freq, persistent=True)
                head_in = embed_dim

            if len(head_hidden) == 0:
                self.proj = nn.Linear(head_in, embed_dim)
            else:
                self.proj = MLP(
                    input_dim=head_in,
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
        # x: (B, L) in radians, range roughly [-2/3 pi, 2/3 pi]
        normed = x * (3.0 / (2.0 * math.pi))  # scale to ~[-1, 1]
        normed = normed.unsqueeze(-1)  # (B, L, 1)
        scaled = normed * (math.pi * self.freqs.to(x))  # broadcast freqs
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected shape (batch, length), got {tuple(x.shape)}")

        if self.use_positional:
            if self.positional_type == "fourier":
                base = self._fourier_features(x)
            else:
                # Legacy sinusoidal lift
                x_exp = x.unsqueeze(-1)
                angles = x_exp * self.legacy_freq.to(x)
                base = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

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
