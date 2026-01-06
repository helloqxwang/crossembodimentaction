from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from networks.mlp import MLP


class ScalarValueEncoder(nn.Module):
    """Encode scalar joint values into a vector embedding.

    If ``use_positional`` is True, applies a fixed sinusoidal lift to ``embed_dim``
    followed by a small MLP. Otherwise, feeds the raw scalar through the MLP.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 128,
        use_positional: bool = True,
        mlp_hidden: Sequence[int] | Iterable[int] = (128,),
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if use_positional and embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even when using positional encoding")

        self.embed_dim = embed_dim
        self.use_positional = use_positional

        base_dim = embed_dim if use_positional else 1
        self.mlp = MLP(
            input_dim=base_dim,
            output_dim=embed_dim,
            hidden_dims=mlp_hidden,
            activation=activation,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

    def _positional_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embedding for scalar values.

        Args:
            x: tensor of shape (B, L)
        Returns:
            Tensor of shape (B, L, embed_dim)
        """

        half = self.embed_dim // 2
        # log space frequencies
        freq = torch.exp(
            torch.arange(half, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        x = x.unsqueeze(-1)
        angles = x * freq
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected shape (batch, length), got {tuple(x.shape)}")

        if self.use_positional:
            base = self._positional_embed(x)
        else:
            base = x.unsqueeze(-1)

        B, L, D = base.shape
        out = self.mlp(base.view(B * L, D))
        return out.view(B, L, -1)
