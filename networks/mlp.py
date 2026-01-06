from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    """General-purpose multilayer perceptron with configurable depth/width."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] | Iterable[int] = (128, 128),
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        final_activation: str | None = None,
    ) -> None:
        super().__init__()

        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")

        hidden_dims = tuple(hidden_dims)
        dims = (input_dim,) + hidden_dims + (output_dim,)

        layers: list[nn.Module] = []
        act = _get_activation(activation)

        for idx in range(len(dims) - 1):
            in_dim, out_dim = dims[idx], dims[idx + 1]
            layers.append(nn.Linear(in_dim, out_dim))

            is_last = idx == len(dims) - 2
            if not is_last:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(act)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        if final_activation is not None:
            layers.append(_get_activation(final_activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError(f"Expected input with at least 2 dims (batch, features), got shape {tuple(x.shape)}")
        return self.net(x)


if __name__ == "__main__":
    torch.manual_seed(0)

    mlp = MLP(
        input_dim=16,
        output_dim=4,
        hidden_dims=(32, 64),
        activation="relu",
        dropout=0.1,
        use_layer_norm=True,
    )

    sample = torch.randn(3, 16)
    output = mlp(sample)
    print("Output shape:", output.shape)
    print("First row:", output[0])
