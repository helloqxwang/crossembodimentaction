from __future__ import annotations

import math
from typing import Callable, Tuple

import torch
import torch.nn as nn


def sample_interpolants(
    x1: torch.Tensor,
    *,
    noise_std: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw flow-matching pairs (x_t, u_t).

    Args:
        x1: Clean targets with arbitrary shape.
        noise_std: Standard deviation for the Gaussian prior p(x0).

    Returns:
        xt: Interpolated point along the straight path.
        ut: Target velocity x1 - x0.
        x0: Sampled noise.
        t:  Time samples in [0, 1], shape (B,).
    """

    if x1.numel() == 0:
        raise ValueError("x1 must be non-empty")

    device, dtype = x1.device, x1.dtype
    x0 = torch.randn_like(x1) * noise_std
    t = torch.rand(x1.shape[0], device=device, dtype=dtype)
    t_unsq = t.view(-1, *([1] * (x1.dim() - 1)))
    xt = (1 - t_unsq) * x0 + t_unsq * x1
    ut = x1 - x0
    return xt, ut, x0, t


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding projected to ``dim``."""

    def __init__(self, dim: int, mlp_hidden: int | None = None) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if dim % 2 != 0:
            raise ValueError("dim must be even for sinusoidal embedding")

        self.dim = dim
        hidden = mlp_hidden or dim
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() != 1:
            t = t.view(-1)
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = t.unsqueeze(1) * freq
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(emb)


def rk4_integrate(
    velocity_fn: Callable[[torch.Tensor, float], torch.Tensor],
    x0: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    """Integrate dx/dt = v(x, t) on t in [0, 1] with fixed RK4."""

    if steps <= 0:
        raise ValueError("steps must be positive")

    dt = 1.0 / steps
    x = x0
    t = 0.0
    for _ in range(steps):
        k1 = velocity_fn(x, t)
        k2 = velocity_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = velocity_fn(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = velocity_fn(x + dt * k3, t + dt)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    return x
