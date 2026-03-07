from __future__ import annotations

import torch
import torch.nn as nn


class MaskedMultiStagePointNetEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 6,
        hidden_dim: int = 128,
        out_channels: int = 128,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.out_channels = int(out_channels)
        self.num_layers = int(num_layers)

        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=False)
        self.conv_in = nn.Conv1d(self.in_channels, self.hidden_dim, kernel_size=1, bias=False)
        self.layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1, bias=False))
            self.global_layers.append(nn.Conv1d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1, bias=False))
        self.conv_out = nn.Conv1d(self.hidden_dim * self.num_layers, self.out_channels, kernel_size=1, bias=False)

    @staticmethod
    def _apply_mask(x: torch.Tensor, point_mask: torch.Tensor | None) -> torch.Tensor:
        if point_mask is None:
            return x
        return x * point_mask.unsqueeze(1).to(dtype=x.dtype)

    @staticmethod
    def _masked_max(x: torch.Tensor, point_mask: torch.Tensor | None) -> torch.Tensor:
        if point_mask is None:
            return x.max(dim=-1, keepdim=True).values

        masked = x.masked_fill(~point_mask.unsqueeze(1), torch.finfo(x.dtype).min)
        pooled = masked.max(dim=-1, keepdim=True).values
        invalid = ~point_mask.any(dim=-1, keepdim=True).unsqueeze(1)
        return torch.where(invalid, torch.zeros_like(pooled), pooled)

    def forward(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if points.ndim != 3:
            raise ValueError(f"Expected points with shape (B, N, C), got {tuple(points.shape)}")
        if int(points.shape[-1]) != self.in_channels:
            raise ValueError(f"Expected last dim {self.in_channels}, got {points.shape[-1]}")

        if point_mask is not None:
            if point_mask.ndim != 2 or point_mask.shape[:2] != points.shape[:2]:
                raise ValueError(
                    f"point_mask must have shape (B, N), got {tuple(point_mask.shape)} for points {tuple(points.shape)}"
                )
            point_mask = point_mask.bool()

        x = points.transpose(1, 2).contiguous()
        y = self._apply_mask(self.act(self.conv_in(x)), point_mask)
        feat_list = []
        for layer, global_layer in zip(self.layers, self.global_layers):
            y = self._apply_mask(self.act(layer(y)), point_mask)
            y_global = self._masked_max(y, point_mask)
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            y = self._apply_mask(self.act(global_layer(y)), point_mask)
            feat_list.append(y)
        x = self._apply_mask(self.conv_out(torch.cat(feat_list, dim=1)), point_mask)
        return self._masked_max(x, point_mask).squeeze(-1)
