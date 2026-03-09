from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.mlp import MLP


def _knn(x: torch.Tensor, k: int) -> torch.Tensor:
    inner = -2.0 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    return pairwise_distance.topk(k=k, dim=-1)[1]


def _get_graph_feature(x: torch.Tensor, k: int) -> torch.Tensor:
    batch_size, num_dims, num_points = x.shape
    k = min(k, num_points)
    idx = _knn(x, k=k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)

    x_transposed = x.transpose(2, 1).contiguous()
    feature = x_transposed.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x_center = x_transposed.view(batch_size, num_points, 1, num_dims).expand(-1, -1, k, -1)
    feature = torch.cat((feature - x_center, x_center), dim=3)
    return feature.permute(0, 3, 1, 2).contiguous()


class DGCNNObjectEncoder(nn.Module):
    def __init__(self, emb_dim: int = 256, k: int = 20) -> None:
        super().__init__()
        self.k = int(k)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, emb_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, object_pc: torch.Tensor) -> torch.Tensor:
        x = object_pc.transpose(1, 2).contiguous()

        x = _get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1)[0]

        x = _get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1)[0]

        x = _get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1)[0]

        x = _get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        return F.adaptive_max_pool1d(x, 1).squeeze(-1)


class _BaseActionCVAE(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        object_emb_dim: int,
        latent_dim: int,
        dgcnn_k: int,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.object_encoder = DGCNNObjectEncoder(emb_dim=object_emb_dim, k=dgcnn_k)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class CrossEmbodimentActionCVAE(_BaseActionCVAE):
    def __init__(
        self,
        *,
        action_dim: int,
        num_embodiments: int = 4,
        embodiment_dim: int = 8,
        object_emb_dim: int = 256,
        latent_dim: int = 32,
        encoder_hidden_dims: tuple[int, ...] = (512, 256),
        decoder_hidden_dims: tuple[int, ...] = (256, 256),
        dgcnn_k: int = 20,
    ) -> None:
        super().__init__(
            action_dim=action_dim,
            object_emb_dim=object_emb_dim,
            latent_dim=latent_dim,
            dgcnn_k=dgcnn_k,
        )
        self.embodiment_embedding = nn.Embedding(num_embodiments, embodiment_dim)

        encoder_input_dim = action_dim + object_emb_dim + embodiment_dim
        self.encoder = MLP(
            input_dim=encoder_input_dim,
            output_dim=2 * latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation="gelu",
            dropout=0.0,
            use_layer_norm=False,
        )
        decoder_input_dim = latent_dim + object_emb_dim + embodiment_dim
        self.decoder = MLP(
            input_dim=decoder_input_dim,
            output_dim=action_dim,
            hidden_dims=decoder_hidden_dims,
            activation="gelu",
            dropout=0.0,
            use_layer_norm=False,
        )

    def encode(
        self,
        object_pc: torch.Tensor,
        action: torch.Tensor,
        embodiment_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embodiment_idx is None:
            raise ValueError("embodiment_idx is required for CrossEmbodimentActionCVAE.encode")
        object_feat = self.object_encoder(object_pc)
        emb_feat = self.embodiment_embedding(embodiment_idx)
        latent_stats = self.encoder(torch.cat([action, object_feat, emb_feat], dim=-1))
        mu, logvar = torch.chunk(latent_stats, chunks=2, dim=-1)
        return mu, logvar

    def decode(
        self,
        object_pc: torch.Tensor,
        embodiment_idx: torch.Tensor | None,
        z: torch.Tensor,
    ) -> torch.Tensor:
        if embodiment_idx is None:
            raise ValueError("embodiment_idx is required for CrossEmbodimentActionCVAE.decode")
        object_feat = self.object_encoder(object_pc)
        emb_feat = self.embodiment_embedding(embodiment_idx)
        return self.decoder(torch.cat([z, object_feat, emb_feat], dim=-1))

    def forward(
        self,
        object_pc: torch.Tensor,
        action: torch.Tensor,
        embodiment_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(object_pc, action, embodiment_idx)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(object_pc, embodiment_idx, z)
        return recon, mu, logvar, z

    @torch.no_grad()
    def sample(
        self,
        object_pc: torch.Tensor,
        embodiment_idx: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embodiment_idx is None:
            raise ValueError("embodiment_idx is required for CrossEmbodimentActionCVAE.sample")
        if z is None:
            z = torch.randn(object_pc.size(0), self.latent_dim, device=object_pc.device)
        return self.decode(object_pc, embodiment_idx, z)


class SingleEmbodimentActionCVAE(_BaseActionCVAE):
    def __init__(
        self,
        *,
        action_dim: int,
        object_emb_dim: int = 256,
        latent_dim: int = 32,
        encoder_hidden_dims: tuple[int, ...] = (512, 256),
        decoder_hidden_dims: tuple[int, ...] = (256, 256),
        dgcnn_k: int = 20,
    ) -> None:
        super().__init__(
            action_dim=action_dim,
            object_emb_dim=object_emb_dim,
            latent_dim=latent_dim,
            dgcnn_k=dgcnn_k,
        )
        encoder_input_dim = action_dim + object_emb_dim
        self.encoder = MLP(
            input_dim=encoder_input_dim,
            output_dim=2 * latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation="gelu",
            dropout=0.0,
            use_layer_norm=False,
        )
        decoder_input_dim = latent_dim + object_emb_dim
        self.decoder = MLP(
            input_dim=decoder_input_dim,
            output_dim=action_dim,
            hidden_dims=decoder_hidden_dims,
            activation="gelu",
            dropout=0.0,
            use_layer_norm=False,
        )

    def encode(
        self,
        object_pc: torch.Tensor,
        action: torch.Tensor,
        embodiment_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del embodiment_idx
        object_feat = self.object_encoder(object_pc)
        latent_stats = self.encoder(torch.cat([action, object_feat], dim=-1))
        mu, logvar = torch.chunk(latent_stats, chunks=2, dim=-1)
        return mu, logvar

    def decode(
        self,
        object_pc: torch.Tensor,
        embodiment_idx: torch.Tensor | None,
        z: torch.Tensor,
    ) -> torch.Tensor:
        del embodiment_idx
        object_feat = self.object_encoder(object_pc)
        return self.decoder(torch.cat([z, object_feat], dim=-1))

    def forward(
        self,
        object_pc: torch.Tensor,
        action: torch.Tensor,
        embodiment_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(object_pc, action, embodiment_idx)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(object_pc, None, z)
        return recon, mu, logvar, z

    @torch.no_grad()
    def sample(
        self,
        object_pc: torch.Tensor,
        embodiment_idx: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del embodiment_idx
        if z is None:
            z = torch.randn(object_pc.size(0), self.latent_dim, device=object_pc.device)
        return self.decode(object_pc, None, z)
