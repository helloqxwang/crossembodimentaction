from __future__ import annotations

from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.mlp import MLP
from networks.transformer import TransformerEncoder


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to rotation matrix.
    rot6d: (..., 6)
    returns: (..., 3, 3)
    """
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def masked_pool(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    mode: Literal["mean", "max"] = "mean",
) -> torch.Tensor:
    if mode == "mean":
        weights = mask.float().unsqueeze(-1)
        summed = (tokens * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1e-6)
        return summed / denom

    if mode == "max":
        neg_inf = torch.finfo(tokens.dtype).min
        masked_tokens = tokens.masked_fill(~mask.unsqueeze(-1), neg_inf)
        pooled = masked_tokens.max(dim=1).values
        has_valid = mask.any(dim=1, keepdim=True)
        return torch.where(has_valid, pooled, torch.zeros_like(pooled))

    raise ValueError(f"Unsupported pooling mode: {mode}")


def apply_global_transform(
    pairs: torch.Tensor,
    mask: torch.Tensor,
    t: torch.Tensor,
    r6d: torch.Tensor,
) -> torch.Tensor:
    """
    Apply global SE(3) to batched point-normal pairs.
    pairs: (B, M, 6), mask: (B, M), t: (B, 3), r6d: (B, 6)
    """
    bsz, m, _ = pairs.shape
    rot = rot6d_to_matrix(r6d)  # (B,3,3)

    pts = pairs[..., :3]
    nrms = pairs[..., 3:]

    pts_out = torch.einsum("bij,bmj->bmi", rot, pts) + t.unsqueeze(1)
    nrms_out = torch.einsum("bij,bmj->bmi", rot, nrms)
    nrms_out = F.normalize(nrms_out, dim=-1)

    out = torch.cat([pts_out, nrms_out], dim=-1)
    out = out * mask.float().unsqueeze(-1)
    if m == 0:
        return out
    return out


class SetTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        token_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        activation: str = "gelu",
        max_length: int = 128,
    ) -> None:
        super().__init__()
        self.in_proj = MLP(
            input_dim=input_dim,
            output_dim=token_dim,
            hidden_dims=(token_dim,),
            activation=activation,
            dropout=dropout,
            use_layer_norm=False,
        )
        self.encoder = TransformerEncoder(
            input_dim=token_dim,
            model_dim=model_dim,
            output_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=4.0,
            dropout=dropout,
            attn_dropout=attn_dropout,
            activation=activation,
            use_positional_encoding=False,
            max_length=max_length,
            norm_first=True,
            final_layer_norm=True,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.in_proj(x)
        return self.encoder(tokens, key_padding_mask=~mask, causal=False)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, embed_dim: int, beta: float = 0.25) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.embed_dim = int(embed_dim)
        self.beta = float(beta)
        self.embedding = nn.Embedding(self.codebook_size, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # z_e: (B, D)
        z = z_e.contiguous()
        z2 = (z ** 2).sum(dim=1, keepdim=True)  # (B,1)
        e = self.embedding.weight  # (K,D)
        e2 = (e ** 2).sum(dim=1)  # (K,)
        dist = z2 + e2.unsqueeze(0) - 2.0 * (z @ e.t())  # (B,K)
        indices = torch.argmin(dist, dim=1)  # (B,)

        z_q = self.embedding(indices)  # (B,D)
        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss = F.mse_loss(z, z_q.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        z_q_st = z + (z_q - z).detach()

        with torch.no_grad():
            one_hot = F.one_hot(indices, num_classes=self.codebook_size).float()
            avg_probs = one_hot.mean(dim=0)
            perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())
            active_codes = (avg_probs > 0).sum()

        stats = {
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss,
            "commit_loss": commit_loss,
            "indices": indices,
            "perplexity": perplexity,
            "active_codes": active_codes.float(),
        }
        return z_q_st, stats


class NFKModel(nn.Module):
    def __init__(
        self,
        *,
        pair_dim: int = 6,
        token_dim: int = 128,
        model_dim: int = 128,
        enc_num_heads: int = 4,
        enc_num_layers: int = 2,
        att_num_heads: int = 4,
        att_num_layers: int = 2,
        codebook_size: int = 32,
        vq_beta: float = 0.25,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        activation: str = "gelu",
        max_points: int = 128,
    ) -> None:
        super().__init__()
        self.pair_dim = int(pair_dim)
        self.token_dim = int(token_dim)

        # E1/E2/E3 are point-cloud encoders.
        self.e1 = SetTokenEncoder(
            input_dim=self.pair_dim,
            token_dim=self.token_dim,
            model_dim=model_dim,
            num_heads=enc_num_heads,
            num_layers=enc_num_layers,
            dropout=dropout,
            attn_dropout=attn_dropout,
            activation=activation,
            max_length=max_points,
        )
        self.e2 = SetTokenEncoder(
            input_dim=self.pair_dim,
            token_dim=self.token_dim,
            model_dim=model_dim,
            num_heads=enc_num_heads,
            num_layers=enc_num_layers,
            dropout=dropout,
            attn_dropout=attn_dropout,
            activation=activation,
            max_length=max_points,
        )
        self.e3 = SetTokenEncoder(
            input_dim=self.pair_dim,
            token_dim=self.token_dim,
            model_dim=model_dim,
            num_heads=enc_num_heads,
            num_layers=enc_num_layers,
            dropout=dropout,
            attn_dropout=attn_dropout,
            activation=activation,
            max_length=max_points,
        )

        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            embed_dim=self.token_dim,
            beta=vq_beta,
        )

        self.att = TransformerEncoder(
            input_dim=self.token_dim,
            model_dim=model_dim,
            output_dim=self.token_dim,
            num_heads=att_num_heads,
            num_layers=att_num_layers,
            mlp_ratio=4.0,
            dropout=dropout,
            attn_dropout=attn_dropout,
            activation=activation,
            use_positional_encoding=False,
            max_length=max_points + 1,
            norm_first=True,
            final_layer_norm=True,
        )

        self.att_decode = MLP(
            input_dim=self.token_dim,
            output_dim=self.pair_dim,
            hidden_dims=(self.token_dim, self.token_dim),
            activation=activation,
            dropout=dropout,
            use_layer_norm=False,
        )
        self.e4_head = MLP(
            input_dim=self.token_dim,
            output_dim=9,
            hidden_dims=(self.token_dim, self.token_dim),
            activation=activation,
            dropout=dropout,
            use_layer_norm=False,
        )

    def forward(
        self,
        zero_pairs_masked: torch.Tensor,  # (B,M,6)
        rand_pairs_masked: torch.Tensor,  # (B,M,6)
        mask_m: torch.Tensor,             # (B,M)
    ) -> Dict[str, torch.Tensor]:
        bsz, m, _ = zero_pairs_masked.shape
        if m == 0:
            zeros = torch.zeros_like(zero_pairs_masked)
            return {
                "pred_pairs_att": zeros,
                "recon_pairs": zeros,
                "t": torch.zeros((bsz, 3), dtype=zeros.dtype, device=zeros.device),
                "r6d": torch.zeros((bsz, 6), dtype=zeros.dtype, device=zeros.device),
                "vq_loss": torch.tensor(0.0, dtype=zeros.dtype, device=zeros.device),
                "codebook_loss": torch.tensor(0.0, dtype=zeros.dtype, device=zeros.device),
                "commit_loss": torch.tensor(0.0, dtype=zeros.dtype, device=zeros.device),
                "perplexity": torch.tensor(0.0, dtype=zeros.dtype, device=zeros.device),
                "active_codes": torch.tensor(0.0, dtype=zeros.dtype, device=zeros.device),
            }

        e1_tokens = self.e1(zero_pairs_masked, mask_m)  # (B,M,D)
        e2_tokens = self.e2(zero_pairs_masked, mask_m)  # (B,M,D)
        pooled_e2 = masked_pool(e2_tokens, mask_m, mode="max")  # (B,D)
        z_q, qstats = self.quantizer(pooled_e2)         # (B,D)

        att_in = torch.cat([e1_tokens, z_q.unsqueeze(1)], dim=1)  # (B,M+1,D)
        att_mask = torch.cat(
            [mask_m, torch.ones((bsz, 1), dtype=torch.bool, device=mask_m.device)],
            dim=1,
        )
        att_out = self.att(att_in, key_padding_mask=~att_mask, causal=False)  # (B,M+1,D)
        att_pairs = self.att_decode(att_out[:, :m, :])  # (B,M,6)

        e3_tokens = self.e3(rand_pairs_masked, mask_m)  # (B,M,D)
        e3_global = masked_pool(e3_tokens, mask_m, mode="max")      # (B,D)
        tr = self.e4_head(e3_global)                    # (B,9)
        t, r6d = tr[:, :3], tr[:, 3:]

        recon_pairs = apply_global_transform(att_pairs, mask_m, t, r6d)  # (B,M,6)

        return {
            "pred_pairs_att": att_pairs,
            "recon_pairs": recon_pairs,
            "t": t,
            "r6d": r6d,
            "vq_loss": qstats["vq_loss"],
            "codebook_loss": qstats["codebook_loss"],
            "commit_loss": qstats["commit_loss"],
            "perplexity": qstats["perplexity"],
            "active_codes": qstats["active_codes"],
        }


def nfk_losses(
    pred_pairs: torch.Tensor,
    target_pairs: torch.Tensor,
    mask_m: torch.Tensor,
    *,
    w_pos: float = 1.0,
    w_nrm: float = 1.0,
) -> Dict[str, torch.Tensor]:
    weights = mask_m.float()
    denom = weights.sum().clamp_min(1.0)

    pred_pos, pred_nrm = pred_pairs[..., :3], pred_pairs[..., 3:]
    tgt_pos, tgt_nrm = target_pairs[..., :3], target_pairs[..., 3:]

    pos_l1 = torch.abs(pred_pos - tgt_pos).sum(dim=-1)
    loss_pos = (pos_l1 * weights).sum() / denom

    pred_nrm = F.normalize(pred_nrm, dim=-1)
    tgt_nrm = F.normalize(tgt_nrm, dim=-1)
    cos_sim = (pred_nrm * tgt_nrm).sum(dim=-1).clamp(-1.0, 1.0)
    loss_nrm = ((1.0 - cos_sim) * weights).sum() / denom

    total = w_pos * loss_pos + w_nrm * loss_nrm
    return {
        "loss_recon_pos": loss_pos,
        "loss_recon_nrm": loss_nrm,
        "loss_recon": total,
    }
