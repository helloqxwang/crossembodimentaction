from __future__ import annotations

from typing import Dict, Optional, Tuple
import math
import os

import hydra
import torch
import torch.nn.functional as F
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

from data_process.dataset import get_ik_dataloader
from networks.mlp import MLP
from networks.transformer import TransformerEncoder
from networks.value_encoder import AxisFourierEncoder


def _apply_mlp_per_joint(mlp: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    B, L, D = x.shape
    out = mlp(x.reshape(B * L, D))
    return out.view(B, L, -1)


def _time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if dim <= 0:
        raise ValueError("time embedding dim must be positive")
    if t.dim() != 2 or t.size(1) != 1:
        raise ValueError(f"Expected t with shape (B,1), got {tuple(t.shape)}")
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half, 1)
    )
    args = t * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1), value=0.0)
    return emb


def _sincos_from_q(chain_q: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.sin(chain_q), torch.cos(chain_q)], dim=-1)


def _prepare_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _load_pretrained(
    models: Dict[str, torch.nn.Module],
    ckpt_path: Optional[str],
    device: torch.device,
    freeze: bool = False,
) -> None:
    if not ckpt_path:
        return
    if not os.path.isfile(ckpt_path):
        print(f"Pretrained checkpoint not found: {ckpt_path}")
        return

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_models = state.get("models", state)

    for key in ("joint_encoder", "link_encoder"):
        if key in state_models:
            try:
                models[key].load_state_dict(state_models[key], strict=False)
                print(f"Loaded pretrained weights for {key} from {ckpt_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to load {key} from {ckpt_path}: {exc}")

    if freeze:
        for key in ("joint_encoder", "link_encoder"):
            for p in models[key].parameters():
                p.requires_grad = False
        print("Frozen pretrained joint and link encoders")

def build_models(cfg: DictConfig, device: torch.device) -> Dict[str, torch.nn.Module]:
    joint_encoder_cfg = cfg.models.joint_encoder
    if getattr(joint_encoder_cfg, "use_fourier", True):
        joint_encoder = AxisFourierEncoder(
            input_dim=joint_encoder_cfg.input_dim,
            embed_dim=joint_encoder_cfg.output_dim,
            num_frequencies=joint_encoder_cfg.num_frequencies,
            sigma=getattr(joint_encoder_cfg, "sigma", 1.0),
            head_hidden_dims=getattr(joint_encoder_cfg, "head_hidden_dims", None),
            activation=getattr(joint_encoder_cfg, "activation", "gelu"),
            dropout=getattr(joint_encoder_cfg, "dropout", 0.0),
            use_layer_norm=getattr(joint_encoder_cfg, "use_layer_norm", False),
        ).to(device)
    else:
        joint_encoder = MLP(
            input_dim=joint_encoder_cfg.input_dim,
            output_dim=joint_encoder_cfg.output_dim,
            hidden_dims=getattr(
                joint_encoder_cfg,
                "mlp_hidden_dims",
                getattr(joint_encoder_cfg, "head_hidden_dims", ()),
            ),
            activation=getattr(joint_encoder_cfg, "activation", "gelu"),
            dropout=getattr(joint_encoder_cfg, "dropout", 0.0),
            use_layer_norm=getattr(joint_encoder_cfg, "use_layer_norm", False),
        ).to(device)

    link_encoder_cfg = cfg.models.link_encoder
    link_encoder = MLP(
        input_dim=link_encoder_cfg.input_dim,
        output_dim=link_encoder_cfg.output_dim,
        hidden_dims=link_encoder_cfg.hidden_dims,
        activation=link_encoder_cfg.activation,
        dropout=link_encoder_cfg.dropout,
        use_layer_norm=link_encoder_cfg.use_layer_norm,
    ).to(device)

    joint_value_encoder_cfg = cfg.models.joint_value_encoder
    joint_value_encoder = MLP(
        input_dim=joint_value_encoder_cfg.input_dim,
        output_dim=joint_value_encoder_cfg.output_dim,
        hidden_dims=getattr(joint_value_encoder_cfg, "hidden_dims", ()),
        activation=getattr(joint_value_encoder_cfg, "activation", "gelu"),
        dropout=getattr(joint_value_encoder_cfg, "dropout", 0.0),
        use_layer_norm=getattr(joint_value_encoder_cfg, "use_layer_norm", False),
    ).to(device)

    transformer_cfg = cfg.models.transformer
    transformer = TransformerEncoder(
        input_dim=transformer_cfg.input_dim,
        model_dim=transformer_cfg.model_dim,
        output_dim=transformer_cfg.output_dim,
        num_heads=transformer_cfg.num_heads,
        num_layers=transformer_cfg.num_layers,
        mlp_ratio=transformer_cfg.mlp_ratio,
        dropout=transformer_cfg.dropout,
        attn_dropout=transformer_cfg.attn_dropout,
        activation=transformer_cfg.activation,
        use_positional_encoding=transformer_cfg.use_positional_encoding,
        max_length=transformer_cfg.max_length,
        norm_first=transformer_cfg.norm_first,
        final_layer_norm=transformer_cfg.final_layer_norm,
    ).to(device)

    flow_head_cfg = cfg.models.flow_head
    flow_head = MLP(
        input_dim=flow_head_cfg.input_dim,
        output_dim=flow_head_cfg.output_dim,
        hidden_dims=flow_head_cfg.hidden_dims,
        activation=flow_head_cfg.activation,
        dropout=flow_head_cfg.dropout,
        use_layer_norm=flow_head_cfg.use_layer_norm,
    ).to(device)

    time_mlp_hidden = int(getattr(cfg.flow_matching, "time_mlp_hidden", 128))
    time_mlp = MLP(
        input_dim=int(cfg.flow_matching.time_embed_dim),
        output_dim=transformer_cfg.input_dim,
        hidden_dims=(time_mlp_hidden,),
        activation=getattr(cfg.flow_matching, "time_mlp_activation", "silu"),
        dropout=0.0,
        use_layer_norm=False,
    ).to(device)

    return {
        "joint_encoder": joint_encoder,
        "link_encoder": link_encoder,
        "joint_value_encoder": joint_value_encoder,
        "transformer": transformer,
        "flow_head": flow_head,
        "time_mlp": time_mlp,
    }


def _build_tokens(
    link_fts: torch.Tensor,
    joint_fts: torch.Tensor,
    joint_value_tokens: torch.Tensor,
    geometry_tokens: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    tokens = []
    joint_indices: list[int] = []
    max_num_links = link_fts.size(1)
    batch_size = link_fts.size(0)

    for i in range(max_num_links):
        tokens.append(link_fts[:, i])
        if i < max_num_links - 1:
            tokens.append(joint_fts[:, i])
            tokens.append(joint_value_tokens[:, i])
            joint_indices.append(len(tokens) - 1)

    tokens.append(geometry_tokens)

    token_tensor = torch.stack(tokens, dim=1)
    token_mask = torch.cat(
        [mask, torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)], dim=1
    )
    key_padding_mask = ~token_mask
    joint_mask = token_mask[:, joint_indices]
    return token_tensor, key_padding_mask, joint_indices, joint_mask


def build_ik_tokens(
    link_fts: torch.Tensor,
    joint_fts: torch.Tensor,
    joint_value_tokens: torch.Tensor,
    geometry_tokens: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    return _build_tokens(link_fts, joint_fts, joint_value_tokens, geometry_tokens, mask)


def _flow_forward(
    *,
    joint_state: torch.Tensor,
    t: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    time_embed_dim: int,
    use_compact_repr: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    joint_encoder = models["joint_encoder"]
    link_encoder = models["link_encoder"]
    joint_value_encoder = models["joint_value_encoder"]
    transformer = models["transformer"]
    flow_head = models["flow_head"]
    time_mlp = models["time_mlp"]

    if use_compact_repr:
        link_features = batch["link_features"].to(device)
    else:
        link_features = batch["link_bps_scdistances"].to(device)
    joint_features = batch["joint_features"].to(device)
    mask = batch["mask"].to(device)
    geometry_tokens = batch["sdf_tokens"].to(device)

    joint_fts = joint_encoder(joint_features)
    link_fts = link_encoder(link_features)
    joint_value_tokens = _apply_mlp_per_joint(joint_value_encoder, joint_state)

    tokens, key_padding_mask, joint_indices, joint_mask = _build_tokens(
        link_fts, joint_fts, joint_value_tokens, geometry_tokens, mask
    )
    time_embed = time_mlp(_time_embedding(t, time_embed_dim))
    tokens = tokens + time_embed[:, None, :]

    transformer_out = transformer(
        tokens,
        key_padding_mask=key_padding_mask,
        causal=False,
    )

    joint_out = transformer_out[:, joint_indices]
    v_pred = _apply_mlp_per_joint(flow_head, joint_out)

    return v_pred, joint_mask, tokens, key_padding_mask


def ik_step(
    batch: Dict[str, torch.Tensor],
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    *,
    noise_std: float,
    time_embed_dim: int,
    use_compact_repr: bool,
    q_min: float | None = None,
    q_max: float | None = None,
) -> Tuple[torch.Tensor, float | None]:
    chain_q = batch["chain_q"].to(device)
    x0 = _sincos_from_q(chain_q)
    noise = torch.randn_like(x0) * float(noise_std)
    B = chain_q.size(0)
    t = torch.rand(B, 1, device=device)
    x_t = (1.0 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * noise
    v_target = noise - x0

    v_pred, joint_mask, _, _ = _flow_forward(
        joint_state=x_t,
        t=t,
        batch=batch,
        models=models,
        device=device,
        time_embed_dim=time_embed_dim,
        use_compact_repr=use_compact_repr,
    )

    loss = (v_pred - v_target).pow(2).sum(dim=-1)
    loss = (loss * joint_mask.float()).sum() / joint_mask.sum().clamp_min(1)

    with torch.no_grad():
        x0_pred = x_t - t.unsqueeze(-1) * v_pred
        q_pred = torch.atan2(x0_pred[..., 0], x0_pred[..., 1])
        if q_min is not None and q_max is not None:
            q_pred = q_pred.clamp(q_min, q_max)
        err = torch.abs(q_pred - chain_q)
        err = (err * joint_mask.float()).sum() / joint_mask.sum().clamp_min(1)

    return loss, float(err.item())


def infer_ik_flow(
    batch: Dict[str, torch.Tensor],
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    *,
    noise_std: float,
    time_embed_dim: int,
    steps: int = 10,
    step_size: float | None = None,
    use_compact_repr: bool,
    return_tokens: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """ODE integration for flow-matching model in joint sin/cos space."""
    if steps <= 0:
        raise ValueError("steps must be >= 1 for flow inference")

    chain_q = batch["chain_q"].to(device)
    x = torch.randn_like(_sincos_from_q(chain_q)) * float(noise_std)

    dt = step_size if step_size is not None else 1.0 / float(steps)
    last_tokens = None
    last_key_padding = None
    joint_mask = None

    for step in range(steps):
        t_val = 1.0 - step / float(steps)
        t = torch.full((x.size(0), 1), t_val, device=device, dtype=x.dtype)
        v_pred, joint_mask, tokens, key_padding_mask = _flow_forward(
            joint_state=x,
            t=t,
            batch=batch,
            models=models,
            device=device,
            time_embed_dim=time_embed_dim,
            use_compact_repr=use_compact_repr,
        )
        x = x - v_pred * dt
        if joint_mask is not None:
            x = torch.where(joint_mask.unsqueeze(-1), x, torch.zeros_like(x))
        last_tokens = tokens
        last_key_padding = key_padding_mask

    if joint_mask is None:
        raise RuntimeError("Flow inference failed to produce joint mask")

    if return_tokens:
        return x, joint_mask, last_tokens, last_key_padding
    return x, joint_mask


@hydra.main(config_path="conf", config_name="config_ik_fm", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = _prepare_device(cfg.training.device)
    torch.manual_seed(0)

    lr = getattr(cfg.training, "lr", 1e-4)
    num_epochs = int(getattr(cfg.training, "num_epochs", 1e6))
    use_wandb = getattr(cfg.training, "wandb", {}).get("enabled", False)
    save_interval = getattr(cfg.training.wandb, "save_interval", 50)
    val_interval = getattr(cfg.training, "validation_interval", 1000)
    q_min = float(getattr(cfg.training, "q_min", -2.0 * math.pi / 3.0))
    q_max = float(getattr(cfg.training, "q_max", 2.0 * math.pi / 3.0))
    noise_std = float(getattr(cfg.flow_matching, "noise_std", 1.0))
    time_embed_dim = int(getattr(cfg.flow_matching, "time_embed_dim", 128))

    indices = list(range(cfg.data.indices.start, cfg.data.indices.end))
    val_indices = list(range(cfg.data.val_indices.start, cfg.data.val_indices.end))
    data_source = to_absolute_path(cfg.data.data_source)
    tokens_dir = to_absolute_path(getattr(cfg.data, "sdf_token_dir", "./data/sdf_tokens"))
    use_compact_repr = bool(getattr(cfg.models.link_encoder, "compact_repr", False))

    ik_loader = get_ik_dataloader(
        data_source=data_source,
        indices=indices,
        num_instances=cfg.data.num_instances,
        tokens_dir=tokens_dir,
        batch_size=cfg.data.batch_size,
        max_num_links=cfg.data.max_num_links,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        drop_last=cfg.data.drop_last,
        link_compact_repr=use_compact_repr,
    )

    val_loader = get_ik_dataloader(
        data_source=data_source,
        indices=val_indices,
        num_instances=cfg.data.num_instances,
        tokens_dir=tokens_dir,
        batch_size=cfg.data.val_batch_size,
        max_num_links=cfg.data.max_num_links,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        link_compact_repr=use_compact_repr,
    )
    val_iter = iter(val_loader)

    models = build_models(cfg, device)
    _load_pretrained(
        models,
        getattr(cfg.training, "pretrained_ckpt", None),
        device,
        freeze=getattr(cfg.training, "freeze_pretrained", False),
    )

    optimizer = torch.optim.Adam(
        [p for m in models.values() for p in m.parameters() if p.requires_grad], lr=lr
    )

    save_dir = os.path.join(cfg.training.wandb.save_dir, cfg.training.wandb.run_name)
    os.makedirs(save_dir, exist_ok=True)

    if use_wandb:
        wandb.init(
            project=cfg.training.wandb.project,
            name=cfg.training.wandb.run_name,
            config=dict(cfg),
        )

    global_step = 0

    for epoch in range(num_epochs):
        for batch in tqdm(ik_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            loss, err = ik_step(
                batch,
                models,
                device,
                noise_std=noise_std,
                time_embed_dim=time_embed_dim,
                use_compact_repr=use_compact_repr,
                q_min=q_min,
                q_max=q_max,
            )
            loss.backward()
            optimizer.step()

            if use_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/angle_err": err,
                        "step": global_step,
                        "epoch": epoch,
                    }
                )
            global_step += 1

            if val_interval > 0 and global_step % val_interval == 0:
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_batch = next(val_iter)

                with torch.no_grad():
                    val_loss, val_err = ik_step(
                        val_batch,
                        models,
                        device,
                        noise_std=noise_std,
                        time_embed_dim=time_embed_dim,
                        use_compact_repr=use_compact_repr,
                        q_min=q_min,
                        q_max=q_max,
                    )

                print(
                    f"[Val @ step {global_step}] loss {val_loss:.4f} | angle_err {val_err:.4f}"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/angle_err": val_err,
                            "step": global_step,
                            "epoch": epoch,
                        }
                    )

        print(f"Epoch {epoch+1}/{num_epochs} loss: {loss.item():.4f}")

        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "models": {k: m.state_dict() for k, m in models.items()},
                    "optimizer": optimizer.state_dict(),
                    "config": dict(cfg),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
