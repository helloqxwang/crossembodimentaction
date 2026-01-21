from __future__ import annotations

from typing import Dict, Optional, Tuple
import os

import hydra
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data_process.dataset import get_dataloader
from networks.mlp import MLP
from networks.transformer import TransformerEncoder
from networks.value_encoder import ScalarValueEncoder
from networks.flow_matching import sample_interpolants, SinusoidalTimeEmbedding, rk4_integrate
from train_fk import build_models as build_fk_models


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


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    masked = diff * mask.float()
    denom = mask.sum().clamp_min(1)
    return masked.sum() / denom


def _load_fk_cfg(cfg: DictConfig) -> DictConfig:
    fk_config_path = getattr(cfg, "fk_config", None)
    if fk_config_path is None:
        fk_config_path = getattr(cfg.models, "fk_config", None)
    if fk_config_path is None:
        raise ValueError("fk_config must be set to the FK config path")
    return OmegaConf.load(to_absolute_path(str(fk_config_path)))


def build_models(cfg: DictConfig, device: torch.device) -> Dict[str, torch.nn.Module]:
    fk_models = build_fk_models(cfg, device)
    joint_encoder = fk_models["joint_encoder"]
    link_encoder = fk_models["link_encoder"]

    joint_value_encoder_cfg = cfg.models.joint_value_encoder
    joint_value_encoder = ScalarValueEncoder(
        embed_dim=joint_value_encoder_cfg.embed_dim,
        use_positional=joint_value_encoder_cfg.use_positional,
        mlp_hidden=joint_value_encoder_cfg.mlp_hidden,
        activation=joint_value_encoder_cfg.activation,
        dropout=joint_value_encoder_cfg.dropout,
        use_layer_norm=joint_value_encoder_cfg.use_layer_norm,
        num_frequencies=getattr(joint_value_encoder_cfg, "num_frequencies", 64),
        positional_head_hidden_dims=getattr(
            joint_value_encoder_cfg, "positional_head_hidden_dims", None
        ),
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

    joint_value_decoder_cfg = cfg.models.joint_value_decoder
    joint_value_decoder = MLP(
        input_dim=joint_value_decoder_cfg.input_dim,
        output_dim=1,
        hidden_dims=joint_value_decoder_cfg.hidden_dims,
        activation=joint_value_decoder_cfg.activation,
        dropout=joint_value_decoder_cfg.dropout,
        use_layer_norm=joint_value_decoder_cfg.use_layer_norm,
    ).to(device)

    time_embed_dim = cfg.flow_matching.time_embed_dim
    if time_embed_dim != transformer_cfg.input_dim:
        raise ValueError("time_embed_dim must match transformer input_dim for additive conditioning")
    time_embedder = SinusoidalTimeEmbedding(
        dim=time_embed_dim,
        mlp_hidden=cfg.flow_matching.get("time_mlp_hidden", None),
    ).to(device)

    return {
        "joint_encoder": joint_encoder,
        "link_encoder": link_encoder,
        "joint_value_encoder": joint_value_encoder,
        "transformer": transformer,
        "joint_value_decoder": joint_value_decoder,
        "time_embedder": time_embedder,
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


def flow_matching_step(
    batch: Dict[str, torch.Tensor],
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    *,
    noise_std: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    joint_encoder = models["joint_encoder"]
    link_encoder = models["link_encoder"]
    joint_value_encoder = models["joint_value_encoder"]
    transformer = models["transformer"]
    joint_value_decoder = models["joint_value_decoder"]
    time_embedder = models["time_embedder"]

    chain_q = batch["chain_q"].to(device)
    link_features = batch["link_bps_scdistances"].to(device)
    joint_features = batch["joint_features"].to(device)
    mask = batch["mask"].to(device)
    geometry_tokens = batch["sdf_tokens"].to(device)

    joint_fts = joint_encoder(joint_features)
    link_fts = link_encoder(link_features)

    xt, ut, _, t = sample_interpolants(chain_q, noise_std=noise_std)
    joint_value_tokens = joint_value_encoder(xt)

    tokens, key_padding_mask, joint_indices, joint_mask = _build_tokens(
        link_fts, joint_fts, joint_value_tokens, geometry_tokens, mask
    )

    time_emb = time_embedder(t).unsqueeze(1)
    conditioned_tokens = tokens + time_emb

    transformer_out = transformer(
        conditioned_tokens,
        key_padding_mask=key_padding_mask,
        causal=False,
    )

    joint_token_out = transformer_out[:, joint_indices]
    pred_velocity = joint_value_decoder(joint_token_out).squeeze(-1)

    loss = masked_mse(pred_velocity, ut, joint_mask)
    return loss, pred_velocity, ut, joint_mask


def run_training_val(
    val_batch: Dict[str, torch.Tensor],
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    noise_std: float,
) -> float:
    with torch.no_grad():
        loss, _, _, _ = flow_matching_step(val_batch, models, device, noise_std=noise_std)
    return loss.item()


def run_inference_val(
    val_batch: Dict[str, torch.Tensor],
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    *,
    steps: int,
    noise_std: float,
) -> float:
    joint_encoder = models["joint_encoder"]
    link_encoder = models["link_encoder"]
    joint_value_encoder = models["joint_value_encoder"]
    transformer = models["transformer"]
    joint_value_decoder = models["joint_value_decoder"]
    time_embedder = models["time_embedder"]

    chain_q = val_batch["chain_q"].to(device)
    link_features = val_batch["link_bps_scdistances"].to(device)
    joint_features = val_batch["joint_features"].to(device)
    mask = val_batch["mask"].to(device)
    geometry_tokens = val_batch["sdf_tokens"].to(device)

    joint_fts = joint_encoder(joint_features)
    link_fts = link_encoder(link_features)

    batch_size = chain_q.size(0)
    device_dtype = chain_q.dtype
    x0 = torch.randn_like(chain_q) * noise_std

    def _velocity_fn(x: torch.Tensor, t_scalar: float) -> torch.Tensor:
        time_tensor = torch.full((batch_size,), float(t_scalar), device=device, dtype=device_dtype)
        joint_value_tokens = joint_value_encoder(x)
        tokens, key_padding_mask, joint_indices, joint_mask = _build_tokens(
            link_fts, joint_fts, joint_value_tokens, geometry_tokens, mask
        )
        conditioned_tokens = tokens + time_embedder(time_tensor).unsqueeze(1)
        transformer_out = transformer(
            conditioned_tokens,
            key_padding_mask=key_padding_mask,
            causal=False,
        )
        joint_token_out = transformer_out[:, joint_indices]
        vel = joint_value_decoder(joint_token_out).squeeze(-1)
        return vel * joint_mask.float()

    with torch.no_grad():
        x1_pred = rk4_integrate(_velocity_fn, x0, steps=steps)
    joint_mask = mask[:, 2::3]  # joint value slots align with every third token
    return masked_mse(x1_pred, chain_q, joint_mask).item()


@hydra.main(config_path="conf", config_name="config_ik_fm", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = _prepare_device(cfg.training.device)
    torch.manual_seed(0)

    lr = getattr(cfg.training, "lr", 1e-4)
    num_epochs = int(getattr(cfg.training, "num_epochs", 1e6))
    use_wandb = getattr(cfg.training, "wandb", {}).get("enabled", False)
    save_interval = getattr(cfg.training.wandb, "save_interval", 50)
    val_interval = getattr(cfg.training, "validation_interval", 1000)
    inference_steps = getattr(cfg.training, "inference_steps", 10)
    noise_std = float(cfg.flow_matching.noise_std)

    indices = list(range(cfg.data.indices.start, cfg.data.indices.end))
    val_indices = list(range(cfg.data.val_indices.start, cfg.data.val_indices.end))
    data_source = to_absolute_path(cfg.data.data_source)

    ik_loader = get_dataloader(
        data_source=data_source,
        indices=indices,
        num_instances=cfg.data.num_instances,
        subsample=cfg.data.subsample,
        batch_size=cfg.data.batch_size,
        max_num_links=cfg.data.max_num_links,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        drop_last=cfg.data.drop_last,
        ik=True,
    )

    val_loader = get_dataloader(
        data_source=data_source,
        indices=val_indices,
        num_instances=cfg.data.num_instances,
        subsample=cfg.data.subsample,
        batch_size=cfg.data.val_batch_size,
        max_num_links=cfg.data.max_num_links,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        ik=True,
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
            loss, _, _, _ = flow_matching_step(batch, models, device, noise_std=noise_std)
            loss.backward()
            optimizer.step()

            if use_wandb:
                wandb.log({"train/loss": loss.item(), "step": global_step, "epoch": epoch})
            global_step += 1

            if val_interval > 0 and global_step % val_interval == 0:
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_batch = next(val_iter)

                train_style_val = run_training_val(val_batch, models, device, noise_std)
                inference_val = run_inference_val(
                    val_batch, models, device, steps=inference_steps, noise_std=noise_std
                )

                print(
                    f"[Val @ step {global_step}] train-style {train_style_val:.4f} | "
                    f"inference {inference_val:.4f}"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "val/train_style": train_style_val,
                            "val/inference": inference_val,
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
