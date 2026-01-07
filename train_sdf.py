from typing import Dict, Tuple
import os

import wandb
import hydra
import torch
from tqdm import tqdm
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from data_process.dataset import get_dataloader
from networks.deep_sdf_decoder import Decoder
from networks.mlp import MLP
from networks.transformer import TransformerEncoder
from networks.value_encoder import ScalarValueEncoder

def _prepare_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def build_models(cfg: DictConfig, device: torch.device) -> Dict[str, torch.nn.Module]:
    joint_encoder_cfg = cfg.models.joint_encoder
    joint_encoder = MLP(
        input_dim=joint_encoder_cfg.input_dim,
        output_dim=joint_encoder_cfg.output_dim,
        hidden_dims=joint_encoder_cfg.hidden_dims,
        activation=joint_encoder_cfg.activation,
        dropout=joint_encoder_cfg.dropout,
        use_layer_norm=joint_encoder_cfg.use_layer_norm,
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

    joint_value_cfg = cfg.models.joint_value_encoder
    joint_value_encoder = ScalarValueEncoder(
        embed_dim=joint_value_cfg.embed_dim,
        use_positional=joint_value_cfg.use_positional,
        mlp_hidden=joint_value_cfg.mlp_hidden,
        activation=joint_value_cfg.activation,
        dropout=joint_value_cfg.dropout,
        use_layer_norm=joint_value_cfg.use_layer_norm,
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

    decoder_cfg = cfg.models.decoder
    decoder = Decoder(
        latent_size=decoder_cfg.latent_size,
        dims=list(decoder_cfg.dims),
        dropout=list(decoder_cfg.dropout),
        dropout_prob=decoder_cfg.dropout_prob,
        norm_layers=list(decoder_cfg.norm_layers),
        latent_in=list(decoder_cfg.latent_in),
        weight_norm=decoder_cfg.weight_norm,
        xyz_in_all=decoder_cfg.xyz_in_all,
        use_tanh=decoder_cfg.use_tanh,
        latent_dropout=decoder_cfg.latent_dropout,
    ).to(device)

    return {
        "joint_encoder": joint_encoder,
        "link_encoder": link_encoder,
        "joint_value_encoder": joint_value_encoder,
        "transformer": transformer,
        "decoder": decoder,
    }


def pooled_latent(
    transformer_out: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """Pool token embeddings into a single latent.

    mode: "mean" (default) or "max". Mask uses True for valid tokens.
    """

    if mode == "mean":
        weights = mask.float().unsqueeze(-1)
        summed = (transformer_out * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1e-6)
        return summed / denom
    if mode == "max":
        masked = transformer_out.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        return masked.max(dim=1).values
    raise ValueError(f"Unsupported pooling mode: {mode}")


def decoder_forward(
    decoder: Decoder,
    latent: torch.Tensor,
    sdf_samples: torch.Tensor,
) -> torch.Tensor:
    """Decode latent + xyz into sdf predictions."""

    xyz = sdf_samples[..., :3]  # (B, N, 3)
    B, N, _ = xyz.shape
    latent_expanded = latent.unsqueeze(1).expand(-1, N, -1)
    decoder_in = torch.cat([latent_expanded, xyz], dim=-1)
    decoder_out = decoder(decoder_in.view(B * N, -1))
    return decoder_out.view(B, N, -1)

def inference(
    batch: Dict[str, torch.Tensor],
    models: Dict[str, torch.nn.Module],
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    joint_encoder = models["joint_encoder"]
    link_encoder = models["link_encoder"]
    joint_value_encoder = models["joint_value_encoder"]
    transformer = models["transformer"]
    decoder = models["decoder"]

    sdf_samples = batch["sdf_samples"].to(device)
    chain_q = batch["chain_q"].to(device)  # (B, L-1)
    link_features = batch["link_bps_scdistances"].to(device)  # (B, L, 257)
    joint_features = batch["joint_features"].to(device)  # (B, L-1, 9)
    mask = batch["mask"].to(device)  # (B, 3*L-2)

    joint_fts = joint_encoder(joint_features)          # (B, L-1, D)
    link_fts = link_encoder(link_features)             # (B, L, D)
    joint_vals = joint_value_encoder(chain_q)          # (B, L-1, D)

    max_num_links = link_features.size(1)
    tokens = []
    for i in range(max_num_links):
        tokens.append(link_fts[:, i])
        if i < max_num_links - 1:
            tokens.append(joint_fts[:, i])
            tokens.append(joint_vals[:, i])

    token_tensor = torch.stack(tokens, dim=1)  # (B, T, D)
    key_padding_mask = ~mask  # transformer expects True for padding

    transformer_out = transformer(
        token_tensor,
        key_padding_mask=key_padding_mask,
        causal=True,
    )

    latent = pooled_latent(transformer_out[:, ::3], mask[:, ::3], mode="max")
    sdf_pred = decoder_forward(decoder, latent, sdf_samples)

    return latent, sdf_pred


def _compute_lr(schedule_cfg, epoch: int, default_lr: float) -> float:
    if schedule_cfg is None:
        return default_lr
    sched_type = str(schedule_cfg.Type).lower()
    if sched_type == "step":
        initial = float(schedule_cfg.Initial)
        interval = int(schedule_cfg.Interval)
        factor = float(schedule_cfg.Factor)
        return initial * (factor ** (epoch // max(interval, 1)))
    raise ValueError(f"Unsupported LR schedule type: {schedule_cfg.Type}")

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = _prepare_device(cfg.training.device)
    torch.manual_seed(0)

    lr = getattr(cfg.training, "lr", 1e-4)
    schedules = cfg.training.get("lr_schedules", []) if hasattr(cfg.training, "lr_schedules") else []
    active_schedule = schedules[0] if len(schedules) > 0 else None
    num_epochs = int(getattr(cfg.training, "num_epochs", 1e8))
    use_wandb = getattr(cfg.training, "wandb", {}).get("enabled", False)
    save_interval = getattr(cfg.training, "save_interval", 50)

    indices = list(range(cfg.data.indices.start, cfg.data.indices.end))
    val_indices = list(range(cfg.data.val_indices.start, cfg.data.val_indices.end))
    data_source = to_absolute_path(cfg.data.data_source)

    sdf_loader = get_dataloader(
        data_source=data_source,
        indices=indices,
        num_instances=cfg.data.num_instances,
        subsample=cfg.data.subsample,
        batch_size=cfg.data.batch_size,
        max_num_links=cfg.data.max_num_links,
        load_ram=cfg.data.load_ram,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        drop_last=cfg.data.drop_last,
    )

    val_loader = get_dataloader(
        data_source=data_source,
        indices=val_indices,
        num_instances=cfg.data.num_instances,
        subsample=cfg.data.subsample,
        batch_size=cfg.data.val_batch_size,
        max_num_links=cfg.data.max_num_links,
        load_ram=cfg.data.load_ram,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    val_iter = iter(val_loader)

    models = build_models(cfg, device)
    init_lr = _compute_lr(active_schedule, 0, lr)
    optimizer = torch.optim.Adam(
        [p for m in models.values() for p in m.parameters()], lr=init_lr
    )
    loss_fn = torch.nn.L1Loss()

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
        # step LR per epoch
        new_lr = _compute_lr(active_schedule, epoch, lr)
        for g in optimizer.param_groups:
            g["lr"] = new_lr

        for batch in tqdm(sdf_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            latent, sdf_pred = inference(batch, models, device=device)
            sdf_gt = batch["sdf_samples"].to(device)[..., 3].unsqueeze(-1)
            loss = loss_fn(sdf_pred, sdf_gt)
            loss.backward()
            optimizer.step()

            if use_wandb:
                wandb.log({"train/loss": loss.item(), "step": global_step, "epoch": epoch})
            global_step += 1

        print(f"Epoch {epoch+1}/{num_epochs} loss: {loss.item():.4f}")

        # single validation batch per epoch, cycling through the loader
        with torch.no_grad():
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_batch = next(val_iter)
            _, val_pred = inference(val_batch, models, device=device)
            val_gt = val_batch["sdf_samples"].to(device)[..., 3].unsqueeze(-1)
            val_loss = loss_fn(val_pred, val_gt).item()

        print(f"Validation loss: {val_loss:.4f}")
        if use_wandb:
            wandb.log({"val/loss": val_loss, "epoch": epoch})

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

