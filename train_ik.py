from typing import Dict, Optional, Tuple
import os

import hydra
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

from data_process.dataset import get_dataloader
from networks.mlp import MLP
from networks.transformer import TransformerEncoder


def _prepare_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


class JointValuePlaceholder(torch.nn.Module):
    """Learnable embedding used as placeholder for joint value tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(1, dim)
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        # Returns (B, dim) filled with the single learned embedding
        return self.embedding.weight[0].unsqueeze(0).expand(batch_size, -1)


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

    placeholder = JointValuePlaceholder(transformer_cfg.input_dim).to(device)

    return {
        "joint_encoder": joint_encoder,
        "link_encoder": link_encoder,
        "transformer": transformer,
        "joint_value_decoder": joint_value_decoder,
        "joint_value_placeholder": placeholder,
    }


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

    state = torch.load(ckpt_path, map_location=device)
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


def inference(
    batch: Dict[str, torch.Tensor],
    models: Dict[str, torch.nn.Module],
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    joint_encoder = models["joint_encoder"]
    link_encoder = models["link_encoder"]
    transformer = models["transformer"]
    joint_value_decoder = models["joint_value_decoder"]
    placeholder = models["joint_value_placeholder"]

    chain_q = batch["chain_q"].to(device)  # (B, L-1)
    link_features = batch["link_bps_scdistances"].to(device)  # (B, L, 1024)
    joint_features = batch["joint_features"].to(device)  # (B, L-1, 9)
    mask = batch["mask"].to(device)  # (B, 3*L-2)

    joint_fts = joint_encoder(joint_features)          # (B, L-1, D)
    link_fts = link_encoder(link_features)             # (B, L, D)

    max_num_links = link_features.size(1)
    batch_size, token_dim = link_fts.shape[0], link_fts.shape[-1]

    placeholder_vec = placeholder(batch_size)          # (B, D)
    placeholder_tokens = placeholder_vec.unsqueeze(1).expand(-1, max_num_links - 1, -1)

    # Place holder
    geometry_token = torch.randn(batch_size, token_dim, device=device)

    tokens = []
    joint_value_token_indices = []

    for i in range(max_num_links):
        tokens.append(link_fts[:, i])
        if i < max_num_links - 1:
            tokens.append(joint_fts[:, i])
            tokens.append(placeholder_tokens[:, i])
            joint_value_token_indices.append(len(tokens) - 1)

    tokens.append(geometry_token)

    token_tensor = torch.stack(tokens, dim=1)  # (B, T, D)

    # Extend mask to include the global geometry token (always valid)
    token_mask = torch.cat(
        [mask, torch.ones(batch_size, 1, device=device, dtype=torch.bool)], dim=1
    )
    key_padding_mask = ~token_mask

    transformer_out = transformer(
        token_tensor,
        key_padding_mask=key_padding_mask,
        causal=True,
    )

    joint_token_out = transformer_out[:, joint_value_token_indices]  # (B, L-1, D)
    joint_mask = token_mask[:, joint_value_token_indices]            # (B, L-1)

    joint_raw = joint_value_decoder(joint_token_out).squeeze(-1)     # (B, L-1)
    joint_preds = torch.tanh(joint_raw) * torch.pi                   # squash to [-pi, pi]

    return joint_preds, joint_mask


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(pred - target)
    masked = diff * mask.float()
    denom = mask.sum().clamp_min(1)
    return masked.sum() / denom


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


@hydra.main(config_path="conf", config_name="config_ik", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = _prepare_device(cfg.training.device)
    torch.manual_seed(0)

    lr = getattr(cfg.training, "lr", 1e-4)
    schedules = cfg.training.get("lr_schedules", []) if hasattr(cfg.training, "lr_schedules") else []
    active_schedule = schedules[0] if len(schedules) > 0 else None
    num_epochs = int(getattr(cfg.training, "num_epochs", 1e8))
    use_wandb = getattr(cfg.training, "wandb", {}).get("enabled", False)
    save_interval = getattr(cfg.training.wandb, "save_interval", 50)

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
    _load_pretrained(
        models,
        getattr(cfg.training, "pretrained_ckpt", None),
        device,
        freeze=getattr(cfg.training, "freeze_pretrained", False),
    )

    init_lr = lr
    optimizer = torch.optim.Adam(
        [p for m in models.values() for p in m.parameters() if p.requires_grad], lr=init_lr
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
        new_lr = _compute_lr(active_schedule, epoch, lr) if active_schedule is not None else lr
        for g in optimizer.param_groups:
            g["lr"] = new_lr

        for batch in tqdm(ik_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            joint_pred, joint_mask = inference(batch, models, device=device)
            joint_gt = batch["chain_q"].to(device)
            loss = masked_l1(joint_pred, joint_gt, joint_mask)
            loss.backward()
            optimizer.step()

            if use_wandb:
                wandb.log({"train/loss": loss.item(), "step": global_step, "epoch": epoch})
            global_step += 1

        print(f"Epoch {epoch+1}/{num_epochs} loss: {loss.item():.4f}")

        with torch.no_grad():
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_batch = next(val_iter)
            val_pred, val_mask = inference(val_batch, models, device=device)
            val_gt = val_batch["chain_q"].to(device)
            val_loss = masked_l1(val_pred, val_gt, val_mask).item()

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
