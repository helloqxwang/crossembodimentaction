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

    return {
        "joint_encoder": joint_encoder,
        "link_encoder": link_encoder,
        "transformer": transformer,
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


def _discretize_joint_values(
    chain_q: torch.Tensor,
    *,
    num_bins: int,
    q_min: float,
    q_max: float,
) -> torch.Tensor:
    if q_max <= q_min:
        raise ValueError("q_max must be greater than q_min")
    q_clamped = chain_q.clamp(q_min, q_max)
    scaled = (q_clamped - q_min) / (q_max - q_min)
    idx = torch.floor(scaled * num_bins).long().clamp(0, num_bins - 1)
    return idx


def _corrupt_one_hot(
    one_hot: torch.Tensor,
    *,
    corrupt_prob: float,
    num_bins: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if corrupt_prob <= 0:
        mask = torch.zeros(one_hot.shape[:2], dtype=torch.bool, device=one_hot.device)
        return one_hot, mask
    mask = torch.rand(one_hot.shape[:2], device=one_hot.device) < corrupt_prob
    rand_idx = torch.randint(0, num_bins, mask.shape, device=one_hot.device)
    corrupted = one_hot.clone()
    corrupted[mask] = F.one_hot(rand_idx[mask], num_bins).float()
    return corrupted, mask


def ik_step(
    batch: Dict[str, torch.Tensor],
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    *,
    num_bins: int,
    q_min: float,
    q_max: float,
    corrupt_prob: float,
    use_compact_repr: bool,
    supervise_only_corrupted: bool,
) -> Tuple[torch.Tensor, float]:
    joint_encoder = models["joint_encoder"]
    link_encoder = models["link_encoder"]
    transformer = models["transformer"]

    chain_q = batch["chain_q"].to(device)
    if use_compact_repr:
        link_features = batch["link_features"].to(device)
    else:
        link_features = batch["link_bps_scdistances"].to(device)
    joint_features = batch["joint_features"].to(device)
    mask = batch["mask"].to(device)
    geometry_tokens = batch["sdf_tokens"].to(device)

    joint_fts = joint_encoder(joint_features)
    link_fts = link_encoder(link_features)

    gt_idx = _discretize_joint_values(chain_q, num_bins=num_bins, q_min=q_min, q_max=q_max)
    gt_one_hot = F.one_hot(gt_idx, num_bins).float()
    corrupted_tokens, corrupt_mask = _corrupt_one_hot(
        gt_one_hot, corrupt_prob=corrupt_prob, num_bins=num_bins
    )

    tokens, key_padding_mask, joint_indices, joint_mask = _build_tokens(
        link_fts, joint_fts, corrupted_tokens, geometry_tokens, mask
    )

    transformer_out = transformer(
        tokens,
        key_padding_mask=key_padding_mask,
        causal=False,
    )

    joint_logits = transformer_out[:, joint_indices]
    B, L, D = joint_logits.shape
    logits_flat = joint_logits.reshape(B * L, D)
    gt_flat = gt_idx.reshape(B * L)

    valid_mask = joint_mask.reshape(B * L)
    if supervise_only_corrupted and corrupt_prob > 0:
        corrupt_flat = corrupt_mask.reshape(B * L)
        valid_mask = valid_mask & corrupt_flat
        if valid_mask.sum() == 0:
            valid_mask = joint_mask.reshape(B * L)

    loss_all = F.cross_entropy(logits_flat, gt_flat, reduction="none")
    loss = (loss_all * valid_mask.float()).sum() / valid_mask.sum().clamp_min(1)

    with torch.no_grad():
        pred_idx = joint_logits.argmax(dim=-1)
        acc_mask = valid_mask.reshape(B, L)
        correct = (pred_idx == gt_idx) & acc_mask
        acc = correct.sum().float() / acc_mask.sum().clamp_min(1)

    return loss, float(acc.item())


@hydra.main(config_path="conf", config_name="config_ik_fm", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = _prepare_device(cfg.training.device)
    torch.manual_seed(0)

    lr = getattr(cfg.training, "lr", 1e-4)
    num_epochs = int(getattr(cfg.training, "num_epochs", 1e6))
    use_wandb = getattr(cfg.training, "wandb", {}).get("enabled", False)
    save_interval = getattr(cfg.training.wandb, "save_interval", 50)
    val_interval = getattr(cfg.training, "validation_interval", 1000)
    num_bins = int(getattr(cfg.models, "sdf_latent_dim", cfg.models.transformer.input_dim))
    if cfg.models.transformer.input_dim != num_bins or cfg.models.transformer.output_dim != num_bins:
        raise ValueError(
            "transformer.input_dim and transformer.output_dim must equal num_bins "
            "for direct one-hot inputs and logits outputs"
        )
    q_min = float(getattr(cfg.training, "q_min", -2.0 * math.pi / 3.0))
    q_max = float(getattr(cfg.training, "q_max", 2.0 * math.pi / 3.0))
    corrupt_prob = float(getattr(cfg.training, "corrupt_prob", 0.15))
    supervise_only_corrupted = bool(getattr(cfg.training, "supervise_only_corrupted", True))

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
            loss, acc = ik_step(
                batch,
                models,
                device,
                num_bins=num_bins,
                q_min=q_min,
                q_max=q_max,
                corrupt_prob=corrupt_prob,
                use_compact_repr=use_compact_repr,
                supervise_only_corrupted=supervise_only_corrupted,
            )
            loss.backward()
            optimizer.step()

            if use_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/acc": acc,
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
                    val_loss, val_acc = ik_step(
                        val_batch,
                        models,
                        device,
                        num_bins=num_bins,
                        q_min=q_min,
                        q_max=q_max,
                        corrupt_prob=0.0,
                        use_compact_repr=use_compact_repr,
                        supervise_only_corrupted=False,
                    )

                print(
                    f"[Val @ step {global_step}] loss {val_loss:.4f} | acc {val_acc:.4f}"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/acc": val_acc,
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
