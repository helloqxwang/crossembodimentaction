from typing import Dict, Tuple
import os

import wandb
import hydra
import torch
import torch.nn.functional as F
from tqdm import tqdm
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from data_process.dataset import get_dataloader
from networks.deep_sdf_decoder import Decoder
from networks.siren_decoder import SirenDecoder
from networks.mlp import MLP
from networks.transformer import TransformerEncoder, SetAggregator
from networks.value_encoder import ScalarValueEncoder, AxisFourierEncoder

def _prepare_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


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

    joint_value_cfg = cfg.models.joint_value_encoder
    joint_value_encoder = ScalarValueEncoder(
        embed_dim=joint_value_cfg.embed_dim,
        use_positional=joint_value_cfg.use_positional,
        positional_type=getattr(joint_value_cfg, "positional_type", "fourier"),
        mlp_hidden=joint_value_cfg.mlp_hidden,
        activation=joint_value_cfg.activation,
        dropout=joint_value_cfg.dropout,
        use_layer_norm=joint_value_cfg.use_layer_norm,
        num_frequencies=getattr(joint_value_cfg, "num_frequencies", 64),
        positional_head_hidden_dims=getattr(
            joint_value_cfg, "positional_head_hidden_dims", None
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

    # aggregator_cfg = cfg.models.aggregator
    # aggregator = SetAggregator(
    #     token_dim=transformer_cfg.output_dim,
    #     num_layers=aggregator_cfg.num_layers,
    #     num_heads=aggregator_cfg.num_heads,
    #     mlp_ratio=aggregator_cfg.mlp_ratio,
    #     dropout=aggregator_cfg.dropout,
    #     attn_dropout=aggregator_cfg.attn_dropout,
    #     activation=aggregator_cfg.activation,
    #     norm_first=aggregator_cfg.norm_first,
    #     final_layer_norm=aggregator_cfg.final_layer_norm,
    #     max_length=aggregator_cfg.max_length,
    # ).to(device)

    decoder_cfg = cfg.models.decoder
    decoder_type = str(getattr(decoder_cfg, "type", "siren")).lower()
    if decoder_type == "siren":
        decoder = SirenDecoder(
            latent_size=decoder_cfg.latent_size,
            hidden_dims=list(decoder_cfg.dims),
            omega_0=getattr(decoder_cfg, "omega_0", 30.0),
            outermost_linear=getattr(decoder_cfg, "outermost_linear", True),
        ).to(device)
    else:
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
        # "aggregator": aggregator,
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
    decoder: torch.nn.Module,
    latent: torch.Tensor,
    xyz: torch.Tensor,
) -> torch.Tensor:
    """Decode latent + xyz into sdf predictions."""
    B, N, _ = xyz.shape
    latent_expanded = latent.unsqueeze(1).expand(-1, N, -1)
    decoder_in = torch.cat([latent_expanded, xyz], dim=-1)
    decoder_out = decoder(decoder_in.view(B * N, -1))
    return decoder_out.view(B, N, -1)

def inference(
    batch: Dict[str, torch.Tensor],
    models: Dict[str, torch.nn.Module],
    cfg: DictConfig,
    device: torch.device = torch.device("cpu"),
    coords_override: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    joint_encoder = models["joint_encoder"]
    link_encoder = models["link_encoder"]
    joint_value_encoder = models["joint_value_encoder"]
    transformer = models["transformer"]
    # aggregator = models["aggregator"]
    decoder = models["decoder"]

    sdf_samples = batch["sdf_samples"].to(device)
    chain_q = batch["chain_q"].to(device)  # (B, L-1)
    if cfg.models.link_encoder.compact_repr:
        link_features = batch['link_features'].to(device)  # (B, L, 4)
    else:
        link_features = batch["link_bps_scdistances"].to(device)  # (B, L, 256 * 4)
    joint_features = batch["joint_features"].to(device)  # (B, L-1, 9)
    mask = batch["mask"].to(device)  # (B, 3*L-2)
    links_mask = batch["link_mask"].to(device)  # (B, L)

    joint_fts = joint_encoder(joint_features)
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
        causal=False,
    )

    link_tokens = transformer_out[:, ::3]  # (B, L, D) link positions only
    # latent = aggregator(link_tokens, links_mask)
    latent = pooled_latent(link_tokens, links_mask, mode="max")
    xyz = coords_override if coords_override is not None else sdf_samples[..., :3]
    sdf_pred = decoder_forward(decoder, latent, xyz)

    return latent, sdf_pred


def siren_sdf_loss(
    pred_sdf: torch.Tensor,
    coords: torch.Tensor,
    gt_sdf: torch.Tensor,
    gt_normals: torch.Tensor,
    normal_mask: torch.Tensor | None = None,
    *,
    sdf_weight: float = 3e3,
    inter_weight: float = 1e2,
    normal_weight: float = 1e2,
    grad_weight: float = 5e1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    gradients = torch.autograd.grad(
        pred_sdf,
        coords,
        grad_outputs=torch.ones_like(pred_sdf),
        create_graph=True,
    )[0]

    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    if normal_mask is None:
        normal_mask = gt_sdf != -1
    normal_constraint = torch.where(
        normal_mask,
        1 - F.cosine_similarity(gradients, gt_normals, dim=-1)[..., None],
        torch.zeros_like(pred_sdf),
    )
    grad_constraint = torch.abs(gradients.norm(dim=-1, keepdim=True) - 1)

    loss_sdf = torch.abs(sdf_constraint).mean() * sdf_weight
    loss_inter = inter_constraint.mean() * inter_weight
    loss_normal = normal_constraint.mean() * normal_weight
    loss_grad = grad_constraint.mean() * grad_weight
    loss = loss_sdf + loss_inter + loss_normal + loss_grad

    return loss, {
        "loss/sdf": loss_sdf.item() / sdf_weight,
        "loss/inter": loss_inter.item() / inter_weight,
        "loss/normal": loss_normal.item() / normal_weight,
        "loss/grad": loss_grad.item() / grad_weight,
    }

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


def _cfg_get(cfg, key: str, default):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

@hydra.main(config_path="conf/conf_fk", config_name="config_fourier_2dfourier_100", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = _prepare_device(cfg.training.device)
    torch.manual_seed(0)

    lr = getattr(cfg.training, "lr", 1e-4)
    schedules = cfg.training.get("lr_schedules", []) if hasattr(cfg.training, "lr_schedules") else []
    active_schedule = schedules[0] if len(schedules) > 0 else None
    num_epochs = int(getattr(cfg.training, "num_epochs", 1e8))
    use_wandb = getattr(cfg.training, "wandb", {}).get("enabled", False)
    test_interval = getattr(cfg.training, "test_interval", -1)
    save_interval = getattr(cfg.training, "save_interval", 50)

    indices = list(range(cfg.data.indices.start, cfg.data.indices.end))
    val_indices = list(range(cfg.data.val_indices.start, cfg.data.val_indices.end))
    data_source = to_absolute_path(cfg.data.data_source)
    loss_type = str(getattr(cfg.training, "loss_type", "siren")).lower()
    sdf_mode = str(getattr(cfg.data, "sdf_mode", "siren")).lower()
    off_surface_center = str(getattr(cfg.data, "off_surface_center", "zero")).lower()
    sdf_target_mode = str(getattr(cfg.data, "sdf_target_mode", "fake")).lower()
    tsdf_band = float(getattr(cfg.data, "tsdf_band", 0.1))
    if loss_type == "siren" and sdf_mode != "siren":
        raise ValueError("siren loss requires data.sdf_mode = 'siren' to provide normals and surface flags")

    sdf_loader = get_dataloader(
        data_source=data_source,
        indices=indices,
        num_instances=cfg.data.num_instances,
        subsample=cfg.data.subsample,
        batch_size=cfg.data.batch_size,
        max_num_links=cfg.data.max_num_links,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        drop_last=cfg.data.drop_last,
        sdf_mode=sdf_mode,
        off_surface_center=off_surface_center,
        sdf_target_mode=sdf_target_mode,
        tsdf_band=tsdf_band,
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
        sdf_mode=sdf_mode,
        off_surface_center=off_surface_center,
        sdf_target_mode=sdf_target_mode,
        tsdf_band=tsdf_band,
    )
    val_iter = iter(val_loader)

    models = build_models(cfg, device)
    # init_lr = _compute_lr(active_schedule, 0, lr)
    init_lr = lr
    optimizer = torch.optim.Adam(
        [p for m in models.values() for p in m.parameters()], lr=init_lr
    )
    loss_fn = torch.nn.L1Loss()
    siren_cfg = getattr(cfg.training, "siren_loss", {})
    siren_weights = {
        "sdf_weight": float(_cfg_get(siren_cfg, "sdf_weight", 3e3)),
        "inter_weight": float(_cfg_get(siren_cfg, "inter_weight", 1e2)),
        "normal_weight": float(_cfg_get(siren_cfg, "normal_weight", 1e2)),
        "grad_weight": float(_cfg_get(siren_cfg, "grad_weight", 5e1)),
    }

    save_dir = os.path.join(cfg.training.wandb.save_dir, cfg.training.wandb.run_name)
    os.makedirs(save_dir, exist_ok=True)

    if use_wandb:
        wandb.init(
            project=cfg.training.wandb.project,
            name=cfg.training.wandb.run_name,
            config=dict(cfg),
        )

    global_step = 0

    def _next_val_batch():
        nonlocal val_iter
        try:
            return next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            return next(val_iter)

    def _run_val_step(step_tag: str) -> float:
        val_batch = _next_val_batch()
        if loss_type == "siren":
            with torch.enable_grad():
                sdf_samples = val_batch["sdf_samples"].to(device)
                coords = sdf_samples[..., :3].clone().detach().requires_grad_(True)
                _, val_pred = inference(val_batch, models, cfg=cfg, device=device, coords_override=coords)
                val_gt = sdf_samples[..., 3:].to(device)
                val_normals = val_batch["sdf_normals"].to(device)
                val_normal_mask = val_batch.get("sdf_normal_mask", None)
                if val_normal_mask is not None:
                    val_normal_mask = val_normal_mask.to(device)
                val_loss, val_terms = siren_sdf_loss(
                    val_pred,
                    coords,
                    val_gt,
                    val_normals,
                    normal_mask=val_normal_mask,
                    **siren_weights,
                )
                val_loss_val = val_loss.item()
        else:
            with torch.no_grad():
                _, val_pred = inference(val_batch, models, cfg=cfg, device=device)
                val_gt = val_batch["sdf_samples"].to(device)[..., 3].unsqueeze(-1)
                val_loss_val = loss_fn(val_pred, val_gt).item()
        if use_wandb:
            wandb.log({"val/loss": val_loss_val, "step": step_tag, "epoch": epoch})
            if loss_type == "siren":
                for k, v in val_terms.items():
                    wandb.log({f"val/{k}": v, "step": step_tag, "epoch": epoch})
        return val_loss_val

    for epoch in range(num_epochs):
        # step LR per epoch
        # new_lr = _compute_lr(active_schedule, epoch, lr)
        # for g in optimizer.param_groups:
        #     g["lr"] = new_lr

        for batch in tqdm(sdf_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            if loss_type == "siren":
                sdf_samples = batch["sdf_samples"].to(device)
                coords = sdf_samples[..., :3].clone().detach().requires_grad_(True)
                latent, sdf_pred = inference(batch, models, cfg=cfg, device=device, coords_override=coords)
                sdf_gt = sdf_samples[..., 3:].to(device)
                sdf_normals = batch["sdf_normals"].to(device)
                sdf_normal_mask = batch.get("sdf_normal_mask", None)
                if sdf_normal_mask is not None:
                    sdf_normal_mask = sdf_normal_mask.to(device)
                loss, loss_terms = siren_sdf_loss(
                    sdf_pred,
                    coords,
                    sdf_gt,
                    sdf_normals,
                    normal_mask=sdf_normal_mask,
                    **siren_weights,
                )
            else:
                latent, sdf_pred = inference(batch, models, cfg=cfg, device=device)
                sdf_gt = batch["sdf_samples"].to(device)[..., 3].unsqueeze(-1)
                loss = loss_fn(sdf_pred, sdf_gt)
            loss.backward()
            optimizer.step()

            if use_wandb:
                wandb.log({"train/loss": loss.item(), "step": global_step, "epoch": epoch})
                if loss_type == "siren":
                    for k, v in loss_terms.items():
                        wandb.log({f"train/{k}": v, "step": global_step, "epoch": epoch})
            global_step += 1

            if test_interval is not None and test_interval > 0 and global_step % test_interval == 0:
                val_loss_iter = _run_val_step(step_tag=global_step)
                print(f"Iter {global_step}: val loss {val_loss_iter:.4f}")

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
