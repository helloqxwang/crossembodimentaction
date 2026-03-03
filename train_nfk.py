from __future__ import annotations

import os
from typing import Dict
from pathlib import Path

import hydra
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

from data_process.nfk_dataset import get_nfk_dataloader
from networks.nfk_model import NFKModel, nfk_losses


def _prepare_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def build_model(cfg: DictConfig, device: torch.device) -> NFKModel:
    model = NFKModel(
        pair_dim=cfg.model.pair_dim,
        token_dim=cfg.model.token_dim,
        model_dim=cfg.model.model_dim,
        enc_num_heads=cfg.model.enc_num_heads,
        enc_num_layers=cfg.model.enc_num_layers,
        att_num_heads=cfg.model.att_num_heads,
        att_num_layers=cfg.model.att_num_layers,
        codebook_size=cfg.model.codebook_size,
        vq_beta=cfg.model.vq_beta,
        usage_temperature=float(getattr(cfg.model, "usage_temperature", 0.5)),
        usage_ema_decay=float(getattr(cfg.model, "usage_ema_decay", 0.99)),
        usage_eps=float(getattr(cfg.model, "usage_eps", 1e-8)),
        dropout=cfg.model.dropout,
        attn_dropout=cfg.model.attn_dropout,
        activation=cfg.model.activation,
        max_points=cfg.model.max_points,
    ).to(device)
    return model


def move_batch_to_device(batch: Dict[str, torch.Tensor | list], device: torch.device) -> Dict[str, torch.Tensor | list]:
    out: Dict[str, torch.Tensor | list] = {}
    for key, val in batch.items():
        if torch.is_tensor(val):
            out[key] = val.to(device)
        else:
            out[key] = val
    return out


def _usage_kl_weight(cfg: DictConfig, global_step: int) -> float:
    max_w = float(getattr(cfg.loss, "w_usage_kl", 0.0))
    if max_w <= 0.0:
        return 0.0

    warmup_steps = int(getattr(cfg.loss, "usage_kl_warmup_steps", 0))
    warmup_start = int(getattr(cfg.loss, "usage_kl_warmup_start_step", 0))

    if global_step < warmup_start:
        return 0.0
    if warmup_steps <= 0:
        return max_w

    progress = float(global_step - warmup_start) / float(max(warmup_steps, 1))
    progress = max(0.0, min(1.0, progress))
    return max_w * progress


def train_one_epoch(
    *,
    model: NFKModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: DictConfig,
    use_wandb: bool,
    global_step_start: int,
    epoch: int,
) -> int:
    model.train()
    global_step = global_step_start

    for batch in tqdm(loader, desc=f"Train {epoch+1}"):
        batch = move_batch_to_device(batch, device)
        zero_pairs = batch["zero_pairs_masked"]
        rand_pairs = batch["rand_pairs_masked"]
        mask_m = batch["mask_m"]

        optimizer.zero_grad()
        outputs = model(zero_pairs, rand_pairs, mask_m)
        recon_terms = nfk_losses(
            outputs["recon_pairs"],
            rand_pairs,
            mask_m,
            w_pos=float(cfg.loss.w_pos),
            w_nrm=float(cfg.loss.w_nrm),
        )
        usage_weight = _usage_kl_weight(cfg, global_step)
        loss_total = (
            recon_terms["loss_recon"]
            + float(cfg.loss.w_vq) * outputs["vq_loss"]
            + usage_weight * outputs["usage_kl_loss"]
        )
        loss_total.backward()
        optimizer.step()

        if use_wandb:
            wandb.log(
                {
                    "train/loss_total": float(loss_total.item()),
                    "train/loss_recon": float(recon_terms["loss_recon"].item()),
                    "train/loss_pos": float(recon_terms["loss_recon_pos"].item()),
                    "train/loss_nrm": float(recon_terms["loss_recon_nrm"].item()),
                    "train/loss_vq": float(outputs["vq_loss"].item()),
                    "train/loss_codebook": float(outputs["codebook_loss"].item()),
                    "train/loss_commit": float(outputs["commit_loss"].item()),
                    "train/loss_usage_kl": float(outputs["usage_kl_loss"].item()),
                    "train/usage_kl_weight": float(usage_weight),
                    "train/usage_entropy": float(outputs["usage_entropy"].item()),
                    "train/perplexity": float(outputs["perplexity"].item()),
                    "train/active_codes": float(outputs["active_codes"].item()),
                    "epoch": epoch,
                    "step": global_step,
                }
            )
        global_step += 1

    return global_step


@torch.no_grad()
def validate(
    *,
    model: NFKModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: DictConfig,
    use_wandb: bool,
    epoch: int,
    step_tag: int,
) -> Dict[str, float]:
    model.eval()
    stats = {
        "val/loss_total": 0.0,
        "val/loss_recon": 0.0,
        "val/loss_pos": 0.0,
        "val/loss_nrm": 0.0,
        "val/loss_vq": 0.0,
        "val/loss_usage_kl": 0.0,
        "val/usage_entropy": 0.0,
        "val/perplexity": 0.0,
        "val/active_codes": 0.0,
    }
    usage_weight = _usage_kl_weight(cfg, int(step_tag))
    n_batches = 0
    for batch in tqdm(loader, desc=f"Val {epoch+1}"):
        batch = move_batch_to_device(batch, device)
        zero_pairs = batch["zero_pairs_masked"]
        rand_pairs = batch["rand_pairs_masked"]
        mask_m = batch["mask_m"]

        outputs = model(zero_pairs, rand_pairs, mask_m)
        recon_terms = nfk_losses(
            outputs["recon_pairs"],
            rand_pairs,
            mask_m,
            w_pos=float(cfg.loss.w_pos),
            w_nrm=float(cfg.loss.w_nrm),
        )
        loss_total = (
            recon_terms["loss_recon"]
            + float(cfg.loss.w_vq) * outputs["vq_loss"]
            + usage_weight * outputs["usage_kl_loss"]
        )

        stats["val/loss_total"] += float(loss_total.item())
        stats["val/loss_recon"] += float(recon_terms["loss_recon"].item())
        stats["val/loss_pos"] += float(recon_terms["loss_recon_pos"].item())
        stats["val/loss_nrm"] += float(recon_terms["loss_recon_nrm"].item())
        stats["val/loss_vq"] += float(outputs["vq_loss"].item())
        stats["val/loss_usage_kl"] += float(outputs["usage_kl_loss"].item())
        stats["val/usage_entropy"] += float(outputs["usage_entropy"].item())
        stats["val/perplexity"] += float(outputs["perplexity"].item())
        stats["val/active_codes"] += float(outputs["active_codes"].item())
        n_batches += 1

    if n_batches > 0:
        for key in stats.keys():
            stats[key] /= n_batches
    stats["val/usage_kl_weight"] = float(usage_weight)

    if use_wandb:
        payload = dict(stats)
        payload["epoch"] = epoch
        payload["step"] = step_tag
        wandb.log(payload)
    return stats


@hydra.main(config_path="conf", config_name="config_nfk", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = _prepare_device(str(cfg.training.device))
    torch.manual_seed(int(cfg.training.seed))
    dataset_device = str(getattr(cfg.data, "dataset_device", "cpu"))

    assets_dir = Path(to_absolute_path(cfg.data.assets_dir))
    robot_list_file = None
    if cfg.data.robot_list_file is not None and str(cfg.data.robot_list_file) != "":
        robot_list_file = Path(to_absolute_path(cfg.data.robot_list_file))

    train_loader = get_nfk_dataloader(
        assets_dir=assets_dir,
        robot_list_key=cfg.data.robot_list_key,
        robot_list_file=robot_list_file,
        dataset_size=cfg.data.train_dataset_size,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        drop_last=cfg.data.drop_last,
        seed=cfg.training.seed,
        device=dataset_device,
        link_num_points=cfg.data.link_num_points,
        box_fit_override=cfg.data.box_fit_override,
    )
    val_loader = get_nfk_dataloader(
        assets_dir=assets_dir,
        robot_list_key=cfg.data.robot_list_key,
        robot_list_file=robot_list_file,
        dataset_size=cfg.data.val_dataset_size,
        batch_size=cfg.data.val_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        seed=cfg.training.seed + 1,
        device=dataset_device,
        link_num_points=cfg.data.link_num_points,
        box_fit_override=False,
    )

    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.training.lr))

    use_wandb = bool(cfg.training.wandb.enabled)
    save_dir = to_absolute_path(os.path.join(cfg.training.wandb.save_dir, cfg.training.wandb.run_name))
    os.makedirs(save_dir, exist_ok=True)

    if use_wandb:
        wandb.init(
            project=cfg.training.wandb.project,
            name=cfg.training.wandb.run_name,
            config=dict(cfg),
        )

    global_step = 0
    num_epochs = int(cfg.training.num_epochs)
    save_interval = int(cfg.training.save_interval)
    val_interval = int(cfg.training.val_interval)

    for epoch in range(num_epochs):
        global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            use_wandb=use_wandb,
            global_step_start=global_step,
            epoch=epoch,
        )

        if val_interval > 0 and ((epoch + 1) % val_interval == 0):
            stats = validate(
                model=model,
                loader=val_loader,
                device=device,
                cfg=cfg,
                use_wandb=use_wandb,
                epoch=epoch,
                step_tag=global_step,
            )
            print(f"Epoch {epoch+1}: val_total={stats['val/loss_total']:.6f}")

        if save_interval > 0 and ((epoch + 1) % save_interval == 0):
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": dict(cfg),
                    "step": global_step,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
