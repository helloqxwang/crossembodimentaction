from __future__ import annotations

import os
import random
import sys
from typing import Dict

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from experiment.cross_cvae_dataset import CrossEmbodimentActionDataset
from networks.cross_cvae_mvp import CrossEmbodimentActionCVAE


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prepare_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # Mean KL over batch.
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def _run_epoch(
    *,
    model: CrossEmbodimentActionCVAE,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    recon_weight: float,
    kl_weight: float,
    grad_clip_norm: float | None,
    log_every: int,
    global_step: int,
    is_train: bool,
    max_batches: int = -1,
) -> tuple[Dict[str, float], int]:
    if is_train:
        model.train()
    else:
        model.eval()

    sum_loss = 0.0
    sum_recon = 0.0
    sum_recon_valid = 0.0
    sum_kl = 0.0
    num_samples = 0

    iterator = tqdm(loader, desc="train" if is_train else "validate")
    for batch_idx, batch in enumerate(iterator):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        object_pc = batch["object_pc"].to(device)
        action = batch["action"].to(device)
        action_mask = batch["action_mask"].to(device).float()
        embodiment_idx = batch["embodiment_idx"].to(device)
        batch_size = action.size(0)

        with torch.set_grad_enabled(is_train):
            recon, mu, logvar, _ = model(object_pc, action, embodiment_idx)
            recon_loss = F.mse_loss(recon, action, reduction="mean")
            sq_error = (recon - action).pow(2)
            recon_valid_loss = (sq_error * action_mask).sum() / action_mask.sum().clamp_min(1.0)
            kl_loss = _kl_divergence(mu, logvar)
            loss = recon_weight * recon_loss + kl_weight * kl_loss

            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        sum_loss += float(loss.item()) * batch_size
        sum_recon += float(recon_loss.item()) * batch_size
        sum_recon_valid += float(recon_valid_loss.item()) * batch_size
        sum_kl += float(kl_loss.item()) * batch_size
        num_samples += batch_size

        if is_train and (global_step % max(log_every, 1) == 0):
            wandb.log(
                {
                    "train/loss": float(loss.item()),
                    "train/recon_mse": float(recon_loss.item()),
                    "train/recon_mse_valid": float(recon_valid_loss.item()),
                    "train/kl": float(kl_loss.item()),
                },
                step=global_step,
            )
        if is_train:
            global_step += 1

    if num_samples == 0:
        raise RuntimeError("Dataloader returned zero samples.")

    metrics = {
        "loss": sum_loss / num_samples,
        "recon_mse": sum_recon / num_samples,
        "recon_mse_valid": sum_recon_valid / num_samples,
        "kl": sum_kl / num_samples,
    }
    return metrics, global_step


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_cross_cvae")
def main(cfg: DictConfig) -> None:
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    _set_seed(int(cfg.seed))
    device = _prepare_device(str(cfg.training.device))

    dro_root = to_absolute_path(str(cfg.dataset.dro_root))
    robot_names = list(cfg.dataset.robot_names)

    train_dataset = CrossEmbodimentActionDataset(
        dro_root=dro_root,
        robot_names=robot_names,
        split="train",
        num_points=int(cfg.dataset.num_points),
    )
    validate_dataset = CrossEmbodimentActionDataset(
        dro_root=dro_root,
        robot_names=robot_names,
        split="validate",
        num_points=int(cfg.dataset.num_points),
        max_dof=train_dataset.max_dof,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.dataset.batch_size),
        shuffle=True,
        num_workers=int(cfg.dataset.num_workers),
        pin_memory=bool(cfg.dataset.pin_memory),
        drop_last=bool(cfg.dataset.drop_last),
        persistent_workers=bool(cfg.dataset.num_workers > 0),
    )
    validate_loader = DataLoader(
        validate_dataset,
        batch_size=int(cfg.dataset.val_batch_size),
        shuffle=False,
        num_workers=int(cfg.dataset.num_workers),
        pin_memory=bool(cfg.dataset.pin_memory),
        drop_last=False,
        persistent_workers=bool(cfg.dataset.num_workers > 0),
    )

    model = CrossEmbodimentActionCVAE(
        action_dim=train_dataset.max_dof,
        num_embodiments=len(robot_names),
        embodiment_dim=int(cfg.model.embodiment_dim),
        object_emb_dim=int(cfg.model.object_emb_dim),
        latent_dim=int(cfg.model.latent_dim),
        encoder_hidden_dims=tuple(int(x) for x in cfg.model.encoder_hidden_dims),
        decoder_hidden_dims=tuple(int(x) for x in cfg.model.decoder_hidden_dims),
        dgcnn_k=int(cfg.model.dgcnn_k),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )

    run_name = str(cfg.wandb.name) if cfg.wandb.name is not None else str(cfg.experiment.name)
    if bool(cfg.wandb.enabled):
        wandb.init(
            project=str(cfg.wandb.project),
            entity=None if cfg.wandb.entity is None else str(cfg.wandb.entity),
            name=run_name,
            mode=str(cfg.wandb.mode),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        wandb.init(mode="disabled")

    save_dir = to_absolute_path(str(cfg.training.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    meta_path = os.path.join(save_dir, "dataset_meta.pt")
    torch.save(
        {
            "robot_names": robot_names,
            "robot_to_idx": train_dataset.robot_to_idx,
            "robot_dofs": train_dataset.robot_dofs,
            "action_dim": train_dataset.max_dof,
        },
        meta_path,
    )

    global_step = 0
    best_val = float("inf")
    for epoch in range(1, int(cfg.training.epochs) + 1):
        train_metrics, global_step = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            recon_weight=float(cfg.training.recon_weight),
            kl_weight=float(cfg.training.kl_weight),
            grad_clip_norm=float(cfg.training.grad_clip_norm),
            log_every=int(cfg.training.log_every),
            global_step=global_step,
            is_train=True,
            max_batches=int(cfg.training.max_train_batches),
        )
        val_metrics, global_step = _run_epoch(
            model=model,
            loader=validate_loader,
            device=device,
            optimizer=None,
            recon_weight=float(cfg.training.recon_weight),
            kl_weight=float(cfg.training.kl_weight),
            grad_clip_norm=None,
            log_every=int(cfg.training.log_every),
            global_step=global_step,
            is_train=False,
            max_batches=int(cfg.training.max_val_batches),
        )

        epoch_log = {
            "epoch": epoch,
            "train/loss_epoch": train_metrics["loss"],
            "train/recon_mse_epoch": train_metrics["recon_mse"],
            "train/recon_mse_valid_epoch": train_metrics["recon_mse_valid"],
            "train/kl_epoch": train_metrics["kl"],
            "validate/loss_epoch": val_metrics["loss"],
            "validate/recon_mse_epoch": val_metrics["recon_mse"],
            "validate/recon_mse_valid_epoch": val_metrics["recon_mse_valid"],
            "validate/kl_epoch": val_metrics["kl"],
        }
        wandb.log(epoch_log, step=global_step)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_recon_valid={val_metrics['recon_mse_valid']:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        torch.save(ckpt, os.path.join(save_dir, "last.pt"))
        if epoch % int(cfg.training.save_every_n_epoch) == 0:
            torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch}.pt"))

        if val_metrics["recon_mse_valid"] < best_val:
            best_val = val_metrics["recon_mse_valid"]
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))

    wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    main()
