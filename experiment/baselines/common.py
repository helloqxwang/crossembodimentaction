from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def run_epoch(
    *,
    model: torch.nn.Module,
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
        embodiment_idx = batch["embodiment_idx"].to(device) if "embodiment_idx" in batch else None
        batch_size = action.size(0)

        with torch.set_grad_enabled(is_train):
            recon, mu, logvar, _ = model(object_pc, action, embodiment_idx)
            recon_loss = F.mse_loss(recon, action, reduction="mean")
            sq_error = (recon - action).pow(2)
            recon_valid_loss = (sq_error * action_mask).sum() / action_mask.sum().clamp_min(1.0)
            kl_loss = kl_divergence(mu, logvar)
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


def init_wandb(cfg: DictConfig, *, run_name: str) -> None:
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


def create_loader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
        persistent_workers=bool(num_workers > 0),
    )


def save_ckpt(
    *,
    cfg: DictConfig,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


def finish_wandb() -> None:
    wandb.finish()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
