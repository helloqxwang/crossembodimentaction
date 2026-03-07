from __future__ import annotations

import contextlib
import copy
import os
import random
import sys
import time
from typing import Any, Dict

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from data_process.contact_policy_dataset import (  # noqa: E402
    OnTheFlySyntheticContactPolicyDataset,
    RealContactPolicyValDataset,
)
from networks.contact_diffusion_policy import ContactDiffusionPolicy  # noqa: E402
from networks.simple_diffusion import SimpleDDIMScheduler  # noqa: E402


class EMAModel:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ) -> None:
        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)
        self.update_after_step = int(update_after_step)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, int(optimization_step) - self.update_after_step - 1)
        value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model: torch.nn.Module) -> None:
        self.decay = self.get_decay(self.optimization_step)
        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                if isinstance(module, _BatchNorm) or not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1.0 - self.decay)
        self.optimization_step += 1


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prepare_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _resolve_project_path(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(ROOT_DIR, path_str))


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _get_amp_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if key in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {name}")


def _autocast_context(device: torch.device, enabled: bool, amp_dtype: torch.dtype):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _build_noise_scheduler(cfg: DictConfig) -> SimpleDDIMScheduler:
    return SimpleDDIMScheduler(
        num_train_timesteps=int(cfg.num_train_timesteps),
        beta_start=float(cfg.beta_start),
        beta_end=float(cfg.beta_end),
        beta_schedule=str(cfg.beta_schedule),
        clip_sample=bool(cfg.clip_sample),
        set_alpha_to_one=bool(cfg.set_alpha_to_one),
        steps_offset=int(cfg.steps_offset),
        prediction_type=str(cfg.prediction_type),
    )


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: str,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    schedule_name = str(name).lower()
    warmup_steps = max(0, int(num_warmup_steps))
    total_steps = max(1, int(num_training_steps))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if schedule_name == "constant":
            return 1.0
        if schedule_name == "linear":
            return max(0.0, 1.0 - progress)
        if schedule_name == "cosine":
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        raise ValueError(f"Unsupported lr_scheduler: {name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _build_policy(cfg: DictConfig, action_dim: int) -> ContactDiffusionPolicy:
    return ContactDiffusionPolicy(
        action_dim=action_dim,
        p_hat_encoder_cfg={
            "in_channels": int(cfg.model.p_hat_encoder.in_channels),
            "hidden_dim": int(cfg.model.p_hat_encoder.hidden_dim),
            "out_channels": int(cfg.model.p_hat_encoder.out_channels),
            "num_layers": int(cfg.model.p_hat_encoder.num_layers),
        },
        contact_encoder_cfg={
            "in_channels": int(cfg.model.contact_encoder.in_channels),
            "hidden_dim": int(cfg.model.contact_encoder.hidden_dim),
            "out_channels": int(cfg.model.contact_encoder.out_channels),
            "num_layers": int(cfg.model.contact_encoder.num_layers),
        },
        fusion_hidden_dims=tuple(int(x) for x in cfg.model.fusion_hidden_dims),
        global_cond_dim=int(cfg.model.global_cond_dim),
        diffusion_step_embed_dim=int(cfg.model.diffusion_step_embed_dim),
        down_dims=tuple(int(x) for x in cfg.model.down_dims),
        kernel_size=int(cfg.model.kernel_size),
        n_groups=int(cfg.model.n_groups),
        condition_type=str(cfg.model.condition_type),
        num_inference_steps=int(cfg.model.num_inference_steps),
        noise_scheduler=_build_noise_scheduler(cfg.model.noise_scheduler),
    )


def _build_loader_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    num_workers = int(cfg.dataset.num_workers)
    kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(cfg.dataset.pin_memory),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(cfg.dataset.prefetch_factor)
    return kwargs


def _build_val_subset_indices(dataset_len: int, subset_size: int, seed: int) -> list[int]:
    n = int(max(0, subset_size))
    if n <= 0 or n >= dataset_len:
        return list(range(dataset_len))
    rng = np.random.default_rng(int(seed))
    return sorted(int(x) for x in rng.choice(dataset_len, size=n, replace=False).tolist())


def _compute_validation_metrics(
    policy: ContactDiffusionPolicy,
    loader: DataLoader,
    metric_models: Dict[str, Any],
    device: torch.device,
    *,
    prefix: str,
    max_batches: int = -1,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    policy.eval()
    full_action_mse = 0.0
    valid_action_mse = 0.0
    contact_point_mse = 0.0
    inactive_surface_mse = 0.0
    fetch_time_total = 0.0
    predict_time_total = 0.0
    metric_time_total = 0.0
    n_action = 0
    n_contact = 0
    n_inactive = 0
    n_batches = 0

    target_batches = len(loader)
    if max_batches > 0:
        target_batches = min(target_batches, int(max_batches))

    with torch.no_grad():
        iterator = iter(loader)
        progress = tqdm(total=target_batches, desc=prefix, leave=False)
        while n_batches < target_batches:
            fetch_t0 = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            fetch_time_total += time.perf_counter() - fetch_t0

            predict_t0 = time.perf_counter()
            batch_device = _move_batch_to_device(batch, device)
            with _autocast_context(device, use_amp, amp_dtype):
                pred_action = policy.predict_action(batch_device)["action_pred"].detach()
            _cuda_sync_if_needed(device)
            predict_time_total += time.perf_counter() - predict_t0
            pred_action = pred_action.cpu()

            metric_t0 = time.perf_counter()
            gt_action = batch["action"].float()
            action_valid_mask = batch["action_valid_mask"].float()
            sq_error = (pred_action - gt_action).pow(2)
            full_action_mse += float(sq_error.mean(dim=1).sum().item())
            valid_action_mse += float(
                ((sq_error * action_valid_mask).sum(dim=1) / action_valid_mask.sum(dim=1).clamp_min(1.0)).sum().item()
            )
            n_action += int(gt_action.shape[0])

            q1_points_centered = batch["p_hat"][:, :, :3].float() - batch["contact_center"].float().unsqueeze(1)
            for i in range(int(gt_action.shape[0])):
                robot_name = str(batch["robot_name"][i])
                model = metric_models[robot_name]
                local_dof = int(batch["local_dof"][i].item())
                pred_local = pred_action[i, :local_dof].to(model.device)
                contact_valid = batch["contact_valid_mask"][i].bool()
                surface_mask = batch["contact_mask_surface"][i].bool()
                contact_indices = batch["contact_point_indices"][i][contact_valid].long()

                pred_points, _ = model.get_surface_points_normals(q=pred_local)
                pred_points = pred_points.detach().cpu()

                if int(contact_indices.numel()) > 0:
                    target_contact = batch["contact_cloud"][i, contact_valid, :3].float()
                    contact_point_mse += float(
                        torch.mean((pred_points[contact_indices, :3] - target_contact) ** 2).item()
                    )
                    n_contact += 1

                inactive_idx = torch.nonzero(~surface_mask, as_tuple=False).view(-1)
                if int(inactive_idx.numel()) > 0:
                    inactive_surface_mse += float(
                        torch.mean((pred_points[inactive_idx, :3] - q1_points_centered[i, inactive_idx, :3]) ** 2).item()
                    )
                    n_inactive += 1

            metric_time_total += time.perf_counter() - metric_t0
            n_batches += 1
            progress.update(1)
        progress.close()

    if n_action == 0:
        raise RuntimeError("Validation dataloader returned zero samples")

    return {
        f"{prefix}/action_mse_full": full_action_mse / max(n_action, 1),
        f"{prefix}/action_mse_valid": valid_action_mse / max(n_action, 1),
        f"{prefix}/contact_point_mse": contact_point_mse / max(n_contact, 1),
        f"{prefix}/inactive_surface_mse": inactive_surface_mse / max(n_inactive, 1),
        f"{prefix}/fetch_s": fetch_time_total / max(n_batches, 1),
        f"{prefix}/predict_s": predict_time_total / max(n_batches, 1),
        f"{prefix}/metric_s": metric_time_total / max(n_batches, 1),
    }


@hydra.main(version_base="1.2", config_path="conf", config_name="config_contact_policy")
def main(cfg: DictConfig) -> None:
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    _set_seed(int(cfg.seed))
    device = _prepare_device(str(cfg.training.device))
    use_amp = bool(cfg.training.use_amp)
    amp_dtype = _get_amp_dtype(str(cfg.training.amp_dtype))

    dataset_kwargs = {
        "robot_names": list(cfg.dataset.robot_names),
        "hparams_path": _resolve_project_path(str(cfg.dataset.hparams_path)),
        "real_masks_path": _resolve_project_path(str(cfg.dataset.real_masks_path)),
        "max_contact_points": int(cfg.dataset.max_contact_points),
        "seed": int(cfg.seed),
        "device": str(cfg.dataset.robot_model_device),
        "check_template_hash": bool(cfg.dataset.check_template_hash),
        "threshold": float(cfg.dataset.threshold),
        "align_exp_scale": float(cfg.dataset.align_exp_scale),
        "sigmoid_scale": float(cfg.dataset.sigmoid_scale),
        "anchor_sampling_mode": str(cfg.dataset.anchor_sampling_mode),
        "anchor_temperature": float(cfg.dataset.anchor_temperature),
        "patch_points": int(cfg.dataset.patch_points),
        "patch_anchor_shift_min": float(cfg.dataset.patch_anchor_shift_min),
        "patch_anchor_shift_max": float(cfg.dataset.patch_anchor_shift_max),
        "patch_extent_min": float(cfg.dataset.patch_extent_min),
        "patch_extent_max": float(cfg.dataset.patch_extent_max),
        "patch_extent_power": float(cfg.dataset.patch_extent_power),
        "patch_shift_power": float(cfg.dataset.patch_shift_power),
        "patch_points_per_anchor_min": int(cfg.dataset.patch_points_per_anchor_min),
        "patch_points_per_anchor_max": int(cfg.dataset.patch_points_per_anchor_max),
        "patch_penetration_clearance": float(cfg.dataset.patch_penetration_clearance),
        "patch_exact_count_prob": float(cfg.dataset.patch_exact_count_prob),
        "patch_count_clamp_ratio": float(cfg.dataset.patch_count_clamp_ratio),
    }
    train_dataset = OnTheFlySyntheticContactPolicyDataset(
        samples_per_epoch=int(cfg.dataset.train_samples_per_epoch),
        train_chunk_size=int(cfg.dataset.train_chunk_size),
        **dataset_kwargs,
    )
    val_dataset = RealContactPolicyValDataset(
        cache_processed_samples=bool(cfg.dataset.cache_validation_samples),
        **dataset_kwargs,
    )

    loader_kwargs = _build_loader_kwargs(cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.dataset.batch_size),
        shuffle=False,
        drop_last=bool(cfg.dataset.drop_last),
        **loader_kwargs,
    )

    full_val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.dataset.val_batch_size),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    subset_indices = _build_val_subset_indices(
        dataset_len=len(val_dataset),
        subset_size=int(cfg.training.val_subset_size),
        seed=int(cfg.seed),
    )
    if len(subset_indices) == len(val_dataset):
        subset_val_loader = full_val_loader
    else:
        subset_val_loader = DataLoader(
            Subset(val_dataset, subset_indices),
            batch_size=int(cfg.dataset.val_batch_size),
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

    model = _build_policy(cfg, action_dim=train_dataset.action_dim).to(device)
    ema_model = None
    ema = None
    if bool(cfg.training.use_ema):
        ema_model = copy.deepcopy(model).to(device)
        ema = EMAModel(
            ema_model,
            update_after_step=int(cfg.training.ema.update_after_step),
            inv_gamma=float(cfg.training.ema.inv_gamma),
            power=float(cfg.training.ema.power),
            min_value=float(cfg.training.ema.min_value),
            max_value=float(cfg.training.ema.max_value),
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        betas=tuple(float(x) for x in cfg.training.betas),
        eps=float(cfg.training.eps),
        weight_decay=float(cfg.training.weight_decay),
    )
    total_steps = len(train_loader) * int(cfg.training.epochs)
    lr_scheduler = _build_lr_scheduler(
        optimizer=optimizer,
        name=str(cfg.training.lr_scheduler),
        num_warmup_steps=int(cfg.training.lr_warmup_steps),
        num_training_steps=int(total_steps),
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

    save_dir = _resolve_project_path(str(cfg.training.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "robot_names": list(cfg.dataset.robot_names),
            "action_dim": int(train_dataset.action_dim),
            "surface_num_points": int(train_dataset.surface_num_points),
            "max_contact_points": int(train_dataset.max_contact_points),
        },
        os.path.join(save_dir, "dataset_meta.pt"),
    )

    metric_models = {name: spec["model"] for name, spec in val_dataset.robot_specs.items()}
    best_val = float("inf")
    best_metric_key = "validate_subset/action_mse_valid"
    global_step = 0
    log_every_n_steps = int(max(1, cfg.training.log_every_n_steps))
    max_train_batches = int(cfg.training.max_train_batches)
    max_val_batches = int(cfg.training.max_val_batches)

    for epoch in range(1, int(cfg.training.epochs) + 1):
        model.train()
        train_losses: list[float] = []
        data_wait_times: list[float] = []
        h2d_times: list[float] = []
        train_step_times: list[float] = []

        train_batches_total = len(train_loader)
        if max_train_batches > 0:
            train_batches_total = min(train_batches_total, max_train_batches)

        iterator = iter(train_loader)
        progress = tqdm(total=train_batches_total, desc=f"train:{epoch}")
        batch_idx = 0
        while batch_idx < train_batches_total:
            fetch_t0 = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            data_wait_s = time.perf_counter() - fetch_t0

            h2d_t0 = time.perf_counter()
            batch_device = _move_batch_to_device(batch, device)
            _cuda_sync_if_needed(device)
            h2d_s = time.perf_counter() - h2d_t0

            optimizer.zero_grad(set_to_none=True)
            step_t0 = time.perf_counter()
            with _autocast_context(device, use_amp, amp_dtype):
                loss, loss_dict = model.compute_loss(batch_device)
            loss.backward()
            if float(cfg.training.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip_norm))
            optimizer.step()
            lr_scheduler.step()
            if ema is not None:
                ema.step(model)
            _cuda_sync_if_needed(device)
            train_step_s = time.perf_counter() - step_t0

            loss_value = float(loss.item())
            train_losses.append(loss_value)
            data_wait_times.append(data_wait_s)
            h2d_times.append(h2d_s)
            train_step_times.append(train_step_s)

            step_log = {
                "train/loss": loss_value,
                "train/lr": float(lr_scheduler.get_last_lr()[0]),
                "train/data_wait_s": data_wait_s,
                "train/h2d_s": h2d_s,
                "train/train_step_s": train_step_s,
                "global_step": global_step,
                "epoch": epoch,
            }
            step_log.update({f"train/{k}": v for k, v in loss_dict.items()})
            should_log_step = (global_step % log_every_n_steps == 0) or (batch_idx + 1 == train_batches_total)
            if should_log_step:
                wandb.log(step_log, step=global_step)

            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                wait=f"{data_wait_s:.3f}",
                step=f"{train_step_s:.3f}",
            )
            global_step += 1
            batch_idx += 1
            progress.update(1)
        progress.close()

        epoch_log = {
            "epoch": epoch,
            "train/loss_epoch": float(np.mean(train_losses)) if train_losses else float("nan"),
            "train/data_wait_s_epoch": float(np.mean(data_wait_times)) if data_wait_times else float("nan"),
            "train/h2d_s_epoch": float(np.mean(h2d_times)) if h2d_times else float("nan"),
            "train/train_step_s_epoch": float(np.mean(train_step_times)) if train_step_times else float("nan"),
        }

        eval_policy = ema_model if ema_model is not None else model
        subset_metrics = _compute_validation_metrics(
            eval_policy,
            subset_val_loader,
            metric_models=metric_models,
            device=device,
            prefix="validate_subset",
            max_batches=max_val_batches,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        epoch_log.update(subset_metrics)

        run_full_validation = False
        if subset_val_loader is not full_val_loader:
            every_n = int(max(1, cfg.training.full_val_every_n_epochs))
            run_full_validation = (epoch % every_n == 0) or (epoch == int(cfg.training.epochs))
        if run_full_validation:
            full_metrics = _compute_validation_metrics(
                eval_policy,
                full_val_loader,
                metric_models=metric_models,
                device=device,
                prefix="validate_full",
                max_batches=max_val_batches,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            epoch_log.update(full_metrics)
        elif subset_val_loader is full_val_loader:
            epoch_log["validate_full/action_mse_valid"] = epoch_log[best_metric_key]

        wandb.log(epoch_log, step=global_step)

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if ema_model is not None:
            ckpt["ema_model"] = ema_model.state_dict()

        torch.save(ckpt, os.path.join(save_dir, "last.pt"))
        if epoch % int(cfg.training.save_every_n_epoch) == 0:
            torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch}.pt"))
        if float(epoch_log[best_metric_key]) < best_val:
            best_val = float(epoch_log[best_metric_key])
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))

        summary = (
            f"[Epoch {epoch}] "
            f"train_loss={epoch_log['train/loss_epoch']:.6f} "
            f"subset_action_valid={epoch_log['validate_subset/action_mse_valid']:.6f} "
            f"data_wait={epoch_log['train/data_wait_s_epoch']:.4f}s "
            f"train_step={epoch_log['train/train_step_s_epoch']:.4f}s"
        )
        if "validate_full/action_mse_valid" in epoch_log:
            summary += f" full_action_valid={epoch_log['validate_full/action_mse_valid']:.6f}"
        print(summary)

    wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
