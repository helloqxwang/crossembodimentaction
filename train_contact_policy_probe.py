from __future__ import annotations

import copy
import os
import sys
from typing import Any, Dict

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Reuse the current contact-policy stack wherever the probe path matches it.
from data_process.contact_policy_probe_dataset import ShadowhandContactPolicyProbeDataset  # noqa: E402
from networks.contact_probe_policy import ContactDiffusionProbePolicy  # noqa: E402
from robot_model.robot_model import create_robot_model  # noqa: E402
from train_contact_policy import (  # noqa: E402
    EMAModel,
    _autocast_context,
    _build_action_normalizer_meta,
    _build_loader_kwargs,
    _build_lr_scheduler,
    _build_noise_scheduler,
    _cuda_sync_if_needed,
    _get_amp_dtype,
    _infer_wandb_run_id,
    _move_batch_to_device,
    _prepare_device,
    _resolve_optional_project_path,
    _resolve_project_path,
    _set_seed,
    build_contact_policy_common_dataset_kwargs,
)


# Build the probe policy with the same action normalization and diffusion utilities
# as the main contact policy, but with contact-only observations.
def _build_probe_policy(
    cfg: DictConfig,
    *,
    action_dim: int,
    action_normalizer_meta: Dict[str, Any],
) -> ContactDiffusionProbePolicy:
    return ContactDiffusionProbePolicy(
        action_dim=int(action_dim),
        robot_names=[str(x) for x in action_normalizer_meta.get("robot_names", [])],
        action_normalizer_scale=action_normalizer_meta.get("scale"),
        action_normalizer_offset=action_normalizer_meta.get("offset"),
        contact_encoder_cfg={
            "in_channels": int(cfg.model.contact_encoder.in_channels),
            "hidden_dim": int(cfg.model.contact_encoder.hidden_dim),
            "out_channels": int(cfg.model.contact_encoder.out_channels),
            "num_layers": int(cfg.model.contact_encoder.num_layers),
        },
        obs_hidden_dims=tuple(int(x) for x in cfg.model.obs_hidden_dims),
        global_cond_dim=int(cfg.model.global_cond_dim),
        diffusion_step_embed_dim=int(cfg.model.diffusion_step_embed_dim),
        down_dims=tuple(int(x) for x in cfg.model.down_dims),
        num_inference_steps=int(cfg.model.num_inference_steps),
        noise_scheduler=_build_noise_scheduler(cfg.model.noise_scheduler),
    )


# Reload a trained probe policy from checkpoint plus the saved dataset metadata.
def load_contact_policy_probe_checkpoint(
    cfg: DictConfig,
    *,
    device: torch.device,
    ckpt_path: str | None = None,
    prefer_ema: bool = True,
) -> tuple[ContactDiffusionProbePolicy, str, Dict[str, Any]]:
    resolved_ckpt_path = (
        _resolve_project_path(os.path.join(str(cfg.training.save_dir), "last.pt"))
        if ckpt_path is None
        else _resolve_project_path(str(ckpt_path))
    )
    ckpt = torch.load(resolved_ckpt_path, map_location=device, weights_only=False)
    build_cfg = cfg
    ckpt_cfg = ckpt.get("config")
    if isinstance(ckpt_cfg, dict):
        try:
            build_cfg = OmegaConf.create(ckpt_cfg)
        except Exception:
            build_cfg = cfg

    meta_path = os.path.join(os.path.dirname(resolved_ckpt_path), "dataset_meta.pt")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Probe dataset metadata not found: {meta_path}")
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid probe dataset metadata: {meta_path}")
    if "action_dim" not in meta or "action_normalizer_scale" not in meta or "action_normalizer_offset" not in meta:
        raise ValueError(f"Probe dataset metadata at {meta_path} is missing action normalization tensors")

    action_normalizer_meta = {
        "robot_names": [str(meta.get("robot_name", str(cfg.probe.robot_name)))],
        "scale": torch.as_tensor(meta["action_normalizer_scale"], dtype=torch.float32),
        "offset": torch.as_tensor(meta["action_normalizer_offset"], dtype=torch.float32),
    }
    model = _build_probe_policy(
        build_cfg,
        action_dim=int(meta["action_dim"]),
        action_normalizer_meta=action_normalizer_meta,
    ).to(device)
    state_dict = ckpt["model"]
    if prefer_ema and "ema_model" in ckpt:
        state_dict = ckpt["ema_model"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, resolved_ckpt_path, meta


# Reuse the shared synthetic-contact dataset settings, then specialize them to one robot.
def _build_probe_dataset_common_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    common_kwargs = build_contact_policy_common_dataset_kwargs(
        cfg,
        robot_names=[str(cfg.probe.robot_name)],
        global_robot_names=[str(cfg.probe.robot_name)],
    )
    common_kwargs.pop("robot_names", None)
    common_kwargs.pop("global_robot_names", None)
    return common_kwargs


# Resolve optional user paths while keeping probe buffers colocated with checkpoints by default.
def _resolve_probe_buffer_path(path_val: Any, *, save_dir: str, default_name: str) -> str:
    resolved = _resolve_optional_project_path(path_val)
    if resolved is not None:
        return resolved
    return os.path.join(save_dir, default_name)


# Build one fixed split, either by loading a saved buffer or synthesizing and saving a new one.
def _build_probe_dataset(
    cfg: DictConfig,
    *,
    common_kwargs: Dict[str, Any],
    save_dir: str,
    split: str,
    generation_device: torch.device,
) -> tuple[ShadowhandContactPolicyProbeDataset, str]:
    split_key = str(split).strip().lower()
    if split_key not in {"train", "test"}:
        raise ValueError(f"Unsupported split={split}")
    num_samples = int(cfg.probe.train_samples if split_key == "train" else cfg.probe.test_samples)
    buffer_path = _resolve_probe_buffer_path(
        getattr(cfg.dataset, f"{split_key}_buffer_path", None),
        save_dir=save_dir,
        default_name=f"probe_{split_key}_buffer.pt",
    )
    dataset_kwargs = dict(common_kwargs)
    dataset_kwargs["device"] = str(generation_device)
    use_saved_buffer = bool(getattr(cfg.dataset, "load_saved_buffers", True)) and os.path.isfile(buffer_path)
    if use_saved_buffer:
        dataset = ShadowhandContactPolicyProbeDataset.from_buffer_file(
            buffer_path,
            **dataset_kwargs,
        )
        print(f"Loaded {split_key} probe buffer from {buffer_path}")
        return dataset, buffer_path

    dataset = ShadowhandContactPolicyProbeDataset(
        task=str(cfg.probe.task),
        robot_name=str(cfg.probe.robot_name),
        samples=num_samples,
        build_batch_size=int(cfg.dataset.build_batch_size),
        progress_label=f"probe_{split_key}",
        fixed_tip_links=list(cfg.probe.fixed_tip_links),
        supervised_joint_prefixes=list(cfg.probe.supervised_joint_prefixes),
        allowed_contact_link_names=list(cfg.probe.allowed_contact_link_names),
        probe_component_range=tuple(int(x) for x in cfg.probe.component_range),
        probe_component_count_values=None
        if cfg.probe.component_count_values is None
        else list(cfg.probe.component_count_values),
        probe_component_count_distribution=None
        if cfg.probe.component_count_distribution is None
        else list(cfg.probe.component_count_distribution),
        probe_contact_count_range=tuple(int(x) for x in cfg.probe.contact_count_range),
        **dataset_kwargs,
    )
    if bool(getattr(cfg.dataset, "save_buffers", True)):
        dataset.save_buffer(buffer_path)
        print(f"Saved {split_key} probe buffer to {buffer_path}")
    return dataset, buffer_path


# Geometry losses run through a robot model on the training device.
def _build_probe_geometry_model(
    dataset: ShadowhandContactPolicyProbeDataset,
    device: torch.device,
) -> Any:
    if str(getattr(dataset.model, "device", "")) == str(device):
        return dataset.model
    return create_robot_model(
        robot_name=str(dataset.robot_model_name),
        device=device,
        num_points=int(dataset.surface_num_points),
    )


# The probe keeps only one geometry term: predicted contact points should match the target contact cloud.
def _compute_probe_contact_geometry_loss(
    pred_action: torch.Tensor,
    batch: Dict[str, Any],
    *,
    geometry_model: Any,
) -> torch.Tensor:
    pred_local = pred_action[:, : int(geometry_model.dof)].to(geometry_model.device, dtype=torch.float32)
    pred_points, _ = geometry_model.get_surface_points_normals_batch(q=pred_local)
    pred_points = pred_points[:, :, :3]
    contact_valid = batch["contact_valid_mask"].to(geometry_model.device).bool()
    contact_indices = batch["contact_point_indices"].to(geometry_model.device, dtype=torch.long)
    safe_indices = contact_indices.clamp(0, int(pred_points.shape[1]) - 1)
    pred_contact = torch.gather(
        pred_points,
        1,
        safe_indices.unsqueeze(-1).expand(-1, -1, 3),
    )
    target_contact = batch["contact_cloud"][..., :3].to(geometry_model.device, dtype=torch.float32)
    mask = contact_valid.unsqueeze(-1).expand_as(target_contact)
    return ((pred_contact - target_contact).pow(2) * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


# Test-time reporting mirrors the probe objective: action-space fit plus contact alignment.
def _compute_probe_test_metrics(
    policy: ContactDiffusionProbePolicy,
    loader: DataLoader,
    geometry_model: Any,
    device: torch.device,
    *,
    prefix: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    contact_geometry_loss_weight: float,
) -> Dict[str, float]:
    policy.eval()
    full_action_mse = 0.0
    valid_action_mse = 0.0
    contact_point_mse = 0.0
    diffusion_loss_sum = 0.0
    contact_geometry_loss_sum = 0.0
    total_samples = 0
    total_contact_batches = 0

    with torch.no_grad():
        progress = tqdm(total=len(loader), desc=prefix, leave=False)
        for batch in loader:
            batch_device = _move_batch_to_device(batch, device)
            with _autocast_context(device, use_amp, amp_dtype):
                diffusion_loss, _, aux = policy.compute_loss(batch_device, return_aux=True)
            pred_action = policy.predict_action(batch_device)["action_pred"].detach()
            _cuda_sync_if_needed(device)

            gt_action = batch_device["action"].to(dtype=torch.float32)
            action_valid_mask = batch_device["action_valid_mask"].to(dtype=torch.float32)
            sq_error = (pred_action - gt_action).pow(2)
            full_action_mse += float(sq_error.mean(dim=1).sum().item())
            valid_action_mse += float(
                ((sq_error * action_valid_mask).sum(dim=1) / action_valid_mask.sum(dim=1).clamp_min(1.0)).sum().item()
            )
            total_samples += int(gt_action.shape[0])

            contact_geometry_loss = _compute_probe_contact_geometry_loss(
                pred_action.to(dtype=torch.float32),
                batch_device,
                geometry_model=geometry_model,
            )
            diffusion_loss_sum += float(diffusion_loss.item()) * int(gt_action.shape[0])
            contact_geometry_loss_sum += float(contact_geometry_loss.item()) * int(gt_action.shape[0])

            pred_local = pred_action[:, : int(geometry_model.dof)].to(geometry_model.device, dtype=torch.float32)
            pred_points, _ = geometry_model.get_surface_points_normals_batch(q=pred_local)
            pred_points = pred_points[:, :, :3]
            contact_valid = batch_device["contact_valid_mask"].to(geometry_model.device).bool()
            contact_indices = batch_device["contact_point_indices"].to(geometry_model.device, dtype=torch.long)
            safe_indices = contact_indices.clamp(0, int(pred_points.shape[1]) - 1)
            pred_contact = torch.gather(
                pred_points,
                1,
                safe_indices.unsqueeze(-1).expand(-1, -1, 3),
            )
            target_contact = batch_device["contact_cloud"][..., :3].to(geometry_model.device, dtype=torch.float32)
            contact_mask = contact_valid.unsqueeze(-1).expand_as(target_contact)
            per_batch_contact_mse = ((pred_contact - target_contact).pow(2) * contact_mask.float()).sum() / contact_mask.float().sum().clamp_min(1.0)
            contact_point_mse += float(per_batch_contact_mse.item()) * int(gt_action.shape[0])
            total_contact_batches += int(gt_action.shape[0])

            progress.update(1)
        progress.close()

    if total_samples == 0:
        raise RuntimeError("Probe test dataloader returned zero samples")

    avg_diffusion_loss = diffusion_loss_sum / float(total_samples)
    avg_contact_geometry_loss = contact_geometry_loss_sum / float(total_samples)
    return {
        f"{prefix}/loss": avg_diffusion_loss + contact_geometry_loss_weight * avg_contact_geometry_loss,
        f"{prefix}/diffusion_loss": avg_diffusion_loss,
        f"{prefix}/contact_geometry_loss": avg_contact_geometry_loss,
        f"{prefix}/action_mse_full": full_action_mse / float(total_samples),
        f"{prefix}/action_mse_valid": valid_action_mse / float(total_samples),
        f"{prefix}/contact_point_mse": contact_point_mse / float(max(total_contact_batches, 1)),
    }


@hydra.main(version_base="1.2", config_path="conf", config_name="config_contact_policy_probe")
def main(cfg: DictConfig) -> None:
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    # Prepare runtime devices and AMP mode first so dataset generation and model setup stay consistent.
    _set_seed(int(cfg.seed))
    device = _prepare_device(str(cfg.training.device))
    generation_device = _prepare_device(str(cfg.dataset.generation_device))
    use_amp = bool(cfg.training.use_amp)
    amp_dtype = _get_amp_dtype(str(cfg.training.amp_dtype))

    # Build the fixed train/test probe splits and remember where the saved buffers live.
    save_dir = _resolve_project_path(str(cfg.training.save_dir))
    os.makedirs(save_dir, exist_ok=True)

    common_dataset_kwargs = _build_probe_dataset_common_kwargs(cfg)
    train_dataset, train_buffer_path = _build_probe_dataset(
        cfg,
        common_kwargs=common_dataset_kwargs,
        save_dir=save_dir,
        split="train",
        generation_device=generation_device,
    )
    test_dataset, test_buffer_path = _build_probe_dataset(
        cfg,
        common_kwargs=common_dataset_kwargs,
        save_dir=save_dir,
        split="test",
        generation_device=generation_device,
    )

    # The probe uses an ordinary map-style loader because the datasets are fixed buffers.
    loader_kwargs = _build_loader_kwargs(cfg, persistent_workers=bool(int(cfg.dataset.num_workers) > 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.dataset.batch_size),
        shuffle=True,
        drop_last=bool(cfg.dataset.drop_last),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg.dataset.test_batch_size),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    # Action normalization is still built through the shared helper so the probe stays in sync
    # with the main policy codepath.
    action_normalizer_meta = _build_action_normalizer_meta(
        cfg,
        robot_specs=train_dataset.robot_specs,
        global_robot_names=list(train_dataset.global_robot_names),
        action_dim=int(train_dataset.action_dim),
    )
    model = _build_probe_policy(
        cfg,
        action_dim=int(train_dataset.action_dim),
        action_normalizer_meta=action_normalizer_meta,
    ).to(device)

    # EMA handling is identical to the main training script so inference-time behavior matches.
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

    # Optimizer and LR schedule follow the same training recipe as the current contact policy.
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

    # Resume restores the full training state but keeps the fixed saved datasets untouched.
    start_epoch = 1
    global_step = 0
    resume_wandb_run_id = None
    if bool(getattr(cfg.training, "resume", False)):
        resume_ckpt_path = _resolve_optional_project_path(getattr(cfg.training, "resume_checkpoint_path", None))
        if resume_ckpt_path is None:
            resume_ckpt_path = os.path.join(save_dir, "last.pt")
        if not os.path.isfile(resume_ckpt_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt_path}")
        resume_ckpt = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model"], strict=True)
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        lr_scheduler.load_state_dict(resume_ckpt["lr_scheduler"])
        if ema_model is not None:
            if "ema_model" not in resume_ckpt:
                raise KeyError("Resume checkpoint is missing ema_model but training.use_ema=true")
            ema_model.load_state_dict(resume_ckpt["ema_model"], strict=True)
            if ema is not None and "ema_state" in resume_ckpt:
                ema.load_state_dict(resume_ckpt["ema_state"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        resume_wandb_run_id = None if cfg.wandb.run_id is None else str(cfg.wandb.run_id)
        if resume_wandb_run_id is None:
            resume_wandb_run_id = resume_ckpt.get("wandb_run_id")
        if resume_wandb_run_id is None:
            resume_wandb_run_id = _infer_wandb_run_id(save_dir)
        print(f"Resuming probe training from checkpoint={resume_ckpt_path} epoch={start_epoch} global_step={global_step}")

    # WandB setup is intentionally parallel to the main contact-policy training flow.
    run_name = str(cfg.wandb.name) if cfg.wandb.name is not None else str(cfg.experiment.name)
    if bool(cfg.wandb.enabled):
        wandb_init_kwargs = {
            "project": str(cfg.wandb.project),
            "entity": None if cfg.wandb.entity is None else str(cfg.wandb.entity),
            "name": run_name,
            "mode": str(cfg.wandb.mode),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if bool(getattr(cfg.training, "resume", False)) and resume_wandb_run_id is not None:
            wandb_init_kwargs["id"] = str(resume_wandb_run_id)
            wandb_init_kwargs["resume"] = str(getattr(cfg.wandb, "resume", "allow"))
        wandb.init(**wandb_init_kwargs)
    else:
        wandb.init(mode="disabled")

    active_wandb_run_id = None
    if bool(cfg.wandb.enabled) and getattr(wandb, "run", None) is not None:
        active_wandb_run_id = str(wandb.run.id)
        with open(os.path.join(save_dir, "wandb_run_id.txt"), "w", encoding="utf-8") as f:
            f.write(active_wandb_run_id)

    # Save probe-specific metadata so later analysis or visualization can recover the exact split files.
    torch.save(
        {
            "task": str(cfg.probe.task),
            "robot_name": str(cfg.probe.robot_name),
            "action_dim": int(train_dataset.action_dim),
            "surface_num_points": int(train_dataset.surface_num_points),
            "max_contact_points": int(train_dataset.max_contact_points),
            "action_normalizer_scale": action_normalizer_meta["scale"].detach().cpu(),
            "action_normalizer_offset": action_normalizer_meta["offset"].detach().cpu(),
            "train_buffer_path": train_buffer_path,
            "test_buffer_path": test_buffer_path,
        },
        os.path.join(save_dir, "dataset_meta.pt"),
    )

    # The probe only optimizes diffusion loss plus contact geometry loss.
    geometry_model = _build_probe_geometry_model(train_dataset, device)
    contact_geometry_loss_weight = float(cfg.training.contact_geometry_loss_weight)
    log_every_n_steps = int(max(1, cfg.training.log_every_n_steps))
    eval_every_n_epochs = int(max(1, cfg.training.eval_every_n_epochs))
    save_every_n_epoch = int(max(1, cfg.training.save_every_n_epoch))

    if start_epoch > int(cfg.training.epochs):
        print(
            f"Checkpoint epoch {start_epoch - 1} already reaches/exceeds configured training.epochs="
            f"{int(cfg.training.epochs)}. Nothing to do."
        )
        wandb.finish()
        return

    # Standard epoch loop over the fixed training split.
    for epoch in range(start_epoch, int(cfg.training.epochs) + 1):
        model.train()
        train_losses: list[float] = []
        train_diffusion_losses: list[float] = []
        train_contact_geometry_losses: list[float] = []

        # Each step combines masked diffusion supervision with direct contact-point alignment.
        progress = tqdm(total=len(train_loader), desc=f"train:{epoch}")
        for batch_idx, batch in enumerate(train_loader):
            batch_device = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, use_amp, amp_dtype):
                diffusion_loss, loss_dict, aux = model.compute_loss(batch_device, return_aux=True)
            contact_geometry_loss = _compute_probe_contact_geometry_loss(
                aux["pred_action"].to(dtype=torch.float32),
                batch_device,
                geometry_model=geometry_model,
            )
            loss = diffusion_loss + contact_geometry_loss_weight * contact_geometry_loss
            loss.backward()
            if float(cfg.training.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip_norm))
            optimizer.step()
            lr_scheduler.step()
            if ema is not None:
                ema.step(model)

            loss_value = float(loss.item())
            train_losses.append(loss_value)
            train_diffusion_losses.append(float(diffusion_loss.item()))
            train_contact_geometry_losses.append(float(contact_geometry_loss.item()))

            step_log = {
                "train/loss": loss_value,
                "train/diffusion_loss": float(diffusion_loss.item()),
                "train/contact_geometry_loss": float(contact_geometry_loss.item()),
                "train/lr": float(lr_scheduler.get_last_lr()[0]),
                "global_step": global_step,
                "epoch": epoch,
            }
            step_log.update({f"train/{k}": v for k, v in loss_dict.items()})
            should_log_step = (global_step % log_every_n_steps == 0) or (batch_idx + 1 == len(train_loader))
            if should_log_step:
                wandb.log(step_log, step=global_step)

            progress.set_postfix(loss=f"{loss_value:.4f}")
            global_step += 1
            progress.update(1)
        progress.close()

        # Epoch summaries are always logged, and the held-out synthetic test set is evaluated periodically.
        epoch_log = {
            "epoch": epoch,
            "train/loss_epoch": float(np.mean(train_losses)) if train_losses else float("nan"),
            "train/diffusion_loss_epoch": float(np.mean(train_diffusion_losses)) if train_diffusion_losses else float("nan"),
            "train/contact_geometry_loss_epoch": float(np.mean(train_contact_geometry_losses))
            if train_contact_geometry_losses
            else float("nan"),
        }

        run_test = (epoch % eval_every_n_epochs == 0) or (epoch == int(cfg.training.epochs))
        if run_test:
            eval_policy = ema_model if ema_model is not None else model
            test_metrics = _compute_probe_test_metrics(
                eval_policy,
                test_loader,
                geometry_model=geometry_model,
                device=device,
                prefix="test",
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                contact_geometry_loss_weight=contact_geometry_loss_weight,
            )
            epoch_log.update(test_metrics)

        wandb.log(epoch_log, step=global_step)

        # Checkpoints are intentionally simple: latest plus periodic snapshots, no best-on-test selection.
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "wandb_run_id": active_wandb_run_id,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if ema_model is not None:
            ckpt["ema_model"] = ema_model.state_dict()
        if ema is not None:
            ckpt["ema_state"] = ema.state_dict()

        torch.save(ckpt, os.path.join(save_dir, "last.pt"))
        if epoch % save_every_n_epoch == 0:
            torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch}.pt"))

        summary = f"[Epoch {epoch}] train_loss={epoch_log['train/loss_epoch']:.6f}"
        if "test/action_mse_valid" in epoch_log:
            summary += (
                f" test_action_valid={epoch_log['test/action_mse_valid']:.6f}"
                f" test_contact_mse={epoch_log['test/contact_point_mse']:.6f}"
            )
        print(summary)

    wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
