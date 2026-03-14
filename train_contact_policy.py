from __future__ import annotations

import contextlib
import copy
import os
import random
import sys
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
from robot_model.robot_model import create_robot_model  # noqa: E402


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

    def state_dict(self) -> Dict[str, float | int]:
        return {
            "decay": float(self.decay),
            "optimization_step": int(self.optimization_step),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = float(state_dict.get("decay", 0.0))
        self.optimization_step = int(state_dict.get("optimization_step", 0))


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


def _resolve_optional_project_path(path_val: Any) -> str | None:
    if path_val is None:
        return None
    path_str = str(path_val).strip()
    if path_str == "" or path_str.lower() == "null":
        return None
    return _resolve_project_path(path_str)


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


def _extract_wandb_run_id_from_path(path: str) -> str | None:
    real_path = os.path.realpath(path)
    base = os.path.basename(real_path.rstrip(os.sep))
    if not base.startswith("run-"):
        return None
    parts = base.split("-")
    if len(parts) < 3:
        return None
    run_id = parts[-1].strip()
    return run_id or None


def _infer_wandb_run_id(save_dir: str) -> str | None:
    run_id_path = os.path.join(save_dir, "wandb_run_id.txt")
    if os.path.isfile(run_id_path):
        with open(run_id_path, "r", encoding="utf-8") as f:
            run_id = f.read().strip()
        if run_id:
            return run_id

    latest_run_path = os.path.join(ROOT_DIR, "wandb", "latest-run")
    if os.path.exists(latest_run_path):
        run_id = _extract_wandb_run_id_from_path(latest_run_path)
        if run_id is not None:
            return run_id
    return None


def _mark_buffered_dataset_resumed(dataset: Any, start_epoch: int) -> None:
    prepared_epoch = max(1, int(start_epoch) - 1)
    if hasattr(dataset, "_prepared_epoch"):
        dataset._prepared_epoch = prepared_epoch


def _build_action_normalizer_meta(
    cfg: DictConfig,
    *,
    robot_specs: Dict[str, Dict[str, Any]],
    global_robot_names: list[str],
    action_dim: int,
) -> Dict[str, Any]:
    num_robots = len(global_robot_names)
    scale = torch.ones((num_robots, int(action_dim)), dtype=torch.float32)
    offset = torch.zeros((num_robots, int(action_dim)), dtype=torch.float32)
    translation_min = torch.as_tensor(cfg.dataset.q2_base_translation_min, dtype=torch.float32).view(3)
    translation_max = torch.as_tensor(cfg.dataset.q2_base_translation_max, dtype=torch.float32).view(3)
    for robot_idx, robot_name in enumerate(global_robot_names):
        spec = robot_specs[robot_name]
        model = spec["model"]
        lower, upper = model.pk_chain.get_joint_limits()
        lower_t = torch.as_tensor(lower, dtype=torch.float32)
        upper_t = torch.as_tensor(upper, dtype=torch.float32)
        finite = torch.isfinite(lower_t) & torch.isfinite(upper_t)
        lower_t = torch.where(finite, lower_t, torch.full_like(lower_t, -float(np.pi)))
        upper_t = torch.where(finite, upper_t, torch.full_like(upper_t, float(np.pi)))
        if len(model.base_translation_indices) > 0:
            lower_t[model.base_translation_indices] = translation_min[: len(model.base_translation_indices)]
            upper_t[model.base_translation_indices] = translation_max[: len(model.base_translation_indices)]
        if len(model.base_rotation_indices) > 0:
            lower_t[model.base_rotation_indices] = -float(np.pi)
            upper_t[model.base_rotation_indices] = float(np.pi)
        input_range = upper_t - lower_t
        ignore_dim = input_range.abs() < 1e-6
        input_range = input_range.clone()
        input_range[ignore_dim] = 2.0
        robot_scale = 2.0 / input_range
        robot_offset = -1.0 - robot_scale * lower_t
        robot_scale[ignore_dim] = 1.0
        robot_offset[ignore_dim] = -lower_t[ignore_dim]
        scale[robot_idx, : int(model.dof)] = robot_scale
        offset[robot_idx, : int(model.dof)] = robot_offset
    return {
        "robot_names": list(global_robot_names),
        "scale": scale,
        "offset": offset,
        "q2_base_translation_min": translation_min,
        "q2_base_translation_max": translation_max,
    }


def _build_policy(
    cfg: DictConfig,
    action_dim: int,
    *,
    action_normalizer_meta: Dict[str, Any] | None = None,
) -> ContactDiffusionPolicy:
    robot_names: list[str] = []
    action_scale = None
    action_offset = None
    if action_normalizer_meta is not None:
        robot_names = [str(x) for x in action_normalizer_meta.get("robot_names", [])]
        action_scale = action_normalizer_meta.get("scale")
        action_offset = action_normalizer_meta.get("offset")
    return ContactDiffusionPolicy(
        action_dim=action_dim,
        robot_names=robot_names,
        action_normalizer_scale=action_scale,
        action_normalizer_offset=action_offset,
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


def build_contact_policy_common_dataset_kwargs(
    cfg: DictConfig,
    *,
    robot_names: list[str] | None = None,
    global_robot_names: list[str] | None = None,
) -> Dict[str, Any]:
    resolved_global_robot_names = (
        list(cfg.dataset.robot_names)
        if global_robot_names is None
        else [str(x) for x in global_robot_names]
    )
    return {
        "robot_names": list(cfg.dataset.robot_names) if robot_names is None else [str(x) for x in robot_names],
        "global_robot_names": resolved_global_robot_names,
        "hparams_path": _resolve_project_path(str(cfg.dataset.hparams_path)),
        "real_masks_path": _resolve_project_path(str(cfg.dataset.real_masks_path)),
        "max_contact_points": int(cfg.dataset.max_contact_points),
        "seed": int(cfg.seed),
        "check_template_hash": bool(cfg.dataset.check_template_hash),
        "threshold": float(cfg.dataset.threshold),
        "align_exp_scale": float(cfg.dataset.align_exp_scale),
        "sigmoid_scale": float(cfg.dataset.sigmoid_scale),
        "component_sampling_mode": str(cfg.dataset.component_sampling_mode),
        "anchor_sampling_mode": str(cfg.dataset.anchor_sampling_mode),
        "anchor_temperature": float(cfg.dataset.anchor_temperature),
        "patch_anchor_shift_min": float(cfg.dataset.patch_anchor_shift_min),
        "patch_anchor_shift_max": float(cfg.dataset.patch_anchor_shift_max),
        "patch_extent_min": float(cfg.dataset.patch_extent_min),
        "patch_extent_max": float(cfg.dataset.patch_extent_max),
        "patch_extent_power": float(cfg.dataset.patch_extent_power),
        "patch_shift_power": float(cfg.dataset.patch_shift_power),
        "patch_normal_jitter_max_deg": float(cfg.dataset.patch_normal_jitter_max_deg),
        "patch_points_per_anchor_min": int(cfg.dataset.patch_points_per_anchor_min),
        "patch_points_per_anchor_max": int(cfg.dataset.patch_points_per_anchor_max),
        "patch_penetration_clearance": float(cfg.dataset.patch_penetration_clearance),
        "q2_base_translation_min": [float(x) for x in cfg.dataset.q2_base_translation_min],
        "q2_base_translation_max": [float(x) for x in cfg.dataset.q2_base_translation_max],
        "q2_base_rotation_mode": str(cfg.dataset.q2_base_rotation_mode),
    }


def load_contact_policy_checkpoint(
    cfg: DictConfig,
    *,
    action_dim: int | None,
    device: torch.device,
    ckpt_path: str | None = None,
    prefer_ema: bool = True,
) -> tuple[ContactDiffusionPolicy, str, Dict[str, Any]]:
    resolved_ckpt_path = (
        _resolve_project_path(os.path.join(str(cfg.training.save_dir), "best.pt"))
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

    resolved_action_dim = action_dim
    meta: Dict[str, Any] = {}
    action_normalizer_meta: Dict[str, Any] | None = None
    meta_path = os.path.join(os.path.dirname(resolved_ckpt_path), "dataset_meta.pt")
    if os.path.isfile(meta_path):
        meta = torch.load(meta_path, map_location="cpu")
        if isinstance(meta, dict) and "action_dim" in meta:
            resolved_action_dim = int(meta["action_dim"])
        if (
            isinstance(meta, dict)
            and "action_normalizer_scale" in meta
            and "action_normalizer_offset" in meta
            and "robot_names" in meta
        ):
            action_normalizer_meta = {
                "robot_names": [str(x) for x in meta["robot_names"]],
                "scale": torch.as_tensor(meta["action_normalizer_scale"], dtype=torch.float32),
                "offset": torch.as_tensor(meta["action_normalizer_offset"], dtype=torch.float32),
            }
    if resolved_action_dim is None:
        raise ValueError("action_dim must be provided when dataset_meta.pt is unavailable")
    if action_normalizer_meta is None:
        raise ValueError(
            f"Checkpoint metadata at {meta_path} is missing action normalizer tensors required by the current policy"
        )

    model = _build_policy(
        build_cfg,
        action_dim=int(resolved_action_dim),
        action_normalizer_meta=action_normalizer_meta,
    ).to(device)
    state_dict = ckpt["model"]
    if prefer_ema and "ema_model" in ckpt:
        state_dict = ckpt["ema_model"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, resolved_ckpt_path, meta


def _build_loader_kwargs(cfg: DictConfig, *, persistent_workers: bool) -> Dict[str, Any]:
    num_workers = int(cfg.dataset.num_workers)
    kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(cfg.dataset.pin_memory),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(cfg.dataset.prefetch_factor)
    return kwargs


def _build_val_subset_indices(dataset_len: int, subset_size: int, seed: int) -> list[int]:
    n = int(max(0, subset_size))
    if n <= 0 or n >= dataset_len:
        return list(range(dataset_len))
    rng = np.random.default_rng(int(seed))
    return sorted(int(x) for x in rng.choice(dataset_len, size=n, replace=False).tolist())


def _build_training_geometry_models(
    train_dataset: OnTheFlySyntheticContactPolicyDataset,
    device: torch.device,
) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for robot_name, spec in train_dataset.robot_specs.items():
        model = spec["model"]
        if str(getattr(model, "device", "")) == str(device):
            models[robot_name] = model
            continue
        models[robot_name] = create_robot_model(
            robot_name=str(spec["robot_model_name"]),
            device=device,
            num_points=int(spec["surface_num_points"]),
        )
    return models


def _copy_base_pose_from_q(
    model: Any,
    q_target: torch.Tensor,
    q_source: torch.Tensor,
) -> torch.Tensor:
    if q_target.ndim == 1:
        q_target = q_target.unsqueeze(0)
    if q_source.ndim == 1:
        q_source = q_source.unsqueeze(0)
    if q_target.shape != q_source.shape:
        raise ValueError(f"q_target and q_source must have the same shape, got {tuple(q_target.shape)} vs {tuple(q_source.shape)}")
    base_translation = None
    base_rotation_euler = None
    if len(model.base_translation_indices) > 0:
        base_translation = q_source[:, model.base_translation_indices]
    if len(model.base_rotation_indices) > 0:
        base_rotation_euler = q_source[:, model.base_rotation_indices]
    return model.set_base_pose_in_q(
        q_target,
        base_translation=base_translation,
        base_rotation_euler=base_rotation_euler,
    )


def _compute_training_geometry_losses(
    pred_action: torch.Tensor,
    batch: Dict[str, Any],
    *,
    geometry_models: Dict[str, Any],
    robot_names: list[str],
) -> Dict[str, torch.Tensor]:
    device = pred_action.device
    contact_sse = torch.zeros((), dtype=torch.float32, device=device)
    contact_count = torch.zeros((), dtype=torch.float32, device=device)
    inactive_sse = torch.zeros((), dtype=torch.float32, device=device)
    inactive_count = torch.zeros((), dtype=torch.float32, device=device)

    robot_index = batch["robot_index"].to(device=device, dtype=torch.long).view(-1)
    q1_padded = batch["q1_padded"].to(device=device, dtype=torch.float32)
    q2_padded = batch["q2_padded"].to(device=device, dtype=torch.float32)

    for ridx in torch.unique(robot_index):
        robot_idx = int(ridx.item())
        sample_mask = robot_index == ridx
        sample_ids = torch.nonzero(sample_mask, as_tuple=False).view(-1)
        if int(sample_ids.numel()) == 0:
            continue

        robot_name = robot_names[robot_idx]
        model = geometry_models[robot_name]
        pred_local = pred_action[sample_ids, : int(model.dof)].to(model.device, dtype=torch.float32)
        q1_local = q1_padded[sample_ids, : int(model.dof)].to(model.device, dtype=torch.float32)
        q2_local = q2_padded[sample_ids, : int(model.dof)].to(model.device, dtype=torch.float32)
        pred_points, _ = model.get_surface_points_normals_batch(q=pred_local)
        pred_points = pred_points[:, :, :3]

        contact_valid = batch["contact_valid_mask"][sample_ids].to(model.device).bool()
        contact_indices = batch["contact_point_indices"][sample_ids].to(model.device, dtype=torch.long)
        safe_contact_indices = contact_indices.clamp(0, int(pred_points.shape[1]) - 1)
        pred_contact = torch.gather(
            pred_points,
            1,
            safe_contact_indices.unsqueeze(-1).expand(-1, -1, 3),
        )
        target_contact = batch["contact_cloud"][sample_ids, :, :3].to(model.device, dtype=torch.float32)
        contact_mask = contact_valid.unsqueeze(-1).expand_as(target_contact)
        contact_sse = contact_sse + ((pred_contact - target_contact).pow(2) * contact_mask.float()).sum()
        contact_count = contact_count + contact_mask.float().sum()

        q1_with_q2_base = _copy_base_pose_from_q(model, q1_local, q2_local)
        q1_with_q2_base_points, _ = model.get_surface_points_normals_batch(q=q1_with_q2_base)
        q1_with_q2_base_points = q1_with_q2_base_points[:, :, :3]

        inactive_link_mask = batch["inactive_link_mask"][sample_ids].to(model.device).bool()
        inactive_link_mask = inactive_link_mask[:, : len(model.mesh_link_names)]
        link_indices = model.surface_template_link_indices.to(model.device, dtype=torch.long)
        point_inactive = torch.gather(
            inactive_link_mask,
            1,
            link_indices.view(1, -1).expand(int(sample_ids.numel()), -1),
        )
        inactive_mask_3 = point_inactive.unsqueeze(-1).expand_as(q1_with_q2_base_points)
        inactive_sse = inactive_sse + ((pred_points - q1_with_q2_base_points).pow(2) * inactive_mask_3.float()).sum()
        inactive_count = inactive_count + inactive_mask_3.float().sum()

    return {
        "contact_geometry_loss": contact_sse / contact_count.clamp_min(1.0),
        "inactive_geometry_loss": inactive_sse / inactive_count.clamp_min(1.0),
    }


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
            try:
                batch = next(iterator)
            except StopIteration:
                break

            batch_device = _move_batch_to_device(batch, device)
            with _autocast_context(device, use_amp, amp_dtype):
                pred_action = policy.predict_action(batch_device)["action_pred"].detach()
            _cuda_sync_if_needed(device)
            pred_action = pred_action.cpu()

            gt_action = batch["action"].float()
            action_valid_mask = batch["action_valid_mask"].float()
            sq_error = (pred_action - gt_action).pow(2)
            full_action_mse += float(sq_error.mean(dim=1).sum().item())
            valid_action_mse += float(
                ((sq_error * action_valid_mask).sum(dim=1) / action_valid_mask.sum(dim=1).clamp_min(1.0)).sum().item()
            )
            n_action += int(gt_action.shape[0])

            for i in range(int(gt_action.shape[0])):
                robot_name = str(batch["robot_name"][i])
                model = metric_models[robot_name]
                local_dof = int(batch["local_dof"][i].item())
                pred_local = pred_action[i, :local_dof].to(model.device)
                q1_local = batch["q1_padded"][i, :local_dof].to(model.device, dtype=torch.float32)
                q2_local = batch["q2_padded"][i, :local_dof].to(model.device, dtype=torch.float32)
                contact_valid = batch["contact_valid_mask"][i].bool()
                contact_indices = batch["contact_point_indices"][i][contact_valid].long()

                pred_points, _ = model.get_surface_points_normals(q=pred_local)
                pred_points = pred_points.detach().cpu()

                if int(contact_indices.numel()) > 0:
                    target_contact = batch["contact_cloud"][i, contact_valid, :3].float()
                    contact_point_mse += float(
                        torch.mean((pred_points[contact_indices, :3] - target_contact) ** 2).item()
                    )
                    n_contact += 1

                q1_with_q2_base = _copy_base_pose_from_q(model, q1_local, q2_local)
                q1_with_q2_base_points, _ = model.get_surface_points_normals(q=q1_with_q2_base)
                q1_with_q2_base_points = q1_with_q2_base_points.detach().cpu()

                inactive_link_mask = batch["inactive_link_mask"][i, : len(model.mesh_link_names)].bool()
                link_indices = model.surface_template_link_indices.detach().cpu()
                inactive_point_mask = inactive_link_mask[link_indices]
                inactive_idx = torch.nonzero(inactive_point_mask, as_tuple=False).view(-1)
                if int(inactive_idx.numel()) > 0:
                    inactive_surface_mse += float(
                        torch.mean((pred_points[inactive_idx, :3] - q1_with_q2_base_points[inactive_idx, :3]) ** 2).item()
                    )
                    n_inactive += 1

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

    common_dataset_kwargs = build_contact_policy_common_dataset_kwargs(cfg)
    train_generation_device = _prepare_device(str(cfg.dataset.train_generation_device))
    real_val_build_device = _prepare_device(
        str(getattr(cfg.dataset, "real_val_build_device", cfg.dataset.robot_model_device))
    )
    train_dataset = OnTheFlySyntheticContactPolicyDataset(
        samples_per_epoch=int(cfg.dataset.train_samples_per_epoch),
        buffer_refresh_fraction=float(cfg.dataset.buffer_refresh_fraction),
        buffer_build_batch_size_per_robot=int(cfg.dataset.buffer_build_batch_size_per_robot),
        buffer_refresh_batch_size_per_robot=int(cfg.dataset.buffer_refresh_batch_size_per_robot),
        device=str(train_generation_device),
        store_aux_metadata=True,
        store_full_metadata=False,
        progress_label="train_synth",
        **common_dataset_kwargs,
    )
    synthetic_val_dataset = OnTheFlySyntheticContactPolicyDataset(
        samples_per_epoch=int(cfg.dataset.synthetic_val_samples_per_epoch),
        buffer_refresh_fraction=float(cfg.dataset.buffer_refresh_fraction),
        buffer_build_batch_size_per_robot=int(cfg.dataset.buffer_build_batch_size_per_robot),
        buffer_refresh_batch_size_per_robot=int(cfg.dataset.buffer_refresh_batch_size_per_robot),
        device=str(train_generation_device),
        store_full_metadata=True,
        progress_label="validate_synth",
        **common_dataset_kwargs,
    )
    val_dataset = RealContactPolicyValDataset(
        cache_processed_samples=bool(cfg.dataset.cache_validation_samples),
        cache_build_batch_size=int(getattr(cfg.dataset, "real_val_cache_build_batch_size", 256)),
        device=str(real_val_build_device),
        progress_label="validate_real",
        **common_dataset_kwargs,
    )

    train_loader_kwargs = _build_loader_kwargs(cfg, persistent_workers=False)
    synthetic_val_loader_kwargs = _build_loader_kwargs(cfg, persistent_workers=False)
    real_val_loader_kwargs = _build_loader_kwargs(
        cfg,
        persistent_workers=bool(cfg.dataset.num_workers > 0),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.dataset.batch_size),
        shuffle=True,
        drop_last=bool(cfg.dataset.drop_last),
        **train_loader_kwargs,
    )
    synthetic_val_loader = DataLoader(
        synthetic_val_dataset,
        batch_size=int(cfg.dataset.val_batch_size),
        shuffle=False,
        drop_last=False,
        **synthetic_val_loader_kwargs,
    )

    full_val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.dataset.val_batch_size),
        shuffle=False,
        drop_last=False,
        **real_val_loader_kwargs,
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
            **real_val_loader_kwargs,
        )

    action_normalizer_meta = _build_action_normalizer_meta(
        cfg,
        robot_specs=train_dataset.robot_specs,
        global_robot_names=list(train_dataset.global_robot_names),
        action_dim=int(train_dataset.action_dim),
    )
    model = _build_policy(
        cfg,
        action_dim=train_dataset.action_dim,
        action_normalizer_meta=action_normalizer_meta,
    ).to(device)
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
    save_dir = _resolve_project_path(str(cfg.training.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    best_metric_key = "validate_real/action_mse_valid"
    best_val = float("inf")
    start_epoch = 1
    global_step = 0
    resume_ckpt_path = None
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
        best_val = float(resume_ckpt.get("best_val", float("inf")))
        resume_wandb_run_id = (
            None
            if getattr(cfg.wandb, "run_id", None) is None
            else str(cfg.wandb.run_id)
        )
        if resume_wandb_run_id is None:
            resume_wandb_run_id = resume_ckpt.get("wandb_run_id")
        if resume_wandb_run_id is None:
            resume_wandb_run_id = _infer_wandb_run_id(save_dir)

        # Synthetic buffers are regenerated on each launch; skip replaying all previous
        # epochs and start refreshing from the next epoch to keep resume practical.
        _mark_buffered_dataset_resumed(train_dataset, start_epoch)
        _mark_buffered_dataset_resumed(synthetic_val_dataset, start_epoch)

        print(
            f"Resuming training from checkpoint={resume_ckpt_path} "
            f"epoch={start_epoch} global_step={global_step}"
        )

    if bool(getattr(cfg.training, "save_frozen_train_buffer", False)):
        if float(cfg.dataset.buffer_refresh_fraction) != 0.0:
            raise ValueError("training.save_frozen_train_buffer requires dataset.buffer_refresh_fraction=0")
        frozen_train_buffer_path = _resolve_optional_project_path(getattr(cfg.training, "frozen_train_buffer_path", None))
        if frozen_train_buffer_path is None:
            frozen_train_buffer_path = os.path.join(save_dir, "frozen_train_buffer.pt")
        train_dataset.save_buffer(frozen_train_buffer_path)
        print(f"Saved frozen train buffer to {frozen_train_buffer_path}")

    run_name = str(cfg.wandb.name) if cfg.wandb.name is not None else str(cfg.experiment.name)
    if bool(cfg.wandb.enabled):
        wandb_init_kwargs = {
            "project": str(cfg.wandb.project),
            "entity": None if cfg.wandb.entity is None else str(cfg.wandb.entity),
            "name": run_name,
            "mode": str(cfg.wandb.mode),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if bool(getattr(cfg.training, "resume", False)):
            if resume_wandb_run_id is None and str(getattr(cfg.wandb, "resume", "allow")).lower() == "must":
                raise ValueError("wandb.resume=must requires a run_id when resuming")
            if resume_wandb_run_id is not None:
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

    torch.save(
        {
            "robot_names": list(train_dataset.global_robot_names),
            "action_dim": int(train_dataset.action_dim),
            "surface_num_points": int(train_dataset.surface_num_points),
            "max_contact_points": int(train_dataset.max_contact_points),
            "action_normalizer_scale": action_normalizer_meta["scale"].detach().cpu(),
            "action_normalizer_offset": action_normalizer_meta["offset"].detach().cpu(),
            "q2_base_translation_min": action_normalizer_meta["q2_base_translation_min"].detach().cpu(),
            "q2_base_translation_max": action_normalizer_meta["q2_base_translation_max"].detach().cpu(),
        },
        os.path.join(save_dir, "dataset_meta.pt"),
    )

    metric_models = {name: spec["model"] for name, spec in val_dataset.robot_specs.items()}
    train_geometry_models = _build_training_geometry_models(train_dataset, device)
    log_every_n_steps = int(max(1, cfg.training.log_every_n_steps))
    max_train_batches = int(cfg.training.max_train_batches)
    max_val_batches = int(cfg.training.max_val_batches)
    contact_geometry_loss_weight = float(getattr(cfg.training, "contact_geometry_loss_weight", 0.0))
    inactive_geometry_loss_weight = float(getattr(cfg.training, "inactive_geometry_loss_weight", 0.0))

    if start_epoch > int(cfg.training.epochs):
        print(
            f"Checkpoint epoch {start_epoch - 1} already reaches/exceeds configured training.epochs="
            f"{int(cfg.training.epochs)}. Nothing to do."
        )
        wandb.finish()
        return

    for epoch in range(start_epoch, int(cfg.training.epochs) + 1):
        train_dataset.prepare_epoch(epoch)
        model.train()
        train_losses: list[float] = []
        train_diffusion_losses: list[float] = []
        train_contact_geometry_losses: list[float] = []
        train_inactive_geometry_losses: list[float] = []

        train_batches_total = len(train_loader)
        if max_train_batches > 0:
            train_batches_total = min(train_batches_total, max_train_batches)

        iterator = iter(train_loader)
        progress = tqdm(total=train_batches_total, desc=f"train:{epoch}")
        batch_idx = 0
        while batch_idx < train_batches_total:
            try:
                batch = next(iterator)
            except StopIteration:
                break

            batch_device = _move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, use_amp, amp_dtype):
                diffusion_loss, loss_dict, loss_aux = model.compute_loss(batch_device, return_aux=True)
            contact_geometry_loss = torch.zeros((), dtype=torch.float32, device=device)
            inactive_geometry_loss = torch.zeros((), dtype=torch.float32, device=device)
            if contact_geometry_loss_weight > 0.0 or inactive_geometry_loss_weight > 0.0:
                geometry_losses = _compute_training_geometry_losses(
                    loss_aux["pred_action"].to(dtype=torch.float32),
                    batch_device,
                    geometry_models=train_geometry_models,
                    robot_names=list(train_dataset.global_robot_names),
                )
                contact_geometry_loss = geometry_losses["contact_geometry_loss"]
                inactive_geometry_loss = geometry_losses["inactive_geometry_loss"]
            loss = diffusion_loss
            if contact_geometry_loss_weight > 0.0:
                loss = loss + contact_geometry_loss_weight * contact_geometry_loss
            if inactive_geometry_loss_weight > 0.0:
                loss = loss + inactive_geometry_loss_weight * inactive_geometry_loss
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
            train_inactive_geometry_losses.append(float(inactive_geometry_loss.item()))

            step_log = {
                "train/loss": loss_value,
                "train/diffusion_loss": float(diffusion_loss.item()),
                "train/contact_geometry_loss": float(contact_geometry_loss.item()),
                "train/inactive_geometry_loss": float(inactive_geometry_loss.item()),
                "train/lr": float(lr_scheduler.get_last_lr()[0]),
                "global_step": global_step,
                "epoch": epoch,
            }
            step_log.update({f"train/{k}": v for k, v in loss_dict.items()})
            should_log_step = (global_step % log_every_n_steps == 0) or (batch_idx + 1 == train_batches_total)
            if should_log_step:
                wandb.log(step_log, step=global_step)

            progress.set_postfix(loss=f"{loss_value:.4f}")
            global_step += 1
            batch_idx += 1
            progress.update(1)
        progress.close()

        epoch_log = {
            "epoch": epoch,
            "train/loss_epoch": float(np.mean(train_losses)) if train_losses else float("nan"),
            "train/diffusion_loss_epoch": float(np.mean(train_diffusion_losses)) if train_diffusion_losses else float("nan"),
            "train/contact_geometry_loss_epoch": float(np.mean(train_contact_geometry_losses)) if train_contact_geometry_losses else float("nan"),
            "train/inactive_geometry_loss_epoch": float(np.mean(train_inactive_geometry_losses)) if train_inactive_geometry_losses else float("nan"),
        }

        eval_policy = ema_model if ema_model is not None else model
        synthetic_val_dataset.prepare_epoch(epoch)
        synth_metrics = _compute_validation_metrics(
            eval_policy,
            synthetic_val_loader,
            metric_models=metric_models,
            device=device,
            prefix="validate_synth",
            max_batches=max_val_batches,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        epoch_log.update(synth_metrics)

        real_metrics = _compute_validation_metrics(
            eval_policy,
            subset_val_loader,
            metric_models=metric_models,
            device=device,
            prefix="validate_real",
            max_batches=max_val_batches,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        epoch_log.update(real_metrics)

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
                prefix="validate_real_full",
                max_batches=max_val_batches,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            epoch_log.update(full_metrics)
        elif subset_val_loader is full_val_loader:
            epoch_log["validate_real_full/action_mse_valid"] = epoch_log[best_metric_key]

        wandb.log(epoch_log, step=global_step)

        if float(epoch_log[best_metric_key]) < best_val:
            best_val = float(epoch_log[best_metric_key])

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val": float(best_val),
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
        if epoch % int(cfg.training.save_every_n_epoch) == 0:
            torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch}.pt"))
        if float(epoch_log[best_metric_key]) <= best_val:
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))

        summary = (
            f"[Epoch {epoch}] "
            f"train_loss={epoch_log['train/loss_epoch']:.6f} "
            f"synth_action_valid={epoch_log['validate_synth/action_mse_valid']:.6f} "
            f"real_action_valid={epoch_log['validate_real/action_mse_valid']:.6f}"
        )
        if "validate_real_full/action_mse_valid" in epoch_log:
            summary += f" real_full_action_valid={epoch_log['validate_real_full/action_mse_valid']:.6f}"
        print(summary)

    wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
