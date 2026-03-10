from __future__ import annotations

import hashlib
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from robot_model.robot_model import create_robot_model, farthest_point_sampling


def _stable_int_seed(key: str) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**31 - 1)


def _normalize_num_points_by_robot(
    num_points_by_robot: Dict[str, int],
) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted((str(robot_name), int(num_points)) for robot_name, num_points in num_points_by_robot.items()))


@lru_cache(maxsize=4)
def _load_real_mask_artifacts_cached(
    real_masks_path: str,
    num_points_items: Tuple[Tuple[str, int], ...],
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    payload = torch.load(real_masks_path, map_location="cpu")
    if not isinstance(payload, dict) or "samples" not in payload:
        raise ValueError(f"Invalid real mask payload: {real_masks_path}")

    num_points_by_robot = {robot_name: int(num_points) for robot_name, num_points in num_points_items}
    masks_by_robot_list: Dict[str, List[torch.Tensor]] = {}
    validation_samples: List[Dict[str, Any]] = []
    for sample_idx, sample in enumerate(payload["samples"]):
        robot_name = str(sample.get("robot_name", ""))
        if robot_name not in num_points_by_robot:
            continue
        if sample.get("invalid_reason") is not None:
            continue
        mask = torch.as_tensor(sample["hand_contact_mask"]).bool().view(-1)
        if int(mask.numel()) != int(num_points_by_robot[robot_name]):
            continue
        masks_by_robot_list.setdefault(robot_name, []).append(mask)
        validation_samples.append(
            {
                "sample_id": int(sample.get("sample_id", sample_idx)),
                "robot_name": robot_name,
                "q2": torch.as_tensor(sample["q"], dtype=torch.float32).view(-1),
                "mask": mask,
            }
        )

    stacked_masks: Dict[str, torch.Tensor] = {}
    for robot_name, masks in masks_by_robot_list.items():
        if len(masks) > 0:
            stacked_masks[robot_name] = torch.stack(masks, dim=0).bool()
    return stacked_masks, validation_samples


def _load_real_mask_artifacts(
    real_masks_path: str | Path,
    num_points_by_robot: Dict[str, int],
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    return _load_real_mask_artifacts_cached(
        str(Path(real_masks_path).resolve()),
        _normalize_num_points_by_robot(num_points_by_robot),
    )


def _extract_contact_count_range_from_real_masks(
    real_masks: torch.Tensor,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> Tuple[int, int]:
    counts = real_masks.detach().bool().cpu().sum(dim=1).float()
    if int(counts.numel()) == 0:
        return 1, 1
    lower = int(torch.quantile(counts, q=float(lower_quantile)).floor().item())
    upper = int(torch.quantile(counts, q=float(upper_quantile)).ceil().item())
    lower = max(1, lower)
    upper = max(lower, upper)
    return lower, upper


def _init_contact_count_interval_stats() -> Dict[str, int]:
    return {
        "num_samples": 0,
        "in_range": 0,
        "too_low": 0,
        "too_high": 0,
    }


def _merge_contact_count_interval_stats(
    acc: Dict[str, int],
    stats: Dict[str, int],
) -> Dict[str, int]:
    for key in acc.keys():
        acc[key] += int(stats.get(key, 0))
    return acc


def _format_contact_count_interval_summary(
    prefix: str,
    stats: Dict[str, int],
) -> str:
    num_samples = max(1, int(stats.get("num_samples", 0)))
    too_low = int(stats.get("too_low", 0))
    too_high = int(stats.get("too_high", 0))
    out_of_range = too_low + too_high
    return (
        f"[{prefix}] contact_count_interval out_of_range={out_of_range}/{num_samples} "
        f"({out_of_range / num_samples:.2%}) too_low={too_low} too_high={too_high}"
    )


def _build_component_distribution_from_range(
    component_range: Tuple[int, int],
    *,
    values: Any | None = None,
    probabilities: Any | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    comp_lo, comp_hi = int(component_range[0]), int(component_range[1])
    support = torch.arange(comp_lo, comp_hi + 1, dtype=torch.long)
    probs = torch.ones((int(support.numel()),), dtype=torch.float32)
    if values is not None and probabilities is not None:
        values_t = torch.as_tensor(values, dtype=torch.long).view(-1)
        probs_t = torch.as_tensor(probabilities, dtype=torch.float32).view(-1)
        if int(values_t.numel()) == int(probs_t.numel()) and int(values_t.numel()) > 0:
            valid = (values_t >= comp_lo) & (values_t <= comp_hi) & (probs_t > 0)
            if bool(valid.any()):
                support = values_t[valid]
                probs = probs_t[valid]
    probs = probs / probs.sum().clamp_min(1e-8)
    return support.long(), probs.float()


def _build_point_prob_from_real_masks(real_masks: torch.Tensor) -> torch.Tensor:
    point_prob = real_masks.detach().bool().cpu().float().mean(dim=0).clamp_min(1e-8)
    return point_prob / point_prob.sum().clamp_min(1e-8)


def _sample_anchor_indices_batch(
    *,
    batch_size: int,
    max_anchors: int,
    num_points: int,
    point_prob: torch.Tensor | None,
    anchor_sampling_mode: str,
    generator: torch.Generator,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    if max_anchors <= 0:
        return torch.zeros((batch_size, 0), dtype=torch.long, device=device)

    max_anchors = int(max(1, min(int(max_anchors), int(num_points))))
    mode = str(anchor_sampling_mode).strip().lower()
    if mode == "uniform":
        weights = torch.ones((batch_size, int(num_points)), dtype=torch.float32, device=device)
    elif mode == "point_prob":
        if point_prob is None:
            raise ValueError("point_prob is required when anchor_sampling_mode='point_prob'")
        weights = point_prob.to(device=device, dtype=torch.float32).clamp_min(1e-8)
        t = float(max(1e-3, float(temperature)))
        if abs(t - 1.0) > 1e-6:
            weights = weights.pow(1.0 / t)
        weights = weights / weights.sum().clamp_min(1e-8)
        weights = weights.unsqueeze(0).expand(int(batch_size), -1)
    else:
        raise ValueError(f"Invalid anchor_sampling_mode={anchor_sampling_mode}")

    return torch.multinomial(weights, num_samples=max_anchors, replacement=False, generator=generator)


def _build_topk_mask_from_sorted(sorted_idx: torch.Tensor, k_values: torch.Tensor, n_points: int) -> torch.Tensor:
    if sorted_idx.ndim != 2:
        raise ValueError(f"sorted_idx must have shape (B, N), got {tuple(sorted_idx.shape)}")
    k_values = k_values.to(device=sorted_idx.device, dtype=torch.long).clamp(0, int(n_points))
    rank = torch.arange(int(n_points), device=sorted_idx.device).unsqueeze(0)
    keep_sorted = rank < k_values.unsqueeze(1)
    out = torch.zeros_like(keep_sorted, dtype=torch.bool)
    out.scatter_(1, sorted_idx, keep_sorted)
    return out


class _BaseContactPolicyDataset:
    def __init__(
        self,
        *,
        robot_names: List[str],
        global_robot_names: List[str] | None = None,
        hparams_path: str | Path,
        real_masks_path: str | Path,
        max_contact_points: int = 256,
        seed: int = 42,
        device: str = "cpu",
        check_template_hash: bool = True,
        threshold: float = 0.4,
        align_exp_scale: float = 2.0,
        sigmoid_scale: float = 10.0,
        component_sampling_mode: str = "distribution",
        anchor_sampling_mode: str = "point_prob",
        anchor_temperature: float = 1.0,
        patch_anchor_shift_min: float = 0.0002,
        patch_anchor_shift_max: float = 0.0030,
        patch_extent_min: float = 0.006,
        patch_extent_max: float = 0.050,
        patch_extent_power: float = 1.0,
        patch_shift_power: float = 1.0,
        patch_normal_jitter_max_deg: float = 10.0,
        patch_points_per_anchor_min: int = 28,
        patch_points_per_anchor_max: int = 84,
        patch_penetration_clearance: float = 0.00008,
        q2_base_translation_min: List[float] | Tuple[float, float, float] = (-0.10, -0.10, -0.10),
        q2_base_translation_max: List[float] | Tuple[float, float, float] = (0.10, 0.10, 0.10),
        q2_base_rotation_mode: str = "uniform_so3",
    ) -> None:
        self.robot_names = [str(x) for x in robot_names]
        if global_robot_names is None:
            global_robot_names = self.robot_names
        self.global_robot_names = [str(x) for x in global_robot_names]
        if len(self.global_robot_names) == 0:
            raise ValueError("global_robot_names must be non-empty")
        missing_global = sorted(set(self.robot_names) - set(self.global_robot_names))
        if missing_global:
            raise ValueError(
                f"robot_names must be a subset of global_robot_names, missing={missing_global}"
            )
        self.robot_name_to_global_index = {
            name: idx for idx, name in enumerate(self.global_robot_names)
        }
        self.hparams_path = Path(hparams_path)
        self.real_masks_path = Path(real_masks_path)
        self.max_contact_points = int(max_contact_points)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.check_template_hash = bool(check_template_hash)

        self.threshold = float(threshold)
        self.align_exp_scale = float(align_exp_scale)
        self.sigmoid_scale = float(sigmoid_scale)
        self.component_sampling_mode = str(component_sampling_mode).strip().lower()
        self.anchor_sampling_mode = str(anchor_sampling_mode)
        self.anchor_temperature = float(anchor_temperature)
        self.patch_anchor_shift_min = float(patch_anchor_shift_min)
        self.patch_anchor_shift_max = float(patch_anchor_shift_max)
        self.patch_extent_min = float(patch_extent_min)
        self.patch_extent_max = float(patch_extent_max)
        self.patch_extent_power = float(patch_extent_power)
        self.patch_shift_power = float(patch_shift_power)
        self.patch_normal_jitter_max_deg = float(patch_normal_jitter_max_deg)
        self.patch_points_per_anchor_min = int(patch_points_per_anchor_min)
        self.patch_points_per_anchor_max = int(patch_points_per_anchor_max)
        self.patch_penetration_clearance = float(patch_penetration_clearance)
        q2_base_translation_min_t = torch.as_tensor(q2_base_translation_min, dtype=torch.float32)
        q2_base_translation_max_t = torch.as_tensor(q2_base_translation_max, dtype=torch.float32)
        if q2_base_translation_min_t.shape != (3,) or q2_base_translation_max_t.shape != (3,):
            raise ValueError("q2_base_translation_min/max must be length-3 vectors")
        if bool((q2_base_translation_max_t < q2_base_translation_min_t).any()):
            raise ValueError("q2_base_translation_max must be >= q2_base_translation_min elementwise")
        self.q2_base_translation_min = q2_base_translation_min_t.to(self.device)
        self.q2_base_translation_max = q2_base_translation_max_t.to(self.device)
        self.q2_base_rotation_mode = str(q2_base_rotation_mode).strip().lower()
        if self.q2_base_rotation_mode != "uniform_so3":
            raise ValueError(
                f"Unsupported q2_base_rotation_mode={q2_base_rotation_mode}. Expected 'uniform_so3'."
            )

        with open(self.hparams_path, "r", encoding="utf-8") as f:
            hparams = yaml.safe_load(f)
        if not isinstance(hparams, dict) or "robots" not in hparams:
            raise ValueError(f"Invalid hparams yaml: {self.hparams_path}")
        robot_cfgs = hparams["robots"]
        if not isinstance(robot_cfgs, dict):
            raise ValueError(f"Invalid robots payload in {self.hparams_path}")

        num_points_by_robot = {
            robot_name: int(robot_cfgs[robot_name]["num_surface_points"])
            for robot_name in self.robot_names
        }
        real_masks_by_robot, raw_validation_samples = _load_real_mask_artifacts(
            self.real_masks_path,
            num_points_by_robot,
        )

        self.robot_specs: Dict[str, Dict[str, Any]] = {}
        self.raw_validation_samples = raw_validation_samples
        surface_num_points: int | None = None
        max_action_dim = 0
        for robot_name in self.robot_names:
            if robot_name not in robot_cfgs:
                raise KeyError(f"{robot_name} not found in {self.hparams_path}")
            if robot_name not in real_masks_by_robot:
                raise RuntimeError(f"{robot_name}: no real masks found in {self.real_masks_path}")

            rcfg = robot_cfgs[robot_name]
            model = create_robot_model(
                robot_name=str(rcfg["robot_model_name"]),
                device=self.device,
                num_points=int(rcfg["num_surface_points"]),
            )
            template_hash = model.get_surface_template_hash()
            expected_hash = str(rcfg["surface_template_hash"])
            if self.check_template_hash and template_hash != expected_hash:
                raise RuntimeError(
                    f"{robot_name}: template hash mismatch ({template_hash} != {expected_hash})"
                )

            real_masks = real_masks_by_robot[robot_name]
            point_prob = _build_point_prob_from_real_masks(real_masks).to(self.device)
            component_range = rcfg.get("component_range", [1, 4])
            if len(component_range) < 2:
                component_range = [1, 4]
            contact_count_range = rcfg.get("contact_count_range")
            if not isinstance(contact_count_range, (list, tuple)) or len(contact_count_range) < 2:
                contact_count_range = _extract_contact_count_range_from_real_masks(real_masks)
                print(
                    f"[contact_count_range] {robot_name}: missing in {self.hparams_path}, "
                    f"using fallback [{int(contact_count_range[0])}, {int(contact_count_range[1])}]"
                )
            contact_count_lo = max(1, int(contact_count_range[0]))
            contact_count_hi = max(contact_count_lo, int(contact_count_range[1]))

            surface_num_points_i = int(rcfg["num_surface_points"])
            if surface_num_points is None:
                surface_num_points = surface_num_points_i
            elif int(surface_num_points) != int(surface_num_points_i):
                raise RuntimeError("All robots must share the same surface point count for this dataset")

            local_dof = int(model.dof)
            max_action_dim = max(max_action_dim, local_dof)
            max_anchors = max(1, min(int(component_range[1]), surface_num_points_i))
            component_range = (
                max(1, min(int(component_range[0]), max_anchors)),
                max(1, min(int(component_range[1]), max_anchors)),
            )
            component_range = (component_range[0], max(component_range[0], component_range[1]))
            component_count_values, component_count_distribution = _build_component_distribution_from_range(
                component_range,
                values=rcfg.get("component_count_values"),
                probabilities=rcfg.get("component_count_distribution"),
            )
            if (
                rcfg.get("component_count_values") is None
                or rcfg.get("component_count_distribution") is None
            ):
                print(
                    f"[component_count_distribution] {robot_name}: missing in {self.hparams_path}, "
                    f"using uniform distribution over [{int(component_range[0])}, {int(component_range[1])}]"
                )
            self.robot_specs[robot_name] = {
                "robot_name": robot_name,
                "robot_model_name": str(rcfg["robot_model_name"]),
                "model": model,
                "surface_num_points": surface_num_points_i,
                "num_links": int(len(model.mesh_link_names)),
                "real_masks": real_masks,
                "point_prob": point_prob,
                "component_range": component_range,
                "component_count_values": component_count_values.to(device=self.device, dtype=torch.long),
                "component_count_distribution": component_count_distribution.to(
                    device=self.device,
                    dtype=torch.float32,
                ),
                "contact_count_range": (contact_count_lo, contact_count_hi),
                "max_anchors": int(max_anchors),
                "local_dof": local_dof,
            }

        self.surface_num_points = int(surface_num_points or 0)
        self.action_dim = int(max_action_dim)
        self.max_num_links = max((int(spec["num_links"]) for spec in self.robot_specs.values()), default=0)

    def _make_generator(self, sample_seed: int) -> torch.Generator:
        generator = torch.Generator(device=self.device.type)
        generator.manual_seed(int(sample_seed))
        return generator

    def _sample_random_q_batch(
        self,
        model: Any,
        batch_size: int,
        generator: torch.Generator,
        *,
        base_pose_mode: str = "canonical",
    ) -> torch.Tensor:
        q = model.sample_random_q(batch_size=int(batch_size), generator=generator).to(self.device)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        mode = str(base_pose_mode).strip().lower()
        if mode == "canonical":
            return model._zero_base_pose_in_q(q)
        if mode == "sampled_q2":
            q = model._zero_base_pose_in_q(q)
            if len(model.base_translation_indices) > 0:
                u = torch.rand(
                    (int(batch_size), len(model.base_translation_indices)),
                    dtype=torch.float32,
                    device=self.device,
                    generator=generator,
                )
                base_translation = self.q2_base_translation_min.view(1, -1) + u * (
                    self.q2_base_translation_max.view(1, -1) - self.q2_base_translation_min.view(1, -1)
                )
            else:
                base_translation = None
            if len(model.base_rotation_indices) > 0:
                base_rotation = model.sample_uniform_base_rotation_euler_xyz(
                    batch_size=int(batch_size),
                    generator=generator,
                )
            else:
                base_rotation = None
            return model.set_base_pose_in_q(
                q,
                base_translation=base_translation,
                base_rotation_euler=base_rotation,
            )
        if mode == "raw":
            return q
        raise ValueError(f"Unsupported base_pose_mode={base_pose_mode}")
        return q

    def _resolve_component_distribution_for_spec(
        self,
        spec: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        comp_lo, comp_hi = spec["component_range"]
        values = torch.as_tensor(spec["component_count_values"], dtype=torch.long, device=self.device).view(-1)
        probs = torch.as_tensor(spec["component_count_distribution"], dtype=torch.float32, device=self.device).view(-1)
        valid = (values >= int(comp_lo)) & (values <= int(comp_hi)) & (probs > 0)
        if bool(valid.any()):
            values = values[valid]
            probs = probs[valid]
            probs = probs / probs.sum().clamp_min(1e-8)
            return values, probs
        support = torch.arange(int(comp_lo), int(comp_hi) + 1, dtype=torch.long, device=self.device)
        uniform = torch.ones((int(support.numel()),), dtype=torch.float32, device=self.device)
        uniform = uniform / uniform.sum().clamp_min(1e-8)
        return support, uniform

    def _sample_component_count_batch(
        self,
        *,
        spec: Dict[str, Any],
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        comp_lo, comp_hi = spec["component_range"]
        max_anchors = int(spec["max_anchors"])
        comp_lo = max(1, min(int(comp_lo), max_anchors))
        comp_hi = max(comp_lo, min(int(comp_hi), max_anchors))
        if self.component_sampling_mode == "uniform":
            return torch.randint(
                comp_lo,
                comp_hi + 1,
                (int(batch_size),),
                device=self.device,
                generator=generator,
            )
        if self.component_sampling_mode == "distribution":
            values, probs = self._resolve_component_distribution_for_spec(spec)
            sample_idx = torch.multinomial(
                probs.unsqueeze(0).expand(int(batch_size), -1),
                num_samples=1,
                replacement=True,
                generator=generator,
            ).squeeze(1)
            return values[sample_idx]
        raise ValueError(
            f"Invalid component_sampling_mode={self.component_sampling_mode}. "
            "Expected 'distribution' or 'uniform'."
        )

    def _sample_synthetic_patch_params(
        self,
        *,
        spec: Dict[str, Any],
        batch_size: int,
        generator: torch.Generator,
    ) -> Dict[str, torch.Tensor]:
        max_anchors = int(spec["max_anchors"])
        active_anchor_count = self._sample_component_count_batch(
            spec=spec,
            batch_size=int(batch_size),
            generator=generator,
        ).clamp(1, max_anchors)
        anchor_valid_mask = (
            torch.arange(max_anchors, device=self.device).unsqueeze(0)
            < active_anchor_count.unsqueeze(1)
        )
        anchor_indices = _sample_anchor_indices_batch(
            batch_size=int(batch_size),
            max_anchors=max_anchors,
            num_points=self.surface_num_points,
            point_prob=spec["point_prob"],
            anchor_sampling_mode=self.anchor_sampling_mode,
            generator=generator,
            temperature=self.anchor_temperature,
            device=self.device,
        )

        points_per_anchor = torch.randint(
            self.patch_points_per_anchor_min,
            self.patch_points_per_anchor_max + 1,
            (int(batch_size),),
            device=self.device,
            generator=generator,
        )
        point_slot_valid = (
            torch.arange(self.patch_points_per_anchor_max, device=self.device).unsqueeze(0)
            < points_per_anchor.unsqueeze(1)
        ) # (B, max_points_per_anchor)
        anchor_point_valid_mask = anchor_valid_mask.unsqueeze(-1) & point_slot_valid.unsqueeze(1) # (B, max_anchors, max_points_per_anchor)
        return {
            "active_anchor_count": active_anchor_count,
            "anchor_indices": anchor_indices,
            "anchor_valid_mask": anchor_valid_mask,
            "anchor_point_valid_mask": anchor_point_valid_mask,
        }

    def _build_surface_mask_batch(
        self,
        contact_values: torch.Tensor,
        contact_count_range: Tuple[int, int] | torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, int]]:
        if contact_values.ndim != 2:
            raise ValueError(f"contact_values must have shape (B, N), got {tuple(contact_values.shape)}")
        batch_size, num_points = contact_values.shape
        base_mask = contact_values >= self.threshold
        sorted_idx = torch.argsort(contact_values, dim=1, descending=True)
        count_range_t = torch.as_tensor(contact_count_range, dtype=torch.long, device=self.device)
        if count_range_t.ndim == 1:
            if int(count_range_t.numel()) != 2:
                raise ValueError(
                    f"contact_count_range must have shape (2,) or (B, 2), got {tuple(count_range_t.shape)}"
                )
            count_range_t = count_range_t.view(1, 2).expand(batch_size, 2)
        elif count_range_t.ndim != 2 or count_range_t.shape != (batch_size, 2):
            raise ValueError(
                f"contact_count_range must have shape (2,) or ({batch_size}, 2), got {tuple(count_range_t.shape)}"
            )
        count_lo = count_range_t[:, 0].clamp(1, int(num_points))
        count_hi = torch.maximum(count_lo, count_range_t[:, 1].clamp(1, int(num_points)))
        cur_k = base_mask.sum(dim=1).long()
        lo_mask = _build_topk_mask_from_sorted(sorted_idx, count_lo, int(num_points))
        hi_mask = _build_topk_mask_from_sorted(sorted_idx, count_hi, int(num_points))

        too_low = cur_k < count_lo
        too_high = cur_k > count_hi
        in_range = (~too_low) & (~too_high)

        out = base_mask.clone()
        if bool(too_low.any()):
            out[too_low] = lo_mask[too_low]
        if bool(too_high.any()):
            out[too_high] = hi_mask[too_high]
        stats = {
            "num_samples": int(batch_size),
            "in_range": int(in_range.sum().item()),
            "too_low": int(too_low.sum().item()),
            "too_high": int(too_high.sum().item()),
        }
        return out.bool(), stats

    def _prepare_contact_cloud_batch(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        surface_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if points.ndim != 3 or normals.ndim != 3:
            raise ValueError(
                f"points and normals must have shape (B, N, 3), got {tuple(points.shape)} and {tuple(normals.shape)}"
            )
        if points.shape != normals.shape:
            raise ValueError(
                f"points and normals must have the same shape, got {tuple(points.shape)} vs {tuple(normals.shape)}"
            )
        if surface_mask.ndim != 2 or surface_mask.shape != points.shape[:2]:
            raise ValueError(
                f"surface_mask must have shape {tuple(points.shape[:2])}, got {tuple(surface_mask.shape)}"
            )

        batch_size = int(points.shape[0])
        lengths = surface_mask.to(device=self.device).bool().sum(dim=1).long()
        contact_cloud = torch.zeros(
            (batch_size, self.max_contact_points, 6),
            dtype=torch.float32,
            device=self.device,
        )
        contact_valid_mask = torch.zeros(
            (batch_size, self.max_contact_points),
            dtype=torch.bool,
            device=self.device,
        )
        contact_point_indices = torch.full(
            (batch_size, self.max_contact_points),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        max_len = int(lengths.max().item()) if int(lengths.numel()) > 0 else 0
        if max_len <= 0:
            return contact_cloud, contact_valid_mask, contact_point_indices

        order = torch.argsort(surface_mask.long(), dim=1, descending=True)
        packed_idx = order[:, :max_len]
        gather_idx = packed_idx.unsqueeze(-1).expand(-1, -1, 3)
        packed_points = torch.gather(points, 1, gather_idx)
        packed_normals = torch.gather(normals, 1, gather_idx)
        packed_cloud = torch.cat([packed_points, packed_normals], dim=-1)
        packed_valid = (
            torch.arange(max_len, device=self.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        packed_cloud = packed_cloud.masked_fill(~packed_valid.unsqueeze(-1), 0.0)

        if max_len > self.max_contact_points:
            contact_cloud, fps_idx = farthest_point_sampling(
                packed_cloud,
                self.max_contact_points,
                lengths=lengths,
            )
            contact_valid_mask = fps_idx >= 0
            safe_idx = fps_idx.clamp_min(0)
            contact_point_indices = torch.gather(packed_idx, 1, safe_idx)
            contact_point_indices = contact_point_indices.masked_fill(~contact_valid_mask, -1)
            contact_cloud = contact_cloud.masked_fill(~contact_valid_mask.unsqueeze(-1), 0.0)
        else:
            contact_cloud[:, :max_len] = packed_cloud
            contact_valid_mask[:, :max_len] = packed_valid
            contact_point_indices[:, :max_len] = packed_idx.masked_fill(~packed_valid, -1)

        contact_cloud = contact_cloud.masked_fill(~contact_valid_mask.unsqueeze(-1), 0.0)
        return contact_cloud, contact_valid_mask, contact_point_indices

    def _pad_action_batch(self, q: torch.Tensor) -> torch.Tensor:
        if q.ndim == 1:
            q = q.unsqueeze(0)
        out = torch.zeros((int(q.shape[0]), self.action_dim), dtype=torch.float32, device=self.device)
        out[:, : int(q.shape[-1])] = q
        return out

    def _build_inactive_link_mask_batch(
        self,
        model: Any,
        surface_mask: torch.Tensor,
    ) -> torch.Tensor:
        if surface_mask.ndim != 2:
            raise ValueError(f"surface_mask must have shape (B, N), got {tuple(surface_mask.shape)}")
        inactive_link_mask = torch.zeros(
            (int(surface_mask.shape[0]), self.max_num_links),
            dtype=torch.bool,
            device=self.device,
        )
        inactive_local = model.get_inactive_link_mask_by_contact_mask(
            surface_mask.to(device=self.device).bool(),
            ignore_base_pose=True,
        )
        inactive_link_mask[:, : int(inactive_local.shape[1])] = inactive_local
        return inactive_link_mask

    def _build_sample_batch(
        self,
        *,
        spec: Dict[str, Any],
        robot_index: int,
        q1: torch.Tensor,
        q2: torch.Tensor,
        surface_mask: torch.Tensor,
        include_metadata: bool,
        include_full_metadata: bool = False,
        q1_points: torch.Tensor | None = None,
        q1_normals: torch.Tensor | None = None,
        q2_points: torch.Tensor | None = None,
        q2_normals: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        model = spec["model"]
        if q1.ndim == 1:
            q1 = q1.unsqueeze(0)
        if q2.ndim == 1:
            q2 = q2.unsqueeze(0)
        if surface_mask.ndim == 1:
            surface_mask = surface_mask.unsqueeze(0)

        if q1_points is None or q1_normals is None:
            q1_points, q1_normals = model.get_surface_points_normals_batch(q=q1)
        if q2_points is None or q2_normals is None:
            q2_points, q2_normals = model.get_surface_points_normals_batch(q=q2)

        p_hat = torch.cat([q1_points, q1_normals], dim=-1)
        contact_cloud, contact_valid_mask, contact_point_indices = self._prepare_contact_cloud_batch(
            points=q2_points,
            normals=q2_normals,
            surface_mask=surface_mask,
        )

        mixed_q = model.mix_q_by_contact_mask(q1, q2, surface_mask).clone()

        out: Dict[str, torch.Tensor] = {
            "p_hat": p_hat,
            "contact_cloud": contact_cloud,
            "contact_valid_mask": contact_valid_mask,
            "action": self._pad_action_batch(mixed_q),
        }
        if include_metadata:
            inactive_link_mask = self._build_inactive_link_mask_batch(model, surface_mask.bool())
            out.update(
                {
                    "q1_padded": self._pad_action_batch(q1),
                    "q2_padded": self._pad_action_batch(q2),
                    "contact_point_indices": contact_point_indices,
                    "inactive_link_mask": inactive_link_mask,
                    "local_dof": torch.full(
                        (int(q1.shape[0]),),
                        int(spec["local_dof"]),
                        dtype=torch.long,
                        device=self.device,
                    ),
                    "robot_index": torch.full(
                        (int(q1.shape[0]),),
                        int(robot_index),
                        dtype=torch.long,
                        device=self.device,
                    ),
                }
            )
        if include_full_metadata:
            action_valid_mask = torch.zeros(
                (int(q1.shape[0]), self.action_dim),
                dtype=torch.bool,
                device=self.device,
            )
            action_valid_mask[:, : int(spec["local_dof"])] = True
            out.update(
                {
                    "action_valid_mask": action_valid_mask,
                    "contact_mask_surface": surface_mask.bool(),
                }
            )
        return out

    def _build_single_sample(
        self,
        *,
        spec: Dict[str, Any],
        robot_index: int,
        q1: torch.Tensor,
        q2: torch.Tensor,
        surface_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor | str]:
        batch = self._build_sample_batch(
            spec=spec,
            robot_index=robot_index,
            q1=q1,
            q2=q2,
            surface_mask=surface_mask,
            include_metadata=True,
            include_full_metadata=True,
        )
        sample: Dict[str, torch.Tensor | str] = {
            key: value[0].detach().cpu()
            for key, value in batch.items()
        }
        sample["robot_name"] = str(spec["robot_name"])
        return sample

    @torch.no_grad()
    def _generate_train_batch(
        self,
        *,
        spec: Dict[str, Any],
        batch_size: int,
        generator: torch.Generator,
        robot_index: int,
        include_metadata: bool = False,
        include_full_metadata: bool = False,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, int]]:
        batch_size = int(max(1, batch_size))
        model = spec["model"]

        q1_batch = self._sample_random_q_batch(
            model=model,
            batch_size=batch_size,
            generator=generator,
            base_pose_mode="canonical",
        )
        q2_batch = self._sample_random_q_batch(
            model=model,
            batch_size=batch_size,
            generator=generator,
            base_pose_mode="sampled_q2",
        )
        q1_points_batch, q1_normals_batch = model.get_surface_points_normals_batch(q=q1_batch)
        q2_points_batch, q2_normals_batch = model.get_surface_points_normals_batch(q=q2_batch)

        patch_params = self._sample_synthetic_patch_params(
            spec=spec,
            batch_size=batch_size,
            generator=generator,
        )
        object_points, object_valid_mask = model._sample_virtual_object_patches_batch(
            hand_points=q2_points_batch,
            hand_normals=q2_normals_batch,
            anchor_indices=patch_params["anchor_indices"],
            anchor_valid_mask=patch_params["anchor_valid_mask"],
            anchor_point_valid_mask=patch_params["anchor_point_valid_mask"],
            generator=generator,
            anchor_shift_min=self.patch_anchor_shift_min,
            anchor_shift_max=self.patch_anchor_shift_max,
            max_plane_extent_min=self.patch_extent_min,
            max_plane_extent_max=self.patch_extent_max,
            patch_shift_power=self.patch_shift_power,
            patch_extent_power=self.patch_extent_power,
            normal_jitter_max_deg=self.patch_normal_jitter_max_deg,
            penetration_clearance=self.patch_penetration_clearance,
        )
        flat_object_points = object_points.view(batch_size, -1, 3)
        flat_object_valid_mask = object_valid_mask.view(batch_size, -1)
        contact_values = model._compute_gendex_contact_value_source_target_batch(
            source_points=q2_points_batch,
            source_normals=q2_normals_batch,
            target_points=flat_object_points,
            target_valid_mask=flat_object_valid_mask,
            align_exp_scale=self.align_exp_scale,
            sigmoid_scale=self.sigmoid_scale,
        )
        surface_mask_batch, count_interval_stats = self._build_surface_mask_batch(
            contact_values,
            spec["contact_count_range"],
        )
        return (
            self._build_sample_batch(
                spec=spec,
                robot_index=robot_index,
                q1=q1_batch,
                q2=q2_batch,
                surface_mask=surface_mask_batch,
                include_metadata=include_metadata,
                include_full_metadata=include_full_metadata,
                q1_points=q1_points_batch,
                q1_normals=q1_normals_batch,
                q2_points=q2_points_batch,
                q2_normals=q2_normals_batch,
            ),
            count_interval_stats,
        )


class SyntheticContactPolicyDataset(Dataset, _BaseContactPolicyDataset):
    def __init__(
        self,
        *,
        samples_per_epoch: int = 10000,
        buffer_refresh_fraction: float = 0.2,
        buffer_build_batch_size_per_robot: int = 32,
        buffer_refresh_batch_size_per_robot: int | None = None,
        store_aux_metadata: bool = False,
        store_full_metadata: bool = False,
        progress_label: str = "synthetic_buffer",
        **kwargs: Any,
    ) -> None:
        Dataset.__init__(self)
        _BaseContactPolicyDataset.__init__(self, **kwargs)

        self.samples_per_epoch = int(samples_per_epoch)
        if self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        self.buffer_refresh_fraction = float(buffer_refresh_fraction)
        if not (0.0 < self.buffer_refresh_fraction <= 1.0):
            raise ValueError(f"buffer_refresh_fraction must be in (0, 1], got {self.buffer_refresh_fraction}")

        cycle_epochs = int(round(1.0 / self.buffer_refresh_fraction))
        if abs(self.buffer_refresh_fraction * cycle_epochs - 1.0) > 1e-6:
            raise ValueError(
                "buffer_refresh_fraction must be an exact reciprocal for deterministic turnover, "
                f"got {self.buffer_refresh_fraction}"
            )
        self.buffer_refresh_cycle_epochs = cycle_epochs
        self.buffer_build_batch_size_per_robot = int(max(1, buffer_build_batch_size_per_robot))
        if buffer_refresh_batch_size_per_robot is None:
            buffer_refresh_batch_size_per_robot = self.buffer_build_batch_size_per_robot
        self.buffer_refresh_batch_size_per_robot = int(max(1, buffer_refresh_batch_size_per_robot))
        self.store_aux_metadata = bool(store_aux_metadata)
        self.store_full_metadata = bool(store_full_metadata)
        self._store_loss_metadata = self.store_aux_metadata or self.store_full_metadata
        self.progress_label = str(progress_label)

        self.robot_slot_ranges: Dict[str, tuple[int, int]] = {}
        self.robot_refresh_blocks: Dict[str, List[torch.Tensor]] = {}
        self._build_slot_schedule()

        self.buffer_p_hat = torch.empty(
            (self.samples_per_epoch, self.surface_num_points, 6),
            dtype=torch.float32,
            device="cpu",
        )
        self.buffer_contact_cloud = torch.empty(
            (self.samples_per_epoch, self.max_contact_points, 6),
            dtype=torch.float32,
            device="cpu",
        )
        self.buffer_contact_valid_mask = torch.empty(
            (self.samples_per_epoch, self.max_contact_points),
            dtype=torch.bool,
            device="cpu",
        )
        self.buffer_action = torch.empty(
            (self.samples_per_epoch, self.action_dim),
            dtype=torch.float32,
            device="cpu",
        )
        self.buffer_q1_padded: torch.Tensor | None = None
        self.buffer_q2_padded: torch.Tensor | None = None
        self.buffer_contact_point_indices: torch.Tensor | None = None
        self.buffer_inactive_link_mask: torch.Tensor | None = None
        self.buffer_local_dof: torch.Tensor | None = None
        self.buffer_robot_index: torch.Tensor | None = None
        self.buffer_action_valid_mask: torch.Tensor | None = None
        self.buffer_contact_mask_surface: torch.Tensor | None = None
        if self._store_loss_metadata:
            self.buffer_q1_padded = torch.empty(
                (self.samples_per_epoch, self.action_dim),
                dtype=torch.float32,
                device="cpu",
            )
            self.buffer_q2_padded = torch.empty(
                (self.samples_per_epoch, self.action_dim),
                dtype=torch.float32,
                device="cpu",
            )
            self.buffer_contact_point_indices = torch.empty(
                (self.samples_per_epoch, self.max_contact_points),
                dtype=torch.int16,
                device="cpu",
            )
            self.buffer_inactive_link_mask = torch.empty(
                (self.samples_per_epoch, self.max_num_links),
                dtype=torch.bool,
                device="cpu",
            )
            self.buffer_local_dof = torch.empty(
                (self.samples_per_epoch,),
                dtype=torch.long,
                device="cpu",
            )
            self.buffer_robot_index = torch.empty(
                (self.samples_per_epoch,),
                dtype=torch.long,
                device="cpu",
            )
        if self.store_full_metadata:
            self.buffer_action_valid_mask = torch.empty(
                (self.samples_per_epoch, self.action_dim),
                dtype=torch.bool,
                device="cpu",
            )
            self.buffer_contact_mask_surface = torch.empty(
                (self.samples_per_epoch, self.surface_num_points),
                dtype=torch.bool,
                device="cpu",
            )

        self._prepared_epoch = 1
        build_t0 = time.perf_counter()
        self._populate_initial_buffer()
        self.buffer_build_time_s = time.perf_counter() - build_t0

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample: Dict[str, torch.Tensor | str] = {
            "p_hat": self.buffer_p_hat[idx],
            "contact_cloud": self.buffer_contact_cloud[idx],
            "contact_valid_mask": self.buffer_contact_valid_mask[idx],
            "action": self.buffer_action[idx],
        }
        if self._store_loss_metadata:
            if (
                self.buffer_q1_padded is None
                or self.buffer_q2_padded is None
                or self.buffer_contact_point_indices is None
                or self.buffer_inactive_link_mask is None
                or self.buffer_local_dof is None
                or self.buffer_robot_index is None
            ):
                raise RuntimeError("Auxiliary metadata buffer storage is not initialized")
            robot_index = int(self.buffer_robot_index[idx].item())
            sample.update(
                {
                    "q1_padded": self.buffer_q1_padded[idx],
                    "q2_padded": self.buffer_q2_padded[idx],
                    "contact_point_indices": self.buffer_contact_point_indices[idx].to(dtype=torch.long),
                    "inactive_link_mask": self.buffer_inactive_link_mask[idx],
                    "local_dof": self.buffer_local_dof[idx],
                    "robot_index": self.buffer_robot_index[idx],
                }
            )
        if self.store_full_metadata:
            if (
                self.buffer_action_valid_mask is None
                or self.buffer_contact_mask_surface is None
            ):
                raise RuntimeError("Full-metadata buffer storage is not initialized")
            sample.update(
                {
                    "action_valid_mask": self.buffer_action_valid_mask[idx],
                    "contact_mask_surface": self.buffer_contact_mask_surface[idx],
                    "robot_name": self.global_robot_names[robot_index],
                }
            )
        return sample

    def _build_slot_schedule(self) -> None:
        num_robots = len(self.robot_names)
        base = self.samples_per_epoch // max(1, num_robots)
        remainder = self.samples_per_epoch % max(1, num_robots)

        start = 0
        for i, robot_name in enumerate(self.robot_names):
            count = base + (1 if i < remainder else 0)
            end = start + count
            self.robot_slot_ranges[robot_name] = (start, end)

            if count <= 0:
                self.robot_refresh_blocks[robot_name] = [
                    torch.zeros((0,), dtype=torch.long)
                    for _ in range(self.buffer_refresh_cycle_epochs)
                ]
                start = end
                continue

            perm_gen = torch.Generator(device="cpu")
            perm_gen.manual_seed(self.seed + _stable_int_seed(f"slot_schedule|{robot_name}"))
            perm = torch.randperm(count, generator=perm_gen, dtype=torch.long) + start
            blocks = [chunk.clone() for chunk in torch.chunk(perm, self.buffer_refresh_cycle_epochs)]
            while len(blocks) < self.buffer_refresh_cycle_epochs:
                blocks.append(torch.zeros((0,), dtype=torch.long))
            self.robot_refresh_blocks[robot_name] = blocks
            start = end

    def _write_train_batch_to_buffer(
        self,
        *,
        slot_indices: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> None:
        if int(slot_indices.numel()) == 0:
            return
        idx = slot_indices.to(device="cpu", dtype=torch.long)
        self.buffer_p_hat[idx] = batch["p_hat"].detach().to(device="cpu", dtype=torch.float32)
        self.buffer_contact_cloud[idx] = batch["contact_cloud"].detach().to(device="cpu", dtype=torch.float32)
        self.buffer_contact_valid_mask[idx] = batch["contact_valid_mask"].detach().to(device="cpu", dtype=torch.bool)
        self.buffer_action[idx] = batch["action"].detach().to(device="cpu", dtype=torch.float32)
        if self._store_loss_metadata:
            if (
                self.buffer_q1_padded is None
                or self.buffer_q2_padded is None
                or self.buffer_contact_point_indices is None
                or self.buffer_inactive_link_mask is None
                or self.buffer_local_dof is None
                or self.buffer_robot_index is None
            ):
                raise RuntimeError("Auxiliary metadata buffer storage is not initialized")
            self.buffer_q1_padded[idx] = batch["q1_padded"].detach().to(device="cpu", dtype=torch.float32)
            self.buffer_q2_padded[idx] = batch["q2_padded"].detach().to(device="cpu", dtype=torch.float32)
            self.buffer_contact_point_indices[idx] = batch["contact_point_indices"].detach().to(device="cpu", dtype=torch.int16)
            self.buffer_inactive_link_mask[idx] = batch["inactive_link_mask"].detach().to(device="cpu", dtype=torch.bool)
            self.buffer_local_dof[idx] = batch["local_dof"].detach().to(device="cpu", dtype=torch.long)
            self.buffer_robot_index[idx] = batch["robot_index"].detach().to(device="cpu", dtype=torch.long)
        if self.store_full_metadata:
            if (
                self.buffer_action_valid_mask is None
                or self.buffer_contact_mask_surface is None
            ):
                raise RuntimeError("Full-metadata buffer storage is not initialized")
            self.buffer_action_valid_mask[idx] = batch["action_valid_mask"].detach().to(device="cpu", dtype=torch.bool)
            self.buffer_contact_mask_surface[idx] = batch["contact_mask_surface"].detach().to(device="cpu", dtype=torch.bool)

    def _fill_slots_for_robot(
        self,
        *,
        robot_name: str,
        robot_index: int,
        slot_indices: torch.Tensor,
        batch_size: int,
        generator: torch.Generator,
        progress: tqdm | None = None,
    ) -> Dict[str, int]:
        if int(slot_indices.numel()) == 0:
            return _init_contact_count_interval_stats()

        spec = self.robot_specs[robot_name]
        offset = 0
        total = int(slot_indices.numel())
        count_interval_stats = _init_contact_count_interval_stats()
        while offset < total:
            cur = min(int(batch_size), total - offset)
            batch, batch_stats = self._generate_train_batch(
                spec=spec,
                batch_size=cur,
                generator=generator,
                robot_index=robot_index,
                include_metadata=self._store_loss_metadata,
                include_full_metadata=self.store_full_metadata,
            )
            _merge_contact_count_interval_stats(count_interval_stats, batch_stats)
            self._write_train_batch_to_buffer(
                slot_indices=slot_indices[offset : offset + cur],
                batch=batch,
            )
            offset += cur
            if progress is not None:
                progress.update(cur)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return count_interval_stats

    def _populate_initial_buffer(self) -> None:
        with torch.no_grad():
            progress = tqdm(total=self.samples_per_epoch, desc=f"{self.progress_label}:build", leave=False)
            for robot_name in self.robot_names:
                robot_index = self.robot_name_to_global_index[robot_name]
                start, end = self.robot_slot_ranges[robot_name]
                slot_indices = torch.arange(start, end, dtype=torch.long)
                generator = self._make_generator(
                    self.seed + _stable_int_seed(f"train_buffer|init|{robot_name}")
                )
                count_interval_stats = self._fill_slots_for_robot(
                    robot_name=robot_name,
                    robot_index=robot_index,
                    slot_indices=slot_indices,
                    batch_size=self.buffer_build_batch_size_per_robot,
                    generator=generator,
                    progress=progress,
                )
                print(
                    _format_contact_count_interval_summary(
                        f"{self.progress_label}:build:{robot_name}",
                        count_interval_stats,
                    )
                )
            progress.close()

    def _refresh_epoch_once(self, epoch: int) -> float:
        refresh_block_idx = (int(epoch) - 2) % self.buffer_refresh_cycle_epochs
        t0 = time.perf_counter()
        with torch.no_grad():
            total_refresh_slots = sum(
                int(self.robot_refresh_blocks[robot_name][refresh_block_idx].numel())
                for robot_name in self.robot_names
            )
            progress = tqdm(
                total=total_refresh_slots,
                desc=f"{self.progress_label}:refresh:{int(epoch)}",
                leave=False,
            )
            for robot_name in self.robot_names:
                robot_index = self.robot_name_to_global_index[robot_name]
                slot_indices = self.robot_refresh_blocks[robot_name][refresh_block_idx]
                if int(slot_indices.numel()) == 0:
                    continue
                generator = self._make_generator(
                    self.seed + _stable_int_seed(f"train_buffer|refresh|epoch={int(epoch)}|{robot_name}")
                )
                count_interval_stats = self._fill_slots_for_robot(
                    robot_name=robot_name,
                    robot_index=robot_index,
                    slot_indices=slot_indices,
                    batch_size=self.buffer_refresh_batch_size_per_robot,
                    generator=generator,
                    progress=progress,
                )
                print(
                    _format_contact_count_interval_summary(
                        f"{self.progress_label}:refresh:{int(epoch)}:{robot_name}",
                        count_interval_stats,
                    )
                )
            progress.close()
        return time.perf_counter() - t0

    def prepare_epoch(self, epoch: int) -> float:
        epoch = int(epoch)
        if epoch <= 1:
            self._prepared_epoch = max(self._prepared_epoch, 1)
            return 0.0
        if epoch <= self._prepared_epoch:
            return 0.0

        total_refresh_s = 0.0
        for refresh_epoch in range(self._prepared_epoch + 1, epoch + 1):
            total_refresh_s += self._refresh_epoch_once(refresh_epoch)
        self._prepared_epoch = epoch
        return total_refresh_s


OnTheFlySyntheticContactPolicyDataset = SyntheticContactPolicyDataset


class RealContactPolicyValDataset(Dataset, _BaseContactPolicyDataset):
    def __init__(
        self,
        *,
        cache_processed_samples: bool = True,
        cache_build_batch_size: int = 256,
        progress_label: str = "validate_real",
        **kwargs: Any,
    ) -> None:
        Dataset.__init__(self)
        self.cache_processed_samples = bool(cache_processed_samples)
        self.cache_build_batch_size = int(max(1, cache_build_batch_size))
        self.progress_label = str(progress_label)
        self.cached_samples: List[Dict[str, torch.Tensor | str]] | None = None
        total_init_steps = 2 + int(self.cache_processed_samples)
        progress = tqdm(
            total=total_init_steps,
            desc=f"{self.progress_label}:base",
            leave=True,
            dynamic_ncols=True,
        )
        try:
            _BaseContactPolicyDataset.__init__(self, **kwargs)
            progress.update(1)
            progress.set_description(f"{self.progress_label}:filter")
            self.samples = self._filter_validation_samples()
            progress.update(1)
            if self.cache_processed_samples:
                progress.set_description(f"{self.progress_label}:cache")
                self.cached_samples = self._build_cached_samples()
                progress.update(1)
        finally:
            progress.close()

    def __len__(self) -> int:
        return len(self.samples)

    def _filter_validation_samples(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for raw_sample in self.raw_validation_samples:
            robot_name = str(raw_sample["robot_name"])
            if robot_name not in self.robot_specs:
                continue
            mask = torch.as_tensor(raw_sample["mask"]).bool().view(-1)
            if int(mask.numel()) != self.surface_num_points:
                continue
            q2 = torch.as_tensor(raw_sample["q2"], dtype=torch.float32).view(-1)
            if int(q2.numel()) != int(self.robot_specs[robot_name]["local_dof"]):
                continue
            samples.append(
                {
                    "sample_id": int(raw_sample["sample_id"]),
                    "robot_name": robot_name,
                    "q2": q2,
                    "mask": mask,
                }
            )
        if len(samples) == 0:
            raise RuntimeError(f"No validation samples found in {self.real_masks_path}")
        return samples

    def _materialize_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor | str]:
        robot_name = str(sample["robot_name"])
        spec = self.robot_specs[robot_name]
        robot_index = self.robot_name_to_global_index[robot_name]
        sample_seed = self.seed + _stable_int_seed(f"{robot_name}|{int(sample['sample_id'])}|q1")
        generator = self._make_generator(sample_seed)
        model = spec["model"]
        q1 = self._sample_random_q_batch(
            model=model,
            batch_size=1,
            generator=generator,
            base_pose_mode="canonical",
        )[0]
        q2 = torch.as_tensor(sample["q2"], dtype=torch.float32, device=self.device)
        surface_mask = torch.as_tensor(sample["mask"], dtype=torch.bool, device=self.device)
        return self._build_single_sample(
            spec=spec,
            robot_index=robot_index,
            q1=q1,
            q2=q2,
            surface_mask=surface_mask,
        )

    def _build_cached_samples(self) -> List[Dict[str, torch.Tensor | str]]:
        cached: List[Dict[str, torch.Tensor | str] | None] = [None] * len(self.samples)
        sample_indices_by_robot: Dict[str, List[int]] = {name: [] for name in self.robot_names}
        for sample_idx, sample in enumerate(self.samples):
            sample_indices_by_robot[str(sample["robot_name"])].append(sample_idx)

        progress = tqdm(
            total=len(self.samples),
            desc=f"{self.progress_label}:cache",
            leave=True,
            dynamic_ncols=True,
        )
        try:
            for robot_name in self.robot_names:
                sample_indices = sample_indices_by_robot.get(robot_name, [])
                if len(sample_indices) == 0:
                    continue
                robot_index = self.robot_name_to_global_index[robot_name]
                spec = self.robot_specs[robot_name]
                model = spec["model"]
                start = 0
                while start < len(sample_indices):
                    chunk_indices = sample_indices[start : start + self.cache_build_batch_size]
                    q1_batch_list: List[torch.Tensor] = []
                    q2_batch_list: List[torch.Tensor] = []
                    mask_batch_list: List[torch.Tensor] = []
                    for sample_idx in chunk_indices:
                        sample = self.samples[sample_idx]
                        sample_seed = self.seed + _stable_int_seed(
                            f"{robot_name}|{int(sample['sample_id'])}|q1"
                        )
                        generator = self._make_generator(sample_seed)
                        q1_batch_list.append(
                            self._sample_random_q_batch(
                                model=model,
                                batch_size=1,
                                generator=generator,
                                base_pose_mode="canonical",
                            )[0]
                        )
                        q2_batch_list.append(
                            torch.as_tensor(sample["q2"], dtype=torch.float32, device=self.device)
                        )
                        mask_batch_list.append(
                            torch.as_tensor(sample["mask"], dtype=torch.bool, device=self.device)
                        )

                    batch = self._build_sample_batch(
                        spec=spec,
                        robot_index=robot_index,
                        q1=torch.stack(q1_batch_list, dim=0),
                        q2=torch.stack(q2_batch_list, dim=0),
                        surface_mask=torch.stack(mask_batch_list, dim=0),
                        include_metadata=True,
                        include_full_metadata=True,
                    )
                    for row_idx, sample_idx in enumerate(chunk_indices):
                        sample_out: Dict[str, torch.Tensor | str] = {
                            key: value[row_idx].detach().cpu()
                            for key, value in batch.items()
                        }
                        sample_out["robot_name"] = robot_name
                        cached[sample_idx] = sample_out
                    start += len(chunk_indices)
                    progress.update(len(chunk_indices))
        finally:
            progress.close()

        if any(sample is None for sample in cached):
            raise RuntimeError("Failed to materialize all cached validation samples")
        return [sample for sample in cached if sample is not None]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        if self.cached_samples is not None:
            return self.cached_samples[idx]
        return self._materialize_sample(self.samples[idx])
