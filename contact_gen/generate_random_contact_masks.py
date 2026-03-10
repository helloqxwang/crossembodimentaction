from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime
from typing import Any, Dict

import hydra
import torch
import yaml
from omegaconf import DictConfig
from tqdm import tqdm


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.append(PROJECT_ROOT)

from data_process.contact_policy_dataset import (  # noqa: E402
    _BaseContactPolicyDataset,
    _format_contact_count_interval_summary,
    _init_contact_count_interval_stats,
    _merge_contact_count_interval_stats,
)


def _abs_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _stable_seed_from_name(name: str) -> int:
    h = hashlib.sha1(name.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="little", signed=False) % (2**31 - 1)


def _compute_sample_count_map(robot_names: list[str], sample_count: int, mode: str) -> dict[str, int]:
    n = len(robot_names)
    if n <= 0:
        return {}
    if mode == "total":
        base = int(sample_count) // n
        rem = int(sample_count) % n
        return {name: int(base + (1 if i < rem else 0)) for i, name in enumerate(robot_names)}
    if mode == "per_robot":
        return {name: int(sample_count) for name in robot_names}
    raise ValueError(f"Invalid sample_count_mode: {mode}. Expected 'per_robot' or 'total'.")


class _RandomMaskSampler(_BaseContactPolicyDataset):
    def resolve_component_range(
        self,
        robot_name: str,
        component_min: int,
        component_max: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        spec = self.robot_specs[robot_name]
        comp_lo_fit, comp_hi_fit = spec["component_range"]
        max_anchors = int(spec["max_anchors"])
        comp_lo = comp_lo_fit if int(component_min) <= 0 else int(component_min)
        comp_hi = comp_hi_fit if int(component_max) <= 0 else int(component_max)
        comp_lo = max(1, min(comp_lo, max_anchors))
        comp_hi = max(comp_lo, min(comp_hi, max_anchors))
        return (int(comp_lo_fit), int(comp_hi_fit)), (int(comp_lo), int(comp_hi))

    def _select_visual_patch_points(
        self,
        object_points: torch.Tensor,
        object_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(object_points.shape[0])
        flat_points = object_points.reshape(batch_size, -1, 3)
        flat_valid_mask = object_valid_mask.reshape(batch_size, -1)
        vis_points = flat_points.masked_fill(~flat_valid_mask.unsqueeze(-1), 0.0)
        vis_mask = flat_valid_mask
        vis_points = vis_points.masked_fill(~vis_mask.unsqueeze(-1), 0.0)
        return vis_points, vis_mask

    @torch.no_grad()
    def generate_batch(
        self,
        *,
        robot_name: str,
        batch_size: int,
        generator: torch.Generator,
        component_min: int,
        component_max: int,
    ) -> Dict[str, Any]:
        base_spec = self.robot_specs[robot_name]
        _, component_range = self.resolve_component_range(
            robot_name=robot_name,
            component_min=component_min,
            component_max=component_max,
        )
        spec = dict(base_spec)
        spec["component_range"] = component_range
        model = spec["model"]

        q_batch = self._sample_random_q_batch(
            model=model,
            batch_size=int(batch_size),
            generator=generator,
            base_pose_mode="sampled_q2",
        )
        hand_points, hand_normals = model.get_surface_points_normals_batch(q=q_batch)
        patch_params = self._sample_synthetic_patch_params(
            spec=spec,
            batch_size=int(batch_size),
            generator=generator,
        )
        object_points, object_valid_mask = model._sample_virtual_object_patches_batch(
            hand_points=hand_points,
            hand_normals=hand_normals,
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
        flat_object_points = object_points.reshape(int(batch_size), -1, 3)
        flat_object_valid_mask = object_valid_mask.reshape(int(batch_size), -1)
        contact_values = model._compute_gendex_contact_value_source_target_batch(
            source_points=hand_points,
            source_normals=hand_normals,
            target_points=flat_object_points,
            target_valid_mask=flat_object_valid_mask,
            align_exp_scale=self.align_exp_scale,
            sigmoid_scale=self.sigmoid_scale,
        )
        masks, count_interval_stats = self._build_surface_mask_batch(
            contact_values=contact_values,
            contact_count_range=spec["contact_count_range"],
        )
        vis_points, vis_mask = self._select_visual_patch_points(
            object_points=object_points,
            object_valid_mask=object_valid_mask,
        )
        anchor_indices = patch_params["anchor_indices"].masked_fill(~patch_params["anchor_valid_mask"], -1)
        patch_types = torch.full_like(anchor_indices, -1)
        patch_types[patch_params["anchor_valid_mask"]] = 0
        return {
            "masks": masks,
            "q": q_batch,
            "hand_points": hand_points,
            "contact_values": contact_values,
            "object_patch_points": vis_points,
            "object_patch_mask": vis_mask,
            "anchor_indices": anchor_indices,
            "patch_types": patch_types,
            "mask_count_interval_stats": count_interval_stats,
        }


@hydra.main(config_path=".", config_name="config_generate_random_contact_masks", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))

    hparams_path = _abs_path(str(cfg.hparams_path))
    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"hparams yaml not found: {hparams_path}")
    with open(hparams_path, "r", encoding="utf-8") as f:
        hparams = yaml.safe_load(f)
    if not isinstance(hparams, dict) or "robots" not in hparams:
        raise ValueError(f"Invalid hparams yaml: {hparams_path}")
    hparams_robots = hparams["robots"]
    if not isinstance(hparams_robots, dict) or len(hparams_robots) == 0:
        raise RuntimeError(f"No robot hyperparameters found in: {hparams_path}")

    robot_names_cfg = list(cfg.robot_names) if cfg.robot_names is not None else []
    robot_names = robot_names_cfg if len(robot_names_cfg) > 0 else sorted(hparams_robots.keys())
    sample_count_mode = str(getattr(cfg, "sample_count_mode", "per_robot"))
    sample_count_map = _compute_sample_count_map(
        robot_names=robot_names,
        sample_count=int(cfg.sample_count),
        mode=sample_count_mode,
    )

    total_jobs = int(sum(sample_count_map.values()))
    print(
        f"[sampling] mode={sample_count_mode} requested={int(cfg.sample_count)} "
        f"robots={len(robot_names)} total_masks_to_generate={total_jobs}"
    )
    for robot_name in robot_names:
        print(f"[sampling] {robot_name}: {sample_count_map[robot_name]} masks")

    real_masks_path = _abs_path(str(cfg.real_masks_path))
    output_dir = _abs_path(str(cfg.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    sampler = _RandomMaskSampler(
        robot_names=robot_names,
        hparams_path=hparams_path,
        real_masks_path=real_masks_path,
        max_contact_points=1,
        seed=int(cfg.seed),
        device=str(cfg.device),
        check_template_hash=bool(cfg.check_template_hash),
        threshold=float(cfg.threshold),
        align_exp_scale=float(cfg.align_exp_scale),
        sigmoid_scale=float(cfg.sigmoid_scale),
        component_sampling_mode=str(cfg.component_sampling_mode),
        anchor_sampling_mode=str(cfg.anchor_sampling_mode),
        anchor_temperature=float(cfg.anchor_temperature),
        patch_anchor_shift_min=float(cfg.patch_anchor_shift_min),
        patch_anchor_shift_max=float(cfg.patch_anchor_shift_max),
        patch_extent_min=float(cfg.patch_extent_min),
        patch_extent_max=float(cfg.patch_extent_max),
        patch_extent_power=float(cfg.patch_extent_power),
        patch_shift_power=float(cfg.patch_shift_power),
        patch_normal_jitter_max_deg=float(cfg.patch_normal_jitter_max_deg),
        patch_points_per_anchor_min=int(cfg.patch_points_per_anchor_min),
        patch_points_per_anchor_max=int(cfg.patch_points_per_anchor_max),
        patch_penetration_clearance=float(cfg.patch_penetration_clearance),
        q2_base_translation_min=[float(x) for x in cfg.q2_base_translation_min],
        q2_base_translation_max=[float(x) for x in cfg.q2_base_translation_max],
        q2_base_rotation_mode=str(cfg.q2_base_rotation_mode),
    )

    for robot_name in robot_names:
        if robot_name not in sampler.robot_specs:
            print(f"[skip] {robot_name}: not found in {hparams_path}")
            continue

        spec = sampler.robot_specs[robot_name]
        resolved_robot_name = str(spec["robot_model_name"])
        num_surface_points = int(spec["surface_num_points"])
        model = spec["model"]
        model_template_hash = model.get_surface_template_hash()
        fit_component_range, used_component_range = sampler.resolve_component_range(
            robot_name=robot_name,
            component_min=int(cfg.component_min),
            component_max=int(cfg.component_max),
        )
        anchor_sampling_mode = str(cfg.anchor_sampling_mode).strip().lower()
        component_sampling_mode = str(cfg.component_sampling_mode).strip().lower()
        print(
            f"[{robot_name}] model={resolved_robot_name} num_surface_points={num_surface_points} "
            f"component_range_fit={list(fit_component_range)} component_range_used={list(used_component_range)} "
            f"contact_count_range={list(spec['contact_count_range'])} "
            f"real_masks={int(spec['real_masks'].shape[0])} "
            f"component_sampling_mode={component_sampling_mode} "
            f"anchor_sampling_mode={anchor_sampling_mode}"
        )

        generator = torch.Generator(device=model.device.type)
        generator.manual_seed(int(cfg.seed) + _stable_seed_from_name(robot_name))

        sampled_masks_list: list[torch.Tensor] = []
        sampled_q_list: list[torch.Tensor] = []
        vis_samples: list[dict[str, torch.Tensor | int | str]] = []
        count_interval_stats = _init_contact_count_interval_stats()
        remain = int(sample_count_map[robot_name])
        chunk = int(max(1, int(cfg.sample_chunk)))
        vis_budget = int(max(0, int(cfg.save_visualization_samples)))
        pbar = tqdm(total=remain, desc=f"sample_masks:{robot_name}")

        while remain > 0:
            b = min(chunk, remain)
            batch = sampler.generate_batch(
                robot_name=robot_name,
                batch_size=b,
                generator=generator,
                component_min=int(cfg.component_min),
                component_max=int(cfg.component_max),
            )
            masks_cpu = batch["masks"].detach().cpu().bool()
            q_cpu = batch["q"].detach().cpu().float()
            sampled_masks_list.append(masks_cpu)
            sampled_q_list.append(q_cpu)
            _merge_contact_count_interval_stats(
                count_interval_stats,
                batch["mask_count_interval_stats"],
            )

            if vis_budget > 0:
                take = min(vis_budget, b)
                hand_points_cpu = batch["hand_points"].detach().cpu().float()
                object_points_cpu = batch["object_patch_points"].detach().cpu().float()
                object_mask_cpu = batch["object_patch_mask"].detach().cpu().bool()
                contact_values_cpu = batch["contact_values"].detach().cpu().float()
                anchor_indices_cpu = batch["anchor_indices"].detach().cpu().long()
                patch_types_cpu = batch["patch_types"].detach().cpu().long()
                for i in range(take):
                    mask_i = masks_cpu[i]
                    hand_points_i = hand_points_cpu[i]
                    vis_samples.append(
                        {
                            "sample_id": len(vis_samples),
                            "robot_name": robot_name,
                            "robot_model_name": resolved_robot_name,
                            "q": q_cpu[i],
                            "hand_points": hand_points_i,
                            "hand_contact_mask": mask_i,
                            "hand_contact_points": hand_points_i[mask_i],
                            "object_patch_points": object_points_cpu[i][object_mask_cpu[i]],
                            "contact_values": contact_values_cpu[i],
                            "anchor_indices": anchor_indices_cpu[i],
                            "patch_types": patch_types_cpu[i],
                        }
                    )
                vis_budget -= take

            remain -= b
            pbar.update(b)
        pbar.close()

        sampled_masks_t = torch.cat(sampled_masks_list, dim=0).bool()
        sampled_q_t = torch.cat(sampled_q_list, dim=0).float()
        sampled_counts = sampled_masks_t.sum(dim=1).long()

        payload = {
            "meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "robot_name": robot_name,
                "robot_model_name": resolved_robot_name,
                "num_surface_points": int(num_surface_points),
                "sample_count_requested": int(cfg.sample_count),
                "sample_count_mode": sample_count_mode,
                "sample_count_effective": int(sample_count_map[robot_name]),
                "component_range_fit": [int(fit_component_range[0]), int(fit_component_range[1])],
                "component_range_used": [int(used_component_range[0]), int(used_component_range[1])],
                "surface_template_hash": model_template_hash,
                "hparams_path": hparams_path,
                "real_masks_path": real_masks_path,
                "threshold": float(cfg.threshold),
                "align_exp_scale": float(cfg.align_exp_scale),
                "sigmoid_scale": float(cfg.sigmoid_scale),
                "component_sampling_mode": component_sampling_mode,
                "anchor_temperature": float(cfg.anchor_temperature),
                "anchor_sampling_mode": anchor_sampling_mode,
                "contact_count_range": [int(spec["contact_count_range"][0]), int(spec["contact_count_range"][1])],
                "patch_normal_jitter_max_deg": float(cfg.patch_normal_jitter_max_deg),
                "sampling_mode": "real_stats_guided_patch_distance",
                "sampling_impl": "contact_policy_dataset_batched",
            },
            "summary": {
                "sampled_num_samples": int(sampled_masks_t.shape[0]),
                "sampled_count_mean": float(sampled_counts.float().mean().item()),
                "sampled_count_std": float(sampled_counts.float().std(unbiased=False).item()),
                "out_of_interval_ratio": float(
                    (int(count_interval_stats["too_low"]) + int(count_interval_stats["too_high"]))
                    / max(1, int(count_interval_stats["num_samples"]))
                ),
            },
            "sampled_masks": sampled_masks_t,
            "sampled_q": sampled_q_t,
            "visualization_samples": vis_samples,
        }
        out_path = os.path.join(output_dir, f"{robot_name}_random_masks.pt")
        torch.save(payload, out_path)
        print(_format_contact_count_interval_summary(f"sample_masks:{robot_name}", count_interval_stats))
        print(f"[saved] {out_path}")
        print(f"[summary] {robot_name}: {payload['summary']}")


if __name__ == "__main__":
    main()
