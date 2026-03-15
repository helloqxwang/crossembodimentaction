from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data_process.contact_policy_dataset import (
    _BaseContactPolicyDataset,
    _build_component_distribution_from_range,
    _format_contact_count_interval_summary,
    _init_contact_count_interval_stats,
    _merge_contact_count_interval_stats,
    _sample_anchor_indices_batch,
)


# Fixed-buffer ShadowHand probe dataset used to isolate task difficulty from
# the original cross-embodiment contact-policy setting.
class ShadowhandContactPolicyProbeDataset(Dataset, _BaseContactPolicyDataset):
    _TASK_FIXED_TIPS = {"tips_fixed_base", "tips_movable_base"}
    _TASK_PATCHES = {"patches_fixed_base", "patches_movable_base"}
    _SUPPORTED_TASKS = _TASK_FIXED_TIPS | _TASK_PATCHES

    @classmethod
    def _parse_task_modes(cls, task: str) -> tuple[str, str, str]:
        task_key = str(task).strip().lower()
        if task_key not in cls._SUPPORTED_TASKS:
            raise ValueError(f"Unsupported probe task={task}. Expected one of {sorted(cls._SUPPORTED_TASKS)}")
        if task_key.startswith("tips_"):
            contact_mode = "tips"
        elif task_key.startswith("patches_"):
            contact_mode = "patches"
        else:
            raise ValueError(f"Unsupported probe task prefix for {task}")
        if task_key.endswith("fixed_base"):
            base_mode = "fixed"
        elif task_key.endswith("movable_base"):
            base_mode = "movable"
        else:
            raise ValueError(f"Unsupported probe task suffix for {task}")
        return task_key, contact_mode, base_mode

    def __init__(
        self,
        *,
        task: str,
        robot_name: str = "shadowhand",
        samples: int,
        build_batch_size: int = 1024,
        progress_label: str = "probe_dataset",
        max_contact_points: int = 256,
        fixed_tip_links: Iterable[str] = ("ffdistal", "thdistal"),
        supervised_joint_prefixes: Iterable[str] = ("FFJ", "THJ"),
        allowed_contact_link_names: Iterable[str] | None = None,
        probe_component_range: Tuple[int, int] = (1, 4),
        probe_component_count_values: Any | None = None,
        probe_component_count_distribution: Any | None = None,
        probe_contact_count_range: Tuple[int, int] = (8, 64),
        load_buffer_path: str | Path | None = None,
        load_buffer_payload: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        Dataset.__init__(self)
        # Probe-task identity determines what observation is built and whether the base pose moves.
        self.task, self.contact_mode, self.base_mode = self._parse_task_modes(task)
        self.robot_name = str(robot_name)
        self.samples = int(samples)
        if self.samples <= 0:
            raise ValueError("samples must be positive")
        self.build_batch_size = int(max(1, build_batch_size))
        self.progress_label = str(progress_label)
        self.fixed_tip_links = tuple(str(x) for x in fixed_tip_links)
        if self.contact_mode == "tips" and len(self.fixed_tip_links) != 2:
            raise ValueError(
                f"Fixed-tip probe tasks require exactly 2 fixed_tip_links, got {self.fixed_tip_links}"
            )
        self.supervised_joint_prefixes = tuple(str(x).upper() for x in supervised_joint_prefixes)
        if allowed_contact_link_names is None:
            allowed_contact_link_names = (
                "palm",
                "ffknuckle",
                "ffproximal",
                "ffmiddle",
                "ffdistal",
                "thbase",
                "thproximal",
                "thmiddle",
                "thdistal",
            )
        self.allowed_contact_link_names = tuple(str(x) for x in allowed_contact_link_names)
        self.supervise_base_pose = self.base_mode == "movable"
        effective_max_contact_points = 2 if self.contact_mode == "tips" else int(max_contact_points)

        if load_buffer_path is not None and load_buffer_payload is not None:
            raise ValueError("Only one of load_buffer_path or load_buffer_payload may be provided")

        # Reuse the shared contact-policy metadata and robot-model initialization for one robot only.
        dataset_kwargs = dict(kwargs)
        dataset_kwargs["robot_names"] = [self.robot_name]
        dataset_kwargs["global_robot_names"] = [self.robot_name]
        dataset_kwargs["max_contact_points"] = effective_max_contact_points
        _BaseContactPolicyDataset.__init__(self, **dataset_kwargs)

        # Precompute the fixed supervision mask and the contact region used by the probe task.
        self.spec = dict(self.robot_specs[self.robot_name])
        self.model = self.spec["model"]
        self.robot_model_name = str(self.spec["robot_model_name"])
        self.local_dof = int(self.model.dof)
        self.robot_index = 0

        self.supervised_joint_indices = self._resolve_supervised_joint_indices()
        self.supervised_action_indices = sorted(
            set(self.supervised_joint_indices + (list(self.model.base_pose_indices) if self.supervise_base_pose else []))
        )
        self.action_valid_mask_template = torch.zeros((self.action_dim,), dtype=torch.bool)
        self.action_valid_mask_template[self.supervised_action_indices] = True

        self.allowed_surface_mask, self.allowed_surface_indices = self._build_allowed_surface_region()
        allowed_point_prob = self.spec["point_prob"][self.allowed_surface_indices].float()
        self.allowed_point_prob = allowed_point_prob / allowed_point_prob.sum().clamp_min(1e-8)
        self.fixed_tip_point_indices = (
            self._resolve_fixed_tip_point_indices() if self.contact_mode == "tips" else None
        )

        self.probe_component_range = tuple(int(x) for x in probe_component_range)
        component_values, component_distribution = _build_component_distribution_from_range(
            self.probe_component_range,
            values=probe_component_count_values,
            probabilities=probe_component_count_distribution,
        )
        self.probe_component_values = component_values.to(device=self.device, dtype=torch.long)
        self.probe_component_distribution = component_distribution.to(device=self.device, dtype=torch.float32)
        contact_lo = max(1, int(probe_contact_count_range[0]))
        contact_hi = max(contact_lo, int(probe_contact_count_range[1]))
        self.probe_contact_count_range = (contact_lo, contact_hi)
        self.max_probe_anchors = max(1, min(int(self.probe_component_range[1]), int(self.allowed_surface_indices.numel())))

        # Keep the stored probe split compact: only contact observations and the full target action are buffered.
        self.buffer_contact_cloud = torch.empty(
            (self.samples, self.max_contact_points, 6),
            dtype=torch.float32,
            device="cpu",
        )
        self.buffer_contact_valid_mask = torch.empty(
            (self.samples, self.max_contact_points),
            dtype=torch.bool,
            device="cpu",
        )
        self.buffer_contact_point_indices = torch.empty(
            (self.samples, self.max_contact_points),
            dtype=torch.int16,
            device="cpu",
        )
        self.buffer_action = torch.empty(
            (self.samples, self.action_dim),
            dtype=torch.float32,
            device="cpu",
        )
        self.buffer_target_q_full = torch.empty(
            (self.samples, self.action_dim),
            dtype=torch.float32,
            device="cpu",
        )

        # Probe splits are fixed once built, but they can also be reloaded later for training or visualization.
        if load_buffer_payload is not None:
            self._load_buffer_payload(load_buffer_payload)
            self.buffer_build_time_s = 0.0
        elif load_buffer_path is not None:
            payload = torch.load(str(Path(load_buffer_path).resolve()), map_location="cpu", weights_only=False)
            self._load_buffer_payload(payload)
            self.buffer_build_time_s = 0.0
        else:
            build_t0 = time.perf_counter()
            self._populate_buffer()
            self.buffer_build_time_s = time.perf_counter() - build_t0

    # Resolve the subset of ShadowHand joints that the probe task is allowed to supervise.
    def _resolve_supervised_joint_indices(self) -> list[int]:
        indices: list[int] = []
        for joint_idx, joint_name in enumerate(self.model.joint_orders):
            upper = str(joint_name).upper()
            if any(upper.startswith(prefix) for prefix in self.supervised_joint_prefixes):
                indices.append(joint_idx)
        if len(indices) == 0:
            raise RuntimeError(
                f"No joints matched supervised_joint_prefixes={self.supervised_joint_prefixes} "
                f"for robot={self.robot_name}"
            )
        return indices

    # Restrict patch-based probes to palm + thumb + index surface points.
    def _build_allowed_surface_region(self) -> tuple[torch.Tensor, torch.Tensor]:
        link_name_to_idx = {str(name): idx for idx, name in enumerate(self.model.mesh_link_names)}
        missing = [name for name in self.allowed_contact_link_names if name not in link_name_to_idx]
        if missing:
            raise KeyError(f"Unknown allowed_contact_link_names for {self.robot_name}: {missing}")
        allowed_link_idx = torch.as_tensor(
            [link_name_to_idx[name] for name in self.allowed_contact_link_names],
            dtype=torch.long,
            device=self.device,
        )
        link_indices = self.model.surface_template_link_indices.to(device=self.device, dtype=torch.long)
        allowed_surface_mask = (link_indices.view(1, -1) == allowed_link_idx.view(-1, 1)).any(dim=0)
        allowed_surface_indices = torch.nonzero(allowed_surface_mask, as_tuple=False).view(-1)
        if int(allowed_surface_indices.numel()) == 0:
            raise RuntimeError(f"No allowed surface points found for {self.robot_name}")
        return allowed_surface_mask.bool(), allowed_surface_indices.long()

    # For the fixed-tip tasks, choose one deterministic template point per fingertip.
    def _resolve_fixed_tip_point_indices(self) -> torch.Tensor:
        link_name_to_idx = {str(name): idx for idx, name in enumerate(self.model.mesh_link_names)}
        missing = [name for name in self.fixed_tip_links if name not in link_name_to_idx]
        if missing:
            raise KeyError(f"Unknown fixed_tip_links for {self.robot_name}: {missing}")
        if "palm" not in link_name_to_idx:
            raise KeyError(f"Robot {self.robot_name} is missing a 'palm' link required for fixed-tip selection")

        q_zero = torch.zeros((1, self.local_dof), dtype=torch.float32, device=self.device)
        points_zero, _ = self.model.get_surface_points_normals_batch(q=q_zero)
        points_zero = points_zero[0]
        link_indices = self.model.surface_template_link_indices.to(device=self.device, dtype=torch.long)
        palm_mask = link_indices == int(link_name_to_idx["palm"])
        if not bool(palm_mask.any()):
            raise RuntimeError(f"Robot {self.robot_name} has no palm surface points")
        palm_centroid = points_zero[palm_mask].mean(dim=0)

        selected: list[int] = []
        for link_name in self.fixed_tip_links:
            link_mask = link_indices == int(link_name_to_idx[link_name])
            if not bool(link_mask.any()):
                raise RuntimeError(f"Robot {self.robot_name} has no surface points on link {link_name}")
            global_indices = torch.nonzero(link_mask, as_tuple=False).view(-1)
            link_points = points_zero[global_indices]
            dist = torch.norm(link_points - palm_centroid.view(1, 3), dim=1)
            selected.append(int(global_indices[int(dist.argmax().item())].item()))
        return torch.as_tensor(selected, dtype=torch.long, device=self.device)

    # Sample a full ShadowHand configuration, then zero out the unsupervised joints.
    def _build_supervised_q_batch(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        base_pose_mode = "sampled_q2" if self.supervise_base_pose else "canonical"
        sampled_q = self._sample_random_q_batch(
            model=self.model,
            batch_size=int(batch_size),
            generator=generator,
            base_pose_mode=base_pose_mode,
        )
        q = torch.zeros_like(sampled_q)
        if self.supervise_base_pose and len(self.model.base_pose_indices) > 0:
            q[:, self.model.base_pose_indices] = sampled_q[:, self.model.base_pose_indices]
        q[:, self.supervised_joint_indices] = sampled_q[:, self.supervised_joint_indices]
        return q

    # Patch-based probes can draw component counts either uniformly or from a fitted distribution.
    def _sample_probe_component_count_batch(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        comp_lo = max(1, min(int(self.probe_component_range[0]), self.max_probe_anchors))
        comp_hi = max(comp_lo, min(int(self.probe_component_range[1]), self.max_probe_anchors))
        if self.component_sampling_mode == "uniform":
            return torch.randint(
                comp_lo,
                comp_hi + 1,
                (int(batch_size),),
                device=self.device,
                generator=generator,
            )
        if self.component_sampling_mode == "distribution":
            valid = (
                (self.probe_component_values >= comp_lo)
                & (self.probe_component_values <= comp_hi)
                & (self.probe_component_distribution > 0)
            )
            values = self.probe_component_values[valid]
            probs = self.probe_component_distribution[valid]
            if int(values.numel()) == 0:
                values = torch.arange(comp_lo, comp_hi + 1, dtype=torch.long, device=self.device)
                probs = torch.ones((int(values.numel()),), dtype=torch.float32, device=self.device)
            probs = probs / probs.sum().clamp_min(1e-8)
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

    # Sample anchors only inside the allowed contact region and expand them into fixed-shape patch masks.
    def _sample_probe_patch_params(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> Dict[str, torch.Tensor]:
        max_anchors = int(self.max_probe_anchors)
        active_anchor_count = self._sample_probe_component_count_batch(
            batch_size=int(batch_size),
            generator=generator,
        ).clamp(1, max_anchors)
        anchor_valid_mask = (
            torch.arange(max_anchors, device=self.device).unsqueeze(0)
            < active_anchor_count.unsqueeze(1)
        )
        anchor_local_indices = _sample_anchor_indices_batch(
            batch_size=int(batch_size),
            max_anchors=max_anchors,
            num_points=int(self.allowed_surface_indices.numel()),
            point_prob=self.allowed_point_prob,
            anchor_sampling_mode=self.anchor_sampling_mode,
            generator=generator,
            temperature=self.anchor_temperature,
            device=self.device,
        )
        anchor_indices = self.allowed_surface_indices.index_select(0, anchor_local_indices.view(-1)).view(
            int(batch_size), max_anchors
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
        )
        anchor_point_valid_mask = anchor_valid_mask.unsqueeze(-1) & point_slot_valid.unsqueeze(1)
        return {
            "anchor_indices": anchor_indices,
            "anchor_valid_mask": anchor_valid_mask,
            "anchor_point_valid_mask": anchor_point_valid_mask,
        }

    # Tasks 1 and 2 observe exactly two fixed fingertip points.
    def _build_fixed_tip_batch(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, int]]:
        q_target = self._build_supervised_q_batch(int(batch_size), generator)
        points, normals = self.model.get_surface_points_normals_batch(q=q_target)
        tip_idx = self.fixed_tip_point_indices.view(1, -1).expand(int(batch_size), -1)
        gather_idx = tip_idx.unsqueeze(-1).expand(-1, -1, 3)
        tip_points = torch.gather(points, 1, gather_idx)
        tip_normals = torch.gather(normals, 1, gather_idx)
        contact_cloud = torch.cat([tip_points, tip_normals], dim=-1)
        batch = {
            "contact_cloud": contact_cloud,
            "contact_valid_mask": torch.ones(
                (int(batch_size), self.max_contact_points),
                dtype=torch.bool,
                device=self.device,
            ),
            "contact_point_indices": tip_idx,
            "action": self._pad_action_batch(q_target),
            "target_q_full": self._pad_action_batch(q_target),
        }
        return batch, _init_contact_count_interval_stats()

    # Task 3 reuses the current synthetic contact sampler, but only on the allowed hand region.
    def _build_patch_batch(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, int]]:
        q_target = self._build_supervised_q_batch(int(batch_size), generator)
        points, normals = self.model.get_surface_points_normals_batch(q=q_target)
        patch_params = self._sample_probe_patch_params(int(batch_size), generator)
        object_points, object_valid_mask = self.model._sample_virtual_object_patches_batch(
            hand_points=points,
            hand_normals=normals,
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
        flat_object_points = object_points.view(int(batch_size), -1, 3)
        flat_object_valid_mask = object_valid_mask.view(int(batch_size), -1)
        contact_values = self.model._compute_gendex_contact_value_source_target_batch(
            source_points=points,
            source_normals=normals,
            target_points=flat_object_points,
            target_valid_mask=flat_object_valid_mask,
            align_exp_scale=self.align_exp_scale,
            sigmoid_scale=self.sigmoid_scale,
        )
        allowed_mask = self.allowed_surface_mask.view(1, -1).expand(int(batch_size), -1)
        contact_values = contact_values.masked_fill(~allowed_mask, -1.0e9)
        surface_mask, count_interval_stats = self._build_surface_mask_batch(
            contact_values,
            self.probe_contact_count_range,
        )
        surface_mask = surface_mask & allowed_mask
        contact_cloud, contact_valid_mask, contact_point_indices = self._prepare_contact_cloud_batch(
            points=points,
            normals=normals,
            surface_mask=surface_mask,
        )
        batch = {
            "contact_cloud": contact_cloud,
            "contact_valid_mask": contact_valid_mask,
            "contact_point_indices": contact_point_indices,
            "action": self._pad_action_batch(q_target),
            "target_q_full": self._pad_action_batch(q_target),
        }
        return batch, count_interval_stats

    # Dispatch to the probe-specific observation builder for the chosen task.
    @torch.no_grad()
    def _generate_batch(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, int]]:
        if self.contact_mode == "tips":
            return self._build_fixed_tip_batch(int(batch_size), generator)
        return self._build_patch_batch(int(batch_size), generator)

    # The on-disk representation mirrors the in-memory probe split exactly.
    def _write_batch_to_buffer(self, slot_indices: slice, batch: Dict[str, torch.Tensor]) -> None:
        self.buffer_contact_cloud[slot_indices] = batch["contact_cloud"].detach().to(device="cpu", dtype=torch.float32)
        self.buffer_contact_valid_mask[slot_indices] = batch["contact_valid_mask"].detach().to(device="cpu", dtype=torch.bool)
        self.buffer_contact_point_indices[slot_indices] = batch["contact_point_indices"].detach().to(
            device="cpu",
            dtype=torch.int16,
        )
        self.buffer_action[slot_indices] = batch["action"].detach().to(device="cpu", dtype=torch.float32)
        self.buffer_target_q_full[slot_indices] = batch["target_q_full"].detach().to(device="cpu", dtype=torch.float32)

    # Build the entire fixed split once, with a progress bar and interval stats for the patch task.
    def _populate_buffer(self) -> None:
        generator = self._make_generator(self.seed)
        count_interval_stats = _init_contact_count_interval_stats()
        progress = tqdm(total=self.samples, desc=f"{self.progress_label}:build", leave=False)
        offset = 0
        while offset < self.samples:
            cur = min(self.build_batch_size, self.samples - offset)
            batch, batch_stats = self._generate_batch(cur, generator)
            _merge_contact_count_interval_stats(count_interval_stats, batch_stats)
            self._write_batch_to_buffer(slice(offset, offset + cur), batch)
            offset += cur
            progress.update(cur)
        progress.close()
        if self.contact_mode == "patches":
            print(
                _format_contact_count_interval_summary(
                    f"{self.progress_label}:{self.robot_name}",
                    count_interval_stats,
                )
            )

    # Each sample exposes the exact fields the probe model and metrics need.
    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        return {
            "contact_cloud": self.buffer_contact_cloud[idx],
            "contact_valid_mask": self.buffer_contact_valid_mask[idx],
            "contact_point_indices": self.buffer_contact_point_indices[idx].to(dtype=torch.long),
            "action": self.buffer_action[idx],
            "action_valid_mask": self.action_valid_mask_template,
            "robot_index": torch.tensor(self.robot_index, dtype=torch.long),
            "local_dof": torch.tensor(self.local_dof, dtype=torch.long),
            "robot_name": self.robot_name,
            "target_q_full": self.buffer_target_q_full[idx],
        }

    # Buffer files use the same lightweight {meta, buffers} pattern as the main contact-policy buffers.
    @staticmethod
    def _read_buffer_payload(buffer_path: str | Path) -> Dict[str, Any]:
        payload = torch.load(str(Path(buffer_path).resolve()), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "meta" not in payload or "buffers" not in payload:
            raise ValueError(f"Invalid probe buffer payload: {buffer_path}")
        return payload

    # Rehydrate a saved probe split without resynthesizing any data.
    @classmethod
    def from_buffer_file(
        cls,
        buffer_path: str | Path,
        **kwargs: Any,
    ) -> "ShadowhandContactPolicyProbeDataset":
        payload = cls._read_buffer_payload(buffer_path)
        meta = payload["meta"]
        init_kwargs = dict(kwargs)
        init_kwargs.pop("max_contact_points", None)
        return cls(
            task=str(meta["task"]),
            robot_name=str(meta["robot_name"]),
            samples=int(meta["samples"]),
            build_batch_size=1,
            progress_label=str(meta.get("progress_label", "probe_dataset")),
            max_contact_points=int(meta["max_contact_points"]),
            fixed_tip_links=meta["fixed_tip_links"],
            supervised_joint_prefixes=meta["supervised_joint_prefixes"],
            allowed_contact_link_names=meta["allowed_contact_link_names"],
            probe_component_range=tuple(meta["probe_component_range"]),
            probe_component_count_values=meta["probe_component_count_values"],
            probe_component_count_distribution=meta["probe_component_count_distribution"],
            probe_contact_count_range=tuple(meta["probe_contact_count_range"]),
            load_buffer_payload=payload,
            **init_kwargs,
        )

    # Keep the serialized payload simple so later visualization can load it directly.
    def _build_buffer_payload(self) -> Dict[str, Any]:
        return {
            "meta": {
                "task": self.task,
                "contact_mode": self.contact_mode,
                "base_mode": self.base_mode,
                "robot_name": self.robot_name,
                "samples": int(self.samples),
                "max_contact_points": int(self.max_contact_points),
                "fixed_tip_links": list(self.fixed_tip_links),
                "supervised_joint_prefixes": list(self.supervised_joint_prefixes),
                "allowed_contact_link_names": list(self.allowed_contact_link_names),
                "probe_component_range": list(self.probe_component_range),
                "probe_component_count_values": self.probe_component_values.detach().cpu(),
                "probe_component_count_distribution": self.probe_component_distribution.detach().cpu(),
                "probe_contact_count_range": list(self.probe_contact_count_range),
                "progress_label": self.progress_label,
            },
            "buffers": {
                "contact_cloud": self.buffer_contact_cloud.detach().cpu(),
                "contact_valid_mask": self.buffer_contact_valid_mask.detach().cpu(),
                "contact_point_indices": self.buffer_contact_point_indices.detach().cpu(),
                "action": self.buffer_action.detach().cpu(),
                "target_q_full": self.buffer_target_q_full.detach().cpu(),
            },
        }

    # Public save/load helpers match the existing fixed-buffer dataset pattern.
    def save_buffer(self, buffer_path: str | Path) -> None:
        path = Path(buffer_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._build_buffer_payload(), str(path))

    # Restore the fixed split tensors directly from disk without touching the sampling path.
    def _load_buffer_payload(self, payload: Dict[str, Any]) -> None:
        meta = payload["meta"]
        buffers = payload["buffers"]
        if int(meta["samples"]) != int(self.samples):
            raise ValueError(f"Saved probe buffer samples mismatch: {meta['samples']} vs {self.samples}")
        if str(meta["task"]) != self.task:
            raise ValueError(f"Saved probe buffer task mismatch: {meta['task']} vs {self.task}")
        if str(meta["robot_name"]) != self.robot_name:
            raise ValueError(f"Saved probe buffer robot mismatch: {meta['robot_name']} vs {self.robot_name}")
        if int(meta["max_contact_points"]) != int(self.max_contact_points):
            raise ValueError(
                f"Saved probe buffer max_contact_points mismatch: {meta['max_contact_points']} vs {self.max_contact_points}"
            )
        self.buffer_contact_cloud.copy_(torch.as_tensor(buffers["contact_cloud"], dtype=torch.float32, device="cpu"))
        self.buffer_contact_valid_mask.copy_(
            torch.as_tensor(buffers["contact_valid_mask"], dtype=torch.bool, device="cpu")
        )
        self.buffer_contact_point_indices.copy_(
            torch.as_tensor(buffers["contact_point_indices"], dtype=torch.int16, device="cpu")
        )
        self.buffer_action.copy_(torch.as_tensor(buffers["action"], dtype=torch.float32, device="cpu"))
        self.buffer_target_q_full.copy_(torch.as_tensor(buffers["target_q_full"], dtype=torch.float32, device="cpu"))
