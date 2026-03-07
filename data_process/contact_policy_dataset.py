from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterator, List

import torch
import yaml
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from robot_model.robot_model import create_robot_model, farthest_point_sampling


def _stable_int_seed(key: str) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**31 - 1)


def _load_real_masks_by_robot(
    real_masks_path: str | Path,
    num_points_by_robot: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    payload = torch.load(real_masks_path, map_location="cpu")
    if not isinstance(payload, dict) or "samples" not in payload:
        raise ValueError(f"Invalid real mask payload: {real_masks_path}")

    out: Dict[str, List[torch.Tensor]] = {}
    for sample in payload["samples"]:
        robot_name = str(sample.get("robot_name", ""))
        if robot_name not in num_points_by_robot:
            continue
        if sample.get("invalid_reason") is not None:
            continue
        mask = torch.as_tensor(sample["hand_contact_mask"]).bool().view(-1)
        if int(mask.numel()) != int(num_points_by_robot[robot_name]):
            continue
        out.setdefault(robot_name, []).append(mask)

    stacked: Dict[str, torch.Tensor] = {}
    for robot_name, masks in out.items():
        if len(masks) > 0:
            stacked[robot_name] = torch.stack(masks, dim=0).bool()
    return stacked


def _extract_real_count_values(real_masks: torch.Tensor) -> torch.Tensor:
    return real_masks.detach().bool().cpu().sum(dim=1).long()


def _build_point_prob_from_real_masks(real_masks: torch.Tensor) -> torch.Tensor:
    point_prob = real_masks.detach().bool().cpu().float().mean(dim=0).clamp_min(1e-8)
    return point_prob / point_prob.sum().clamp_min(1e-8)


def _sample_anchor_indices(
    n_anchor: int,
    num_points: int,
    point_prob: torch.Tensor | None,
    anchor_sampling_mode: str,
    generator: torch.Generator,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    n = int(max(1, min(int(n_anchor), int(num_points))))
    mode = str(anchor_sampling_mode).strip().lower()
    if mode == "uniform":
        return torch.randperm(int(num_points), device=device, generator=generator)[:n]
    if mode != "point_prob":
        raise ValueError(f"Invalid anchor_sampling_mode={anchor_sampling_mode}")
    if point_prob is None:
        raise ValueError("point_prob is required when anchor_sampling_mode='point_prob'")

    p = point_prob.to(device=device, dtype=torch.float32).clamp_min(1e-8)
    t = float(max(1e-3, float(temperature)))
    if abs(t - 1.0) > 1e-6:
        p = p.pow(1.0 / t)
    p = p / p.sum().clamp_min(1e-8)
    return torch.multinomial(p, num_samples=n, replacement=False, generator=generator)


def _sample_target_count(
    count_values: torch.Tensor,
    num_points: int,
    generator: torch.Generator,
    device: torch.device,
) -> int:
    idx = int(
        torch.randint(
            0,
            int(count_values.numel()),
            (1,),
            device=device,
            generator=generator,
        ).item()
    )
    return int(max(1, min(int(num_points), int(count_values[idx].item()))))


def _resize_mask_to_target_count(mask: torch.Tensor, scores: torch.Tensor, target_count: int) -> torch.Tensor:
    n = int(mask.numel())
    k = int(max(1, min(n, int(target_count))))
    order = torch.argsort(scores, descending=True)
    out = torch.zeros_like(mask, dtype=torch.bool)
    out[order[:k]] = True
    return out


class _BaseContactPolicyDataset:
    def __init__(
        self,
        *,
        robot_names: List[str],
        hparams_path: str | Path,
        real_masks_path: str | Path,
        max_contact_points: int = 256,
        seed: int = 42,
        device: str = "cpu",
        check_template_hash: bool = True,
        threshold: float = 0.4,
        align_exp_scale: float = 2.0,
        sigmoid_scale: float = 10.0,
        anchor_sampling_mode: str = "point_prob",
        anchor_temperature: float = 1.0,
        patch_points: int = 96,
        patch_anchor_shift_min: float = 0.0002,
        patch_anchor_shift_max: float = 0.0030,
        patch_extent_min: float = 0.006,
        patch_extent_max: float = 0.050,
        patch_extent_power: float = 1.0,
        patch_shift_power: float = 1.0,
        patch_points_per_anchor_min: int = 28,
        patch_points_per_anchor_max: int = 84,
        patch_penetration_clearance: float = 0.00008,
        patch_exact_count_prob: float = 1.0,
        patch_count_clamp_ratio: float = 0.3,
    ) -> None:
        self.robot_names = [str(x) for x in robot_names]
        self.hparams_path = Path(hparams_path)
        self.real_masks_path = Path(real_masks_path)
        self.max_contact_points = int(max_contact_points)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.check_template_hash = bool(check_template_hash)

        self.threshold = float(threshold)
        self.align_exp_scale = float(align_exp_scale)
        self.sigmoid_scale = float(sigmoid_scale)
        self.anchor_sampling_mode = str(anchor_sampling_mode)
        self.anchor_temperature = float(anchor_temperature)
        self.patch_points = int(patch_points)
        self.patch_anchor_shift_min = float(patch_anchor_shift_min)
        self.patch_anchor_shift_max = float(patch_anchor_shift_max)
        self.patch_extent_min = float(patch_extent_min)
        self.patch_extent_max = float(patch_extent_max)
        self.patch_extent_power = float(patch_extent_power)
        self.patch_shift_power = float(patch_shift_power)
        self.patch_points_per_anchor_min = int(patch_points_per_anchor_min)
        self.patch_points_per_anchor_max = int(patch_points_per_anchor_max)
        self.patch_penetration_clearance = float(patch_penetration_clearance)
        self.patch_exact_count_prob = float(patch_exact_count_prob)
        self.patch_count_clamp_ratio = float(patch_count_clamp_ratio)

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
        real_masks_by_robot = _load_real_masks_by_robot(self.real_masks_path, num_points_by_robot)

        self.robot_specs: Dict[str, Dict[str, Any]] = {}
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
            count_values = _extract_real_count_values(real_masks)
            point_prob = _build_point_prob_from_real_masks(real_masks)
            component_range = rcfg.get("component_range", [1, 4])
            if len(component_range) < 2:
                component_range = [1, 4]

            surface_num_points_i = int(rcfg["num_surface_points"])
            if surface_num_points is None:
                surface_num_points = surface_num_points_i
            elif int(surface_num_points) != int(surface_num_points_i):
                raise RuntimeError("All robots must share the same surface point count for this dataset")

            local_dof = int(model.dof)
            max_action_dim = max(max_action_dim, local_dof)
            self.robot_specs[robot_name] = {
                "robot_name": robot_name,
                "robot_model_name": str(rcfg["robot_model_name"]),
                "model": model,
                "surface_num_points": surface_num_points_i,
                "real_masks": real_masks,
                "count_values": count_values,
                "point_prob": point_prob,
                "component_range": (int(component_range[0]), int(component_range[1])),
                "local_dof": local_dof,
            }

        self.surface_num_points = int(surface_num_points or 0)
        self.action_dim = int(max_action_dim)

    def _make_generator(self, sample_seed: int) -> torch.Generator:
        generator = torch.Generator(device=self.device.type)
        generator.manual_seed(int(sample_seed))
        return generator

    def _sample_synthetic_contact(
        self,
        spec: Dict[str, Any],
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model = spec["model"]
        q2 = model.sample_random_q(generator=generator).to(self.device)
        hand_points, hand_normals = model.get_surface_points_normals(q=q2)
        comp_lo, comp_hi = spec["component_range"]
        n_anchor = int(
            torch.randint(
                int(comp_lo),
                int(comp_hi) + 1,
                (1,),
                device=self.device,
                generator=generator,
            ).item()
        )
        anchor_idx = _sample_anchor_indices(
            n_anchor=n_anchor,
            num_points=self.surface_num_points,
            point_prob=spec["point_prob"],
            anchor_sampling_mode=self.anchor_sampling_mode,
            generator=generator,
            temperature=self.anchor_temperature,
            device=self.device,
        )
        target_count = _sample_target_count(
            count_values=spec["count_values"].to(self.device),
            num_points=self.surface_num_points,
            generator=generator,
            device=self.device,
        )

        us = float(torch.rand(1, device=self.device, generator=generator).item()) ** max(1e-3, self.patch_shift_power)
        ue = float(torch.rand(1, device=self.device, generator=generator).item()) ** max(1e-3, self.patch_extent_power)
        anchor_shift = float(
            self.patch_anchor_shift_min
            + us * max(1e-8, self.patch_anchor_shift_max - self.patch_anchor_shift_min)
        )
        max_extent = float(
            self.patch_extent_min
            + ue * max(1e-8, self.patch_extent_max - self.patch_extent_min)
        )
        points_per_anchor = int(
            torch.randint(
                self.patch_points_per_anchor_min,
                self.patch_points_per_anchor_max + 1,
                (1,),
                device=self.device,
                generator=generator,
            ).item()
        )
        patch_points_gen = int(max(self.patch_points, points_per_anchor * n_anchor))
        object_points, _ = model._sample_virtual_object_patches(
            hand_points=hand_points,
            hand_normals=hand_normals,
            num_anchors=n_anchor,
            total_patch_points=patch_points_gen,
            generator=generator,
            anchor_indices=anchor_idx,
            anchor_shift=anchor_shift,
            max_plane_extent=max_extent,
            penetration_clearance=self.patch_penetration_clearance,
        )
        contact_values = model._compute_gendex_contact_value_source_target(
            source_points=hand_points,
            source_normals=hand_normals,
            target_points=object_points,
            align_exp_scale=self.align_exp_scale,
            sigmoid_scale=self.sigmoid_scale,
        )
        mask = contact_values >= self.threshold
        if float(torch.rand(1, device=self.device, generator=generator).item()) < self.patch_exact_count_prob:
            mask = _resize_mask_to_target_count(mask=mask, scores=contact_values, target_count=target_count)
        else:
            clamp_ratio = max(0.0, self.patch_count_clamp_ratio)
            k_lo = int(max(1, round((1.0 - clamp_ratio) * float(target_count))))
            k_hi = int(min(self.surface_num_points, round((1.0 + clamp_ratio) * float(target_count))))
            cur_k = int(mask.sum().item())
            if cur_k < k_lo:
                mask = _resize_mask_to_target_count(mask=mask, scores=contact_values, target_count=k_lo)
            elif cur_k > k_hi:
                mask = _resize_mask_to_target_count(mask=mask, scores=contact_values, target_count=k_hi)
        return q2, hand_points, hand_normals, mask.bool()

    def _prepare_contact_cloud(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        surface_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_idx = torch.nonzero(surface_mask, as_tuple=False).view(-1)
        cloud_full = torch.cat([points[valid_idx], normals[valid_idx]], dim=-1)
        if int(cloud_full.shape[0]) == 0:
            cloud = torch.zeros((self.max_contact_points, 6), dtype=torch.float32, device=self.device)
            valid_mask = torch.zeros((self.max_contact_points,), dtype=torch.bool, device=self.device)
            padded_idx = torch.full((self.max_contact_points,), -1, dtype=torch.long, device=self.device)
            center = torch.zeros((3,), dtype=torch.float32, device=self.device)
            return cloud, valid_mask, padded_idx, center

        if int(cloud_full.shape[0]) > self.max_contact_points:
            cloud_sel, sel_idx = farthest_point_sampling(cloud_full, self.max_contact_points)
            valid_idx = valid_idx[sel_idx]
        else:
            cloud_sel = cloud_full

        center = cloud_sel[:, :3].mean(dim=0)
        cloud_sel = cloud_sel.clone()
        cloud_sel[:, :3] = cloud_sel[:, :3] - center

        n_sel = int(cloud_sel.shape[0])
        cloud = torch.zeros((self.max_contact_points, 6), dtype=torch.float32, device=self.device)
        valid_mask = torch.zeros((self.max_contact_points,), dtype=torch.bool, device=self.device)
        padded_idx = torch.full((self.max_contact_points,), -1, dtype=torch.long, device=self.device)
        cloud[:n_sel] = cloud_sel
        valid_mask[:n_sel] = True
        padded_idx[:n_sel] = valid_idx.long()
        return cloud, valid_mask, padded_idx, center

    def _pad_action(self, q: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((self.action_dim,), dtype=torch.float32, device=self.device)
        out[: int(q.numel())] = q
        return out

    def _build_sample(
        self,
        *,
        spec: Dict[str, Any],
        q1: torch.Tensor,
        q2: torch.Tensor,
        surface_mask: torch.Tensor,
        q1_points: torch.Tensor | None = None,
        q1_normals: torch.Tensor | None = None,
        q2_points: torch.Tensor | None = None,
        q2_normals: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | str]:
        model = spec["model"]
        if q1_points is None or q1_normals is None:
            q1_points, q1_normals = model.get_surface_points_normals(q=q1)
        if q2_points is None or q2_normals is None:
            q2_points, q2_normals = model.get_surface_points_normals(q=q2)

        p_hat = torch.cat([q1_points, q1_normals], dim=-1)
        contact_cloud, contact_valid_mask, contact_point_indices, contact_center = self._prepare_contact_cloud(
            points=q2_points,
            normals=q2_normals,
            surface_mask=surface_mask,
        )

        mixed_q = model.mix_q_by_contact_mask(q1, q2, surface_mask)
        if bool(contact_valid_mask.any()) and len(model.base_translation_indices) > 0:
            for axis, joint_idx in enumerate(model.base_translation_indices[:3]):
                mixed_q[joint_idx] = mixed_q[joint_idx] - contact_center[axis]

        action = self._pad_action(mixed_q)
        q1_padded = self._pad_action(q1)
        q2_padded = self._pad_action(q2)
        action_valid_mask = torch.zeros((self.action_dim,), dtype=torch.bool, device=self.device)
        action_valid_mask[: int(spec["local_dof"])] = True

        return {
            "p_hat": p_hat.detach().cpu(),
            "contact_cloud": contact_cloud.detach().cpu(),
            "contact_valid_mask": contact_valid_mask.detach().cpu(),
            "action": action.detach().cpu(),
            "action_valid_mask": action_valid_mask.detach().cpu(),
            "q1_padded": q1_padded.detach().cpu(),
            "q2_padded": q2_padded.detach().cpu(),
            "contact_mask_surface": surface_mask.detach().cpu(),
            "contact_point_indices": contact_point_indices.detach().cpu(),
            "contact_center": contact_center.detach().cpu(),
            "local_dof": torch.tensor(spec["local_dof"], dtype=torch.long),
            "robot_name": str(spec["robot_name"]),
            "robot_model_name": str(spec["robot_model_name"]),
        }


class OnTheFlySyntheticContactPolicyDataset(IterableDataset, _BaseContactPolicyDataset):
    def __init__(
        self,
        *,
        samples_per_epoch: int = 10000,
        train_chunk_size: int = 256,
        **kwargs: Any,
    ) -> None:
        IterableDataset.__init__(self)
        _BaseContactPolicyDataset.__init__(self, **kwargs)
        self.samples_per_epoch = int(samples_per_epoch)
        self.train_chunk_size = int(max(1, train_chunk_size))
        self._iter_counter = 0

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _generate_train_sample(self, generator: torch.Generator) -> Dict[str, torch.Tensor | str]:
        robot_idx = int(
            torch.randint(
                0,
                len(self.robot_names),
                (1,),
                device=self.device,
                generator=generator,
            ).item()
        )
        robot_name = self.robot_names[robot_idx]
        spec = self.robot_specs[robot_name]
        model = spec["model"]
        q1 = model.sample_random_q(generator=generator).to(self.device)
        q2, q2_points, q2_normals, surface_mask = self._sample_synthetic_contact(spec, generator=generator)
        return self._build_sample(
            spec=spec,
            q1=q1,
            q2=q2,
            surface_mask=surface_mask,
            q2_points=q2_points,
            q2_normals=q2_normals,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor | str]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        samples_per_worker = self.samples_per_epoch // num_workers
        if worker_id < self.samples_per_epoch % num_workers:
            samples_per_worker += 1

        iter_idx = self._iter_counter
        self._iter_counter += 1
        generator = self._make_generator(
            self.seed + _stable_int_seed(f"train|worker={worker_id}|iter={iter_idx}")
        )

        produced = 0
        while produced < samples_per_worker:
            chunk_size = min(self.train_chunk_size, samples_per_worker - produced)
            buffer = [self._generate_train_sample(generator=generator) for _ in range(chunk_size)]
            for sample in buffer:
                yield sample
            produced += chunk_size


class RealContactPolicyValDataset(Dataset, _BaseContactPolicyDataset):
    def __init__(
        self,
        *,
        cache_processed_samples: bool = True,
        **kwargs: Any,
    ) -> None:
        Dataset.__init__(self)
        _BaseContactPolicyDataset.__init__(self, **kwargs)
        payload = torch.load(self.real_masks_path, map_location="cpu")
        if not isinstance(payload, dict) or "samples" not in payload:
            raise ValueError(f"Invalid validation payload: {self.real_masks_path}")

        samples: List[Dict[str, Any]] = []
        for sample in payload["samples"]:
            if sample.get("invalid_reason") is not None:
                continue
            robot_name = str(sample.get("robot_name", ""))
            if robot_name not in self.robot_specs:
                continue
            mask = torch.as_tensor(sample["hand_contact_mask"]).bool().view(-1)
            if int(mask.numel()) != self.surface_num_points:
                continue
            q2 = torch.as_tensor(sample["q"]).float().view(-1)
            if int(q2.numel()) != int(self.robot_specs[robot_name]["local_dof"]):
                continue
            samples.append(
                {
                    "sample_id": int(sample.get("sample_id", len(samples))),
                    "robot_name": robot_name,
                    "q2": q2,
                    "mask": mask,
                }
            )
        if len(samples) == 0:
            raise RuntimeError(f"No validation samples found in {self.real_masks_path}")

        self.samples = samples
        self.cache_processed_samples = bool(cache_processed_samples)
        self.cached_samples: List[Dict[str, torch.Tensor | str]] | None = None
        if self.cache_processed_samples:
            self.cached_samples = [self._materialize_sample(sample) for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _materialize_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor | str]:
        robot_name = str(sample["robot_name"])
        spec = self.robot_specs[robot_name]
        sample_seed = self.seed + _stable_int_seed(f"{robot_name}|{int(sample['sample_id'])}|q1")
        generator = self._make_generator(sample_seed)
        model = spec["model"]
        q1 = model.sample_random_q(generator=generator).to(self.device)
        q2 = torch.as_tensor(sample["q2"], dtype=torch.float32, device=self.device)
        surface_mask = torch.as_tensor(sample["mask"], dtype=torch.bool, device=self.device)
        return self._build_sample(
            spec=spec,
            q1=q1,
            q2=q2,
            surface_mask=surface_mask,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        if self.cached_samples is not None:
            return self.cached_samples[idx]
        return self._materialize_sample(self.samples[idx])
