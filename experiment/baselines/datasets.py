from __future__ import annotations

import json
import os
from typing import Dict, Iterable

import torch
from torch.utils.data import Dataset


def load_split_objects(dro_root: str, split: str) -> set[str]:
    split_path = os.path.join(dro_root, "data/CMapDataset_filtered/split_train_validate_objects.json")
    split_json = json.load(open(split_path, "r", encoding="utf-8"))
    if split not in split_json:
        raise KeyError(f"Split '{split}' not found in {split_path}")
    return set(split_json[split])


def load_cmap_metadata(dro_root: str) -> list[tuple[torch.Tensor, str, str]]:
    dataset_path = os.path.join(dro_root, "data/CMapDataset_filtered/cmap_dataset.pt")
    dataset = torch.load(dataset_path, map_location="cpu")
    metadata = dataset["metadata"]
    out: list[tuple[torch.Tensor, str, str]] = []
    for q, object_name, robot_name in metadata:
        q_tensor = q if torch.is_tensor(q) else torch.tensor(q, dtype=torch.float32)
        out.append((q_tensor.float(), str(object_name), str(robot_name)))
    return out


def infer_robot_dofs(dro_root: str, robot_names: Iterable[str]) -> Dict[str, int]:
    robot_name_set = set(str(name) for name in robot_names)
    robot_dofs: Dict[str, int] = {name: 0 for name in robot_name_set}
    for q, _object_name, robot_name in load_cmap_metadata(dro_root):
        if robot_name not in robot_name_set:
            continue
        robot_dofs[robot_name] = max(robot_dofs[robot_name], int(q.numel()))
    missing = sorted(name for name, dof in robot_dofs.items() if dof <= 0)
    if missing:
        raise RuntimeError(f"No q samples found for robots: {missing}")
    return {name: robot_dofs[name] for name in sorted(robot_dofs.keys())}


class CVAEActionDataset(Dataset):
    def __init__(
        self,
        *,
        dro_root: str,
        robot_names: list[str],
        split: str,
        num_points: int = 512,
        action_dim: int | None = None,
        robot_to_idx: Dict[str, int] | None = None,
        include_embodiment_idx: bool = True,
    ) -> None:
        super().__init__()
        if split not in {"train", "validate"}:
            raise ValueError(f"split must be 'train' or 'validate', got: {split}")

        self.dro_root = str(dro_root)
        self.robot_names = [str(name) for name in robot_names]
        self.robot_name_set = set(self.robot_names)
        self.split = split
        self.num_points = int(num_points)
        self.include_embodiment_idx = bool(include_embodiment_idx)
        self.robot_to_idx = None if robot_to_idx is None else {str(k): int(v) for k, v in robot_to_idx.items()}

        if self.include_embodiment_idx:
            if self.robot_to_idx is None:
                self.robot_to_idx = {name: i for i, name in enumerate(self.robot_names)}
            missing = sorted(name for name in self.robot_names if name not in self.robot_to_idx)
            if missing:
                raise ValueError(f"robot_to_idx is missing robots: {missing}")

        split_objects = load_split_objects(self.dro_root, self.split)
        selected_samples: list[tuple[torch.Tensor, str, str]] = []
        robot_dofs: Dict[str, int] = {name: 0 for name in self.robot_names}
        for q, object_name, robot_name in load_cmap_metadata(self.dro_root):
            if robot_name not in self.robot_name_set:
                continue
            if object_name not in split_objects:
                continue
            selected_samples.append((q, object_name, robot_name))
            robot_dofs[robot_name] = max(robot_dofs[robot_name], int(q.numel()))

        if len(selected_samples) == 0:
            raise RuntimeError("No samples selected. Check robot_names, split, and DRO-Grasp data path.")

        self.samples = selected_samples
        self.robot_dofs = robot_dofs
        required_action_dim = max(robot_dofs.values())
        if action_dim is None:
            self.action_dim = required_action_dim
        else:
            if int(action_dim) < required_action_dim:
                raise ValueError(
                    f"Provided action_dim={action_dim} is smaller than required dof={required_action_dim}"
                )
            self.action_dim = int(action_dim)

        self.object_pc_cache = self._build_object_cache()

    def _build_object_cache(self) -> Dict[str, torch.Tensor]:
        unique_objects = sorted({object_name for _q, object_name, _robot_name in self.samples})
        cache: Dict[str, torch.Tensor] = {}
        for object_name in unique_objects:
            dataset_name, mesh_name = object_name.split("+")
            object_pc_path = os.path.join(
                self.dro_root,
                "data/PointCloud/object",
                dataset_name,
                f"{mesh_name}.pt",
            )
            if not os.path.exists(object_pc_path):
                raise FileNotFoundError(f"Object point cloud not found: {object_pc_path}")
            pc = torch.load(object_pc_path, map_location="cpu").float()
            if pc.ndim != 2 or pc.shape[1] < 3:
                raise ValueError(f"Invalid object point cloud shape {tuple(pc.shape)} at {object_pc_path}")
            cache[object_name] = pc[:, :3]
        return cache

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        q, object_name, robot_name = self.samples[idx]
        dof = int(q.numel())

        action = torch.zeros(self.action_dim, dtype=torch.float32)
        action[:dof] = q

        action_mask = torch.zeros(self.action_dim, dtype=torch.bool)
        action_mask[:dof] = True

        object_pc_full = self.object_pc_cache[object_name]
        if object_pc_full.size(0) >= self.num_points:
            indices = torch.randperm(object_pc_full.size(0))[: self.num_points]
        else:
            indices = torch.randint(0, object_pc_full.size(0), (self.num_points,))
        object_pc = object_pc_full[indices].clone()

        sample: Dict[str, torch.Tensor | str] = {
            "object_pc": object_pc,
            "action": action if self.action_dim != dof else q.clone(),
            "action_mask": action_mask if self.action_dim != dof else torch.ones(dof, dtype=torch.bool),
            "robot_name": robot_name,
            "object_name": object_name,
            "dof": torch.tensor(dof, dtype=torch.long),
        }
        if self.include_embodiment_idx:
            assert self.robot_to_idx is not None
            sample["embodiment_idx"] = torch.tensor(self.robot_to_idx[robot_name], dtype=torch.long)
        return sample
