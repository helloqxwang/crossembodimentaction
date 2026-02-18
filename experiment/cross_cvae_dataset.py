from __future__ import annotations

import json
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class CrossEmbodimentActionDataset(Dataset):
    """Dataset for cross-embodiment CVAE with zero-padded joint vectors."""

    def __init__(
        self,
        *,
        dro_root: str,
        robot_names: List[str],
        split: str,
        num_points: int = 512,
        max_dof: int | None = None,
        robot_to_idx: Dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        if split not in {"train", "validate"}:
            raise ValueError(f"split must be 'train' or 'validate', got: {split}")

        self.dro_root = dro_root
        self.robot_names = list(robot_names)
        self.split = split
        self.num_points = int(num_points)
        self.robot_name_set = set(self.robot_names)
        if robot_to_idx is None:
            self.robot_to_idx = {name: i for i, name in enumerate(self.robot_names)}
        else:
            self.robot_to_idx = {str(k): int(v) for k, v in robot_to_idx.items()}
            for name in self.robot_names:
                if name not in self.robot_to_idx:
                    raise ValueError(f"robot_names includes '{name}' but robot_to_idx has no such key")

        split_path = os.path.join(self.dro_root, "data/CMapDataset_filtered/split_train_validate_objects.json")
        dataset_split = json.load(open(split_path, "r", encoding="utf-8"))
        split_objects = set(dataset_split[self.split])

        dataset_path = os.path.join(self.dro_root, "data/CMapDataset_filtered/cmap_dataset.pt")
        dataset = torch.load(dataset_path, map_location="cpu")
        metadata = dataset["metadata"]

        selected_samples = []
        robot_dofs: Dict[str, int] = {name: 0 for name in self.robot_names}
        for q, object_name, robot_name in metadata:
            if robot_name not in self.robot_name_set:
                continue
            if object_name not in split_objects:
                continue

            q_tensor = q if torch.is_tensor(q) else torch.tensor(q, dtype=torch.float32)
            q_tensor = q_tensor.float()
            selected_samples.append((q_tensor, object_name, robot_name))
            robot_dofs[robot_name] = max(robot_dofs[robot_name], int(q_tensor.numel()))

        if len(selected_samples) == 0:
            raise RuntimeError("No samples selected. Check robot_names, split, and DRO-Grasp data path.")

        self.samples = selected_samples
        self.robot_dofs = robot_dofs
        local_max_dof = max(robot_dofs.values())
        if max_dof is None:
            self.max_dof = local_max_dof
        else:
            if int(max_dof) < local_max_dof:
                raise ValueError(
                    f"Provided max_dof={max_dof} is smaller than split-required dof={local_max_dof}"
                )
            self.max_dof = int(max_dof)
        self.object_pc_cache = self._build_object_cache()

    def _build_object_cache(self) -> Dict[str, torch.Tensor]:
        unique_objects = sorted({obj_name for _, obj_name, _ in self.samples})
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        q, object_name, robot_name = self.samples[idx]
        dof = int(q.numel())

        action = torch.zeros(self.max_dof, dtype=torch.float32)
        action[:dof] = q

        action_mask = torch.zeros(self.max_dof, dtype=torch.bool)
        action_mask[:dof] = True

        object_pc_full = self.object_pc_cache[object_name]
        if object_pc_full.size(0) >= self.num_points:
            indices = torch.randperm(object_pc_full.size(0))[: self.num_points]
        else:
            indices = torch.randint(0, object_pc_full.size(0), (self.num_points,))
        object_pc = object_pc_full[indices].clone()

        return {
            "object_pc": object_pc,  # (P, 3)
            "action": action,  # (max_dof,)
            "action_mask": action_mask,  # (max_dof,)
            "embodiment_idx": torch.tensor(self.robot_to_idx[robot_name], dtype=torch.long),
            "robot_name": robot_name,
            "object_name": object_name,
            "dof": torch.tensor(dof, dtype=torch.long),
        }
