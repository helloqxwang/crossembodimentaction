from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.utils.data

from robot_model.robot_model import create_robot_model


class NFKBoxFaceDataset(torch.utils.data.Dataset):
    """
    Dataset for random robot/q sampling with box-face point-normal pairs.

    Each sample:
    - randomly selects one robot from a configured manipulator list,
    - samples random q and zero q,
    - samples one valid face point+normal per link,
    - samples a contact-like mask over links (at most one active face per link).
    """

    def __init__(
        self,
        *,
        assets_dir: str | Path,
        robot_list_key: str = "right_hands",
        robot_list_file: str | Path | None = None,
        dataset_size: int = 10000,
        seed: int = 0,
        device: str = "cpu",
        link_num_points: int = 512,
        box_fit_override: bool = False,
    ) -> None:
        self.assets_dir = Path(assets_dir)
        self.robot_list_file = (
            Path(robot_list_file)
            if robot_list_file is not None
            else self.assets_dir / "manipulator_robot_lists.json"
        )
        self.robot_list_key = robot_list_key
        self.dataset_size = int(dataset_size)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.link_num_points = int(link_num_points)
        self.box_fit_override = bool(box_fit_override)

        with open(self.robot_list_file, "r", encoding="utf-8") as f:
            robot_lists = json.load(f)
        if self.robot_list_key not in robot_lists:
            raise KeyError(
                f"robot_list_key='{self.robot_list_key}' not found in {self.robot_list_file}. "
                f"available={list(robot_lists.keys())}"
            )
        self.robot_names: List[str] = list(robot_lists[self.robot_list_key])
        if not self.robot_names:
            raise ValueError(f"Robot list '{self.robot_list_key}' is empty")

        self.robot_models = {}
        for robot_name in self.robot_names:
            model = create_robot_model(
                robot_name=robot_name,
                device=self.device,
                num_points=self.link_num_points,
                assets_dir=self.assets_dir,
            )
            model.switch_to_box_links_only_mode(override=self.box_fit_override)
            self.robot_models[robot_name] = model

    def __len__(self) -> int:
        return self.dataset_size

    @staticmethod
    def _sample_q(model, generator: torch.Generator) -> torch.Tensor:
        lower, upper = model.pk_chain.get_joint_limits()
        lower_t = torch.tensor(lower, dtype=torch.float32, device=model.device)
        upper_t = torch.tensor(upper, dtype=torch.float32, device=model.device)
        finite = torch.isfinite(lower_t) & torch.isfinite(upper_t)

        q = torch.zeros(model.dof, dtype=torch.float32, device=model.device)
        if finite.any():
            u = torch.rand(model.dof, dtype=torch.float32, device=model.device, generator=generator)
            q[finite] = lower_t[finite] + u[finite] * (upper_t[finite] - lower_t[finite])
        if (~finite).any():
            n_unbounded = int((~finite).sum().item())
            q[~finite] = (
                torch.rand(n_unbounded, device=model.device, generator=generator) * 2.0 - 1.0
            ) * torch.pi
        return q

    @staticmethod
    def _pick_one_face_per_link(
        points: torch.Tensor,
        normals: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert per-link 6-face samples to one point-normal pair per link.
        Returns:
            pairs: (N, 6)
            valid_mask: (N,)
            picked_face_idx: (N,)
        """
        if points.ndim != 3 or normals.ndim != 3:
            raise ValueError("Expected points/normals shape (N, 6, 3)")

        n_links = points.shape[0]
        pairs = torch.zeros((n_links, 6), dtype=torch.float32, device=points.device)
        valid_mask = torch.zeros((n_links,), dtype=torch.bool, device=points.device)
        picked_face_idx = torch.full((n_links,), -1, dtype=torch.long, device=points.device)

        for i in range(n_links):
            p = points[i]   # (6, 3)
            n = normals[i]  # (6, 3)
            finite = torch.isfinite(p).all(dim=-1) & torch.isfinite(n).all(dim=-1)
            valid_idx = torch.where(finite)[0]
            if valid_idx.numel() == 0:
                continue
            chosen = valid_idx[
                torch.randint(
                    0,
                    valid_idx.numel(),
                    (1,),
                    device=points.device,
                    generator=generator,
                ).item()
            ]
            pairs[i, :3] = p[chosen]
            pairs[i, 3:] = n[chosen]
            valid_mask[i] = True
            picked_face_idx[i] = chosen

        return pairs, valid_mask, picked_face_idx

    @staticmethod
    def _sample_contact_mask(
        valid_mask: torch.Tensor,
        *,
        rng: random.Random,
        generator: torch.Generator,
    ) -> torch.Tensor:
        n_links = int(valid_mask.numel())
        valid_idx = torch.where(valid_mask)[0]
        n_valid = int(valid_idx.numel())
        contact_mask = torch.zeros((n_links,), dtype=torch.bool, device=valid_mask.device)
        if n_valid == 0:
            return contact_mask

        lower = min(4, n_links // 2)
        lower = max(1, lower)
        lower = min(lower, n_valid)
        upper = n_valid
        m = rng.randint(lower, upper)

        picked = valid_idx[torch.randperm(n_valid, device=valid_mask.device, generator=generator)[:m]]
        contact_mask[picked] = True
        return contact_mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        rng = random.Random(self.seed + int(idx))
        robot_name = rng.choice(self.robot_names)
        model = self.robot_models[robot_name]
        generator = torch.Generator(device=model.device.type)
        generator.manual_seed(self.seed + int(idx))

        q_zero = torch.zeros(model.dof, dtype=torch.float32, device=model.device)
        q_rand = self._sample_q(model, generator=generator)

        points_zero, normals_zero = model.get_box_face_points_normals(q=q_zero)
        points_rand, normals_rand = model.get_box_face_points_normals(q=q_rand)

        zero_pairs, valid_zero, chosen_faces = self._pick_one_face_per_link(
            points_zero,
            normals_zero,
            generator=generator,
        )
        rand_pairs = torch.zeros_like(zero_pairs)
        valid_rand = torch.zeros_like(valid_zero)

        for i in range(zero_pairs.shape[0]):
            face_id = int(chosen_faces[i].item())
            if face_id < 0:
                continue
            p = points_rand[i, face_id]
            n = normals_rand[i, face_id]
            ok = torch.isfinite(p).all() and torch.isfinite(n).all()
            if ok:
                rand_pairs[i, :3] = p
                rand_pairs[i, 3:] = n
                valid_rand[i] = True

        pair_valid_mask = valid_zero & valid_rand
        contact_mask = self._sample_contact_mask(
            pair_valid_mask,
            rng=rng,
            generator=generator,
        )

        return {
            "robot_name": robot_name,
            "q_zero": q_zero.detach().cpu(),
            "q_rand": q_rand.detach().cpu(),
            "zero_pairs": zero_pairs.detach().cpu(),          # (N, 6)
            "rand_pairs": rand_pairs.detach().cpu(),          # (N, 6)
            "pair_valid_mask": pair_valid_mask.detach().cpu(),  # (N,)
            "contact_mask": contact_mask.detach().cpu(),      # (N,)
        }


def make_nfk_collate_fn():
    def collate_fn(batch: List[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | List[str]]:
        masked_zero: List[torch.Tensor] = []
        masked_rand: List[torch.Tensor] = []
        robot_names: List[str] = []
        q_zeros: List[torch.Tensor] = []
        q_rands: List[torch.Tensor] = []

        for item in batch:
            zero_pairs = item["zero_pairs"]
            rand_pairs = item["rand_pairs"]
            contact_mask = item["contact_mask"]
            pair_valid_mask = item["pair_valid_mask"]
            active = contact_mask & pair_valid_mask
            masked_zero.append(zero_pairs[active])
            masked_rand.append(rand_pairs[active])
            robot_names.append(item["robot_name"])
            q_zeros.append(item["q_zero"])
            q_rands.append(item["q_rand"])

        max_m = max((x.shape[0] for x in masked_zero), default=0)
        bsz = len(batch)
        zero_padded = torch.zeros((bsz, max_m, 6), dtype=torch.float32)
        rand_padded = torch.zeros((bsz, max_m, 6), dtype=torch.float32)
        mask_m = torch.zeros((bsz, max_m), dtype=torch.bool)

        for i in range(bsz):
            m_i = masked_zero[i].shape[0]
            if m_i == 0:
                continue
            zero_padded[i, :m_i] = masked_zero[i]
            rand_padded[i, :m_i] = masked_rand[i]
            mask_m[i, :m_i] = True

        return {
            "zero_pairs_masked": zero_padded,  # (B, M_max, 6)
            "rand_pairs_masked": rand_padded,  # (B, M_max, 6)
            "mask_m": mask_m,                  # (B, M_max)
            "robot_names": robot_names,
            "q_zero_list": q_zeros,
            "q_rand_list": q_rands,
        }

    return collate_fn


def get_nfk_dataloader(
    *,
    assets_dir: str | Path,
    robot_list_key: str,
    robot_list_file: str | Path | None,
    dataset_size: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    seed: int = 0,
    device: str = "cpu",
    link_num_points: int = 512,
    box_fit_override: bool = False,
) -> torch.utils.data.DataLoader:
    dataset = NFKBoxFaceDataset(
        assets_dir=assets_dir,
        robot_list_key=robot_list_key,
        robot_list_file=robot_list_file,
        dataset_size=dataset_size,
        seed=seed,
        device=device,
        link_num_points=link_num_points,
        box_fit_override=box_fit_override,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=make_nfk_collate_fn(),
    )
