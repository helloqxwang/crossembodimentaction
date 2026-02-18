from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from experiment.cross_cvae_utils import (
    create_model_from_ckpt,
    ensure_parent_dir,
    load_ckpt,
    load_dataset_meta,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prepare_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sq = (pred - target).pow(2)
    denom = mask.float().sum().clamp_min(1.0)
    return (sq * mask.float()).sum() / denom


def _load_split_objects(dro_root: str, split: str) -> list[str]:
    split_path = os.path.join(dro_root, "data/CMapDataset_filtered/split_train_validate_objects.json")
    split_json = json.load(open(split_path, "r", encoding="utf-8"))
    if split not in split_json:
        raise KeyError(f"Split '{split}' not found in {split_path}")
    return list(split_json[split])


def _load_object_pc_cache(dro_root: str, object_names: list[str]) -> dict[str, torch.Tensor]:
    cache: dict[str, torch.Tensor] = {}
    for object_name in object_names:
        dataset_name, mesh_name = object_name.split("+")
        pc_path = os.path.join(dro_root, "data/PointCloud/object", dataset_name, f"{mesh_name}.pt")
        if not os.path.exists(pc_path):
            raise FileNotFoundError(f"Object point cloud file not found: {pc_path}")
        pc = torch.load(pc_path, map_location="cpu").float()
        if pc.ndim != 2 or pc.shape[1] < 3:
            raise ValueError(f"Invalid object point cloud shape at {pc_path}: {tuple(pc.shape)}")
        cache[object_name] = pc[:, :3]
    return cache


def _sample_object_pc(pc_full: torch.Tensor, num_points: int) -> torch.Tensor:
    if pc_full.size(0) >= num_points:
        idx = torch.randperm(pc_full.size(0))[:num_points]
    else:
        idx = torch.randint(0, pc_full.size(0), (num_points,))
    return pc_full[idx].clone()


def _load_pair_samples(
    dro_root: str,
    *,
    split_objects: set[str],
    selected_robots: set[str],
) -> dict[tuple[str, str], list[torch.Tensor]]:
    dataset_path = os.path.join(dro_root, "data/CMapDataset_filtered/cmap_dataset.pt")
    metadata = torch.load(dataset_path, map_location="cpu")["metadata"]
    pair_to_qs: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
    for q, object_name, robot_name in metadata:
        if robot_name not in selected_robots:
            continue
        if object_name not in split_objects:
            continue
        q_tensor = q if torch.is_tensor(q) else torch.tensor(q, dtype=torch.float32)
        pair_to_qs[(robot_name, object_name)].append(q_tensor.float())
    return pair_to_qs


def _to_python_float(v: Any) -> float:
    if isinstance(v, torch.Tensor):
        return float(v.item())
    return float(v)


def _chamfer_distance(pc_a: torch.Tensor, pc_b: torch.Tensor) -> torch.Tensor:
    # pc_a: (Na, 3), pc_b: (Nb, 3)
    d = torch.cdist(pc_a, pc_b, p=2)
    return d.min(dim=1)[0].mean() + d.min(dim=0)[0].mean()


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_validate_cross_cvae")
def main(cfg: DictConfig) -> None:
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    _set_seed(int(cfg.seed))
    rng = random.Random(int(cfg.seed))
    device = _prepare_device(str(cfg.validation.device))

    ckpt_path = to_absolute_path(str(cfg.model.ckpt_path))
    meta_path = (
        to_absolute_path(str(cfg.model.dataset_meta_path))
        if cfg.model.dataset_meta_path is not None
        else os.path.join(os.path.dirname(ckpt_path), "dataset_meta.pt")
    )
    dro_root = to_absolute_path(str(cfg.dataset.dro_root))
    split = str(cfg.dataset.split)

    meta = load_dataset_meta(meta_path)
    ckpt = load_ckpt(ckpt_path, device=device)
    model, used_model_kwargs, model_source = create_model_from_ckpt(
        ckpt=ckpt,
        cfg_model=cfg.model,
        action_dim=int(meta["action_dim"]),
        num_embodiments=len(meta["robot_names"]),
        device=device,
    )

    model_robot_names = list(meta["robot_names"])
    selected_robots = (
        list(model_robot_names)
        if cfg.validation.robot_names is None
        else [str(x) for x in cfg.validation.robot_names]
    )
    unknown = sorted(set(selected_robots) - set(model_robot_names))
    if unknown:
        raise ValueError(f"Unknown robots in validation.robot_names: {unknown}")

    object_names = _load_split_objects(dro_root, split=split)
    if cfg.validation.object_names is not None:
        object_filter = set(str(x) for x in cfg.validation.object_names)
        object_names = [x for x in object_names if x in object_filter]
    if int(cfg.validation.max_objects) >= 0:
        object_names = object_names[: int(cfg.validation.max_objects)]
    if len(object_names) == 0:
        raise RuntimeError("No objects selected for validation.")

    object_pc_cache = _load_object_pc_cache(dro_root, object_names)
    pair_to_qs = _load_pair_samples(
        dro_root,
        split_objects=set(object_names),
        selected_robots=set(selected_robots),
    )
    # Shuffle once so round-0 uses random sample, round-1 uses another random sample, etc.
    for key in pair_to_qs:
        rng.shuffle(pair_to_qs[key])

    pred_mode = str(cfg.validation.pred_mode).lower()
    if pred_mode not in {"sample", "recon"}:
        raise ValueError(f"Unsupported pred_mode: {pred_mode}")
    test_rounds = int(cfg.validation.test_rounds)
    if test_rounds <= 0:
        raise ValueError("validation.test_rounds must be >= 1")

    object_pose6 = torch.tensor(cfg.validation.object_pose6, dtype=torch.float32)
    robot_to_idx = {str(k): int(v) for k, v in meta["robot_to_idx"].items()}
    robot_dofs = {str(k): int(v) for k, v in meta["robot_dofs"].items()}
    action_dim = int(meta["action_dim"])
    only_pairs_with_gt = bool(cfg.validation.only_pairs_with_gt)
    compute_chamfer = bool(cfg.validation.compute_chamfer)

    # lazy import DRO hand model for geometry-aware losses
    sys.path.append(dro_root)
    from utils.hand_model import create_hand_model  # type: ignore

    hand_cache: dict[str, Any] = {}

    def _get_hand(robot_name: str):
        if robot_name not in hand_cache:
            hand_cache[robot_name] = create_hand_model(robot_name, torch.device("cpu"))
        return hand_cache[robot_name]

    samples: list[Dict[str, Any]] = []
    metrics_data_mse_full: list[float] = []
    metrics_data_mse_valid: list[float] = []
    metrics_data_joint_l1: list[float] = []
    metrics_data_chamfer: list[float] = []
    per_robot_pair_data = defaultdict(int)
    per_robot_pair_no_data = defaultdict(int)
    per_object_pair_data = defaultdict(int)
    per_object_pair_no_data = defaultdict(int)
    skipped_no_data_pairs = 0

    total_pairs = len(selected_robots) * len(object_names)
    progress_total = total_pairs * test_rounds
    pbar = tqdm(total=progress_total, desc="validate_cross_cvae_rounds")

    with torch.no_grad():
        for round_idx in range(test_rounds):
            for object_name in object_names:
                object_pc = _sample_object_pc(object_pc_cache[object_name], int(cfg.dataset.num_points))
                object_pc = object_pc.unsqueeze(0).to(device)  # (1, P, 3)
                for robot_name in selected_robots:
                    q_list = pair_to_qs.get((robot_name, object_name), [])
                    has_data = round_idx < len(q_list)
                    if (not has_data) and only_pairs_with_gt:
                        skipped_no_data_pairs += 1
                        pbar.update(1)
                        continue

                    dof = int(robot_dofs[robot_name])
                    emb_idx = torch.tensor([robot_to_idx[robot_name]], dtype=torch.long, device=device)

                    q_gt_padded = torch.zeros((1, action_dim), dtype=torch.float32, device=device)
                    action_mask = torch.zeros((1, action_dim), dtype=torch.bool, device=device)
                    if has_data:
                        q_gt = q_list[round_idx].to(device)
                        local_dof = int(q_gt.numel())
                        q_gt_padded[:, :local_dof] = q_gt
                        action_mask[:, :local_dof] = True
                    else:
                        local_dof = dof

                    # no-data pairs are still tested using sampling mode.
                    if pred_mode == "recon" and has_data:
                        if bool(cfg.validation.recon_use_mean_latent):
                            mu, _ = model.encode(object_pc, q_gt_padded, emb_idx)
                            q_pred_padded = model.decode(object_pc, emb_idx, mu)
                        else:
                            q_pred_padded, _, _, _ = model(object_pc, q_gt_padded, emb_idx)
                    else:
                        q_pred_padded = model.sample(object_pc, emb_idx)

                    sample: Dict[str, Any] = {
                        "sample_id": len(samples),
                        "round_idx": round_idx,
                        "robot_name": robot_name,
                        "object_name": object_name,
                        "embodiment_idx": int(emb_idx.item()),
                        "dof": int(local_dof),
                        "pair_status": "has-data" if has_data else "no-data",
                        "source": "pred" if has_data else "no-data",
                        "pred_mode": pred_mode,
                        "q_pred_padded": q_pred_padded[0].detach().cpu(),
                        "q_pred": q_pred_padded[0, :local_dof].detach().cpu(),
                        "object_pose6": object_pose6.clone(),
                    }

                    if has_data:
                        mse_full = torch.mean((q_pred_padded - q_gt_padded) ** 2)
                        mse_valid = _masked_mse(q_pred_padded, q_gt_padded, action_mask)
                        joint_l1 = torch.mean(torch.abs(q_pred_padded[:, :local_dof] - q_gt_padded[:, :local_dof]))
                        sample["mse_full"] = _to_python_float(mse_full)
                        sample["mse_valid"] = _to_python_float(mse_valid)
                        sample["joint_l1"] = _to_python_float(joint_l1)
                        metrics_data_mse_full.append(sample["mse_full"])
                        metrics_data_mse_valid.append(sample["mse_valid"])
                        metrics_data_joint_l1.append(sample["joint_l1"])

                        if compute_chamfer:
                            hand = _get_hand(robot_name)
                            q_pred = q_pred_padded[0, :local_dof].detach().cpu()
                            q_gt_eval = q_gt_padded[0, :local_dof].detach().cpu()
                            pred_pc = hand.get_transformed_links_pc(q_pred)[:, :3].float().cpu()
                            gt_pc = hand.get_transformed_links_pc(q_gt_eval)[:, :3].float().cpu()
                            chamfer = _chamfer_distance(pred_pc, gt_pc)
                            sample["chamfer"] = _to_python_float(chamfer)
                            metrics_data_chamfer.append(sample["chamfer"])
                        else:
                            sample["chamfer"] = None

                        per_robot_pair_data[robot_name] += 1
                        per_object_pair_data[object_name] += 1

                        if bool(cfg.validation.save_gt):
                            sample["q_gt_padded"] = q_gt_padded[0].detach().cpu()
                            sample["q_gt"] = q_gt_padded[0, :local_dof].detach().cpu()
                    else:
                        sample["mse_full"] = None
                        sample["mse_valid"] = None
                        sample["joint_l1"] = None
                        sample["chamfer"] = None
                        per_robot_pair_no_data[robot_name] += 1
                        per_object_pair_no_data[object_name] += 1

                    samples.append(sample)
                    pbar.update(1)
    pbar.close()

    if len(samples) == 0:
        raise RuntimeError("No samples generated. Check config.")

    save_path = to_absolute_path(str(cfg.output.save_path))
    ensure_parent_dir(save_path)
    save_json_path = os.path.splitext(save_path)[0] + ".json"
    ensure_parent_dir(save_json_path)

    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "ckpt_path": ckpt_path,
            "dataset_meta_path": meta_path,
            "dro_root": dro_root,
            "split": split,
            "num_points": int(cfg.dataset.num_points),
            "model_robot_names": model_robot_names,
            "selected_robots": selected_robots,
            "selected_objects": object_names,
            "robot_to_idx": robot_to_idx,
            "robot_dofs": robot_dofs,
            "action_dim": action_dim,
            "pred_mode": pred_mode,
            "test_rounds": test_rounds,
            "model_source": model_source,
            "model_kwargs_used": used_model_kwargs,
        },
        "summary": {
            "num_samples": len(samples),
            "num_pairs_per_round": total_pairs,
            "num_data_pairs": int(sum(1 for s in samples if s["pair_status"] == "has-data")),
            "num_no_data_pairs": int(sum(1 for s in samples if s["pair_status"] == "no-data")),
            "num_skipped_no_data_pairs": int(skipped_no_data_pairs),
            "mean_mse_full_data_only": float(np.mean(metrics_data_mse_full))
            if metrics_data_mse_full
            else None,
            "mean_mse_valid_data_only": float(np.mean(metrics_data_mse_valid))
            if metrics_data_mse_valid
            else None,
            "mean_joint_l1_data_only": float(np.mean(metrics_data_joint_l1))
            if metrics_data_joint_l1
            else None,
            "mean_chamfer_data_only": float(np.mean(metrics_data_chamfer))
            if metrics_data_chamfer
            else None,
            "per_robot_data_count": {str(k): int(v) for k, v in per_robot_pair_data.items()},
            "per_robot_no_data_count": {str(k): int(v) for k, v in per_robot_pair_no_data.items()},
            "per_object_data_count": {str(k): int(v) for k, v in per_object_pair_data.items()},
            "per_object_no_data_count": {str(k): int(v) for k, v in per_object_pair_no_data.items()},
        },
        "samples": samples,
    }
    torch.save(payload, save_path)

    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": payload["meta"], "summary": payload["summary"]}, f, indent=2)

    print("\n=== Validation Saved ===")
    print(f"save_path: {save_path}")
    print(f"summary_path: {save_json_path}")
    print(f"num_samples: {payload['summary']['num_samples']}")
    print(f"num_data_pairs: {payload['summary']['num_data_pairs']}")
    print(f"num_no_data_pairs: {payload['summary']['num_no_data_pairs']}")
    print(f"mean_joint_l1_data_only: {payload['summary']['mean_joint_l1_data_only']}")
    print(f"mean_chamfer_data_only: {payload['summary']['mean_chamfer_data_only']}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    main()
