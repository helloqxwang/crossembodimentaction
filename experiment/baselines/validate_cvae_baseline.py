from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from experiment.baselines.datasets import load_cmap_metadata, load_split_objects  # noqa: E402
from experiment.baselines.io import (  # noqa: E402
    create_model_from_ckpt,
    ensure_parent_dir,
    load_ckpt,
    load_dataset_meta,
)


@dataclass
class LoadedModel:
    robot_name: str
    model: torch.nn.Module
    meta: dict[str, Any]
    ckpt_path: str
    dataset_meta_path: str
    model_kwargs_used: dict[str, Any]
    model_source: str


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
    pair_to_qs: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
    for q, object_name, robot_name in load_cmap_metadata(dro_root):
        if robot_name not in selected_robots:
            continue
        if object_name not in split_objects:
            continue
        pair_to_qs[(robot_name, object_name)].append(q.float())
    return pair_to_qs


def _to_python_float(v: Any) -> float:
    if isinstance(v, torch.Tensor):
        return float(v.item())
    return float(v)


def _chamfer_distance(pc_a: torch.Tensor, pc_b: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(pc_a, pc_b, p=2)
    return d.min(dim=1)[0].mean() + d.min(dim=0)[0].mean()


def _load_cross_model(cfg: DictConfig, device: torch.device) -> tuple[LoadedModel, list[str]]:
    ckpt_path = to_absolute_path(str(cfg.policy_paths.cross.ckpt_path))
    meta_path = (
        to_absolute_path(str(cfg.policy_paths.cross.dataset_meta_path))
        if cfg.policy_paths.cross.dataset_meta_path is not None
        else os.path.join(os.path.dirname(ckpt_path), "dataset_meta.pt")
    )
    meta = load_dataset_meta(meta_path)
    ckpt = load_ckpt(ckpt_path, device=device)
    model, used_model_kwargs, model_source = create_model_from_ckpt(
        ckpt=ckpt,
        cfg_arch=None,
        meta=meta,
        device=device,
    )
    all_robot_names = list(meta["all_robot_names"])
    return (
        LoadedModel(
            robot_name="__shared__",
            model=model,
            meta=meta,
            ckpt_path=ckpt_path,
            dataset_meta_path=meta_path,
            model_kwargs_used=used_model_kwargs,
            model_source=model_source,
        ),
        all_robot_names,
    )


def _discover_single_robot_names(checkpoint_root: str, checkpoint_name: str) -> list[str]:
    root = Path(checkpoint_root)
    out = []
    if not root.exists():
        raise FileNotFoundError(f"single checkpoint_root not found: {checkpoint_root}")
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / checkpoint_name).is_file():
            out.append(child.name)
    if not out:
        raise RuntimeError(f"No per-robot checkpoints found under {checkpoint_root}")
    return out


def _load_single_models(cfg: DictConfig, device: torch.device, selected_robots: list[str] | None) -> dict[str, LoadedModel]:
    checkpoint_root = to_absolute_path(str(cfg.policy_paths.single.checkpoint_root))
    checkpoint_name = str(cfg.policy_paths.single.checkpoint_name)
    dataset_meta_name = str(cfg.policy_paths.single.dataset_meta_name)
    robot_names = selected_robots or _discover_single_robot_names(checkpoint_root, checkpoint_name)

    out: dict[str, LoadedModel] = {}
    for robot_name in robot_names:
        ckpt_path = os.path.join(checkpoint_root, robot_name, checkpoint_name)
        meta_path = os.path.join(checkpoint_root, robot_name, dataset_meta_name)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found for robot '{robot_name}': {ckpt_path}")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"dataset_meta not found for robot '{robot_name}': {meta_path}")
        meta = load_dataset_meta(meta_path)
        if str(meta["baseline_type"]) != "single_embodiment":
            raise ValueError(f"Expected single_embodiment dataset_meta at {meta_path}")
        ckpt = load_ckpt(ckpt_path, device=device)
        model, used_model_kwargs, model_source = create_model_from_ckpt(
            ckpt=ckpt,
            cfg_arch=None,
            meta=meta,
            device=device,
        )
        out[robot_name] = LoadedModel(
            robot_name=robot_name,
            model=model,
            meta=meta,
            ckpt_path=ckpt_path,
            dataset_meta_path=meta_path,
            model_kwargs_used=used_model_kwargs,
            model_source=model_source,
        )
    return out


def _local_eval_tensors(
    *,
    bundle: LoadedModel,
    robot_name: str,
    q_gt: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, int]:
    baseline_type = str(bundle.meta["baseline_type"])
    dof = int(bundle.meta["robot_dofs"][robot_name])
    if q_gt is None:
        return None, None, None, dof
    q_gt = q_gt.to(device)
    if baseline_type == "cross_embodiment":
        action_dim = int(bundle.meta["action_dim"])
        q_gt_padded = torch.zeros((1, action_dim), dtype=torch.float32, device=device)
        action_mask = torch.zeros((1, action_dim), dtype=torch.bool, device=device)
        q_gt_padded[:, :dof] = q_gt
        action_mask[:, :dof] = True
        return q_gt_padded, action_mask, q_gt, dof
    return q_gt.unsqueeze(0), torch.ones((1, dof), dtype=torch.bool, device=device), q_gt, dof


def _predict_action(
    *,
    bundle: LoadedModel,
    object_pc: torch.Tensor,
    q_gt_model: torch.Tensor | None,
    robot_name: str,
    pred_mode: str,
    recon_use_mean_latent: bool,
) -> torch.Tensor:
    meta = bundle.meta
    model = bundle.model
    if str(meta["baseline_type"]) == "cross_embodiment":
        emb_idx = torch.tensor([int(meta["robot_to_idx"][robot_name])], dtype=torch.long, device=object_pc.device)
    else:
        emb_idx = None

    if pred_mode == "recon" and q_gt_model is not None:
        if recon_use_mean_latent:
            mu, _ = model.encode(object_pc, q_gt_model, emb_idx)
            return model.decode(object_pc, emb_idx, mu)
        recon, _mu, _logvar, _z = model(object_pc, q_gt_model, emb_idx)
        return recon
    return model.sample(object_pc, emb_idx)


@hydra.main(version_base="1.2", config_path="../conf/baselines", config_name="validate_cvae_baseline")
def main(cfg: DictConfig) -> None:
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    _set_seed(int(cfg.seed))
    rng = random.Random(int(cfg.seed))
    device = _prepare_device(str(cfg.validation.device))

    dro_root = to_absolute_path(str(cfg.dataset.dro_root))
    split = str(cfg.dataset.split)
    baseline_type = str(cfg.policy.type)
    selected_robots_cfg = None if cfg.validation.robot_names is None else [str(x) for x in cfg.validation.robot_names]

    if baseline_type == "cross_embodiment":
        shared_bundle, all_robot_names = _load_cross_model(cfg, device)
        selected_robots = selected_robots_cfg or list(all_robot_names)
        unknown = sorted(set(selected_robots) - set(all_robot_names))
        if unknown:
            raise ValueError(f"Unknown robots in validation.robot_names: {unknown}")
        loaded_models = {"__shared__": shared_bundle}
    elif baseline_type == "single_embodiment":
        loaded_models = _load_single_models(cfg, device, selected_robots_cfg)
        selected_robots = selected_robots_cfg or sorted(loaded_models.keys())
        unknown = sorted(set(selected_robots) - set(loaded_models.keys()))
        if unknown:
            raise ValueError(f"Missing single-baseline checkpoints for robots: {unknown}")
    else:
        raise ValueError(f"Unsupported policy.type: {baseline_type}")

    object_names = sorted(load_split_objects(dro_root, split=split))
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
    for key in pair_to_qs:
        rng.shuffle(pair_to_qs[key])

    pred_mode = str(cfg.validation.pred_mode).lower()
    if pred_mode not in {"sample", "recon"}:
        raise ValueError(f"Unsupported pred_mode: {pred_mode}")
    test_rounds = int(cfg.validation.test_rounds)
    if test_rounds <= 0:
        raise ValueError("validation.test_rounds must be >= 1")

    object_pose6 = torch.tensor(cfg.validation.object_pose6, dtype=torch.float32)
    only_pairs_with_gt = bool(cfg.validation.only_pairs_with_gt)
    compute_chamfer = bool(cfg.validation.compute_chamfer)

    sys.path.append(dro_root)
    from utils.hand_model import create_hand_model  # type: ignore

    hand_cache: dict[str, Any] = {}

    def _get_hand(robot_name: str):
        if robot_name not in hand_cache:
            hand_cache[robot_name] = create_hand_model(robot_name, torch.device("cpu"))
        return hand_cache[robot_name]

    samples: list[dict[str, Any]] = []
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
    pbar = tqdm(total=progress_total, desc="validate_cvae_rounds")

    with torch.no_grad():
        for round_idx in range(test_rounds):
            for object_name in object_names:
                object_pc = _sample_object_pc(object_pc_cache[object_name], int(cfg.dataset.num_points))
                object_pc = object_pc.unsqueeze(0).to(device)
                for robot_name in selected_robots:
                    bundle = loaded_models["__shared__"] if baseline_type == "cross_embodiment" else loaded_models[robot_name]
                    q_list = pair_to_qs.get((robot_name, object_name), [])
                    has_data = round_idx < len(q_list)
                    if (not has_data) and only_pairs_with_gt:
                        skipped_no_data_pairs += 1
                        pbar.update(1)
                        continue

                    q_gt_model = None
                    action_mask = None
                    q_gt_local = None
                    q_gt_padded_cpu = None
                    q_gt_cpu = None
                    q_gt = q_list[round_idx] if has_data else None
                    q_gt_model, action_mask, q_gt_local, dof = _local_eval_tensors(
                        bundle=bundle,
                        robot_name=robot_name,
                        q_gt=q_gt,
                        device=device,
                    )

                    q_pred_model = _predict_action(
                        bundle=bundle,
                        object_pc=object_pc,
                        q_gt_model=q_gt_model,
                        robot_name=robot_name,
                        pred_mode=pred_mode,
                        recon_use_mean_latent=bool(cfg.validation.recon_use_mean_latent),
                    )
                    q_pred_local = q_pred_model[0, :dof].detach().cpu()
                    if q_gt_local is not None:
                        q_gt_cpu = q_gt_local.detach().cpu()
                    if q_gt_model is not None and q_gt_model.size(1) != dof:
                        q_gt_padded_cpu = q_gt_model[0].detach().cpu()

                    sample: dict[str, Any] = {
                        "sample_id": len(samples),
                        "round_idx": round_idx,
                        "robot_name": robot_name,
                        "eval_robot_name": robot_name,
                        "object_name": object_name,
                        "model_robot_name": None if baseline_type == "cross_embodiment" else robot_name,
                        "baseline_type": baseline_type,
                        "trained_robot_names": list(bundle.meta["trained_robot_names"]),
                        "conditioning_mode": str(bundle.meta["conditioning_mode"]),
                        "dof": int(dof),
                        "pair_status": "has-data" if has_data else "no-data",
                        "source": "pred" if has_data else "no-data",
                        "pred_mode": pred_mode,
                        "q_pred": q_pred_local,
                        "object_pose6": object_pose6.clone(),
                    }
                    if q_pred_model.size(1) != dof:
                        sample["q_pred_padded"] = q_pred_model[0].detach().cpu()
                    if baseline_type == "cross_embodiment":
                        sample["embodiment_idx"] = int(bundle.meta["robot_to_idx"][robot_name])

                    if has_data:
                        assert q_gt_model is not None and action_mask is not None and q_gt_cpu is not None
                        mse_full = torch.mean((q_pred_model - q_gt_model) ** 2)
                        mse_valid = _masked_mse(q_pred_model, q_gt_model, action_mask)
                        joint_l1 = torch.mean(torch.abs(q_pred_model[0, :dof] - q_gt_model[0, :dof]))
                        sample["mse_full"] = _to_python_float(mse_full)
                        sample["mse_valid"] = _to_python_float(mse_valid)
                        sample["joint_l1"] = _to_python_float(joint_l1)
                        if bool(cfg.validation.save_gt):
                            sample["q_gt"] = q_gt_cpu
                            if q_gt_padded_cpu is not None:
                                sample["q_gt_padded"] = q_gt_padded_cpu
                        metrics_data_mse_full.append(sample["mse_full"])
                        metrics_data_mse_valid.append(sample["mse_valid"])
                        metrics_data_joint_l1.append(sample["joint_l1"])

                        if compute_chamfer:
                            hand = _get_hand(robot_name)
                            pred_pc = hand.get_transformed_links_pc(q_pred_local)[:, :3].float().cpu()
                            gt_pc = hand.get_transformed_links_pc(q_gt_cpu)[:, :3].float().cpu()
                            chamfer = _chamfer_distance(pred_pc, gt_pc)
                            sample["chamfer"] = _to_python_float(chamfer)
                            metrics_data_chamfer.append(sample["chamfer"])
                        else:
                            sample["chamfer"] = None

                        per_robot_pair_data[robot_name] += 1
                        per_object_pair_data[object_name] += 1
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

    if baseline_type == "cross_embodiment":
        model_meta = loaded_models["__shared__"].meta
        model_ckpt_info: Any = {
            "ckpt_path": loaded_models["__shared__"].ckpt_path,
            "dataset_meta_path": loaded_models["__shared__"].dataset_meta_path,
            "model_source": loaded_models["__shared__"].model_source,
            "model_kwargs_used": loaded_models["__shared__"].model_kwargs_used,
        }
    else:
        model_meta = {"robot_names": selected_robots}
        model_ckpt_info = {
            robot_name: {
                "ckpt_path": bundle.ckpt_path,
                "dataset_meta_path": bundle.dataset_meta_path,
                "model_source": bundle.model_source,
                "model_kwargs_used": bundle.model_kwargs_used,
            }
            for robot_name, bundle in loaded_models.items()
        }

    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "baseline_type": baseline_type,
            "dro_root": dro_root,
            "split": split,
            "num_points": int(cfg.dataset.num_points),
            "trained_robot_names": model_meta.get("trained_robot_names", []),
            "all_robot_names": model_meta.get("all_robot_names", model_meta.get("robot_names", selected_robots)),
            "selected_robots": selected_robots,
            "selected_objects": object_names,
            "pred_mode": pred_mode,
            "test_rounds": test_rounds,
            "model_info": model_ckpt_info,
        },
        "summary": {
            "num_samples": len(samples),
            "num_pairs_per_round": total_pairs,
            "num_data_pairs": int(sum(1 for s in samples if s["pair_status"] == "has-data")),
            "num_no_data_pairs": int(sum(1 for s in samples if s["pair_status"] == "no-data")),
            "num_skipped_no_data_pairs": int(skipped_no_data_pairs),
            "mean_mse_full_data_only": float(np.mean(metrics_data_mse_full)) if metrics_data_mse_full else None,
            "mean_mse_valid_data_only": float(np.mean(metrics_data_mse_valid)) if metrics_data_mse_valid else None,
            "mean_joint_l1_data_only": float(np.mean(metrics_data_joint_l1)) if metrics_data_joint_l1 else None,
            "mean_chamfer_data_only": float(np.mean(metrics_data_chamfer)) if metrics_data_chamfer else None,
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
    print(f"baseline_type: {baseline_type}")
    print(f"num_samples: {payload['summary']['num_samples']}")
    print(f"num_data_pairs: {payload['summary']['num_data_pairs']}")
    print(f"num_no_data_pairs: {payload['summary']['num_no_data_pairs']}")
    print(f"mean_joint_l1_data_only: {payload['summary']['mean_joint_l1_data_only']}")
    print(f"mean_chamfer_data_only: {payload['summary']['mean_chamfer_data_only']}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    main()
