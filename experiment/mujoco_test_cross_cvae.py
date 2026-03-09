from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from experiment.cross_cvae_utils import ensure_parent_dir, get_robot_asset_paths
from experiment.sim.assets import prepare_hand_path_for_mujoco
from experiment.sim.grasp_validation import resolve_disturbance_directions, run_static_grasp_validation_cpu
from experiment.sim.mjwarp_batch import evaluate_static_grasp_batch_mjwarp
from experiment.sim.sample_io import load_samples_from_config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _classify_error(err_msg: str) -> str:
    msg = err_msg.lower()
    if "error opening file" in msg or "no such file or directory" in msg:
        return "asset_missing"
    if "dof does not match hand qpos dof" in msg:
        return "dof_mismatch"
    if "cuda" in msg or "warp" in msg:
        return "backend_error"
    return "runtime_error"


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record.get("error") is None]
    by_hand: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_robot_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid:
        by_hand[str(record["robot_name"])].append(record)
        by_robot_source[f"{record['robot_name']}|{record['source']}"].append(record)

    def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {
                "count": 0,
                "in_hand_rate": 0.0,
                "robust_in_hand_rate": float("nan"),
                "final_contact_rate": 0.0,
                "unstable_rate": 0.0,
                "pos_err_final_mean": float("nan"),
                "pos_err_final_max": float("nan"),
                "rot_err_final_mean": float("nan"),
                "rot_err_final_max": float("nan"),
                "final_hand_obj_distance_mean": float("nan"),
            }
        in_hand = [bool(row["summary"]["final_in_hand"]) for row in rows]
        final_contact = [bool(row["summary"]["final_contact"]) for row in rows]
        unstable = [bool(row.get("unstable", False)) for row in rows]
        robust = [
            bool(row.get("disturbance", {}).get("all_directions_in_hand"))
            for row in rows
            if isinstance(row.get("disturbance"), dict)
        ]
        pos_final = [float(row["summary"]["pos_err_final"]) for row in rows]
        rot_final = [float(row["summary"]["rot_err_final"]) for row in rows]
        hand_obj_dist = [float(row["summary"]["final_hand_obj_distance"]) for row in rows]
        return {
            "count": len(rows),
            "in_hand_rate": float(np.mean(in_hand)),
            "robust_in_hand_rate": float(np.mean(robust)) if robust else float("nan"),
            "final_contact_rate": float(np.mean(final_contact)),
            "unstable_rate": float(np.mean(unstable)),
            "pos_err_final_mean": float(np.mean(pos_final)),
            "pos_err_final_max": float(np.max(pos_final)),
            "rot_err_final_mean": float(np.mean(rot_final)),
            "rot_err_final_max": float(np.max(rot_final)),
            "final_hand_obj_distance_mean": float(np.mean(hand_obj_dist)),
        }

    aggregate = {
        "count_total": len(records),
        "count_valid": len(valid),
        "count_error": len(records) - len(valid),
        "in_hand_rate": float(np.mean([bool(row["summary"]["final_in_hand"]) for row in valid])) if valid else 0.0,
        "robust_in_hand_rate": (
            float(
                np.mean(
                    [
                        bool(row.get("disturbance", {}).get("all_directions_in_hand"))
                        for row in valid
                        if isinstance(row.get("disturbance"), dict)
                    ]
                )
            )
            if any(isinstance(row.get("disturbance"), dict) for row in valid)
            else float("nan")
        ),
        "final_contact_rate": float(np.mean([bool(row["summary"]["final_contact"]) for row in valid])) if valid else 0.0,
        "unstable_rate": float(np.mean([bool(row.get("unstable", False)) for row in valid])) if valid else 0.0,
        "pos_err_final_mean": (
            float(np.mean([float(row["summary"]["pos_err_final"]) for row in valid])) if valid else float("nan")
        ),
        "pos_err_final_max": (
            float(np.max([float(row["summary"]["pos_err_final"]) for row in valid])) if valid else float("nan")
        ),
        "rot_err_final_mean": (
            float(np.mean([float(row["summary"]["rot_err_final"]) for row in valid])) if valid else float("nan")
        ),
        "rot_err_final_max": (
            float(np.max([float(row["summary"]["rot_err_final"]) for row in valid])) if valid else float("nan")
        ),
        "final_hand_obj_distance_mean": (
            float(np.mean([float(row["summary"]["final_hand_obj_distance"]) for row in valid]))
            if valid
            else float("nan")
        ),
        "per_hand": {hand: _stats(rows) for hand, rows in sorted(by_hand.items())},
        "per_robot_source": {key: _stats(rows) for key, rows in sorted(by_robot_source.items())},
    }
    return aggregate


def _group_samples_by_pair(samples: list[dict[str, Any]]) -> list[tuple[tuple[str, str], list[dict[str, Any]]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[(str(sample["robot_name"]), str(sample["object_name"]))].append(sample)
    return list(grouped.items())


def _build_success_record(sample: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": int(sample["index"]),
        "robot_name": str(sample["robot_name"]),
        "object_name": str(sample["object_name"]),
        "source": str(sample["source"]),
        "dof": int(sample["dof"]),
        "ordered_idx": int(sample["ordered_idx"]),
        "error": None,
        "error_type": None,
        "summary": result["summary"],
        "unstable": bool(result.get("unstable", False)),
        "unstable_step": result.get("unstable_step"),
        "disturbance": result.get("disturbance"),
        "meta": dict(sample.get("meta", {})),
    }


def _build_error_record(sample: dict[str, Any], error: Exception) -> dict[str, Any]:
    error_message = str(error)
    return {
        "index": int(sample["index"]),
        "robot_name": str(sample["robot_name"]),
        "object_name": str(sample["object_name"]),
        "source": str(sample["source"]),
        "dof": int(sample["dof"]),
        "ordered_idx": int(sample["ordered_idx"]),
        "error": error_message,
        "error_type": _classify_error(error_message),
        "summary": None,
        "unstable": None,
        "unstable_step": None,
        "disturbance": None,
        "meta": dict(sample.get("meta", {})),
    }


def _run_cpu_group(
    samples: list[dict[str, Any]],
    *,
    robot_name: str,
    hand_path: str,
    object_mesh_path: str,
    cfg: DictConfig,
    disturbance_directions: list[tuple[str, np.ndarray]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    object_pose6 = np.asarray(cfg.mujoco.object_pose6, dtype=np.float64)
    for sample in samples:
        try:
            result = run_static_grasp_validation_cpu(
                robot_name=robot_name,
                q_target=np.asarray(sample["q_eval"], dtype=np.float64),
                object_pose6=object_pose6,
                hand_path=hand_path,
                object_mesh_path=object_mesh_path,
                sim_dt=float(cfg.mujoco.sim_dt),
                num_steps=int(cfg.mujoco.num_steps),
                steps_per_frame=int(cfg.mujoco.steps_per_frame),
                in_hand_distance_threshold=float(cfg.mujoco.in_hand_distance_threshold),
                hand_kp=float(cfg.mujoco.hand_kp),
                hand_kd=float(cfg.mujoco.hand_kd),
                hand_max_force=float(cfg.mujoco.hand_max_force),
                gravity_enabled=bool(cfg.mujoco.gravity_enabled),
                squeezing_enabled=bool(cfg.mujoco.squeezing.enabled),
                object_density=float(cfg.mujoco.object_density),
                model_nconmax=int(cfg.mujoco.nconmax_per_world),
                model_njmax=int(cfg.mujoco.njmax_per_world),
                disturbance_enabled=bool(cfg.mujoco.disturbance.enabled),
                disturbance_push_steps=int(cfg.mujoco.disturbance.push_steps),
                disturbance_recovery_steps=int(cfg.mujoco.disturbance.recovery_steps),
                disturbance_force_acceleration=float(cfg.mujoco.disturbance.force_acceleration),
                disturbance_force_cap=float(cfg.mujoco.disturbance.force_cap),
                disturbance_directions=disturbance_directions,
                show_rerun=False,
                use_visual_mesh_for_rerun=False,
                save_rerun_path=None,
            )
            records.append(_build_success_record(sample, result))
        except Exception as exc:  # noqa: BLE001
            records.append(_build_error_record(sample, exc))
    return records


@hydra.main(version_base="1.2", config_path="conf", config_name="config_mujoco_test_cross_cvae")
def main(cfg: DictConfig) -> None:
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    _set_seed(int(cfg.seed))
    dro_root = to_absolute_path(str(cfg.paths.dro_root))
    cfg.paths.dro_root = dro_root
    if str(cfg.input.mode).lower() == "generated":
        cfg.input.generated.results_path = to_absolute_path(str(cfg.input.generated.results_path))
    elif cfg.input.cmap_gt.get("data_pt_path") is not None:
        cfg.input.cmap_gt.data_pt_path = to_absolute_path(str(cfg.input.cmap_gt.data_pt_path))
    samples = load_samples_from_config(cfg)
    if not samples:
        raise RuntimeError("No samples to evaluate.")

    for index, sample in enumerate(samples):
        sample["index"] = int(index)

    backend = str(cfg.mujoco.backend).lower()
    device = str(cfg.mujoco.device)
    disturbance_directions = resolve_disturbance_directions(cfg.mujoco.disturbance.directions)
    patch_dir = Path(to_absolute_path(str(cfg.output.urdf_patch_dir))).resolve()
    hand_path_cache: dict[str, str] = {}

    eval_records: list[dict[str, Any]] = []
    pair_groups = _group_samples_by_pair(samples)
    print(
        f"[MuJoCo Test] backend={backend} device={device} "
        f"samples={len(samples)} pair_groups={len(pair_groups)}"
    )
    total_start = time.perf_counter()
    for group_idx, ((robot_name, object_name), rows) in enumerate(pair_groups, start=1):
        group_start = time.perf_counter()
        print(
            f"[Pair {group_idx}/{len(pair_groups)}] "
            f"robot={robot_name} object={object_name} samples={len(rows)}"
        )
        hand_path_orig, object_mesh_path = get_robot_asset_paths(dro_root, robot_name, object_name)
        hand_path = prepare_hand_path_for_mujoco(
            hand_path_orig,
            patch_dir=patch_dir,
            cache=hand_path_cache,
        )
        try:
            if backend == "mjwarp":
                pair_records = evaluate_static_grasp_batch_mjwarp(
                    rows,
                    robot_name=robot_name,
                    hand_path=hand_path,
                    object_mesh_path=object_mesh_path,
                    object_pose6=np.asarray(cfg.mujoco.object_pose6, dtype=np.float64),
                    sim_dt=float(cfg.mujoco.sim_dt),
                    num_steps=int(cfg.mujoco.num_steps),
                    steps_per_frame=int(cfg.mujoco.steps_per_frame),
                    in_hand_distance_threshold=float(cfg.mujoco.in_hand_distance_threshold),
                    hand_kp=float(cfg.mujoco.hand_kp),
                    hand_kd=float(cfg.mujoco.hand_kd),
                    hand_max_force=float(cfg.mujoco.hand_max_force),
                    gravity_enabled=bool(cfg.mujoco.gravity_enabled),
                    squeezing_enabled=bool(cfg.mujoco.squeezing.enabled),
                    object_density=float(cfg.mujoco.object_density),
                    disturbance_enabled=bool(cfg.mujoco.disturbance.enabled),
                    disturbance_push_steps=int(cfg.mujoco.disturbance.push_steps),
                    disturbance_recovery_steps=int(cfg.mujoco.disturbance.recovery_steps),
                    disturbance_force_acceleration=float(cfg.mujoco.disturbance.force_acceleration),
                    disturbance_force_cap=float(cfg.mujoco.disturbance.force_cap),
                    disturbance_directions=disturbance_directions,
                    device=device,
                    max_worlds_per_batch=int(cfg.mujoco.max_worlds_per_batch),
                    nconmax_per_world=int(cfg.mujoco.nconmax_per_world),
                    njmax_per_world=int(cfg.mujoco.njmax_per_world),
                    capture_cuda_graph=bool(cfg.mujoco.capture_cuda_graph),
                )
            elif backend == "cpu":
                pair_records = _run_cpu_group(
                    rows,
                    robot_name=robot_name,
                    hand_path=hand_path,
                    object_mesh_path=object_mesh_path,
                    cfg=cfg,
                    disturbance_directions=disturbance_directions,
                )
            else:
                raise ValueError(f"Unsupported mujoco.backend: {backend}")
            eval_records.extend(pair_records)
        except Exception as exc:  # noqa: BLE001
            for sample in rows:
                eval_records.append(_build_error_record(sample, exc))
        print(
            f"[Pair {group_idx}/{len(pair_groups)}] "
            f"completed in {time.perf_counter() - group_start:.2f}s"
        )

    eval_records.sort(key=lambda record: int(record["index"]))
    aggregate = _aggregate(eval_records)

    save_path = to_absolute_path(str(cfg.output.save_path))
    ensure_parent_dir(save_path)
    json_path = str(Path(save_path).with_suffix(".json"))
    ensure_parent_dir(json_path)
    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": str(cfg.input.mode),
            "backend": backend,
            "device": device,
            "dro_root": dro_root,
                "num_input_samples": len(samples),
                "num_eval_records": len(eval_records),
                "disturbance_enabled": bool(cfg.mujoco.disturbance.enabled),
                "gravity_enabled": bool(cfg.mujoco.gravity_enabled),
                "squeezing_enabled": bool(cfg.mujoco.squeezing.enabled),
            },
        "aggregate": aggregate,
        "records": eval_records,
    }
    torch.save(payload, save_path)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"meta": payload["meta"], "aggregate": aggregate}, handle, indent=2)

    print("\n=== MuJoCo Test Saved ===")
    print(f"save_path: {save_path}")
    print(f"summary_path: {json_path}")
    print(f"backend: {backend}")
    print(f"device: {device}")
    print(f"elapsed_sec: {time.perf_counter() - total_start:.2f}")
    print(f"count_total: {aggregate['count_total']}")
    print(f"count_valid: {aggregate['count_valid']}")
    print(f"in_hand_rate: {aggregate['in_hand_rate']:.4f}")
    if np.isfinite(float(aggregate["robust_in_hand_rate"])):
        print(f"robust_in_hand_rate: {aggregate['robust_in_hand_rate']:.4f}")
    error_counts = defaultdict(int)
    for record in eval_records:
        if record["error"] is not None:
            error_counts[str(record["error_type"])] += 1
    if error_counts:
        print("error_breakdown:")
        for key, value in sorted(error_counts.items()):
            print(f"  {key}: {value}")
    print("per_hand:")
    for hand_name, hand_stats in sorted(aggregate["per_hand"].items()):
        print(
            f"  {hand_name}: count={hand_stats['count']}, "
            f"in_hand_rate={hand_stats['in_hand_rate']:.4f}, "
            f"robust_in_hand_rate={hand_stats['robust_in_hand_rate']:.4f}, "
            f"contact_rate={hand_stats['final_contact_rate']:.4f}, "
            f"unstable_rate={hand_stats['unstable_rate']:.4f}, "
            f"pos_err_mean={hand_stats['pos_err_final_mean']:.6f}, "
            f"rot_err_mean={hand_stats['rot_err_final_mean']:.6f}"
        )


if __name__ == "__main__":
    sys.exit(main())
