from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from experiment.cross_cvae_utils import get_robot_asset_paths
from experiment.sim.assets import prepare_hand_path_for_mujoco
from experiment.sim.config_utils import load_mujoco_test_config
from experiment.sim.grasp_validation import resolve_disturbance_directions, run_static_grasp_validation_cpu
from experiment.sim.sample_io import detect_data_pt_mode, select_sample_from_data_pt


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Play one selected grasp sample live in MuJoCo + rerun.")
    parser.add_argument("--robot-name", type=str, required=True)
    parser.add_argument("--sample-idx", "--ordered-idx", dest="sample_idx", type=int, default=0)
    parser.add_argument("--data-pt-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--q-key", choices=["q_pred", "q_gt"], default=None)
    parser.add_argument("--show-viewer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-rrd-path", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    cfg = load_mujoco_test_config(args.config_path)
    dro_root = str(Path(str(cfg.paths.dro_root)).expanduser().resolve())
    q_key = str(args.q_key or cfg.input.generated.q_key)
    entry = select_sample_from_data_pt(
        data_pt_path=args.data_pt_path,
        dro_root=dro_root,
        robot_name=str(args.robot_name),
        sample_idx=int(args.sample_idx),
        split=str(cfg.input.cmap_gt.split),
        q_key=q_key,
    )
    data_mode = detect_data_pt_mode(args.data_pt_path)

    robot_name = str(entry["robot_name"])
    object_name = str(entry["object_name"])
    hand_path_orig, object_mesh_path = get_robot_asset_paths(dro_root, robot_name, object_name)
    patch_dir = Path(str(cfg.output.urdf_patch_dir)).expanduser().resolve()
    patch_dir.mkdir(parents=True, exist_ok=True)
    hand_path = prepare_hand_path_for_mujoco(hand_path_orig, patch_dir=patch_dir, cache={})

    disturbance_directions = resolve_disturbance_directions(cfg.mujoco.disturbance.directions)
    result = run_static_grasp_validation_cpu(
        robot_name=robot_name,
        q_target=np.asarray(entry["q_eval"], dtype=np.float64),
        object_pose6=np.asarray(cfg.mujoco.object_pose6, dtype=np.float64),
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
        show_rerun=bool(args.show_viewer),
        use_visual_mesh_for_rerun=True,
        save_rerun_path=args.save_rrd_path,
    )

    print("=== Play Selection ===")
    print(f"data_mode: {data_mode}")
    print(f"data_pt_path: {Path(args.data_pt_path).expanduser().resolve()}")
    print(f"robot_name: {robot_name}")
    print(f"sample_idx: {entry['ordered_idx']}")
    print(f"object_name: {object_name}")
    print(f"source: {entry.get('source', 'n/a')}")
    print(f"sample_id: {entry.get('raw_sample', {}).get('sample_id', 'n/a')}")
    print(f"dataset_index: {entry.get('meta', {}).get('dataset_index', 'n/a')}")
    print("=== Play Summary ===")
    print(json.dumps(result["summary"], indent=2))
    if result.get("disturbance") is not None:
        print("=== Disturbance Summary ===")
        disturbance = result["disturbance"]
        print(f"all_directions_in_hand: {disturbance['all_directions_in_hand']}")
        for direction in disturbance["directions"]:
            print(
                f"  {direction['direction']}: final_in_hand={direction['summary']['final_in_hand']} "
                f"unstable={direction['unstable']} force={direction['force_magnitude']:.4f}"
            )
    if result.get("viewer_error") is not None:
        print(f"viewer_error: {result['viewer_error']}")

    if args.output_json is not None:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(
                _to_jsonable(
                    {
                        "selection": {
                            "data_mode": data_mode,
                            "data_pt_path": str(Path(args.data_pt_path).expanduser().resolve()),
                            "robot_name": robot_name,
                            "sample_idx": int(entry["ordered_idx"]),
                            "object_name": object_name,
                            "source": entry.get("source"),
                        },
                        "summary": result["summary"],
                        "disturbance": result.get("disturbance"),
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
