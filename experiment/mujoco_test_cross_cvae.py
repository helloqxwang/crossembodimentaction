from __future__ import annotations

import importlib.util
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import hydra
import numpy as np
import torch
from xml.etree import ElementTree
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from experiment.cross_cvae_utils import ensure_parent_dir, get_robot_asset_paths


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ensure_python_bin_in_path() -> None:
    # rr.spawn() locates `rerun` via PATH. When launching with an absolute python
    # (e.g., /path/to/.venv/bin/python), that bin dir may not be present in PATH.
    py_bin = str(Path(sys.executable).resolve().parent)
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if py_bin not in parts:
        os.environ["PATH"] = py_bin + (os.pathsep + current if current else "")


def _import_spider_validator(spider_root: str):
    script_path = os.path.join(spider_root, "examples", "run_grasp_validation.py")
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Could not find: {script_path}")

    spec = importlib.util.spec_from_file_location("spider_run_grasp_validation", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "run_grasp_validation"):
        raise AttributeError(f"run_grasp_validation not found in {script_path}")
    return module


def _to_numpy_q(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _classify_error(err_msg: str) -> str:
    msg = err_msg.lower()
    if "error opening file" in msg or "no such file or directory" in msg:
        return "asset_missing"
    if "trajectory dof does not match hand qpos dof" in msg:
        return "dof_mismatch"
    return "runtime_error"


def _find_existing_mesh_source(hand_dir: Path, original_rel: str) -> Path | None:
    target = (hand_dir / original_rel).resolve()
    if target.exists():
        return target

    basename = Path(original_rel).name
    stem = Path(original_rel).stem

    matches = list(hand_dir.rglob(basename))
    if not matches:
        matches = [p for p in hand_dir.rglob(f"{stem}.*") if p.suffix.lower() in {".obj", ".stl", ".dae"}]
    if not matches:
        return None
    return sorted(matches)[0]


def _prepare_hand_path_for_mujoco(
    hand_path: str,
    *,
    patch_dir: Path,
    cache: dict[str, str],
) -> str:
    hand_path = str(Path(hand_path).expanduser().resolve())
    if hand_path in cache:
        return cache[hand_path]

    src = Path(hand_path)
    if src.suffix.lower() != ".urdf":
        cache[hand_path] = hand_path
        return hand_path

    tree = ElementTree.parse(src)
    root = tree.getroot()
    hand_dir = src.parent

    robot_patch_dir = patch_dir / src.stem
    robot_patch_dir.mkdir(parents=True, exist_ok=True)
    patched_urdf_path = robot_patch_dir / src.name

    unresolved = []
    mesh_files = []
    for mesh in root.findall(".//mesh"):
        rel = mesh.attrib.get("filename", "")
        if not rel:
            continue
        source = _find_existing_mesh_source(hand_dir, rel)
        if source is None:
            unresolved.append(rel)
            continue
        mesh_files.append((rel, source))

    # Copy referenced meshes into patch folder while preserving URDF relative paths.
    for rel, source in mesh_files:
        dst = (robot_patch_dir / rel).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(source, dst)

        # MuJoCo URDF import with collision meshdir may strip subdir paths.
        # Ensure basename also exists under collision mesh folder.
        if "/visual/" in rel:
            rel_collision = rel.replace("/visual/", "/collision/")
            dst_collision = (robot_patch_dir / rel_collision).resolve()
            dst_collision.parent.mkdir(parents=True, exist_ok=True)
            if not dst_collision.exists():
                shutil.copy2(source, dst_collision)

    # If still unresolved filenames, keep original and let caller record error explicitly.
    if unresolved:
        cache[hand_path] = hand_path
        return hand_path

    tree.write(patched_urdf_path, encoding="utf-8", xml_declaration=True)
    cache[hand_path] = str(patched_urdf_path)
    return str(patched_urdf_path)


def _load_generated_samples(
    results_path: str,
    *,
    source_filter: Iterable[str] | None,
    q_key: str,
    max_samples_per_robot: int,
    shuffle: bool,
    seed: int,
) -> list[Dict[str, Any]]:
    payload = torch.load(results_path, map_location="cpu")
    if not isinstance(payload, dict) or "samples" not in payload:
        raise ValueError(f"Invalid generated results file: {results_path}")

    raw_samples = list(payload["samples"])
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(raw_samples)

    source_filter_set = None if source_filter is None else set(source_filter)
    per_robot = defaultdict(int)
    out: list[Dict[str, Any]] = []

    for s in raw_samples:
        robot_name = str(s["robot_name"])
        source = str(s.get("source", "pred"))
        if source_filter_set is not None and source not in source_filter_set:
            continue
        if max_samples_per_robot >= 0 and per_robot[robot_name] >= max_samples_per_robot:
            continue

        if q_key not in s:
            # fallback to padded if available
            if q_key == "q_pred" and "q_pred_padded" in s:
                q_data = s["q_pred_padded"][: int(s["dof"])]
            elif q_key == "q_gt" and "q_gt_padded" in s:
                q_data = s["q_gt_padded"][: int(s["dof"])]
            else:
                continue
        else:
            q_data = s[q_key]

        out.append(
            {
                "robot_name": robot_name,
                "object_name": str(s["object_name"]),
                "q_eval": _to_numpy_q(q_data),
                "dof": int(s["dof"]),
                "source": source if q_key == "q_pred" else f"{source}_as_{q_key}",
                "meta": {
                    "sample_id": int(s.get("sample_id", len(out))),
                    "gen_index": int(s.get("gen_index", s.get("round_idx", 0))),
                    "pred_mode": s.get("pred_mode"),
                    "mse_valid": (
                        float(s["mse_valid"])
                        if (s.get("mse_valid") is not None)
                        else float("nan")
                    ),
                    "pair_status": s.get("pair_status"),
                },
            }
        )
        per_robot[robot_name] += 1

    return out


def _load_cmap_gt_samples(
    dro_root: str,
    *,
    split: str,
    robot_names: list[str],
    max_samples_per_robot: int,
    shuffle: bool,
    seed: int,
) -> list[Dict[str, Any]]:
    split_path = os.path.join(dro_root, "data/CMapDataset_filtered/split_train_validate_objects.json")
    split_json = json.load(open(split_path, "r", encoding="utf-8"))
    if split not in split_json:
        raise KeyError(f"split '{split}' not found in {split_path}")
    split_objects = set(split_json[split])

    dataset_path = os.path.join(dro_root, "data/CMapDataset_filtered/cmap_dataset.pt")
    metadata = torch.load(dataset_path, map_location="cpu")["metadata"]

    filtered = []
    for q, object_name, robot_name in metadata:
        if robot_name not in robot_names:
            continue
        if object_name not in split_objects:
            continue
        filtered.append((q, object_name, robot_name))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(filtered)

    per_robot = defaultdict(int)
    out: list[Dict[str, Any]] = []
    for q, object_name, robot_name in filtered:
        if max_samples_per_robot >= 0 and per_robot[robot_name] >= max_samples_per_robot:
            continue
        q_np = _to_numpy_q(q)
        out.append(
            {
                "robot_name": str(robot_name),
                "object_name": str(object_name),
                "q_eval": q_np,
                "dof": int(q_np.shape[0]),
                "source": "cmap_gt",
                "meta": {},
            }
        )
        per_robot[robot_name] += 1
    return out


def _aggregate(records: list[dict]) -> dict:
    valid = [r for r in records if r.get("error") is None]
    by_key: dict[str, list[dict]] = defaultdict(list)
    by_hand: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_key[f"{r['robot_name']}|{r['source']}"].append(r)
        by_hand[str(r["robot_name"])].append(r)

    def _stats(rows: list[dict]) -> dict:
        in_hand = [bool(x["summary"]["final_in_hand"]) for x in rows]
        final_contact = [bool(x["summary"]["final_contact"]) for x in rows]
        unstable = [bool(x.get("unstable", False)) for x in rows]
        pos_final = [float(x["summary"]["pos_err_final"]) for x in rows]
        rot_final = [float(x["summary"]["rot_err_final"]) for x in rows]
        hand_obj_dist = [float(x["summary"]["final_hand_obj_distance"]) for x in rows]
        return {
            "count": len(rows),
            "in_hand_rate": float(np.mean(in_hand)) if rows else 0.0,
            "final_contact_rate": float(np.mean(final_contact)) if rows else 0.0,
            "unstable_rate": float(np.mean(unstable)) if rows else 0.0,
            "pos_err_final_mean": float(np.mean(pos_final)) if rows else float("nan"),
            "pos_err_final_max": float(np.max(pos_final)) if rows else float("nan"),
            "rot_err_final_mean": float(np.mean(rot_final)) if rows else float("nan"),
            "rot_err_final_max": float(np.max(rot_final)) if rows else float("nan"),
            "final_hand_obj_distance_mean": float(np.mean(hand_obj_dist)) if rows else float("nan"),
        }

    per_group = {k: _stats(v) for k, v in by_key.items()}
    per_hand = {k: _stats(v) for k, v in by_hand.items()}

    all_in_hand = [bool(x["summary"]["final_in_hand"]) for x in valid]
    all_contact = [bool(x["summary"]["final_contact"]) for x in valid]
    all_unstable = [bool(x.get("unstable", False)) for x in valid]
    all_pos_final = [float(x["summary"]["pos_err_final"]) for x in valid]
    all_rot_final = [float(x["summary"]["rot_err_final"]) for x in valid]
    all_hand_obj_dist = [float(x["summary"]["final_hand_obj_distance"]) for x in valid]

    return {
        "count_total": len(records),
        "count_valid": len(valid),
        "count_error": len(records) - len(valid),
        "in_hand_rate": float(np.mean(all_in_hand)) if valid else 0.0,
        "final_contact_rate": float(np.mean(all_contact)) if valid else 0.0,
        "unstable_rate": float(np.mean(all_unstable)) if valid else 0.0,
        "pos_err_final_mean": float(np.mean(all_pos_final)) if valid else float("nan"),
        "pos_err_final_max": float(np.max(all_pos_final)) if valid else float("nan"),
        "rot_err_final_mean": float(np.mean(all_rot_final)) if valid else float("nan"),
        "rot_err_final_max": float(np.max(all_rot_final)) if valid else float("nan"),
        "final_hand_obj_distance_mean": float(np.mean(all_hand_obj_dist)) if valid else float("nan"),
        "per_hand": per_hand,
        "per_robot_source": per_group,
    }


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_mujoco_test_cross_cvae")
def main(cfg: DictConfig) -> None:
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    _set_seed(int(cfg.seed))
    _ensure_python_bin_in_path()
    spider_root = to_absolute_path(str(cfg.paths.spider_root))
    dro_root = to_absolute_path(str(cfg.paths.dro_root))
    validator_module = _import_spider_validator(spider_root)
    run_grasp_validation = validator_module.run_grasp_validation
    if bool(cfg.mujoco.show_viewer):
        print("Viewer enabled: only the first evaluated sample opens the Rerun viewer (idx == 0).")
        if (os.name != "nt") and (not os.environ.get("DISPLAY")) and (not os.environ.get("WAYLAND_DISPLAY")):
            print("Warning: no DISPLAY/WAYLAND_DISPLAY detected; interactive viewer may fail to launch.")

    mode = str(cfg.input.mode).lower()
    if mode == "generated":
        samples = _load_generated_samples(
            to_absolute_path(str(cfg.input.generated.results_path)),
            source_filter=None
            if cfg.input.generated.source_filter is None
            else list(cfg.input.generated.source_filter),
            q_key=str(cfg.input.generated.q_key),
            max_samples_per_robot=int(cfg.input.max_samples_per_robot),
            shuffle=bool(cfg.input.shuffle),
            seed=int(cfg.seed),
        )
    elif mode == "cmap_gt":
        robot_names = list(cfg.input.cmap_gt.robot_names)
        samples = _load_cmap_gt_samples(
            dro_root=dro_root,
            split=str(cfg.input.cmap_gt.split),
            robot_names=robot_names,
            max_samples_per_robot=int(cfg.input.max_samples_per_robot),
            shuffle=bool(cfg.input.shuffle),
            seed=int(cfg.seed),
        )
    else:
        raise ValueError(f"Unsupported input.mode: {mode}")

    if len(samples) == 0:
        raise RuntimeError("No samples to evaluate.")

    if int(cfg.input.max_total_samples) >= 0:
        samples = samples[: int(cfg.input.max_total_samples)]

    object_pose6 = np.asarray(cfg.mujoco.object_pose6, dtype=np.float64)
    num_steps = int(cfg.mujoco.num_steps)
    eval_records: list[Dict[str, Any]] = []
    patch_dir = Path(to_absolute_path(str(cfg.output.urdf_patch_dir))).resolve()
    hand_path_cache: dict[str, str] = {}

    for idx, s in enumerate(tqdm(samples, desc="mujoco_test_cross_cvae")):
        robot_name = s["robot_name"]
        object_name = s["object_name"]
        q_eval = _to_numpy_q(s["q_eval"])
        hand_path_orig, object_mesh_path = get_robot_asset_paths(dro_root, robot_name, object_name)
        hand_path = _prepare_hand_path_for_mujoco(
            hand_path_orig,
            patch_dir=patch_dir,
            cache=hand_path_cache,
        )

        trajectory = np.repeat(q_eval[None, :], num_steps, axis=0)
        object_trajectory = np.repeat(object_pose6[None, :], num_steps, axis=0)

        show_viewer_this = bool(cfg.mujoco.show_viewer)
        save_video_this = bool(cfg.mujoco.save_video)
        video_path = None
        if save_video_this:
            video_dir = to_absolute_path(str(cfg.output.video_dir))
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"{idx:05d}_{robot_name}_{object_name.replace('+', '_')}.mp4")

        record = {
            "index": idx,
            "robot_name": robot_name,
            "object_name": object_name,
            "source": s["source"],
            "dof": int(s["dof"]),
            "hand_path": hand_path,
            "hand_path_original": hand_path_orig,
            "object_mesh_path": object_mesh_path,
            "error": None,
            "error_type": None,
            "summary": None,
            "unstable": None,
            "unstable_step": None,
            "meta": s.get("meta", {}),
        }
        try:
            result = run_grasp_validation(
                trajectory=trajectory,
                object_trajectory=object_trajectory,
                object_mesh_path=object_mesh_path,
                hand_path=hand_path,
                sim_dt=float(cfg.mujoco.sim_dt),
                steps_per_frame=int(cfg.mujoco.steps_per_frame),
                show_viewer=show_viewer_this,
                in_hand_distance_threshold=float(cfg.mujoco.in_hand_distance_threshold),
                save_video=save_video_this,
                video_path=video_path,
                render_fps=None if cfg.mujoco.render_fps is None else float(cfg.mujoco.render_fps),
                render_width=int(cfg.mujoco.render_width),
                render_height=int(cfg.mujoco.render_height),
                render_every_n_steps=int(cfg.mujoco.render_every_n_steps),
                use_visual_mesh_for_viewer=bool(cfg.mujoco.use_visual_mesh_for_viewer),
                hand_kp=float(cfg.mujoco.hand_kp),
                hand_kd=float(cfg.mujoco.hand_kd),
                hand_max_force=float(cfg.mujoco.hand_max_force),
            )
            record["summary"] = result["summary"]
            record["unstable"] = bool(result.get("unstable", False))
            record["unstable_step"] = result.get("unstable_step")
            record["video_path"] = result.get("video_path")
            record["video_error"] = result.get("video_error")
            record["viewer_error"] = result.get("viewer_error")
            if bool(cfg.output.save_full_rollout):
                record["rollout"] = {
                    "num_steps": int(result["num_steps"]),
                    "pos_err": np.asarray(result["pos_err"]),
                    "rot_err": np.asarray(result["rot_err"]),
                }
        except Exception as exc:  # noqa: BLE001
            record["error"] = str(exc)
            record["error_type"] = _classify_error(record["error"])
        eval_records.append(record)

    aggregate = _aggregate(eval_records)

    save_path = to_absolute_path(str(cfg.output.save_path))
    ensure_parent_dir(save_path)
    json_path = os.path.splitext(save_path)[0] + ".json"
    ensure_parent_dir(json_path)

    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "spider_root": spider_root,
            "dro_root": dro_root,
            "num_input_samples": len(samples),
            "num_eval_records": len(eval_records),
        },
        "aggregate": aggregate,
        "records": eval_records,
    }
    torch.save(payload, save_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": payload["meta"], "aggregate": aggregate}, f, indent=2)

    print("\n=== MuJoCo Test Saved ===")
    print(f"save_path: {save_path}")
    print(f"summary_path: {json_path}")
    print(f"count_total: {aggregate['count_total']}")
    print(f"count_valid: {aggregate['count_valid']}")
    print(f"in_hand_rate: {aggregate['in_hand_rate']:.4f}")
    viewer_errors = [
        r.get("viewer_error")
        for r in eval_records
        if (isinstance(r, dict) and r.get("viewer_error"))
    ]
    if viewer_errors:
        print("viewer_error:")
        print(f"  {viewer_errors[0]}")
    error_counts = defaultdict(int)
    for r in eval_records:
        if r["error"] is not None:
            error_counts[str(r.get("error_type", "runtime_error"))] += 1
    if error_counts:
        print("error_breakdown:")
        for k, v in sorted(error_counts.items()):
            print(f"  {k}: {v}")
    print("per_hand:")
    for hand_name, hand_stats in sorted(aggregate["per_hand"].items()):
        print(
            f"  {hand_name}: count={hand_stats['count']}, "
            f"in_hand_rate={hand_stats['in_hand_rate']:.4f}, "
            f"contact_rate={hand_stats['final_contact_rate']:.4f}, "
            f"unstable_rate={hand_stats['unstable_rate']:.4f}, "
            f"pos_err_mean={hand_stats['pos_err_final_mean']:.6f}, "
            f"rot_err_mean={hand_stats['rot_err_final_mean']:.6f}"
        )
        
if __name__ == "__main__":
    main()