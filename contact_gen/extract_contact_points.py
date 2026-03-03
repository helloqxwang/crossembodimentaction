from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Any

import torch
from tqdm import tqdm


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract manipulator-side contact point clouds from DRO-Grasp demonstrations "
            "using GenDex-style continuous contact map computation."
        )
    )
    parser.add_argument(
        "--dro-root",
        type=str,
        default=os.path.join(PROJECT_ROOT, "..", "DRO-Grasp"),
        help="Path to DRO-Grasp repository root.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help=(
            "Optional metadata dataset path. Defaults to "
            "<dro_root>/data/CMapDataset_filtered/cmap_dataset.pt."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "validate", "all"],
        help="Which split to extract.",
    )
    parser.add_argument(
        "--robot-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional robot whitelist. If omitted, keep all robots.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help=(
            "Threshold converting continuous hand contact-map values to a 0/1 mask. "
            "Default 0.4 follows common GenDex post-process practice."
        ),
    )
    parser.add_argument(
        "--align-exp-scale",
        type=float,
        default=2.0,
        help="Scale in exp(scale * (1 - align)) for GenDex align-distance.",
    )
    parser.add_argument(
        "--sigmoid-scale",
        type=float,
        default=10.0,
        help="Scale in sigmoid(scale * contact_dist) for contact value conversion.",
    )
    parser.add_argument(
        "--num-hand-points",
        type=int,
        default=1024,
        help=(
            "Whole-hand uniform surface sample count per grasp sample. "
            "This controls final hand surface point count exactly."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Optional limit for debug. -1 means all selected samples.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "contact_gen", "contact_points_validate.pt"),
        help="Output .pt file path.",
    )
    return parser


def _to_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().float().cpu()
    return torch.tensor(x, dtype=torch.float32)


def _entry_to_q_object_robot(entry: tuple[Any, ...]) -> tuple[torch.Tensor, str, str]:
    # Supports both:
    # - (q, object_name, robot_name)
    # - (map_value, q, object_name, robot_name)
    if len(entry) == 3:
        q, object_name, robot_name = entry
        return _to_tensor(q), str(object_name), str(robot_name)
    if len(entry) == 4:
        _, q, object_name, robot_name = entry
        return _to_tensor(q), str(object_name), str(robot_name)
    raise ValueError(f"Unsupported metadata entry length: {len(entry)}")


def _load_split_objects(dro_root: str, split: str) -> set[str]:
    if split == "all":
        return set()
    split_path = os.path.join(dro_root, "data/CMapDataset_filtered/split_train_validate_objects.json")
    with open(split_path, "r", encoding="utf-8") as f:
        split_json = json.load(f)
    if split not in split_json:
        raise KeyError(f"split '{split}' not found in {split_path}")
    return set(str(x) for x in split_json[split])


def _resolve_robot_model_name(robot_name: str, asset_names: set[str]) -> str:
    if robot_name in asset_names:
        return robot_name

    preferred = {
        "allegro": [
            "dro/allegro/allegro_hand_left_extended",
            "dro/allegro/allegro_hand_left",
        ],
        "barrett": [
            "dro/barrett/model_extended",
            "dro/barrett/model",
        ],
        "ezgripper": [
            "dro/ezgripper/ezgripper_extended",
            "dro/ezgripper/ezgripper",
        ],
        "robotiq_3finger": [
            "dro/urdf/robotiq_3finger_description_extended",
            "dro/urdf/robotiq_3finger_description",
        ],
        "shadowhand": [
            "dro/shadowhand/shadow_hand_right_extended",
            "dro/shadowhand/shadow_hand_right",
        ],
    }
    for candidate in preferred.get(robot_name, []):
        if candidate in asset_names:
            return candidate

    candidates = [k for k in asset_names if (f"/{robot_name}/" in k) or (robot_name in k)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        candidates = sorted(
            candidates,
            key=lambda x: (
                int(x.startswith("dro/")),
                int("extended" in x),
                int("right" in x),
                -len(x),
            ),
            reverse=True,
        )
        return candidates[0]
    raise KeyError(f"Cannot resolve robot model for '{robot_name}'.")


def _normalize_normals(normals: torch.Tensor) -> torch.Tensor:
    return normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-8)


def _object_mesh_path(dro_root: str, object_name: str) -> str:
    dataset_name, mesh_name = object_name.split("+")
    return os.path.join(
        dro_root,
        "data/data_urdf/object",
        dataset_name,
        mesh_name,
        f"{mesh_name}.stl",
    )


def _load_object_points_and_normals(
    dro_root: str,
    object_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_name, mesh_name = object_name.split("+")
    pc_path = os.path.join(dro_root, "data/PointCloud/object", dataset_name, f"{mesh_name}.pt")
    if not os.path.exists(pc_path):
        raise FileNotFoundError(f"Object point cloud not found: {pc_path}")

    object_pc = torch.load(pc_path, map_location="cpu").float()
    if object_pc.ndim != 2 or object_pc.shape[1] < 3:
        raise ValueError(f"Invalid object point cloud shape at {pc_path}: {tuple(object_pc.shape)}")
    object_xyz = object_pc[:, :3]

    if object_pc.shape[1] >= 6:
        object_normal = _normalize_normals(object_pc[:, 3:6])
        return object_xyz, object_normal

    # Fallback: estimate normals by nearest sampled mesh points.
    import trimesh

    mesh_path = _object_mesh_path(dro_root, object_name)
    mesh = trimesh.load_mesh(mesh_path)
    sampled_xyz, face_indices = mesh.sample(max(4096, object_xyz.shape[0]), return_index=True)
    sampled_xyz = torch.tensor(sampled_xyz, dtype=torch.float32)
    sampled_normal = torch.tensor(mesh.face_normals[face_indices], dtype=torch.float32)
    sampled_normal = _normalize_normals(sampled_normal)

    nearest = torch.cdist(object_xyz, sampled_xyz, p=2).argmin(dim=1)
    object_normal = sampled_normal[nearest]
    return object_xyz, object_normal


def _compute_gendex_contact_value_source_target(
    source_points: torch.Tensor,
    source_normals: torch.Tensor,
    target_points: torch.Tensor,
    align_exp_scale: float,
    sigmoid_scale: float,
) -> torch.Tensor:
    # source_points/source_normals: (Ns, 3), target_points: (Nt, 3)
    # Equivalent to pairwise diff, but avoids allocating (Ns, Nt, 3).
    if source_points.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    if target_points.numel() == 0:
        return torch.zeros((source_points.shape[0],), dtype=torch.float32)

    source_points = source_points.float()
    source_normals = source_normals.float()
    target_points = target_points.float()

    ss = (source_points * source_points).sum(dim=1, keepdim=True)  # (Ns,1)
    tt = (target_points * target_points).sum(dim=1).unsqueeze(0)  # (1,Nt)
    st = source_points @ target_points.T  # (Ns,Nt)
    source_target_dist = torch.sqrt((ss + tt - 2.0 * st).clamp_min(0.0))  # (Ns,Nt)

    target_dot_normal = target_points @ source_normals.T  # (Nt,Ns)
    source_dot_normal = (source_points * source_normals).sum(dim=1, keepdim=True)  # (Ns,1)
    source_target_align = target_dot_normal.T - source_dot_normal  # (Ns,Nt)
    source_target_align = source_target_align / (source_target_dist + 1e-5)

    source_target_align_dist = source_target_dist * torch.exp(
        float(align_exp_scale) * (1.0 - source_target_align)
    )
    contact_dist = torch.sqrt(source_target_align_dist.min(dim=1).values.clamp_min(0.0))
    contact_value = 1.0 - 2.0 * (torch.sigmoid(float(sigmoid_scale) * contact_dist) - 0.5)
    return contact_value.clamp(0.0, 1.0)


def _empty_contact(
    sample_id: int,
    object_name: str,
    robot_name: str,
    robot_model_name: str,
    q: torch.Tensor,
    num_hand_points: int,
    threshold: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "object_name": object_name,
        "robot_name": robot_name,
        "robot_model_name": robot_model_name,
        "q": q,
        "threshold": float(threshold),
        "num_hand_points": int(num_hand_points),
        "hand_surface_points": torch.zeros((num_hand_points, 3), dtype=torch.float32),
        "hand_surface_normals": torch.zeros((num_hand_points, 3), dtype=torch.float32),
        "hand_contact_points": torch.zeros((0, 3), dtype=torch.float32),
        "hand_contact_mask": torch.zeros((num_hand_points,), dtype=torch.bool),
        "hand_contact_value": torch.zeros((num_hand_points,), dtype=torch.float32),
        "num_hand_contacts": 0,
        "surface_template_hash": None,
        "contact_value_stats": {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        },
        "invalid_reason": reason,
    }


def main() -> None:
    args = _build_parser().parse_args()

    if int(args.num_hand_points) <= 0:
        raise ValueError("--num-hand-points must be positive.")

    dro_root = os.path.abspath(args.dro_root)
    if not os.path.exists(dro_root):
        raise FileNotFoundError(f"dro_root does not exist: {dro_root}")

    if args.dataset_path is None:
        dataset_path = os.path.join(dro_root, "data/CMapDataset_filtered/cmap_dataset.pt")
    else:
        dataset_path = os.path.abspath(args.dataset_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset path does not exist: {dataset_path}")

    sys.path.append(PROJECT_ROOT)
    from robot_model.robot_model import create_robot_model, discover_robot_assets  # type: ignore

    assets_meta = discover_robot_assets()
    asset_names = set(assets_meta.keys())

    dataset = torch.load(dataset_path, map_location="cpu")
    metadata: list[tuple[Any, ...]] = list(dataset["metadata"])
    split_objects = _load_split_objects(dro_root, args.split)
    allowed_robots = None if args.robot_names is None else set(args.robot_names)

    selected: list[tuple[torch.Tensor, str, str]] = []
    for entry in metadata:
        q, object_name, robot_name = _entry_to_q_object_robot(entry)
        if args.split != "all" and object_name not in split_objects:
            continue
        if allowed_robots is not None and robot_name not in allowed_robots:
            continue
        selected.append((q, object_name, robot_name))

    if args.max_samples > 0:
        selected = selected[: args.max_samples]

    if len(selected) == 0:
        raise RuntimeError("No samples selected. Check split/robot filters.")

    hand_cache: dict[str, Any] = {}
    hand_template_hash: dict[str, str] = {}
    object_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    samples_out: list[dict[str, Any]] = []
    pbar = tqdm(total=len(selected), desc="extract_contact_points")
    with torch.inference_mode():
        for sample_id, (q, object_name, robot_name) in enumerate(selected):
            resolved_robot_name = robot_name
            try:
                if robot_name not in hand_cache:
                    resolved_robot_name = _resolve_robot_model_name(robot_name, asset_names)
                    hand_cache[robot_name] = create_robot_model(
                        robot_name=resolved_robot_name,
                        device=torch.device("cpu"),
                        num_points=int(args.num_hand_points),
                    )
                    hand_template_hash[robot_name] = hand_cache[robot_name].get_surface_template_hash()
                hand = hand_cache[robot_name]
                resolved_robot_name = str(hand.robot_name)

                q = q.float().cpu()
                # Use fixed per-robot surface template from create_robot_model(num_points=...),
                # transformed by q for this grasp.
                hand_surface_points, hand_surface_normals = hand.get_surface_points_normals(q)
                hand_surface_points = hand_surface_points.float().cpu()
                hand_surface_normals = hand_surface_normals.float().cpu()

                if object_name not in object_cache:
                    object_cache[object_name] = _load_object_points_and_normals(dro_root, object_name)
                object_points, _ = object_cache[object_name]

                # Directly compute hand contact values (hand as source, object as target).
                hand_contact_value = _compute_gendex_contact_value_source_target(
                    source_points=hand_surface_points,
                    source_normals=hand_surface_normals,
                    target_points=object_points,
                    align_exp_scale=float(args.align_exp_scale),
                    sigmoid_scale=float(args.sigmoid_scale),
                )
                hand_contact_mask = hand_contact_value >= float(args.threshold)
                hand_contact_points = hand_surface_points[hand_contact_mask]

                samples_out.append(
                    {
                        "sample_id": sample_id,
                        "object_name": object_name,
                        "robot_name": robot_name,
                        "robot_model_name": resolved_robot_name,
                        "q": q,
                        "threshold": float(args.threshold),
                        "num_hand_points": int(args.num_hand_points),
                        "hand_surface_points": hand_surface_points,
                        "hand_surface_normals": hand_surface_normals,
                        "hand_contact_points": hand_contact_points,
                        "hand_contact_mask": hand_contact_mask,
                        "hand_contact_value": hand_contact_value,
                        "num_hand_contacts": int(hand_contact_mask.sum().item()),
                        "surface_template_hash": hand_template_hash.get(robot_name),
                        "contact_value_stats": {
                            "min": float(hand_contact_value.min().item()) if hand_contact_value.numel() > 0 else None,
                            "max": float(hand_contact_value.max().item()) if hand_contact_value.numel() > 0 else None,
                            "mean": float(hand_contact_value.mean().item()) if hand_contact_value.numel() > 0 else None,
                            "std": float(hand_contact_value.std(unbiased=False).item())
                            if hand_contact_value.numel() > 0
                            else None,
                        },
                    }
                )
            except Exception as exc:  # Keep long extraction jobs robust.
                print(f"Warning: failed to process sample {sample_id} (object={object_name}, robot={robot_name}): {exc}")
                samples_out.append(
                    _empty_contact(
                        sample_id=sample_id,
                        object_name=object_name,
                        robot_name=robot_name,
                        robot_model_name=resolved_robot_name,
                        q=q,
                        num_hand_points=int(args.num_hand_points),
                        threshold=float(args.threshold),
                        reason=str(exc),
                    )
                )
            pbar.update(1)
    pbar.close()

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dro_root": dro_root,
            "dataset_path": dataset_path,
            "split": args.split,
            "robot_names": args.robot_names,
            "threshold": float(args.threshold),
            "align_exp_scale": float(args.align_exp_scale),
            "sigmoid_scale": float(args.sigmoid_scale),
            "num_hand_points": int(args.num_hand_points),
            "num_samples": len(samples_out),
            "num_invalid_samples": int(sum(1 for s in samples_out if s.get("invalid_reason") is not None)),
            "surface_template_hash_by_robot": hand_template_hash,
            "method": "GenDex align-distance contact map on hand surface -> threshold mask on hand",
        },
        "samples": samples_out,
    }
    torch.save(payload, output_path)

    valid = [s for s in samples_out if s.get("invalid_reason") is None]
    invalid = [s for s in samples_out if s.get("invalid_reason") is not None]
    avg_hand_contacts = (
        float(sum(s["num_hand_contacts"] for s in valid) / max(len(valid), 1))
        if valid
        else math.nan
    )
    print("Extraction finished.")
    print(f"  output: {output_path}")
    print(f"  samples: {len(samples_out)} (valid={len(valid)}, invalid={len(invalid)})")
    print(f"  threshold: {args.threshold}")
    print(f"  num_hand_points: {int(args.num_hand_points)}")
    print(f"  avg hand contacts: {avg_hand_contacts:.2f}")


if __name__ == "__main__":
    main()
