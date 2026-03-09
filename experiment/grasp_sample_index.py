from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Iterable, Sequence

import torch
from hydra.utils import to_absolute_path

from experiment.baselines.datasets import load_split_objects


def _to_tensor_q(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu().float()
    return torch.tensor(value, dtype=torch.float32)


def _infer_q_from_cmap_record(item: Sequence[Any]) -> torch.Tensor:
    for value in item[:-2]:
        if torch.is_tensor(value):
            return value.detach().cpu().float()
        if isinstance(value, (list, tuple)):
            return torch.tensor(value, dtype=torch.float32)
    raise ValueError(f"Could not infer grasp tensor from CMap record: {item}")


def _first_seen_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _assign_ordered_idx(entries: list[dict[str, Any]]) -> None:
    per_robot = defaultdict(int)
    for entry in entries:
        robot_name = str(entry["robot_name"])
        entry["ordered_idx"] = per_robot[robot_name]
        per_robot[robot_name] += 1


def _apply_per_robot_cap(entries: list[dict[str, Any]], max_samples_per_robot: int) -> list[dict[str, Any]]:
    if max_samples_per_robot < 0:
        return list(entries)
    per_robot = defaultdict(int)
    out: list[dict[str, Any]] = []
    for entry in entries:
        robot_name = str(entry["robot_name"])
        if per_robot[robot_name] >= max_samples_per_robot:
            continue
        out.append(entry)
        per_robot[robot_name] += 1
    return out


def group_entries_by_robot(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["robot_name"])].append(entry)
    return dict(grouped)


def ordered_robot_names(
    entries: list[dict[str, Any]],
    *,
    preferred_order: Sequence[str] | None = None,
) -> list[str]:
    present = {str(entry["robot_name"]) for entry in entries}
    if preferred_order is not None:
        ordered = [str(name) for name in preferred_order if str(name) in present]
        extras = [name for name in _first_seen_order(entry["robot_name"] for entry in entries) if name not in set(ordered)]
        return ordered + extras
    return _first_seen_order(str(entry["robot_name"]) for entry in entries)


def load_ordered_results_entries(
    results_path: str,
    *,
    robot_names: Sequence[str] | None = None,
    max_samples: int = -1,
    max_samples_per_robot: int = -1,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    abs_path = to_absolute_path(str(results_path))
    payload = torch.load(abs_path, map_location="cpu")
    if not isinstance(payload, dict) or "samples" not in payload:
        raise ValueError(f"Expected dict payload with 'samples': {abs_path}")

    filter_set = None if robot_names is None else {str(name) for name in robot_names}
    entries: list[dict[str, Any]] = []
    for global_index, sample in enumerate(payload["samples"]):
        if not isinstance(sample, dict):
            continue
        robot_name = str(sample["robot_name"])
        if filter_set is not None and robot_name not in filter_set:
            continue
        entries.append(
            {
                "input_mode": "results",
                "global_index": int(global_index),
                "robot_name": robot_name,
                "object_name": str(sample["object_name"]),
                "dof": int(sample["dof"]),
                "raw_sample": sample,
                "source": str(sample.get("source", "pred")),
            }
        )

    entries = _apply_per_robot_cap(entries, int(max_samples_per_robot))
    if max_samples >= 0:
        entries = entries[: int(max_samples)]
    _assign_ordered_idx(entries)
    robots = ordered_robot_names(entries, preferred_order=robot_names)
    return payload, entries, robots


def load_ordered_cmap_entries(
    dro_root: str,
    *,
    split: str,
    dataset_path: str | None = None,
    robot_names: Sequence[str] | None = None,
    object_names: Sequence[str] | None = None,
    max_samples_per_robot: int = -1,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    abs_dro_root = to_absolute_path(str(dro_root))
    if dataset_path is None:
        filtered_path = os.path.join(abs_dro_root, "data/CMapDataset_filtered/cmap_dataset.pt")
        raw_path = os.path.join(abs_dro_root, "data/CMapDataset/cmap_dataset.pt")
        resolved_dataset_path = filtered_path if os.path.exists(filtered_path) else raw_path
    else:
        resolved_dataset_path = to_absolute_path(str(dataset_path))
    if not os.path.exists(resolved_dataset_path):
        raise FileNotFoundError(f"Could not find CMap dataset under {abs_dro_root}")

    loaded = torch.load(resolved_dataset_path, map_location="cpu")
    if not isinstance(loaded, dict) or "metadata" not in loaded:
        raise ValueError(f"Expected dict payload with 'metadata': {resolved_dataset_path}")

    split_objects = load_split_objects(abs_dro_root, split=split)
    object_filter = None if object_names is None else {str(name) for name in object_names}
    robot_filter = None if robot_names is None else {str(name) for name in robot_names}

    per_robot_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dataset_index, item in enumerate(loaded["metadata"]):
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        robot_name = item[-1]
        object_name = item[-2]
        if not isinstance(robot_name, str) or not isinstance(object_name, str):
            continue
        if robot_filter is not None and robot_name not in robot_filter:
            continue
        if object_name not in split_objects:
            continue
        if object_filter is not None and object_name not in object_filter:
            continue
        q_tensor = _infer_q_from_cmap_record(item)
        per_robot_rows[robot_name].append(
            {
                "input_mode": "cmap_gt",
                "dataset_index": int(dataset_index),
                "robot_name": robot_name,
                "object_name": object_name,
                "dof": int(q_tensor.numel()),
                "q_gt": q_tensor,
                "q_eval": q_tensor,
                "source": "cmap_gt",
                "raw_sample": {
                    "dataset_index": int(dataset_index),
                    "robot_name": robot_name,
                    "object_name": object_name,
                    "q_gt": q_tensor,
                    "dof": int(q_tensor.numel()),
                    "source": "cmap_gt",
                },
            }
        )

    robots = ordered_robot_names(
        [{"robot_name": name} for name in per_robot_rows.keys()],
        preferred_order=robot_names,
    )
    entries: list[dict[str, Any]] = []
    for robot_name in robots:
        rows = sorted(
            per_robot_rows.get(robot_name, []),
            key=lambda row: (str(row["object_name"]), int(row["dataset_index"])),
        )
        rows = _apply_per_robot_cap(rows, int(max_samples_per_robot))
        entries.extend(rows)
    _assign_ordered_idx(entries)

    meta = {
        "dro_root": abs_dro_root,
        "split": str(split),
        "dataset_path": resolved_dataset_path,
        "source": "cmap_gt",
    }
    return meta, entries, robots


def get_sample_for_robot(entries_by_robot: dict[str, list[dict[str, Any]]], robot_name: str, ordered_idx: int) -> dict[str, Any]:
    rows = entries_by_robot.get(str(robot_name), [])
    if len(rows) == 0:
        raise KeyError(f"No samples for robot '{robot_name}'")
    idx = max(0, min(int(ordered_idx), len(rows) - 1))
    return rows[idx]
