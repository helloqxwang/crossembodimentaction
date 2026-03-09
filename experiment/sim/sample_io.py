from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from omegaconf import DictConfig

from experiment.grasp_sample_index import (
    get_sample_for_robot,
    group_entries_by_robot,
    load_ordered_cmap_entries,
    load_ordered_results_entries,
)


def _resolve_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _to_numpy_q(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float64)
    return np.asarray(value, dtype=np.float64)


def detect_data_pt_mode(data_pt_path: str | Path) -> str:
    payload = torch.load(_resolve_path(data_pt_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload: {data_pt_path}")
    if "samples" in payload:
        return "results"
    if "metadata" in payload:
        return "cmap_dataset"
    if "records" in payload:
        raise ValueError(
            "MuJoCo eval payloads are not valid play/test inputs. "
            "Pass a generated grasp payload or a CMap dataset .pt file instead."
        )
    raise ValueError(f"Unsupported payload schema for {data_pt_path}")


def _apply_total_limit(samples: list[dict[str, Any]], max_total_samples: int) -> list[dict[str, Any]]:
    if max_total_samples < 0:
        return samples
    return samples[: int(max_total_samples)]


def load_result_samples(
    results_path: str,
    *,
    robot_names: Sequence[str] | None = None,
    source_filter: Iterable[str] | None = None,
    q_key: str = "q_pred",
    max_samples_per_robot: int = -1,
    max_total_samples: int = -1,
    shuffle: bool = False,
    seed: int = 0,
) -> list[dict[str, Any]]:
    _, entries, _ = load_ordered_results_entries(
        results_path,
        robot_names=robot_names,
        max_samples=-1,
        max_samples_per_robot=-1,
    )
    rows = list(entries)
    if shuffle:
        random.Random(seed).shuffle(rows)

    source_filter_set = None if source_filter is None else {str(x) for x in source_filter}
    per_robot = defaultdict(int)
    samples: list[dict[str, Any]] = []
    for entry in rows:
        raw_sample = entry["raw_sample"]
        robot_name = str(entry["robot_name"])
        source = str(raw_sample.get("source", entry.get("source", "pred")))
        if source_filter_set is not None and source not in source_filter_set:
            continue
        if max_samples_per_robot >= 0 and per_robot[robot_name] >= max_samples_per_robot:
            continue

        q_data = raw_sample.get(q_key)
        if q_data is None and q_key == "q_pred":
            q_data = raw_sample.get("q_pred_padded")
        if q_data is None and q_key == "q_gt":
            q_data = raw_sample.get("q_gt_padded")
        if q_data is None:
            continue

        dof = int(entry["dof"])
        q_eval = _to_numpy_q(q_data)[:dof]
        samples.append(
            {
                "input_mode": "results",
                "robot_name": robot_name,
                "object_name": str(entry["object_name"]),
                "dof": dof,
                "ordered_idx": int(entry["ordered_idx"]),
                "q_eval": q_eval,
                "source": source if q_key == "q_pred" else f"{source}_as_{q_key}",
                "raw_sample": raw_sample,
                "meta": {
                    "pair_status": raw_sample.get("pair_status"),
                    "sample_id": raw_sample.get("sample_id"),
                    "global_index": int(entry.get("global_index", -1)),
                    "ordered_idx": int(entry["ordered_idx"]),
                },
            }
        )
        per_robot[robot_name] += 1

    return _apply_total_limit(samples, max_total_samples)


def load_cmap_dataset_samples(
    *,
    dro_root: str,
    split: str,
    dataset_path: str | None = None,
    robot_names: Sequence[str] | None = None,
    object_names: Sequence[str] | None = None,
    max_samples_per_robot: int = -1,
    max_total_samples: int = -1,
    shuffle: bool = False,
    seed: int = 0,
) -> list[dict[str, Any]]:
    _, entries, _ = load_ordered_cmap_entries(
        dro_root=dro_root,
        split=split,
        dataset_path=dataset_path,
        robot_names=robot_names,
        object_names=object_names,
        max_samples_per_robot=-1,
    )
    rows = list(entries)
    if shuffle:
        random.Random(seed).shuffle(rows)

    per_robot = defaultdict(int)
    samples: list[dict[str, Any]] = []
    for entry in rows:
        robot_name = str(entry["robot_name"])
        if max_samples_per_robot >= 0 and per_robot[robot_name] >= max_samples_per_robot:
            continue
        q_eval = _to_numpy_q(entry["q_eval"])
        samples.append(
            {
                "input_mode": "cmap_gt",
                "robot_name": robot_name,
                "object_name": str(entry["object_name"]),
                "dof": int(entry["dof"]),
                "ordered_idx": int(entry["ordered_idx"]),
                "q_eval": q_eval,
                "source": "cmap_gt",
                "raw_sample": entry["raw_sample"],
                "meta": {
                    "pair_status": "has-data",
                    "dataset_index": int(entry.get("dataset_index", -1)),
                    "ordered_idx": int(entry["ordered_idx"]),
                },
            }
        )
        per_robot[robot_name] += 1

    return _apply_total_limit(samples, max_total_samples)


def load_samples_from_config(cfg: DictConfig) -> list[dict[str, Any]]:
    mode = str(cfg.input.mode).lower()
    if mode == "generated":
        return load_result_samples(
            str(cfg.input.generated.results_path),
            robot_names=None,
            source_filter=cfg.input.generated.source_filter,
            q_key=str(cfg.input.generated.q_key),
            max_samples_per_robot=int(cfg.input.max_samples_per_robot),
            max_total_samples=int(cfg.input.max_total_samples),
            shuffle=bool(cfg.input.shuffle),
            seed=int(cfg.seed),
        )
    if mode == "cmap_gt":
        object_names = None
        if cfg.input.cmap_gt.get("object_names") is not None:
            object_names = [str(x) for x in cfg.input.cmap_gt.object_names]
        data_pt_path = cfg.input.cmap_gt.get("data_pt_path")
        return load_cmap_dataset_samples(
            dro_root=str(cfg.paths.dro_root),
            split=str(cfg.input.cmap_gt.split),
            dataset_path=None if data_pt_path is None else str(data_pt_path),
            robot_names=[str(x) for x in cfg.input.cmap_gt.robot_names],
            object_names=object_names,
            max_samples_per_robot=int(cfg.input.max_samples_per_robot),
            max_total_samples=int(cfg.input.max_total_samples),
            shuffle=bool(cfg.input.shuffle),
            seed=int(cfg.seed),
        )
    raise ValueError(f"Unsupported input.mode: {mode}")


def select_sample_from_data_pt(
    *,
    data_pt_path: str,
    dro_root: str,
    robot_name: str,
    sample_idx: int,
    split: str,
    q_key: str,
) -> dict[str, Any]:
    data_mode = detect_data_pt_mode(data_pt_path)
    if data_mode == "results":
        rows = load_result_samples(
            data_pt_path,
            robot_names=[robot_name],
            source_filter=None,
            q_key=q_key,
            max_samples_per_robot=-1,
            max_total_samples=-1,
            shuffle=False,
            seed=0,
        )
    else:
        rows = load_cmap_dataset_samples(
            dro_root=dro_root,
            split=split,
            dataset_path=data_pt_path,
            robot_names=[robot_name],
            max_samples_per_robot=-1,
            max_total_samples=-1,
            shuffle=False,
            seed=0,
        )
    grouped = group_entries_by_robot(rows)
    return get_sample_for_robot(grouped, robot_name, int(sample_idx))
