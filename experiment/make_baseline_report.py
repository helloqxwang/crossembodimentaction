from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_pt(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload: {path}")
    return payload


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _validate_per_hand(payload: dict[str, Any]) -> dict[str, dict[str, float | int | None]]:
    per_hand: dict[str, dict[str, list[float] | int]] = {}
    summary = payload.get("summary", {})
    for hand, count in dict(summary.get("per_robot_data_count", {})).items():
        per_hand.setdefault(str(hand), {})
        per_hand[str(hand)]["data_count"] = int(count)
    for hand, count in dict(summary.get("per_robot_no_data_count", {})).items():
        per_hand.setdefault(str(hand), {})
        per_hand[str(hand)]["no_data_count"] = int(count)

    for sample in payload.get("samples", []):
        if not isinstance(sample, dict):
            continue
        hand = str(sample.get("robot_name", "unknown"))
        stats = per_hand.setdefault(hand, {})
        if sample.get("pair_status") != "has-data":
            continue
        for key in ("mse_full", "mse_valid", "joint_l1", "chamfer"):
            value = sample.get(key)
            if value is None:
                continue
            stats.setdefault(key, [])
            stats[key].append(float(value))

    out: dict[str, dict[str, float | int | None]] = {}
    for hand, stats in per_hand.items():
        out[hand] = {
            "data_count": int(stats.get("data_count", 0)),
            "no_data_count": int(stats.get("no_data_count", 0)),
            "mse_full": _mean_or_none(list(stats.get("mse_full", []))),
            "mse_valid": _mean_or_none(list(stats.get("mse_valid", []))),
            "joint_l1": _mean_or_none(list(stats.get("joint_l1", []))),
            "chamfer": _mean_or_none(list(stats.get("chamfer", []))),
        }
    return out


def _sim_per_hand(payload: dict[str, Any]) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    per_hand = payload.get("aggregate", {}).get("per_hand", {})
    if not isinstance(per_hand, dict):
        return out
    for hand, stats in per_hand.items():
        if not isinstance(stats, dict):
            continue
        out[str(hand)] = {
            "count": int(stats.get("count", 0)),
            "in_hand_rate": None if stats.get("in_hand_rate") is None else float(stats.get("in_hand_rate")),
            "robust_in_hand_rate": (
                None if stats.get("robust_in_hand_rate") is None else float(stats.get("robust_in_hand_rate"))
            ),
            "final_contact_rate": (
                None if stats.get("final_contact_rate") is None else float(stats.get("final_contact_rate"))
            ),
            "unstable_rate": None if stats.get("unstable_rate") is None else float(stats.get("unstable_rate")),
            "pos_err_final_mean": (
                None if stats.get("pos_err_final_mean") is None else float(stats.get("pos_err_final_mean"))
            ),
            "rot_err_final_mean": (
                None if stats.get("rot_err_final_mean") is None else float(stats.get("rot_err_final_mean"))
            ),
            "final_hand_obj_distance_mean": (
                None
                if stats.get("final_hand_obj_distance_mean") is None
                else float(stats.get("final_hand_obj_distance_mean"))
            ),
        }
    return out


def _select_sim_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in payload.get("records", []):
        if not isinstance(record, dict):
            continue
        if record.get("error") is not None or not isinstance(record.get("summary"), dict):
            continue
        rows.append(record)
    return rows


def _sim_stats_from_rows(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "count": 0,
            "in_hand_rate": None,
            "robust_in_hand_rate": None,
            "final_contact_rate": None,
            "unstable_rate": None,
            "pos_err_final_mean": None,
            "rot_err_final_mean": None,
            "final_hand_obj_distance_mean": None,
        }
    robust = [
        bool(row.get("disturbance", {}).get("all_directions_in_hand"))
        for row in rows
        if isinstance(row.get("disturbance"), dict)
    ]
    finite_hand_dist = [
        float(row["summary"]["final_hand_obj_distance"])
        for row in rows
        if np.isfinite(float(row["summary"]["final_hand_obj_distance"]))
    ]
    return {
        "count": len(rows),
        "in_hand_rate": float(np.mean([bool(row["summary"]["final_in_hand"]) for row in rows])),
        "robust_in_hand_rate": float(np.mean(robust)) if robust else None,
        "final_contact_rate": float(np.mean([bool(row["summary"]["final_contact"]) for row in rows])),
        "unstable_rate": float(np.mean([bool(row.get("unstable", False)) for row in rows])),
        "pos_err_final_mean": float(np.mean([float(row["summary"]["pos_err_final"]) for row in rows])),
        "rot_err_final_mean": float(np.mean([float(row["summary"]["rot_err_final"]) for row in rows])),
        "final_hand_obj_distance_mean": float(np.mean(finite_hand_dist)) if finite_hand_dist else None,
    }


def _sim_per_hand_from_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    per_hand: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        per_hand.setdefault(str(row["robot_name"]), []).append(row)
    return {hand: _sim_stats_from_rows(hand_rows) for hand, hand_rows in sorted(per_hand.items())}


def _sample_lookup_by_result_index(validate_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for idx, sample in enumerate(validate_payload.get("samples", [])):
        if not isinstance(sample, dict):
            continue
        lookup[int(idx)] = sample
        sample_id = sample.get("sample_id")
        if sample_id is not None:
            lookup[int(sample_id)] = sample
    return lookup


def _gt_match_selectors(
    *,
    validate_payload: dict[str, Any],
    sim_payload: dict[str, Any],
) -> set[tuple[str, str, int]]:
    sample_lookup = _sample_lookup_by_result_index(validate_payload)
    selectors: set[tuple[str, str, int]] = set()
    for record in _select_sim_records(sim_payload):
        meta = record.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
        if str(meta.get("pair_status", "")) != "has-data":
            continue
        sample_ref = meta.get("global_index")
        if sample_ref is None or int(sample_ref) < 0:
            sample_ref = meta.get("sample_id")
        if sample_ref is None or int(sample_ref) < 0:
            continue
        sample = sample_lookup.get(int(sample_ref))
        if not isinstance(sample, dict):
            continue
        if sample.get("pair_status") != "has-data":
            continue
        selectors.add(
            (
                str(sample["robot_name"]),
                str(sample["object_name"]),
                int(sample["round_idx"]),
            )
        )
    return selectors


def _gt_record_sort_key(record: dict[str, Any]) -> tuple[int, int]:
    meta = record.get("meta", {})
    dataset_index = -1
    if isinstance(meta, dict) and meta.get("dataset_index") is not None:
        dataset_index = int(meta["dataset_index"])
    return (dataset_index, int(record.get("index", -1)))


def _select_gt_records(
    payload: dict[str, Any],
    selectors: set[tuple[str, str, int]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in _select_sim_records(payload):
        grouped[(str(record["robot_name"]), str(record["object_name"]))].append(record)

    matched: list[dict[str, Any]] = []
    for (robot_name, object_name), rows in grouped.items():
        rows_sorted = sorted(rows, key=_gt_record_sort_key)
        for pair_rank, record in enumerate(rows_sorted):
            if (robot_name, object_name, int(pair_rank)) in selectors:
                matched.append(record)
    return matched


def _overall_validate(payload: dict[str, Any]) -> dict[str, float | int | None]:
    summary = payload.get("summary", {})
    return {
        "num_samples": int(summary.get("num_samples", 0)),
        "num_data_pairs": int(summary.get("num_data_pairs", 0)),
        "num_no_data_pairs": int(summary.get("num_no_data_pairs", 0)),
        "mean_mse_full_data_only": (
            None
            if summary.get("mean_mse_full_data_only") is None
            else float(summary.get("mean_mse_full_data_only"))
        ),
        "mean_mse_valid_data_only": (
            None
            if summary.get("mean_mse_valid_data_only") is None
            else float(summary.get("mean_mse_valid_data_only"))
        ),
        "mean_joint_l1_data_only": (
            None
            if summary.get("mean_joint_l1_data_only") is None
            else float(summary.get("mean_joint_l1_data_only"))
        ),
        "mean_chamfer_data_only": (
            None if summary.get("mean_chamfer_data_only") is None else float(summary.get("mean_chamfer_data_only"))
        ),
    }


def _overall_sim(payload: dict[str, Any]) -> dict[str, float | int | None]:
    agg = payload.get("aggregate", {})
    return {
        "count_total": int(agg.get("count_total", 0)),
        "count_valid": int(agg.get("count_valid", 0)),
        "in_hand_rate": None if agg.get("in_hand_rate") is None else float(agg.get("in_hand_rate")),
        "robust_in_hand_rate": (
            None if agg.get("robust_in_hand_rate") is None else float(agg.get("robust_in_hand_rate"))
        ),
        "final_contact_rate": (
            None if agg.get("final_contact_rate") is None else float(agg.get("final_contact_rate"))
        ),
        "unstable_rate": None if agg.get("unstable_rate") is None else float(agg.get("unstable_rate")),
        "pos_err_final_mean": None if agg.get("pos_err_final_mean") is None else float(agg.get("pos_err_final_mean")),
        "rot_err_final_mean": None if agg.get("rot_err_final_mean") is None else float(agg.get("rot_err_final_mean")),
        "final_hand_obj_distance_mean": (
            None
            if agg.get("final_hand_obj_distance_mean") is None
            else float(agg.get("final_hand_obj_distance_mean"))
        ),
    }


def _to_text(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _overall_table_md(split: str, rows: list[dict[str, Any]]) -> str:
    lines = [f"## Split: {split}", "", "### Overall", ""]
    lines.append(
        "| model | type | robot | data_pairs | no_data_pairs | mse_valid | joint_l1 | chamfer | "
        "sim_in_hand | sim_robust | sim_contact | sim_unstable | gt_in_hand | gt_robust |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['model_label']} | {row['policy_type']} | {_to_text(row.get('policy_robot'))} | "
            f"{_to_text(row.get('num_data_pairs'))} | {_to_text(row.get('num_no_data_pairs'))} | "
            f"{_to_text(row.get('mean_mse_valid_data_only'))} | {_to_text(row.get('mean_joint_l1_data_only'))} | "
            f"{_to_text(row.get('mean_chamfer_data_only'))} | {_to_text(row.get('sim_in_hand_rate'))} | "
            f"{_to_text(row.get('sim_robust_in_hand_rate'))} | {_to_text(row.get('sim_final_contact_rate'))} | "
            f"{_to_text(row.get('sim_unstable_rate'))} | {_to_text(row.get('gt_in_hand_rate'))} | "
            f"{_to_text(row.get('gt_robust_in_hand_rate'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _per_hand_table_md(split: str, rows: list[dict[str, Any]]) -> str:
    lines = [f"### Per-hand ({split})", ""]
    lines.append(
        "| model | hand | data_count | no_data_count | mse_valid | joint_l1 | chamfer | "
        "sim_in_hand | sim_robust | sim_contact | sim_unstable | gt_in_hand | gt_robust |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['model_label']} | {row['hand']} | {_to_text(row.get('data_count'))} | "
            f"{_to_text(row.get('no_data_count'))} | {_to_text(row.get('mse_valid'))} | "
            f"{_to_text(row.get('joint_l1'))} | {_to_text(row.get('chamfer'))} | "
            f"{_to_text(row.get('sim_in_hand_rate'))} | {_to_text(row.get('sim_robust_in_hand_rate'))} | "
            f"{_to_text(row.get('sim_final_contact_rate'))} | {_to_text(row.get('sim_unstable_rate'))} | "
            f"{_to_text(row.get('gt_in_hand_rate'))} | {_to_text(row.get('gt_robust_in_hand_rate'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a multi-model validation + sim comparison report.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = [str(x) for x in manifest["splits"]]

    gt_payloads_by_split: dict[str, dict[str, Any]] = {}
    gt_by_split: dict[str, dict[str, Any]] = {}
    for split in splits:
        gt_sim_payload = _load_pt(str(manifest["gt"][split]["sim_gt_pt"]))
        gt_payloads_by_split[split] = gt_sim_payload
        gt_by_split[split] = {
            "overall": _overall_sim(gt_sim_payload),
            "per_hand": _sim_per_hand(gt_sim_payload),
        }

    overall_rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
    per_hand_rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}

    for model_label, model_info in manifest["policies"].items():
        policy_type = str(model_info["type"])
        policy_robot = model_info.get("robot_name")
        for split in splits:
            split_info = model_info["splits"][split]
            validate_payload = _load_pt(str(split_info["validate_pt"]))
            sim_payload = _load_pt(str(split_info["sim_pred_pt"]))
            validate_overall = _overall_validate(validate_payload)
            validate_per_hand = _validate_per_hand(validate_payload)
            sim_overall = _overall_sim(sim_payload)
            sim_per_hand = _sim_per_hand(sim_payload)
            selected_selectors = _gt_match_selectors(
                validate_payload=validate_payload,
                sim_payload=sim_payload,
            )
            gt_rows = _select_gt_records(
                gt_payloads_by_split[split],
                selectors=selected_selectors,
            )
            gt_overall = _sim_stats_from_rows(gt_rows)
            gt_per_hand = _sim_per_hand_from_rows(gt_rows)

            overall_rows_by_split[split].append(
                {
                    "model_label": model_label,
                    "policy_type": policy_type,
                    "policy_robot": policy_robot,
                    **validate_overall,
                    "sim_in_hand_rate": sim_overall["in_hand_rate"],
                    "sim_robust_in_hand_rate": sim_overall["robust_in_hand_rate"],
                    "sim_final_contact_rate": sim_overall["final_contact_rate"],
                    "sim_unstable_rate": sim_overall["unstable_rate"],
                    "sim_pos_err_final_mean": sim_overall["pos_err_final_mean"],
                    "sim_rot_err_final_mean": sim_overall["rot_err_final_mean"],
                    "sim_final_hand_obj_distance_mean": sim_overall["final_hand_obj_distance_mean"],
                    "gt_in_hand_rate": gt_overall["in_hand_rate"],
                    "gt_robust_in_hand_rate": gt_overall["robust_in_hand_rate"],
                }
            )

            all_hands = sorted(set(validate_per_hand.keys()) | set(sim_per_hand.keys()))
            for hand in all_hands:
                v = validate_per_hand.get(hand, {})
                s = sim_per_hand.get(hand, {})
                g = gt_per_hand.get(hand, {})
                per_hand_rows_by_split[split].append(
                    {
                        "model_label": model_label,
                        "policy_type": policy_type,
                        "policy_robot": policy_robot,
                        "hand": hand,
                        "data_count": v.get("data_count"),
                        "no_data_count": v.get("no_data_count"),
                        "mse_full": v.get("mse_full"),
                        "mse_valid": v.get("mse_valid"),
                        "joint_l1": v.get("joint_l1"),
                        "chamfer": v.get("chamfer"),
                        "sim_in_hand_rate": s.get("in_hand_rate"),
                        "sim_robust_in_hand_rate": s.get("robust_in_hand_rate"),
                        "sim_final_contact_rate": s.get("final_contact_rate"),
                        "sim_unstable_rate": s.get("unstable_rate"),
                        "sim_pos_err_final_mean": s.get("pos_err_final_mean"),
                        "sim_rot_err_final_mean": s.get("rot_err_final_mean"),
                        "gt_in_hand_rate": g.get("in_hand_rate"),
                        "gt_robust_in_hand_rate": g.get("robust_in_hand_rate"),
                    }
                )

    report_payload = {
        "meta": {
            "manifest": str(manifest_path),
            "splits": splits,
            "models": list(manifest["policies"].keys()),
        },
        "gt": gt_by_split,
        "overall": overall_rows_by_split,
        "per_hand": per_hand_rows_by_split,
    }

    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    md_parts = ["# Baseline Comparison Report", ""]
    for split in splits:
        md_parts.append(_overall_table_md(split, overall_rows_by_split[split]))
        md_parts.append(_per_hand_table_md(split, per_hand_rows_by_split[split]))
    output_md = Path(args.output_md).expanduser().resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(md_parts), encoding="utf-8")

    print(f"manifest: {manifest_path}")
    print(f"output_md: {output_md}")
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
