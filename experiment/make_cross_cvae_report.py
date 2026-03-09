from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_pt(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload: {path}")
    return payload


def _expected_hands(validate_payload: dict[str, Any], mujoco_payloads: list[dict[str, Any]]) -> list[str]:
    hands = set()
    meta = validate_payload.get("meta", {})
    if isinstance(meta, dict):
        for key in ("selected_robots", "all_robot_names", "trained_robot_names"):
            value = meta.get(key)
            if isinstance(value, list):
                hands.update(str(x) for x in value)
    for sample in validate_payload.get("samples", []):
        if isinstance(sample, dict) and isinstance(sample.get("robot_name"), str):
            hands.add(str(sample["robot_name"]))
    for payload in mujoco_payloads:
        per_hand = payload.get("aggregate", {}).get("per_hand", {})
        if isinstance(per_hand, dict):
            hands.update(str(x) for x in per_hand.keys())
    return sorted(hands)


def _grasp_losses_per_hand(
    validate_payload: dict[str, Any],
    *,
    expected_hands: list[str],
) -> dict[str, dict[str, float | None]]:
    per_hand_joint = {}
    per_hand_chamfer = {}

    for s in validate_payload.get("samples", []):
        if s.get("pair_status") != "has-data":
            continue
        hand = str(s["robot_name"])
        per_hand_joint.setdefault(hand, [])
        per_hand_chamfer.setdefault(hand, [])
        if s.get("joint_l1") is not None:
            per_hand_joint[hand].append(float(s["joint_l1"]))
        if s.get("chamfer") is not None:
            per_hand_chamfer[hand].append(float(s["chamfer"]))

    out: dict[str, dict[str, float | None]] = {}
    all_hands = sorted(set(expected_hands) | set(per_hand_joint.keys()) | set(per_hand_chamfer.keys()))
    for hand in all_hands:
        j = per_hand_joint.get(hand, [])
        c = per_hand_chamfer.get(hand, [])
        out[hand] = {
            "joint_l1": float(np.mean(j)) if len(j) > 0 else None,
            "chamfer": float(np.mean(c)) if len(c) > 0 else None,
        }
    return out


def _in_hand_rate_from_records(
    mujoco_payload: dict[str, Any],
    *,
    pair_status: str | None,
) -> dict[str, float | None]:
    per_hand: dict[str, list[float]] = {}
    for r in mujoco_payload.get("records", []):
        if not isinstance(r, dict):
            continue
        if r.get("error") is not None:
            continue
        if pair_status is not None:
            meta = r.get("meta", {})
            if not isinstance(meta, dict) or meta.get("pair_status") != pair_status:
                continue
        summary = r.get("summary", {})
        if not isinstance(summary, dict):
            continue
        hand = str(r.get("robot_name", "unknown"))
        per_hand.setdefault(hand, []).append(float(bool(summary.get("final_in_hand", False))))
    out: dict[str, float | None] = {}
    for hand, vals in per_hand.items():
        out[hand] = float(np.mean(vals)) if vals else None
    return out


def _robust_in_hand_rate_from_records(
    mujoco_payload: dict[str, Any],
    *,
    pair_status: str | None,
) -> dict[str, float | None]:
    per_hand: dict[str, list[float]] = {}
    for r in mujoco_payload.get("records", []):
        if not isinstance(r, dict):
            continue
        if r.get("error") is not None:
            continue
        if pair_status is not None:
            meta = r.get("meta", {})
            if not isinstance(meta, dict) or meta.get("pair_status") != pair_status:
                continue
        disturbance = r.get("disturbance", {})
        if not isinstance(disturbance, dict):
            continue
        hand = str(r.get("robot_name", "unknown"))
        per_hand.setdefault(hand, []).append(float(bool(disturbance.get("all_directions_in_hand", False))))
    out: dict[str, float | None] = {}
    for hand, vals in per_hand.items():
        out[hand] = float(np.mean(vals)) if vals else None
    return out


def _in_hand_rate_per_hand_aggregate(mujoco_payload: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    per_hand = mujoco_payload.get("aggregate", {}).get("per_hand", {})
    if not isinstance(per_hand, dict):
        return out
    for hand, stats in per_hand.items():
        if not isinstance(stats, dict):
            continue
        v = stats.get("in_hand_rate")
        out[str(hand)] = None if v is None else float(v)
    return out


def _robust_in_hand_rate_per_hand_aggregate(mujoco_payload: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    per_hand = mujoco_payload.get("aggregate", {}).get("per_hand", {})
    if not isinstance(per_hand, dict):
        return out
    for hand, stats in per_hand.items():
        if not isinstance(stats, dict):
            continue
        v = stats.get("robust_in_hand_rate")
        if v is None:
            out[str(hand)] = None
        else:
            out[str(hand)] = float(v)
    return out


def _merge_metrics(
    grasp_losses: dict[str, dict[str, float | None]],
    gt_in_hand_rates: dict[str, float | None],
    gt_robust_in_hand_rates: dict[str, float | None],
    pred_in_hand_rates_has_data: dict[str, float | None],
    pred_in_hand_rates_all: dict[str, float | None],
    pred_robust_in_hand_rates_has_data: dict[str, float | None],
    pred_robust_in_hand_rates_all: dict[str, float | None],
    *,
    expected_hands: list[str],
) -> dict[str, dict[str, float | None]]:
    hands = sorted(
        set(expected_hands)
        | set(grasp_losses.keys())
        | set(gt_in_hand_rates.keys())
        | set(gt_robust_in_hand_rates.keys())
        | set(pred_in_hand_rates_has_data.keys())
        | set(pred_in_hand_rates_all.keys())
        | set(pred_robust_in_hand_rates_has_data.keys())
        | set(pred_robust_in_hand_rates_all.keys())
    )
    out = {}
    for hand in hands:
        g = grasp_losses.get(hand, {})
        out[hand] = {
            "grasp_joint_l1": g.get("joint_l1"),
            "grasp_chamfer": g.get("chamfer"),
            "gt_in_hand_rate": gt_in_hand_rates.get(hand),
            "gt_robust_in_hand_rate": gt_robust_in_hand_rates.get(hand),
            "in_hand_rate": pred_in_hand_rates_has_data.get(hand),
            "in_hand_rate_all": pred_in_hand_rates_all.get(hand),
            "robust_in_hand_rate": pred_robust_in_hand_rates_has_data.get(hand),
            "robust_in_hand_rate_all": pred_robust_in_hand_rates_all.get(hand),
        }
    return out


def _to_text(v: float | None) -> str:
    return "n/a" if v is None else f"{float(v):.6f}"


def _table_md(title: str, rows: dict[str, dict[str, float | None]]) -> str:
    lines = [f"## {title}", ""]
    lines.append(
        "| hand | grasp_joint_l1 | grasp_chamfer | GT-in-hand-rate | GT-robust-rate | "
        "in-hand-rate | in-hand-rate all | robust-rate | robust-rate all |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for hand in sorted(rows.keys()):
        r = rows[hand]
        lines.append(
            f"| {hand} | {_to_text(r.get('grasp_joint_l1'))} | "
            f"{_to_text(r.get('grasp_chamfer'))} | {_to_text(r.get('gt_in_hand_rate'))} | "
            f"{_to_text(r.get('gt_robust_in_hand_rate'))} | {_to_text(r.get('in_hand_rate'))} | "
            f"{_to_text(r.get('in_hand_rate_all'))} | {_to_text(r.get('robust_in_hand_rate'))} | "
            f"{_to_text(r.get('robust_in_hand_rate_all'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a simple per-hand report: grasping loss + GT/predicted in-hand-rate metrics."
    )
    parser.add_argument("--train-validate-pt", type=str, required=True)
    parser.add_argument("--test-validate-pt", type=str, required=True)
    parser.add_argument("--train-mujoco-pred-pt", type=str, required=True)
    parser.add_argument("--test-mujoco-pred-pt", type=str, required=True)
    parser.add_argument("--train-mujoco-gt-pt", type=str, required=True)
    parser.add_argument("--test-mujoco-gt-pt", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    train_validate = _load_pt(args.train_validate_pt)
    test_validate = _load_pt(args.test_validate_pt)
    train_mujoco_pred = _load_pt(args.train_mujoco_pred_pt)
    test_mujoco_pred = _load_pt(args.test_mujoco_pred_pt)
    train_mujoco_gt = _load_pt(args.train_mujoco_gt_pt)
    test_mujoco_gt = _load_pt(args.test_mujoco_gt_pt)

    train_expected_hands = _expected_hands(train_validate, [train_mujoco_pred, train_mujoco_gt])
    test_expected_hands = _expected_hands(test_validate, [test_mujoco_pred, test_mujoco_gt])

    train_rows = _merge_metrics(
        _grasp_losses_per_hand(train_validate, expected_hands=train_expected_hands),
        _in_hand_rate_per_hand_aggregate(train_mujoco_gt),
        _robust_in_hand_rate_per_hand_aggregate(train_mujoco_gt),
        _in_hand_rate_from_records(train_mujoco_pred, pair_status="has-data"),
        _in_hand_rate_per_hand_aggregate(train_mujoco_pred),
        _robust_in_hand_rate_from_records(train_mujoco_pred, pair_status="has-data"),
        _robust_in_hand_rate_per_hand_aggregate(train_mujoco_pred),
        expected_hands=train_expected_hands,
    )
    test_rows = _merge_metrics(
        _grasp_losses_per_hand(test_validate, expected_hands=test_expected_hands),
        _in_hand_rate_per_hand_aggregate(test_mujoco_gt),
        _robust_in_hand_rate_per_hand_aggregate(test_mujoco_gt),
        _in_hand_rate_from_records(test_mujoco_pred, pair_status="has-data"),
        _in_hand_rate_per_hand_aggregate(test_mujoco_pred),
        _robust_in_hand_rate_from_records(test_mujoco_pred, pair_status="has-data"),
        _robust_in_hand_rate_per_hand_aggregate(test_mujoco_pred),
        expected_hands=test_expected_hands,
    )

    report = {
        "train": train_rows,
        "validate": test_rows,
    }

    output_json = Path(args.output_json).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = ["# CVAE Baseline Report", ""]
    md.append(_table_md("Train Split", train_rows))
    md.append(_table_md("Validate Split", test_rows))
    output_md.write_text("\n".join(md), encoding="utf-8")

    print(f"Saved report markdown: {output_md}")
    print(f"Saved report json: {output_json}")


if __name__ == "__main__":
    main()
