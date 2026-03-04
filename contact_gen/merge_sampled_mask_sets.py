from __future__ import annotations

import argparse
import os
from typing import List

import torch


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge two sampled-mask directories with a fixed ratio.")
    p.add_argument("--dir-a", type=str, required=True, help="Directory A with *_random_masks.pt files.")
    p.add_argument("--dir-b", type=str, required=True, help="Directory B with *_random_masks.pt files.")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    p.add_argument("--ratio-a", type=float, default=0.5, help="Fraction from A in [0,1].")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--robot-names",
        type=str,
        nargs="*",
        default=["allegro", "barrett", "ezgripper", "robotiq_3finger", "shadowhand"],
    )
    p.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Per-robot output count. 0 means min(count_a, count_b).",
    )
    return p


def _load_payload(path: str) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid payload: {path}")
    if "sampled_masks" not in payload or "sampled_q" not in payload:
        raise ValueError(f"Missing sampled fields in: {path}")
    return payload


def _merge_robot(
    payload_a: dict,
    payload_b: dict,
    ratio_a: float,
    out_count: int,
    seed: int,
) -> dict:
    masks_a = payload_a["sampled_masks"].bool()
    masks_b = payload_b["sampled_masks"].bool()
    q_a = payload_a["sampled_q"].float()
    q_b = payload_b["sampled_q"].float()

    if int(masks_a.shape[1]) != int(masks_b.shape[1]):
        raise ValueError("Mask point dimensions mismatch between sources.")
    if int(q_a.shape[1]) != int(q_b.shape[1]):
        raise ValueError("q dimensions mismatch between sources.")

    max_count = min(int(masks_a.shape[0]), int(masks_b.shape[0]))
    n = int(max_count if out_count <= 0 else min(out_count, max_count))
    if n <= 0:
        raise ValueError("No samples available to merge.")

    n_a = int(round(float(ratio_a) * float(n)))
    n_a = max(0, min(n, n_a))
    n_b = n - n_a

    g = torch.Generator().manual_seed(int(seed))
    idx_a = torch.randperm(int(masks_a.shape[0]), generator=g)[:n_a]
    idx_b = torch.randperm(int(masks_b.shape[0]), generator=g)[:n_b]

    masks = torch.cat([masks_a[idx_a], masks_b[idx_b]], dim=0)
    q = torch.cat([q_a[idx_a], q_b[idx_b]], dim=0)
    perm = torch.randperm(int(masks.shape[0]), generator=g)
    masks = masks[perm]
    q = q[perm]

    meta = dict(payload_a.get("meta", {}))
    meta["merge_ratio_a"] = float(ratio_a)
    meta["merge_sources"] = [
        str(payload_a.get("meta", {}).get("sampling_mode", "source_a")),
        str(payload_b.get("meta", {}).get("sampling_mode", "source_b")),
    ]
    meta["merge_note"] = "merged_by_merge_sampled_mask_sets.py"

    counts = masks.sum(dim=1).float()
    summary = {
        "sampled_num_samples": int(masks.shape[0]),
        "sampled_count_mean": float(counts.mean().item()),
        "sampled_count_std": float(counts.std(unbiased=False).item()),
    }

    return {
        "meta": meta,
        "summary": summary,
        "sampled_masks": masks,
        "sampled_q": q,
        "visualization_samples": [],
    }


def main() -> None:
    args = _build_parser().parse_args()
    ratio_a = float(max(0.0, min(1.0, float(args.ratio_a))))
    os.makedirs(args.output_dir, exist_ok=True)

    for i, robot_name in enumerate(args.robot_names):
        a_path = os.path.join(args.dir_a, f"{robot_name}_random_masks.pt")
        b_path = os.path.join(args.dir_b, f"{robot_name}_random_masks.pt")
        if not os.path.exists(a_path):
            print(f"[skip] {robot_name}: missing {a_path}")
            continue
        if not os.path.exists(b_path):
            print(f"[skip] {robot_name}: missing {b_path}")
            continue

        pa = _load_payload(a_path)
        pb = _load_payload(b_path)
        merged = _merge_robot(
            payload_a=pa,
            payload_b=pb,
            ratio_a=ratio_a,
            out_count=int(args.sample_count),
            seed=int(args.seed) + i * 997,
        )
        out_path = os.path.join(args.output_dir, f"{robot_name}_random_masks.pt")
        torch.save(merged, out_path)
        print(f"[saved] {out_path} summary={merged['summary']}")


if __name__ == "__main__":
    main()
