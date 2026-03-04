from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rank sampler experiments with blended IoU/span score.")
    p.add_argument(
        "--eval-root",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "coverage_eval_explore"),
    )
    p.add_argument("--output-json", type=str, default="")
    return p


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _score_robot(m: Dict[str, float]) -> Dict[str, float]:
    cov03 = float(m.get("coverage_iou_ge_03", 0.0))
    cov05 = float(m.get("coverage_iou_ge_05", 0.0))
    med = float(m.get("median_best_iou", 0.0))

    span2d = float(m.get("pc2d_span_ratio_mean", 0.0))
    span_inter = float(m.get("pc2d_intersection_cover_mean", 0.0))
    tr = float(m.get("span_trace_ratio", 0.0))
    ld = float(m.get("span_logdet_ratio", 0.0))
    nn_ratio = float(m.get("nn_p95_ratio", 2.0))

    iou_score = cov03 + 1.6 * cov05 + 0.7 * med
    span_score = (
        0.35 * _clip(span2d, 0.0, 1.2)
        + 0.25 * _clip(span_inter, 0.0, 1.2)
        + 0.25 * _clip(tr, 0.0, 1.2)
        + 0.15 * _clip(ld, 0.0, 1.2)
    )
    # If real->sampled nearest-neighbor tails are much larger than real->real, down-weight.
    ood_factor = 1.0 / (1.0 + max(0.0, nn_ratio - 1.0))
    blended = (0.6 * iou_score + 0.4 * span_score) * ood_factor
    return {
        "iou_score": float(iou_score),
        "span_score": float(span_score),
        "ood_factor": float(ood_factor),
        "blended_score": float(blended),
    }


def main() -> None:
    args = _build_parser().parse_args()
    root = os.path.abspath(args.eval_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(root)

    results: List[Dict[str, object]] = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "summary_metrics.json")
        if not os.path.exists(p):
            continue
        metrics = json.load(open(p, "r", encoding="utf-8"))
        if not isinstance(metrics, dict) or len(metrics) == 0:
            continue
        robot_scores = {r: _score_robot(m) for r, m in metrics.items()}
        mean_blended = sum(v["blended_score"] for v in robot_scores.values()) / len(robot_scores)
        mean_iou = sum(v["iou_score"] for v in robot_scores.values()) / len(robot_scores)
        mean_span = sum(v["span_score"] for v in robot_scores.values()) / len(robot_scores)
        mean_ood = sum(v["ood_factor"] for v in robot_scores.values()) / len(robot_scores)
        results.append(
            {
                "experiment": name,
                "mean_blended_score": float(mean_blended),
                "mean_iou_score": float(mean_iou),
                "mean_span_score": float(mean_span),
                "mean_ood_factor": float(mean_ood),
                "robot_scores": robot_scores,
            }
        )

    results.sort(key=lambda x: x["mean_blended_score"], reverse=True)
    for i, r in enumerate(results, 1):
        print(
            f"{i:02d}. {r['experiment']}: blended={r['mean_blended_score']:.4f} "
            f"(iou={r['mean_iou_score']:.4f}, span={r['mean_span_score']:.4f}, ood={r['mean_ood_factor']:.4f})"
        )

    if args.output_json:
        out_path = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()

