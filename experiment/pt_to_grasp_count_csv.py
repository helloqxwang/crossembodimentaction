from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def _extract_pairs(data: Any) -> list[tuple[str, str]]:
    """Return [(robot_name, object_name), ...] from common .pt layouts."""
    pairs: list[tuple[str, str]] = []

    if isinstance(data, dict) and "samples" in data:
        # Our generated validation result format.
        for sample in data["samples"]:
            robot = sample.get("robot_name")
            obj = sample.get("object_name")
            if isinstance(robot, str) and isinstance(obj, str):
                pairs.append((robot, obj))
        return pairs

    if isinstance(data, dict) and "metadata" in data:
        # DRO CMap format, e.g. (q, object_name, robot_name) or (idx, q, object_name, robot_name)
        for item in data["metadata"]:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            robot = item[-1]
            obj = item[-2]
            if isinstance(robot, str) and isinstance(obj, str):
                pairs.append((robot, obj))
        return pairs

    if isinstance(data, list):
        # Fallback: list of dict samples.
        for item in data:
            if not isinstance(item, dict):
                continue
            robot = item.get("robot_name")
            obj = item.get("object_name")
            if isinstance(robot, str) and isinstance(obj, str):
                pairs.append((robot, obj))
        return pairs

    return pairs


def _write_csv(
    *,
    out_csv: Path,
    robots: list[str],
    objects: list[str],
    counts: Counter[tuple[str, str]],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hand/object", *objects])
        for robot in robots:
            row = [robot]
            for obj in objects:
                row.append(counts[(robot, obj)])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load a .pt file and export a CSV count table. "
            "Rows: hands, Columns: objects, Entries: number of grasps."
        )
    )
    parser.add_argument("--input-pt", type=str, required=True, help="Input .pt file path.")
    parser.add_argument("--output-csv", type=str, default="_.csv", help="Output .csv file path.")
    parser.add_argument(
        "--sort-by",
        type=str,
        default="name",
        choices=["name", "count_desc"],
        help="Sort hands/objects alphabetically or by total counts.",
    )
    args = parser.parse_args()

    input_pt = Path(args.input_pt).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    if not input_pt.exists():
        raise FileNotFoundError(f"Input file not found: {input_pt}")

    data = torch.load(input_pt, map_location="cpu")
    pairs = _extract_pairs(data)
    if len(pairs) == 0:
        raise RuntimeError(
            "Could not find (robot_name, object_name) records in the input file. "
            "Expected one of: {'samples': [...]}, {'metadata': [...]}, or list of sample dicts."
        )

    counts: Counter[tuple[str, str]] = Counter(pairs)
    robot_totals: Counter[str] = Counter([r for r, _ in pairs])
    object_totals: Counter[str] = Counter([o for _, o in pairs])

    robots = sorted(robot_totals.keys())
    objects = sorted(object_totals.keys())
    if args.sort_by == "count_desc":
        robots = sorted(robot_totals.keys(), key=lambda x: (-robot_totals[x], x))
        objects = sorted(object_totals.keys(), key=lambda x: (-object_totals[x], x))

    _write_csv(out_csv=output_csv, robots=robots, objects=objects, counts=counts)

    print(f"Input: {input_pt}")
    print(f"Output CSV: {output_csv}")
    print(f"Total grasps: {len(pairs)}")
    print(f"Total hands: {len(robots)}")
    print(f"Total objects: {len(objects)}")


if __name__ == "__main__":
    main()
