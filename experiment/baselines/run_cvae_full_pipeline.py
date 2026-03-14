"""End-to-end CVAE baseline evaluation pipeline.

Orchestrates: grasp generation (validate) -> grasp counting -> MuJoCo simulation
-> report generation, for every (policy, split) combination specified in the config.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[2]  # project root (two levels up)


@dataclass(frozen=True)
class PolicySpec:
    """Immutable descriptor for one policy to evaluate."""

    label: str                              # human-readable tag used in output paths
    policy_type: str                        # "cross_embodiment" or "single_embodiment"
    robot_name: str | None                  # None for cross-embodiment policies
    validate_overrides: tuple[str, ...]     # Hydra CLI overrides for the validate script


def _run(cmd: Sequence[str], *, workdir: Path) -> None:
    """Print and execute a shell command, raising on non-zero exit."""
    print("[RUN]", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(list(cmd), cwd=str(workdir), check=True)


def _ensure_parent(path: Path) -> None:
    """Create parent directories of *path* if they don't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _policy_specs(cfg: DictConfig) -> list[PolicySpec]:
    """Build the list of PolicySpec objects from the Hydra config.

    Reads the ``policies.cross`` and ``policies.single`` sections and
    produces one spec per enabled policy (one cross-embodiment policy and/or
    one spec per robot for single-embodiment).
    """
    out: list[PolicySpec] = []

    # --- cross-embodiment policy (at most one) ---
    if bool(cfg.policies.cross.enabled):
        overrides = [
            "policy.type=cross_embodiment",
            f"policy_paths.cross.ckpt_path={to_absolute_path(str(cfg.policies.cross.ckpt_path))}",
        ]
        if cfg.policies.cross.dataset_meta_path is not None:
            overrides.append(
                f"policy_paths.cross.dataset_meta_path={to_absolute_path(str(cfg.policies.cross.dataset_meta_path))}"
            )
        out.append(
            PolicySpec(
                label=str(cfg.policies.cross.label),
                policy_type="cross_embodiment",
                robot_name=None,
                validate_overrides=tuple(overrides),
            )
        )

    # --- single-embodiment policies (one per robot) ---
    if bool(cfg.policies.single.enabled):
        checkpoint_root = to_absolute_path(str(cfg.policies.single.checkpoint_root))
        checkpoint_name = str(cfg.policies.single.checkpoint_name)
        dataset_meta_name = str(cfg.policies.single.dataset_meta_name)
        for robot_name in [str(x) for x in cfg.policies.single.robot_names]:
            out.append(
                PolicySpec(
                    label=f"{str(cfg.policies.single.label_prefix)}{robot_name}",
                    policy_type="single_embodiment",
                    robot_name=robot_name,
                    validate_overrides=(
                        "policy.type=single_embodiment",
                        f"policy_paths.single.checkpoint_root={checkpoint_root}",
                        f"policy_paths.single.checkpoint_name={checkpoint_name}",
                        f"policy_paths.single.dataset_meta_name={dataset_meta_name}",
                        f"validation.robot_names=[{robot_name}]",
                    ),
                )
            )
    if not out:
        raise RuntimeError("No policies enabled in run_cvae_full_pipeline config.")
    return out


def _split_dir(root_dir: Path, label: str, split: str) -> Path:
    """Return ``root_dir/label/split`` — the per-policy, per-split output directory."""
    return root_dir / label / split


def _json_sidecar(path: Path) -> Path:
    """Return the companion .json path for a .pt file (same stem, .json suffix)."""
    return path.with_suffix(".json")


@hydra.main(version_base="1.2", config_path="../conf/baselines", config_name="run_cvae_full_pipeline")
def main(cfg: DictConfig) -> None:
    """Run the full CVAE evaluation pipeline driven by a Hydra config."""

    # ── 0. Print resolved config for reproducibility ──
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    # ── 1. Resolve Python interpreters and script paths ──
    repr_python = to_absolute_path(str(cfg.env.repr_python))      # python for representation / policy code
    sim_python = to_absolute_path(str(cfg.env.sim_python))         # python for MuJoCo simulation
    validate_script = ROOT_DIR / "experiment" / "baselines" / "validate_cvae_baseline.py"
    sim_script = ROOT_DIR / "experiment" / "mujoco_grasp_test.py"
    report_script = ROOT_DIR / "experiment" / "make_baseline_report.py"
    count_script = ROOT_DIR / "experiment" / "pt_to_grasp_count_csv.py"

    # ── 2. Resolve output paths and ensure parent dirs exist ──
    root_dir = Path(to_absolute_path(str(cfg.outputs.root_dir))).resolve()
    manifest_path = Path(to_absolute_path(str(cfg.outputs.manifest_path))).resolve()
    report_md = Path(to_absolute_path(str(cfg.outputs.report_md))).resolve()
    report_json = Path(to_absolute_path(str(cfg.outputs.report_json))).resolve()
    for path in (manifest_path, report_md, report_json):
        _ensure_parent(path)

    # ── 3. Parse policy specs and shared CLI overrides from config ──
    policies = _policy_specs(cfg)
    splits = [str(x) for x in cfg.splits]                          # e.g. ["train", "validate"]
    validate_common = [str(x) for x in cfg.validate.common_overrides]
    sim_common = [str(x) for x in cfg.sim.common_overrides]

    # Manifest tracks every artifact produced by the pipeline.
    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root_dir": str(root_dir),
        "splits": splits,
        "policies": {},
        "gt": {},
    }

    # ── 4. Ground-truth simulation: run MuJoCo sim on GT grasps per split ──
    for split in splits:
        gt_dir = _split_dir(root_dir, "gt", split)
        sim_gt_pt = gt_dir / "sim_gt.pt"
        _ensure_parent(sim_gt_pt)
        _run(
            [
                sim_python,
                str(sim_script),
                *sim_common,
                "input.mode=cmap_gt",                              # use ground-truth contact maps
                f"input.cmap_gt.split={split}",
                f"output.save_path={sim_gt_pt}",
            ],
            workdir=ROOT_DIR,
        )
        manifest["gt"][split] = {
            "sim_gt_pt": str(sim_gt_pt),
            "sim_gt_json": str(_json_sidecar(sim_gt_pt)),
        }

    # ── 5. Per-policy evaluation loop ──
    for policy in policies:
        policy_entry = {
            "type": policy.policy_type,
            "robot_name": policy.robot_name,
            "splits": {},
        }
        manifest["policies"][policy.label] = policy_entry

        for split in splits:
            split_dir = _split_dir(root_dir, policy.label, split)
            validate_pt = split_dir / "generated_grasps.pt"        # CVAE-generated grasps
            count_csv = split_dir / "count_table.csv"              # per-object grasp counts
            sim_pred_pt = split_dir / "sim_pred.pt"                # simulation results on predictions
            for path in (validate_pt, count_csv, sim_pred_pt):
                _ensure_parent(path)

            # Step A: Generate grasps with the CVAE policy
            _run(
                [
                    repr_python,
                    str(validate_script),
                    *policy.validate_overrides,
                    *validate_common,
                    f"dataset.split={split}",
                    f"output.save_path={validate_pt}",
                ],
                workdir=ROOT_DIR,
            )

            # Step B: Summarise generated grasps into a count table
            _run(
                [
                    repr_python,
                    str(count_script),
                    "--input-pt",
                    str(validate_pt),
                    "--output-csv",
                    str(count_csv),
                ],
                workdir=ROOT_DIR,
            )

            # Step C: Simulate predicted grasps in MuJoCo
            _run(
                [
                    sim_python,
                    str(sim_script),
                    *sim_common,
                    "input.mode=generated",
                    "input.generated.q_key=q_pred",
                    f"input.generated.results_path={validate_pt}",
                    f"output.save_path={sim_pred_pt}",
                ],
                workdir=ROOT_DIR,
            )

            # Record artifact paths in the manifest
            policy_entry["splits"][split] = {
                "validate_pt": str(validate_pt),
                "validate_json": str(_json_sidecar(validate_pt)),
                "count_csv": str(count_csv),
                "sim_pred_pt": str(sim_pred_pt),
                "sim_pred_json": str(_json_sidecar(sim_pred_pt)),
            }

    # ── 6. Write manifest and generate the final comparison report ──
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _run(
        [
            repr_python,
            str(report_script),
            "--manifest",
            str(manifest_path),
            "--output-md",
            str(report_md),
            "--output-json",
            str(report_json),
        ],
        workdir=ROOT_DIR,
    )

    # ── 7. Summary ──
    print("\n=== Full Pipeline Done ===")
    print(f"manifest: {manifest_path}")
    print(f"report_md: {report_md}")
    print(f"report_json: {report_json}")
    print("policies:")
    for policy in policies:
        robot_suffix = "" if policy.robot_name is None else f" robot={policy.robot_name}"
        print(f"  {policy.label}: type={policy.policy_type}{robot_suffix}")


if __name__ == "__main__":
    sys.exit(main())
