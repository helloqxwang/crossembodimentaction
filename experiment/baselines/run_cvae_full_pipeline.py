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

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PolicySpec:
    label: str
    policy_type: str
    robot_name: str | None
    validate_overrides: tuple[str, ...]


def _run(cmd: Sequence[str], *, workdir: Path) -> None:
    print("[RUN]", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(list(cmd), cwd=str(workdir), check=True)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _policy_specs(cfg: DictConfig) -> list[PolicySpec]:
    out: list[PolicySpec] = []
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
    return root_dir / label / split


def _json_sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


@hydra.main(version_base="1.2", config_path="../conf/baselines", config_name="run_cvae_full_pipeline")
def main(cfg: DictConfig) -> None:
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    repr_python = to_absolute_path(str(cfg.env.repr_python))
    sim_python = to_absolute_path(str(cfg.env.sim_python))
    validate_script = ROOT_DIR / "experiment" / "baselines" / "validate_cvae_baseline.py"
    sim_script = ROOT_DIR / "experiment" / "mujoco_grasp_test.py"
    report_script = ROOT_DIR / "experiment" / "make_baseline_report.py"
    count_script = ROOT_DIR / "experiment" / "pt_to_grasp_count_csv.py"

    root_dir = Path(to_absolute_path(str(cfg.outputs.root_dir))).resolve()
    manifest_path = Path(to_absolute_path(str(cfg.outputs.manifest_path))).resolve()
    report_md = Path(to_absolute_path(str(cfg.outputs.report_md))).resolve()
    report_json = Path(to_absolute_path(str(cfg.outputs.report_json))).resolve()
    for path in (manifest_path, report_md, report_json):
        _ensure_parent(path)

    policies = _policy_specs(cfg)
    splits = [str(x) for x in cfg.splits]
    validate_common = [str(x) for x in cfg.validate.common_overrides]
    sim_common = [str(x) for x in cfg.sim.common_overrides]

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root_dir": str(root_dir),
        "splits": splits,
        "policies": {},
        "gt": {},
    }

    for split in splits:
        gt_dir = _split_dir(root_dir, "gt", split)
        sim_gt_pt = gt_dir / "sim_gt.pt"
        _ensure_parent(sim_gt_pt)
        _run(
            [
                sim_python,
                str(sim_script),
                *sim_common,
                "input.mode=cmap_gt",
                f"input.cmap_gt.split={split}",
                f"output.save_path={sim_gt_pt}",
            ],
            workdir=ROOT_DIR,
        )
        manifest["gt"][split] = {
            "sim_gt_pt": str(sim_gt_pt),
            "sim_gt_json": str(_json_sidecar(sim_gt_pt)),
        }

    for policy in policies:
        policy_entry = {
            "type": policy.policy_type,
            "robot_name": policy.robot_name,
            "splits": {},
        }
        manifest["policies"][policy.label] = policy_entry
        for split in splits:
            split_dir = _split_dir(root_dir, policy.label, split)
            validate_pt = split_dir / "generated_grasps.pt"
            count_csv = split_dir / "count_table.csv"
            sim_pred_pt = split_dir / "sim_pred.pt"
            for path in (validate_pt, count_csv, sim_pred_pt):
                _ensure_parent(path)

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

            policy_entry["splits"][split] = {
                "validate_pt": str(validate_pt),
                "validate_json": str(_json_sidecar(validate_pt)),
                "count_csv": str(count_csv),
                "sim_pred_pt": str(sim_pred_pt),
                "sim_pred_json": str(_json_sidecar(sim_pred_pt)),
            }

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
