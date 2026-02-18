from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]


def _run(cmd: Sequence[str], *, workdir: Path) -> None:
    print("[RUN]", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(list(cmd), cwd=str(workdir), check=True)


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_run_cross_cvae_full_pipeline")
def main(cfg: DictConfig) -> None:
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    repr_python = to_absolute_path(str(cfg.env.repr_python))
    mujoco_python = to_absolute_path(str(cfg.env.mujoco_python))

    train_validate_pt = to_absolute_path(str(cfg.outputs.train_validate_pt))
    test_validate_pt = to_absolute_path(str(cfg.outputs.test_validate_pt))
    train_mujoco_pred_pt = to_absolute_path(str(cfg.outputs.train_mujoco_pred_pt))
    test_mujoco_pred_pt = to_absolute_path(str(cfg.outputs.test_mujoco_pred_pt))
    train_mujoco_gt_pt = to_absolute_path(str(cfg.outputs.train_mujoco_gt_pt))
    test_mujoco_gt_pt = to_absolute_path(str(cfg.outputs.test_mujoco_gt_pt))
    report_md = to_absolute_path(str(cfg.outputs.report_md))
    report_json = to_absolute_path(str(cfg.outputs.report_json))

    Path(train_validate_pt).parent.mkdir(parents=True, exist_ok=True)
    Path(test_validate_pt).parent.mkdir(parents=True, exist_ok=True)
    Path(train_mujoco_pred_pt).parent.mkdir(parents=True, exist_ok=True)
    Path(test_mujoco_pred_pt).parent.mkdir(parents=True, exist_ok=True)
    Path(train_mujoco_gt_pt).parent.mkdir(parents=True, exist_ok=True)
    Path(test_mujoco_gt_pt).parent.mkdir(parents=True, exist_ok=True)
    Path(report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(report_json).parent.mkdir(parents=True, exist_ok=True)

    validate_script = ROOT_DIR / "experiment" / "validate_cross_cvae.py"
    vis_script = ROOT_DIR / "experiment" / "visualize_cross_cvae_results.py"
    mujoco_script = ROOT_DIR / "experiment" / "mujoco_test_cross_cvae.py"
    report_script = ROOT_DIR / "experiment" / "make_cross_cvae_report.py"
    count_script = ROOT_DIR / "experiment" / "pt_to_grasp_count_csv.py"

    train_count_csv = to_absolute_path(str(cfg.outputs.train_count_csv))
    test_count_csv = to_absolute_path(str(cfg.outputs.test_count_csv))
    Path(train_count_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(test_count_csv).parent.mkdir(parents=True, exist_ok=True)

    validate_common = [str(x) for x in cfg.validate.common_overrides]
    validate_train = [str(x) for x in cfg.validate.train_overrides]
    validate_test = [str(x) for x in cfg.validate.test_overrides]

    _run(
        [
            repr_python,
            str(validate_script),
            *validate_common,
            *validate_train,
            f"output.save_path={train_validate_pt}",
        ],
        workdir=ROOT_DIR,
    )
    _run(
        [
            repr_python,
            str(validate_script),
            *validate_common,
            *validate_test,
            f"output.save_path={test_validate_pt}",
        ],
        workdir=ROOT_DIR,
    )
    _run(
        [
            repr_python,
            str(count_script),
            "--input-pt",
            train_validate_pt,
            "--output-csv",
            train_count_csv,
        ],
        workdir=ROOT_DIR,
    )
    _run(
        [
            repr_python,
            str(count_script),
            "--input-pt",
            test_validate_pt,
            "--output-csv",
            test_count_csv,
        ],
        workdir=ROOT_DIR,
    )

    vis_procs: list[subprocess.Popen[str]] = []
    if bool(cfg.visualization.launch):
        train_vis_cmd = [
            repr_python,
            str(vis_script),
            f"visualization.results_path={train_validate_pt}",
            f"visualization.port={int(cfg.visualization.train_port)}",
            *[str(x) for x in cfg.visualization.extra_overrides],
        ]
        test_vis_cmd = [
            repr_python,
            str(vis_script),
            f"visualization.results_path={test_validate_pt}",
            f"visualization.port={int(cfg.visualization.test_port)}",
            *[str(x) for x in cfg.visualization.extra_overrides],
        ]
        print("[SPAWN]", " ".join(shlex.quote(x) for x in train_vis_cmd))
        vis_procs.append(subprocess.Popen(train_vis_cmd, cwd=str(ROOT_DIR)))
        print("[SPAWN]", " ".join(shlex.quote(x) for x in test_vis_cmd))
        vis_procs.append(subprocess.Popen(test_vis_cmd, cwd=str(ROOT_DIR)))

    mujoco_common = [str(x) for x in cfg.mujoco.common_overrides]
    _run(
        [
            mujoco_python,
            str(mujoco_script),
            *mujoco_common,
            "input.mode=generated",
            "input.generated.q_key=q_pred",
            f"input.generated.results_path={train_validate_pt}",
            f"output.save_path={train_mujoco_pred_pt}",
        ],
        workdir=ROOT_DIR,
    )
    _run(
        [
            mujoco_python,
            str(mujoco_script),
            *mujoco_common,
            "input.mode=generated",
            "input.generated.q_key=q_pred",
            f"input.generated.results_path={test_validate_pt}",
            f"output.save_path={test_mujoco_pred_pt}",
        ],
        workdir=ROOT_DIR,
    )
    _run(
        [
            mujoco_python,
            str(mujoco_script),
            *mujoco_common,
            "input.mode=generated",
            "input.generated.q_key=q_gt",
            "input.generated.source_filter=[pred]",
            f"input.generated.results_path={train_validate_pt}",
            f"output.save_path={train_mujoco_gt_pt}",
        ],
        workdir=ROOT_DIR,
    )
    _run(
        [
            mujoco_python,
            str(mujoco_script),
            *mujoco_common,
            "input.mode=generated",
            "input.generated.q_key=q_gt",
            "input.generated.source_filter=[pred]",
            f"input.generated.results_path={test_validate_pt}",
            f"output.save_path={test_mujoco_gt_pt}",
        ],
        workdir=ROOT_DIR,
    )

    _run(
        [
            repr_python,
            str(report_script),
            "--train-validate-pt",
            train_validate_pt,
            "--test-validate-pt",
            test_validate_pt,
            "--train-mujoco-pred-pt",
            train_mujoco_pred_pt,
            "--test-mujoco-pred-pt",
            test_mujoco_pred_pt,
            "--train-mujoco-gt-pt",
            train_mujoco_gt_pt,
            "--test-mujoco-gt-pt",
            test_mujoco_gt_pt,
            "--output-md",
            report_md,
            "--output-json",
            report_json,
        ],
        workdir=ROOT_DIR,
    )

    print("\n=== Full Pipeline Done ===")
    print(f"train_validate_pt: {train_validate_pt}")
    print(f"test_validate_pt: {test_validate_pt}")
    print(f"train_mujoco_pred_pt: {train_mujoco_pred_pt}")
    print(f"test_mujoco_pred_pt: {test_mujoco_pred_pt}")
    print(f"train_mujoco_gt_pt: {train_mujoco_gt_pt}")
    print(f"test_mujoco_gt_pt: {test_mujoco_gt_pt}")
    print(f"train_count_csv: {train_count_csv}")
    print(f"test_count_csv: {test_count_csv}")
    print(f"report_md: {report_md}")
    print(f"report_json: {report_json}")
    if vis_procs:
        print("Visualization processes:")
        for p in vis_procs:
            print(f"  pid={p.pid}")


if __name__ == "__main__":
    sys.exit(main())
