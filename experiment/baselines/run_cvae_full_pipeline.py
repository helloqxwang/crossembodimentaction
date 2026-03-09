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

ROOT_DIR = Path(__file__).resolve().parents[2]


def _run(cmd: Sequence[str], *, workdir: Path) -> None:
    print("[RUN]", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(list(cmd), cwd=str(workdir), check=True)


def _baseline_validate_overrides(cfg: DictConfig) -> list[str]:
    baseline_type = str(cfg.baseline.type)
    overrides = [f"baseline.type={baseline_type}"]
    if baseline_type == "cross_embodiment":
        overrides.append(f"model.cross.ckpt_path={to_absolute_path(str(cfg.model.cross.ckpt_path))}")
        if cfg.model.cross.dataset_meta_path is not None:
            overrides.append(
                f"model.cross.dataset_meta_path={to_absolute_path(str(cfg.model.cross.dataset_meta_path))}"
            )
    elif baseline_type == "single_embodiment":
        overrides.append(
            f"model.single.checkpoint_root={to_absolute_path(str(cfg.model.single.checkpoint_root))}"
        )
        overrides.append(f"model.single.checkpoint_name={str(cfg.model.single.checkpoint_name)}")
        overrides.append(f"model.single.dataset_meta_name={str(cfg.model.single.dataset_meta_name)}")
    else:
        raise ValueError(f"Unsupported baseline.type: {baseline_type}")
    return overrides


@hydra.main(version_base="1.2", config_path="../conf/baselines", config_name="run_cvae_full_pipeline")
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

    for path in [
        train_validate_pt,
        test_validate_pt,
        train_mujoco_pred_pt,
        test_mujoco_pred_pt,
        train_mujoco_gt_pt,
        test_mujoco_gt_pt,
        report_md,
        report_json,
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    validate_script = ROOT_DIR / "experiment" / "baselines" / "validate_cvae_baseline.py"
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
    validate_baseline = _baseline_validate_overrides(cfg)

    _run(
        [
            repr_python,
            str(validate_script),
            *validate_baseline,
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
            *validate_baseline,
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
    print(f"baseline_type: {cfg.baseline.type}")
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
        for proc in vis_procs:
            print(f"  pid={proc.pid}")


if __name__ == "__main__":
    sys.exit(main())
