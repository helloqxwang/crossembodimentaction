# Experiment

This folder contains the active training, validation, visualization, and MuJoCo testing pipeline for grasp generation.

## Environments

- `repr` env: training, model validation, Viser visualization, report generation
- `spider` env: MuJoCo batch testing and single-sample rerun play

All active Hydra configs live under `experiment/conf/`.

## Main Pipelines

### 1. Model pipeline

`train_* -> checkpoint + dataset_meta.pt -> validate_cvae_baseline.py -> generated_grasps.pt`

- Cross-embodiment training: `baselines/train_cross_embodiment_cvae.py`
- Single-embodiment training: `baselines/train_single_embodiment_cvae.py`
- Shared validation: `baselines/validate_cvae_baseline.py`

The validator writes a unified `generated_grasps.pt` format. That file is the standard input for visualization, MuJoCo testing, and reporting.

### 2. GT pipeline

`DRO-Grasp filtered CMap .pt -> visualize_cross_cvae_results.py or mujoco_grasp_test.py`

This path bypasses model validation and uses DRO-Grasp filtered grasps directly.

## Ordered Sample Selection

The shared selector is:

- `robot_name`
- `ordered_idx`

This is used consistently across Viser, MuJoCo batch testing, and rerun play. The indexing logic lives in `grasp_sample_index.py`.

## Main Scripts

### Training and validation

- `baselines/train_cross_embodiment_cvae.py`
  - Train one cross-embodiment CVAE.
  - Output: checkpoint directory and `dataset_meta.pt`.

- `baselines/train_single_embodiment_cvae.py`
  - Train one single-embodiment CVAE.
  - Output: checkpoint directory and `dataset_meta.pt`.

- `baselines/validate_cvae_baseline.py`
  - Run a trained baseline on a split and save `generated_grasps.pt`.
  - Input: checkpoint(s), dataset metadata, DRO-Grasp assets.

- `baselines/run_cvae_full_pipeline.py`
  - Orchestrate validation, count CSV export, simulation testing, and comparison report generation across models.

### Visualization

- `visualize_cross_cvae_results.py`
  - Viser viewer for either:
    - validation results (`input_mode=results`)
    - DRO-Grasp GT (`input_mode=cmap_gt`)
  - Main controls: `robot_index`, `ordered_idx`, `display_mode`
  - Uses `repr` env.

### Simulation

- `mujoco_grasp_test.py`
  - Headless MuJoCo batch testing.
  - Input modes:
    - `generated`: use `generated_grasps.pt`
    - `cmap_gt`: use DRO-Grasp filtered dataset
  - Output: `mujoco_eval.pt` and `mujoco_eval.json`
  - Uses `spider` env.

- `replay_grasp_sample.py`
  - Play one selected sample live in MuJoCo and visualize it in rerun.
  - Inputs:
    - `--robot-name`
    - `--sample-idx`
    - `--data-pt-path`
  - Reads sim hyperparameters from `config_mujoco_grasp_test.yaml`.
  - Uses `spider` env.

### Reporting and utilities

- `make_baseline_report.py`
  - Merge multi-model validation and sim outputs into a comparison report.

- `pt_to_grasp_count_csv.py`
  - Export a hand-object count table from a `.pt` file.

## Sim Core

The local MuJoCo implementation is under `sim/`.

- `sim/grasp_validation.py`
  - Shared CPU validation core
  - Validation model construction
  - Gravity toggle
  - Disturbance testing

- `sim/mjwarp_batch.py`
  - Batched `mjwarp` backend for fast headless testing

- `sim/squeeze_control.py`
  - Shared `outer_q` / `inner_q` squeeze controller
  - Used by both CPU and `mjwarp` paths

- `sim/assets.py`
  - Hand URDF patching for MuJoCo

- `sim/rerun_scene.py`
  - Rerun scene logging for single-sample play

- `sim/sample_io.py`
  - Load source samples from result payloads or DRO-Grasp CMap payloads

## Important Configs

- `conf/config_mujoco_grasp_test.yaml`
  - Batch sim and play settings
  - Includes:
    - `mujoco.gravity_enabled`
    - `mujoco.squeezing.enabled`
    - disturbance settings

- `conf/config_visualize_cross_cvae.yaml`
  - Viser settings and input mode

- `conf/baselines/*.yaml`
  - Training / validation / full-pipeline configs

## Typical Commands

Cross-embodiment training:

```bash
python experiment/baselines/train_cross_embodiment_cvae.py
```

Single-embodiment training:

```bash
python experiment/baselines/train_single_embodiment_cvae.py
```

Validation to `generated_grasps.pt`:

```bash
python experiment/baselines/validate_cvae_baseline.py
```

GT Viser:

```bash
python experiment/visualize_cross_cvae_results.py visualization.input_mode=cmap_gt
```

GT MuJoCo batch test:

```bash
python experiment/mujoco_grasp_test.py input.mode=cmap_gt
```

Play one sample in rerun:

```bash
python experiment/replay_grasp_sample.py \
  --robot-name allegro \
  --sample-idx 0 \
  --data-pt-path /path/to/data.pt
```

## Notes

- `generated_grasps.pt` and DRO-Grasp `cmap_dataset.pt` are valid source inputs.
- MuJoCo eval payloads (`records`) are report outputs, not play inputs.
- `mujoco_test_cross_cvae.py` is kept as a compatibility wrapper for the old command name.
- `experiment/conf/legacy/` contains old config files kept only for reference.
