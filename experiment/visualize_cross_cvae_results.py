from __future__ import annotations

import os
import sys
import time

import hydra
import torch
import trimesh
import viser
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from experiment.grasp_sample_index import (
    get_sample_for_robot,
    group_entries_by_robot,
    load_ordered_cmap_entries,
    load_ordered_results_entries,
)
def _find_dro_root(cfg_dro_root: str | None, payload: dict) -> str:
    if cfg_dro_root is not None:
        return cfg_dro_root
    meta = payload.get("meta", {})
    if isinstance(meta, dict) and "dro_root" in meta:
        return str(meta["dro_root"])
    raise ValueError("Cannot infer dro_root. Set visualization.dro_root in config.")


def _mesh_path_from_object_name(dro_root: str, object_name: str) -> str:
    dataset_name, mesh_name = object_name.split("+")
    return os.path.join(
        dro_root,
        "data/data_urdf/object",
        dataset_name,
        mesh_name,
        f"{mesh_name}.stl",
    )


def _to_tensor_q(sample: dict, key: str) -> torch.Tensor | None:
    q = sample.get(key)
    if q is None:
        return None
    if torch.is_tensor(q):
        return q.float().cpu()
    return torch.tensor(q, dtype=torch.float32)


@hydra.main(version_base="1.2", config_path="conf", config_name="config_visualize_cross_cvae")
def main(cfg: DictConfig) -> None:
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    vis_cfg = cfg.visualization
    input_mode = str(vis_cfg.get("input_mode", "results")).lower()
    filter_robots = None if vis_cfg.robot_names is None else [str(x) for x in vis_cfg.robot_names]
    max_samples = int(vis_cfg.max_samples)
    max_samples_per_robot = int(vis_cfg.get("max_samples_per_robot", -1))

    if input_mode == "results":
        payload, entries, robot_names = load_ordered_results_entries(
            str(vis_cfg.results_path),
            robot_names=filter_robots,
            max_samples=max_samples,
            max_samples_per_robot=max_samples_per_robot,
        )
        dro_root = _find_dro_root(
            None if vis_cfg.dro_root is None else to_absolute_path(str(vis_cfg.dro_root)),
            payload,
        )
    elif input_mode == "cmap_gt":
        if vis_cfg.dro_root is None:
            raise ValueError("visualization.dro_root must be set for input_mode=cmap_gt")
        dro_root = to_absolute_path(str(vis_cfg.dro_root))
        cmap_object_names = None
        cmap_cfg = vis_cfg.get("cmap_gt", {})
        if cmap_cfg.get("object_names") is not None:
            cmap_object_names = [str(x) for x in cmap_cfg.object_names]
        payload, entries, robot_names = load_ordered_cmap_entries(
            dro_root=dro_root,
            split=str(cmap_cfg.get("split", "validate")),
            robot_names=filter_robots,
            object_names=cmap_object_names,
            max_samples_per_robot=max_samples_per_robot,
        )
    else:
        raise ValueError(f"Unsupported visualization.input_mode: {input_mode}")

    if len(entries) == 0:
        raise RuntimeError("No samples left after filtering.")

    # Import DRO hand model lazily after root resolution.
    sys.path.append(dro_root)
    from utils.hand_model import create_hand_model  # type: ignore

    entries_by_robot = group_entries_by_robot(entries)
    max_ordered_idx = max(len(rows) for rows in entries_by_robot.values()) - 1

    hand_cache: dict[str, object] = {}
    object_mesh_cache: dict[str, trimesh.Trimesh] = {}

    server = viser.ViserServer(
        host=str(cfg.visualization.host),
        port=int(cfg.visualization.port),
    )

    any_gt = input_mode == "cmap_gt" or any(
        ("q_gt" in entry["raw_sample"] or "q_gt_padded" in entry["raw_sample"]) for entry in entries
    )
    mode_label = "display_mode (0=pred,1=gt,2=both)"
    mode_max = 2 if any_gt else 0
    if input_mode == "cmap_gt":
        mode_label = "display_mode (0=grasp)"
        mode_max = 0

    initial_robot_index = int(vis_cfg.get("initial_robot_index", 0))
    initial_robot_name = vis_cfg.get("initial_robot_name")
    if initial_robot_name is not None:
        initial_robot_name = str(initial_robot_name)
        if initial_robot_name in robot_names:
            initial_robot_index = robot_names.index(initial_robot_name)
    initial_robot_index = max(0, min(initial_robot_index, len(robot_names) - 1))

    robot_slider = server.gui.add_slider(
        label="robot_index",
        min=0,
        max=max(0, len(robot_names) - 1),
        step=1,
        initial_value=initial_robot_index,
    )
    sample_slider = server.gui.add_slider(
        label="ordered_idx",
        min=0,
        max=max(0, max_ordered_idx),
        step=1,
        initial_value=max(0, int(vis_cfg.get("initial_ordered_idx", 0))),
    )
    mode_slider = server.gui.add_slider(
        label=mode_label,
        min=0,
        max=mode_max,
        step=1,
        initial_value=min(int(cfg.visualization.initial_display_mode), mode_max),
    )

    def _get_hand(robot_name: str):
        if robot_name not in hand_cache:
            hand_cache[robot_name] = create_hand_model(robot_name)
        return hand_cache[robot_name]

    def _get_mesh(object_name: str) -> trimesh.Trimesh:
        if object_name not in object_mesh_cache:
            path = _mesh_path_from_object_name(dro_root, object_name)
            object_mesh_cache[object_name] = trimesh.load_mesh(path)
        return object_mesh_cache[object_name]

    def _selected_robot_name() -> str:
        return robot_names[max(0, min(int(robot_slider.value), len(robot_names) - 1))]

    def _render(_robot_idx: int, display_mode: int) -> None:
        robot_name = _selected_robot_name()
        robot_entries = entries_by_robot[robot_name]
        requested_ordered_idx = int(sample_slider.value)
        actual_ordered_idx = max(0, min(requested_ordered_idx, len(robot_entries) - 1))
        if actual_ordered_idx != requested_ordered_idx:
            sample_slider.value = actual_ordered_idx
            return

        entry = get_sample_for_robot(entries_by_robot, robot_name, actual_ordered_idx)
        sample = entry["raw_sample"]
        object_name = str(entry["object_name"])
        dof = int(entry["dof"])

        object_mesh = _get_mesh(object_name)
        server.scene.add_mesh_simple(
            "object",
            object_mesh.vertices,
            object_mesh.faces,
            color=(239, 132, 167),
            opacity=float(cfg.visualization.object_opacity),
        )

        hand = _get_hand(robot_name)

        q_pred = _to_tensor_q(sample, "q_pred")
        q_gt = _to_tensor_q(sample, "q_gt")
        if input_mode == "cmap_gt":
            q_gt = _to_tensor_q(sample, "q_gt")
            if q_gt is None:
                q_gt = _to_tensor_q(entry, "q_gt")

        if q_pred is None:
            q_pred_padded = _to_tensor_q(sample, "q_pred_padded")
            if q_pred_padded is not None:
                q_pred = q_pred_padded[:dof]

        if q_gt is None:
            q_gt_padded = _to_tensor_q(sample, "q_gt_padded")
            if q_gt_padded is not None:
                q_gt = q_gt_padded[:dof]

        show_pred = display_mode in (0, 2)
        show_gt = display_mode in (1, 2)
        if input_mode == "cmap_gt":
            show_pred = False
            show_gt = True
        # For no-data entries, GT may be missing; always show generated if available.
        if q_gt is None and q_pred is not None:
            show_pred = True
            show_gt = False

        if show_pred and q_pred is not None:
            pred_mesh = hand.get_trimesh_q(q_pred)["visual"]
            server.scene.add_mesh_simple(
                "robot_pred",
                pred_mesh.vertices,
                pred_mesh.faces,
                color=tuple(cfg.visualization.pred_color),
                opacity=float(cfg.visualization.hand_opacity),
            )
        else:
            server.scene.add_mesh_simple(
                "robot_pred",
                torch.zeros((0, 3)).numpy(),
                torch.zeros((0, 3), dtype=torch.int32).numpy(),
                color=(0, 0, 0),
                opacity=0.0,
            )

        if show_gt and q_gt is not None:
            gt_mesh = hand.get_trimesh_q(q_gt)["visual"]
            server.scene.add_mesh_simple(
                "robot_gt",
                gt_mesh.vertices,
                gt_mesh.faces,
                color=tuple(cfg.visualization.gt_color),
                opacity=float(cfg.visualization.hand_opacity),
            )
        else:
            server.scene.add_mesh_simple(
                "robot_gt",
                torch.zeros((0, 3)).numpy(),
                torch.zeros((0, 3), dtype=torch.int32).numpy(),
                color=(0, 0, 0),
                opacity=0.0,
            )

        print(
            f"[robot={robot_name} ordered_idx={actual_ordered_idx}] "
            f"object={object_name} dof={dof} input_mode={input_mode} mode={display_mode} "
            f"sample_id={sample.get('sample_id', 'n/a')} global_index={entry.get('global_index', 'n/a')} "
            f"dataset_index={entry.get('dataset_index', 'n/a')} mse_valid={sample.get('mse_valid', 'n/a')}"
        )

    robot_slider.on_update(lambda _: _render(int(robot_slider.value), int(mode_slider.value)))
    sample_slider.on_update(lambda _: _render(int(robot_slider.value), int(mode_slider.value)))
    mode_slider.on_update(lambda _: _render(int(robot_slider.value), int(mode_slider.value)))

    _render(int(robot_slider.value), int(mode_slider.value))
    print(f"Viewer URL: http://{cfg.visualization.host}:{cfg.visualization.port}")
    if bool(vis_cfg.get("exit_after_first_render", False)):
        return
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
