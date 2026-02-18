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


def _load_payload(path: str) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}")
    if "samples" not in payload:
        raise KeyError(f"Payload at {path} missing 'samples'")
    return payload


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


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_visualize_cross_cvae")
def main(cfg: DictConfig) -> None:
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    results_path = to_absolute_path(str(cfg.visualization.results_path))
    payload = _load_payload(results_path)
    all_samples = list(payload["samples"])
    if len(all_samples) == 0:
        raise RuntimeError(f"No samples in results file: {results_path}")

    dro_root = _find_dro_root(
        None if cfg.visualization.dro_root is None else to_absolute_path(str(cfg.visualization.dro_root)),
        payload,
    )

    # Import DRO hand model lazily after root resolution.
    sys.path.append(dro_root)
    from utils.hand_model import create_hand_model  # type: ignore

    filter_robots = None if cfg.visualization.robot_names is None else set(cfg.visualization.robot_names)
    samples = []
    for s in all_samples:
        if filter_robots is not None and s["robot_name"] not in filter_robots:
            continue
        samples.append(s)
    if len(samples) == 0:
        raise RuntimeError("No samples left after filtering by robot_names.")

    if int(cfg.visualization.max_samples) >= 0:
        samples = samples[: int(cfg.visualization.max_samples)]

    hand_cache: dict[str, object] = {}
    object_mesh_cache: dict[str, trimesh.Trimesh] = {}

    server = viser.ViserServer(
        host=str(cfg.visualization.host),
        port=int(cfg.visualization.port),
    )

    any_gt = any(("q_gt" in s or "q_gt_padded" in s) for s in samples)
    mode_max = 2 if any_gt else 0
    sample_slider = server.gui.add_slider(
        label="sample_index",
        min=0,
        max=max(0, len(samples) - 1),
        step=1,
        initial_value=0,
    )
    mode_slider = server.gui.add_slider(
        label="display_mode (0=pred,1=gt,2=both)",
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

    def _render(idx: int, display_mode: int) -> None:
        sample = samples[idx]
        robot_name = sample["robot_name"]
        object_name = sample["object_name"]
        dof = int(sample["dof"])

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
            f"[sample {idx}] robot={robot_name} object={object_name} dof={dof} "
            f"mode={display_mode} mse_valid={sample.get('mse_valid', 'n/a')}"
        )

    sample_slider.on_update(lambda _: _render(int(sample_slider.value), int(mode_slider.value)))
    mode_slider.on_update(lambda _: _render(int(sample_slider.value), int(mode_slider.value)))

    _render(0, int(mode_slider.value))
    print(f"Viewer URL: http://{cfg.visualization.host}:{cfg.visualization.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
