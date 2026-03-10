from __future__ import annotations

import os
import sys
import time
from typing import Any

import hydra
import numpy as np
import torch
import viser
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from data_process.contact_policy_dataset import OnTheFlySyntheticContactPolicyDataset  # noqa: E402
from train_contact_policy import (  # noqa: E402
    _prepare_device,
    build_contact_policy_common_dataset_kwargs,
)


def _empty_pc(server: viser.ViserServer, name: str, point_size: float) -> None:
    server.scene.add_point_cloud(
        name=name,
        points=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.float32),
        point_size=float(point_size),
        point_shape="circle",
    )


def _to_numpy_points(points: torch.Tensor) -> np.ndarray:
    return points.detach().cpu().numpy().astype(np.float32)


def _uniform_colors(n: int, rgb: tuple[float, float, float]) -> np.ndarray:
    colors = np.zeros((n, 3), dtype=np.float32)
    if n > 0:
        colors[:, 0] = rgb[0]
        colors[:, 1] = rgb[1]
        colors[:, 2] = rgb[2]
    return colors


def _get_mesh_arrays(model: Any, q: torch.Tensor, translation: torch.Tensor | None) -> tuple[np.ndarray, np.ndarray]:
    mesh = model.get_trimesh_q(q.to(model.device), mode="original")["visual"].copy()
    if translation is not None:
        mesh.apply_translation(translation.detach().cpu().numpy().astype(np.float64))
    return (
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
    )


def _load_training_dataset(cfg: DictConfig) -> OnTheFlySyntheticContactPolicyDataset:
    robot_names = [str(x) for x in cfg.dataset.robot_names]
    samples_per_hand = int(max(1, getattr(cfg.visualization, "train_data_samples_per_hand", 200)))
    samples_per_epoch = samples_per_hand * max(1, len(robot_names))
    common_dataset_kwargs = build_contact_policy_common_dataset_kwargs(cfg)
    return OnTheFlySyntheticContactPolicyDataset(
        samples_per_epoch=samples_per_epoch,
        buffer_refresh_fraction=float(cfg.dataset.buffer_refresh_fraction),
        buffer_build_batch_size_per_robot=min(
            int(cfg.dataset.buffer_build_batch_size_per_robot),
            samples_per_hand,
        ),
        buffer_refresh_batch_size_per_robot=min(
            int(cfg.dataset.buffer_refresh_batch_size_per_robot),
            samples_per_hand,
        ),
        device=str(_prepare_device(str(cfg.dataset.train_generation_device))),
        store_full_metadata=True,
        progress_label="visualize_train_data",
        **common_dataset_kwargs,
    )


def _render_training_sample(
    *,
    server: viser.ViserServer,
    dataset: OnTheFlySyntheticContactPolicyDataset,
    sample_idx: int,
    point_size: float,
    hand_opacity: float,
) -> None:
    sample = dataset[int(sample_idx)]
    robot_name = str(sample["robot_name"])
    model = dataset.robot_specs[robot_name]["model"]

    local_dof = int(sample["local_dof"].item())
    q1 = sample["q1_padded"][:local_dof].float().to(model.device)
    q2 = sample["q2_padded"][:local_dof].float().to(model.device)
    q_action = sample["action"][:local_dof].float().to(model.device)

    contact_valid = sample["contact_valid_mask"].bool()
    gt_contact = sample["contact_cloud"][contact_valid, :3].float()
    contact_indices = sample["contact_point_indices"][contact_valid].long().to(model.device)

    with torch.no_grad():
        action_points, _ = model.get_surface_points_normals(q=q_action)
    action_points = action_points.detach()
    if int(contact_indices.numel()) > 0:
        action_contact = action_points[contact_indices]
    else:
        action_contact = torch.zeros((0, 3), dtype=torch.float32, device=model.device)

    q1_v, q1_f = _get_mesh_arrays(model, q1, None)
    server.scene.add_mesh_simple(
        name="robot_q1",
        vertices=q1_v,
        faces=q1_f,
        color=(90, 140, 255),
        opacity=float(hand_opacity),
        visible=True,
    )

    q2_v, q2_f = _get_mesh_arrays(model, q2, None)
    server.scene.add_mesh_simple(
        name="robot_q2",
        vertices=q2_v,
        faces=q2_f,
        color=(255, 170, 80),
        opacity=float(hand_opacity),
        visible=True,
    )

    action_v, action_f = _get_mesh_arrays(model, q_action, translation=None)
    server.scene.add_mesh_simple(
        name="robot_action",
        vertices=action_v,
        faces=action_f,
        color=(80, 220, 140),
        opacity=float(hand_opacity),
        visible=True,
    )

    if int(gt_contact.shape[0]) > 0:
        server.scene.add_point_cloud(
            name="gt_contact",
            points=_to_numpy_points(gt_contact),
            colors=_uniform_colors(int(gt_contact.shape[0]), (1.0, 0.25, 0.25)),
            point_size=float(point_size),
            point_shape="circle",
        )
    else:
        _empty_pc(server, "gt_contact", point_size)

    if int(action_contact.shape[0]) > 0:
        server.scene.add_point_cloud(
            name="action_contact",
            points=_to_numpy_points(action_contact),
            colors=_uniform_colors(int(action_contact.shape[0]), (0.15, 0.95, 0.35)),
            point_size=float(point_size),
            point_shape="circle",
        )
    else:
        _empty_pc(server, "action_contact", point_size)

    print(
        f"[sample {sample_idx}] robot={robot_name} "
        f"valid_contacts={int(contact_valid.sum().item())} local_dof={local_dof}"
    )


@hydra.main(version_base="1.2", config_path="conf", config_name="config_contact_policy")
def main(cfg: DictConfig) -> None:
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    dataset = _load_training_dataset(cfg)
    server = viser.ViserServer(
        host=str(cfg.visualization.host),
        port=int(cfg.visualization.port),
    )

    sample_slider = server.gui.add_slider(
        label="sample_index",
        min=0,
        max=max(0, len(dataset) - 1),
        step=1,
        initial_value=0,
    )

    def _render() -> None:
        _render_training_sample(
            server=server,
            dataset=dataset,
            sample_idx=int(sample_slider.value),
            point_size=float(cfg.visualization.point_size),
            hand_opacity=float(cfg.visualization.hand_opacity),
        )

    sample_slider.on_update(lambda _: _render())
    _render()

    print(
        f"Viewer URL: http://{cfg.visualization.host}:{cfg.visualization.port} | "
        f"samples={len(dataset)}"
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
