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
from torch.utils.data import DataLoader, Subset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from data_process.contact_policy_dataset import (  # noqa: E402
    OnTheFlySyntheticContactPolicyDataset,
    RealContactPolicyValDataset,
)
from train_contact_policy import (  # noqa: E402
    _move_batch_to_device,
    _prepare_device,
    build_contact_policy_common_dataset_kwargs,
    load_contact_policy_checkpoint,
)


VIS_MODE_NAMES = {
    0: "all",
    1: "po",
    2: "q1a",
    3: "q2a",
}


def _empty_mesh(server: viser.ViserServer, name: str) -> None:
    server.scene.add_mesh_simple(
        name=name,
        vertices=np.zeros((0, 3), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.int32),
        color=(0.0, 0.0, 0.0),
        opacity=0.0,
        visible=False,
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


def _load_visualization_batch(
    cfg: DictConfig,
    *,
    global_robot_names: list[str] | None = None,
) -> tuple[Dict[str, Any], dict[str, Any]]:
    robot_name = str(cfg.visualization.robot_name)
    num_samples = int(max(1, cfg.visualization.num_samples))
    common_dataset_kwargs = build_contact_policy_common_dataset_kwargs(
        cfg,
        robot_names=[robot_name],
        global_robot_names=global_robot_names,
    )

    mode = str(cfg.visualization.mode).strip().lower()
    if mode == "synthesized":
        dataset = OnTheFlySyntheticContactPolicyDataset(
            samples_per_epoch=num_samples,
            buffer_refresh_fraction=float(cfg.dataset.buffer_refresh_fraction),
            buffer_build_batch_size_per_robot=min(
                int(cfg.dataset.buffer_build_batch_size_per_robot),
                num_samples,
            ),
            buffer_refresh_batch_size_per_robot=min(
                int(cfg.dataset.buffer_refresh_batch_size_per_robot),
                num_samples,
            ),
            device=str(_prepare_device(str(cfg.dataset.train_generation_device))),
            store_full_metadata=True,
            progress_label="visualize_synth",
            **common_dataset_kwargs,
        )
        indices = list(range(num_samples))
    elif mode == "real":
        dataset = RealContactPolicyValDataset(
            cache_processed_samples=False,
            cache_build_batch_size=int(getattr(cfg.dataset, "real_val_cache_build_batch_size", 256)),
            device=str(_prepare_device(str(getattr(cfg.dataset, "real_val_build_device", cfg.dataset.robot_model_device)))),
            progress_label="visualize_real",
            **common_dataset_kwargs,
        )
        start_index = int(max(0, cfg.visualization.start_index))
        end_index = min(len(dataset), start_index + num_samples)
        indices = list(range(start_index, end_index))
        if len(indices) == 0:
            raise IndexError(
                f"No real samples available for robot={robot_name} at start_index={start_index}. "
                f"Dataset length={len(dataset)}."
            )
    else:
        raise ValueError(f"Unsupported visualization.mode={cfg.visualization.mode}")

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=len(indices),
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    meta = {
        "dataset": dataset,
        "indices": indices,
        "mode": mode,
        "robot_name": robot_name,
    }
    return batch, meta


def _get_local_q(batch: Dict[str, Any], key: str, sample_idx: int) -> torch.Tensor:
    local_dof = int(batch["local_dof"][sample_idx].item())
    return batch[key][sample_idx, :local_dof].float()


def _get_mesh_arrays(model: Any, q: torch.Tensor, translation: torch.Tensor | None) -> tuple[np.ndarray, np.ndarray]:
    mesh = model.get_trimesh_q(q.to(model.device), mode="original")["visual"].copy()
    if translation is not None:
        mesh.apply_translation(translation.detach().cpu().numpy().astype(np.float64))
    return (
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
    )


def _align_pred_base_pose(model: Any, q_pred: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
    aligned = q_pred.detach().clone()
    base_pose_indices = list(getattr(model, "base_pose_indices", []))
    if len(base_pose_indices) > 0:
        aligned[base_pose_indices] = q_ref[base_pose_indices].to(aligned.device)
    return aligned


def _render_contact_policy_sample(
    *,
    server: viser.ViserServer,
    batch: Dict[str, Any],
    pred_action: torch.Tensor,
    dataset: Any,
    sample_idx: int,
    vis_mode: int,
    point_size: float,
    hand_opacity: float,
) -> None:
    robot_name = str(batch["robot_name"][sample_idx])
    model = dataset.robot_specs[robot_name]["model"]
    q1 = _get_local_q(batch, "q1_padded", sample_idx).to(model.device)
    q2 = _get_local_q(batch, "q2_padded", sample_idx).to(model.device)
    q_pred = pred_action[sample_idx, : int(batch["local_dof"][sample_idx].item())].float().to(model.device)

    contact_valid = batch["contact_valid_mask"][sample_idx].bool()
    gt_contact = batch["contact_cloud"][sample_idx, contact_valid, :3].float()
    contact_indices = batch["contact_point_indices"][sample_idx][contact_valid].long()

    mode_name = VIS_MODE_NAMES.get(int(vis_mode), f"unknown-{vis_mode}")
    q_pred_vis = q_pred

    if vis_mode == 2:
        q_pred_vis = _align_pred_base_pose(model, q_pred, q1)

    pred_points, _ = model.get_surface_points_normals(q=q_pred_vis)
    pred_points = pred_points.detach().cpu()
    pred_contact = pred_points[contact_indices] if int(contact_indices.numel()) > 0 else torch.zeros((0, 3))

    show_q1 = vis_mode in (0, 2)
    show_q2 = vis_mode in (0, 3)
    show_pred = vis_mode in (0, 2, 3)
    show_gt_contact = vis_mode in (0, 1)
    show_pred_contact = vis_mode in (0, 1)

    if show_q1:
        v, f = _get_mesh_arrays(model, q1, None)
        server.scene.add_mesh_simple(
            name="robot_q1",
            vertices=v,
            faces=f,
            color=(90, 140, 255),
            opacity=float(hand_opacity),
            visible=True,
        )
    else:
        _empty_mesh(server, "robot_q1")

    if show_q2:
        v, f = _get_mesh_arrays(model, q2, None)
        server.scene.add_mesh_simple(
            name="robot_q2",
            vertices=v,
            faces=f,
            color=(255, 170, 80),
            opacity=float(hand_opacity),
            visible=True,
        )
    else:
        _empty_mesh(server, "robot_q2")

    if show_pred:
        v, f = _get_mesh_arrays(model, q_pred_vis, translation=None)
        server.scene.add_mesh_simple(
            name="robot_pred",
            vertices=v,
            faces=f,
            color=(80, 220, 140),
            opacity=float(hand_opacity),
            visible=True,
        )
    else:
        _empty_mesh(server, "robot_pred")

    if show_gt_contact and int(gt_contact.shape[0]) > 0:
        server.scene.add_point_cloud(
            name="gt_contact",
            points=_to_numpy_points(gt_contact),
            colors=_uniform_colors(int(gt_contact.shape[0]), (1.0, 0.25, 0.25)),
            point_size=float(point_size),
            point_shape="circle",
        )
    else:
        _empty_pc(server, "gt_contact", point_size)

    if show_pred_contact and int(pred_contact.shape[0]) > 0:
        server.scene.add_point_cloud(
            name="pred_contact",
            points=_to_numpy_points(pred_contact),
            colors=_uniform_colors(int(pred_contact.shape[0]), (0.15, 0.95, 0.35)),
            point_size=float(point_size),
            point_shape="circle",
        )
    else:
        _empty_pc(server, "pred_contact", point_size)

    print(
        f"[sample {sample_idx}] robot={robot_name} "
        f"vis_mode={mode_name} valid_contacts={int(contact_valid.sum().item())}"
    )


@hydra.main(version_base="1.2", config_path="conf", config_name="config_contact_policy")
def main(cfg: DictConfig) -> None:
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    device = _prepare_device(str(cfg.training.device))
    policy, ckpt_path, ckpt_meta = load_contact_policy_checkpoint(
        cfg,
        action_dim=None,
        device=device,
        ckpt_path=None if cfg.visualization.checkpoint_path is None else str(cfg.visualization.checkpoint_path),
        prefer_ema=bool(cfg.visualization.checkpoint_prefer_ema),
    )
    batch, batch_meta = _load_visualization_batch(
        cfg,
        global_robot_names=[str(x) for x in ckpt_meta.get("robot_names", list(cfg.dataset.robot_names))],
    )

    with torch.no_grad():
        pred = policy.predict_action(_move_batch_to_device(batch, device))
    pred_action = pred["action_pred"].detach().cpu()

    server = viser.ViserServer(
        host=str(cfg.visualization.host),
        port=int(cfg.visualization.port),
    )

    num_samples = int(pred_action.shape[0])
    sample_slider = server.gui.add_slider(
        label="sample_index",
        min=0,
        max=max(0, num_samples - 1),
        step=1,
        initial_value=0,
    )
    mode_slider = server.gui.add_slider(
        label="vis_mode (0=all,1=po,2=q1a,3=q2a)",
        min=0,
        max=3,
        step=1,
        initial_value=int(min(max(0, cfg.visualization.initial_vis_mode), 3)),
    )

    def _render() -> None:
        _render_contact_policy_sample(
            server=server,
            batch=batch,
            pred_action=pred_action,
            dataset=batch_meta["dataset"],
            sample_idx=int(sample_slider.value),
            vis_mode=int(mode_slider.value),
            point_size=float(cfg.visualization.point_size),
            hand_opacity=float(cfg.visualization.hand_opacity),
        )

    sample_slider.on_update(lambda _: _render())
    mode_slider.on_update(lambda _: _render())

    _render()
    print(
        f"Viewer URL: http://{cfg.visualization.host}:{cfg.visualization.port} | "
        f"mode={batch_meta['mode']} robot={batch_meta['robot_name']} ckpt={ckpt_path}"
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
