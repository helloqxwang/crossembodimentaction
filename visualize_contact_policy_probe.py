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

from data_process.contact_policy_probe_dataset import ShadowhandContactPolicyProbeDataset  # noqa: E402
from train_contact_policy import _move_batch_to_device, _prepare_device, _resolve_project_path  # noqa: E402
from train_contact_policy_probe import (  # noqa: E402
    _build_probe_dataset_common_kwargs,
    _resolve_probe_buffer_path,
    load_contact_policy_probe_checkpoint,
)


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


def _get_mesh_arrays(model: Any, q: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    mesh = model.get_trimesh_q(q.to(model.device), mode="original")["visual"].copy()
    return (
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
    )


def _resolve_visualization_split_name(split_name: str) -> str:
    split_key = str(split_name).strip().lower()
    if split_key == "validate":
        return "test"
    if split_key in {"train", "test"}:
        return split_key
    raise ValueError(f"Unsupported visualization.split={split_name}. Expected train or validate.")


def _load_probe_visualization_batch(
    cfg: DictConfig,
    *,
    ckpt_meta: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    split_key = _resolve_visualization_split_name(str(cfg.visualization.split))
    buffer_meta_key = f"{split_key}_buffer_path"
    saved_buffer_path = ckpt_meta.get(buffer_meta_key)
    if saved_buffer_path is None:
        saved_buffer_path = _resolve_probe_buffer_path(
            getattr(cfg.dataset, f"{split_key}_buffer_path", None),
            save_dir=_resolve_project_path(str(cfg.training.save_dir)),
            default_name=f"probe_{split_key}_buffer.pt",
        )
    common_dataset_kwargs = _build_probe_dataset_common_kwargs(cfg)
    common_dataset_kwargs["device"] = str(_prepare_device(str(cfg.dataset.generation_device)))
    dataset = ShadowhandContactPolicyProbeDataset.from_buffer_file(
        saved_buffer_path,
        **common_dataset_kwargs,
    )

    start_index = int(max(0, cfg.visualization.start_index))
    end_index = min(len(dataset), start_index + int(max(1, cfg.visualization.num_samples)))
    indices = list(range(start_index, end_index))
    if len(indices) == 0:
        raise IndexError(
            f"No probe samples available for split={split_key} at start_index={start_index}. "
            f"Dataset length={len(dataset)}."
        )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=len(indices),
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    return batch, {
        "dataset": dataset,
        "indices": indices,
        "split": split_key,
        "buffer_path": str(saved_buffer_path),
    }


def _render_probe_sample(
    *,
    server: viser.ViserServer,
    batch: dict[str, Any],
    pred_action: torch.Tensor,
    dataset: ShadowhandContactPolicyProbeDataset,
    sample_idx: int,
    point_size: float,
    hand_opacity: float,
) -> None:
    model = dataset.model
    target_q = batch["target_q_full"][sample_idx, : int(dataset.local_dof)].float().to(model.device)
    pred_q = pred_action[sample_idx, : int(dataset.local_dof)].float().to(model.device)

    contact_valid = batch["contact_valid_mask"][sample_idx].bool()
    gt_contact = batch["contact_cloud"][sample_idx, contact_valid, :3].float()
    contact_indices = batch["contact_point_indices"][sample_idx][contact_valid].long()

    pred_points, _ = model.get_surface_points_normals(q=pred_q)
    pred_points = pred_points.detach().cpu()
    pred_contact = pred_points[contact_indices] if int(contact_indices.numel()) > 0 else torch.zeros((0, 3))

    v_gt, f_gt = _get_mesh_arrays(model, target_q)
    server.scene.add_mesh_simple(
        name="robot_gt",
        vertices=v_gt,
        faces=f_gt,
        color=(255, 170, 80),
        opacity=float(hand_opacity),
        visible=True,
    )

    v_pred, f_pred = _get_mesh_arrays(model, pred_q)
    server.scene.add_mesh_simple(
        name="robot_pred",
        vertices=v_pred,
        faces=f_pred,
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

    if int(pred_contact.shape[0]) > 0:
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
        f"[sample {sample_idx}] split=probe "
        f"valid_contacts={int(contact_valid.sum().item())}"
    )


@hydra.main(version_base="1.2", config_path="conf", config_name="config_contact_policy_probe")
def main(cfg: DictConfig) -> None:
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    device = _prepare_device(str(cfg.training.device))
    policy, ckpt_path, ckpt_meta = load_contact_policy_probe_checkpoint(
        cfg,
        device=device,
        ckpt_path=None if cfg.visualization.checkpoint_path is None else str(cfg.visualization.checkpoint_path),
        prefer_ema=bool(cfg.visualization.checkpoint_prefer_ema),
    )
    batch, batch_meta = _load_probe_visualization_batch(cfg, ckpt_meta=ckpt_meta)

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

    def _render() -> None:
        _render_probe_sample(
            server=server,
            batch=batch,
            pred_action=pred_action,
            dataset=batch_meta["dataset"],
            sample_idx=int(sample_slider.value),
            point_size=float(cfg.visualization.point_size),
            hand_opacity=float(cfg.visualization.hand_opacity),
        )

    sample_slider.on_update(lambda _: _render())
    _render()

    print(
        f"Viewer URL: http://{cfg.visualization.host}:{cfg.visualization.port} | "
        f"split={batch_meta['split']} buffer={batch_meta['buffer_path']} ckpt={ckpt_path}"
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
