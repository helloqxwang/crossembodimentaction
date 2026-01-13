"""Visualize IK predictions against ground-truth meshes.

Loads the validation split, runs the trained IK model to predict joint angles,
reconstructs the chain mesh via ``ChainModel`` at the predicted configuration,
and visualizes predicted vs. GT meshes in Viser (side-by-side).

Configuration: uses ``conf/config_ik.yaml``; override at runtime for checkpoint
or dataset tweaks, e.g. ``python visualize_ik.py training.pretrained_ckpt=...``.
"""

from typing import Dict, Optional, Tuple
import logging
import os
import numpy as np
import hydra
import torch
import trimesh
import viser
import matplotlib.pyplot as plt
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from networks.transformer import plot_token_dependency

from data_process.dataset import get_dataloader
from robot_model.chain_model import ChainModel
from train_ik import _prepare_device, build_models, inference_ik


def _load_models(cfg: DictConfig, device: torch.device) -> Tuple[Dict[str, torch.nn.Module], Optional[int]]:
	models = build_models(cfg, device)
	ckpt_path = getattr(cfg.training, "test_ckpt", None)
	if not ckpt_path:
		raise FileNotFoundError("training.test_ckpt must be set for visualization")
	ckpt_path = to_absolute_path(ckpt_path)
	if not os.path.isfile(ckpt_path):
		raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

	# Allow OmegaConf objects inside checkpoints saved from train_ik/train_sdf.
	torch.serialization.add_safe_globals([DictConfig])

	state = torch.load(ckpt_path, map_location=device, weights_only=False)
	epoch = state.get("epoch") if isinstance(state, dict) else None
	state_dicts = state.get("models", state)

	for name, module in models.items():
		if name in state_dicts:
			module.load_state_dict(state_dicts[name], strict=False)
		else:
			raise KeyError(f"model '{name}' not found in checkpoint")

	return models, epoch


def _resolve_gt_mesh_path(data_root: str, class_idx: int, instance_idx: int) -> Optional[str]:
	path = os.path.join(data_root, "chain_meshes", f"chain_{class_idx}_{instance_idx}.obj")
	return path if os.path.isfile(path) else None


def _get_chain(chain_cache: Dict[int, ChainModel], data_root: str, class_idx: int, device: torch.device) -> ChainModel:
	if class_idx in chain_cache:
		return chain_cache[class_idx]
	urdf_path = os.path.join(data_root, "out_chains", f"chain_{class_idx}.urdf")
	if not os.path.isfile(urdf_path):
		raise FileNotFoundError(f"URDF not found for class {class_idx}: {urdf_path}")
	chain_cache[class_idx] = ChainModel(urdf_path=urdf_path, device=device)
	return chain_cache[class_idx]


def visualize_meshes(
	pred_mesh: trimesh.Trimesh,
	gt_mesh: Optional[trimesh.Trimesh] = None,
	*,
	port: int = 8100,
	show_pred_only: bool = False,
):
	server = viser.ViserServer(port=port)
	pred_mesh_vis = pred_mesh.copy()
	# Viser lacks a color kwarg; bake vertex colors directly into the mesh.
	pred_mesh_vis.visual.face_colors = [51, 179, 255, 255]  # RGBA ~ (0.2, 0.7, 1.0)
	server.scene.add_mesh_trimesh(name="pred_mesh", mesh=pred_mesh_vis)

	# Shift GT mesh to the +X direction so both are visible together.
	if gt_mesh is not None and not show_pred_only:
		bounds = pred_mesh.bounds
		size_x = float((bounds[1] - bounds[0])[0])
		shift = size_x * 1.4 if size_x > 0 else 0.2
		gt_shifted = gt_mesh.copy()
		gt_shifted.apply_translation([shift, 0, 0])
		gt_shifted.visual.face_colors = [255, 77, 77, 255]  # RGBA ~ (1.0, 0.3, 0.3)
		server.scene.add_mesh_trimesh(name="gt_mesh", mesh=gt_shifted)
		server.scene.add_frame(
            name="pred_frame",
            wxyz=np.array([1, 0, 0, 0]),
            position=np.array([0, 0, 0]),
            axes_length=shift * 0.3,
            axes_radius=0.005,
        )
		server.scene.add_frame(
				name="gt_frame",
				wxyz=np.array([1, 0, 0, 0]),
				position=np.array([shift, 0, 0]),
				axes_length=shift * 0.3,
				axes_radius=0.005,
			)

	logging.info(f"Viser running on http://localhost:{port}")
	return server


def _pred_q_for_sample(
	joint_pred: torch.Tensor,
	joint_mask: torch.Tensor,
	sample_idx: int,
) -> torch.Tensor:
	# Extract valid joints using the mask (True for valid positions).
	valid_count = int(joint_mask[sample_idx].sum().item())
	return joint_pred[sample_idx, :valid_count]


@hydra.main(config_path="conf", config_name="config_ik", version_base="1.3")
def main(cfg: DictConfig) -> None:
	logging.basicConfig(level=logging.INFO)

	device = _prepare_device(cfg.training.device)
	torch.manual_seed(0)

	models, epoch = _load_models(cfg, device)
	logging.info(f"loaded checkpoint epoch={epoch}")

	data_root = to_absolute_path(cfg.data.data_source)
	val_indices = list(range(cfg.data.val_indices.start, cfg.data.val_indices.end))

	val_loader = get_dataloader(
		data_source=data_root,
		indices=[1, ],
		num_instances=cfg.data.num_instances,
		subsample=cfg.data.subsample,
		batch_size=cfg.data.val_batch_size,
		max_num_links=cfg.data.max_num_links,
		load_ram=cfg.data.load_ram,
		shuffle=False,
		num_workers=cfg.data.num_workers,
		drop_last=False,
		ik=True,
	)

	val_loader_another = get_dataloader(
		data_source=data_root,
		indices=[2, ],
		num_instances=cfg.data.num_instances,
		subsample=cfg.data.subsample,
		batch_size=cfg.data.val_batch_size,
		max_num_links=cfg.data.max_num_links,
		load_ram=cfg.data.load_ram,
		shuffle=False,
		num_workers=cfg.data.num_workers,
		drop_last=False,
		ik=True,
	)

	vis_cfg = getattr(cfg, "visualize", {})
	max_mesh = int(getattr(vis_cfg, "max_mesh", 64))
	port = int(getattr(vis_cfg, "port", 9010))
	show_pred_only = bool(getattr(vis_cfg, "show_pred_only", False))
	plot_dependency = bool(getattr(vis_cfg, "plot_dependency", False))
	dep_plot_path = getattr(vis_cfg, "dependency_plot_path", "outputs/dependency.png")

	chain_cache: Dict[int, ChainModel] = {}
	shown = 0

	for batch, batch_another in zip(val_loader, val_loader_another):
		batch["sdf_tokens"] = batch_another["sdf_tokens"]
		if plot_dependency:
			joint_pred, joint_mask, dependencies = inference_ik(batch, models, device=device, return_dependency=True)
		else:
			joint_pred, joint_mask = inference_ik(batch, models, device=device)

		class_indices = batch["class_idx"]
		instance_indices = batch["instance_idx"]

		for i in range(len(class_indices)):
			if shown >= max_mesh:
				logging.info("Reached visualization limit; exiting.")
				return

			cls = class_indices[i]
			inst = instance_indices[i]

			q_pred = _pred_q_for_sample(joint_pred, joint_mask, i)
			chain = _get_chain(chain_cache, data_root, cls, "cpu")
			chain.update_status(q_pred.unsqueeze(0).detach().to(chain.device))
			pred_mesh = chain.get_trimesh_q(idx=0, boolean_merged=True)

			gt_mesh_path = _resolve_gt_mesh_path(data_root, batch_another["class_idx"][i], batch_another["instance_idx"][i])
			gt_mesh = trimesh.load(gt_mesh_path, force="mesh") if gt_mesh_path else None

			visualize_meshes(
				pred_mesh=pred_mesh,
				gt_mesh=gt_mesh,
				port=port + shown,  # avoid port collisions when showing multiple meshes
				show_pred_only=show_pred_only,
			)

			if plot_dependency and shown == 0:
				os.makedirs(os.path.dirname(dep_plot_path), exist_ok=True)
				fig, _ = plot_token_dependency(
					dependencies[i],
					title=f"Token dependency class {cls} instance {inst}",
				)
				fig.savefig(dep_plot_path, dpi=150)
				plt.close(fig)
				logging.info(f"Saved dependency plot to {dep_plot_path}")

			logging.info(f"Visualized class {cls} instance {inst} (sample {shown+1}/{max_mesh})")
			shown += 1

		if shown >= max_mesh:
			break

if __name__ == "__main__":
	main()