from typing import Dict, Tuple, Optional
import os
import logging

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import torch
import trimesh
import viser

from data_process.dataset import get_dataloader
from data_process.visualize_samples import visualize_sdf, plot_mesh
from train_fk import build_models, inference
import time
import numpy as np
import plyfile
import skimage


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )

def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)

    sdf = decoder(inputs)

    return sdf

def decode_to_mesh(
    decoder, latent_vec, filename, N=256, max_batch=32 ** 3, grid_scale=2, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-grid_scale, -grid_scale, -grid_scale]
    voxel_size = 2 * grid_scale / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            decode_sdf(decoder, latent_vec, sample_subset).squeeze(1).detach().cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def _prepare_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA unavailable, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def _load_models(cfg: DictConfig, device: torch.device) -> Tuple[Dict[str, torch.nn.Module], Optional[int]]:
    models = build_models(cfg, device)
    ckpt_path = to_absolute_path(cfg.validation.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    # Allow OmegaConf objects inside checkpoints saved from train_sdf.
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


def reconstruct_mesh(
    decoder: torch.nn.Module,
    latent: torch.Tensor,
    out_path: str,
    N: int = 256,
    max_batch: int = 262144,
    grid_scale: int = 1,
    offset=None,
    scale=None,
):
    """Decode a single latent to a mesh using deep_sdf.mesh.create_mesh."""

    if not torch.cuda.is_available():
        raise RuntimeError("Mesh extraction requires CUDA because deep_sdf.mesh uses .cuda() internally.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    decoder.eval()
    with torch.no_grad():
        decode_to_mesh(
            decoder,
            latent,
            out_path,
            N=N,
            max_batch=max_batch,
            grid_scale=grid_scale,
            offset=offset,
            scale=scale,
        )


def _resolve_gt_mesh_path(template: Optional[str], class_idx: int, instance_idx: int) -> Optional[str]:
    if template is None:
        return None
    path = template.format(class_idx=class_idx, instance_idx=instance_idx)
    return path if os.path.isfile(path) else None


def visualize_meshes(
    pred_mesh_path: str,
    gt_mesh: Optional[trimesh.Trimesh] = None,
    port: int = 8080,
    show_pred_only: bool = False,
):
    if not os.path.isfile(pred_mesh_path):
        raise FileNotFoundError(f"pred mesh not found: {pred_mesh_path}")

    server = viser.ViserServer(port=port)
    pred_mesh = trimesh.load(pred_mesh_path, force="mesh")
    server.scene.add_mesh_trimesh(
        name="pred_mesh",
        mesh=pred_mesh,
        # material=viser.materials.MeshStandardMaterial(color=(0.2, 0.7, 1.0), opacity=0.9),
    )
    # Calculate mesh bounds and shift distance
    pred_bounds = pred_mesh.bounds
    pred_size = pred_bounds[1] - pred_bounds[0]
    shift_distance = pred_size[0] * 1.2  # 20% padding

    # Shift gt_mesh along x-axis
    if gt_mesh is not None and not show_pred_only:
        gt_mesh_shifted = gt_mesh.copy()
        gt_mesh_shifted.apply_translation([shift_distance, 0, 0])
        server.scene.add_mesh_trimesh(
            name="gt_mesh",
            mesh=gt_mesh_shifted,
        )
        
        # Visualize world frames
        server.scene.add_frame(
            name="pred_frame",
            wxyz=np.array([1, 0, 0, 0]),
            position=np.array([0, 0, 0]),
            axes_length=shift_distance * 0.3,
            axes_radius=0.005,
        )
        server.scene.add_frame(
            name="gt_frame",
            wxyz=np.array([1, 0, 0, 0]),
            position=np.array([shift_distance, 0, 0]),
            axes_length=shift_distance * 0.3,
            axes_radius=0.005,
        )
    logging.info(f"Viser running on http://localhost:{port}")


def visualize_sdf_with_mesh(
    pred_mesh: trimesh.Trimesh,
    sdf_pred: np.ndarray,
    sdf_gt: Optional[np.ndarray] = None,
    title: str = "SDF + Mesh",
    downsample_ratio: float = 0.1,
    max_points: int = 50000,
    marker_size: int = 3,
    seed: int = 0,
):
    """Visualize predicted SDF + mesh in same figure.
    
    Args:
        pred_mesh: trimesh object.
        sdf_pred: Predicted SDF points, shape (N, 4) with [x, y, z, sdf].
        sdf_gt: Optional GT SDF points, shape (M, 4) with [x, y, z, sdf].
        title: Plot title.
        downsample_ratio: Fraction of points to visualize [0, 1].
        max_points: Cap on visualized points after downsampling.
        marker_size: Scatter marker size.
        seed: Random seed for subsampling.
    """
    visualize_sdf(
        mesh=pred_mesh,
        sdf_pred=sdf_pred,
        sdf_gt=sdf_gt,
        title=title,
        downsample_ratio=downsample_ratio,
        max_points=max_points,
        marker_size=marker_size,
        seed=seed,
    )


def _validate_once(
    cfg: DictConfig,
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    batch: Dict[str, torch.Tensor],
    mesh_counter: int,
) -> int:
    loss_fn = torch.nn.L1Loss()
    with torch.no_grad():
        latent, sdf_pred = inference(batch, models, device=device)
        sdf_gt = batch["sdf_samples"].to(device)[..., 3].unsqueeze(-1)
        loss_val = loss_fn(sdf_pred, sdf_gt).item()

    logging.info(f"batch loss: {loss_val:.6f}")

    class_indices = batch["class_idx"]
    instance_indices = batch["instance_idx"]
    gt_meshes = [trimesh.load(os.path.join(cfg.data.data_source, f"chain_meshes", f"chain_{class_idx}_{instance_idx}.obj"), force="mesh") for class_idx, instance_idx in zip(class_indices, instance_indices)]
    radius = [float(np.linalg.norm(mesh.vertices, axis=1).max()) for mesh in gt_meshes]
    
    max_mesh_num = int(getattr(cfg.validation, "max_mesh_per_batch", 0))
    if max_mesh_num > 0:
        save_root = to_absolute_path(cfg.validation.save_mesh_dir)
        mesh_cfg = cfg.validation.mesh
        for i in range(min(max_mesh_num, latent.shape[0])):
            cls = batch["class_idx"][i]
            inst = batch["instance_idx"][i]
            name = f"cls{cls}_inst{inst}_m{mesh_counter+i}"
            out_path = os.path.join(save_root, name)
            reconstruct_mesh(
                decoder=models["decoder"],
                latent=latent[i : i + 1],
                out_path=out_path,
                N=mesh_cfg.N,
                max_batch=mesh_cfg.max_batch,
                grid_scale=radius[i] * 1.1,
                offset=mesh_cfg.get("offset", None),
                scale=mesh_cfg.get("scale", None),
            )

            vis_cfg = cfg.validation.visualize
            if getattr(vis_cfg, "enable", False):
                visualize_meshes(
                    pred_mesh_path=out_path + ".ply",
                    gt_mesh=gt_meshes[i],
                    port=vis_cfg.port,
                    show_pred_only=vis_cfg.get("show_pred_only", False),
                )

    return mesh_counter + latent.shape[0]


@hydra.main(config_path="conf", config_name="config_validation", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    device = _prepare_device(cfg.validation.device)
    torch.manual_seed(0)

    models, epoch = _load_models(cfg, device)
    logging.info(f"loaded checkpoint epoch={epoch}")

    data_source = to_absolute_path(cfg.data.data_source)
    val_indices = list(range(cfg.data.val_indices.start, cfg.data.val_indices.end))

    val_loader = get_dataloader(
        data_source=data_source,
        indices=val_indices,
        num_instances=cfg.data.num_instances,
        subsample=cfg.data.subsample,
        batch_size=cfg.data.val_batch_size,
        max_num_links=cfg.data.max_num_links,
        load_ram=cfg.data.load_ram,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        drop_last=False,
    )

    mesh_counter = 0
    num_batches = cfg.validation.get("num_batches", None)
    for b_idx, batch in enumerate(val_loader):
        mesh_counter = _validate_once(cfg, models, device, batch, mesh_counter)
        if num_batches is not None and (b_idx + 1) >= num_batches:
            break

    logging.info("validation finished")


if __name__ == "__main__":
    main()