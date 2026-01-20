from typing import Dict, Tuple, Optional
import os
import logging

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import torch
import trimesh
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

from data_process.dataset import get_dataloader
from data_process.visualize_samples import visualize_sdf_points_plotly
from train_fk import build_models, inference
import time
import numpy as np
import plyfile
import skimage.measure
from robot_model.chain_model import ChainModel
from visualize_ik import visualize_meshes

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    marching_step: int = 2,
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
        numpy_3d_sdf_tensor,
        level=0.0,
        spacing=[voxel_size] * 3,
        # step_size=max(int(marching_step), 1),
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
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        logging.warning("no surface extracted; skipping ply write to %s", ply_filename_out)
        return

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
    decoder,
    latent_vec,
    filename,
    N=256,
    max_batch=32 ** 3,
    grid_scale=2,
    offset=None,
    scale=None,
    marching_step: int = 2,
    normalize_center=None,
    normalize_scale=None,
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
        if normalize_center is not None and normalize_scale is not None:
            sample_subset = (sample_subset - normalize_center) / normalize_scale

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
        marching_step=marching_step,
    )


def _prepare_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA unavailable, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def _load_models(cfg: DictConfig, device: torch.device) -> Tuple[Dict[str, torch.nn.Module], Optional[int]]:
    models = build_models(cfg, device)
    if cfg.validation.checkpoint_epoch is None:
        logging.info("no checkpoint specified, skipping model loading")
        return models, None
    ckpt_path = to_absolute_path(os.path.join(cfg.validation.checkpoint_dir, f"epoch_{cfg.validation.checkpoint_epoch}.pth"))
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
    marching_step: int = 2,
    normalize_center=None,
    normalize_scale=None,
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
            marching_step=marching_step,
            normalize_center=normalize_center,
            normalize_scale=normalize_scale,
        )


def _validate_once(
    cfg: DictConfig,
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    batch: Dict[str, torch.Tensor],
    mesh_counter: int,
    chain_error_dict: Dict,
    sample_vis_state: Dict,
) -> int:
    loss_fn = torch.nn.L1Loss(reduction='none')
    with torch.no_grad():
        # batch['link_mask'] = torch.ones_like(batch['link_mask'])
        latent, sdf_pred = inference(batch, models, cfg=cfg, device=device)
        sdf_gt = batch["sdf_samples"].to(device)[..., 3].unsqueeze(-1)
        loss_points_vals = loss_fn(sdf_pred, sdf_gt) # (B, num_samples, 1)
        loss_vals = loss_points_vals.flatten(1).mean(dim=1)  # (B, )
        for i in range(loss_vals.shape[0]):
            cls = batch["class_idx"][i]
            if cls not in chain_error_dict:
                chain_error_dict[cls] = []
            chain_error_dict[cls].append(loss_vals[i].item())

    vis_cfg = getattr(cfg.validation, "sample_visualize", None)
    if vis_cfg and getattr(vis_cfg, "enable", False):
        out_dir = to_absolute_path(vis_cfg.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        rng = sample_vis_state["rng"]
        remaining = vis_cfg.num_samples - sample_vis_state["count"]
        if remaining > 0:
            B = sdf_pred.shape[0]
            idxs = rng.choice(B, size=min(remaining, B), replace=False).tolist()
            for i in idxs:
                cls = batch["class_idx"][i]
                inst = batch["instance_idx"][i]
                chain_model = ChainModel(
                    urdf_path=os.path.join(
                        cfg.data.data_source, f'out_chains_v2', f"chain_{cls}.urdf"),
                    samples_per_link=128,
                    device="cpu",
                )
                chain_model.update_status(batch['chain_q'][i, :chain_model.dof].cpu())
                gt_mesh = chain_model.get_trimesh_q(0)

                pts = batch["sdf_samples"][i, :, :3].cpu().numpy()
                sdf_pred_i = sdf_pred[i, :, 0].cpu().numpy()
                sdf_gt_i = sdf_gt[i, :, 0].cpu().numpy()
                normals = batch.get("sdf_normals", None)
                normals_np = normals[i].cpu().numpy() if normals is not None else None

                out_path = os.path.join(out_dir, f"samples_cls{cls}_inst{inst}.png")
                html_path = os.path.join(out_dir, f"samples_cls{cls}_inst{inst}.html")
                visualize_sdf_points_plotly(
                    mesh=gt_mesh,
                    points=pts,
                    sdf=sdf_pred_i,
                    gt_sdf=sdf_gt_i,
                    normals=normals_np,
                    title=f"cls {cls} inst {inst} | abs error",
                    downsample_ratio=vis_cfg.downsample_ratio,
                    max_points=vis_cfg.max_points,
                    normal_stride=vis_cfg.normal_stride,
                    normal_length=vis_cfg.normal_length,
                    save_path=out_path,
                    save_html_path=html_path if vis_cfg.save_html else None,
                    show=False,
                )
                sample_vis_state["count"] += 1
                if sample_vis_state["count"] >= vis_cfg.num_samples:
                    break
    
    max_mesh_num = int(getattr(cfg.validation, "max_mesh_per_batch", 0))
    if max_mesh_num > 0:
        save_root = to_absolute_path(cfg.validation.save_mesh_dir)
        mesh_cfg = cfg.validation.mesh
        for i in range(min(max_mesh_num, latent.shape[0])):
            cls = batch["class_idx"][i]
            inst = batch["instance_idx"][i]
            chain_model = ChainModel(
                urdf_path=os.path.join(
                    cfg.data.data_source, f'out_chains_v2', f"chain_{cls}.urdf"),
                    samples_per_link=128,
                    device="cpu",
            )
            chain_model.update_status(batch['chain_q'][i, :chain_model.dof].cpu())
            gt_mesh = chain_model.get_trimesh_q(0)
            radius = float(np.linalg.norm(gt_mesh.vertices, axis=1).max())
            name = f"cls{cls}_inst{inst}_m{mesh_counter+i}"
            out_path = os.path.join(save_root, name)
            normalize_center = None
            normalize_scale = None
            if str(getattr(cfg.data, "normalize_mode", "none")).lower() != "none":
                normalize_center = batch["coord_center"][i].to(device=models["decoder"].parameters().__next__().device)
                normalize_scale = batch["coord_scale"][i].to(device=models["decoder"].parameters().__next__().device)

            reconstruct_mesh(
                decoder=models["decoder"],
                latent=latent[i : i + 1],
                out_path=out_path,
                N=mesh_cfg.N,
                max_batch=mesh_cfg.max_batch,
                grid_scale=radius * 1.5,
                offset=mesh_cfg.get("offset", None),
                scale=mesh_cfg.get("scale", None),
                normalize_center=normalize_center,
                normalize_scale=normalize_scale,
            )

            vis_cfg = cfg.validation.visualize
            if getattr(vis_cfg, "enable", False):
                server = visualize_meshes(
                    pred_mesh=trimesh.load(out_path + ".ply", force="mesh"),
                    gt_mesh=gt_mesh,
                    port=vis_cfg.port,
                    show_pred_only=vis_cfg.get("show_pred_only", False),
                )
                print(f"Visualizing mesh on port {vis_cfg.port}...")

    return mesh_counter + latent.shape[0]


def _save_plotly_figure(fig: go.Figure, out_path: str | None, save_html: bool = False) -> None:
    if out_path is None:
        return
    try:
        fig.write_image(out_path)
    except Exception as exc:
        html_path = out_path + ".html"
        fig.write_html(html_path)
        logging.warning("write_image failed (%s); wrote HTML to %s", exc, html_path)
    if save_html:
        fig.write_html(out_path + ".html")


def plot_chain_error_boxplot(
    chain_error_dict: Dict[int, list],
    out_path: str | None = None,
    show: bool = False,
    title: str | None = None,
    save_html: bool = False,
):
    """Create a Plotly box plot of per-class error lists."""

    if not chain_error_dict:
        return None

    records = [
        {"class": cls, "error": err}
        for cls, errs in chain_error_dict.items()
        for err in errs
    ]

    fig = px.box(records, x="class", y="error", points="outliers")
    fig.update_layout(
        title=title or "Per-chain error distribution",
        xaxis_title="chain id",
        yaxis_title="mean abs error",
    )
    _save_plotly_figure(fig, out_path, save_html=save_html)
    if show:
        fig.show()
    return fig


def plot_zero_pose_chains(
    val_indices: list[int],
    data_source: str,
    out_path: str | None = None,
    show: bool = False,
    gap: float = 0.2,
    save_html: bool = False,
):
    """Visualize zero-pose meshes for given chain indices using Plotly.

    Meshes are laid out along the y-axis without overlap (with a configurable gap).
    """

    if not val_indices:
        return None

    fig = go.Figure()
    current_y = 0.0
    for idx in val_indices:
        urdf_path = os.path.join(data_source, "out_chains_v2", f"chain_{idx}.urdf")
        if not os.path.isfile(urdf_path):
            logging.warning("URDF not found for chain %s at %s", idx, urdf_path)
            continue

        chain_model = ChainModel(urdf_path=urdf_path, samples_per_link=128, device="cpu")
        q_zero = torch.zeros(chain_model.dof)
        chain_model.update_status(q_zero)
        mesh = chain_model.get_trimesh_q(0)

        verts = mesh.vertices.copy()
        faces = mesh.faces

        min_y, max_y = verts[:, 1].min(), verts[:, 1].max()
        height = float(max_y - min_y)
        shift = current_y - float(min_y)
        verts[:, 1] = verts[:, 1] + shift

        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                name=f"chain_{idx}",
                opacity=0.6,
            )
        )

        current_y += height + gap

    fig.update_layout(scene=dict(aspectmode="data"), title="Zero-pose chains (stacked along y)")
    _save_plotly_figure(fig, out_path, save_html=save_html)
    if show:
        fig.show()
    return fig


def plot_error_histogram(
    chain_error_dict: Dict[int, list],
    out_path: str | None = None,
    show: bool = False,
    save_html: bool = False,
):
    errors = [err for errs in chain_error_dict.values() for err in errs]
    if not errors:
        return None
    fig = px.histogram(x=errors, nbins=40)
    fig.update_layout(
        title="Error histogram (all instances)",
        xaxis_title="mean abs error",
        yaxis_title="count",
    )
    _save_plotly_figure(fig, out_path, save_html=save_html)
    if show:
        fig.show()
    return fig


def plot_error_vs_links(
    chain_error_dict: Dict[int, list],
    data_source: str,
    out_path: str | None = None,
    show: bool = False,
    save_html: bool = False,
):
    records = []
    for cls, errs in chain_error_dict.items():
        urdf_path = os.path.join(data_source, "out_chains_v2", f"chain_{cls}.urdf")
        if not os.path.isfile(urdf_path):
            continue
        chain_model = ChainModel(urdf_path=urdf_path, samples_per_link=128, device="cpu")
        records.append(
            {
                "chain_id": cls,
                "num_links": chain_model.num_links,
                "error": float(np.mean(errs)),
            }
        )
    if not records:
        return None
    fig = px.scatter(records, x="num_links", y="error", hover_data=["chain_id"])
    fig.update_layout(
        title="Mean error vs chain length",
        xaxis_title="num links",
        yaxis_title="mean abs error",
    )
    _save_plotly_figure(fig, out_path, save_html=save_html)
    if show:
        fig.show()
    return fig


def summarize_errors_by_length_and_boxes(
    chain_error_dict: Dict[int, list],
    data_source: str,
    lengths: Tuple[int, ...] = (2, 3, 4, 5),
) -> Tuple[Dict[int, list], list[Dict[int, list]]]:
    """Group errors by chain length and by number of box links per chain.

    Returns
    -------
    length_errors: dict
        key: chain length (e.g., 2–5); value: list of errors for all chains of that length.
    boxes_per_length: list of dict
        For each length in ``lengths`` (in order), a dict mapping ``num_boxes`` -> list of errors.
    """

    length_errors: Dict[int, list] = {L: [] for L in lengths}
    boxes_per_length: list[Dict[int, list]] = [{ } for _ in lengths]

    for cls, errs in chain_error_dict.items():
        urdf_path = os.path.join(data_source, "out_chains_v2", f"chain_{cls}.urdf")
        if not os.path.isfile(urdf_path):
            logging.warning("URDF not found for chain %s at %s", cls, urdf_path)
            continue

        chain_model = ChainModel(urdf_path=urdf_path, samples_per_link=128, device="cpu")
        length = chain_model.num_links
        num_boxes = sum(1 for spec in chain_model.geom_specs.values() if spec.get("type") == "box")

        if length in length_errors:
            length_errors[length].extend(errs)
            length_idx = lengths.index(length)
            buckets = boxes_per_length[length_idx]
            if num_boxes not in buckets:
                buckets[num_boxes] = []
            buckets[num_boxes].extend(errs)
        else:
            length_errors.setdefault(length, []).extend(errs)

    return length_errors, boxes_per_length


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
        shuffle=False,
        num_workers=cfg.data.num_workers,
        drop_last=cfg.data.drop_last,
        fix_q_samples=cfg.data.get("fix_q_samples", False),
    )

    mesh_counter = 0
    chain_error_dict = {}
    vis_cfg = getattr(cfg.validation, "sample_visualize", None)
    sample_vis_state = {
        "count": 0,
        "rng": np.random.default_rng(getattr(vis_cfg, "seed", 0) if vis_cfg else 0),
    }
    for b_idx, batch in enumerate(tqdm(val_loader, desc="Validating", leave=False)):
        mesh_counter = _validate_once(cfg, models, device, batch, mesh_counter, chain_error_dict, sample_vis_state)

    length_errors, boxes_per_length = summarize_errors_by_length_and_boxes(
        chain_error_dict,
        data_source,
    )

    output_dir = to_absolute_path(getattr(cfg.validation, "output_dir", "./outputs/validation"))
    os.makedirs(output_dir, exist_ok=True)

    plot_chain_error_boxplot(
        chain_error_dict,
        out_path=os.path.join(output_dir, "boxplot_by_chain.png"),
        show=False,
        title="Error distribution by chain",
        save_html=True,
    )
    plot_chain_error_boxplot(
        length_errors,
        out_path=os.path.join(output_dir, "boxplot_by_length.png"),
        show=False,
        title="Error distribution by chain length",
        save_html=True,
    )
    for length_idx, buckets in enumerate(boxes_per_length):
        plot_chain_error_boxplot(
            buckets,
            out_path=os.path.join(output_dir, f"boxplot_boxes_len{length_idx + 2}.png"),
            show=False,
            title=f"Error by #boxes (length {length_idx + 2})",
            save_html=True,
        )

    plot_error_histogram(
        chain_error_dict,
        out_path=os.path.join(output_dir, "error_histogram.png"),
        show=False,
        save_html=True,
    )
    plot_error_vs_links(
        chain_error_dict,
        data_source,
        out_path=os.path.join(output_dir, "error_vs_links.png"),
        show=False,
        save_html=True,
    )

    plot_zero_pose_chains(
        val_indices=val_indices,
        data_source=data_source,
        out_path=os.path.join(output_dir, "zero_pose_chains.png"),
        show=False,
        gap=0.08,
        save_html=True,
    )
    
    logging.info("validation finished")

if __name__ == "__main__":
    main()
