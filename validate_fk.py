from __future__ import annotations

from dataclasses import dataclass
import itertools
import logging
import os
from typing import Dict, Iterable, List

import hydra
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import trimesh
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from robot_model.chain_model import ChainModel
from data_process.dataset import (
    apply_normalization,
    compute_chain_normalization,
    load_chain_properties,
    pad_chain_features,
    resolve_normalization,
)
from train_fk import build_models, decoder_forward, infer_link_tokens, pooled_latent
from visualize_fk import reconstruct_mesh


@dataclass
class MethodBundle:
    name: str
    cfg: DictConfig
    models: Dict[str, torch.nn.Module]
    norm_mode: str
    norm_center_mode: str
    chain_centers: Dict[int, torch.Tensor]
    chain_scales: Dict[int, torch.Tensor]


def _prepare_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA unavailable, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def _load_checkpoint(models: Dict[str, torch.nn.Module], ckpt_path: str) -> Optional[int]:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    torch.serialization.add_safe_globals([DictConfig])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = state.get("epoch") if isinstance(state, dict) else None
    state_dicts = state.get("models", state)
    for name, module in models.items():
        if name in state_dicts:
            module.load_state_dict(state_dicts[name], strict=False)
        else:
            raise KeyError(f"model '{name}' not found in checkpoint")
    for module in models.values():
        module.eval()
    return epoch


def _enumerate_link_masks(num_links: int, *, min_visible: int = 1) -> List[torch.Tensor]:
    if num_links <= 0:
        return [torch.zeros((0,), dtype=torch.bool)]
    if num_links <= min_visible:
        return [torch.ones((num_links,), dtype=torch.bool)]
    max_mask = num_links - min_visible
    masks = []
    link_indices = list(range(num_links))
    for k in range(max_mask + 1):
        for masked_indices in itertools.combinations(link_indices, k):
            mask = torch.ones(num_links, dtype=torch.bool)
            if masked_indices:
                mask[list(masked_indices)] = False
            masks.append(mask)
    return masks


def _select_masks(
    num_links: int,
    mask_cfg: DictConfig,
    rng: np.random.Generator,
) -> List[Dict[str, torch.Tensor]]:
    mode = str(getattr(mask_cfg, "mode", "none")).lower()
    min_visible = int(getattr(mask_cfg, "min_visible", 1))

    if mode == "none":
        mask = torch.ones(num_links, dtype=torch.bool)
        return [{"mask": mask, "mask_idx": 0, "num_masked": 0}]

    if mode == "specific":
        raw = list(getattr(mask_cfg, "specific", []))
        if len(raw) != num_links:
            raise ValueError(f"specific mask length {len(raw)} != num_links {num_links}")
        mask = torch.tensor(raw, dtype=torch.bool)
        num_masked = int((~mask).sum().item())
        return [{"mask": mask, "mask_idx": 0, "num_masked": num_masked}]

    if mode == "index":
        idx = int(getattr(mask_cfg, "index", 0))
        masks = _enumerate_link_masks(num_links, min_visible=min_visible)
        if idx < 0 or idx >= len(masks):
            raise ValueError(f"mask index {idx} out of range for {len(masks)} masks")
        mask = masks[idx]
        num_masked = int((~mask).sum().item())
        return [{"mask": mask, "mask_idx": idx, "num_masked": num_masked}]

    if mode == "num_masked":
        num_masked = int(getattr(mask_cfg, "num_masked", 0))
        if num_masked > num_links - min_visible:
            raise ValueError("num_masked too large for min_visible constraint")
        link_indices = list(range(num_links))
        masked_indices = rng.choice(link_indices, size=num_masked, replace=False).tolist()
        mask = torch.ones(num_links, dtype=torch.bool)
        if masked_indices:
            mask[masked_indices] = False
        return [{"mask": mask, "mask_idx": 0, "num_masked": num_masked}]

    if mode == "all":
        masks = _enumerate_link_masks(num_links, min_visible=min_visible)
        out = []
        for idx, mask in enumerate(masks):
            num_masked = int((~mask).sum().item())
            out.append({"mask": mask, "mask_idx": idx, "num_masked": num_masked})
        return out

    raise ValueError(f"Unsupported mask mode: {mode}")


def _build_grid_points(points: torch.Tensor, grid_res: int, scale_factor: float) -> torch.Tensor:
    if points.numel() == 0:
        raise ValueError("No points to build grid from.")
    radius = float(points.norm(dim=-1).max().item())
    grid_scale = radius * scale_factor
    coords = torch.linspace(-grid_scale, grid_scale, grid_res)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)


def _load_method_bundle(
    method_cfg: DictConfig,
    device: torch.device,
    eval_indices: Iterable[int],
    data_source: str,
) -> MethodBundle:
    cfg = OmegaConf.load(to_absolute_path(str(method_cfg.config)))
    models = build_models(cfg, device)
    ckpt_path = to_absolute_path(str(method_cfg.checkpoint))
    epoch = _load_checkpoint(models, ckpt_path)
    logging.info("Loaded %s (epoch=%s)", method_cfg.name, epoch)

    norm_mode = str(getattr(cfg.data, "normalize_mode", "none")).lower()
    center_mode = str(getattr(cfg.data, "normalize_center_mode", "zero")).lower()
    norm_indices = _resolve_indices(cfg.data.get("indices")) or list(eval_indices)
    chain_centers, chain_scales = compute_chain_normalization(
        data_source,
        norm_indices,
        normalize_mode=norm_mode,
        normalize_center_mode=center_mode,
    )

    return MethodBundle(
        name=str(method_cfg.name),
        cfg=cfg,
        models=models,
        norm_mode=norm_mode,
        norm_center_mode=center_mode,
        chain_centers=chain_centers,
        chain_scales=chain_scales,
    )


def _resolve_indices(entry) -> List[int]:
    if entry is None:
        return []
    if isinstance(entry, DictConfig):
        if "start" in entry and "end" in entry:
            return list(range(int(entry.start), int(entry.end)))
    if isinstance(entry, (list, tuple)):
        return [int(x) for x in entry]
    raise ValueError(f"Unsupported indices format: {entry}")


def _category_for_class(cls: int, seen_set: set[int], unseen_set: set[int]) -> str:
    if cls in seen_set:
        return "seen"
    if cls in unseen_set:
        return "unseen"
    return "unknown"


def _plot_boxplot(records: List[Dict], title: str, out_path: str, save_html: bool) -> None:
    if not records:
        return
    fig = px.box(records, x="method", y="value", points="outliers")
    fig.update_layout(title=title, xaxis_title="method", yaxis_title="metric")
    try:
        fig.write_image(out_path)
    except Exception as exc:
        html_path = out_path + ".html"
        fig.write_html(html_path)
        logging.warning("write_image failed (%s); wrote HTML to %s", exc, html_path)
    if save_html:
        fig.write_html(out_path + ".html")


def _save_plotly_figure(fig: go.Figure, out_path: str, save_html: bool) -> None:
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


def _mesh_trace(mesh: trimesh.Trimesh, *, name: str, color: str, opacity: float) -> go.Mesh3d:
    verts = mesh.vertices
    faces = mesh.faces
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name=name,
        opacity=opacity,
        color=color,
    )


def save_mesh_comparison(
    *,
    pred_mesh: trimesh.Trimesh,
    gt_mesh: trimesh.Trimesh,
    out_path: str,
    save_html: bool,
    show_pred_only: bool = False,
) -> None:
    pred = pred_mesh.copy()
    gt = gt_mesh.copy()

    if not show_pred_only and not gt.is_empty and not pred.is_empty:
        bounds = pred.bounds
        shift = bounds[1, 0] - bounds[0, 0]
        gt.apply_translation([shift * 1.2, 0.0, 0.0])

    fig = go.Figure()
    fig.add_trace(_mesh_trace(pred, name="pred", color="#33B3FF", opacity=0.7))
    if not show_pred_only:
        fig.add_trace(_mesh_trace(gt, name="gt", color="#CCCCCC", opacity=0.5))
    fig.update_layout(scene=dict(aspectmode="data"), title="Pred vs GT")
    _save_plotly_figure(fig, out_path, save_html)


def _filter_records(
    records: List[Dict],
    *,
    metric: str,
    category: str,
    length: str | int,
    mask_mode: str,
    mask_value: str | int,
) -> List[Dict]:
    filtered = []
    for rec in records:
        if metric != "all" and rec["metric"] != metric:
            continue
        if category != "all" and rec["category"] != category:
            continue
        if length != "all" and rec["length"] != int(length):
            continue
        if mask_mode == "num_masked" and mask_value != "all":
            if rec["num_masked"] != int(mask_value):
                continue
        if mask_mode == "mask_idx" and mask_value != "all":
            if rec["mask_idx"] != int(mask_value):
                continue
        filtered.append(rec)
    return filtered


def _stringify_token(value) -> str:
    if value == "all":
        return "all"
    return str(value)


def _save_plots(
    records: List[Dict],
    *,
    output_dir: str,
    plot_cfg: DictConfig,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    metrics_available = sorted({rec["metric"] for rec in records})
    categories_available = sorted({rec["category"] for rec in records})
    lengths_available = sorted({rec["length"] for rec in records})
    mask_nums_available = sorted({rec["num_masked"] for rec in records})
    mask_idxs_available = sorted({rec["mask_idx"] for rec in records})

    metric_cfg = getattr(plot_cfg, "metric", "all")
    category_cfg = getattr(plot_cfg, "category", "all")
    length_cfg = getattr(plot_cfg, "length", "all")
    mask_mode = str(getattr(plot_cfg, "mask_mode", "none")).lower()
    save_all = bool(getattr(plot_cfg, "save_all_combinations", False))
    save_html = bool(getattr(plot_cfg, "save_html", False))

    if save_all:
        metric_list = metrics_available
        category_list = ["all"] + categories_available if len(categories_available) > 1 else categories_available
        length_list = ["all"] + lengths_available if len(lengths_available) > 1 else lengths_available
        if mask_mode == "num_masked":
            mask_list = ["all"] + mask_nums_available if len(mask_nums_available) > 1 else mask_nums_available
        elif mask_mode == "mask_idx":
            mask_list = ["all"] + mask_idxs_available if len(mask_idxs_available) > 1 else mask_idxs_available
        else:
            mask_list = ["all"]
    else:
        metric_list = metric_cfg if isinstance(metric_cfg, list) else [metric_cfg]
        category_list = category_cfg if isinstance(category_cfg, list) else [category_cfg]
        length_list = length_cfg if isinstance(length_cfg, list) else [length_cfg]
        if mask_mode == "num_masked":
            mask_list = getattr(plot_cfg, "mask_values", "all")
        elif mask_mode == "mask_idx":
            mask_list = getattr(plot_cfg, "mask_indices", "all")
        else:
            mask_list = ["all"]
        if not isinstance(mask_list, list):
            mask_list = [mask_list]

    for metric in tqdm(metric_list, desc="Plot metrics"):
        for category in category_list:
            for length in length_list:
                for mask_value in mask_list:
                    filtered = _filter_records(
                        records,
                        metric=metric,
                        category=category,
                        length=length,
                        mask_mode=mask_mode,
                        mask_value=mask_value,
                    )
                    if not filtered:
                        print(f"Skipping empty plot for {metric}, {category}, {length}, {mask_value}")
                        continue
                    title = f"{metric} | cat={category} | len={length} | mask={mask_value}"
                    fname = (
                        f"box_{metric}_cat-{_stringify_token(category)}"
                        f"_len-{_stringify_token(length)}"
                        f"_mask-{_stringify_token(mask_value)}.png"
                    )
                    out_path = os.path.join(output_dir, fname)
                    _plot_boxplot(filtered, title=title, out_path=out_path, save_html=save_html)


@hydra.main(config_path="conf", config_name="config_validate_fk", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    device = _prepare_device(cfg.validation.device)
    seed = int(getattr(cfg.validation, "seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_source = to_absolute_path(cfg.data.data_source)
    max_num_links = int(cfg.data.max_num_links)
    num_instances = int(cfg.data.num_instances)

    train_indices = _resolve_indices(cfg.data.get("indices"))
    val_indices = _resolve_indices(cfg.data.get("val_indices"))
    eval_indices = _resolve_indices(cfg.data.get("eval_indices", None))
    if not eval_indices:
        eval_indices = val_indices

    seen_set = set(train_indices)
    unseen_set = set(val_indices)

    methods = []
    for method_cfg in cfg.validation.methods:
        methods.append(_load_method_bundle(method_cfg, device, eval_indices, data_source))

    fix_q_samples = bool(cfg.data.get("fix_q_samples", False))
    q_cache_dir = os.path.join(data_source, "chain_qs")
    os.makedirs(q_cache_dir, exist_ok=True)

    mask_cfg = cfg.validation.mask
    metric_cfg = cfg.validation.metrics
    vis_cfg = getattr(cfg.validation, "visualize_mesh", None)
    vis_enabled = bool(getattr(vis_cfg, "enabled", False)) if vis_cfg is not None else False
    if vis_enabled and not torch.cuda.is_available():
        logging.warning("Mesh visualization disabled (CUDA required for reconstruction).")
        vis_enabled = False
    vis_instance_cfg = getattr(vis_cfg, "instance_idx", 0) if vis_cfg is not None else 0
    vis_all_instances = str(vis_instance_cfg).lower() == "all"
    vis_instance_idx = 0 if vis_all_instances else int(vis_instance_cfg)
    mesh_out_dir = (
        to_absolute_path(getattr(vis_cfg, "output_dir", "./outputs/validate_fk/meshes"))
        if vis_cfg is not None
        else None
    )
    if vis_enabled and mesh_out_dir is not None:
        os.makedirs(mesh_out_dir, exist_ok=True)

    results: List[Dict] = []
    rng = np.random.default_rng(seed)

    for class_idx in tqdm(eval_indices, desc="Classes"):
        link_features, joint_features, link_bps, chain_props = load_chain_properties(
            data_source, class_idx
        )
        link_features, joint_features, link_bps, num_links = pad_chain_features(
            link_features,
            joint_features,
            link_bps,
            max_num_links=max_num_links,
        )

        urdf_path = os.path.join(data_source, "out_chains_v2", f"chain_{class_idx}.urdf")
        chain_model = ChainModel(urdf_path=urdf_path, samples_per_link=128, device="cpu")

        if fix_q_samples:
            q_samples_path = os.path.join(
                q_cache_dir, f"chain_{class_idx}_{num_instances}_samples.pt"
            )
            if os.path.exists(q_samples_path):
                q_samples = torch.load(q_samples_path, weights_only=False).float()
            else:
                q_samples = chain_model.sample_q(num_instances).float().cpu()
                torch.save(q_samples, q_samples_path)
        else:
            q_samples = chain_model.sample_q(num_instances).float().cpu()

        for inst_idx in tqdm(range(num_instances), desc=f"instances in {class_idx}", leave=False):
            q = q_samples[inst_idx]
            chain_model.update_status(q)

            mask_infos = _select_masks(num_links, mask_cfg, rng)

            # Precompute per-method link tokens for this instance
            method_tokens: Dict[str, torch.Tensor] = {}
            for method in methods:
                link_tokens, _ = infer_link_tokens(
                    class_real_idx=class_idx,
                    q_values=q.view(1, -1),
                    cfg=method.cfg,
                    models=method.models,
                    data_source=data_source,
                    device=device,
                    chain_props=chain_props,
                    max_num_links=max_num_links,
                )
                method_tokens[method.name] = link_tokens

            for mask_info in mask_infos:
                mask = mask_info["mask"]
                mask_padded = torch.zeros(max_num_links, dtype=torch.bool)
                mask_padded[:num_links] = mask
                num_masked = mask_info["num_masked"]
                mask_idx = mask_info["mask_idx"]

                sample_seed = seed + class_idx * 100000 + inst_idx * 1000 + mask_idx
                torch.manual_seed(sample_seed)

                surface_pts = None
                deepsdf_pts = None
                deepsdf_sdf = None
                iou_pts = None
                iou_sdf = None

                if metric_cfg.sdf_l1_surface.enabled:
                    surface_pts = chain_model.sample_surface_points(
                        metric_cfg.sdf_l1_surface.num_samples,
                        mask=mask,
                    )[0]

                if metric_cfg.sdf_l1_deepsdf.enabled:
                    deepsdf_pts = chain_model.sample_query_points(
                        metric_cfg.sdf_l1_deepsdf.num_samples,
                        mask=mask,
                        var=metric_cfg.sdf_l1_deepsdf.var,
                        near_surface_ratio=metric_cfg.sdf_l1_deepsdf.near_surface_ratio,
                    )[0]
                    deepsdf_sdf = chain_model.query_sdf(
                        deepsdf_pts.unsqueeze(0),
                        mask=mask,
                    )[0]

                if metric_cfg.iou.enabled:
                    pc_full = chain_model.get_transformed_links_pc(
                        num_points=None,
                        mask=mask,
                    )[0, :, :3]
                    iou_pts = _build_grid_points(
                        pc_full,
                        metric_cfg.iou.grid_res,
                        metric_cfg.iou.grid_scale_factor,
                    )
                    iou_sdf = chain_model.query_sdf(
                        iou_pts.unsqueeze(0),
                        mask=mask,
                    )[0]

                gt_mesh = None
                gt_radius = None
                if vis_enabled and (vis_all_instances or inst_idx == vis_instance_idx):
                    gt_mesh = chain_model.get_trimesh_q(
                        0,
                        boolean_merged=True,
                        mask=mask,
                    )
                    if not gt_mesh.is_empty and len(gt_mesh.vertices) > 0:
                        gt_radius = float(np.linalg.norm(gt_mesh.vertices, axis=1).max())

                for method in methods:
                    link_tokens = method_tokens[method.name]
                    latent = pooled_latent(
                        link_tokens,
                        mask_padded.unsqueeze(0).to(device),
                        mode="max",
                    )

                    center, scale = resolve_normalization(
                        chain_model,
                        class_real_idx=class_idx,
                        normalize_mode=method.norm_mode,
                        normalize_center_mode=method.norm_center_mode,
                        chain_centers=method.chain_centers,
                        chain_scales=method.chain_scales,
                        link_mask=mask,
                    )

                    category = _category_for_class(class_idx, seen_set, unseen_set)
                    length = num_links

                    if (
                        vis_enabled
                        and (vis_all_instances or inst_idx == vis_instance_idx)
                        and gt_mesh is not None
                        and gt_radius is not None
                    ):
                        vis_cfg = cfg.validation.visualize_mesh
                        method_dir = os.path.join(mesh_out_dir, method.name)
                        os.makedirs(method_dir, exist_ok=True)
                        out_stem = os.path.join(
                            method_dir, f"cls{class_idx}_mask{mask_idx}_m{num_masked}"
                        )
                        decoder_device = next(method.models["decoder"].parameters()).device
                        norm_center = center.to(decoder_device) if center is not None else None
                        norm_scale = scale.to(decoder_device) if scale is not None else None

                        reconstruct_mesh(
                            decoder=method.models["decoder"],
                            latent=latent,
                            out_path=out_stem,
                            N=getattr(vis_cfg, "grid_res", 128),
                            max_batch=getattr(vis_cfg, "max_batch", 262144),
                            grid_scale=gt_radius * float(getattr(vis_cfg, "grid_scale_factor", 1.4)),
                            offset=None,
                            scale=None,
                            normalize_center=norm_center,
                            normalize_scale=norm_scale,
                        )

                        pred_mesh = trimesh.load(out_stem + ".ply", force="mesh")
                        save_mesh_comparison(
                            pred_mesh=pred_mesh,
                            gt_mesh=gt_mesh,
                            out_path=out_stem + ".png",
                            save_html=bool(getattr(vis_cfg, "save_html", False)),
                            show_pred_only=bool(getattr(vis_cfg, "show_pred_only", False)),
                        )

                    if metric_cfg.sdf_l1_surface.enabled and surface_pts is not None:
                        coords = surface_pts.clone()
                        coords, _ = apply_normalization(
                            coords,
                            None,
                            center=center,
                            scale=scale,
                        )
                        pred = decoder_forward(
                            method.models["decoder"],
                            latent,
                            coords.unsqueeze(0).to(device),
                        ).squeeze(0)
                        value = pred.abs().mean().item()
                        results.append(
                            {
                                "method": method.name,
                                "metric": "sdf_l1_surface",
                                "value": value,
                                "class_idx": class_idx,
                                "instance_idx": inst_idx,
                                "category": category,
                                "length": length,
                                "num_masked": num_masked,
                                "mask_idx": mask_idx,
                            }
                        )

                    if metric_cfg.sdf_l1_deepsdf.enabled and deepsdf_pts is not None:
                        coords = deepsdf_pts.clone()
                        sdf = deepsdf_sdf.clone() if deepsdf_sdf is not None else None
                        coords, sdf = apply_normalization(
                            coords,
                            sdf,
                            center=center,
                            scale=scale,
                        )
                        pred = decoder_forward(
                            method.models["decoder"],
                            latent,
                            coords.unsqueeze(0).to(device),
                        ).squeeze(0)
                        value = (pred.squeeze(-1).cpu() - sdf).abs().mean().item()
                        results.append(
                            {
                                "method": method.name,
                                "metric": "sdf_l1_deepsdf",
                                "value": value,
                                "class_idx": class_idx,
                                "instance_idx": inst_idx,
                                "category": category,
                                "length": length,
                                "num_masked": num_masked,
                                "mask_idx": mask_idx,
                            }
                        )

                    if metric_cfg.iou.enabled and iou_pts is not None:
                        coords = iou_pts.clone()
                        sdf = iou_sdf.clone() if iou_sdf is not None else None
                        coords, sdf = apply_normalization(
                            coords,
                            sdf,
                            center=center,
                            scale=scale,
                        )
                        pred = decoder_forward(
                            method.models["decoder"],
                            latent,
                            coords.unsqueeze(0).to(device),
                        ).squeeze(0)
                        pred_occ = pred.squeeze(-1).cpu() <= 0
                        gt_occ = sdf <= 0
                        intersection = (pred_occ & gt_occ).sum().item()
                        union = (pred_occ | gt_occ).sum().item()
                        value = float(intersection / max(union, 1))
                        results.append(
                            {
                                "method": method.name,
                                "metric": "iou_sdf",
                                "value": value,
                                "class_idx": class_idx,
                                "instance_idx": inst_idx,
                                "category": category,
                                "length": length,
                                "num_masked": num_masked,
                                "mask_idx": mask_idx,
                            }
                        )

    output_dir = to_absolute_path(cfg.validation.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    torch.save(results, os.path.join(output_dir, "results.pt"))
    with open(os.path.join(output_dir, "results.csv"), "w", encoding="utf-8") as f:
        header = [
            "method",
            "metric",
            "value",
            "class_idx",
            "instance_idx",
            "category",
            "length",
            "num_masked",
            "mask_idx",
        ]
        f.write(",".join(header) + "\n")
        for rec in results:
            row = [str(rec[h]) for h in header]
            f.write(",".join(row) + "\n")

    _save_plots(results, output_dir=output_dir, plot_cfg=cfg.validation.plots)


if __name__ == "__main__":
    main()
