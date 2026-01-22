from __future__ import annotations

from dataclasses import dataclass
import itertools
import logging
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import hydra
import numpy as np
import plotly.express as px
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from robot_model.chain_model import ChainModel
from train_fk import build_models, decoder_forward, inference, pooled_latent


@dataclass
class MethodBundle:
    name: str
    cfg: DictConfig
    models: Dict[str, torch.nn.Module]
    norm_mode: str
    norm_center_mode: str
    chain_centers: Dict[int, torch.Tensor]
    chain_scales: Dict[int, torch.Tensor]
    dataset_center: Optional[torch.Tensor]
    dataset_scale: Optional[torch.Tensor]


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


def _prepare_link_bps(bps_info) -> torch.Tensor:
    link_bps_scdistances = [
        np.concatenate(
            [
                bp["offsets"],
                np.ones((bp["offsets"].shape[0], 1)) * (bp["scale_to_unit"] * 1.0),
            ],
            axis=-1,
        )
        for bp in bps_info
    ]
    return torch.tensor(np.stack(link_bps_scdistances, axis=0)).float().flatten(1)


def _pad_tensor(x: torch.Tensor, *, length: int) -> torch.Tensor:
    if x.size(0) > length:
        raise ValueError(f"Cannot pad length {x.size(0)} to smaller length {length}")
    if x.size(0) == length:
        return x
    pad = torch.zeros((length - x.size(0), *x.shape[1:]), dtype=x.dtype)
    return torch.cat([x, pad], dim=0)


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


def _compute_center_scale(
    pc: torch.Tensor,
    center_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pc.numel() == 0:
        return torch.zeros(3), torch.tensor(1.0)
    pts = pc[:, :3]
    if center_mode == "mesh":
        pts_min = pts.min(dim=0).values
        pts_max = pts.max(dim=0).values
        center = 0.5 * (pts_min + pts_max)
    elif center_mode == "zero":
        center = torch.zeros_like(pts[0])
    else:
        raise ValueError(f"Unsupported normalize_center_mode: {center_mode}")
    scale = (pts - center).norm(dim=-1).max().clamp_min(1e-6)
    return center, scale


def _build_normalization_cache(
    indices: Iterable[int],
    data_source: str,
    norm_mode: str,
    center_mode: str,
) -> tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    chain_centers: Dict[int, torch.Tensor] = {}
    chain_scales: Dict[int, torch.Tensor] = {}
    dataset_center = None
    dataset_scale = None

    if norm_mode not in {"chain", "dataset"}:
        return chain_centers, chain_scales, dataset_center, dataset_scale

    pcs = []
    for class_idx in indices:
        urdf_path = os.path.join(data_source, "out_chains_v2", f"chain_{class_idx}.urdf")
        chain_model = ChainModel(urdf_path=urdf_path, samples_per_link=128, device="cpu")
        chain_model.update_status(torch.zeros(chain_model.dof))
        pc_full = chain_model.get_transformed_links_pc(
            num_points=None,
            mask=torch.ones(chain_model.num_links, dtype=torch.bool),
        )[0, :, :3]
        pcs.append(pc_full)
        if norm_mode == "chain":
            center, scale = _compute_center_scale(pc_full, center_mode)
            chain_centers[class_idx] = center
            chain_scales[class_idx] = scale

    if norm_mode == "dataset":
        all_pts = torch.cat(pcs, dim=0)
        center, scale = _compute_center_scale(all_pts, center_mode)
        dataset_center = center
        dataset_scale = scale
        for class_idx in indices:
            chain_centers[class_idx] = center
            chain_scales[class_idx] = scale

    return chain_centers, chain_scales, dataset_center, dataset_scale


def _apply_normalization(
    coords: torch.Tensor,
    sdf: Optional[torch.Tensor],
    *,
    norm_mode: str,
    center_mode: str,
    class_idx: int,
    chain_model: ChainModel,
    link_mask: torch.Tensor,
    chain_centers: Dict[int, torch.Tensor],
    chain_scales: Dict[int, torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if norm_mode == "none":
        return coords, sdf

    if norm_mode == "instance":
        pc_full = chain_model.get_transformed_links_pc(num_points=None, mask=link_mask)[0, :, :3]
        center, scale = _compute_center_scale(pc_full, center_mode)
    else:
        if class_idx not in chain_centers:
            current_q = chain_model.q.clone()
            chain_model.update_status(torch.zeros(chain_model.dof))
            pc_full = chain_model.get_transformed_links_pc(
                num_points=None,
                mask=torch.ones(chain_model.num_links, dtype=torch.bool),
            )[0, :, :3]
            center, scale = _compute_center_scale(pc_full, center_mode)
            chain_centers[class_idx] = center
            chain_scales[class_idx] = scale
            chain_model.update_status(current_q.squeeze(0))
        center = chain_centers[class_idx]
        scale = chain_scales[class_idx]

    coords = (coords - center) / scale
    if sdf is not None:
        sdf = sdf / scale
    return coords, sdf


def _build_token_mask(num_links: int, max_num_links: int) -> torch.Tensor:
    token_mask = torch.zeros(3 * max_num_links - 2, dtype=torch.bool)
    token_mask[: 3 * num_links - 2] = True
    return token_mask


def _build_link_mask_full(num_links: int, max_num_links: int) -> torch.Tensor:
    mask = torch.zeros(max_num_links, dtype=torch.bool)
    mask[:num_links] = True
    return mask


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
    chain_centers, chain_scales, dataset_center, dataset_scale = _build_normalization_cache(
        norm_indices,
        data_source,
        norm_mode,
        center_mode,
    )

    return MethodBundle(
        name=str(method_cfg.name),
        cfg=cfg,
        models=models,
        norm_mode=norm_mode,
        norm_center_mode=center_mode,
        chain_centers=chain_centers,
        chain_scales=chain_scales,
        dataset_center=dataset_center,
        dataset_scale=dataset_scale,
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
        category_list = ["all"] + categories_available
        length_list = ["all"] + lengths_available
        if mask_mode == "num_masked":
            mask_list = ["all"] + mask_nums_available
        elif mask_mode == "mask_idx":
            mask_list = ["all"] + mask_idxs_available
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

    for metric in metric_list:
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

    results: List[Dict] = []
    rng = np.random.default_rng(seed)

    for class_idx in tqdm(eval_indices, desc="Classes"):
        chain_props = np.load(
            os.path.join(data_source, "out_chains_v2", f"chain_{class_idx}_properties.npz"),
            allow_pickle=True,
        )
        link_features = torch.tensor(chain_props["links_property"]).float()
        joint_features = torch.tensor(chain_props["joints_property"]).float()
        link_bps = _prepare_link_bps(chain_props["bpses"])

        num_links = link_features.size(0)
        if num_links > max_num_links:
            raise ValueError(
                f"class {class_idx} has {num_links} links > max_num_links {max_num_links}"
            )

        link_features = _pad_tensor(link_features, length=max_num_links)
        link_bps = _pad_tensor(link_bps, length=max_num_links)
        joint_features = _pad_tensor(joint_features, length=max_num_links - 1)

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

        token_mask = _build_token_mask(num_links, max_num_links)
        link_mask_full = _build_link_mask_full(num_links, max_num_links)

        for inst_idx in tqdm(range(num_instances), desc=f"cls {class_idx}", leave=False):
            q = q_samples[inst_idx]
            chain_model.update_status(q)

            mask_infos = _select_masks(num_links, mask_cfg, rng)

            # Precompute per-method link tokens for this instance
            method_tokens: Dict[str, torch.Tensor] = {}
            for method in methods:
                batch = {
                    "sdf_samples": torch.zeros(1, 1, 4, device=device),
                    "chain_q": q.view(1, -1).to(device),
                    "link_features": link_features.unsqueeze(0).to(device),
                    "joint_features": joint_features.unsqueeze(0).to(device),
                    "link_bps_scdistances": link_bps.unsqueeze(0).to(device),
                    "mask": token_mask.unsqueeze(0).to(device),
                    "link_mask": link_mask_full.unsqueeze(0).to(device),
                }
                _, _, link_tokens = inference(
                    batch,
                    method.models,
                    cfg=method.cfg,
                    device=device,
                    coords_override=torch.zeros(1, 1, 3, device=device),
                    return_link_tokens=True,
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

                for method in methods:
                    link_tokens = method_tokens[method.name]
                    latent = pooled_latent(
                        link_tokens,
                        mask_padded.unsqueeze(0).to(device),
                        mode="max",
                    )

                    category = _category_for_class(class_idx, seen_set, unseen_set)
                    length = num_links

                    if metric_cfg.sdf_l1_surface.enabled and surface_pts is not None:
                        coords = surface_pts.clone()
                        coords, _ = _apply_normalization(
                            coords,
                            None,
                            norm_mode=method.norm_mode,
                            center_mode=method.norm_center_mode,
                            class_idx=class_idx,
                            chain_model=chain_model,
                            link_mask=mask,
                            chain_centers=method.chain_centers,
                            chain_scales=method.chain_scales,
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
                        coords, sdf = _apply_normalization(
                            coords,
                            sdf,
                            norm_mode=method.norm_mode,
                            center_mode=method.norm_center_mode,
                            class_idx=class_idx,
                            chain_model=chain_model,
                            link_mask=mask,
                            chain_centers=method.chain_centers,
                            chain_scales=method.chain_scales,
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
                        coords, sdf = _apply_normalization(
                            coords,
                            sdf,
                            norm_mode=method.norm_mode,
                            center_mode=method.norm_center_mode,
                            class_idx=class_idx,
                            chain_model=chain_model,
                            link_mask=mask,
                            chain_centers=method.chain_centers,
                            chain_scales=method.chain_scales,
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
