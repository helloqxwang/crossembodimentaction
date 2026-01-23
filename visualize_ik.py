"""Visualize IK predictions against ground-truth meshes (flow matching).

Runs flow-matching style inference to predict joint values, converts them to
continuous angles if needed, and visualizes predicted vs. GT meshes. Also
supports attention map visualization per transformer layer.

Config: uses ``conf/config_visualize_ik.yaml`` by default; override at runtime for
checkpoint or dataset tweaks, e.g.
``python visualize_ik.py visualize.test_ckpt=... visualize.attention.enabled=true``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, List
import logging
import os
import math

import hydra
import numpy as np
import torch
import trimesh
import viser
import matplotlib.pyplot as plt
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from data_process.dataset import get_ik_dataloader, load_chain_properties
from robot_model.chain_model import ChainModel
from networks.transformer import TransformerEncoder
from train_ik_fm import _prepare_device, build_models, infer_ik_flow
from train_fk import load_models_from_checkpoint, infer_link_tokens, pooled_latent


def _load_models(cfg: DictConfig, device: torch.device) -> Tuple[Dict[str, torch.nn.Module], Optional[int]]:
    models = build_models(cfg, device)
    ckpt_path = getattr(cfg.visualize, "test_ckpt", None)
    if not ckpt_path:
        ckpt_path = getattr(cfg.training, "pretrained_ckpt", None)
    if not ckpt_path:
        raise FileNotFoundError("visualize.test_ckpt must be set for visualization")
    ckpt_path = to_absolute_path(str(ckpt_path))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    torch.serialization.add_safe_globals([DictConfig])
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = state.get("epoch") if isinstance(state, dict) else None
    state_dicts = state.get("models", state)

    for name, module in models.items():
        if name in state_dicts:
            module.load_state_dict(state_dicts[name], strict=False)
        else:
            raise KeyError(f"model '{name}' not found in checkpoint")

    for m in models.values():
        m.eval()
    return models, epoch


def _resolve_urdf_path(data_root: str, class_idx: int) -> str:
    urdf_path = os.path.join(data_root, "out_chains_v2", f"chain_{class_idx}.urdf")
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF not found for class {class_idx}: {urdf_path}")
    return urdf_path


def _get_chain(chain_cache: Dict[int, ChainModel], data_root: str, class_idx: int, device: torch.device) -> ChainModel:
    if class_idx in chain_cache:
        return chain_cache[class_idx]
    chain_cache[class_idx] = ChainModel(urdf_path=_resolve_urdf_path(data_root, class_idx), device=device)
    return chain_cache[class_idx]


def _override_batch_with_fk_tokens(
    batch: Dict[str, torch.Tensor],
    *,
    fk_cfg: DictConfig,
    fk_models: Dict[str, torch.nn.Module],
    fk_data_root: str,
    fk_chain_props_cache: Dict[int, Dict],
    fk_chain_cache: Dict[int, ChainModel],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Sample new q for each instance and recompute sdf_tokens via FK inference."""
    class_indices = batch["class_idx"]
    batch_size = len(class_indices)
    chain_q = batch["chain_q"].clone()
    sdf_tokens = batch["sdf_tokens"].clone()
    link_mask = batch.get("link_mask", None)

    class_groups: Dict[int, List[int]] = {}
    for i in range(batch_size):
        cls = int(class_indices[i])
        class_groups.setdefault(cls, []).append(i)

    for cls, idxs in class_groups.items():
        chain = _get_chain(fk_chain_cache, fk_data_root, cls, "cpu")
        q_samples = chain.sample_q(len(idxs)).float().cpu()
        dof = int(chain.dof)

        for local_i, global_i in enumerate(idxs):
            chain_q[global_i, :dof] = q_samples[local_i]
            if chain_q.size(1) > dof:
                chain_q[global_i, dof:] = 0.0

        if cls not in fk_chain_props_cache:
            _, _, _, chain_props = load_chain_properties(fk_data_root, cls)
            fk_chain_props_cache[cls] = chain_props
        chain_props = fk_chain_props_cache[cls]

        link_tokens, _ = infer_link_tokens(
            class_real_idx=cls,
            q_values=q_samples,
            cfg=fk_cfg,
            models=fk_models,
            data_source=fk_data_root,
            device=device,
            chain_props=chain_props,
            max_num_links=int(fk_cfg.data.max_num_links),
        )

        if link_mask is None:
            mask_batch = torch.ones(
                (len(idxs), link_tokens.size(1)), dtype=torch.bool, device=device
            )
        else:
            mask_batch = link_mask[idxs].to(device)
            if mask_batch.size(1) != link_tokens.size(1):
                if mask_batch.size(1) > link_tokens.size(1):
                    mask_batch = mask_batch[:, : link_tokens.size(1)]
                else:
                    pad = torch.zeros(
                        (mask_batch.size(0), link_tokens.size(1) - mask_batch.size(1)),
                        dtype=mask_batch.dtype,
                        device=mask_batch.device,
                    )
                    mask_batch = torch.cat([mask_batch, pad], dim=1)

        latent = pooled_latent(link_tokens, mask_batch, mode="max").detach().cpu()
        sdf_tokens[idxs] = latent

    batch["chain_q"] = chain_q
    batch["sdf_tokens"] = sdf_tokens
    return batch


def _split_gt_mesh_by_mask(
    chain: ChainModel,
    link_mask: torch.Tensor | np.ndarray | None,
) -> Tuple[trimesh.Trimesh, Optional[trimesh.Trimesh]]:
    if link_mask is None:
        return chain.get_trimesh_q(idx=0, boolean_merged=True), None

    mask_full = torch.as_tensor(link_mask).bool().cpu().flatten()
    mask_full = mask_full[: chain.num_links]
    if mask_full.numel() == 0:
        return chain.get_trimesh_q(idx=0, boolean_merged=True), None

    if mask_full.any():
        gt_mesh = chain.get_trimesh_q(idx=0, boolean_merged=True, mask=mask_full)
    else:
        gt_mesh = chain.get_trimesh_q(idx=0, boolean_merged=True)

    masked_mesh = None
    if (~mask_full).any():
        masked_mesh = chain.get_trimesh_q(idx=0, boolean_merged=True, mask=~mask_full)

    return gt_mesh, masked_mesh


def visualize_meshes(
    pred_mesh: trimesh.Trimesh,
    gt_mesh: Optional[trimesh.Trimesh] = None,
    masked_mesh: Optional[trimesh.Trimesh] = None,
    *,
    port: int = 8100,
    show_pred_only: bool = False,
):
    server = viser.ViserServer(port=port)
    pred_mesh_vis = pred_mesh.copy()
    pred_mesh_vis.visual.face_colors = [51, 179, 255, 255]
    server.scene.add_mesh_trimesh(name="pred_mesh", mesh=pred_mesh_vis)

    if gt_mesh is not None and not show_pred_only:
        bounds = pred_mesh.bounds
        size_x = float((bounds[1] - bounds[0])[0])
        shift = size_x * 1.4 if size_x > 0 else 0.2
        gt_shifted = gt_mesh.copy()
        gt_shifted.apply_translation([shift, 0, 0])
        gt_shifted.visual.face_colors = [255, 77, 77, 255]
        server.scene.add_mesh_trimesh(name="gt_mesh", mesh=gt_shifted)
        if masked_mesh is not None and not masked_mesh.is_empty:
            masked_shifted = masked_mesh.copy()
            masked_shifted.apply_translation([shift, 0, 0])
            masked_shifted.visual.face_colors = [255, 200, 51, 255]
            server.scene.add_mesh_trimesh(name="gt_masked_mesh", mesh=masked_shifted)
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


def _sincos_to_angle(joint_state: torch.Tensor) -> torch.Tensor:
    return torch.atan2(joint_state[..., 0], joint_state[..., 1])


def _collect_attention_maps(
    model: TransformerEncoder,
    x: torch.Tensor,
    *,
    key_padding_mask: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> List[torch.Tensor]:
    attn_maps: List[torch.Tensor] = []

    def _save_attn(_: torch.nn.Module, __: tuple, output: tuple) -> None:
        if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
            attn_maps.append(output[1].detach().cpu())

    patched_forwards = []
    handles = []
    try:
        for layer in model.encoder.layers:
            mha = layer.self_attn
            orig_forward = mha.forward

            def _patched_forward(*args, orig_forward=orig_forward, **kwargs):
                # TransformerEncoderLayer passes need_weights=False; override to collect maps.
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = True
                return orig_forward(*args, **kwargs)

            patched_forwards.append((mha, orig_forward))
            mha.forward = _patched_forward  # type: ignore[assignment]
            handles.append(mha.register_forward_hook(_save_attn))

        if torch.cuda.is_available() and hasattr(torch.nn, "attention"):
            sdp_ctx = torch.nn.attention.sdpa_kernel(
                [torch.nn.attention.SDPBackend.MATH]
            )
        elif torch.cuda.is_available():
            sdp_ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            )
        else:
            sdp_ctx = torch.backends.cuda.sdp_kernel()
        with sdp_ctx, torch.no_grad():
            _ = model(
                x,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                causal=causal,
            )
    finally:
        for h in handles:
            h.remove()
        for mha, orig_forward in patched_forwards:
            mha.forward = orig_forward

    return attn_maps


def _token_labels(max_num_links: int, valid_tokens: int) -> List[str]:
    labels: List[str] = []
    for i in range(max_num_links):
        labels.append(f"L{i}")
        if i < max_num_links - 1:
            labels.append(f"J{i}")
            labels.append(f"Q{i}")
    labels.append("G")
    return labels[:valid_tokens]


def _save_attention_plots(
    attn_maps: List[torch.Tensor],
    *,
    out_dir: str,
    sample_idx: int,
    token_labels: Optional[List[str]] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    for layer_idx, attn in enumerate(attn_maps):
        if attn.dim() != 3:
            continue
        mat = attn[sample_idx].numpy()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, cmap="viridis", origin="lower")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Attention layer {layer_idx}")
        if token_labels is not None and len(token_labels) == mat.shape[0]:
            ax.set_xticks(range(len(token_labels)))
            ax.set_yticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, rotation=90, fontsize=6)
            ax.set_yticklabels(token_labels, fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"attn_layer_{layer_idx}.png"), dpi=150)
        plt.close(fig)

def slice_batch(batch: Dict[str, torch.Tensor | list], idx: int) -> Dict[str, torch.Tensor | list]:
    """Return a batch_size=1 batch containing only item idx."""
    if idx < 0:
        raise IndexError("idx must be non-negative")
    out: Dict[str, torch.Tensor | list] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.size(0) <= idx:
                raise IndexError(f"idx {idx} out of range for {key} with size {value.size(0)}")
            out[key] = value[idx : idx + 1]
        elif isinstance(value, list):
            if len(value) <= idx:
                raise IndexError(f"idx {idx} out of range for {key} with length {len(value)}")
            out[key] = [value[idx]]
        else:
            out[key] = value
    return out

@hydra.main(config_path="conf", config_name="config_visualize_ik", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    device = _prepare_device(cfg.training.device)
    torch.manual_seed(0)

    models, epoch = _load_models(cfg, device)
    logging.info(f"loaded checkpoint epoch={epoch}")

    data_root = to_absolute_path(cfg.data.data_source)

    val_indices = list(range(cfg.data.val_indices.start, cfg.data.val_indices.end))
    val_loader = get_ik_dataloader(
        data_source=data_root,
        indices=val_indices,
        num_instances=cfg.data.num_instances,
        tokens_dir=to_absolute_path(cfg.data.sdf_token_dir),
        batch_size=cfg.data.val_batch_size,
        max_num_links=cfg.data.max_num_links,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        drop_last=False,
        link_compact_repr=bool(getattr(cfg.models.link_encoder, "compact_repr", False)),
    )

    vis_cfg = getattr(cfg, "visualize", {})
    max_mesh = int(getattr(vis_cfg, "max_mesh", 64))
    port = int(getattr(vis_cfg, "port", 9010))
    show_pred_only = bool(getattr(vis_cfg, "show_pred_only", False))
    infer_cfg = getattr(vis_cfg, "inference", {})
    flow_steps = int(getattr(infer_cfg, "steps", 10))
    flow_step_size = getattr(infer_cfg, "step_size", None)
    q_min = float(getattr(vis_cfg, "q_min", -2.0 * math.pi / 3.0))
    q_max = float(getattr(vis_cfg, "q_max", 2.0 * math.pi / 3.0))
    noise_std = float(getattr(infer_cfg, "noise_std", cfg.flow_matching.noise_std))
    time_embed_dim = int(getattr(cfg.flow_matching, "time_embed_dim", 128))

    fk_cfg = None
    fk_models = None
    fk_data_root = None
    fk_chain_props_cache: Dict[int, Dict] = {}
    fk_chain_cache: Dict[int, ChainModel] = {}
    fk_opts = getattr(vis_cfg, "fk_tokens", {})
    fk_enabled = bool(getattr(fk_opts, "enabled", False))
    if fk_enabled:
        fk_cfg_path = getattr(fk_opts, "config", None)
        fk_ckpt = getattr(fk_opts, "checkpoint", None)
        if not fk_cfg_path or not fk_ckpt:
            raise ValueError("visualize.fk_tokens.config and visualize.fk_tokens.checkpoint must be set")
        fk_cfg = OmegaConf.load(to_absolute_path(str(fk_cfg_path)))
        fk_models, _ = load_models_from_checkpoint(fk_cfg, device, fk_ckpt)
        fk_data_root = to_absolute_path(getattr(fk_cfg.data, "data_source", data_root))

    attn_cfg = getattr(vis_cfg, "attention", {})
    attn_enabled = bool(getattr(attn_cfg, "enabled", False))
    attn_out_dir = to_absolute_path(getattr(attn_cfg, "output_dir", "./outputs/ik_attention"))
    attn_sample_idx = int(getattr(attn_cfg, "sample_idx", 0))

    chain_cache: Dict[int, ChainModel] = {}
    shown = 0

    for batch in val_loader:
        if fk_enabled:
            batch = _override_batch_with_fk_tokens(
                batch,
                fk_cfg=fk_cfg,
                fk_models=fk_models,
                fk_data_root=fk_data_root,
                fk_chain_props_cache=fk_chain_props_cache,
                fk_chain_cache=fk_chain_cache,
                device=device,
            )

        joint_state, joint_mask = infer_ik_flow(
            batch,
            models,
            device,
            noise_std=noise_std,
            time_embed_dim=time_embed_dim,
            steps=flow_steps,
            step_size=flow_step_size,
            use_compact_repr=bool(getattr(cfg.models.link_encoder, "compact_repr", False)),
            return_tokens=False,
        )
        pred_q = _sincos_to_angle(joint_state).clamp(q_min, q_max)

        class_indices = batch["class_idx"]
        instance_indices = batch["instance_idx"]

        if attn_enabled:
            transformer = models["transformer"]
            for i in range(len(class_indices)):
                sliced_batch = slice_batch(batch, i)
                _, _, tokens, key_padding_mask = infer_ik_flow(
                    sliced_batch,
                    models,
                    device,
                    noise_std=noise_std,
                    time_embed_dim=time_embed_dim,
                    steps=flow_steps,
                    step_size=flow_step_size,
                    use_compact_repr=bool(getattr(cfg.models.link_encoder, "compact_repr", False)),
                    return_tokens=True,
                )

                token_labels = _token_labels(cfg.data.max_num_links, tokens.size(1))
                attn_maps = _collect_attention_maps(
                    transformer,
                    tokens,
                    key_padding_mask=key_padding_mask,
                    causal=False,
                )
                _save_attention_plots(
                    attn_maps,
                    out_dir=attn_out_dir,
                    sample_idx=attn_sample_idx,
                    token_labels=token_labels,
                )
                logging.info(f"The current link mask is {sliced_batch['link_mask']}")
                logging.info(f"Saved attention maps for class {class_indices[i]} instance {instance_indices[i]} to {attn_out_dir}")
            # attn_enabled = False

        # for i in range(len(class_indices)):
        #     if shown >= max_mesh:
        #         logging.info("Reached visualization limit; exiting.")
        #         return

        #     cls = class_indices[i]
        #     inst = instance_indices[i]

        #     valid_count = int(joint_mask[i].sum().item())
        #     q_pred = pred_q[i, :valid_count]
        #     chain = _get_chain(chain_cache, data_root, cls, "cpu")
        #     chain.update_status(q_pred.unsqueeze(0).detach().to(chain.device))
        #     pred_mesh = chain.get_trimesh_q(idx=0, boolean_merged=True)

        #     gt_q = batch["chain_q"][i, :valid_count]
        #     chain.update_status(gt_q.unsqueeze(0).detach().to(chain.device))
        #     link_mask = batch.get("link_mask", None)
        #     gt_mesh, masked_mesh = _split_gt_mesh_by_mask(
        #         chain,
        #         link_mask[i] if link_mask is not None else None,
        #     )

        #     visualize_meshes(
        #         pred_mesh=pred_mesh,
        #         gt_mesh=gt_mesh,
        #         masked_mesh=masked_mesh,
        #         port=port + shown,
        #         show_pred_only=show_pred_only,
        #     )
            
        #     logging.info(f"Current mask: {link_mask[i] if link_mask is not None else 'None'}")
        #     logging.info(f"Visualized class {cls} instance {inst} (sample {shown+1}/{max_mesh})")
        #     shown += 1

        # if shown >= max_mesh:
        #     break

if __name__ == "__main__":
    main()
