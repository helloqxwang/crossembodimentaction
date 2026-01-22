"""Extract and store SDF latents for all samples in a dataset.

Loads a checkpoint trained with train_fk.py in SIREN mode, runs inference over
the dataset defined by a Hydra config, and saves per-sample latent vectors
along with link masks, joint values, and class indices.
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from robot_model.chain_model import ChainModel
from train_fk import build_models, inference, pooled_latent, _prepare_device
from tqdm import tqdm


def _load_models(
    cfg: DictConfig, device: torch.device, checkpoint: str
) -> Tuple[Dict[str, torch.nn.Module], Optional[int]]:
    models = build_models(cfg, device)
    ckpt_path = to_absolute_path(checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    # Allow OmegaConf objects inside checkpoints saved from train_fk.
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SDF latents for an entire dataset")
    parser.add_argument(
        "--config",
        default="conf/conf_siren/config_siren.yaml",
        help="Path to Hydra config used for data/model params",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/mfk_siren/epoch_830.pth",
        help="Path to checkpoint from train_fk.py",
    )
    parser.add_argument(
        "--save_dir",
        default="./data/sdf_tokens",
        help="Directory to store latent .pt files",
    )
    parser.add_argument(
        "--instance_nums",
        type=int,
        default=1000,
        help="Override samples per chain (num_instances)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (cpu/cuda). If unset, use cfg.training.device",
    )
    return parser.parse_args()


def _enumerate_link_masks(num_links: int, *, min_visible: int = 1) -> torch.Tensor:
    """Enumerate all link masks with at least ``min_visible`` links kept."""

    if num_links <= 0:
        return torch.zeros((1, 0), dtype=torch.bool)
    if num_links < min_visible:
        return torch.ones((1, num_links), dtype=torch.bool)

    max_mask = num_links - min_visible
    masks = []
    link_indices = list(range(num_links))
    for k in range(max_mask + 1):
        for masked_indices in itertools.combinations(link_indices, k):
            mask = torch.ones(num_links, dtype=torch.bool)
            if masked_indices:
                mask[list(masked_indices)] = False
            masks.append(mask)
    return torch.stack(masks, dim=0)


def _pad_tensor(x: torch.Tensor, *, length: int) -> torch.Tensor:
    if x.size(0) > length:
        raise ValueError(f"Cannot pad length {x.size(0)} to smaller length {length}")
    if x.size(0) == length:
        return x
    pad = torch.zeros((length - x.size(0), *x.shape[1:]), dtype=x.dtype)
    return torch.cat([x, pad], dim=0)


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


def main() -> None:
    args = _parse_args()

    cfg = OmegaConf.load(args.config)
    device_str = args.device if args.device is not None else cfg.training.device
    device = _prepare_device(str(device_str))

    # indices = list(range(cfg.data.indices.start, cfg.data.indices.end))
    indices = list(range(cfg.data.indices.start, 110))
    data_source = to_absolute_path(cfg.data.data_source)
    num_instances = args.instance_nums or cfg.data.num_instances
    max_num_links = int(cfg.data.max_num_links)

    models, epoch = _load_models(cfg, device, args.checkpoint)
    if epoch is not None:
        print(f"Loaded checkpoint at epoch {epoch}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for class_real_idx in tqdm(indices, desc="Classes"):
            chain_props = np.load(
                os.path.join(data_source, "out_chains_v2", f"chain_{class_real_idx}_properties.npz"),
                allow_pickle=True,
            )
            link_features = torch.tensor(chain_props["links_property"]).float()
            joint_features = torch.tensor(chain_props["joints_property"]).float()
            link_bps = _prepare_link_bps(chain_props["bpses"])

            num_links = link_features.size(0)
            if num_links > max_num_links:
                raise ValueError(
                    f"class {class_real_idx} has {num_links} links > max_num_links {max_num_links}"
                )

            urdf_path = Path(os.path.join(data_source, "out_chains_v2", f"chain_{class_real_idx}.urdf"))
            chain_model = ChainModel(urdf_path=urdf_path, samples_per_link=128, device="cpu")
            q_samples = chain_model.sample_q(num_instances).float().cpu()

            link_features = _pad_tensor(link_features, length=max_num_links)
            link_bps = _pad_tensor(link_bps, length=max_num_links)
            joint_features = _pad_tensor(joint_features, length=max_num_links - 1)

            chain_q = torch.zeros(num_instances, max_num_links - 1)
            chain_q[:, : q_samples.size(1)] = q_samples
            chain_q = chain_q.to(device)
            token_mask = torch.zeros(3 * max_num_links - 2, dtype=torch.bool, device=device)
            token_mask[: 3 * num_links - 2] = True
            link_mask_full = torch.zeros(max_num_links, dtype=torch.bool, device=device)
            link_mask_full[:num_links] = True

            batch = {
                "sdf_samples": torch.zeros(num_instances, 1, 4, device=device),
                "chain_q": chain_q,
                "link_features": link_features.unsqueeze(0).expand(num_instances, -1, -1).to(device),
                "joint_features": joint_features.unsqueeze(0).expand(num_instances, -1, -1).to(device),
                "link_bps_scdistances": link_bps.unsqueeze(0).expand(num_instances, -1, -1).to(device),
                "mask": token_mask.unsqueeze(0).expand(num_instances, -1),
                "link_mask": link_mask_full.unsqueeze(0).expand(num_instances, -1),
            }

            _, _, link_tokens = inference(
                batch,
                models,
                cfg=cfg,
                device=device,
                coords_override=torch.zeros(num_instances, 1, 3, device=device),
                return_link_tokens=True,
            )

            masks = _enumerate_link_masks(num_links, min_visible=1)
            masks_padded = torch.zeros(masks.size(0), max_num_links, dtype=torch.bool)
            masks_padded[:, :num_links] = masks
            masks_padded = masks_padded.to(device)

            latents_by_mask = []
            for mask_idx in range(masks_padded.size(0)):
                mask_batch = masks_padded[mask_idx].unsqueeze(0).expand(num_instances, -1)
                latent = pooled_latent(link_tokens, mask_batch, mode="max")
                latents_by_mask.append(latent.cpu())

            latents_by_mask = torch.stack(latents_by_mask, dim=0).permute(1, 0, 2)
            masks_cpu = masks.cpu()

            for inst_idx in range(num_instances):
                out_path = save_dir / f"cls{class_real_idx}_inst{inst_idx}.pt"
                torch.save(
                    {
                        "sdf_tokens": latents_by_mask[inst_idx],
                        "link_mask": masks_cpu,
                        "q_values": q_samples[inst_idx],
                        "cls_idx": class_real_idx,
                        "instance_idx": inst_idx,
                    },
                    out_path,
                )

    print(f"Saved latents to {save_dir}")


if __name__ == "__main__":
    main()
 
