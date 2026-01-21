"""Extract and store SDF latents for all samples in a dataset.

Loads a checkpoint trained with train_fk.py in SIREN mode, runs inference over
the dataset defined by a Hydra config, and saves per-sample latent vectors.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data_process.dataset import get_dataloader
from train_fk import build_models, inference, _prepare_device


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
        "--device",
        default=None,
        help="Override device (cpu/cuda). If unset, use cfg.training.device",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override dataloader workers",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size; defaults to cfg.data.batch_size",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = OmegaConf.load(args.config)
    device_str = args.device if args.device is not None else cfg.training.device
    device = _prepare_device(str(device_str))

    indices = list(range(cfg.data.indices.start, cfg.data.indices.end))
    data_source = to_absolute_path(cfg.data.data_source)

    batch_size = args.batch_size or cfg.data.batch_size
    num_workers = args.num_workers if args.num_workers is not None else cfg.data.num_workers

    loader = get_dataloader(
        data_source=data_source,
        indices=indices,
        num_instances=cfg.data.num_instances,
        subsample=cfg.data.subsample,
        batch_size=batch_size,
        max_num_links=cfg.data.max_num_links,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        sdf_mode=getattr(cfg.data, "sdf_mode", "siren"),
        off_surface_center=getattr(cfg.data, "off_surface_center", "zero"),
        siren_sdf_mode=getattr(cfg.data, "siren_sdf_mode", "true"),
        tsdf_band=getattr(cfg.data, "tsdf_band", 0.1),
        normalize_mode=getattr(cfg.data, "normalize_mode", "none"),
        normalize_center_mode=getattr(cfg.data, "normalize_center_mode", "zero"),
    )

    models, epoch = _load_models(cfg, device, args.checkpoint)
    if epoch is not None:
        print(f"Loaded checkpoint at epoch {epoch}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting latents"):
            latent, _ = inference(batch, models, cfg=cfg, device=device)
            latent = latent.cpu()
            for i, (cls, inst) in enumerate(zip(batch["class_idx"], batch["instance_idx"])):
                out_path = save_dir / f"cls{cls}_inst{inst}.pt"
                torch.save(latent[i], out_path)

    print(f"Saved latents to {save_dir}")


if __name__ == "__main__":
    main()
 