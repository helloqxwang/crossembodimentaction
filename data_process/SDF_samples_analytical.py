"""
Generate DeepSDF-style samples directly from analytic SDFs of kinematic chains.

This bypasses mesh export by using `ChainModel` to sample configurations and
query signed distances analytically. Outputs mirror the format of
`data_generate_samples.py`: each chain gets a folder containing `deepsdf.npz`
with `pos` and `neg` arrays (Nx4, xyz + sdf) and `scale_to` (set to 1.0).
"""

from __future__ import annotations

import argparse
import os
from math import ceil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# Make sure we can import ChainModel.
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / ""))
from robot_model.chain_model import ChainModel  # noqa: E402
from data_process.SDF_samples import visualize_sdf
from tqdm import tqdm


def _chain_sort_key(name: str) -> Tuple[int, int]:
    """Sort names like `chain_{class}_{idx}` consistently."""
    parts = name.split("_")
    try:
        return (int(parts[1]))
    except Exception:
        return (int(1e9), int(1e9))


def sample_chain_deepsdf(
    urdf_path: Path,
    *,
    n_configs: int,
    n_samples: int,
    near_surface_ratio: float,
    var: float,
    device: str,
    batch_size: int = 8,
    samples_per_link: int = 256,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Sample configurations and generate pos/neg SDF points per configuration.

    Returns a list of tuples (pos, neg, q_single) for each configuration.
    """
    torch_device = torch.device(device)
    model = ChainModel(
        urdf_path,
        device=torch_device,
        samples_per_link=samples_per_link,
    )

    per_config: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    remaining = n_configs
    while remaining > 0:
        cur_batch = min(batch_size, remaining)
        q = model.sample_q(B=cur_batch)
        model.update_status(q)
        pts = model.sample_query_points(n=n_samples, var=var, near_surface_ratio=near_surface_ratio)
        sdf = model.query_sdf(pts)

        for b in range(cur_batch):
            samples = torch.cat([pts[b], sdf[b].unsqueeze(-1)], dim=-1).cpu().numpy().astype(np.float32)
            mask_pos = samples[:, 3] >= 0
            pos = samples[mask_pos]
            neg = samples[~mask_pos]
            per_config.append((pos, neg, q[b].cpu().numpy()))

        remaining -= cur_batch

    return per_config


def generate_all(
    urdf_dir: Path,
    dest_dir: Path,
    *,
    n_configs: int,
    n_samples: int,
    near_surface_ratio: float,
    var: float,
    device: str,
    batch_size: int,
    max_chains: int,
    samples_per_link: int,
) -> None:
    urdf_files = sorted([p for p in urdf_dir.glob("*.urdf")], key=lambda p: _chain_sort_key(p.stem))
    if max_chains > 0:
        urdf_files = urdf_files[:max_chains]
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(urdf_files)} URDFs in {urdf_dir}")
    for cls_idx, urdf_path in enumerate(urdf_files):
        name = urdf_path.stem
        out_root = dest_dir / name
        out_root.mkdir(parents=True, exist_ok=True)

        print(f"[{cls_idx+1}/{len(urdf_files)}] {name}: sampling {n_configs} configs × {n_samples} pts")
        configs = sample_chain_deepsdf(
            urdf_path,
            n_configs=n_configs,
            n_samples=n_samples,
            near_surface_ratio=near_surface_ratio,
            var=var,
            device=device,
            batch_size=batch_size,
            samples_per_link=samples_per_link,
        )

        for ins_idx, (pos, neg, q_single) in enumerate(configs):
            cfg_dir = out_root / f"config_{ins_idx}"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            out_npz = cfg_dir / "deepsdf.npz"

            if out_npz.exists():
                print(f"    config {ins_idx}: exists, skipping")
                continue

            ### For visualization/debugging purposes only:
            # chain_model = ChainModel(
            #     urdf_path,
            #     device="cpu",
            #     samples_per_link=samples_per_link,
            # )
            # chain_model.update_status(torch.from_numpy(q_single).unsqueeze(0))
            # mesh = chain_model.get_trimesh_q(idx=0, boolean_merged=True)
            # sdf_samples = np.concatenate([pos, neg], axis=0)
            # visualize_sdf(mesh=mesh, sdf_samples=sdf_samples, downsample_ratio=0.001, title=f"DeepSDF: {ins_idx}")
            
            np.savez(
                out_npz,
                pos=pos,
                neg=neg,
                scale_to=1.0,
                q=q_single,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate analytic DeepSDF samples from URDF chains.")
    parser.add_argument("datadir", type=str, help="dataset root (expects URDFs under datadir/out_chains)")
    parser.add_argument("--urdf-dir", type=str, default=None, help="override URDF directory; defaults to datadir/out_chains")
    parser.add_argument("--dest", type=str, default=None, help="output dir; defaults to datadir/chain_samples_normalized")
    parser.add_argument("-n", "--n-samples", default=250000, type=int, help="query points per configuration")
    parser.add_argument("--n-configs", default=32, type=int, help="number of random configurations to sample per chain")
    parser.add_argument("--near-surface-ratio", default=0.95, type=float, help="fraction of near-surface samples")
    parser.add_argument("--var", default=0.0025, type=float, help="variance for near-surface noise")
    parser.add_argument("--batch-size", default=8, type=int, help="configurations per forward pass")
    parser.add_argument("--samples-per-link", default=256, type=int, help="points per link to seed near-surface sampling")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="compute device")
    parser.add_argument("--max-chains", default=-1, type=int, help="limit number of chains; -1 for all")
    args = parser.parse_args()

    urdf_dir = Path(args.urdf_dir) if args.urdf_dir else Path(args.datadir) / "out_chains"
    dest_dir = Path(args.dest) if args.dest else Path(args.datadir) / "chain_samples_normalized"

    generate_all(
        urdf_dir,
        dest_dir,
        n_configs=args.n_configs,
        n_samples=args.n_samples,
        near_surface_ratio=args.near_surface_ratio,
        var=args.var,
        device=args.device,
        batch_size=args.batch_size,
        max_chains=args.max_chains,
        samples_per_link=args.samples_per_link,
    )


if __name__ == "__main__":
    main()
