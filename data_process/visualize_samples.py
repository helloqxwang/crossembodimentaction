"""Visualization for mesh + SDF samples and predictions.

Provides reusable functions for 3D visualization of meshes and SDF values.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def downsample(points: np.ndarray, ratio: float, max_points: int | None, rng: np.random.Generator) -> np.ndarray:
    """Return a random subset of points according to ratio/max_points."""
    if points is None or len(points) == 0:
        return points
    keep = int(len(points) * ratio)
    if max_points is not None:
        keep = min(keep, max_points)
    keep = max(1, keep) if len(points) > 0 else 0
    if keep >= len(points):
        return points
    idx = rng.choice(len(points), size=keep, replace=False)
    return points[idx]


def load_samples(path: str, ratio: float, max_points: int | None, rng: np.random.Generator):
    """Load and downsample pos/neg samples from npz file."""
    data = np.load(path)
    pos = data.get("pos")
    neg = data.get("neg")
    pos = downsample(pos, ratio, max_points, rng) if pos is not None else None
    neg = downsample(neg, ratio, max_points, rng) if neg is not None else None
    return pos, neg


def load_surface(surface_path: str, ratio: float, max_points: int | None, rng: np.random.Generator) -> np.ndarray | None:
    """Load and downsample surface points from npy file."""
    if surface_path is None:
        return None
    if not os.path.isfile(surface_path):
        raise FileNotFoundError(surface_path)
    pts = np.load(surface_path)
    pts = pts[:, :3]  # drop normals
    pts = downsample(pts, ratio, max_points, rng)
    return pts


def plot_mesh(ax, mesh: trimesh.Trimesh, color: str = "lightgray", alpha: float = 0.3) -> None:
    """Add mesh to 3D axes."""
    faces = mesh.faces
    verts = mesh.vertices
    tri_vertices = verts[faces]
    collection = Poly3DCollection(tri_vertices, facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_collection3d(collection)


def visualize_sdf(
    mesh: trimesh.Trimesh,
    sdf_samples: np.ndarray,
    title: str = "SDF Visualization",
    downsample_ratio: float = 0.1,
    max_points: int = 50000,
    marker_size: int = 3,
    seed: int = 0,
) -> None:
    """Visualize mesh with SDF samples.
    
    Args:
        mesh: trimesh object.
        sdf_samples: SDF points, shape (N, 4) with [x, y, z, sdf].
                     Positive SDF → blue, Negative SDF → red.
                     Color intensity by absolute SDF value.
        title: Plot title.
        downsample_ratio: Fraction of points to visualize [0, 1].
        max_points: Cap on visualized points after downsampling.
        marker_size: Scatter marker size.
        seed: Random seed for subsampling.
    """
    rng = np.random.default_rng(seed)
    sdf_samples = downsample(sdf_samples, downsample_ratio, max_points, rng)
    
    if sdf_samples is None or len(sdf_samples) == 0:
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    plot_mesh(ax, mesh)

    # Split into positive and negative SDF
    xyz = sdf_samples[:, :3]
    sdf = sdf_samples[:, 3]
    
    pos_mask = sdf >= 0
    neg_mask = sdf < 0
    
    # Plot positive SDF with blue (deeper = larger abs value)
    if pos_mask.sum() > 0:
        ax.scatter(
            xyz[pos_mask, 0], xyz[pos_mask, 1], xyz[pos_mask, 2],
            c=sdf[pos_mask], cmap="Blues", s=marker_size, alpha=0.8, label="sdf >= 0"
        )
    
    # Plot negative SDF with red (deeper = larger abs value)
    if neg_mask.sum() > 0:
        ax.scatter(
            xyz[neg_mask, 0], xyz[neg_mask, 1], xyz[neg_mask, 2],
            c=-sdf[neg_mask], cmap="Reds", s=marker_size, alpha=0.8, label="sdf < 0"
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=35)
    ax.legend(loc="upper right")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def main():
    """CLI: visualize SDF samples from DeepSDF npz files."""
    parser = argparse.ArgumentParser(description="Visualize mesh with sampled SDF points.")
    parser.add_argument("--mesh", required=True, help="Path to the mesh .obj")
    parser.add_argument("--samples", required=True, help="Path to the npz file (e.g., deepsdf.npz)")
    parser.add_argument("--surface", default=None, help="Optional surface.npy to overlay")
    parser.add_argument("--downsample", type=float, default=0.02, help="Fraction of samples to keep (0,1]")
    parser.add_argument("--max-points", type=int, default=50000, help="Upper cap on plotted points after downsampling")
    parser.add_argument("--marker-size", type=int, default=3, help="Scatter marker size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling")
    args = parser.parse_args()

    if not (0 < args.downsample <= 1):
        raise ValueError("--downsample must be in (0,1]")

    rng = np.random.default_rng(args.seed)

    mesh = trimesh.load(args.mesh, force='mesh')
    pos, neg = load_samples(args.samples, args.downsample, args.max_points, rng)
    surf_pts = load_surface(args.surface, args.downsample, args.max_points, rng) if args.surface else None

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    plot_mesh(ax, mesh)

    # Combine pos/neg samples for visualization
    samples = []
    if pos is not None:
        samples.append(pos)
    if neg is not None:
        samples.append(neg)
    if samples:
        combined = np.vstack(samples)
        xyz = combined[:, :3]
        sdf = combined[:, 3]
        
        pos_mask = sdf >= 0
        if pos_mask.sum() > 0:
            ax.scatter(xyz[pos_mask, 0], xyz[pos_mask, 1], xyz[pos_mask, 2],
                      c=sdf[pos_mask], cmap="Blues", s=args.marker_size, alpha=0.8, label="sdf >= 0")
        
        neg_mask = sdf < 0
        if neg_mask.sum() > 0:
            ax.scatter(xyz[neg_mask, 0], xyz[neg_mask, 1], xyz[neg_mask, 2],
                      c=-sdf[neg_mask], cmap="Reds", s=args.marker_size, alpha=0.8, label="sdf < 0")

    if surf_pts is not None and len(surf_pts) > 0:
        ax.scatter(surf_pts[:, 0], surf_pts[:, 1], surf_pts[:, 2], 
                  c="green", s=args.marker_size, alpha=0.5, label="surface")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=35)
    ax.legend(loc="upper right")
    ax.set_title(os.path.basename(args.samples))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
