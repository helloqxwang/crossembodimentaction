"""Quick visualization for mesh + SDF samples (e.g., deepsdf.npz).

Usage example:
    python scripts/visualize_samples.py \
        --mesh dataset/meshes/000.obj \
        --samples dataset/samples/000/deepsdf.npz \
        --surface dataset/samples/000/surface.npy \
        --downsample 0.02
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _downsample(points: np.ndarray, ratio: float, max_points: int | None, rng: np.random.Generator):
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
    data = np.load(path)
    pos = data.get("pos")
    neg = data.get("neg")
    pos = _downsample(pos, ratio, max_points, rng) if pos is not None else None
    neg = _downsample(neg, ratio, max_points, rng) if neg is not None else None
    return pos, neg


def plot_mesh(ax, mesh: trimesh.Trimesh, color: str = "lightgray", alpha: float = 0.3):
    faces = mesh.faces
    verts = mesh.vertices
    tri_vertices = verts[faces]
    collection = Poly3DCollection(tri_vertices, facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_collection3d(collection)


def plot_points(ax, pts: np.ndarray, cmap, label: str, size: int):
    if pts is None or len(pts) == 0:
        return
    xyz = pts[:, :3]
    sdf = pts[:, 3] if pts.shape[1] > 3 else np.zeros(len(pts))
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=sdf, cmap=cmap, s=size, alpha=0.8, label=label)
    return sc


def load_surface(surface_path: str, ratio: float, max_points: int | None, rng: np.random.Generator):
    if surface_path is None:
        return None
    if not os.path.isfile(surface_path):
        raise FileNotFoundError(surface_path)
    pts = np.load(surface_path)
    pts = pts[:, :3]  # drop normals
    pts = _downsample(pts, ratio, max_points, rng)
    return pts


def main():
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
    pos_sc = plot_points(ax, pos, cmap="Blues", label="pos (sdf>=0)", size=args.marker_size)
    neg_sc = plot_points(ax, neg, cmap="Reds", label="neg (sdf<0)", size=args.marker_size)

    if surf_pts is not None and len(surf_pts) > 0:
        ax.scatter(surf_pts[:, 0], surf_pts[:, 1], surf_pts[:, 2], c="green", s=args.marker_size, alpha=0.5, label="surface")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=35)

    handles = []
    labels = []
    for sc in [pos_sc, neg_sc]:
        if sc is not None:
            handles.append(sc)
            labels.append(sc.get_label())
    if surf_pts is not None and len(surf_pts) > 0:
        handles.append(ax.scatter([], [], [], c="green", s=args.marker_size, alpha=0.5))
        labels.append("surface")
    if handles:
        ax.legend(handles, labels, loc="upper right")

    ax.set_title(os.path.basename(args.samples))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
