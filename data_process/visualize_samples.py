"""Visualization for mesh + SDF samples and predictions.

Provides reusable functions for 3D visualization of meshes and SDF values.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
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


def visualize_sdf_points_plotly(
    *,
    mesh: trimesh.Trimesh | None = None,
    points: np.ndarray | None = None,
    sdf: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    gt_sdf: np.ndarray | None = None,
    title: str = "SDF Visualization",
    downsample_ratio: float = 0.1,
    max_points: int = 50000,
    marker_size: int = 3,
    show_sign: str = "all",
    normal_stride: int = 10,
    normal_length: float = 0.02,
    seed: int = 0,
    save_path: str | None = None,
    save_html_path: str | None = None,
    show: bool = False,
) -> go.Figure:
    """Plot mesh + points with SDF colors and optional normals using Plotly.

    If gt_sdf is provided, point colors represent abs(pred - gt) with a color bar.
    Otherwise, points are colored by signed distance (blue negative, red positive).
    show_sign controls whether to plot all points or only negative/zero/positive.
    """

    rng = np.random.default_rng(seed)
    fig = go.Figure()

    if mesh is not None:
        verts = mesh.vertices
        faces = mesh.faces
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="lightgray",
                opacity=0.35,
                name="mesh",
            )
        )

    if points is not None:
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3); got shape {points.shape}")

        keep_count = int(len(points) * downsample_ratio)
        if max_points is not None:
            keep_count = min(keep_count, max_points)
        keep_count = max(1, keep_count) if len(points) > 0 else 0
        if keep_count == 0:
            pts = None
            idx = None
        elif keep_count >= len(points):
            idx = np.arange(len(points))
            pts = points
        else:
            idx = rng.choice(len(points), size=keep_count, replace=False)
            pts = points[idx]

        if pts is not None and len(pts) > 0:
            
            if sdf is not None:
                sdf_vals = np.asarray(sdf).reshape(-1)[idx]
            else:
                sdf_vals = None

            gt_vals_full = None
            if gt_sdf is not None:
                gt_vals_full = np.asarray(gt_sdf).reshape(-1)[idx]
            normals_full = None
            if normals is not None:
                normals_full = np.asarray(normals)[idx]

            sign_mask = None
            if sdf_vals is not None:
                show_sign = str(show_sign).lower()
                if show_sign == "negative":
                    sign_mask = sdf_vals < 0
                elif show_sign == "zero":
                    sign_mask = sdf_vals == 0
                elif show_sign == "positive":
                    sign_mask = sdf_vals > 0
                elif show_sign == "all":
                    sign_mask = None
                else:
                    raise ValueError(f"Unsupported show_sign: {show_sign}")
                if sign_mask is not None:
                    pts = pts[sign_mask]
                    sdf_vals = sdf_vals[sign_mask]
                    if gt_vals_full is not None:
                        gt_vals_full = gt_vals_full[sign_mask]
                    if normals_full is not None:
                        normals_full = normals_full[sign_mask]

            cmid = None
            valid_mask = None
            color = "rgb(80,80,80)"
            cmin = cmax = None
            colorbar = None
            colorscale = None
            if gt_sdf is not None and sdf_vals is not None:
                gt_vals = gt_vals_full
                valid_mask = gt_vals != -1
                err = np.abs(sdf_vals - gt_vals)
                err_valid = err[valid_mask] if np.any(valid_mask) else err
                color = err_valid
                cmax = float(np.percentile(err_valid, 95)) if np.any(err_valid > 0) else 1.0
                cmin = 0.0
                colorbar = dict(title="|pred - gt|")
                colorscale = "Viridis"
            elif sdf_vals is not None:
                valid_mask = sdf_vals != -1
                sdf_valid = sdf_vals[valid_mask] if np.any(valid_mask) else sdf_vals
                abs_sdf = np.abs(sdf_valid)
                max_abs = float(np.percentile(abs_sdf, 95)) if np.any(abs_sdf > 0) else 1.0
                color = sdf_valid
                cmin, cmax = -max_abs, max_abs
                cmid = 0.0
                colorbar = dict(title="sdf")
                colorscale = [
                    [0.0, "rgb(0,0,180)"],
                    [0.5, "rgb(230,230,230)"],
                    [1.0, "rgb(180,0,0)"],
                ]

            if valid_mask is None:
                valid_mask = np.ones(len(pts), dtype=bool)
            pts_valid = pts[valid_mask]
            pts_unknown = pts[~valid_mask]

            if len(pts_valid) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=pts_valid[:, 0],
                        y=pts_valid[:, 1],
                        z=pts_valid[:, 2],
                        mode="markers",
                        marker=dict(
                            size=marker_size,
                            color=color,
                            colorscale=colorscale,
                            cmin=cmin,
                            cmax=cmax,
                            cmid=cmid,
                            colorbar=colorbar,
                            opacity=0.9,
                        ),
                        name="samples",
                    )
                )
            if len(pts_unknown) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=pts_unknown[:, 0],
                        y=pts_unknown[:, 1],
                        z=pts_unknown[:, 2],
                        mode="markers",
                        marker=dict(
                            size=max(1, int(marker_size * 0.8)),
                            color="rgb(120,120,120)",
                            opacity=0.5,
                        ),
                        name="sdf=-1 (unknown)",
                    )
                )

            if normals_full is not None:
                normals = normals_full
                norm_pts = pts
                if gt_sdf is not None:
                    gt_vals = gt_vals_full
                    surface_mask = gt_vals != -1
                elif sdf_vals is not None:
                    surface_mask = sdf_vals != -1
                else:
                    surface_mask = np.ones(len(norm_pts), dtype=bool)
                normals = normals[surface_mask]
                norm_pts = norm_pts[surface_mask]
                if len(norm_pts) > 0:
                    step = max(1, int(normal_stride))
                    line_pts = []
                    for i in range(0, len(norm_pts), step):
                        p = norm_pts[i]
                        n = normals[i]
                        line_pts.append(p)
                        line_pts.append(p + normal_length * n)
                        line_pts.append([np.nan, np.nan, np.nan])
                    line_pts = np.array(line_pts)
                    fig.add_trace(
                        go.Scatter3d(
                            x=line_pts[:, 0],
                            y=line_pts[:, 1],
                            z=line_pts[:, 2],
                            mode="lines",
                            line=dict(color="black", width=2),
                            name="normals",
                        )
                    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
    )

    if save_path:
        try:
            fig.write_image(save_path)
        except Exception as exc:
            if save_html_path is None:
                save_html_path = save_path + ".html"
            fig.write_html(save_html_path)
            print(f"[visualize_sdf_points_plotly] write_image failed ({exc}); wrote HTML to {save_html_path}")

    if save_html_path and (save_path is None or save_html_path != save_path):
        fig.write_html(save_html_path)

    if show:
        fig.show()

    return fig


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
