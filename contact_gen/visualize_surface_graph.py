from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize robot surface connectivity graph directly from robot models."
    )
    parser.add_argument(
        "--robot-names",
        type=str,
        nargs="*",
        default=["allegro", "barrett", "ezgripper", "robotiq_3finger", "shadowhand"],
        help="Robots to visualize.",
    )
    parser.add_argument(
        "--num-surface-points",
        type=int,
        default=512,
        help="Whole-hand point count used to create robot models.",
    )
    parser.add_argument("--vis-host", type=str, default="127.0.0.1")
    parser.add_argument("--vis-port", type=int, default=8092)
    parser.add_argument("--vis-point-size", type=float, default=0.002)
    parser.add_argument("--vis-hand-opacity", type=float, default=0.35)
    return parser


def _resolve_robot_model_name(robot_name: str, asset_names: set[str]) -> str:
    if robot_name in asset_names:
        return robot_name
    preferred = {
        "allegro": ["dro/allegro/allegro_hand_left_extended", "dro/allegro/allegro_hand_left"],
        "barrett": ["dro/barrett/model_extended", "dro/barrett/model"],
        "ezgripper": ["dro/ezgripper/ezgripper_extended", "dro/ezgripper/ezgripper"],
        "robotiq_3finger": [
            "dro/urdf/robotiq_3finger_description_extended",
            "dro/urdf/robotiq_3finger_description",
        ],
        "shadowhand": ["dro/shadowhand/shadow_hand_right_extended", "dro/shadowhand/shadow_hand_right"],
    }
    for c in preferred.get(robot_name, []):
        if c in asset_names:
            return c
    candidates = [k for k in asset_names if (f"/{robot_name}/" in k) or (robot_name in k)]
    if len(candidates) == 0:
        raise KeyError(f"Cannot resolve robot model for '{robot_name}'")
    candidates = sorted(
        candidates,
        key=lambda x: (int(x.startswith("dro/")), int("extended" in x), int("right" in x), -len(x)),
        reverse=True,
    )
    return candidates[0]


def _visualize_surface_graph(
    models: dict[str, Any],
    host: str,
    port: int,
    point_size: float,
    hand_opacity: float,
) -> None:
    import numpy as np
    import viser

    if len(models) == 0:
        raise RuntimeError("No models available for graph visualization.")

    cache: dict[str, dict[str, Any]] = {}
    for robot_name, model in models.items():
        q = model.get_canonical_q().detach().clone()
        if hasattr(model, "base_translation_indices") and len(model.base_translation_indices) > 0:
            q[model.base_translation_indices] = 0.0

        points, normals = model.get_surface_points_normals(q)
        points = points.detach().cpu()
        normals = normals.detach().cpu()

        simplified_mesh = getattr(model, "surface_union_mesh_canonical", None)
        if simplified_mesh is None:
            simplified_mesh = model.get_trimesh_q(q, mode="original")["visual"]
        original_mesh = model.get_trimesh_q(q, mode="original")["visual"]

        if len(simplified_mesh.vertices) > 0:
            center = torch.tensor(simplified_mesh.vertices, dtype=torch.float32).mean(dim=0, keepdim=True)
        elif len(original_mesh.vertices) > 0:
            center = torch.tensor(original_mesh.vertices, dtype=torch.float32).mean(dim=0, keepdim=True)
        else:
            center = points.mean(dim=0, keepdim=True)

        cache[robot_name] = {
            "points": points,
            "normals": normals,
            "neighbors": model.surface_graph_neighbors,
            "simplified_mesh": simplified_mesh,
            "original_mesh": original_mesh,
            "center": center,
        }

    robot_names = sorted(models.keys())
    max_points = max(int(cache[n]["points"].shape[0]) for n in robot_names)
    max_points = max(1, max_points)

    server = viser.ViserServer(host=host, port=int(port))
    robot_slider = server.gui.add_slider(
        "robot_index",
        min=0,
        max=max(0, len(robot_names) - 1),
        step=1,
        initial_value=0,
    )
    point_slider = server.gui.add_slider(
        "point_index",
        min=0,
        max=max_points - 1,
        step=1,
        initial_value=0,
    )
    vis_mode_slider = server.gui.add_slider(
        "vis_mode (0=graph,1=graph+outer,2=graph+outer+orig,3=outer,4=outer+orig)",
        min=0,
        max=4,
        step=1,
        initial_value=2,
    )
    normal_scale_slider = server.gui.add_slider(
        "normal_scale",
        min=0.0,
        max=0.03,
        step=0.0005,
        initial_value=0.0,
    )

    def _render() -> None:
        ridx = int(robot_slider.value)
        robot_name = robot_names[ridx]
        item = cache[robot_name]
        points = item["points"]
        normals = item["normals"]
        n = int(points.shape[0])
        if n == 0:
            return
        pidx = int(max(0, min(int(point_slider.value), n - 1)))

        graph = item["neighbors"]
        neighbors = graph[pidx] if pidx < len(graph) else []

        colors = np.zeros((n, 3), dtype=np.float32)
        colors[:, 0] = 0.25
        colors[:, 1] = 0.62
        colors[:, 2] = 1.00

        vis_mode = int(vis_mode_slider.value)
        show_graph = vis_mode in (0, 1, 2)
        show_simplified = vis_mode in (1, 2, 3, 4)
        show_original = vis_mode in (2, 4)

        simplified_mesh = item["simplified_mesh"]
        original_mesh = item["original_mesh"]
        center = item["center"]

        points_vis = points - center
        if len(neighbors) > 0:
            neigh_np = np.array(neighbors, dtype=np.int64)
            neigh_np = neigh_np[(neigh_np >= 0) & (neigh_np < n)]
            colors[neigh_np, :] = np.array([1.00, 0.85, 0.20], dtype=np.float32)
        colors[pidx, :] = np.array([1.00, 0.20, 0.20], dtype=np.float32)

        if show_graph:
            server.scene.add_point_cloud(
                name="surface_graph_points",
                points=points_vis.numpy().astype(np.float32),
                colors=colors,
                point_size=float(point_size),
                point_shape="circle",
            )
            if len(neighbors) > 0:
                neigh_idx = np.array(neighbors, dtype=np.int64)
                neigh_idx = neigh_idx[(neigh_idx >= 0) & (neigh_idx < n)]
                neigh_pts = points_vis[torch.as_tensor(neigh_idx, dtype=torch.long)]
                anchor = points_vis[pidx].unsqueeze(0).repeat(neigh_pts.shape[0], 1)
                seg = torch.stack([anchor, neigh_pts], dim=1).numpy().astype(np.float32)
                server.scene.add_line_segments(
                    name="surface_graph_edges",
                    points=seg,
                    colors=(1.0, 0.25, 0.25),
                    line_width=2.0,
                )
            else:
                server.scene.add_line_segments(
                    name="surface_graph_edges",
                    points=np.zeros((0, 2, 3), dtype=np.float32),
                    colors=(1.0, 0.25, 0.25),
                    line_width=2.0,
                )
        else:
            server.scene.add_point_cloud(
                name="surface_graph_points",
                points=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.float32),
                point_size=float(point_size),
                point_shape="circle",
            )
            server.scene.add_line_segments(
                name="surface_graph_edges",
                points=np.zeros((0, 2, 3), dtype=np.float32),
                colors=(1.0, 0.25, 0.25),
                line_width=2.0,
            )

        normal_scale = float(normal_scale_slider.value)
        if (normal_scale > 0.0) and show_graph:
            end = points_vis + normals * normal_scale
            seg = torch.stack([points_vis, end], dim=1).numpy().astype(np.float32)
            server.scene.add_line_segments(
                name="surface_normals",
                points=seg,
                colors=(0.30, 0.95, 0.95),
                line_width=1.5,
            )
            query_seg = torch.stack([points_vis[pidx], end[pidx]], dim=0).unsqueeze(0).numpy().astype(np.float32)
            server.scene.add_line_segments(
                name="query_normal",
                points=query_seg,
                colors=(1.0, 0.05, 0.95),
                line_width=3.0,
            )
        else:
            server.scene.add_line_segments(
                name="surface_normals",
                points=np.zeros((0, 2, 3), dtype=np.float32),
                colors=(0.0, 0.0, 0.0),
                line_width=0.0,
            )
            server.scene.add_line_segments(
                name="query_normal",
                points=np.zeros((0, 2, 3), dtype=np.float32),
                colors=(0.0, 0.0, 0.0),
                line_width=0.0,
            )

        if show_simplified:
            v = torch.tensor(simplified_mesh.vertices, dtype=torch.float32) - center
            f = torch.tensor(simplified_mesh.faces, dtype=torch.int32)
            outer_opacity = float(max(hand_opacity, 0.72))
            server.scene.add_mesh_simple(
                name="hand_mesh_simplified",
                vertices=v.numpy(),
                faces=f.numpy(),
                color=(0.20, 0.78, 0.35),
                opacity=outer_opacity,
            )
        else:
            server.scene.add_mesh_simple(
                name="hand_mesh_simplified",
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
                color=(0.0, 0.0, 0.0),
                opacity=0.0,
            )

        if show_original:
            v = torch.tensor(original_mesh.vertices, dtype=torch.float32) - center
            f = torch.tensor(original_mesh.faces, dtype=torch.int32)
            original_opacity = float(max(0.45, 0.8 * hand_opacity))
            server.scene.add_mesh_simple(
                name="hand_mesh_original",
                vertices=v.numpy(),
                faces=f.numpy(),
                color=(0.72, 0.72, 0.72),
                opacity=original_opacity,
            )
        else:
            server.scene.add_mesh_simple(
                name="hand_mesh_original",
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
                color=(0.0, 0.0, 0.0),
                opacity=0.0,
            )

        print(
            f"[graph-vis] robot={robot_name} point={pidx} degree={len(neighbors)} vis_mode={vis_mode} "
            f"normal_scale={normal_scale:.4f} num_points={n}"
        )

    @robot_slider.on_update
    def _(_: Any) -> None:
        _render()

    @point_slider.on_update
    def _(_: Any) -> None:
        _render()

    @vis_mode_slider.on_update
    def _(_: Any) -> None:
        _render()

    @normal_scale_slider.on_update
    def _(_: Any) -> None:
        _render()

    _render()
    print(f"Graph viewer URL: http://{host}:{port}")
    print("Use sliders: robot_index and point_index.")
    while True:
        time.sleep(1.0)


def main() -> None:
    args = _build_parser().parse_args()

    sys.path.append(PROJECT_ROOT)
    from robot_model.robot_model import create_robot_model, discover_robot_assets  # type: ignore

    assets_meta = discover_robot_assets()
    asset_names = set(assets_meta.keys())
    vis_models: dict[str, Any] = {}
    num_surface_points = int(max(1, args.num_surface_points))

    for robot_name in args.robot_names:
        resolved_robot_name = _resolve_robot_model_name(robot_name, asset_names)

        model = create_robot_model(
            robot_name=resolved_robot_name,
            device=torch.device("cpu"),
            num_points=int(num_surface_points),
        )
        vis_models[robot_name] = model
        print(
            f"[{robot_name}] model={resolved_robot_name} points={num_surface_points} "
            f"degree_mean={sum(len(x) for x in model.surface_graph_neighbors)/max(1,len(model.surface_graph_neighbors)):.2f}"
        )

    if len(vis_models) == 0:
        raise RuntimeError("No robot models available for visualization.")

    _visualize_surface_graph(
        models=vis_models,
        host=str(args.vis_host),
        port=int(args.vis_port),
        point_size=float(args.vis_point_size),
        hand_opacity=float(args.vis_hand_opacity),
    )


if __name__ == "__main__":
    main()
