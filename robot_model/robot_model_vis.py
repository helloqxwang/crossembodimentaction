from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import viser

ROOT_DIR = Path(__file__).resolve().parents[1]

try:
    from robot_model.robot_model import create_robot_model, discover_robot_assets
except Exception:
    from robot_model import create_robot_model, discover_robot_assets


def _empty_mesh(server: viser.ViserServer, name: str) -> None:
    server.scene.add_mesh_simple(
        name=name,
        vertices=np.zeros((0, 3), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.int32),
        color=(0.7, 0.8, 1.0),
        opacity=0.0,
    )


def _empty_pc(server: viser.ViserServer, name: str, point_size: float = 0.004) -> None:
    server.scene.add_point_cloud(
        name=name,
        points=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.float32),
        point_size=float(point_size),
        point_shape="circle",
    )


def _empty_lines(server: viser.ViserServer, name: str) -> None:
    server.scene.add_line_segments(
        name=name,
        points=np.zeros((0, 2, 3), dtype=np.float32),
        colors=(1.0, 1.0, 1.0),
        line_width=1.0,
    )


def _uniform_colors(n: int, rgb: tuple[float, float, float]) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    colors = np.zeros((n, 3), dtype=np.float32)
    colors[:, 0] = rgb[0]
    colors[:, 1] = rgb[1]
    colors[:, 2] = rgb[2]
    return colors


def _is_humanoid_name(robot_name: str) -> bool:
    n = robot_name.lower()
    keywords = ["unitree_g1", "unitree_h1", "h1_2", "booster_t1", "fourier_n1", "humanoid"]
    return any(k in n for k in keywords)


def _is_humanoid_model(robot_name: str, model) -> bool:
    if _is_humanoid_name(robot_name):
        return True
    jnames = " ".join(j.lower() for j in model.get_joint_orders())
    has_leg_pattern = ("hip" in jnames) and ("knee" in jnames) and ("ankle" in jnames)
    return bool(has_leg_pattern and model.dof >= 20)


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


def _run_surface_graph_view(
    models: dict[str, Any],
    host: str,
    port: int,
    point_size: float,
    hand_opacity: float,
) -> None:
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
            _empty_mesh(server, "hand_mesh_simplified")

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
            _empty_mesh(server, "hand_mesh_original")

    @robot_slider.on_update
    def _(_: viser.GuiEvent) -> None:
        _render()

    @point_slider.on_update
    def _(_: viser.GuiEvent) -> None:
        _render()

    @vis_mode_slider.on_update
    def _(_: viser.GuiEvent) -> None:
        _render()

    @normal_scale_slider.on_update
    def _(_: viser.GuiEvent) -> None:
        _render()

    _render()
    print(f"Graph viewer URL: http://{host}:{port}")
    while True:
        time.sleep(1.0)

def _load_robot_subset_from_list(
    all_names: list[str],
    list_file: Path,
    list_key: str,
) -> list[str]:
    if list_key == "all":
        return all_names

    if not list_file.exists():
        raise FileNotFoundError(f"Robot list file not found: {list_file}")

    with open(list_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if list_key not in payload:
        raise KeyError(f"Key '{list_key}' not in {list_file}. Available: {list(payload.keys())}")

    names = payload[list_key]
    if not isinstance(names, list):
        raise ValueError(f"Expected list at key '{list_key}' in {list_file}")

    names = [n for n in names if n in all_names]
    return sorted(names)


def main_surface_graph_cli(argv: list[str] | None = None) -> None:
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
    parser.add_argument("--assets-dir", type=str, default=str(ROOT_DIR / "assets"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    assets_dir = Path(args.assets_dir)
    assets_meta = discover_robot_assets(assets_dir)
    asset_names = set(assets_meta.keys())

    vis_models: dict[str, Any] = {}
    num_surface_points = int(max(1, args.num_surface_points))
    for robot_name in args.robot_names:
        resolved_robot_name = _resolve_robot_model_name(robot_name, asset_names)
        model = create_robot_model(
            robot_name=resolved_robot_name,
            device=torch.device(args.device),
            num_points=int(num_surface_points),
            assets_dir=assets_dir,
        )
        vis_models[robot_name] = model
        print(
            f"[{robot_name}] model={resolved_robot_name} points={num_surface_points} "
            f"degree_mean={sum(len(x) for x in model.surface_graph_neighbors)/max(1,len(model.surface_graph_neighbors)):.2f}"
        )
    _run_surface_graph_view(
        models=vis_models,
        host=str(args.vis_host),
        port=int(args.vis_port),
        point_size=float(args.vis_point_size),
        hand_opacity=float(args.vis_hand_opacity),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize URDF/XML robots with RobotModel + viser")
    parser.add_argument("--assets-dir", type=str, default=str(ROOT_DIR / "assets"))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9402)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--vis-point-size", type=float, default=0.002)
    parser.add_argument("--only-manipulators", action="store_true")
    parser.add_argument("--robot-list-file", type=str, default=str(ROOT_DIR / "assets" / "manipulator_robot_lists.json"))
    parser.add_argument("--robot-list-key", type=str, default="right_hands", choices=["all", "both_hands", "right_hands"])
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir)
    device = torch.device(args.device)

    all_assets = discover_robot_assets(assets_dir)
    if not all_assets:
        raise RuntimeError(f"No robot assets found in: {assets_dir}")

    candidate_names = sorted(all_assets.keys())
    candidate_names = _load_robot_subset_from_list(
        candidate_names,
        list_file=Path(args.robot_list_file),
        list_key=args.robot_list_key,
    )

    if args.only_manipulators:
        candidate_names = [n for n in candidate_names if not _is_humanoid_name(n)]

    models = {}
    failed = {}

    for name in candidate_names:
        try:
            model = create_robot_model(name, device=device, num_points=args.num_points, assets_dir=assets_dir)
            if args.only_manipulators and _is_humanoid_model(name, model):
                continue
            models[name] = model
            n_pts = int(model.surface_template_points_local.shape[0])
            n_links = int(len(model.meshes_original))
            print(
                f"[loaded] {name} (model: {model.robot_name}, points: {n_pts}, "
                f"links: {n_links}, dof: {model.dof})"
            )
        except Exception as exc:
            failed[name] = str(exc)
            print(f"[failed] {name}: {exc}")

    if not models:
        raise RuntimeError(f"No model available. First errors: {list(failed.items())[:3]}")

    robot_names = sorted(models.keys())

    print(f"Loaded robots: {len(models)}")
    print(f"Robot list key: {args.robot_list_key}")
    if args.only_manipulators:
        print("Filter: only manipulators")
    if failed:
        print(f"Failed robots: {len(failed)}")
        for k, v in list(failed.items())[:10]:
            print(f"  - {k}: {v}")

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_frame("world", show_axes=True, axes_length=0.12, axes_radius=0.004)

    robot_slider = server.gui.add_slider(
        "robot_index",
        min=0,
        max=len(robot_names) - 1,
        step=1,
        initial_value=0,
    )
    vis_slider = server.gui.add_slider(
        "vis_mode (0=mesh,1=pc(filtered),2=both,3=pts+normals@orig,4=pts+normals@box,5=surface_graph)",
        min=0,
        max=5,
        step=1,
        initial_value=2,
    )
    link_slider = server.gui.add_slider(
        "link_mode (0=original, 1=box, 2=both)",
        min=0,
        max=2,
        step=1,
        initial_value=0,
    )
    graph_point_slider = server.gui.add_slider(
        "graph_point_index",
        min=0,
        max=max(0, int(args.num_points) - 1),
        step=1,
        initial_value=0,
    )
    graph_mesh_mode_slider = server.gui.add_slider(
        "graph_mesh_mode (0=none,1=outer,2=original,3=both)",
        min=0,
        max=3,
        step=1,
        initial_value=3,
    )
    graph_normal_scale_slider = server.gui.add_slider(
        "graph_normal_scale",
        min=0.0,
        max=0.03,
        step=0.0005,
        initial_value=0.0,
    )

    def render() -> None:
        ridx = int(robot_slider.value)
        vis_mode = int(vis_slider.value)
        link_mode = int(link_slider.value)
        vis_point_size = float(args.vis_point_size)

        robot_name = robot_names[ridx]
        model = models[robot_name]
        q = model.get_canonical_q().detach().clone()
        if hasattr(model, "base_translation_indices") and len(model.base_translation_indices) > 0:
            q[model.base_translation_indices] = 0.0

        if vis_mode == 5:
            points, normals = model.get_surface_points_normals(q=q)
            points = points.detach().cpu()
            normals = normals.detach().cpu()
            n = int(points.shape[0])
            pidx = int(max(0, min(int(graph_point_slider.value), max(0, n - 1))))
            neighbors = model.surface_graph_neighbors[pidx] if (n > 0 and pidx < len(model.surface_graph_neighbors)) else []

            simplified_mesh = getattr(model, "surface_union_mesh_canonical", None)
            if simplified_mesh is None:
                simplified_mesh = model.get_trimesh_q(q, mode="original")["visual"]
            original_mesh = model.get_trimesh_q(q, mode="original")["visual"]

            graph_mesh_mode = int(graph_mesh_mode_slider.value)
            show_simplified = graph_mesh_mode in (1, 3)
            show_original = graph_mesh_mode in (2, 3)
            normal_scale = float(graph_normal_scale_slider.value)

            center = points.mean(dim=0, keepdim=True) if n > 0 else torch.zeros((1, 3), dtype=torch.float32)
            if show_simplified and len(simplified_mesh.vertices) > 0:
                center = torch.tensor(simplified_mesh.vertices, dtype=torch.float32).mean(dim=0, keepdim=True)
            elif show_original and len(original_mesh.vertices) > 0:
                center = torch.tensor(original_mesh.vertices, dtype=torch.float32).mean(dim=0, keepdim=True)

            # Clear normal overview layers.
            _empty_mesh(server, "robot_mesh_original")
            _empty_mesh(server, "robot_mesh_box")
            _empty_pc(server, "robot_pc_original", point_size=vis_point_size)
            _empty_pc(server, "robot_pc_box", point_size=vis_point_size)
            _empty_pc(server, "box_face_points", point_size=vis_point_size)
            _empty_lines(server, "box_face_normals")

            if n > 0:
                points_vis = points - center
                colors = _uniform_colors(n, (0.25, 0.62, 1.00))
                if len(neighbors) > 0:
                    neigh_np = np.array(neighbors, dtype=np.int64)
                    neigh_np = neigh_np[(neigh_np >= 0) & (neigh_np < n)]
                    colors[neigh_np] = np.array([1.00, 0.85, 0.20], dtype=np.float32)
                colors[pidx] = np.array([1.00, 0.20, 0.20], dtype=np.float32)
                server.scene.add_point_cloud(
                    name="surface_graph_points",
                    points=points_vis.numpy().astype(np.float32),
                    colors=colors.astype(np.float32),
                    point_size=vis_point_size,
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
                    _empty_lines(server, "surface_graph_edges")

                if normal_scale > 0.0:
                    end = points_vis + normals * normal_scale
                    seg = torch.stack([points_vis, end], dim=1).numpy().astype(np.float32)
                    server.scene.add_line_segments(
                        name="surface_normals",
                        points=seg,
                        colors=(0.30, 0.95, 0.95),
                        line_width=1.5,
                    )
                    qseg = torch.stack([points_vis[pidx], end[pidx]], dim=0).unsqueeze(0).numpy().astype(np.float32)
                    server.scene.add_line_segments(
                        name="query_normal",
                        points=qseg,
                        colors=(1.0, 0.05, 0.95),
                        line_width=3.0,
                    )
                else:
                    _empty_lines(server, "surface_normals")
                    _empty_lines(server, "query_normal")
            else:
                _empty_pc(server, "surface_graph_points", point_size=vis_point_size)
                _empty_lines(server, "surface_graph_edges")
                _empty_lines(server, "surface_normals")
                _empty_lines(server, "query_normal")

            if show_simplified:
                v = torch.tensor(simplified_mesh.vertices, dtype=torch.float32) - center
                f = torch.tensor(simplified_mesh.faces, dtype=torch.int32)
                server.scene.add_mesh_simple(
                    name="hand_mesh_simplified",
                    vertices=v.numpy(),
                    faces=f.numpy(),
                    color=(0.20, 0.78, 0.35),
                    opacity=0.75,
                )
            else:
                _empty_mesh(server, "hand_mesh_simplified")

            if show_original:
                v = torch.tensor(original_mesh.vertices, dtype=torch.float32) - center
                f = torch.tensor(original_mesh.faces, dtype=torch.int32)
                server.scene.add_mesh_simple(
                    name="hand_mesh_original",
                    vertices=v.numpy(),
                    faces=f.numpy(),
                    color=(0.72, 0.72, 0.72),
                    opacity=0.45,
                )
            else:
                _empty_mesh(server, "hand_mesh_original")
            return

        _empty_pc(server, "surface_graph_points", point_size=vis_point_size)
        _empty_lines(server, "surface_graph_edges")
        _empty_lines(server, "surface_normals")
        _empty_lines(server, "query_normal")
        _empty_mesh(server, "hand_mesh_simplified")
        _empty_mesh(server, "hand_mesh_original")

        # Modes 3 and 4 show pre-sampled box-face points/normals.
        if vis_mode in (3, 4):
            model.switch_to_box_links_only_mode(override=False)

        original_pc = np.zeros((0, 3), dtype=np.float32)
        box_pc = np.zeros((0, 3), dtype=np.float32)
        original_mesh = None
        box_mesh = None

        if vis_mode in (0, 1, 2):
            include_original = link_mode in (0, 2)
            include_box = link_mode in (1, 2)

            if include_original:
                # Use filtered whole-hand surface template points.
                pc, _ = model.get_surface_points_normals(q=q)
                original_pc = pc.detach().cpu().numpy() if pc.numel() else np.zeros((0, 3), dtype=np.float32)
                original_mesh = model.get_trimesh_q(q, mode="original")["visual"]

            if include_box:
                model.switch_to_box_links_only_mode(override=False)
                pc = model.get_transformed_links_pc(q=q, mode="box")
                box_pc = pc[:, :3].detach().cpu().numpy() if pc.numel() else np.zeros((0, 3), dtype=np.float32)
                box_mesh = model.get_trimesh_q(q, mode="box")["visual"]

        elif vis_mode == 3:
            original_mesh = model.get_trimesh_q(q, mode="original")["visual"]
        elif vis_mode == 4:
            box_mesh = model.get_trimesh_q(q, mode="box")["visual"]

        center = np.zeros(3, dtype=np.float32)
        if original_pc.shape[0] > 0:
            center = original_pc.mean(axis=0).astype(np.float32)
        elif box_pc.shape[0] > 0:
            center = box_pc.mean(axis=0).astype(np.float32)
        elif original_mesh is not None and len(original_mesh.vertices) > 0:
            center = original_mesh.vertices.mean(axis=0).astype(np.float32)
        elif box_mesh is not None and len(box_mesh.vertices) > 0:
            center = box_mesh.vertices.mean(axis=0).astype(np.float32)

        # Mesh/point modes 0,1,2
        show_mesh = vis_mode in (0, 2)
        show_pc = vis_mode in (1, 2)

        if show_mesh and original_mesh is not None:
            mesh_vis = original_mesh.copy()
            if len(mesh_vis.vertices) > 0:
                mesh_vis.apply_translation(-center)
            server.scene.add_mesh_simple(
                name="robot_mesh_original",
                vertices=mesh_vis.vertices.astype(np.float32),
                faces=mesh_vis.faces.astype(np.int32),
                color=(0.30, 0.65, 1.00),
                opacity=0.55 if box_mesh is not None else 0.85,
            )
        elif vis_mode == 3 and original_mesh is not None:
            mesh_vis = original_mesh.copy()
            mesh_vis.apply_translation(-center)
            server.scene.add_mesh_simple(
                name="robot_mesh_original",
                vertices=mesh_vis.vertices.astype(np.float32),
                faces=mesh_vis.faces.astype(np.int32),
                color=(0.30, 0.65, 1.00),
                opacity=0.85,
            )
        else:
            _empty_mesh(server, "robot_mesh_original")

        if show_mesh and box_mesh is not None:
            mesh_vis = box_mesh.copy()
            if len(mesh_vis.vertices) > 0:
                mesh_vis.apply_translation(-center)
            server.scene.add_mesh_simple(
                name="robot_mesh_box",
                vertices=mesh_vis.vertices.astype(np.float32),
                faces=mesh_vis.faces.astype(np.int32),
                color=(1.00, 0.55, 0.20),
                opacity=0.45 if original_mesh is not None else 0.85,
            )
        elif vis_mode == 4 and box_mesh is not None:
            mesh_vis = box_mesh.copy()
            mesh_vis.apply_translation(-center)
            server.scene.add_mesh_simple(
                name="robot_mesh_box",
                vertices=mesh_vis.vertices.astype(np.float32),
                faces=mesh_vis.faces.astype(np.int32),
                color=(1.00, 0.55, 0.20),
                opacity=0.85,
            )
        else:
            _empty_mesh(server, "robot_mesh_box")

        if show_pc and original_pc.shape[0] > 0:
            pc_show = original_pc - center
            colors = _uniform_colors(pc_show.shape[0], (0.25, 0.60, 1.00))
            server.scene.add_point_cloud(
                name="robot_pc_original",
                points=pc_show.astype(np.float32),
                colors=colors,
                point_size=vis_point_size,
                point_shape="circle",
            )
        else:
            _empty_pc(server, "robot_pc_original", point_size=vis_point_size)

        if show_pc and box_pc.shape[0] > 0:
            pc_show = box_pc - center
            colors = _uniform_colors(pc_show.shape[0], (1.00, 0.60, 0.15))
            server.scene.add_point_cloud(
                name="robot_pc_box",
                points=pc_show.astype(np.float32),
                colors=colors,
                point_size=vis_point_size,
                point_shape="circle",
            )
        else:
            _empty_pc(server, "robot_pc_box", point_size=vis_point_size)

        # Mode 3/4: box-face points + normals.
        if vis_mode in (3, 4):
            face_pts, face_nrm = model.get_box_face_points_normals(q=q)
            face_pts = face_pts.detach().cpu().numpy()
            face_nrm = face_nrm.detach().cpu().numpy()
            if face_pts.size > 0:
                pts = face_pts.reshape(-1, 3)
                nrm = face_nrm.reshape(-1, 3)
                valid = np.isfinite(pts).all(axis=1) & np.isfinite(nrm).all(axis=1)
                valid = valid & (np.linalg.norm(nrm, axis=1) > 1e-8)
                pts = pts[valid]
                nrm = nrm[valid]
                pts = pts - center

                if pts.shape[0] > 0:
                    server.scene.add_point_cloud(
                        name="box_face_points",
                        points=pts.astype(np.float32),
                        colors=_uniform_colors(pts.shape[0], (1.0, 0.95, 0.25)),
                        point_size=vis_point_size,
                        point_shape="circle",
                    )

                    seg = np.stack([pts, pts + 0.03 * nrm], axis=1).astype(np.float32)
                    server.scene.add_line_segments(
                        name="box_face_normals",
                        points=seg,
                        colors=(1.0, 0.2, 0.2),
                        line_width=2.0,
                    )
                else:
                    _empty_pc(server, "box_face_points", point_size=vis_point_size)
                    _empty_lines(server, "box_face_normals")
            else:
                _empty_pc(server, "box_face_points", point_size=vis_point_size)
                _empty_lines(server, "box_face_normals")
        else:
            _empty_pc(server, "box_face_points", point_size=vis_point_size)
            _empty_lines(server, "box_face_normals")

    @robot_slider.on_update
    def _(_: viser.GuiEvent) -> None:
        render()

    @vis_slider.on_update
    def _(_: viser.GuiEvent) -> None:
        render()

    @link_slider.on_update
    def _(_: viser.GuiEvent) -> None:
        render()

    @graph_point_slider.on_update
    def _(_: viser.GuiEvent) -> None:
        render()

    @graph_mesh_mode_slider.on_update
    def _(_: viser.GuiEvent) -> None:
        render()

    @graph_normal_scale_slider.on_update
    def _(_: viser.GuiEvent) -> None:
        render()

    render()
    print(f"Viser running at http://{args.host}:{args.port}")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    # Compatibility mode:
    # - `python robot_model/robot_model_vis.py` -> robot overview visualizer
    # - `python robot_model/robot_model_vis.py surface_graph ...` -> surface graph visualizer
    if len(sys.argv) > 1 and sys.argv[1] == "surface_graph":
        main_surface_graph_cli(argv=sys.argv[2:])
    else:
        main()
