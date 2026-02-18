from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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


def _empty_pc(server: viser.ViserServer, name: str) -> None:
    server.scene.add_point_cloud(
        name=name,
        points=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.float32),
        point_size=0.004,
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


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = np.random.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _downsample_points_normals(points: np.ndarray, normals: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= max_points:
        return points, normals
    idx = np.random.choice(points.shape[0], size=max_points, replace=False)
    return points[idx], normals[idx]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize URDF/XML robots with RobotModel + viser")
    parser.add_argument("--assets-dir", type=str, default=str(ROOT_DIR / "assets"))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9402)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--only-manipulators", action="store_true")
    parser.add_argument("--robot-list-file", type=str, default=str(ROOT_DIR / "assets" / "manipulator_robot_lists.json"))
    parser.add_argument("--robot-list-key", type=str, default="all", choices=["all", "both_hands", "right_hands"])
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
            model = create_robot_model(name, device=device, num_points=512, assets_dir=assets_dir)
            if args.only_manipulators and _is_humanoid_model(name, model):
                continue
            models[name] = model
        except Exception as exc:
            failed[name] = str(exc)

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
        "vis_mode (0=mesh,1=pc,2=both,3=pts+normals@orig,4=pts+normals@box)",
        min=0,
        max=4,
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

    def render() -> None:
        ridx = int(robot_slider.value)
        vis_mode = int(vis_slider.value)
        link_mode = int(link_slider.value)

        robot_name = robot_names[ridx]
        model = models[robot_name]
        q = torch.zeros(model.get_canonical_q().shape, device=model.device)

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
                pc = model.get_transformed_links_pc(q=q, mode="original")
                original_pc = pc[:, :3].detach().cpu().numpy() if pc.numel() else np.zeros((0, 3), dtype=np.float32)
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
            pc_show = _downsample_points(original_pc - center, args.num_points)
            colors = _uniform_colors(pc_show.shape[0], (0.25, 0.60, 1.00))
            server.scene.add_point_cloud(
                name="robot_pc_original",
                points=pc_show.astype(np.float32),
                colors=colors,
                point_size=0.0035,
                point_shape="circle",
            )
        else:
            _empty_pc(server, "robot_pc_original")

        if show_pc and box_pc.shape[0] > 0:
            pc_show = _downsample_points(box_pc - center, args.num_points)
            colors = _uniform_colors(pc_show.shape[0], (1.00, 0.60, 0.15))
            server.scene.add_point_cloud(
                name="robot_pc_box",
                points=pc_show.astype(np.float32),
                colors=colors,
                point_size=0.0035,
                point_shape="circle",
            )
        else:
            _empty_pc(server, "robot_pc_box")

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
                pts, nrm = _downsample_points_normals(pts, nrm, args.num_points)
                pts = pts - center

                if pts.shape[0] > 0:
                    server.scene.add_point_cloud(
                        name="box_face_points",
                        points=pts.astype(np.float32),
                        colors=_uniform_colors(pts.shape[0], (1.0, 0.95, 0.25)),
                        point_size=0.006,
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
                    _empty_pc(server, "box_face_points")
                    _empty_lines(server, "box_face_normals")
            else:
                _empty_pc(server, "box_face_points")
                _empty_lines(server, "box_face_normals")
        else:
            _empty_pc(server, "box_face_points")
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

    render()
    print(f"Viser running at http://{args.host}:{args.port}")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
