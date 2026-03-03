from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import torch
import trimesh
import viser


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize manipulator contact point clouds extracted by extract_contact_points.py."
        )
    )
    parser.add_argument(
        "--contact-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "contact_gen", "contact_points_validate.pt"),
        help="Path to extracted contact payload (.pt).",
    )
    parser.add_argument(
        "--dro-root",
        type=str,
        default=None,
        help="Optional DRO-Grasp root. If omitted, read from payload meta.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point-size", type=float, default=0.004)
    parser.add_argument("--hand-opacity", type=float, default=0.8)
    parser.add_argument("--object-opacity", type=float, default=1.0)
    return parser


def _to_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().float().cpu()
    return torch.tensor(x, dtype=torch.float32)


def _resolve_robot_model_name(robot_name: str, asset_names: set[str]) -> str:
    if robot_name in asset_names:
        return robot_name

    preferred = {
        "allegro": [
            "dro/allegro/allegro_hand_left_extended",
            "dro/allegro/allegro_hand_left",
        ],
        "barrett": [
            "dro/barrett/model_extended",
            "dro/barrett/model",
        ],
        "ezgripper": [
            "dro/ezgripper/ezgripper_extended",
            "dro/ezgripper/ezgripper",
        ],
        "robotiq_3finger": [
            "dro/urdf/robotiq_3finger_description_extended",
            "dro/urdf/robotiq_3finger_description",
        ],
        "shadowhand": [
            "dro/shadowhand/shadow_hand_right_extended",
            "dro/shadowhand/shadow_hand_right",
        ],
    }
    for candidate in preferred.get(robot_name, []):
        if candidate in asset_names:
            return candidate

    candidates = [k for k in asset_names if (f"/{robot_name}/" in k) or (robot_name in k)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        candidates = sorted(
            candidates,
            key=lambda x: (
                int(x.startswith("dro/")),
                int("extended" in x),
                int("right" in x),
                -len(x),
            ),
            reverse=True,
        )
        return candidates[0]
    raise KeyError(f"Cannot resolve robot model for '{robot_name}'.")


def _object_mesh_path(dro_root: str, object_name: str) -> str:
    dataset_name, mesh_name = object_name.split("+")
    return os.path.join(
        dro_root,
        "data/data_urdf/object",
        dataset_name,
        mesh_name,
        f"{mesh_name}.stl",
    )


def _empty_mesh() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros((0, 3), dtype=torch.float32), torch.zeros((0, 3), dtype=torch.int32)


def _empty_points() -> torch.Tensor:
    return torch.zeros((0, 3), dtype=torch.float32)


def main() -> None:
    args = _build_parser().parse_args()
    contact_path = os.path.abspath(args.contact_path)
    if not os.path.exists(contact_path):
        raise FileNotFoundError(f"contact payload not found: {contact_path}")

    payload = torch.load(contact_path, map_location="cpu")
    samples: list[dict[str, Any]] = list(payload["samples"])
    if len(samples) == 0:
        raise RuntimeError(f"No samples in payload: {contact_path}")

    if args.dro_root is None:
        meta = payload.get("meta", {})
        if not isinstance(meta, dict) or "dro_root" not in meta:
            raise ValueError("Cannot infer dro_root from payload meta. Please set --dro-root.")
        dro_root = os.path.abspath(str(meta["dro_root"]))
    else:
        dro_root = os.path.abspath(args.dro_root)
    if not os.path.exists(dro_root):
        raise FileNotFoundError(f"dro_root does not exist: {dro_root}")

    sys.path.append(PROJECT_ROOT)
    from robot_model.robot_model import create_robot_model, discover_robot_assets  # type: ignore

    assets_meta = discover_robot_assets()
    asset_names = set(assets_meta.keys())

    hand_cache: dict[str, Any] = {}
    mesh_cache: dict[str, trimesh.Trimesh] = {}

    server = viser.ViserServer(host=args.host, port=int(args.port))

    sample_slider = server.gui.add_slider(
        label="sample_index",
        min=0,
        max=max(0, len(samples) - 1),
        step=1,
        initial_value=0,
    )
    mode_slider = server.gui.add_slider(
        label="vis_mode (0=points,1=points+hand,2=points+hand+object)",
        min=0,
        max=2,
        step=1,
        initial_value=2,
    )

    def _get_hand(robot_name: str, num_hand_points: int, robot_model_name_hint: str | None = None):
        if robot_name not in hand_cache:
            resolved = (
                robot_model_name_hint
                if (robot_model_name_hint is not None and robot_model_name_hint in asset_names)
                else _resolve_robot_model_name(robot_name, asset_names)
            )
            hand_cache[robot_name] = create_robot_model(
                robot_name=resolved,
                device=torch.device("cpu"),
                num_points=int(num_hand_points),
            )
        return hand_cache[robot_name]

    def _get_object_mesh(object_name: str) -> trimesh.Trimesh:
        if object_name not in mesh_cache:
            mesh_cache[object_name] = trimesh.load_mesh(_object_mesh_path(dro_root, object_name))
        return mesh_cache[object_name]

    def _render(sample_index: int, vis_mode: int) -> None:
        sample = samples[sample_index]
        robot_name = str(sample["robot_name"])
        robot_model_name = (
            None if sample.get("robot_model_name") is None else str(sample.get("robot_model_name"))
        )
        object_name = str(sample["object_name"])
        q = _to_tensor(sample["q"])
        num_hand_points = int(sample.get("num_hand_points", payload.get("meta", {}).get("num_hand_points", 256)))
        hand_contact_points = _to_tensor(sample["hand_contact_points"])
        if hand_contact_points.ndim != 2:
            hand_surface_points = _to_tensor(sample.get("hand_surface_points", _empty_points()))
            hand_mask = torch.as_tensor(sample.get("hand_contact_mask", torch.zeros((0,), dtype=torch.bool))).bool().view(-1)
            if hand_surface_points.ndim == 2 and hand_mask.numel() == hand_surface_points.shape[0]:
                hand_contact_points = hand_surface_points[hand_mask]
            else:
                hand_contact_points = _empty_points()
        has_contact_points = hand_contact_points.ndim == 2 and hand_contact_points.shape[0] > 0

        server.scene.add_point_cloud(
            "hand_contact_points",
            hand_contact_points.numpy() if has_contact_points else _empty_points().numpy(),
            point_size=float(args.point_size),
            point_shape="circle",
            colors=(255, 90, 90),
            visible=bool(has_contact_points),
        )

        if vis_mode >= 1:
            hand = _get_hand(robot_name, num_hand_points, robot_model_name)
            hand_mesh = hand.get_trimesh_q(q)["visual"]
            hand_vertices = torch.tensor(hand_mesh.vertices, dtype=torch.float32)
            hand_faces = torch.tensor(hand_mesh.faces, dtype=torch.int32)
            server.scene.add_mesh_simple(
                "hand_mesh",
                hand_vertices.numpy(),
                hand_faces.numpy(),
                color=(102, 192, 255),
                opacity=float(args.hand_opacity),
                visible=True,
            )
        else:
            vertices, faces = _empty_mesh()
            server.scene.add_mesh_simple(
                "hand_mesh",
                vertices.numpy(),
                faces.numpy(),
                color=(0, 0, 0),
                opacity=0.0,
                visible=False,
            )

        if vis_mode >= 2:
            object_mesh = _get_object_mesh(object_name)
            obj_vertices = torch.tensor(object_mesh.vertices, dtype=torch.float32)
            obj_faces = torch.tensor(object_mesh.faces, dtype=torch.int32)
            server.scene.add_mesh_simple(
                "object_mesh",
                obj_vertices.numpy(),
                obj_faces.numpy(),
                color=(239, 132, 167),
                opacity=float(args.object_opacity),
                visible=True,
            )
        else:
            vertices, faces = _empty_mesh()
            server.scene.add_mesh_simple(
                "object_mesh",
                vertices.numpy(),
                faces.numpy(),
                color=(0, 0, 0),
                opacity=0.0,
                visible=False,
            )

        print(
            f"[sample {sample_index}] robot={robot_name} object={object_name} "
            f"hand_contacts={sample.get('num_hand_contacts', 'n/a')} "
            f"mode={vis_mode}"
        )
        if sample.get("invalid_reason") is not None:
            print(f"  invalid_reason: {sample['invalid_reason']}")

    sample_slider.on_update(lambda _: _render(int(sample_slider.value), int(mode_slider.value)))
    mode_slider.on_update(lambda _: _render(int(sample_slider.value), int(mode_slider.value)))

    _render(0, int(mode_slider.value))
    print(f"Viewer URL: http://{args.host}:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
