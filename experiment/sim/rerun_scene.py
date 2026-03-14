from __future__ import annotations

import warnings
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr
import trimesh

DEFAULT_FLOOR_COLOR = np.array([200, 200, 200], dtype=np.uint8)
DEFAULT_OBJECT_COLOR = np.array([100, 149, 237], dtype=np.uint8)
DEFAULT_HAND_COLOR = np.array([200, 200, 200], dtype=np.uint8)


def _xyzw_from_wxyz(wxyz: np.ndarray) -> np.ndarray:
    return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float32)


def _get_mesh_group_path(
    geom_name: str,
    entity_root: str,
    *,
    is_collision: bool,
) -> str:
    name = geom_name.lower()
    if "collision" in name:
        return f"{entity_root}/collision"
    if "visual" in name:
        return f"{entity_root}/visual"
    if is_collision:
        return f"{entity_root}/collision"
    return f"{entity_root}/visual"


def _get_entity_color(entity_name: str) -> np.ndarray:
    entity_lower = entity_name.lower()
    if "object" in entity_lower:
        return DEFAULT_OBJECT_COLOR
    if any(part in entity_lower for part in ("hand", "thumb", "index", "middle", "ring", "pinky")):
        return DEFAULT_HAND_COLOR
    return np.array([240, 240, 240], dtype=np.uint8)


def _vertex_colors_from_rgba(mesh: trimesh.Trimesh, rgba: np.ndarray | None) -> np.ndarray | None:
    if rgba is None:
        return None
    rgba_arr = np.asarray(rgba)
    if rgba_arr.size == 3:
        rgba_arr = np.concatenate([rgba_arr, [1.0]], axis=0)
    if rgba_arr.size != 4:
        return None
    if rgba_arr.dtype.kind == "f":
        rgba_arr = np.clip(rgba_arr, 0.0, 1.0) * 255.0
    rgba_uint8 = rgba_arr.astype(np.uint8)
    return np.tile(rgba_uint8[None, :], (len(mesh.vertices), 1))


def _mujoco_mesh_to_trimesh(model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh | None:
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return None

    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])

    vertices = np.asarray(model.mesh_vert[vert_start : vert_start + vert_count], dtype=np.float32)
    faces = np.asarray(model.mesh_face[face_start : face_start + face_count], dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    rgba = None
    mat_id = int(model.geom_matid[geom_id])
    if mat_id >= 0 and mat_id < int(model.nmat):
        rgba = np.asarray(model.mat_rgba[mat_id], dtype=np.float32)
    else:
        rgba = np.asarray(model.geom_rgba[geom_id], dtype=np.float32)
    vertex_colors = _vertex_colors_from_rgba(mesh, rgba)
    if vertex_colors is not None:
        mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
    return mesh


def _trimesh_from_primitive(geom_type: int, size: np.ndarray, rgba: np.ndarray | None) -> trimesh.Trimesh | None:
    geom_type_enum = mujoco.mjtGeom
    if geom_type == geom_type_enum.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=float(size[0]), subdivisions=2)
    elif geom_type == geom_type_enum.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=float(size[0]), height=float(2.0 * size[1]))
    elif geom_type == geom_type_enum.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=float(2.0 * size[1]))
    elif geom_type == geom_type_enum.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2.0 * np.asarray(size[:3], dtype=np.float32))
    elif geom_type == geom_type_enum.mjGEOM_PLANE:
        mesh = trimesh.creation.box(extents=[20.0, 20.0, 0.01])
    else:
        return None

    vertex_colors = _vertex_colors_from_rgba(mesh, rgba)
    if vertex_colors is not None:
        mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
    return mesh


def _log_trimesh_entity(entity_path: str, mesh: trimesh.Trimesh, fallback_color: np.ndarray) -> None:
    vertex_positions = np.asarray(mesh.vertices, dtype=np.float32)
    triangle_indices = np.asarray(mesh.faces, dtype=np.uint32)
    if vertex_positions.size == 0 or triangle_indices.size == 0:
        return
    vertex_normals = None
    if hasattr(mesh, "vertex_normals") and len(mesh.vertex_normals) == len(mesh.vertices):
        vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

    vertex_colors = None
    if hasattr(mesh.visual, "vertex_colors"):
        colors = np.asarray(mesh.visual.vertex_colors)
        if colors.ndim == 2 and colors.shape[0] == len(mesh.vertices):
            vertex_colors = colors.astype(np.uint8)

    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=vertex_positions,
            triangle_indices=triangle_indices,
            vertex_normals=vertex_normals,
            vertex_colors=vertex_colors if vertex_colors is not None else fallback_color,
            albedo_factor=fallback_color,
        ),
        static=True,
    )


def build_and_log_scene_from_spec(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    *,
    entity_root: str = "mujoco",
) -> list[tuple[str, int]]:
    body_placeholder_idx = 0
    geom_placeholder_idx = 0
    for body in spec.bodies[1:]:
        if not body.name:
            body.name = f"RERUN_BODY_{body_placeholder_idx}"
            body_placeholder_idx += 1
        for geom in body.geoms:
            if not geom.name:
                geom.name = f"RERUN_GEOM_{geom_placeholder_idx}"
                geom_placeholder_idx += 1

    rr.log(f"{entity_root}/world", rr.Transform3D(translation=[0.0, 0.0, 0.0]), static=True)
    rr.log(f"{entity_root}/collision", rr.Transform3D(), static=True)
    rr.log(f"{entity_root}/visual", rr.Transform3D(), static=True)
    rr.log(
        f"{entity_root}/floor",
        rr.Boxes3D(half_sizes=[[0.3, 0.3, 0.001]], colors=DEFAULT_FLOOR_COLOR, fill_mode=3),
        static=True,
    )

    geom_name_to_id: dict[str, int] = {}
    for geom_id in range(int(model.ngeom)):
        try:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        except Exception:
            name = None
        if name:
            geom_name_to_id[name] = geom_id

    geom_by_body_index: dict[tuple[int, int], int] = {}
    body_geom_counts: dict[int, int] = {}
    for geom_id in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[geom_id])
        geom_index = body_geom_counts.get(body_id, 0)
        geom_by_body_index[(body_id, geom_index)] = geom_id
        body_geom_counts[body_id] = geom_index + 1

    body_entity_and_ids: list[tuple[str, int]] = []
    for body in spec.bodies[1:]:
        body_name = str(body.name)
        collision_entity = f"{entity_root}/collision/{body_name}"
        visual_entity = f"{entity_root}/visual/{body_name}"
        body_entity_and_ids.append((collision_entity, int(body.id)))
        body_entity_and_ids.append((visual_entity, int(body.id)))
        rr.log(collision_entity, rr.Transform3D())
        rr.log(visual_entity, rr.Transform3D())

        for geom_idx, geom in enumerate(body.geoms):
            geom_name = str(geom.name)
            model_geom_id = geom_name_to_id.get(geom_name, -1)
            if model_geom_id < 0:
                model_geom_id = geom_by_body_index.get((int(body.id), int(geom_idx)), -1)
            is_collision = False
            if model_geom_id >= 0:
                is_collision = bool(model.geom_contype[model_geom_id] != 0 or model.geom_conaffinity[model_geom_id] != 0)
            geom_entity = f"{_get_mesh_group_path(geom_name, entity_root, is_collision=is_collision)}/{body_name}/{geom_name}"
            if model_geom_id >= 0:
                local_pos = np.asarray(model.geom_pos[model_geom_id], dtype=np.float32)
                local_quat = np.asarray(model.geom_quat[model_geom_id], dtype=np.float32)
                geom_type = int(model.geom_type[model_geom_id])
                geom_size = np.asarray(model.geom_size[model_geom_id], dtype=np.float32)
            else:
                local_pos = np.asarray(getattr(geom, "pos", [0.0, 0.0, 0.0]), dtype=np.float32)
                local_quat = np.asarray(getattr(geom, "quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
                geom_type = int(geom.type)
                geom_size = np.asarray(getattr(geom, "size", [0.0, 0.0, 0.0]), dtype=np.float32)
            rr.log(
                geom_entity,
                rr.Transform3D(translation=local_pos, quaternion=_xyzw_from_wxyz(local_quat)),
                static=True,
            )
            tm = None
            fallback_color = _get_entity_color(geom_entity)
            rgba = None
            if model_geom_id >= 0:
                rgba = np.asarray(model.geom_rgba[model_geom_id], dtype=np.float32)
            if geom_type == mujoco.mjtGeom.mjGEOM_MESH and model_geom_id >= 0:
                try:
                    tm = _mujoco_mesh_to_trimesh(model, model_geom_id)
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(f"Failed to convert mesh geom '{geom_name}': {exc}", stacklevel=1)
            if tm is None and model_geom_id >= 0:
                tm = _trimesh_from_primitive(geom_type, geom_size, rgba)
            if tm is None:
                continue
            _log_trimesh_entity(geom_entity, tm, fallback_color)

    return body_entity_and_ids


def init_rerun(*, app_name: str, spawn: bool) -> None:
    rr.init(app_name)
    if spawn:
        rr.spawn()


def save_rerun(path: str | Path) -> None:
    rr.save(str(Path(path).expanduser().resolve()))


def log_frame(
    data: mujoco.MjData,
    *,
    sim_time: float,
    viewer_body_entity_and_ids: list[tuple[str, int]],
) -> None:
    rr.set_time("sim_time", timestamp=float(sim_time))
    for entity, body_id in viewer_body_entity_and_ids:
        pos = np.asarray(data.xpos[body_id], dtype=np.float32)
        quat_xyzw = _xyzw_from_wxyz(np.asarray(data.xquat[body_id], dtype=np.float32))
        rr.log(entity, rr.Transform3D(translation=pos, quaternion=quat_xyzw))
