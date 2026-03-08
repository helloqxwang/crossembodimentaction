from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation

from pytorch3d.ops import sample_farthest_points


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS_DIR = ROOT_DIR / "assets"
DEFAULT_MANIPULATOR_LISTS_JSON = DEFAULT_ASSETS_DIR / "manipulator_robot_lists.json"


def _stable_int_seed(key: str) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**31 - 1)


@contextlib.contextmanager
def _temporary_global_seed(seed: int):
    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    np.random.seed(int(seed) % (2**32 - 1))
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _build_chain_quiet(urdf_data: bytes | str) -> pk.chain.Chain:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        return pk.build_chain_from_urdf(urdf_data)


def _as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            return None
        return trimesh.util.concatenate(
            tuple(
                trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in scene_or_mesh.geometry.values()
            )
        )
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh
    return None


def farthest_point_sampling(
    pos: torch.Tensor,
    n_sampling: int,
    lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pos.ndim == 2:
        pos = pos.unsqueeze(0)
        squeeze = True
    elif pos.ndim == 3:
        squeeze = False
    else:
        raise ValueError("pos must be shape (N, C) or (B, N, C)")

    n_points = pos.shape[1]
    k = min(int(n_sampling), n_points)
    if k <= 0:
        raise ValueError("n_sampling must be positive")

    if lengths is not None:
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=pos.device).view(-1)
        if int(lengths.numel()) != int(pos.shape[0]):
            raise ValueError(
                f"lengths must have one entry per batch element, got {tuple(lengths.shape)} for batch {pos.shape[0]}"
            )
        lengths = lengths.clamp_min(0).clamp_max(int(n_points))

    if sample_farthest_points is not None:
        sampled_xyz, sampled_idx = sample_farthest_points(
            pos[..., :3].contiguous(),
            lengths=lengths,
            K=k,
        )
        if pos.shape[-1] > 3:
            gather_idx = sampled_idx.clamp_min(0)[..., :, None].expand(*sampled_idx.shape, pos.shape[-1])
            sampled = torch.gather(pos, -2, gather_idx)
            sampled = sampled.masked_fill(sampled_idx[..., None] < 0, 0.0)
        else:
            sampled = sampled_xyz
    else:
        sampled_rows = []
        sampled_idx_rows = []
        for batch_idx in range(int(pos.shape[0])):
            cur_len = int(lengths[batch_idx].item()) if lengths is not None else int(n_points)
            cur_idx = torch.full((k,), -1, dtype=torch.long, device=pos.device)
            cur_sample = torch.zeros((k, pos.shape[-1]), dtype=pos.dtype, device=pos.device)
            if cur_len > 0:
                keep = torch.randperm(cur_len, device=pos.device)[: min(k, cur_len)]
                cur_idx[: int(keep.numel())] = keep
                cur_sample[: int(keep.numel())] = pos[batch_idx, keep]
            sampled_rows.append(cur_sample)
            sampled_idx_rows.append(cur_idx)
        sampled = torch.stack(sampled_rows, dim=0)
        sampled_idx = torch.stack(sampled_idx_rows, dim=0)

    if squeeze:
        sampled = sampled.squeeze(0)
        sampled_idx = sampled_idx.squeeze(0)

    return sampled, sampled_idx


def _matrix_to_euler_xyz(matrix: torch.Tensor) -> torch.Tensor:
    device = matrix.device
    euler = Rotation.from_matrix(matrix.detach().cpu().numpy()).as_euler("XYZ")
    return torch.tensor(euler, dtype=torch.float32, device=device)


def _euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    device = euler.device
    mat = Rotation.from_euler("XYZ", euler.detach().cpu().numpy()).as_matrix()
    return torch.tensor(mat, dtype=torch.float32, device=device)


def _matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.T.reshape(9)[:6]


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / torch.norm(v, dim=-1, keepdim=True).clamp_min(1e-8)


def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    x = _normalize(rot6d[..., 0:3])
    y = _normalize(rot6d[..., 3:6])
    a = _normalize(x + y)
    b = _normalize(x - y)
    x = _normalize(a + b)
    y = _normalize(a - b)
    z = _normalize(torch.cross(x, y, dim=-1))
    return torch.stack([x, y, z], dim=-2).mT


def _euler_to_rot6d(euler: torch.Tensor) -> torch.Tensor:
    return _matrix_to_rot6d(_euler_xyz_to_matrix(euler))


def _rot6d_to_euler(rot6d: torch.Tensor) -> torch.Tensor:
    return _matrix_to_euler_xyz(_rot6d_to_matrix(rot6d))


def _axisangle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    (x, y, z), c, s = axis, torch.cos(angle), torch.sin(angle)
    return torch.tensor(
        [
            [(1 - c) * x * x + c, (1 - c) * x * y - s * z, (1 - c) * x * z + s * y],
            [(1 - c) * x * y + s * z, (1 - c) * y * y + c, (1 - c) * y * z - s * x],
            [(1 - c) * x * z - s * y, (1 - c) * y * z + s * x, (1 - c) * z * z + c],
        ],
        dtype=torch.float32,
        device=axis.device,
    )


def q_euler_to_q_rot6d(q_euler: torch.Tensor) -> torch.Tensor:
    if q_euler.shape[-1] < 6:
        return q_euler
    return torch.cat([q_euler[..., :3], _euler_to_rot6d(q_euler[..., 3:6]), q_euler[..., 6:]], dim=-1)


def q_rot6d_to_q_euler(q_rot6d: torch.Tensor) -> torch.Tensor:
    if q_rot6d.shape[-1] < 9:
        return q_rot6d
    return torch.cat([q_rot6d[..., :3], _rot6d_to_euler(q_rot6d[..., 3:9]), q_rot6d[..., 9:]], dim=-1)


def _parse_vec(attr_val: Optional[str], n: int, default: List[float]) -> np.ndarray:
    if attr_val is None:
        return np.array(default, dtype=np.float64)
    vals = np.fromstring(attr_val, sep=" ", dtype=np.float64)
    if vals.size != n:
        return np.array(default, dtype=np.float64)
    return vals


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    return Rotation.from_quat(quat_xyzw).as_matrix()


def _parse_pose_element(elem: ET.Element, angle_in_radian: bool = True) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    pos = _parse_vec(elem.attrib.get("pos"), 3, [0.0, 0.0, 0.0])
    t[:3, 3] = pos

    if "quat" in elem.attrib:
        q = _parse_vec(elem.attrib.get("quat"), 4, [1.0, 0.0, 0.0, 0.0])
        t[:3, :3] = _quat_wxyz_to_matrix(q)
    elif "euler" in elem.attrib:
        e = _parse_vec(elem.attrib.get("euler"), 3, [0.0, 0.0, 0.0])
        if not angle_in_radian:
            e = np.deg2rad(e)
        t[:3, :3] = Rotation.from_euler("xyz", e).as_matrix()
    elif "axisangle" in elem.attrib:
        aa = _parse_vec(elem.attrib.get("axisangle"), 4, [1.0, 0.0, 0.0, 0.0])
        axis = aa[:3]
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-8:
            axis = axis / axis_norm
        angle = aa[3]
        if not angle_in_radian:
            angle = np.deg2rad(angle)
        t[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()

    return t


def _resolve_file_path(path_str: str, base_dir: Path) -> Path:
    p = path_str.replace("package://", "").replace("file://", "")
    candidate = Path(p)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _load_mesh_geometry(mesh_path: Path, scale: Optional[np.ndarray] = None) -> Optional[trimesh.Trimesh]:
    if not mesh_path.exists():
        return None
    try:
        mesh = _as_mesh(trimesh.load(str(mesh_path), force="mesh"))
    except Exception:
        return None
    if mesh is None:
        return None
    if scale is not None:
        mesh.apply_scale(scale)
    return mesh


def _load_urdf_link_geometries(urdf_path: Path, link_names: List[str]) -> Dict[str, trimesh.Trimesh]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = urdf_path.parent

    link_meshes: Dict[str, trimesh.Trimesh] = {}

    for link in root.findall("link"):
        link_name = link.attrib.get("name")
        if link_name not in link_names:
            continue

        meshes: List[trimesh.Trimesh] = []
        for visual in link.findall("visual"):
            origin = visual.find("origin")
            origin_tf = np.eye(4, dtype=np.float64)
            if origin is not None:
                xyz = _parse_vec(origin.attrib.get("xyz"), 3, [0.0, 0.0, 0.0])
                rpy = _parse_vec(origin.attrib.get("rpy"), 3, [0.0, 0.0, 0.0])
                origin_tf[:3, 3] = xyz
                origin_tf[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()

            geometry = visual.find("geometry")
            if geometry is None or len(list(geometry)) == 0:
                continue

            geom_node = list(geometry)[0]
            mesh: Optional[trimesh.Trimesh] = None

            if geom_node.tag.endswith("mesh"):
                fname = geom_node.attrib.get("filename", "")
                scale = _parse_vec(geom_node.attrib.get("scale"), 3, [1.0, 1.0, 1.0])
                mesh_path = _resolve_file_path(fname, urdf_dir)
                mesh = _load_mesh_geometry(mesh_path, scale)
            elif geom_node.tag.endswith("box"):
                size = _parse_vec(geom_node.attrib.get("size"), 3, [0.01, 0.01, 0.01])
                mesh = trimesh.creation.box(extents=size)
            elif geom_node.tag.endswith("sphere"):
                radius = float(geom_node.attrib.get("radius", "0.01"))
                mesh = trimesh.creation.icosphere(radius=radius)
            elif geom_node.tag.endswith("cylinder"):
                radius = float(geom_node.attrib.get("radius", "0.01"))
                length = float(geom_node.attrib.get("length", "0.01"))
                mesh = trimesh.creation.cylinder(radius=radius, height=length)

            if mesh is None:
                continue

            mesh = mesh.copy()
            mesh.apply_transform(origin_tf)
            meshes.append(mesh)

        if meshes:
            if len(meshes) == 1:
                link_meshes[link_name] = meshes[0]
            else:
                link_meshes[link_name] = trimesh.util.concatenate(meshes)

    return link_meshes


def _parse_mjcf_defaults(root: ET.Element) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], Dict[str, Dict[str, str]]]:
    class_defaults: Dict[str, Dict[str, Dict[str, str]]] = {}
    base_defaults = {"geom": {}, "joint": {}}

    default_root = root.find("default")
    if default_root is None:
        return class_defaults, base_defaults

    def walk_default(node: ET.Element, inherited: Dict[str, Dict[str, str]]) -> None:
        current = {"geom": dict(inherited["geom"]), "joint": dict(inherited["joint"])}

        geom_node = node.find("geom")
        if geom_node is not None:
            current["geom"].update(geom_node.attrib)

        joint_node = node.find("joint")
        if joint_node is not None:
            current["joint"].update(joint_node.attrib)

        cls = node.attrib.get("class")
        if cls is not None:
            class_defaults[cls] = {"geom": dict(current["geom"]), "joint": dict(current["joint"])}

        for child_default in node.findall("default"):
            walk_default(child_default, current)

    walk_default(default_root, base_defaults)

    geom_node = default_root.find("geom")
    if geom_node is not None:
        base_defaults["geom"].update(geom_node.attrib)

    joint_node = default_root.find("joint")
    if joint_node is not None:
        base_defaults["joint"].update(joint_node.attrib)

    return class_defaults, base_defaults


def _get_effective_attrs(
    elem: ET.Element,
    kind: str,
    active_childclass: Optional[str],
    class_defaults: Dict[str, Dict[str, Dict[str, str]]],
    base_defaults: Dict[str, Dict[str, str]],
) -> Dict[str, str]:
    attrs: Dict[str, str] = dict(base_defaults[kind])

    if active_childclass and active_childclass in class_defaults:
        attrs.update(class_defaults[active_childclass][kind])

    elem_class = elem.attrib.get("class")
    if elem_class and elem_class in class_defaults:
        attrs.update(class_defaults[elem_class][kind])

    attrs.update(elem.attrib)
    return attrs


def _build_mjcf_mesh_assets(root: ET.Element, xml_path: Path) -> Dict[str, Tuple[Path, np.ndarray]]:
    compiler = root.find("compiler")
    meshdir = ""
    if compiler is not None:
        meshdir = compiler.attrib.get("meshdir", "")

    asset = root.find("asset")
    if asset is None:
        return {}

    mesh_map: Dict[str, Tuple[Path, np.ndarray]] = {}
    for mesh in asset.findall("mesh"):
        mesh_name = mesh.attrib.get("name")
        mesh_file = mesh.attrib.get("file")
        if not mesh_name or not mesh_file:
            continue

        scale = _parse_vec(mesh.attrib.get("scale"), 3, [1.0, 1.0, 1.0])
        mesh_path = Path(mesh_file)
        if not mesh_path.is_absolute():
            if meshdir:
                mesh_path = (xml_path.parent / meshdir / mesh_file).resolve()
            else:
                mesh_path = (xml_path.parent / mesh_file).resolve()

        mesh_map[mesh_name] = (mesh_path, scale)

    return mesh_map


def _create_mjcf_primitive_mesh(geom_type: str, attrs: Dict[str, str]) -> Optional[trimesh.Trimesh]:
    if geom_type == "box":
        size = _parse_vec(attrs.get("size"), 3, [0.01, 0.01, 0.01])
        return trimesh.creation.box(extents=2.0 * size)
    if geom_type == "sphere":
        size = _parse_vec(attrs.get("size"), 1, [0.01])
        return trimesh.creation.icosphere(radius=float(size[0]))
    if geom_type == "cylinder":
        size = _parse_vec(attrs.get("size"), 2, [0.01, 0.01])
        radius, half = float(size[0]), float(size[1])
        return trimesh.creation.cylinder(radius=radius, height=2.0 * half)
    if geom_type == "capsule":
        if "fromto" in attrs:
            fromto = _parse_vec(attrs.get("fromto"), 6, [0, 0, 0, 0, 0, 0])
            p0 = fromto[:3]
            p1 = fromto[3:]
            length = float(np.linalg.norm(p1 - p0))
            size = _parse_vec(attrs.get("size"), 1, [0.01])
            radius = float(size[0])
            mesh = trimesh.creation.capsule(radius=radius, height=max(length, 1e-6))

            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            dir_vec = p1 - p0
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-8:
                dir_unit = dir_vec / norm
                axis = np.cross(z_axis, dir_unit)
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-8:
                    axis /= axis_norm
                    angle = math.acos(float(np.clip(np.dot(z_axis, dir_unit), -1.0, 1.0)))
                    rot = Rotation.from_rotvec(axis * angle).as_matrix()
                else:
                    dot = float(np.dot(z_axis, dir_unit))
                    rot = np.eye(3) if dot > 0 else Rotation.from_euler("x", math.pi).as_matrix()
            else:
                rot = np.eye(3)

            tf = np.eye(4)
            tf[:3, :3] = rot
            tf[:3, 3] = 0.5 * (p0 + p1)
            mesh.apply_transform(tf)
            return mesh

        size = _parse_vec(attrs.get("size"), 2, [0.01, 0.01])
        radius, half = float(size[0]), float(size[1])
        return trimesh.creation.capsule(radius=radius, height=2.0 * half)

    return None


def _mjcf_geom_to_trimesh(
    geom_attrs: Dict[str, str],
    mesh_assets: Dict[str, Tuple[Path, np.ndarray]],
    angle_in_radian: bool,
) -> Optional[trimesh.Trimesh]:
    geom_type = geom_attrs.get("type", "sphere")

    if geom_type == "mesh":
        mesh_name = geom_attrs.get("mesh")
        if not mesh_name or mesh_name not in mesh_assets:
            return None
        mesh_path, mesh_scale = mesh_assets[mesh_name]
        mesh = _load_mesh_geometry(mesh_path, mesh_scale)
        if mesh is None:
            return None
    else:
        mesh = _create_mjcf_primitive_mesh(geom_type, geom_attrs)
        if mesh is None:
            return None

    tf = _parse_pose_element(_dict_to_elem(geom_attrs), angle_in_radian=angle_in_radian)
    mesh = mesh.copy()
    mesh.apply_transform(tf)
    return mesh


def _dict_to_elem(attrs: Dict[str, str]) -> ET.Element:
    e = ET.Element("tmp")
    e.attrib.update(attrs)
    return e


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def _matrix_to_xyz_rpy(tf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xyz = tf[:3, 3]
    rpy = Rotation.from_matrix(tf[:3, :3]).as_euler("xyz")
    return xyz, rpy


def _build_pk_and_meshes_from_mjcf(xml_path: Path) -> Tuple[pk.chain.Chain, Dict[str, trimesh.Trimesh]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    angle_mode = "degree"
    if compiler is not None:
        angle_mode = compiler.attrib.get("angle", "degree").lower()
    angle_in_radian = angle_mode == "radian"

    class_defaults, base_defaults = _parse_mjcf_defaults(root)
    mesh_assets = _build_mjcf_mesh_assets(root, xml_path)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF has no <worldbody>: {xml_path}")

    urdf_robot = ET.Element("robot", attrib={"name": root.attrib.get("model", xml_path.stem)})
    ET.SubElement(urdf_robot, "link", attrib={"name": "world"})

    mesh_records: Dict[str, List[trimesh.Trimesh]] = {}
    joint_id = 0

    def add_link(name: str) -> None:
        ET.SubElement(urdf_robot, "link", attrib={"name": name})

    def add_joint(
        name: str,
        parent: str,
        child: str,
        jtype: str,
        origin_tf: np.ndarray,
        axis: Optional[np.ndarray] = None,
        limit: Optional[Tuple[float, float]] = None,
    ) -> None:
        joint_el = ET.SubElement(urdf_robot, "joint", attrib={"name": name, "type": jtype})
        ET.SubElement(joint_el, "parent", attrib={"link": parent})
        ET.SubElement(joint_el, "child", attrib={"link": child})

        xyz, rpy = _matrix_to_xyz_rpy(origin_tf)
        ET.SubElement(
            joint_el,
            "origin",
            attrib={
                "xyz": f"{xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g}",
                "rpy": f"{rpy[0]:.9g} {rpy[1]:.9g} {rpy[2]:.9g}",
            },
        )

        if axis is not None and jtype in ("revolute", "prismatic"):
            ET.SubElement(joint_el, "axis", attrib={"xyz": f"{axis[0]:.9g} {axis[1]:.9g} {axis[2]:.9g}"})

        if jtype in ("revolute", "prismatic"):
            if limit is None:
                limit = (-math.pi, math.pi) if jtype == "revolute" else (-1.0, 1.0)
            lower, upper = float(limit[0]), float(limit[1])
            if lower > upper:
                lower, upper = upper, lower
            ET.SubElement(
                joint_el,
                "limit",
                attrib={
                    "lower": f"{lower:.9g}",
                    "upper": f"{upper:.9g}",
                    "effort": "1000",
                    "velocity": "1000",
                },
            )

    def collect_body_meshes(body: ET.Element, link_name: str, active_childclass: Optional[str]) -> None:
        body_meshes: List[trimesh.Trimesh] = []
        fallback_meshes: List[trimesh.Trimesh] = []

        for geom in body.findall("geom"):
            attrs = _get_effective_attrs(
                geom,
                kind="geom",
                active_childclass=active_childclass,
                class_defaults=class_defaults,
                base_defaults=base_defaults,
            )
            m = _mjcf_geom_to_trimesh(attrs, mesh_assets, angle_in_radian=angle_in_radian)
            if m is None:
                continue

            name_l = attrs.get("name", "").lower()
            is_collision = ("collision" in name_l) or (attrs.get("group") == "3")
            fallback_meshes.append(m)
            if not is_collision:
                body_meshes.append(m)

        use_meshes = body_meshes if body_meshes else fallback_meshes
        if use_meshes:
            mesh_records[link_name] = use_meshes

    def parse_body(body: ET.Element, parent_link: str, inherited_childclass: Optional[str]) -> None:
        nonlocal joint_id

        body_name = _safe_name(body.attrib.get("name", f"body_{joint_id}"))
        body_tf = _parse_pose_element(body, angle_in_radian=angle_in_radian)

        active_childclass = body.attrib.get("childclass", inherited_childclass)

        joints = body.findall("joint")
        freejoint = body.find("freejoint")

        if freejoint is not None:
            free_joint_names = [
                f"{body_name}_free_tx",
                f"{body_name}_free_ty",
                f"{body_name}_free_tz",
                f"{body_name}_free_rx",
                f"{body_name}_free_ry",
                f"{body_name}_free_rz",
            ]
            free_axes = [
                ("prismatic", np.array([1.0, 0.0, 0.0])),
                ("prismatic", np.array([0.0, 1.0, 0.0])),
                ("prismatic", np.array([0.0, 0.0, 1.0])),
                ("revolute", np.array([1.0, 0.0, 0.0])),
                ("revolute", np.array([0.0, 1.0, 0.0])),
                ("revolute", np.array([0.0, 0.0, 1.0])),
            ]

            chain_parent = parent_link
            for idx, (jname, (jtype, axis)) in enumerate(zip(free_joint_names, free_axes)):
                child = body_name if idx == len(free_joint_names) - 1 else f"{body_name}__free_{idx}"
                add_link(child)
                add_joint(
                    name=jname,
                    parent=chain_parent,
                    child=child,
                    jtype=jtype,
                    origin_tf=body_tf if idx == 0 else np.eye(4),
                    axis=axis,
                    limit=(-2.0, 2.0) if jtype == "prismatic" else (-math.pi, math.pi),
                )
                chain_parent = child

            current_link = body_name
        elif not joints:
            add_link(body_name)
            add_joint(
                name=f"{parent_link}_to_{body_name}_fixed",
                parent=parent_link,
                child=body_name,
                jtype="fixed",
                origin_tf=body_tf,
            )
            current_link = body_name
        else:
            supported_joints = []
            for joint in joints:
                attrs = _get_effective_attrs(
                    joint,
                    kind="joint",
                    active_childclass=active_childclass,
                    class_defaults=class_defaults,
                    base_defaults=base_defaults,
                )
                mj_type = attrs.get("type", "hinge")
                if mj_type in ("hinge", "slide"):
                    supported_joints.append((joint, attrs))

            if not supported_joints:
                add_link(body_name)
                add_joint(
                    name=f"{parent_link}_to_{body_name}_fixed",
                    parent=parent_link,
                    child=body_name,
                    jtype="fixed",
                    origin_tf=body_tf,
                )
                current_link = body_name
            else:
                chain_parent = parent_link
                for idx, (_, attrs) in enumerate(supported_joints):
                    mj_type = attrs.get("type", "hinge")
                    jtype = "revolute" if mj_type == "hinge" else "prismatic"
                    jname = _safe_name(attrs.get("name", f"joint_{joint_id}"))
                    joint_id += 1

                    child = body_name if idx == len(supported_joints) - 1 else f"{body_name}__joint_{idx}"
                    add_link(child)

                    axis = _parse_vec(attrs.get("axis"), 3, [0.0, 0.0, 1.0])
                    axis_norm = np.linalg.norm(axis)
                    if axis_norm > 1e-8:
                        axis = axis / axis_norm

                    limit = None
                    if "range" in attrs:
                        rg = _parse_vec(attrs.get("range"), 2, [0.0, 0.0])
                        limit = (float(rg[0]), float(rg[1]))
                        if not angle_in_radian and jtype == "revolute":
                            limit = (math.radians(limit[0]), math.radians(limit[1]))

                    add_joint(
                        name=jname,
                        parent=chain_parent,
                        child=child,
                        jtype=jtype,
                        origin_tf=body_tf if idx == 0 else np.eye(4),
                        axis=axis,
                        limit=limit,
                    )
                    chain_parent = child

                current_link = body_name

        collect_body_meshes(body, current_link, active_childclass)

        for child_body in body.findall("body"):
            parse_body(child_body, current_link, active_childclass)

    top_bodies = worldbody.findall("body")
    if not top_bodies:
        raise ValueError(f"MJCF has no body in worldbody: {xml_path}")

    for top_body in top_bodies:
        parse_body(top_body, "world", None)

    urdf_text = ET.tostring(urdf_robot, encoding="unicode")
    chain = _build_chain_quiet(urdf_text)

    mesh_dict: Dict[str, trimesh.Trimesh] = {}
    link_names = set(chain.get_link_names())
    for link_name, meshes in mesh_records.items():
        if link_name not in link_names or not meshes:
            continue
        if len(meshes) == 1:
            mesh_dict[link_name] = meshes[0]
        else:
            mesh_dict[link_name] = trimesh.util.concatenate(meshes)

    return chain, mesh_dict


def discover_robot_assets(assets_dir: Path = DEFAULT_ASSETS_DIR) -> Dict[str, Dict[str, str]]:
    assets_dir = Path(assets_dir)
    all_assets: Dict[str, Dict[str, str]] = {}

    dro_root = assets_dir / "dro_urdf_robot"
    spider_root = assets_dir / "spider_robots"

    if dro_root.exists():
        for urdf in sorted(dro_root.rglob("*.urdf")):
            if urdf.stem.endswith("_nobase"):
                continue
            rel = urdf.relative_to(assets_dir)
            folder = urdf.parent.name
            key = f"dro/{folder}/{urdf.stem}"
            all_assets[key] = {
                "name": key,
                "format": "urdf",
                "path": str(rel),
                "source": "dro",
            }

    if spider_root.exists():
        for urdf in sorted(spider_root.rglob("*.urdf")):
            if urdf.stem.endswith("_nobase"):
                continue
            rel = urdf.relative_to(assets_dir)
            folder = urdf.parent.name
            key = f"spider/{folder}/{urdf.stem}"
            all_assets[key] = {
                "name": key,
                "format": "urdf",
                "path": str(rel),
                "source": "spider",
            }

        for xml in sorted(spider_root.rglob("*.xml")):
            if xml.name == "bimanual.xml":
                continue
            if xml.stem.endswith("_nobase"):
                continue
            try:
                root = ET.parse(xml).getroot()
                worldbody = root.find("worldbody")
                if worldbody is None or worldbody.find("body") is None:
                    continue
            except Exception:
                continue

            rel = xml.relative_to(assets_dir)
            folder = xml.parent.name
            key = f"spider/{folder}/{xml.stem}"
            all_assets[key] = {
                "name": key,
                "format": "xml",
                "path": str(rel),
                "source": "spider",
            }

    return dict(sorted(all_assets.items(), key=lambda kv: kv[0]))


def _is_humanoid_asset_name(robot_name: str) -> bool:
    name = robot_name.lower()
    keywords = ["unitree_g1", "unitree_h1", "h1_2", "booster_t1", "fourier_n1", "humanoid"]
    return any(k in name for k in keywords)


def _robot_kind_key(robot_name: str) -> str:
    parts = robot_name.split("/")
    if len(parts) >= 2:
        return parts[1].lower()
    return robot_name.lower()


def _robot_side(robot_name: str) -> str:
    name = robot_name.lower()
    right_markers = ["_right", "/right", " right", "right_", "_r_", "_r", "r_"]
    left_markers = ["_left", "/left", " left", "left_", "_l_", "_l", "l_"]

    if any(m in name for m in right_markers):
        return "right"
    if any(m in name for m in left_markers):
        return "left"
    return "neutral"


def _base_removal_targets(robot_name: str) -> Optional[Dict[str, List[str]]]:
    name = robot_name.lower()
    if "ezgripper" in name:
        return {"urdf_links": ["base_link"], "xml_bodies": ["base_link", "base"]}
    if "shadow" in name:
        return {"urdf_links": ["forearm"], "xml_bodies": ["forearm"]}
    return None


def _recompose_pose(parent_tf: np.ndarray, child_elem: ET.Element, angle_in_radian: bool) -> None:
    child_tf = _parse_pose_element(child_elem, angle_in_radian=angle_in_radian)
    new_tf = parent_tf @ child_tf

    pos = new_tf[:3, 3]
    quat_xyzw = Rotation.from_matrix(new_tf[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)

    child_elem.attrib["pos"] = f"{pos[0]:.9g} {pos[1]:.9g} {pos[2]:.9g}"
    child_elem.attrib["quat"] = f"{quat_wxyz[0]:.9g} {quat_wxyz[1]:.9g} {quat_wxyz[2]:.9g} {quat_wxyz[3]:.9g}"
    child_elem.attrib.pop("euler", None)
    child_elem.attrib.pop("axisangle", None)


def ensure_base_removed_asset(
    robot_name: str,
    robot_path: Path,
    force_override: bool = False,
) -> Path:
    """
    Create and reuse base-removed urdf/xml assets for selected hands.

    Supported targets:
    - ezgripper: remove `base_link`
    - shadow hand: remove `forearm`

    If generated file exists, it is reused unless `force_override=True`.
    """
    robot_path = Path(robot_path)
    targets = _base_removal_targets(robot_name)
    if targets is None:
        return robot_path

    out_path = robot_path.with_name(f"{robot_path.stem}_nobase{robot_path.suffix}")
    if out_path.exists() and (not force_override):
        return out_path

    suffix = robot_path.suffix.lower()
    if suffix == ".urdf":
        remove_links = set(targets["urdf_links"])
        tree = ET.parse(robot_path)
        root = tree.getroot()

        def _joint_parent_child(joint_el: ET.Element) -> Tuple[Optional[str], Optional[str]]:
            parent_el = joint_el.find("parent")
            child_el = joint_el.find("child")
            parent = parent_el.attrib.get("link") if parent_el is not None else None
            child = child_el.attrib.get("link") if child_el is not None else None
            return parent, child

        # Rewire children of removed links to removed-link parent and drop incoming joints.
        changed = True
        while changed:
            changed = False
            for joint_el in list(root.findall("joint")):
                parent, child = _joint_parent_child(joint_el)
                if child not in remove_links:
                    continue

                for other in list(root.findall("joint")):
                    if other is joint_el:
                        continue
                    p2_el = other.find("parent")
                    if p2_el is not None and p2_el.attrib.get("link") == child and parent is not None:
                        p2_el.attrib["link"] = parent

                root.remove(joint_el)
                changed = True

        # Remove joints that still reference removed links.
        for joint_el in list(root.findall("joint")):
            parent, child = _joint_parent_child(joint_el)
            if parent in remove_links or child in remove_links:
                root.remove(joint_el)

        # Remove target links.
        for link_el in list(root.findall("link")):
            if link_el.attrib.get("name") in remove_links:
                root.remove(link_el)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out_path, encoding="utf-8", xml_declaration=True)
        return out_path

    if suffix == ".xml":
        remove_bodies = {_safe_name(x) for x in targets["xml_bodies"]}
        tree = ET.parse(robot_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            return robot_path

        compiler = root.find("compiler")
        angle_mode = compiler.attrib.get("angle", "degree").lower() if compiler is not None else "degree"
        angle_in_radian = angle_mode == "radian"

        def process_parent(parent_elem: ET.Element) -> None:
            for body in list(parent_elem.findall("body")):
                process_parent(body)
                body_name = _safe_name(body.attrib.get("name", ""))
                if body_name not in remove_bodies:
                    continue

                body_tf = _parse_pose_element(body, angle_in_radian=angle_in_radian)
                for child_body in list(body.findall("body")):
                    _recompose_pose(body_tf, child_body, angle_in_radian=angle_in_radian)
                    parent_elem.append(child_body)
                    body.remove(child_body)

                parent_elem.remove(body)

        process_parent(worldbody)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out_path, encoding="utf-8", xml_declaration=True)
        return out_path

    return robot_path


def _candidate_priority(robot_name: str, meta: Dict[str, Dict[str, str]]) -> Tuple[int, int, int]:
    item = meta[robot_name]
    fmt = item.get("format", "").lower()
    src = item.get("source", "").lower()
    fmt_score = 1 if fmt == "xml" else 0
    src_score = 1 if src == "spider" else 0
    return (fmt_score, src_score, -len(robot_name))


def _pick_best(candidates: List[str], meta: Dict[str, Dict[str, str]]) -> Optional[str]:
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda x: (_candidate_priority(x, meta), x), reverse=True)
    return candidates[0]


def _build_link_tree_from_urdf(urdf_path: Path, link_names: List[str]) -> Tuple[Dict[str, Optional[str]], Dict[str, List[str]]]:
    link_set = set(link_names)
    parent_map: Dict[str, Optional[str]] = {name: None for name in link_names}
    children_map: Dict[str, List[str]] = {name: [] for name in link_names}

    root = ET.parse(urdf_path).getroot()
    for joint_el in root.findall("joint"):
        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        if parent_el is None or child_el is None:
            continue
        p = parent_el.attrib.get("link")
        c = child_el.attrib.get("link")
        if c in link_set:
            parent_map[c] = p if p in link_set else None
        if p in link_set and c in link_set:
            children_map[p].append(c)

    return parent_map, children_map


def _build_link_tree_from_mjcf(xml_path: Path, link_names: List[str]) -> Tuple[Dict[str, Optional[str]], Dict[str, List[str]]]:
    link_set = set(link_names)
    parent_map: Dict[str, Optional[str]] = {name: None for name in link_names}
    children_map: Dict[str, List[str]] = {name: [] for name in link_names}

    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        return parent_map, children_map

    def rec(parent_body: ET.Element, parent_name: Optional[str]) -> None:
        for body in parent_body.findall("body"):
            name = _safe_name(body.attrib.get("name", ""))
            if name in link_set:
                if parent_name in link_set:
                    parent_map[name] = parent_name
                    children_map[parent_name].append(name)
                else:
                    parent_map[name] = None
                rec(body, name)
            else:
                rec(body, parent_name)

    rec(worldbody, None)
    return parent_map, children_map


def build_manipulator_robot_lists(
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    out_path: Path = DEFAULT_MANIPULATOR_LISTS_JSON,
) -> Dict[str, List[str]]:
    """
    Build two manipulator robot lists:
    - both_hands: one urdf/xml per manipulator kind, include both left and right when both exist.
    - right_hands: one urdf/xml per manipulator kind, prefer right; fallback to neutral/left if needed.
    """
    meta = discover_robot_assets(assets_dir)
    manipulators = [name for name in meta.keys() if not _is_humanoid_asset_name(name)]

    by_kind: Dict[str, List[str]] = {}
    for name in manipulators:
        kind = _robot_kind_key(name)
        by_kind.setdefault(kind, []).append(name)

    list_both: List[str] = []
    list_right: List[str] = []

    for kind in sorted(by_kind.keys()):
        names = by_kind[kind]
        lefts = [n for n in names if _robot_side(n) == "left"]
        rights = [n for n in names if _robot_side(n) == "right"]
        neutrals = [n for n in names if _robot_side(n) == "neutral"]

        left_pick = _pick_best(lefts, meta)
        right_pick = _pick_best(rights, meta)
        neutral_pick = _pick_best(neutrals, meta)

        # List 1: include both hands when both are available.
        if left_pick and right_pick:
            list_both.extend([left_pick, right_pick])
        elif right_pick:
            list_both.append(right_pick)
        elif left_pick:
            list_both.append(left_pick)
        elif neutral_pick:
            list_both.append(neutral_pick)

        # List 2: prefer right hand only for handed robots.
        if right_pick:
            list_right.append(right_pick)
        elif neutral_pick:
            list_right.append(neutral_pick)
        elif left_pick:
            list_right.append(left_pick)

    # Remove duplicates while preserving order.
    def uniq(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for item in seq:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    result = {
        "both_hands": uniq(list_both),
        "right_hands": uniq(list_right),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


class RobotModel:
    def __init__(
        self,
        robot_name: str,
        robot_path: Path,
        device: torch.device,
        num_points: int = 512,
    ) -> None:
        self.robot_name = robot_name
        self.robot_path = Path(robot_path)
        self.device = device

        if self.robot_path.suffix.lower() == ".urdf":
            chain = _build_chain_quiet(self.robot_path.read_bytes())
            chain = chain.to(dtype=torch.float32, device=device)
            meshes = _load_urdf_link_geometries(self.robot_path, chain.get_link_names())
        elif self.robot_path.suffix.lower() == ".xml":
            chain, meshes = _build_pk_and_meshes_from_mjcf(self.robot_path)
            chain = chain.to(dtype=torch.float32, device=device)
            meshes = {k: v for k, v in meshes.items()}
        else:
            raise ValueError(f"Unsupported robot file format: {self.robot_path}")

        self.pk_chain = chain
        self.dof = len(self.pk_chain.get_joint_parameter_names())
        self.joint_orders = [joint.name for joint in self.pk_chain.get_joints()]
        self.base_pose_indices = self._infer_base_pose_indices()
        self.base_translation_indices = self._infer_base_translation_indices()
        # NOTE: num_points now means whole-hand surface point count.
        self.link_num_points = int(num_points)
        self.surface_num_points = int(num_points)

        # Keep original-link assets and box-link assets separately.
        self.meshes_original = {k: v.copy() for k, v in meshes.items()}
        self.vertices_original = self._sample_points_from_meshes(self.meshes_original)
        mesh_link_names = list(self.meshes_original.keys())
        self.mesh_link_names = list(mesh_link_names)
        if self.robot_path.suffix.lower() == ".urdf":
            self.link_parent_map, self.link_children_map = _build_link_tree_from_urdf(self.robot_path, mesh_link_names)
        elif self.robot_path.suffix.lower() == ".xml":
            self.link_parent_map, self.link_children_map = _build_link_tree_from_mjcf(self.robot_path, mesh_link_names)
        else:
            self.link_parent_map = {name: None for name in mesh_link_names}
            self.link_children_map = {name: [] for name in mesh_link_names}

        self.meshes_box: Dict[str, trimesh.Trimesh] = {}
        self.vertices_box: Dict[str, torch.Tensor] = {}
        self.box_fit_info: Dict[str, Dict[str, object]] = {}
        self.box_face_centers_local: Dict[str, torch.Tensor] = {}
        self.box_face_normals_local: Dict[str, torch.Tensor] = {}

        self.link_mode = "original"
        self.meshes = self.meshes_original
        self.vertices = self.vertices_original
        self.box_fits_default_dir = DEFAULT_ASSETS_DIR / "box_fits"

        # Whole-hand surface template (local frame) and graph.
        self.surface_template_by_link: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.surface_template_points_local = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        self.surface_template_normals_local = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        self.surface_template_link_indices = torch.zeros((0,), dtype=torch.long, device=self.device)
        self.surface_template_points_canonical_world = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        self.surface_template_normals_canonical_world = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        self.surface_graph_points_canonical_world = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        self.surface_graph_normals_canonical_world = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        self.surface_union_mesh_canonical: Optional[trimesh.Trimesh] = None
        self.surface_graph_neighbors_strict: List[List[int]] = []
        self.surface_graph_neighbors: List[List[int]] = []

        # Contact-mask sampler fitted state.
        self.contact_mask_sampler_state: Optional[Dict[str, torch.Tensor]] = None

        self.frame_status = None
        self.surface_template_seed = _stable_int_seed(
            f"{self.robot_name}|{self.surface_num_points}|surface_template_v2"
        )
        self.surface_artifact_path: Optional[str] = None
        self.surface_artifact_built: bool = False
        with _temporary_global_seed(self.surface_template_seed):
            self._load_or_build_surface_assets(num_points=self.surface_num_points)
        self.joint_descendant_link_mask = self._build_joint_descendant_link_mask()

    def _allocate_points_over_meshes(
        self,
        mesh_dict: Dict[str, trimesh.Trimesh],
        total_points: int,
    ) -> Dict[str, int]:
        if total_points <= 0:
            raise ValueError("total_points must be positive")
        link_names = list(mesh_dict.keys())
        if len(link_names) == 0:
            return {}

        areas = np.array([float(max(mesh_dict[n].area, 1e-8)) for n in link_names], dtype=np.float64)
        areas = areas / np.clip(areas.sum(), 1e-8, None)
        raw = areas * float(total_points)
        counts = np.floor(raw).astype(np.int64)
        remain = int(total_points - counts.sum())
        if remain > 0:
            frac = raw - counts
            order = np.argsort(-frac)
            for i in range(remain):
                counts[order[i % len(order)]] += 1

        return {name: int(count) for name, count in zip(link_names, counts)}

    def _surface_artifact_cache_path(self, num_points: int) -> Path:
        try:
            from .robot_model_process import get_surface_artifact_path
        except Exception:
            from robot_model_process import get_surface_artifact_path  # type: ignore

        return get_surface_artifact_path(
            robot_path=self.robot_path,
            robot_name=self.robot_name,
            num_points=int(num_points),
        )

    def _load_or_build_surface_assets(self, num_points: int) -> None:
        try:
            from .robot_model_process import apply_surface_artifact_to_model, load_or_build_surface_artifact
        except Exception:
            from robot_model_process import apply_surface_artifact_to_model, load_or_build_surface_artifact  # type: ignore

        artifact, cache_path, built = load_or_build_surface_artifact(
            model=self,
            num_points=int(num_points),
            seed=int(self.surface_template_seed),
            cache_path=self._surface_artifact_cache_path(num_points=int(num_points)),
        )
        apply_surface_artifact_to_model(self, artifact)
        self.surface_artifact_path = str(cache_path)
        self.surface_artifact_built = bool(built)

    def _sample_mesh_points_normals(
        self,
        mesh: trimesh.Trimesh,
        count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if count <= 0:
            empty = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            return empty, empty

        try:
            pts_np, face_idx = mesh.sample(int(count), return_index=True)
            nrm_np = mesh.face_normals[face_idx]
            pts = torch.tensor(pts_np, dtype=torch.float32, device=self.device)
            nrms = torch.tensor(nrm_np, dtype=torch.float32, device=self.device)
        except Exception:
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            if verts.shape[0] == 0:
                empty = torch.zeros((count, 3), dtype=torch.float32, device=self.device)
                return empty, empty
            idx = np.random.randint(0, verts.shape[0], size=(count,))
            pts = torch.tensor(verts[idx], dtype=torch.float32, device=self.device)
            nrms = torch.zeros_like(pts)

        nrms = nrms / torch.norm(nrms, dim=1, keepdim=True).clamp_min(1e-8)
        return pts, nrms

    def _get_canonical_link_meshes_world(
        self,
    ) -> Tuple[List[str], Dict[str, trimesh.Trimesh]]:
        q_ref = self._zero_base_translation_in_q(self.get_canonical_q().detach().clone())
        self.update_status(q_ref)
        link_names = list(self.meshes_original.keys())
        link_meshes_world: Dict[str, trimesh.Trimesh] = {}
        for link_name in link_names:
            if link_name not in self.frame_status:
                continue
            mesh_local = self.meshes_original.get(link_name, None)
            if mesh_local is None or len(mesh_local.faces) == 0:
                continue
            tf = self.frame_status[link_name].get_matrix()[0].detach().cpu().numpy()
            mesh_world = mesh_local.copy()
            mesh_world.apply_transform(tf)
            if len(mesh_world.faces) == 0:
                continue
            link_meshes_world[link_name] = mesh_world
        if len(link_meshes_world) == 0:
            raise RuntimeError(f"Failed to build canonical meshes for robot: {self.robot_name}")
        return link_names, link_meshes_world

    def _sample_points_from_meshes(self, mesh_dict: Dict[str, trimesh.Trimesh]) -> Dict[str, torch.Tensor]:
        counts = self._allocate_points_over_meshes(mesh_dict, total_points=self.surface_num_points)
        vertices: Dict[str, torch.Tensor] = {}
        for link_name, link_mesh in mesh_dict.items():
            c = int(counts.get(link_name, 0))
            if c <= 0:
                vertices[link_name] = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
                continue
            pts, _ = self._sample_mesh_points_normals(link_mesh, c)
            vertices[link_name] = pts
        return vertices

    def get_surface_template(self) -> Dict[str, torch.Tensor]:
        return {
            "points_local": self.surface_template_points_local.detach().clone(),
            "normals_local": self.surface_template_normals_local.detach().clone(),
            "link_indices": self.surface_template_link_indices.detach().clone(),
        }

    def get_surface_template_hash(self) -> str:
        h = hashlib.sha1()
        pts = np.ascontiguousarray(self.surface_template_points_local.detach().cpu().numpy().astype(np.float32))
        nrms = np.ascontiguousarray(self.surface_template_normals_local.detach().cpu().numpy().astype(np.float32))
        idx = np.ascontiguousarray(self.surface_template_link_indices.detach().cpu().numpy().astype(np.int64))
        h.update(pts.tobytes())
        h.update(nrms.tobytes())
        h.update(idx.tobytes())
        return h.hexdigest()

    def _set_active_mode_assets(self, mode: str) -> None:
        if mode == "original":
            self.link_mode = "original"
            self.meshes = self.meshes_original
            self.vertices = self.vertices_original
            return
        if mode == "box":
            self.link_mode = "box"
            self.meshes = self.meshes_box
            self.vertices = self.vertices_box
            return
        raise ValueError(f"Unknown mode: {mode}")

    def _resolve_mode_assets(self, mode: str = "active") -> Tuple[Dict[str, trimesh.Trimesh], Dict[str, torch.Tensor]]:
        if mode == "active":
            return self.meshes, self.vertices
        if mode == "original":
            return self.meshes_original, self.vertices_original
        if mode == "box":
            if not self.meshes_box:
                self.switch_to_box_links_only_mode(override=False)
            return self.meshes_box, self.vertices_box
        raise ValueError(f"Unsupported mode: {mode}")

    def _box_cache_file(self, cache_dir: Optional[Path] = None) -> Path:
        cache_root = Path(cache_dir) if cache_dir is not None else self.box_fits_default_dir
        safe_name = self.robot_name.replace("/", "__")
        return cache_root / f"{safe_name}.json"

    def _estimate_mesh_iou(self, mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, num_samples: int = 20000) -> float:
        bounds = np.vstack([mesh_a.bounds, mesh_b.bounds])
        low = bounds.min(axis=0)
        high = bounds.max(axis=0)
        if np.any(high <= low + 1e-9):
            return 0.0

        try:
            rng = np.random.default_rng(0)
            points = rng.uniform(low=low, high=high, size=(int(num_samples), 3))
            in_a = mesh_a.contains(points)
            in_b = mesh_b.contains(points)
            union = np.logical_or(in_a, in_b).sum()
            if union == 0:
                return 0.0
            inter = np.logical_and(in_a, in_b).sum()
            return float(inter / union)
        except Exception:
            # Fallback: rough AABB-based IoU using mesh volumes.
            try:
                a0, a1 = mesh_a.bounds
                b0, b1 = mesh_b.bounds
                inter_extent = np.clip(np.minimum(a1, b1) - np.maximum(a0, b0), 0.0, None)
                inter_vol = float(np.prod(inter_extent))
                va = float(max(mesh_a.volume, 0.0))
                vb = float(max(mesh_b.volume, 0.0))
                denom = va + vb - inter_vol
                return float(inter_vol / denom) if denom > 1e-9 else 0.0
            except Exception:
                return 0.0

    def _fit_box_for_mesh(self, mesh: trimesh.Trimesh, iou_samples: int = 20000) -> Dict[str, object]:
        obb = mesh.bounding_box_oriented
        transform = np.asarray(obb.primitive.transform, dtype=np.float64)
        extents = np.asarray(obb.primitive.extents, dtype=np.float64)

        box_mesh = trimesh.creation.box(extents=extents)
        box_mesh.apply_transform(transform)
        iou = self._estimate_mesh_iou(mesh, box_mesh, num_samples=iou_samples)

        return {
            "transform": transform.tolist(),
            "extents": extents.tolist(),
            "iou": float(iou),
        }

    def _build_box_meshes_from_payload(self, payload: Dict[str, object], iou_samples: int = 20000) -> Tuple[Dict[str, trimesh.Trimesh], Dict[str, Dict[str, object]]]:
        links_payload = payload.get("links", {})
        if not isinstance(links_payload, dict):
            links_payload = {}

        meshes_box: Dict[str, trimesh.Trimesh] = {}
        fit_info: Dict[str, Dict[str, object]] = {}

        for link_name, original_mesh in self.meshes_original.items():
            fit = links_payload.get(link_name)
            if not isinstance(fit, dict) or ("transform" not in fit) or ("extents" not in fit):
                fit = self._fit_box_for_mesh(original_mesh, iou_samples=iou_samples)

            transform = np.asarray(fit["transform"], dtype=np.float64)
            extents = np.asarray(fit["extents"], dtype=np.float64)
            if transform.shape != (4, 4) or extents.shape != (3,):
                fit = self._fit_box_for_mesh(original_mesh, iou_samples=iou_samples)
                transform = np.asarray(fit["transform"], dtype=np.float64)
                extents = np.asarray(fit["extents"], dtype=np.float64)

            box_mesh = trimesh.creation.box(extents=extents)
            box_mesh.apply_transform(transform)
            meshes_box[link_name] = box_mesh
            fit_info[link_name] = {
                "transform": transform.tolist(),
                "extents": extents.tolist(),
                "iou": float(fit.get("iou", 0.0)),
            }

        return meshes_box, fit_info

    def _prepare_box_face_samples(self) -> None:
        self.box_face_centers_local = {}
        self.box_face_normals_local = {}

        # Axis-aligned box faces in local canonical box frame.
        base_normals = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )

        for link_name, fit in self.box_fit_info.items():
            extents = np.asarray(fit["extents"], dtype=np.float64)
            transform = np.asarray(fit["transform"], dtype=np.float64)
            half = 0.5 * extents

            base_centers = np.array(
                [
                    [half[0], 0.0, 0.0],
                    [-half[0], 0.0, 0.0],
                    [0.0, half[1], 0.0],
                    [0.0, -half[1], 0.0],
                    [0.0, 0.0, half[2]],
                    [0.0, 0.0, -half[2]],
                ],
                dtype=np.float64,
            )

            rot = transform[:3, :3]
            trans = transform[:3, 3]

            centers_local = (base_centers @ rot.T) + trans[None, :]
            normals_local = base_normals @ rot.T
            norms = np.linalg.norm(normals_local, axis=1, keepdims=True)
            normals_local = normals_local / np.clip(norms, 1e-8, None)

            self.box_face_centers_local[link_name] = torch.tensor(
                centers_local, dtype=torch.float32, device=self.device
            )
            self.box_face_normals_local[link_name] = torch.tensor(
                normals_local, dtype=torch.float32, device=self.device
            )

    def switch_to_box_links_only_mode(
        self,
        override: bool = False,
        cache_dir: Optional[Path] = None,
        iou_samples: int = 20000,
    ) -> Dict[str, float]:
        """
        Switch active assets to box-links-only mode.

        If cached fitted boxes exist, load them unless ``override=True``.
        """
        cache_file = self._box_cache_file(cache_dir)
        payload: Optional[Dict[str, object]] = None

        if cache_file.exists() and not override:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = None

        if payload is None:
            payload = {
                "robot_name": self.robot_name,
                "robot_path": str(self.robot_path),
                "links": {},
            }
            links_payload: Dict[str, Dict[str, object]] = {}
            for link_name, mesh in self.meshes_original.items():
                links_payload[link_name] = self._fit_box_for_mesh(mesh, iou_samples=iou_samples)
            payload["links"] = links_payload

            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

        self.meshes_box, self.box_fit_info = self._build_box_meshes_from_payload(payload, iou_samples=iou_samples)
        self.vertices_box = self._sample_points_from_meshes(self.meshes_box)
        self._prepare_box_face_samples()
        self._set_active_mode_assets("box")

        return {k: float(v.get("iou", 0.0)) for k, v in self.box_fit_info.items()}

    def switch_to_box_links_mode(
        self,
        override: bool = False,
        cache_dir: Optional[Path] = None,
        iou_samples: int = 20000,
    ) -> Dict[str, float]:
        return self.switch_to_box_links_only_mode(override=override, cache_dir=cache_dir, iou_samples=iou_samples)

    def switch_to_original_links_mode(self) -> None:
        self._set_active_mode_assets("original")

    def get_box_face_points_normals(self, q: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return box-surface centers and normals in world frame.

        Returns:
            points_world: (num_links, 6, 3)
            normals_world: (num_links, 6, 3)
        """
        if self.link_mode != "box":
            raise RuntimeError("get_box_face_points_normals requires box_links_mode. Call switch_to_box_links_only_mode first.")

        if not self.box_face_centers_local or not self.box_face_normals_local:
            self._prepare_box_face_samples()

        if q is None:
            q = torch.zeros(self.dof, dtype=torch.float32, device=self.device)
        self.update_status(q)

        points_all: List[torch.Tensor] = []
        normals_all: List[torch.Tensor] = []
        for link_name in self.meshes_box.keys():
            if link_name not in self.frame_status:
                continue
            if link_name not in self.box_face_centers_local:
                continue

            centers_local = self.box_face_centers_local[link_name]  # (6,3)
            normals_local = self.box_face_normals_local[link_name]  # (6,3)
            tf = self.frame_status[link_name].get_matrix()[0].to(self.device)  # (4,4)
            rot = tf[:3, :3]

            centers_h = torch.cat(
                [centers_local, torch.ones((centers_local.shape[0], 1), device=self.device)], dim=1
            )  # (6,4)
            centers_world = (centers_h @ tf.T)[:, :3]
            normals_world = normals_local @ rot.T
            normals_world = normals_world / torch.norm(normals_world, dim=1, keepdim=True).clamp_min(1e-8)

            # Rule-based removal:
            # 1) if exactly one child link -> remove face toward child.
            # 2) if has parent link -> remove face toward parent.
            remove_indices: set[int] = set()
            link_pos = tf[:3, 3]

            child_candidates = [
                c for c in self.link_children_map.get(link_name, [])
                if c in self.frame_status and c in self.meshes_box
            ]
            if len(child_candidates) == 1:
                child_tf = self.frame_status[child_candidates[0]].get_matrix()[0].to(self.device)
                vec = child_tf[:3, 3] - link_pos
                if torch.norm(vec) > 1e-8:
                    dir_unit = vec / torch.norm(vec)
                    dots = normals_world @ dir_unit
                    remove_indices.add(int(torch.argmax(dots).item()))

            parent_name = self.link_parent_map.get(link_name, None)
            if (parent_name is not None) and (parent_name in self.frame_status) and (parent_name in self.meshes_box):
                parent_tf = self.frame_status[parent_name].get_matrix()[0].to(self.device)
                vec = parent_tf[:3, 3] - link_pos
                if torch.norm(vec) > 1e-8:
                    dir_unit = vec / torch.norm(vec)
                    dots = normals_world @ dir_unit
                    remove_indices.add(int(torch.argmax(dots).item()))

            # Keep fixed output shape: mark removed samples as NaN.
            if remove_indices:
                ridx = torch.tensor(sorted(remove_indices), dtype=torch.long, device=self.device)
                centers_world[ridx] = float("nan")
                normals_world[ridx] = float("nan")

            points_all.append(centers_world)
            normals_all.append(normals_world)

        if not points_all:
            empty = torch.zeros((0, 6, 3), dtype=torch.float32, device=self.device)
            return empty, empty
        return torch.stack(points_all, dim=0), torch.stack(normals_all, dim=0)

    def get_joint_orders(self) -> List[str]:
        return list(self.joint_orders)

    def _infer_base_pose_indices(self) -> List[int]:
        idx: List[int] = []
        for i, name in enumerate(self.joint_orders):
            n = name.lower().replace("-", "_").replace(" ", "")
            has_base_token = any(t in n for t in ("root", "base", "world", "global", "floating", "trans", "virtual"))
            if not has_base_token:
                continue
            if any(
                t in n
                for t in (
                    "roll",
                    "pitch",
                    "yaw",
                    "rot",
                    "_rx",
                    "_ry",
                    "_rz",
                    "_x",
                    "_y",
                    "_z",
                    "x",
                    "y",
                    "z",
                )
            ):
                idx.append(i)
        return sorted(set(idx))

    def _build_joint_descendant_link_mask(self) -> torch.Tensor:
        link_to_idx = {name: i for i, name in enumerate(self.mesh_link_names)}
        joint_to_links: Dict[str, List[str]] = {}
        for joint, child_links in self.pk_chain.get_joints_and_child_links():
            joint_name = str(joint.name)
            descendants: List[str] = []
            for link in child_links:
                link_name = getattr(link, "name", None)
                if link_name in link_to_idx:
                    descendants.append(str(link_name))
            joint_to_links[joint_name] = descendants

        mask = torch.zeros(
            (self.dof, len(self.mesh_link_names)),
            dtype=torch.bool,
            device=self.device,
        )
        for joint_idx, joint_name in enumerate(self.joint_orders):
            for link_name in joint_to_links.get(joint_name, []):
                mask[joint_idx, link_to_idx[link_name]] = True
        if len(self.base_pose_indices) > 0 and len(self.mesh_link_names) > 0:
            mask[self.base_pose_indices] = True
        return mask

    def _infer_base_translation_indices(self) -> List[int]:
        idx: List[int] = []
        direct = {
            "rootx",
            "rooty",
            "rootz",
            "root_x",
            "root_y",
            "root_z",
            "base_x",
            "base_y",
            "base_z",
            "world_x",
            "world_y",
            "world_z",
            "global_x",
            "global_y",
            "global_z",
            "trans_x",
            "trans_y",
            "trans_z",
            "translation_x",
            "translation_y",
            "translation_z",
            "floating_x",
            "floating_y",
            "floating_z",
            "virtual_joint_x",
            "virtual_joint_y",
            "virtual_joint_z",
            "virtual_x",
            "virtual_y",
            "virtual_z",
        }
        for i, name in enumerate(self.joint_orders):
            n = name.lower().replace("-", "_").replace(" ", "")
            if n in direct:
                idx.append(i)
                continue
            has_base_token = any(t in n for t in ("root", "base", "world", "global", "floating", "trans", "virtual"))
            if not has_base_token:
                continue
            axis_suffix = n.endswith("_x") or n.endswith("_y") or n.endswith("_z") or n.endswith("x") or n.endswith("y") or n.endswith("z")
            if not axis_suffix:
                continue
            if any(t in n for t in ("roll", "pitch", "yaw", "rot", "_rx", "_ry", "_rz")):
                continue
            idx.append(i)
        return sorted(set(idx))

    def _zero_base_translation_in_q(self, q: torch.Tensor) -> torch.Tensor:
        if len(self.base_translation_indices) == 0:
            return q
        q_out = q.clone()
        if q_out.ndim == 1:
            q_out[self.base_translation_indices] = 0.0
        else:
            q_out[..., self.base_translation_indices] = 0.0
        return q_out

    def update_status(self, q: torch.Tensor) -> None:
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if q.shape[-1] == self.dof + 3:
            q = q_rot6d_to_q_euler(q)
        if q.shape[-1] != self.dof:
            raise ValueError(f"Expected joint dim {self.dof}, got {q.shape[-1]}")
        self.frame_status = self.pk_chain.forward_kinematics(q.to(self.device))

    def get_transformed_links_pc(
        self,
        q: Optional[torch.Tensor] = None,
        links_pc: Optional[Dict[str, torch.Tensor]] = None,
        mode: str = "active",
    ) -> torch.Tensor:
        if links_pc is None:
            _, mode_vertices = self._resolve_mode_assets(mode)
            links_pc = mode_vertices

        if q is None:
            q = torch.zeros(self.dof, dtype=torch.float32, device=self.device)

        self.update_status(q)

        all_pc_se3 = []
        for link_index, (link_name, link_pc) in enumerate(links_pc.items()):
            if link_name not in self.frame_status:
                continue
            if not torch.is_tensor(link_pc):
                link_pc = torch.tensor(link_pc, dtype=torch.float32, device=self.device)
            n_link = link_pc.shape[0]
            se3 = self.frame_status[link_name].get_matrix()[0].to(self.device)
            homogeneous_tensor = torch.ones(n_link, 1, device=self.device)
            link_pc_homogeneous = torch.cat([link_pc.to(self.device), homogeneous_tensor], dim=1)
            link_pc_se3 = (link_pc_homogeneous @ se3.T)[:, :3]
            index_tensor = torch.full([n_link, 1], float(link_index), device=self.device)
            link_pc_se3_index = torch.cat([link_pc_se3, index_tensor], dim=1)
            all_pc_se3.append(link_pc_se3_index)

        if not all_pc_se3:
            return torch.zeros((0, 4), dtype=torch.float32, device=self.device)

        return torch.cat(all_pc_se3, dim=0)

    def get_surface_points_normals_batch(self, q: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if q is None:
            q = torch.zeros((1, self.dof), dtype=torch.float32, device=self.device)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        self.update_status(q)

        n_total = int(self.surface_template_points_local.shape[0])
        if n_total <= 0:
            empty = torch.zeros((int(q.shape[0]), self.surface_num_points, 3), dtype=torch.float32, device=self.device)
            return empty, empty

        batch_size = int(q.shape[0])
        points = torch.zeros((batch_size, n_total, 3), dtype=torch.float32, device=self.device)
        normals = torch.zeros((batch_size, n_total, 3), dtype=torch.float32, device=self.device)
        link_idx = self.surface_template_link_indices
        link_names = self.mesh_link_names

        for link_i, link_name in enumerate(link_names):
            mask = link_idx == int(link_i)
            if not bool(mask.any()):
                continue
            if link_name not in self.frame_status:
                continue
            pts_local = self.surface_template_points_local[mask]
            nrms_local = self.surface_template_normals_local[mask]

            tf = self.frame_status[link_name].get_matrix().to(self.device)
            if tf.ndim == 2:
                tf = tf.unsqueeze(0)
            rot = tf[:, :3, :3]
            trans = tf[:, :3, 3]
            pts_world = torch.matmul(pts_local.unsqueeze(0), rot.transpose(1, 2)) + trans.unsqueeze(1)
            nrms_world = torch.matmul(nrms_local.unsqueeze(0), rot.transpose(1, 2))
            nrms_world = nrms_world / torch.norm(nrms_world, dim=-1, keepdim=True).clamp_min(1e-8)

            points[:, mask] = pts_world
            normals[:, mask] = nrms_world

        return points, normals

    def get_surface_points_normals(self, q: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        points, normals = self.get_surface_points_normals_batch(q=q)
        return points[0], normals[0]

    def sample_random_q(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        batch_size = int(batch_size)
        lower, upper = self.pk_chain.get_joint_limits()
        lower_t = torch.tensor(lower, dtype=torch.float32, device=self.device)
        upper_t = torch.tensor(upper, dtype=torch.float32, device=self.device)
        finite = torch.isfinite(lower_t) & torch.isfinite(upper_t)
        q = torch.zeros((batch_size, self.dof), dtype=torch.float32, device=self.device)
        if finite.any():
            u = torch.rand((batch_size, self.dof), device=self.device, generator=generator)
            mode = torch.rand((batch_size, 1), device=self.device, generator=generator)
            correlated = 0.75 * torch.rand((batch_size, 1), device=self.device, generator=generator) + 0.25 * u
            extreme_choice = torch.randint(
                0,
                2,
                (batch_size, 1),
                device=self.device,
                generator=generator,
            ).bool()
            extreme = torch.where(extreme_choice, u.pow(2.5), 1.0 - u.pow(2.5))
            center = 0.5 + 0.5 * (2.0 * u - 1.0).pow(3)
            t = torch.where(
                mode < 0.25,
                u,
                torch.where(mode < 0.55, correlated, torch.where(mode < 0.80, extreme, center)),
            ).clamp(0.0, 1.0)
            finite_mask = finite.unsqueeze(0).expand(batch_size, -1)
            q[finite_mask] = (
                lower_t.unsqueeze(0).expand(batch_size, -1)[finite_mask]
                + t[finite_mask]
                * (upper_t.unsqueeze(0).expand(batch_size, -1)[finite_mask] - lower_t.unsqueeze(0).expand(batch_size, -1)[finite_mask])
            )
        if (~finite).any():
            q[:, ~finite] = (torch.rand((batch_size, int((~finite).sum().item())), device=self.device, generator=generator) * 2.0 - 1.0) * torch.pi
        q = self._zero_base_translation_in_q(q)
        if batch_size == 1:
            return q[0]
        return q

    def mix_q_by_contact_mask(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        squeeze = False
        if q1.ndim == 1:
            q1 = q1.unsqueeze(0)
            squeeze = True
        if q2.ndim == 1:
            q2 = q2.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)

        if q1.shape != q2.shape:
            raise ValueError(f"q1 and q2 must have the same shape, got {tuple(q1.shape)} vs {tuple(q2.shape)}")
        if q1.ndim != 2:
            raise ValueError(f"Expected q tensors with shape (B, dof), got {tuple(q1.shape)}")
        if q1.shape[-1] != self.dof:
            raise ValueError(f"Expected q last dim {self.dof}, got {q1.shape[-1]}")
        if mask.ndim != 2 or int(mask.shape[0]) != int(q1.shape[0]):
            raise ValueError(
                f"mask must have shape (B, {self.surface_num_points}), got {tuple(mask.shape)} for batch {q1.shape[0]}"
            )
        if int(mask.shape[-1]) != int(self.surface_num_points):
            raise ValueError(
                f"mask last dim must match surface_num_points={self.surface_num_points}, got {mask.shape[-1]}"
            )

        q1 = q1.to(self.device)
        q2 = q2.to(self.device)
        mask = mask.to(self.device).bool()

        active_links = torch.zeros(
            (mask.shape[0], len(self.mesh_link_names)),
            dtype=torch.bool,
            device=self.device,
        )
        link_idx = self.surface_template_link_indices.to(self.device)
        for link_i in range(len(self.mesh_link_names)):
            surface_mask = link_idx == int(link_i)
            if bool(surface_mask.any()):
                active_links[:, link_i] = mask[:, surface_mask].any(dim=1)

        joint_active = (
            active_links.unsqueeze(1)
            & self.joint_descendant_link_mask.unsqueeze(0)
        ).any(dim=-1)
        mixed = torch.where(joint_active, q2, q1)
        if squeeze:
            return mixed.squeeze(0)
        return mixed

    @staticmethod
    def _compute_gendex_contact_value_source_target_batch(
        source_points: torch.Tensor,
        source_normals: torch.Tensor,
        target_points: torch.Tensor,
        target_valid_mask: Optional[torch.Tensor] = None,
        align_exp_scale: float = 2.0,
        sigmoid_scale: float = 10.0,
    ) -> torch.Tensor:
        if source_points.ndim != 3 or source_normals.ndim != 3 or target_points.ndim != 3:
            raise ValueError(
                "Expected batched tensors with shapes "
                "(B, N, 3), (B, N, 3), and (B, K, 3)"
            )
        if source_points.shape != source_normals.shape:
            raise ValueError(
                f"source_points and source_normals must have the same shape, "
                f"got {tuple(source_points.shape)} vs {tuple(source_normals.shape)}"
            )
        if source_points.shape[0] != target_points.shape[0]:
            raise ValueError(
                f"source_points batch and target_points batch must match, "
                f"got {source_points.shape[0]} vs {target_points.shape[0]}"
            )
        if source_points.shape[-1] != 3 or target_points.shape[-1] != 3:
            raise ValueError("Expected xyz points/normals with last dim = 3")

        batch_size = int(source_points.shape[0])
        num_source = int(source_points.shape[1])
        num_target = int(target_points.shape[1])
        if num_source == 0:
            return torch.zeros((batch_size, 0), dtype=torch.float32, device=source_points.device)
        if num_target == 0:
            return torch.zeros((batch_size, num_source), dtype=torch.float32, device=source_points.device)

        source_points = source_points.float()
        source_normals = source_normals.float()
        target_points = target_points.float()

        if target_valid_mask is not None:
            target_valid_mask = target_valid_mask.to(device=source_points.device).bool()
            if target_valid_mask.shape != target_points.shape[:2]:
                raise ValueError(
                    f"target_valid_mask must have shape {tuple(target_points.shape[:2])}, "
                    f"got {tuple(target_valid_mask.shape)}"
                )

        ss = (source_points * source_points).sum(dim=2, keepdim=True)
        tt = (target_points * target_points).sum(dim=2).unsqueeze(1)
        st = torch.matmul(source_points, target_points.transpose(1, 2))
        source_target_dist = torch.sqrt((ss + tt - 2.0 * st).clamp_min(0.0))

        target_dot_normal = torch.matmul(target_points, source_normals.transpose(1, 2))
        source_dot_normal = (source_points * source_normals).sum(dim=2, keepdim=True)
        source_target_align = target_dot_normal.transpose(1, 2) - source_dot_normal
        source_target_align = source_target_align / (source_target_dist + 1e-5)

        source_target_align_dist = source_target_dist * torch.exp(
            float(align_exp_scale) * (1.0 - source_target_align)
        )
        if target_valid_mask is not None:
            source_target_align_dist = source_target_align_dist.masked_fill(
                ~target_valid_mask.unsqueeze(1),
                float("inf"),
            )

        contact_dist = torch.sqrt(source_target_align_dist.min(dim=2).values.clamp_min(0.0))
        contact_value = 1.0 - 2.0 * (torch.sigmoid(float(sigmoid_scale) * contact_dist) - 0.5)
        return contact_value.clamp(0.0, 1.0)

    @staticmethod
    def _compute_gendex_contact_value_source_target(
        source_points: torch.Tensor,
        source_normals: torch.Tensor,
        target_points: torch.Tensor,
        align_exp_scale: float = 2.0,
        sigmoid_scale: float = 10.0,
    ) -> torch.Tensor:
        if source_points.ndim != 2 or source_normals.ndim != 2 or target_points.ndim != 2:
            raise ValueError(
                "Expected unbatched tensors with shapes (N, 3), (N, 3), and (K, 3)"
            )

        values = RobotModel._compute_gendex_contact_value_source_target_batch(
            source_points=source_points.unsqueeze(0),
            source_normals=source_normals.unsqueeze(0),
            target_points=target_points.unsqueeze(0),
            target_valid_mask=None,
            align_exp_scale=align_exp_scale,
            sigmoid_scale=sigmoid_scale,
        )
        return values[0]

    def _mask_component_count(self, mask: torch.Tensor) -> int:
        mask_bool = mask.detach().bool().cpu()
        n = int(mask_bool.numel())
        if n == 0 or int(mask_bool.sum()) == 0:
            return 0
        neighbors = self.surface_graph_neighbors
        if len(neighbors) != n:
            raise RuntimeError(
                "Surface connectivity graph is not aligned with mask size. "
                "Rebuild the graph or check surface template consistency."
            )

        active = mask_bool.numpy()
        visited = np.zeros((n,), dtype=bool)
        components = 0

        for root in np.where(active)[0].tolist():
            if visited[root]:
                continue
            components += 1
            stack = [root]
            visited[root] = True
            while stack:
                i = stack.pop()
                for j in neighbors[i]:
                    if (not visited[j]) and active[j]:
                        visited[j] = True
                        stack.append(j)
        return int(components)

    @staticmethod
    def _random_tangent_basis(normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ref = (
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=normals.device)
            .unsqueeze(0)
            .repeat(normals.shape[0], 1)
        )
        alt = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=normals.device).unsqueeze(0).repeat(normals.shape[0], 1)
        use_alt = torch.abs((normals * ref).sum(dim=1)) > 0.9
        ref[use_alt] = alt[use_alt]
        t1 = torch.cross(normals, ref, dim=1)
        t1 = t1 / torch.norm(t1, dim=1, keepdim=True).clamp_min(1e-8)
        t2 = torch.cross(normals, t1, dim=1)
        t2 = t2 / torch.norm(t2, dim=1, keepdim=True).clamp_min(1e-8)
        return t1, t2

    @staticmethod
    def _broadcast_batch_param(
        value: float | torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        out = torch.as_tensor(value, dtype=torch.float32, device=device)
        if out.ndim == 0:
            out = out.repeat(batch_size)
        else:
            out = out.view(-1)
        if int(out.numel()) != int(batch_size):
            raise ValueError(f"{name} must broadcast to batch size {batch_size}, got {tuple(out.shape)}")
        return out

    def _sample_virtual_patch_geometry_batch(
        self,
        *,
        anchor_points: torch.Tensor,
        anchor_normals: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        anchor_shift_min: float | torch.Tensor,
        anchor_shift_max: float | torch.Tensor,
        max_plane_extent_min: float | torch.Tensor,
        max_plane_extent_max: float | torch.Tensor,
        patch_shift_power: float,
        patch_extent_power: float,
        normal_jitter_max_deg: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, max_anchors, _ = anchor_points.shape
        anchor_shift_min_t = self._broadcast_batch_param(
            anchor_shift_min,
            batch_size=batch_size,
            device=self.device,
            name="anchor_shift_min",
        )
        anchor_shift_max_t = self._broadcast_batch_param(
            anchor_shift_max,
            batch_size=batch_size,
            device=self.device,
            name="anchor_shift_max",
        )
        anchor_shift_max_t = torch.maximum(anchor_shift_max_t, anchor_shift_min_t)
        extent_min_t = self._broadcast_batch_param(
            max_plane_extent_min,
            batch_size=batch_size,
            device=self.device,
            name="max_plane_extent_min",
        )
        extent_max_t = self._broadcast_batch_param(
            max_plane_extent_max,
            batch_size=batch_size,
            device=self.device,
            name="max_plane_extent_max",
        )
        extent_max_t = torch.maximum(extent_max_t, extent_min_t)

        us = torch.rand((batch_size,), device=self.device, generator=generator).pow(max(1e-3, float(patch_shift_power)))
        ue = torch.rand((batch_size,), device=self.device, generator=generator).pow(max(1e-3, float(patch_extent_power)))
        anchor_shift_t = anchor_shift_min_t + us * (anchor_shift_max_t - anchor_shift_min_t).clamp_min(1e-8)
        max_plane_extent_t = extent_min_t + ue * (extent_max_t - extent_min_t).clamp_min(1e-8)

        base_t1, base_t2 = self._random_tangent_basis(anchor_normals.reshape(-1, 3))
        base_t1 = base_t1.view(batch_size, max_anchors, 3)
        base_t2 = base_t2.view(batch_size, max_anchors, 3)
        max_jitter_rad = math.radians(max(0.0, float(normal_jitter_max_deg)))
        if max_jitter_rad > 0.0:
            jitter_phi = torch.rand((batch_size, max_anchors), device=self.device, generator=generator) * (2.0 * math.pi)
            jitter_theta = torch.rand((batch_size, max_anchors), device=self.device, generator=generator) * max_jitter_rad
            jitter_tangent = (
                torch.cos(jitter_phi).unsqueeze(-1) * base_t1
                + torch.sin(jitter_phi).unsqueeze(-1) * base_t2
            )
            perturbed_normals = (
                torch.cos(jitter_theta).unsqueeze(-1) * anchor_normals
                + torch.sin(jitter_theta).unsqueeze(-1) * jitter_tangent
            )
            perturbed_normals = perturbed_normals / perturbed_normals.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            perturbed_normals = anchor_normals

        flat_t1, flat_t2 = self._random_tangent_basis(perturbed_normals.reshape(-1, 3))
        t1 = flat_t1.view(batch_size, max_anchors, 3)
        t2 = flat_t2.view(batch_size, max_anchors, 3)
        centers = anchor_points + anchor_shift_t[:, None, None] * perturbed_normals

        u0 = torch.rand((batch_size, max_anchors), device=self.device, generator=generator)
        u1 = torch.rand((batch_size, max_anchors), device=self.device, generator=generator)
        ex = max_plane_extent_t[:, None] * u0.pow(0.32)
        ey = max_plane_extent_t[:, None] * u1.pow(0.32)

        boost_mask = torch.rand((batch_size, max_anchors), device=self.device, generator=generator) < 0.18
        boost = torch.empty((batch_size, max_anchors), device=self.device).uniform_(1.2, 1.8, generator=generator)
        ex = torch.where(boost_mask, ex * boost, ex)
        ey = torch.where(boost_mask, ey * boost, ey)

        edge_mask = torch.rand((batch_size, max_anchors), device=self.device, generator=generator) < 0.35
        edge_scale = torch.empty((batch_size, max_anchors), device=self.device).uniform_(0.04, 0.25, generator=generator)
        edge_axis = torch.randint(
            0,
            2,
            (batch_size, max_anchors),
            device=self.device,
            generator=generator,
        ).bool()
        ex = torch.where(edge_mask & edge_axis, ex * edge_scale, ex)
        ey = torch.where(edge_mask & (~edge_axis), ey * edge_scale, ey)
        ex = ex.clamp_min(2e-4)
        ey = ey.clamp_min(2e-4)
        return centers, t1, t2, ex, ey

    def _sample_virtual_object_patches_batch(
        self,
        hand_points: torch.Tensor,
        hand_normals: torch.Tensor,
        anchor_indices: torch.Tensor,
        anchor_valid_mask: torch.Tensor,
        anchor_point_valid_mask: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        anchor_shift_min: float | torch.Tensor = 0.0002,
        anchor_shift_max: float | torch.Tensor = 0.0030,
        max_plane_extent_min: float | torch.Tensor = 0.006,
        max_plane_extent_max: float | torch.Tensor = 0.050,
        patch_shift_power: float = 1.0,
        patch_extent_power: float = 1.0,
        normal_jitter_max_deg: float = 10.0,
        penetration_clearance: float = 0.0001,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hand_points.ndim != 3 or hand_normals.ndim != 3:
            raise ValueError(
                f"hand_points and hand_normals must have shape (B, N, 3), got {tuple(hand_points.shape)} and {tuple(hand_normals.shape)}"
            )
        if hand_points.shape != hand_normals.shape:
            raise ValueError(
                f"hand_points and hand_normals must have the same shape, got {tuple(hand_points.shape)} vs {tuple(hand_normals.shape)}"
            )
        if anchor_indices.ndim != 2 or anchor_valid_mask.ndim != 2:
            raise ValueError(
                f"anchor_indices and anchor_valid_mask must have shape (B, A), got {tuple(anchor_indices.shape)} and {tuple(anchor_valid_mask.shape)}"
            )
        if anchor_indices.shape != anchor_valid_mask.shape:
            raise ValueError(
                f"anchor_indices and anchor_valid_mask must have the same shape, got {tuple(anchor_indices.shape)} vs {tuple(anchor_valid_mask.shape)}"
            )
        if anchor_point_valid_mask.ndim != 3 or anchor_point_valid_mask.shape[:2] != anchor_indices.shape:
            raise ValueError(
                "anchor_point_valid_mask must have shape (B, A, P) aligned with anchor_indices, "
                f"got {tuple(anchor_point_valid_mask.shape)} for anchors {tuple(anchor_indices.shape)}"
            )

        batch_size, num_points, _ = hand_points.shape
        _, max_anchors = anchor_indices.shape
        max_points_per_anchor = int(anchor_point_valid_mask.shape[-1])
        if int(batch_size) == 0 or max_anchors == 0 or max_points_per_anchor == 0:
            empty_points = torch.zeros(
                (batch_size, max_anchors, max_points_per_anchor, 3),
                dtype=torch.float32,
                device=self.device,
            )
            empty_mask = torch.zeros(
                (batch_size, max_anchors, max_points_per_anchor),
                dtype=torch.bool,
                device=self.device,
            )
            return empty_points, empty_mask

        anchor_indices = anchor_indices.to(device=self.device, dtype=torch.long) # (B, A)
        anchor_valid_mask = anchor_valid_mask.to(device=self.device).bool() # (B, A)
        anchor_point_valid_mask = anchor_point_valid_mask.to(device=self.device).bool() # (B, A, P)
        safe_anchor_indices = anchor_indices.clamp(0, max(0, int(num_points) - 1))
        gather_idx = safe_anchor_indices.unsqueeze(-1).expand(-1, -1, 3)
        anchor_points = torch.gather(hand_points, 1, gather_idx)
        anchor_normals = torch.gather(hand_normals, 1, gather_idx)
        centers, t1, t2, ex, ey = self._sample_virtual_patch_geometry_batch(
            anchor_points=anchor_points,
            anchor_normals=anchor_normals,
            generator=generator,
            anchor_shift_min=anchor_shift_min,
            anchor_shift_max=anchor_shift_max,
            max_plane_extent_min=max_plane_extent_min,
            max_plane_extent_max=max_plane_extent_max,
            patch_shift_power=patch_shift_power,
            patch_extent_power=patch_extent_power,
            normal_jitter_max_deg=normal_jitter_max_deg,
        )

        u = (torch.rand((batch_size, max_anchors, max_points_per_anchor), device=self.device, generator=generator) * 2.0 - 1.0) * ex.unsqueeze(-1)
        v = (torch.rand((batch_size, max_anchors, max_points_per_anchor), device=self.device, generator=generator) * 2.0 - 1.0) * ey.unsqueeze(-1)
        object_points = (
            centers.unsqueeze(2)
            + u.unsqueeze(-1) * t1.unsqueeze(2)
            + v.unsqueeze(-1) * t2.unsqueeze(2)
        )

        point_valid_mask = anchor_point_valid_mask & anchor_valid_mask.unsqueeze(-1)
        flat_points = object_points.view(batch_size, max_anchors * max_points_per_anchor, 3)
        flat_valid_mask = point_valid_mask.view(batch_size, max_anchors * max_points_per_anchor)
        if bool(flat_valid_mask.any()):
            nearest_idx = torch.cdist(flat_points, hand_points).argmin(dim=-1)
            nearest_gather = nearest_idx.unsqueeze(-1).expand(-1, -1, 3)
            ref_points = torch.gather(hand_points, 1, nearest_gather)
            ref_normals = torch.gather(hand_normals, 1, nearest_gather)
            signed = torch.sum((flat_points - ref_points) * ref_normals, dim=-1)
            flat_valid_mask = flat_valid_mask & (signed >= float(penetration_clearance))
        flat_points = flat_points.masked_fill(~flat_valid_mask.unsqueeze(-1), 0.0)
        return (
            flat_points.view(batch_size, max_anchors, max_points_per_anchor, 3),
            flat_valid_mask.view(batch_size, max_anchors, max_points_per_anchor),
        )

    def _sample_virtual_object_patches(
        self,
        hand_points: torch.Tensor,
        hand_normals: torch.Tensor,
        num_anchors: int,
        total_patch_points: int,
        generator: Optional[torch.Generator] = None,
        anchor_indices: Optional[torch.Tensor] = None,
        anchor_shift: float = 0.002,
        max_plane_extent: float = 0.026,
        penetration_clearance: float = 0.0001,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        n_pts = int(hand_points.shape[0])
        if n_pts == 0:
            empty = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            return empty, {"anchor_indices": torch.zeros((0,), dtype=torch.long), "patch_types": torch.zeros((0,), dtype=torch.long)}

        n_anchors = int(max(1, min(num_anchors, n_pts)))
        if anchor_indices is None:
            anchor_indices = torch.randperm(n_pts, device=self.device, generator=generator)[:n_anchors]
        else:
            anchor_indices = anchor_indices.to(device=self.device, dtype=torch.long).view(-1)
            if int(anchor_indices.numel()) == 0:
                anchor_indices = torch.randperm(n_pts, device=self.device, generator=generator)[:n_anchors]
            else:
                anchor_indices = anchor_indices[:n_anchors]
        n_anchors = int(anchor_indices.numel())
        total_patch_points = max(int(total_patch_points), n_anchors)
        counts = torch.full((n_anchors,), total_patch_points // n_anchors, dtype=torch.long, device=self.device)
        counts[: total_patch_points % n_anchors] += 1
        max_points_per_anchor = int(counts.max().item())
        anchor_valid_mask = torch.ones((1, n_anchors), dtype=torch.bool, device=self.device)
        anchor_point_valid_mask = (
            torch.arange(max_points_per_anchor, device=self.device).view(1, 1, max_points_per_anchor)
            < counts.view(1, n_anchors, 1)
        )
        object_points, object_valid_mask = self._sample_virtual_object_patches_batch(
            hand_points=hand_points.unsqueeze(0),
            hand_normals=hand_normals.unsqueeze(0),
            anchor_indices=anchor_indices.view(1, n_anchors),
            anchor_valid_mask=anchor_valid_mask,
            anchor_point_valid_mask=anchor_point_valid_mask,
            generator=generator,
            anchor_shift_min=anchor_shift,
            anchor_shift_max=anchor_shift,
            max_plane_extent_min=max_plane_extent,
            max_plane_extent_max=max_plane_extent,
            patch_shift_power=1.0,
            patch_extent_power=1.0,
            normal_jitter_max_deg=0.0,
            penetration_clearance=penetration_clearance,
        )
        patch_types_t = torch.zeros((n_anchors,), dtype=torch.long, device=self.device)
        return object_points[0][object_valid_mask[0]], {"anchor_indices": anchor_indices, "patch_types": patch_types_t}

    def _sample_patch_anchor_indices(
        self,
        hand_points: torch.Tensor,
        n_anchor: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        n_pts = int(hand_points.shape[0])
        n_anchor = int(max(1, min(int(n_anchor), n_pts)))
        mode_u = float(torch.rand(1, device=self.device, generator=generator).item())

        # 1) Random anchors.
        if mode_u < 0.35 or n_anchor <= 1:
            return torch.randperm(n_pts, device=self.device, generator=generator)[:n_anchor]

        # 2) Spread anchors (greedy farthest-point).
        if mode_u < 0.70:
            first = int(torch.randint(0, n_pts, (1,), device=self.device, generator=generator).item())
            selected = [first]
            dmin = torch.cdist(hand_points[first : first + 1], hand_points).squeeze(0)
            while len(selected) < n_anchor:
                nxt = int(torch.argmax(dmin).item())
                selected.append(nxt)
                dnew = torch.cdist(hand_points[nxt : nxt + 1], hand_points).squeeze(0)
                dmin = torch.minimum(dmin, dnew)
            return torch.tensor(selected, dtype=torch.long, device=self.device)

        # 3) Clustered anchors via graph random-walk.
        neighbors = self.surface_graph_neighbors
        if len(neighbors) != n_pts:
            return torch.randperm(n_pts, device=self.device, generator=generator)[:n_anchor]
        seed = int(torch.randint(0, n_pts, (1,), device=self.device, generator=generator).item())
        picked = {seed}
        order = [seed]
        cur = seed
        steps = 0
        max_steps = 10 * n_anchor + 50
        while len(order) < n_anchor and steps < max_steps:
            steps += 1
            nb = neighbors[cur]
            if len(nb) == 0:
                cur = seed
                continue
            j = int(torch.randint(0, len(nb), (1,), device=self.device, generator=generator).item())
            nxt = int(nb[j])
            if nxt not in picked:
                picked.add(nxt)
                order.append(nxt)
            cur = nxt
            if bool(torch.rand(1, device=self.device, generator=generator).item() < 0.2):
                cur = order[int(torch.randint(0, len(order), (1,), device=self.device, generator=generator).item())]
        if len(order) < n_anchor:
            extra = torch.randperm(n_pts, device=self.device, generator=generator).tolist()
            for idx in extra:
                if idx not in picked:
                    picked.add(int(idx))
                    order.append(int(idx))
                if len(order) >= n_anchor:
                    break
        return torch.tensor(order[:n_anchor], dtype=torch.long, device=self.device)

    @staticmethod
    def _estimate_contact_count_range(
        counts: torch.Tensor,
        *,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
    ) -> Tuple[int, int]:
        counts = counts.detach().view(-1).float().cpu()
        if int(counts.numel()) == 0:
            return 1, 1
        lower = int(torch.quantile(counts, q=float(lower_quantile)).floor().item())
        upper = int(torch.quantile(counts, q=float(upper_quantile)).ceil().item())
        lower = max(1, lower)
        upper = max(lower, upper)
        return lower, upper

    def fit_contact_mask_sampler(
        self,
        real_masks: torch.Tensor,
        max_component_samples: int = 5000,
    ) -> Dict[str, torch.Tensor]:
        masks = real_masks.detach().bool().cpu()
        if masks.ndim != 2:
            raise ValueError(f"Expected real_masks shape (M,N), got {tuple(masks.shape)}")
        if int(masks.shape[1]) != int(self.surface_num_points):
            raise ValueError(
                f"real_masks N={masks.shape[1]} does not match surface_num_points={self.surface_num_points}"
            )

        counts = masks.sum(dim=1).long()
        count_hist = torch.bincount(counts, minlength=self.surface_num_points + 1).float()
        count_lo, count_hi = self._estimate_contact_count_range(counts)

        max_comp = 32
        comp_hist = torch.zeros((max_comp + 1,), dtype=torch.float32)
        comp_values: List[int] = []
        n_comp_eval = min(int(max_component_samples), int(masks.shape[0]))
        if n_comp_eval > 0:
            idx = torch.randperm(int(masks.shape[0]))[:n_comp_eval]
            for i in idx.tolist():
                c = self._mask_component_count(masks[i].to(self.device))
                comp_values.append(int(c))
                c = int(min(c, max_comp))
                comp_hist[c] += 1.0
        if comp_hist.sum() <= 0:
            comp_hist[1] = 1.0
            comp_values = [1]

        comp_t = torch.tensor(comp_values, dtype=torch.float32)
        if comp_t.numel() == 0:
            comp_lo = 1
            comp_hi = 4
        else:
            comp_t = comp_t.clamp_min(1.0)
            comp_lo = int(torch.quantile(comp_t, q=0.1).round().item())
            comp_hi = int(torch.quantile(comp_t, q=0.9).round().item())
            comp_lo = max(1, comp_lo)
            comp_hi = max(comp_lo, comp_hi)
        comp_lo = min(comp_lo, 8)
        comp_hi = max(comp_lo, min(comp_hi, 8))
        component_count_values = torch.arange(comp_lo, comp_hi + 1, dtype=torch.long)
        component_count_distribution = comp_hist[comp_lo : comp_hi + 1].float()
        if float(component_count_distribution.sum().item()) <= 0.0:
            component_count_distribution = torch.ones_like(component_count_distribution)
        component_count_distribution = component_count_distribution / component_count_distribution.sum().clamp_min(1e-8)

        state: Dict[str, torch.Tensor] = {
            "count_hist": count_hist,
            "component_hist": comp_hist,
            "component_range": torch.tensor([comp_lo, comp_hi], dtype=torch.long),
            "component_count_values": component_count_values,
            "component_count_distribution": component_count_distribution,
            "contact_count_range": torch.tensor([count_lo, count_hi], dtype=torch.long),
            "num_real_samples": torch.tensor([int(masks.shape[0])], dtype=torch.long),
        }
        self.contact_mask_sampler_state = {
            k: (v.detach().cpu() if torch.is_tensor(v) else torch.tensor(v))
            for k, v in state.items()
        }
        return self.contact_mask_sampler_state

    def _sample_q_for_contact(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return self.sample_random_q(batch_size=1, generator=generator)

    def get_sampled_pc(
        self,
        q: Optional[torch.Tensor] = None,
        num_points: int = 512,
        mode: str = "active",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q is None:
            q = self.get_canonical_q()
        sampled_pc = self.get_transformed_links_pc(q=q, links_pc=None, mode=mode)
        return farthest_point_sampling(sampled_pc, num_points)

    def get_canonical_q(self) -> torch.Tensor:
        lower, upper = self.pk_chain.get_joint_limits()
        lower_t = torch.tensor(lower, dtype=torch.float32, device=self.device)
        upper_t = torch.tensor(upper, dtype=torch.float32, device=self.device)
        finite = torch.isfinite(lower_t) & torch.isfinite(upper_t)

        canonical = torch.zeros(self.dof, dtype=torch.float32, device=self.device)
        canonical[finite] = lower_t[finite] * 0.75 + upper_t[finite] * 0.25
        return canonical

    def get_initial_q(self, q: Optional[torch.Tensor] = None, max_angle: float = math.pi / 6) -> torch.Tensor:
        if q is None:
            lower, upper = self.pk_chain.get_joint_limits()
            lower_t = torch.tensor(lower, dtype=torch.float32, device=self.device)
            upper_t = torch.tensor(upper, dtype=torch.float32, device=self.device)
            q_initial = torch.zeros(self.dof, dtype=torch.float32, device=self.device)
            finite = torch.isfinite(lower_t) & torch.isfinite(upper_t)
            portion = random.uniform(0.65, 0.85)
            q_initial[finite] = lower_t[finite] * portion + upper_t[finite] * (1 - portion)
            return q_initial

        if len(q.shape) > 1:
            q = q.squeeze(0)
        if q.shape[-1] == self.dof + 3:
            q = q_rot6d_to_q_euler(q)
        if q.shape[-1] != self.dof:
            raise ValueError(f"Expected joint dim {self.dof}, got {q.shape[-1]}")

        q_initial = q.clone().to(self.device)

        if self.dof >= 9:
            eps = 1e-8
            direction_raw = -q_initial[:3]
            direction_norm = torch.norm(direction_raw)
            if direction_norm < eps:
                direction = torch.randn(3, device=self.device)
                direction = direction / (torch.norm(direction) + eps)
            else:
                direction = direction_raw / direction_norm
            angle = torch.tensor(random.uniform(0, max_angle), device=self.device)
            axis = torch.randn(3, device=self.device)
            axis -= torch.dot(axis, direction) * direction
            axis_norm = torch.norm(axis)
            if axis_norm < eps:
                fallback = torch.tensor([1.0, 0.0, 0.0], device=self.device)
                if torch.abs(torch.dot(fallback, direction)) > 0.9:
                    fallback = torch.tensor([0.0, 1.0, 0.0], device=self.device)
                axis = torch.cross(direction, fallback)
                axis_norm = torch.norm(axis)
            axis = axis / (axis_norm + eps)
            random_rotation = _axisangle_to_matrix(axis, angle).to(self.device)
            rotation_matrix = random_rotation @ _rot6d_to_matrix(q_initial[3:9])
            q_initial[3:9] = _matrix_to_rot6d(rotation_matrix)
            q_initial = q_rot6d_to_q_euler(q_initial)

        return q_initial

    def get_trimesh_q(self, q: torch.Tensor, mode: str = "active") -> Dict[str, object]:
        self.update_status(q)
        mode_meshes, _ = self._resolve_mode_assets(mode)

        vertices = []
        faces = []
        vertex_offset = 0
        parts: Dict[str, trimesh.Trimesh] = {}

        for link_name, mesh in mode_meshes.items():
            if link_name not in self.frame_status:
                continue
            tf = self.frame_status[link_name].get_matrix()[0].detach().cpu().numpy()
            part_mesh = mesh.copy()
            part_mesh.apply_transform(tf)
            parts[link_name] = part_mesh

            vertices.append(part_mesh.vertices)
            faces.append(part_mesh.faces + vertex_offset)
            vertex_offset += len(part_mesh.vertices)

        if vertices:
            all_vertices = np.vstack(vertices)
            all_faces = np.vstack(faces)
            visual = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
        else:
            visual = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))

        return {"visual": visual, "parts": parts}

    def get_trimesh_se3(self, transform: Dict[str, torch.Tensor], index: int, mode: str = "active") -> trimesh.Trimesh:
        mode_meshes, _ = self._resolve_mode_assets(mode)
        vertices = []
        faces = []
        vertex_offset = 0

        for link_name, mesh in mode_meshes.items():
            if link_name not in transform:
                continue
            tf = transform[link_name][index].detach().cpu().numpy()
            m = mesh.copy()
            m.apply_transform(tf)
            vertices.append(m.vertices)
            faces.append(m.faces + vertex_offset)
            vertex_offset += len(m.vertices)

        if not vertices:
            return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))

        return trimesh.Trimesh(vertices=np.vstack(vertices), faces=np.vstack(faces))


def create_robot_model(
    robot_name: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_points: int = 512,
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    remove_special_bases: bool = True,
    remove_special_bases_override: bool = False,
) -> RobotModel:
    meta = discover_robot_assets(assets_dir)

    if robot_name not in meta:
        matches = [k for k in meta if k.endswith(f"/{robot_name}") or robot_name in k]
        if len(matches) == 1:
            robot_name = matches[0]
        else:
            raise KeyError(
                f"Robot '{robot_name}' not found. Available count={len(meta)}. "
                f"Example keys: {list(meta.keys())[:8]}"
            )

    robot_rel_path = Path(meta[robot_name]["path"])
    robot_path = Path(assets_dir) / robot_rel_path
    if not robot_path.exists():
        raise FileNotFoundError(f"Robot asset does not exist: {robot_path}")

    if remove_special_bases:
        robot_path = ensure_base_removed_asset(
            robot_name=robot_name,
            robot_path=robot_path,
            force_override=remove_special_bases_override,
        )

    return RobotModel(robot_name, robot_path, device=device, num_points=num_points)


def dump_robot_assets_index(
    out_path: Path = DEFAULT_ASSETS_DIR / "robot_assets_index.json",
    assets_dir: Path = DEFAULT_ASSETS_DIR,
) -> None:
    meta = discover_robot_assets(assets_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    dump_robot_assets_index()
    build_manipulator_robot_lists()
    print(f"Wrote {DEFAULT_ASSETS_DIR / 'robot_assets_index.json'}")
    print(f"Wrote {DEFAULT_MANIPULATOR_LISTS_JSON}")
