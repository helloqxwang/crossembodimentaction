from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

try:
    from pytorch3d.ops import sample_farthest_points
except Exception:
    sample_farthest_points = None


SURFACE_ARTIFACT_VERSION = "surface_template_graph_v1"


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


def _farthest_point_sampling(pos: torch.Tensor, n_sampling: int) -> Tuple[torch.Tensor, torch.Tensor]:
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

    if sample_farthest_points is not None:
        sampled_xyz, sampled_idx = sample_farthest_points(pos[..., :3].contiguous(), K=k)
        if pos.shape[-1] > 3:
            gather_idx = sampled_idx[..., :, None].expand(*sampled_idx.shape, pos.shape[-1])
            sampled = torch.gather(pos, -2, gather_idx)
        else:
            sampled = sampled_xyz
    else:
        sampled_idx = torch.randperm(n_points, device=pos.device)[:k]
        sampled = pos[:, sampled_idx, :]
        sampled_idx = sampled_idx.unsqueeze(0).expand(pos.shape[0], -1)

    if squeeze:
        sampled = sampled.squeeze(0)
        sampled_idx = sampled_idx.squeeze(0)
    return sampled, sampled_idx


def get_surface_artifact_path(robot_path: Path, robot_name: str, num_points: int) -> Path:
    robot_path = Path(robot_path)
    key = f"{robot_name}|{str(robot_path.resolve())}|{int(num_points)}|{SURFACE_ARTIFACT_VERSION}"
    short = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return robot_path.parent / f"surface_template_graph_n{int(num_points)}_{short}.pt"


def _robot_scale(meshes: List[trimesh.Trimesh]) -> float:
    if len(meshes) == 0:
        return 0.2
    mins = []
    maxs = []
    for mesh in meshes:
        if mesh is None or len(mesh.vertices) == 0:
            continue
        b = np.asarray(mesh.bounds, dtype=np.float64)
        mins.append(b[0])
        maxs.append(b[1])
    if len(mins) == 0:
        return 0.2
    bb_min = np.min(np.stack(mins, axis=0), axis=0)
    bb_max = np.max(np.stack(maxs, axis=0), axis=0)
    return float(max(np.linalg.norm(bb_max - bb_min), 1e-3))


def _surface_proxy_pitch_from_scale(scale: float) -> float:
    return float(np.clip(scale / 260.0, 0.0003, 0.0010))


def _wrapper_closing_structure() -> np.ndarray:
    st = np.zeros((3, 3, 3), dtype=bool)
    st[1, 1, 1] = True
    st[0, 1, 1] = True
    st[2, 1, 1] = True
    st[1, 0, 1] = True
    st[1, 2, 1] = True
    st[1, 1, 0] = True
    st[1, 1, 2] = True
    return st


def _build_voxel_wrapper_mesh(
    robot_name: str,
    meshes_world: List[trimesh.Trimesh],
    pitch: float,
) -> trimesh.Trimesh:
    occ_points = []
    for mesh in meshes_world:
        if mesh is None or len(mesh.faces) == 0 or len(mesh.vertices) == 0:
            continue
        try:
            vg = mesh.voxelized(pitch=float(pitch))
            pts = np.asarray(vg.points, dtype=np.float64)
        except Exception:
            pts = np.zeros((0, 3), dtype=np.float64)
        if pts.shape[0] > 0:
            occ_points.append(pts)

    if len(occ_points) == 0:
        raise RuntimeError(f"Failed to voxelize canonical meshes for robot: {robot_name}")

    all_pts = np.concatenate(occ_points, axis=0)
    origin = all_pts.min(axis=0) - 2.0 * float(pitch)
    idx = np.floor((all_pts - origin[None, :]) / float(pitch) + 1e-6).astype(np.int64)
    idx = np.unique(idx, axis=0)
    if idx.shape[0] == 0:
        raise RuntimeError(f"Empty occupancy indices for robot: {robot_name}")

    shape = idx.max(axis=0) + 3
    occ = np.zeros(tuple(int(x) for x in shape.tolist()), dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    structure = _wrapper_closing_structure()
    occ_closed = ndi.binary_closing(occ, structure=structure, iterations=1)

    occ_count = int(occ.sum())
    added = int(np.logical_and(occ_closed, np.logical_not(occ)).sum())
    add_ratio = float(added) / max(1, occ_count)
    if add_ratio <= 0.06:
        occ = occ_closed

    try:
        occ = ndi.binary_fill_holes(occ)
    except Exception:
        pass

    idx_final = np.argwhere(occ)
    if idx_final.shape[0] == 0:
        raise RuntimeError(f"Wrapper occupancy became empty for robot: {robot_name}")

    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = float(pitch)
    transform[1, 1] = float(pitch)
    transform[2, 2] = float(pitch)
    transform[:3, 3] = origin

    enc = trimesh.voxel.encoding.SparseBinaryEncoding(idx_final.astype(np.int64))
    vg_union = trimesh.voxel.VoxelGrid(encoding=enc, transform=transform)
    mesh_union = _as_mesh(vg_union.marching_cubes)
    if mesh_union is None or len(mesh_union.faces) == 0:
        raise RuntimeError(f"Failed to build wrapper marching-cubes mesh for robot: {robot_name}")

    scale = _robot_scale(meshes_world)
    mesh_diag = float(np.linalg.norm(np.asarray(mesh_union.bounds[1] - mesh_union.bounds[0], dtype=np.float64)))
    if mesh_diag > 4.0 * max(scale, 1e-6):
        mesh_union = mesh_union.copy()
        mesh_union.apply_transform(vg_union.transform)

    mesh_union = trimesh.Trimesh(
        vertices=np.asarray(mesh_union.vertices, dtype=np.float64),
        faces=np.asarray(mesh_union.faces, dtype=np.int64),
        process=True,
    )
    if hasattr(trimesh.repair, "fix_normals"):
        trimesh.repair.fix_normals(mesh_union)
    if len(mesh_union.faces) == 0:
        raise RuntimeError(f"Wrapper mesh is empty for robot: {robot_name}")
    return mesh_union


def _orient_watertight_components_outward(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    try:
        comps = list(mesh.split(only_watertight=False))
    except Exception:
        comps = []
    if len(comps) == 0:
        out = mesh.copy()
    else:
        fixed = []
        for comp in comps:
            c = comp.copy()
            try:
                vol = float(c.volume)
                if np.isfinite(vol) and vol < 0.0:
                    c.invert()
            except Exception:
                pass
            fixed.append(c)
        out = _as_mesh(trimesh.util.concatenate(fixed))
        if out is None:
            out = mesh.copy()
    if hasattr(trimesh.repair, "fix_normals"):
        trimesh.repair.fix_normals(out)
    return out


def _build_boolean_union_mesh(
    robot_name: str,
    meshes_world: List[trimesh.Trimesh],
    pitch: Optional[float] = None,
) -> trimesh.Trimesh:
    if len(meshes_world) == 0:
        raise RuntimeError("Cannot build union mesh from empty mesh list.")
    scale = _robot_scale(meshes_world)
    wrapper_pitch = float(_surface_proxy_pitch_from_scale(scale) if pitch is None else pitch)
    mesh_union = _build_voxel_wrapper_mesh(robot_name=robot_name, meshes_world=meshes_world, pitch=wrapper_pitch)
    mesh_union = _orient_watertight_components_outward(mesh_union)
    if len(mesh_union.faces) == 0:
        raise RuntimeError(f"Final union mesh is empty for robot: {robot_name}")
    return mesh_union


def _sample_surface_points_normals_from_mesh(
    mesh: trimesh.Trimesh,
    num_points: int,
    device: torch.device,
    oversample_ratio: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(num_points) <= 0:
        raise ValueError("num_points must be positive.")
    n_query = int(max(num_points, num_points * max(1, int(oversample_ratio))))
    try:
        pts_np, face_idx = mesh.sample(n_query, return_index=True)
        nrms_np = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)
    except Exception:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        if verts.shape[0] == 0:
            raise RuntimeError("Cannot sample surface points from empty union mesh.")
        rand_idx = np.random.randint(0, verts.shape[0], size=(n_query,))
        pts_np = verts[rand_idx]
        nrms_np = np.zeros_like(pts_np, dtype=np.float32)

    pts = torch.tensor(pts_np, dtype=torch.float32, device=device)
    nrms = torch.tensor(nrms_np, dtype=torch.float32, device=device)
    nrms = nrms / torch.norm(nrms, dim=1, keepdim=True).clamp_min(1e-8)

    if int(pts.shape[0]) > int(num_points):
        _, keep = _farthest_point_sampling(pts, int(num_points))
        keep = keep.long()
        pts = pts[keep]
        nrms = nrms[keep]
    elif int(pts.shape[0]) < int(num_points):
        extra_n = int(num_points - pts.shape[0])
        extra_idx = torch.randint(0, int(pts.shape[0]), (extra_n,), device=device)
        pts = torch.cat([pts, pts[extra_idx]], dim=0)
        nrms = torch.cat([nrms, nrms[extra_idx]], dim=0)

    if bool(getattr(mesh, "is_watertight", False)) and int(pts.shape[0]) > 0:
        try:
            pts_np = pts.detach().cpu().numpy().astype(np.float64)
            nrms_np = nrms.detach().cpu().numpy().astype(np.float64)
            mesh_diag = float(np.linalg.norm(np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64)))
            eps = max(1e-5, 1e-4 * mesh_diag)
            plus_inside = np.asarray(mesh.contains(pts_np + nrms_np * eps), dtype=bool).reshape(-1)
            minus_inside = np.asarray(mesh.contains(pts_np - nrms_np * eps), dtype=bool).reshape(-1)
            flip = plus_inside & (~minus_inside)
            if bool(np.any(flip)):
                nrms_np[flip] *= -1.0
                nrms = torch.tensor(nrms_np, dtype=torch.float32, device=device)
        except Exception:
            pass
    return pts, nrms


def _assign_union_points_to_links(
    model: Any,
    points_world: torch.Tensor,
    normals_world: torch.Tensor,
    link_meshes_world: Dict[str, trimesh.Trimesh],
    link_names: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_total = int(points_world.shape[0])
    if n_total == 0:
        raise RuntimeError("No union points to assign to robot links.")

    query_np = points_world.detach().cpu().numpy().astype(np.float64)
    dist_stack = np.full((len(link_names), n_total), np.inf, dtype=np.float64)
    for link_i, link_name in enumerate(link_names):
        mesh_world = link_meshes_world.get(link_name, None)
        if mesh_world is None or len(mesh_world.vertices) == 0:
            continue
        try:
            _, dist, _ = trimesh.proximity.closest_point(mesh_world, query_np)
            d = np.asarray(dist, dtype=np.float64).reshape(-1)
        except Exception:
            verts = np.asarray(mesh_world.vertices, dtype=np.float64)
            if verts.shape[0] == 0:
                continue
            tree = cKDTree(verts)
            d, _ = tree.query(query_np, k=1)
            d = np.asarray(d, dtype=np.float64).reshape(-1)
        if d.shape[0] == n_total:
            dist_stack[link_i] = d

    if not bool(np.isfinite(dist_stack).any()):
        raise RuntimeError("Failed to compute link assignment distances for union points.")

    link_idx_np = np.argmin(dist_stack, axis=0).astype(np.int64)
    link_idx = torch.tensor(link_idx_np, dtype=torch.long, device=model.device)

    points_local = torch.zeros_like(points_world)
    normals_local = torch.zeros_like(normals_world)
    for link_i, link_name in enumerate(link_names):
        sel = link_idx == int(link_i)
        if not bool(sel.any()):
            continue
        if link_name not in model.frame_status:
            continue
        tf = model.frame_status[link_name].get_matrix()[0].to(model.device)
        rot = tf[:3, :3]
        trans = tf[:3, 3].view(1, 3)
        points_local[sel] = (points_world[sel] - trans) @ rot
        normals_local[sel] = normals_world[sel] @ rot
    normals_local = normals_local / torch.norm(normals_local, dim=1, keepdim=True).clamp_min(1e-8)
    return points_local, normals_local, link_idx


def _build_concatenated_mesh(meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    if len(meshes) == 0:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))
    out = _as_mesh(trimesh.util.concatenate(meshes))
    if out is None:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))
    return trimesh.Trimesh(
        vertices=np.asarray(out.vertices, dtype=np.float64),
        faces=np.asarray(out.faces, dtype=np.int64),
        process=True,
    )


def _raycast_project_points_to_mesh(
    mesh_world: trimesh.Trimesh,
    query_points_world: np.ndarray,
    query_normals_world: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(query_points_world.shape[0])
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    q = np.asarray(query_points_world, dtype=np.float64)
    nrm = np.asarray(query_normals_world, dtype=np.float64)
    nrm_norm = np.linalg.norm(nrm, axis=1, keepdims=True)
    bad = nrm_norm.reshape(-1) < 1e-8
    if bool(np.any(bad)):
        nrm[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        nrm_norm = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.clip(nrm_norm, 1e-8, None)

    diag = float(np.linalg.norm(np.asarray(mesh_world.bounds[1] - mesh_world.bounds[0], dtype=np.float64)))
    eps = max(1e-5, 1e-4 * max(diag, 1e-6))

    origins_pos = q + eps * nrm
    dirs_pos = -nrm
    origins_neg = q - eps * nrm
    dirs_neg = nrm
    origins = np.concatenate([origins_pos, origins_neg], axis=0)
    dirs = np.concatenate([dirs_pos, dirs_neg], axis=0)

    best_pts = np.zeros((n, 3), dtype=np.float64)
    best_nrms = np.zeros((n, 3), dtype=np.float64)
    best_dist = np.full((n,), np.inf, dtype=np.float64)
    hit = np.zeros((n,), dtype=bool)

    try:
        loc, ray_idx, tri_idx = mesh_world.ray.intersects_location(
            origins,
            dirs,
            multiple_hits=False,
        )
        loc = np.asarray(loc, dtype=np.float64)
        ray_idx = np.asarray(ray_idx, dtype=np.int64).reshape(-1)
        tri_idx = np.asarray(tri_idx, dtype=np.int64).reshape(-1)
        face_normals = np.asarray(mesh_world.face_normals, dtype=np.float64)
        for k in range(int(ray_idx.shape[0])):
            rid = int(ray_idx[k])
            sid = rid if rid < n else (rid - n)
            if sid < 0 or sid >= n:
                continue
            p_hit = loc[k]
            dist = float(np.linalg.norm(p_hit - origins[rid]))
            if dist >= best_dist[sid]:
                continue
            best_dist[sid] = dist
            best_pts[sid] = p_hit
            tid = int(tri_idx[k])
            if tid >= 0 and tid < face_normals.shape[0]:
                best_nrms[sid] = face_normals[tid]
            hit[sid] = True
    except Exception:
        pass

    missing = ~hit
    if bool(np.any(missing)):
        q_miss = q[missing]
        try:
            cp, _, tri = trimesh.proximity.closest_point(mesh_world, q_miss)
            cp = np.asarray(cp, dtype=np.float64)
            tri = np.asarray(tri, dtype=np.int64).reshape(-1)
            best_pts[missing] = cp
            face_normals = np.asarray(mesh_world.face_normals, dtype=np.float64)
            miss_normals = np.zeros((cp.shape[0], 3), dtype=np.float64)
            valid = (tri >= 0) & (tri < face_normals.shape[0])
            if bool(np.any(valid)):
                miss_normals[valid] = face_normals[tri[valid]]
            if bool(np.any(~valid)):
                miss_normals[~valid] = nrm[missing][~valid]
            best_nrms[missing] = miss_normals
        except Exception:
            best_pts[missing] = q_miss
            best_nrms[missing] = nrm[missing]

    dots = np.sum(best_nrms * nrm, axis=1)
    flip = dots < 0.0
    if bool(np.any(flip)):
        best_nrms[flip] *= -1.0
    best_nrm_norm = np.linalg.norm(best_nrms, axis=1, keepdims=True)
    badn = best_nrm_norm.reshape(-1) < 1e-8
    if bool(np.any(badn)):
        best_nrms[badn] = nrm[badn]
        best_nrm_norm = np.linalg.norm(best_nrms, axis=1, keepdims=True)
    best_nrms = best_nrms / np.clip(best_nrm_norm, 1e-8, None)
    return best_pts, best_nrms


def _wrapper_outward_clearance_keep_mask(
    mesh_world: trimesh.Trimesh,
    points_world: torch.Tensor,
    normals_world: torch.Tensor,
    min_clearance: float,
) -> torch.Tensor:
    n = int(points_world.shape[0])
    if n == 0:
        return torch.zeros((0,), dtype=torch.bool, device=points_world.device)

    pts = points_world.detach().cpu().numpy().astype(np.float64)
    nrms = normals_world.detach().cpu().numpy().astype(np.float64)
    nrms_norm = np.linalg.norm(nrms, axis=1, keepdims=True)
    bad = nrms_norm.reshape(-1) < 1e-8
    if bool(np.any(bad)):
        nrms[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        nrms_norm = np.linalg.norm(nrms, axis=1, keepdims=True)
    nrms = nrms / np.clip(nrms_norm, 1e-8, None)

    diag = float(np.linalg.norm(np.asarray(mesh_world.bounds[1] - mesh_world.bounds[0], dtype=np.float64)))
    eps = max(1e-5, 1e-4 * max(diag, 1e-6))
    origins = pts + eps * nrms
    dirs = nrms

    best = np.full((n,), np.inf, dtype=np.float64)
    try:
        loc, ray_idx, _ = mesh_world.ray.intersects_location(
            origins,
            dirs,
            multiple_hits=False,
        )
        loc = np.asarray(loc, dtype=np.float64)
        ray_idx = np.asarray(ray_idx, dtype=np.int64).reshape(-1)
        for k in range(int(ray_idx.shape[0])):
            rid = int(ray_idx[k])
            if rid < 0 or rid >= n:
                continue
            d = float(np.linalg.norm(loc[k] - origins[rid]))
            if d < best[rid]:
                best[rid] = d
    except Exception:
        return torch.ones((n,), dtype=torch.bool, device=points_world.device)

    keep = (~np.isfinite(best)) | (best >= float(min_clearance))
    return torch.tensor(keep, dtype=torch.bool, device=points_world.device)


def _build_strict_surface_graph_from_union_mesh(
    union_mesh: trimesh.Trimesh,
    sample_points_world: np.ndarray,
    sample_normals_world: np.ndarray,
) -> List[List[int]]:
    n = int(sample_points_world.shape[0])
    strict_sets = [set() for _ in range(n)]
    if n <= 1:
        return [[] for _ in range(n)]

    verts = np.asarray(union_mesh.vertices, dtype=np.float64)
    edges = np.asarray(union_mesh.edges_unique, dtype=np.int64)
    if verts.shape[0] == 0 or edges.shape[0] == 0:
        return [[] for _ in range(n)]

    e0 = edges[:, 0].astype(np.int64)
    e1 = edges[:, 1].astype(np.int64)
    w = np.linalg.norm(verts[e0] - verts[e1], axis=1).astype(np.float64)
    valid_edge = np.isfinite(w) & (w > 0.0)
    if not bool(np.any(valid_edge)):
        return [[] for _ in range(n)]
    e0 = e0[valid_edge]
    e1 = e1[valid_edge]
    w = w[valid_edge]

    rows = np.concatenate([e0, e1], axis=0)
    cols = np.concatenate([e1, e0], axis=0)
    data = np.concatenate([w, w], axis=0)
    mesh_graph = coo_matrix((data, (rows, cols)), shape=(verts.shape[0], verts.shape[0])).tocsr()

    vertex_tree = cKDTree(verts)
    _, sample_vidx = vertex_tree.query(sample_points_world, k=1)
    sample_vidx = np.asarray(sample_vidx, dtype=np.int64).reshape(-1)
    sample_vidx = np.clip(sample_vidx, 0, max(0, verts.shape[0] - 1))

    point_tree = cKDTree(sample_points_world)
    d_nn, _ = point_tree.query(sample_points_world, k=2)
    d_nn = np.asarray(d_nn, dtype=np.float64)
    nn = d_nn[:, 1] if d_nn.ndim == 2 and d_nn.shape[1] > 1 else np.full((n,), np.inf, dtype=np.float64)
    finite_nn = nn[np.isfinite(nn) & (nn > 0.0)]
    if finite_nn.size == 0:
        return [[] for _ in range(n)]
    nn_med = float(np.median(finite_nn))
    nn_q90 = float(np.quantile(finite_nn, 0.90))
    geodesic_radius = max(2.60 * nn_med, 2.20 * nn_q90)
    geodesic_radius = max(geodesic_radius, float(np.median(w) * 6.0))
    geodesic_limit = float(1.35 * geodesic_radius)
    euclid_radius = max(1.90 * nn_q90, 2.20 * nn_med)
    same_anchor_euclid_radius = max(1.25 * nn_q90, 1.50 * nn_med)

    geo_to_vertices = dijkstra(
        csgraph=mesh_graph,
        directed=False,
        indices=sample_vidx,
        limit=geodesic_limit,
    )
    geo_sample = np.asarray(geo_to_vertices[:, sample_vidx], dtype=np.float64)

    normals = np.asarray(sample_normals_world, dtype=np.float64)
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    valid_normal = normal_norm.reshape(-1) > 1e-8
    normals[valid_normal] = normals[valid_normal] / normal_norm[valid_normal]
    normals[~valid_normal] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    max_neighbors = 24
    min_neighbors = 8
    n_angle_bins = 8

    def _tangent_basis(nrm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(ref, nrm))) > 0.85:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        t1 = np.cross(nrm, ref)
        t1_norm = float(np.linalg.norm(t1))
        if t1_norm < 1e-8:
            t1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            t1_norm = float(np.linalg.norm(t1))
        t1 = t1 / max(t1_norm, 1e-8)
        t2 = np.cross(nrm, t1)
        t2 = t2 / max(float(np.linalg.norm(t2)), 1e-8)
        return t1, t2

    for i in range(n):
        row = geo_sample[i]
        valid_geo = np.isfinite(row) & (row <= geodesic_radius)
        valid_geo[i] = False
        cand_geo = np.where(valid_geo)[0]
        if cand_geo.size == 0:
            continue

        same_anchor = cand_geo[row[cand_geo] <= 1e-12]
        if same_anchor.size > 0:
            d_same = np.linalg.norm(sample_points_world[same_anchor] - sample_points_world[i][None, :], axis=1)
            same_anchor = same_anchor[d_same <= same_anchor_euclid_radius]

        local = np.asarray(point_tree.query_ball_point(sample_points_world[i], r=float(euclid_radius)), dtype=np.int64)
        if local.ndim == 0:
            local = local.reshape(1)
        local = local[(local >= 0) & (local < n) & (local != i)]
        local = local[np.isfinite(row[local]) & (row[local] <= geodesic_radius)]

        cand = np.unique(np.concatenate([cand_geo, same_anchor, local], axis=0))
        if cand.size == 0:
            continue

        geo = row[cand]
        eu = np.linalg.norm(sample_points_world[cand] - sample_points_world[i][None, :], axis=1)
        order = np.lexsort((eu, geo))
        cand = cand[order]
        geo = geo[order]

        keep_list: List[int] = []
        used = set()
        t1, t2 = _tangent_basis(normals[i])
        best_by_bin: Dict[int, Tuple[float, int]] = {}
        for c_idx, j in enumerate(cand.tolist()):
            v = sample_points_world[j] - sample_points_world[i]
            v_tan = v - normals[i] * float(np.dot(v, normals[i]))
            v_norm = float(np.linalg.norm(v_tan))
            if v_norm < 1e-10:
                continue
            x = float(np.dot(v_tan, t1))
            y = float(np.dot(v_tan, t2))
            ang = math.atan2(y, x)
            b = int(math.floor((ang + math.pi) / (2.0 * math.pi / float(n_angle_bins))))
            b = max(0, min(n_angle_bins - 1, b))
            g = float(geo[c_idx])
            prev = best_by_bin.get(b, None)
            if (prev is None) or (g < prev[0]):
                best_by_bin[b] = (g, int(j))

        for _, (_, j) in sorted(best_by_bin.items(), key=lambda kv: kv[1][0]):
            if j in used:
                continue
            keep_list.append(int(j))
            used.add(int(j))
            if len(keep_list) >= max_neighbors:
                break

        for j in cand.tolist():
            if len(keep_list) >= max_neighbors:
                break
            if int(j) in used:
                continue
            keep_list.append(int(j))
            used.add(int(j))

        if len(keep_list) < min_neighbors:
            cand_all = np.where(np.isfinite(row) & (np.arange(n) != i))[0]
            if cand_all.size > 0:
                geo_all = row[cand_all]
                eu_all = np.linalg.norm(sample_points_world[cand_all] - sample_points_world[i][None, :], axis=1)
                order_all = np.lexsort((eu_all, geo_all))
                for j in cand_all[order_all].tolist():
                    if int(j) in used:
                        continue
                    if not np.isfinite(row[int(j)]):
                        continue
                    keep_list.append(int(j))
                    used.add(int(j))
                    if len(keep_list) >= min_neighbors:
                        break

        keep = np.asarray(keep_list[:max_neighbors], dtype=np.int64)
        for j in keep.tolist():
            if i == j:
                continue
            strict_sets[i].add(int(j))
            strict_sets[int(j)].add(i)

    for i in range(n):
        if len(strict_sets[i]) > 0:
            continue
        _, nn_idx = point_tree.query(sample_points_world[i], k=2)
        nn_idx = np.asarray(nn_idx, dtype=np.int64).reshape(-1)
        if nn_idx.size < 2:
            continue
        j = int(nn_idx[1])
        if i == j:
            continue
        strict_sets[i].add(j)
        strict_sets[j].add(i)

    return [sorted(s) for s in strict_sets]


def _build_surface_artifact(model: Any, num_points: int, seed: int) -> Dict[str, Any]:
    if int(num_points) <= 0:
        raise ValueError("num_points must be positive.")

    link_names, link_meshes_world = model._get_canonical_link_meshes_world()
    robot_scale = _robot_scale(list(link_meshes_world.values()))
    wrapper_pitch = float(_surface_proxy_pitch_from_scale(robot_scale))

    union_mesh = _build_boolean_union_mesh(
        robot_name=str(model.robot_name),
        meshes_world=list(link_meshes_world.values()),
    )
    sample_candidates = int(max(int(num_points) * 6, int(num_points)))
    simplified_points_world, simplified_normals_world = _sample_surface_points_normals_from_mesh(
        mesh=union_mesh,
        num_points=sample_candidates,
        device=model.device,
        oversample_ratio=2,
    )
    keep_mask = _wrapper_outward_clearance_keep_mask(
        mesh_world=union_mesh,
        points_world=simplified_points_world,
        normals_world=simplified_normals_world,
        min_clearance=float(max(4.0 * wrapper_pitch, 2.5e-3)),
    )
    if int(keep_mask.sum().item()) >= int(num_points):
        simplified_points_world = simplified_points_world[keep_mask]
        simplified_normals_world = simplified_normals_world[keep_mask]

    if int(simplified_points_world.shape[0]) > int(num_points):
        _, keep = _farthest_point_sampling(simplified_points_world, int(num_points))
        keep = keep.long()
        simplified_points_world = simplified_points_world[keep]
        simplified_normals_world = simplified_normals_world[keep]
    elif int(simplified_points_world.shape[0]) < int(num_points):
        extra_n = int(num_points - simplified_points_world.shape[0])
        if int(simplified_points_world.shape[0]) == 0:
            raise RuntimeError(f"Outer-envelope filtering removed all points for robot: {model.robot_name}")
        extra_idx = torch.randint(0, int(simplified_points_world.shape[0]), (extra_n,), device=model.device)
        simplified_points_world = torch.cat([simplified_points_world, simplified_points_world[extra_idx]], dim=0)
        simplified_normals_world = torch.cat([simplified_normals_world, simplified_normals_world[extra_idx]], dim=0)

    original_full_mesh = _build_concatenated_mesh(list(link_meshes_world.values()))
    if len(original_full_mesh.faces) > 0:
        proj_pts_np, proj_nrms_np = _raycast_project_points_to_mesh(
            mesh_world=original_full_mesh,
            query_points_world=simplified_points_world.detach().cpu().numpy().astype(np.float64),
            query_normals_world=simplified_normals_world.detach().cpu().numpy().astype(np.float64),
        )
        proj_pts = torch.tensor(proj_pts_np, dtype=torch.float32, device=model.device)
        proj_nrms = torch.tensor(proj_nrms_np, dtype=torch.float32, device=model.device)

        proj_shift = torch.norm(proj_pts - simplified_points_world, dim=1)
        max_proj_shift = float(max(4.0 * wrapper_pitch, 2e-3))
        nrm_dot = torch.sum(proj_nrms * simplified_normals_world, dim=1)
        use_proj = (proj_shift <= max_proj_shift) & (nrm_dot >= 0.2)

        template_points_world = torch.where(use_proj.unsqueeze(1), proj_pts, simplified_points_world)
        template_normals_world = torch.where(use_proj.unsqueeze(1), proj_nrms, simplified_normals_world)
        template_normals_world = template_normals_world / torch.norm(
            template_normals_world,
            dim=1,
            keepdim=True,
        ).clamp_min(1e-8)
    else:
        template_points_world = simplified_points_world
        template_normals_world = simplified_normals_world

    points_local, normals_local, link_idx = _assign_union_points_to_links(
        model=model,
        points_world=template_points_world,
        normals_world=template_normals_world,
        link_meshes_world=link_meshes_world,
        link_names=link_names,
    )
    strict_neighbors = _build_strict_surface_graph_from_union_mesh(
        union_mesh=union_mesh,
        sample_points_world=simplified_points_world.detach().cpu().numpy().astype(np.float64),
        sample_normals_world=simplified_normals_world.detach().cpu().numpy().astype(np.float64),
    )

    return {
        "meta": {
            "version": SURFACE_ARTIFACT_VERSION,
            "robot_name": str(model.robot_name),
            "robot_path": str(model.robot_path),
            "seed": int(seed),
            "num_surface_points": int(num_points),
        },
        "link_names": [str(x) for x in link_names],
        "surface_template_points_local": points_local.detach().cpu(),
        "surface_template_normals_local": normals_local.detach().cpu(),
        "surface_template_link_indices": link_idx.detach().cpu(),
        "surface_template_points_canonical_world": template_points_world.detach().cpu(),
        "surface_template_normals_canonical_world": template_normals_world.detach().cpu(),
        "surface_graph_points_canonical_world": simplified_points_world.detach().cpu(),
        "surface_graph_normals_canonical_world": simplified_normals_world.detach().cpu(),
        "surface_graph_neighbors_strict": strict_neighbors,
        "surface_graph_neighbors": strict_neighbors,
        "surface_union_mesh_vertices": torch.tensor(
            np.asarray(union_mesh.vertices, dtype=np.float32),
            dtype=torch.float32,
        ),
        "surface_union_mesh_faces": torch.tensor(
            np.asarray(union_mesh.faces, dtype=np.int64),
            dtype=torch.long,
        ),
    }


def _validate_surface_artifact(artifact: Dict[str, Any], model: Any, num_points: int, seed: int) -> bool:
    if not isinstance(artifact, dict):
        return False
    meta = artifact.get("meta", {})
    if not isinstance(meta, dict):
        return False
    if str(meta.get("version", "")) != SURFACE_ARTIFACT_VERSION:
        return False
    if str(meta.get("robot_name", "")) != str(model.robot_name):
        return False
    if int(meta.get("num_surface_points", -1)) != int(num_points):
        return False
    if int(meta.get("seed", -1)) != int(seed):
        return False

    required = [
        "surface_template_points_local",
        "surface_template_normals_local",
        "surface_template_link_indices",
        "surface_template_points_canonical_world",
        "surface_template_normals_canonical_world",
        "surface_graph_points_canonical_world",
        "surface_graph_normals_canonical_world",
        "surface_graph_neighbors",
        "surface_union_mesh_vertices",
        "surface_union_mesh_faces",
        "link_names",
    ]
    for k in required:
        if k not in artifact:
            return False
    n = int(torch.as_tensor(artifact["surface_template_link_indices"]).numel())
    if n <= 0:
        return False
    if int(torch.as_tensor(artifact["surface_template_points_local"]).shape[0]) != n:
        return False
    if int(torch.as_tensor(artifact["surface_template_normals_local"]).shape[0]) != n:
        return False
    if len(artifact["surface_graph_neighbors"]) != n:
        return False
    return True


def load_or_build_surface_artifact(
    model: Any,
    num_points: int,
    seed: int,
    cache_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Path, bool]:
    path = Path(cache_path) if cache_path is not None else get_surface_artifact_path(
        robot_path=Path(model.robot_path),
        robot_name=str(model.robot_name),
        num_points=int(num_points),
    )
    artifact = None
    if path.exists():
        try:
            loaded = torch.load(path, map_location="cpu")
            if _validate_surface_artifact(loaded, model=model, num_points=num_points, seed=seed):
                artifact = loaded
        except Exception:
            artifact = None

    built = False
    if artifact is None:
        artifact = _build_surface_artifact(model=model, num_points=int(num_points), seed=int(seed))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, path)
        built = True

    return artifact, path, built


def apply_surface_artifact_to_model(model: Any, artifact: Dict[str, Any]) -> None:
    device = model.device
    link_names = [str(x) for x in list(artifact["link_names"])]

    model.surface_template_points_local = torch.as_tensor(
        artifact["surface_template_points_local"],
        dtype=torch.float32,
        device=device,
    )
    model.surface_template_normals_local = torch.as_tensor(
        artifact["surface_template_normals_local"],
        dtype=torch.float32,
        device=device,
    )
    model.surface_template_link_indices = torch.as_tensor(
        artifact["surface_template_link_indices"],
        dtype=torch.long,
        device=device,
    )
    model.surface_template_points_canonical_world = torch.as_tensor(
        artifact["surface_template_points_canonical_world"],
        dtype=torch.float32,
        device=device,
    )
    model.surface_template_normals_canonical_world = torch.as_tensor(
        artifact["surface_template_normals_canonical_world"],
        dtype=torch.float32,
        device=device,
    )
    model.surface_graph_points_canonical_world = torch.as_tensor(
        artifact["surface_graph_points_canonical_world"],
        dtype=torch.float32,
        device=device,
    )
    model.surface_graph_normals_canonical_world = torch.as_tensor(
        artifact["surface_graph_normals_canonical_world"],
        dtype=torch.float32,
        device=device,
    )

    neighbors = artifact.get("surface_graph_neighbors", [])
    neighbors_strict = artifact.get("surface_graph_neighbors_strict", neighbors)
    model.surface_graph_neighbors = [[int(v) for v in row] for row in neighbors]
    model.surface_graph_neighbors_strict = [[int(v) for v in row] for row in neighbors_strict]

    union_v = torch.as_tensor(artifact["surface_union_mesh_vertices"], dtype=torch.float32).cpu().numpy()
    union_f = torch.as_tensor(artifact["surface_union_mesh_faces"], dtype=torch.long).cpu().numpy().astype(np.int64)
    model.surface_union_mesh_canonical = trimesh.Trimesh(
        vertices=np.asarray(union_v, dtype=np.float64),
        faces=np.asarray(union_f, dtype=np.int64),
        process=False,
    )
    if hasattr(trimesh.repair, "fix_normals"):
        trimesh.repair.fix_normals(model.surface_union_mesh_canonical)

    by_link: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for link_i, link_name in enumerate(link_names):
        mask = model.surface_template_link_indices == int(link_i)
        by_link[link_name] = (
            model.surface_template_points_local[mask],
            model.surface_template_normals_local[mask],
        )
    model.surface_template_by_link = by_link

