"""Minimal chain model tailored to chain_generation.py outputs.

Supports:
- Load a generated URDF (boxes, cylinders, or capsules) via pytorch_kinematics.
- Sample joint configs, forward points, export a merged mesh, and visualize in Viser.
"""

from __future__ import annotations

import math
import colorsys
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import trimesh
import pytorch_kinematics as pk
import viser
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.ops import sample_farthest_points


def farthest_point_sampling(pos: torch.Tensor,
                            n_sampling: int,
                            *,
                            return_indices_only: bool = False,
                            random_start: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Batch-aware farthest point sampling using PyTorch3D."""
    if pos.ndim == 2:
        pos = pos.unsqueeze(0)
        squeeze = True
    elif pos.ndim == 3:
        squeeze = False
    else:
        raise ValueError('pos must be a tensor of shape (N, C) or (B, N, C).')

    if n_sampling <= 0:
        raise ValueError('n_sampling must be positive.')

    coords = pos[..., :3].contiguous()
    max_points = coords.shape[-2]
    k = min(n_sampling, max_points)

    sampled_coords, sampled_idx = sample_farthest_points(coords, K=k, random_start_point=random_start)

    if pos.shape[-1] > 3:
        gather_idx = sampled_idx[..., :, None].expand(*sampled_idx.shape, pos.shape[-1])
        sampled_points = torch.gather(pos, -2, gather_idx)
    else:
        sampled_points = sampled_coords

    if squeeze:
        sampled_points = sampled_points.squeeze(0)
        sampled_idx = sampled_idx.squeeze(0)

    if return_indices_only:
        return sampled_idx

    return sampled_points, sampled_idx


def generate_distinct_colors(n: int, saturation: float = 0.65, value: float = 0.9) -> List[Tuple[float, float, float]]:
    if n <= 0:
        return []
    golden_ratio_conjugate = (math.sqrt(5) - 1) / 2
    hue = 0.0
    colors = []
    for _ in range(n):
        hue = (hue + golden_ratio_conjugate) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors


def _parse_meshes_from_urdf(urdf_path: str, device: torch.device) -> Tuple[Dict[str, trimesh.Trimesh], Dict[str, Dict[str, torch.Tensor]]]:
    """Parse link geometries (box/capsule/cylinder) and return mesh plus analytic params.

    Returns a tuple: (meshes, geom_specs) where geom_specs[link] stores ``type``,
    ``params`` (tensor), and ``tf`` (4x4 transform from link frame to geom frame).
    """

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    meshes: Dict[str, trimesh.Trimesh] = {}
    geom_specs: Dict[str, Dict[str, torch.Tensor]] = {}

    for link in root.findall('link'):
        name = link.get('name')
        visual = link.find('visual')
        if visual is None:
            continue
        geometry = visual.find('geometry')
        if geometry is None:
            continue

        box = geometry.find('box')
        cylinder = geometry.find('cylinder')
        capsule = geometry.find('capsule')

        geom_tf = np.eye(4)
        geom_type: Optional[str] = None
        params: Optional[torch.Tensor] = None

        if box is not None:
            sx, sy, sz = map(float, box.get('size').split())
            mesh = trimesh.creation.box(extents=(sx, sy, sz))
            geom_type = 'box'
            params = torch.tensor([sx * 0.5, sy * 0.5, sz * 0.5], dtype=torch.float32, device=device)
        elif cylinder is not None:
            radius = float(cylinder.get('radius'))
            length = float(cylinder.get('length'))
            mesh = trimesh.creation.cylinder(radius=radius, height=length)
            rot = trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0])  # align axis to +X
            mesh.apply_transform(rot)
            geom_tf = rot @ geom_tf
            geom_type = 'cylinder'
            params = torch.tensor([length * 0.5, radius], dtype=torch.float32, device=device)  # half-length, radius
        elif capsule is not None:
            radius = float(capsule.get('radius'))
            length = float(capsule.get('length'))
            mesh = trimesh.creation.capsule(radius=radius, height=length)
            rot = trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0])  # align axis to +X
            # word_frame = trimesh.creation.axis(origin_size=0.02)
            mesh.apply_transform(rot)
            # Derive half-length from the actual mesh extent to stay consistent with URDF semantics.
            # geom_tf = rot @ geom_tf
            geom_type = 'capsule'
            params = torch.tensor([length, radius], dtype=torch.float32, device=device)  # cyl half-length, radius
        else:
            continue

        origin = visual.find('origin')
        if origin is not None and origin.get('xyz') is not None:
            ox, oy, oz = map(float, origin.get('xyz').split())
            trans = trimesh.transformations.translation_matrix([ox, oy, oz])
            mesh.apply_transform(trans)
            geom_tf = trans @ geom_tf
        else:
            raise KeyError(f"Visual origin missing for link '{name}' in URDF.")

        if geom_type is None or params is None:
            continue

        meshes[name] = mesh
        geom_specs[name] = {
            'type': geom_type,
            'params': params,
            'tf': torch.tensor(geom_tf, dtype=torch.float32, device=device),
        }

    return meshes, geom_specs


def _parse_joint_axes_from_urdf(urdf_path: str, device: torch.device) -> List[Dict[str, torch.Tensor]]:
    """Collect joint axis vectors and their child links from URDF."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints: List[Dict[str, torch.Tensor]] = []
    for joint in root.findall('joint'):
        joint_type = joint.get('type', '')
        if joint_type not in ('revolute', 'continuous', 'prismatic'):
            continue

        axis_el = joint.find('axis')
        if axis_el is not None and axis_el.get('xyz') is not None:
            axis = np.fromstring(axis_el.get('xyz'), sep=' ', dtype=float)
        else:
            axis = np.array([1.0, 0.0, 0.0], dtype=float)
            print(f"Warning: Joint '{joint.get('name')}' has no axis specified; defaulting to [1,0,0].")

        child_el = joint.find('child')
        if child_el is None:
            continue
        child_name = child_el.get('link')
        joints.append({
            'name': joint.get('name', child_name),
            'child': child_name,
            'axis': torch.as_tensor(axis, dtype=torch.float32, device=device),
        })

    return joints

class ChainModel:
    def __init__(
        self,
        urdf_path: str | Path,
        *,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        samples_per_link: int = 256,
    ) -> None:
        self.urdf_path = str(urdf_path)
        self.device = device

        self.pk_chain = pk.build_chain_from_urdf(open(self.urdf_path).read()).to(dtype=torch.float32, device=device)
        self.dof = len(self.pk_chain.get_joint_parameter_names())

        self.meshes, self.geom_specs = _parse_meshes_from_urdf(self.urdf_path, device)
        self.joint_axes = _parse_joint_axes_from_urdf(self.urdf_path, device)
        self.link_points = self._sample_rest_pose_points(self.meshes, samples_per_link)
        self.frame_status = None
        self.num_links = len(self.meshes)

    # Utilities
    def sample_q(self, B: int = 1) -> torch.Tensor:
        try:
            lower, upper = self.pk_chain.get_joint_limits()
            lower_t = torch.tensor(lower, dtype=torch.float32, device=self.device)
            upper_t = torch.tensor(upper, dtype=torch.float32, device=self.device)
            rand = torch.rand(B, self.dof, device=self.device)
            return lower_t * (1 - rand) + upper_t * rand
        except Exception:
            # Fallback: no limits available; return zeros.
            return torch.zeros((B, self.dof), dtype=torch.float32, device=self.device)

    def update_status(self, q: torch.Tensor) -> None:
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if q.shape[-1] != self.dof:
            raise ValueError(f"Expected last dim {self.dof}, got {q.shape[-1]}")
        self.q = q.to(self.device)
        self.frame_status = self.pk_chain.forward_kinematics(self.q)

    # SDF sampling and queries
    def sample_query_points(
        self,
        n: int,
        var: float = 0.0025,
        near_surface_ratio: float = 0.95,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample query points for each pose in the current batch.

        Args:
            n: total number of query points per batch element.
            var: variance used for near-surface Gaussian noise (mirrors data_generate_samples.py).
            near_surface_ratio: fraction of points sampled near the surface; rest are uniform in a bounding sphere.

        Returns:
            (B, n, 3) tensor on ``self.device`` matching the batch size of the last ``update_status`` call.
        """

        if self.frame_status is None:
            raise RuntimeError('Call update_status() before sampling query points.')

        if not (0.0 < near_surface_ratio < 1.0):
            raise ValueError('near_surface_ratio must be in (0, 1).')

        n_near = int(n * near_surface_ratio)
        n_uniform = n - n_near

        pc_full = self.get_transformed_links_pc(num_points=None, mask=mask)  # (B, N, 4)
        B, M, _ = pc_full.shape

        if M == 0:
            raise ValueError('Mask removed all links; no points to sample from.')

        all_batches = []
        for b in range(B):
            pts = pc_full[b, :, :3]

            radius = float(pts.norm(dim=-1).max().item())
            base_radius = 0.9
            scale_ratio = radius / base_radius if base_radius > 0 else 1.0

            sigma1 = math.sqrt(var) * scale_ratio
            sigma2 = math.sqrt(var / 10.0) * scale_ratio

            # Sample base surface points with replacement to allow large n.
            idx = torch.randint(0, M, (max(1, n_near),), device=self.device)
            base = pts[idx]

            near_pts = torch.cat([
                base + torch.randn_like(base) * sigma1,
                base + torch.randn_like(base) * sigma2,
            ], dim=0)[:n_near]

            if n_uniform > 0:
                dir_vec = torch.randn(n_uniform, 3, device=self.device)
                dir_vec = dir_vec / dir_vec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                radii = torch.rand(n_uniform, 1, device=self.device) ** (1.0 / 3.0)
                uniform_pts = dir_vec * radii * radius
                all_pts = torch.cat([near_pts, uniform_pts], dim=0)
            else:
                all_pts = near_pts

            all_batches.append(all_pts)

        return torch.stack(all_batches, dim=0)

    def query_sdf(self, points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute signed distance for ``points`` (B,N,3 or N,3) to the articulated chain.

        Uses analytic SDFs per link geometry (box/cylinder/capsule) and returns the
        minimum distance over links for each point.
        """

        if self.frame_status is None:
            raise RuntimeError('Call update_status() before querying SDF.')

        if points.dim() == 2:
            points = points.unsqueeze(0)
        if points.size(0) != self.q.size(0):
            raise ValueError(f'Batch mismatch: points batch {points.size(0)} vs q batch {self.q.size(0)}')

        B, N, _ = points.shape
        device = self.device
        dtype = points.dtype
        points = points.to(device)

        link_names = list(self.geom_specs.keys())
        mask_list = None
        if mask is not None:
            mask_bool = mask.to(dtype=torch.bool)
            if mask_bool.numel() != len(link_names):
                raise ValueError(f"Mask length {mask_bool.numel()} != number of links {len(link_names)}")
            mask_list = mask_bool.flatten().tolist()

        sdf_all_links = []
        for idx, (link_name, spec) in enumerate(self.geom_specs.items()):
            if mask_list is not None and not mask_list[idx]:
                continue
            tf_world = self.frame_status[link_name].get_matrix().to(device=device, dtype=dtype)  # world_T_link (B,4,4)
            tf_world_inv = torch.linalg.inv(tf_world)  # link_T_world

            tf_geom = spec['tf'].to(device=device, dtype=dtype)  # link_T_geom (4,4)
            tf_geom_inv = torch.linalg.inv(tf_geom)              # geom_T_link (4,4)

            # world -> link -> geom
            homog = torch.cat([points, torch.ones(B, N, 1, device=device, dtype=dtype)], dim=-1)  # (B,N,4)
            p_link = torch.matmul(homog, tf_world_inv.transpose(1, 2))
            p_geom = torch.matmul(p_link, tf_geom_inv.transpose(0, 1))[:, :, :3]

            gtype = spec['type']
            params = spec['params'].to(device=device, dtype=dtype)

            if gtype == 'box':
                half_extents = params  # (3,)
                qv = p_geom.abs() - half_extents
                outside = torch.clamp(qv, min=0.0)
                outside_norm = torch.linalg.norm(outside, dim=-1)
                inside = torch.clamp(qv.max(dim=-1).values, max=0.0)
                sdf = outside_norm + inside
            elif gtype == 'capsule':
                height, radius = params[0], params[1]
                px = p_geom[..., 0].clamp(-height * 0.5, height * 0.5)
                closest = torch.stack([px, torch.zeros_like(px), torch.zeros_like(px)], dim=-1)
                sdf = torch.linalg.norm(p_geom - closest, dim=-1) - radius
            elif gtype == 'cylinder':
                raise ValueError(f'Unsupported geometry type: {gtype}')
                half_len, radius = params[0], params[1]
                radial = torch.linalg.norm(p_geom[..., 1:], dim=-1) - radius
                axial = p_geom[..., 0].abs() - half_len
                h = torch.stack([radial, axial], dim=-1)
                outside = torch.clamp(h, min=0.0)
                outside_norm = torch.linalg.norm(outside, dim=-1)
                inside = torch.clamp(h.max(dim=-1).values, max=0.0)
                sdf = outside_norm + inside
            else:
                raise ValueError(f'Unsupported geometry type: {gtype}')

            sdf_all_links.append(sdf)

        if not sdf_all_links:
            raise RuntimeError('No link geometries parsed; cannot query SDF.')

        sdf_stack = torch.stack(sdf_all_links, dim=-1)  # (B,N,L)
        sdf_min, _ = sdf_stack.min(dim=-1)
        return sdf_min

    # Points
    def _sample_rest_pose_points(self, mesh_dict: Dict[str, trimesh.Trimesh], samples_per_link: int) -> Dict[str, torch.Tensor]:
        points: Dict[str, torch.Tensor] = {}
        for name, mesh in mesh_dict.items():
            sampled = mesh.sample(samples_per_link)
            points[name] = torch.as_tensor(sampled, dtype=torch.float32, device=self.device)
        return points

    def get_transformed_links_pc(
        self,
        links_pc: Optional[Dict[str, torch.Tensor]] = None,
        num_points: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if links_pc is None:
            links_pc = self.link_points

        batch_size = self.q.shape[0]
        all_pc = []
        link_names = list(links_pc.keys())

        if mask is not None:
            mask_bool = mask.to(dtype=torch.bool)
            if mask_bool.numel() != len(link_names):
                raise ValueError(f"Mask length {mask_bool.numel()} != number of links {len(link_names)}")
            mask_flags = mask_bool.flatten().tolist()
        else:
            mask_flags = [True] * len(link_names)

        for link_idx, link_name in enumerate(link_names):
            if not mask_flags[link_idx]:
                continue
            local_pts = links_pc[link_name].to(self.device)
            n_pts = local_pts.shape[0]
            hom = torch.cat([local_pts, torch.ones((n_pts, 1), device=self.device, dtype=torch.float32)], dim=-1)

            tf = self.frame_status[link_name].get_matrix()  # (B,4,4)
            world = torch.matmul(hom.unsqueeze(0), tf.transpose(1, 2))[..., :3]
            idx_col = torch.full((batch_size, n_pts, 1), float(link_idx), device=self.device)
            all_pc.append(torch.cat([world, idx_col], dim=-1))

        pc = torch.cat(all_pc, dim=1)
        if num_points is not None and pc.shape[1] > num_points and farthest_point_sampling is not None:
            pc, _ = farthest_point_sampling(pc, num_points)
        return pc

    # Mesh export
    def get_trimesh_q(self, idx: int, boolean_merged: bool=True, mask: Optional[torch.Tensor] = None) -> trimesh.Trimesh:
        """Return mesh for configuration ``idx``; optionally boolean-merge to avoid overlaps."""
        link_names = list(self.meshes.keys())
        if mask is not None:
            mask_bool = mask.to(dtype=torch.bool)
            if mask_bool.numel() != len(link_names):
                raise ValueError(f"Mask length {mask_bool.numel()} != number of links {len(link_names)}")
            mask_flags = mask_bool.flatten().tolist()
        else:
            mask_flags = [True] * len(link_names)

        transformed_meshes = []
        for link_idx, (link_name, mesh) in enumerate(self.meshes.items()):
            if not mask_flags[link_idx]:
                continue
            tf = self.frame_status[link_name].get_matrix()[idx].cpu().numpy()
            m = mesh.copy()
            m.apply_transform(tf)
            transformed_meshes.append(m)
        if boolean_merged:
            try:
                merged = trimesh.boolean.union(transformed_meshes)
                if merged is not None:
                    return merged
            except Exception as exc:
                print(f"[get_trimesh_q] boolean union failed, falling back to concat: {exc}")

        return trimesh.util.concatenate(transformed_meshes)

    def get_joint_axes_world(self, axis_length: float = 0.08) -> List[Dict[str, torch.Tensor]]:
        """Return world-frame joint axes for the current configuration.

        Call ``update_status`` before this to ensure ``frame_status`` is valid.
        """
        if self.frame_status is None:
            raise RuntimeError('Call update_status() before requesting joint axes.')

        axes_world: List[Dict[str, torch.Tensor]] = []
        for joint in self.joint_axes:
            child_name = joint['child']
            tf = self.frame_status[child_name].get_matrix()  # (B,4,4)
            rot = tf[:, :3, :3]
            pos = tf[:, :3, 3]

            axis_local = joint['axis'].to(self.device).view(1, 3, 1).expand(tf.shape[0], -1, -1)
            axis_world = torch.matmul(rot, axis_local).squeeze(-1)
            axis_world = axis_world / torch.linalg.norm(axis_world, dim=1, keepdim=True).clamp_min(1e-8)

            axes_world.append({
                'name': joint['name'],
                'start': pos,
                'direction': axis_world,
                'length': float(axis_length),
            })

        return axes_world

def visualize_chain_pc(
        pc: torch.Tensor,
        link_names: List[str],
        point_size: float = 0.003,
        mesh: trimesh.Trimesh = None,
        joint_axes: Optional[List[Dict[str, torch.Tensor]]] = None,
        axis_radius: float = 0.003,
        host: str = '127.0.0.1',
        port: int = 9100,
    ) -> viser.ViserServer:
    if pc.shape[0] != 1:
        raise ValueError('visualize_chain expects a single configuration (batch=1).')
    pc = pc[0]

    server = viser.ViserServer(host=host, port=port)
    server.scene.add_frame('world_frame', show_axes=['x', 'y', 'z'], axes_length=0.08, axes_radius=0.004, position=(0, 0, 0))

    colors = generate_distinct_colors(len(link_names))

    pts_np = pc[:, :3].detach().cpu().numpy()
    link_idx = pc[:, 3].long()
    for idx in range(len(link_names)):
        mask = (link_idx == idx)
        server.scene.add_point_cloud(
            f'link_{idx}_points',
            pts_np[mask.cpu().numpy()],
            point_size=point_size,
            point_shape='circle',
            colors=colors[idx],
        )
    if mesh is not None:
        verts = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.int32)
        server.scene.add_mesh_simple(
            name='chain_mesh',
            vertices=verts,
            faces=faces,
            color=(0.7, 0.7, 0.9),
            )

    if joint_axes is not None:
        for j_idx, joint in enumerate(joint_axes):
            start = joint['start'][0]
            direction = joint['direction'][0]
            direction = direction / direction.norm().clamp_min(1e-8)

            # Build an orientation whose X-axis aligns with the joint axis.
            up = torch.tensor([0.0, 0.0, 1.0], device=direction.device)
            if torch.abs(torch.dot(direction, up)) > 0.95:
                up = torch.tensor([0.0, 1.0, 0.0], device=direction.device)
            y_axis = torch.cross(up, direction)
            y_axis = y_axis / y_axis.norm().clamp_min(1e-8)
            z_axis = torch.cross(direction, y_axis)
            rot = torch.stack([direction, y_axis, z_axis], dim=1)  # columns
            quat = matrix_to_quaternion(rot.unsqueeze(0)).squeeze(0)

            server.scene.add_frame(
                f'joint_axis_{j_idx}_{joint["name"]}',
                position=start.detach().cpu().numpy(),
                wxyz=quat.detach().cpu().numpy(),
                show_axes=['x'],
                axes_length=float(joint['length']),
                axes_radius=axis_radius,
            )
    return server

def visualize_sdf_viser(
    mesh: trimesh.Trimesh,
    sdf_samples: np.ndarray,
    downsample_ratio: float = 0.1,
    max_points: int = 50000,
    marker_size: float = 0.003,
    seed: int = 0,
    *,
    host: str = "127.0.0.1",
    port: int = 9300,
) -> viser.ViserServer:
    """Visualize a mesh with SDF points using Viser.

    Matches the signature of ``visualize_samples.visualize_sdf`` but renders via
    Viser instead of matplotlib. Points are downsampled, colored blue for
    positive SDF and red for negative, with intensity based on |sdf|.
    """

    rng = np.random.default_rng(seed)
    n_total = len(sdf_samples)
    keep = int(n_total * downsample_ratio)
    if max_points is not None:
        keep = min(keep, max_points)
    keep = max(1, keep) if n_total > 0 else 0
    if keep == 0:
        raise ValueError("No points available for visualization")
    if keep < n_total:
        idx = rng.choice(n_total, size=keep, replace=False)
        sdf_samples = sdf_samples[idx]

    xyz = sdf_samples[:, :3]
    sdf = sdf_samples[:, 3]

    pos_mask = sdf >= 0
    neg_mask = ~pos_mask

    server = viser.ViserServer(host=host, port=port)

    # Make the mesh translucent so interior points remain visible.
    mesh_vis = mesh.copy()
    mesh_vis.visual.face_colors = [200, 200, 210, 80]  # RGBA with alpha ~0.31
    server.scene.add_mesh_trimesh(name="mesh", mesh=mesh_vis)

    def _add_points(mask: np.ndarray, base_color: Tuple[float, float, float], name: str) -> None:
        count = int(mask.sum())
        if count == 0:
            return
        vals = np.abs(sdf[mask])

        # Normalize with a percentile to reduce dominance of a few far samples.
        scale = np.percentile(vals, 99.0)
        vals = vals / (scale + 1e-9)
        vals = np.clip(vals, 0.0, 1.0)

        # Lift near-surface values so they are still visible and bias toward bright ends.
        min_intensity = 0.18
        gamma = 0.6
        intensities = min_intensity + (1.0 - min_intensity) * np.power(vals, gamma)

        colors = np.stack([
            base_color[0] * intensities,
            base_color[1] * intensities,
            base_color[2] * intensities,
        ], axis=-1)
        server.scene.add_point_cloud(
            name=name,
            points=xyz[mask],
            colors=colors,
            point_size=marker_size,
        )

    _add_points(pos_mask, (0.2, 0.6, 1.0), "sdf_pos")
    _add_points(neg_mask, (1.0, 0.0, 0.0), "sdf_neg")

    return server

def generate_linkage_datasets(
        urdf_path: str | Path,
        data_save_dir: str | Path,
        B: int = 10,
        visualize: bool = False,
    ) -> Tuple[torch.Tensor, trimesh.Trimesh]:
    """Generate point cloud dataset for a chain model defined by the URDF.

    Returns:
        pc: (B, N, 4) point clouds with link indices in last channel.
        merged_mesh: boolean-merged trimesh of the entire chain at rest pose.
    """
    model = ChainModel(
        urdf_path,
        samples_per_link=256,
    )
    q_sample = model.sample_q(B=B)
    # q_sample = torch.zeros_like(q_sample)  # all zeros for testing
    link_names = list(model.link_points.keys())
    
    model.update_status(q_sample)
    meshes = []
    for b in range(B):
        meshes.append(model.get_trimesh_q(b, boolean_merged=True))
        if visualize:
            joint_axes = model.get_joint_axes_world(axis_length=0.06)
            chain_pc = model.get_transformed_links_pc(None, num_points=256)
            server = visualize_chain_pc(
                chain_pc,
                link_names,
                mesh=meshes[-1],
                joint_axes=joint_axes,
                host='127.0.0.1',
                port=9100,
            )
    
    for idx, mesh in enumerate(meshes):
        mesh.export(Path(data_save_dir) / f"{Path(urdf_path).stem}_{idx}.obj")
    np.savez(
        Path(data_save_dir) / f"{Path(urdf_path).stem}_q.npz",
        q=q_sample.cpu().numpy(),
    )


def _smoke_test_sdf():
    """Minimal SDF sampling/query test using chain_0 if available."""

    urdf_path = Path("data/out_chains/chain_1.urdf")
    if not urdf_path.is_file():
        print("[SDF test] Skipped (urdf not found)")
        return

    B = 128
    model = ChainModel(urdf_path, samples_per_link=128)
    q = model.sample_q(B=B)
    model.update_status(q)

    mask = torch.ones((model.num_links, ), dtype=torch.bool)
    mask[1::2] = False  # test link masking
    pts = model.sample_query_points(n=275000, mask=mask)
    sdf = model.query_sdf(pts, mask=mask)

    for idx in range(1, B):
        visualize_sdf_viser(
            mesh=model.get_trimesh_q(idx, boolean_merged=True, mask=mask),
            sdf_samples=torch.cat([pts[idx], sdf[idx].unsqueeze(-1)], dim=-1).cpu().numpy(),
            host="127.0.0.1",
            port=9200 + idx,
            downsample_ratio=0.003
        )
        visualize_sdf_viser(
            mesh=model.get_trimesh_q(idx, boolean_merged=True),
            sdf_samples=torch.cat([pts[idx], sdf[idx].unsqueeze(-1)], dim=-1).cpu().numpy(),
            host="127.0.0.1",
            port=9200 + idx,
            downsample_ratio=0.003
        )

        print(f"[SDF test] pts shape {pts.shape}, sdf stats mean={sdf.mean():.4f}, min={sdf.min():.4f}, max={sdf.max():.4f}")

if __name__ == '__main__':
    _smoke_test_sdf()

    # urdf_dir = "data/out_chains_v2/"
    # data_save_dir = "data/chain_meshes/"
    # Path(data_save_dir).mkdir(parents=True, exist_ok=True)
    # num_links = 1100
    # num_configs = 100
    # for i in range(num_links):
    #     urdf_path = Path(urdf_dir) / f"chain_{i}.urdf"
    #     generate_linkage_datasets(
    #         urdf_path,
    #         data_save_dir,
    #         B=num_configs,
    #         visualize=False,
    #     )
    #     print(f"Saved chain {i} data to {data_save_dir}")