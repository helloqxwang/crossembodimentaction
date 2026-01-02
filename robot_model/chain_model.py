"""Minimal chain model tailored to chain_generation.py outputs.

Supports:
- Load a generated URDF (boxes, cylinders, or capsules) via pytorch_kinematics.
- Sample joint configs, forward points, export a merged mesh, and visualize in Viser.
"""

from __future__ import annotations

import math
import colorsys
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


def _parse_meshes_from_urdf(urdf_path: str, device: torch.device) -> Dict[str, trimesh.Trimesh]:
    """Parse link geometries (box or capsule) directly from generated URDF."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    meshes: Dict[str, trimesh.Trimesh] = {}
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
        if box is not None:
            sx, sy, sz = map(float, box.get('size').split())
            mesh = trimesh.creation.box(extents=(sx, sy, sz))
        elif cylinder is not None:
            radius = float(cylinder.get('radius'))
            length = float(cylinder.get('length'))
            mesh = trimesh.creation.cylinder(radius=radius, height=length)
            # Align cylinder to +X (trimesh cylinder is along +Z by default).
            rot = trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0])
            mesh.apply_transform(rot)
        elif capsule is not None:
            radius = float(capsule.get('radius'))
            length = float(capsule.get('length'))  # cylindrical part
            mesh = trimesh.creation.capsule(radius=radius, height=length)
            # Align capsule to +X (trimesh capsule is along +Z by default).
            rot = trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0])
            mesh.apply_transform(rot)
        else:
            continue

        meshes[name] = mesh
    return meshes


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

        self.meshes = _parse_meshes_from_urdf(self.urdf_path, device)
        self.link_points = self._sample_rest_pose_points(self.meshes, samples_per_link)
        self.frame_status = None

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
    ) -> torch.Tensor:
        if links_pc is None:
            links_pc = self.link_points

        batch_size = self.q.shape[0]
        all_pc = []
        link_names = list(links_pc.keys())

        for link_idx, link_name in enumerate(link_names):
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
    def get_trimesh_q(self, idx: int) -> trimesh.Trimesh:
        scene = trimesh.Scene()
        for link_name, mesh in self.meshes.items():
            tf = self.frame_status[link_name].get_matrix()[idx].cpu().numpy()
            scene.add_geometry(mesh, transform=tf)

        combined = trimesh.util.concatenate(scene.dump())
        return combined


def visualize_chain_pc(
        pc: torch.Tensor,
        link_names: List[str],
        point_size: float = 0.003,
        mesh: trimesh.Trimesh = None,
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
    return server


if __name__ == '__main__':
    urdf_path = "out_chains/chain_0.urdf"
    model = ChainModel(
        urdf_path,
        samples_per_link=256,
    )
    q_sample = model.sample_q(B=1)
    link_names = list(model.link_points.keys())
    
    model.update_status(q_sample)
    chain_pc = model.get_transformed_links_pc(None, num_points=256)
    chain_mesh = model.get_trimesh_q(0)

    server = visualize_chain_pc(chain_pc, link_names, mesh=chain_mesh, host='127.0.0.1', port=9100)