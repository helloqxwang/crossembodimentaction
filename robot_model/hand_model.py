import os
import sys
import json
import math
import random
import numpy as np
import torch
import trimesh
import pytorch_kinematics as pk
import viser
from pytorch3d.transforms import matrix_to_quaternion

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# from utils.func_utils import farthest_point_sampling
from data_utils.utils import farthest_point_sampling
from utils.mesh_utils import load_link_geometries
from utils.rotation import *
import colorsys
from typing import Dict, List, Optional, Set, Tuple

def generate_distinct_colors(n: int,
                             saturation: float = 0.65,
                             value: float = 0.9) -> List[Tuple[float, float, float]]:
    """
    Evenly sample hues using the golden ratio to keep points spaced on the HSV circle.
    Returns RGB triples in [0, 1].
    """
    if n <= 0:
        return []

    golden_ratio_conjugate = (math.sqrt(5) - 1) / 2  # ≈0.618
    hue = 0.0
    colors = []
    for _ in range(n):
        hue = (hue + golden_ratio_conjugate) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors


def xyz_to_polar_direction(vectors: torch.Tensor) -> torch.Tensor:
    """Convert 3D vectors to 2D polar (azimuth, elevation) directions.

    Args:
        vectors: Tensor with final dimension of size 3.

    Returns:
        Tensor with final dimension 2 containing azimuth and elevation in radians.
    """
    if vectors.shape[-1] != 3:
        raise ValueError('vectors must have last dimension size 3.')

    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]

    xy_norm = torch.sqrt(x * x + y * y).clamp_min(1e-8)

    azimuth = torch.atan2(y, x)
    elevation = torch.atan2(z, xy_norm)

    return torch.stack((azimuth, elevation), dim=-1)


def fibonacci_sphere(samples: int,
                     radius: float,
                     *,
                     device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
    """Generate approximately uniform points on a sphere using a Fibonacci lattice."""
    if samples <= 0:
        raise ValueError('samples must be positive.')

    indices = torch.arange(samples, dtype=dtype, device=device) + 0.5
    phi = (1.0 + math.sqrt(5.0)) / 2.0

    z = 1.0 - 2.0 * indices / samples
    theta = 2.0 * math.pi * indices / phi

    r_xy = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    x = torch.cos(theta) * r_xy
    y = torch.sin(theta) * r_xy

    points = torch.stack((x, y, z), dim=-1)
    return points * radius

class HandModel:
    def __init__(
        self,
        robot_name,
        urdf_path,
        meshes_path,
        links_pc_path,
        device,
        sample_points_n=128
    ):
        self.robot_name = robot_name
        self.urdf_path = urdf_path
        self.meshes_path = meshes_path
        self.device = device

        self.pk_chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(dtype=torch.float32, device=device)
        self.dof = len(self.pk_chain.get_joint_parameter_names())
        if os.path.exists(links_pc_path):  # In case of generating robot links pc, the file doesn't exist.
            links_pc_data = torch.load(links_pc_path, map_location=device, weights_only=False)
            self.links_pc = links_pc_data['filtered']
            self.links_pc_original = links_pc_data['original']
        else:
            self.links_pc = None
            self.links_pc_original = None

        self.meshes = load_link_geometries(robot_name, self.urdf_path, self.pk_chain.get_link_names())

        self.vertices = {}
        removed_links = json.load(open(os.path.join(ROOT_DIR, 'data_utils/removed_links.json')))[robot_name]
        # for link_name, link_mesh in self.meshes.items():
        #     if link_name in removed_links:  # remove links unrelated to contact
        #         continue
        #     v = link_mesh.sample(link_num_points)
        #     self.vertices[link_name] = v
        self.meshes = {k: v for k, v in self.meshes.items() if k not in removed_links}
        self.vertices = self.sample_rest_pose_points(self.meshes, sample_points_n)

        self.frame_status = None
        if robot_name == 'shadowhand':
            self.base_joint_indices = [0, 1, 2, 3, 4, 5, 6, 7] # root translation & rotation & wrist rotation
        elif robot_name == 'allegro':
            self.base_joint_indices = [0, 1, 2, 3, 4, 5] # root translation & rotation
        else:
            self.base_joint_indices = [0, 1, 2, 3, 4, 5]

        self.kcs_indices = self.prepare_kinematic_chain()
        self._link_joint_index_cache = {}
        self._link_path_cache = {}
        self._link_neighbor_cache = {}
        self._link_order_cache = None

    def prepare_kinematic_chain(self):
        chain_indices = [] 
        link_names = list(self.vertices.keys())
        for link_name in link_names:
            link_chain_indices = self.related_joint_indices(link_name)
            chain_indices.append(link_chain_indices)
        filtered = []
        seen = set()
        for idx, idx_list in enumerate(chain_indices):
            fs = frozenset(idx_list)
            if fs in seen:
                continue
            as_set = set(idx_list)
            if any(as_set < set(other) for j, other in enumerate(chain_indices) if j != idx):
                continue
            seen.add(fs)
            filtered.append(idx_list)

        return filtered

    def sample_q(self, B: int = 1, fix_root: bool = True):
        q = torch.zeros(B, self.dof, dtype=torch.float32, device=self.device)
        lower, upper = self.pk_chain.get_joint_limits()
        lower_tensor = torch.tensor(lower, dtype=torch.float32, device=self.device)
        upper_tensor = torch.tensor(upper, dtype=torch.float32, device=self.device)

        random_portion = torch.rand(B, len(lower), dtype=torch.float32, device=self.device)
        q = lower_tensor.unsqueeze(0) * random_portion + upper_tensor.unsqueeze(0) * (1 - random_portion)
        if fix_root:
            q[:, self.base_joint_indices] = 0.0  # fix root translation & rotation
        return q

    def get_leaves(self):
        neighbors = self._link_neighbors()
        return [name for name, adj in neighbors.items() if len(adj) <= 1]

    def get_chain_from_leaves(self, leaf_a, leaf_b):
        neighbors = self._link_neighbors()
        if leaf_a not in neighbors or leaf_b not in neighbors:
            raise ValueError('Both leaf_a and leaf_b must be valid link names.')

        if leaf_a == leaf_b:
            return [leaf_a]

        link_order = list(self.vertices.keys())
        order = {name: idx for idx, name in enumerate(link_order)}
        stack = [(leaf_a, [leaf_a])]
        paths: List[List[str]] = []
        while stack:
            current, path = stack.pop()
            for nxt in neighbors[current]:
                if nxt in path:
                    continue
                new_path = path + [nxt]
                if nxt == leaf_b:
                    paths.append(new_path)
                    continue
                stack.append((nxt, new_path))

        if not paths:
            return []
        paths.sort(key=lambda p: [order[name] for name in p])
        if len(paths) == 1:
            return paths[0]
        return tuple(paths)

    def get_chain_transformations(self, q, links):
        """Return stacked world points and relative transforms for the given links.

        Args:
            q: Joint configuration tensor, accepting ``(B, DOF)`` Euler or
                ``(B, 9 + DOF)`` rotation-6D formats. A 1-D input becomes batch 1.
            links: Dictionary mapping link name to points in its local frame. Each
                entry must provide the same number of points ``N``. 
                ATTENTION: The points here are still in the local link frame, not the geocentered frame.

        Returns:
            world_points: Tensor of shape ``(B, L, N, 3)`` with per-link world points
                ordered by the input dictionary keys.
            relative_transforms: Tensor of shape ``(B, L, 4, 4)`` where each slice
                stores the transform from the parent link (identity for roots).
        """

        if not torch.is_tensor(q):
            q_tensor = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        else:
            q_tensor = q.to(dtype=torch.float32, device=self.device)
        if q_tensor.ndim == 1:
            q_tensor = q_tensor.unsqueeze(0)
        if q_tensor.shape[-1] != self.dof:
            q_tensor = q_rot6d_to_q_euler(q_tensor)

        self.update_status(q_tensor)
        batch_size = q_tensor.shape[0]

        link_names = list(links.keys())

        world_points_per_link: List[torch.Tensor] = []
        center_transforms_per_link: List[torch.Tensor] = []

        identity4 = torch.eye(4, dtype=torch.float32, device=self.device)

        for link_name in link_names:
            local_points = links[link_name].to(dtype=torch.float32, device=self.device).unsqueeze(0).expand(batch_size, -1, -1).contiguous()

            num_points = local_points.shape[1]
            transform = self.frame_status[link_name].get_matrix().to(self.device)
            ones = torch.ones((batch_size, num_points, 1), dtype=torch.float32, device=self.device)
            homogeneous = torch.cat([local_points, ones], dim=-1)
            world = torch.matmul(homogeneous, transform.transpose(1, 2))[..., :3]
            world_points_per_link.append(world)

            geocenter_np = self.meshes[link_name].center_mass
            geocenter = torch.as_tensor(geocenter_np, dtype=torch.float32, device=self.device)
            center_shift = identity4.unsqueeze(0).expand(batch_size, -1, -1).clone()
            center_shift[:, :3, 3] = geocenter
            center_transforms_per_link.append(transform @ center_shift)

        def invert_homogeneous(matrix: torch.Tensor) -> torch.Tensor:
            rot = matrix[..., :3, :3]
            trans = matrix[..., :3, 3]
            rot_t = rot.transpose(-1, -2)
            inv = torch.zeros_like(matrix)
            inv[..., :3, :3] = rot_t
            inv[..., 3, 3] = 1.0
            inv[..., :3, 3] = -(rot_t @ trans.unsqueeze(-1)).squeeze(-1)
            return inv

        world_points = torch.stack(world_points_per_link, dim=1) # (B, N_links, N_points, 3)
        center_transforms = torch.stack(center_transforms_per_link, dim=1) # (B, N_links, 4, 4)

        relative_transforms: List[torch.Tensor] = []
        for idx, link_name in enumerate(link_names):
            if idx == 0:
                relative_transforms.append(identity4.unsqueeze(0).expand(batch_size, -1, -1).clone())
                continue

            parent_tf = center_transforms[:, idx - 1]
            child_tf = center_transforms[:, idx]
            relative_transforms.append(invert_homogeneous(parent_tf) @ child_tf)

        relative_transforms = torch.stack(relative_transforms, dim=1)

        return world_points, relative_transforms

    def visualize_chain_debug(
        self,
        q,
        links,
        batch_index: int = 0,
        host: str = '127.0.0.1',
        port: int = 9100
    ):
        """Visualize sampled points, meshes, and reconstructed frames for a chain."""
        if not isinstance(links, dict) or not links:
            raise ValueError('links must be a non-empty dictionary of link-local points.')

        world_points, relative_transforms = self.get_chain_transformations(q, links)
        world_points = world_points.to(dtype=torch.float32, device=self.device)
        relative_transforms = relative_transforms.to(dtype=torch.float32, device=self.device)
        self.update_status(q)

        batch_size = world_points.shape[0]
        link_names = list(links.keys())
        num_links = len(link_names)
        identity4 = torch.eye(4, dtype=torch.float32, device=self.device)

        center_transforms_per_link: List[torch.Tensor] = []
        expected_world_points: List[torch.Tensor] = []

        for link_name in link_names:
            local_points:torch.Tensor = links[link_name].to(dtype=torch.float32, device=self.device)
            local_points = local_points.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

            transform = self.frame_status[link_name].get_matrix().to(self.device)

            geocenter_np = self.meshes[link_name].center_mass
            geocenter = torch.as_tensor(geocenter_np, dtype=torch.float32, device=self.device)
            center_shift = identity4.unsqueeze(0).expand(batch_size, -1, -1).clone()
            center_shift[:, :3, 3] = geocenter
            center_transform = transform @ center_shift
            center_transforms_per_link.append(center_transform)

            if local_points.shape[1] == 0:
                expected_world_points.append(torch.empty((batch_size, 0, 3), dtype=torch.float32, device=self.device))
            else:
                ones = torch.ones((batch_size, local_points.shape[1], 1), dtype=torch.float32, device=self.device)
                homogeneous = torch.cat([local_points, ones], dim=-1)
                expected_world = torch.matmul(homogeneous, transform.transpose(1, 2))[..., :3]
                expected_world_points.append(expected_world)

        def invert_homogeneous(matrix: torch.Tensor) -> torch.Tensor:
            rot = matrix[..., :3, :3]
            trans = matrix[..., :3, 3]
            rot_t = rot.transpose(-1, -2)
            inv = torch.zeros_like(matrix)
            inv[..., :3, :3] = rot_t
            inv[..., 3, 3] = 1.0
            inv[..., :3, 3] = -(rot_t @ trans.unsqueeze(-1)).squeeze(-1)
            return inv

        center_transforms = torch.stack(center_transforms_per_link, dim=1)
        expected_world_points_tensor = torch.stack(expected_world_points, dim=1)

        root_idx = 0
        root_name = link_names[root_idx]
        root_world_tf = center_transforms[:, root_idx]
        root_inv_tf = invert_homogeneous(root_world_tf)

        identity_batch = identity4.unsqueeze(0).expand(batch_size, -1, -1).clone()
        relative_chain = torch.empty_like(center_transforms)
        relative_chain[:, 0] = identity_batch
        # Each relative transform maps the child frame into its parent frame. Chaining them
        # with right-multiplication matches the "left-multiply inverse" rule under the
        # row-vector convention while keeping the code in the column-vector form used here.
        for idx in range(1, num_links):
            rel_tf = relative_transforms[:, idx]
            relative_chain[:, idx] = torch.matmul(relative_chain[:, idx - 1], rel_tf)

        reconstructed_world = torch.matmul(root_world_tf.unsqueeze(1), relative_chain)

        point_error = (
            (world_points - expected_world_points_tensor)
            .norm(dim=-1)
            .amax() if world_points.numel() > 0 else torch.tensor(0.0, device=self.device)
        )
        frame_error = (
            (reconstructed_world - center_transforms)
            .norm(dim=(-1, -2))
            .amax()
        )
        print(
            f'[visualize_chain_debug] max point deviation: {point_error.item():.6e}, '
            f'max frame deviation: {frame_error.item():.6e}'
        )

        colors = generate_distinct_colors(num_links)
        server = viser.ViserServer(host=host, port=port)
        server.scene.add_frame(
            'world_frame',
            show_axes=['x', 'y', 'z'],
            axes_length=0.08,
            axes_radius=0.004,
            position=(0, 0, 0)
        )

        root_tf = root_world_tf[batch_index]
        root_rot_tensor = root_tf[:3, :3]
        root_quat = matrix_to_quaternion(root_rot_tensor.unsqueeze(0)).squeeze(0)
        server.scene.add_frame(
            f'root_{link_names[root_idx]}',
            position=root_tf[:3, 3].detach().cpu().numpy(),
            wxyz=root_quat.detach().cpu().numpy(),
            axes_length=0.07,
            axes_radius=0.003
        )

        for link_idx, link_name in enumerate(link_names):
            color_rgb = colors[link_idx]
            color255 = tuple(int(channel * 255) for channel in color_rgb)

            points_np = world_points[batch_index, link_idx].detach().cpu().numpy()
            if points_np.size > 0:
                point_colors = np.tile(np.array(color255, dtype=np.uint8), (points_np.shape[0], 1))
                server.scene.add_point_cloud(
                    f'pc_{link_idx}_{link_name}',
                    points_np,
                    point_size=0.003,
                    point_shape='square',
                    colors=point_colors,
                )

            mesh = self.meshes[link_name]
            transform_np = self.frame_status[link_name].get_matrix()[batch_index].detach().cpu().numpy()
            mesh_transformed = mesh.copy().apply_transform(transform_np)
            server.scene.add_mesh_simple(
                f'mesh_{link_idx}_{link_name}',
                mesh_transformed.vertices,
                mesh_transformed.faces,
                color=color255,
                opacity=0.25
            )

            actual_tf = center_transforms[batch_index, link_idx]
            actual_quat = matrix_to_quaternion(actual_tf[:3, :3].unsqueeze(0)).squeeze(0)
            server.scene.add_frame(
                f'actual_{link_idx}_{link_name}',
                position=actual_tf[:3, 3].detach().cpu().numpy(),
                wxyz=actual_quat.detach().cpu().numpy(),
                axes_length=0.035,
                axes_radius=0.0015
            )

            recon_tf = reconstructed_world[batch_index, link_idx]
            recon_quat = matrix_to_quaternion(recon_tf[:3, :3].unsqueeze(0)).squeeze(0)
            server.scene.add_frame(
                f'recon_{link_idx}_{link_name}',
                position=recon_tf[:3, 3].detach().cpu().numpy(),
                wxyz=recon_quat.detach().cpu().numpy(),
                axes_length=0.05,
                axes_radius=0.001
            )

        # aligned_actual = torch.matmul(root_inv_tf.unsqueeze(1), center_transforms)
        # aligned_reconstructed = relative_chain
        # for link_idx, link_name in enumerate(link_names):
        #     aligned_tf = aligned_actual[batch_index, link_idx]
        #     recon_aligned_tf = aligned_reconstructed[batch_index, link_idx]

        #     aligned_quat = matrix_to_quaternion(aligned_tf[:3, :3].unsqueeze(0)).squeeze(0)
        #     recon_aligned_quat = matrix_to_quaternion(recon_aligned_tf[:3, :3].unsqueeze(0)).squeeze(0)

        #     server.scene.add_frame(
        #         f'aligned_actual_{link_idx}_{link_name}',
        #         position=aligned_tf[:3, 3].detach().cpu().numpy(),
        #         wxyz=aligned_quat.detach().cpu().numpy(),
        #         axes_length=0.035,
        #         axes_radius=0.0015
        #     )
        #     server.scene.add_frame(
        #         f'aligned_recon_{link_idx}_{link_name}',
        #         position=recon_aligned_tf[:3, 3].detach().cpu().numpy(),
        #         wxyz=recon_aligned_quat.detach().cpu().numpy(),
        #         axes_length=0.05,
        #         axes_radius=0.001
        #     )

        return server

    def _link_neighbors(self) -> Dict[str, List[str]]:
        if self._link_neighbor_cache:
            return {k: list(v) for k, v in self._link_neighbor_cache.items()}

        link_names = list(self.vertices.keys())
        link_set = set(link_names)
        order = {name: idx for idx, name in enumerate(link_names)}
        neighbors: Dict[str, Set[str]] = {name: set() for name in link_names}

        for link_name in link_names:
            path = self._link_path_frames(link_name)
            filtered_path = [name for name in path if name in link_set]
            if not filtered_path:
                continue
            try:
                pos = filtered_path.index(link_name)
            except ValueError:
                continue
            if pos > 0:
                parent = filtered_path[pos - 1]
                neighbors[link_name].add(parent)
                neighbors[parent].add(link_name)

        cached = {
            name: tuple(sorted(adj, key=lambda item: order[item]))
            for name, adj in neighbors.items()
        }
        self._link_neighbor_cache = cached
        return {k: list(v) for k, v in cached.items()}

    def _link_path_frames(self, link_name: str) -> Tuple[str, ...]:
        cache = self._link_path_cache
        if link_name not in cache:
            serial_chain = pk.SerialChain(self.pk_chain, link_name)
            cache[link_name] = tuple(serial_chain.frame_to_idx.keys())
        return cache[link_name]

    def related_joint_indices(self, link_name, mask_6d=True):
        assert link_name in self.pk_chain.get_link_names(), f'Link name {link_name} not in the robot model.'
        parent_links = list(pk.SerialChain(self.pk_chain, link_name).frame_to_idx.keys())
        all_jointed_link_names = self.pk_chain.get_joint_parent_frame_names()
        whole_joints_indices = [all_jointed_link_names.index(p) if p in all_jointed_link_names else -1 for p in parent_links]

        if mask_6d:
            masked_indices = self.base_joint_indices + [-1]
        else:
            masked_indices = [-1]
        related_joint_indices = [n for n in whole_joints_indices if n not in masked_indices]
        return related_joint_indices

    def get_joint_orders(self):
        return [joint.name for joint in self.pk_chain.get_joints()]

    def update_status(self, q):
        if q.shape[-1] != self.dof:
            q = q_rot6d_to_q_euler(q)
        self.frame_status = self.pk_chain.forward_kinematics(q.to(self.device))

    def get_transformed_links_pc(self, q=None, links_pc=None):
        """
        Use robot link pc & q value to get point cloud.
        SUPPORT BATCHING NOW!

        :param q: (B, 6 + DOF,), joint values (euler representation)
        :param links_pc: {link_name: (N_link, 3)}, robot links pc dict, not None only for get_sampled_pc()
        :return: point cloud: (B, N, 4), with link index
        """
        if q is None:
            q = torch.zeros((1, self.dof), dtype=torch.float32, device=self.device)
        self.update_status(q)
        if links_pc is None:
            links_pc = self.links_pc

        all_pc_se3 = []
        for link_index, (link_name, link_pc) in enumerate(links_pc.items()):
            if not torch.is_tensor(link_pc):
                link_pc = torch.tensor(link_pc, dtype=torch.float32, device=q.device)
            n_link = link_pc.shape[0]
            se3 = self.frame_status[link_name].get_matrix().to(q.device)
            homogeneous_tensor = torch.ones(n_link, 1, device=q.device)
            link_pc_homogeneous = torch.cat([link_pc.to(q.device), homogeneous_tensor], dim=1)
            link_pc_se3 = (link_pc_homogeneous @ se3.transpose(-1, -2))[..., :3]
            index_tensor = torch.full([q.shape[0], n_link, 1], float(link_index), device=q.device)
            link_pc_se3_index = torch.cat([link_pc_se3, index_tensor], dim=-1)
            all_pc_se3.append(link_pc_se3_index)
        all_pc_se3 = torch.cat(all_pc_se3, dim=-2)

        return all_pc_se3

    def sample_rest_pose_points(self, link_mesh_dict, total_samples: int):
        """Sample per-link points, transform to rest pose, then downsample to a total of ``total_samples`` points.

        Args:
            link_mesh_dict (Dict[str, trimesh.Trimesh]): Mapping from link name to mesh geometry.
            total_samples (int): Desired total number of points after downsampling.

        Returns:
            Dict[str, torch.Tensor]: Mapping from link name to downsampled points in the original link frame.
        """
        if not link_mesh_dict:
            return {}

        if total_samples <= 0:
            return {
                link_name: torch.empty((0, 3), dtype=torch.float32, device=self.device)
                for link_name in link_mesh_dict
            }

        link_names = list(link_mesh_dict.keys())
        per_link_samples = max(total_samples, 20)
        sampled_local: Dict[str, torch.Tensor] = {}
        local_points_list = []
        for link_name in link_names:
            mesh = link_mesh_dict[link_name]
            sampled_np = np.asarray(mesh.sample(per_link_samples), dtype=np.float32)
            local_tensor = torch.from_numpy(sampled_np).to(self.device)
            sampled_local[link_name] = local_tensor
            local_points_list.append(local_tensor)

        local_points = torch.cat(local_points_list, dim=0)

        rest_pose = torch.zeros((1, self.dof), dtype=torch.float32, device=self.device)
        transformed = self.get_transformed_links_pc(rest_pose, sampled_local)
        transformed = transformed.squeeze(0)

        if transformed.ndim == 1:
            transformed = transformed.unsqueeze(0)

        if transformed.shape[0] == 0:
            return {
                link_name: torch.empty((0, 3), dtype=torch.float32, device=self.device)
                for link_name in link_names
            }

        total_points = transformed.shape[0]
        if total_points > total_samples:
            transformed_down, selected_indices = farthest_point_sampling(transformed, total_samples)
        else:
            transformed_down = transformed
            selected_indices = torch.arange(total_points, device=self.device)

        selected_indices_tensor = selected_indices.to(dtype=torch.long, device=self.device)
        link_indices = transformed_down[:, -1].round().to(torch.long)
        points_local = local_points[selected_indices_tensor]

        output = {}
        for idx, link_name in enumerate(link_names):
            mask = link_indices == idx
            if torch.any(mask):
                output[link_name] = points_local[mask]
            else:
                output[link_name] = torch.empty((0, 3), dtype=torch.float32, device=self.device)

        return output

    def sample_mesh_points_with_bps(
        self,
        link_mesh_dict: Dict[str, trimesh.Trimesh],
        samples_per_link: int,
        bps_point_count: Optional[int] = None,
        bps_mesh_point_count: Optional[int] = None,
        bps_basis_points: Optional[torch.Tensor] = None
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Sample mesh points, compute BPS features, and azimuth/elevation directions.

        Args:
            link_mesh_dict: Mapping from link name to ``trimesh.Trimesh`` geometry.
            samples_per_link: Number of farthest-point-sampled points to retain per link.
            bps_point_count: Number of basis points for BPS. Defaults to ``samples_per_link``.
            bps_mesh_point_count: Number of mesh samples used when building the BPS descriptor.
                Defaults to ``max(samples_per_link, bps_point_count)``.
            bps_basis_points: Optional precomputed basis points in the mesh local frame.
                They will be uniformly scaled to match each link's ``bps_radius``.

    Returns:
        Dict mapping each link name to a tuple ``(points, bps_representation, polar_directions, radius)``.
        ``points`` has shape ``(samples_per_link, 3)`` in the mesh local frame.
        ``bps_representation`` has shape ``(bps_point_count, 6)`` (basis point and displacement).
        ``polar_directions`` has shape ``(samples_per_link, 2)`` holding azimuth and elevation (radians).
        ``radius`` is a scalar tensor storing the BPS sphere radius used for that link.
        """

        if not link_mesh_dict:
            return {}
        if samples_per_link <= 0:
            raise ValueError('samples_per_link must be positive.')

        if bps_basis_points is not None:
            if bps_basis_points.ndim != 2 or bps_basis_points.shape[1] != 3:
                raise ValueError('bps_basis_points must have shape (N, 3).')
            if bps_point_count is not None and bps_point_count != bps_basis_points.shape[0]:
                raise ValueError('bps_point_count must match the number of provided basis points.')
            bps_count = int(bps_basis_points.shape[0])
            if bps_count <= 0:
                raise ValueError('bps_basis_points must contain at least one point.')
            basis_template = bps_basis_points.to(dtype=torch.float32, device=self.device)
            template_radius = basis_template.norm(dim=-1).amax().clamp_min(1e-8)
        else:
            bps_count = samples_per_link if bps_point_count is None else bps_point_count
            if bps_count <= 0:
                raise ValueError('bps_point_count must be positive.')
            bps_count = int(bps_count)
            basis_template = None
            template_radius = None

        default_mesh_point_count = max(samples_per_link, bps_count)
        mesh_point_count = default_mesh_point_count if bps_mesh_point_count is None else bps_mesh_point_count
        if mesh_point_count <= 0:
            raise ValueError('bps_mesh_point_count must be positive.')
        mesh_point_count = int(mesh_point_count)

        results: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        for link_name, mesh in link_mesh_dict.items():
            if not isinstance(mesh, trimesh.Trimesh):
                raise TypeError(f'Expected trimesh.Trimesh for link {link_name}.')

            sampled_np = np.asarray(mesh.sample(mesh_point_count), dtype=np.float32)
            raw_samples = torch.from_numpy(sampled_np).to(dtype=torch.float32, device=self.device)

            if raw_samples.shape[0] == 0:
                sampled_points = torch.zeros((samples_per_link, 3), dtype=torch.float32, device=self.device)
            else:
                if raw_samples.shape[0] >= samples_per_link:
                    sampled_points, _ = farthest_point_sampling(raw_samples, samples_per_link)
                else:
                    repeat_count = samples_per_link - raw_samples.shape[0]
                    repeat_indices = torch.randint(raw_samples.shape[0], (repeat_count,), device=self.device)
                    sampled_points = torch.cat((raw_samples, raw_samples[repeat_indices]), dim=0)

            geocenter = torch.as_tensor(mesh.center_mass, dtype=torch.float32, device=self.device)

            if raw_samples.shape[0] == 0:
                bps_radius = torch.tensor(1e-4, dtype=torch.float32, device=self.device)
            else:
                relative_points_bps = raw_samples - geocenter
                bps_radius = torch.clamp(relative_points_bps.norm(dim=-1).amax(), min=1e-4)

            if basis_template is None:
                basis_points_local = fibonacci_sphere(
                    bps_count,
                    bps_radius.item(),
                    device=self.device,
                    dtype=torch.float32
                )
            else:
                scale = (bps_radius / template_radius).to(dtype=torch.float32)
                basis_points_local = basis_template * scale
            basis_points = basis_points_local + geocenter

            if raw_samples.shape[0] == 0:
                displacement = torch.zeros_like(basis_points)
            else:
                diff = raw_samples.unsqueeze(0) - basis_points.unsqueeze(1)
                dists = diff.norm(dim=-1)
                nearest_idx = dists.argmin(dim=1)
                nearest_points = raw_samples[nearest_idx]
                displacement = nearest_points - basis_points

            bps_representation = torch.cat((basis_points, displacement), dim=-1)

            relative_points = sampled_points - geocenter
            polar_directions = xyz_to_polar_direction(relative_points)

            radius_tensor = bps_radius.to(dtype=torch.float32)

            results[link_name] = (
                sampled_points,
                bps_representation,
                polar_directions,
                radius_tensor
            )

        return results

    def get_sampled_pc(self, q=None, num_points=None):
        """
        :param q: (B, 9 + DOF,), joint values (rot6d representation)
        :param num_points: int, number of sampled points
        :return: ((B, N, 3), list), sampled point cloud (numpy) & index
        """
        if q is None:
            q = self.get_canonical_q()

        sampled_pc = self.get_transformed_links_pc(q, self.vertices)
        if num_points is None:
            return sampled_pc
        return farthest_point_sampling(sampled_pc, num_points)

    def get_canonical_q(self):
        """ For visualization purposes only. """
        lower, upper = self.pk_chain.get_joint_limits()
        canonical_q = torch.tensor(lower) * 0.75 + torch.tensor(upper) * 0.25
        canonical_q[:6] = 0
        return canonical_q

    def get_mesh_geocenters(self):
        """ Return the geocenter of each link mesh in rest pose. """
        geocenters = {}
        for link_name in self.vertices:
            mesh = self.meshes[link_name]
            geocenter = torch.tensor(mesh.center_mass, dtype=torch.float32, device=self.device)
            geocenters[link_name] = geocenter.unsqueeze(0)
        return geocenters

    def get_initial_q(self, q=None, max_angle: float = math.pi / 6):
        """
        Compute the robot initial joint value q based on the target grasp.
        Root translation is not considered since the point cloud will be normalized to zero-mean.

        :param q: (6 + DOF,) or (9 + DOF,), joint values (euler/rot6d representation)
        :param max_angle: float, maximum angle of the random rotation
        :return: initial q: (6 + DOF,), euler representation
        """
        if q is None:  # random sample root rotation and joint values
            q_initial = torch.zeros(self.dof, dtype=torch.float32, device=self.device)

            q_initial[3:6] = (torch.rand(3) * 2 - 1) * torch.pi
            q_initial[5] /= 2

            lower_joint_limits, upper_joint_limits = self.pk_chain.get_joint_limits()
            lower_joint_limits = torch.tensor(lower_joint_limits[6:], dtype=torch.float32)
            upper_joint_limits = torch.tensor(upper_joint_limits[6:], dtype=torch.float32)
            portion = random.uniform(0.65, 0.85)
            q_initial[6:] = lower_joint_limits * portion + upper_joint_limits * (1 - portion)
        else:
            if len(q) == self.dof:
                q = q_euler_to_q_rot6d(q)
            q_initial = q.clone()

            # compute random initial rotation
            direction = - q_initial[:3] / torch.norm(q_initial[:3])
            angle = torch.tensor(random.uniform(0, max_angle), device=q.device)  # sample rotation angle
            axis = torch.randn(3).to(q.device)  # sample rotation axis
            axis -= torch.dot(axis, direction) * direction  # ensure orthogonality
            axis = axis / torch.norm(axis)
            random_rotation = axisangle_to_matrix(axis, angle).to(q.device)
            rotation_matrix = random_rotation @ rot6d_to_matrix(q_initial[3:9])
            q_initial[3:9] = matrix_to_rot6d(rotation_matrix)

            # compute random initial joint values
            lower_joint_limits, upper_joint_limits = self.pk_chain.get_joint_limits()
            lower_joint_limits = torch.tensor(lower_joint_limits[6:], dtype=torch.float32)
            upper_joint_limits = torch.tensor(upper_joint_limits[6:], dtype=torch.float32)
            portion = random.uniform(0.65, 0.85)
            q_initial[9:] = lower_joint_limits * portion + upper_joint_limits * (1 - portion)
            # q_initial[9:] = torch.zeros_like(q_initial[9:], dtype=q.dtype, device=q.device)

            q_initial = q_rot6d_to_q_euler(q_initial)

        return q_initial

    def get_trimesh_q(self, q):
        """ Return the hand trimesh object corresponding to the input joint value q. """
        self.update_status(q)

        scene = trimesh.Scene()
        for link_name in self.vertices:
            # the mesh transform matrix is the same as the frame status matrix
            mesh_transform_matrix = self.frame_status[link_name].get_matrix()[0].cpu().numpy()
            scene.add_geometry(self.meshes[link_name].copy().apply_transform(mesh_transform_matrix))

        ## merge all geometries mannually
        vertices = []
        faces = []
        vertex_offset = 0
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                vertices.append(geom.vertices)
                faces.append(geom.faces + vertex_offset)
                vertex_offset += len(geom.vertices)
        all_vertices = np.vstack(vertices)
        all_faces = np.vstack(faces)

        parts = {}
        for link_name in self.meshes:
            mesh_transform_matrix = self.frame_status[link_name].get_matrix()[0].cpu().numpy()
            part_mesh = self.meshes[link_name].copy().apply_transform(mesh_transform_matrix)
            parts[link_name] = part_mesh

        return_dict = {
            'visual': trimesh.Trimesh(vertices=all_vertices, faces=all_faces),
            'parts': parts
        }
        return return_dict

    def get_trimesh_se3(self, transform, index):
        """ Return the hand trimesh object corresponding to the input transform. """
        scene = trimesh.Scene()
        for link_name in transform:
            mesh_transform_matrix = transform[link_name][index].cpu().numpy()
            scene.add_geometry(self.meshes[link_name].copy().apply_transform(mesh_transform_matrix))

        vertices = []
        faces = []
        vertex_offset = 0
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                vertices.append(geom.vertices)
                faces.append(geom.faces + vertex_offset)
                vertex_offset += len(geom.vertices)
        all_vertices = np.vstack(vertices)
        all_faces = np.vstack(faces)

        return trimesh.Trimesh(vertices=all_vertices, faces=all_faces)


def create_hand_model(
    robot_name,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    num_points=512
):
    json_path = os.path.join(ROOT_DIR, 'data/data_urdf/robot/urdf_assets_meta.json')
    urdf_assets_meta = json.load(open(json_path))
    urdf_path = os.path.join(ROOT_DIR, urdf_assets_meta['urdf_path'][robot_name])
    meshes_path = os.path.join(ROOT_DIR, urdf_assets_meta['meshes_path'][robot_name])
    links_pc_path = os.path.join(ROOT_DIR, f'data/PointCloud/robot/{robot_name}.pt')
    hand_model = HandModel(robot_name, urdf_path, meshes_path, links_pc_path, device, num_points)
    return hand_model

def sample_random_link_q(hand_model, link_name, num_samples=10, extreme=True):
    q = torch.zeros(num_samples, hand_model.dof)
    related_joint_indices = hand_model.related_joint_indices(link_name)
    if len(related_joint_indices) == 0:
        return q
    lower, upper = hand_model.pk_chain.get_joint_limits()
    lower_related = [lower[i] for i in related_joint_indices]
    upper_related = [upper[i] for i in related_joint_indices]
    
    samples_ls = []
    for idx in range(len(related_joint_indices)):
        samples = torch.rand(num_samples) * (upper_related[idx] - lower_related[idx]) + lower_related[idx]
        samples_ls.append(samples)
    samples_tensor = torch.stack(samples_ls, dim=1)  # (num_samples, num_joints)

    if extreme:
        lower_tensor = torch.tensor(lower_related, dtype=torch.float32)
        upper_tensor = torch.tensor(upper_related, dtype=torch.float32)

        scale_up_positive = upper_tensor / samples_tensor
        scale_up_negative = lower_tensor / samples_tensor
        scale_up = torch.where(samples_tensor >= 0, scale_up_positive, scale_up_negative)
        scale = torch.min(scale_up, dim=1).values

        samples_tensor = samples_tensor * scale.unsqueeze(-1)
    
    q[:, related_joint_indices] = samples_tensor
    return q

def sample_reachability_pc(hand_model:HandModel, point_num=16834, ratio=5, multi_points=False):
    q = hand_model.sample_q(B=point_num * ratio, fix_root=True)
    if multi_points:
        joint_dict = hand_model.vertices
    else:
        joint_dict = hand_model.get_mesh_geocenters()

    hand_pc = hand_model.get_transformed_links_pc(q, joint_dict)[..., :3].transpose(0, 1)
    sampled_pc, _ = farthest_point_sampling(hand_pc, point_num)

    sampled_pc_dict = {}
    for idx, link_name in enumerate(joint_dict):
        sampled_pc_dict[link_name] = sampled_pc[idx].cpu()
    return sampled_pc_dict

def vis_point_cloud(pc: torch.Tensor):
    """Visualize a single or batched point cloud in Viser."""
    if pc.ndim == 2:
        pc = pc.unsqueeze(0)
    elif pc.ndim != 3:
        raise ValueError('pc must have shape (N, 3) or (B, N, 3).')

    batch_size = pc.shape[0]
    colors = generate_distinct_colors(batch_size)
    server = viser.ViserServer(host='127.0.0.1', port=9090)
    server.scene.add_frame(
        'world_frame',
        show_axes=['x', 'y', 'z'],
        axes_length=0.05,
        axes_radius=0.005,
        position=(0, 0, 0)
    )

    for idx in range(batch_size):
        points_np = pc[idx].detach().cpu().numpy()
        color = tuple(int(channel * 255) for channel in colors[idx])
        server.scene.add_point_cloud(
            f'vertices_{idx}',
            points_np,
            point_size=0.002,
            point_shape="circle",
            colors=color,
        )

def generate_hand_model_data(hand_name_ls, save_path, point_num=16834, ratio=10, vis=False, multi_points=False):
    os.makedirs(save_path, exist_ok=True)
    for mode_name in hand_name_ls:
        hand_model = create_hand_model(mode_name)
        sampled_pc_dict = sample_reachability_pc(hand_model, point_num=point_num, ratio=ratio, multi_points=multi_points)
        if vis:
            sampled_pc = torch.stack([sampled_pc_dict[k] for k in sampled_pc_dict], dim=0)
            vis_point_cloud(sampled_pc)
        torch.save(sampled_pc_dict, f'{save_path}/{mode_name}_reachability_pc_{point_num}.pt')

def generate_hand_chain_data(hand_name_ls, save_path, n_point=128, n_q=1e7, batch_size=1e3):
    for mode_name in hand_name_ls:
        hand_model = create_hand_model(mode_name, num_points=n_point)
        # Reuse a unit-sphere template so every call shares the same basis ordering.
        bps_basis_points = fibonacci_sphere(
            128,
            1.0,
            device=hand_model.device,
            dtype=torch.float32
        )
        leaves_ls = hand_model.get_leaves()
        link_info_dict = hand_model.sample_mesh_points_with_bps(
            hand_model.meshes,
            n_point,
            bps_point_count=128,
            bps_mesh_point_count=512,
            bps_basis_points=bps_basis_points
        )
        all_chain_dict = {k: link_info_dict[k][0] for k in link_info_dict}  # only keep the sampled points
        if n_point == 1:
            all_chain_dict = hand_model.get_mesh_geocenters()

        chain_point_positions_full_ls = []
        chain_transformations_full_ls = []
        chain_point_delta_positions_full_ls = []
        chain_links_ls = []
        for i, leaf_a in enumerate(leaves_ls):
            for j, leaf_b in enumerate(leaves_ls):
                if i >= j:
                    continue
                chain_point_positions_ls = []
                chain_transformations_ls = []
                chain_point_delta_positions_ls = []
                chain = hand_model.get_chain_from_leaves(leaf_a, leaf_b)
                chain_links_ls.append(chain)
                for start_idx in range(0, int(n_q), int(batch_size)):
                    q_samples = hand_model.sample_q(int(batch_size), fix_root=True)
                    q_add_delta = q_samples + 0.1
                    chain_dict = {k: all_chain_dict[k] for k in chain}
                    points, transformations = hand_model.get_chain_transformations(q_samples, chain_dict)
                    points_delta = hand_model.get_transformed_links_pc(q_add_delta, chain_dict)[..., :3].reshape(*points.shape)

                    chain_point_positions_ls.append(points)
                    chain_transformations_ls.append(transformations[:, 1:])  # exclude the root identity transform
                    chain_point_delta_positions_ls.append(points_delta)

                chain_point_positions_full_ls.append(torch.cat(chain_point_positions_ls, dim=0).cpu())
                chain_transformations_full_ls.append(torch.cat(chain_transformations_ls, dim=0).cpu())
                chain_point_delta_positions_full_ls.append(torch.cat(chain_point_delta_positions_ls, dim=0).cpu())

        bps_info_dict = {k: link_info_dict[k][1].cpu() for k in link_info_dict}  # only keep the bps representation
        orientation_info_dict = {k: link_info_dict[k][2].cpu() for k in link_info_dict}  # only keep the polar directions
        bps_radius_dict = {k: link_info_dict[k][3].item() for k in link_info_dict}  # only keep the bps radius
        torch.save({
            'chain_point_positions': chain_point_positions_full_ls, # [(B, L_c, n_p, 3), ...]
            'chain_point_delta_positions': chain_point_delta_positions_full_ls, # [(B, L_c, n_p, 3), ...]
            'chain_transformations': chain_transformations_full_ls, # [(B, L_c - 1, 4, 4), ...]
            'chain_link_names': chain_links_ls, # [[link1, link2, ...], ...]
            'bps_representation': bps_info_dict, # {link_name: (bps_n, 6), ...}
            'orientation_directions': orientation_info_dict, # {link_name: (n_p, 2), ...}
            'bps_radius': bps_radius_dict # {link_name: scalar, ...}
        }, f'{save_path}/{mode_name}_chain_point{int(n_point)}_{int(n_q)}.pt' if n_point !=1 else f'{save_path}/{mode_name}_chain_{int(n_q)}.pt')

def visualize_all_hands(hand_name_ls, ee_point_num=128):
    for mode_name in hand_name_ls:
        hand_model = create_hand_model(mode_name, num_points=ee_point_num)
        q_rest = torch.zeros((1, hand_model.dof))

        ## the original center of links
        links_points = {k: torch.zeros(1, 3, device=hand_model.device) for k in hand_model.links_pc}
        ## the geocenter of meshes
        geocenters = hand_model.get_mesh_geocenters()

        pc = hand_model.get_transformed_links_pc(q_rest, geocenters)[0, :, :3]     
        hand_mesh = hand_model.get_trimesh_q(q_rest[0])['visual'] 
        server = viser.ViserServer(host='127.0.0.1', port=9090)
        server.scene.add_frame(
            'world_frame',
            show_axes=['x', 'y', 'z'],
            axes_length=0.05,
            axes_radius=0.005,
            position=(0, 0, 0)
        )
        server.scene.add_point_cloud(
            'vertices',
            pc.numpy(),
            point_size=0.002,
            point_shape="circle",
            colors=(255, 0, 0),
        )
        server.scene.add_mesh_simple(
            'robot_initial',
            hand_mesh.vertices,
            hand_mesh.faces,
            color=(102, 192, 255),
            opacity=0.2
        )
        print(f'Hand model {mode_name} created.')

# Test
if __name__ == '__main__':
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    # hand_model_ls = ["shadowhand", "allegro", "barrett"]
    hand_name_ls = [
        "allegro", "shadowhand", "barrett", 
        "ezgripper", "robotiq_3finger", "leaphand"]
    save_path='./data/EEReprData'
    n_q = 1e6
    os.makedirs(save_path, exist_ok=True)
    
    # generate_hand_model_data(hand_name_ls, save_path=save_path, point_num=16834, vis=False, multi_points=False)
    generate_hand_chain_data(hand_name_ls, save_path=save_path, n_point=1, n_q=n_q, batch_size=1e4)
    generate_hand_chain_data(hand_name_ls, save_path=save_path, n_point=32, n_q=n_q, batch_size=1e4)