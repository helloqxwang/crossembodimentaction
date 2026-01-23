#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
from bisect import bisect_right
import math
import numpy as np
import os
import random
import torch
import torch.utils.data
from tqdm import tqdm
from pathlib import Path
import sys
from typing import Dict, Optional
sys.path.append(str(Path(__file__).resolve().parent.parent / ""))
from robot_model.chain_model import ChainModel, visualize_sdf_viser
import torch.nn.functional as F

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
    scale_to = npz.get("scale_to", 1.0)
    q = npz.get("q", None)

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples, scale_to, q


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def _prepare_link_bps_scdistances(bps_info):
    link_bps_scdistances = [
        np.concatenate(
            [
                bp_info["offsets"],
                np.ones((bp_info["offsets"].shape[0], 1)) * bp_info["scale_to_unit"],
            ],
            axis=-1,
        )
        for bp_info in bps_info
    ]
    return torch.tensor(np.stack(link_bps_scdistances, axis=0)).float().flatten(1)


def prepare_link_bps_scdistances(bps_info):
    return _prepare_link_bps_scdistances(bps_info)


def _pad_tensor(x: torch.Tensor, *, length: int) -> torch.Tensor:
    if x.size(0) > length:
        raise ValueError(f"Cannot pad length {x.size(0)} to smaller length {length}")
    if x.size(0) == length:
        return x
    pad = torch.zeros((length - x.size(0), *x.shape[1:]), dtype=x.dtype)
    return torch.cat([x, pad], dim=0)


def load_chain_properties(data_source: str, class_real_idx: int):
    chain_properties = np.load(
        os.path.join(data_source, "out_chains_v2", f"chain_{class_real_idx}_properties.npz"),
        allow_pickle=True,
    )
    link_features = torch.tensor(chain_properties["links_property"]).float()
    joint_features = torch.tensor(chain_properties["joints_property"]).float()
    link_bps = _prepare_link_bps_scdistances(chain_properties["bpses"])
    return link_features, joint_features, link_bps, chain_properties


def pad_chain_features(
    link_features: torch.Tensor,
    joint_features: torch.Tensor,
    link_bps: torch.Tensor,
    *,
    max_num_links: int,
):
    num_links = link_features.size(0)
    if num_links > max_num_links:
        raise ValueError(f"num_links {num_links} exceeds max_num_links {max_num_links}")
    link_features = _pad_tensor(link_features, length=max_num_links)
    link_bps = _pad_tensor(link_bps, length=max_num_links)
    joint_features = _pad_tensor(joint_features, length=max_num_links - 1)
    return link_features, joint_features, link_bps, num_links


def compute_center_scale_from_points(
    points: torch.Tensor,
    *,
    center_mode: str = "zero",
) -> tuple[torch.Tensor, torch.Tensor]:
    if points.numel() == 0:
        return torch.zeros(3), torch.tensor(1.0)
    pts = points[:, :3]
    if center_mode == "mesh":
        pts_min = pts.min(dim=0).values
        pts_max = pts.max(dim=0).values
        center = 0.5 * (pts_min + pts_max)
    elif center_mode == "zero":
        center = torch.zeros_like(pts[0])
    else:
        raise ValueError(f"Unsupported normalize_center_mode: {center_mode}")
    scale = (pts - center).norm(dim=-1).max().clamp_min(1e-6)
    return center, scale


def compute_chain_normalization(
    data_source: str,
    indices: list[int],
    *,
    normalize_mode: str,
    normalize_center_mode: str,
) -> tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    chain_centers: Dict[int, torch.Tensor] = {}
    chain_scales: Dict[int, torch.Tensor] = {}
    if normalize_mode not in ("chain", "dataset"):
        return chain_centers, chain_scales

    cached_chain_pcs = []
    for class_real_idx in indices:
        urdf_path = os.path.join(data_source, "out_chains_v2", f"chain_{class_real_idx}.urdf")
        chain_model = ChainModel(urdf_path=urdf_path, samples_per_link=128, device="cpu")
        chain_model.update_status(torch.zeros(chain_model.dof))
        pc_full = chain_model.get_transformed_links_pc(
            num_points=None,
            mask=torch.ones(chain_model.num_links, dtype=torch.bool),
        )[0, :, :3]
        cached_chain_pcs.append((class_real_idx, pc_full))
        if normalize_mode == "chain":
            center, scale = compute_center_scale_from_points(
                pc_full,
                center_mode=normalize_center_mode,
            )
            chain_centers[class_real_idx] = center
            chain_scales[class_real_idx] = scale

    if normalize_mode == "dataset":
        all_pts = torch.cat([pc for _, pc in cached_chain_pcs], dim=0)
        center, scale = compute_center_scale_from_points(
            all_pts,
            center_mode=normalize_center_mode,
        )
        for class_real_idx, _ in cached_chain_pcs:
            chain_centers[class_real_idx] = center.clone()
            chain_scales[class_real_idx] = scale.clone()

    return chain_centers, chain_scales


def compute_instance_center_scale(
    chain_model: ChainModel,
    *,
    link_mask: torch.Tensor,
    normalize_center_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    pc_full = chain_model.get_transformed_links_pc(num_points=None, mask=link_mask)[0, :, :3]
    return compute_center_scale_from_points(pc_full, center_mode=normalize_center_mode)


def resolve_normalization(
    chain_model: ChainModel,
    *,
    class_real_idx: int,
    normalize_mode: str,
    normalize_center_mode: str,
    chain_centers: Dict[int, torch.Tensor],
    chain_scales: Dict[int, torch.Tensor],
    link_mask: torch.Tensor,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if normalize_mode == "none":
        return None, None
    if normalize_mode == "instance":
        center, scale = compute_instance_center_scale(
            chain_model,
            link_mask=link_mask,
            normalize_center_mode=normalize_center_mode,
        )
        return center, scale

    if class_real_idx not in chain_centers:
        current_q = chain_model.q.clone()
        chain_model.update_status(torch.zeros(chain_model.dof))
        pc_full = chain_model.get_transformed_links_pc(
            num_points=None,
            mask=torch.ones(chain_model.num_links, dtype=torch.bool),
        )[0, :, :3]
        center, scale = compute_center_scale_from_points(
            pc_full,
            center_mode=normalize_center_mode,
        )
        chain_centers[class_real_idx] = center
        chain_scales[class_real_idx] = scale
        chain_model.update_status(current_q.squeeze(0))

    return chain_centers[class_real_idx], chain_scales[class_real_idx]


def apply_normalization(
    coords: torch.Tensor,
    sdf: Optional[torch.Tensor],
    *,
    center: Optional[torch.Tensor],
    scale: Optional[torch.Tensor],
):
    if center is None or scale is None:
        return coords, sdf
    coords = (coords - center) / scale
    if sdf is not None:
        sdf = sdf / scale
    return coords, sdf


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        indices,
        num_instances,
        subsample,
        ik=False,
        pose_mode=False,
        fix_q_samples=False,
        sdf_mode: str = "deepsdf",
        off_surface_center: str = "zero",
        siren_sdf_mode: str = "fake",
        tsdf_band: float = 0.1,
        normalize_mode: str = "none",
        normalize_center_mode: str = "zero",
    ):
        self.subsample = subsample
        self.indices = indices
        self.num_instances = num_instances
        self.data_source = data_source
        self.ik = ik
        self.pose_mode = pose_mode
        self.fix_q_samples = fix_q_samples
        self.sdf_mode = sdf_mode
        self.off_surface_center = off_surface_center
        self.siren_sdf_mode = siren_sdf_mode
        self.tsdf_band = float(tsdf_band)
        self.normalize_mode = str(normalize_mode).lower()
        self.normalize_center_mode = str(normalize_center_mode).lower()
        if self.normalize_mode not in ("none", "instance", "chain", "dataset"):
            raise ValueError(f"Unsupported normalize_mode: {self.normalize_mode}")

        self.chain_qs = []
        self.links_properties = []
        self.joints_properties = []
        self.bpses = []
        self.chain_centers = []
        self.chain_scales = []
        cached_chain_pcs = []

        self.chain_model_ls: list[tuple[ChainModel, int]] = []
        for class_idx, class_real_idx in enumerate(indices):
            chain_properties = np.load(os.path.join(
                data_source, f'out_chains_v2', f"chain_{class_real_idx}_properties.npz"), allow_pickle=True)
            self.links_properties.append(chain_properties['links_property'])
            self.joints_properties.append(chain_properties['joints_property'])
            self.bpses.append(chain_properties['bpses'])

            chain_model = ChainModel(
                urdf_path=Path(os.path.join(
                    data_source, f'out_chains_v2',
                    f"chain_{class_real_idx}.urdf")),
                    samples_per_link=128,
                    device="cpu",
            )
            self.chain_model_ls.append((chain_model, class_real_idx))
            
            if self.normalize_mode in ("chain", "dataset"):
                chain_model.update_status(torch.zeros(chain_model.dof))
                pc_full = chain_model.get_transformed_links_pc(
                    num_points=None,
                    mask=torch.ones(chain_model.num_links, dtype=torch.bool),
                )[0, :, :3]
                cached_chain_pcs.append(pc_full)
                if self.normalize_mode == "chain":
                    if self.normalize_center_mode == "mesh":
                        # compute center of each chain at zero configuration
                        pts_min = pc_full.min(dim=0).values
                        pts_max = pc_full.max(dim=0).values
                        center = 0.5 * (pts_min + pts_max)
                    elif self.normalize_center_mode == "zero":
                        center = torch.zeros_like(pc_full[0])
                    else:
                        raise ValueError(f"Unsupported normalize_center_mode: {self.normalize_center_mode}")
                    scale = (pc_full - center).norm(dim=-1).max().clamp_min(1e-6)
                    self.chain_centers.append(center)
                    self.chain_scales.append(scale)
            else:
                self.chain_centers.append(torch.zeros(3))
                self.chain_scales.append(torch.tensor(1.0))

            if self.fix_q_samples:
                os.makedirs(os.path.join(data_source, f'chain_qs'), exist_ok=True)
                q_samples_path = os.path.join(
                    data_source, f'chain_qs',
                    f"chain_{class_real_idx}_{num_instances}_samples.pt")
                if os.path.exists(q_samples_path):
                    q_samples = torch.load(q_samples_path, weights_only=False).float()
                else:
                    q_samples = chain_model.sample_q(num_instances).float()
                    torch.save(q_samples, q_samples_path)
                self.chain_qs.append(q_samples)
            else:
                self.chain_qs.append(chain_model.sample_q(num_instances).float())

        if self.normalize_mode == "dataset":
            if not cached_chain_pcs:
                raise ValueError("normalize_mode=dataset requires chain point clouds")
            all_pts = torch.cat(cached_chain_pcs, dim=0)
            if self.normalize_center_mode == "mesh":
                pts_min = all_pts.min(dim=0).values
                pts_max = all_pts.max(dim=0).values
                center = 0.5 * (pts_min + pts_max)
            elif self.normalize_center_mode == "zero":
                center = torch.zeros_like(all_pts[0])
            else:
                raise ValueError(f"Unsupported normalize_center_mode: {self.normalize_center_mode}")
            scale = (all_pts - center).norm(dim=-1).max().clamp_min(1e-6)
            self.chain_centers = [center.clone() for _ in self.indices]
            self.chain_scales = [scale.clone() for _ in self.indices]
    
    def __len__(self):
        return len(self.indices) * self.num_instances

    def __getitem__(self, idx):
        class_idx = idx // self.num_instances
        instance_idx = idx % self.num_instances
        model, class_real_idx = self.chain_model_ls[class_idx]
        q = self.chain_qs[class_idx][instance_idx]
        
        # Randomly mask 0–80% of links for SDF/mesh queries; keep at least one link visible.
        link_mask = torch.ones((model.num_links,), dtype=torch.bool)
        frac = random.random() * 0.8
        n_to_mask = int(math.floor(frac * model.num_links))
        n_to_mask = min(n_to_mask, model.num_links - 1)  # ensure at least one remains
        # n_to_mask = 0
        if n_to_mask > 0:
            perm = torch.randperm(model.num_links)[:n_to_mask]
            link_mask[perm] = False

        scale_to = 1
        model.update_status(q)
        sdf_normals = None
        normal_mask = None

        if self.sdf_mode == "siren":
            n_surface = self.subsample // 2
            n_off = self.subsample - n_surface
            surface_pts = model.sample_surface_points(n=n_surface, mask=link_mask)
            off_pts = model.sample_off_surface_points(
                n=n_off,
                mask=link_mask,
                center_mode=self.off_surface_center,
            )
            _, surface_normals = model.query_sdf_and_normals(surface_pts, mask=link_mask)

            coords = torch.cat([surface_pts[0], off_pts[0]], dim=0)
            sdf_true = model.query_sdf(torch.cat([surface_pts, off_pts], dim=1), mask=link_mask)[0].unsqueeze(-1)

            if self.siren_sdf_mode == "true":
                sdf_labels = sdf_true
            elif self.siren_sdf_mode == "tsdf":
                band = max(self.tsdf_band, 1e-6)
                sdf_labels = torch.where(
                    sdf_true.abs() <= band,
                    torch.clamp(sdf_true, -band, band),
                    torch.full_like(sdf_true, -1.0),
                )
            elif self.siren_sdf_mode == "fake":
                sdf_surface = torch.zeros(n_surface, 1, device=surface_pts.device)
                sdf_off = -torch.ones(n_off, 1, device=off_pts.device)
                sdf_labels = torch.cat([sdf_surface, sdf_off], dim=0)
            else:
                raise ValueError(f"Unsupported siren_sdf_mode: {self.siren_sdf_mode}")

            sdf_normals = torch.zeros(self.subsample, 3, device=surface_pts.device)
            sdf_normals[:n_surface] = surface_normals[0]
            normal_mask = torch.zeros(self.subsample, 1, device=surface_pts.device, dtype=torch.bool)
            normal_mask[:n_surface] = True
            sdf = torch.cat([coords, sdf_labels], dim=-1)
        else:
            pts = model.sample_query_points(n=self.subsample, mask=link_mask, var=0.005)
            sdf_data = model.query_sdf(pts, mask=link_mask)
            sdf = torch.cat([pts[0], sdf_data[0].unsqueeze(-1)], dim=-1)

        if self.normalize_mode == "instance":
            pc_full = model.get_transformed_links_pc(num_points=None, mask=link_mask)[0, :, :3]
            if self.normalize_center_mode == "mesh":
                pts_min = pc_full.min(dim=0).values
                pts_max = pc_full.max(dim=0).values
                coord_center = 0.5 * (pts_min + pts_max)
            elif self.normalize_center_mode == "zero":
                coord_center = torch.zeros_like(pc_full[0])
            else:
                raise ValueError(f"Unsupported normalize_center_mode: {self.normalize_center_mode}")
            coord_scale = (pc_full - coord_center).norm(dim=-1).max().clamp_min(1e-6)

        coord_center = self.chain_centers[class_idx].clone()
        coord_scale = self.chain_scales[class_idx].clone()
        if self.normalize_mode != "none":
            sdf[:, :3] = (sdf[:, :3] - coord_center) / coord_scale
            if self.sdf_mode != "siren" or self.siren_sdf_mode != "fake":
                sdf[:, 3] = sdf[:, 3] / coord_scale
        if self.pose_mode:
            link_6d_poses = model.get_link_poses_world().squeeze(0)
            link_pose_repr = torch.cat([
                link_6d_poses[:, :3, 3], # translation
                matrix_to_rotation_6d(link_6d_poses[:, :3, :3]).reshape(model.num_links, 6) # 6D rotation
            ], dim=-1)

        link_features = torch.tensor(self.links_properties[class_idx]).float()
        joint_features = torch.tensor(self.joints_properties[class_idx]).float()
        # scale joint origins to match normalized mesh coordinates
        joint_features[:, :3] = joint_features[:, :3] * float(scale_to)
        
        link_bps_info = self.bpses[class_idx]
        # Offsets are already in normalized (unit-sphere) coordinates from generation; keep them.
        # Adjust scale_to_unit to account for the runtime normalization scale_to from SDF samples.
        link_bps_scdistances = [
            np.concatenate(
                [bp_info['offsets'], torch.ones((bp_info['offsets'].shape[0], 1)) * (bp_info['scale_to_unit'] * (1.0 / float(scale_to)))],
                axis=-1,
            )
            for bp_info in link_bps_info
        ]
        # link_bps_scdistances = torch.tensor(np.stack([np.concatenate([np.array((bp_info['scale_to_unit'], )), bp_info['distances']]) for bp_info in link_bps_info])).float()
        link_bps_scdistances = torch.tensor(np.stack(link_bps_scdistances, axis=0)).float().flatten(1)
        return {
            'sdf_samples': sdf, # Torch.Tensor of shape (subsample, 4)
            'sdf_normals': sdf_normals, # Torch.Tensor of shape (subsample, 3) or None
            'sdf_normal_mask': normal_mask, # Torch.Tensor of shape (subsample, 1) or None
            'coord_center': coord_center, # Torch.Tensor of shape (3,)
            'coord_scale': coord_scale, # scalar tensor
            'chain_q': q, # Torch.Tensor of shape (num_links m - 1, )
            'link_features': link_features, # Torch.Tensor of shape (num_links m, 4)
            'joint_features': joint_features, # Torch.Tensor of shape (num_links m - 1, 9)
            'link_bps_scdistances': link_bps_scdistances, # Torch.Tensor of shape (num_links m, 257)
            'sdf_tokens': self.sdf_tokens[class_idx, instance_idx] if self.ik else None, # Torch.Tensor of shape (token_dim, )
            'class_idx': class_real_idx, # scalar int
            'instance_idx': instance_idx, # scalar int
            'link_mask': link_mask, # Torch.Tensor of shape (num_links m,)
            'links_poses': link_pose_repr if self.pose_mode else None, # Torch.Tensor of shape (num_links m, 9) 
        }


class IKTokenSamples(torch.utils.data.Dataset):
    """Dataset that loads precomputed SDF tokens with link masks for IK training."""

    def __init__(
        self,
        data_source: str,
        indices,
        num_instances: int,
        tokens_dir: str,
        link_compact_repr: bool = False,
    ) -> None:
        self.indices = indices
        self.num_instances = num_instances
        self.data_source = data_source
        self.tokens_dir = Path(tokens_dir)
        self.link_compact_repr = link_compact_repr

        self.link_features = []
        self.joint_features = []
        self.link_bps = []

        for class_real_idx in indices:
            chain_properties = np.load(
                os.path.join(data_source, "out_chains_v2", f"chain_{class_real_idx}_properties.npz"),
                allow_pickle=True,
            )
            links_property = torch.tensor(chain_properties["links_property"]).float()
            joints_property = torch.tensor(chain_properties["joints_property"]).float()
            bpses = chain_properties["bpses"]

            self.link_features.append(links_property)
            self.joint_features.append(joints_property)
            self.link_bps.append(_prepare_link_bps_scdistances(bpses))

    def __len__(self):
        return len(self.indices) * self.num_instances

    def __getitem__(self, idx):
        class_idx = idx // self.num_instances
        instance_idx = idx % self.num_instances
        class_real_idx = self.indices[class_idx]

        token_path = self.tokens_dir / f"cls{class_real_idx}_inst{instance_idx}.pt"
        if not token_path.exists():
            raise FileNotFoundError(f"Missing token file: {token_path}")
        token_data = torch.load(token_path, weights_only=False, map_location="cpu")

        sdf_tokens = torch.as_tensor(token_data["sdf_tokens"]).float()
        link_mask = torch.as_tensor(token_data["link_mask"]).bool()
        q_values = torch.as_tensor(token_data["q_values"]).float()

        return {
            "chain_q": q_values,  # (num_links-1,)
            "link_features": self.link_features[class_idx],  # (num_links, 4)
            "joint_features": self.joint_features[class_idx],  # (num_links-1, 9)
            "link_bps_scdistances": self.link_bps[class_idx],  # (num_links, 257)
            "sdf_tokens": sdf_tokens,  # (num_masks, token_dim)
            "link_mask": link_mask,  # (num_masks, num_links)
            "class_idx": int(token_data.get("cls_idx", class_real_idx)),
            "instance_idx": int(token_data.get("instance_idx", instance_idx)),
        }

def make_collate_fn(max_num_links: int):
    def collate_fn(batch):
        B = len(batch)
        # infer shapes
        sdf_shape = batch[0]["sdf_samples"].shape  # (subsample, 4)
        has_normals = batch[0]["sdf_normals"] is not None
        has_normal_mask = batch[0]["sdf_normal_mask"] is not None
        link_feat_dim = batch[0]["link_features"].shape[-1]          # 4
        joint_feat_dim = batch[0]["joint_features"].shape[-1]        # 9
        bps_dim = batch[0]["link_bps_scdistances"].shape[-1]         # 257
        links_poses_dim = batch[0]["links_poses"].shape[-1] if batch[0]["links_poses"] is not None else 0

        # allocate
        sdf_samples = torch.stack([b["sdf_samples"] for b in batch], dim=0)
        chain_q = torch.zeros(B, max_num_links - 1, dtype=torch.float32)
        link_features = torch.zeros(B, max_num_links, link_feat_dim, dtype=torch.float32)
        joint_features = torch.zeros(B, max_num_links - 1, joint_feat_dim, dtype=torch.float32)
        link_bps = torch.zeros(B, max_num_links, bps_dim, dtype=torch.float32)
        links_poses = torch.zeros(B, max_num_links, links_poses_dim, dtype=torch.float32) if links_poses_dim > 0 else None
        sdf_normals = torch.stack([b["sdf_normals"] for b in batch], dim=0) if has_normals else None
        sdf_normal_mask = torch.stack([b["sdf_normal_mask"] for b in batch], dim=0) if has_normal_mask else None
        coord_center = torch.stack([b["coord_center"] for b in batch], dim=0)
        coord_scale = torch.stack([b["coord_scale"] for b in batch], dim=0)
        masks = torch.zeros(B, 3 * max_num_links - 2, dtype=torch.bool)
        link_masks = torch.zeros(B, max_num_links, dtype=torch.bool)

        class_idx = []
        instance_idx = []

        for i, b in enumerate(batch):
            n_links = b["link_features"].shape[0]
            if n_links > max_num_links:
                raise ValueError(f"n_links {n_links} exceeds max_num_links {max_num_links}")
            n_joints = n_links - 1

            chain_q[i, :n_joints] = b["chain_q"]
            link_features[i, :n_links] = b["link_features"]
            joint_features[i, :n_joints] = b["joint_features"]
            link_bps[i, :n_links] = b["link_bps_scdistances"]
            link_masks[i, :n_links] = b["link_mask"]
            if links_poses is not None:
                links_poses[i, :n_links] = b["links_poses"]
            masks[i, :3 * n_links - 2] = True

            class_idx.append(b["class_idx"])
            instance_idx.append(b["instance_idx"])

        return {
            "sdf_samples": sdf_samples,                 # (B, subsample, 4)
            "sdf_normals": sdf_normals,                 # (B, subsample, 3) or None
            "sdf_normal_mask": sdf_normal_mask,         # (B, subsample, 1) or None
            "coord_center": coord_center,               # (B, 3)
            "coord_scale": coord_scale,                 # (B,)
            "chain_q": chain_q,                         # (B, max_num_links-1)
            "link_features": link_features,             # (B, max_num_links, 4)
            "joint_features": joint_features,           # (B, max_num_links-1, 9)
            "link_bps_scdistances": link_bps,           # (B, max_num_links, 257)
            "sdf_tokens": torch.stack([b["sdf_tokens"] for b in batch], dim=0) if batch[0]["sdf_tokens"] is not None else None, # (B, token_dim)
            "class_idx": class_idx,                     # list of scalars
            "instance_idx": instance_idx,               # list of scalars
            "mask": masks,                              # (B, 3*max_num_links-2)
            "link_mask": link_masks,                    # (B, max_num_links)
            "links_poses": links_poses,               # (B, max_num_links, 9) or None
        }
    return collate_fn

def make_ik_collate_fn(max_num_links: int):
    def collate_fn(batch):
        total_masks = sum(item["sdf_tokens"].shape[0] for item in batch)
        if total_masks == 0:
            raise ValueError("Empty batch received in IK collate.")

        link_feat_dim = batch[0]["link_features"].shape[-1]
        joint_feat_dim = batch[0]["joint_features"].shape[-1]
        bps_dim = batch[0]["link_bps_scdistances"].shape[-1]
        token_dim = batch[0]["sdf_tokens"].shape[-1]

        chain_q = torch.zeros(total_masks, max_num_links - 1, dtype=torch.float32)
        link_features = torch.zeros(total_masks, max_num_links, link_feat_dim, dtype=torch.float32)
        joint_features = torch.zeros(total_masks, max_num_links - 1, joint_feat_dim, dtype=torch.float32)
        link_bps = torch.zeros(total_masks, max_num_links, bps_dim, dtype=torch.float32)
        masks = torch.zeros(total_masks, 3 * max_num_links - 2, dtype=torch.bool)
        link_masks = torch.zeros(total_masks, max_num_links, dtype=torch.bool)
        sdf_tokens = torch.zeros(total_masks, token_dim, dtype=torch.float32)

        class_idx = []
        instance_idx = []

        cursor = 0
        for item in batch:
            n_links = item["link_features"].shape[0]
            if n_links > max_num_links:
                raise ValueError(f"n_links {n_links} exceeds max_num_links {max_num_links}")
            n_joints = n_links - 1
            n_masks = item["sdf_tokens"].shape[0]

            for m in range(n_masks):
                chain_q[cursor, :n_joints] = item["chain_q"]
                link_features[cursor, :n_links] = item["link_features"]
                joint_features[cursor, :n_joints] = item["joint_features"]
                link_bps[cursor, :n_links] = item["link_bps_scdistances"]
                masks[cursor, : 3 * n_links - 2] = True
                link_masks[cursor, :n_links] = item["link_mask"][m]
                sdf_tokens[cursor] = item["sdf_tokens"][m]

                class_idx.append(item["class_idx"])
                instance_idx.append(item["instance_idx"])
                cursor += 1

        return {
            "chain_q": chain_q,  # (B, max_num_links-1)
            "link_features": link_features,  # (B, max_num_links, 4)
            "joint_features": joint_features,  # (B, max_num_links-1, 9)
            "link_bps_scdistances": link_bps,  # (B, max_num_links, 257)
            "sdf_tokens": sdf_tokens,  # (B, token_dim)
            "class_idx": class_idx,
            "instance_idx": instance_idx,
            "mask": masks,  # (B, 3*max_num_links-2)
            "link_mask": link_masks,  # (B, max_num_links)
        }

    return collate_fn

def get_dataloader(
    data_source: str,
    indices,
    num_instances: int,
    subsample: int,
    batch_size: int,
    max_num_links: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    ik=False,
    pose_mode=False,
    fix_q_samples=False,
    sdf_mode: str = "deepsdf",
    off_surface_center: str = "zero",
    siren_sdf_mode: str = "fake",
    tsdf_band: float = 0.1,
    normalize_mode: str = "none",
    normalize_center_mode: str = "zero",
):
    dataset = SDFSamples(
        data_source=data_source,
        indices=indices,
        num_instances=num_instances,
        subsample=subsample,
        ik=ik,
        pose_mode=pose_mode,
        fix_q_samples=fix_q_samples,
        sdf_mode=sdf_mode,
        off_surface_center=off_surface_center,
        siren_sdf_mode=siren_sdf_mode,
        tsdf_band=tsdf_band,
        normalize_mode=normalize_mode,
        normalize_center_mode=normalize_center_mode,
    )
    collate_fn = make_collate_fn(max_num_links)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    return loader


def get_ik_dataloader(
    data_source: str,
    indices,
    num_instances: int,
    tokens_dir: str,
    batch_size: int,
    max_num_links: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    link_compact_repr: bool = False,
):
    dataset = IKTokenSamples(
        data_source=data_source,
        indices=indices,
        num_instances=num_instances,
        tokens_dir=tokens_dir,
        link_compact_repr=link_compact_repr,
    )
    collate_fn = make_ik_collate_fn(max_num_links)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    return loader

if __name__ == "__main__":
    ### Test the dataset loading.
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_source = "./data/"
    indices = list(range(1, 30))
    num_instances = 100
    subsample = 16384
    batch_size = 32
    max_num_links = 5
    num_workers = 0

    from data_process.visualize_samples import visualize_sdf_points_plotly

    siren_loader = get_dataloader(
        data_source=data_source,
        indices=indices,
        num_instances=num_instances,
        subsample=subsample,
        batch_size=1,
        max_num_links=max_num_links,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pose_mode=False,
        # parameters
        sdf_mode="deepsdf",
        off_surface_center="zero",
        siren_sdf_mode="true",
        tsdf_band=0.05,
        normalize_mode="dataset",
        normalize_center_mode="zero",
    )
    for data_batch in siren_loader:
        vis_cls = data_batch["class_idx"][0]
        urdf_path = Path(f"data/out_chains_v2/chain_{vis_cls}.urdf")
        model = ChainModel(urdf_path, samples_per_link=128, device="cpu")
        q = data_batch["chain_q"][0]
        model.update_status(q[: model.dof])
        mesh = model.get_trimesh_q(0, boolean_merged=True, mask=data_batch["link_mask"][0, :model.num_links]).copy()

        # Apply the same normalization used for SDF samples so visualization stays consistent.
        center = data_batch["coord_center"][0].cpu().numpy()
        scale = float(data_batch["coord_scale"][0])
        mesh.apply_translation(-center)
        mesh.apply_scale(1.0 / max(scale, 1e-8))

        out_dir = Path("./outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        visualize_sdf_points_plotly(
            mesh=mesh,
            points=data_batch["sdf_samples"][0, :, :3].cpu().numpy(),
            sdf=data_batch["sdf_samples"][0, :, 3].cpu().numpy(),
            normals=data_batch["sdf_normals"][0].cpu().numpy() if data_batch["sdf_normals"] is not None else None,
            title="SIREN mode samples",
            downsample_ratio=0.05,
            max_points=20000,
            normal_stride=1,
            save_path=str(out_dir / "siren_samples.png"),
            save_html_path=str(out_dir / "siren_samples.html"),
            show=True,
            show_sign="all",  # "all" | "negative" | "zero" | "positive"
        )

        print("Done")
