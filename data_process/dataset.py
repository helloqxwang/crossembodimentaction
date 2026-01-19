#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import math
import numpy as np
import os
import random
import torch
import torch.utils.data
from tqdm import tqdm
from pathlib import Path
import sys
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
        sdf_mode: str = "l1",
        off_surface_center: str = "zero",
        sdf_target_mode: str = "fake",
        tsdf_band: float = 0.1,
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
        self.sdf_target_mode = sdf_target_mode
        self.tsdf_band = float(tsdf_band)

        self.chain_qs = []
        self.links_properties = []
        self.joints_properties = []
        self.bpses = []

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
        n_to_mask = 0
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

            if self.sdf_target_mode == "true":
                sdf_labels = sdf_true
            elif self.sdf_target_mode == "tsdf":
                band = max(self.tsdf_band, 1e-6)
                sdf_labels = torch.where(
                    sdf_true.abs() <= band,
                    torch.clamp(sdf_true, -band, band),
                    torch.full_like(sdf_true, -1.0),
                )
            elif self.sdf_target_mode == "fake":
                sdf_surface = torch.zeros(n_surface, 1, device=surface_pts.device)
                sdf_off = -torch.ones(n_off, 1, device=off_pts.device)
                sdf_labels = torch.cat([sdf_surface, sdf_off], dim=0)
            else:
                raise ValueError(f"Unsupported sdf_target_mode: {self.sdf_target_mode}")

            sdf_normals = torch.zeros(self.subsample, 3, device=surface_pts.device)
            sdf_normals[:n_surface] = surface_normals[0]
            normal_mask = torch.zeros(self.subsample, 1, device=surface_pts.device, dtype=torch.bool)
            normal_mask[:n_surface] = True
            sdf = torch.cat([coords, sdf_labels], dim=-1)
        else:
            pts = model.sample_query_points(n=self.subsample, mask=link_mask, var=0.02)
            sdf_data = model.query_sdf(pts, mask=link_mask)
            sdf = torch.cat([pts[0], sdf_data[0].unsqueeze(-1)], dim=-1)
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
    sdf_mode: str = "l1",
    off_surface_center: str = "zero",
    sdf_target_mode: str = "fake",
    tsdf_band: float = 0.1,
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
        sdf_target_mode=sdf_target_mode,
        tsdf_band=tsdf_band,
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

    # loader = get_dataloader(
    #     data_source=data_source,
    #     indices=indices,
    #     num_instances=num_instances,
    #     subsample=subsample,
    #     batch_size=batch_size,
    #     max_num_links=max_num_links,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     drop_last=True,
    #     pose_mode=True,
    # )

    # print(f"Dataset size: {len(loader.dataset)}")
    # for i, batch in enumerate(tqdm(loader)):
    #     pass
    #     # print("Batch keys:", batch.keys())
    #     # print("sdf_samples:", batch["sdf_samples"].shape)
    #     # print("chain_q:", batch["chain_q"].shape)
    #     # print("link_features:", batch["link_features"].shape)
    #     # print("joint_features:", batch["joint_features"].shape)
    #     # print("link_bps_scdistances:", batch["link_bps_scdistances"].shape)
    #     # print("mask:", batch["mask"].shape)
    #     # print("links_poses:", batch["links_poses"].shape if batch["links_poses"] is not None else None)
    #     ### Visulization code
    #     vis_cls = batch["class_idx"][0]
    #     urdf_path = Path(f"data/out_chains_v2/chain_{vis_cls}.urdf")
    #     model = ChainModel(urdf_path, samples_per_link=128)
    #     q = batch["chain_q"][0] # Here is padded q. So need some inspection to get the real q.
    #     model.update_status(q[:1])

    #     visualize_sdf_viser(
    #         mesh=model.get_trimesh_q(0, boolean_merged=True, mask=batch['link_mask'][0, :1]),
    #         sdf_samples=batch["sdf_samples"][0].cpu().numpy(),
    #         host="127.0.0.1",
    #         port=9200,
    #         downsample_ratio=0.03
    #     )

    #     break

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
        sdf_mode="siren",
        off_surface_center="mesh",
        sdf_target_mode="tsdf",
        tsdf_band=0.05,
    )
    siren_batch = next(iter(siren_loader))
    vis_cls = siren_batch["class_idx"][0]
    urdf_path = Path(f"data/out_chains_v2/chain_{vis_cls}.urdf")
    model = ChainModel(urdf_path, samples_per_link=128, device="cpu")
    q = siren_batch["chain_q"][0]
    model.update_status(q[: model.dof])
    mesh = model.get_trimesh_q(0, boolean_merged=True, mask=siren_batch["link_mask"][0, :model.num_links])

    out_dir = Path("./outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    visualize_sdf_points_plotly(
        mesh=mesh,
        points=siren_batch["sdf_samples"][0, :, :3].cpu().numpy(),
        sdf=siren_batch["sdf_samples"][0, :, 3].cpu().numpy(),
        normals=siren_batch["sdf_normals"][0].cpu().numpy(),
        title="SIREN mode samples",
        downsample_ratio=0.05,
        max_points=20000,
        normal_stride=1,
        save_path=str(out_dir / "siren_samples.png"),
        save_html_path=str(out_dir / "siren_samples.html"),
        show=False,
        show_sign="all",  # "all" | "negative" | "zero" | "positive"
    )

    print("Done")
