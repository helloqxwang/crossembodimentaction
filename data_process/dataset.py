#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
from tqdm import tqdm


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

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples, scale_to


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
        load_ram=False,

    ):
        self.subsample = subsample
        self.indices = indices
        self.num_instances = num_instances
        self.data_source = data_source

        self.npzfiles = []
        self.chain_qs = []
        self.links_properties = []
        self.joints_properties = []
        self.bpses = []

        for class_idx, class_real_idx in enumerate(indices):
            chain_qs = np.load(os.path.join(
                data_source, f'chain_meshes',
                f"chain_{class_real_idx}_q.npz",))['q']
            chain_properties = np.load(os.path.join(
                data_source, f'out_chains',
                f"chain_{class_real_idx}_properties.npz"), allow_pickle=True)
            self.chain_qs.append(chain_qs)
            self.links_properties.append(chain_properties['links_property'])
            self.joints_properties.append(chain_properties['joints_property'])
            self.bpses.append(chain_properties['bpses'])

            for instance_idx in range(num_instances):
                filename = os.path.join(
                    data_source,'chain_samples_normalized',
                    f'chain_{class_real_idx}_{instance_idx}',
                    f"deepsdf.npz",
                )
                if not os.path.isfile(filename):
                    continue
                    raise FileNotFoundError(filename)
                self.npzfiles.append((filename, class_real_idx, class_idx, instance_idx))

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for filename, class_idx, instance_idx in tqdm(self.npzfiles):
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

    def __len__(self):
        return len(self.npzfiles)

    def __getitem__(self, idx):
        filename, class_real_idx, class_idx, instance_idx = self.npzfiles[idx]
        if self.load_ram:
            sdf, scale_to = unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample)
        else:
            sdf, scale_to = unpack_sdf_samples(filename, self.subsample)

        q = torch.tensor(self.chain_qs[class_idx][instance_idx]).float()
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
        # link_bps_scdistances = [np.concatenate([np.array((bp_info['scale_to_unit'], )), bp_info['distances']]) for bp_info in link_bps_info]
        link_bps_scdistances = torch.tensor(np.stack(link_bps_scdistances, axis=0)).float().flatten(1)
        return {
            'sdf_samples': sdf, # Torch.Tensor of shape (subsample, 4)
            'chain_q': q, # Torch.Tensor of shape (num_links m - 1, )
            'link_features': link_features, # Torch.Tensor of shape (num_links m, 4)
            'joint_features': joint_features, # Torch.Tensor of shape (num_links m - 1, 9)
            'link_bps_scdistances': link_bps_scdistances, # Torch.Tensor of shape (num_links m, 257)
            'class_idx': class_real_idx, # scalar int
            'instance_idx': instance_idx, # scalar int
        }

def make_collate_fn(max_num_links: int):
    def collate_fn(batch):
        B = len(batch)
        # infer shapes
        sdf_shape = batch[0]["sdf_samples"].shape  # (subsample, 4)
        link_feat_dim = batch[0]["link_features"].shape[-1]          # 4
        joint_feat_dim = batch[0]["joint_features"].shape[-1]        # 9
        bps_dim = batch[0]["link_bps_scdistances"].shape[-1]         # 257

        # allocate
        sdf_samples = torch.stack([b["sdf_samples"] for b in batch], dim=0)
        chain_q = torch.zeros(B, max_num_links - 1, dtype=torch.float32)
        link_features = torch.zeros(B, max_num_links, link_feat_dim, dtype=torch.float32)
        joint_features = torch.zeros(B, max_num_links - 1, joint_feat_dim, dtype=torch.float32)
        link_bps = torch.zeros(B, max_num_links, bps_dim, dtype=torch.float32)

        masks = torch.zeros(B, 3 * max_num_links - 2, dtype=torch.bool)

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

            masks[i, :3 * n_links - 2] = True

            class_idx.append(b["class_idx"])
            instance_idx.append(b["instance_idx"])

        return {
            "sdf_samples": sdf_samples,                 # (B, subsample, 4)
            "chain_q": chain_q,                         # (B, max_num_links-1)
            "link_features": link_features,             # (B, max_num_links, 4)
            "joint_features": joint_features,           # (B, max_num_links-1, 9)
            "link_bps_scdistances": link_bps,           # (B, max_num_links, 257)
            "class_idx": class_idx,                     # list of scalars
            "instance_idx": instance_idx,               # list of scalars
            "mask": masks,                               # (B, 3*max_num_links-2)
        }
    return collate_fn

def get_dataloader(
    data_source: str,
    indices,
    num_instances: int,
    subsample: int,
    batch_size: int,
    max_num_links: int,
    load_ram: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
):
    dataset = SDFSamples(
        data_source=data_source,
        indices=indices,
        num_instances=num_instances,
        subsample=subsample,
        load_ram=load_ram,
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
    data_source = "./data/"
    indices = list(range(100))
    num_instances = 100
    subsample = 16384
    batch_size = 64
    max_num_links = 5
    num_workers = 0

    loader = get_dataloader(
        data_source=data_source,
        indices=indices,
        num_instances=num_instances,
        subsample=subsample,
        batch_size=batch_size,
        max_num_links=max_num_links,
        load_ram=False,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    print(f"Dataset size: {len(loader.dataset)}")
    for i, batch in enumerate(tqdm(loader)):
        print("Batch keys:", batch.keys())
        print("sdf_samples:", batch["sdf_samples"].shape)
        print("chain_q:", batch["chain_q"].shape)
        print("link_features:", batch["link_features"].shape)
        print("joint_features:", batch["joint_features"].shape)
        print("link_bps_scdistances:", batch["link_bps_scdistances"].shape)
        print("mask:", batch["mask"].shape)

    print("Done")