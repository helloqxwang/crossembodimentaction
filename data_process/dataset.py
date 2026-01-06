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

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


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

        for class_idx in indices:
            for intance_idx in range(num_instances):
                filename = os.path.join(
                    data_source,
                    f'chain_{class_idx}_{intance_idx}',
                    f"deepsdf.npz",
                )
                if not os.path.isfile(filename):
                    continue
                    raise FileNotFoundError(filename)
                self.npzfiles.append(filename)

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for filename in tqdm(self.npzfiles):
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
        filename = self.npzfiles[idx]
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, self.subsample), idx

if __name__ == "__main__":
    ### Test the dataset loading.
    data_source = "./data/chain_samples/"
    indices = list(range(100))
    num_instances = 100
    subsample = 16384
    dataset = SDFSamples(
        data_source,
        indices,
        num_instances,
        subsample,
        load_ram=False,
    )
    print(f"Dataset size: {len(dataset)}")
    a = dataset[0]
    print(f"Sample shape: {a[0].shape}, index: {a[1]}")

    sdf_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size= 64,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    for i, data in enumerate(tqdm(sdf_loader)):
        samples, indices = data
        print(f"Batch {i}: samples shape: {samples.shape}, indices: {indices}")
    
    print("Done")