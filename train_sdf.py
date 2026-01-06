import torch
from tqdm import tqdm

from data_process.dataset import SDFSamples
from networks.deep_sdf_decoder import DeepSDFDecoder


def main():
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