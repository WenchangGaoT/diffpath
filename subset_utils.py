import torch
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import numpy as np


def load_subset_data(loader: DataLoader, subset_ratio: float):
    '''
    Returns a new dataloader loading a random subset from the original dataloader.

    params:
        loader: torch.utils.data.DataLoader, the original dataloder containing whole dataset
        subset ratio: float, the ratio of subset length to dataset length

    returns:
        torch.utils.data.DataLoader

    Examples:
        from ood_utils import load_data
        cifar_loader = load_data(args.dataset, args.data_dir, args.batch_size, args.image_size, train=True)
        cifar_subset_loader = subset_loader(cifar_loader, 0.1)
    '''
    num_samples = len(loader.dataset)
    subset_samples = int(subset_ratio * num_samples)
    indices = torch.randperm(num_samples).numpy()[:subset_samples]
    sampler = SubsetRandomSampler(indices)
    return DataLoader(
        dataset=loader.dataset,
        num_workers=loader.num_workers,
        batch_size=loader.batch_size,
        shuffle=False,
        sampler=sampler
    )
