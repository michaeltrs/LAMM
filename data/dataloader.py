from __future__ import print_function, division
import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import trimesh


def get_dataloader(paths_file, root_dir, transform=None, batch_size=32, num_workers=4, shuffle=True,
                   return_paths=False, my_collate=None):
    dataset = Mesh3DDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, return_paths=return_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             collate_fn=my_collate)
    return dataloader


class Mesh3DDataset(Dataset):
    """
    3D Mesh Dataset.

    Args:
    csv_file (string): Path to the csv file with .obj mesh paths.
    root_dir (string): Root directory with all the meshes. Paths found in csv_file are defined relative to this path.
    transform (callable, optional): Optional transform to be applied on a sample.
    Returns:
        dict: __getitem__ returns a dict containing a set of 3D points and optionally its file path.
    """
    def __init__(self, csv_file, root_dir, transform=None, return_paths=False):
        if type(csv_file) == str:
            self.data_paths = pd.read_csv(csv_file, header=None)
        elif type(csv_file) in [list, tuple]:
            self.data_paths = pd.concat([pd.read_csv(csv_file_, header=None) for csv_file_ in csv_file], axis=0).reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx, no_transform=False):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Join root_dir and provided path to create sample absolute path
        sample_path = os.path.join(self.root_dir, self.data_paths.iloc[idx, 0])

        # Read mesh object and extract vertices
        verts = np.array(trimesh.load(sample_path).vertices)

        sample = {
            'verts': verts
        }

        # Optionally apply a transformation (e.g., normalization, augmentation, etc.)
        if self.transform:
            sample = self.transform(sample)

        # Optionally return paths
        if self.return_paths:
            sample['path'] = sample_path
        
        return sample
