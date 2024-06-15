from __future__ import print_function, division

import pickle
import torch
from torchvision import transforms


def mesh3D_transform(mean_std_file):
    """
    Constructs a composition of transforms for 3D mesh data.

    This function creates a pipeline of transformations to convert 3D mesh data into a
    format suitable for model training or inference. The pipeline includes converting
    the data to tensors and normalizing it using provided mean and standard deviation values.

    Parameters:
    - mean_std_file (str): Path to a file containing mean and standard deviation values
      for normalization.

    Returns:
    - torchvision.transforms.Compose: A composed transform consisting of ToTensor and
      Normalize transforms.
    """
    transform_list = []

    transform_list.append(ToTensor())
    transform_list.append(Normalize(mean_std_file=mean_std_file))

    return transforms.Compose(transform_list)


class Normalize(object):
    """
    Normalize a 3D mesh sample.

    This transform normalizes each vertex of the mesh by subtracting the mean and dividing
    by the standard deviation. These values are loaded from a provided file.

    Attributes:
    - mean (torch.Tensor): The mean to subtract.
    - std (torch.Tensor): The standard deviation for division, ensuring numerical stability.

    Parameters:
    - mean_std_file (str): Path to the file containing mean and standard deviation values.
    - h (float, optional): A small value added to the standard deviation to ensure numerical
      stability. Defaults to 1e-7.
    """
    def __init__(self, mean_std_file, h=1e-7):

        with open(mean_std_file, 'rb') as handle:
            mean_std = pickle.load(handle, encoding='latin1')

        self.mean = torch.tensor(mean_std['mean']).to(torch.float32)
        self.std = torch.tensor(mean_std['std']).to(torch.float32) + h

    def __call__(self, sample):
        for key in sample:
            if 'verts' in key:
                sample[key] = (sample[key] - self.mean) / self.std
        return sample


# 1
class ToTensor(object):
    """
    Convert numerical data in a sample to PyTorch tensors.

    This transform iterates over each key-value pair in the input sample and converts it into a PyTorch tensor of
    float32 type. This is typically used to convert numpy arrays or lists to tensors.
    """
    def __call__(self, sample):
        for key in sample:
            sample[key] = torch.tensor(sample[key]).to(torch.float32)
        return sample
