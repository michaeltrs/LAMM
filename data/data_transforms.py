from __future__ import print_function, division
import torch
from torchvision import transforms
import pickle


def mesh3D_transform(mean_std_file):
    """
    """
    transform_list = []

    transform_list.append(ToTensor())
    transform_list.append(Normalize(mean_std_file=mean_std_file))

    return transforms.Compose(transform_list)


class Normalize(object):
    def __init__(self, mean_std_file, h=1e-7):

        with open(mean_std_file, 'rb') as handle:
            mean_std = pickle.load(handle, encoding='latin1')

        self.mean = torch.tensor(mean_std['mean']).to(torch.float32)
        self.std = torch.tensor(mean_std['std']).to(torch.float32) + h
        print(self.mean.shape, self.std.shape)

    def __call__(self, sample):
        for key in sample:
            if 'verts' in key:
                sample[key] = (sample[key] - self.mean) / self.std
        return sample


# 1
class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """

    def __call__(self, sample):
        for key in sample:
            sample[key] = torch.tensor(sample[key]).to(torch.float32)
        return sample
