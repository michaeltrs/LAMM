import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lstsq
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import get_dataloaders
from models import LAMM
from utils.config_utils import read_yaml, get_template_mean_std
from utils.mesh_utils import get_region_boundaries
from utils.torch_utils import load_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples using a neural network.")
    parser.add_argument('--config_file', type=Path, required=True,
                        help="Path to the configuration file.")
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint.pth',
                        help="Name of the checkpoint file.")
    parser.add_argument('--machine', type=str, default='local',
                        help="Machine to run genration and retrieve data from, defaults to 'local")
    parser.add_argument('--num_epochs', type=int, default=5,
                        help="Number of epochs to gather displacement statistics from. Recommended >1, default 5.")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to use for computations ('cpu' or 'cuda:index').")
    args = parser.parse_args()
    return args


def affine_transform(source_points, target_points):
    """
    Calculate the affine transformation matrix that maps a set of source points to a set of target points.
    This function computes the affine transformation matrix that best maps the source points to the target points
    in a least squares sense. The function returns an augmented affine transformation matrix of size 3x4, which includes
    the translation components.

    Parameters:
    - source_points (np.ndarray): A Nx3 matrix of N source points.
    - target_points (np.ndarray): A Nx3 matrix of N target points, corresponding to the source points.

    Returns:
    - np.ndarray: The augmented affine transformation matrix of size 3x4.

    Raises:
    - AssertionError: If the source points and target points do not have the same shape.

    Note:
    - The source and target points matrices should have the same shape, and the number of points (N) should be at least 3
      for a 2D transformation and 4 for a 3D transformation to uniquely determine the affine transformation.
    """
    # Number of points should be the same
    assert source_points.shape == target_points.shape, "Source points and target points must have the same shape."

    # Append ones to the source points matrix
    num_points = source_points.shape[0]
    ones_column = np.ones((num_points, 1))
    source_points_augmented = np.hstack([source_points, ones_column])

    # Solve for the affine transformation matrix
    affine_matrix, _, _, _ = lstsq(source_points_augmented, target_points)

    # Append a row to make it a 3x3 matrix
    affine_matrix_augmented = np.vstack([affine_matrix.T, [0, 0, 0, 1]])

    return affine_matrix_augmented


def transform_points(points, affine_matrix):
    """
    Apply an affine transformation to a set of points using a given affine transformation matrix.
    This function transforms a set of points in Cartesian coordinates to a new set of points using the given affine
    transformation matrix. The transformation matrix should be an augmented matrix of size 3x4 for 3D points
    transformation.

    Parameters:
    - points (np.ndarray): A Nx3 matrix of N points in Cartesian coordinates to be transformed.
    - affine_matrix (np.ndarray): The augmented affine transformation matrix of size 3x4 used to perform the transformation.

    Returns:
    - np.ndarray: The transformed points in a Nx2 or Nx3 matrix, depending on the input points' dimensionality.

    Note:
    - The points matrix and the affine transformation matrix must be compatible in dimensions for the transformation to be applied.
    - This function assumes that the affine_matrix is already in the correct form to be applied to the points, including
      the translation components.
    """
    # Convert points to homogeneous coordinates by adding a column of ones
    # points, affine_matrix = trg_fp, M
    num_points = points.shape[0]
    ones_column = np.ones((num_points, 1))
    points_homogeneous = np.hstack([points, ones_column])

    # Perform matrix multiplication to apply the transformation
    transformed_points_homogeneous = np.dot(points_homogeneous, affine_matrix.T)

    # Convert back to Cartesian coordinates by removing the last column
    transformed_points = transformed_points_homogeneous[:, :3]

    return transformed_points


def prepare_inference_files(args):
    """
    Prepares and saves necessary model files for inference, focusing on global and local geometry control as discussed
    in Section 4.3 of the documentation or paper. This function saves severarl files later used for inference:
        - per-vertex mean and standard deviation
        - region boundaries
        - global latent code distribution
        - local displacement distributions for control landmarks per region

    Parameters:
    - args (Namespace): Command line arguments or any object with attributes config_file, checkpoint_name, machine,
                        num_epochs, and device, which are used to specify the configuration file, model checkpoint,
                        computation machine, number of epochs for processing, and the device for computation, respectively.

    Returns:
    - tuple: A tuple containing paths to the saved files for region boundaries, displacement and global
             latent code distribution. These files are crucial for the inference stage for geometry manipulation.

    Raises:
    - FileNotFoundError: If the specified config_file or checkpoint_name does not exist or is inaccessible.
    - ValueError: If the configuration file is invalid or incompatible with the expected format.

    Note:
    - This function is specifically designed to work with the training process and model architecture described in
      Section 4.3 of the associated documentation or paper.
    """
    config_file = args.config_file
    checkpoint_name = args.checkpoint_name
    machine = args.machine
    num_epochs = args.num_epochs
    device = f'cuda:{args.device}'

    savedir = f'{os.path.dirname(config_file)}/files'

    os.makedirs(savedir, exist_ok=True)

    config = read_yaml(config_file)

    with open(config['MODEL']['region_ids_file'], 'rb') as handle:
        region_ids = {int(k): v for k, v in pickle.load(handle, encoding='latin1').items()}

    config['MACHINE'] = machine

    control_lms = {int(k): v for k, v in config['MODEL']['control_vertices'].items()}
    print('control lms: ', control_lms.keys())

    mesh, mean, std, mm_mult = get_template_mean_std(config)

    # Data per-vertex mean and std
    mean_std_savename = f'{savedir}/mean_std.pickle'
    with open(mean_std_savename, 'wb') as file:
        print(type(mean), type(std))
        pickle.dump({'mean': mean, 'std': std, 'mm_mult': 1000}, file)

    # Region boundaries
    boundaries = {int(k): v for k, v in get_region_boundaries(mesh, region_ids).items()}

    boundaries_savename = f'{savedir}/region_boundaries.pickle'
    with open(boundaries_savename, 'wb') as file:
        pickle.dump(boundaries, file)

    train_loader = get_dataloaders(config)['training']

    config['MODEL']['manipulation'] = True
    net = LAMM(config['MODEL']).to(device)
    load_from_checkpoint(net, f'{os.path.dirname(config_file)}/{checkpoint_name}', device=device, partial_restore=False)

    Z = []
    Dlms = {key: [] for key in control_lms.keys()}

    for epoch in range(num_epochs):

        print(f'epoch {epoch + 1} of {num_epochs}')

        for step, sample in enumerate(tqdm(train_loader)):

            source = sample['verts']

            # gather latent codes only during the first epoch as these will be the same in remaining epochs
            if epoch == 0:
                z = net.encode(source.to(device))[0]
                Z.append(z[:, 0].detach().cpu())

            source = source.cpu()
            target = torch.roll(source, 1, 0)
            source = source.numpy()
            target = target.numpy()
            B = source.shape[0]

            for k in range(B):

                for i, name in enumerate(control_lms.keys()):

                    source_ = source[k]
                    modified_ = source[k].copy()
                    target_ = target[k]
                    modified_naive = source_.copy()

                    idx = boundaries[name]

                    trg_fp = target_[region_ids[name]]

                    modified_naive[region_ids[name]] = trg_fp

                    src_bound = source_[idx]
                    trg_bound = target_[idx]

                    M = affine_transform(trg_bound, src_bound)

                    if (np.abs(np.diag(M)) > 1).any():
                        continue

                    # Apply transformation to the entire source mesh
                    target_aligned = transform_points(trg_fp, M)
                    modified_[region_ids[name]] = target_aligned  # src_fp

                    if name in control_lms:
                        Dlms[name].append((modified_ - source_)[control_lms[name]])

    # Global latent code distribution
    # Fit a multivariate gaussian to latent codes. Will be used to generate new faces (global)
    Z = torch.cat(Z).cpu().detach().numpy()
    mu = np.mean(Z, axis=0)
    sigma = np.cov(Z, rowvar=0)
    gaussian_id_savename = f'{savedir}/gaussian_id.pickle'  # os.path.dirname(config_file)}/files
    with open(gaussian_id_savename, 'wb') as f:
        pickle.dump({'mean': mu, 'sigma': sigma}, f)

    # Local displacement statistics
    # Gather displacement statistics for control landmarks per region. Will be used to randomly sample new regions (local)
    displacement_stats = {}
    for i, key in enumerate(Dlms.keys()):
        X = np.stack(Dlms[key])
        N = len(Dlms[key])
        X = X.reshape((N, -1))
        mu = np.mean(X, axis=0)
        sigma = np.cov(X, rowvar=0)
        displacement_stats[key] = {'mean': mu, 'std': sigma}

    displ_savename = f'{savedir}/displacement_stats.pickle'  # {os.path.dirname(config_file)}/files
    with open(displ_savename, 'wb') as file:
        pickle.dump(displacement_stats, file)

    return boundaries_savename, displ_savename, gaussian_id_savename


if __name__ == "__main__":

    args = parse_args()
    prepare_inference_files(args)
