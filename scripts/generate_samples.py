import os
import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import LAMM
from utils.config_utils import read_yaml, get_template_mean_std
from utils.torch_utils import load_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples using a neural network.")
    parser.add_argument('--config_file', type=Path, required=True,
                        help="Path to the configuration file.")
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint.pth',
                        help="Name of the checkpoint file.")
    parser.add_argument('--savedir_name', type=str, default='generated_global',
                        help="Name of the output directory wrt config_file base directory. "
                             "Pass 'None' to disable saving generated sampes altogether.")
    parser.add_argument('--num_samples', type=int, default=30,
                        help="Number of samples to generate.")
    parser.add_argument('--k_std', type=float, default=1.0,
                        help="Standard deviation multiplier for generating displacements.")
    parser.add_argument('--machine', type=str, default='local',
                        help="Machine to run genration and retrieve data from, defaults to 'local")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to use for computations ('cpu' or 'cuda:index').")
    args = parser.parse_args()
    return args


def save_mesh(x, faces, name):
    """Saves a mesh to an OBJ file."""
    x = trimesh.Trimesh(vertices=x, faces=faces)
    x.export(name, file_type='obj')


def generate_global(args):
    """
    Generates and optionally saves a specified number of global shape meshes using the LAMM neural network model.

    Parameters:
    - args (argparse.Namespace): Command-line arguments containing the following keys:
      - config_file (Path): Path to the YAML configuration file for the model and generation settings.
      - checkpoint_name (str): Name of the checkpoint file for loading the model weights. Defaults to 'checkpoint.pth'.
      - savedir_name (str): Name of the directory where generated meshes will be saved. This path is relative to the
        configuration file's base directory. If set to 'None', saving is disabled. Defaults to 'generated_global'.
      - num_samples (int): Number of mesh samples to generate. Defaults to 30.
      - k_std (float): A standard deviation multiplier for generating displacements in other contexts.
      - machine (str): Identifier for the machine where generation and data retrieval are performed. Defaults to 'local'.
      - device (str): The device on which to perform computations ('cpu' or 'cuda:index'). Defaults to 'cpu'.

    Returns:
    - meshes (list of dict): A list of dictionaries, each containing a single key 'verts' with a value of the generated
      mesh vertices as a torch.Tensor. These vertices are adjusted according to the template mean and standard deviation.

    Side Effects:
    - May create a new directory and save generated mesh files in OBJ format if saving is enabled through `savedir_name`.

    Raises:
    - Various exceptions related to file reading, network loading, and tensor operations if configurations are incorrect
      or if the computation device is not properly set or unsupported.

    """
    config_file = args.config_file
    checkpoint_name = args.checkpoint_name
    savedir_name = args.savedir_name
    k_std = args.k_std
    num_samples = args.num_samples
    device = f'cuda:{args.device}'

    if savedir_name != "None":
        savedir = f'{os.path.dirname(config_file)}/{savedir_name}'
        os.makedirs(savedir, exist_ok=True)
    else:
        savedir = None

    config = read_yaml(config_file)
    config['MODEL']['manipulation'] = False
    config['MACHINE'] = args.machine

    net = LAMM(config['MODEL']).to(device)
    load_from_checkpoint(net, f'{os.path.dirname(config_file)}/{checkpoint_name}', device=device, partial_restore=True)

    mesh, mean, std, mm_mult = get_template_mean_std(config)
    faces = mesh.faces

    net.eval()

    with open(f'{os.path.dirname(config_file)}/files/gaussian_id.pickle', 'rb') as handle:
        gaussian_id = pickle.load(handle, encoding='latin1')
        # Extract mean (mu) and standard deviation (sigma) of latent code distribution
        mu = gaussian_id['mean']
        sigma = gaussian_id['sigma']

    # Generate random latent codes for global sample generation
    z = torch.tensor(np.random.multivariate_normal(mu, k_std * sigma, num_samples)).to(device)

    out = net.decode(z.unsqueeze(1).to(torch.float32))[-1].detach().cpu()

    meshes = []
    for i, out_ in enumerate(out):
        out_ = mm_mult * (out_ * (std + 1e-7) + mean)
        meshes.append({'verts': out_.unsqueeze(0).to(torch.float32)})
        if savedir:
            save_mesh(out_, faces, f'{savedir}/{i}.obj')

    return meshes


if __name__ == "__main__":

    args = parse_args()
    meshes = generate_global(args)
