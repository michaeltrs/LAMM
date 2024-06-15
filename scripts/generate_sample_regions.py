import os
import sys
import argparse
from copy import copy

import torch
import numpy as np
import pickle
import trimesh

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data import get_dataloaders
from models import LAMM
from scripts import prepare_inference_files, generate_global
from utils.config_utils import read_yaml, get_template_mean_std
from utils.torch_utils import load_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Process input arguments for the script")
    parser.add_argument('--config_file', type=str, default='assets/checkpoints/UHM17k_v1/config_file.yaml',
                        help='Path to the configuration file.')
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint.pth',
                        help='Name of the checkpoint file.')
    parser.add_argument('--savedir_name', type=str, default='generated_local',
                        help='Name of the directory to save generated meshes. Pass "None" for not saving outputs.')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to generate.')
    parser.add_argument('--num_random_generations', type=int, default=5,
                        help='Number of random generations.')
    parser.add_argument('--k_std', type=float, default=1,
                        help='Standard deviation multiplier (k_std)')
    parser.add_argument('--machine', type=str, default='local',
                        help='Machine where the script is run.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run the model on.')
    parser.add_argument('--prepare_files', action='store_true',
                        help='Whether to prepare inference files before running if these are not found in "files"'
                             'subdir of config_file base dir. You will need to have access to some mesh data for'
                             'choosing this option.')
    # parser.set_defaults(prepare_files=True)
    parser.add_argument('--generate_random_source', action='store_true',
                        help='The script will first generate random mesh instances and generate random regions for these.'
                             'Otherwise use a set of provided data paths in evaluation section of config file.')
    args = parser.parse_args()
    return args


def save_mesh(x, faces, name):
    """Saves a mesh to an OBJ file."""
    x = trimesh.Trimesh(vertices=x, faces=faces)
    x.export(name, file_type='obj')


def get_random_displacements(delta_stats, key, control_lms, k_std=1, device='cpu'):
    """
    Generates random displacements for the control landmarks of a single region (key) from a Gaussian distribution
    fitted on training set displacement data.

    Parameters:
    - delta_stats (dict): A dictionary containing mean and standard deviation for the control vertices of each mesh region.
    - key (int): The specific key defining the region for which to generate random displacements.
    - control_lms (dict): A dictionary of control vertices for each mesh region.
    - k_std (float): A multiplier for the standard deviation to control displacement variance, defaults to 1.
    - device (str): The PyTorch device (e.g., 'cpu' or 'cuda:0') to use for generated tensors.

    Returns:
    - list: A list of tensor displacements for the control vertices of each mesh region.
    """
    delta = []
    for idx in control_lms.keys():
        if idx != key:
            delta.append(torch.zeros(3 * len(control_lms[idx]), device=device))
        else:
            delta.append(torch.tensor(np.random.multivariate_normal(delta_stats[key]['mean'], k_std * delta_stats[key]['std'], 1),
                                      dtype=torch.float32, device=device))
    return delta


def generate_local(args):
    """
    Generates local manipulations of 3D meshes based on the configurations provided through command-line arguments.
    The function generates random displacements to control landmarks and leverages a pre-trained LAMM model to apply
    these displacements and generate random mesh regions, producing variations of the input mesh.
    The function supports processing multiple samples and saving the manipulated meshes to disk.

    Parameters:
    - args (argparse.Namespace): Command-line arguments specifying configuration settings. Relevant arguments include:
        * config_file (str): Path to the YAML configuration file specifying model and data parameters.
        * checkpoint_name (str): Filename of the model checkpoint to load for mesh manipulation.
        * savedir_name (str): Directory name where generated meshes should be saved. If "None", outputs are not saved.
        * num_samples (int): Number of samples to generate random regions for from the provided data or randomly
            generated ones.
        * num_random_generations (int): Number of random generations to produce for each sample.
        * k_std (float): Standard deviation multiplier for controlling the variance of displacements.
        * machine (str): Identifier for the machine where the script is run (for configuration purposes).
        * device (str): PyTorch device identifier (e.g., 'cuda:0', 'cpu') for computation.
        * prepare_files (bool): Flag indicating whether to prepare inference files before generation.
        * generate_random_source (bool): Flag indicating whether to generate source meshes randomly before apllying
            region manipulations or use specific data paths from the config file for evaluation.

    Returns:
    None. Generated meshes are saved to disk if `savedir_name` is not "None".

    Note:
    The function assumes the presence of several globally available resources and utilities, including data loaders,
    model definitions, and utility functions for configuration and Torch manipulation. It is part of a larger framework
    for 3D mesh manipulation and assumes that necessary data preprocessing and model preparation steps have been completed.
    """
    config_file = args.config_file
    checkpoint_name = args.checkpoint_name
    savedir_name = args.savedir_name
    num_samples = args.num_samples
    num_random_generations = args.num_random_generations
    k_std = args.k_std
    machine = args.machine
    device = f'cuda:{args.device}'
    prepare_files = args.prepare_files
    generate_random_source = args.generate_random_source

    config = read_yaml(config_file)
    config['MACHINE'] = machine
    control_lms = config['MODEL']['control_vertices']

    if savedir_name != "None":
        savedir = f'{os.path.dirname(config_file)}/{savedir_name}'
        os.makedirs(savedir, exist_ok=True)
    else:
        savedir = None

    mesh, mean, std, mm_mult = get_template_mean_std(config)
    faces = mesh.faces

    if prepare_files:
        args_ = copy(args)
        args_.num_epochs = 5
        _ = prepare_inference_files(args_)
    delta_stats_file = f'{os.path.dirname(config_file)}/displacement_stats.pickle'
    with open(delta_stats_file, 'rb') as file:
        delta_stats = pickle.load(file)

    if not generate_random_source:
        data_loader = get_dataloaders(config)['eval']
    else:
        args_ = copy(args)
        args_.savedir_name = "None"  # Do not save now, will be saved later next to samples with random regions
        data_loader = generate_global(args_)

    # Set manipulation flag to True, initialize model and load from provided checkpoint
    config['MODEL']['manipulation'] = True
    net = LAMM(config['MODEL']).to(device)
    load_from_checkpoint(net, f'{os.path.dirname(config_file)}/{checkpoint_name}', device=device,
                         partial_restore=False)
    net.eval()

    num_saved_samples = 0
    source_meshes = []
    manipulated_meshes = []
    for step, sample in enumerate(data_loader):

        remaining_samples = num_samples - num_saved_samples
        if remaining_samples <= 0:
            break

        source = sample['verts'].to(device)

        B = source.shape[0]

        if B > remaining_samples:
            source = source[:num_samples]
            B = remaining_samples

        for i in range(B):

            num_saved_samples += 1

            print(f'generating identity {num_saved_samples} of {num_samples}')

            x = source[i].unsqueeze(0)

            source_ = mm_mult * (x[0].detach().cpu() * (std + 1e-7) + mean).cpu()
            source_meshes.append(source_)
            if savedir:
                save_mesh(source_, faces, f'{savedir}/id_{i}_source.obj')

            manipulated_meshes.append([])
            for k, key in enumerate(control_lms.keys()):

                for m in range(num_random_generations):

                    delta = get_random_displacements(delta_stats, key, control_lms, k_std, device)

                    out = net((x, delta))[-1]

                    out_ = mm_mult * (out[0].detach().cpu() * (std + 1e-7) + mean)

                    manipulated_meshes[-1].append({'verts': out_.unsqueeze(0).to(torch.float32)})
                    if savedir:
                        save_mesh(out_, faces, f'{savedir}/id_{i}_region_{key}_sample_{m}.obj')

    return source_meshes, manipulated_meshes


if __name__ == "__main__":

    args = parse_args()
    source_meshes, manipulated_meshes = generate_local(args)
