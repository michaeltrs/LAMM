import argparse
import csv
import os
import sys

import numpy as np
import torch
import trimesh

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data import get_dataloaders
from models import LAMM
from utils.config_utils import read_yaml, get_template_mean_std
from utils.torch_utils import load_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', type=str, default='config/manipulation.yaml',
                        help='configuration (.yaml) file to use')
    parser.add_argument('--checkpoint', type=str, default='best_global.pth', help='filepath for pretrained checkpoint')
    parser.add_argument('--machine', type=str, default='local', help='machine name to run experiment')
    parser.add_argument('--device', type=str, default='0',
                        help='devices to use, use comma separated values for multiple gpus e.g. "0,1"')
    parser.add_argument('--save_output', action='store_true', help="Save reconstruction examples")
    return parser.parse_args()


def save_mesh(x, faces, name):
    """Saves a mesh to an OBJ file."""
    x = trimesh.Trimesh(vertices=x, faces=faces)
    x.export(name, file_type='obj')


def evaluate_autoencoder(args):
    """
    Evaluates a trained autoencoder model on a specified dataset. Also, optionally, saves source and reconstructed
    meshes for the first evaluation batch in args.save_out directory if one is provided.

    Args:
        args (argparse.Namespace): Command line arguments.
    """

    config_file = args.config
    checkpoint = args.checkpoint
    machine = args.machine
    device = f'cuda:{args.device}'
    save_output = args.save_output

    config = read_yaml(config_file)
    config['MACHINE'] = machine
    config['MODEL']['manipulation'] = False

    dname = config['DATASETS']['eval']['dataset']

    eval_loader = get_dataloaders(config)['eval']

    # Intialize model and load from checkpoint
    net = LAMM(config['MODEL']).to(device)
    net = net.to(device)
    load_from_checkpoint(net, f'{os.path.dirname(config_file)}/{checkpoint}', partial_restore=True)

    mesh, mean, std, mm_mult = get_template_mean_std(config)
    faces = mesh.faces

    savedir = f'{os.path.dirname(config_file)}/AE'
    os.makedirs(savedir, exist_ok=True)

    losses_l2 = []
    losses_l1 = []
    distances = []

    net.eval()
    with torch.no_grad():
        for step, sample in enumerate(eval_loader):

            if step % 10 == 0:
                print(f'step {step + 1} of {len(eval_loader)}')

            x = sample['verts'].to(device)
            B = x.shape[0]

            out = net(x)[-1]

            # Transform normalized outputs to unormalized ones in (m) and then to mm (x 1000).
            x = mm_mult * (sample['verts'] * (std + 1e-7) + mean)
            out = mm_mult * (out.detach().cpu() * (std + 1e-7) + mean)

            # Save reconstructed outputs (optional) only for the first step
            if save_output and step == 0:
                for i in range(B):
                    save_mesh(x[i].cpu().numpy(), faces, f'{savedir}/{i}_source.obj')
                    save_mesh(out[i].cpu().numpy(), faces, f'{savedir}/{i}_reconstructed.obj')

            loss_L1 = torch.nn.functional.l1_loss(x, out)
            loss_L2 = torch.nn.functional.mse_loss(x, out)
            distance = torch.sqrt(torch.sum((x - out) ** 2, dim=2)).mean()

            losses_l2.append(loss_L2.cpu().detach().numpy())
            losses_l1.append(loss_L1.cpu().detach().numpy())
            distances.append(distance.cpu().detach().numpy())

    print(f'L2: {np.mean(losses_l2)} mm2, L1: {np.mean(losses_l1)} mm, Euclidean Distance: {np.mean(distances)} mm')

    with open(f'{savedir}/eval_loss.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        data = [['Autoencoding Evaluation Losses in mm.'],
                [config_file],
                [checkpoint],
                [dname, 'eval'],
                ['L1', np.mean(losses_l1)],
                ['L2', np.mean(losses_l2)],
                ['Euclidean Distance', np.mean(distances)]]
        for row in data:
            writer.writerow(row)


if __name__ == "__main__":

    args = parse_args()
    evaluate_autoencoder(args)
