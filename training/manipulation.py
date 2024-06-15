"""
This script implements the 3D manipulation training and evaluation procedure described in sec.3.2 of the paper:

    "Locally Adaptive Neural 3D Morphable Models"
    by Michail Tarasiou et. al., accepted in CVPR 2024.

Link to the paper: https://arxiv.org/pdf/2401.02937.pdf

Usage:
python training/manipulation.py --config path_to_config.yaml --device device_id
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data import get_dataloaders
from models import LAMM
from utils import (build_scheduler,
                   get_loss,
                   read_yaml, copy_yaml, get_params_values,
                   write_mean_summaries,
                   get_net_trainable_params, load_from_checkpoint,
                   set_deterministic_behavior)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', type=str, default="config/manipulation.yaml",
                        help='configuration (.yaml) file to use')
    parser.add_argument('--device', type=str, default='0',
                        help='devices to use, use comma separated values for multiple gpus e.g. "0,1"')
    return parser.parse_args()


def train_step(net, sample, loss_fn, alpha_min_max, lambda_target, optimizer, device='cuda:0'):
    """
    Performs a single training step for mesh manipulation using the LAMM network.

    This function executes a forward pass, computes the loss, performs a backward pass,
    and updates the network's weights accordingly.

    Parameters:
    - net (torch.nn.Module): The LAMM network instance to be trained.
    - sample (dict): A batch of training samples.
    - loss_fn (torch.nn.modules.loss._Loss): The loss function used for training.
    - alpha_min_max (tuple of float): A tuple containing the minimum and maximum values for alpha,
      used in calculating the manipulation loss.
    - lambda_target (tuple of torch.Tensor): A tuple containing tensors representing per-layer
      loss weights for the encoder and decoder parts of the network, respectively.
    - optimizer (torch.optim.Optimizer): The optimizer used for adjusting the network's weights.
    - device (str): The device on which to perform calculations. Accepts strings like 'cpu' or 'cuda:0'.
    """
    # Get source (input) and target (at output level) sample. For manipulation target are a random permutation of source.
    # Since the dataloader selects samples randomly taking the next sample in the batch as target has an equivalent effect.
    Vs = sample['verts'].to(device)
    Vt = torch.roll(Vs, 1, 0)
    B = Vs.shape[0]

    # Model inputs and targets. As discussed in sec.3.2, half of the training batch is dedicated to continuing
    # autoencoder (dim reduction/ mesh reconstruction) training, and half for mesh source-to-target manipulation.
    X = torch.cat((Vs, Vs), dim=0)
    Y = torch.cat((Vs, Vt), dim=0)

    # Randomly sample alpha values for linearly combining source and target values to generated updated target values
    # for each training data sample. Concatenate x2 similar to how the training batch (X, Y) will be constructed below.
    alpha_min, alpha_max = alpha_min_max
    alpha = alpha_min + (alpha_max - alpha_min) * torch.rand(B).unsqueeze(-1).to(device)

    # Calculate source to target displacements at control vertices Vc
    delta_Vc = [alpha * (Vt - Vs)[:, torch.tensor(net.control_vertices[idx], device=X.device)].reshape(B, -1)
                for idx in net.control_region_keys]
    delta_Vc = [torch.cat((torch.zeros(d.shape).to(d.device), d)) for d in delta_Vc]

    # Apply linear weights to create updated target values
    alpha = torch.cat((alpha, alpha)).unsqueeze(-1)
    Y = alpha * Y + (1 - alpha) * X

    # Calculate target geometry at every layer of the network for multilayer loss (eqs.1,2). We use a zero tensor object
    # of the same dimensionality as data samples to define mean geometry assuming all data are centered by the
    # dataloader which this codebase assumes to be the case.
    lambda_target_encoder, lambda_target_decoder = lambda_target
    Y_expanded = torch.cat(((1 - lambda_target_encoder) * torch.zeros_like(X) + lambda_target_encoder * X,
                            (1 - lambda_target_decoder) * torch.zeros_like(Y) + lambda_target_decoder * Y), dim=0)

    outputs = net((X, delta_Vc))

    # Calculate loss: (Nlayers x 2B x N x 3)
    loss = loss_fn(outputs, Y_expanded)

    # Mask loss component corresponding to learned decoder tokens (at layer net.encoder_depth + 1) for the second half
    # (B:) part of the batch which is used for manipulation training
    mask = torch.ones(loss.shape, device=device)
    mask[net.encoder_depth + 1, B:] = 0

    optimizer.zero_grad()
    ((mask * loss).sum() / mask.sum()).backward()
    optimizer.step()

    # Return the mean loss per layer (averaged over num batches, num vertices and spatial dimensions).
    return loss.mean(dim=[1, 2, 3])


def evaluate(net, evalloader, loss_fn, alpha_max=1, device='cuda:0'):
    """
    Performs model evaluation for mesh manipulation using the LAMM network.

    Parameters:
    - net (torch.nn.Module): The LAMM network instance to be trained.
    - evalloader (torch.utils.data.dataloader.DataLoader): A torch DataLoader.
    - loss_fn (torch.nn.modules.loss._Loss): The loss function used for training.
    - alpha_max (tuple of float): A tuple containing the minimum and maximum values for alpha,
      used in calculating the manipulation loss.
    - device (str): The device on which to perform calculations. Accepts strings like 'cpu' or 'cuda:0'.
    """

    losses_ae = []
    losses_alpha_max = []

    net.eval()
    with torch.no_grad():
        for step, sample in enumerate(evalloader):

            # Get source (input) and target (at output level) sample. For manipulation target are a random permutation of source.
            # Since the dataloader selects samples randomly taking the next sample in the batch as target has an equivalent effect.
            Vs = sample['verts'].to(device)
            Vt = torch.roll(Vs, 1, 0)
            B = Vs.shape[0]

            # Calculate source to target displacements at control vertices Vc, get model outputs and append results
            # Source to target
            delta_Vc_max = [
                alpha_max * (Vt - Vs)[:, torch.tensor(net.control_vertices[idx], device=Vs.device)].reshape(B, -1)
                for idx in net.control_region_keys
            ]
            outputs_max = net((Vs, delta_Vc_max))[-1]
            losses_alpha_max.append(loss_fn(outputs_max, alpha_max * Vt + (1 - alpha_max) * Vs).mean().cpu().detach().numpy())
            # Autoencoding
            delta_Vc_ae = [
                torch.zeros(d.shape).to(d.device) for d in delta_Vc_max
            ]
            outputs_ae = net((Vs, delta_Vc_ae))[-1]
            losses_ae.append(loss_fn(outputs_ae, Vs).mean().cpu().detach().numpy())

    return np.mean(losses_ae), np.mean(losses_alpha_max)


def main():
    """
    Parses command-line arguments to configure the execution environment, initializes the model and data loaders,
    and starts the training and evaluation process based on the provided configuration. The script's behavior,
    including input parameters and training settings, can be adjusted via command-line arguments or configuration files.
    """

    args = parse_args()

    # This script should work fine in most cases using only single GPU training since LAMM is very lightweight in terms
    # of memory. Training on 12k vertex meshes with batch size 32 requires ~2.5GB of GPU RAM.
    device_ids = [int(d) for d in args.device.split(',')]
    device = f'cuda:{device_ids[0]}'

    config = read_yaml(args.config)
    config['MODEL']['manipulation'] = True

    seed = config['SOLVER']['seed']
    if seed is not None:
        print(f'setting seed={seed}')
        set_deterministic_behavior(seed=seed)

    # dataloaders is a dict containing two DataLoader objects {'training': ..., 'eval': ...}
    dataloaders = get_dataloaders(config)

    net = LAMM(config['MODEL']).to(device)

    # Training hyperparameters
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    save_dir = config['CHECKPOINT']["save_dir"]
    num_steps_train = len(dataloaders['training'])
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)

    # lambda_target is used to weight the contribution of the ground truths in the loss at each layer (l) in the
    # multilayer loss formulation presented in eqs.(1,2) (equivalent to the term l/L). Compared to dimensionality
    # reduction training, here, we separate between encoder and decoder weights since source (Vs) and target meshes (Vt)
    # are different.
    lambda_target_encoder = torch.linspace(1, 0, config['MODEL']['encoder_depth'] + 1).view(
        config['MODEL']['encoder_depth'] + 1, 1, 1, 1).to(device)
    lambda_target_decoder = torch.linspace(0, 1, config['MODEL']['decoder_depth'] + 1).view(
        config['MODEL']['decoder_depth'] + 1, 1, 1, 1).to(device)
    lambda_target = (lambda_target_encoder, lambda_target_decoder)

    # Linear mixing ratio between source and target samples to create updated target values
    alpha_min = float(config['SOLVER']['alpha_min'])
    alpha_max = float(config['SOLVER']['alpha_max'])
    alpha_max_epoch = float(config['SOLVER']['alpha_max_epoch'])  # epoch when alpha_max_ takes maximum defined value

    checkpoint_file = config['CHECKPOINT']["load_from_checkpoint"]
    if checkpoint_file:
        load_from_checkpoint(net, checkpoint_file, device=device, partial_restore=True)
    print("current learn rate: ", lr)

    if save_dir and (not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    copy_yaml(config)

    if len(device_ids) > 1:
        net = nn.DataParallel(net, device_ids=device_ids)
    net.to(device)

    loss_fn = get_loss(config, reduction="none")
    trainable_params = get_net_trainable_params(net)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad()
    scheduler = build_scheduler(config, optimizer, num_steps_train)
    writer = SummaryWriter(save_dir)

    best_eval_loss_ae = 1e10
    best_eval_loss_alpha_max = 1e10
    net.train()
    for epoch in range(1, num_epochs + 1):
        alpha_max_ = alpha_min + (alpha_max - alpha_min) * min(1, epoch / alpha_max_epoch)
        alpha_min_max = (alpha_min, alpha_max_)

        for step, sample in enumerate(dataloaders['training']):

            abs_step = (epoch - 1) * num_steps_train + step + 1

            loss = train_step(net, sample, loss_fn, alpha_min_max, lambda_target, optimizer, device)

            if abs_step % train_metrics_steps == 0:

                write_mean_summaries(writer, {f'train_loss_{i}': loss_.item() for i, loss_ in enumerate(loss)},
                                     abs_step, mode="training", optimizer=optimizer)
                print(
                    f"abs_step: {abs_step}, epoch: {epoch}, step: {step + 1}, loss: {str(loss.tolist())}, "
                    f"learn rate: {optimizer.param_groups[0]['lr']}, alpha_max_curr: {alpha_max_}"
                )

            if abs_step % eval_steps == 0:

                print('EVAL ------------------------------------------------------------------------------------------')
                eval_loss_ae, eval_loss_alpha_max = evaluate(net, dataloaders['eval'], loss_fn, alpha_max, device)

                if eval_loss_ae < best_eval_loss_ae:
                    if len(device_ids) > 1:
                        torch.save(net.module.state_dict(), f"{save_dir}/best_ae.pth")
                    else:
                        torch.save(net.state_dict(), f"{save_dir}/best_ae.pth")
                    best_eval_loss_ae = eval_loss_ae

                if eval_loss_alpha_max < best_eval_loss_alpha_max:
                    if len(device_ids) > 1:
                        torch.save(net.module.state_dict(), f"{save_dir}/best_alpha_max.pth")
                    else:
                        torch.save(net.state_dict(), f"{save_dir}/best_alpha_max.pth")
                    best_eval_loss_alpha_max = eval_loss_alpha_max

                write_mean_summaries(writer, {'eval_loss_ae': eval_loss_ae,
                                              'eval_loss_alpha_max': eval_loss_alpha_max},
                                     abs_step, mode="eval", optimizer=None)
                print(
                    f"abs_step: {abs_step}, epoch: {epoch}, step: {step + 1}, "
                    f"loss_ae: {eval_loss_ae}, loss_alpha_max: {eval_loss_alpha_max}"
                )
                net.train()

        scheduler.step_update(abs_step)


if __name__ == "__main__":
    main()
