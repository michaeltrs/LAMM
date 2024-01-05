import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_net_trainable_params, load_from_checkpoint
from models import LAMM
from data import get_dataloaders
from utils.loss_functions import get_loss
from utils.summaries import write_mean_summaries


def train_and_evaluate(net, dataloaders, config, device):
    def train_step(net, sample, loss_fn, optimizer, device):
        # zero the parameter gradients
        optimizer.zero_grad()
        # model forward pass
        x = sample['verts'].to(device)
        outputs = net(x)
        if dinput == 3:
            gts = sample['verts'].to(device)
        else:
            gts = sample['verts'][:, :, :3].to(device)
        gts_new = (1 - a) * mean + a * gts

        loss = loss_fn(outputs, gts_new)

        (w * loss).sum().backward()
        # run optimizer
        optimizer.step()
        return loss.mean(dim=[1,2,3])

    def evaluate(net, evalloader, loss_fn):
        losses_all = []
        net.eval()
        with torch.no_grad():
            for step, sample in enumerate(evalloader):
                x = sample['verts'].to(device)
                outputs = net(x)[-1]
                loss = loss_fn(outputs, x).mean()
                losses_all.append(loss.cpu().detach().numpy())
        mean_loss = np.mean(losses_all)
        return mean_loss

    # ------------------------------------------------------------------------------------------------------------------#
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    save_path = config['CHECKPOINT']["save_path"]
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]  #f"{save_path}/best.pth"  #
    num_steps_train = len(dataloaders['train'])
    local_device_ids = config['local_device_ids']
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)
    a = torch.cat((torch.linspace(1, 0, config['MODEL']['encoder_depth'] + 1),
                   torch.linspace(0, 1, config['MODEL']['decoder_depth'] + 1))).view(
        config['MODEL']['encoder_depth'] + config['MODEL']['decoder_depth'] + 2, 1, 1, 1).to(device)

    w = torch.tensor(config['SOLVER']['weights']).to(torch.float32).to(device).view(a.shape)
    dinput = get_params_values(config['MODEL'], "Dinput", 3)

    start_global = 1
    start_epoch = 1
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False)
    print("current learn rate: ", lr)

    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)

    if save_path and (not os.path.exists(save_path)):
        os.makedirs(save_path)

    copy_yaml(config)

    loss_fn = get_loss(config, device, reduction='none')

    trainable_params = get_net_trainable_params(net)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad()

    scheduler = build_scheduler(config, optimizer, num_steps_train)

    writer = SummaryWriter(save_path)

    best_eval_loss = 1e10
    net.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):  # loop over the dataset multiple times
        for step, sample in enumerate(dataloaders['train']):
            abs_step = start_global + (epoch - start_epoch) * num_steps_train + step

            loss = train_step(net, sample, loss_fn, optimizer, device)


            # print batch statistics ----------------------------------------------------------------------------------#
            if abs_step % train_metrics_steps == 0:

                write_mean_summaries(writer, {f'train_loss_{i}': loss[i].item() for i in range(loss.shape[0])}, abs_step, mode="train", optimizer=optimizer)
                print(
                    f"abs_step: {abs_step}, epoch: {epoch}, step: {step+1}, loss: {loss.data.tolist()}, learn rate: %.8f" % optimizer.param_groups[0]["lr"]
                )

            # evaluate model ------------------------------------------------------------------------------------------#
            if abs_step % eval_steps == 0:  # evaluate model every eval_steps batches
                print('EVAL -----------------------------------------------------')

                eval_loss = evaluate(net, dataloaders['eval'], loss_fn)

                if eval_loss < best_eval_loss:
                    if len(local_device_ids) > 1:
                        torch.save(net.module.state_dict(), "%s/best.pth" % (save_path))
                    else:
                        torch.save(net.state_dict(), "%s/best.pth" % (save_path))
                    best_eval_loss = eval_loss

                write_mean_summaries(writer, {'eval_loss': eval_loss}, abs_step, mode="eval_micro", optimizer=None)
                print(
                    "abs_step: %d, epoch: %d, step: %5d, loss: %.7f" %
                    (abs_step, epoch, step + 1, eval_loss))
                net.train()

        scheduler.step_update(abs_step)


if __name__ == "__main__":

    config_file = "configs/dim_reduction.yaml"
    device_ids = [0]

    device = f'cuda:{device_ids[0]}'
    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids
    config['MODEL']['manipulation'] = False

    DATASET_INFO = read_yaml("data/currently_implemented_datasets.yaml")[config['MACHINE']][config['DATASETS']['train']['dataset']]
    mean = torch.zeros([DATASET_INFO['num_vertices'], 3]).to(torch.float32).to(device)

    dataloaders = get_dataloaders(config)

    net = LAMM(config['MODEL']).to(device)

    train_and_evaluate(net, dataloaders, config, device)
