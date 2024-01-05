import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from utils.loss_functions import get_loss
from utils.summaries import write_mean_summaries
torch.autograd.set_detect_anomaly(True)
from models import LAMM


def train_and_evaluate(net, dataloaders, config, device):
    def train_step(net, sample, loss_fn, optimizer, device):
        # zero the parameter gradients
        optimizer.zero_grad()
        # model forward pass
        x = sample['verts'].to(device)
        B = x.shape[0]
        y = torch.roll(x, 1, 0)

        b = b_min + (b_max_current - b_min) * torch.rand(B).unsqueeze(-1).to(device)

        delta = [b * (y - x)[:, torch.tensor(control_lms[idx])].reshape(B, -1).to(x.device) for idx in region_keys]
        delta = [torch.cat((torch.zeros(d.shape).to(d.device), d)) for d in delta]
        X = torch.cat((x, x), dim=0)

        outputs = net((X, delta))

        Y = torch.cat((x, y), dim=0)

        # Better sample b_ ~ U(0, b) ------------------------------------------------------------------------
        b2 = torch.cat((torch.zeros_like(b), b)).unsqueeze(-1)
        Y = b2 * Y + (1 - b2) * X

        gts_enc = (1 - aenc) * mean + aenc * X
        gts_dec = (1 - adec) * mean + adec * Y
        gts = torch.cat((gts_enc, gts_dec), dim=0)

        lms_loss = torch.tensor(0).to(device)
        loss = loss_fn(outputs, gts)
        loss_out = loss.mean(dim=(1, 2, 3)).clone()

        mask = torch.ones(loss.shape, device=device)
        mask[semantic_token_idx, B:, manip_ids, :] = 0
        loss = (mask * loss).sum() / mask.sum()

        loss.backward(loss)
        optimizer.step()
        return loss_out, lms_loss

    def evaluate(net, evalloader, loss_fn):

        losses_ae = []
        losses_bmin = []
        losses_bmax = []

        net.eval()
        with torch.no_grad():
            for step, sample in enumerate(evalloader):

                x = sample['verts'].to(device)
                B = x.shape[0]

                y = torch.roll(x, 1, 0)  # .clone()

                delta_min = [b_min * (y - x)[:, torch.tensor(control_lms[idx])].reshape(B, -1).to(x.device) for idx in region_keys]
                outputs_min = net((x, delta_min))[-1]
                losses_bmin.append(loss_fn(outputs_min, b_min * y + (1 - b_min) * x).mean().cpu().detach().numpy())

                delta_max = [b_max * (y - x)[:, torch.tensor(control_lms[idx])].reshape(B, -1).to(x.device) for idx in region_keys]
                outputs_max = net((x, delta_max))[-1]
                losses_bmax.append(loss_fn(outputs_max, b_max * y + (1 - b_max) * x).mean().cpu().detach().numpy())

                delta_ae = [torch.zeros(d.shape).to(d.device) for d in delta_min]
                outputs_ae = net((x, delta_ae))[-1]
                losses_ae.append(loss_fn(outputs_ae, x).mean().cpu().detach().numpy())

        return np.mean(losses_ae), np.mean(losses_bmin), np.mean(losses_bmax)


    # ------------------------------------------------------------------------------------------------------------------#
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    save_path = config['CHECKPOINT']["save_path"]
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    num_steps_train = len(dataloaders['train'])
    local_device_ids = config['local_device_ids']
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)
    semantic_token_idx = config['MODEL']['encoder_depth'] + 1
    aenc = torch.linspace(1, 0, config['MODEL']['encoder_depth'] + 1).view(config['MODEL']['encoder_depth'] + 1, 1, 1,
                                                                           1).to(device)
    adec = torch.linspace(0, 1, config['MODEL']['decoder_depth'] + 1).view(config['MODEL']['decoder_depth'] + 1, 1, 1,
                                                                           1).to(device)
    b_min = float(config['SOLVER']['b_min'])
    b_max = float(config['SOLVER']['b_max'])
    b_max_epoch = float(config['SOLVER']['b_max_epoch'])
    region_keys = net.region_keys

    control_lms = config['MODEL']['regions']
    control_lms = dict(sorted(control_lms.items()))

    start_global = 1
    start_epoch = 1
    if checkpoint:
        load_from_checkpoint(net, checkpoint, device=device, partial_restore=True)
    print("current learn rate: ", lr)

    manip_ids = torch.cat([net.face_part_ids[i] for i in control_lms.keys()])

    if save_path and (not os.path.exists(save_path)):
        os.makedirs(save_path)

    copy_yaml(config)

    loss_fn = get_loss(config, device, reduction="none")

    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)

    trainable_params = get_net_trainable_params(net)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad()

    scheduler = build_scheduler(config, optimizer, num_steps_train)

    writer = SummaryWriter(save_path)

    best_eval_loss_ae = 1e10
    best_eval_loss_bmin = 1e10
    best_eval_loss_bmax = 1e10

    net.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        b_max_current = b_min + (b_max - b_min) * min(1, epoch / b_max_epoch)

        for step, sample in enumerate(dataloaders['train']):

            abs_step = start_global + (epoch - start_epoch) * num_steps_train + step  #+ 120600

            loss, lms_loss = train_step(net, sample, loss_fn, optimizer, device)

            # print batch statistics ----------------------------------------------------------------------------------#
            if abs_step % train_metrics_steps == 0:

                write_mean_summaries(writer, {'train_loss_%d' % i: loss_.item() for i, loss_ in enumerate(loss)},
                                     abs_step, mode="train", optimizer=optimizer)
                print(
                    "abs_step: %d, epoch: %d, step: %5d, loss: %s, learn rate: %.8f, b_max_curr: %.3f" %
                    (abs_step, epoch, step + 1, str(loss.tolist()), optimizer.param_groups[0]["lr"], b_max_current))

            if abs_step % eval_steps == 0:  # evaluate model every eval_steps batches
                eval_loss_ae, eval_loss_bmin, eval_loss_bmax = evaluate(net, dataloaders['eval'], loss_fn)
                if eval_loss_ae < best_eval_loss_ae:
                    if len(local_device_ids) > 1:
                        torch.save(net.module.state_dict(), "%s/best_ae.pth" % save_path)
                    else:
                        torch.save(net.state_dict(), "%s/best_ae.pth" % save_path)
                    best_eval_loss_ae = eval_loss_ae

                if eval_loss_bmin < best_eval_loss_bmin:
                    if len(local_device_ids) > 1:
                        torch.save(net.module.state_dict(), "%s/best_bmin.pth" % save_path)
                    else:
                        torch.save(net.state_dict(), "%s/best_bmin.pth" % save_path)
                    best_eval_loss_bmin = eval_loss_bmin

                if eval_loss_bmax < best_eval_loss_bmax:
                    if len(local_device_ids) > 1:
                        torch.save(net.module.state_dict(), "%s/best_bmax.pth" % save_path)
                    else:
                        torch.save(net.state_dict(), "%s/best_bmax.pth" % save_path)
                    best_eval_loss_bmax = eval_loss_bmax

                write_mean_summaries(writer, {'eval_loss_ae': eval_loss_ae,
                                              'eval_loss_bmin': eval_loss_bmin,
                                              'eval_loss_bmax': eval_loss_bmax},
                                     abs_step, mode="eval", optimizer=None)
                print(
                    "abs_step: %d, epoch: %d, step: %5d, loss_ae: %.7f, loss_bmin: %.7f, loss_bmax: %.7f" %
                    (abs_step, epoch, step + 1, eval_loss_ae, eval_loss_bmin, eval_loss_bmax))
                net.train()

        scheduler.step_update(abs_step)


if __name__ == "__main__":

    config_file = "configs/manipulation.yaml"
    device_ids = [0]

    device = f'cuda:{device_ids[0]}'
    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids
    config['MODEL']['manipulation'] = True

    DATASET_INFO = read_yaml("data/currently_implemented_datasets.yaml")[config['MACHINE']][
        config['DATASETS']['train']['dataset']]
    mean = torch.zeros([DATASET_INFO['num_vertices'], 3]).to(torch.float32).to(device)

    dataloaders = get_dataloaders(config)

    net = LAMM(config['MODEL']).to(device)

    train_and_evaluate(net, dataloaders, config, device)
