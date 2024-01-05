import torch
from copy import deepcopy


def get_loss(config, device, reduction='mean'):
    loss_config = config['SOLVER']

    print(loss_config['loss_function'])

    if type(loss_config['loss_function']) in [list, tuple]:
        loss_fun = []
        loss_types = deepcopy(loss_config['loss_function'])
        config_ = deepcopy(config)
        for loss_fun_type in loss_types:
            config_['SOLVER']['loss_function'] = loss_fun_type
            loss_fun.append(get_loss(config_, device, reduction=reduction))
        return loss_fun

    # MSE Loss -------------------------------------------------------------------------------
    if loss_config['loss_function'] == 'mse':
        return torch.nn.MSELoss(size_average=None, reduce=None, reduction=reduction)

    if loss_config['loss_function'] == 'l1':
        return torch.nn.L1Loss(size_average=None, reduce=None, reduction=reduction)
