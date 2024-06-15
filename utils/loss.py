import torch
from copy import deepcopy


def get_loss(config, reduction='mean'):
    """
    Dynamically creates and returns a loss function object based on the configuration provided.
    This function supports returning a single loss function or a list of loss functions if the
    configuration specifies multiple loss types. The loss function(s) returned are based on PyTorch's
    built-in loss functions, with the 'reduction' parameter applied as specified.

    Parameters:
    - config (dict): Configuration dictionary with keys that include 'SOLVER', which itself is a
                     dictionary containing at least a 'loss_function' key. The 'loss_function' key's
                     value can be a string specifying a single loss function or a list/tuple of strings
                     for multiple loss functions.
    - reduction (str, optional): Specifies the reduction to apply to the output of the loss function.
                                 Can be 'none', 'mean', or 'sum'. Default is 'mean'.

    Returns:
    - torch.nn.modules.loss._Loss or list of torch.nn.modules.loss._Loss: The loss function object
      corresponding to the specified 'loss_function' in the config. If multiple loss functions are specified,
      a list of loss function objects is returned. Each object is an instance of a PyTorch loss class,
      configured with the specified 'reduction' parameter.

    Raises:
    - ValueError: If the 'loss_function' specified in the config is not recognized (not implemented in this
                  function).

    Note:
    - This implementation prints the loss function configuration for debugging purposes.
    - If adding support for more loss functions, ensure to handle them in the conditional statements.
    """
    loss_config = config['SOLVER']

    print(loss_config['loss_function'])

    if type(loss_config['loss_function']) in [list, tuple]:
        loss_fun = []
        loss_types = deepcopy(loss_config['loss_function'])
        config_ = deepcopy(config)
        for loss_fun_type in loss_types:
            config_['SOLVER']['loss_function'] = loss_fun_type
            loss_fun.append(get_loss(config_, reduction=reduction))
        return loss_fun

    if loss_config['loss_function'] == 'mse':
        return torch.nn.MSELoss(size_average=None, reduce=None, reduction=reduction)

    if loss_config['loss_function'] == 'l1':
        return torch.nn.L1Loss(size_average=None, reduce=None, reduction=reduction)
