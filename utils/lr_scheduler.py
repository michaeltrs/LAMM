from timm.scheduler.cosine_lr import CosineLRScheduler


def build_scheduler(config, optimizer, n_iter_per_epoch):
    """
    Constructs and returns a learning rate scheduler based on the provided configuration. Currently,
    this function supports creating a CosineLRScheduler from the 'timm' library, with parameters
    specified in the 'config' dictionary. The scheduler adjusts the learning rate for each training
    epoch, following a cosine decay schedule with warmup.

    Parameters:
    - config (dict): Configuration dictionary that includes 'SOLVER' key with scheduler settings:
                     'num_epochs', 'num_cycles', 'num_warmup_epochs', 'lr_min', 'lr_start', and
                     'lr_scheduler' type.
    - optimizer (torch.optim.Optimizer): The optimizer for which the scheduler will adjust the
                                         learning rate.
    - n_iter_per_epoch (int): Number of iterations (batches) per epoch. This is used to calculate
                              the total number of steps ('t_initial') for the scheduler.

    Returns:
    - lr_scheduler (timm.scheduler.cosine_lr.CosineLRScheduler or None): A CosineLRScheduler object
      configured as per the 'config', if 'lr_scheduler' is set to 'cosine'. Returns None for any other
      'lr_scheduler' value not implemented in this function.

    Example Usage:
    - Given a configuration dictionary 'config', an optimizer 'optimizer', and the number of iterations
      per epoch 'n_iter_per_epoch', this function can be used as follows:
        scheduler = build_scheduler(config, optimizer, n_iter_per_epoch)

    Note:
    - This function currently only supports the 'cosine' learning rate scheduler. To support additional
      schedulers, extend the conditional statements to instantiate and return other types of schedulers.
    - The 'lr_scheduler' configuration must match the expected values ('cosine' for now) to create a
      scheduler. Otherwise, the function will return None.
    """
    t_initial = int(config['SOLVER']['num_epochs'] * n_iter_per_epoch / config['SOLVER']['num_cycles'])
    warmup_steps = int(config['SOLVER']['num_warmup_epochs'] * n_iter_per_epoch)

    lr_scheduler = None
    if config['SOLVER']['lr_scheduler'] == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=float(config['SOLVER']['lr_min']),
            warmup_lr_init=float(config['SOLVER']['lr_start']),
            warmup_t=warmup_steps,
            cycle_limit=int(config['SOLVER']['num_cycles']),
            t_in_epochs=False,
        )

    return lr_scheduler
