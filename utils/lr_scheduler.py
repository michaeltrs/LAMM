from timm.scheduler.cosine_lr import CosineLRScheduler


def build_scheduler(config, optimizer, n_iter_per_epoch):
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
