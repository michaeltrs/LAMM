from data.dataloader import get_dataloader
from data.transforms import mesh3D_transform
from utils.config_utils import read_yaml


def get_dataloaders(config, return_paths=False):
    DATASET_INFO = read_yaml("data/implemented_datasets.yaml")[config['MACHINE']]

    dataloaders = {}

    # TRAIN data -------------------------------------------------------------------------------------------------------
    train_config = config['DATASETS']['train']
    TRAIN_DATA = DATASET_INFO[train_config['dataset']]
    dataloaders['training'] = get_dataloader(paths_file=TRAIN_DATA['paths_train'],
                                          root_dir=TRAIN_DATA['basedir'],
                                          transform=mesh3D_transform(mean_std_file=TRAIN_DATA['mean_std_file']),
                                          batch_size=train_config['batch_size'], num_workers=4, shuffle=True,
                                          return_paths=return_paths, my_collate=None)

    # EVAL data -------------------------------------------------------------------------------------------------------
    eval_config = config['DATASETS']['eval']
    EVAL_DATA = DATASET_INFO[eval_config['dataset']]
    dataloaders['eval'] = get_dataloader(paths_file=EVAL_DATA['paths_eval'],
                                         root_dir=EVAL_DATA['basedir'],
                                         transform=mesh3D_transform(mean_std_file=EVAL_DATA['mean_std_file']),
                                         batch_size=eval_config['batch_size'], num_workers=4, shuffle=False,
                                         return_paths=return_paths, my_collate=None)

    return dataloaders
