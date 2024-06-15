from yaml import dump, safe_load
import os
import pickle
import trimesh


class ConfigParams:
    """
    A class for managing configuration parameters with default values.

    Attributes:
        config (dict): A dictionary holding configuration parameters.

    Methods:
        __init__(self, config, default_values): Initializes the configuration with default values for missing parameters.
        __getitem__(self, item): Allows retrieval of configuration parameters using dictionary-like indexing.
    """
    def __init__(self, config, default_values):
        self.config = config
        for item in default_values:
            if item not in config:
                self.config[item] = default_values[item]

    def __getitem__(self, item):
        return self.config[item]


def get_params_values(args, key, default=None):
    """
    Retrieves the value of a specified key from a dictionary, returning a default value if the key is not found.

    Parameters:
        args (dict): The dictionary from which to retrieve the value.
        key (str): The key for which to retrieve the value.
        default (any, optional): The default value to return if the key is not found. Defaults to None.

    Returns:
        The value associated with the specified key in the dictionary, or the default value if the key is not found.
    """
    if (key in args) and (args[key] is not None):
            return args[key]
    return default


def read_yaml(yaml_file):
    """
    Reads a YAML file and returns its contents as a dictionary.

    Parameters:
        yaml_file (str): The path to the YAML file to be read.

    Returns:
        dict: The YAML file contents as a dictionary.
    """
    with open(yaml_file, 'r') as config_file:
        yaml_dict = safe_load(config_file)
    return yaml_dict


def copy_yaml(config_file, save_name='config_file.yaml'):
    """
    Copies a configuration file to a specified save directory, automatically renaming to avoid overwrites.

    Parameters:
        config_file (str or dict): The configuration file path or dictionary to be copied.
        save_name (str, optional): The base name for the saved configuration file. Defaults to 'config_file.yaml'.

    Notes:
        The target save directory is determined by the 'CHECKPOINT' key's 'save_dir' value within the configuration.
        If the target file name already exists, a numerical suffix is appended to create a unique file name.
    """
    if type(config_file) is str:
        yfile = read_yaml(config_file)
    elif type(config_file) is dict:
        yfile = config_file
    save_name = f"{yfile['CHECKPOINT']['save_dir']}/{save_name}"
    i = 1
    while os.path.isfile(save_name):
        save_name = f"{save_name[:-5]}_{i}.yaml"
        i += 1
    with open(save_name, 'w') as outfile:
        dump(yfile, outfile, default_flow_style=False)


def get_template_mean_std(config):
    """
    Retrieves the mean and standard deviation of a dataset, along with its template mesh, based on configuration.

    Parameters:
        config (dict): The configuration dictionary specifying the machine and dataset to be used.

    Returns:
        tuple: A tuple containing the template mesh (as a trimesh object), the mean, the standard deviation of
        the dataset and a multiplier to transform data set mesh dimensions to mm.
    """
    DATASET_INFO = read_yaml("data/implemented_datasets.yaml")[config['MACHINE']][
        config['DATASETS']['eval']['dataset']]
    mean_std_file = DATASET_INFO['mean_std_file']
    with open(mean_std_file, 'rb') as handle:
        mean_std = pickle.load(handle, encoding='latin1')
    mean = mean_std['mean']  # .numpy()
    std = mean_std['std']  # .numpy()  # + h
    mm_mult = mean_std['mm_mult']
    template_path = DATASET_INFO['template']
    mesh = trimesh.load(template_path)
    return mesh, mean, std, mm_mult
