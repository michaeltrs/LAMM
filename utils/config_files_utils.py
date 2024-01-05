from yaml import dump, safe_load
import os


class ConfigParams:
    def __init__(self, config, default_values):
        self.config = config
        for item in default_values:
            if item not in config:
                self.config[item] = default_values[item]

    def __getitem__(self, item):
        return self.config[item]


def get_params_values(args, key, default=None):
    """
    set default to None if a value is required in the config file
    """
    if (key in args) and (args[key] is not None):
            return args[key]
    return default


def read_yaml(yaml_file):
    with open(yaml_file, 'r') as config_file:
        yaml_dict = safe_load(config_file)
    return yaml_dict


def copy_yaml(config_file, save_name='config_file.yaml'):
    """
    copies config file to training savedir
    """
    if type(config_file) is str:
        yfile = read_yaml(config_file)
    elif type(config_file) is dict:
        yfile = config_file
    save_name = yfile['CHECKPOINT']['save_path'] + "/%s" % save_name
    i = 1
    while os.path.isfile(save_name):
        save_name = "%s_%d.yaml" % (save_name[:-5], i)
        i += 1
    with open(save_name, 'w') as outfile:
        dump(yfile, outfile, default_flow_style=False)
