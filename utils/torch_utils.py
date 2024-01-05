import torch
import os
import glob
import sys


def load_from_checkpoint(net, checkpoint, partial_restore=False, device=None):
    
    assert checkpoint is not None, "no path provided for checkpoint, value is None"

    if os.path.isdir(checkpoint):
        latest_checkpoint = max(glob.iglob(checkpoint + '/*.pth'), key=os.path.getctime)
        print("loading model from %s" % latest_checkpoint)
        saved_net = torch.load(latest_checkpoint)
    elif os.path.isfile(checkpoint):
        print("loading model from %s" % checkpoint)
        if device is None:
            saved_net = torch.load(checkpoint)
        else:
            saved_net = torch.load(checkpoint, map_location=device)
    else:
        raise FileNotFoundError("provided checkpoint not found, does not mach any directory or file")

    if partial_restore:
        net_dict = net.state_dict()
        if 'state_dict' in saved_net.keys():
            if any(['encoder_q' in key for key in saved_net['state_dict'].keys()]):
                saved_net2 = {}
                for k, v in saved_net['state_dict'].items():
                    try:
                        if k.split(".")[:2] == ['module', 'encoder_q']:
                            k = ".".join(k.split(".")[2:])
                            if k.split(".")[0] == 'crnn':
                                k_ = k.split(".")
                                k_[0] = 'crnn_main'
                                k = ".".join(k_)
                            if k in net_dict:
                                saved_net2[k] = v
                    except:
                        continue
                saved_net = saved_net2
            else:
                saved_net = {k: v for k, v in saved_net.items() if k in net_dict}
        else:
            saved_net = {k: v for k, v in saved_net.items() if (k in net_dict)}

        print("params to keep from checkpoint:")
        print(saved_net.keys())
        extra_params = {k: v for k, v in net_dict.items() if k not in saved_net}
        print("params to randomly init:")
        print(extra_params.keys())
        for param in extra_params:
            saved_net[param] = net_dict[param]

    net.load_state_dict(saved_net, strict=True)


def get_net_trainable_params(net):
    try:
        trainable_params = net.trainable_params
    except AttributeError:
        trainable_params = list(net.parameters())
    print("Trainable params shapes are:")
    print([trp.shape for trp in trainable_params])
    return trainable_params
    
    
def get_device(device_ids, allow_cpu=False):
    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % device_ids[0])
    elif allow_cpu:
        device = torch.device("cpu")
    else:
        sys.exit("No allowed device is found")
    return device
