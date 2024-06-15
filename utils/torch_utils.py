import os
import glob
import sys
import torch
import torch.nn.functional as F


def resize_match2d(target_size, source, dim=[2, 3], mode='bilinear'):
    """
    source must have shape [..., H, W]
    :param mode: 'nearest'
    """
    target_h, target_w = target_size
    source_h, source_w = source.shape[dim[0]], source.shape[dim[1]]
    if (source_h != target_h) or (source_w != target_w):
        source_type = source.dtype
        if source_type != torch.float32:
            source = source.to(torch.float32)
            return F.interpolate(source, size=(target_h, target_w), mode=mode).to(source_type)
        return F.interpolate(source, size=(target_h, target_w), mode=mode)
    return source


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
        raise FileNotFoundError(f"provided checkpoint {checkpoint} not found, does not mach any directory or file.")

    # For partially restoring a model from checkpoint restore only the common parameters and randomly initialize the
    # remaining ones
    if partial_restore:
        net_dict = net.state_dict()
        saved_net = {k: v for k, v in saved_net.items() if (k in net_dict)}
        print("parameters to keep from checkpoint:")
        print(saved_net.keys())
        extra_params = {k: v for k, v in net_dict.items() if k not in saved_net}
        print("parameters to randomly init:")
        print(extra_params.keys())
        for param in extra_params:
            saved_net[param] = net_dict[param]

    net.load_state_dict(saved_net, strict=True)


def get_net_trainable_params(net):
    try:
        trainable_params = net.trainable_params
    except AttributeError:
        trainable_params = list(net.parameters())
    print("Trainable parameters shapes are:")
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
