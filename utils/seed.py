import torch
import numpy as np
import random
import os


def set_deterministic_behavior(seed=42):
    # Python RNG
    random.seed(seed)

    # Numpy RNG
    np.random.seed(seed)

    # PyTorch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Configure PyTorch to use deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Avoid nondeterministic algorithms (newer versions of PyTorch)
    # Note: This option is available from PyTorch 1.8 and later
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if torch.__version__ >= '1.8':
        torch.use_deterministic_algorithms(True)



