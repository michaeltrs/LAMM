from torch import nn
from functools import partial

# Converts input into a tuple if not already one.
pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    """
    Implements a Pre-Normalization Residual block. This module applies layer normalization before passing the input
    through a function (fn) and adding the result to the original input (residual connection).

    Parameters:
    - dim (int): Dimensionality of the input features.
    - fn (callable): The function to apply to the input after normalization. This is typically
      a neural network layer or module.

    Attributes:
    - fn (callable): The function provided as a parameter.
    - norm (nn.LayerNorm): The layer normalization module.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    """
    Creates a feedforward neural network module. This function returns a sequential container of a dense (linear) layer,
    a GELU activation function, a dropout layer, another dense layer, and another dropout layer. The first dense layer
    expands the dimensionality of the input by the expansion factor, and the second dense layer projects it back.

    Parameters:
    - dim (int): The input and output dimensionality of the feedforward network.
    - expansion_factor (float, optional): Factor by which to expand the dimensionality in the hidden layer. Defaults to 4.
    - dropout (float, optional): Dropout rate. Defaults to 0.
    - dense (callable, optional): The dense layer constructor. Defaults to nn.Linear.

    Returns:
    - nn.Sequential: The feedforward network module.
    """
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


class MLPMixerPerLayerOut(nn.Module):
    """
    Implements an MLPMixer architecture with per-layer outputs. This class constructs an MLPMixer model with a specified
    number of layers, where each layer consists of two types of operations: mixing the features across patches and
    mixing across channels. It outputs the intermediate representations from all layers.

    Parameters:
    - dim (int): Dimensionality of the input features.
    - depth (int): Number of layers in the mixer model.
    - num_patches (int): Number of patches into which the input is divided.
    - expansion_factor (float, optional): Expansion factor for patch mixing feedforward networks. Defaults to 4.
    - expansion_factor_token (float, optional): Expansion factor for channel mixing feedforward networks. Defaults to 0.5.
    - dropout (float, optional): Dropout rate. Defaults to 0.

    Attributes:
    - layers (nn.ModuleList): List of mixer layers, each consisting of patch and channel mixing operations.
    - norm (nn.LayerNorm): Final layer normalization.
    - depth (int): Depth of the model, for internal tracking.
    """
    def __init__(self, dim, depth, num_patches, expansion_factor=4, expansion_factor_token=0.5, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.depth = depth
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
            ]))

    def forward(self, x):
        out = [x]
        for i, (fN, fD) in enumerate(self.layers):
            x = fN(x)
            x = fD(x)
            if i < (self.depth -1):
                out.append(x)
        out.append(self.norm(x))
        return out
