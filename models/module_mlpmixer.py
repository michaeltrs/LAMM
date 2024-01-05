from torch import nn
from functools import partial


pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


class MLPMixerPerLayerOut(nn.Module):
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
