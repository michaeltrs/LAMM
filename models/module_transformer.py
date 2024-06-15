from torch import nn, einsum
from einops import rearrange


class Residual(nn.Module):
    """
    A residual connection wrapper for a neural network function.

    Args:
        fn (callable): The neural network function to wrap with a residual connection.
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    """
    Applies Layer Normalization before passing the input through the given function.

    Args:
        dim (int): The dimension of the LayerNorm normalization.
        fn (callable): The neural network function to apply after normalization.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    A simple feed-forward block with a linear transformation, GELU activation, and optional dropout.

    Args:
        dim (int): Number of input and output features.
        hidden_dim (int): Number of hidden units.
        dropout (float, optional): Dropout rate. Default: 0.
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Implements a multi-head self-attention mechanism.

    Args:
        dim (int): Dimension of input features.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        dropout (float, optional): Dropout rate for attention weights. Default: 0.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class TransformerPerLayerOut(nn.Module):
    """
    A Transformer module that returns intermediate layer outputs. This module stacks several blocks, each consisting of
    a PreNorm-Attention and PreNorm-FeedForward layers, and applies Layer Normalization at the end. It is capable
    of returning all intermediate outputs for each block if specified.

    Args:
        dim (int): The feature dimension of the input and output.
        depth (int): The number of layers in the transformer.
        heads (int): The number of attention heads in the Attention mechanism.
        dim_head (int): The dimension of each attention head.
        mlp_dim (int): The dimension of the hidden layer in the FeedForward block.
        dropout (float, optional): Dropout rate applied in Attention and FeedForward blocks. Default: 0.
        return_input (bool, optional): If True, includes the input in the list of returned outputs. Default: True.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., return_input=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.return_input = return_input

    def forward(self, x):
        if self.return_input:
            out = [x]
        else:
            out = []
        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
            if i < (self.depth -1):
                out.append(x)
        out.append(self.norm(x))
        return out
