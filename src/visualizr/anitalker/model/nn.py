from math import log

from torch import (
    Tensor,
    arange,
    cat,
    cos,
    exp,
    float32,
    nn,
    sin,
    zeros_like,
)
from torch.utils.checkpoint import checkpoint


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs) -> nn.Conv1d | nn.Conv2d | nn.Conv3d:
    """Create a 1D, 2D, or 3D convolution module."""
    match dims:
        case 1:
            return nn.Conv1d(*args, **kwargs)
        case 2:
            return nn.Conv2d(*args, **kwargs)
        case 3:
            return nn.Conv3d(*args, **kwargs)
    msg: str = f"unsupported dimensions: {dims}"
    raise ValueError(msg)


def avg_pool_nd(
    dims: int, *args, **kwargs
) -> nn.AvgPool1d | nn.AvgPool2d | nn.AvgPool3d:
    """Create a 1D, 2D, or 3D average pooling module."""
    match dims:
        case 1:
            return nn.AvgPool1d(*args, **kwargs)
        case 2:
            return nn.AvgPool2d(*args, **kwargs)
        case 3:
            return nn.AvgPool3d(*args, **kwargs)
    msg: str = f"unsupported dimensions: {dims}"
    raise ValueError(msg)


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels) -> GroupNorm32:
    """
    Make a standard normalization layer.

    :param channels: Number of input channels.
    :return: A `nn.Module` for normalization.
    """
    return GroupNorm32(min(32, channels), channels)


def timestep_embedding(timesteps, dim, max_period: int = 10000) -> Tensor:
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: A 1D Tensor of N indexes, one per batch element.
                      These may be fractional.
    :param dim: The dimension of the output.
    :param max_period: Controls the minimum frequency of the embeddings.
    :return: An [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs: Tensor = exp(-log(max_period) * arange(start=0, end=half, dtype=float32) / half).to(
        device=timesteps.device,
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding: Tensor = cat([cos(args), sin(args)], dim=-1)
    if dim % 2:
        embedding = cat([embedding, zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def torch_checkpoint(func, args, flag, preserve_rng_state: bool = False):
    # torch's gradient checkpoint works with automatic mixed precision, given `torch>=1.8`
    if flag:
        return checkpoint(func, *args, preserve_rng_state=preserve_rng_state)
    return func(*args)
