import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Number

import torch as th
from torch import nn
from torch.nn.functional import interpolate

from visualizr.anitalker.config_base import BaseConfig
from visualizr.anitalker.model.nn import (
    avg_pool_nd,
    conv_nd,
    normalization,
    torch_checkpoint,
    zero_module,
)


class TimestepBlock(nn.Module, ABC):
    """Any module where forward() takes timestep embeddings as a second argument."""

    @abstractmethod
    def forward(self, x, emb=None, cond=None, lateral=None):
        """Apply the module to `x` given `emb` timestep embeddings."""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb=None, cond=None, lateral=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb=emb, cond=cond, lateral=lateral)
            else:
                x = layer(x)
        return x


@dataclass
class ResBlockConfig(BaseConfig):
    channels: int
    emb_channels: int
    dropout: float
    out_channels: int | None = None
    # condition the resblock with time and encoder's output
    use_condition: bool = True
    # whether to use 3×3 conv for a skip path when the channels aren't matched.
    use_conv: bool = False
    # dimension of conv (always 2 = 2D)
    dims: int = 2
    # gradient checkpoint
    use_checkpoint: bool = False
    up: bool = False
    down: bool = False
    # whether to condition with both time and encoder's output
    two_cond: bool = False
    # number of encoders' output channels
    cond_emb_channels: int | None = None
    has_lateral: bool = False
    # if to init the convolution with zero weights,
    # this is defaulted from BeatGANs and seems to help learning.
    use_zero_module: bool = True

    def __post_init__(self):
        self.out_channels = self.out_channels or self.channels
        self.cond_emb_channels = self.cond_emb_channels or self.emb_channels

    def make_model(self):
        return ResBlock(self)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    Total layers:
        in_layers
            norm
            act
            conv
        out_layers
            norm
            (modulation)
            act
            conv

    """

    def __init__(self, conf: ResBlockConfig):
        super().__init__()
        self.conf = conf

        #############################
        # IN LAYERS
        #############################
        layers = [
            normalization(conf.channels),
            nn.SiLU(),
            conv_nd(conf.dims, conf.channels, conf.out_channels, 3, padding=1),
        ]
        self.in_layers = nn.Sequential(*layers)

        self.updown = conf.up or conf.down

        if conf.up:
            self.h_upd = Upsample(conf.channels, False, conf.dims)
            self.x_upd = Upsample(conf.channels, False, conf.dims)
        elif conf.down:
            self.h_upd = Downsample(conf.channels, False, conf.dims)
            self.x_upd = Downsample(conf.channels, False, conf.dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        #############################
        # OUT LAYERS CONDITIONS
        #############################
        if conf.use_condition:
            # condition layers for the `out_layers`
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(conf.emb_channels, 2 * conf.out_channels),
            )

            if conf.two_cond:
                self.cond_emb_layers = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(conf.cond_emb_channels, conf.out_channels),
                )
            #############################
            # OUT LAYERS (ignored when there is no condition)
            #############################
            # original version
            conv = conv_nd(
                conf.dims,
                conf.out_channels,
                conf.out_channels,
                3,
                padding=1,
            )
            if conf.use_zero_module:
                # zero out the weights, it seems to help training
                conv = zero_module(conv)

            # construct the layers
            # - norm
            # - modulation
            # - act
            # - dropout
            # - conv
            layers = []
            layers += [
                normalization(conf.out_channels),
                nn.SiLU(),
                nn.Dropout(p=conf.dropout),
                conv,
            ]
            self.out_layers = nn.Sequential(*layers)

        #############################
        # SKIP LAYERS
        #############################
        if conf.out_channels == conf.channels:
            # can't be used with `gatedconv`,
            # also `gatedconv` is always used as the first block.
            self.skip_connection = nn.Identity()
        else:
            if conf.use_conv:
                kernel_size = 3
                padding = 1
            else:
                kernel_size = 1
                padding = 0

            self.skip_connection = conv_nd(
                conf.dims,
                conf.channels,
                conf.out_channels,
                kernel_size,
                padding=padding,
            )

    def forward(self, x, emb=None, cond=None, lateral=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: input
            lateral: lateral connection from the encoder
        """
        return torch_checkpoint(
            self._forward,
            (x, emb, cond, lateral),
            self.conf.use_checkpoint,
        )

    def _forward(
        self,
        x,
        emb=None,
        cond=None,
        lateral=None,
    ):
        # lateral: required if `has_lateral` and non-gated,
        # with gated, it can be supplied optionally.
        if self.conf.has_lateral:
            # lateral may be supplied even if it doesn't require
            # the model will take the lateral only if `has_lateral`
            if lateral is None:
                raise ValueError("`lateral` is required")
            x = th.cat([x, lateral], dim=1)

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.conf.use_condition:
            # it's possible that the network may not receive the time emb
            # this happens with autoenc and setting the time_at
            emb_out = self.emb_layers(emb).type(h.dtype) if emb is not None else None
            if self.conf.two_cond:
                # it's possible that the network is two_cond
                # but it doesn't get the second condition,
                # in which case, we ignore the second condition
                # and treat as if the network has one condition.
                cond_out = (
                    None if cond is None else self.cond_emb_layers(cond).type(h.dtype)
                )
                if cond_out is not None:
                    while len(cond_out.shape) < len(h.shape):
                        cond_out = cond_out[..., None]
            else:
                cond_out = None

            # this is the new refactored code
            h = apply_conditions(
                h=h,
                emb=emb_out,
                cond=cond_out,
                layers=self.out_layers,
                in_channels=self.conf.out_channels,
            )

        return self.skip_connection(x) + h


def apply_conditions(
    h,
    emb=None,
    cond=None,
    layers: nn.Sequential = None,
    scale_bias: float = 1,
    in_channels: int = 512,
    up_down_layer: nn.Module = None,
):
    """
    Apply conditions on the feature maps.

    Args:
        emb: time conditional (ready to scale + shift)
        cond: encoder's conditional (read to scale + shift)
    """
    two_cond = emb is not None and cond is not None

    if emb is not None:
        # adjusting shapes
        while len(emb.shape) < len(h.shape):
            emb = emb[..., None]

    if two_cond:
        # adjusting shapes
        while len(cond.shape) < len(h.shape):
            cond = cond[..., None]
        # time first
        scale_shifts = [emb, cond]
    else:
        # `cond` is not used with a single cond mode.
        scale_shifts = [emb]

    # support scale, shift or shift only
    for i, each in enumerate(scale_shifts):
        if each is None:
            # special case: the condition is not provided
            a = None
            b = None
        elif each.shape[1] == in_channels * 2:
            a, b = th.chunk(each, 2, dim=1)
        else:
            a = each
            b = None
        scale_shifts[i] = (a, b)

    # condition scale bias could be a list
    if isinstance(scale_bias, Number):
        biases = [scale_bias] * len(scale_shifts)

    # by default, the scale and shift are applied after the group norm but BEFORE `SiLU`
    pre_layers, post_layers = layers[0], layers[1:]

    # spilt the post-layer to be able to scale up or down before conv
    # post-layers will contain only the conv
    mid_layers, post_layers = post_layers[:-2], post_layers[-2:]

    h = pre_layers(h)
    # `scale` and `shift` for each condition
    for i, (scale, shift) in enumerate(scale_shifts):
        # if `scale` is None, it indicates that the condition is not provided.
        if scale is not None:
            h = h * (biases[i] + scale)
            if shift is not None:
                h = h + shift
    h = mid_layers(h)

    # upscale or downscale if any just before the last conv
    if up_down_layer is not None:
        h = up_down_layer(h)
    h = post_layers(h)
    return h


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: Channels in the inputs and outputs.
    :param use_conv: A bool determining if a convolution is applied.
    :param dims: Determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        if x.shape[1] != self.channels:
            msg = f"Input has {x.shape[1]} channels but layer has {self.channels}"
            raise ValueError(msg)
        if self.dims == 3:
            x = interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2))
        else:
            x = interpolate(x, scale_factor=2)
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: Channels in the inputs and outputs.
    :param use_conv: A bool determining if a convolution is applied.
    :param dims: Determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=1,
            )
        elif self.channels == self.out_channels:
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)
        else:
            msg: str = (
                "Downsampling with no convolution requires channel reduction. "
                f"Layer has {self.channels} but `out_channels` is {self.out_channels}"
            )
            raise ValueError(msg)

    def forward(self, x):
        if x.shape[1] != self.channels:
            msg = f"Input has {x.shape[1]} channels but layer has {self.channels}"
            raise ValueError(msg)
        return self.op(x)


class AttentionBlock(nn.Module):
    """An attention block that allows spatial positions to attend to each other."""

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        elif channels % num_head_channels == 0:
            self.num_heads = channels // num_head_channels
        else:
            msg: str = (
                f"q,k,v channels {channels} is not "
                f"divisible by `num_head_channels` {num_head_channels}"
            )
            raise ValueError(msg)
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return torch_checkpoint(self._forward, (x,), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """
    A module, which performs QKV attention.

    Matches legacy QKVAttention + input/output heads shaping.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: An `[N x (H × 3 × C) x T]` tensor of Qs, Ks, and Vs.
        :return: an `[N x (H × C) x T]` tensor after attention.
        """
        bs, width, length = qkv.shape
        if width % (3 * self.n_heads) != 0:
            raise ValueError(f"Invalid qkv shape {qkv.shape}")
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        # More stable with f16 than dividing afterward
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    """A module, which performs QKV attention and splits in a different order."""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: An `[N x (3 × H × C) x T]` tensor of Qs, Ks, and Vs.
        :return: An `[N x (H × C) x T]` tensor after attention.
        """
        bs, width, length = qkv.shape
        if width % (3 * self.n_heads) != 0:
            raise ValueError(f"Invalid qkv shape {qkv.shape}")
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterward
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)
