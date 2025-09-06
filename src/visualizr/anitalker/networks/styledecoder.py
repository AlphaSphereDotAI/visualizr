"""
This module provides functions and classes for filtering, up/down sampling, and
LeakyReLU activation operations for image processing and neural networks.
"""

import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.functional import (
    conv2d,
    conv_transpose2d,
    grid_sample,
    leaky_relu,
    linear,
    pad,
)


def fused_leaky_relu(_input, bias, negative_slope=0.2, scale=2**0.5):
    """
    Apply fused leaky ReLU activation with bias and scaling.

    Args:
        _input (Tensor): Input tensor.
        bias (Tensor): Bias tensor to add to input.
        negative_slope (float): Negative slope for leaky ReLU.
        scale (float): Scaling factor after activation.

    Returns:
        Tensor: Activated and scaled output tensor.
    """
    return leaky_relu(_input + bias, negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, _input):
        """
        Apply fused leaky ReLU activation on the input.

        Args:
            _input (Tensor): Input tensor.

        Returns:
            Tensor: Activated output tensor.
        """
        return fused_leaky_relu(_input, self.bias, self.negative_slope, self.scale)


def upfirdn2d_native(
    _input,
    kernel,
    up_x,
    up_y,
    down_x,
    down_y,
    pad_x0,
    pad_x1,
    pad_y0,
    pad_y1,
):
    """
    Perform upsample, FIR filter, and downsample on 2D input tensor.

    Args:
        _input (Tensor): Input tensor of shape (N, C, H, W).
        kernel (Tensor): FIR filter kernel.
        up_x (int): Upsampling factor in width dimension.
        up_y (int): Upsampling factor in height dimension.
        down_x (int): Downsampling factor in width dimension.
        down_y (int): Downsampling factor in height dimension.
        pad_x0 (int): Left padding in width dimension.
        pad_x1 (int): Right padding in width dimension.
        pad_y0 (int): Top padding in height dimension.
        pad_y1 (int): Bottom padding in height dimension.

    Returns:
        Tensor: A processed tensor after upfirdn operations.
    """
    _, minor, in_h, in_w = _input.shape
    kernel_h, kernel_w = kernel.shape

    out = _input.view(-1, minor, in_h, 1, in_w, 1)
    out = pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
    out = out.view(-1, minor, in_h * up_y, in_w * up_x)

    out = pad(out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[
        :,
        :,
        max(-pad_y0, 0) : out.shape[2] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[3] - max(-pad_x1, 0),
    ]

    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1],
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    return out[:, :, ::down_y, ::down_x]


def upfirdn2d(_input, kernel, up=1, down=1, _pad=(0, 0)):
    """
    Wrapper for upfirdn2d_native with same up/down and symmetric padding.

    Args:
        _input (Tensor): Input tensor.
        kernel (Tensor): FIR filter kernel.
        up (int): Upsampling factor.
        down (int): Downsampling factor.
        _pad (tuple): Padding for both dimensions (pad_x, pad_y).

    Returns:
        Tensor: Processed tensor after upfirdn operations.
    """
    return upfirdn2d_native(
        _input,
        kernel,
        up,
        up,
        down,
        down,
        _pad[0],
        _pad[1],
        _pad[0],
        _pad[1],
    )


def make_kernel(k):
    """
    Create a 2D convolution kernel tensor from 1D or 2D input.

    Args:
        k (list or Tensor): 1D or 2D kernel coefficients.

    Returns:
        Tensor: Normalized 2D kernel tensor.
    """
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor**2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, _input):
        return upfirdn2d(_input, self.kernel, up=self.factor, _pad=self.pad)


class Blur(nn.Module):
    def __init__(self, kernel, _pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer("kernel", kernel)

        self.pad = _pad

    def forward(self, _input):
        return upfirdn2d(_input, self.kernel, _pad=self.pad)


class EqualConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size),
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding

        self.bias = nn.Parameter(torch.zeros(out_channel)) if bias else None

    def forward(self, _input):
        """
        Apply equalized learning rate 2D convolution to the input tensor.

        Args:
            _input (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Convolved output tensor.
        """
        return conv2d(
            _input,
            self.weight * self.scale,
            self.bias,
            self.stride,
            self.padding,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
        bias_init=0,
        lr_mul=1,
        activation=None,
    ):
        """
        Initialize the EqualLinear layer.

        Args:
            in_dim (int): Number of input features.
            out_dim (int): Number of output features.
            bias (bool): Whether to include bias term.
            bias_init (float): Initial bias value.
            lr_mul (float): Learning rate multiplier.
            activation (callable, optional): Activation function to apply.
        """
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, _input):
        """
        Perform the forward pass of the EqualLinear layer.

        Args:
            _input (Tensor): Input tensor of shape (..., in_dim).

        Returns:
            Tensor: Output tensor of shape (..., out_dim).
        """
        if self.activation:
            out = linear(_input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = linear(_input, self.weight * self.scale, self.bias * self.lr_mul)
        return out

    def __repr__(self):
        """
        Return a string representation of the EqualLinear layer.

        Returns:
            str: Representation string showing input and output dimensions.
        """
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        """
        Initialize the ScaledLeakyReLU activation module.

        Args:
            negative_slope (float): Controls the angle of the negative
                                    slope for negative inputs.
        """
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, _input):
        """
        Apply the scaled LeakyReLU activation to the input tensor.

        Args:
            _input (Tensor): Input tensor to apply activation.

        Returns:
            Tensor: An activated tensor after applying scaled LeakyReLU.
        """
        return leaky_relu(_input, negative_slope=self.negative_slope)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel: list = None,
    ):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, _pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, _pad=(pad0, pad1))

        fan_in = in_channel * kernel_size**2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size),
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate

    def forward(self, _input, style):
        batch, in_channel, height, width = _input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight: Tensor = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel,
            in_channel,
            self.kernel_size,
            self.kernel_size,
        )

        if self.upsample:
            _input = _input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch,
                self.out_channel,
                in_channel,
                self.kernel_size,
                self.kernel_size,
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel,
                self.out_channel,
                self.kernel_size,
                self.kernel_size,
            )
            out = conv_transpose2d(_input, weight, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        elif self.downsample:
            _input = self.blur(_input)
            _, _, height, width = _input.shape
            _input = _input.view(1, batch * in_channel, height, width)
            out = conv2d(_input, weight, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        else:
            _input = _input.view(1, batch * in_channel, height, width)
            out = conv2d(_input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        return image if noise is None else image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, _input):
        batch = _input.shape[0]
        return self.input.repeat(batch, 1, 1, 1)


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel: list = None,
        demodulate=True,
    ):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, _input, style, noise=None):
        out = self.conv(_input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        return out


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel: list = None,
        bias=True,
        activate=True,
    ):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, _pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            ),
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU())

        super().__init__(*layers)


class ToRGB(nn.Module):
    def __init__(self, in_channel, upsample=True, blur_kernel: list = None):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ConvLayer(in_channel, 3, 1)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, _input, skip=None):
        out = self.conv(_input)
        out += self.bias
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


class ToFlow(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel: list = None):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.style_dim = style_dim
        self.in_channel = in_channel
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, _input, style, feat, skip=None):
        out = self.conv(_input, style)
        out += self.bias

        # warping
        xs = np.linspace(-1, 1, _input.size(2))

        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)

        xs = (
            torch.tensor(xs)
            .float()
            .unsqueeze(0)
            .repeat(_input.size(0), 1, 1, 1)
            .to(_input.device)
        )
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        sampler = torch.tanh(out[:, 0:2, :, :])
        mask = torch.sigmoid(out[:, 2:3, :, :])
        flow = sampler.permute(0, 2, 3, 1) + xs
        feat_warp = grid_sample(feat, flow) * mask
        return feat_warp, feat_warp + _input * (1.0 - mask), out


class Direction(nn.Module):
    def __init__(self, motion_dim):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(512, motion_dim))

    def forward(self, _input):
        # input: (bs*t) x 512
        weight = self.weight + 1e-8
        # get eigenvector, orthogonal [n1, n2, n3, n4]
        q, _ = torch.linalg.qr(weight)
        if _input is None:
            return q
        input_diag = torch.diag_embed(_input)  # alpha, diagonal matrix
        out = torch.matmul(input_diag, q.T)
        out = torch.sum(out, dim=1)
        return out


class Synthesis(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        motion_dim,
        blur_kernel: list = None,
        channel_multiplier=1,
    ):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        super().__init__()

        self.size = size
        self.style_dim = style_dim
        self.motion_dim = motion_dim
        # Linear Motion Decomposition (LMD) from LIA
        self.direction = Direction(motion_dim)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4],
            self.channels[4],
            3,
            style_dim,
            blur_kernel=blur_kernel,
        )

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.to_flows = nn.ModuleList()

        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2**i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                ),
            )
            self.convs.append(
                StyledConv(
                    out_channel,
                    out_channel,
                    3,
                    style_dim,
                    blur_kernel=blur_kernel,
                ),
            )
            self.to_rgbs.append(ToRGB(out_channel))

            self.to_flows.append(ToFlow(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def forward(self, source_before_decoupling, target_motion, feats):
        skip_flow = None
        skip = None
        directions = self.direction(target_motion)
        latent = source_before_decoupling + directions  # wa + directions

        inject_index = self.n_latent
        latent = latent.unsqueeze(1).repeat(1, inject_index, 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0])

        i = 1
        for conv1, conv2, to_rgb, to_flow, feat in zip(
            self.convs[::2],
            self.convs[1::2],
            self.to_rgbs,
            self.to_flows,
            feats,
            strict=False,
        ):
            out = conv1(out, latent[:, i])
            out = conv2(out, latent[:, i + 1])
            if out.size(2) == 8:
                out_warp, out, skip_flow = to_flow(out, latent[:, i + 2], feat)
                skip = to_rgb(out_warp)
            else:
                out_warp, out, skip_flow = to_flow(
                    out,
                    latent[:, i + 2],
                    feat,
                    skip_flow,
                )
                skip = to_rgb(out_warp, skip)
            i += 2

        return skip
