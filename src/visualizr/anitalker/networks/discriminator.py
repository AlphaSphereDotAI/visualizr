import math

from regex import T
import torch
from torch import Tensor, nn
from torch.nn.functional import conv2d, leaky_relu, pad


def fused_leaky_relu(
    _input: Tensor,
    bias: nn.Parameter,
    negative_slope: float = 0.2,
    scale: float = 2**0.5,
) -> Tensor:
    return leaky_relu(_input + bias, negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(
        self,
        channel,
        negative_slope: float = 0.2,
        scale: float = 2**0.5,
    ) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.negative_slope: float = negative_slope
        self.scale: float = scale

    def forward(self, _input: Tensor) -> Tensor:
        return fused_leaky_relu(_input, self.bias, self.negative_slope, self.scale)


def upfirdn2d_native(
    _input: Tensor,
    kernel: Tensor,
    up_x: int,
    up_y: int,
    down_x: int,
    down_y: int,
    pad_x0: int,
    pad_x1: int,
    pad_y0: int,
    pad_y1: int,
) -> Tensor:
    _, minor, in_h, in_w = _input.shape
    kernel_h, kernel_w = kernel.shape

    out: Tensor = _input.view(-1, minor, in_h, 1, in_w, 1)
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
    w: Tensor = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    return out[:, :, ::down_y, ::down_x]


def upfirdn2d(
    _input: Tensor,
    kernel: Tensor,
    up: int = 1,
    down: int = 1,
    _pad: tuple[int] = (0, 0),
) -> Tensor:
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


def make_kernel(k: Tensor) -> Tensor:
    k: Tensor = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class Blur(nn.Module):
    def __init__(
        self,
        kernel: Tensor,
        _pad: tuple[int],
        upsample_factor: int = 1,
    ) -> None:
        super().__init__()
        kernel: Tensor = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)
        self.register_buffer("kernel", kernel)
        self.pad: tuple[int] = _pad

    def forward(self, _input: Tensor) -> Tensor:
        return upfirdn2d(_input, self.kernel, _pad=self.pad)


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.negative_slope: float = negative_slope

    def forward(self, _input: Tensor) -> Tensor:
        return leaky_relu(_input, negative_slope=self.negative_slope)


class EqualConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size),
        )
        self.scale: float = 1 / math.sqrt(in_channel * kernel_size**2)
        self.stride: int = stride
        self.padding: int = padding
        self.bias: nn.Parameter | None = (
            nn.Parameter(torch.zeros(out_channel)) if bias else None
        )

    def forward(self, _input: Tensor) -> Tensor:
        return conv2d(
            _input,
            self.weight * self.scale,
            self.bias,
            self.stride,
            self.padding,
        )


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample: bool = False,
        blur_kernel: list | None = None,
        bias: bool = True,
        activate: bool = True,
    ) -> None:
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        layers: list[nn.Module] = []

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
