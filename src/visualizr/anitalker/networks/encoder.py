from math import log, sqrt

from gradio import Info
from torch import Tensor, flip, float32, nn, randn, softmax, tensor, zeros, zeros_like
from torch.nn.functional import conv2d, leaky_relu, linear, pad

from visualizr.app.logger import logger


def fused_leaky_relu(_input, bias, negative_slope=0.2, scale=2**0.5):
    return leaky_relu(_input + bias, negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        super().__init__()
        self.bias = nn.Parameter(zeros(1, channel, 1, 1))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, _input):
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
    w = flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )

    return out[:, :, ::down_y, ::down_x]


def upfirdn2d(_input, kernel, up=1, down=1, _pad=(0, 0)):
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
    k = tensor(k, dtype=float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


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


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, _input):
        return leaky_relu(_input, negative_slope=self.negative_slope)


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
            randn(out_channel, in_channel, kernel_size, kernel_size),
        )
        self.scale = 1 / sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding

        self.bias = nn.Parameter(zeros(out_channel)) if bias else None

    def forward(self, _input):
        return conv2d(
            _input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

    def __repr__(self):
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
        super().__init__()

        self.weight = nn.Parameter(randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(zeros(out_dim).fill_(bias_init)) if bias else None
        self.activation = activation

        self.scale = (1 / sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, _input):
        if self.activation:
            out = linear(_input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = linear(_input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


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


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel,
            out_channel,
            1,
            True,
            activate=False,
            bias=False,
        )

    def forward(self, _input):
        out = self.conv1(_input)
        out = self.conv2(out)
        skip = self.skip(_input)
        out = (out + skip) / sqrt(2)
        return out


class WeightedSumLayer(nn.Module):
    def __init__(self, num_tensors=8):
        super().__init__()
        self.weights = nn.Parameter(randn(num_tensors))

    def forward(self, tensor_list):
        weights = softmax(self.weights, dim=0)
        weighted_sum: Tensor = zeros_like(tensor_list[0])
        for _tensor, weight in zip(tensor_list, weights, strict=False):
            weighted_sum += _tensor * weight
        return weighted_sum


class EncoderApp(nn.Module):
    def __init__(self, size, w_dim=512, fusion_type=""):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16,
        }

        self.w_dim = w_dim
        log_size = int(log(size, 2))

        self.convs = nn.ModuleList()
        self.convs.append(ConvLayer(3, channels[size], 1))

        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel

        self.convs.append(EqualConv2d(in_channel, self.w_dim, 4, bias=False))

        self.fusion_type = fusion_type

        if self.fusion_type != "weighted_sum":
            raise ValueError(
                f"Unsupported `fusion_type`: {self.fusion_type}. "
                "Expected 'weighted_sum'.",
            )
        _msg = "HAL layer is enabled!"
        logger.info(_msg)
        Info(_msg)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = EqualLinear(64, 512)
        self.fc2 = EqualLinear(128, 512)
        self.fc3 = EqualLinear(256, 512)
        self.ws = WeightedSumLayer()

    def forward(self, x):
        res = []
        h = x
        pooled_h_lists = []
        for i, conv in enumerate(self.convs):
            h = conv(h)
            if self.fusion_type == "weighted_sum":
                pooled_h = self.adaptive_pool(h).view(x.size(0), -1)
                if i == 0:
                    pooled_h_lists.append(self.fc1(pooled_h))
                elif i == 1:
                    pooled_h_lists.append(self.fc2(pooled_h))
                elif i == 2:
                    pooled_h_lists.append(self.fc3(pooled_h))
                else:
                    pooled_h_lists.append(pooled_h)
            res.append(h)

        if self.fusion_type == "weighted_sum":
            last_layer = self.ws(pooled_h_lists)
        else:
            last_layer = res[-1].squeeze(-1).squeeze(-1)
        layer_features = res[::-1][2:]

        return last_layer, layer_features


class DecouplingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # identity_excluded_net is called identity encoder in the paper
        self.identity_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.identity_net_density = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # identity_excluded_net is called motion encoder in the paper
        self.identity_excluded_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        id_, id_rm = self.identity_net(x), self.identity_excluded_net(x)
        id_density = self.identity_net_density(id_)
        return id_, id_rm, id_density


class Encoder(nn.Module):
    def __init__(self, size, dim=512, dim_motion=20, weighted_sum=False):
        super().__init__()

        # image encoder
        self.net_app = EncoderApp(size, dim, weighted_sum)

        # decouping network
        self.net_decouping = DecouplingModel(dim, dim, dim)

        # part of the motion encoder
        fc = [EqualLinear(dim, dim)]
        fc.extend(EqualLinear(dim, dim) for _ in range(3))
        fc.append(EqualLinear(dim, dim_motion))
        self.fc = nn.Sequential(*fc)

    def forward(self, input_source, input_target, input_face, input_aug):
        h_target_motion_target = None
        h_another_face_target = None
        h_face = None
        h_source_id_emb = None
        h_source_idrm_emb = None
        h_source_id_density_emb = None
        h_target_id_emb = None
        h_target_idrm_emb = None
        h_target_id_density_emb = None
        h_face_id_emb = None
        h_face_idrm_emb = None
        h_face_id_density_emb = None
        h_aug_id_emb = None
        h_aug_idrm_emb = None
        h_aug_id_density_emb = None
        if input_target is not None:
            h_source, feats = self.net_app(input_source)
            h_target, _ = self.net_app(input_target)
            h_face, _ = self.net_app(input_face)
            h_aug, _ = self.net_app(input_aug)

            h_source_id_emb, h_source_idrm_emb, h_source_id_density_emb = (
                self.net_decouping(h_source)
            )
            h_target_id_emb, h_target_idrm_emb, h_target_id_density_emb = (
                self.net_decouping(h_target)
            )
            h_face_id_emb, h_face_idrm_emb, h_face_id_density_emb = self.net_decouping(
                h_face,
            )
            h_aug_id_emb, h_aug_idrm_emb, h_aug_id_density_emb = self.net_decouping(
                h_aug,
            )

            h_target_motion_target = self.fc(h_target_idrm_emb)
            h_another_face_target = self.fc(h_face_idrm_emb)

        else:
            h_source, feats = self.net_app(input_source)

        return {
            "h_source": h_source,
            "h_motion": h_target_motion_target,
            "feats": feats,
            "h_another_face_target": h_another_face_target,
            "h_face": h_face,
            "h_source_id_emb": h_source_id_emb,
            "h_source_idrm_emb": h_source_idrm_emb,
            "h_source_id_density_emb": h_source_id_density_emb,
            "h_target_id_emb": h_target_id_emb,
            "h_target_idrm_emb": h_target_idrm_emb,
            "h_target_id_density_emb": h_target_id_density_emb,
            "h_face_id_emb": h_face_id_emb,
            "h_face_idrm_emb": h_face_idrm_emb,
            "h_face_id_density_emb": h_face_id_density_emb,
            "h_aug_id_emb": h_aug_id_emb,
            "h_aug_idrm_emb": h_aug_idrm_emb,
            "h_aug_id_density_emb": h_aug_id_density_emb,
        }
