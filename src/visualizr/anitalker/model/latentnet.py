from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import init

from visualizr.anitalker.choices import Activation
from visualizr.anitalker.config_base import BaseConfig
from visualizr.anitalker.model.nn import timestep_embedding


class LatentNetType(Enum):
    none: str = "none"
    # injecting inputs into the hidden layers
    skip: str = "skip"


class LatentNetReturn(NamedTuple):
    pred: torch.Tensor | None = None


@dataclass
class MLPSkipNetConfig(BaseConfig):
    """Default MLP for the latent DPM in the paper."""

    num_channels: int
    skip_layers: tuple[int]
    num_hid_channels: int
    num_layers: int
    num_time_emb_channels: int = 64
    activation: Activation = Activation.silu
    use_norm: bool = True
    condition_bias: float = 1
    dropout: float = 0
    last_act: Activation = Activation.none
    num_time_layers: int = 2
    time_last_act: bool = False

    def make_model(self) -> "MLPSkipNet":
        return MLPSkipNet(self)


class MLPSkipNet(nn.Module):
    """
    concat x to hidden layers.

    Default MLP for the latent DPM in the paper.
    """

    def __init__(self, conf: MLPSkipNetConfig) -> None:
        super().__init__()
        self.conf: MLPSkipNetConfig = conf

        layers: list[nn.Module] = []
        for i in range(conf.num_time_layers):
            a: int = conf.num_time_emb_channels if i == 0 else conf.num_channels
            b: int = conf.num_channels
            layers.append(nn.Linear(a, b))
            if i < conf.num_time_layers - 1 or conf.time_last_act:
                layers.append(conf.activation.get_act())
        self.time_embed: nn.Sequential = nn.Sequential(*layers)
        self.layers: nn.ModuleList = nn.ModuleList([])

        act: Activation | None = None
        norm: bool | None = None
        cond: bool | None = None
        a: int | None = None
        b: int | None = None
        dropout: float | None = None
        for i in range(conf.num_layers):
            if i == 0:
                act = conf.activation
                norm = conf.use_norm
                cond = True
                a, b = conf.num_channels, conf.num_hid_channels
                dropout = conf.dropout
            elif i == conf.num_layers - 1:
                act = Activation.none
                norm = False
                cond = False
                a, b = conf.num_hid_channels, conf.num_channels
                dropout = 0
            else:
                act = conf.activation
                norm = conf.use_norm
                cond = True
                a, b = conf.num_hid_channels, conf.num_hid_channels
                dropout = conf.dropout

            if i in conf.skip_layers:
                a += conf.num_channels

            self.layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    cond_channels=conf.num_channels,
                    use_cond=cond,
                    condition_bias=conf.condition_bias,
                    dropout=dropout,
                ),
            )
        self.last_act: nn.Identity | nn.ReLU | nn.LeakyReLU | nn.SiLU | nn.Tanh = (
            conf.last_act.get_act()
        )

    def forward(self, x, t) -> LatentNetReturn:
        t: torch.Tensor = timestep_embedding(t, self.conf.num_time_emb_channels)
        cond = self.time_embed(t)
        h = x
        for i in range(len(self.layers)):
            if i in self.conf.skip_layers:
                # injecting input the hidden layers
                h = torch.cat([h, x], dim=1)
            h = self.layers[i].forward(x=h, cond=cond)
        h = self.last_act(h)
        return LatentNetReturn(h)


class MLPLNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        use_cond: bool,
        activation: Activation,
        cond_channels: int,
        condition_bias: float = 0,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.activation: Activation = activation
        self.condition_bias: float = condition_bias
        self.use_cond: bool = use_cond

        self.linear = nn.Linear(in_channels, out_channels)
        self.act: nn.Identity | nn.ReLU | nn.LeakyReLU | nn.SiLU | nn.Tanh = (
            activation.get_act()
        )
        if self.use_cond:
            self.linear_emb = nn.Linear(cond_channels, out_channels)
            self.cond_layers = nn.Sequential(self.act, self.linear_emb)
        self.norm: nn.LayerNorm | nn.Identity = (
            nn.LayerNorm(out_channels) if norm else nn.Identity()
        )
        self.dropout: nn.Dropout | nn.Identity = (
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        )
        self.init_weights()

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation in [Activation.relu, Activation.silu]:
                    init.kaiming_normal_(module.weight, nonlinearity="relu")
                elif self.activation == Activation.lrelu:
                    init.kaiming_normal_(module.weight, a=0.2)  # leaky_relu

    def forward(self, x, cond=None):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)
            cond = (cond, None)

            # scale shift first
            x = x * (self.condition_bias + cond[0])
            if cond[1] is not None:
                x = x + cond[1]
        # then norm
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
