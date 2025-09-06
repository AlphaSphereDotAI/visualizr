from dataclasses import dataclass
from typing import NamedTuple

from torch import Tensor, nn

from visualizr.anitalker.model.latentnet import MLPSkipNetConfig
from visualizr.anitalker.model.nn import timestep_embedding
from visualizr.anitalker.model.unet import (
    BeatGANsEncoderConfig,
    BeatGANsUNetConfig,
    BeatGANsUNetModel,
)


@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    # number of style channels
    enc_out_channels: int = 512
    enc_attn_resolutions: tuple[int] | None = None
    enc_pool: str = "depthconv"
    enc_num_res_block: int = 2
    enc_channel_mult: tuple[int] | None = None
    enc_grad_checkpoint: bool = False
    latent_net_conf: MLPSkipNetConfig | None = None


class BeatGANsAutoencModel(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        super().__init__(conf)
        self.conf = conf

        # having only time, cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )

        self.encoder = BeatGANsEncoderConfig(
            image_size=conf.image_size,
            in_channels=conf.in_channels,
            model_channels=conf.model_channels,
            out_channels=conf.enc_out_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(
                conf.enc_attn_resolutions or conf.attention_resolutions
            ),
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            pool=conf.enc_pool,
        ).make_model()

    def noise_to_cond(self, noise: Tensor):
        raise NotImplementedError

    def encode(self, x):
        cond = self.encoder.forward(x)
        return {"cond": cond}

    def forward(
        self,
        x,
        t,
        y=None,
        x_start=None,
        cond=None,
        style=None,
        noise=None,
        t_cond=None,
        **kwargs,
    ):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """
        if t_cond is None:
            t_cond = t
        if noise is not None:
            # if the noise is given, we predict the cond from noise.
            cond = self.noise_to_cond(noise)
        if cond is None:
            if x is not None and len(x) != len(x_start):
                raise ValueError(f"{len(x)} != {len(x_start)}")

            tmp = self.encode(x_start)
            cond = tmp["cond"]
        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels)
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        else:
            # this happens when training only autoenc
            _t_emb = None
            _t_cond_emb = None
        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(time_emb=_t_emb, cond=cond)
        else:
            raise NotImplementedError
        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None

        if (y is not None) != (self.conf.num_classes is not None):
            msg = "must specify y if and only if the model is class-conditional"
            raise ValueError(msg)

        if self.conf.num_classes is not None:
            raise NotImplementedError

        # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        enc_cond_emb = cond_emb
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb

        hs = [[] for _ in range(len(self.conf.channel_mult))]

        if x is not None:
            h = x.type(self.dtype)

            # input blocks
            k = 0
            for i in range(len(self.input_num_blocks)):
                for _ in range(self.input_num_blocks[i]):
                    h = self.input_blocks[k](h, emb=enc_time_emb, cond=enc_cond_emb)
                    hs[i].append(h)
                    k += 1
            if k != len(self.input_blocks):
                raise ValueError(f"expected {len(self.input_blocks)} blocks, got {k}")

            # middle blocks
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
        else:
            # no lateral connections
            # happen when training only the autoencoder
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]

        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for _ in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None.
                try:
                    lateral = hs[-i - 1].pop()
                except IndexError:
                    lateral = None

                h = self.output_blocks[k](
                    h,
                    emb=dec_time_emb,
                    cond=dec_cond_emb,
                    lateral=lateral,
                )
                k += 1

        pred = self.out(h)
        return AutoencReturn(pred=pred, cond=cond)


class AutoencReturn(NamedTuple):
    pred: Tensor
    cond: Tensor = None


class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor = None
    # time only
    time_emb: Tensor = None
    # style only (but could depend on time)
    style: Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_channels, time_out_channels),
            nn.SiLU(),
            nn.Linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None):
        time_emb = None if time_emb is None else self.time_embed(time_emb)
        style = self.style(cond)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)
