from dataclasses import dataclass
from typing import Literal

from visualizr.anitalker.choices import (
    Activation,
    GenerativeType,
    LossType,
    ModelMeanType,
    ModelName,
    ModelType,
    ModelVarType,
    TrainMode,
)
from visualizr.anitalker.config_base import BaseConfig
from visualizr.anitalker.diffusion.base import get_named_beta_schedule
from visualizr.anitalker.diffusion.diffusion import (
    SpacedDiffusionBeatGansConfig,
    space_timesteps,
)
from visualizr.anitalker.diffusion.resample import UniformSampler
from visualizr.anitalker.model import (
    BeatGANsAutoencConfig,
    BeatGANsUNetConfig,
    ModelConfig,
)
from visualizr.anitalker.model.latentnet import LatentNetType, MLPSkipNetConfig


@dataclass
class TrainConfig(BaseConfig):
    infer_type: Literal[
        "mfcc_full_control",
        "mfcc_pose_only",
        "hubert_pose_only",
        "hubert_audio_only",
        "hubert_full_control",
    ] = None
    # random seed
    seed: int = 0
    train_mode: TrainMode = TrainMode.diffusion
    batch_size: int = 16
    batch_size_eval: int | None = None
    beatgans_gen_type: GenerativeType = GenerativeType.ddim
    beatgans_loss_type: LossType = LossType.mse
    beatgans_model_mean_type: ModelMeanType = ModelMeanType.eps
    beatgans_model_var_type: ModelVarType = ModelVarType.fixed_large
    beatgans_rescale_timesteps: bool = False
    beta_scheduler: str = "linear"
    data_name: str = ""
    data_val_name: str | None = None
    diffusion_type: str | None = None
    dropout: float = 0.1
    fp16: bool = False
    img_size: int = 64
    model_conf: ModelConfig = None
    model_name: ModelName | None = None
    model_type: ModelType | None = None
    net_attn: tuple[int] | None = None
    net_beatgans_attn_head: int = 1
    # not necessarily the same as the number of style channels
    net_beatgans_embed_channels: int = 512
    net_resblock_updown: bool = True
    net_enc_pool: str = "adaptivenonzero"
    net_beatgans_gradient_checkpoint: bool = False
    net_beatgans_resnet_two_cond: bool = False
    net_beatgans_resnet_use_zero_module: bool = True
    net_beatgans_resnet_cond_channels: int | None = None
    net_ch_mult: tuple[int] | None = None
    net_ch: int = 64
    net_enc_attn: tuple[int] | None = None
    # number of resblocks for the encoder (half-unet)
    net_enc_num_res_blocks: int = 2
    net_enc_channel_mult: tuple[int] | None = None
    net_enc_grad_checkpoint: bool = False
    net_latent_activation: Activation = Activation.silu
    net_latent_condition_bias: float = 0
    net_latent_dropout: float = 0
    net_latent_layers: int | None = None
    net_latent_net_last_act: Activation = Activation.none
    net_latent_net_type: LatentNetType = LatentNetType.none
    net_latent_num_hid_channels: int = 1024
    net_latent_num_time_layers: int = 2
    net_latent_skip_layers: tuple[int] | None = None
    net_latent_time_emb_channels: int = 64
    net_latent_use_norm: bool = False
    net_latent_time_last_act: bool = False
    net_num_res_blocks: int = 2
    # number of ResBlocks for the UNET
    net_num_input_res_blocks: int | None = None
    sample_size: int = 64
    style_ch: int = 512
    T_eval: int = 1_000
    T_sampler: str = "uniform"
    T: int = 1_000
    # to be overridden
    name: str = ""
    decoder_layers = None
    motion_dim = None

    def __post_init__(self):
        self.batch_size_eval = self.batch_size_eval or self.batch_size
        self.data_val_name = self.data_val_name or self.data_name

    def scale_up_gpus(self, num_gpus, num_nodes=1):
        self.batch_size *= num_gpus * num_nodes
        self.batch_size_eval *= num_gpus * num_nodes
        return self

    def _make_diffusion_conf(self, t: int):
        if self.diffusion_type != "beatgans":
            raise NotImplementedError
        # can use t < `self.t` for evaluation
        # follows the guided-diffusion repo conventions
        # `t` is evenly spaced.
        if self.beatgans_gen_type == GenerativeType.ddpm:
            section_counts = [t]
        elif self.beatgans_gen_type == GenerativeType.ddim:
            section_counts = f"ddim{t}"
        return SpacedDiffusionBeatGansConfig(
            gen_type=self.beatgans_gen_type,
            model_type=self.model_type,
            betas=get_named_beta_schedule(self.beta_scheduler, self.T),
            model_mean_type=self.beatgans_model_mean_type,
            model_var_type=self.beatgans_model_var_type,
            loss_type=self.beatgans_loss_type,
            rescale_timesteps=self.beatgans_rescale_timesteps,
            use_timesteps=space_timesteps(
                num_timesteps=self.T,
                section_counts=section_counts,
            ),
            fp16=self.fp16,
        )

    @property
    def model_out_channels(self):
        return 3

    def make_t_sampler(self) -> UniformSampler:
        if self.T_sampler != "uniform":
            raise NotImplementedError
        return UniformSampler(self.T)

    def make_diffusion_conf(self):
        return self._make_diffusion_conf(self.T)

    def make_eval_diffusion_conf(self):
        return self._make_diffusion_conf(self.T_eval)

    def make_model_conf(self):
        if self.model_name == ModelName.beatgans_ddpm:
            self.model_type = ModelType.ddpm
            self.model_conf = BeatGANsUNetConfig(
                attention_resolutions=self.net_attn,
                channel_mult=self.net_ch_mult,
                conv_resample=True,
                dims=2,
                dropout=self.dropout,
                embed_channels=self.net_beatgans_embed_channels,
                image_size=self.img_size,
                in_channels=3,
                model_channels=self.net_ch,
                num_classes=None,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.net_beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.net_beatgans_gradient_checkpoint,
                use_new_attention_order=False,
                resnet_two_cond=self.net_beatgans_resnet_two_cond,
                resnet_use_zero_module=self.net_beatgans_resnet_use_zero_module,
            )
        elif self.model_name in [
            ModelName.beatgans_autoenc,
        ]:
            cls = BeatGANsAutoencConfig
            # supports both autoenc and vaeddpm
            if self.model_name == ModelName.beatgans_autoenc:
                self.model_type = ModelType.autoencoder
            else:
                raise NotImplementedError

            if self.net_latent_net_type == LatentNetType.none:
                latent_net_conf = None
            elif self.net_latent_net_type == LatentNetType.skip:
                latent_net_conf = MLPSkipNetConfig(
                    num_channels=self.style_ch,
                    skip_layers=self.net_latent_skip_layers,
                    num_hid_channels=self.net_latent_num_hid_channels,
                    num_layers=self.net_latent_layers,
                    num_time_emb_channels=self.net_latent_time_emb_channels,
                    activation=self.net_latent_activation,
                    use_norm=self.net_latent_use_norm,
                    condition_bias=self.net_latent_condition_bias,
                    dropout=self.net_latent_dropout,
                    last_act=self.net_latent_net_last_act,
                    num_time_layers=self.net_latent_num_time_layers,
                    time_last_act=self.net_latent_time_last_act,
                )
            self.model_conf = cls(
                attention_resolutions=self.net_attn,
                channel_mult=self.net_ch_mult,
                conv_resample=True,
                dims=2,
                dropout=self.dropout,
                embed_channels=self.net_beatgans_embed_channels,
                enc_out_channels=self.style_ch,
                enc_pool=self.net_enc_pool,
                enc_num_res_block=self.net_enc_num_res_blocks,
                enc_channel_mult=self.net_enc_channel_mult,
                enc_grad_checkpoint=self.net_enc_grad_checkpoint,
                enc_attn_resolutions=self.net_enc_attn,
                image_size=self.img_size,
                in_channels=3,
                model_channels=self.net_ch,
                num_classes=None,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.net_beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.net_beatgans_gradient_checkpoint,
                use_new_attention_order=False,
                resnet_two_cond=self.net_beatgans_resnet_two_cond,
                resnet_use_zero_module=self.net_beatgans_resnet_use_zero_module,
                latent_net_conf=latent_net_conf,
                resnet_cond_channels=self.net_beatgans_resnet_cond_channels,
            )
        else:
            raise NotImplementedError(self.model_name)

        return self.model_conf
