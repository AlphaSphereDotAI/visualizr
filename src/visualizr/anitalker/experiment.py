from copy import deepcopy
from typing import TYPE_CHECKING

from pytorch_lightning import LightningModule, seed_everything
from torch import randn
from torch.cuda import amp

from visualizr.anitalker.choices import TrainMode
from visualizr.anitalker.config import TrainConfig
from visualizr.anitalker.diffusion.diffusion import SpacedDiffusionBeatGans
from visualizr.anitalker.model.seq2seq import DiffusionPredictor
from visualizr.anitalker.renderer import render_condition

if TYPE_CHECKING:
    from visualizr.anitalker.diffusion.resample import UniformSampler


class LitModel(LightningModule):
    def __init__(self, conf: TrainConfig) -> None:
        super().__init__()
        if conf.train_mode == TrainMode.manipulate:
            msg = "`conf.train_mode` cannot be `manipulate`"
            raise ValueError(msg)
        if conf.seed is not None:
            seed_everything(conf.seed)
        self.save_hyperparameters(conf.as_dict_jsonable())
        self.conf: TrainConfig = conf
        self.model = DiffusionPredictor(conf)
        self.ema_model: DiffusionPredictor = deepcopy(self.model)
        self.ema_model.requires_grad_(requires_grad=False)
        self.ema_model.eval()
        self.sampler: SpacedDiffusionBeatGans = (
            conf.make_diffusion_conf().make_sampler()
        )
        self.eval_sampler: SpacedDiffusionBeatGans = (
            conf.make_eval_diffusion_conf().make_sampler()
        )
        # this is shared for both model and latent
        self.T_sampler: UniformSampler = conf.make_t_sampler()
        # initial variables for consistent sampling
        self.register_buffer(
            "x_T",
            randn(
                conf.sample_size,
                3,
                conf.img_size,
                conf.img_size,
            ),
        )

    def render(
        self,
        start,
        motion_direction_start,
        audio_driven,
        face_location,
        face_scale,
        ypr_info,
        noisy_t,
        step_t,
        control_flag,
    ):
        sampler = (
            self.conf._make_diffusion_conf(step_t).make_sampler()
            if step_t is not None
            else self.eval_sampler
        )

        return render_condition(
            self.conf,
            self.ema_model,
            sampler,
            start,
            motion_direction_start,
            audio_driven,
            face_location,
            face_scale,
            ypr_info,
            noisy_t,
            control_flag,
        )

    def forward(self, noise=None, x_start=None, ema_model: bool = False):
        with amp.autocast(False):
            model: DiffusionPredictor = (
                self.model if self.disable_ema else self.ema_model
            )
            return self.eval_sampler.sample(model=model, noise=noise, x_start=x_start)
