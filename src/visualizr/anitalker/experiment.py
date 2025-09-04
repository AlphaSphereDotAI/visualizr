import copy
import os

import numpy as np
import torch
from gradio import Info
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.cuda import amp
from torch.utils.data.dataset import TensorDataset

from visualizr.anitalker.choices import TrainMode
from visualizr.anitalker.config import TrainConfig
from visualizr.anitalker.dist_utils import get_world_size
from visualizr.anitalker.model.seq2seq import DiffusionPredictor
from visualizr.anitalker.renderer import render_condition


class LitModel(LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        if conf.train_mode == TrainMode.manipulate:
            raise ValueError("`conf.train_mode` cannot be `manipulate`")
        if conf.seed is not None:
            seed_everything(conf.seed)
        self.save_hyperparameters(conf.as_dict_jsonable())
        self.conf = conf
        self.model = DiffusionPredictor(conf)
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()
        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()
        # this is shared for both model and latent
        self.T_sampler = conf.make_t_sampler()
        if conf.train_mode.use_latent_net():
            self.latent_sampler = conf.make_latent_diffusion_conf().make_sampler()
            self.eval_latent_sampler = (
                conf.make_latent_eval_diffusion_conf().make_sampler()
            )
        else:
            self.latent_sampler = None
            self.eval_latent_sampler = None
        # initial variables for consistent sampling
        self.register_buffer(
            "x_T",
            torch.randn(conf.sample_size, 3, conf.img_size, conf.img_size),
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
            model = self.model if self.disable_ema else self.ema_model
            return self.eval_sampler.sample(model=model, noise=noise, x_start=x_start)

    def setup(self) -> None:  # TODO
        """Make datasets & seeding each worker."""
        ##############################################
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            Info(f"local seed: {seed}")
        ##############################################

        self.train_data = self.conf.make_dataset()
        Info(f"train data: {len(self.train_data)}")
        self.val_data = self.train_data
        Info(f"val data: {len(self.val_data)}")

    def _train_dataloader(self, drop_last=True):
        """Make the dataloader."""
        # make sure to use the fraction of batch size
        # the batch size is global.
        conf = self.conf.clone()
        conf.batch_size = self.batch_size
        return conf.make_loader(self.train_data, shuffle=True, drop_last=drop_last)

    def train_dataloader(self):
        """
        Return the dataloader.

        If diffusion mode → return image dataset
        if latent mode → return the inferred latent dataset.
        """
        Info("on train dataloader start ...")
        if not self.conf.train_mode.require_dataset_infer():
            return self._train_dataloader()
        if self.conds is None:
            # usually we load self.conds from a file,
            # so we don't need to do this again.
            self.conds = self.infer_whole_dataset()
            # Need to use float32. Unless the mean & std will be off.
            # (1, c)
            self.conds_mean.data = self.conds.float().mean(dim=0, keepdim=True)
            self.conds_std.data = self.conds.float().std(dim=0, keepdim=True)
        Info(f"mean: {self.conds_mean.mean()}, std: {self.conds_std.mean()}")

        # return the dataset with pre-calculated conds
        conf = self.conf.clone()
        conf.batch_size = self.batch_size
        data = TensorDataset(self.conds)
        return conf.make_loader(data, shuffle=True)

    @property
    def batch_size(self):
        """Local batch size for each worker."""
        ws = get_world_size()
        if self.conf.batch_size % ws != 0:
            raise ValueError("batch size must be divisible by world size")
        return self.conf.batch_size // ws


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup


def train(conf: TrainConfig, gpus, nodes=1):
    Info(f"conf: {conf.name}")
    model = LitModel(conf)

    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)
    checkpoint = ModelCheckpoint(
        dirpath=f"{conf.logdir}",
        save_last=True,
        save_top_k=-1,
        every_n_epochs=10,
    )
    checkpoint_path = f"{conf.logdir}/last.ckpt"
    Info(f"ckpt path: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        resume = checkpoint_path
        Info("resume!")
    else:
        resume = conf.continue_from.pathcd if conf.continue_from is not None else None
    tb_logger = TensorBoardLogger(save_dir=conf.logdir, name=None, version="")

    # from pytorch_lightning.

    plugins = []
    if len(gpus) == 1 and nodes == 1:
        accelerator = None
    else:
        accelerator = "ddp"
        # important for working with gradient checkpoint
        plugins.append(DDPStrategy(find_unused_parameters=True))

    trainer = Trainer(
        max_steps=conf.total_samples // conf.batch_size_effective,
        resume_from_checkpoint=resume,
        gpus=gpus,
        num_nodes=nodes,
        accelerator=accelerator,
        precision=16 if conf.fp16 else 32,
        callbacks=[
            checkpoint,
            LearningRateMonitor(),
        ],
        # clip in the model instead
        # gradient_clip_val=conf.grad_clip,
        replace_sampler_ddp=True,
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        plugins=plugins,
    )

    trainer.fit(model)
