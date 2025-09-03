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
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import TensorDataset

from visualizr.anitalker.choices import OptimizerType, TrainMode
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

    def setup(self) -> None:
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

    @property
    def num_samples(self):
        """(global) batch size * iterations."""
        # Batch size here is global.
        # `global_step` already takes into account the accum batches.
        return self.global_step * self.conf.batch_size_effective

    def is_last_accum(self, batch_idx):
        """
        If it is the last gradient accumulation loop.

        Used with `gradient_accum > 1` and to see if the optimizer will perform “step”
        in this iteration or not.
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0

    def training_step(self, batch, batch_idx):
        """Calculate the loss function. No optimization at this stage."""
        with amp.autocast(False):
            motion_start = batch["motion_start"]  # Size [B, 512]
            motion_direction = batch["motion_direction"]  # Size [B, 125, 20]
            audio_feats = batch["audio_feats"].float()  # Size [B, 25, 250, 1024]
            face_location = batch["face_location"].float()  # Size [B, 125]
            face_scale = batch["face_scale"].float()  # Size [B, 125, 1]
            yaw_pitch_roll = batch["yaw_pitch_roll"].float()  # Size [B, 125, 3]
            motion_direction_start = batch["motion_direction_start"].float()
            if self.conf.train_mode == TrainMode.diffusion:
                # main training mode
                # with numpy seed we have the problem that the sample t's are related.
                t, _ = self.T_sampler.sample(len(motion_start), motion_start.device)
                losses = self.sampler.training_losses(
                    model=self.model,
                    motion_direction_start=motion_direction_start,
                    motion_target=motion_direction,
                    motion_start=motion_start,
                    audio_feats=audio_feats,
                    face_location=face_location,
                    face_scale=face_scale,
                    yaw_pitch_roll=yaw_pitch_roll,
                    t=t,
                )
            else:
                raise NotImplementedError

            loss = losses["loss"].mean()
            # divide by accum batches to make the accumulated gradient exact.
            for key in losses.keys():
                losses[key] = self.all_gather(losses[key]).mean()

            if self.global_rank == 0:
                self.logger.experiment.add_scalar(
                    "loss",
                    losses["loss"],
                    self.num_samples,
                )
                for key in losses:
                    self.logger.experiment.add_scalar(
                        f"loss/{key}",
                        losses[key],
                        self.num_samples,
                    )

        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """After each training step."""
        if self.is_last_accum(batch_idx):
            if self.conf.train_mode == TrainMode.latent_diffusion:
                # it trains only the latent hence change only the latent
                ema(
                    self.model.latent_net,
                    self.ema_model.latent_net,
                    self.conf.ema_decay,
                )
            else:
                ema(self.model, self.ema_model, self.conf.ema_decay)

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # fix the fp16 + clip grad norm problem with pytorch lighting
        # this is the currently correct way to do it
        if self.conf.grad_clip > 0:
            params = [p for group in optimizer.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.conf.grad_clip)

    def configure_optimizers(self):
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(
                self.model.parameters(),
                lr=self.conf.lr,
                weight_decay=self.conf.weight_decay,
            )
        elif self.conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.conf.lr,
                weight_decay=self.conf.weight_decay,
            )
        out = {"optimizer": optim}
        if self.conf.warmup > 0:
            sched = torch.optim.lr_scheduler.LambdaLR(
                optim,
                lr_lambda=WarmupLR(self.conf.warmup),
            )
            out["lr_scheduler"] = {"scheduler": sched, "interval": "step"}
        return out

    def split_tensor(self, x):
        """
        Extract the tensor for a corresponding “worker” in the batch dimension.

        Args:
            x: (n, c)

        Returns:
            x: (`n_local`, c)
        """
        n = len(x)
        rank = self.global_rank
        world_size = get_world_size()
        per_rank = n // world_size
        return x[rank * per_rank : (rank + 1) * per_rank]


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay),
        )


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup


def is_time(num_samples, every, step_size):
    closest = (num_samples // every) * every
    return num_samples - closest < step_size


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
