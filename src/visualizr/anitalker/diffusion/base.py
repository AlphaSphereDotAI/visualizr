import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch as th
from torch.cuda.amp import autocast
from tqdm import tqdm

from visualizr.anitalker.choices import (
    GenerativeType,
    LossType,
    ModelMeanType,
    ModelType,
    ModelVarType,
)
from visualizr.anitalker.config_base import BaseConfig
from visualizr.anitalker.model import Model
from visualizr.anitalker.model.base import BaseModule
from visualizr.anitalker.model.nn import mean_flat


@dataclass
class GaussianDiffusionBeatGansConfig(BaseConfig):
    """
    Configuration for the `GaussianDiffusionBeatGans` diffusion process.

    This class holds all hyperparameters and settings required to initialize and
    control a Gaussian diffusion process for BeatGans models.

    Args:
        betas: A 1D numpy array of betas for each diffusion timestep,
               starting at T and going to 1.
        model_mean_type: A ModelMeanType determining what the model outputs.
        model_var_type: A ModelVarType determining how variance is output.
        loss_type: A LossType determining the loss function to use.
        rescale_timesteps: If True, pass floating point timesteps into the
                           model so that they're always scaled like in the
                           original paper (0 to 1000).
    """

    gen_type: GenerativeType
    betas: tuple[float]
    model_type: ModelType
    model_mean_type: ModelMeanType
    model_var_type: ModelVarType
    loss_type: LossType
    rescale_timesteps: bool
    fp16: bool
    train_pred_xstart_detach: bool = True

    def make_sampler(self):
        """
        Create a `GaussianDiffusionBeatGans` sampler based on this configuration.

        Returns:
            GaussianDiffusionBeatGans: A diffusion sampler
                                       initialized with this config.
        """
        return GaussianDiffusionBeatGans(self)


class GaussianDiffusionBeatGans:
    """Utilities for training and sampling diffusion models."""

    def __init__(self, conf: GaussianDiffusionBeatGansConfig):
        self.conf = conf
        self.model_mean_type = conf.model_mean_type
        self.model_var_type = conf.model_var_type
        self.loss_type = conf.loss_type
        self.rescale_timesteps = conf.rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(conf.betas, dtype=np.float64)
        self.betas = betas
        if len(betas.shape) != 1:
            raise ValueError("betas must be 1D")
        if not ((betas > 0).all() and (betas <= 1).all()):
            raise ValueError("betas must be positive and less than or equal to 1")
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        if self.alphas_cumprod_prev.shape != (self.num_timesteps,):
            msg: str = "`alphas_cumprod_prev` must have the same shape as `betas`"
            raise ValueError(msg)

        # calculations for diffusion and others.
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior `q(x_{t-1} | x_t, x_0)`
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:]),
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def training_losses(
        self,
        model: BaseModule,
        motion_direction_start: th.Tensor,
        motion_target: th.Tensor,
        motion_start: th.Tensor,
        audio_feats: th.Tensor,
        face_location: th.Tensor,
        face_scale: th.Tensor,
        yaw_pitch_roll: th.Tensor,
        t: th.Tensor,
        noise: th.Tensor = None,
    ):
        """
        Compute training losses for a single timestep.

        :param model: The model to evaluate loss on.
        :param t: A batch of timestep indexes.
        :param noise: If specified, the specific Gaussian noise to try to remove.
        :return: A dict with the key “loss” containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if noise is None:
            noise = th.randn_like(motion_target)

        x_t = self.q_sample(motion_target, t, noise=noise)

        terms = {"x_t": x_t}

        if self.loss_type in [
            LossType.mse,
            LossType.l1,
        ]:
            with autocast(self.conf.fp16):
                # x_t is static wrt. To the diffusion process.
                (
                    predicted_direction,
                    predicted_location,
                    predicted_scale,
                    predicted_pose,
                ) = model.forward(
                    motion_start,
                    motion_direction_start,
                    audio_feats,
                    face_location,
                    face_scale,
                    yaw_pitch_roll,
                    x_t.detach(),
                    self._scale_timesteps(t),
                    control_flag=False,
                )

            target_types = {
                ModelMeanType.eps: noise,
            }
            target = target_types[self.model_mean_type]
            if not predicted_direction.shape == target.shape == motion_target.shape:
                msg: str = (
                    f"Shape mismatch: "
                    f"predicted_direction {predicted_direction.shape}, "
                    f"target {target.shape}, "
                    f"motion_target {motion_target.shape}"
                )
                raise ValueError(msg)

            if self.loss_type == LossType.mse:
                if self.model_mean_type == ModelMeanType.eps:
                    direction_loss = mean_flat((target - predicted_direction) ** 2)
                    location_loss = mean_flat(
                        (face_location.unsqueeze(-1) - predicted_location) ** 2,
                    )
                    scale_loss = mean_flat((face_scale - predicted_scale) ** 2)
                    pose_loss = mean_flat((yaw_pitch_roll - predicted_pose) ** 2)

                    terms["mse"] = (
                        direction_loss + location_loss + scale_loss + pose_loss
                    )
            elif self.loss_type == LossType.l1:
                # (n, c, h, w) => (n, )
                terms["mse"] = mean_flat((target - predicted_direction).abs())

            if "vb" in terms:
                # if learning the variance, also use the vlb loss
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def sample(
        self,
        model: Model,
        shape=None,
        noise=None,
        cond=None,
        x_start=None,
        clip_denoised=True,
        model_kwargs=None,
        progress=False,
    ):
        """
        Generate samples from the diffusion model using either DDPM or DDIM sampling.

        This function selects the appropriate sampling loop based on
        the generative type in the configuration.

        Args:
            model: The model to use for sampling.
            shape: The shape of the samples to generate.
            noise: Optional initial noise tensor for sampling.
            cond: Optional conditioning input for the model.
            x_start: Optional starting tensor for autoencoder models.
            clip_denoised: If True, clip denoised outputs to [-1, 1].
            model_kwargs: Optional dictionary of extra keyword arguments for the model.
            progress: If True, display a progress bar during sampling.

        Returns:
            torch.Tensor: The generated samples from the model.
        """
        if model_kwargs is None:
            model_kwargs = {}
            if self.conf.model_type.has_autoenc():
                model_kwargs["x_start"] = x_start
                model_kwargs["cond"] = cond

        match self.conf.gen_type:
            case GenerativeType.ddpm:
                return self.p_sample_loop(
                    model,
                    shape=shape,
                    noise=noise,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    progress=progress,
                )
            case GenerativeType.ddim:
                return self.ddim_sample_loop(
                    model,
                    shape=shape,
                    noise=noise,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    progress=progress,
                )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution `q(x_t | x_0)`.

        :param x_start: The `[N × C × ...]` Tensor of noiseless inputs.
        :param t: The number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, `log_variance`), all `x_start`'s shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod,
            t,
            x_start.shape,
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from `q(x_t | x_0)`.

        :param x_start: The initial data batch.
        :param t: The number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: If specified, the split-out normal noise.
        :return: A noisy version of `x_start`.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        if noise.shape != x_start.shape:
            raise ValueError(f"Shape mismatch: {noise.shape} vs {x_start.shape}")
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior.

            `q(x_{t-1} | x_t, x_0)`
        """
        if x_start.shape != x_t.shape:
            msg: str = f"Shape mismatch: {x_start.shape} vs {x_t.shape}"
            raise ValueError(msg)
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(
            self.posterior_variance,
            t,
            x_t.shape,
        )
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped,
            t,
            x_t.shape,
        )
        if not (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        ):
            msg: str = (
                "Shape mismatch: "
                f"{posterior_mean.shape} vs "
                f"{posterior_variance.shape} vs "
                f"{posterior_log_variance_clipped.shape} vs "
                f"{x_start.shape}"
            )
            raise ValueError(msg)
        return (
            posterior_mean,
            posterior_variance,
            posterior_log_variance_clipped,
        )

    def p_mean_variance(
        self,
        model,
        x,
        t: th.Tensor,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: The model, which takes a signal, and a batch of timesteps
                      as input.
        :param x: The `[N × C × ...]` Tensor at time t.
        :param t: A 1-D Tensor of timesteps.
        :param clip_denoised: If True, clip the denoised signal into [-1, 1].
        :param denoised_fn: If not None, a function, which applies to the
            `x_start` prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: If not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: A dict with the following keys:
                 * 'mean': the model means output.
                 * 'variance': the model variance output.
                 * 'log_variance': the log of 'variance'.
                 * 'pred_xstart': the prediction for `x_0`.
        """
        model_log_variance = None
        model_variance = None
        if model_kwargs is None:
            model_kwargs = {}

        motion_start = model_kwargs["start"]
        audio_feats = model_kwargs["audio_driven"]
        face_location = model_kwargs["face_location"]
        face_scale = model_kwargs["face_scale"]
        yaw_pitch_roll = model_kwargs["yaw_pitch_roll"]
        motion_direction_start = model_kwargs["motion_direction_start"]
        control_flag = model_kwargs["control_flag"]

        b, _ = x.shape[:2]
        if t.shape != (b,):
            raise ValueError(f"Shape mismatch: {t.shape} vs {(b,)}")
        with autocast(self.conf.fp16):
            model_forward, _, _, _ = model.forward(
                motion_start,
                motion_direction_start,
                audio_feats,
                face_location,
                face_scale,
                yaw_pitch_roll,
                x,
                self._scale_timesteps(t),
                control_flag,
            )
        model_output = model_forward

        if self.model_var_type in [
            ModelVarType.fixed_large,
            ModelVarType.fixed_small,
        ]:
            model_variance, model_log_variance = {
                # for `fixed_large`, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.fixed_large: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.fixed_small: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(_x):
            if denoised_fn is not None:
                _x = denoised_fn(_x)
            return _x.clamp(-1, 1) if clip_denoised else _x

        pred_xstart = None
        if self.model_mean_type in [ModelMeanType.eps]:
            if self.model_mean_type == ModelMeanType.eps:  # TODO
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output),
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart,
                x_t=x,
                t=t,
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        if not (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        ):
            msg: str = (
                "Shape mismatch: "
                f"{model_mean.shape} vs "
                f"{model_log_variance.shape} vs "
                f"{pred_xstart.shape} vs "
                f"{x.shape}"
            )
            raise ValueError(msg)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "model_forward": model_forward,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        if x_t.shape != eps.shape:
            raise ValueError(f"Shape mismatch: {x_t.shape} vs {eps.shape}")
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        if x_t.shape != xprev.shape:
            raise ValueError(f"Shape mismatch: {x_t.shape} vs {xprev.shape}")
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1,
                t,
                x_t.shape,
            )
            * x_t
        )

    def _predict_xstart_from_scaled_xstart(self, t, scaled_xstart):
        return scaled_xstart * _extract_into_tensor(
            self.sqrt_recip_alphas_cumprod,
            t,
            scaled_xstart.shape,
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _predict_eps_from_scaled_xstart(self, x_t, t, scaled_xstart):
        # scaled_xstart: is supposed to be `sqrt(alphacum) * x_0`
        # 1 / sqrt(1-alphabar) * (x_t - scaled xstart)
        return (x_t - scaled_xstart) / _extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_t.shape,
        )

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            # scale t to be maxed out at 1000 steps
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function `cond_fn` that
        computes the gradient of a conditional log probability about
        `x`. In particular, `cond_fn` computes `grad(log(p(y|x)))`, and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        return p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by `cond_fn`.

        See `condition_mean()` for details on `cond_fn`.

        Unlike `condition_mean()`, this instead uses the conditioning strategy
        from Song et al. (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x,
            self._scale_timesteps(t),
            **model_kwargs,
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"],
            x_t=x,
            t=t,
        )
        return out

    def p_sample(
        self,
        model: Model,
        x,
        t: th.Tensor,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample `x_{t-1}` from the model at the given timestep.

        :param model: The model to sample from.
        :param x: The current tensor is at `x_{t-1}`.
        :param t: The value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: If True, clip the `x_start` prediction to [-1, 1].
        :param denoised_fn: If not None, a function, which applies to the
                            `x_start` prediction before it is used to sample.
        :param cond_fn: If not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: If not None, a dict of extra keyword arguments to
                             pass to the model. This can be used for conditioning.
        :return: A dict containing the following keys:
                 * 'sample': a random sample from the model.
                 * 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn,
                out,
                x,
                t,
                model_kwargs,
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model: Model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: The model module.
        :param shape: The shape of the samples, (N, C, H, W).
        :param noise: If specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: If True, clip `x_start` predictions to [-1, 1].
        :param denoised_fn: If not None, a function, which applies to the
                            `x_start` prediction before it is used to sample.
        :param cond_fn: If not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: If not None, a dict of extra keyword arguments to
                             pass to the model. This can be used for conditioning.
        :param device: If specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: If True, show a tqdm progress bar.
        :return: A non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model: Model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as `p_sample_loop()`.
        Returns a generator over dicts, where each dict is the return value of
        `p_sample()`.
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            if not isinstance(shape, (tuple, list)):
                raise TypeError(f"Shape must be a tuple or list, not a {type(shape)}")
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * len(img), device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised,
                    denoised_fn,
                    cond_fn,
                    model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model: Model,
        x,
        t: th.Tensor,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample `x_{t-1}` from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # if we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model: Model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        NOTE: never used?
        """
        if not np.isclose(eta, 0.0, rtol=1e-09, atol=1e-09):
            msg: str = (
                f"eta must be 0, but got {eta}, Reverse ODE only for deterministic path"
            )
            raise ValueError(msg)
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised,
            denoised_fn,
            model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # if we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed  (DDIM paper)  (th.sqrt == torch.sqrt)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample_loop(
        self,
        model: Model,
        x,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        device=None,
    ):
        if device is None:
            device = next(model.parameters()).device
        sample_t = []
        xstart_t = []
        _t = []
        indices = list(range(self.num_timesteps))
        sample = x
        for i in indices:
            t = th.tensor([i] * len(sample), device=device)
            with th.no_grad():
                out = self.ddim_reverse_sample(
                    model,
                    sample,
                    t,
                    clip_denoised,
                    denoised_fn,
                    model_kwargs,
                    eta,
                )
                sample = out["sample"]
                # [1, ..., _t]
                sample_t.append(sample)
                # [0, ...., _t-1]
                xstart_t.append(out["pred_xstart"])
                # [0, ..., _t-1] ready to use
                _t.append(t)

        return {
            #  xT
            "sample": sample,
            # (1, ..., _t)
            "sample_t": sample_t,
            # xstart here is a bit different from sampling from _t = _t-1 to _t = 0
            # may not be exact.
            "xstart_t": xstart_t,
            "T": _t,
        }

    def ddim_sample_loop(
        self,
        model: Model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model: Model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as `p_sample_loop_progressive()`.
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            if not isinstance(shape, (tuple, list)):
                raise TypeError(f"Shape must be a tuple or list, not a {type(shape)}")
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            indices = tqdm(indices)

        for i in indices:
            _kwargs = (
                model_kwargs[i] if isinstance(model_kwargs, list) else model_kwargs
            )
            t = th.tensor([i] * len(img), device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised,
                    denoised_fn,
                    cond_fn,
                    _kwargs,
                    eta,
                )
                out["t"] = t
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self,
        model: Model,
        x_start,
        x_t,
        t: th.Tensor,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: A dict with the following keys:
                 * 'output': a shape [N] tensor of NLLs or KLs.
                 * 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start,
            x_t=x_t,
            t=t,
        )
        out = self.p_mean_variance(
            model,
            x_t,
            t,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )
        kl = normal_kl(
            true_mean,
            true_log_variance_clipped,
            out["mean"],
            out["log_variance"],
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start,
            means=out["mean"],
            log_scales=0.5 * out["log_variance"],
        )
        if decoder_nll.shape != x_start.shape:
            msg = f"Shape mismatch: {decoder_nll.shape} vs {x_start.shape}"
            raise ValueError(msg)
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {
            "output": output,
            "pred_xstart": out["pred_xstart"],
            "model_forward": out["model_forward"],
        }

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim. This term can't be optimized, as it only depends on the encoder.

        :param x_start: The `[N × C × ...]` Tensor of inputs.
        :return: A batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean,
            logvar1=qt_log_variance,
            mean2=0.0,
            logvar2=0.0,
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(
        self,
        model: Model,
        x_start,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Compute the entire variational lower-bound.

        Measured in bits-per-dim, as well as other related quantities.

        :param model: The model to evaluate loss on.
        :param x_start: The `[N × C × ...]` Tensor of inputs.
        :param clip_denoised: If True, clip denoised samples.
        :param model_kwargs: If not None, a dict of extra keyword arguments to
                             pass to the model. This can be used for conditioning.
        :return: A dict containing the following keys:
                 * total_bpd: the total variational lower-bound, per batch element.
                 * prior_bpd: the prior term in the lower-bound.
                 * vb: an [N × T] tensor of terms in the lower-bound.
                 * xstart_mse: an [N × T] tensor of x_0 MSEs for each timestep.
                 * mse: an [N × T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1D numpy array for a batch of indexes.

    :param arr: The 1D numpy array.
    :param timesteps: A tensor of indexes into the array to extract.
    :param broadcast_shape: A larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: A tensor of shape [batch_size, 1, ...] Where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules, which remain like
    the limit of num_diffusion_timesteps.
    Beta schedules may be added but shouldn't be removed or changed once
    they're committed to maintain backwards compatibility.
    """
    match schedule_name:
        case "linear":
            # Linear schedule from Ho et al., extended to work for any number of
            # diffusion steps.
            scale = 1000 / num_diffusion_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return np.linspace(
                beta_start,
                beta_end,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
        case "cosine":
            return betas_for_alpha_bar(
                num_diffusion_timesteps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        case "const0.01":
            scale = 1000 / num_diffusion_timesteps
            return np.array([scale * 0.01] * num_diffusion_timesteps, dtype=np.float64)
        case "const0.015":
            scale = 1000 / num_diffusion_timesteps
            return np.array([scale * 0.015] * num_diffusion_timesteps, dtype=np.float64)
        case "const0.008":
            scale = 1000 / num_diffusion_timesteps
            return np.array([scale * 0.008] * num_diffusion_timesteps, dtype=np.float64)
        case "const0.0065":
            scale = 1000 / num_diffusion_timesteps
            return np.array(
                [scale * 0.0065] * num_diffusion_timesteps,
                dtype=np.float64,
            )
        case "const0.0055":
            scale = 1000 / num_diffusion_timesteps
            return np.array(
                [scale * 0.0055] * num_diffusion_timesteps,
                dtype=np.float64,
            )
        case "const0.0045":
            scale = 1000 / num_diffusion_timesteps
            return np.array(
                [scale * 0.0045] * num_diffusion_timesteps,
                dtype=np.float64,
            )
        case "const0.0035":
            scale = 1000 / num_diffusion_timesteps
            return np.array(
                [scale * 0.0035] * num_diffusion_timesteps,
                dtype=np.float64,
            )
        case "const0.0025":
            scale = 1000 / num_diffusion_timesteps
            return np.array(
                [scale * 0.0025] * num_diffusion_timesteps,
                dtype=np.float64,
            )
        case "const0.0015":
            scale = 1000 / num_diffusion_timesteps
            return np.array(
                [scale * 0.0015] * num_diffusion_timesteps,
                dtype=np.float64,
            )
        case _:
            raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from `t = [0,1]`.

    :param num_diffusion_timesteps: The number of betas to produce.
    :param alpha_bar: A lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: The maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two Gaussian's.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = next(
        (obj for obj in (mean1, logvar1, mean2, logvar2) if isinstance(obj, th.Tensor)),
        None,
    )
    if tensor is None:
        raise ValueError("`tensor` is required, at least one argument must be a Tensor")

    # Force variances to be Tensors.
    # Broadcasting helps convert scalars to
    # Tensors, but it doesn't work for `th.exp()`.
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: The target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: The Gaussian means Tensor.
    :param log_scales: The Gaussian log stddev Tensor.
    :return: A tensor like x of log probabilities (in nats).
    """
    if not x.shape == means.shape == log_scales.shape:
        msg: str = (
            "Shape mismatch: "
            f"x {x.shape}, "
            f"means {means.shape}, "
            f"log_scales {log_scales.shape}"
        )
        raise ValueError(msg)
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(
            x > 0.999,
            log_one_minus_cdf_min,
            th.log(cdf_delta.clamp(min=1e-12)),
        ),
    )
    if log_probs.shape != x.shape:
        raise ValueError(f"Shape mismatch: {log_probs.shape} vs {x.shape}")
    return log_probs


class DummyModel(th.nn.Module):
    def __init__(self, pred):
        super().__init__()
        self.pred = pred

    def forward(self):
        return DummyReturn(pred=self.pred)


class DummyReturn(NamedTuple):
    pred: th.Tensor
