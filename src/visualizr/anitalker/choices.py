from enum import Enum

from torch.nn import Identity, LeakyReLU, ReLU, SiLU, Tanh


class TrainMode(Enum):
    # manipulate mode = training the classifier
    manipulate = "manipulate"
    # default training mode!
    diffusion = "diffusion"
    # Default latent training mode.
    # Fitting a DDPM to a given latent.
    latent_diffusion = "latentdiffusion"

    def is_manipulate(self):
        return self in [TrainMode.manipulate]

    def is_diffusion(self):
        return self in [TrainMode.diffusion, TrainMode.latent_diffusion]

    def is_autoenc(self):
        # the network possibly does autoencoding
        return self in [TrainMode.diffusion]

    def is_latent_diffusion(self):
        return self in [TrainMode.latent_diffusion]

    def use_latent_net(self):
        return self.is_latent_diffusion()

    def require_dataset_infer(self):
        """Whether training in this mode requires latent variables to be available."""
        # this will precalculate all the latents beforehand,
        # and the dataset will be all the predicted latents.
        return self in [TrainMode.latent_diffusion, TrainMode.manipulate]


class ManipulateMode(Enum):
    """how to train the classifier to manipulate."""

    # train on whole celeba attr dataset
    celebahq_all = "celebahq_all"
    # celeba with D2C's crop
    d2c_fewshot = "d2cfewshot"
    d2c_fewshot_allneg = "d2cfewshotallneg"

    def is_celeba_attr(self):
        return self in [
            ManipulateMode.d2c_fewshot,
            ManipulateMode.d2c_fewshot_allneg,
            ManipulateMode.celebahq_all,
        ]

    def is_single_class(self):
        return self in [ManipulateMode.d2c_fewshot, ManipulateMode.d2c_fewshot_allneg]

    def is_fewshot(self):
        return self in [ManipulateMode.d2c_fewshot, ManipulateMode.d2c_fewshot_allneg]

    def is_fewshot_allneg(self):
        return self in [ManipulateMode.d2c_fewshot_allneg]


class ModelType(Enum):
    """Kinds of the backbone models."""

    # unconditional ddpm
    ddpm = "ddpm"
    # autoencoding ddpm cannot do unconditional generation
    autoencoder = "autoencoder"

    def has_autoenc(self):
        return self in [ModelType.autoencoder]

    def can_sample(self):
        return self in [ModelType.ddpm]


class ModelName(Enum):
    """List of all supported model classes."""

    beatgans_ddpm = "beatgans_ddpm"
    beatgans_autoenc = "beatgans_autoenc"


class ModelMeanType(Enum):
    """Which type of output the model predicts."""

    eps = "eps"  # the model predicts epsilon


class ModelVarType(Enum):
    """
    What is used as the model's output variance.

    The `LEARNED_RANGE` option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    # posterior beta_t
    fixed_small = "fixed_small"
    # beta_t
    fixed_large = "fixed_large"


class LossType(Enum):
    # use raw MSE loss and KL when learning variances
    mse = "mse"
    l1 = "l1"


class GenerativeType(Enum):
    """where how a sample is generated."""

    ddpm = "ddpm"
    ddim = "ddim"


class OptimizerType(Enum):
    adam = "adam"
    adamw = "adamw"


class ManipulateLossType(Enum):
    bce = "bce"
    mse = "mse"


class Activation(Enum):
    none = "none"
    relu = "relu"
    lrelu = "lrelu"
    silu = "silu"
    tanh = "tanh"

    def get_act(self) -> Identity | ReLU | LeakyReLU | SiLU | Tanh:
        match self:
            case Activation.none:
                return Identity()
            case Activation.relu:
                return ReLU()
            case Activation.lrelu:
                return LeakyReLU(negative_slope=0.2)
            case Activation.silu:
                return SiLU()
            case Activation.tanh:
                return Tanh()
