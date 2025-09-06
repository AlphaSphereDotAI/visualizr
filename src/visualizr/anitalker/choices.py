from enum import Enum

from torch.nn import Identity, LeakyReLU, ReLU, SiLU, Tanh


class TrainMode(Enum):
    # manipulate mode = training the classifier
    manipulate = "manipulate"
    # default training mode!
    diffusion = "diffusion"


class ModelType(Enum):
    """Kinds of the backbone models."""

    # unconditional ddpm
    ddpm = "ddpm"
    # autoencoding ddpm cannot do unconditional generation
    autoencoder = "autoencoder"

    def has_autoenc(self):
        return self in [ModelType.autoencoder]


class ModelName(Enum):
    """List of all supported model classes."""

    beatgans_ddpm = "beatgans_ddpm"
    beatgans_autoenc = "beatgans_autoenc"


class ModelMeanType(Enum):
    """Which type of output the model predicts."""

    # the model predicts epsilon
    eps = "eps"


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


class GenerativeType(Enum):
    """where how a sample is generated."""

    ddpm = "ddpm"
    ddim = "ddim"


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
