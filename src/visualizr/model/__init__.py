from typing import Union

from .unet import BeatGANsUNetConfig, BeatGANsUNetModel
from .unet_autoenc import BeatGANsAutoencConfig, BeatGANsAutoencModel

Model = Union[BeatGANsUNetModel, BeatGANsAutoencModel]
ModelConfig = Union[BeatGANsUNetConfig, BeatGANsAutoencConfig]
