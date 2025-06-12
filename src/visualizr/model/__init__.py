from typing import Union

from visualizr.model.unet import BeatGANsUNetConfig, BeatGANsUNetModel
from visualizr.model.unet_autoenc import BeatGANsAutoencConfig, BeatGANsAutoencModel

Model = Union[BeatGANsUNetModel, BeatGANsAutoencModel]
ModelConfig = Union[BeatGANsUNetConfig, BeatGANsAutoencConfig]
