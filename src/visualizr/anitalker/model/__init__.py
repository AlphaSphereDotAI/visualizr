from typing import Union

from visualizr.anitalker.model.unet import BeatGANsUNetConfig, BeatGANsUNetModel
from visualizr.anitalker.model.unet_autoenc import (
    BeatGANsAutoencConfig,
    BeatGANsAutoencModel,
)

Model = Union[BeatGANsUNetModel, BeatGANsAutoencModel]
ModelConfig = Union[BeatGANsUNetConfig, BeatGANsAutoencConfig]
