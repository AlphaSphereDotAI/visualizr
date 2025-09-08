from visualizr.anitalker.model.unet import BeatGANsUNetConfig, BeatGANsUNetModel
from visualizr.anitalker.model.unet_autoenc import (
    BeatGANsAutoencConfig,
    BeatGANsAutoencModel,
)

Model = BeatGANsUNetModel | BeatGANsAutoencModel
ModelConfig = BeatGANsUNetConfig | BeatGANsAutoencConfig
