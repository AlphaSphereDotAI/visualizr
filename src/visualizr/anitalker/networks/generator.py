from torch import nn

from visualizr.anitalker.networks.encoder import Encoder
from visualizr.anitalker.networks.styledecoder import Synthesis


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim=512,
        motion_dim=20,
        channel_multiplier=1,
        blur_kernel: list = None,
    ):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        super(Generator, self).__init__()

        # encoder
        self.enc = Encoder(size, style_dim, motion_dim)
        self.dec = Synthesis(
            size, style_dim, motion_dim, blur_kernel, channel_multiplier
        )

    def get_direction(self):
        return self.dec.direction(None)

    def synthesis(self, wa, alpha, feat):
        return self.dec(wa, alpha, feat)

    def forward(self, img_source, img_drive, h_start=None):
        wa, alpha, feats = self.enc(img_source, img_drive, h_start)
        return self.dec(wa, alpha, feats)
