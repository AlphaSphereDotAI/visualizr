from torch import load, nn

from visualizr.anitalker.networks.encoder import Encoder
from visualizr.anitalker.networks.styledecoder import Synthesis
from visualizr.settings import logger


class LIA_Model(nn.Module):
    def __init__(
        self,
        size=256,
        style_dim=512,
        motion_dim=20,
        channel_multiplier=1,
        blur_kernel=[1, 3, 3, 1],
        fusion_type="",
    ):
        super().__init__()
        self.enc = Encoder(size, style_dim, motion_dim, fusion_type)
        self.dec = Synthesis(
            size, style_dim, motion_dim, blur_kernel, channel_multiplier
        )

    def get_start_direction_code(self, x_start, x_target, x_face, x_aug):
        enc_dic = self.enc(x_start, x_target, x_face, x_aug)

        wa, alpha, feats = (
            enc_dic["h_source"],
            enc_dic["h_motion"],
            enc_dic["feats"],
        )

        return wa, alpha, feats

    def render(self, start, direction, feats):
        return self.dec(start, direction, feats)

    def load_lightning_model(self, lia_pretrained_model_path):
        selfState = self.state_dict()

        state = load(lia_pretrained_model_path, map_location="cpu")
        for name, param in state.items():
            origName = name
            if name not in selfState:
                name = name.replace("lia.", "")
                if name not in selfState:
                    logger.exception("%s is not in the model." % origName)
                    # You can ignore those errors as some parameters are only used for training
                    continue
            if selfState[name].size() != state[origName].size():
                logger.exception(
                    "Wrong parameter length: %s, model: %s, loaded: %s"
                    % (
                        origName,
                        selfState[name].size(),
                        state[origName].size(),
                    )
                )
                continue
            selfState[name].copy_(param)
