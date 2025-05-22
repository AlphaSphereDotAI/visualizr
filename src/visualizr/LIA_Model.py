from torch import load, nn

from visualizr.networks.encoder import Encoder
from visualizr.networks.styledecoder import Synthesis

# This part is modified from: https://github.com/wyhsirius/LIA


class LIA_Model(nn.Module):

    def __init__(
        self,
        size: int = 256,
        style_dim: int = 512,
        motion_dim: int = 20,
        channel_multiplier: int = 1,
        blur_kernel: list[int] = [1, 3, 3, 1],
        fusion_type: str = "",
    ) -> None:
        super().__init__()
        self.enc = Encoder(size=size,
                           dim=style_dim,
                           dim_motion=motion_dim,
                           weighted_sum=fusion_type)
        self.dec = Synthesis(size=size,
                             style_dim=style_dim,
                             motion_dim=motion_dim,
                             blur_kernel=blur_kernel,
                             channel_multiplier=channel_multiplier)

    def get_start_direction_code(self, x_start, x_target, x_face, x_aug):
        enc_dic = self.enc(x_start, x_target, x_face, x_aug)
        wa, alpha, feats = enc_dic["h_source"], enc_dic["h_motion"], enc_dic[
            "feats"]

        return wa, alpha, feats

    def render(self, start, direction, feats):
        return self.dec(start, direction, feats)

    def load_lightning_model(self, lia_pretrained_model_path):
        selfState = self.state_dict()

        state = load(f=lia_pretrained_model_path, map_location="cpu")
        for name, param in state.items():
            origName = name
            if name not in selfState:
                name = name.replace("lia.", "")
                if name not in selfState:
                    print("%s is not in the model." % origName)
                    # You can ignore those errors as some parameters are only used for training
                    continue
            if selfState[name].size() != state[origName].size():
                print(
                    "Wrong parameter length: %s, model: %s, loaded: %s" %
                    (origName, selfState[name].size(), state[origName].size()))
                continue
            selfState[name].copy_(param)
