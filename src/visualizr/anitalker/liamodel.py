from gradio import Error
from torch import load, nn

from visualizr.anitalker.networks.encoder import Encoder
from visualizr.anitalker.networks.styledecoder import Synthesis


class LiaModel(nn.Module):
    def __init__(
        self,
        size=256,
        style_dim=512,
        motion_dim=20,
        channel_multiplier=1,
        blur_kernel: list = None,
        fusion_type="",
    ):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        super().__init__()
        self.enc = Encoder(size, style_dim, motion_dim, fusion_type)
        self.dec = Synthesis(
            size, style_dim, motion_dim, blur_kernel, channel_multiplier
        )

    def get_start_direction_code(self, x_start, x_target, x_face, x_aug):
        enc_dic = self.enc(x_start, x_target, x_face, x_aug)
        wa, alpha, feats = enc_dic["h_source"], enc_dic["h_motion"], enc_dic["feats"]
        return wa, alpha, feats

    def render(self, start, direction, feats):
        return self.dec(start, direction, feats)

    def load_lightning_model(self, lia_pretrained_model_path):
        self_state = self.state_dict()

        state = load(lia_pretrained_model_path, map_location="cpu")
        for name, param in state.items():
            orig_name = name
            if name not in self_state:
                name = name.replace("lia.", "")
            if name not in self_state:
                Error(f"{orig_name} is not in the model.")
                # You can ignore those errors as some parameters are only used for training.
                continue
            if self_state[name].size() != state[orig_name].size():
                Error(
                    f"Wrong parameter length: {orig_name}, "
                    + f"model: {self_state[name].size()}, "
                    + f"loaded: {state[orig_name].size()}"
                )
                continue
            self_state[name].copy_(param)
