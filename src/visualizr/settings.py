from typing import Literal

from pydantic import BaseModel


class Args(BaseModel):
    infer_type: str
    test_image_path: str
    test_audio_path: str
    test_hubert_path: str
    result_path: str = "./outputs/"
    stage1_checkpoint_path: str = "ckpts/stage1.ckpt"
    stage2_checkpoint_path: str
    seed: int
    control_flag: bool = True
    pose_yaw: float
    pose_pitch: float
    pose_roll: float
    face_location: float
    pose_driven_path: str = "not_supported_in_this_mode"
    face_scale: float
    step_T: int
    image_size: int = 256
    device: Literal["cuda", "cpu"] = "cuda"
    motion_dim: int = 20
    decoder_layers: int = 2
    face_sr: bool


class DefaultValues(BaseModel):
    pose_yaw: float = 0.0
    pose_pitch: float = 0.0
    pose_roll: float = 0.0
    face_location: float = 0.5
    face_scale: float = 0.5
    step_T: int = 50
    seed: int = 0
