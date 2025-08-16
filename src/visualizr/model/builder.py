from pathlib import Path
from typing import Literal

from huggingface_hub import snapshot_download

from visualizr.anitalker.utils import generate_video
from visualizr.settings import Settings


class Model:
    def __init__(self, settings: Settings):
        self.settings: Settings = settings
        snapshot_download(
            repo_id=settings.model.repo_id,
            local_dir=settings.directory.checkpoint,
            repo_type="model",
        )

    def __call__(
        self,
        image_path: Path = None,
        audio_path: Path = None,
        infer_type: Literal[
            "mfcc_full_control",
            "mfcc_pose_only",
            "hubert_pose_only",
            "hubert_audio_only",
            "hubert_full_control",
        ] = None,
        pose_yaw: float = None,
        pose_pitch: float = None,
        pose_roll: float = None,
        face_location: float = None,
        face_scale: float = None,
        step_t: int = None,
        face_sr: bool = None,
        seed: int = None,
    ):
        return generate_video(
            image_path or self.settings.model.image_path,
            audio_path or self.settings.model.audio_path,
            infer_type or self.settings.model.infer_type,
            pose_yaw or self.settings.model.pose_yaw,
            pose_pitch or self.settings.model.pose_pitch,
            pose_roll or self.settings.model.pose_roll,
            face_location or self.settings.model.face_location,
            face_scale or self.settings.model.face_scale,
            step_t or self.settings.model.step_t,
            face_sr or self.settings.model.face_sr,
            seed or self.settings.model.seed,
        )
