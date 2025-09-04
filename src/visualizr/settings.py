"""This module contains the settings for the Visualizr app."""

from pathlib import Path
from sys import exit
from typing import Literal

from dotenv import load_dotenv
from gradio import Error, Info
from pydantic import BaseModel, DirectoryPath, Field, FilePath, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.cuda import is_available

load_dotenv()


class DirectorySettings(BaseModel):
    base: DirectoryPath = Field(default_factory=lambda: Path.cwd())
    results: DirectoryPath = Field(default_factory=lambda: Path.cwd() / "results")
    frames: DirectoryPath = Field(
        default_factory=lambda: Path.cwd() / "results" / "frames",
    )
    checkpoint: DirectoryPath = Field(default_factory=lambda: Path.cwd() / "ckpts")
    log: DirectoryPath = Field(default_factory=lambda: Path.cwd() / "logs")
    assets: DirectoryPath = Field(default_factory=lambda: Path.cwd() / "assets")
    image: DirectoryPath = Field(
        default_factory=lambda: Path.cwd() / "assets" / "image",
    )
    audio: DirectoryPath = Field(
        default_factory=lambda: Path.cwd() / "assets" / "audio",
    )
    video: DirectoryPath = Field(
        default_factory=lambda: Path.cwd() / "assets" / "video",
    )

    @model_validator(mode="after")
    def create_missing_dirs(self) -> "DirectorySettings":
        """
        Ensure that all specified directories exist, creating them if necessary.
        This method checks and creates any missing directories defined in the DirectorySettings.

        Returns:
            Self: The validated DirectorySettings instance.
        """
        for directory in [
            self.base,
            self.results,
            self.frames,
            self.checkpoint,
            self.assets,
            self.log,
            self.image,
            self.audio,
            self.video,
        ]:
            directory.mkdir(exist_ok=True)
            Info(f"Created directory: {directory}")
        return self


class Checkpoint(BaseModel):
    stage_1: FilePath = Field(
        default_factory=lambda: Path.cwd() / "ckpts" / "stage1.ckpt",
    )
    mfcc_pose_only: FilePath = Field(
        default_factory=lambda: Path.cwd() / "ckpts" / "stage2_pose_only_mfcc.ckpt",
    )
    mfcc_full_control: FilePath = Field(
        default_factory=lambda: Path.cwd() / "ckpts" / "stage2_full_control_mfcc.ckpt",
    )
    hubert_audio_only: FilePath = Field(
        default_factory=lambda: Path.cwd() / "ckpts" / "stage2_audio_only_hubert.ckpt",
    )
    hubert_pose_only: FilePath = Field(
        default_factory=lambda: Path.cwd() / "ckpts" / "stage2_pose_only_hubert.ckpt",
    )
    hubert_full_control: FilePath = Field(
        default_factory=lambda: Path.cwd()
        / "ckpts"
        / "stage2_full_control_hubert.ckpt",
    )


class ModelSettings(BaseModel):
    pose_yaw: float = 0.0
    pose_pitch: float = 0.0
    pose_roll: float = 0.0
    face_location: float = 0.5
    face_scale: float = 0.5
    step_t: int = 50
    seed: int = 0
    motion_dim: int = 20

    image_path: FilePath = Field(default=None)
    audio_path: FilePath = Field(default=None)

    control_flag: bool = True
    pose_driven_path: str = "not_supported_in_this_mode"
    image_size: int = 256
    device: Literal["cuda", "cpu"] = "cuda" if is_available() else "cpu"
    motion_dim: int = 20
    decoder_layers: int = 2

    repo_id: str = "taocode/anitalker_ckpts"
    infer_type: Literal[
        "mfcc_full_control",
        "mfcc_pose_only",
        "hubert_pose_only",
        "hubert_audio_only",
        "hubert_full_control",
    ] = Field(default="mfcc_full_control")
    face_sr: bool = False
    checkpoint: Checkpoint = Checkpoint()

    @model_validator(mode="after")
    def check_image_path(self) -> "ModelSettings":
        if self.image_path and not self.image_path.exists():
            Error("Image path does not exist.")
            exit(0)
        if self.audio_path and not self.audio_path.exists():
            Error("Audio path does not exist.")
            exit(0)
        return self


class Settings(BaseSettings):
    """Configuration for the Visualizr app."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_parse_none_str="None",
        env_file=".env",
        extra="ignore",
    )
    directory: DirectorySettings = DirectorySettings()
    model: ModelSettings = ModelSettings()


if __name__ == "__main__":
    print(Settings().model_dump())
    print(__package__)
