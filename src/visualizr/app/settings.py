"""Settings for the Visualizr app."""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, DirectoryPath, Field, FilePath, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.cuda import is_available

from visualizr.app.logger import logger
from visualizr.app.types import InferenceType

load_dotenv()


class DirectorySettings(BaseModel):
    """Directory settings for the Visualizr app."""

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

        Checks and creates any missing directories defined in the `DirectorySettings`.

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
            if not directory.exists():
                directory.mkdir(exist_ok=True)
                logger.info("Created directory %s.", directory)
        return self


class Checkpoint(BaseModel):
    """Checkpoint settings for the Visualizr app."""

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
    """Model settings for the Visualizr app."""

    pose_yaw: float = 0.0
    pose_pitch: float = 0.0
    pose_roll: float = 0.0
    face_location: float = 0.5
    face_scale: float = 0.5
    step_t: int = 50
    seed: int = 0
    motion_dim: int = 20
    image_path: FilePath | None = Field(
        default=None,
        description="The path to the image to generate a video from.",
    )
    audio_path: FilePath | None = Field(
        default=None,
        description="The path to the audio to generate a video from.",
    )
    control_flag: bool = True
    pose_driven_path: str = "not_supported_in_this_mode"
    image_size: int = 256
    device: Literal["cuda", "cpu"] = "cuda" if is_available() else "cpu"
    decoder_layers: int = 2
    repo_id: str = "taocode/anitalker_ckpts"
    revision: str = "main"
    infer_type: InferenceType = Field(default="mfcc_full_control")
    face_sr: bool = False
    checkpoint: Checkpoint = Checkpoint()

    @model_validator(mode="after")
    def check_image_path(self) -> "ModelSettings":
        """
        Check if the image and audio paths exist.

        Args:
            self (ModelSettings): The ModelSettings instance to check.

        Returns:
            Self: The validated ModelSettings instance.
        """
        if self.image_path and not self.image_path.exists():
            msg_image_err: str = f"Image path does not exist: {self.image_path}"
            logger.error(msg_image_err)
            raise FileNotFoundError(msg_image_err)
        if self.audio_path and not self.audio_path.exists():
            msg_audio_err: str = f"Audio path does not exist: {self.audio_path}"
            logger.error(msg_audio_err)
            raise FileNotFoundError(msg_audio_err)
        return self


class Settings(BaseSettings):
    """Configuration for the Visualizr app."""

    model_config: SettingsConfigDict = SettingsConfigDict(
        env_nested_delimiter="__",
        env_parse_none_str="None",
        env_file=".env",
        extra="ignore",
    )
    directory: DirectorySettings = DirectorySettings()
    model: ModelSettings = ModelSettings()
