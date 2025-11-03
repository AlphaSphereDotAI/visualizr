"""Settings for the Visualizr app."""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from gradio import Error
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    PositiveInt,
    StrictStr,
    computed_field,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.cuda import is_available

from visualizr import APP_NAME
from visualizr.anitalker.checkpoint import ModelName
from visualizr.app.logger import logger
from visualizr.app.types import InferenceType

load_dotenv()


class DirectorySettings(BaseModel):
    """Settings for application directories."""

    base: DirectoryPath = Field(default_factory=Path.cwd, frozen=True)

    @computed_field
    @property
    def results(self) -> DirectoryPath:
        """Path to the results directory."""
        return self.base / "results" / APP_NAME

    @computed_field
    @property
    def frames(self) -> DirectoryPath:
        """Path to the frames directory."""
        return self.results / "frames"

    @computed_field
    @property
    def checkpoint(self) -> DirectoryPath:
        """Path to the checkpoint directory."""
        return self.base / "ckpts"

    @computed_field
    @property
    def log(self) -> DirectoryPath:
        """Path to the log directory."""
        return self.base / "logs" / APP_NAME

    @computed_field
    @property
    def assets(self) -> DirectoryPath:
        """Path to the assets directory."""
        return self.base / "assets"

    @computed_field
    @property
    def image(self) -> DirectoryPath:
        """Path to the image directory."""
        return self.assets / "image"

    @computed_field
    @property
    def audio(self) -> DirectoryPath:
        """Path to the audio directory."""
        return self.assets / "audio"

    @computed_field
    @property
    def video(self) -> DirectoryPath:
        """Path to the video directory."""
        return self.assets / "video"

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
                directory.mkdir(exist_ok=True, parents=True)
                logger.info("Created directory %s.", directory)
        return self


class Checkpoint(BaseModel):
    """Settings for model checkpoints."""

    base: DirectoryPath = Field(
        default_factory=lambda: Path.cwd() / "ckpts",
        frozen=True,
        exclude=True,
    )

    @computed_field
    @property
    def stage_1(self) -> FilePath:
        """Path to the stage 1 checkpoint file."""
        return self.base / ModelName.stage_1

    @computed_field
    @property
    def mfcc_pose_only(self) -> FilePath:
        """Path to the MFCC pose-only checkpoint file."""
        return self.base / ModelName.mfcc_pose_only

    @computed_field
    @property
    def mfcc_full_control(self) -> FilePath:
        """Path to the MFCC full-control checkpoint file."""
        return self.base / ModelName.mfcc_full_control

    @computed_field
    @property
    def hubert_audio_only(self) -> FilePath:
        """Path to the Hubert audio-only checkpoint file."""
        return self.base / ModelName.hubert_audio_only

    @computed_field
    @property
    def hubert_pose_only(self) -> FilePath:
        """Path to the Hubert pose-only checkpoint file."""
        return self.base / ModelName.hubert_pose_only

    @computed_field
    @property
    def hubert_full_control(self) -> FilePath:
        """Path to the Hubert full-control checkpoint file."""
        return self.base / ModelName.hubert_full_control


class ModelSettings(BaseModel):
    """Settings for the model configuration."""

    pose_yaw: float = Field(default=0.0, ge=-1, le=1)
    pose_pitch: float = Field(default=0.0, ge=-1, le=1)
    pose_roll: float = Field(default=0.0, ge=-1, le=1)
    face_location: float = Field(default=0.5, ge=0, le=1)
    face_scale: float = Field(default=0.5, ge=0, le=1)
    step_t: PositiveInt = Field(default=50, ge=1, le=100)
    seed: int = Field(default=0)
    motion_dim: PositiveInt = Field(default=20)
    image_path: FilePath | None = Field(default=None)
    audio_path: FilePath | None = Field(default=None)
    control_flag: bool = Field(default=True)
    pose_driven_path: StrictStr = Field(
        default="not_supported_in_this_mode",
        frozen=True,
        exclude=True,
    )
    image_size: PositiveInt = Field(default=256)
    device: Literal["cuda", "cpu"] = Field(
        default_factory=lambda: "cuda" if is_available() else "cpu",
    )
    decoder_layers: PositiveInt = Field(default=2)
    repo_id: str = Field(default="taocode/anitalker_ckpts")
    revision: str = Field(default="main")
    infer_type: InferenceType = Field(default="mfcc_full_control")
    face_sr: bool = Field(default=False)
    checkpoint: Checkpoint = Field(default_factory=Checkpoint, frozen=True)

    @model_validator(mode="after")
    def check_missing_paths(self) -> "ModelSettings":
        """
        Validate that the image and audio paths exist if provided.

        Checks the existence of the image_path and audio_path attributes.

        Returns:
            ModelSettings: The validated ModelSettings instance.

        Raises:
            FileNotFoundError: If the image_path or audio_path does not exist.
        """
        if self.image_path and not self.image_path.exists():
            _msg: StrictStr = f"Image path does not exist: {self.image_path}"
            logger.error(_msg)
            Error(_msg)
            raise FileNotFoundError(_msg)
        if self.audio_path and not self.audio_path.exists():
            _msg: StrictStr = f"Audio path does not exist: {self.audio_path}"
            logger.error(_msg)
            Error(_msg)
            raise FileNotFoundError(_msg)
        return self


class Settings(BaseSettings):
    """Configuration for the Visualizr app."""

    model_config: SettingsConfigDict = SettingsConfigDict(
        env_nested_delimiter="__",
        env_parse_none_str="None",
        env_file=".env",
        extra="ignore",
    )
    directory: DirectorySettings = Field(default_factory=DirectorySettings, frozen=True)
    model: ModelSettings = Field(default_factory=ModelSettings)


if __name__ == "__main__":
    from rich import print as rprint

    rprint(Settings().model_dump_json(indent=4))
