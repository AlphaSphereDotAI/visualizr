from importlib.util import find_spec
from pathlib import Path

from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    concatenate_videoclips,
)
from numpy import asarray, ndarray, transpose
from PIL import Image
from torch import Tensor, from_numpy
from torchvision.transforms import ToPILImage

from visualizr.settings import logger


def check_package_installed(package_name: str) -> bool:
    return find_spec(package_name) is not None


def frames_to_video(
    input_path: Path,
    audio_path: Path,
    output_path: Path,
    fps: int = 25,
):
    image_files = [input_path / img for img in sorted(input_path.iterdir())]
    clips = [ImageClip(m.as_posix()).set_duration(1 / fps) for m in image_files]
    video = concatenate_videoclips(clips, method="compose")
    audio = AudioFileClip(audio_path)
    final_video = video.set_audio(audio)
    final_video.write_videofile(
        output_path.as_posix(), fps, "libx264", audio_codec="aac"
    )


def load_image(filename: str, size: int) -> ndarray:
    img: Image.Image = Image.open(filename).convert("RGB")
    img_resized: Image.Image = img.resize((size, size))
    img_np: ndarray = asarray(img_resized)
    img_transposed: ndarray = transpose(img_np, (2, 0, 1))  # 3 x 256 x 256
    return img_transposed / 255.0


def img_preprocessing(img_path: str, size: int) -> Tensor:
    img_np: ndarray = load_image(img_path, size)  # [0, 1]
    img: Tensor = from_numpy(img_np).unsqueeze(0).float()  # [0, 1]
    normalized_image: Tensor = (img - 0.5) * 2.0  # [-1, 1]
    return normalized_image


def saved_image(img_tensor: Tensor, img_path: Path) -> None:
    pil_image_converter: ToPILImage = ToPILImage()
    img = pil_image_converter(img_tensor.detach().cpu().squeeze(0))
    img.save(img_path)


def remove_frames(frames_path: Path):
    for frame in frames_path.iterdir():
        try:
            frame.unlink()
            logger.info(f"Deleted {frame}")
        except OSError:
            logger.exception(f"Error while deleting file {frame}")
            continue
