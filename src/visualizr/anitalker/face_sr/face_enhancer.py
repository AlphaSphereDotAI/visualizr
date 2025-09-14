"""
Provides face image enhancement capability.

Using GFPGAN, RestoreFormer, and CodeFormer models, and background upsampling with
RealESRGAN. It includes functions to generate enhanced images as lists or
generators to optimize memory usage.
"""

from collections.abc import Generator
from pathlib import Path
from typing import Literal

import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from gradio import Info, Warning as grWarning
from numpy import ndarray
from realesrgan import RealESRGANer
from torch.cuda import is_available
from tqdm import tqdm

from visualizr.anitalker.face_sr.videoio import load_video_to_cv2
from visualizr.app.logger import logger

GH: str = "https://github.com"
REAL_ESRGAN_X_2_PLUS_MODEL_PATH: str = (
    f"{GH}/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
)
GFPGAN_V_1_4_MODEL_URL: str = (
    f"{GH}/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
)
RESTORE_FORMER_MODEL_URL: str = (
    f"{GH}/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
)
CODE_FORMER_MODEL_URL: str = (
    f"{GH}/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
)
GFPGAN_WEIGHTS: Path = Path("gfpgan/weights")


def enhancer_list(
    images: Path,
    method: str = "gfpgan",
    bg_upsampler: str | None = "realesrgan",
) -> list:
    """
    Generate a list of enhanced images.

    From given images using the specified
    face enhancement method and background upsampler.

    Args:
        images: A path of images to be processed.
        method: The face enhancement model to use
            ("gfpgan", "RestoreFormer", or "codeformer").
        bg_upsampler: The background upsampler to use ("realesrgan").

    Returns:
        list: A list of enhanced images.
    """
    gen = enhancer_generator_no_len(images, method, bg_upsampler)
    return list(gen)


def setup_gfpgan_restorer(method: str):
    channel_multiplier: int | None = None
    model_name: str | None = None
    url: str | None = None
    arch: str | None = None
    match method:
        case "gfpgan":
            arch = "clean"
            channel_multiplier = 2
            model_name = "GFPGANv1.4"
            url = GFPGAN_V_1_4_MODEL_URL
        case "RestoreFormer":
            arch = "RestoreFormer"
            channel_multiplier = 2
            model_name = "RestoreFormer"
            url = RESTORE_FORMER_MODEL_URL
        case "codeformer":
            arch = "CodeFormer"
            channel_multiplier = 2
            model_name = "CodeFormer"
            url = CODE_FORMER_MODEL_URL
    if model_name is None or url is None or arch is None or channel_multiplier is None:
        msg: str = "`model_name`, `url`, `arch`, and `channel_multiplier` must be set"
        raise ValueError(msg)
    return channel_multiplier, model_name, url, arch


def setup_background_upsampler(bg_upsampler: str) -> RealESRGANer | None:
    _bg_upsampler: RealESRGANer | None = None
    if bg_upsampler == "realesrgan":
        if not is_available():  # CPU
            _msg: str = (
                "The unoptimized RealESRGAN is slow on CPU. "
                "We do not use it. "
                "If you really want to use it, "
                "please modify the corresponding codes."
            )
            logger.warning(_msg)
            grWarning(_msg)
            _bg_upsampler = None
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2)
            # need to set False in CPU mode
            _bg_upsampler = RealESRGANer(
                scale=2,
                model_path=REAL_ESRGAN_X_2_PLUS_MODEL_PATH,
                model=model,
                tile=400,
                pre_pad=0,
                half=True,
            )
    else:
        _bg_upsampler = None
    return _bg_upsampler


def enhancer_generator_no_len(
    images: Path,
    method: Literal["gfpgan", "RestoreFormer", "codeformer"] = "gfpgan",
    bg_upsampler: str = "realesrgan",
) -> Generator[ndarray]:
    """
    Generate enhanced face images as a generator without a defined length.

    This function yields enhanced images one by one using the specified face
    enhancement method and background upsampler, optimizing memory usage for
    large datasets.

    Args:
        images: A path of images to be processed.
        method: The face enhancement model to
                use ("gfpgan", "RestoreFormer", or "codeformer").
        bg_upsampler: The background upsampler to use ("realesrgan").

    Yields:
        ndarray: An enhanced image as a NumPy array.

    Raises:
        ValueError: If an unsupported model version is specified.
    """
    if method not in ["gfpgan", "RestoreFormer", "codeformer"]:
        msg: str = (
            f"Wrong model version {method}. "
            "Expected one of: gfpgan, RestoreFormer, codeformer."
        )
        raise ValueError(msg)
    _msg = f"face enhancer: {method}"
    logger.info(_msg)
    Info(_msg)
    if not isinstance(images, list) and images.is_file():
        # handle video to images
        images = load_video_to_cv2(images.as_posix())

    # Setup GFPGAN restorer
    channel_multiplier, model_name, url, arch = setup_gfpgan_restorer(method)

    # Setup background upsampler
    _bg_upsampler = setup_background_upsampler(bg_upsampler)

    # determine model paths
    model_path: Path = GFPGAN_WEIGHTS / f"{model_name}.pth"

    if not model_path.is_file():
        # download pre-trained models from URL
        model_path: str = url

    restorer = GFPGANer(
        model_path=model_path if isinstance(model_path, str) else model_path.as_posix(),
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=_bg_upsampler,
    )

    # restore
    for idx in tqdm(range(len(images)), "Face Enhancer:"):
        img = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)
        # restore faces and background if necessary
        _, _, r_img = restorer.enhance(img)
        yield cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
