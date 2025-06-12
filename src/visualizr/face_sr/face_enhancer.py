from os import path
from typing import Literal

import cv2
import torch
from gfpgan import GFPGANer
from numpy import dtype, generic, ndarray
from tqdm import tqdm

from visualizr import logger
from visualizr.face_sr.videoio import load_video_to_cv2


class GeneratorWithLen(object):
    """From https://stackoverflow.com/a/7460929"""

    def __init__(self, gen, length) -> None:
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen


def enhancer_list(
    images,
    method: Literal["gfpgan", "RestoreFormer", "codeformer"] = "gfpgan",
    bg_upsampler="realesrgan",
):
    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler)
    return list(gen)


def enhancer_generator_with_len(
    images,
    method: Literal["gfpgan", "RestoreFormer", "codeformer"] = "gfpgan",
    bg_upsampler="realesrgan",
) -> GeneratorWithLen:
    """Provide a generator with a __len__ method so that it can be passed to functions that
    call len()"""

    if path.isfile(path=images):  # handle video to images
        images: list[ndarray[tuple[float, float, float], dtype[generic]]] = (
            load_video_to_cv2(input_path=images)
        )

    gen = enhancer_generator_no_len(
        images_path=images, method=method, bg_upsampler=bg_upsampler
    )
    gen_with_len = GeneratorWithLen(gen=gen, length=len(images))
    return gen_with_len


def enhancer_generator_no_len(
    images_path: str,
    method: Literal["gfpgan", "RestoreFormer", "codeformer"] = "gfpgan",
    bg_upsampler: str = "realesrgan",
):
    """Provide a generator function so that all the enhanced images don't need
    to be stored in memory at the same time. This can save tons of RAM compared to
    the enhancer function."""
    images = []
    arch = None
    channel_multiplier = None
    model_name = None
    url = None

    if method not in ["gfpgan", "RestoreFormer", "codeformer"]:
        raise ValueError(f"method {method} is not supported yet.")

    logger.info("face enhancer....")
    if not isinstance(images_path, list) and path.isfile(path=images_path):
        images: list[ndarray[tuple[float, float, float], dtype[generic]]] = (
            load_video_to_cv2(input_path=images_path)
        )

    # ------------------------ set up GFPGAN restorer ------------------------
    match method:
        case "gfpgan":
            arch = "clean"
            channel_multiplier = 2
            model_name = "GFPGANv1.4"
            url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
        case "RestoreFormer":
            arch = "RestoreFormer"
            channel_multiplier = 2
            model_name = "RestoreFormer"
            url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
        case "codeformer":
            arch = "CodeFormer"
            channel_multiplier = 2
            model_name = "CodeFormer"
            url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"

    # ------------------------ set up background upsampler ------------------------
    if bg_upsampler == "realesrgan":
        if not torch.cuda.is_available():  # CPU
            import warnings

            warnings.warn(
                message="The unoptimized RealESRGAN is slow on CPU. We do not use it. "
                "If you really want to use it, please modify the corresponding codes."
            )
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                model=model,
                tile=400,
                pre_pad=0,
                half=True,
            )  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # determine model paths
    model_path = path.join("gfpgan/weights", model_name + ".pth")

    if not path.isfile(model_path):
        model_path = path.join("checkpoints", model_name + ".pth")

    if not path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler,
    )

    # ------------------------ restore ------------------------
    for idx in tqdm(range(len(images)), "Face Enhancer:"):
        img = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)

        # restore faces and background if necessary
        _, _, r_img = restorer.enhance(img)

        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        yield r_img
