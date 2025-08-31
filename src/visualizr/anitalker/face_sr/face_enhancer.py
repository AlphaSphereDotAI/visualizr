"""
This module provides face image enhancement functionality using GFPGAN,
RestoreFormer, and CodeFormer models, and background upsampling with
RealESRGAN. It includes functions to generate enhanced images as lists or
generators to optimize memory usage.
"""

import os

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from gradio import Info
from realesrgan import RealESRGANer
from tqdm import tqdm

from visualizr.anitalker.face_sr.videoio import load_video_to_cv2


class GeneratorWithLen:
    """From https://stackoverflow.com/a/7460929"""

    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen


def enhancer_list(images, method="gfpgan", bg_upsampler="realesrgan"):
    """
    Generate a list of enhanced images from the input images using the specified
    face enhancement method and background upsampler.

    Args:
        images (Union[list, str]): A list of images or a file path to a video
            to be processed.
        method (str): The face enhancement model to use ("gfpgan",
            "RestoreFormer", or "codeformer").
        bg_upsampler (str): The background upsampler to use ("realesrgan").

    Returns:
        list: A list of enhanced images.
    """
    gen = enhancer_generator_no_len(images, method, bg_upsampler)
    return list(gen)


def enhancer_generator_with_len(images, method="gfpgan", bg_upsampler="realesrgan"):
    """
    Provide a generator with a `__len__` method
    so that it can be passed to functions that
    call `len()`
    """
    if os.path.isfile(images):  # handle video to images
        images = load_video_to_cv2(images)

    gen = enhancer_generator_no_len(images, method, bg_upsampler)
    return GeneratorWithLen(gen, len(images))


def enhancer_generator_no_len(images, method="gfpgan", bg_upsampler="realesrgan"):
    """
    Provide a generator function so that all the enhanced images don't need
    to be stored in memory at the same time. This can save tons of RAM compared to
    the enhancer function.
    """
    if method not in ["gfpgan", "RestoreFormer", "codeformer"]:
        raise ValueError(f"Wrong model version {method}.")
    Info("face enhancer....")
    if not isinstance(images, list) and os.path.isfile(
        images
    ):  # handle video to images
        images = load_video_to_cv2(images)
    channel_multiplier = None
    model_name = None
    url = None
    arch = None
    # ------------------------ set up GFPGAN restorer ------------------------
    match method:
        case "gfpgan":
            arch = "clean"
            channel_multiplier = 2
            model_name = "GFPGANv1.4"
            url = (
                "https://github.com/TencentARC/GFPGAN/releases/download/"
                "v1.3.0/GFPGANv1.4.pth"
            )
        case "RestoreFormer":
            arch = "RestoreFormer"
            channel_multiplier = 2
            model_name = "RestoreFormer"
            url = (
                "https://github.com/TencentARC/GFPGAN/releases/download/"
                "v1.3.4/RestoreFormer.pth"
            )
        case "codeformer":
            arch = "CodeFormer"
            channel_multiplier = 2
            model_name = "CodeFormer"
            url = (
                "https://github.com/sczhou/CodeFormer/releases/download/"
                "v0.1.0/codeformer.pth"
            )
    # ------------------------ set up background upsampler ------------------------
    if bg_upsampler == "realesrgan":
        if not torch.cuda.is_available():  # CPU
            import warnings

            warnings.warn(
                (
                    "The unoptimized RealESRGAN is slow on CPU. "
                    "We do not use it. "
                    "If you really want to use it, "
                    "please modify the corresponding codes."
                )
            )
            bg_upsampler = None
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path=(
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
                    "v0.2.1/RealESRGAN_x2plus.pth"
                ),
                model=model,
                tile=400,
                pre_pad=0,
                half=True,
            )  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # determine model paths
    model_path = os.path.join("gfpgan/weights", f"{model_name}.pth")

    if not os.path.isfile(model_path):
        model_path = os.path.join("checkpoints", f"{model_name}.pth")

    if not os.path.isfile(model_path):
        # download pre-trained models from URL
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

        yield cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
