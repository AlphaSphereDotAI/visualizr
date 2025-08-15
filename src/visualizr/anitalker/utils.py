import os
from importlib.util import find_spec
from pathlib import Path
from shutil import rmtree
from sys import exit
from time import time
from typing import Literal

import librosa
import numpy as np
import python_speech_features
import torch
from gradio import Markdown, Video
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)
from PIL import Image
from torch import Tensor
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from visualizr.anitalker.config import TrainConfig
from visualizr.anitalker.experiment import LitModel
from visualizr.anitalker.LIA_Model import LIA_Model
from visualizr.anitalker.templates import ffhq256_autoenc
from visualizr.settings import Checkpoint, Settings, logger


def check_package_installed(package_name: str) -> bool:
    return find_spec(package_name) is not None


def frames_to_video(input_path, audio_path, output_path, fps=25):
    image_files = [
        os.path.join(input_path, img) for img in sorted(os.listdir(input_path))
    ]
    clips = [ImageClip(m).set_duration(1 / fps) for m in image_files]
    video = concatenate_videoclips(clips, method="compose")
    audio = AudioFileClip(audio_path)
    final_video = video.set_audio(audio)
    final_video.write_videofile(output_path, fps, "libx264", audio_codec="aac")


def load_image(filename: str, size: int) -> np.ndarray:
    img: Image.Image = Image.open(filename).convert("RGB")
    img_resized: Image.Image = img.resize((size, size))
    img_np: np.ndarray = np.asarray(img_resized)
    img_transposed: np.ndarray = np.transpose(img_np, (2, 0, 1))  # 3 x 256 x 256
    return img_transposed / 255.0


def img_preprocessing(img_path: str, size: int) -> Tensor:
    img_np: np.ndarray = load_image(img_path, size)  # [0, 1]
    img: Tensor = torch.from_numpy(img_np).unsqueeze(0).float()  # [0, 1]
    normalized_image: Tensor = (img - 0.5) * 2.0  # [-1, 1]
    return normalized_image


def saved_image(img_tensor: Tensor, img_path: str) -> None:
    pil_image_converter: ToPILImage = ToPILImage()
    img = pil_image_converter(img_tensor.detach().cpu().squeeze(0))
    img.save(img_path)


def load_stage_1_model() -> LIA_Model:
    logger.info("Loading stage 1 model... ")
    lia: LIA_Model = LIA_Model(
        motion_dim=Settings.model.motion_dim, fusion_type="weighted_sum"
    )
    lia.load_lightning_model(Settings.directory.checkpoint_stage_1)
    lia.to("cuda")
    return lia


def load_stage_2_model(conf: TrainConfig, stage2_checkpoint_path: str) -> LitModel:
    logger.info("Loading stage 2 model... ")
    model = LitModel(conf)
    state = torch.load(stage2_checkpoint_path, "cpu")
    model.load_state_dict(state)
    model.ema_model.eval()
    model.ema_model.to("cuda")
    return model


def init_conf(
    infer_type: Literal[
        "mfcc_full_control",
        "mfcc_pose_only",
        "hubert_pose_only",
        "hubert_audio_only",
        "hubert_full_control",
    ],
    seed: int,
) -> TrainConfig:
    logger.info("Initializing configuration... ")
    conf: TrainConfig = ffhq256_autoenc()
    conf.seed = seed
    conf.decoder_layers = 2
    conf.infer_type = infer_type
    conf.motion_dim = Settings.model.motion_dim
    logger.info(f"infer_type: {infer_type}")
    match infer_type:
        case "mfcc_full_control":
            conf.face_location = True
            conf.face_scale = True
            conf.mfcc = True
        case "mfcc_pose_only":
            conf.face_location = False
            conf.face_scale = False
            conf.mfcc = True
        case "hubert_pose_only":
            conf.face_location = False
            conf.face_scale = False
            conf.mfcc = False
        case "hubert_audio_only":
            conf.face_location = False
            conf.face_scale = False
            conf.mfcc = False
        case "hubert_full_control":
            conf.face_location = True
            conf.face_scale = True
            conf.mfcc = False
    return conf


def get_checkpoint_stage_2_path(
    infer_type: Literal[
        "mfcc_full_control",
        "mfcc_pose_only",
        "hubert_pose_only",
        "hubert_audio_only",
        "hubert_full_control",
    ],
) -> Path:
    match infer_type:
        case "mfcc_full_control":
            return Checkpoint.mfcc_full_control
        case "mfcc_pose_only":
            return Checkpoint.mfcc_pose_only
        case "hubert_pose_only":
            return Checkpoint.hubert_pose_only
        case "hubert_audio_only":
            return Checkpoint.hubert_audio_only
        case "hubert_full_control":
            return Checkpoint.hubert_full_control


def main(
    infer_type: Literal[
        "mfcc_full_control",
        "mfcc_pose_only",
        "hubert_pose_only",
        "hubert_audio_only",
        "hubert_full_control",
    ],
    image_path: str,
    audio_path: str,
    face_sr: bool,
    pose_yaw: float,
    pose_pitch: float,
    pose_roll: float,
    face_location: float,
    face_scale: float,
    step_t: int,
    seed: int,
):
    frame_end = None
    audio_driven = None
    if not os.path.exists(image_path):
        logger.exception(f"{image_path} does not exist!")
        exit(0)
    if not os.path.exists(audio_path):
        logger.exception(f"{audio_path} does not exist!")
        exit(0)

    image_name: str = Path(image_path).stem
    audio_name: str = Path(audio_path).stem

    predicted_video_256_path: Path = (
        Settings.directory.results / f"{image_name}-{audio_name}.mp4"
    )
    predicted_video_512_path: Path = (
        Settings.directory.results / f"{image_name}-{audio_name}_SR.mp4"
    )

    lia: LIA_Model = load_stage_1_model()

    conf: TrainConfig = init_conf(infer_type, seed)

    img_source: Tensor = img_preprocessing(image_path, 256).to("cuda")
    one_shot_lia_start, one_shot_lia_direction, feats = lia.get_start_direction_code(
        img_source, img_source, img_source, img_source
    )

    model = load_stage_2_model(conf, get_checkpoint_stage_2_path(infer_type))

    if conf.infer_type.startswith("mfcc"):
        # MFCC features
        wav, sr = librosa.load(audio_path, sr=16000)
        input_values = python_speech_features.mfcc(
            signal=wav, samplerate=sr, numcep=13, winlen=0.025, winstep=0.01
        )
        d_mfcc_feat = python_speech_features.base.delta(input_values, 1)
        d_mfcc_feat2 = python_speech_features.base.delta(input_values, 2)
        audio_driven_obj: np.ndarray = np.hstack(
            (input_values, d_mfcc_feat, d_mfcc_feat2)
        )
        frame_start, frame_end = 0, int(audio_driven_obj.shape[0] / 4)
        # The video frame is fixed to 25 hz, and the audio is fixed to 100 hz.
        audio_start, audio_end = (
            int(frame_start * 4),
            int(frame_end * 4),
        )

        audio_driven = (
            torch.Tensor(audio_driven_obj[audio_start:audio_end, :])
            .unsqueeze(0)
            .float()
            .to("cuda")
        )

    elif conf.infer_type.startswith("hubert"):
        # Hubert features
        if not check_package_installed("transformers"):
            logger.exception("Please install transformers module first.")
            exit(0)
        hubert_model_path = "ckpts/chinese-hubert-large"
        if not os.path.exists(hubert_model_path):
            logger.exception(
                "Please download the hubert weight into the ckpts path first."
            )
            exit(0)
        logger.info(
            "You did not extract the audio features in advance, "
            + "extracting online now, which will increase processing delay"
        )

        start_time = time()

        # load hubert model
        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        audio_model = HubertModel.from_pretrained(hubert_model_path).to("cuda")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_path)
        audio_model.feature_extractor._freeze_parameters()  # skipcq: PYL-W0212
        audio_model.eval()

        # hubert model forward pass
        audio, sr = librosa.load(audio_path, sr=16000)
        input_values = feature_extractor(
            audio,
            sampling_rate=16000,
            padding=True,
            do_normalize=True,
            return_tensors="pt",
        ).input_values
        input_values = input_values.to("cuda")
        ws_feats = []
        with torch.no_grad():
            outputs = audio_model(input_values, output_hidden_states=True)
            for i in range(len(outputs.hidden_states)):
                ws_feats.append(outputs.hidden_states[i].detach().cpu().numpy())
            ws_feat_obj = np.array(ws_feats)
            ws_feat_obj = np.squeeze(ws_feat_obj, 1)
            ws_feat_obj = np.pad(
                ws_feat_obj, ((0, 0), (0, 1), (0, 0)), "edge"
            )  # align the audio length with the video frame

        execution_time = time() - start_time
        logger.info(f"Extraction Audio Feature: {execution_time:.2f} Seconds")

        audio_driven_obj = ws_feat_obj

        frame_start, frame_end = 0, int(audio_driven_obj.shape[1] / 2)
        audio_start, audio_end = (
            int(frame_start * 2),
            int(frame_end * 2),
        )  # The video frame is fixed to 25 hz, and the audio is fixed to 50 hz

        audio_driven = (
            torch.Tensor(audio_driven_obj[:, audio_start:audio_end, :])
            .unsqueeze(0)
            .float()
            .to("cuda")
        )

    # Diffusion Noise
    noisy_t = torch.randn((1, frame_end, Settings.model.motion_dim)).to("cuda")

    # ======Inputs for Attribute Control=========
    yaw_signal = torch.zeros(1, frame_end, 1).to("cuda") + pose_yaw
    pitch_signal = torch.zeros(1, frame_end, 1).to("cuda") + pose_pitch
    roll_signal = torch.zeros(1, frame_end, 1).to("cuda") + pose_roll
    pose_signal = torch.cat((yaw_signal, pitch_signal, roll_signal), dim=-1)

    pose_signal = torch.clamp(pose_signal, -1, 1)

    face_location_signal = torch.zeros(1, frame_end, 1).to("cuda") + face_location
    face_scale_tensor = torch.zeros(1, frame_end, 1).to("cuda") + face_scale
    # ===========================================
    start_time = time()
    # ======Diffusion De-nosing Process=========
    generated_directions = model.render(
        one_shot_lia_start,
        one_shot_lia_direction,
        audio_driven,
        face_location_signal,
        face_scale_tensor,
        pose_signal,
        noisy_t,
        step_t,
        True,
    )
    # =========================================

    execution_time = time() - start_time
    logger.info(f"Motion Diffusion Model: {execution_time:.2f} Seconds")

    generated_directions = generated_directions.detach().cpu().numpy()

    start_time = time()
    # ======Rendering images frame-by-frame=========
    for pred_index in tqdm(range(generated_directions.shape[1])):
        ori_img_recon = lia.render(
            one_shot_lia_start,
            torch.Tensor(generated_directions[:, pred_index, :]).to("cuda"),
            feats,
        )
        ori_img_recon = ori_img_recon.clamp(-1, 1)
        wav_pred = (ori_img_recon.detach() + 1) / 2
        saved_image(
            wav_pred,
            os.path.join(Settings.directory.frames, f"{pred_index:06d}.png"),
        )
    # ==============================================

    execution_time = time() - start_time
    logger.info(f"Renderer Model: {execution_time:.2f} Seconds")
    logger.info(f"Saving video at {predicted_video_256_path}")

    frames_to_video(
        str(Settings.directory.frames),
        audio_path,
        str(predicted_video_256_path),
    )

    rmtree(Settings.directory.frames)

    # Enhancer
    if face_sr and check_package_installed("gfpgan"):
        from imageio import mimsave

        from visualizr.anitalker.face_sr.face_enhancer import enhancer_list

        # Super-resolution
        mimsave(
            predicted_video_512_path / Settings.directory.tmp_extension,
            enhancer_list(predicted_video_256_path, bg_upsampler=None),
            fps=25.0,
        )

        # Merge audio and video
        video_clip = VideoFileClip(
            predicted_video_512_path / Settings.directory.tmp_extension
        )
        audio_clip = AudioFileClip(predicted_video_256_path)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(
            predicted_video_512_path, codec="libx264", audio_codec="aac"
        )

        os.remove(predicted_video_512_path / Settings.directory.tmp_extension)

    if face_sr:
        return predicted_video_256_path, predicted_video_512_path
    return predicted_video_256_path, predicted_video_256_path


def generate_video(
    uploaded_img: str,
    uploaded_audio: str,
    infer_type: Literal[
        "mfcc_full_control",
        "mfcc_pose_only",
        "hubert_pose_only",
        "hubert_audio_only",
        "hubert_full_control",
    ],
    pose_yaw: float,
    pose_pitch: float,
    pose_roll: float,
    face_location: float,
    face_scale: float,
    step_t: int,
    face_sr: bool,
    seed: int,
) -> tuple[Video | None, Video | None, Markdown]:
    if not uploaded_img or not uploaded_audio:
        return (
            None,
            None,
            Markdown(
                "Error: Input image or audio file is empty. "
                + "Please check and upload both files."
            ),
        )
    try:
        output_256_video_path, output_512_video_path = main(
            infer_type,
            uploaded_img,
            uploaded_audio,
            face_sr,
            pose_yaw,
            pose_pitch,
            pose_roll,
            face_location,
            face_scale,
            step_t,
            seed,
        )
        if not os.path.exists(output_256_video_path):
            return (
                None,
                None,
                Markdown(
                    "Error: Video generation failed. "
                    + "Please check your inputs and try again."
                ),
            )
        if output_256_video_path == output_512_video_path:
            return (
                Video(value=output_256_video_path),
                None,
                Markdown("Video (256*256 only) generated successfully!"),
            )
        return (
            Video(value=output_256_video_path),
            Video(value=output_512_video_path),
            Markdown("Video generated successfully!"),
        )

    except Exception as e:
        return (
            None,
            None,
            Markdown(f"Error: An unexpected error occurred - {str(e)}"),
        )
