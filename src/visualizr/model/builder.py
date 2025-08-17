from os import path, remove
from pathlib import Path
from sys import exit
from time import time
from typing import Literal, Optional

import librosa
import numpy as np
import torch
from gradio import Markdown, Video
from huggingface_hub import snapshot_download
from imageio import mimsave
from moviepy.editor import AudioFileClip, VideoFileClip
from python_speech_features import mfcc
from python_speech_features.base import delta
from torch import Tensor
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from visualizr.anitalker.config import TrainConfig
from visualizr.anitalker.experiment import LitModel
from visualizr.anitalker.face_sr.face_enhancer import enhancer_list
from visualizr.anitalker.LIA_Model import LIA_Model
from visualizr.anitalker.templates import ffhq256_autoenc
from visualizr.anitalker.utils import (
    check_package_installed,
    frames_to_video,
    img_preprocessing,
    remove_frames,
    saved_image,
)
from visualizr.settings import Settings, logger


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
        return self.generate_video(
            infer_type or self.settings.model.infer_type,
            image_path or self.settings.model.image_path,
            audio_path or self.settings.model.audio_path,
            face_sr or self.settings.model.face_sr,
            pose_yaw or self.settings.model.pose_yaw,
            pose_pitch or self.settings.model.pose_pitch,
            pose_roll or self.settings.model.pose_roll,
            face_location or self.settings.model.face_location,
            face_scale or self.settings.model.face_scale,
            step_t or self.settings.model.step_t,
            seed or self.settings.model.seed,
        )

    def generate_video(
        self,
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
    ) -> tuple[Video | None, Video | None, Markdown]:
        if not image_path or not audio_path:
            return (
                None,
                None,
                Markdown(
                    "Error: Input image or audio file is empty. "
                    + "Please check and upload both files."
                ),
            )
        if not Path(image_path).exists():
            logger.exception(f"{image_path} does not exist!")
            exit(0)
        if not Path(audio_path).exists():
            logger.exception(f"{audio_path} does not exist!")
            exit(0)

        image_name: str = Path(image_path).stem
        audio_name: str = Path(audio_path).stem

        predicted_video_256_path: Path = (
            self.settings.directory.results / f"{image_name}-{audio_name}.mp4"
        )
        predicted_video_512_path: Path = (
            self.settings.directory.results / f"{image_name}-{audio_name}_SR.mp4"
        )

        lia: LIA_Model = self._load_stage_1_model()

        conf: TrainConfig = self._init_conf(infer_type, seed)

        img_source: Tensor = img_preprocessing(image_path, 256).to("cuda")
        one_shot_lia_start, one_shot_lia_direction, feats = (
            lia.get_start_direction_code(img_source, img_source, img_source, img_source)
        )

        model = self._load_stage_2_model(
            conf, self._get_checkpoint_stage_2_path(infer_type)
        )

        frame_end: int = 0
        audio_driven: Optional[Tensor] = None

        if conf.infer_type.startswith("mfcc"):
            # MFCC features
            wav, sr = librosa.load(audio_path, sr=16000)
            input_values = mfcc(wav, sr)
            d_mfcc_feat = delta(input_values, 1)
            d_mfcc_feat2 = delta(input_values, 2)
            audio_driven_obj: np.ndarray = np.hstack(
                (input_values, d_mfcc_feat, d_mfcc_feat2)
            )
            frame_start: int = 0
            frame_end: int = int(audio_driven_obj.shape[0] / 4)
            # The video frame is fixed to 25 hz, and the audio is fixed to 100 hz.
            audio_start: int = int(frame_start * 4)
            audio_end: int = int(frame_end * 4)
            audio_driven: Tensor = (
                Tensor(audio_driven_obj[audio_start:audio_end, :])
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
            if not Path(hubert_model_path).exists():
                logger.exception(
                    "Please download the hubert weight into the ckpts path first."
                )
                exit(0)
            logger.info(
                "You did not extract the audio features in advance, "
                + "extracting online now, which will increase processing delay"
            )

            start_time = time()

            audio_model = HubertModel.from_pretrained(hubert_model_path).to("cuda")
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                hubert_model_path
            )
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
        noisy_t = torch.randn((1, frame_end, self.settings.model.motion_dim)).to("cuda")

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
                wav_pred, self.settings.directory.frames / f"{pred_index:06d}.png"
            )
        # ==============================================

        execution_time = time() - start_time
        logger.info(f"Renderer Model: {execution_time:.2f} Seconds")
        logger.info(f"Saving video at {predicted_video_256_path}")

        frames_to_video(
            self.settings.directory.frames, audio_path, predicted_video_256_path
        )

        remove_frames(self.settings.directory.frames)

        # Enhancer
        if face_sr and check_package_installed("gfpgan"):
            # Super-resolution
            mimsave(
                predicted_video_512_path / self.settings.directory.tmp_extension,
                enhancer_list(predicted_video_256_path, bg_upsampler=None),
                fps=25.0,
            )
            # Merge audio and video
            video_clip = VideoFileClip(
                predicted_video_512_path / self.settings.directory.tmp_extension
            )
            audio_clip = AudioFileClip(predicted_video_256_path)
            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(
                predicted_video_512_path, codec="libx264", audio_codec="aac"
            )
            remove(predicted_video_512_path / self.settings.directory.tmp_extension)
        if not path.exists(predicted_video_256_path):
            return (
                None,
                None,
                Markdown(
                    "Error: Video generation failed. "
                    + "Please check your inputs and try again."
                ),
            )
        if face_sr:
            return (
                Video(value=predicted_video_256_path),
                Video(value=predicted_video_512_path),
                Markdown("Video generated successfully!"),
            )
        return (
            Video(value=predicted_video_256_path),
            None,
            Markdown("Video (256*256 only) generated successfully!"),
        )

    def _load_stage_1_model(self) -> LIA_Model:
        logger.info("Loading stage 1 model")
        lia: LIA_Model = LIA_Model(
            motion_dim=self.settings.model.motion_dim, fusion_type="weighted_sum"
        )
        lia.load_lightning_model(self.settings.model.checkpoint.stage_1)
        lia.to("cuda")
        return lia

    def _load_stage_2_model(
        self,
        conf: TrainConfig,
        stage2_checkpoint_path: str,
    ) -> LitModel:
        logger.info("Loading stage 2 model")
        model = LitModel(conf)
        state = torch.load(stage2_checkpoint_path, "cpu")
        model.load_state_dict(state)
        model.ema_model.eval()
        model.ema_model.to("cuda")
        return model

    def _init_conf(
        self,
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
        conf.motion_dim = self.settings.model.motion_dim
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

    def _get_checkpoint_stage_2_path(
        self,
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
                return self.settings.model.checkpoint.mfcc_full_control
            case "mfcc_pose_only":
                return self.settings.model.checkpoint.mfcc_pose_only
            case "hubert_pose_only":
                return self.settings.model.checkpoint.hubert_pose_only
            case "hubert_audio_only":
                return self.settings.model.checkpoint.hubert_audio_only
            case "hubert_full_control":
                return self.settings.model.checkpoint.hubert_full_control
