from pathlib import Path
from sys import exit as sys_exit
from time import time
from typing import Literal, Optional

from gradio import (
    Audio,
    Blocks,
    Button,
    Checkbox,
    Column,
    Dropdown,
    Error,
    Image,
    Info,
    Markdown,
    Number,
    Row,
    Slider,
    Tab,
    Video,
)
from huggingface_hub import snapshot_download
from librosa import load as librosa_load
from numpy import (
    array as np_array,
)
from numpy import (
    hstack as np_hstack,
)
from numpy import (
    ndarray,
)
from numpy import (
    pad as np_pad,
)
from numpy import (
    squeeze as np_squeeze,
)
from python_speech_features import mfcc
from python_speech_features.base import delta
from torch import (
    Tensor,
    no_grad,
)
from torch import (
    cat as torch_cat,
)
from torch import (
    clamp as torch_clamp,
)
from torch import (
    randn as torch_randn,
)
from torch import (
    zeros as torch_zeros,
)
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from visualizr.anitalker.config import TrainConfig
from visualizr.anitalker.LIA_Model import LIA_Model
from visualizr.anitalker.utils import (
    check_package_installed,
    frames_to_video,
    img_preprocessing,
    init_configuration,
    load_stage_2_model,
    remove_frames,
    saved_image,
    super_resolution,
)
from visualizr.settings import Settings, logger


class App:
    def __init__(self, settings: Settings):
        self.settings: Settings = settings
        snapshot_download(
            repo_id=settings.model.repo_id,
            local_dir=settings.directory.checkpoint,
            repo_type="model",
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
        if image_path is None or not Path(image_path).exists():
            logger.error(f"{image_path} does not exist or is invalid!")
            Error(f"{image_path} does not exist or is invalid!")
            return (
                None,
                None,
                Markdown(
                    f"Error: image_path '{image_path}' does not exist or is invalid."
                ),
            )
        if audio_path is None or not Path(audio_path).exists():
            logger.error(f"{audio_path} does not exist or is invalid!")
            Error(f"{audio_path} does not exist or is invalid!")
            return (
                None,
                None,
                Markdown(
                    f"Error: audio_path '{audio_path}' does not exist or is invalid."
                ),
            )

        predicted_video_256_path: Path = (
            self.settings.directory.results
            / f"{Path(image_path).stem}-{Path(audio_path).stem}.mp4"
        )
        predicted_video_512_path: Path = (
            self.settings.directory.results
            / f"{Path(image_path).stem}-{Path(audio_path).stem}_SR.mp4"
        )

        lia: LIA_Model = self._load_stage_1_model()

        conf: TrainConfig = init_configuration(
            infer_type, seed, 2, self.settings.model.motion_dim
        )

        img_source: Tensor = img_preprocessing(image_path, 256).to("cuda")
        one_shot_lia_start, one_shot_lia_direction, feats = (
            lia.get_start_direction_code(img_source, img_source, img_source, img_source)
        )

        model = load_stage_2_model(conf, self._get_checkpoint_stage_2_path(infer_type))

        frame_end: int = 0
        audio_driven: Optional[Tensor] = None

        if conf.infer_type.startswith("mfcc"):
            # MFCC features
            wav, sr = librosa_load(audio_path, sr=16000)
            input_values = mfcc(wav, sr)
            d_mfcc_feat = delta(input_values, 1)
            d_mfcc_feat2 = delta(input_values, 2)
            audio_driven_obj: ndarray = np_hstack(
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
                Error("Please install transformers module first.")
                sys_exit(0)
            hubert_model_path = "ckpts/chinese-hubert-large"
            if not Path(hubert_model_path).exists():
                logger.exception(
                    "Please download the hubert weight into the ckpts path first."
                )
                Error("Please download the hubert weight into the ckpts path first.")
                sys_exit(0)
            logger.info(
                "You did not extract the audio features in advance, "
                + "extracting online now, which will increase processing delay"
            )
            Info(
                "You did not extract the audio features in advance, "
                + "extracting online now, which will increase processing delay"
            )

            start_time = time()

            audio_model = HubertModel.from_pretrained(hubert_model_path).to("cuda")
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                hubert_model_path
            )
            audio_model.feature_extractor._freeze_parameters()
            audio_model.eval()

            # hubert model forward pass
            audio, sr = librosa_load(audio_path, sr=16000)
            input_values = feature_extractor(
                audio,
                sampling_rate=16000,
                padding=True,
                do_normalize=True,
                return_tensors="pt",
            ).input_values
            input_values = input_values.to("cuda")
            ws_feats = []
            with no_grad():
                outputs = audio_model(input_values, output_hidden_states=True)
                ws_feats.extend(
                    outputs.hidden_states[i].detach().cpu().numpy()
                    for i in range(len(outputs.hidden_states))
                )
                ws_feat_obj = np_array(ws_feats)
                ws_feat_obj = np_squeeze(ws_feat_obj, 1)
                # align the audio length with the video frame
                ws_feat_obj = np_pad(ws_feat_obj, ((0, 0), (0, 1), (0, 0)), "edge")

            execution_time = time() - start_time
            logger.info(f"Extraction Audio Feature: {execution_time:.2f} Seconds")
            Info(f"Extraction Audio Feature: {execution_time:.2f} Seconds")

            audio_driven_obj = ws_feat_obj

            frame_start, frame_end = 0, int(audio_driven_obj.shape[1] / 2)
            # The video frame is fixed to 25 hz, and the audio is fixed to 50 hz.
            audio_start, audio_end = (
                int(frame_start * 2),
                int(frame_end * 2),
            )

            audio_driven = (
                Tensor(audio_driven_obj[:, audio_start:audio_end, :])
                .unsqueeze(0)
                .float()
                .to("cuda")
            )

        # Diffusion Noise
        noisy_t = torch_randn((1, frame_end, self.settings.model.motion_dim)).to("cuda")

        # ======Inputs for Attribute Control=========
        yaw_signal = torch_zeros(1, frame_end, 1).to("cuda") + pose_yaw
        pitch_signal = torch_zeros(1, frame_end, 1).to("cuda") + pose_pitch
        roll_signal = torch_zeros(1, frame_end, 1).to("cuda") + pose_roll
        pose_signal = torch_cat((yaw_signal, pitch_signal, roll_signal), dim=-1)

        pose_signal = torch_clamp(pose_signal, -1, 1)

        face_location_signal = torch_zeros(1, frame_end, 1).to("cuda") + face_location
        face_scale_tensor = torch_zeros(1, frame_end, 1).to("cuda") + face_scale
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
        Info(f"Motion Diffusion Model: {execution_time:.2f} Seconds")

        generated_directions = generated_directions.detach().cpu().numpy()

        start_time = time()
        # ======Rendering images frame-by-frame=========
        for pred_index in tqdm(range(generated_directions.shape[1])):
            ori_img_recon = lia.render(
                one_shot_lia_start,
                Tensor(generated_directions[:, pred_index, :]).to("cuda"),
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
        Info(f"Renderer Model: {execution_time:.2f} Seconds")
        logger.info(f"Saving video at {predicted_video_256_path}")
        Info(f"Saving video at {predicted_video_256_path}")

        frames_to_video(
            self.settings.directory.frames, audio_path, predicted_video_256_path
        )

        remove_frames(self.settings.directory.frames)

        # Enhancer
        if face_sr and check_package_installed("gfpgan"):
            # Super-resolution
            super_resolution(
                predicted_video_512_path / self.settings.directory.tmp_extension,
                predicted_video_256_path,
                predicted_video_512_path,
            )
        if not predicted_video_256_path.exists():
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
        Info("Loading stage 1 model")
        lia: LIA_Model = LIA_Model(
            motion_dim=self.settings.model.motion_dim, fusion_type="weighted_sum"
        )
        lia.load_lightning_model(self.settings.model.checkpoint.stage_1)
        lia.to("cuda")
        return lia

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

    def gui(self) -> Blocks:
        """Create the Gradio interface for the voice generation web app."""
        with Blocks() as app:
            with Tab("AniTalker"):
                with Column():
                    with Row():
                        with Column():
                            image_path: Image = Image(
                                value=self.settings.model.image_path.as_posix(),
                                type="filepath",
                                label="Reference Image",
                            )
                            audio_path = Audio(
                                value=self.settings.model.audio_path.as_posix(),
                                type="filepath",
                                label="Input Audio",
                                show_download_button=True,
                            )
                        with Column():
                            output_video_256 = Video(label="Generated Video (256)")
                            output_video_512 = Video(label="Generated Video (512)")
                            output_message = Markdown()
                    with Row():
                        generate_button = Button(value="Generate", variant="primary")
                        stop_button: Button = Button(value="Stop", variant="stop")
            with Tab("Configuration"):
                infer_type = Dropdown(
                    label="Inference Type",
                    choices=[
                        "mfcc_full_control",
                        "mfcc_pose_only",
                        "hubert_pose_only",
                        "hubert_audio_only",
                        "hubert_full_control",
                    ],
                    value="hubert_audio_only",
                )
                face_sr = Checkbox(label="Enable Face Super-Resolution (512*512)")
                seed = Number(
                    label="Seed",
                    value=self.settings.model.seed,
                )
                pose_yaw = Slider(
                    label="pose_yaw",
                    minimum=-1,
                    maximum=1,
                    value=self.settings.model.pose_yaw,
                )
                pose_pitch = Slider(
                    label="pose_pitch",
                    minimum=-1,
                    maximum=1,
                    value=self.settings.model.pose_pitch,
                )
                pose_roll = Slider(
                    label="pose_roll",
                    minimum=-1,
                    maximum=1,
                    value=self.settings.model.pose_roll,
                )
                face_location = Slider(
                    label="face_location",
                    maximum=1,
                    value=self.settings.model.face_location,
                )
                face_scale = Slider(
                    label="face_scale",
                    maximum=1,
                    value=self.settings.model.face_scale,
                )
                step_t = Slider(
                    label="step_T",
                    minimum=1,
                    step=1,
                    value=self.settings.model.step_t,
                )
            generate_button = generate_button.click(
                self.generate_video,
                [
                    infer_type,
                    image_path,
                    audio_path,
                    face_sr,
                    pose_yaw,
                    pose_pitch,
                    pose_roll,
                    face_location,
                    face_scale,
                    step_t,
                    seed,
                ],
                [
                    output_video_256,
                    output_video_512,
                    output_message,
                ],
            )
            stop_button.click(cancels=generate_button)
            return app
