from pathlib import Path
from sys import exit as sys_exit
from time import time
from typing import Literal

from gradio import (
    Audio,
    Blocks,
    Button,
    Checkbox,
    Column,
    Dropdown,
    Error,
    Group,
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
    hstack as np_hstack,
    ndarray,
    pad as np_pad,
    squeeze as np_squeeze,
)
from python_speech_features import delta, mfcc
from torch import (
    Tensor,
    cat as torch_cat,
    clamp as torch_clamp,
    no_grad,
    randn as torch_randn,
    zeros as torch_zeros,
)
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from visualizr.anitalker.config import TrainConfig
from visualizr.anitalker.liamodel import LiaModel
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
from visualizr.settings import Settings


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
        image_path: str | Path,
        audio_path: str | Path,
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
            Error(f"{image_path} does not exist or is invalid!")
            return (
                None,
                None,
                Markdown(
                    f"Error: image_path '{image_path}' does not exist or is invalid.",
                ),
            )
        if audio_path is None or not Path(audio_path).exists():
            Error(f"{audio_path} does not exist or is invalid!")
            return (
                None,
                None,
                Markdown(
                    f"Error: audio_path '{audio_path}' does not exist or is invalid.",
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

        lia: LiaModel = self._load_stage_1_model()

        conf: TrainConfig = init_configuration(
            infer_type,
            seed,
            2,
            self.settings.model.motion_dim,
        )

        img_source: Tensor = img_preprocessing(image_path, 256).to("cuda")
        one_shot_lia_start, one_shot_lia_direction, feats = (
            lia.get_start_direction_code(
                img_source,
                img_source,
                img_source,
                img_source,
            )
        )

        model = load_stage_2_model(
            conf,
            self._get_checkpoint_stage_2_path(infer_type),
        )

        frame_end: int = 0
        audio_driven: Tensor | None = None

        if conf.infer_type.startswith("mfcc"):
            # MFCC features
            wav, sr = librosa_load(audio_path, sr=16000)
            input_values = mfcc(wav, sr)
            d_mfcc_feat = delta(input_values, 1)
            d_mfcc_feat2 = delta(input_values, 2)
            audio_driven_obj: ndarray = np_hstack(
                (input_values, d_mfcc_feat, d_mfcc_feat2),
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
                Error("Please install transformers module first.")
                sys_exit(0)
            hubert_model_path = "ckpts/chinese-hubert-large"
            if not Path(hubert_model_path).exists():
                Error("Please download the hubert weight into the ckpts path first.")
                sys_exit(0)
            Info(
                (
                    "You did not extract the audio features in advance, "
                    "extracting online now, which will increase processing delay"
                ),
            )

            start_time = time()

            audio_model = HubertModel.from_pretrained(hubert_model_path).to("cuda")
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                hubert_model_path,
            )
            audio_model.feature_extractor._freeze_parameters()  # noqa: SLF001
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
                wav_pred,
                self.settings.directory.frames / f"{pred_index:06d}.png",
            )
        # ==============================================

        execution_time = time() - start_time
        Info(f"Renderer Model: {execution_time:.2f} Seconds")
        Info(f"Saving video at {predicted_video_256_path}")

        frames_to_video(
            self.settings.directory.frames,
            audio_path,
            predicted_video_256_path,
        )

        remove_frames(self.settings.directory.frames)

        # Enhancer
        if face_sr and check_package_installed("gfpgan"):
            # Super-resolution
            super_resolution(
                predicted_video_512_path.with_suffix(".tmp.mp4"),
                predicted_video_256_path,
                predicted_video_512_path,
            )
        if not predicted_video_256_path.exists():
            return (
                None,
                None,
                Markdown(
                    (
                        "Error: Video generation failed. "
                        "Please check your inputs and try again."
                    ),
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
            Markdown("Video (256 ✕ 256 only) generated successfully!"),
        )

    def generate_video_from_name(
        self,
        name: str,
        infer_type: Literal[
            "mfcc_full_control",
            "mfcc_pose_only",
            "hubert_pose_only",
            "hubert_audio_only",
            "hubert_full_control",
        ],
        audio_path: str | Path,
        face_sr: bool,
        pose_yaw: float,
        pose_pitch: float,
        pose_roll: float,
        face_location: float,
        face_scale: float,
        step_t: int,
        seed: int,
    ) -> tuple[Video | None, Video | None, Markdown]:
        """
        Generate a video for a character by name using the provided settings and audio.

        Args:
            name (str): The base name of the character image (without extension).
            infer_type (Literal[
                'mfcc_full_control',
                'mfcc_pose_only',
                'hubert_pose_only',
                'hubert_audio_only',
                'hubert_full_control',
            ]): The type of inference mode.
            audio_path (str | Path): Path to the input audio file.
            face_sr (bool): Whether to apply a face super-resolution.
            pose_yaw (float): Yaw angle for the character's pose.
            pose_pitch (float): Pitch angle for the character's pose.
            pose_roll (float): Roll angle for the character's pose.
            face_location (float): Relative location parameter for a face positioning.
            face_scale (float): Scaling factor for the face.
            step_t (int): Number of diffusion steps.
            seed (int): Random seed for reproducibility.

        Returns:
            tuple[Video | None, Video | None, Markdown]: A tuple
                        containing the generated 256x256 video,
                        the high-resolution video (if `face_sr` is True),
                        and a Markdown status message.
        """
        return self.generate_video(
            infer_type,
            self._get_image_path(name),
            audio_path,
            face_sr,
            pose_yaw,
            pose_pitch,
            pose_roll,
            face_location,
            face_scale,
            step_t,
            seed,
        )

    def _get_image_path(self, name: str) -> Path:
        """
        Retrieve the image path for a given character name by checking
        supported extensions.

        Args:
            name (str): The base name of the image file (without extension).

        Returns:
            Path: The path to the existing image file.
                  Defaults to .jpg if none is found.
        """
        for ext in (".jpg", ".jpeg", ".png"):
            path = self.settings.directory.image / f"{name}{ext}"
            if path.is_file():
                return path
        return self.settings.directory.image / f"{name}.jpg"

    def _get_character_names(self) -> list[str]:
        """
        List all character names available in the image directory.

        Returns:
            list[str]: Sorted list of unique character names (file stems)
                       from supported image files.
        """
        extensions = ("*.jpg", "*.jpeg", "*.png")
        paths = (
            p for ext in extensions for p in self.settings.directory.image.glob(ext)
        )
        # Use a set to handle cases where an image exists with multiple supported
        # extensions (for example, napoleon.jpg, napoleon.png)
        return sorted({p.stem for p in paths})

    def _load_stage_1_model(self) -> LiaModel:
        Info("Loading stage 1 model")
        lia: LiaModel = LiaModel(
            motion_dim=self.settings.model.motion_dim,
            fusion_type="weighted_sum",
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
            with Tab("AniTalker (Generate Video from Paths)"):
                with Row():
                    with Column():
                        image_path: Image = Image(
                            value=(
                                self.settings.model.image_path.as_posix()
                                if self.settings.model.image_path
                                else None
                            ),
                            type="filepath",
                            label="Reference Image",
                        )
                        audio_path = Audio(
                            value=(
                                self.settings.model.audio_path.as_posix()
                                if self.settings.model.audio_path
                                else None
                            ),
                            type="filepath",
                            label="Input Audio",
                            show_download_button=True,
                        )
                    with Column():
                        output_video_256 = Video(label="Generated Video (256)")
                        output_video_512 = Video(label="Generated Video (512)")
                        output_message = Markdown()
                with Row():
                    generate_button = Button("Generate", variant="primary")
                    stop_button: Button = Button("Stop", variant="stop")
            with Tab("AniTalker (Generate Video from Name)"):
                with Row():
                    with Column():
                        name = Dropdown(
                            self._get_character_names(),
                            label="Character",
                            info=(
                                "Choose character, More characters will be added later."
                            ),
                        )
                    with Column():
                        output_video_256_from_name = Video(
                            label="Generated Video (256)",
                        )
                        output_video_512_from_name = Video(
                            label="Generated Video (512)",
                        )
                        output_message_from_name = Markdown()
                with Row():
                    generate_from_name_button = Button("Generate", variant="primary")
                    stop_from_name_button: Button = Button("Stop", variant="stop")
            with Tab("Configuration"):
                with Row():
                    infer_type = Dropdown(
                        [
                            "mfcc_full_control",
                            "mfcc_pose_only",
                            "hubert_pose_only",
                            "hubert_audio_only",
                            "hubert_full_control",
                        ],
                        value="hubert_audio_only",
                        label="Inference Type",
                    )
                    seed = Number(self.settings.model.seed, label="Seed")
                    face_sr = Checkbox(label="Enable Face Super-Resolution (512*512)")
                with Row():
                    with Group(), Row():
                        pose_yaw = Slider(
                            -1,
                            1,
                            self.settings.model.pose_yaw,
                            label="pose_yaw",
                        )
                        pose_pitch = Slider(
                            -1,
                            1,
                            self.settings.model.pose_pitch,
                            label="pose_pitch",
                        )
                        pose_roll = Slider(
                            -1,
                            1,
                            self.settings.model.pose_roll,
                            label="pose_roll",
                        )
                    with Row():
                        face_location = Slider(
                            maximum=1,
                            value=self.settings.model.face_location,
                            label="face_location",
                        )
                        face_scale = Slider(
                            maximum=1,
                            value=self.settings.model.face_scale,
                            label="face_scale",
                        )
                        step_t = Slider(
                            minimum=1,
                            step=1,
                            value=self.settings.model.step_t,
                            label="step_T",
                        )
            generate_button_event = generate_button.click(
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
            stop_button.click(cancels=generate_button_event)
            generate_from_name_button_event = generate_from_name_button.click(
                self.generate_video_from_name,
                [
                    name,
                    infer_type,
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
                    output_video_256_from_name,
                    output_video_512_from_name,
                    output_message_from_name,
                ],
            )
            stop_from_name_button.click(cancels=generate_from_name_button_event)
            return app
