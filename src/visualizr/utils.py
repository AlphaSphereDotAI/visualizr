import argparse
import shutil
import time
from argparse import Namespace
from importlib.util import find_spec
from os import listdir, makedirs, path, remove
from pathlib import Path

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
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.VideoClip import VideoClip
from numpy import ndarray
from PIL import Image
from PIL.ImageFile import ImageFile
from torch import Tensor
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from visualizr import FRAMES_RESULT_SAVED_PATH, RESULTS_DIR, model_mapping
from visualizr.experiment import LitModel
from visualizr.LIA_Model import LIA_Model
from visualizr.templates import ffhq256_autoenc


def check_package_installed(package_name: str) -> bool:
    return find_spec(package_name) is not None


def frames_to_video(
    input_path: str, audio_path: str, output_path: str, fps: int = 25
) -> None:
    image_files: list[str] = [
        path.join(input_path, img) for img in sorted(listdir(path=input_path))
    ]
    clips = [ImageClip(img=m).set_duration(1 / fps) for m in image_files]
    video: VideoClip | CompositeVideoClip = concatenate_videoclips(
        clips=clips, method="compose"
    )
    audio: AudioFileClip = AudioFileClip(audio_path)
    video.set_audio(audio)
    video.write_videofile(output_path, fps, "libx264", "aac")


def load_image(filename: str, size: int) -> np.ndarray:
    img: ImageFile = Image.open(filename).convert("RGB")
    img_resized: ImageFile = img.resize((size, size))
    img_np: ndarray = np.asarray(img_resized)
    img_transposed: ndarray = np.transpose(img_np, (2, 0, 1))  # 3 x 256 x 256
    return img_transposed / 255.0


def img_preprocessing(img_path: str, size: int) -> Tensor:
    img_np: ndarray = load_image(img_path, size)  # [0, 1]
    img: Tensor = torch.from_numpy(img_np).unsqueeze(0).float()  # [0, 1]
    normalized_image: Tensor = (img - 0.5) * 2.0  # [-1, 1]
    return normalized_image


def saved_image(img_tensor: Tensor, img_path: str) -> None:
    pil_image_converter: ToPILImage = ToPILImage()
    img = pil_image_converter(img_tensor.detach().cpu().squeeze(0))
    img.save(img_path)

def load_stage1_model(motion_dim: int, stage1_checkpoint_path: str) -> LIA_Model:
    lia: LIA_Model = LIA_Model(motion_dim=motion_dim, fusion_type="weighted_sum")
    lia.load_lightning_model(stage1_checkpoint_path)
    lia.to("cuda")
    return lia

def main(infer_type,
         image_path,
         test_audio_path,
         test_hubert_path="",
         stage1_checkpoint_path="ckpts/stage1.ckpt",
         stage2_checkpoint_path=model_mapping.get(
                infer_type, "default_checkpoint.ckpt"
            ),
         seed,
         control_flag=True,
         pose_yaw: float,
         pose_pitch,
         pose_roll,
         face_location,
         face_scale,
         step_T,
         image_size=256,
         device="cuda",
         motion_dim=20,
         decoder_layers=2,
         face_sr):
    
    image_name: str = Path(image_path).stem
    audio_name: str = Path(test_audio_path).stem
    predicted_video_256_path: Path = RESULTS_DIR/ f"{image_name}-{audio_name}.mp4"
    predicted_video_512_path: Path = RESULTS_DIR/ f"{image_name}-{audio_name}_SR.mp4"

    # ======Loading Stage 1 model=========
    lia: LIA_Model = load_stage1_model(motion_dim, stage1_checkpoint_path)
    # ============================

    conf = ffhq256_autoenc()
    conf.seed = seed
    conf.decoder_layers = decoder_layers
    conf.infer_type = infer_type
    conf.motion_dim = motion_dim

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
        case _:
            print("Type NOT Found!")
            exit(0)

    if not path.exists(image_path):
        print(f"{image_path} does not exist!")
        exit(0)

    if not path.exists(test_audio_path):
        print(f"{test_audio_path} does not exist!")
        exit(0)

    img_source = img_preprocessing(image_path, image_size).to("cuda")
    one_shot_lia_start, one_shot_lia_direction, feats = lia.get_start_direction_code(
        img_source, img_source, img_source, img_source
    )

    # ======Loading Stage 2 model=========
    model = LitModel(conf)
    state = torch.load(stage2_checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.ema_model.eval()
    model.ema_model.to("cuda")
    # =================================

    # ======Audio Input=========
    if conf.infer_type.startswith("mfcc"):
        # MFCC features
        wav, sr = librosa.load(test_audio_path, sr=16000)
        input_values = python_speech_features.mfcc(
            signal=wav, samplerate=sr, numcep=13, winlen=0.025, winstep=0.01
        )
        d_mfcc_feat = python_speech_features.base.delta(input_values, 1)
        d_mfcc_feat2 = python_speech_features.base.delta(input_values, 2)
        audio_driven_obj = np.hstack((input_values, d_mfcc_feat, d_mfcc_feat2))
        frame_start, frame_end = 0, int(audio_driven_obj.shape[0] / 4)
        audio_start, audio_end = (
            int(frame_start * 4),
            int(frame_end * 4),
        )  # The video frame is fixed to 25 hz and the audio is fixed to 100 hz

        audio_driven = (
            torch.Tensor(audio_driven_obj[audio_start:audio_end, :])
            .unsqueeze(0)
            .float()
            .to("cuda")
        )

    elif conf.infer_type.startswith("hubert"):
        # Hubert features
        if not path.exists(test_hubert_path):
            if not check_package_installed("transformers"):
                print("Please install transformers module first.")
                exit(0)
            hubert_model_path = "ckpts/chinese-hubert-large"
            if not path.exists(hubert_model_path):
                print("Please download the hubert weight into the ckpts path first.")
                exit(0)
            print(
                "You did not extract the audio features in advance, extracting online now, which will increase processing delay"
            )

            start_time = time.time()

            # load hubert model
            from transformers import HubertModel, Wav2Vec2FeatureExtractor

            audio_model = HubertModel.from_pretrained(hubert_model_path).to("cuda")
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                hubert_model_path
            )
            audio_model.feature_extractor._freeze_parameters()
            audio_model.eval()

            # hubert model forward pass
            audio, sr = librosa.load(test_audio_path, sr=16000)
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
                )  # align the audio length with video frame

            execution_time = time.time() - start_time
            print(f"Extraction Audio Feature: {execution_time:.2f} Seconds")

            audio_driven_obj = ws_feat_obj
        else:
            print(f"Using audio feature from path: {test_hubert_path}")
            audio_driven_obj = np.load(test_hubert_path)

        frame_start, frame_end = 0, int(audio_driven_obj.shape[1] / 2)
        audio_start, audio_end = (
            int(frame_start * 2),
            int(frame_end * 2),
        )  # The video frame is fixed to 25 hz and the audio is fixed to 50 hz

        audio_driven = (
            torch.Tensor(audio_driven_obj[:, audio_start:audio_end, :])
            .unsqueeze(0)
            .float()
            .to("cuda")
        )
    # ============================

    # Diffusion Noise
    noisyT = torch.randn((1, frame_end, motion_dim)).to("cuda")

    # ======Inputs for Attribute Control=========
    yaw_signal = torch.zeros(1, frame_end, 1).to("cuda") + pose_yaw
    pitch_signal = torch.zeros(1, frame_end, 1).to("cuda") + pose_pitch
    roll_signal = torch.zeros(1, frame_end, 1).to("cuda") + pose_roll
    pose_signal = torch.cat((yaw_signal, pitch_signal, roll_signal), dim=-1)

    pose_signal = torch.clamp(pose_signal, -1, 1)

    face_location_signal = torch.zeros(1, frame_end, 1).to("cuda") + face_location
    face_scae_signal = torch.zeros(1, frame_end, 1).to("cuda") + face_scale
    # ===========================================

    start_time = time.time()

    # ======Diffusion De-nosing Process=========
    generated_directions = model.render(
        one_shot_lia_start,
        one_shot_lia_direction,
        audio_driven,
        face_location_signal,
        face_scae_signal,
        pose_signal,
        noisyT,
        step_T,
        control_flag=control_flag,
    )
    # =========================================

    execution_time = time.time() - start_time
    print(f"Motion Diffusion Model: {execution_time:.2f} Seconds")

    generated_directions = generated_directions.detach().cpu().numpy()

    start_time = time.time()
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
            wav_pred, path.join(FRAMES_RESULT_SAVED_PATH, "%06d.png" % (pred_index))
        )
    # ==============================================

    execution_time = time.time() - start_time
    print(f"Renderer Model: {execution_time:.2f} Seconds")

    frames_to_video(
        FRAMES_RESULT_SAVED_PATH, test_audio_path, predicted_video_256_path
    )

    shutil.rmtree(FRAMES_RESULT_SAVED_PATH)

    # Enhancer
    if face_sr and check_package_installed("gfpgan"):
        from imageio import mimsave

        from visualizr.face_sr.face_enhancer import enhancer_list

        # Super-resolution
        mimsave(
            predicted_video_512_path + ".tmp.mp4",
            enhancer_list(predicted_video_256_path, method="gfpgan", bg_upsampler=None),
            fps=25.0,
        )

        # Merge audio and video
        video_clip = VideoFileClip(predicted_video_512_path + ".tmp.mp4")
        audio_clip = AudioFileClip(predicted_video_256_path)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(
            predicted_video_512_path, codec="libx264", audio_codec="aac"
        )

        remove(predicted_video_512_path + ".tmp.mp4")

    if face_sr:
        return predicted_video_256_path, predicted_video_512_path
    else:
        return predicted_video_256_path, predicted_video_256_path


def generate_video(
    uploaded_img: str,
    uploaded_audio: str,
    infer_type: str,
    pose_yaw: float,
    pose_pitch: float,
    pose_roll: float,
    face_location: float,
    face_scale: float,
    step_t: int,
    face_sr: bool,
    seed: int,
):
    if not uploaded_img or not uploaded_audio:
        return None, Markdown(
            "Error: Input image or audio file is empty. Please check and upload both files."
        )

    try:
        output_256_video_path, output_512_video_path = main(infer_type=infer_type, image_path=uploaded_img,
                                                            test_audio_path=uploaded_audio,
                                                            stage2_checkpoint_path=model_mapping.get(
                                                                infer_type, "default_checkpoint.ckpt"
                                                            ), seed=seed, pose_yaw=pose_yaw, pose_pitch=pose_pitch,
                                                            pose_roll=pose_roll, face_location=face_location,
                                                            face_scale=face_scale, step_T=step_t, face_sr=face_sr,)

        if not path.exists(path=output_256_video_path):
            return None, Markdown(
                value="Error: Video generation failed. Please check your inputs and try again."
            )
        if output_256_video_path == output_512_video_path:
            return (
                Video(value=output_256_video_path),
                None,
                Markdown(value="Video (256*256 only) generated successfully!"),
            )
        return (
            Video(value=output_256_video_path),
            Video(value=output_512_video_path),
            Markdown(value="Video generated successfully!"),
        )

    except Exception as e:
        return (
            None,
            None,
            Markdown(value=f"Error: An unexpected error occurred - {str(e)}"),
        )
