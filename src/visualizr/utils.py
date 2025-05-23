import argparse
import os
import shutil
import sys
import time
from argparse import Namespace
from importlib.util import find_spec

import librosa
import numpy as np
import python_speech_features
import torch
from gradio import Markdown, Video
from moviepy.editor import concatenate_videoclips  # type: ignore
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    VideoFileClip,
)
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.VideoClip import VideoClip
from numpy import asarray, float32, transpose  # type: ignore
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm

from visualizr import model_mapping
from visualizr.config import TrainConfig
from visualizr.experiment import LitModel
from visualizr.LIA_Model import LIA_Model
from visualizr.templates import ffhq256_autoenc


def check_package_installed(package_name: str) -> bool:
    if find_spec(name=package_name) is None:
        print(f"{package_name} is not installed.")
        return False
    else:
        print(f"{package_name} is installed.")
        return True


def frames_to_video(
    input_path: str, audio_path: str, output_path: str, fps: int = 25
) -> None:
    image_files: list[str] = [
        os.path.join(input_path, img) for img in sorted(os.listdir(path=input_path))
    ]
    clips = [ImageClip(img=m).set_duration(1 / fps) for m in image_files]
    video: VideoClip | CompositeVideoClip = concatenate_videoclips(
        clips=clips, method="compose"
    )

    audio = AudioFileClip(filename=audio_path)
    final_video = video.set_audio(audio)
    final_video.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
    )


def load_image(filename: str, size: int) -> NDArray[float32]:
    img: Image.Image = Image.open(fp=filename).convert(mode="RGB")
    img_resized: Image.Image = img.resize(size=(size, size))
    img_np: NDArray[float32] = asarray(a=img_resized)
    img_transposed: NDArray[float32] = transpose(
        a=img_np, axes=(2, 0, 1)
    )  # 3 x 256 x 256
    return img_transposed / 255.0


def img_preprocessing(img_path: str, size: int) -> Tensor:
    img_np: NDArray[float32] = load_image(filename=img_path, size=size)  # [0, 1]
    img: Tensor = torch.from_numpy(ndarray=img_np).unsqueeze(dim=0).float()  # [0, 1]
    imgs_norm: Tensor = (img - 0.5) * 2.0  # [-1, 1]
    return imgs_norm


def saved_image(img_tensor: Tensor, img_path: str) -> None:
    to_pil = transforms.ToPILImage()
    img = to_pil(pic=img_tensor.detach().cpu().squeeze(0))
    img.save(img_path)


def main(args: Namespace):
    frames_result_saved_path: str = os.path.join(args.result_path, "frames")
    os.makedirs(name=frames_result_saved_path, exist_ok=True)
    test_image_name: str = os.path.splitext(p=os.path.basename(p=args.test_image_path))[
        0
    ]
    audio_name: str = os.path.splitext(p=os.path.basename(p=args.test_audio_path))[0]
    predicted_video_256_path: str = os.path.join(
        args.result_path, f"{test_image_name}-{audio_name}.mp4"
    )
    predicted_video_512_path: str = os.path.join(
        args.result_path, f"{test_image_name}-{audio_name}_SR.mp4"
    )

    # ======Loading Stage 1 model=========
    lia = LIA_Model(motion_dim=args.motion_dim, fusion_type="weighted_sum")
    lia.load_lightning_model(lia_pretrained_model_path=args.stage1_checkpoint_path)
    lia.to(device="cuda")
    # ============================

    conf: TrainConfig = ffhq256_autoenc()
    conf.seed = args.seed
    conf.decoder_layers = args.decoder_layers
    conf.infer_type = args.infer_type
    conf.motion_dim = args.motion_dim

    if args.infer_type == "mfcc_full_control":
        conf.face_location = True
        conf.face_scale = True
        conf.mfcc = True
    elif args.infer_type == "mfcc_pose_only":
        conf.face_location = False
        conf.face_scale = False
        conf.mfcc = True
    elif args.infer_type == "hubert_pose_only":
        conf.face_location = False
        conf.face_scale = False
        conf.mfcc = False
    elif args.infer_type == "hubert_audio_only":
        conf.face_location = False
        conf.face_scale = False
        conf.mfcc = False
    elif args.infer_type == "hubert_full_control":
        conf.face_location = True
        conf.face_scale = True
        conf.mfcc = False
    else:
        print("Type NOT Found!")
        sys.exit(0)

    if not os.path.exists(args.test_image_path):
        print(f"{args.test_image_path} does not exist!")
        sys.exit(0)

    if not os.path.exists(args.test_audio_path):
        print(f"{args.test_audio_path} does not exist!")
        sys.exit(0)

    img_source = img_preprocessing(args.test_image_path, args.image_size).to("cuda")
    one_shot_lia_start, one_shot_lia_direction, feats = lia.get_start_direction_code(
        img_source, img_source, img_source, img_source
    )

    # ======Loading Stage 2 model=========
    model: LitModel = LitModel(conf)
    state = torch.load(args.stage2_checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.ema_model.eval()
    model.ema_model.to("cuda")
    # =================================

    # ======Audio Input=========
    if conf.infer_type.startswith("mfcc"):
        # MFCC features
        wav, sr = librosa.load(path=args.test_audio_path, sr=16000)
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
        if not os.path.exists(args.test_hubert_path):
            if not check_package_installed("transformers"):
                print("Please install transformers module first.")
                sys.exit(0)
            hubert_model_path: str = "ckpt/chinese-hubert-large"
            if not os.path.exists(hubert_model_path):
                print("Please download the hubert weight into the ckpts path first.")
                sys.exit(0)
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
            audio, sr = librosa.load(path=args.test_audio_path, sr=16000)
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
            print(f"Using audio feature from path: {args.test_hubert_path}")
            audio_driven_obj = np.load(file=args.test_hubert_path)

        frame_start, frame_end = 0, int(audio_driven_obj.shape[1] / 2)
        audio_start, audio_end = (
            int(frame_start * 2),
            int(frame_end * 2),
        )  # The video frame is fixed to 25 hz and the audio is fixed to 50 hz

        audio_driven: Tensor = (
            torch.Tensor(audio_driven_obj[:, audio_start:audio_end, :])
            .unsqueeze(dim=0)
            .float()
            .to(device="cuda")
        )
    # ============================

    # Diffusion Noise
    noisyT: Tensor = torch.randn((1, frame_end, args.motion_dim)).to("cuda")

    # ======Inputs for Attribute Control=========
    if os.path.exists(args.pose_driven_path):
        pose_obj = np.load(args.pose_driven_path)

        if len(pose_obj.shape) != 2:
            print("please check your pose information. The shape must be like (T, 3).")
            sys.exit(0)
        if pose_obj.shape[1] != 3:
            print("please check your pose information. The shape must be like (T, 3).")
            sys.exit(0)

        if pose_obj.shape[0] >= frame_end:
            pose_obj = pose_obj[:frame_end, :]
        else:
            padding = np.tile(
                A=pose_obj[-1, :], reps=(frame_end - pose_obj.shape[0], 1)
            )
            pose_obj = np.vstack(tup=(pose_obj, padding))

        pose_signal = (
            torch.Tensor(pose_obj).unsqueeze(dim=0).to(device="cuda") / 90
        )  # 90 is for normalization here
    else:
        yaw_signal: Tensor = (
            torch.zeros(1, frame_end, 1).to(device="cuda") + args.pose_yaw
        )
        pitch_signal: Tensor = (
            torch.zeros(1, frame_end, 1).to(device="cuda") + args.pose_pitch
        )
        roll_signal: Tensor = (
            torch.zeros(1, frame_end, 1).to(device="cuda") + args.pose_roll
        )
        pose_signal = torch.cat(tensors=(yaw_signal, pitch_signal, roll_signal), dim=-1)

    pose_signal: Tensor = torch.clamp(input=pose_signal, min=-1, max=1)

    face_location_signal: Tensor = (
        torch.zeros(1, frame_end, 1).to(device="cuda") + args.face_location
    )
    face_scae_signal: Tensor = (
        torch.zeros(1, frame_end, 1).to(device="cuda") + args.face_scale
    )
    # ===========================================

    start_time: float = time.time()

    # ======Diffusion Denosing Process=========
    generated_directions = model.render(
        start=one_shot_lia_start,
        motion_direction_start=one_shot_lia_direction,
        audio_driven=audio_driven,
        face_location=face_location_signal,
        face_scale=face_scae_signal,
        ypr_info=pose_signal,
        noisyT=noisyT,
        step_T=args.step_T,
        control_flag=args.control_flag,
    )
    # =========================================

    execution_time: float = time.time() - start_time
    print(f"Motion Diffusion Model: {execution_time:.2f} Seconds")

    generated_directions: NDArray = generated_directions.detach().cpu().numpy()

    start_time = time.time()
    # ======Rendering images frame-by-frame=========
    for pred_index in tqdm(iterable=range(generated_directions.shape[1])):
        ori_img_recon = lia.render(
            start=one_shot_lia_start,
            direction=torch.Tensor(generated_directions[:, pred_index, :]).to(
                device="cuda"
            ),
            feats=feats,
        )
        ori_img_recon = ori_img_recon.clamp(-1, 1)
        wav_pred = (ori_img_recon.detach() + 1) / 2
        saved_image(
            wav_pred, os.path.join(frames_result_saved_path, "%06d.png" % (pred_index))
        )
    # ==============================================

    execution_time = time.time() - start_time
    print(f"Renderer Model: {execution_time:.2f} Seconds")

    frames_to_video(
        input_path=frames_result_saved_path,
        audio_path=args.test_audio_path,
        output_path=predicted_video_256_path,
    )

    shutil.rmtree(path=frames_result_saved_path)

    # Enhancer
    if args.face_sr and check_package_installed(package_name="gfpgan"):
        import imageio

        from visualizr.face_sr.face_enhancer import enhancer_list

        # Super-resolution
        imageio.mimsave(
            uri=predicted_video_512_path + ".tmp.mp4",
            ims=enhancer_list(
                images=predicted_video_256_path, method="gfpgan", bg_upsampler=None
            ),
            fps=float(25),
        )

        # Merge audio and video
        video_clip: VideoFileClip = VideoFileClip(predicted_video_512_path + ".tmp.mp4")
        audio_clip: AudioFileClip = AudioFileClip(predicted_video_256_path)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(
            predicted_video_512_path, codec="libx264", audio_codec="aac"
        )

        os.remove(predicted_video_512_path + ".tmp.mp4")

    if args.face_sr:
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
    step_T: int,
    face_sr: bool,
    seed: int,
) -> (
    tuple[None, Markdown]
    | tuple[Video, None, Markdown]
    | tuple[Video, Video, Markdown]
    | tuple[None, None, Markdown]
):
    if not uploaded_img or not uploaded_audio:
        return None, Markdown(
            value="Error: Input image or audio file is empty. Please check and upload both files."
        )
    stage2_checkpoint_path: str = model_mapping.get(
        infer_type, "default_checkpoint.ckpt"
    )
    try:
        args: Namespace = argparse.Namespace(
            infer_type=infer_type,
            test_image_path=uploaded_img,
            test_audio_path=uploaded_audio,
            test_hubert_path="",
            result_path="./outputs/",
            stage1_checkpoint_path="ckpt/stage1.ckpt",
            stage2_checkpoint_path=stage2_checkpoint_path,
            seed=seed,
            control_flag=True,
            pose_yaw=pose_yaw,
            pose_pitch=pose_pitch,
            pose_roll=pose_roll,
            face_location=face_location,
            pose_driven_path="not_supported_in_this_mode",
            face_scale=face_scale,
            step_T=step_T,
            image_size=256,
            device="cuda",
            motion_dim=20,
            decoder_layers=2,
            face_sr=face_sr,
        )
        video_path: tuple[str, str] = main(args=args)
        output_256_video_path: str = video_path[0]
        output_512_video_path: str = video_path[1]

        if not os.path.exists(path=output_256_video_path):
            return (
                None,
                Markdown(
                    value="Error: Video generation failed. Please check your inputs and try again."
                ),
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
