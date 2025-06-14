from gradio import (
    Accordion,
    Audio,
    Blocks,
    Button,
    Checkbox,
    Column,
    Dropdown,
    Image,
    Markdown,
    Number,
    Row,
    Slider,
    Video,
)

from visualizr.settings import DefaultValues
from visualizr.utils import generate_video


def app_block() -> Blocks:
    """Create the Gradio interface for the voice generation web application."""
    with Blocks() as app:
        Markdown(value="# AniTalker")
        with Row():
            with Column():
                uploaded_img: Image = Image(type="filepath", label="Reference Image")
                uploaded_audio = Audio(
                    type="filepath", label="Input Audio", show_download_button=True
                )
            with Column():
                output_video_256 = Video(label="Generated Video (256)")
                output_video_512 = Video(label="Generated Video (512)")
                output_message = Markdown()

        generate_button = Button(value="Generate Video")

        with Accordion(label="Configuration"):
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
            seed = Number(label="Seed", value=DefaultValues().seed)
            pose_yaw = Slider(
                label="pose_yaw",
                minimum=-1,
                maximum=1,
                value=DefaultValues().pose_yaw,
            )
            pose_pitch = Slider(
                label="pose_pitch",
                minimum=-1,
                maximum=1,
                value=DefaultValues().pose_pitch,
            )
            pose_roll = Slider(
                label="pose_roll",
                minimum=-1,
                maximum=1,
                value=DefaultValues().pose_roll,
            )
            face_location = Slider(
                label="face_location", maximum=1, value=DefaultValues().face_location
            )
            face_scale = Slider(
                label="face_scale", maximum=1, value=DefaultValues().face_scale
            )
            step_t = Slider(
                label="step_T", minimum=1, step=1, value=DefaultValues().step_T
            )

        generate_button.click(
            fn=generate_video,
            inputs=[
                uploaded_img,
                uploaded_audio,
                infer_type,
                pose_yaw,
                pose_pitch,
                pose_roll,
                face_location,
                face_scale,
                step_t,
                face_sr,
                seed,
            ],
            outputs=[output_video_256, output_video_512, output_message],
        )
        return app
