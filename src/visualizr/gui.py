from gradio import (
    HTML,
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

from visualizr import default_values
from visualizr.utils import generate_video


def app_block() -> Blocks:
    """Create the Gradio interface for the voice generation web application."""
    with Blocks() as app:
        Markdown(value="# AniTalker")
        Markdown(value="![]()")
        Markdown(
            value="credits: [X-LANCE](https://github.com/X-LANCE/AniTalker) (creators of the github repository), [Yuhan Xu](https://github.com/yuhanxu01)(webui), Delik"
        )
        Markdown(
            value="AniTalker: Animate Vivid and Diverse Talking Faces through Identity-Decoupled Facial Motion Encoding. [[arXiv]](https://arxiv.org/abs/2405.03121) [[project]](https://x-lance.github.io/AniTalker/)"
        )
        HTML(
            value='<a href="https://discord.gg/AQsmBmgEPy"> <img src="https://img.shields.io/discord/1198701940511617164?color=%23738ADB&label=Discord&style=for-the-badge" alt="Discord"> </a>'
        )
        with Row():
            with Column():
                uploaded_img = Image(type="filepath", label="Reference Image")
                uploaded_audio = Audio(type="filepath", label="Input Audio")
            with Column():
                output_video_256 = Video(label="Generated Video (256)")
                output_video_512 = Video(label="Generated Video (512)")
                output_message = Markdown()

        generate_button = Button(value="Generate Video")

        with Accordion(label="Configuration", open=True):
            infer_type = Dropdown(
                label="Inference Type",
                choices=[
                    "mfcc_pose_only",
                    "mfcc_full_control",
                    "hubert_audio_only",
                    "hubert_pose_only",
                ],
                value="hubert_audio_only",
            )
            face_sr = Checkbox(
                label="Enable Face Super-Resolution (512*512)", value=False
            )
            seed = Number(label="Seed", value=default_values["seed"])
            pose_yaw = Slider(
                label="pose_yaw", minimum=-1, maximum=1, value=default_values["pose_yaw"],
            )
            pose_pitch = Slider(
                label="pose_pitch",
                minimum=-1,
                maximum=1,
                value=default_values["pose_pitch"],
            )
            pose_roll = Slider(
                label="pose_roll", minimum=-1, maximum=1, value=default_values["pose_roll"],
            )
            face_location = Slider(
                label="face_location",
                minimum=0,
                maximum=1,
                value=default_values["face_location"],
            )
            face_scale = Slider(
                label="face_scale", minimum=0, maximum=1, value=default_values["face_scale"],
            )
            step_T = Slider(
                label="step_T",
                minimum=1,
                maximum=100,
                step=1,
                value=default_values["step_T"],
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
                step_T,
                face_sr,
                seed,
            ],
            outputs=[output_video_256, output_video_512, output_message],
        )
        return app