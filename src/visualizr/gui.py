from gradio import (
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
    Tab,
    Video,
)

from visualizr.model.runner import model, settings


def app_block() -> Blocks:
    """Create the Gradio interface for the voice generation web app."""
    with Blocks() as app:
        with Tab("AniTalker"):
            with Row():
                with Column():
                    image_path: Image = Image(
                        type="filepath",
                        label="Reference Image",
                    )
                    audio_path = Audio(
                        type="filepath",
                        label="Input Audio",
                        show_download_button=True,
                    )
                with Column():
                    output_video_256 = Video(label="Generated Video (256)")
                    output_video_512 = Video(label="Generated Video (512)")
                    output_message = Markdown()
            generate_button = Button(value="Generate Video", variant="primary")

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
                value=settings.model.seed,
            )
            pose_yaw = Slider(
                label="pose_yaw",
                minimum=-1,
                maximum=1,
                value=settings.model.pose_yaw,
            )
            pose_pitch = Slider(
                label="pose_pitch",
                minimum=-1,
                maximum=1,
                value=settings.model.pose_pitch,
            )
            pose_roll = Slider(
                label="pose_roll",
                minimum=-1,
                maximum=1,
                value=settings.model.pose_roll,
            )
            face_location = Slider(
                label="face_location",
                maximum=1,
                value=settings.model.face_location,
            )
            face_scale = Slider(
                label="face_scale",
                maximum=1,
                value=settings.model.face_scale,
            )
            step_t = Slider(
                label="step_T",
                minimum=1,
                step=1,
                value=settings.model.step_t,
            )

        generate_button.click(
            model,
            [
                image_path,
                audio_path,
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
            [
                output_video_256,
                output_video_512,
                output_message,
            ],
        )
        return app
