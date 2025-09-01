"""This module contains tests for the app's HTTP endpoints."""

from os import getenv

from gradio_client import Client, handle_file


def test_app() -> None:
    """
    Test the reachability of Visualizr.

    Returns:
        None
    """
    client = Client(getenv("VISUALIZR_URL", "http://localhost:7860/"))
    assert client.heartbeat.is_alive()


def test_generate_video() -> None:
    client = Client("http://127.0.0.1:7860/")
    result = client.predict(
        infer_type="hubert_audio_only",
        image_path=handle_file(
            "https://raw.githubusercontent.com/AlphaSphereDotAI/chattr/main/assets/image/Napoleon.jpg"
        ),
        audio_path=handle_file(
            "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav"
        ),
        face_sr=False,
        pose_yaw=0,
        pose_pitch=0,
        pose_roll=0,
        face_location=0.5,
        face_scale=0.5,
        step_t=50,
        seed=0,
        api_name="/generate_video",
    )
    print(result)
