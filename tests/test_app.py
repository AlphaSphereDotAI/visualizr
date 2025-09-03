"""Tests for the app's HTTP endpoints."""

from os import getenv
from pathlib import Path

from gradio_client import Client, handle_file

URL = getenv("VISUALIZR_URL", "http://localhost:7860/")
IMAGE_PATH = handle_file(
    "https://github.com/AlphaSphereDotAI/chattr/raw/main/assets/image/Napoleon.jpg",
)
AUDIO_PATH = handle_file(
    "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",
)


def test_app() -> None:
    """Test the reachability of Visualizr."""
    client = Client(URL)
    assert client.heartbeat.is_alive()


def test_generate_video_mfcc_full_control_no_face_sr() -> None:
    """Test MFCC full-control inference without a face super-resolution."""
    client = Client(URL)
    result = client.predict(
        infer_type="mfcc_full_control",
        image_path=IMAGE_PATH,
        audio_path=AUDIO_PATH,
        face_sr=False,
        api_name="/generate_video",
    )
    assert Path(result[0]["value"]["video"]).is_file()
    assert result[1] is None
    assert result[2]["value"] == "Video (256 ✕ 256 only) generated successfully!"


def test_generate_video_mfcc_full_control_face_sr() -> None:
    """Test MFCC full-control inference with a face super-resolution."""
    client = Client(URL)
    result = client.predict(
        infer_type="mfcc_full_control",
        image_path=IMAGE_PATH,
        audio_path=AUDIO_PATH,
        face_sr=True,
        api_name="/generate_video",
    )
    assert Path(result[0]["value"]["video"]).is_file()
    assert Path(result[1]["value"]["video"]).is_file()
    assert result[2]["value"] == "Video generated successfully!"


def test_generate_video_mfcc_pose_only_no_face_sr() -> None:
    """Test MFCC pose-only inference without a face super-resolution."""
    client = Client(URL)
    result = client.predict(
        infer_type="mfcc_pose_only",
        image_path=IMAGE_PATH,
        audio_path=AUDIO_PATH,
        face_sr=False,
        api_name="/generate_video",
    )
    assert Path(result[0]["value"]["video"]).is_file()
    assert result[1] is None
    assert result[2]["value"] == "Video (256 ✕ 256 only) generated successfully!"


def test_generate_video_mfcc_pose_only_face_sr() -> None:
    """Test MFCC pose-only inference with a face super-resolution."""
    client = Client(URL)
    result = client.predict(
        infer_type="mfcc_pose_only",
        image_path=IMAGE_PATH,
        audio_path=AUDIO_PATH,
        face_sr=True,
        api_name="/generate_video",
    )
    assert Path(result[0]["value"]["video"]).is_file()
    assert Path(result[1]["value"]["video"]).is_file()
    assert result[2]["value"] == "Video generated successfully!"


def test_generate_video_hubert_pose_only_no_face_sr() -> None:
    """Test HuBERT pose-only inference without a face super-resolution."""
    client = Client(URL)
    result = client.predict(
        infer_type="hubert_pose_only",
        image_path=IMAGE_PATH,
        audio_path=AUDIO_PATH,
        face_sr=False,
        api_name="/generate_video",
    )
    assert Path(result[0]["value"]["video"]).is_file()
    assert result[1] is None
    assert result[2]["value"] == "Video (256 ✕ 256 only) generated successfully!"


def test_generate_video_hubert_pose_only_face_sr() -> None:
    """Test HuBERT pose-only inference with a face super-resolution."""
    client = Client(URL)
    result = client.predict(
        infer_type="hubert_pose_only",
        image_path=IMAGE_PATH,
        audio_path=AUDIO_PATH,
        face_sr=True,
        api_name="/generate_video",
    )
    assert Path(result[0]["value"]["video"]).is_file()
    assert Path(result[1]["value"]["video"]).is_file()
    assert result[2]["value"] == "Video generated successfully!"


def test_generate_video_hubert_audio_only_no_face_sr() -> None:
    """Test HuBERT audio-only inference without a face super-resolution."""
    client = Client(URL)
    result = client.predict(
        infer_type="hubert_audio_only",
        image_path=IMAGE_PATH,
        audio_path=AUDIO_PATH,
        face_sr=False,
        api_name="/generate_video",
    )
    assert Path(result[0]["value"]["video"]).is_file()
    assert result[1] is None
    assert result[2]["value"] == "Video (256 ✕ 256 only) generated successfully!"


def test_generate_video_hubert_audio_only_face_sr() -> None:
    """Test HuBERT audio-only inference with a face super-resolution."""
    client = Client(URL)
    result = client.predict(
        infer_type="hubert_audio_only",
        image_path=IMAGE_PATH,
        audio_path=AUDIO_PATH,
        face_sr=True,
        api_name="/generate_video",
    )
    assert Path(result[0]["value"]["video"]).is_file()
    assert Path(result[1]["value"]["video"]).is_file()
    assert result[2]["value"] == "Video generated successfully!"


def test_generate_video_hubert_full_control_no_face_sr() -> None:
    """Test HuBERT full-control inference without a face super-resolution."""
    client = Client(URL)
    result = client.predict(
        infer_type="hubert_full_control",
        image_path=IMAGE_PATH,
        audio_path=AUDIO_PATH,
        face_sr=False,
        api_name="/generate_video",
    )
    assert Path(result[0]["value"]["video"]).is_file()
    assert result[1] is None
    assert result[2]["value"] == "Video (256 ✕ 256 only) generated successfully!"


def test_generate_video_hubert_full_control_face_sr() -> None:
    """Test HuBERT full-control inference with a face super-resolution."""
    client = Client(URL)
    result = client.predict(
        infer_type="hubert_full_control",
        image_path=IMAGE_PATH,
        audio_path=AUDIO_PATH,
        face_sr=True,
        api_name="/generate_video",
    )
    assert Path(result[0]["value"]["video"]).is_file()
    assert Path(result[1]["value"]["video"]).is_file()
    assert result[2]["value"] == "Video generated successfully!"
