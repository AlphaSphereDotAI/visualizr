"""This module contains tests for the app's HTTP endpoints."""

from os import getenv

from gradio_client import Client


def test_app() -> None:
    """
    Test the reachability of Visualizr.

    Returns:
        None
    """
    client = Client(getenv("VISUALIZR_URL", "http://localhost:7860/"))
    assert client.heartbeat.is_alive()
