"""Utility functions for the Visualizr application."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from httpx import URL, Client, HTTPStatusError, RequestError, Response

from visualizr.app.logger import logger


def download_file(url: URL, suffix: str = ".wav") -> Path:
    """
    Download a file from a given URL and save it as a temporary file.

    Args:
        url (URL): The URL to download the file from.
        suffix (str): The suffix for the temporary file. Defaults to ".wav".

    Returns:
        Path: The path to the downloaded temporary file.
    """
    try:
        with Client() as client:
            response: Response = client.get(url)
            response.raise_for_status()
            with NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(response.content)
                audio_path = Path(f.name)
            logger.info(f"Downloaded audio to {audio_path}")
    except (RequestError, HTTPStatusError, OSError) as e:
        msg: str = f"Failed to download audio from {url}: {e}"
        logger.error(msg)
        raise
    return audio_path
