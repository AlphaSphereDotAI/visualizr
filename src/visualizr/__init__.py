from datetime import datetime
from os import getenv
from pathlib import Path
from warnings import filterwarnings

from dotenv import load_dotenv
from loguru import logger
from torch import cuda

filterwarnings(
    action="ignore",
    message="dropout option adds dropout after all but last recurrent layer",
)
filterwarnings(
    action="ignore",
    message="`torch.nn.utils.weight_norm` is deprecated",
)

load_dotenv()

DEBUG: bool = getenv(key="DEBUG", default="True").lower() == "true"
SERVER_NAME: str = getenv(key="GRADIO_SERVER_NAME", default="localhost")
SERVER_PORT: int = int(getenv(key="GRADIO_SERVER_PORT", default="8080"))
CURRENT_DATE: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

BASE_DIR: Path = Path.cwd()
RESULTS_DIR: Path = BASE_DIR / "results"
LOG_DIR: Path = BASE_DIR / "logs"
AUDIO_FILE_PATH: Path = RESULTS_DIR / f"{CURRENT_DATE}.wav"
LOG_FILE_PATH: Path = LOG_DIR / f"{CURRENT_DATE}.log"

RESULTS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

CUDA_AVAILABLE: bool = cuda.is_available()
logger.add(
    LOG_FILE_PATH,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    colorize=True,
)
logger.info(f"CUDA Available: {CUDA_AVAILABLE}")
logger.info(f"Current date: {CURRENT_DATE}")
logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Results directory: {RESULTS_DIR}")
logger.info(f"Log directory: {LOG_DIR}")
logger.info(f"Audio file path: {AUDIO_FILE_PATH}")
logger.info(f"Log file path: {LOG_FILE_PATH}")
