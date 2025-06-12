from datetime import datetime
from os import getenv
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from loguru import logger
from torch import cuda

load_dotenv()

DEBUG: bool = getenv(key="DEBUG", default="True").lower() == "true"
SERVER_NAME: str = getenv(key="GRADIO_SERVER_NAME", default="localhost")
SERVER_PORT: int = int(getenv(key="GRADIO_SERVER_PORT", default="8080"))
CURRENT_DATE: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

BASE_DIR: Path = Path.cwd()
RESULTS_DIR: Path = BASE_DIR / "results"
LOG_DIR: Path = BASE_DIR / "logs"
CHECKPOINT_DIR: Path = BASE_DIR / "ckpts"
AUDIO_FILE_PATH: Path = RESULTS_DIR / f"{CURRENT_DATE}.wav"
LOG_FILE_PATH: Path = LOG_DIR / f"{CURRENT_DATE}.log"
CUDA_AVAILABLE: bool = cuda.is_available()

FRAMES_RESULT_SAVED_PATH: Path = RESULTS_DIR / "frames"
STAGE_1_CHECKPOINT_PATH = CHECKPOINT_DIR / "stage1.ckpt"

RESULTS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
FRAMES_RESULT_SAVED_PATH.mkdir(exist_ok=True)

MOTION_DIM: int = 20
TMP_MP4: str = ".tmp.mp4"

logger.add(
    sink=LOG_FILE_PATH,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    colorize=True,
)
logger.info(f"CUDA Available: {CUDA_AVAILABLE}")
logger.info(f"Current date: {CURRENT_DATE}")
logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Results directory: {RESULTS_DIR}")
logger.info(f"Log directory: {LOG_DIR}")
logger.info(f"Checkpoint directory: {CHECKPOINT_DIR}")

model_mapping: dict[str, str] = {
    "mfcc_pose_only": f"{CHECKPOINT_DIR}/stage2_pose_only_mfcc.ckpt",
    "mfcc_full_control": f"{CHECKPOINT_DIR}/stage2_more_controllable_mfcc.ckpt",
    "hubert_audio_only": f"{CHECKPOINT_DIR}/stage2_audio_only_hubert.ckpt",
    "hubert_pose_only": f"{CHECKPOINT_DIR}/stage2_pose_only_hubert.ckpt",
    "hubert_full_control": f"{CHECKPOINT_DIR}/stage2_full_control_hubert.ckpt",
}

snapshot_download(
    repo_id="taocode/anitalker_ckpts", local_dir=CHECKPOINT_DIR, repo_type="model"
)
