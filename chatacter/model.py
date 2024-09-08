import subprocess
import time

from transformers import logging

logging.set_verbosity_debug()


def generate_video(character: str) -> str:
    start_time = time.time()
    command = [
        "python",
        "chatacter/inference.py",
        "--driven_audio",
        "./assets/audio/AUDIO.wav",
        "--source_image",
        f"./assets/image/{character}.jpg",
        "--result_dir",
        "./assets/results",
        "--still",
        "--preprocess",
        "full",
        "--enhancer",
        "gfpgan",
    ]
    print(" ".join(command))
    subprocess.run(command)
    end_time = time.time()
    return str(end_time - start_time)
