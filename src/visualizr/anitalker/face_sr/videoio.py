import cv2
import numpy as np


def load_video_to_cv2(input_path: str) -> list[np.ndarray]:
    video_stream = cv2.VideoCapture(input_path)
    full_frames: list[np.ndarray] = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        full_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return full_frames
