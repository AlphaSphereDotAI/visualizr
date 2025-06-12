import cv2
from numpy import dtype, generic, ndarray


def load_video_to_cv2(
    input_path: str,
) -> list[ndarray[tuple[float, float, float], dtype[generic]]]:
    video_stream = cv2.VideoCapture(filename=input_path)
    video_stream.get(propId=cv2.CAP_PROP_FPS)
    full_frames: list[ndarray[tuple[float, float, float], dtype[generic]]] = []
    while 1:
        video_result: tuple[
            bool, ndarray[tuple[float, float, float], dtype[generic]]
        ] = video_stream.read()
        still_reading: bool = video_result[0]
        frame: ndarray[tuple[float, float, float], dtype[generic]] = video_result[1]
        if not still_reading:
            video_stream.release()
            break
        full_frames.append(cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB))
    return full_frames
