import os
import shutil
import uuid

import cv2
from numpy import dtype, float32, generic, ndarray, number


def load_video_to_cv2(
    input_path: str,
) -> list[ndarray[tuple[float, float, float], dtype[generic]]]:
    video_stream = cv2.VideoCapture(filename=input_path)
    fps: float = video_stream.get(propId=cv2.CAP_PROP_FPS)
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


def save_video_with_watermark(video, audio, save_path, watermark=False):
    temp_file: str = f"{str(uuid.uuid4())}.mp4"
    cmd: str = (
        r'ffmpeg -y -hide_banner -loglevel error -i "%s" -i "%s" -vcodec copy "%s"'
        % (video, audio, temp_file)
    )
    os.system(command=cmd)

    if watermark is False:
        shutil.move(src=temp_file, dst=save_path)
    else:
        # watermark
        try:
            ##### check if stable-diffusion-webui
            from modules import paths

            watarmark_path = (
                f"{paths.script_path}/extensions/SadTalker/docs/sadtalker_logo.png"
            )

        except Exception:
            # get the root path of sadtalker.
            dir_path: str = os.path.dirname(p=os.path.realpath(path=__file__))
            watarmark_path: str = dir_path + "/../../docs/sadtalker_logo.png"

        cmd = (
            r'ffmpeg -y -hide_banner -loglevel error -i "%s" -i "%s" -filter_complex "[1]scale=100:-1[wm];[0][wm]overlay=(main_w-overlay_w)-10:10" "%s"'
            % (temp_file, watarmark_path, save_path)
        )
        os.system(command=cmd)
        os.remove(path=temp_file)
