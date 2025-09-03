from os import listdir, path
from random import randint

import numpy as np
from librosa import load as librosa_load
from PIL import Image
from python_speech_features import delta, mfcc
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm


class LatentDataLoader:
    """
    Data loader for latent features that loads image frames, audio features,
    and motion latents from specified directories. Applies image transformations,
    computes MFCC features if enabled,
    and prepares data windows for training or evaluation.
    """

    def __init__(
        self,
        window_size,
        frame_jpgs,
        lmd_feats_prefix,
        audio_prefix,
        raw_audio_prefix,
        motion_latents_prefix,
        pose_prefix,
        db_name,
        video_fps: int = 25,
        audio_hz: int = 50,
        size: int = 256,
        mfcc_mode: bool = False,
    ):
        self.window_size = window_size
        self.lmd_feats_prefix = lmd_feats_prefix
        self.audio_prefix = audio_prefix
        self.pose_prefix = pose_prefix
        self.video_fps = video_fps
        self.audio_hz = audio_hz
        self.db_name = db_name
        self.raw_audio_prefix = raw_audio_prefix
        self.mfcc_mode = mfcc_mode

        self.transform = Compose(
            [
                Resize((size, size)),
                ToTensor(),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ],
        )

        self.data = []
        for _db_name in ["VoxCeleb2", "HDTF"]:
            db_png_path = path.join(frame_jpgs, _db_name)
            for clip_name in tqdm(listdir(db_png_path)):
                item_dict: dict = {
                    "clip_name": clip_name,
                    "frame_count": len(
                        list(
                            listdir(
                                path.join(frame_jpgs, _db_name, clip_name),
                            ),
                        ),
                    ),
                    "hubert_path": path.join(
                        audio_prefix,
                        _db_name,
                        f"{clip_name}.npy",
                    ),
                    "wav_path": path.join(
                        raw_audio_prefix,
                        _db_name,
                        f"{clip_name}.wav",
                    ),
                    "yaw_pitch_roll_path": path.join(
                        pose_prefix,
                        _db_name,
                        "raw_videos_pose_yaw_pitch_roll",
                        f"{clip_name}.npy",
                    ),
                }

                if not path.exists(item_dict["yaw_pitch_roll_path"]):
                    print(f"{_db_name}'s {clip_name} miss yaw_pitch_roll_path")
                    continue

                item_dict["yaw_pitch_roll"] = np.load(item_dict["yaw_pitch_roll_path"])
                item_dict["yaw_pitch_roll"] = (
                    np.clip(item_dict["yaw_pitch_roll"], -90, 90) / 90.0
                )

                if not path.exists(item_dict["wav_path"]):
                    print(f"{_db_name}'s {clip_name} miss wav_path")
                    continue

                if not path.exists(item_dict["hubert_path"]):
                    print(f"{_db_name}'s {clip_name} miss hubert_path")
                    continue

                if self.mfcc_mode:
                    wav, sr = librosa_load(item_dict["wav_path"], sr=16000)
                    input_values = mfcc(signal=wav, samplerate=sr)
                    d_mfcc_feat = delta(input_values, 1)
                    d_mfcc_feat2 = delta(input_values, 2)
                    input_values = np.hstack((input_values, d_mfcc_feat, d_mfcc_feat2))
                    item_dict["hubert_obj"] = input_values
                else:
                    item_dict["hubert_obj"] = np.load(
                        item_dict["hubert_path"],
                        mmap_mode="r",
                    )
                item_dict["lmd_path"] = path.join(
                    lmd_feats_prefix,
                    _db_name,
                    f"{clip_name}.txt",
                )
                item_dict["lmd_obj_full"] = self.read_landmark_info(
                    item_dict["lmd_path"],
                    upper_face=False,
                )

                motion_start_path = path.join(
                    motion_latents_prefix,
                    _db_name,
                    "motions",
                    f"{clip_name}.npy",
                )
                motion_direction_path = path.join(
                    motion_latents_prefix,
                    _db_name,
                    "directions",
                    f"{clip_name}.npy",
                )

                if not path.exists(motion_start_path):
                    print(f"{_db_name}'s {clip_name} miss motion_start_path")
                    continue
                if not path.exists(motion_direction_path):
                    print(f"{_db_name}'s {clip_name} miss motion_direction_path")
                    continue

                item_dict["motion_start_obj"] = np.load(motion_start_path)
                item_dict["motion_direction_obj"] = np.load(motion_direction_path)

                if self.mfcc_mode:
                    min_len = min(
                        item_dict["lmd_obj_full"].shape[0],
                        item_dict["yaw_pitch_roll"].shape[0],
                        item_dict["motion_start_obj"].shape[0],
                        item_dict["motion_direction_obj"].shape[0],
                        int(item_dict["hubert_obj"].shape[0] / 4),
                        item_dict["frame_count"],
                    )
                    item_dict["frame_count"] = min_len
                    item_dict["hubert_obj"] = item_dict["hubert_obj"][: min_len * 4, :]
                else:
                    min_len = min(
                        item_dict["lmd_obj_full"].shape[0],
                        item_dict["yaw_pitch_roll"].shape[0],
                        item_dict["motion_start_obj"].shape[0],
                        item_dict["motion_direction_obj"].shape[0],
                        int(item_dict["hubert_obj"].shape[1] / 2),
                        item_dict["frame_count"],
                    )

                    item_dict["frame_count"] = min_len
                    item_dict["hubert_obj"] = item_dict["hubert_obj"][
                        :,
                        : min_len * 2,
                        :,
                    ]

                if min_len < self.window_size * self.video_fps + 5:
                    continue

        print("Db count:", len(self.data))

    def get_single_image(self, image_path):
        """
        Load an image from a given path, convert it to RGB format,
        apply the configured transform, and return the processed image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            Tensor: Transformed image tensor in RGB format.
        """
        img_source = Image.open(image_path).convert("RGB")
        return self.transform(img_source)

    @staticmethod
    def get_multiple_ranges(lists, multi_ranges):
        """
        Extract elements from a list based on multiple start-end index ranges.

        Args:
            lists (list): The list from which to extract elements.
            multi_ranges (list of tuple): List of (start, end) index tuples.

        Returns:
            list: Flattened list of extracted elements from the specified ranges.

        Raises:
            ValueError: If `multi_ranges` is not a list of (start, end) tuples.

        """
        # Ensure that `multi_ranges` is a list of tuples
        if not all(isinstance(item, tuple) and len(item) == 2 for item in multi_ranges):
            msg: str = (
                "multi_ranges must be a list of (start, end) "
                "tuples with exactly two elements each"
            )
            raise ValueError(msg)
        extracted_elements = [lists[start:end] for start, end in multi_ranges]
        return [item for sublist in extracted_elements for item in sublist]

    def read_landmark_info(self, lmd_path, upper_face=True):
        with open(lmd_path) as file:
            lmd_lines = file.readlines()
        lmd_lines.sort()

        total_lmd_obj = []
        for line in lmd_lines:
            # Split the coordinates and filter out any empty strings
            coords = [c for c in line.strip().split(" ") if c]
            coords = coords[1:]  # don't include the filename in the first row
            lmd_obj = []
            if upper_face:
                # Ensure that the coordinates are parsed as integers
                for coord_pair in self.get_multiple_ranges(
                    coords,
                    [(0, 3), (14, 27), (36, 48)],
                ):  # 28
                    x, y = coord_pair.split("_")
                    lmd_obj.append((int(x) / 512, int(y) / 512))
            else:
                for coord_pair in coords:
                    x, y = coord_pair.split("_")
                    lmd_obj.append((int(x) / 512, int(y) / 512))
            total_lmd_obj.append(lmd_obj)

        return np.array(total_lmd_obj, dtype=np.float32)

    @staticmethod
    def calculate_face_height(landmarks):
        forehead_center = (landmarks[:, 21, :] + landmarks[:, 22, :]) / 2
        chin_bottom = landmarks[:, 8, :]
        return np.linalg.norm(forehead_center - chin_bottom, axis=1, keepdims=True)

    def __getitem__(self, index):
        data_item = self.data[index]
        hubert_obj = data_item["hubert_obj"]
        frame_count = data_item["frame_count"]
        lmd_obj_full = data_item["lmd_obj_full"]
        yaw_pitch_roll = data_item["yaw_pitch_roll"]
        motion_start_obj = data_item["motion_start_obj"]
        motion_direction_obj = data_item["motion_direction_obj"]

        frame_end_index = randint(
            self.window_size * self.video_fps + 1,
            frame_count - 1,
        )
        frame_start_index = frame_end_index - self.window_size * self.video_fps
        frame_hint_index = frame_start_index - 1

        audio_start_index = int(frame_start_index * (self.audio_hz / self.video_fps))
        audio_end_index = int(frame_end_index * (self.audio_hz / self.video_fps))

        if self.mfcc_mode:
            audio_feats = hubert_obj[audio_start_index:audio_end_index, :]
        else:
            audio_feats = hubert_obj[:, audio_start_index:audio_end_index, :]

        lmd_obj_full = lmd_obj_full[frame_hint_index:frame_end_index, :]

        yaw_pitch_roll = yaw_pitch_roll[frame_start_index:frame_end_index, :]

        motion_start = motion_start_obj[frame_hint_index]
        motion_direction_start = motion_direction_obj[frame_hint_index]
        motion_direction = motion_direction_obj[frame_start_index:frame_end_index, :]

        return {
            "motion_start": motion_start,
            "motion_direction": motion_direction,
            "audio_feats": audio_feats,
            # '1' → means taking the first frame as the driven frame.
            # '30' → is the noise location,
            # '0' → means x coordinate.
            "face_location": lmd_obj_full[1:, 30, 0],
            "face_scale": self.calculate_face_height(lmd_obj_full[1:, :, :]),
            "yaw_pitch_roll": yaw_pitch_roll,
            "motion_direction_start": motion_direction_start,
        }

    def __len__(self) -> int:
        return len(self.data)
