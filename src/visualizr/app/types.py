from typing import Literal

InferenceType = Literal[
    "mfcc_full_control",
    "mfcc_pose_only",
    "hubert_pose_only",
    "hubert_audio_only",
    "hubert_full_control",
]
