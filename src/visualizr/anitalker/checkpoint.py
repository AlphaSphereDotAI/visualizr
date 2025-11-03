"""Enumeration of model checkpoint names."""

from enum import Enum


class ModelName(str, Enum):
    """Enumeration of available model checkpoints."""

    stage_1: str = "stage1.ckpt"
    mfcc_pose_only: str = "stage2_pose_only_mfcc.ckpt"
    mfcc_full_control: str = "stage2_full_control_mfcc.ckpt"
    hubert_audio_only: str = "stage2_audio_only_hubert.ckpt"
    hubert_pose_only: str = "stage2_pose_only_hubert.ckpt"
    hubert_full_control: str = "stage2_full_control_hubert.ckpt"
