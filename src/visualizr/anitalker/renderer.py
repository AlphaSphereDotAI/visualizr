"""
This module provides rendering functionality for the AniTalker model, generating
samples conditioned on various inputs using a diffusion sampler.
"""

from visualizr.anitalker.choices import TrainMode
from visualizr.anitalker.config import TrainConfig


def render_condition(
    conf: TrainConfig,
    model,
    sampler,
    start,
    motion_direction_start,
    audio_driven,
    face_location,
    face_scale,
    yaw_pitch_roll,
    noisy_t,
    control_flag,
):
    """
    Render a conditional diffusion sample using the provided sampler and model.

    Args:
        conf (TrainConfig): Configuration for training specifying model and mode.
        model: Autoencoder-capable model for diffusion sampling.
        sampler: Sampler instance for generating diffusion samples.
        start: Initial latent or frame to start the sampling process.
        motion_direction_start: Initial motion direction data for conditioning.
        audio_driven: Audio-driven control input for the sampler.
        face_location: Coordinates of the face location for conditioning.
        face_scale: Scale factor for the face region.
        yaw_pitch_roll: Tuple specifying yaw, pitch, and roll angles for head pose.
        noisy_t: Noise tensor representing the diffusion noise level.
        control_flag: Flags controlling additional conditioning behavior.

    Returns:
        The generated sample from the sampler based on provided conditioning parameters.

    Raises:
        NotImplementedError: If the training mode in conf is not diffusion.
        ValueError: If the model type is not autoencoder-capable.
    """
    if conf.train_mode != TrainMode.diffusion:
        raise NotImplementedError()
    if not conf.model_type.has_autoenc():
        msg: str = (
            "TrainMode.diffusion requires an "
            "autoencoder-capable `model_type`; "
            f"got {conf.model_type!r}"
        )
        raise ValueError(msg)
    return sampler.sample(
        model=model,
        noise=noisy_t,
        model_kwargs={
            "motion_direction_start": motion_direction_start,
            "yaw_pitch_roll": yaw_pitch_roll,
            "start": start,
            "audio_driven": audio_driven,
            "face_location": face_location,
            "face_scale": face_scale,
            "control_flag": control_flag,
        },
    )
