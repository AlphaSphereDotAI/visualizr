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
    if conf.train_mode != TrainMode.diffusion:
        raise NotImplementedError()
    if not conf.model_type.has_autoenc():
        raise ValueError(
            f"TrainMode.diffusion requires an autoencoder-capable model_type; got {conf.model_type!r}"
        )
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
