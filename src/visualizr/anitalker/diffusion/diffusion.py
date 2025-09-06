from dataclasses import dataclass

import numpy as np

from visualizr.anitalker.diffusion.base import (
    GaussianDiffusionBeatGans,
    GaussianDiffusionBeatGansConfig,
)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process.

    Given the number of timesteps we want to take from equally sized portions
    of the original process.

    For example, if there are 300 timesteps, and the section counts are [10,15,20],
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: The number of diffusion steps in the original
                          process to divide up.
    :param section_counts: List of numbers or string containing
                           comma-separated numbers, indicating the step count
                           per a section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: A set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            msg = f"cannot create exactly {num_timesteps} steps with an integer stride"
            raise ValueError(msg)
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            msg = f"cannot divide section of {size} steps into {section_count}"
            raise ValueError(msg)
        frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


@dataclass
class SpacedDiffusionBeatGansConfig(GaussianDiffusionBeatGansConfig):
    """
    Configuration for a spaced diffusion process.

    This class holds the parameters for creating a spaced diffusion sampler, including the timesteps to use.

    Args:
        use_timesteps: A collection (sequence or set) of timesteps from the
                       original diffusion process to retain.
    """

    use_timesteps: tuple[int] | None = None

    def make_sampler(self):
        return SpacedDiffusionBeatGans(self)


class SpacedDiffusionBeatGans(GaussianDiffusionBeatGans):
    """A diffusion process, which can skip steps in a base diffusion process."""

    def __init__(self, conf: SpacedDiffusionBeatGansConfig):
        self.conf = conf
        self.use_timesteps = set(conf.use_timesteps)
        # how the new t's mapped to the old t's
        self.timestep_map = []
        base_diffusion = GaussianDiffusionBeatGans(conf)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                # getting the new betas of the new timesteps
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        conf.betas = np.array(new_betas)
        super().__init__(conf)
