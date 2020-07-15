from morphing_agents.mujoco.dkitty.elements import LEG
from morphing_agents.mujoco.dkitty.elements import LEG_UPPER_BOUND
from morphing_agents.mujoco.dkitty.elements import LEG_LOWER_BOUND
from typing import List, NamedTuple
import numpy as np


PI = np.pi


DEFAULT_DESIGN = (
    LEG(x=0.09,
        y=0.122,
        z=0,
        a=0,
        b=-3.14,
        c=0,
        hip_center=-0.1105,
        hip_range=0.3895,
        thigh_center=PI / 4,
        thigh_range=PI / 4,
        ankle_center=-1.0,
        ankle_range=1.0,
        thigh_size=0.0965,
        ankle_size=0.0945),
    LEG(x=-0.09,
        y=0.122,
        z=0,
        a=0,
        b=3.14,
        c=0,
        hip_center=0.1105,
        hip_range=0.3895,
        thigh_center=PI / 4,
        thigh_range=PI / 4,
        ankle_center=-1.0,
        ankle_range=1.0,
        thigh_size=0.0965,
        ankle_size=0.0945),
    LEG(x=-0.09,
        y=-0.122,
        z=0,
        a=0,
        b=3.14,
        c=0,
        hip_center=0.1105,
        hip_range=0.3895,
        thigh_center=PI / 4,
        thigh_range=PI / 4,
        ankle_center=-1.0,
        ankle_range=1.0,
        thigh_size=0.0965,
        ankle_size=0.0945),
    LEG(x=0.09,
        y=-0.122,
        z=0,
        a=0,
        b=-3.14,
        c=0,
        hip_center=-0.1105,
        hip_range=0.3895,
        thigh_center=PI / 4,
        thigh_range=PI / 4,
        ankle_center=-1.0,
        ankle_range=1.0,
        thigh_size=0.0965,
        ankle_size=0.0945))


def sample_uniformly(num_legs=4) -> List[NamedTuple]:
    """Sample new designs uniformly from the design space with the
    provided number of legs

    Args:

    num_legs: int
        the number of legs in the agent, used if fixed_design is None

    Returns:

    design: list
        a list of design elements, which are named tuples such as
        [LEG(x=0, y=0, z=0, a=0, b=0, c=0, ...), ...]
    """

    return [LEG(*np.random.uniform(
        low=LEG_LOWER_BOUND, high=LEG_UPPER_BOUND))
        for n in range(num_legs)]


def sample_centered(noise_std=0.125,
                    center=DEFAULT_DESIGN) -> List[NamedTuple]:
    """Sample new designs uniformly from the design space with the
    provided number of legs

    Args:

    noise_std: float
        a fraction of the design space the noise std takes
    center: list
        a default morphology centered morphologies are sampled from

    Returns:

    design: list
        a list of design elements, which are named tuples such as
        [LEG(x=0, y=0, z=0, a=0, b=0, c=0, ...), ...]
    """

    ub = np.array(list(LEG_UPPER_BOUND))
    lb = np.array(list(LEG_LOWER_BOUND))
    return [LEG(*np.clip(
        np.array(leg) + np.random.normal(0, (ub - lb) / 2) * noise_std,
        lb, ub)) for leg in center]
