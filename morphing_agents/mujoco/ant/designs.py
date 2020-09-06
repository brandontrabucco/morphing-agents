from morphing_agents.mujoco.ant.elements import LEG
from morphing_agents.mujoco.ant.elements import LEG_UPPER_BOUND
from morphing_agents.mujoco.ant.elements import LEG_LOWER_BOUND
from typing import List, NamedTuple
import numpy as np


DEFAULT_DESIGN = (
    LEG(x=0.0,
        y=0.0,
        z=0.0,
        a=0,
        b=0,
        c=0,
        hip_center=0,
        hip_range=5,
        thigh_center=0,
        thigh_range=30,
        ankle_center=50,
        ankle_range=20,
        hip_size=0.2,
        thigh_size=0.2,
        ankle_size=0.4),
    LEG(x=0.0,
        y=0.0,
        z=0.0,
        a=0,
        b=0,
        c=90,
        hip_center=0,
        hip_range=5,
        thigh_center=0,
        thigh_range=30,
        ankle_center=50,
        ankle_range=20,
        hip_size=0.2,
        thigh_size=0.2,
        ankle_size=0.4),
    LEG(x=0.0,
        y=0.0,
        z=0.0,
        a=0,
        b=0,
        c=180,
        hip_center=0,
        hip_range=5,
        thigh_center=0,
        thigh_range=30,
        ankle_center=50,
        ankle_range=20,
        hip_size=0.2,
        thigh_size=0.2,
        ankle_size=0.4),
    LEG(x=0.0,
        y=0.0,
        z=0.0,
        a=0,
        b=0,
        c=-90,
        hip_center=0,
        hip_range=5,
        thigh_center=0,
        thigh_range=30,
        ankle_center=50,
        ankle_range=20,
        hip_size=0.2,
        thigh_size=0.2,
        ankle_size=0.4))


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
