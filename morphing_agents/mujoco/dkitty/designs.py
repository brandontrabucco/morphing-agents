from morphing_agents.mujoco.dkitty.elements import LEG
from morphing_agents.mujoco.dkitty.elements import LEG_UPPER_BOUND
from morphing_agents.mujoco.dkitty.elements import LEG_LOWER_BOUND
from typing import List, NamedTuple
import numpy as np


DEFAULT_DKITTY = (
    LEG(x=0,
        y=0,
        z=0,
        a=0,
        b=0,
        c=0,
        hip_upper=0.0,
        hip_lower=-0.0,
        thigh_upper=0.0,
        thigh_lower=0.0,
        ankle_upper=0.0,
        ankle_lower=0.0,
        thigh_size=0.0,
        ankle_size=0.0),
    LEG(x=0,
        y=0,
        z=0,
        a=0,
        b=0,
        c=0,
        hip_upper=0.0,
        hip_lower=-0.0,
        thigh_upper=0.0,
        thigh_lower=0.0,
        ankle_upper=0.0,
        ankle_lower=0.0,
        thigh_size=0.0,
        ankle_size=0.0),
    LEG(x=0,
        y=0,
        z=0,
        a=0,
        b=0,
        c=0,
        hip_upper=0.0,
        hip_lower=-0.0,
        thigh_upper=0.0,
        thigh_lower=0.0,
        ankle_upper=0.0,
        ankle_lower=0.0,
        thigh_size=0.0,
        ankle_size=0.0),
    LEG(x=0,
        y=0,
        z=0,
        a=0,
        b=0,
        c=0,
        hip_upper=0.0,
        hip_lower=-0.0,
        thigh_upper=0.0,
        thigh_lower=0.0,
        ankle_upper=0.0,
        ankle_lower=0.0,
        thigh_size=0.0,
        ankle_size=0.0))


def sample_uniformly() -> List[NamedTuple]:
    return [LEG(*np.random.uniform(
        low=LEG_LOWER_BOUND, high=LEG_UPPER_BOUND))
        for n in range(4)]
