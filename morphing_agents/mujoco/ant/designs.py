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
        hip_upper=0,
        hip_lower=-5,
        thigh_upper=30,
        thigh_lower=-30,
        ankle_upper=70,
        ankle_lower=30,
        hip_size=0.2,
        thigh_size=0.2,
        ankle_size=0.4),
    LEG(x=0.0,
        y=0.0,
        z=0.0,
        a=0,
        b=0,
        c=90,
        hip_upper=0,
        hip_lower=-5,
        thigh_upper=30,
        thigh_lower=-30,
        ankle_upper=70,
        ankle_lower=30,
        hip_size=0.2,
        thigh_size=0.2,
        ankle_size=0.4),
    LEG(x=0.0,
        y=0.0,
        z=0.0,
        a=0,
        b=0,
        c=180,
        hip_upper=0,
        hip_lower=-5,
        thigh_upper=30,
        thigh_lower=-30,
        ankle_upper=70,
        ankle_lower=30,
        hip_size=0.2,
        thigh_size=0.2,
        ankle_size=0.4),
    LEG(x=0.0,
        y=0.0,
        z=0.0,
        a=0,
        b=0,
        c=270,
        hip_upper=0,
        hip_lower=-5,
        thigh_upper=30,
        thigh_lower=-30,
        ankle_upper=70,
        ankle_lower=30,
        hip_size=0.2,
        thigh_size=0.2,
        ankle_size=0.4))


def sample_uniformly(num_legs=4) -> List[NamedTuple]:
    return [LEG(*np.random.uniform(
        low=LEG_LOWER_BOUND, high=LEG_UPPER_BOUND))
        for n in range(num_legs)]
