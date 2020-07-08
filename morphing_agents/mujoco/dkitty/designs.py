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
        y=0.1220,
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


def sample_uniformly() -> List[NamedTuple]:
    return [LEG(*np.random.uniform(
        low=LEG_LOWER_BOUND, high=LEG_UPPER_BOUND))
        for n in range(4)]
