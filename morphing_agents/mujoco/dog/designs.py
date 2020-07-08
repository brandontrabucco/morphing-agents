from morphing_agents.mujoco.dog.elements import LEG
from morphing_agents.mujoco.dog.elements import LEG_UPPER_BOUND
from morphing_agents.mujoco.dog.elements import LEG_LOWER_BOUND
from typing import List, NamedTuple
import numpy as np


DEFAULT_DESIGN = (
    LEG(x=-0.4,
        y=-0.2,
        z=0.0,
        a=0,
        b=0,
        c=0,
        hip_center=0,
        hip_range=30,
        thigh_center=45,
        thigh_range=30,
        ankle_center=-90,
        ankle_range=30,
        thigh_size=0.4,
        ankle_size=0.4),
    LEG(x=0.4,
        y=-0.2,
        z=0.0,
        a=0,
        b=0,
        c=0,
        hip_center=0,
        hip_range=30,
        thigh_center=45,
        thigh_range=30,
        ankle_center=-90,
        ankle_range=30,
        thigh_size=0.4,
        ankle_size=0.4),
    LEG(x=0.4,
        y=0.2,
        z=0.0,
        a=0,
        b=0,
        c=0,
        hip_center=0,
        hip_range=30,
        thigh_center=45,
        thigh_range=30,
        ankle_center=-90,
        ankle_range=30,
        thigh_size=0.4,
        ankle_size=0.4),
    LEG(x=-0.4,
        y=0.2,
        z=0.0,
        a=0,
        b=0,
        c=0,
        hip_center=0,
        hip_range=30,
        thigh_center=45,
        thigh_range=30,
        ankle_center=-90,
        ankle_range=30,
        thigh_size=0.4,
        ankle_size=0.4))


def sample_uniformly(num_legs=4) -> List[NamedTuple]:
    return [LEG(*np.random.uniform(
        low=LEG_LOWER_BOUND, high=LEG_UPPER_BOUND))
        for n in range(num_legs)]
