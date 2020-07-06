from collections import namedtuple
import numpy as np


PI = np.pi


LEG = namedtuple("LEG", [
    "x",
    "y",
    "z",
    "a",
    "b",
    "c",
    "hip_upper",
    "hip_lower",
    "thigh_upper",
    "thigh_lower",
    "ankle_upper",
    "ankle_lower",
    "thigh_size",
    "ankle_size"])

LEG_UPPER_BOUND = LEG(
    x=0.05,
    y=0.05,
    z=0,
    a=PI / 4,
    b=PI / 4,
    c=PI / 4,
    hip_upper=0.0,
    hip_lower=0.0,
    thigh_upper=0.0,
    thigh_lower=0.0,
    ankle_upper=0.0,
    ankle_lower=0.0,
    thigh_size=0.0,
    ankle_size=0.0)


LEG_LOWER_BOUND = LEG(
    x=-0.05,
    y=-0.05,
    z=0,
    a=-PI / 4,
    b=-PI / 4,
    c=-PI / 4,
    hip_upper=0.0,
    hip_lower=0.0,
    thigh_upper=0.0,
    thigh_lower=0.0,
    ankle_upper=0.0,
    ankle_lower=0.0,
    thigh_size=0.0,
    ankle_size=0.0)
