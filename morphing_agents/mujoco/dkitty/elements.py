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
    "hip_center",
    "hip_range",
    "thigh_center",
    "thigh_range",
    "ankle_center",
    "ankle_range",
    "thigh_size",
    "ankle_size"])


LEG_UPPER_BOUND = LEG(
    x=0.1,
    y=0.1,
    z=0,
    a=PI,
    b=PI,
    c=PI,
    hip_center=PI,
    hip_range=PI / 4,
    thigh_center=PI,
    thigh_range=PI / 4,
    ankle_center=PI,
    ankle_range=PI / 4,
    thigh_size=0.1965,
    ankle_size=0.1945)


LEG_LOWER_BOUND = LEG(
    x=-0.1,
    y=-0.1,
    z=0,
    a=-PI,
    b=-PI,
    c=-PI,
    hip_center=-PI,
    hip_range=0,
    thigh_center=-PI,
    thigh_range=0,
    ankle_center=-PI,
    ankle_range=0,
    thigh_size=0.0965,
    ankle_size=0.0945)
