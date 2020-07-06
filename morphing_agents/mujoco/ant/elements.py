from collections import namedtuple


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
    "hip_size",
    "thigh_size",
    "ankle_size"])

LEG_UPPER_BOUND = LEG(
    x=0.1,
    y=0.1,
    z=0.1,
    a=180,
    b=180,
    c=180,
    hip_upper=60,
    hip_lower=0,
    thigh_upper=60,
    thigh_lower=0,
    ankle_upper=120,
    ankle_lower=60,
    hip_size=0.4,
    thigh_size=0.4,
    ankle_size=0.8)


LEG_LOWER_BOUND = LEG(
    x=-0.1,
    y=-0.1,
    z=-0.1,
    a=-180,
    b=-180,
    c=-180,
    hip_upper=0,
    hip_lower=-60,
    thigh_upper=0,
    thigh_lower=-60,
    ankle_upper=60,
    ankle_lower=0,
    hip_size=0.1,
    thigh_size=0.1,
    ankle_size=0.2)
