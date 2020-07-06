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
    "thigh_size",
    "ankle_size"])

LEG_LOWER_BOUND = LEG(
    x=-0.4,
    y=-0.2,
    z=0,
    a=0,
    b=0,
    c=10,
    hip_upper=20,
    hip_lower=-40,
    thigh_upper=65,
    thigh_lower=5,
    ankle_upper=-70,
    ankle_lower=-130,
    thigh_size=0.2,
    ankle_size=0.2)


LEG_UPPER_BOUND = LEG(
    x=0.4,
    y=0.2,
    z=0,
    a=0,
    b=0,
    c=-10,
    hip_upper=40,
    hip_lower=-20,
    thigh_upper=85,
    thigh_lower=25,
    ankle_upper=-50,
    ankle_lower=-110,
    thigh_size=0.4,
    ankle_size=0.4)
