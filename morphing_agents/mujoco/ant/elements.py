from collections import namedtuple


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
    hip_center=180,
    hip_range=45,
    thigh_center=180,
    thigh_range=45,
    ankle_center=180,
    ankle_range=45,
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
    hip_center=-180,
    hip_range=5,
    thigh_center=-180,
    thigh_range=5,
    ankle_center=-180,
    ankle_range=5,
    hip_size=0.1,
    thigh_size=0.1,
    ankle_size=0.2)
