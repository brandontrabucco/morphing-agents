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
    "thigh_size",
    "ankle_size"])


LEG_UPPER_BOUND = LEG(
    x=0.4,
    y=0.2,
    z=0.2,
    a=180,
    b=180,
    c=180,
    hip_center=180,
    hip_range=45,
    thigh_center=180,
    thigh_range=45,
    ankle_center=180,
    ankle_range=45,
    thigh_size=0.4,
    ankle_size=0.4)


LEG_LOWER_BOUND = LEG(
    x=-0.4,
    y=-0.2,
    z=-0.2,
    a=-180,
    b=-180,
    c=-180,
    hip_center=-180,
    hip_range=5,
    thigh_center=-180,
    thigh_range=5,
    ankle_center=-180,
    ankle_range=5,
    thigh_size=0.2,
    ankle_size=0.2)
