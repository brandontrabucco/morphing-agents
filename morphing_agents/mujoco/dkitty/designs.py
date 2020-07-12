from morphing_agents.mujoco.dkitty.elements import LEG
from morphing_agents.mujoco.dkitty.elements import LEG_UPPER_BOUND
from morphing_agents.mujoco.dkitty.elements import LEG_LOWER_BOUND
from typing import List, NamedTuple
import numpy as np
import random


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
        y=0.122,
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


CURATED_DESIGNS = [
    DEFAULT_DESIGN,
    [LEG(x=0.0918536458535748, y=-0.09255236534560901, z=0.0, a=0.8353093265804032, b=-2.5962518162013177,
         c=1.3168499164626946, hip_center=-1.3366355767132567, hip_range=0.2139926263961598,
         thigh_center=0.8030804713374127, thigh_range=0.12579937656972343, ankle_center=1.0554621796782984,
         ankle_range=0.3840714465580822, thigh_size=0.12481724976653479, ankle_size=0.13256750245509402),
     LEG(x=0.09819165447852674, y=-0.05316155478221499, z=0.0, a=-2.1197510272771005, b=-2.324962621320358,
         c=0.7247900863229795, hip_center=2.7462209420062154, hip_range=0.7850165826537762,
         thigh_center=2.61019729010098, thigh_range=0.11259108922343891, ankle_center=2.717670285998757,
         ankle_range=0.04366712179677797, thigh_size=0.10642200908214423, ankle_size=0.164411624378425),
     LEG(x=-0.007974277893839893, y=0.01108069525118996, z=0.0, a=0.8403897718315272, b=-0.9401639784150309,
         c=-2.5243025493425706, hip_center=2.0418933230896075, hip_range=0.7394256460942709,
         thigh_center=-0.49163356076703746, thigh_range=0.7779744903851609, ankle_center=0.9155332293308645,
         ankle_range=0.33107507994519525, thigh_size=0.09955960650505095, ankle_size=0.16167361370767785),
     LEG(x=-0.09671812690951426, y=-0.08050365754439688, z=0.0, a=-2.414409999022001, b=-2.0813717012589565,
         c=-0.9699402075835453, hip_center=1.3440637072713262, hip_range=0.3808626779507646,
         thigh_center=1.5358422817346291, thigh_range=0.10438948987789577, ankle_center=-3.134763793746379,
         ankle_range=0.15978536528340748, thigh_size=0.12585997446689673, ankle_size=0.10120244333069603)],
    [LEG(x=0.010282905102950796, y=0.04796540518588688, z=0.0, a=1.2655434354142123, b=1.5265015371294925,
         c=-2.34046625641939, hip_center=2.3955574906053636, hip_range=0.6854683311062671,
         thigh_center=-0.5039904639251356, thigh_range=0.44188362842590384, ankle_center=2.849718807366984,
         ankle_range=0.32692348571779806, thigh_size=0.13738517556294766, ankle_size=0.17187617410401662),
     LEG(x=-0.08320425406903763, y=0.016094550626303522, z=0.0, a=-0.43421590631105333, b=3.0867524909462825,
         c=0.23286314846740863, hip_center=1.3024861794110931, hip_range=0.5602345295421548,
         thigh_center=0.016679514137927676, thigh_range=0.032161837768302184, ankle_center=3.135526812222941,
         ankle_range=0.7648598036137605, thigh_size=0.15411911993641392, ankle_size=0.13356917497341114),
     LEG(x=0.05492909339493682, y=0.04719837253839104, z=0.0, a=-0.16632864758121224, b=-2.9616882911420106,
         c=1.8006238888600565, hip_center=2.1288682442766236, hip_range=0.5711949721724231,
         thigh_center=0.469719561991798, thigh_range=0.6116094523595166, ankle_center=-0.30653441502929146,
         ankle_range=0.40318374416467073, thigh_size=0.11731753396614504, ankle_size=0.12481009734211646),
     LEG(x=-0.04269576436531924, y=0.07988511663161438, z=0.0, a=1.294955828318642, b=1.5267481170780695,
         c=-2.8440380937567724, hip_center=1.1105983676582065, hip_range=0.49372097728991615,
         thigh_center=1.9817464820538842, thigh_range=0.1382256496041114, ankle_center=-1.7458455992961324,
         ankle_range=0.5635025721357284, thigh_size=0.18839981754018215, ankle_size=0.11166351626226828)],
    [LEG(x=0.03050293485515468, y=0.034493211254673334, z=0.0, a=2.731863488251663, b=-1.7611810500811609,
         c=2.717890395106859, hip_center=0.7191469795505028, hip_range=0.39752892332898776,
         thigh_center=2.6951480399746295, thigh_range=0.12359197712744416, ankle_center=-2.346857743867222,
         ankle_range=0.30988421251055315, thigh_size=0.12914510598302753, ankle_size=0.1904853084466563),
     LEG(x=0.060923435530762876, y=-0.05607283796784166, z=0.0, a=1.7351366906497345, b=1.0154631234243165,
         c=0.2016295989491499, hip_center=-0.8113303042888105, hip_range=0.24904538035680077,
         thigh_center=1.1620382071210225, thigh_range=0.2925229552646683, ankle_center=-1.8917694374481158,
         ankle_range=0.3359438017847097, thigh_size=0.1523593102212865, ankle_size=0.18150126396779473),
     LEG(x=0.02077757683640162, y=0.08546093668888421, z=0.0, a=-1.3053919434997328, b=0.21144896746342345,
         c=0.1980019902202006, hip_center=1.1993137814533394, hip_range=0.2880589799729674,
         thigh_center=-2.9396691312072045, thigh_range=0.09132873850092031, ankle_center=2.000946664566002,
         ankle_range=0.520140509450801, thigh_size=0.11168886794963653, ankle_size=0.13581439215259355),
     LEG(x=0.0835223687541215, y=0.03982279068870176, z=0.0, a=-0.6396880575500989, b=2.5029573497899813,
         c=1.7443706308610496, hip_center=-2.708274565029213, hip_range=0.2896815925216841,
         thigh_center=0.8213167048462484, thigh_range=0.3540266429872243, ankle_center=0.8029226987749758,
         ankle_range=0.6681472785233653, thigh_size=0.1115360730271178, ankle_size=0.1794822507991541)]]


def sample_uniformly(num_legs=4) -> List[NamedTuple]:
    return [LEG(*np.random.uniform(
        low=LEG_LOWER_BOUND, high=LEG_UPPER_BOUND))
        for n in range(num_legs)]


def sample_curated() -> List[NamedTuple]:
    return random.choice(CURATED_DESIGNS)
