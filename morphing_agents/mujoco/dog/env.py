from morphing_agents.mujoco.dog.designs import sample_uniformly
from morphing_agents.mujoco.dog.designs import DEFAULT_DESIGN
from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
import tempfile
import xml.etree.ElementTree as ET
import os
import gym


class DogEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, design=DEFAULT_DESIGN, expose_design=True):
        self.design = np.concatenate(design)
        self.expose_design = expose_design

        xml_name = 'base.xml'
        xml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), xml_name)

        tree = ET.parse(xml_path)
        torso = tree.find(".//body[@name='torso']")
        actuator = tree.find(".//actuator")

        for i, leg in enumerate(design):

            leg_i_body = ET.SubElement(
                torso,
                "body",
                name=f"leg_{i}_body",
                pos=f"{leg.x} {leg.y} {leg.z}",
                euler=f"{leg.a} {leg.b} {leg.c}")

            ET.SubElement(
                leg_i_body,
                "joint",
                axis="1 0 0",
                name=f"hip_{i}_joint",
                pos="0.0 0.0 0.0",
                range=f"{leg.hip_center - leg.hip_range} {leg.hip_center + leg.hip_range}",
                type="hinge")

            ET.SubElement(
                leg_i_body,
                "joint",
                axis="0 1 0",
                name=f"thigh_{i}_joint",
                pos="0.0 0.0 0.0",
                range=f"{leg.thigh_center - leg.thigh_range} {leg.thigh_center + leg.thigh_range}",
                type="hinge")

            ET.SubElement(
                leg_i_body,
                "geom",
                fromto=f"0.0 0.0 -{leg.thigh_size} 0.0 0.0 0.0",
                name=f"thigh_{i}_geom",
                size="0.08",
                type="capsule")

            ankle_i_body = ET.SubElement(
                leg_i_body,
                "body",
                pos=f"0 0 -{leg.thigh_size}",
                name=f"ankle_{i}_geom")

            ET.SubElement(
                ankle_i_body,
                "joint",
                axis="0 1 0",
                name=f"ankle_{i}_joint",
                pos="0.0 0.0 0.0",
                range=f"{leg.ankle_center - leg.ankle_range} {leg.ankle_center + leg.ankle_range}",
                type="hinge")

            ET.SubElement(
                ankle_i_body,
                "geom",
                fromto=f"0.0 0.0 -{leg.ankle_size} 0.0 0.0 0.0",
                name=f"ankle_{i}_geom",
                size="0.08",
                type="capsule")

            ET.SubElement(
                actuator,
                "motor",
                ctrllimited="true",
                ctrlrange="-1.0 1.0",
                joint=f"hip_{i}_joint",
                gear="150")

            ET.SubElement(
                actuator,
                "motor",
                ctrllimited="true",
                ctrlrange="-1.0 1.0",
                joint=f"thigh_{i}_joint",
                gear="150")

            ET.SubElement(
                actuator,
                "motor",
                ctrllimited="true",
                ctrlrange="-1.0 1.0",
                joint=f"ankle_{i}_joint",
                gear="150")

        _, file_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(file_path)

        mujoco_env.MujocoEnv.__init__(self, file_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))

        forward_reward = (xposafter - xposbefore) / self.dt
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        state = self.state_vector()
        notdone = np.isfinite(state).all() and 0.1 <= state[2] <= 1.75
        done = not notdone
        ob = self._get_obs()

        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        obs_list = [
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat]
        if self.expose_design:
            obs_list.append(self.design)
        return np.concatenate(obs_list)

    def reset_model(self):
        qpos = self.init_qpos + \
               self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + \
               self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class MorphingDogEnv(gym.Wrapper, utils.EzPickle):

    def __init__(self, num_legs=4, fixed_design=None, **kwargs):
        self.num_legs = num_legs
        self.fixed_design = fixed_design
        self.kwargs = kwargs
        self.reset()
        utils.EzPickle.__init__(self)

    def reset(self, **kwargs):
        try:
            design = sample_uniformly(num_legs=self.num_legs) \
                if self.fixed_design is None else self.fixed_design
            gym.Wrapper.__init__(self, DogEnv(design=design, **self.kwargs))
            return self.env.reset(**kwargs)
        except AssertionError:
            return self.reset(**kwargs)
