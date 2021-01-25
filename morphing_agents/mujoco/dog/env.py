from morphing_agents.mujoco.dog.designs import sample_centered
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

    def __init__(self,
                 design=DEFAULT_DESIGN,
                 expose_design=True,):
        """Build a Dog environment that has a parametric design for training
        morphology-conditioned agents

        Args:

        design: list
            a list of dog design elements, which are named tuples such as
            [LEG(x=0, y=0, z=0, a=0, b=0, c=0, ...), ...]
        expose_design: bool
            a boolean that indicates whether the design parameters are to
            be concatenated with the observation
        """

        # save the design parameters as a vectorized array
        self.design = np.concatenate(design)
        self.expose_design = expose_design

        # load the base agent xml file
        xml_name = 'base.xml'
        xml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), xml_name)

        # find the agent body and add elements to it
        tree = ET.parse(xml_path)
        torso = tree.find(".//body[@name='torso']")
        actuator = tree.find(".//actuator")

        # iterate through every design element
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
                range=f"{leg.hip_center - leg.hip_range} "
                      f"{leg.hip_center + leg.hip_range}",
                type="hinge")

            ET.SubElement(
                leg_i_body,
                "joint",
                axis="0 1 0",
                name=f"thigh_{i}_joint",
                pos="0.0 0.0 0.0",
                range=f"{leg.thigh_center - leg.thigh_range} "
                      f"{leg.thigh_center + leg.thigh_range}",
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
                range=f"{leg.ankle_center - leg.ankle_range} "
                      f"{leg.ankle_center + leg.ankle_range}",
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

        # make a temporary xml file and write the agent to it
        fd, file_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(file_path)

        # build the mujoco environment
        mujoco_env.MujocoEnv.__init__(self, file_path, 5)
        utils.EzPickle.__init__(self)

        # remove the temporary file
        os.close(fd)
        os.remove(file_path)

    def step(self, a):
        """Step the simulator using the provided actions. The action space
        depends on the number of design elements in the agent

        Args:

        a: np.ndarray
            an array corresponding to actions for each design element

        Returns:

        obs: np.ndarray
            an array corresponding to observations for each design element
        reward: float
            a reward that encourages the agent to run as fast as possible
        done: bool
            a boolean that specifies if the agent has died
        info: dict
            extra statistics for measuring components of the reward
        """

        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))

        forward_reward = (xposafter - xposbefore) / self.dt
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        # only terminate if the simulator generates NaN values
        # we do this because some morphologies may jump or crawl
        # and terminating based on height may cause degenerate behavior
        state = self.state_vector()
        done = not np.isfinite(state).all()
        ob = self._get_obs()

        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        """Get the agent's current observation, which may also include
        the design parameters

        Returns:

        obs: np.ndarray
            an array corresponding to observations for each design element
        """

        obs_list = [
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat]
        if self.expose_design:
            obs_list.append(self.design)
        return np.concatenate(obs_list)

    def reset_model(self):
        """Reset the agent to a default location, with slight noise added
        to each of the agent's joints

        Returns:

        obs: np.ndarray
            an array corresponding to observations for each design element
        """

        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()


class MorphingDogEnv(gym.Wrapper, utils.EzPickle):

    def __init__(self,
                 num_legs=4,
                 fixed_design=None,
                 centered=True,
                 center=DEFAULT_DESIGN,
                 noise_std=0.125,
                 retry_at_fail=False,
                 **kwargs):
        """Wrap around the DogEnv and provide an interface for randomly
        sampling the design every reset

        Args:

        num_legs: int
            the number of legs in the agent, used if fixed_design is None
        fixed_design: list
            a static design that the agent is initialized with every reset
        centered: bool
            if random morphologies are centered around a default morphology
        center: list
            a default morphology centered morphologies are sampled from
        noise_std: float
            a fraction of the design space the noise std takes
        retry_at_fail: bool
            if a new design should be sampled if initialization fails
        """

        self.num_legs = num_legs
        self.fixed_design = fixed_design
        self.centered = centered
        self.center = center
        self.noise_std = noise_std
        self.retry_at_fail = retry_at_fail
        self.kwargs = kwargs
        self.is_initialized = False

        self.reset()
        utils.EzPickle.__init__(self)

    def reset(self, **kwargs):
        """Reset the inner environment and possibly rebuild that environment
        if a new morphology is provided

        Returns:

        obs: np.ndarray
            an array corresponding to observations for each design element
        """

        try:

            if self.fixed_design is None and self.centered:
                self.is_initialized = False
                design = sample_centered(
                    noise_std=self.noise_std, center=self.center)

            elif self.fixed_design is None:
                self.is_initialized = False
                design = sample_uniformly(num_legs=self.num_legs)

            else:
                design = self.fixed_design

            if not self.is_initialized:
                self.is_initialized = True
                gym.Wrapper.__init__(
                    self, DogEnv(design=design, **self.kwargs))

            return self.env.reset(**kwargs)

        except AssertionError as e:

            if self.retry_at_fail:
                return self.reset(**kwargs)

            else:
                raise e
