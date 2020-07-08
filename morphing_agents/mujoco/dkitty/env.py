from morphing_agents.mujoco.dkitty.designs import DEFAULT_DESIGN
from morphing_agents.mujoco.dkitty.designs import sample_curated
from morphing_agents.mujoco.dkitty.designs import sample_uniformly
from morphing_agents.mujoco.utils import load_xml_tree
from typing import Dict, Optional, Sequence, Tuple, Union
from robel.dkitty.base_env import BaseDKittyEnv
from robel.dkitty.base_env import DEFAULT_DKITTY_CALIBRATION_MAP
from robel.components.robot import RobotComponentBuilder
from robel.components.tracking import TrackerComponentBuilder, TrackerState
from robel.simulation.randomize import SimRandomizer
from robel.utils.configurable import configurable
from robel.utils.math_utils import calculate_cosine
from robel.utils.resources import get_asset_path
from gym import utils
import abc
import numpy as np
import os
import tempfile
import collections
import xml.etree.ElementTree as ET
import gym


DKITTY_ASSET_PATH = 'robel/dkitty/assets/dkitty_walk-v0.xml'


DEFAULT_OBSERVATION_KEYS = (
    'root_pos',
    'root_euler',
    'kitty_qpos',
    'root_vel',
    'root_angular_vel',
    'kitty_qvel',
    'last_action',
    'upright',
    'heading',
    'target_error',
    'design')


PI = np.pi


class DKittyEnv(BaseDKittyEnv):
    """Base environment for all DKitty morphology tasks."""

    def step(self, a):
        """Hack to get the env to work with softlearning, which assumes
        the info dict contains only scalars"""

        o, r, d, i = super().step(a)
        i = {key: value for key, value in i.items() if np.isscalar(value)}
        return o, r, d, i

    def __init__(self,
                 sim_model,
                 design=DEFAULT_DESIGN,
                 **kwargs):
        self.design = np.concatenate(design)
        self._legs = design

        assert len(design) == 4, "dkitty must always have 4 legs"
        tree = ET.ElementTree(element=load_xml_tree(sim_model))
        root = tree.getroot()

        # fix when many sims are running
        ET.SubElement(root, "size", njmax="1200")

        # modify settings for Front Right Leg
        spec = design[0]
        leg = tree.find(".//body[@name='A:FR10']")

        leg.attrib['pos'] = f"{spec.x} {spec.y} {spec.z}"
        leg.attrib['euler'] = f"{spec.a} {spec.b} {spec.c}"
        leg_joint = tree.find(".//joint[@name='A:FRJ10']")
        leg_joint.attrib['range'] = f"{spec.hip_center - spec.hip_range} " \
                                    f"{spec.hip_center + spec.hip_range}"
        leg_ctrl = tree.find(".//position[@name='A:FRJ10']")
        leg_ctrl.attrib['ctrlrange'] = f"{spec.hip_center - spec.hip_range} " \
                                       f"{spec.hip_center + spec.hip_range}"

        thigh = tree.find(".//body[@name='A:FR11']")
        thigh.attrib['pos'] = f"0 0 {spec.thigh_size}"
        thigh_joint = tree.find(".//joint[@name='A:FRJ11']")
        thigh_joint.attrib['range'] = f"{spec.thigh_center - spec.hip_range} " \
                                      f"{spec.thigh_center + spec.hip_range}"
        thigh_ctrl = tree.find(".//position[@name='A:FRJ11']")
        thigh_ctrl.attrib['ctrlrange'] = f"{spec.thigh_center - spec.hip_range} " \
                                         f"{spec.thigh_center + spec.hip_range}"

        ankle = tree.find(".//body[@name='A:FR12']")
        ankle.attrib['pos'] = f"0 0 {spec.ankle_size}"
        ankle_joint = tree.find(".//joint[@name='A:FRJ12']")
        ankle_joint.attrib['range'] = f"{spec.ankle_center - spec.ankle_range} " \
                                      f"{spec.ankle_center + spec.ankle_range}"
        ankle_ctrl = tree.find(".//position[@name='A:FRJ12']")
        ankle_ctrl.attrib['ctrlrange'] = f"{spec.ankle_center - spec.ankle_range} " \
                                         f"{spec.ankle_center + spec.ankle_range}"

        # modify settings for Front Left Leg
        spec = design[1]
        leg = tree.find(".//body[@name='A:FL20']")

        leg.attrib['pos'] = f"{spec.x} {spec.y} {spec.z}"
        leg.attrib['euler'] = f"{spec.a} {spec.b} {spec.c}"
        leg_joint = tree.find(".//joint[@name='A:FLJ20']")
        leg_joint.attrib['range'] = f"{spec.hip_center - spec.hip_range} " \
                                    f"{spec.hip_center + spec.hip_range}"
        leg_ctrl = tree.find(".//position[@name='A:FLJ20']")
        leg_ctrl.attrib['ctrlrange'] = f"{spec.hip_center - spec.hip_range} " \
                                       f"{spec.hip_center + spec.hip_range}"

        thigh = tree.find(".//body[@name='A:FL21']")
        thigh.attrib['pos'] = f"0 0 {spec.thigh_size}"
        thigh_joint = tree.find(".//joint[@name='A:FLJ21']")
        thigh_joint.attrib['range'] = f"{spec.thigh_center - spec.hip_range} " \
                                      f"{spec.thigh_center + spec.hip_range}"
        thigh_ctrl = tree.find(".//position[@name='A:FLJ21']")
        thigh_ctrl.attrib['ctrlrange'] = f"{spec.thigh_center - spec.hip_range} " \
                                         f"{spec.thigh_center + spec.hip_range}"

        ankle = tree.find(".//body[@name='A:FL22']")
        ankle.attrib['pos'] = f"0 0 {spec.ankle_size}"
        ankle_joint = tree.find(".//joint[@name='A:FLJ22']")
        ankle_joint.attrib['range'] = f"{spec.ankle_center - spec.ankle_range} " \
                                      f"{spec.ankle_center + spec.ankle_range}"
        ankle_ctrl = tree.find(".//position[@name='A:FLJ22']")
        ankle_ctrl.attrib['ctrlrange'] = f"{spec.ankle_center - spec.ankle_range} " \
                                         f"{spec.ankle_center + spec.ankle_range}"

        # modify settings for Back Left Leg
        spec = design[2]
        leg = tree.find(".//body[@name='A:BL30']")

        leg.attrib['pos'] = f"{spec.x} {spec.y} {spec.z}"
        leg.attrib['euler'] = f"{spec.a} {spec.b} {spec.c}"
        leg_joint = tree.find(".//joint[@name='A:BLJ30']")
        leg_joint.attrib['range'] = f"{spec.hip_center - spec.hip_range} " \
                                    f"{spec.hip_center + spec.hip_range}"
        leg_ctrl = tree.find(".//position[@name='A:BLJ30']")
        leg_ctrl.attrib['ctrlrange'] = f"{spec.hip_center - spec.hip_range} " \
                                       f"{spec.hip_center + spec.hip_range}"

        thigh = tree.find(".//body[@name='A:BL31']")
        thigh.attrib['pos'] = f"0 0 {spec.thigh_size}"
        thigh_joint = tree.find(".//joint[@name='A:BLJ31']")
        thigh_joint.attrib['range'] = f"{spec.thigh_center - spec.hip_range} " \
                                      f"{spec.thigh_center + spec.hip_range}"
        thigh_ctrl = tree.find(".//position[@name='A:BLJ31']")
        thigh_ctrl.attrib['ctrlrange'] = f"{spec.thigh_center - spec.hip_range} " \
                                         f"{spec.thigh_center + spec.hip_range}"

        ankle = tree.find(".//body[@name='A:BL32']")
        ankle.attrib['pos'] = f"0 0 {spec.ankle_size}"
        ankle_joint = tree.find(".//joint[@name='A:BLJ32']")
        ankle_joint.attrib['range'] = f"{spec.ankle_center - spec.ankle_range} " \
                                      f"{spec.ankle_center + spec.ankle_range}"
        ankle_ctrl = tree.find(".//position[@name='A:BLJ32']")
        ankle_ctrl.attrib['ctrlrange'] = f"{spec.ankle_center - spec.ankle_range} " \
                                         f"{spec.ankle_center + spec.ankle_range}"

        # modify settings for Back Right Leg
        spec = design[3]
        leg = tree.find(".//body[@name='A:BR40']")

        leg.attrib['pos'] = f"{spec.x} {spec.y} {spec.z}"
        leg.attrib['euler'] = f"{spec.a} {spec.b} {spec.c}"
        leg_joint = tree.find(".//joint[@name='A:BRJ40']")
        leg_joint.attrib['range'] = f"{spec.hip_center - spec.hip_range} " \
                                    f"{spec.hip_center + spec.hip_range}"
        leg_ctrl = tree.find(".//position[@name='A:BRJ40']")
        leg_ctrl.attrib['ctrlrange'] = f"{spec.hip_center - spec.hip_range} " \
                                       f"{spec.hip_center + spec.hip_range}"

        thigh = tree.find(".//body[@name='A:BR41']")
        thigh.attrib['pos'] = f"0 0 {spec.thigh_size}"
        thigh_joint = tree.find(".//joint[@name='A:BRJ41']")
        thigh_joint.attrib['range'] = f"{spec.thigh_center - spec.hip_range} " \
                                      f"{spec.thigh_center + spec.hip_range}"
        thigh_ctrl = tree.find(".//position[@name='A:BRJ41']")
        thigh_ctrl.attrib['ctrlrange'] = f"{spec.thigh_center - spec.hip_range} " \
                                         f"{spec.thigh_center + spec.hip_range}"

        ankle = tree.find(".//body[@name='A:BR42']")
        ankle.attrib['pos'] = f"0 0 {spec.ankle_size}"
        ankle_joint = tree.find(".//joint[@name='A:BRJ42']")
        ankle_joint.attrib['range'] = f"{spec.ankle_center - spec.ankle_range} " \
                                      f"{spec.ankle_center + spec.ankle_range}"
        ankle_ctrl = tree.find(".//position[@name='A:BRJ42']")
        ankle_ctrl.attrib['ctrlrange'] = f"{spec.ankle_center - spec.ankle_range} " \
                                         f"{spec.ankle_center + spec.ankle_range}"

        _, file_path = tempfile.mkstemp(text=True,
                                        suffix='.xml',
                                        dir=os.path.dirname(sim_model))
        tree.write(file_path)

        super().__init__(file_path, **kwargs)

    # Copyright 2019 The ROBEL Authors.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    def _configure_robot(self, builder: RobotComponentBuilder):
        """Configures the robot component."""
        builder.add_group(
            'dkitty',
            actuator_indices=range(12),
            qpos_indices=range(6, 18),
            qpos_range=[
                # FR
                (self._legs[0].hip_center - self._legs[0].hip_range,
                 self._legs[0].hip_center + self._legs[0].hip_range),
                (self._legs[0].thigh_center - self._legs[0].thigh_range,
                 self._legs[0].thigh_center + self._legs[0].thigh_range),
                (self._legs[0].ankle_center - self._legs[0].ankle_range,
                 self._legs[0].ankle_center + self._legs[0].ankle_range),
                # FL
                (self._legs[1].hip_center - self._legs[1].hip_range,
                 self._legs[1].hip_center + self._legs[1].hip_range),
                (self._legs[1].thigh_center - self._legs[1].thigh_range,
                 self._legs[1].thigh_center + self._legs[1].thigh_range),
                (self._legs[1].ankle_center - self._legs[1].ankle_range,
                 self._legs[1].ankle_center + self._legs[1].ankle_range),
                # BL
                (self._legs[2].hip_center - self._legs[2].hip_range,
                 self._legs[2].hip_center + self._legs[2].hip_range),
                (self._legs[2].thigh_center - self._legs[2].thigh_range,
                 self._legs[2].thigh_center + self._legs[2].thigh_range),
                (self._legs[2].ankle_center - self._legs[2].ankle_range,
                 self._legs[2].ankle_center + self._legs[2].ankle_range),
                # BR
                (self._legs[3].hip_center - self._legs[3].hip_range,
                 self._legs[3].hip_center + self._legs[3].hip_range),
                (self._legs[3].thigh_center - self._legs[3].thigh_range,
                 self._legs[3].thigh_center + self._legs[3].thigh_range),
                (self._legs[3].ankle_center - self._legs[3].ankle_range,
                 self._legs[3].ankle_center + self._legs[3].ankle_range),
            ],
            qvel_range=[(-PI, PI)] * 12,
        )
        if self._sim_observation_noise is not None:
            builder.update_group(
                'dkitty', sim_observation_noise=self._sim_observation_noise)
        # If a device path is given, set the motor IDs and calibration map.
        if self._device_path is not None:
            builder.set_dynamixel_device_path(self._device_path)
            builder.set_hardware_calibration_map(DEFAULT_DKITTY_CALIBRATION_MAP)
            builder.update_group(
                'dkitty',
                motor_ids=[10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42])


class DKittyUprightEnv(DKittyEnv):
    """Base environment for D'Kitty tasks where the D'Kitty must be upright."""

    def __init__(
            self,
            *args,
            torso_tracker_id: Optional[Union[str, int]] = None,
            upright_obs_key: str = 'upright',
            upright_threshold: float = 0,  # cos(90deg).
            upright_reward: float = 1,
            falling_reward: float = -100,
            **kwargs):
        """Initializes the environment.
        Args:
            torso_tracker_id: The device index or serial of the tracking device
                for the D'Kitty torso.
            upright_obs_key: The observation key for uprightnedness.
            upright_threshold: The threshold (in [0, 1]) above which the D'Kitty
                is considered to be upright. If the cosine similarity of the
                D'Kitty's z-axis with the global z-axis is below this threshold,
                the D'Kitty is considered to have fallen.
            upright_reward: The reward multiplier for uprightedness.
            falling_reward: The reward multipler for falling.
            **kwargs: Arguemnts to pass to BaseDKittyEnv.
        """
        self._torso_tracker_id = torso_tracker_id
        super().__init__(*args, **kwargs)

        self._upright_obs_key = upright_obs_key
        self._upright_threshold = upright_threshold
        self._upright_reward = upright_reward
        self._falling_reward = falling_reward

    def _configure_tracker(self, builder: TrackerComponentBuilder):
        """Configures the tracker component."""
        super()._configure_tracker(builder)
        builder.add_tracker_group(
            'torso',
            hardware_tracker_id=self._torso_tracker_id,
            sim_params=dict(
                element_name='torso',
                element_type='joint',
                qpos_indices=range(6),
            ),
            hardware_params=dict(
                is_origin=True,
                # tracked_rotation_offset=(-1.57, 0, 1.57),
            ))

    def _get_upright_obs(self,
                         torso_track_state: TrackerState) -> Dict[str, float]:
        """Returns a dictionary of uprightedness observations."""
        return {self._upright_obs_key: torso_track_state.rot[2, 2]}

    def _get_upright_rewards(
            self,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        upright = obs_dict[self._upright_obs_key]
        return {
            'upright': (
                self._upright_reward * (upright - self._upright_threshold) /
                (1 - self._upright_threshold)),
            'falling': self._falling_reward *
                       (upright < self._upright_threshold),
        }

    def get_done(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Returns whether the episode should terminate."""
        return obs_dict[self._upright_obs_key] < self._upright_threshold


class DKittyWalk(DKittyUprightEnv, metaclass=abc.ABCMeta):
    """Shared logic for Morphing DKitty walk tasks."""

    def __init__(self,
                 asset_path: str = DKITTY_ASSET_PATH,
                 observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
                 target_tracker_id: Optional[Union[str, int]] = None,
                 heading_tracker_id: Optional[Union[str, int]] = None,
                 frame_skip: int = 40,
                 upright_threshold: float = 0.9,
                 upright_reward: float = 1,
                 falling_reward: float = -500,
                 expose_design=True,
                 **kwargs):
        """Initializes the environment.
        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            target_tracker_id: The device index or serial of the tracking device
                for the target location.
            heading_tracker_id: The device index or serial of the tracking
                device for the heading direction. This defaults to the target
                tracker.
            frame_skip: The number of simulation steps per environment step.
            upright_threshold: The threshold (in [0, 1]) above which the D'Kitty
                is considered to be upright. If the cosine similarity of the
                D'Kitty's z-axis with the global z-axis is below this threshold,
                the D'Kitty is considered to have fallen.
            upright_reward: The reward multiplier for uprightedness.
            falling_reward: The reward multipler for falling.
        """
        self._target_tracker_id = target_tracker_id
        self._heading_tracker_id = heading_tracker_id
        if self._heading_tracker_id is None:
            self._heading_tracker_id = self._target_tracker_id
        observation_keys = list(observation_keys)
        if not expose_design:
            observation_keys.remove("design")

        super().__init__(
            sim_model=get_asset_path(asset_path),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            upright_threshold=upright_threshold,
            upright_reward=upright_reward,
            falling_reward=falling_reward,
            **kwargs)

        self._initial_target_pos = np.zeros(3)
        self._initial_heading_pos = None

    def _configure_tracker(self, builder: TrackerComponentBuilder):
        """Configures the tracker component."""
        super()._configure_tracker(builder)
        builder.add_tracker_group(
            'target',
            hardware_tracker_id=self._target_tracker_id,
            sim_params=dict(
                element_name='target',
                element_type='site',
            ),
            mimic_xy_only=True)
        builder.add_tracker_group(
            'heading',
            hardware_tracker_id=self._heading_tracker_id,
            sim_params=dict(
                element_name='heading',
                element_type='site',
            ),
            mimic_xy_only=True)

    def _reset(self):
        """Resets the environment."""
        self._reset_dkitty_standing()

        # If no heading is provided, head towards the target.
        target_pos = self._initial_target_pos
        heading_pos = self._initial_heading_pos
        if heading_pos is None:
            heading_pos = target_pos

        # Set the tracker locations.
        self.tracker.set_state({
            'torso': TrackerState(pos=np.zeros(3), rot=np.identity(3)),
            'target': TrackerState(pos=target_pos),
            'heading': TrackerState(pos=heading_pos),
        })

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        # Apply action.
        self.robot.step({
            'dkitty': action,
        })

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        """Returns the current observation of the environment.
        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """
        robot_state = self.robot.get_state('dkitty')
        target_state, heading_state, torso_track_state = self.tracker.get_state(
            ['target', 'heading', 'torso'])

        target_xy = target_state.pos[:2]
        kitty_xy = torso_track_state.pos[:2]

        # Get the heading of the torso (the y-axis).
        current_heading = torso_track_state.rot[:2, 1]

        # Get the direction towards the heading location.
        desired_heading = heading_state.pos[:2] - kitty_xy

        # Calculate the alignment of the heading with the desired direction.
        heading = calculate_cosine(current_heading, desired_heading)

        return collections.OrderedDict((
            # Add observation terms relating to being upright.
            *self._get_upright_obs(torso_track_state).items(),
            ('root_pos', torso_track_state.pos),
            ('root_euler', torso_track_state.rot_euler),
            ('root_vel', torso_track_state.vel),
            ('root_angular_vel', torso_track_state.angular_vel),
            ('kitty_qpos', robot_state.qpos),
            ('kitty_qvel', robot_state.qvel),
            ('last_action', self._get_last_action()),
            ('heading', heading),
            ('target_pos', target_xy),
            ('target_error', target_xy - kitty_xy),
            ('design', self.design),
        ))

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        target_xy_dist = np.linalg.norm(obs_dict['target_error'])
        heading = obs_dict['heading']

        reward_dict = collections.OrderedDict((
            # Add reward terms for being upright.
            *self._get_upright_rewards(obs_dict).items(),
            # Reward for proximity to the target.
            ('target_dist_cost', -4 * target_xy_dist),
            # Heading - 1 @ cos(0) to 0 @ cos(25deg).
            ('heading', 2 * (heading - 0.9) / 0.1),
            # Bonus
            ('bonus_small', 5 * ((target_xy_dist < 0.5) + (heading > 0.9))),
            ('bonus_big', 10 * (target_xy_dist < 0.5) * (heading > 0.9)),
        ))
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        return collections.OrderedDict((
            ('points', -np.linalg.norm(obs_dict['target_error'])),
            ('success', reward_dict['bonus_big'] > 0.0),
        ))


@configurable(pickleable=True)
class DKittyWalkFixed(DKittyWalk):
    """Walk straight towards a fixed location."""

    def _reset(self):
        """Resets the environment."""
        target_dist = 2.0
        target_theta = np.pi / 2  # Point towards y-axis
        self._initial_target_pos = target_dist * np.array([
            np.cos(target_theta), np.sin(target_theta), 0
        ])
        super()._reset()


@configurable(pickleable=True)
class DKittyWalkRandom(DKittyWalk):
    """Walk straight towards a random location."""

    def __init__(
            self,
            *args,
            target_distance_range: Tuple[float, float] = (1.0, 2.0),
            # +/- 60deg
            target_angle_range: Tuple[float, float] = (-np.pi / 3, np.pi / 3),
            **kwargs):
        """Initializes the environment.
        Args:
            target_distance_range: The range in which to sample the target
                distance.
            target_angle_range: The range in which to sample the angle between
                the initial D'Kitty heading and the target.
        """
        super().__init__(*args, **kwargs)
        self._target_distance_range = target_distance_range
        self._target_angle_range = target_angle_range

    def _reset(self):
        """Resets the environment."""
        target_dist = self.np_random.uniform(*self._target_distance_range)
        # Offset the angle by 90deg since D'Kitty looks towards +y-axis.
        target_theta = np.pi / 2 + self.np_random.uniform(
            *self._target_angle_range)
        self._initial_target_pos = target_dist * np.array([
            np.cos(target_theta), np.sin(target_theta), 0
        ])
        super()._reset()


@configurable(pickleable=True)
class DKittyWalkRandomDynamics(DKittyWalkRandom):
    """Walk straight towards a random location."""

    def __init__(self,
                 *args,
                 sim_observation_noise: Optional[float] = 0.05,
                 **kwargs):
        super().__init__(
            *args, sim_observation_noise=sim_observation_noise, **kwargs)
        self._randomizer = SimRandomizer(self)
        self._dof_indices = (
            self.robot.get_config('dkitty').qvel_indices.tolist())

    def _reset(self):
        """Resets the environment."""
        # Randomize joint dynamics.
        self._randomizer.randomize_dofs(
            self._dof_indices,
            all_same=True,
            damping_range=(0.1, 0.2),
            friction_loss_range=(0.001, 0.005),
        )
        self._randomizer.randomize_actuators(
            all_same=True,
            kp_range=(2.8, 3.2),
        )
        # Randomize friction on all geoms in the scene.
        self._randomizer.randomize_geoms(
            all_same=True,
            friction_slide_range=(0.8, 1.2),
            friction_spin_range=(0.003, 0.007),
            friction_roll_range=(0.00005, 0.00015),
        )
        # Generate a random height field.
        self._randomizer.randomize_global(
            total_mass_range=(1.6, 2.0),
            height_field_range=(0, 0.05),
        )
        self.sim_scene.upload_height_field(0)
        super()._reset()


class MorphingDKittyEnv(gym.Wrapper, utils.EzPickle):

    def __init__(self,
                 num_legs=4,
                 fixed_design=None,
                 curated=True,
                 **kwargs):
        self.num_legs = num_legs
        self.fixed_design = fixed_design
        self.curated = curated
        self.kwargs = kwargs
        self.reset()
        utils.EzPickle.__init__(self)

    def reset(self, **kwargs):
        try:
            if self.fixed_design is None and self.curated:
                design = sample_curated()
            elif self.fixed_design is None:
                design = sample_uniformly(num_legs=self.num_legs)
            else:
                design = self.fixed_design
            gym.Wrapper.__init__(self, DKittyWalkFixed(design=design, **self.kwargs))
            return self.env.reset(**kwargs)
        except AssertionError:
            return self.reset(**kwargs)
