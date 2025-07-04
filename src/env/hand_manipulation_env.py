# src/env/hand_manipulation_env.py
import gymnasium as gym
from gymnasium import spaces
from dm_control import mujoco
import numpy as np

# Import dependencies
from src.robot.base_robot import BaseRobot
from src.objects.base_object import BaseObject
from src.env.observation_builder import ObservationBuilder
from src.env.reward_composer import RewardComposer

class HandManipulationEnv(gym.Env):
    """
    Gymnasium environment for Dex Hand object manipulation.
    Orchestrates the robot, object, observation, and reward components.
    """
    def __init__(self, physics: mujoco.Physics,
                 robot: BaseRobot,
                 manipulable_object: BaseObject,
                 observation_builder: ObservationBuilder,
                 reward_composer: RewardComposer,
                 env_config: dict):

        super().__init__()
        self._physics = physics
        self._model = physics.model
        self._robot = robot
        self._object = manipulable_object
        self._observation_builder = observation_builder
        self._reward_composer = reward_composer
        self._env_config = env_config

        self.control_timestep = self._env_config["control_timestep"]
        self.episode_duration = self._env_config["episode_duration"]

        # Define Observation Space based on the ObservationBuilder
        dummy_obs = self._observation_builder.build_observation()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=dummy_obs.shape, dtype=np.float32)

        # Define Action Space based on the robot's action space limits
        low_action, high_action = self._robot.action_space_limits
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        # Store full initial qpos/qvel for MuJoCo's direct reset,
        # then let robot/object components handle their specific parts.
        self._initial_qpos_all = self._physics.data.qpos.copy()
        self._initial_qvel_all = self._physics.data.qvel.copy()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._physics.reset() # Reset MuJoCo's internal state to model defaults
        
        # Reset all qpos/qvel to known initial states (important for deterministic restarts)
        self._physics.data.qpos[:] = self._initial_qpos_all
        self._physics.data.qvel[:] = self._initial_qvel_all

        # Delegate reset logic to specific components
        self._robot.reset_state()
        self._object.reset_state()

        # Step physics a few times to stabilize after reset
        for _ in range(10): # Simulate 10 steps to allow physics to settle
            self._physics.step()

        observation = self._observation_builder.build_observation()
        info = {}
        return observation, info

    def step(self, action):
        # Clip actions to stay within the action space boundaries
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply actions to the robot's actuators
        self._robot.set_commands(clipped_action)

        # Simulate for 'num_substeps' to cover 'control_timestep' duration
        num_substeps = int(self.control_timestep / self._model.opt.timestep)
        for _ in range(num_substeps):
            self._physics.step()

        reward = self._reward_composer.get_total_reward()

        # Check for episode termination (time limit)
        terminated = False
        if self._physics.data.time >= self.episode_duration:
            terminated = True

        truncated = False # In gymnasium, time limits typically result in 'truncated=True' but for simplicity initially, 'terminated=True' works.

        observation = self._observation_builder.build_observation()
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        # Render the MuJoCo scene for visualization
        return self._physics.render(width=640, height=480, camera_id=0) # You might need to adjust camera_id

    def close(self):
        # No explicit close needed for dm_control.Physics, but good practice
        pass