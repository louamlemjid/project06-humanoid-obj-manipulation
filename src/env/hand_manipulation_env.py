import gymnasium as gym
from gymnasium import spaces
from dm_control import mujoco
import numpy as np
import time

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

        # Set simulation timestep for stability
        self._model.opt.timestep = 0.001  # Reduced for numerical stability

        # Define Observation Space
        dummy_obs = self._observation_builder.build_observation()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=dummy_obs.shape, dtype=np.float32)

        # Define Action Space
        low_action, high_action = self._robot.action_space_limits
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        # Store initial state
        self._initial_qpos_all = self._physics.data.qpos.copy()
        self._initial_qvel_all = self._physics.data.qvel.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._physics.reset()

        # Reset to initial state
        #self._physics.data.qpos[:] = self._initial_qpos_all
        #self._physics.data.qvel[:] = self._initial_qvel_all

        # Set base state (fixed to avoid instability)
        #self._physics.data.qpos[0:3] = [0.25, 0.0, 0.03]
        #self._physics.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        #self._physics.data.qvel[0:6] = 0.0
        self._reward_composer._in_contact = False
        # Delegate reset to components
        self._robot.reset_state()
        self._object.reset_state()

        # Increased stabilization steps
        for _ in range(50):
            self._physics.step()

        observation = self._observation_builder.build_observation()
        info = {"reset_time": self._physics.data.time}

        return observation, info

    def step(self, action):
        start_time = time.time()
        clipped_action = action #np.clip(action, self.action_space.low, self.action_space.high)
        self._robot.set_commands(clipped_action)
        num_substeps = int(self.control_timestep / self._model.opt.timestep)
        for _ in range(num_substeps):
            self._physics.step()

        # Check for instability
        qacc = self._physics.data.qacc[6:20]


        reward = self._reward_composer.get_total_reward()
        terminated = self._physics.data.time >= self.episode_duration
        truncated = False
        observation = self._observation_builder.build_observation()
        contacts = self._physics.data.ncon
        info = {
            "step_time": time.time() - start_time,
            "contacts": contacts,
            "time": self._physics.data.time
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        return self._physics.render(width=640, height=480, camera_id=0)

    def close(self):
        pass