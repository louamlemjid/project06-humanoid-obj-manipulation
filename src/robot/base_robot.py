# src/robot/base_robot.py
from abc import ABC, abstractmethod
import numpy as np
from dm_control import mujoco

class BaseRobot(ABC):
    """Abstract base class for any robot in the simulation."""

    def __init__(self, physics: mujoco.Physics):
        self._physics = physics
        self._model = physics.model
        self._controllable_joint_ids = []
        self._actuator_ids = []

    @property
    def physics(self) -> mujoco.Physics:
        return self._physics

    @property
    def model(self) -> mujoco.wrapper.core.MjModel:
        return self._model

    @property
    def controllable_joint_ids(self) -> list[int]:
        """Returns a list of MuJoCo IDs for joints controlled by this robot."""
        return self._controllable_joint_ids

    @property
    def actuator_ids(self) -> list[int]:
        """Returns a list of MuJoCo IDs for actuators controlling this robot."""
        return self._actuator_ids

    @property
    def action_space_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (low, high) bounds for the robot's action space."""
        low = self._model.actuator_ctrlrange[self.actuator_ids, 0]
        high = self._model.actuator_ctrlrange[self.actuator_ids, 1]
        return low, high

    @abstractmethod
    def get_observation_data(self) -> np.ndarray:
        """Returns the part of the observation related to this robot's state."""
        pass

    @abstractmethod
    def set_commands(self, actions: np.ndarray):
        """Applies actions to the robot's actuators."""
        pass

    @abstractmethod
    def reset_state(self):
        """Resets the robot's joint positions and velocities to initial state."""
        pass