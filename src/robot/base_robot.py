# src/robot/base_robot.py
from abc import ABC, abstractmethod
import numpy as np
from dm_control import mujoco

class BaseRobot:
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
        return self._controllable_joint_ids

    @property
    def actuator_ids(self) -> list[int]:
        return self._actuator_ids

    @property
    def action_space_limits(self) -> tuple[np.ndarray, np.ndarray]:
        # Manually set control ranges for the first 7 actuators
        manual_ctrl_low = np.array([-2,-2,-1.5,-2] + [-1.5]*3) # [-3 , -3, -3, -2, -1, -1, -1]
        manual_ctrl_high = np.array([2, 2, 1.5, 2] + [1.5]*3)  # [ 3 ,  3 , 3 , 2 , 1 , 1 , 1]

        # Assume actuator_ids = [0, 1, 2, ..., 28]
        # Skip first 7 and get the rest dynamically from the model
        auto_ids = self.actuator_ids[7:]
        auto_ctrl_low = self._model.actuator_ctrlrange[auto_ids, 0]
        auto_ctrl_high = self._model.actuator_ctrlrange[auto_ids, 1]

        # Combine manual and auto
        low = np.concatenate([manual_ctrl_low, auto_ctrl_low])
        high = np.concatenate([manual_ctrl_high, auto_ctrl_high])

        return low, high


    def get_observation_data(self) -> np.ndarray:
        pass

    def set_commands(self, actions: np.ndarray):
        pass

    def reset_state(self):
        pass