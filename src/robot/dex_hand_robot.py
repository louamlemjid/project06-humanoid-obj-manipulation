# src/robot/dex_hand_robot.py
import numpy as np
from dm_control import mujoco
from src.robot.base_robot import BaseRobot # Relative import

class DexHandRobot(BaseRobot):
    """Concrete implementation for the Dex Hand robot."""

    def __init__(self, physics: mujoco.Physics):
        super().__init__(physics)
        # Identify specific joint and actuator IDs for the Dex Hand
        # Filter for names starting with 'lh_' (left hand) and exclude wrist for finger-only control
        self._controllable_joint_ids = [
            self._model.joint_name2id(name) for name in self._model.joint_names if 'lh_' in name and 'WRJ' not in name
        ]
        self._actuator_ids = [
            self._model.actuator_name2id(name) for name in self._model.actuator_name if 'lh_A_' in name and 'WRJ' not in name
        ]

        # Store initial qpos for the Dex Hand for reset (only for controllable joints)
        self._initial_qpos = self.physics.data.qpos[self.controllable_joint_ids].copy()

    def get_observation_data(self) -> np.ndarray:
        """Returns Dex Hand's qpos and qvel for controllable joints."""
        qpos_obs = self.physics.data.qpos[self.controllable_joint_ids]
        qvel_obs = self.physics.data.qvel[self.controllable_joint_ids]
        return np.concatenate([qpos_obs, qvel_obs]).astype(np.float32)

    def set_commands(self, actions: np.ndarray):
        """Applies actions (target joint positions) to Dex Hand actuators."""
        # Actions array should directly correspond to the order of actuator_ids
        for i, act_id in enumerate(self.actuator_ids):
            self.physics.data.ctrl[act_id] = actions[i]

    def reset_state(self):
        """Resets the Dex Hand's joint positions and velocities."""
        # Set qpos for controllable joints to their initial state
        self.physics.data.qpos[self.controllable_joint_ids] = self._initial_qpos
        # Set qvel for controllable joints to zero
        self.physics.data.qvel[self.controllable_joint_ids] = 0.0