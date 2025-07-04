# src/objects/cube_object.py
import numpy as np
from dm_control import mujoco
from src.objects.base_object import BaseObject # Relative import

class CubeObject(BaseObject):
    """Concrete implementation for a manipulable cube object."""

    def __init__(self, physics: mujoco.Physics, object_config: dict):
        super().__init__(physics, object_config)
        self._initial_pos_relative = np.array(object_config["initial_pos_relative"])
        self._pos_randomization_range = {
            k: np.array(v) for k, v in object_config["pos_randomization_range"].items()
        }

    def get_observation_data(self) -> np.ndarray:
        """Returns cube's position, orientation, linear and angular velocities."""
        object_pos = self._physics.data.body_xpos[self.body_id]
        object_quat = self._physics.data.body_xquat[self.body_id]
        object_lin_vel = self._physics.data.body_xvelp[self.body_id]
        object_ang_vel = self._physics.data.body_xvelr[self.body_id]
        return np.concatenate([object_pos, object_quat, object_lin_vel, object_ang_vel]).astype(np.float32)

    def reset_state(self):
        """Resets the cube's position to initial (with randomization) and velocity to zero."""
        object_initial_qpos_idx = self._model.joint_qposadr[self.joint_id]

        # Apply randomization to initial position
        random_offset_x = np.random.uniform(self._pos_randomization_range['x'][0], self._pos_randomization_range['x'][1])
        random_offset_y = np.random.uniform(self._pos_randomization_range['y'][0], self._pos_randomization_range['y'][1])
        random_offset_z = np.random.uniform(self._pos_randomization_range['z'][0], self._pos_randomization_range['z'][1])

        new_pos = self._initial_pos_relative + np.array([random_offset_x, random_offset_y, random_offset_z])

        # Set the object's position (first 3 values of its free joint qpos)
        self._physics.data.qpos[object_initial_qpos_idx:object_initial_qpos_idx+3] = new_pos
        # Set the object's orientation (quaternion, next 4 values). Keep it upright.
        self._physics.data.qpos[object_initial_qpos_idx+3:object_initial_qpos_idx+7] = np.array([1.0, 0.0, 0.0, 0.0]) # WXYZ identity quaternion

        # Reset object's velocities (6 values for a free joint)
        object_initial_qvel_idx = self._model.joint_dofadr[self.joint_id]
        self._physics.data.qvel[object_initial_qvel_idx:object_initial_qvel_idx+6] = 0.0