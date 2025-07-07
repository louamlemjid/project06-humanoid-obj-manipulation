#src/objects/base_object.py
from abc import ABC, abstractmethod
import numpy as np
from dm_control import mujoco

class BaseObject(ABC):
    """Abstract base class for any manipulable object."""

    def __init__(self, physics: mujoco.Physics, object_config: dict):
        self._physics = physics
        self._model = physics.model
        self._object_config = object_config
        self._body_id = self._model.name2id('cube','body')
        self._joint_id = self._model.name2id('cube_joint','joint')

    @property
    def physics(self) -> mujoco.Physics:
        return self._physics

    @property
    def model(self) -> mujoco.wrapper.core.MjModel:
        return self._model

    @property
    def body_id(self) -> int:
        return self._body_id

    @property
    def joint_id(self) -> int:
        return self._joint_id

    @abstractmethod
    def get_observation_data(self) -> np.ndarray:
        """Returns the part of the observation related to this object's state."""
        pass

    @abstractmethod
    def reset_state(self):
        """Resets the object's position and orientation to an initial (possibly randomized) state."""
        pass