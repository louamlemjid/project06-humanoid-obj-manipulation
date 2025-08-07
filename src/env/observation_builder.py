# src/env/observation_builder.py
import numpy as np
from src.robot.base_robot import BaseRobot # Relative import
from src.objects.base_object import BaseObject # Relative import

# src/env/observation_builder.py
import numpy as np


class ObservationBuilder:
    """
    Responsible for building the complete observation array from various components.
    """
    def __init__(self, robot: BaseRobot, manipulable_object: BaseObject):
        self._robot = robot
        self._object = manipulable_object

    def build_observation(self) -> np.ndarray:
        """Collects observation data from robot and object and concatenates them."""
        robot_obs = self._robot.get_observation_data()
        object_obs = self._object.get_observation_data()
        deltas = robot_obs[-3:] - object_obs
        obs = np.concatenate([
            robot_obs,
            object_obs,
            deltas
        ]).astype(np.float32)
        return obs