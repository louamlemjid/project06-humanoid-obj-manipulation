import numpy as np
from dm_control import mujoco

class TaskManager:
    def __init__(self, physics, object_body_id, finger_body_ids, palm_id):
        self.physics = physics
        self.object_body_id = object_body_id
        self.finger_body_ids = finger_body_ids
        self.palm_id = palm_id

    def get_current_task(self) -> str:
        object_pos = self.physics.data.body_xpos[self.object_body_id]
        palm_pos = self.physics.data.body_xpos[self.palm_id]
        finger_positions = self.physics.data.body_xpos[self.finger_body_ids]

        # Task 1: Reaching
        finger_dists = [np.linalg.norm(f - object_pos) for f in finger_positions]
        if np.mean(finger_dists) > 0.03:
            return "reaching"

        # Task 2: Grasping
        if self._is_gripper_closed() and self._is_touching_object(finger_positions, object_pos):
            return "grasping"

        # Task 3: Lifting
        if object_pos[2] > 0.1:
            return "lifting"

        # Task 4: Moving (optional)
        if np.linalg.norm(object_pos[:2] - np.array([0.3, 0.3])) < 0.05:
            return "placing"

        return "unknown"  # Fallback
