import numpy as np
from dm_control import mujoco


class RewardComposer:
    def __init__(
        self,
        physics: mujoco.Physics,
        object_body_id: int,
        hand_joint_ids: list[int],
        finger_body_ids: list[int],
        hand_palm_body_id: int,
        reward_config: dict
    ):
        self._physics = physics
        self._object_body_id = object_body_id
        self._hand_joint_ids = hand_joint_ids
        self._finger_body_ids = finger_body_ids
        self._hand_palm_body_id = hand_palm_body_id
        self._reward_config = reward_config
        self._in_contact = False  # For logging only

    def get_distance(self) -> float:
        palm_pos = self._physics.data.xpos[self._hand_palm_body_id]
        cube_pos = self._physics.data.xpos[self._object_body_id]
        return np.linalg.norm(palm_pos - cube_pos)

    import numpy as np

    def _get_orientation_reward(self) -> float:
        try:
            # Get rotation matrix
            palm_rot = self._physics.data.xmat[self._hand_palm_body_id].reshape(3, 3)

            # --- FORWARD DIRECTION: Assume -Z is forward ---
            palm_forward = -palm_rot[2]  # -Z axis = forward

            # Vector from palm to cube
            cube_pos = self._physics.data.xpos[self._object_body_id]
            palm_pos = self._physics.data.xpos[self._hand_palm_body_id]
            to_cube = cube_pos - palm_pos

            # Safe normalization with epsilon
            to_cube_norm = np.linalg.norm(to_cube)
            to_cube = to_cube / (to_cube_norm + 1e-6)  # ← Added 1e-6 for safety

            # Alignment: how well palm is facing the cube
            alignment = np.dot(palm_forward, to_cube)

            # Only reward if facing toward the cube
            # if alignment < 0.0:
            #     return 0.0  # Back of hand facing → no reward

            # # Optional: gate by distance (only reward when close)


            # Output in [0, 1]
            return float(np.exp(0.1 * (alignment - 1)))  # Already bounded by dot product

        except Exception:
            return 0.0
    import numpy as np

    def _get_finger_reach_distance(self):
          # Get the object's position
          cube_pos = self._physics.data.xpos[self._object_body_id]

          # Indices of the distal links (replace these with actual site/body IDs if needed)
          distal_indices = [17, 13, 21, 31, 26]

          # Get the positions of each distal segment
          finger_positions = [self._physics.data.xpos[i] for i in distal_indices]

          # Compute Euclidean distances from each finger to the cube
          distances = [np.linalg.norm(finger_pos - cube_pos) for finger_pos in finger_positions]

          # Compute average distance
          avg_distance = np.mean(distances)

          if (self.get_distance() < 0.1 ):
            return float(1.0 * np.exp(-1.0 * avg_distance))

          return 0.0

    def _get_smoothness_reward(self) -> float:
        """Encourage smooth motion by penalizing high joint velocities."""
        qvel = self._physics.data.qvel[:7]  # Arm joint velocities
        return -0.002 * np.sum(np.square(qvel))

    def reward_for_task(self, task_name: str) -> float:
        distance = self.get_distance()
        avg_finger_distance = self._get_finger_reach_distance()

        # --- REACHING: Strong shaping to get close ---
        reward_reach = 15.0 * np.exp(-8.0 * distance)

        # --- CONTACT: Only reward very close proximity ---
        reward_contact = 50.0 if distance < 0.07 else 0.0  # Touch threshold

        # --- ORIENTATION: Encourage correct palm pose ---
        reward_orientation = self._get_orientation_reward()

        # --- CONTROL COST ---
        action = self._physics.data.ctrl[:7]
        reward_ctrl = -0.001 * np.sum(np.square(action))

        # --- SMOOTHNESS ---
        reward_smooth = self._get_smoothness_reward()

        total = (
            reward_reach +
            #reward_contact +
            reward_orientation +
            reward_ctrl 

            #reward_smooth
        )
        return float(total)

    def get_total_reward(self) -> float:
        return self.reward_for_task("reaching")

    def get_info(self) -> dict:
        distance = self.get_distance()

        avg_finger_distance = self._get_finger_reach_distance()
        # Recompute all components (same as in reward_for_task)
        reward_reach = 15.0 * np.exp(-8.0 * distance)

        reward_contact = 50.0 if distance < 0.07 else 0.0

        reward_orientation = self._get_orientation_reward()

        action = self._physics.data.ctrl[:7]
        reward_ctrl = -0.001 * np.sum(np.square(action))
        reward_smooth = self._get_smoothness_reward()



        total_reward = (
            reward_reach +
            #reward_contact +
            reward_orientation +
            reward_ctrl
            #reward_smooth
        )

        return {
            "reach_distance": distance,
            "success": distance < 0.07,  # Success at 10 cm
            "in_contact": self._in_contact,
            "total_reward": total_reward,
            "reward_components": {
                "reward_reach": reward_reach,
                "reward_contact": reward_contact,
                "reward_orientation": reward_orientation,
                "reward_ctrl": reward_ctrl,
                #"reward_smooth": reward_smooth,
            },
            # Optional debug: uncomment to log orientation
            # "debug_palm_x": palm_rot[:, 0].tolist(),
            # "debug_palm_y": palm_rot[:, 1].tolist(),
            # "debug_palm_z": palm_rot[:, 2].tolist(),
        }