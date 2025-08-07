class UnitreeG1Robot(BaseRobot):
    def __init__(self, physics: mujoco.Physics):
        super().__init__(physics)
        self._controllable_joint_ids = list(range(1, 8)) + list(range(9,31)) #29 joints
        self._actuator_ids = list(range(7)) + list(range(15, 33))  # Arm + hand/fingers

        # Save actuator ctrlrange for non-arm joints
        self._ctrl_range = self.model.actuator_ctrlrange[self._actuator_ids]

        # Indices for qpos and qvel
        self._arm_qpos_indices = list(range(7,14)) + list(range(15,37))
        self._arm_qvel_indices = list(range(6,13)) + list(range(14,36))

        # Initial positions
        self._initial_qpos = self.physics.data.qpos[self._arm_qpos_indices].copy()  # Exclude index 14

        # Manually set the range for the first 7 joints (IDs 1–7)
        manual_joint_min = np.array([
            -3.0892,     # left_shoulder_pitch_joint
            -1.5882,     # left_shoulder_roll_joint
            -2.618,      # left_shoulder_yaw_joint
            -1.0472,     # left_elbow_joint
            -1.97222,    # left_wrist_roll_joint
            -1.61443,    # left_wrist_pitch_joint
            -1.61443     # left_wrist_yaw_joint
        ])
        manual_joint_max = np.array([
            2.6704,      # left_shoulder_pitch_joint
            2.2515,      # left_shoulder_roll_joint
            2.618,       # left_shoulder_yaw_joint
            2.0944,      # left_elbow_joint
            1.97222,     # left_wrist_roll_joint
            1.61443,     # left_wrist_pitch_joint
            1.61443      # left_wrist_yaw_joint
        ])

        # Use model.jnt_range for the rest (IDs 9–30)
        auto_ids = list(range(9, 31))
        auto_joint_min = self.model.jnt_range[auto_ids, 0]
        auto_joint_max = self.model.jnt_range[auto_ids, 1]

        # Concatenate manual and auto joint ranges
        self._joint_min = np.concatenate([manual_joint_min, auto_joint_min])
        self._joint_max = np.concatenate([manual_joint_max, auto_joint_max])




        # For reference, xpos[2:31] assumed used elsewhere in the agent code

    def get_observation_data(self) -> np.ndarray:
        #arm_qpos = self.physics.data.qpos[self._arm_qpos_indices]
        #arm_qvel = self.physics.data.qvel[self._arm_qvel_indices]
        palm_xpos = self.physics.data.xpos[16]
        arm_qpos = self.physics.data.qpos[7:14]
        arm_qvel = self.physics.data.qvel[6:13]

        return  np.concatenate([
        arm_qpos,
        arm_qvel,
        palm_xpos
    ])


    # src/robot/unitree_g1_robot.py in set_commands

    def set_commands(self, actions: np.ndarray):
        """Applies actions to the actuators, with proper clipping and damping."""
        if actions.shape[0] != len(self._actuator_ids):
            raise ValueError(f"Expected {len(self._actuator_ids)} actions, got {actions.shape[0]}")

        # 1. Get the true actuator control ranges
        low, high = self.action_space_limits
        arm_low, arm_high = low[:7], high[:7]
        finger_low, finger_high = low[7:], high[7:]

        # 2. Separate arm and finger actions
        raw_arm_actions = actions[:7]
        raw_finger_actions = actions[7:]

        # 3. Apply Damping (if you choose to use it)
        # Note: Ensure arm_qvel indices correctly match arm actuators if you re-enable this
        damped_arm_actions = raw_arm_actions

        # 4. Clip BOTH arm and finger actions to their valid ranges
        clipped_arm_actions = np.clip(damped_arm_actions, arm_low, arm_high)
        clipped_finger_actions = np.clip(raw_finger_actions, finger_low, finger_high)

        #calculate distance
        palm_pos = self._physics.data.xpos[9]
        cube_pos = self._physics.data.xpos[1]

        palm_cube_distance =  np.linalg.norm(palm_pos - cube_pos)

        # 5. Apply the final, safe control signals
        final_ctrl = np.concatenate([clipped_arm_actions, clipped_finger_actions])
        self.physics.data.ctrl[:7] = final_ctrl[:7]

        #ony ɔve fingers when pal close enough
        if(palm_cube_distance < 0.09):
            self.physics.data.ctrl[15:33] = final_ctrl[7:]

    def reset_state(self):
        #margin = 0.1 * (self._joint_max - self._joint_min)
        #random_qpos = np.random.uniform(self._joint_min + margin, self._joint_max - margin)
        randomPos = random.uniform(0.4,0.7)
        self.physics.data.qpos[self._arm_qpos_indices] = self._initial_qpos

        self.physics.data.ctrl[:] = 0.0
        self.physics.data.qvel[self._arm_qvel_indices] = 0.0
        self.physics.data.qpos[9]=randomPos