# src/env/reward_components.py
import numpy as np
from dm_control import mujoco

# Define individual reward functions here.
# Each function takes physics, relevant object_body_id, and config.

def object_height_reward(physics: mujoco.Physics, object_body_id: int, config: dict) -> float:
    """Rewards the agent based on the object's height."""
    object_height = physics.data.xpos[object_body_id][2] # Z-coordinate
    reward = config.get("object_height_factor", 100) * (object_height - 0.05) # Assume table is at Z=0.05
    return float(reward)

def target_height_bonus_reward(physics: mujoco.Physics, object_body_id: int, config: dict) -> float:
    """Gives a bonus if the object reaches a target height."""
    object_height = physics.data.xpos[object_body_id][2]
    if object_height >= config.get("target_height", 0.05):
        return float(config.get("target_height_bonus", 500))
    return 0.0

def drop_penalty_reward(physics: mujoco.Physics, object_body_id: int, config: dict) -> float:
    """Penalizes the agent if the object falls too low."""
    object_height = physics.data.xpos[object_body_id][2]
    if object_height < config.get("drop_threshold", 0.02): # Below a very low threshold
        return float(-config.get("drop_penalty", 200))
    return 0.0

def hand_movement_reward(physics: mujoco.Physics, hand_joint_ids: list[int], config: dict) -> float:
    """Rewards the agent for moving the hand's joints to encourage exploration."""
    # Get velocities of the specified hand joints
    hand_joint_velocities = physics.data.qvel[hand_joint_ids]

    # Calculate the magnitude of the joint velocities (L2 norm)
    movement_magnitude = np.linalg.norm(hand_joint_velocities)

    # Apply a scaling factor from the config
    scaling_factor = config.get("hand_joint_velocity_scaling", 1.0)

    # The reward could be the movement magnitude scaled by the scaling factor
    reward = scaling_factor * movement_magnitude

    return float(reward)

def hand_to_object_distance_reward(physics: mujoco.Physics, hand_palm_body_id: int, object_body_id: int, config: dict) -> float:
    """Rewards the agent for minimizing the distance between the hand palm and the object."""
    # Get the position of the hand palm and the object
    hand_palm_pos = physics.data.xpos[hand_palm_body_id]
    object_pos = physics.data.xpos[object_body_id]

    # Calculate the Euclidean distance
    distance = np.linalg.norm(hand_palm_pos - object_pos)

    # Use an exponential decay function for the reward.
    # The reward is close to the factor when distance is 0, and decays as distance increases.
    distance_scale = config.get("distance_scale", 10.0) # Controls how fast the reward decays
    reward_factor = config.get("distance_reward_factor", 1.0) # Max reward
    
    reward = reward_factor * np.exp(-distance_scale * distance)

    return float(reward)