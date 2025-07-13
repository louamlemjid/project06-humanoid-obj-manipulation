# src/env/reward_components.py
import numpy as np
from dm_control import mujoco

# Define individual reward functions here.
# Each function takes physics, relevant object_body_id, and config.

# def object_height_reward(physics: mujoco.Physics, object_body_id: int, config: dict) -> float:
#     """Rewards the agent for lifting the object to a certain height."""
#     object_height = physics.data.xpos[object_body_id][2]
#     target_height = config.get("target_height", 0.15)
#     
#     # Reward is proportional to how close the object is to the target height
#     reward = max(0, 1 - abs(object_height - target_height) / target_height)
#     return float(reward * config.get("height_reward_factor", 10))

# def target_height_reward(physics: mujoco.Physics, object_body_id: int, config: dict) -> float:
#     """Gives a sparse reward when the object reaches the target height."""
#     object_height = physics.data.xpos[object_body_id][2]
#     if object_height >= config.get("target_height", 0.15):
#         return float(config.get("target_achieved_bonus", 50))
#     return 0.0

# def drop_penalty_reward(physics: mujoco.Physics, object_body_id: int, config: dict) -> float:
#     """Penalizes the agent if the object falls too low."""
#     object_height = physics.data.xpos[object_body_id][2]
#     if object_height < config.get("drop_threshold", 0.02): # Below a very low threshold
#         return float(-config.get("drop_penalty", 200))
#     return 0.0

def hand_z_position_reward(
    physics: mujoco.Physics, 
    hand_palm_body_id: int, 
    config: dict
) -> float:
    """
    Rewards the agent for maintaining the hand's height above a certain threshold.
    This can be used to prevent the hand from dropping too low.
    """
    # Get the z-position of the hand's palm
    hand_z_pos = physics.data.xpos[hand_palm_body_id][2]
    
    # Get configuration parameters
    target_height = config.get("hand_target_z", 0.05)
    reward_factor = config.get("hand_z_reward_factor", 300.0)
    
    # Calculate the reward
    # The reward is positive if the hand is above the target height
    reward = (hand_z_pos - target_height) * reward_factor
    
    return float(reward)
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

def finger_proximity_reward(
    physics: mujoco.Physics, 
    finger_body_ids: list[int], 
    object_body_id: int, 
    config: dict
) -> float:
    """
    Rewards the agent for minimizing the distance between multiple finger parts and the object.
    This encourages the hand to shape itself around the object for grasping.
    """
    # Get the position of the object
    object_pos = physics.data.xpos[object_body_id]
    
    distances = []
    for finger_id in finger_body_ids:
        # Get the position of the finger part
        finger_pos = physics.data.xpos[finger_id]
        
        # Calculate the Euclidean distance
        distance = np.linalg.norm(finger_pos - object_pos)
        distances.append(distance)
        
    # Calculate the average distance from the finger parts to the object
    avg_distance = np.mean(distances)

    # Use an exponential decay function for the reward.
    # The reward is close to the factor when distance is 0, and decays as distance increases.
    distance_scale = config.get("distance_scale", 20.0)  # Sharper decay for proximity
    reward_factor = config.get("distance_reward_factor", 5.0)  # Higher reward for close proximity
    
    reward = reward_factor * np.exp(-distance_scale * avg_distance)

    return float(reward)

def object_y_position_reward(
    physics: mujoco.Physics,
    object_body_id: int,
    config: dict
) -> float:
    """
    Rewards the agent for moving the object to a target y-position.
    This encourages the agent to not only lift the object but also move it to a specific location.
    """
    # Get the object's current y-position
    object_y_pos = physics.data.xpos[object_body_id][1]

    # Get configuration parameters from the dictionary
    target_y = config.get("target_y_position", 0.2)
    distance_scale = config.get("y_pos_distance_scale", 15.0)
    reward_factor = config.get("y_pos_reward_factor", 10.0)

    # Calculate the absolute difference in the y-position
    y_distance = abs(object_y_pos - target_y)

    # Use an exponential decay function for the reward.
    # The reward is highest when the object is at the target y-position.
    reward = reward_factor * np.exp(-distance_scale * y_distance)

    return float(reward)