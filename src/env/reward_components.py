# src/env/reward_components.py
import numpy as np
from dm_control import mujoco

# Define individual reward functions here.
# Each function takes physics, relevant object_body_id, and config.

def object_height_reward(physics: mujoco.Physics, object_body_id: int, config: dict) -> float:
    """Rewards the agent based on the object's height."""
    object_height = physics.data.body_xpos[object_body_id][2] # Z-coordinate
    reward = config.get("object_height_factor", 100) * (object_height - 0.05) # Assume table is at Z=0.05
    return float(reward)

def target_height_bonus_reward(physics: mujoco.Physics, object_body_id: int, config: dict) -> float:
    """Gives a bonus if the object reaches a target height."""
    object_height = physics.data.body_xpos[object_body_id][2]
    if object_height >= config.get("target_height", 0.15):
        return float(config.get("target_height_bonus", 500))
    return 0.0

def drop_penalty_reward(physics: mujoco.Physics, object_body_id: int, config: dict) -> float:
    """Penalizes the agent if the object falls too low."""
    object_height = physics.data.body_xpos[object_body_id][2]
    if object_height < config.get("drop_threshold", 0.02): # Below a very low threshold
        return float(-config.get("drop_penalty", 200))
    return 0.0