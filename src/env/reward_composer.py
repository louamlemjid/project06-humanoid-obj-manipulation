# src/env/reward_composer.py
from typing import Callable, List
from dm_control import mujoco
from src.env.reward_components import (
    object_height_reward,
    target_height_bonus_reward,
    drop_penalty_reward
) # Import individual components

class RewardComposer:
    """
    Composes multiple individual reward components into a single total reward.
    """
    def __init__(self, physics: mujoco.Physics, object_body_id: int, reward_config: dict):
        self._physics = physics
        self._object_body_id = object_body_id
        self._reward_config = reward_config

        # Define the list of active reward functions to use
        self._reward_functions: List[Callable[[mujoco.Physics, int, dict], float]] = [
            object_height_reward,
            target_height_bonus_reward,
            drop_penalty_reward,
        ]

    def get_total_reward(self) -> float:
        """Calculates the sum of all active reward components."""
        total_reward = 0.0
        for func in self._reward_functions:
            total_reward += func(self._physics, self._object_body_id, self._reward_config)
        return total_reward