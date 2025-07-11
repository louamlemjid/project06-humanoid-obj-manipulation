# src/env/reward_composer.py
from typing import Callable, List
from dm_control import mujoco
from src.env.reward_components import (
    object_height_reward,
    target_height_bonus_reward,
    drop_penalty_reward,
    hand_movement_reward,
    finger_proximity_reward
) # Import individual components

class RewardComposer:
    """
    Composes multiple individual reward components into a single total reward.
    """
    def __init__(
        self, 
        physics: mujoco.Physics, 
        object_body_id: int, 
        hand_joint_ids: list[int],
        finger_body_ids: list[int],
        reward_config: dict
    ):
        self._physics = physics
        self._object_body_id = object_body_id
        self._hand_joint_ids = hand_joint_ids
        self._finger_body_ids = finger_body_ids
        self._reward_config = reward_config

        # Define the list of active reward functions to use
        self._reward_functions: List[Callable[[mujoco.Physics, int, dict], float]] = [
            object_height_reward,
            target_height_bonus_reward,
            drop_penalty_reward,
            hand_movement_reward,
            finger_proximity_reward,
        ]

        # Store reward functions and their specific arguments
        self._reward_functions = [
            (hand_movement_reward, {'hand_joint_ids': self._hand_joint_ids}),
            (finger_proximity_reward, {'finger_body_ids': self._finger_body_ids, 'object_body_id': self._object_body_id})
        ]

    def get_total_reward(self) -> float:
        """Calculates the sum of all active reward components."""
        total_reward = 0.0
        for func, kwargs in self._reward_functions:
            # Pass the specific arguments required by each reward function
            total_reward += func(
                physics=self._physics, 
                config=self._reward_config,
                **kwargs
            )
        return total_reward