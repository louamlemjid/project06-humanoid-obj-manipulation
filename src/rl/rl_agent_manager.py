# src/rl/rl_agent_manager.py
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env # Not used in this minimal test, but useful
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import gymnasium as gym

class RLAgentManager:
    """Manages the lifecycle of an RL agent (training, saving, loading)."""

    def __init__(self, env: gym.Env, rl_config: dict):
        self._env = env
        self._rl_config = rl_config
        self._model = None

        os.makedirs(self._rl_config["log_dir"], exist_ok=True)
        os.makedirs(os.path.dirname(self._rl_config["model_save_path"]), exist_ok=True)

    def train_agent(self):
        """Initializes and trains the SAC agent."""
        print("Initializing SAC agent...")
        
        self._model = SAC(
            "MlpPolicy",
            self._env,
            verbose=1,
            learning_rate=self._rl_config["learning_rate"],
            buffer_size=self._rl_config["buffer_size"],
            learning_starts=self._rl_config["learning_starts"],
            train_freq=tuple(self._rl_config["train_freq"]),
            gradient_steps=self._rl_config["gradient_steps"],
            ent_coef=self._rl_config["ent_coef"],
            tensorboard_log=self._rl_config["log_dir"]
        )

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=self._rl_config["save_freq"],
            save_path=os.path.join(self._rl_config["log_dir"], "checkpoints"),
            name_prefix='sac_model'
        )

        print(f"Starting training for {self._rl_config['total_timesteps']} timesteps...")
        self._model.learn(
            total_timesteps=self._rl_config["total_timesteps"],
            progress_bar=True,
            callback=[checkpoint_callback]
        )
        print("Training finished!")
        self._model.save(self._rl_config["model_save_path"])
        print(f"Model saved to {self._rl_config['model_save_path']}")

    def evaluate_agent(self, num_episodes: int = 1, render: bool = True):
        """Evaluates the loaded agent."""
        if self._model is None:
            print("No model loaded. Please train or load an agent first.")
            return

        print(f"Evaluating agent for {num_episodes} episodes...")
        for i in range(num_episodes):
            obs, info = self._env.reset()
            done = False
            total_reward = 0
            frames = []
            while not done:
                action, _states = self._model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self._env.step(action)
                total_reward += reward
                if render:
                    frames.append(self._env.render())
                done = terminated or truncated
            print(f"Episode {i+1}: Total Reward = {total_reward:.2f}")
            # For Colab, you'd typically save frames and then display them as a video.
            # Example (if you have a helper function like display_video):
            # if render and frames:
            #     from your_utils_file import display_video
            #     display_video(frames, fps=1/self._env.control_timestep)
        print("Evaluation complete.")