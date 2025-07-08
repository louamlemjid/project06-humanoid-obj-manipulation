# main.py
import yaml
import gymnasium as gym

# Import all your modular components
from src.sim.scene_builder import SceneBuilder
from src.robot.dex_hand_robot import DexHandRobot
from src.objects.cube_object import CubeObject
from src.env.observation_builder import ObservationBuilder
from src.env.reward_composer import RewardComposer
from src.env.hand_manipulation_env import HandManipulationEnv
from src.rl.rl_agent_manager import RLAgentManager

def load_config(config_path: str) -> dict:
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    pathToModel = "./project06-humanoid-obj-manipulation/models/dex_hand/cube_scene.xml"
    sceneBuilder = SceneBuilder()
    physics = sceneBuilder.build(pathToModel)
    print("MuJoCo scene built successfully.")

    # --- 2. Initialize Robot and Object Interfaces ---
    robot = DexHandRobot(physics)

    object_config = {
    "initial_pos_relative": [0.0, 0.0, 0.0],
    "pos_randomization_range": {
        "x": [-0.1, 0.1],
        "y": [-0.02, 0.02],
    "z": [-0.005, 0.005]},
    "name": "cube" ,
  "joint_name": "cube_joint" ,
  "geom_name": "cube_geom",
  "size": [0.03, 0.04, 0.02],
    "rewards":{
            "object_height_factor": 100,
                "target_height": 0.06,
                "target_height_bonus": 500,
                "drop_penalty": 200
            },
       "env":{
           "hand_model_path": "models/dex_hand/scene.xml",
           "control_timestep": 0.002,
           "episode_duration": 2.0 # seconds
       },
    "rl":
    {
      "total_timesteps": 50000 ,# Keep low for initial test, increase to millions for actual training
      "log_dir": "./sac_hand_manipulation_logs/",
      "model_save_path": "./trained_models/hand_policy.zip",
      "learning_rate": 0.0003,
      "buffer_size": 10000,
      "learning_starts": 100, # Start learning quickly for test
      "train_freq": [1, "episode"],
      "gradient_steps": 1,
      "ent_coef": "auto",
      "save_freq": 5000, # How often to save checkpoints during training}
    }}
    
    manipulable_object = CubeObject(physics, object_config)
    print("Robot and Object interfaces initialized.")

    # --- 3. Setup Observation and Reward Composers ---
    observation_builder = ObservationBuilder(robot, manipulable_object)

    reward_composer = RewardComposer(physics, manipulable_object.body_id, object_config["rewards"])
    print("Observation and Reward composers set up.")

    # --- 4. Create the Custom Gym Environment ---
    env = HandManipulationEnv(
        physics=physics,
        robot=robot,
        manipulable_object=manipulable_object,
        observation_builder=observation_builder,
        reward_composer=reward_composer,
        env_config=config["env"]
    )
    print("Gym environment created.")

    # --- 5. Initialize and Manage the RL Agent ---
    rl_manager = RLAgentManager(env, config["rl"])
    print("RL Agent Manager initialized.")

    # --- Action: Train the agent ---
    rl_manager.train_agent()

    # --- Optional: Evaluate the trained agent (uncomment to run) ---
    # rl_manager.load_agent() # Loads the recently saved model
    # rl_manager.evaluate_agent(num_episodes=2, render=True)

    env.close() # Clean up the environment

if __name__ == "__main__":
    main()