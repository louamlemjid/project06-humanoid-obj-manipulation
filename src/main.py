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
            "z": [-0.005, 0.005]
        },
        "name": "cube" ,          
        "joint_name": "cube_joint" ,  
        "geom_name": "cube_geom",  
        "size": [0.03, 0.04, 0.02]
    }
    manipulable_object = CubeObject(physics, object_config)
    print("Robot and Object interfaces initialized.")

    # --- 3. Setup Observation and Reward Composers ---
    observation_builder = ObservationBuilder(robot, manipulable_object)
    reward_composer = RewardComposer(physics, manipulable_object.body_id, config["rewards"])
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