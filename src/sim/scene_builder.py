# src/sim/scene_builder.py

from dm_control import mujoco
import numpy as np

class SceneBuilder:
    """
    Responsible for building the complete MuJoCo scene from MJCF elements.
    """
    def __init__(self, hand_model_path: str, object_config: dict):
        self.hand_model_path = hand_model_path
        self.object_config = object_config

    def build(self) -> mujoco.Physics:
        """Loads the hand model and attaches the manipulable object."""
        # Load the existing Dex Hand scene
        dex_hand_mjcf_root = mujoco.Physics.from_xml_path(self.hand_model_path)

        # Create an MJCF element for your manipulable object (e.g., a cube)
        object_mjcf = mujoco.Physics.RootElement(model=self.object_config["name"])
        cube_body = object_mjcf.worldbody.add(
            'body',
            name=self.object_config["name"],
            pos=self.object_config["initial_pos_relative"]
        )
        cube_body.add('joint', name=self.object_config["joint_name"], type="free")
        cube_body.add('geom',
                      name=self.object_config["name"] + "_geom",
                      type="box",
                      size=self.object_config["geom_size"],
                      mass=self.object_config["mass"],
                      rgba="1 0 0 1")

        # Attach the object to the existing Dex Hand scene's worldbody
        dex_hand_mjcf_root.worldbody.attach(object_mjcf)

        # Create the final physics object from the combined model
        physics = mujoco.Physics.from_mjcf_model(dex_hand_mjcf_root)
        return physics