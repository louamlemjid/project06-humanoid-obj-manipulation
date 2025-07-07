# src/sim/scene_builder.py

from dm_control import mujoco
import os

class SceneBuilder:
    """
    Responsible for building a complete MuJoCo scene directly 
    from a given XML file.
    """
    
    @staticmethod
    def build(scene_xml_path: str) -> mujoco.Physics:
        """
        Loads and compiles a MuJoCo scene from a specified XML file.

        This is a static method, so you can call it directly on the class
        without creating an instance: `SceneBuilder.build_from_xml(...)`.

        Args:
            scene_xml_path: The full path to the scene's .xml file.

        Returns:
            A compiled dm_control.mujoco.Physics instance.
            
        Raises:
            FileNotFoundError: If the provided XML file path does not exist.
        """
        if not os.path.exists(scene_xml_path):
            raise FileNotFoundError(
                f"The specified scene file was not found: {scene_xml_path}"
            )
            
        print(f"Loading scene from: {scene_xml_path}")
        
        # The core function to load a model from a complete XML file.
        physics = mujoco.Physics.from_xml_path(scene_xml_path)
        
        print("Scene built successfully.")
        return physics