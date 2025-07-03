import mujoco
import os

model_path = os.path.join("models", "unitree_h1", "h1.xml")

try:
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    print("Model loaded successfully! :D ")
except Exception as e:
    print("Failed to load MuJoCo model :(")
    print(e)
