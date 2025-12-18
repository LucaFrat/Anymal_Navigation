from isaaclab.managers import SceneEntityCfg
from .vision_encoder import load_encoder
import torch
import torchvision.utils as vutils
import os
import random

from isaaclab.utils.math import quat_apply_inverse

def cone_position_b(env, robot_cfg: SceneEntityCfg, cone_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Observation: The position of the cone relative to the robot's base frame.
    """

    robot = env.scene[robot_cfg.name]
    cone = env.scene[cone_cfg.name]

    robot_pos = robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w
    cone_pos = cone.data.root_pos_w

    vec_w = cone_pos - robot_pos
    vec_b = quat_apply_inverse(robot_quat, vec_w)

    return vec_b


def visual_latent(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reads camera image, passes it through the frozen encoder, returns latent vector.
    """
    # Load Model
    encoder = load_encoder(env.device)

    sensor = env.scene.sensors[sensor_cfg.name]
    # Shape: (Num_Envs, H, W, 4) -> RGB + Alpha
    images = sensor.data.output["rgb"]

    if images.shape[-1] == 4:
        images = images[..., :3]

    images = images.permute(0, 3, 1, 2).float() / 255.0

    save_image(images[0])
    save_image(images[1])
    save_image(images[2])

    with torch.no_grad():
        latents = encoder(images)

    return latents


def save_image(image):
    save_dir = "debug_images"
    os.makedirs(save_dir, exist_ok=True)
    num = random.randint(1, 1000)
    filename = os.path.join(save_dir, f"step_{num:03d}.png")
    vutils.save_image(image, filename)
    print(f"Saved {filename}")
