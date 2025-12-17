from isaaclab.managers import SceneEntityCfg
from .vision_encoder import load_encoder
import torch
import torchvision.utils as vutils
import os
import random



def visual_latent(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reads camera image, passes it through the frozen encoder, returns latent vector.
    """
    # 1. Load Model (Cached)
    encoder = load_encoder(env.device)

    # 2. Get Raw Data
    sensor = env.scene.sensors[sensor_cfg.name]
    # Shape: (Num_Envs, H, W, 4) -> RGB + Alpha
    images = sensor.data.output["rgb"]

    # 3. Preprocess
    # Remove Alpha channel if present: (N, H, W, 3)
    if images.shape[-1] == 4:
        images = images[..., :3]

    # PyTorch expects (N, C, H, W). Permute dimensions.
    # Also normalize 0-255 to 0.0-1.0
    images = images.permute(0, 3, 1, 2).float() / 255.0

    # save_image(images[0])
    # save_image(images[1])
    # save_image(images[2])

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
