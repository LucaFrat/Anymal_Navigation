import torch
from typing import Tuple
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv

def reset_cone_pos_donut(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    radius_range: Tuple[float, float],
):

    asset: RigidObject = env.scene[asset_cfg.name]
    root_state = asset.data.default_root_state[env_ids].clone()

    min_r, max_r = radius_range

    r_squared = torch.rand(len(env_ids), device=env.device) * (max_r**2 - min_r**2) + min_r**2
    r = torch.sqrt(r_squared)

    theta = torch.rand(len(env_ids), device=env.device) * 2 * torch.pi - torch.pi

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)


    env_origins = env.scene.env_origins[env_ids]
    root_state[:, 0] = env_origins[:, 0] + x
    root_state[:, 1] = env_origins[:, 1] + y
    root_state[:, 2] = env_origins[:, 2] + 0.5

    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)