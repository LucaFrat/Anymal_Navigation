# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def terrain_levels_progress(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "pose_command",
) -> torch.Tensor:
    """
    Curriculum that increases difficulty if the robot covers > 50% of the distance to the goal.
    Decreases difficulty if the robot barely moves (fails early).
    """

    # get Robot Position
    asset = env.scene[asset_cfg.name]
    root_pos_w = asset.data.root_pos_w[env_ids, :2]  # XY position in World

    # get the Goal Position (from the episode that just finished)
    cmd_term = env.command_manager.get_term(command_name)

    # Safety check: ensure the command term has the world position attribute
    if not hasattr(cmd_term, "pos_command_w"):
        return env.scene.terrain.terrain_levels[env_ids]

    target_pos_w = cmd_term.pos_command_w[env_ids, :2]

    # assumes the robot spawns relatively close to the origin (0,0) of its env instance
    start_pos_w = env.scene.env_origins[env_ids, :2]

    # calculate distances
    total_mission_dist = torch.norm(target_pos_w - start_pos_w, dim=1)
    # The distance currently remaining to the goal
    dist_to_goal = torch.norm(target_pos_w - root_pos_w, dim=1)
    # The distance traveled from start (used for failure check)
    dist_from_start = torch.norm(root_pos_w - start_pos_w, dim=1)

    # if remaining distance is less than 20% the total distance
    move_up = dist_to_goal < (0.2 * total_mission_dist)

    # if covered less than 30% of the distance
    move_down = dist_from_start < (0.2 * total_mission_dist)


    terrain_levels = env.scene.terrain.terrain_levels[env_ids]
    terrain_levels += 1 * move_up
    terrain_levels -= 1 * move_down

    return torch.mean(terrain_levels.float())


def distance_level(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "pose_command",
    ) -> torch.Tensor:

    asset = env.scene[asset_cfg.name]
    root_pos_w = asset.data.root_pos_w[env_ids, :2]

    cmd_term = env.command_manager.get_term(command_name)

    if not hasattr(cmd_term, "pos_command_w"):
        return env.scene.terrain.terrain_levels[env_ids]

    target_pos_w = cmd_term.pos_command_w[env_ids, :2]

    start_pos_w = env.scene.env_origins[env_ids, :2]

    total_mission_dist = torch.norm(target_pos_w - start_pos_w, dim=1)
    dist_to_goal = torch.norm(target_pos_w - root_pos_w, dim=1)
    dist_from_start = torch.norm(root_pos_w - start_pos_w, dim=1)

    # if remaining distance is less than 30% the total distance
    move_up = torch.mean(1.0*((dist_to_goal < (0.3 * total_mission_dist))))
    # if covered less than 20% of the distance
    move_down = torch.mean(1.0 * (dist_from_start < (0.2 * total_mission_dist)))
    # print(f"UP - DOWN: {move_up - move_down}")

    mean_level_increment = move_up - move_down
    pose_command = env.command_manager.get_term(command_name)

    pos_x = pose_command.cfg.ranges.pos_x
    current_val = pos_x[1]
    new_val = current_val + mean_level_increment * 0.01

    # Clamp the tensor
    new_pos_x_abs = torch.clamp(new_val, min=4.0, max=6.0)

    # convert to standard Python float for the config
    new_limit = new_pos_x_abs.item()

    # Update config with floats
    pose_command.cfg.ranges.pos_x = (-new_limit, new_limit)
    pose_command.cfg.ranges.pos_y = (-new_limit, new_limit)

    return new_pos_x_abs



def obstacle_angle_level(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    command_name: str = "pose_command",
) -> torch.Tensor:
    """
    Curriculum that decreases the goal offset angle as the robot improves.
    Starts at high angle (easy), goes to 0 (hard).
    """

    cmd_term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]

    target_pos_w = cmd_term.pos_command_w[env_ids, :2]
    root_pos_w = asset.data.root_pos_w[env_ids, :2]
    start_pos_w = env.scene.env_origins[env_ids, :2]

    total_dist = torch.norm(target_pos_w - start_pos_w, dim=1)
    current_dist = torch.norm(target_pos_w - root_pos_w, dim=1)

    # Success: Remaining distance is < 20% of total (Robot reached goal)
    is_success = current_dist < (0.2 * total_dist)
    success_rate = torch.mean(is_success.float())

    # Get current max angle
    current_max_angle = cmd_term.cfg.goal_pose_angle_range[1]

    # Simple Logic:
    # If success rate > 70%, decrease angle by 0.05 radians
    # If success rate < 40%, increase angle by 0.01 radians (make easier)

    if success_rate > 0.7:
        new_angle = max(0.0, current_max_angle - 0.05)
    elif success_rate < 0.4:
        new_angle = min(1.57, current_max_angle + 0.01)
    else:
        new_angle = current_max_angle

    cmd_term.cfg.goal_pose_angle_range = (-new_angle, new_angle)

    return torch.tensor(new_angle)

