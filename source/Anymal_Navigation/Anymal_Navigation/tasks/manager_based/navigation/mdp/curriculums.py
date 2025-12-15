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
    dist_to_goal = torch.norm(target_pos_w - root_pos_w, dim=1)
    dist_from_start = torch.norm(root_pos_w - start_pos_w, dim=1)

    # if remaining distance is less than 30% the total distance
    move_up = torch.mean(1.0*((dist_to_goal < (0.3 * total_mission_dist))))
    # if covered less than 20% of the distance
    move_down = torch.mean(1.0 * (dist_from_start < (0.2 * total_mission_dist)))
    print(f"UP - DOWN: {move_up - move_down}")

    mean_level_increment = move_up - move_down
    pose_command = env.command_manager.get_term(command_name)

    pos_x = pose_command.cfg.ranges.pos_x
    new_pos_x_abs = torch.clamp(torch.tensor(pos_x[1] + mean_level_increment*0.01), min=1.0, max=4.0)
    pose_command.cfg.ranges.pos_x = (-new_pos_x_abs, new_pos_x_abs)
    pose_command.cfg.ranges.pos_y = (-new_pos_x_abs, new_pos_x_abs)

    return torch.tensor(pose_command.cfg.ranges.pos_x[1])


