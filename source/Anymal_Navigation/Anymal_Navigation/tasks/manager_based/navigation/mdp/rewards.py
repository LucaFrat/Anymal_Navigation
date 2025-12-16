# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""

    asset = env.scene[asset_cfg.name]
    root_pos_w = asset.data.root_pos_w[:, :2]  # XY position in World

    # get the Goal Position (from the episode that just finished)
    cmd_term = env.command_manager.get_term(command_name)

    target_pos_w = cmd_term.pos_command_w[:, :2]

    # assumes the robot spawns relatively close to the origin (0,0) of its env instance
    start_pos_w = env.scene.env_origins[:, :2]

    # calculate distances
    total_mission_dist = torch.norm(target_pos_w - start_pos_w, dim=1)
    dist_to_goal = torch.norm(target_pos_w - root_pos_w, dim=1)
    distance = dist_to_goal / total_mission_dist * 3.0
    # print(f"DISTANCE: {distance}")
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()


def face_target(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize not facing the target when walking toward it"""
    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]
    target_angle = torch.atan2(goal_pos[:, 1], goal_pos[:, 0])
    return torch.abs(target_angle)
