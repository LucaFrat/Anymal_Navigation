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


def progress_toward_goal(env, command_name: str) -> torch.Tensor:
    """Reward for moving closer to the goal compared to the previous step."""

    cmd_term = env.command_manager.get_term(command_name)
    target_pos_b = cmd_term.command[:, :2] # (x, y) in base frame

    robot = env.scene["robot"]
    vel_b = robot.data.root_lin_vel_b[:, :2]

    dist = torch.norm(target_pos_b, dim=1, keepdim=True)
    dir_to_goal = target_pos_b / (dist + 1e-6)

    # how much of our velocity is towards the goal?
    progress = torch.sum(vel_b * dir_to_goal, dim=1)

    return progress


def cone_proximity_penalty(
    env,
    robot_cfg: SceneEntityCfg,
    cone_cfg: SceneEntityCfg,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Penalize getting closer than 'threshold' to the cone.
    Reward = -(threshold - distance) if distance < threshold, else 0.
    """
    robot = env.scene[robot_cfg.name]
    cone = env.scene[cone_cfg.name]

    dist = torch.norm(robot.data.root_pos_w[:, :2] - cone.data.root_pos_w[:, :2], dim=-1)

    # how much closer than threshold are we?
    violation = torch.clamp(threshold - dist, min=0.0) / threshold

    return violation


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""

    asset = env.scene[asset_cfg.name]
    root_pos_w = asset.data.root_pos_w[:, :2]

    cmd_term = env.command_manager.get_term(command_name)

    target_pos_w = cmd_term.pos_command_w[:, :2]

    start_pos_w = env.scene.env_origins[:, :2]

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
