# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


import Anymal_Navigation.tasks.manager_based.navigation.mdp as mdp
from Anymal_Navigation.tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import AnymalCRoughEnvCfg
from Anymal_Navigation.tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg


LOW_LEVEL_ENV_CFG = AnymalCRoughEnvCfg()
# LOW_LEVEL_ENV_CFG = AnymalCFlatEnvCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_cone_pos = EventTerm(
        func=mdp.reset_cone_pos_donut,
        mode="reset",  # Randomize on every reset (or use "startup" for once-only)
        params={
            "asset_cfg": SceneEntityCfg("cone"), # Must match the name in MySceneCfg
            "radius_range": (1.0, 2.5),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path="logs/rsl_rl/anymal_c_rough/2025-12-15_13-20-18/exported/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        visual_features = ObsTerm(
            func=mdp.visual_latent,
            params={"sensor_cfg": SceneEntityCfg("camera")},
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=5.0,
        params={"std": 2.0,
                "command_name": "pose_command",
                "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=1.0,
        params={"std": 0.2,
                "command_name": "pose_command",
                "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )

    # penalize high velocity commands to encourage efficiency
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.001)

    # penalize jerky command changes
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)

    # face_target = RewTerm(
    #     func=mdp.face_target,
    #     weight= -0.1,
    #     params={"command_name": "pose_command"}
    # )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-5.0, 5.0), pos_y=(-5.0, 5.0), heading=(-math.pi, math.pi)),
    )
    # pose_command = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="base",
    #     resampling_time_range=(8.0, 8.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(-2.0, 2.0),
    #         pos_y=(-2.0, 2.0),
    #         pos_z=(1.0, 1.1),
    #         roll=(0.0, 0.0),
    #         pitch=(0.0, 0.0),
    #         yaw=(-math.pi, math.pi)),
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(
    #     func=mdp.terrain_levels_progress,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "command_name": "pose_command",
    #     },
    # )
    goal_distance = CurrTerm(
        func=mdp.distance_level,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "pose_command",
        }
    )


@configclass
class NavigationRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # environment settings
    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        self.scene.terrain.max_init_terrain_level = None


        # if getattr(self.curriculum, "terrain_levels", None) is not None:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = True
        # else:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = False


class NavigationEnvCfg_PLAY(NavigationRoughEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
