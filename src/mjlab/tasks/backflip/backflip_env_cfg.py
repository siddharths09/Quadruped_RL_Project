"""Backflip task configuration for Go2."""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.entity.entity import EntityCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
    ActionTermCfg,
    CommandTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.viewer import ViewerConfig

# Import Go2 robot
from mjlab.asset_zoo.robots.unitree_go2 import get_go2_robot_cfg, GO2_ACTION_SCALE

# Import MDP functions
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.backflip.mdp import BackflipCommandCfg
from mjlab.envs import mdp as common_mdp

from mjlab.tasks.backflip import mdp

# Scene configuration - flat terrain for backflip
SCENE_CFG = SceneCfg(
    terrain=TerrainImporterCfg(
        terrain_type="plane",  # Flat ground for precision maneuver
    ),
    num_envs=4096,
    extent=2.0,
)

# Viewer configuration
VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    asset_name="robot",
    body_name="trunk",
    distance=3.0,
    elevation=-5.0,
    azimuth=90.0,
)

# Simulation configuration
SIM_CFG = SimulationCfg(
    nconmax=35,
    njmax=300,
    mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
    ),
)


def create_backflip_env_cfg(
  robot_cfg: EntityCfg,
  action_scale: float | dict[str, float],
  viewer_body_name: str,
  site_names: tuple[str, ...],
  feet_sensor_cfg: ContactSensorCfg,
  self_collision_sensor_cfg: ContactSensorCfg,
  foot_friction_geom_names: tuple[str, ...] | str,
) -> ManagerBasedRlEnvCfg:
    """Create backflip task configuration for Go2."""
    
    scene = deepcopy(SCENE_CFG)
    scene.entities = {"robot": get_go2_robot_cfg()}
    
    scene.sensors = (
      feet_sensor_cfg,
      self_collision_sensor_cfg,
    )
    
    viewer = deepcopy(VIEWER_CONFIG)
    
    # ---------------------------------------------------------------------------
    # Actions
    # ---------------------------------------------------------------------------
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            asset_name="robot",
            actuator_names=(".*",),
            scale=GO2_ACTION_SCALE,
            use_default_offset=True,
        )
    }
    
    # ---------------------------------------------------------------------------
    # Part4 (b) - Backflip Command
    # ---------------------------------------------------------------------------
    commands: dict[str, CommandTermCfg] = {
        "backflip": BackflipCommandCfg(
            flip_duration=1.5,
            initial_height=0.35,
            jump_height=0.6,
            landing_height=0.35,
            initial_pitch=0.0,
            flip_rotation=-2 * math.pi,  # Full backflip
            landing_pitch=0.0,
            resampling_time_range=(3.0, 8.0)
        )
    }
    
    # ---------------------------------------------------------------------------
    # Observations
    # ---------------------------------------------------------------------------
    policy_terms: dict[str, ObservationTermCfg] = {
        # Robot state observations (no noise for precision control)
        "base_lin_vel": ObservationTermCfg(
            func=velocity_mdp.base_lin_vel,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=velocity_mdp.base_ang_vel,
        ),
        "projected_gravity": ObservationTermCfg(
            func=velocity_mdp.projected_gravity,
        ),
        "joint_pos": ObservationTermCfg(
            func=velocity_mdp.joint_pos_rel,
        ),
        "joint_vel": ObservationTermCfg(
            func=velocity_mdp.joint_vel_rel,
        ),
        "last_action": ObservationTermCfg(
            func=velocity_mdp.last_action,
        ),
        # Backflip-specific observations
        "backflip_command": ObservationTermCfg(
            func=velocity_mdp.generated_commands,
            params={"command_name": "backflip"},
        ),
    }
    critic_terms = {
        # Robot state observations (no noise for precision control)
        "base_lin_vel": ObservationTermCfg(
            func=velocity_mdp.base_lin_vel,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=velocity_mdp.base_ang_vel,
        ),
        "projected_gravity": ObservationTermCfg(
            func=velocity_mdp.projected_gravity,
        ),
        "joint_pos": ObservationTermCfg(
            func=velocity_mdp.joint_pos_rel,
        ),
        "joint_vel": ObservationTermCfg(
            func=velocity_mdp.joint_vel_rel,
        ),
        "last_action": ObservationTermCfg(
            func=velocity_mdp.last_action,
        ),
        # Backflip-specific observations
        "backflip_command": ObservationTermCfg(
            func=velocity_mdp.generated_commands,
            params={"command_name": "backflip"},
        ),
    }
    
    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }
    
    # ---------------------------------------------------------------------------
    # Rewards
    # ---------------------------------------------------------------------------
    rewards = {
        # --- Main Objectives: Track desired height and pitch ---
        "track_height": RewardTermCfg(
            func=mdp.track_base_height,
            weight=2.0,
            params={
                "std": 0.3,
                "command_name": "backflip",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "track_pitch": RewardTermCfg(
            func= mdp.track_base_pitch,
            weight=2.0,
            params={
                "std": 0.5,
                "command_name": "backflip",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        
        # --- Regularizations for smooth execution ---
        "action_rate": RewardTermCfg(
            func=velocity_mdp.action_rate_l2,
            weight=-0.01,
        ),

        # --- Landing bonus ---
        "landing_bonus": RewardTermCfg(
            func=mdp.landing_success_bonus,
            weight=10.0,
            params={
                "command_name": "backflip",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
    }
    
    # ---------------------------------------------------------------------------
    # Terminations
    # ---------------------------------------------------------------------------
    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,
        ),
        "base_contact": TerminationTermCfg(
            func=mdp.illegal_contact,
            time_out=False,
            params={
                "sensor_name":  self_collision_sensor_cfg.name,
            },
        ),
    }
    
    # ---------------------------------------------------------------------------
    # Events
    # ---------------------------------------------------------------------------
    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }
    
    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        actions=actions,
        commands=commands,
        rewards=rewards,
        terminations=terminations,
        events=events,
        sim=SIM_CFG,
        viewer=viewer,
        decimation=4,
        episode_length_s=2.0,  # Short episode for backflip
    )