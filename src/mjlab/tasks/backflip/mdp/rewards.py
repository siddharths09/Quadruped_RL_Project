from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.third_party.isaaclab.isaaclab.utils.math import euler_xyz_from_quat

import math

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

_CMD_IDX_PHASE = 0
_CMD_IDX_Z_DES = 1
_CMD_IDX_SIN_PITCH_DES = 2
_CMD_IDX_COS_PITCH_DES = 3
_CMD_IDX_PITCH_RATE_DES = 4


def track_base_height(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  assert cmd is not None, f"Command '{command_name}' not found."
  z = asset.data.root_link_pos_w[:, 2]
  z_des = cmd[:, _CMD_IDX_Z_DES]
  return torch.exp(-torch.square(z - z_des) / (std**2))


def track_projected_gravity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  assert cmd is not None, f"Command '{command_name}' not found."
  sin_pitch = cmd[:, _CMD_IDX_SIN_PITCH_DES]
  cos_pitch = cmd[:, _CMD_IDX_COS_PITCH_DES]
  g_des = torch.stack(
    [sin_pitch, torch.zeros_like(sin_pitch), -cos_pitch], dim=-1
  )
  g_act = asset.data.projected_gravity_b
  err = torch.sum(torch.square(g_act - g_des), dim=1)
  return torch.exp(-err / (std**2))


def track_pitch_rate(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  start_phase: float,
  end_phase: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  assert cmd is not None, f"Command '{command_name}' not found."
  phase = cmd[:, _CMD_IDX_PHASE]
  rate_des = cmd[:, _CMD_IDX_PITCH_RATE_DES]
  rate_act = asset.data.root_link_ang_vel_b[:, 1]
  active = (phase >= start_phase) & (phase <= end_phase)
  reward = torch.exp(-torch.square(rate_act - rate_des) / (std**2))
  return reward * active.to(dtype=torch.float32)


def feet_airborne(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  start_phase: float,
  end_phase: float,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  in_contact = (found > 0).to(dtype=torch.float32)  # [B, N]
  airborne_frac = 1.0 - torch.mean(in_contact, dim=1)
  cmd = env.command_manager.get_command(command_name)
  assert cmd is not None, f"Command '{command_name}' not found."
  phase = cmd[:, _CMD_IDX_PHASE]
  active = (phase >= start_phase) & (phase <= end_phase)
  return airborne_frac * active.to(dtype=torch.float32)

# BACKFLIP REWARDS

def track_base_pitch(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward tracking desired pitch angle during flip."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    
    # Command format: [phase, desired_height, desired_pitch]
    desired_pitch = command[:, 2]
    
    # Get actual pitch from quaternion
    quat = asset.data.root_link_quat_w
    euler = euler_xyz_from_quat(quat)
    _, actual_pitch, _ = euler_xyz_from_quat(quat)   # Pitch is the y-axis rotation
    
    err = torch.atan2(torch.sin(actual_pitch - desired_pitch),
                      torch.cos(actual_pitch - desired_pitch))

    pitch_error = err * err
    return torch.exp(-pitch_error / std**2)

def track_base_yaw(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward tracking desiredyaw angle at end of flip."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    
    phase = command[:, 0]
    desired_yaw = torch.zeros_like(phase) 
    
    # Get actual pitch from quaternion
    quat = asset.data.root_link_quat_w
    euler = euler_xyz_from_quat(quat)
    _, _, actual_yaw = euler_xyz_from_quat(quat) 
    
    err = torch.atan2(
        torch.sin(actual_yaw - desired_yaw),
        torch.cos(actual_yaw - desired_yaw),
    )
    yaw_error = err * err
    active = (phase >= 0.8).to(dtype=torch.float32) 
    
    return torch.exp(-yaw_error / std**2) * active


def joint_velocity_penalty(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize excessive joint velocities."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_vel), dim=1)


def landing_success_bonus(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Bonus reward for successful landing at end of flip."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    
    # Command format: [phase, desired_height, desired_pitch]
    phase = command[:, 0]
    
    # Only give bonus near end of flip
    near_end = (phase > 0.9).float()
    
    # Check if upright
    quat = asset.data.root_link_quat_w
    euler = euler_xyz_from_quat(quat)
    pitch = euler[1]
    roll = euler[0]
    
    upright = (torch.abs(pitch) < 0.3) & (torch.abs(roll) < 0.3)

    #check if rotation happened
    did_invert = (torch.abs(torch.abs(pitch) - math.pi) < 0.7)
    
    # Check if at correct height
    actual_height = asset.data.root_link_pos_w[:, 2]
    at_correct_height = torch.abs(actual_height - 0.35) < 0.1
    
    # Bonus if upright and at correct height near end
    success = upright.float() * near_end * did_invert.float()
    
    return success
