"""Backflip command generator for Go2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

import torch
import math

from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass
class BackflipCommandCfg(CommandTermCfg):
    """Configuration for backflip command generator.
    
    Provides phase-based reference trajectories for:
    - Phase variable (0 to 1)
    - Desired base height
    - Desired pitch angle (rotation)
    """
    
    flip_duration: float = 1.5  # Total duration of backflip in seconds
    
    # Height trajectory parameters
    initial_height: float = 0.35  # Starting height (m)
    jump_height: float = 0.6     # Peak height during flip (m)
    landing_height: float = 0.35  # Target landing height (m)
    
    # Pitch trajectory parameters (in radians)
    initial_pitch: float = 0.0           # Starting pitch
    flip_rotation: float = -2 * math.pi  # Full backflip rotation (negative = backward)
    landing_pitch: float = 0.0           # Target landing pitch
    
    class_type: type = field(default_factory=lambda: BackflipCommand)


class BackflipCommand(CommandTerm):
    """Backflip command generator.
    
    Generates phase-based reference trajectories for backflip execution:
    - Phase ∈ [0, 1]: Progress through the flip
    - Desired height: Jump up, peak at mid-flip, land
    - Desired pitch: Rotate backward throughout flip
    """
    
    def __init__(self, cfg: BackflipCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        
        # Phase variable (0 to 1 representing flip progress)
        self.phase = torch.zeros(env.num_envs, device=env.device)
        
        # Time elapsed in current flip
        self.time_elapsed = torch.zeros(env.num_envs, device=env.device)
        
        # Command output: [phase, desired_height, desired_pitch]
        self._command = torch.zeros(env.num_envs, 3, device=env.device)

    @property
    def command(self):
        return self._command
    
    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        """Reset command for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        
        self.phase[env_ids] = 0.0
        self.time_elapsed[env_ids] = 0.0
        self._compute_command(env_ids)

        return {}
    
    def compute(self, dt: float):
        """Update command based on elapsed time."""
        # Update time and phase
        self.time_elapsed += dt
        self.phase = torch.clamp(self.time_elapsed / self.cfg.flip_duration, 0.0, 1.0)
        
        # Update command for all environments
        env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        self._compute_command(env_ids)
    
    
    def _compute_command(self, env_ids: torch.Tensor):
        """Compute reference trajectories based on phase.
        
        Height trajectory:
        - Phase 0.0-0.3: Jump up (takeoff)
        - Phase 0.3-0.7: At peak height (rotating)
        - Phase 0.7-1.0: Descend and land
        
        Pitch trajectory:
        - Phase 0.0-0.2: Prepare for rotation
        - Phase 0.2-0.8: Main rotation (full backflip)
        - Phase 0.8-1.0: Stabilize for landing
        """
        phase = self.phase[env_ids]
        
        # --- Desired Height Trajectory ---
        # Use a parabolic jump: go up, peak at mid-flip, come down
        desired_height = torch.zeros_like(phase)
        
        # Takeoff phase (0.0 - 0.3): Jump from initial to peak height
        takeoff_mask = phase < 0.3
        takeoff_progress = phase[takeoff_mask] / 0.3
        desired_height[takeoff_mask] = (
            self.cfg.initial_height + 
            (self.cfg.jump_height - self.cfg.initial_height) * takeoff_progress
        )
        
        # Mid-air phase (0.3 - 0.7): Maintain peak height
        midair_mask = (phase >= 0.3) & (phase < 0.7)
        desired_height[midair_mask] = self.cfg.jump_height
        
        # Landing phase (0.7 - 1.0): Descend to landing height
        landing_mask = phase >= 0.7
        landing_progress = (phase[landing_mask] - 0.7) / 0.3
        desired_height[landing_mask] = (
            self.cfg.jump_height - 
            (self.cfg.jump_height - self.cfg.landing_height) * landing_progress
        )
        
        # --- Desired Pitch Trajectory ---
        # Smoothly rotate backward throughout the flip
        desired_pitch = torch.zeros_like(phase)
        
        # Preparation phase (0.0 - 0.2): Minimal rotation
        prep_mask = phase < 0.2
        prep_progress = phase[prep_mask] / 0.2
        desired_pitch[prep_mask] = (
            self.cfg.initial_pitch + 
            0.1 * self.cfg.flip_rotation * prep_progress
        )
        
        # Main rotation phase (0.2 - 0.8): Full backflip rotation
        rotation_mask = (phase >= 0.2) & (phase < 0.8)
        rotation_progress = (phase[rotation_mask] - 0.2) / 0.6
        desired_pitch[rotation_mask] = (
            self.cfg.initial_pitch + 
            self.cfg.flip_rotation * rotation_progress
        )
        
        # Stabilization phase (0.8 - 1.0): Finish rotation and stabilize
        stabilize_mask = phase >= 0.8
        stabilize_progress = (phase[stabilize_mask] - 0.8) / 0.2
        desired_pitch[stabilize_mask] = (
            self.cfg.initial_pitch + self.cfg.flip_rotation + 
            (self.cfg.landing_pitch - self.cfg.initial_pitch - self.cfg.flip_rotation) * stabilize_progress
        )
        
        # Update command: [phase, desired_height, desired_pitch]
        self.command[env_ids, 0] = phase
        self.command[env_ids, 1] = desired_height
        self.command[env_ids, 2] = desired_pitch