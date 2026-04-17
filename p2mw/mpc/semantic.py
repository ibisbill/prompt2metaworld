"""
Semantic state parsing: convert raw MetaWorld observations into
structured, human-readable representations that LLMs can reason about.

MetaWorld 39-dim observation layout:
  [0:3]   gripper position (x, y, z)
  [3]     gripper state — 1.0 = fully open, 0.0 = fully closed
  [4:7]   object-1 position (x, y, z)
  [7:11]  object-1 quaternion (w, x, y, z)
  [11:14] object-2 position (x, y, z)
  [14:18] object-2 quaternion (w, x, y, z)
  [18:36] previous timestep (same layout as [0:18])
  [36:39] goal position (x, y, z)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SemanticState:
    gripper_pos: np.ndarray         # (3,)  current gripper xyz
    gripper_state: float            # [0,1] 1 = open
    obj1_pos: np.ndarray            # (3,)
    obj1_quat: np.ndarray           # (4,)  w,x,y,z
    obj1_rotation_deg: float        # scalar rotation magnitude in degrees
    obj2_pos: np.ndarray            # (3,)
    goal_pos: np.ndarray            # (3,)
    gripper_to_obj1: np.ndarray     # (3,)  displacement vector
    dist_gripper_to_obj1: float
    obj1_to_goal: np.ndarray        # (3,)  displacement vector
    dist_obj1_to_goal: float
    gripper_vel: Optional[np.ndarray] = None   # estimated from prev_obs
    obj1_vel: Optional[np.ndarray] = None

    def to_text(self) -> str:
        lines = [
            "--- Semantic State ---",
            (f"Gripper:  pos=({self.gripper_pos[0]:.3f}, {self.gripper_pos[1]:.3f},"
             f" {self.gripper_pos[2]:.3f})  "
             f"{'open' if self.gripper_state > 0.5 else 'closed'}"
             f" (state={self.gripper_state:.2f})"),
            (f"Object 1: pos=({self.obj1_pos[0]:.3f}, {self.obj1_pos[1]:.3f},"
             f" {self.obj1_pos[2]:.3f})  rotation={self.obj1_rotation_deg:.1f}°"),
            (f"Object 2: pos=({self.obj2_pos[0]:.3f}, {self.obj2_pos[1]:.3f},"
             f" {self.obj2_pos[2]:.3f})"),
            (f"Goal:     pos=({self.goal_pos[0]:.3f}, {self.goal_pos[1]:.3f},"
             f" {self.goal_pos[2]:.3f})"),
            (f"Gripper → Object 1: "
             f"[{self.gripper_to_obj1[0]:.3f}, {self.gripper_to_obj1[1]:.3f},"
             f" {self.gripper_to_obj1[2]:.3f}]  dist={self.dist_gripper_to_obj1:.3f} m"),
            (f"Object 1 → Goal:    "
             f"[{self.obj1_to_goal[0]:.3f}, {self.obj1_to_goal[1]:.3f},"
             f" {self.obj1_to_goal[2]:.3f}]  dist={self.dist_obj1_to_goal:.3f} m"),
        ]
        if self.gripper_vel is not None:
            lines.append(
                f"Gripper vel:  [{self.gripper_vel[0]:.4f}, {self.gripper_vel[1]:.4f},"
                f" {self.gripper_vel[2]:.4f}]"
            )
            lines.append(
                f"Object 1 vel: [{self.obj1_vel[0]:.4f}, {self.obj1_vel[1]:.4f},"
                f" {self.obj1_vel[2]:.4f}]"
            )
        lines.append("----------------------")
        return "\n".join(lines)

    def progress_estimate(self) -> float:
        """Rough [0,1] progress: how close object-1 is to the goal."""
        max_dist = 1.0  # normalization constant (metres)
        return float(max(0.0, 1.0 - self.dist_obj1_to_goal / max_dist))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotation_deg(q: np.ndarray) -> float:
    """Rotation magnitude in degrees from a (w,x,y,z) quaternion."""
    q = np.array(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return 0.0
    q = q / norm
    angle_rad = 2.0 * np.arccos(float(np.clip(abs(q[0]), 0.0, 1.0)))
    return float(np.degrees(angle_rad))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_obs(obs: np.ndarray, prev_obs: Optional[np.ndarray] = None) -> SemanticState:
    """
    Parse a raw 39-dim MetaWorld observation into a SemanticState.

    Args:
        obs:      Current 39-dim observation.
        prev_obs: Previous 39-dim observation, used to estimate velocity.
                  Pass None on the first step.
    """
    obs = np.array(obs, dtype=float)

    gripper_pos   = obs[0:3]
    gripper_state = float(obs[3])
    obj1_pos      = obs[4:7]
    obj1_quat     = obs[7:11]
    obj2_pos      = obs[11:14]
    goal_pos      = obs[36:39]

    gripper_to_obj1      = obj1_pos - gripper_pos
    dist_gripper_to_obj1 = float(np.linalg.norm(gripper_to_obj1))
    obj1_to_goal         = goal_pos - obj1_pos
    dist_obj1_to_goal    = float(np.linalg.norm(obj1_to_goal))
    obj1_rotation_deg    = _rotation_deg(obj1_quat)

    gripper_vel = obj1_vel = None
    if prev_obs is not None:
        prev_obs    = np.array(prev_obs, dtype=float)
        gripper_vel = obs[0:3] - prev_obs[0:3]
        obj1_vel    = obs[4:7] - prev_obs[4:7]

    return SemanticState(
        gripper_pos=gripper_pos,
        gripper_state=gripper_state,
        obj1_pos=obj1_pos,
        obj1_quat=obj1_quat,
        obj1_rotation_deg=obj1_rotation_deg,
        obj2_pos=obj2_pos,
        goal_pos=goal_pos,
        gripper_to_obj1=gripper_to_obj1,
        dist_gripper_to_obj1=dist_gripper_to_obj1,
        obj1_to_goal=obj1_to_goal,
        dist_obj1_to_goal=dist_obj1_to_goal,
        gripper_vel=gripper_vel,
        obj1_vel=obj1_vel,
    )
