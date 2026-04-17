"""Unit tests for p2mw.mpc.semantic — no MetaWorld install required."""

import numpy as np
import pytest

from p2mw.mpc.semantic import parse_obs, SemanticState


def _make_obs(
    gripper_pos=(0.1, 0.2, 0.3),
    gripper_state=1.0,
    obj1_pos=(0.5, 0.0, 0.1),
    obj1_quat=(1.0, 0.0, 0.0, 0.0),   # identity
    obj2_pos=(0.0, 0.5, 0.0),
    goal_pos=(0.8, 0.0, 0.1),
) -> np.ndarray:
    """Build a synthetic 39-dim observation with controllable fields."""
    obs = np.zeros(39)
    obs[0:3]   = gripper_pos
    obs[3]     = gripper_state
    obs[4:7]   = obj1_pos
    obs[7:11]  = obj1_quat
    obs[11:14] = obj2_pos
    # [14:36] previous timestep — leave as zeros
    obs[36:39] = goal_pos
    return obs


class TestParseObs:
    def test_basic_fields_extracted(self):
        obs = _make_obs()
        s = parse_obs(obs)

        np.testing.assert_allclose(s.gripper_pos, [0.1, 0.2, 0.3])
        assert s.gripper_state == pytest.approx(1.0)
        np.testing.assert_allclose(s.obj1_pos, [0.5, 0.0, 0.1])
        np.testing.assert_allclose(s.goal_pos, [0.8, 0.0, 0.1])

    def test_distances_computed(self):
        obs = _make_obs(
            gripper_pos=(0.0, 0.0, 0.0),
            obj1_pos=(3.0, 4.0, 0.0),
            goal_pos=(3.0, 4.0, 0.0),
        )
        s = parse_obs(obs)
        assert s.dist_gripper_to_obj1 == pytest.approx(5.0)
        assert s.dist_obj1_to_goal == pytest.approx(0.0)

    def test_displacement_vectors(self):
        obs = _make_obs(
            gripper_pos=(1.0, 0.0, 0.0),
            obj1_pos=(2.0, 0.0, 0.0),
            goal_pos=(5.0, 0.0, 0.0),
        )
        s = parse_obs(obs)
        np.testing.assert_allclose(s.gripper_to_obj1, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(s.obj1_to_goal,    [3.0, 0.0, 0.0])

    def test_velocity_none_without_prev(self):
        obs = _make_obs()
        s = parse_obs(obs)
        assert s.gripper_vel is None
        assert s.obj1_vel is None

    def test_velocity_estimated_with_prev(self):
        prev = _make_obs(gripper_pos=(0.0, 0.0, 0.0), obj1_pos=(0.0, 0.0, 0.0))
        curr = _make_obs(gripper_pos=(0.1, 0.0, 0.0), obj1_pos=(0.0, 0.2, 0.0))
        s = parse_obs(curr, prev_obs=prev)
        np.testing.assert_allclose(s.gripper_vel, [0.1, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(s.obj1_vel,    [0.0, 0.2, 0.0], atol=1e-6)

    def test_identity_quaternion_gives_zero_rotation(self):
        obs = _make_obs(obj1_quat=(1.0, 0.0, 0.0, 0.0))
        s = parse_obs(obs)
        assert s.obj1_rotation_deg == pytest.approx(0.0, abs=1e-4)

    def test_to_text_contains_key_fields(self):
        obs = _make_obs()
        s = parse_obs(obs)
        text = s.to_text()
        assert "Gripper" in text
        assert "Object 1" in text
        assert "Goal" in text
        assert "dist=" in text

    def test_progress_estimate_zero_at_goal(self):
        obs = _make_obs(obj1_pos=(0.8, 0.0, 0.1), goal_pos=(0.8, 0.0, 0.1))
        s = parse_obs(obs)
        assert s.progress_estimate() == pytest.approx(1.0)

    def test_progress_estimate_bounded(self):
        obs = _make_obs(obj1_pos=(0.0, 0.0, 0.0), goal_pos=(10.0, 0.0, 0.0))
        s = parse_obs(obs)
        assert 0.0 <= s.progress_estimate() <= 1.0
