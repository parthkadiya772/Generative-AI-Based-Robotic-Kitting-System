"""
Unit tests for the Execution Layer (safety, motion library, controller).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.safety import (
    WorkspaceBoundary,
    check_gripper_force,
    ActionTimeout,
    SafetyMonitor,
)
from execution.motion_library import (
    compute_pre_grasp_approach,
    compute_place_sequence,
    compute_transit_path,
    compute_full_pick_place_sequence,
)


# ═════════════════════════════════════════════════════════════
# WORKSPACE BOUNDARY TESTS
# ═════════════════════════════════════════════════════════════

class TestWorkspaceBoundary:
    """Tests for workspace boundary validation."""

    def setup_method(self):
        self.bounds = WorkspaceBoundary({
            "x_min": -0.2, "x_max": 0.8,
            "y_min": -0.5, "y_max": 0.5,
            "z_min": 0.0, "z_max": 0.4,
        })

    def test_valid_position(self):
        assert self.bounds.check(0.3, 0.1, 0.2) is True

    def test_x_too_low(self):
        assert self.bounds.check(-0.5, 0.0, 0.1) is False

    def test_x_too_high(self):
        assert self.bounds.check(1.5, 0.0, 0.1) is False

    def test_y_too_low(self):
        assert self.bounds.check(0.3, -0.8, 0.1) is False

    def test_z_too_low(self):
        assert self.bounds.check(0.3, 0.0, -0.1) is False

    def test_z_too_high(self):
        assert self.bounds.check(0.3, 0.0, 0.6) is False

    def test_boundary_inclusive(self):
        """Positions exactly at the boundary should be valid."""
        assert self.bounds.check(-0.2, -0.5, 0.0) is True
        assert self.bounds.check(0.8, 0.5, 0.4) is True

    def test_clamp(self):
        """Clamping should bring values within bounds."""
        cx, cy, cz = self.bounds.clamp(1.5, -1.0, 0.6)
        assert cx == 0.8
        assert cy == -0.5
        assert cz == 0.4

    def test_clamp_unchanged(self):
        """Clamping within-bounds values should not change them."""
        cx, cy, cz = self.bounds.clamp(0.3, 0.1, 0.2)
        assert cx == 0.3
        assert cy == 0.1
        assert cz == 0.2


# ═════════════════════════════════════════════════════════════
# GRIPPER FORCE TESTS
# ═════════════════════════════════════════════════════════════

class TestGripperForce:

    def test_valid_force(self):
        assert check_gripper_force(30.0, 40.0) is True

    def test_zero_force(self):
        assert check_gripper_force(0.0, 40.0) is True

    def test_excessive_force(self):
        assert check_gripper_force(50.0, 40.0) is False

    def test_negative_force(self):
        assert check_gripper_force(-5.0, 40.0) is False


# ═════════════════════════════════════════════════════════════
# MOTION LIBRARY TESTS
# ═════════════════════════════════════════════════════════════

class TestMotionLibrary:
    """Tests for motion waypoint generation."""

    def test_pre_grasp_waypoint_count(self):
        """Pre-grasp approach should produce 3 waypoints."""
        waypoints = compute_pre_grasp_approach(
            target_x=0.3, target_y=0.1, target_z=0.02,
            safe_z=0.3,
        )
        assert len(waypoints) == 3
        assert waypoints[0]["name"] == "safe_height"
        assert waypoints[1]["name"] == "hover"
        assert waypoints[2]["name"] == "grasp"

    def test_pre_grasp_z_ordering(self):
        """Waypoints should descend from safe_z to grasp_z."""
        waypoints = compute_pre_grasp_approach(
            target_x=0.3, target_y=0.1, target_z=0.02,
            safe_z=0.3,
        )
        assert waypoints[0]["z"] > waypoints[1]["z"] > waypoints[2]["z"]

    def test_place_sequence_waypoint_count(self):
        """Place sequence should produce 3 waypoints."""
        waypoints = compute_place_sequence(
            target_x=0.5, target_y=0.0, target_z=0.02,
            safe_z=0.3,
        )
        assert len(waypoints) == 3
        assert waypoints[0]["name"] == "safe_above_dest"
        assert waypoints[1]["name"] == "place_descend"
        assert waypoints[2]["name"] == "retract"

    def test_transit_path_waypoint_count(self):
        """Transit path should produce 2 waypoints."""
        waypoints = compute_transit_path(
            start_x=0.3, start_y=0.1,
            end_x=0.5, end_y=0.0,
            transit_z=0.5,
        )
        assert len(waypoints) == 2
        assert waypoints[0]["name"] == "lift_to_transit"
        assert waypoints[1]["name"] == "lateral_move"

    def test_transit_same_z(self):
        """Both transit waypoints should be at the same Z."""
        waypoints = compute_transit_path(0.3, 0.1, 0.5, 0.0, 0.5)
        assert waypoints[0]["z"] == waypoints[1]["z"]

    def test_full_pick_place_sequence(self):
        """Full sequence should produce all three phases."""
        result = compute_full_pick_place_sequence(
            pick_x=0.3, pick_y=0.1, pick_z=0.02,
            place_x=0.5, place_y=0.0, place_z=0.02,
            safe_z=0.3, transit_z=0.5,
        )
        assert "approach" in result
        assert "transit" in result
        assert "place" in result
        assert len(result["approach"]) == 3
        assert len(result["transit"]) == 2
        assert len(result["place"]) == 3

    def test_tcp_z_offset_applied(self):
        """TCP Z offset should shift all waypoints."""
        wp_no_offset = compute_pre_grasp_approach(0.3, 0.1, 0.02, 0.3, tcp_z_offset=0.0)
        wp_with_offset = compute_pre_grasp_approach(0.3, 0.1, 0.02, 0.3, tcp_z_offset=0.15)

        for i in range(len(wp_no_offset)):
            assert wp_with_offset[i]["z"] == pytest.approx(
                wp_no_offset[i]["z"] + 0.15, abs=0.001
            )


# ═════════════════════════════════════════════════════════════
# SAFETY MONITOR TESTS
# ═════════════════════════════════════════════════════════════

class TestSafetyMonitor:

    def setup_method(self):
        self.monitor = SafetyMonitor({
            "workspace_bounds": {
                "x_min": -0.2, "x_max": 0.8,
                "y_min": -0.5, "y_max": 0.5,
                "z_min": 0.0, "z_max": 0.4,
            },
            "gripper_force": 40.0,
            "action_timeout_seconds": 30.0,
        })

    def test_valid_pose(self):
        assert self.monitor.validate_pose(0.3, 0.1, 0.2) is True

    def test_invalid_pose(self):
        assert self.monitor.validate_pose(99.0, 0.0, 0.0) is False

    def test_valid_force(self):
        assert self.monitor.validate_force(30.0) is True

    def test_excessive_force(self):
        assert self.monitor.validate_force(50.0) is False

    def test_estop_flag(self):
        assert self.monitor.estop_requested is False
        self.monitor.request_estop()
        assert self.monitor.estop_requested is True
        self.monitor.clear_estop()
        assert self.monitor.estop_requested is False

    def test_timeout_context(self):
        ctx = self.monitor.get_timeout_context("test_action")
        assert isinstance(ctx, ActionTimeout)
        assert ctx.timeout == 30.0
