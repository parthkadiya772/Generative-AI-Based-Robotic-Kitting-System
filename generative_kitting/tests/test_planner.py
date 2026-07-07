"""
Unit tests for the Orchestration Layer (task plan validation).
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.validators import (
    validate_task_plan,
    PlanValidationError,
    ALLOWED_ACTIONS,
)


# ═════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════

VALID_PLAN = {
    "task_summary": "Pick motor valve and place in kit tray",
    "total_steps": 7,
    "plan": [
        {"step": 1, "action": "move_home", "params": {}},
        {"step": 2, "action": "open_gripper", "params": {}},
        {"step": 3, "action": "pick_object", "params": {"object_id": "obj_001", "x": 0.35, "y": -0.12, "z": 0.02}},
        {"step": 4, "action": "verify_grasp", "params": {}},
        {"step": 5, "action": "place_object", "params": {"object_id": "obj_001", "x": 0.5, "y": 0.0, "z": 0.02}},
        {"step": 6, "action": "open_gripper", "params": {}},
        {"step": 7, "action": "move_home", "params": {}},
    ],
}

SCENE_OBJECTS = ["obj_001", "obj_002", "obj_003"]

WORKSPACE_BOUNDS = {
    "x_min": -0.2, "x_max": 0.8,
    "y_min": -0.5, "y_max": 0.5,
    "z_min": 0.0, "z_max": 0.4,
}


# ═════════════════════════════════════════════════════════════
# VALID PLAN TESTS
# ═════════════════════════════════════════════════════════════

class TestValidPlan:
    """Tests that valid plans pass validation."""

    def test_valid_plan_passes(self):
        """A correctly structured plan should pass."""
        result = validate_task_plan(
            VALID_PLAN,
            scene_objects=SCENE_OBJECTS,
            workspace_bounds=WORKSPACE_BOUNDS,
        )
        assert result["total_steps"] == 7
        assert result["task_summary"] == "Pick motor valve and place in kit tray"

    def test_total_steps_corrected(self):
        """total_steps should be auto-corrected to match plan length."""
        plan = {**VALID_PLAN, "total_steps": 999}
        result = validate_task_plan(plan, scene_objects=SCENE_OBJECTS)
        assert result["total_steps"] == 7


# ═════════════════════════════════════════════════════════════
# INVALID PLAN TESTS
# ═════════════════════════════════════════════════════════════

class TestInvalidPlan:
    """Tests that invalid plans trigger PlanValidationError."""

    def test_invalid_action_rejected(self):
        """Actions not in the whitelist should be rejected."""
        plan = {
            "task_summary": "test",
            "plan": [
                {"step": 1, "action": "move_home", "params": {}},
                {"step": 2, "action": "launch_missile", "params": {"target": "moon"}},
                {"step": 3, "action": "move_home", "params": {}},
            ],
        }
        with pytest.raises(PlanValidationError) as exc_info:
            validate_task_plan(plan)
        assert "launch_missile" in str(exc_info.value)

    def test_no_start_move_home(self):
        """Plan not starting with move_home should fail."""
        plan = {
            "task_summary": "test",
            "plan": [
                {"step": 1, "action": "open_gripper", "params": {}},
                {"step": 2, "action": "move_home", "params": {}},
            ],
        }
        with pytest.raises(PlanValidationError) as exc_info:
            validate_task_plan(plan)
        assert "START" in str(exc_info.value)

    def test_no_end_move_home(self):
        """Plan not ending with move_home should fail."""
        plan = {
            "task_summary": "test",
            "plan": [
                {"step": 1, "action": "move_home", "params": {}},
                {"step": 2, "action": "open_gripper", "params": {}},
            ],
        }
        with pytest.raises(PlanValidationError) as exc_info:
            validate_task_plan(plan)
        assert "END" in str(exc_info.value)

    def test_pick_without_open_gripper(self):
        """pick_object without preceding open_gripper should fail."""
        plan = {
            "task_summary": "test",
            "plan": [
                {"step": 1, "action": "move_home", "params": {}},
                {"step": 2, "action": "pick_object", "params": {"object_id": "obj_001", "x": 0.3, "y": 0.1, "z": 0.02}},
                {"step": 3, "action": "verify_grasp", "params": {}},
                {"step": 4, "action": "move_home", "params": {}},
            ],
        }
        with pytest.raises(PlanValidationError) as exc_info:
            validate_task_plan(plan, scene_objects=SCENE_OBJECTS)
        assert "preceded by open_gripper" in str(exc_info.value)

    def test_pick_without_verify_grasp(self):
        """pick_object without following verify_grasp should fail."""
        plan = {
            "task_summary": "test",
            "plan": [
                {"step": 1, "action": "move_home", "params": {}},
                {"step": 2, "action": "open_gripper", "params": {}},
                {"step": 3, "action": "pick_object", "params": {"object_id": "obj_001", "x": 0.3, "y": 0.1, "z": 0.02}},
                {"step": 4, "action": "move_home", "params": {}},
            ],
        }
        with pytest.raises(PlanValidationError) as exc_info:
            validate_task_plan(plan, scene_objects=SCENE_OBJECTS)
        assert "followed by verify_grasp" in str(exc_info.value)

    def test_coordinate_out_of_bounds(self):
        """Coordinates outside workspace bounds should fail."""
        plan = {
            "task_summary": "test",
            "plan": [
                {"step": 1, "action": "move_home", "params": {}},
                {"step": 2, "action": "move_to_pose", "params": {"x": 99.0, "y": 0.0, "z": 0.0}},
                {"step": 3, "action": "move_home", "params": {}},
            ],
        }
        with pytest.raises(PlanValidationError) as exc_info:
            validate_task_plan(plan, workspace_bounds=WORKSPACE_BOUNDS)
        assert "outside workspace" in str(exc_info.value)

    def test_unknown_object_id(self):
        """Object IDs not in the scene should fail."""
        plan = {
            "task_summary": "test",
            "plan": [
                {"step": 1, "action": "move_home", "params": {}},
                {"step": 2, "action": "open_gripper", "params": {}},
                {"step": 3, "action": "pick_object", "params": {"object_id": "obj_999", "x": 0.3, "y": 0.1, "z": 0.02}},
                {"step": 4, "action": "verify_grasp", "params": {}},
                {"step": 5, "action": "move_home", "params": {}},
            ],
        }
        with pytest.raises(PlanValidationError) as exc_info:
            validate_task_plan(plan, scene_objects=SCENE_OBJECTS)
        assert "obj_999" in str(exc_info.value)

    def test_too_many_steps(self):
        """Plans exceeding max_steps should fail."""
        steps = [{"step": 1, "action": "move_home", "params": {}}]
        steps += [{"step": i, "action": "open_gripper", "params": {}} for i in range(2, 62)]
        steps.append({"step": 62, "action": "move_home", "params": {}})

        plan = {"task_summary": "test", "plan": steps}
        with pytest.raises(PlanValidationError) as exc_info:
            validate_task_plan(plan, max_steps=50)
        assert "exceeds maximum" in str(exc_info.value)

    def test_joint_angle_rejection(self):
        """Plans containing raw joint angle data should be rejected."""
        plan = {
            "task_summary": "test",
            "plan": [
                {"step": 1, "action": "move_home", "params": {}},
                {"step": 2, "action": "move_to_pose", "params": {
                    "x": 0.3, "y": 0.1, "z": 0.2,
                    "joint_angles": [0.1, -1.5, 1.2, -0.8, 0.5, 0.0],
                }},
                {"step": 3, "action": "move_home", "params": {}},
            ],
        }
        with pytest.raises(PlanValidationError) as exc_info:
            validate_task_plan(plan)
        assert "joint" in str(exc_info.value).lower()

    def test_empty_plan_fails(self):
        """Plan with zero steps should fail."""
        plan = {"task_summary": "test", "plan": []}
        with pytest.raises(PlanValidationError) as exc_info:
            validate_task_plan(plan)
        assert "zero steps" in str(exc_info.value)


# ═════════════════════════════════════════════════════════════
# MISC TESTS
# ═════════════════════════════════════════════════════════════

class TestPlanValidationError:
    """Tests for the PlanValidationError exception class."""

    def test_error_has_errors_list(self):
        """PlanValidationError should carry the list of errors."""
        err = PlanValidationError(["error1", "error2"])
        assert len(err.errors) == 2
        assert "error1" in err.errors

    def test_error_has_plan(self):
        """PlanValidationError should optionally carry the plan."""
        plan = {"plan": []}
        err = PlanValidationError(["test"], plan=plan)
        assert err.plan is plan

    def test_error_message_truncated(self):
        """Error message should truncate if there are many errors."""
        errors = [f"error_{i}" for i in range(10)]
        err = PlanValidationError(errors)
        assert "and 7 more" in str(err)

    def test_allowed_actions_complete(self):
        """Verify the allowed actions set contains all 8 primitives."""
        expected = {
            "move_home", "move_to_pose", "open_gripper", "close_gripper",
            "pick_object", "place_object", "verify_grasp", "request_perception_update",
        }
        assert ALLOWED_ACTIONS == expected
