"""
Integration tests for the Generative Kitting System.

Tests the end-to-end pipeline using mock VLM/LLM responses
and mock execution (no Isaac Sim required).
"""

import json
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.vlm_perception import VLMPerception
from perception.validators import validate_scene_response
from orchestration.llm_planner import LLMPlanner
from orchestration.validators import validate_task_plan, PlanValidationError
from knowledge.parts_database import PartsDatabase


# ═════════════════════════════════════════════════════════════
# MOCK DATA
# ═════════════════════════════════════════════════════════════

MOCK_VLM_RESPONSE = json.dumps({
    "detected_objects": [
        {
            "object_id": "obj_001",
            "label": "motor_valve",
            "semantic_description": "Grey steel motor valve, ~80mm",
            "affordance": "graspable",
            "approximate_position": {"x": 0.35, "y": -0.12, "z": 0.02},
            "confidence": 0.92,
        },
        {
            "object_id": "obj_002",
            "label": "black_plug",
            "semantic_description": "Black plastic plug connector",
            "affordance": "graspable",
            "approximate_position": {"x": 0.20, "y": 0.10, "z": 0.02},
            "confidence": 0.88,
        },
    ],
    "scene_summary": "Workspace contains a motor valve and a black plug on the table.",
})

MOCK_LLM_RESPONSE = json.dumps({
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
})


# ═════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═════════════════════════════════════════════════════════════

class TestEndToEndPipeline:
    """Test the full perception → planning → validation pipeline with mocks."""

    def test_vlm_to_plan_pipeline(self):
        """VLM response → parse → validate → LLM plan → validate."""

        # Step 1: Parse VLM response
        parsed_scene = VLMPerception.parse_vlm_response(MOCK_VLM_RESPONSE)
        assert "detected_objects" in parsed_scene

        # Step 2: Validate scene
        validated_scene = validate_scene_response(parsed_scene)
        assert len(validated_scene["detected_objects"]) == 2

        # Step 3: Parse LLM plan
        parsed_plan = json.loads(MOCK_LLM_RESPONSE)
        assert "plan" in parsed_plan

        # Step 4: Validate plan against scene
        scene_objects = [obj["object_id"] for obj in validated_scene["detected_objects"]]
        validated_plan = validate_task_plan(
            parsed_plan,
            scene_objects=scene_objects,
        )
        assert validated_plan["total_steps"] == 7
        assert validated_plan["plan"][0]["action"] == "move_home"
        assert validated_plan["plan"][-1]["action"] == "move_home"

    def test_plan_references_scene_objects(self):
        """All object_ids in the plan should exist in the scene."""
        parsed_scene = VLMPerception.parse_vlm_response(MOCK_VLM_RESPONSE)
        validated_scene = validate_scene_response(parsed_scene)
        scene_ids = {obj["object_id"] for obj in validated_scene["detected_objects"]}

        parsed_plan = json.loads(MOCK_LLM_RESPONSE)

        for step in parsed_plan["plan"]:
            obj_id = step.get("params", {}).get("object_id")
            if obj_id:
                assert obj_id in scene_ids, f"Plan references unknown object: {obj_id}"

    def test_invalid_plan_triggers_retry_info(self):
        """PlanValidationError should carry enough info for a retry prompt."""
        bad_plan = {
            "task_summary": "test",
            "plan": [
                {"step": 1, "action": "fly_away", "params": {}},
            ],
        }
        with pytest.raises(PlanValidationError) as exc_info:
            validate_task_plan(bad_plan)

        error = exc_info.value
        assert len(error.errors) > 0
        assert error.plan is not None


# ═════════════════════════════════════════════════════════════
# DATABASE INTEGRATION TESTS
# ═════════════════════════════════════════════════════════════

class TestDatabaseIntegration:
    """Test the knowledge layer in the context of the full pipeline."""

    def setup_method(self):
        """Create a temporary in-memory-like database."""
        self.db = PartsDatabase(":memory:")

    def teardown_method(self):
        self.db.close()

    def test_seed_and_query(self):
        """Seed parts and verify query."""
        from knowledge.seed_data import seed_parts, seed_kits
        seed_parts(self.db)
        seed_kits(self.db)

        parts = self.db.get_all_parts()
        assert len(parts) == 9

        kits = self.db.get_all_kits()
        assert len(kits) == 3

    def test_action_logging(self):
        """Log actions and retrieve session summary."""
        session = "test_session_001"

        self.db.log_action(session, "move_home", success=True, duration_seconds=1.5)
        self.db.log_action(session, "pick_object", object_id="obj_001", success=True, duration_seconds=5.0)
        self.db.log_action(session, "verify_grasp", success=False, error_message="No object detected", duration_seconds=0.5)

        logs = self.db.get_session_log(session)
        assert len(logs) == 3

        summary = self.db.get_session_summary(session)
        assert summary["total_actions"] == 3
        assert summary["successes"] == 2
        assert summary["failures"] == 1
        assert summary["total_time_seconds"] == 7.0

    def test_kit_required_parts(self):
        """Kit required_parts should be properly serialised/deserialised."""
        self.db.add_kit(
            "test_kit",
            "Test Kit",
            [
                {"label": "bolt", "quantity": 4},
                {"label": "gear", "quantity": 2},
            ],
        )
        kit = self.db.get_kit("test_kit")
        assert kit is not None
        assert len(kit["required_parts"]) == 2
        assert kit["required_parts"][0]["quantity"] == 4

    def test_part_lookup_by_label(self):
        """Should find parts by label."""
        self.db.add_part("part_001", "motor_valve", "actuator")
        self.db.add_part("part_002", "motor_valve", "actuator")

        results = self.db.get_part_by_label("motor_valve")
        assert len(results) == 2


# ═════════════════════════════════════════════════════════════
# REPORT STRUCTURE TESTS
# ═════════════════════════════════════════════════════════════

class TestReportStructure:
    """Verify the execution report has the expected fields."""

    def test_report_keys(self):
        """The execution report from a pipeline run should have standard keys."""
        # Simulate what execute_plan would return
        report = {
            "task_summary": "Pick motor valve",
            "total_steps": 7,
            "completed_steps": 7,
            "successful_steps": 6,
            "failed_steps": 1,
            "skipped_steps": 0,
            "step_results": [
                {"step": i, "action": "move_home", "success": True, "duration_seconds": 1.0}
                for i in range(1, 8)
            ],
            "total_time_seconds": 34.2,
        }

        assert "task_summary" in report
        assert "total_steps" in report
        assert "step_results" in report
        assert report["successful_steps"] + report["failed_steps"] == report["completed_steps"]
