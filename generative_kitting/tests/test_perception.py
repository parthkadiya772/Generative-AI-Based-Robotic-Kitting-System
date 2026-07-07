"""
Unit tests for the Perception Layer (VLM parsing and validation).
"""

import json
import os
import sys
import pytest
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.vlm_perception import VLMPerception
from perception.validators import (
    validate_scene_response,
    validate_single_object,
    count_by_label,
)


# ═════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════

VALID_SCENE_RESPONSE = {
    "detected_objects": [
        {
            "object_id": "obj_001",
            "label": "motor_valve",
            "semantic_description": "Grey steel motor valve, approximately 80mm diameter",
            "affordance": "graspable",
            "approximate_position": {"x": 0.35, "y": -0.12, "z": 0.02},
            "confidence": 0.92,
        },
        {
            "object_id": "obj_002",
            "label": "black_hose",
            "semantic_description": "Flexible black rubber hose, ~150mm",
            "affordance": "graspable",
            "approximate_position": {"x": 0.20, "y": 0.10, "z": 0.02},
            "confidence": 0.85,
        },
        {
            "object_id": "obj_003",
            "label": "small_hinge",
            "semantic_description": "Small silver hinge",
            "affordance": "graspable",
            "approximate_position": {"x": 0.45, "y": 0.05, "z": 0.01},
            "confidence": 0.78,
        },
    ],
    "scene_summary": "Workspace contains a motor valve, a black hose, and a small hinge.",
}


VALID_RAW_JSON = json.dumps(VALID_SCENE_RESPONSE)

MARKDOWN_WRAPPED_JSON = f"```json\n{VALID_RAW_JSON}\n```"

MALFORMED_JSON = '{"detected_objects": [{"object_id": "obj_001", INVALID'

PARTIAL_RESPONSE = f"Here is the analysis:\n\n{VALID_RAW_JSON}\n\nLet me know if you need more details."


# ═════════════════════════════════════════════════════════════
# VLM RESPONSE PARSING TESTS
# ═════════════════════════════════════════════════════════════

class TestVLMParsing:
    """Tests for VLMPerception.parse_vlm_response()."""

    def test_valid_json(self):
        """Parse clean JSON string."""
        result = VLMPerception.parse_vlm_response(VALID_RAW_JSON)
        assert "detected_objects" in result
        assert len(result["detected_objects"]) == 3

    def test_markdown_fenced_json(self):
        """Parse JSON wrapped in markdown code fences."""
        result = VLMPerception.parse_vlm_response(MARKDOWN_WRAPPED_JSON)
        assert "detected_objects" in result
        assert len(result["detected_objects"]) == 3

    def test_json_with_surrounding_text(self):
        """Extract JSON from a response with surrounding text."""
        result = VLMPerception.parse_vlm_response(PARTIAL_RESPONSE)
        assert "detected_objects" in result

    def test_malformed_json_raises(self):
        """Malformed JSON should raise ValueError."""
        with pytest.raises(ValueError, match="Failed to parse"):
            VLMPerception.parse_vlm_response(MALFORMED_JSON)

    def test_empty_string_raises(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError):
            VLMPerception.parse_vlm_response("")

    def test_whitespace_trimming(self):
        """Handles leading/trailing whitespace."""
        result = VLMPerception.parse_vlm_response(f"  \n  {VALID_RAW_JSON}  \n  ")
        assert "detected_objects" in result

    def test_qwen_image_prep_resizes_to_28_multiple(self):
        """Qwen prep snaps both dims to multiples of 28 within Qwen's
        pixel budget so smart_resize is a no-op and bbox coords land in
        the (W, H) we sent. Long edge cap is 1260 (45 * 28)."""
        vlm = VLMPerception({"vlm_provider": "ollama_qwen"})
        image = Image.new("RGB", (1920, 1080), color="white")

        prepared = vlm._prepare_image_for_qwen(image)

        w, h = prepared.size
        assert w % 28 == 0 and h % 28 == 0
        assert max(w, h) <= 1260
        # Pixel budget should stay below ~1 MP for typical 16:9.
        assert w * h <= 1_000_000
        assert vlm._last_qwen_image_size == (w, h)


# ═════════════════════════════════════════════════════════════
# VLM VALIDATION TESTS
# ═════════════════════════════════════════════════════════════

class TestVLMValidation:
    """Tests for perception validators."""

    def test_valid_scene_passes(self):
        """Full valid scene response should pass validation."""
        result = validate_scene_response(VALID_SCENE_RESPONSE)
        assert len(result["detected_objects"]) == 3
        assert "scene_summary" in result

    def test_low_confidence_rejected(self):
        """Objects below confidence threshold are filtered out."""
        response = {
            "detected_objects": [
                {
                    "object_id": "obj_001",
                    "label": "ghost_part",
                    "semantic_description": "A ghost",
                    "affordance": "graspable",
                    "approximate_position": {"x": 0.1, "y": 0.1, "z": 0.0},
                    "confidence": 0.3,
                },
            ],
            "scene_summary": "test",
        }
        result = validate_scene_response(response, confidence_threshold=0.5)
        assert len(result["detected_objects"]) == 0

    def test_out_of_bounds_rejected(self):
        """Objects with coordinates outside workspace bounds are rejected."""
        response = {
            "detected_objects": [
                {
                    "object_id": "obj_001",
                    "label": "far_part",
                    "semantic_description": "Too far away",
                    "affordance": "graspable",
                    "approximate_position": {"x": 999.0, "y": 0.0, "z": 0.0},
                    "confidence": 0.95,
                },
            ],
            "scene_summary": "test",
        }
        bounds = {"x_min": -1, "x_max": 2, "y_min": -1, "y_max": 1, "z_min": -0.1, "z_max": 1}
        result = validate_scene_response(response, workspace_bounds=bounds)
        assert len(result["detected_objects"]) == 0

    def test_missing_field_rejected(self):
        """Objects missing required fields are rejected."""
        response = {
            "detected_objects": [
                {
                    "object_id": "obj_001",
                    "label": "incomplete",
                    # Missing: semantic_description, affordance, location, confidence
                },
            ],
            "scene_summary": "test",
        }
        result = validate_scene_response(response)
        assert len(result["detected_objects"]) == 0

    def test_bbox_pixels_satisfies_location(self):
        """Bin-zoom Qwen schema (bbox_pixels, no approximate_position)
        must pass validation — downstream depth projection produces
        world coords from the pixel bbox."""
        response = {
            "detected_objects": [
                {
                    "object_id": "obj_1",
                    "label": "large_gear",
                    "semantic_description": "black flat disc",
                    "affordance": "graspable",
                    "bbox_pixels": [100, 500, 200, 600],
                    "confidence": 0.95,
                },
            ],
            "scene_summary": "test",
        }
        result = validate_scene_response(response)
        assert len(result["detected_objects"]) == 1
        assert result["detected_objects"][0]["bbox_pixels"] == [100, 500, 200, 600]

    def test_no_location_field_rejected(self):
        """An object with every other field but no location is rejected."""
        response = {
            "detected_objects": [
                {
                    "object_id": "obj_1",
                    "label": "ghost",
                    "semantic_description": "where is it?",
                    "affordance": "graspable",
                    "confidence": 0.95,
                },
            ],
            "scene_summary": "test",
        }
        result = validate_scene_response(response)
        assert len(result["detected_objects"]) == 0

    def test_missing_detected_objects_raises(self):
        """Missing 'detected_objects' key should raise ValueError."""
        with pytest.raises(ValueError, match="missing 'detected_objects'"):
            validate_scene_response({"scene_summary": "nothing here"})

    def test_non_dict_raises(self):
        """Non-dict input should raise ValueError."""
        with pytest.raises(ValueError, match="must be a dict"):
            validate_scene_response("not a dict")

    def test_invalid_affordance_normalised(self):
        """Unknown affordance should be normalised to 'graspable'."""
        response = {
            "detected_objects": [
                {
                    "object_id": "obj_001",
                    "label": "test",
                    "semantic_description": "Test part",
                    "affordance": "teleportable",
                    "approximate_position": {"x": 0.1, "y": 0.1, "z": 0.0},
                    "confidence": 0.9,
                },
            ],
            "scene_summary": "test",
        }
        result = validate_scene_response(response)
        assert result["detected_objects"][0]["affordance"] == "graspable"

    def test_count_by_label(self):
        """Count helper should group objects by label."""
        objects = VALID_SCENE_RESPONSE["detected_objects"]
        counts = count_by_label(objects)
        assert counts["motor_valve"] == 1
        assert counts["black_hose"] == 1
        assert counts["small_hinge"] == 1

    def test_confidence_clamped_to_one(self):
        """Confidence > 1.0 should be clamped."""
        response = {
            "detected_objects": [
                {
                    "object_id": "obj_001",
                    "label": "test",
                    "semantic_description": "Test",
                    "affordance": "graspable",
                    "approximate_position": {"x": 0.1, "y": 0.1, "z": 0.0},
                    "confidence": 1.5,
                },
            ],
            "scene_summary": "test",
        }
        result = validate_scene_response(response)
        assert result["detected_objects"][0]["confidence"] == 1.0
