"""
VLM Output Validators — Hallucination Guard (Layer 1).

Validates and sanitises the structured JSON returned by the VLM
to prevent downstream errors from:
  - Low-confidence detections
  - Out-of-bounds spatial coordinates
  - Missing or malformed fields
  - Hallucinated objects
"""

from typing import Any, Dict, List, Optional

from utils.logger import log


# ─── Required fields for each detected object ───────────────
REQUIRED_OBJECT_FIELDS = {
    "object_id": str,
    "label": str,
    "semantic_description": str,
    "affordance": str,
    "approximate_position": dict,
    "confidence": (int, float),
}

VALID_AFFORDANCES = {"graspable", "stackable", "fragile", "fixed", "destination"}

# Default workspace bounds for position validation (UR10 + gantry reach)
DEFAULT_WORKSPACE_BOUNDS = {
    "x_min": -2.5, "x_max": 3.5,
    "y_min": -2.0, "y_max": 2.0,
    "z_min": -0.1, "z_max": 2.0,
}


def validate_scene_response(
    response: Dict[str, Any],
    confidence_threshold: float = 0.5,
    workspace_bounds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Validate and sanitise a complete VLM scene response.

    Parameters
    ----------
    response : dict
        Parsed VLM JSON response.
    confidence_threshold : float
        Minimum confidence to keep a detected object.
    workspace_bounds : dict, optional
        Workspace bounds for spatial validation.

    Returns
    -------
    dict
        Validated response with filtered ``detected_objects``
        and guaranteed ``scene_summary``.

    Raises
    ------
    ValueError
        If the response structure is fundamentally invalid.
    """
    if not isinstance(response, dict):
        raise ValueError(f"VLM response must be a dict, got {type(response).__name__}")

    # Ensure required top-level keys
    if "detected_objects" not in response:
        raise ValueError("VLM response missing 'detected_objects' key")

    objects = response.get("detected_objects", [])
    if not isinstance(objects, list):
        raise ValueError(
            f"'detected_objects' must be a list, got {type(objects).__name__}"
        )

    scene_summary = response.get("scene_summary", "No scene summary provided.")
    if not isinstance(scene_summary, str):
        scene_summary = str(scene_summary)

    bounds = workspace_bounds or DEFAULT_WORKSPACE_BOUNDS

    # Validate each object
    validated_objects = []
    for i, obj in enumerate(objects):
        try:
            validated = validate_single_object(
                obj, index=i,
                confidence_threshold=confidence_threshold,
                workspace_bounds=bounds,
            )
            if validated is not None:
                validated_objects.append(validated)
        except Exception as e:
            log.warning(f"Rejected object at index {i}: {e}")

    log.info(
        f"VLM validation: {len(validated_objects)}/{len(objects)} objects accepted "
        f"(threshold={confidence_threshold})"
    )

    return {
        "detected_objects": validated_objects,
        "scene_summary": scene_summary,
    }


def validate_single_object(
    obj: Dict[str, Any],
    index: int = 0,
    confidence_threshold: float = 0.5,
    workspace_bounds: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Validate a single detected object from the VLM response.

    Returns
    -------
    dict or None
        The validated (possibly normalised) object dict,
        or None if the object should be rejected.
    """
    if not isinstance(obj, dict):
        log.warning(f"Object {index}: not a dict — rejected")
        return None

    # ── Check required fields ────────────────────────────────
    for field, expected_type in REQUIRED_OBJECT_FIELDS.items():
        if field not in obj:
            log.warning(f"Object {index}: missing field '{field}' — rejected")
            return None
        if not isinstance(obj[field], expected_type):
            # Try type coercion for numeric fields
            if field == "confidence":
                try:
                    obj[field] = float(obj[field])
                except (ValueError, TypeError):
                    log.warning(
                        f"Object {index}: field '{field}' has wrong type "
                        f"({type(obj[field]).__name__}) — rejected"
                    )
                    return None
            else:
                log.warning(
                    f"Object {index}: field '{field}' expected {expected_type}, "
                    f"got {type(obj[field]).__name__} — rejected"
                )
                return None

    # ── Confidence filter ────────────────────────────────────
    confidence = float(obj["confidence"])
    if confidence < confidence_threshold:
        log.debug(
            f"Object {index} ('{obj['label']}'): confidence {confidence:.2f} "
            f"< threshold {confidence_threshold} — rejected"
        )
        return None

    # Clamp confidence to [0, 1]
    obj["confidence"] = max(0.0, min(1.0, confidence))

    # ── Affordance validation ────────────────────────────────
    affordance = obj.get("affordance", "graspable").lower()
    if affordance not in VALID_AFFORDANCES:
        log.debug(
            f"Object {index}: unknown affordance '{affordance}' — defaulting to 'graspable'"
        )
        obj["affordance"] = "graspable"

    # ── Spatial position validation ──────────────────────────
    pos = obj.get("approximate_position", {})
    if not isinstance(pos, dict):
        log.warning(f"Object {index}: approximate_position is not a dict — rejected")
        return None

    bounds = workspace_bounds or DEFAULT_WORKSPACE_BOUNDS

    for axis in ("x", "y", "z"):
        if axis not in pos:
            log.warning(
                f"Object {index}: approximate_position missing '{axis}' — rejected"
            )
            return None
        try:
            val = float(pos[axis])
            pos[axis] = val
        except (ValueError, TypeError):
            log.warning(
                f"Object {index}: coordinate '{axis}' = {pos[axis]} is not numeric — rejected"
            )
            return None

        lo = bounds.get(f"{axis}_min", -float("inf"))
        hi = bounds.get(f"{axis}_max", float("inf"))
        if val < lo or val > hi:
            log.warning(
                f"Object {index} ('{obj['label']}'): {axis}={val:.3f} "
                f"outside bounds [{lo}, {hi}] — rejected"
            )
            return None

    obj["approximate_position"] = pos

    # ── Ensure object_id is unique-looking ───────────────────
    if not obj["object_id"]:
        obj["object_id"] = f"obj_{index + 1:03d}"

    return obj


def count_by_label(objects: List[Dict]) -> Dict[str, int]:
    """
    Count detected objects grouped by label.

    Useful for comparing VLM output against kit requirements.
    """
    counts = {}
    for obj in objects:
        label = obj.get("label", "unknown")
        counts[label] = counts.get(label, 0) + 1
    return counts
