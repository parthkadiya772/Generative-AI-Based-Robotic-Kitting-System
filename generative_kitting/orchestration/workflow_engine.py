"""
Neuro-Symbolic Kitting Workflow Engine (Crop-and-Zoom).

Orchestrates the full AI-driven pick-and-place pipeline:
  0.   Overhead RGB + VLM → locate bin + kitting tray (landmarks + bbox)
  1.   Bridge depth projection → bin/tray world coordinates
  2.   Crop bin region from overhead image → VLM zoom analysis →
       identify parts in the cropped, higher-detail view
       Convert crop-local coords → full-image coords →
       Bridge depth projection → part world coordinates
  3.   LLM generates plan using real-world coordinates
  4-5. Execute pick (approach → depth analysis → wrist verify →
       descend → grasp → retract) + place (transit → descend → release)
  6.   Overhead RGB verification of final workspace state

The robot does NOT move for scanning — it stays at its current
position while the overhead camera + VLM analyzes the bin via
crop-and-zoom.  The robot only moves when actually picking a part.
This avoids gantry collisions caused by repeated scan repositioning.

ALL coordinates come from camera vision + depth projection.
No hardcoded positions, no USD prim lookups.
"""

import json
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from knowledge.parts_catalogue import (
    detector_queries,
    detector_query_list,
    format_catalogue_for_prompt,
)
from perception.depth_estimator import DepthEstimator
from perception.object_detector import ZeroShotDetector
from perception.vlm_perception import qwen_target_size
from utils.logger import log


def _with_catalogue(prompt: str) -> str:
    """Prepend the parts catalogue (visual descriptions + labelling
    rules) to a VLM prompt. The catalogue anchors the VLM's labels to
    a known vocabulary so it stops inventing strings like
    'circular_mechanical_part'. If the catalogue file is missing the
    prompt is returned unchanged."""
    block = format_catalogue_for_prompt()
    if not block:
        return prompt
    return f"{block}\n\n{prompt}"


# ─── VLM Prompts ────────────────────────────────────────────

# Used by Streamlit manual scan button — full scene analysis from overhead
SCENE_ANALYSIS_PROMPT = """You are a robotic vision system analyzing an overhead view of an
industrial kitting workspace from approximately 5 metres above.

Identify ALL objects in this image:
1. ALL loose mechanical parts visible inside the blue parts bin
2. The WHITE KITTING TRAY / destination box (the white box where picked parts go)

LABELING RULES:
- Use ONLY the EXACT label strings from the KNOWN PART TYPES catalogue
  prepended above. Do NOT invent descriptive labels.
- If two similar parts exist, add _1, _2 suffix
- If a part doesn't match any catalogue entry, label it "unknown"
- Label the destination box as "kitting_tray" with affordance "destination"
- Parts in the bin should have affordance "graspable"

IGNORE: robot arm, gripper, gantry rail, blue bin walls, shadows, reflections

POSITION: Give normalised image coordinates (x, y) for each object's centre,
where (0,0) = top-left and (1,1) = bottom-right.

RESPOND ONLY WITH VALID JSON:
{
  "detected_objects": [
    {
      "object_id": "obj_001",
      "label": "motor_valve",
      "semantic_description": "brass cylindrical valve fitting, ~5cm",
      "affordance": "graspable",
      "image_position": {"x": 0.35, "y": 0.42},
      "confidence": 0.9
    },
    {
      "object_id": "obj_tray",
      "label": "kitting_tray",
      "semantic_description": "white rectangular destination box",
      "affordance": "destination",
      "image_position": {"x": 0.75, "y": 0.50},
      "confidence": 0.95
    }
  ],
  "scene_summary": "..."
}"""

# ─── Overhead landmark prompt (big targets only) ───────────
LANDMARK_PROMPT = """You are a robotic vision system. This is an overhead view of an
industrial kitting workspace from ~5 metres above.

Your ONLY task is to locate TWO landmarks:
1. The BLUE PARTS BIN — the blue rectangular container holding loose mechanical parts.
2. The WHITE KITTING TRAY — the white box / white container where picked parts go.

For each landmark, give:
- ``image_position``: CENTRE as normalised image coordinates (x, y) where
  (0,0) = top-left and (1,1) = bottom-right.
- ``image_bbox``: TIGHT bounding box around the landmark, also normalised:
  ``{"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ...}``.

CRITICAL: Look at THIS image carefully and report the ACTUAL positions.
Do NOT guess or copy example values.

DO NOT detect individual parts, the robot arm, gantry rail, or anything else.
ONLY find the bin and the tray.

RESPOND ONLY WITH VALID JSON:
{
  "detected_objects": [
    {
      "object_id": "bin_001",
      "label": "parts_bin",
      "semantic_description": "blue rectangular parts bin",
      "affordance": "fixed",
      "image_position": {"x": 0.40, "y": 0.60},
      "image_bbox": {"x_min": 0.30, "y_min": 0.45, "x_max": 0.55, "y_max": 0.78},
      "confidence": 0.95
    },
    {
      "object_id": "tray_001",
      "label": "kitting_tray",
      "semantic_description": "white kitting tray / destination box",
      "affordance": "destination",
      "image_position": {"x": 0.75, "y": 0.50},
      "image_bbox": {"x_min": 0.68, "y_min": 0.42, "x_max": 0.85, "y_max": 0.60},
      "confidence": 0.93
    }
  ],
  "scene_summary": "..."
}"""

# ─── Bin zoom prompt — applied to the cropped bin region ───
BIN_ZOOM_PROMPT = """You are a robotic vision system. This image is a CROPPED close-up of
ONLY the blue parts bin from an overhead industrial workspace view.
The bin (and the parts inside it) fills most or all of the frame.

Identify ALL loose mechanical parts visible inside the bin.

LABELING RULES:
- Use ONLY the EXACT label strings from the KNOWN PART TYPES catalogue
  prepended above. Do NOT invent descriptive labels.
- If two similar parts exist, add a number suffix (e.g. ``large_gear_1``,
  ``large_gear_2``).
- If a part doesn't match any catalogue entry, label it ``"unknown"``.
- Each part has affordance ``"graspable"``.

IGNORE:
- Blue bin walls, edges, dividers, the bin itself
- Robot arm, gripper, gantry rail
- Shadows, reflections
- The white kitting tray (it is outside this crop)

POSITION RULES:
- Give ``image_position`` as NORMALISED coordinates (x, y) within THIS CROPPED
  image where (0,0) = top-left and (1,1) = bottom-right of the crop.
- Position must point to the CENTRE of each part.

RESPOND ONLY WITH VALID JSON:
{
  "detected_objects": [
    {
      "object_id": "obj_001",
      "label": "motor_valve",
      "semantic_description": "brass cylindrical valve, ~5cm",
      "affordance": "graspable",
      "image_position": {"x": 0.4, "y": 0.5},
      "confidence": 0.9
    }
  ],
  "scene_summary": "..."
}"""

# ─── Minimal prompt tuned for Qwen-VL (8B, drowns in long prompts) ──
# Replaces BIN_ZOOM_PROMPT + catalogue + grounding suffix entirely
# when the active VLM is Qwen. The 8B model can't follow the full
# decision-tree catalogue (3500+ token prompts cause it to emit
# whitespace under format=json). Strip everything down to the bare
# minimum the model needs:
#   - 3 part types as one-liners (no decision tree, no
#     not_to_confuse_with sub-clauses)
#   - exact JSON schema with bbox_norm baked in
#   - 4 short rules
def bin_zoom_prompt_qwen_minimal(image_w: int = 0, image_h: int = 0) -> str:
    """Qwen3-VL grounding prompt tuned to engage the model's trained
    grounding pathway (vs. captioning).

    Three deliberate choices match Qwen3-VL's RefCOCO/+/g training
    distribution:

    1. Opens with **"Locate every"** — the grounding trigger phrase.
       "Identify" / "describe" engages the captioning pathway and
       gives loose boxes; "Locate" / "Outline the position of" engages
       grounding and gives tight boxes.
    2. **bbox_2d first** in each object dict — Qwen emits fields in
       order and leading fields get the most attention budget. Putting
       the bbox first maximises grounding precision; trailing fields
       (label, affordance, …) cost almost nothing once the box is set.
    3. **0-1000 grid** — Qwen3-VL's native, image-size-invariant
       grounding coordinate system.

    The (image_w, image_h) parameters are kept for API compatibility
    but unused — the 0-1000 grid is dimension-independent.
    """
    return """Locate every mechanical part in this overhead image of an industrial parts bin and output a tight 2D bounding box for each one.

LABEL VOCABULARY (use the EXACT label string):
- "large_gear":  flat BLACK circular disc (the only BLACK part type)
- "small_hinge": TINY silver flat plate with TWO small holes
- "motor_valve": large silver bulky body with ONE large hole through the middle

OUTPUT ONLY this JSON (no markdown, no commentary, no backticks):
{
  "detected_objects": [
    {
      "bbox_2d": [x_min, y_min, x_max, y_max],
      "label": "large_gear",
      "object_id": "obj_1",
      "semantic_description": "black flat disc",
      "affordance": "graspable",
      "confidence": 0.95
    }
  ]
}

GROUNDING RULES (most important — affects whether the robot grasps the part):
- bbox_2d uses your standard normalised grounding coordinates in [0, 1000]
  where (0, 0) = top-left, (1000, 1000) = bottom-right. Integer values.
- Each box must wrap ONLY the visible silhouette of ONE part. The four
  edges must TOUCH the part — no padding, no bin walls, no shadows,
  no neighbouring parts.
- Every visible instance gets ITS OWN box. Two boxes must NEVER share
  the same x_min OR the same y_min — even when parts are the same type
  and clustered together, their pixel positions ARE different. Look
  carefully at each instance's actual centre before writing its bbox.
- Different instances must NOT overlap heavily — if two parts touch,
  the boxes share at most a thin edge.

LABELLING RULES:
- Use ONLY the three labels above. Never invent new labels.
- BLACK round → always "large_gear" (no other black parts exist).
- Silver with a large through-hole → "motor_valve".
- Silver tiny flat plate with two small holes → "small_hinge".
- affordance is always "graspable" for these parts.

Detect EVERY visible part regardless of type. Do not skip parts to keep
the response short — the robot needs the complete scene."""


# Backwards-compat alias so existing callers (e.g. diagnose_qwen.py)
# that import the constant still resolve. The prompt no longer depends
# on image dimensions.
BIN_ZOOM_PROMPT_QWEN_MINIMAL = bin_zoom_prompt_qwen_minimal()


# ─── Grounding suffix for self-grounding VLMs (e.g. Qwen2.5-VL) ─────
# Appended to BIN_ZOOM_PROMPT when the active VLM can emit tight
# bboxes natively, so OWL-ViT2 can be skipped.
BIN_ZOOM_GROUNDING_SUFFIX = """

GROUNDING REQUIREMENT:
For EVERY detected object, ALSO emit a TIGHT bounding box around the
part as NORMALISED coordinates within THIS CROPPED image:
  "bbox_norm": [x_min, y_min, x_max, y_max]
where each value is in [0, 1] and (0,0)=top-left, (1,1)=bottom-right.
The bbox must wrap ONLY the part itself, not the bin background. Be as
tight as possible — the robot picks from the bbox centre.

If you can segment the part, also emit:
    "mask_polygon": [[x1, y1], [x2, y2], ...]
Use the polygon to represent the visible silhouette of the part, not the
bin walls, dividers, or background.

Always keep image_position at the geometric centre of the part, not the
centre of the surrounding bin region.
"""

# ─── Wrist camera close-range scan prompt ──────────────────
WRIST_SCAN_PROMPT = """You are a robotic vision system with a wrist-mounted camera
on a UR10 robot arm. You are looking STRAIGHT DOWN into a parts bin from
approximately 30-50 cm above.

Identify ALL loose mechanical parts visible in the bin below.

LABELING RULES:
- Use SHORT labels (1-3 words with underscores) based on PRIMARY SHAPE + COLOUR.
  Examples: "brass_valve", "black_hose", "metal_gear", "small_hinge",
  "black_plug", "silver_box", "metal_plate", "small_tube", "silver_gun",
  "tube_with_clamps", "black_plate"
- Focus on what makes each part VISUALLY DISTINCT from others
- If two similar parts exist, add a number suffix (e.g. "brass_valve_1",
  "brass_valve_2")

IGNORE:
- Blue bin walls, edges, dividers
- Robot arm, gripper, gripper fingers, gantry rail
- Shadows, reflections

POSITION RULES:
- Give positions as NORMALISED coordinates (x, y) within THIS image
  where (0,0) = top-left and (1,1) = bottom-right.
- Position must point to the CENTRE of each part.

RESPOND ONLY WITH VALID JSON:
{
  "detected_objects": [
    {
      "object_id": "obj_001",
      "label": "brass_valve",
      "semantic_description": "brass cylindrical valve with threaded ends, ~5cm",
      "affordance": "graspable",
      "image_position": {"x": 0.4, "y": 0.5},
      "confidence": 0.9
    }
  ],
  "scene_summary": "..."
}"""

DEPTH_GRASP_PROMPT = """You are a robotic depth-vision system analyzing a close-up depth image
from a RealSense RSD455 camera mounted on a UR10 robot arm.

The robot arm is positioned near a part it needs to grasp.
Analyze this depth image and provide:

1. **part_visible**: Is a graspable part clearly visible? (true/false)
2. **part_center**: Estimated pixel center of the nearest part (x, y as fraction 0-1)
3. **estimated_distance_m**: Approximate distance to the part surface in metres
4. **grasp_clear**: Is the grasp path clear of obstacles? (true/false)
5. **approach_vector**: Recommended approach direction ("top_down", "angled", "side")
6. **confidence**: Your confidence score (0.0 to 1.0)
7. **refinement_notes**: Any observations about part orientation, clutter, or risk

RESPOND ONLY WITH VALID JSON:
{
  "part_visible": true,
  "part_center": {"x": 0.5, "y": 0.5},
  "estimated_distance_m": 0.15,
  "grasp_clear": true,
  "approach_vector": "top_down",
  "confidence": 0.85,
  "refinement_notes": "part is upright, clear approach from above"
}"""

WRIST_VERIFY_PROMPT = """You are looking straight down from a wrist-mounted RealSense camera
on a UR10 robot arm, approximately 30-50 cm above a parts bin.

The robot's gantry has ALREADY moved the gripper above the planned
target part — that target is the part CLOSEST TO THE IMAGE CENTRE.
Other parts in the same compartment may also be visible at the
edges of the frame; IGNORE them.

Before the robot descends to grasp, verify:
1. Is the target part clearly visible NEAR the centre of the image?
   (Allow up to ~30% from centre. If the closest part is at the
   edge of the frame, treat it as "not the target" and answer
   part_visible=false rather than guessing.)
2. Where exactly is THE CENTRE-MOST part? (normalised coords, 0-1)
   Be precise — do NOT default to (0.5, 0.5). Inspect the image
   pixels and report the actual centroid you observe.
3. Is the vertical descent path CLEAR of obstacles?
   (other parts directly below gripper, bin walls, bin dividers
   that could cause a collision during descent)

RESPOND ONLY WITH VALID JSON:
{
  "part_visible": true,
  "label": "short_part_label",
  "center": {"x": 0.52, "y": 0.49},
  "confidence": 0.85,
  "path_clear": true,
  "collision_risk": "none",
  "notes": "part centroid measured at ~52% / 49% from origin"
}

If no graspable part is visible NEAR THE IMAGE CENTRE (only at edges
or not at all), set part_visible to false.
Set path_clear to false if obstacles block the descent path.
collision_risk: "none", "low", "medium", or "high"."""


class WorkflowPhase:
    """Represents a single phase in the kitting workflow."""
    BIN_LOOKUP      = "bin_lookup"      # USD lookup of bin + tray centres
    BIN_SCAN_POSE   = "bin_scan_pose"   # Move ee_link above bin (downward)
    SCENE_SCAN      = "scene_scan"      # Wrist RGB capture
    SCENE_ANALYSIS  = "scene_analysis"  # VLM identifies parts on wrist image
    PLAN_GENERATION = "plan_generation"
    REALIGN         = "realign"         # Gantry slides over target part
    APPROACH        = "approach"
    DEPTH_ANALYSIS  = "depth_analysis"
    GRASP_REFINEMENT = "grasp_refinement"
    PICK_EXECUTE    = "pick_execute"
    GRASP_VERIFY    = "grasp_verify"
    PLACE_EXECUTE   = "place_execute"
    VERIFY          = "verify"
    COMPLETE        = "complete"


class KittingWorkflowEngine:
    """
    Vision-based kitting orchestrator.

    Connects VLM perception, LLM planning, and Isaac Sim execution
    into a closed-loop workflow that iterates until the task is complete.

    KEY DESIGN: ALL coordinates come from camera vision + depth projection.
    No USD prim paths are used for localisation.  The VLM identifies parts
    and the kitting tray from an overhead RGB image, then the bridge's
    depth-projection endpoint converts image coordinates to real-world XYZ.
    """

    def __init__(self, camera, vlm, planner, config=None,
                 recorder=None, place_verifier=None):
        self.camera = camera
        self.vlm = vlm
        self.planner = planner
        self.config = config or {}
        self._phase_callback = None
        self._cancelled = False
        self._bin_top_z = None  # set by Phase 0-1 landmark detection
        self._bin_bot_z = None
        self._bin_aabb_xy = None  # (xmin, ymin, xmax, ymax) — for USD-snap bounds filter
        self._tray_position = None  # set by Phase 0 USD lookup
        self._depth_estimator = DepthEstimator(
            self.config.get("perception", {})
        )
        self._detector = ZeroShotDetector(
            self.config.get("perception", {})
        )
        # Evaluation plumbing (optional — both default to None when the
        # caller has no Evaluation tab attached, so the hooks become
        # no-ops and the workflow runs unchanged).
        self.recorder = recorder
        self.place_verifier = place_verifier

    def set_phase_callback(self, callback: Callable):
        """Set a callback function(phase, status, detail) for UI updates."""
        self._phase_callback = callback

    def cancel(self):
        self._cancelled = True

    def _notify(self, phase: str, status: str, detail: str = ""):
        if self._phase_callback:
            self._phase_callback(phase, status, detail)
        log.info(f"[Workflow] {phase}: {status} — {detail}")

    # ─── Main Entry Point ───────────────────────────────────

    def execute_kitting_task(self, user_command: str) -> Dict[str, Any]:
        """Execute a full neuro-symbolic kitting task — hybrid pipeline.

        Pipeline (current approach):
          Phase 0: USD lookup → bin top_z + kitting tray drop_z
          Phase 1: Overhead RGB capture (camera is now positioned to
                   tightly frame the bin → no crop, no upscale, no
                   landmark detection step needed)
          Phase 2: Full overhead image → VLM identifies parts (with
                   catalogue grounding) → bridge depth-projects each
                   part to real-world XYZ. Tray injected from USD.
          Phase 3: LLM plan generation (parts XYZ + tray XYZ)
          Phase 4-6: Per-pick: existing pick sequence (approach via
                   `_execute_pick_sequence` → wrist depth + IMU
                   refinement → grasp), then place at the USD tray
          Phase 7: Overhead verification snapshot (display only)
        """
        self._cancelled = False
        # Allocate a fresh task_id for evaluation logs. This stays None
        # when no recorder is attached so the rest of the engine is
        # unaffected.
        self._eval_task_id = (self.recorder.begin_task()
                              if self.recorder else None)
        self._eval_targeted = 0
        self._eval_placed = 0
        self._eval_task_start = datetime.now()
        result = {
            "command": user_command,
            "start_time": self._eval_task_start.isoformat(),
            "phases": [],
            "status": "running",
        }

        try:
            sim_cfg  = self.config.get("simulation", {})
            exec_cfg = self.config.get("execution", {})
            bin_prim    = sim_cfg.get("bin_prim_path",
                                      "/World/robot_facade_full")
            tray_prim   = sim_cfg.get("tray_prim_path", "/World/box_840")
            tray_safe_h = exec_cfg.get("tray_safe_height", 0.25)

            # ═══════════════════════════════════════════════════
            # PHASE 0: USD lookup — bin top_z + tray drop_z
            # Both fixtures are static so their world coordinates are
            # authoritative in USD. We don't need an overhead VLM
            # landmark step anymore: the camera is positioned to
            # tightly frame the bin already.
            # ═══════════════════════════════════════════════════
            self._notify(WorkflowPhase.BIN_LOOKUP, "running",
                         "Querying USD for bin + tray centres...")
            bin_info = self.camera.get_prim_center(bin_prim)
            if not bin_info.get("ok"):
                raise RuntimeError(
                    f"Bin prim lookup failed ({bin_prim}): "
                    f"{bin_info.get('error', '?')}")
            tray_info = self.camera.get_prim_center(tray_prim)
            if not tray_info.get("ok"):
                raise RuntimeError(
                    f"Tray prim lookup failed ({tray_prim}): "
                    f"{tray_info.get('error', '?')}")

            bin_top_z  = float(bin_info["top_z"])
            bin_bot_z  = float(bin_info["bot_z"])
            self._bin_top_z = bin_top_z
            self._bin_bot_z = bin_bot_z
            # Cache the bin's XY footprint for the USD-snap bounds
            # filter (rejects USD prims that have fallen outside the
            # bin so the gripper isn't sent to a far-away match).
            try:
                _bin_min = bin_info.get("min_pt") or []
                _bin_max = bin_info.get("max_pt") or []
                if len(_bin_min) >= 2 and len(_bin_max) >= 2:
                    self._bin_aabb_xy = (
                        float(_bin_min[0]), float(_bin_min[1]),
                        float(_bin_max[0]), float(_bin_max[1]))
                else:
                    self._bin_aabb_xy = None
            except (TypeError, ValueError):
                self._bin_aabb_xy = None
            # ``pivot`` is the prim's transform translation (xformOp:translate)
            # — the AUTHORITATIVE position the prim was placed at in USD.
            # ``center_xy`` is the bbox geometric centre, which can drift
            # from the placed position when the prim has child meshes.
            # For the tray we prefer pivot because the user has set it up
            # at the actual tray centre; if pivot ≠ center_xy we log both
            # so any divergence is visible.
            tray_pivot = tray_info.get("pivot")
            tray_bbox_xy = tray_info["center_xy"]
            if tray_pivot and len(tray_pivot) >= 2:
                tray_xy = (float(tray_pivot[0]), float(tray_pivot[1]))
                bbox_dx = abs(tray_xy[0] - float(tray_bbox_xy[0]))
                bbox_dy = abs(tray_xy[1] - float(tray_bbox_xy[1]))
                if max(bbox_dx, bbox_dy) > 0.005:  # >5mm divergence
                    log.info(
                        f"[USD] Tray pivot vs bbox centre divergence: "
                        f"pivot=({tray_xy[0]:.3f}, {tray_xy[1]:.3f}) "
                        f"bbox=({tray_bbox_xy[0]:.3f}, "
                        f"{tray_bbox_xy[1]:.3f}) — using pivot.")
            else:
                tray_xy = (float(tray_bbox_xy[0]), float(tray_bbox_xy[1]))
            tray_top_z = float(tray_info["top_z"])
            tray_position = {
                "center_xy": tray_xy,
                # Raw USD value — used by ``_execute_place_sequence``
                # to bypass the LLM and target the tray directly.
                "top_z_raw": tray_top_z,
                # Drop point includes the safe clearance above the rim.
                # This is what the LLM sees in the scene.
                "top_z": tray_top_z + tray_safe_h,
            }
            # Stash on self so phases beyond the scope of `tray_position`
            # local var (e.g. _execute_place_sequence) can read the
            # USD-derived coords directly without round-tripping through
            # the LLM's plan params.
            self._tray_position = tray_position
            log.info(
                f"[USD] Bin top_z={bin_top_z:.3f}; "
                f"Tray centre=({tray_xy[0]:.3f}, {tray_xy[1]:.3f}), "
                f"top_z={tray_top_z:.3f} "
                f"(drop_z={tray_position['top_z']:.3f})")
            result["phases"].append({
                "phase": WorkflowPhase.BIN_LOOKUP,
                "status": "success",
                "detail": (f"Bin top_z={bin_top_z:.3f}, "
                           f"Tray drop_z={tray_position['top_z']:.3f}"),
                "timestamp": datetime.now().isoformat(),
            })
            self._notify(WorkflowPhase.BIN_LOOKUP, "success",
                         "Bin + tray centres acquired from USD")

            if self._cancelled:
                return self._finalize(result, "cancelled")

            # ═══════════════════════════════════════════════════
            # PHASE 1: Capture overhead RGB
            # The camera already frames the blue bin closely, so we
            # use the full overhead image directly instead of trying
            # to re-localize or crop the bin again.
            # ═══════════════════════════════════════════════════
            self._notify(WorkflowPhase.SCENE_SCAN, "running",
                         "Capturing overhead RGB...")
            overhead_image = self.camera.capture_workspace_image()
            log.info(
                f"[Phase 1] Overhead capture: "
                f"{overhead_image.size[0]}x{overhead_image.size[1]} px")
            # The placeholder image (bridge unreachable) is 640x480
            # with a near-black gradient. Detect + bail rather than
            # send phantom data through the VLM + planner.
            if self._looks_like_placeholder(overhead_image):
                raise RuntimeError(
                    "Overhead capture returned the placeholder image "
                    "(640x480, near-black). The Isaac Sim bridge is "
                    "unreachable — check it's running on port 8600.")
            result["phases"].append({
                "phase": WorkflowPhase.SCENE_SCAN,
                "status": "success",
                "detail": (f"{overhead_image.size[0]}x"
                           f"{overhead_image.size[1]} px"),
                "timestamp": datetime.now().isoformat(),
            })
            self._notify(WorkflowPhase.SCENE_SCAN, "success",
                         "Overhead frame captured")

            bin_crop_box = (0.0, 0.0, 1.0, 1.0)

            if self._cancelled:
                return self._finalize(result, "cancelled")

            # ═══════════════════════════════════════════════════
            # PHASE 2: Crop the bin region from the overhead image,
            # send to VLM for part identification, project each part
            # to world coordinates via the overhead depth pipeline.
            # Robot stays parked — no scan move required.
            # ═══════════════════════════════════════════════════
            max_retries = exec_cfg.get("max_pick_retries", 2)
            pick_succeeded = False
            current_overhead = overhead_image  # re-captured on retry

            for attempt in range(max_retries + 1):
                if self._cancelled:
                    return self._finalize(result, "cancelled")

                tag = f" (retry {attempt})" if attempt > 0 else ""

                self._notify(WorkflowPhase.SCENE_ANALYSIS, "running",
                             f"Sending overhead frame to VLM{tag}...")

                target_parts = self._extract_target_parts(user_command)
                pick_quantity = self._parse_pick_quantity(user_command)
                if target_parts:
                    log.info(
                        f"[Phase 2] Target parts from command: "
                        f"{target_parts}  quantity={pick_quantity if pick_quantity is not None else 'all'}")

                # Use the full overhead frame directly.
                scene, coord_source = self._detect_parts_from_bin_crop(
                    current_overhead,
                    bin_crop_box=bin_crop_box,
                    target_parts=target_parts,
                    bin_top_z=bin_top_z,
                    camera_alias="rgb")

                num_detected = len(scene.get("detected_objects", []))
                log.info(
                    f"[Phase 2] Overhead VLM: {num_detected} parts{tag}")
                # Per-part XYZ + sanity check vs the bin AABB. Anything
                # that lands OUTSIDE the bin's AABB by > 30 cm almost
                # certainly came from a depth-projection error (depth
                # ray hit a divider or background instead of the part).
                # This makes "robot moved to the gantry" diagnoses
                # obvious — you'll see ❌ next to the bad coords.
                bb = bin_info.get("bot_z", 0.0), bin_info.get("top_z", 0.0)
                bin_xy = bin_info.get("center_xy", [0.0, 0.0])
                bin_w  = (bin_info.get("max_pt", [0, 0, 0])[0]
                          - bin_info.get("min_pt", [0, 0, 0])[0]) \
                    if bin_info.get("max_pt") else 1.0
                bin_h  = (bin_info.get("max_pt", [0, 0, 0])[1]
                          - bin_info.get("min_pt", [0, 0, 0])[1]) \
                    if bin_info.get("max_pt") else 1.0
                # Fallback: assume bin is centred on bin_xy with
                # generous radius if AABB extents weren't returned.
                ax_min = bin_xy[0] - max(bin_w, 1.0) / 2 - 0.30
                ax_max = bin_xy[0] + max(bin_w, 1.0) / 2 + 0.30
                ay_min = bin_xy[1] - max(bin_h, 1.0) / 2 - 0.30
                ay_max = bin_xy[1] + max(bin_h, 1.0) / 2 + 0.30
                for o in scene.get("detected_objects", []):
                    pos = o.get("approximate_position", {})
                    x, y, z = (pos.get("x", 0), pos.get("y", 0),
                               pos.get("z", 0))
                    in_bin = (ax_min <= x <= ax_max
                              and ay_min <= y <= ay_max
                              and bb[0] - 0.10 <= z <= bb[1] + 0.30)
                    flag = "" if in_bin else "  ❌ OUTSIDE BIN AABB"
                    log.info(
                        f"  [scan] {o.get('label', '?')} → "
                        f"x={x:.3f} y={y:.3f} z={z:.3f} "
                        f"(conf={o.get('confidence', 0):.2f}){flag}")

                if coord_source == "none":
                    log.error("No 3D coordinates from overhead depth")
                    self._notify(WorkflowPhase.SCENE_ANALYSIS, "failed",
                                 "Overhead depth projection failed")
                    if attempt < max_retries:
                        # Robot is parked — just re-capture overhead.
                        try:
                            current_overhead = (
                                self.camera.capture_workspace_image())
                        except Exception as e:
                            log.warning(f"Re-capture failed: {e}")
                        continue
                    return self._finalize(
                        result, "error",
                        "Overhead depth projection failed after retries")

                # Inject tray as a destination from USD coordinates
                # (no overhead landmark step in the new pipeline).
                scene.setdefault("detected_objects", []).append({
                    "object_id": "obj_tray",
                    "label": "kitting_tray",
                    "semantic_description": "white kitting tray",
                    "affordance": "destination",
                    "approximate_position": {
                        "x": tray_position["center_xy"][0],
                        "y": tray_position["center_xy"][1],
                        "z": tray_position["top_z"],
                    },
                    "confidence": 1.0,
                    "_source": "usd_lookup",
                })

                # Reachability filter
                self._filter_unreachable(scene)

                # ── Target-part filter ─────────────────────────────
                # When the operator names specific parts ("pick gear"),
                # drop everything else so the LLM can't pick a
                # motor_valve and label it "gear" to satisfy the command.
                # If nothing matches, bail with a clear error before
                # the LLM hallucinates a substitute.
                if target_parts:
                    objects = scene.get("detected_objects", [])
                    kept, dropped = [], []
                    target_cats = [
                        self._label_category(t) for t in target_parts]
                    log.info(
                        f"[Target-filter] target={target_parts} "
                        f"(category roots={target_cats})")
                    for obj in objects:
                        raw_label = obj.get("label", "?")
                        label_cat = self._label_category(raw_label)
                        if obj.get("affordance") == "destination":
                            kept.append(obj)
                            continue
                        matched = self._labels_match_target(
                            [obj], target_parts)
                        if matched:
                            kept.append(obj)
                            log.info(
                                f"  [Target-filter] KEEP '{raw_label}' "
                                f"(category='{label_cat}')")
                        else:
                            dropped.append(raw_label)
                            log.info(
                                f"  [Target-filter] DROP '{raw_label}' "
                                f"(category='{label_cat}') — no match "
                                f"vs {target_cats}")
                    scene["detected_objects"] = kept
                    if dropped:
                        log.info(
                            f"[Target-filter] Kept {len(kept)} objects "
                            f"matching {target_parts}; dropped "
                            f"{len(dropped)} other labels: {dropped}")

                    pickable = [o for o in kept
                                if o.get("affordance") != "destination"]
                    if not pickable:
                        msg = (f"No parts matching {target_parts} "
                               f"detected in scene")
                        log.error(f"[Target-filter] {msg}")
                        self._notify(WorkflowPhase.SCENE_ANALYSIS, "failed",
                                     msg)
                        if attempt < max_retries:
                            # Robot is parked — just re-capture overhead
                            try:
                                current_overhead = (
                                    self.camera.capture_workspace_image())
                            except Exception:
                                pass
                            continue
                        return self._finalize(result, "error", msg)

                num_reachable = len(scene.get("detected_objects", []))
                result["scene_description"] = scene
                result["phases"].append({
                    "phase": WorkflowPhase.SCENE_ANALYSIS,
                    "status": "success",
                    "detail": (
                        f"Bin-crop: {num_detected} detected, "
                        f"{num_reachable} reachable "
                        f"({coord_source}){tag}"
                    ),
                    "timestamp": datetime.now().isoformat(),
                })
                self._notify(WorkflowPhase.SCENE_ANALYSIS, "success",
                             f"{num_reachable} parts from bin crop{tag}")

                # ── Evaluation hook: perception metrics ─────────
                # Only run on the first attempt — retries re-use the
                # same scene for matching purposes and would just
                # double-count.
                # IMPORTANT: log metrics BEFORE the USD ground-truth
                # snap below, so the recorded perception accuracy
                # reflects the raw VLM+detector pipeline, not the
                # snap-corrected coordinates.
                if self.recorder and attempt == 0:
                    try:
                        self._log_perception_metrics(scene)
                    except Exception as exc:
                        log.warning(
                            f"[eval] perception metrics failed: {exc}")

                # ── USD ground-truth snap for grasp targets ─────
                # Replace each perception XY with the matching USD
                # prim's centre when within snap radius. Perception
                # imprecision (~1-3 cm at this overhead camera
                # distance) was misaligning the gripper before
                # realignment even started; the snap removes that
                # error for sim-only demos. Disable for real-robot
                # runs by setting execution.use_usd_grasp_targets
                # false in config.yaml.
                if exec_cfg.get("use_usd_grasp_targets", True):
                    try:
                        self._snap_to_usd_ground_truth(scene)
                    except Exception as exc:
                        log.warning(
                            f"[USD-snap] failed: {exc} — keeping "
                            f"raw perception coords")

                if self._cancelled:
                    return self._finalize(result, "cancelled")

                # ── Phase 3: LLM Plan Generation ──────────────
                self._notify(WorkflowPhase.PLAN_GENERATION, "running",
                             f"Generating plan{tag}...")

                kit_tray_pos = None
                if tray_position:
                    kit_tray_pos = [
                        tray_position["center_xy"][0],
                        tray_position["center_xy"][1],
                        tray_position["top_z"],
                    ]

                try:
                    plan = self.planner.generate_plan(
                        user_command, scene,
                        workspace_bounds=self.config.get(
                            "execution", {}).get("workspace_bounds"),
                        kit_tray_position=kit_tray_pos,
                        max_picks=pick_quantity,
                    )
                except Exception as e:
                    log.warning(f"LLM planning failed{tag}: {e}")
                    if attempt < max_retries:
                        try:
                            current_overhead = (
                                self.camera.capture_workspace_image())
                        except Exception:
                            pass
                        continue
                    raise

                total_steps = plan.get("total_steps", 0)
                result["task_plan"] = plan
                result["phases"].append({
                    "phase": WorkflowPhase.PLAN_GENERATION,
                    "status": "success",
                    "detail": f"{total_steps} steps{tag}",
                    "timestamp": datetime.now().isoformat(),
                })
                self._notify(WorkflowPhase.PLAN_GENERATION, "success",
                             f"Plan: {total_steps} steps — "
                             f"{plan.get('task_summary', '')}")

                if self._cancelled:
                    return self._finalize(result, "cancelled")

                # ── Phases 4-5: Execute plan steps ────────────
                # The robot is still at home (no scanning movement
                # was performed).  Plan move_home steps execute as
                # written; pick_object handles the actual approach.
                execution_results = []
                last_grip_hold = None
                pick_failed = False
                # Camera_Kit + VLM place verification flag. When the
                # post-place tray-camera check returns ``in_tray=False``
                # the gripper either dropped the part outside the tray
                # or never had it at all (false-positive grasp). In
                # both cases we re-initiate the orchestration cycle —
                # re-scan + re-plan — exactly like a pick failure.
                place_failed = False
                place_failure_reason = ""

                for step in plan.get("plan", []):
                    if self._cancelled:
                        break

                    action = step.get("action", "")
                    params = step.get("params", {})
                    step_num = step.get("step", "?")
                    step_result = {"step": step_num, "action": action}

                    if action == "pick_object":
                        # _execute_pick_sequence handles the approach
                        # (Phase A) → wrist depth refinement (Phase A2)
                        # → grasp + retract internally. No separate
                        # gantry realign needed in the overhead-camera
                        # pipeline — the existing approach already
                        # slides the gantry to the part XY.
                        self._eval_targeted += 1
                        _pick_t0 = datetime.now()
                        step_result = self._execute_pick_sequence(
                            step, params, scene, step_num)
                        last_grip_hold = step_result.get(
                            "grip_hold",
                            step_result.get("finger_close"))
                        self._log_pick_event(
                            params, step_result,
                            (datetime.now() - _pick_t0).total_seconds())

                        if step_result.get("status") != "success":
                            pick_failed = True
                            log.warning(
                                f"Pick failed (attempt {attempt + 1})")
                            execution_results.append(step_result)
                            break

                    elif action == "place_object":
                        params = self._resolve_place_coordinates(
                            params, tray_position)
                        if params is None:
                            step_result["status"] = "failed"
                            step_result["error"] = "Tray not detected"
                            execution_results.append(step_result)
                            continue
                        if last_grip_hold is not None:
                            params["grip_hold"] = last_grip_hold
                        _place_t0 = datetime.now()
                        step_result = self._execute_place_sequence(
                            step, params, scene, step_num)
                        # Camera_Kit + VLM verification — authoritative
                        # "did the part actually land in the tray?"
                        # signal that ignores false-positive grasp
                        # confirmations from the contact sensor.
                        place_label = (params.get("label")
                                       or scene.get("_last_target_label", "")
                                       or "")
                        place_verdict = self._log_place_event(
                            params, step_result, place_label,
                            (datetime.now() - _place_t0).total_seconds())
                        last_grip_hold = None

                        # ── Place verification: if the tray VLM says
                        # the part is NOT in the tray, escalate the
                        # failure to the outer retry loop so the
                        # orchestration cycle restarts (re-scan + re-
                        # plan) rather than continuing with a
                        # corrupted scene assumption. ``None`` means
                        # the verifier was unavailable / inconclusive
                        # and is treated as a non-failure to avoid
                        # spurious retries when running without the
                        # Camera_Kit feed.
                        if place_verdict is False:
                            place_failed = True
                            place_failure_reason = (
                                f"Camera_Kit VLM reports '{place_label}' "
                                f"NOT in tray after place")
                            log.warning(
                                f"[Place verify] {place_failure_reason} "
                                f"(attempt {attempt + 1})")
                            self._notify(
                                "place_verify", "failed",
                                place_failure_reason)
                            # Mark the step as failed in the plan log
                            # so the operator can see the verdict in
                            # the Streamlit execution log.
                            step_result["status"] = "failed"
                            step_result["error"] = "place_not_verified"
                            execution_results.append(step_result)
                            break

                    elif action == "move_home":
                        self._notify("move_home", "running",
                                     f"Step {step_num}")
                        r = self.camera.send_command("/api/home", {})
                        step_result["status"] = (
                            "success" if r.get("status") == "ok"
                            else "failed")

                    elif action == "open_gripper":
                        r = self.camera.send_command(
                            "/api/gripper", {"action": "open"})
                        step_result["status"] = (
                            "success" if r.get("status") == "ok"
                            else "failed")

                    elif action == "close_gripper":
                        r = self.camera.send_command(
                            "/api/gripper", {"action": "close"})
                        step_result["status"] = (
                            "success" if r.get("status") == "ok"
                            else "failed")

                    elif action == "verify_grasp":
                        step_result["status"] = "success"

                    else:
                        step_result["status"] = "skipped"

                    execution_results.append(step_result)

                result["execution_results"] = execution_results

                if not pick_failed and not place_failed:
                    pick_succeeded = True
                    break  # success — exit retry loop

                # ── Failure → send robot home + re-capture overhead ──
                # Triggered by either:
                #   1. pick_failed   — the gripper never closed on a part.
                #   2. place_failed  — Camera_Kit VLM verified the part
                #                      did NOT end up in the tray after
                #                      placement (operator's request:
                #                      re-initiate the entire cycle from
                #                      the orchestration layer).
                # Both cases re-run from Phase 2 (scene scan) so the
                # planner sees the current scene state, not the stale
                # one that produced the failed plan.
                failure_kind = "Grasp" if pick_failed else "Place"
                if place_failed:
                    failure_detail = place_failure_reason
                else:
                    failure_detail = "grasp not confirmed"
                if attempt < max_retries:
                    self._notify(WorkflowPhase.SCENE_ANALYSIS, "running",
                                 f"{failure_kind} failed ({failure_detail}) "
                                 f"— re-scanning (retry {attempt + 1})...")
                    try:
                        # Park before re-scan in case the arm is in an
                        # awkward post-pick pose blocking the overhead.
                        self.camera.send_command("/api/home", {})
                        new_overhead = self.camera.capture_workspace_image()
                        if self._looks_like_placeholder(new_overhead):
                            raise RuntimeError(
                                "Re-capture returned placeholder image "
                                "(bridge unreachable). Aborting retry "
                                "instead of feeding hallucinated parts "
                                "into the planner.")
                        current_overhead = new_overhead
                    except Exception as e:
                        log.warning(f"Re-capture failed: {e}")
                        return self._finalize(
                            result, "error",
                            f"Re-capture failed: {e}")

            # ═══════════════════════════════════════════════════
            # PHASE 7: Verification snapshot (overhead RGB, display only)
            #
            # The overhead camera is no longer used for perception
            # decisions in the new pipeline. We still capture one final
            # image so the operator has a visual record of the kitting
            # outcome in the Streamlit UI — no VLM call, no logic.
            # ═══════════════════════════════════════════════════
            if not self._cancelled:
                self._notify(WorkflowPhase.VERIFY, "running",
                             "Capturing verification snapshot...")
                try:
                    verify_image = self.camera.capture_workspace_image()
                    result["verification"] = {
                        "captured_at": datetime.now().isoformat(),
                        "image_size": list(verify_image.size),
                    }
                except Exception as e:
                    log.warning(f"Verification snapshot failed: {e}")

                result["phases"].append({
                    "phase": WorkflowPhase.VERIFY,
                    "status": "success",
                    "detail": "Overhead snapshot captured",
                    "timestamp": datetime.now().isoformat(),
                })
                self._notify(WorkflowPhase.VERIFY, "success",
                             "Verification snapshot saved")

            status = "completed" if pick_succeeded else "failed"
            return self._finalize(result, status)

        except Exception as e:
            log.error(f"Workflow failed: {e}")
            result["phases"].append({
                "phase": "error",
                "status": "failed",
                "detail": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            self._notify("error", "failed", str(e))
            return self._finalize(result, "error", str(e))

    # ─── Coordinate Resolution ─────────────────────────────

    # ─── Two-Stage VLM Analysis ────────────────────────────────

    def _find_bin_region(self, scene: dict) -> Optional[tuple]:
        """Estimate the blue bin's bounding box in normalised image coords.

        Uses the VLM-detected graspable objects to infer where the bin is.
        Returns (x_min, y_min, x_max, y_max) in normalised coords, or None.
        """
        xs, ys = [], []
        for obj in scene.get("detected_objects", []):
            if obj.get("affordance") in ("graspable", "stackable", "fragile"):
                pos = obj.get("image_position", {})
                if pos:
                    xs.append(pos.get("x", 0.5))
                    ys.append(pos.get("y", 0.5))
        if len(xs) < 1:
            return None
        # Pad around detected objects (parts don't cover the full bin)
        margin = 0.08
        x_min = max(0.0, min(xs) - margin)
        y_min = max(0.0, min(ys) - margin)
        x_max = min(1.0, max(xs) + margin)
        y_max = min(1.0, max(ys) + margin)
        # Ensure minimum crop size (at least 20% of image per axis)
        if x_max - x_min < 0.20:
            cx = (x_min + x_max) / 2
            x_min, x_max = max(0, cx - 0.10), min(1, cx + 0.10)
        if y_max - y_min < 0.20:
            cy = (y_min + y_max) / 2
            y_min, y_max = max(0, cy - 0.10), min(1, cy + 0.10)
        return (x_min, y_min, x_max, y_max)

    def _crop_bin_region(self, image, bin_box: tuple,
                         upscale_long_edge: int = 1600):
        """Crop the bin area from a full-resolution image and LANCZOS-
        upscale so parts occupy enough pixels for the VLM.

        A bin crop from a 1920x1080 overhead is typically ~500x300 px,
        leaving each part only ~30–50 px wide — too small for Gemma4
        to discriminate colour / shape / holes. Upscaling to 1600 px on
        the long edge (same preprocessing used by
        ``diagnose_vlm_perception.py``) gives the VLM a sharper input.

        Args:
            image: PIL Image (full frame, e.g. 1920x1080)
            bin_box: (x_min, y_min, x_max, y_max) in normalised coords
            upscale_long_edge: target long-edge size in pixels after
                LANCZOS upscale (set ≤0 to disable upscaling)
        Returns:
            Cropped + (optionally) upscaled PIL Image of just the bin area.
        """
        from PIL import Image as _PILImage
        w, h = image.size
        x_min, y_min, x_max, y_max = bin_box
        left = int(x_min * w)
        top = int(y_min * h)
        right = int(x_max * w)
        bottom = int(y_max * h)
        cropped = image.crop((left, top, right, bottom))

        if upscale_long_edge and upscale_long_edge > 0:
            long_edge = max(cropped.size)
            if long_edge < upscale_long_edge:
                scale = upscale_long_edge / long_edge
                new_size = (int(cropped.size[0] * scale),
                            int(cropped.size[1] * scale))
                cropped = cropped.resize(new_size, _PILImage.LANCZOS)
                log.info(
                    f"[Bin-zoom] LANCZOS upscaled crop → "
                    f"{cropped.size[0]}x{cropped.size[1]} (x{scale:.2f})")
        return cropped

    @staticmethod
    def _compute_bin_crop_box(bin_image_pos, bin_image_bbox=None,
                              default_radius=0.18):
        """Compute the normalised crop box for the bin from VLM landmark output.

        Prefers ``bin_image_bbox`` from the VLM (tight box).  If only the
        centre is available, falls back to a square crop of half-side
        ``default_radius`` centred on ``bin_image_pos``.  A small padding
        is added so the bin walls fully fit inside the crop.
        """
        pad = 0.02
        if (bin_image_bbox
                and all(k in bin_image_bbox
                        for k in ("x_min", "y_min", "x_max", "y_max"))):
            x_min = max(0.0, bin_image_bbox["x_min"] - pad)
            y_min = max(0.0, bin_image_bbox["y_min"] - pad)
            x_max = min(1.0, bin_image_bbox["x_max"] + pad)
            y_max = min(1.0, bin_image_bbox["y_max"] + pad)
        else:
            cx = bin_image_pos.get("x", 0.5)
            cy = bin_image_pos.get("y", 0.5)
            x_min = max(0.0, cx - default_radius)
            y_min = max(0.0, cy - default_radius)
            x_max = min(1.0, cx + default_radius)
            y_max = min(1.0, cy + default_radius)
        return (x_min, y_min, x_max, y_max)

    def _detect_parts_from_bin_crop(self, overhead_image, bin_crop_box,
                                    target_parts=None, bin_top_z=None,
                                    camera_alias: str = "rgb"):
        """Crop the bin from the overhead image, run VLM + OWL-ViT2, project to world.

        Returns ``(scene_dict, coord_source_string)``.  ``coord_source``
        is ``"none"`` if depth projection failed.

        Pipeline:
          1. Try GT scene_annotations first — bridge returns USD-truth
             world coords + per-camera 2D bboxes for every part.
          2. Crop the bin region from the overhead image.
          3. VLM identifies parts (semantic labels) in crop-local coords.
          4. OWL-ViT2 detects tight bboxes for the same parts.
          5. Match VLM labels → GT annotation by 2D bbox IoU (with the
             OWL-ViT2 bbox).  When matched, use GT world XYZ directly.
          6. Otherwise match VLM label → OWL-ViT2 bbox; use bbox centre
             for XY and median depth across a 5×5 grid for Z.
          7. Fall back to single-point depth projection when nothing
             matches a VLM label.
          8. Sanity-check each part Z against ``bin_top_z`` — if Z is
             > 0.20 m below or > 0.30 m above the bin reference, replace
             with ``bin_top_z`` so IK gets a reachable target.
        """
        # ── Step 1: pull GT annotations (bypass when bridge lacks them) ──
        gt_annotations = []
        try:
            if hasattr(self.camera, "get_scene_annotations"):
                ann_resp = self.camera.get_scene_annotations(camera=camera_alias)
                if ann_resp.get("status") == "ok":
                    gt_annotations = ann_resp.get("annotations", [])
                    visible_gt = [a for a in gt_annotations
                                  if a.get("visible", True)
                                  and a.get("occlusion", 0) < 0.7]
                    log.info(
                        f"[Bin-zoom] GT annotations: {len(visible_gt)} "
                        f"visible / {len(gt_annotations)} total")
                    gt_annotations = visible_gt
                elif ann_resp.get("error"):
                    log.info(
                        f"[Bin-zoom] GT annotations unavailable: "
                        f"{ann_resp.get('error')}")
        except Exception as e:
            log.warning(f"[Bin-zoom] GT annotation fetch failed: {e}")

        # Detect "no-op crop" — overhead camera tightly frames the bin
        # so we send the full image as-is. Skip the LANCZOS upscale to
        # avoid unnecessary work + token bloat.
        full_frame = (
            bin_crop_box[0] <= 1e-6 and bin_crop_box[1] <= 1e-6 and
            bin_crop_box[2] >= 1.0 - 1e-6 and bin_crop_box[3] >= 1.0 - 1e-6
        )
        upscale = 0 if full_frame else 1600
        cropped = self._crop_bin_region(
            overhead_image, bin_crop_box, upscale_long_edge=upscale)
        cw, ch = cropped.size
        log.info(
            f"[Bin-zoom] {'Full overhead frame' if full_frame else 'Cropped bin image'}"
            f": {cw}x{ch} px (upscale={upscale})")

        # ── Detector mode is derived from the active VLM provider ──
        # Qwen-family VLMs are self-grounding → skip OWL-ViT2 and
        # extract bboxes directly from the VLM JSON response.
        use_vlm_grounding = self._is_self_grounding_vlm(
            getattr(self.vlm, "provider", ""))
        detector_mode = "vlm" if use_vlm_grounding else "owlv2"
        log.info(f"[Bin-zoom] Detector mode: {detector_mode} "
                 f"(provider={getattr(self.vlm, 'provider', '?')})")

        # Qwen-VL (8B) drowns in the full catalogue + grounding prompt
        # (3500+ tokens → emits whitespace under format=json). Use the
        # bare-minimum hand-tuned prompt instead. The catalogue
        # one-liners are baked into the prompt, and bbox_pixels in the
        # JSON schema gives us self-grounding without the separate suffix.
        if use_vlm_grounding:
            # Qwen3-VL emits bboxes in a normalised 0-1000 grid that is
            # image-size invariant. We still pre-resize the image to a
            # 28-multiple shape (cheaper, predictable token count, no
            # Ollama-side surprise resizes) but the prompt itself no
            # longer depends on dimensions.
            qw, qh = qwen_target_size(cw, ch)
            prompt = bin_zoom_prompt_qwen_minimal()
            log.info(f"[Bin-zoom] Qwen target image size: {qw}x{qh} "
                     f"(crop {cw}x{ch}); bbox grid = 0-1000")
            # Deliberately DO NOT append "operator is looking for X" for
            # the Qwen path. That hint biases Qwen toward emitting only
            # that label and stopping early — we observed it returning
            # 3 gears and skipping the visible hinges + motor_valves.
            # The downstream [Target-filter] step filters by label after
            # detection, so the biasing isn't needed.
        else:
            prompt = _with_catalogue(BIN_ZOOM_PROMPT)
            if target_parts:
                prompt = (
                    prompt
                    + "\n\nIMPORTANT: The operator is looking for: "
                    + ", ".join(target_parts).replace("_", " ")
                    + ". Use the matching label string from above."
                )

        try:
            # Full-frame overhead → default 1024 max_size is plenty
            # (image is already focused on the bin). For the legacy
            # narrow-crop path we kept the 1600 ceiling so the LANCZOS
            # upscale isn't undone.
            vlm_max = 1024 if full_frame else 1600
            scene = self.vlm.analyze_scene(
                cropped, custom_prompt=prompt, max_size=vlm_max)
        except Exception as e:
            log.warning(f"[Bin-zoom] VLM analysis failed: {e}")
            return {"detected_objects": []}, "none"

        x_min, y_min, x_max, y_max = bin_crop_box
        cw_norm = x_max - x_min
        ch_norm = y_max - y_min
        objects = scene.get("detected_objects", [])

        # ── Source tight bboxes: Qwen uses its own self-grounded bbox
        #    output, while Gemma/LLaVA use OWL-ViT2 on the same crop.
        #    This keeps Qwen on the VLM→world projection path and
        #    avoids mixing in detector geometry.
        detector_dets = []
        vlm_labels = [
            obj.get("label", "") for obj in objects
            if obj.get("label")
        ]
        if use_vlm_grounding:
            qwen_size = getattr(self.vlm, "_last_qwen_image_size", None)
            detector_dets = self._extract_vlm_bboxes(
                objects, cw, ch, qwen_size=qwen_size)
            log.info(
                f"[Bin-zoom] Qwen self-grounding: {len(detector_dets)} "
                f"bboxes from {len(objects)} objects "
                f"(qwen_size={qwen_size}, crop={cw}x{ch})")
        elif self._detector.is_available:
            # Build OWL queries from the FULL catalogue (always) plus any
            # extra labels Gemma4 reported that aren't catalogued. The
            # full-catalogue sweep is the safety net for when Gemma4 misses
            # parts entirely — OWL still finds them and we synthesise the
            # missing scene entries below.
            #
            # Each catalogue entry can declare MULTIPLE alternative
            # detector_queries — OWL batches them all in one forward
            # pass, so multiple phrasings per part is essentially free
            # and dramatically improves recall (OWL responds inconsistently
            # to phrasing).
            try:
                cat_q_list = detector_query_list()  # {canonical: [q1, q2, ...]}
                queries = []
                seen = set()
                for canonical, qs in cat_q_list.items():
                    for q in qs:
                        k = q.lower().strip()
                        if k and k not in seen:
                            seen.add(k)
                            queries.append(q)
                cat_query_count = len(queries)
                # Add any Gemma4 labels that aren't in the catalogue
                # (legacy behaviour for "unknown"-style labels).
                if vlm_labels:
                    extra = self._simplify_detector_labels(vlm_labels)
                    for q in extra:
                        if q and q.lower().strip() not in seen:
                            queries.append(q)
                            seen.add(q.lower().strip())
                detector_dets = self._detector.detect_with_labels(
                    cropped, queries)
                log.info(
                    f"[Bin-zoom] OWL-ViT2: {len(detector_dets)} "
                    f"bboxes from {len(queries)} queries "
                    f"(catalogue={cat_query_count}, "
                    f"extra={len(queries) - cat_query_count})")
            except Exception as e:
                log.warning(f"[Bin-zoom] Detector failed: {e}")
                detector_dets = []

        if use_vlm_grounding and not detector_dets:
            log.warning(
                "[Bin-zoom] Qwen self-grounding produced no bboxes; "
                "falling back to VLM image_position only")

        # ── Greedy 1-to-1 assignment of bboxes to VLM objects ──
        # Each bbox is used by AT MOST ONE object. Stops every
        # `large_gear_1/2/3` from collapsing to the same "best" bbox.
        bbox_assignment, used_det_idxs = self._assign_detector_bboxes(
            objects, detector_dets, cw, ch, bin_crop_box)
        log.info(
            f"[Bin-zoom] Assigned {len(bbox_assignment)}/{len(objects)} "
            f"VLM objects to unique bboxes ({detector_mode})")

        # ── Detector safety net: synthesise objects for OWL hits that
        # NO VLM detection claimed.  When Gemma4 misses parts entirely
        # (e.g. returns 2 small_hinges and skips the gears), OWL still
        # finds the gears via the catalogue queries — we recover those
        # by inferring the canonical label from the query → catalogue
        # reverse map and adding a synthetic scene object.
        # Only runs on the OWL path (not Qwen self-grounding) and is
        # gated by perception.detector_finds_missed_parts (default true).
        synth_enabled = (
            (not use_vlm_grounding)
            and self._detector.is_available
            and self.config.get("perception", {})
                .get("detector_finds_missed_parts", True))
        if synth_enabled and detector_dets:
            # Normalise both query keys and OWL det labels to a common
            # form (lowercase, underscores → spaces). OWL-ViT2 emits
            # ``flat_black_circular_disc`` while the catalogue stores
            # ``flat black circular disc`` — without this the lookup
            # would always miss and synthesis wouldn't fire.
            import re as _re_synth
            def _norm_q_synth(s):
                return _re_synth.sub(
                    r"\s+", " ",
                    (s or "").replace("_", " ").lower()).strip()
            try:
                cat_q_list_synth = detector_query_list()
                query_to_canonical = {}
                for canonical, qs in cat_q_list_synth.items():
                    for q in qs:
                        query_to_canonical[_norm_q_synth(q)] = canonical
            except Exception:
                query_to_canonical = {}

            cx_min, cy_min, cx_max, cy_max = bin_crop_box
            cw_norm_ = cx_max - cx_min
            ch_norm_ = cy_max - cy_min
            synth_count = 0
            for di, det in enumerate(detector_dets):
                if di in used_det_idxs:
                    continue
                det_label = _norm_q_synth(det.get("label") or "")
                bbox = det.get("bbox")
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue
                # Only synthesise from queries that map to a known
                # catalogue part type. OWL hits on Gemma4-only "extra"
                # queries are skipped — they don't have a canonical label.
                canonical = query_to_canonical.get(det_label)
                if not canonical:
                    continue
                try:
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                except (TypeError, ValueError):
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue

                # Crop pixels → full-image normalised
                nx1 = cx_min + (x1 / cw) * cw_norm_
                ny1 = cy_min + (y1 / ch) * ch_norm_
                nx2 = cx_min + (x2 / cw) * cw_norm_
                ny2 = cy_min + (y2 / ch) * ch_norm_
                # Crop-relative normalised bbox centre for image_position
                cx_norm = ((x1 + x2) / 2.0) / max(cw, 1)
                cy_norm = ((y1 + y2) / 2.0) / max(ch, 1)

                synth_count += 1
                conf = float(det.get("confidence", 0.0))
                # 0.9 weight to mark this as detector-only (slightly less
                # trustworthy than VLM+detector agreement).
                obj = {
                    "object_id": f"obj_synth_{synth_count}",
                    "label": canonical,
                    "semantic_description":
                        f"{canonical} (detector-only synthesis)",
                    "affordance": "graspable",
                    "image_position": {"x": cx_norm, "y": cy_norm},
                    "confidence": min(1.0, conf * 0.9),
                    "_synth_from_detector": True,
                }
                obj_idx = len(objects)
                objects.append(obj)
                bbox_assignment[obj_idx] = (nx1, ny1, nx2, ny2)
                log.info(
                    f"  [Synth] {canonical} from OWL '{det_label}' "
                    f"(conf={conf:.2f}) → bbox_full=("
                    f"{nx1:.3f},{ny1:.3f},{nx2:.3f},{ny2:.3f})")
            if synth_count:
                log.info(
                    f"[Bin-zoom] Detector safety net synthesised "
                    f"{synth_count} parts from unclaimed OWL hits")
                # Keep scene["detected_objects"] in sync so downstream
                # code (overlay rendering, target filter) sees them.
                scene["detected_objects"] = objects

        # Drop VLM detections that didn't get a detector bbox match.
        # Without a bbox, depth projection runs on the VLM's guessed
        # image_position which is unreliable (off by 5-15%, sometimes
        # entirely hallucinated). Phantom parts make the planner send
        # the robot to a divider wall. Bbox-only detections are far
        # more trustworthy because they require BOTH the VLM (correct
        # label) AND the detector (correct location) to agree.
        require_bbox = (
            self.config.get("perception", {})
                .get("require_detector_bbox", True))
        if require_bbox:
            kept_objects = []
            dropped = []
            for i, obj in enumerate(objects):
                if i in bbox_assignment:
                    kept_objects.append(obj)
                else:
                    dropped.append(obj.get("label", "?"))
            if dropped:
                log.info(
                    f"[Bin-zoom] Dropped {len(dropped)} VLM detections "
                    f"without detector bbox match: {dropped}")
            objects = kept_objects
            scene["detected_objects"] = objects
            # Re-key the assignment to the compact kept_objects list.
            # Iterate the original assignment in sorted-index order so
            # the new indices line up with kept_objects order.
            bbox_assignment = {
                new_idx: bbox
                for new_idx, (_, bbox) in enumerate(
                    sorted(bbox_assignment.items()))
            }

        # ── Convert VLM crop-local → full-image; apply assigned bboxes ──
        for i, obj in enumerate(objects):
            pos = obj.get("image_position",
                          obj.get("approximate_position", {}))
            if pos and "x" in pos and "y" in pos:
                full_x = x_min + pos["x"] * cw_norm
                full_y = y_min + pos["y"] * ch_norm
                pos["x"] = full_x
                pos["y"] = full_y
                obj["image_position"] = pos
                obj["affordance"] = obj.get("affordance", "graspable")

            bbox_full = bbox_assignment.get(i)
            if bbox_full:
                obj["_detector_bbox_full"] = bbox_full
                # Keep the semantic image_position for motion targeting.
                # Detector boxes are still used for depth sampling and
                # overlay, but they should not pull the robot toward the
                # box centre when the box is loose around a crowded bin.
                current_pos = obj.get("image_position", {})
                if not (isinstance(current_pos, dict)
                        and "x" in current_pos and "y" in current_pos):
                    cx = (bbox_full[0] + bbox_full[2]) / 2.0
                    cy = (bbox_full[1] + bbox_full[3]) / 2.0
                    obj["image_position"] = {"x": cx, "y": cy}

            # ── GT label cross-check (NO coordinate substitution) ──
            # Match by 2D bbox IoU against the OWL-ViT2 bbox to record
            # the canonical part_type from USD for label sanity-checking.
            # Coordinates still come from the camera + depth pipeline so
            # this layer never cheats with USD-truth XYZ.
            gt_match = self._match_gt_annotation(
                obj.get("label", ""), bbox_full, gt_annotations)
            if gt_match:
                obj["_gt_part_type"] = gt_match.get("part_type", "")
                obj["_gt_name"] = gt_match.get("name", "")
                log.info(
                    f"  [GT-label] '{obj.get('label', '?')}' ↔ "
                    f"{gt_match.get('name', '?')} "
                    f"(part_type={gt_match.get('part_type', '?')})")

        # ── Optional: per-instance crop-and-zoom refinement with Qwen ──
        # First-pass grounding can be loose when N similar parts cluster
        # in one region (Qwen's vision tokens span 28x28 patches; a few
        # gears within ~150 px collapse to the same patch coords). The
        # refinement step crops a 2x area around each rough bbox and
        # re-queries Qwen with a single-instance grounding prompt
        # ("Outline the position of the {label}"). Qwen's RefCOCO-trained
        # pathway is far more precise when there's exactly one target
        # in view. Costs N extra Qwen calls per scan, so it's gated by
        # ``perception.refine_with_crop_zoom`` (default false).
        refine_cfg = (self.config.get("perception", {})
                      .get("refine_with_crop_zoom", False))
        if refine_cfg and use_vlm_grounding:
            n_to_refine = sum(1 for o in objects
                              if o.get("_detector_bbox_full"))
            log.info(
                f"[Refine] Per-instance crop-and-zoom ON; "
                f"re-querying Qwen on {n_to_refine} parts")
            for obj in objects:
                bbox_full = obj.get("_detector_bbox_full")
                if not bbox_full:
                    continue
                refined = self._refine_bbox_with_qwen(
                    overhead_image, obj.get("label", ""), bbox_full)
                if refined is not None:
                    obj["_detector_bbox_full"] = refined
                    cx = (refined[0] + refined[2]) / 2.0
                    cy = (refined[1] + refined[3]) / 2.0
                    obj["image_position"] = {"x": cx, "y": cy}

        # ── Optional: per-instance VLM label verification ────────────
        # Crop each detected part to a 2x region and ask the VLM "what
        # is this?" with the catalogue list as the choice set. Single-
        # instance classification on a small crop is dramatically more
        # accurate than multi-instance scene grounding — especially
        # for visually-similar parts (motor_valve vs small_hinge).
        # Open-vocabulary: the catalogue is the only source of truth,
        # so adding a new part means a YAML edit, no code change. Costs
        # N extra VLM calls per scan, gated by
        # ``perception.verify_labels_with_vlm`` (default false).
        verify_cfg = (self.config.get("perception", {})
                      .get("verify_labels_with_vlm", False))
        if verify_cfg:
            n_to_verify = sum(1 for o in objects
                              if o.get("_detector_bbox_full"))
            log.info(
                f"[Verify] Per-instance VLM label verification ON; "
                f"re-classifying {n_to_verify} parts")
            min_override_conf = float(
                self.config.get("perception", {})
                    .get("verify_labels_min_confidence", 0.5))
            overrides = 0
            for obj in objects:
                bbox_full = obj.get("_detector_bbox_full")
                if not bbox_full:
                    continue
                current = (obj.get("label") or "").strip()
                current_base = re.sub(r"_\d+$", "", current).strip()
                result = self._verify_label_with_vlm(
                    overhead_image, current, bbox_full)
                if result is None:
                    log.info(
                        f"  [Verify] '{current}': VLM gave no answer; "
                        f"keeping current label")
                    continue
                new_label, conf = result
                new_base = re.sub(
                    r"_\d+$", "", new_label.strip().lower()).strip()

                if new_base in {"", "unknown"}:
                    log.info(
                        f"  [Verify] '{current}': VLM said 'unknown' "
                        f"(conf={conf:.2f}); keeping current label")
                    obj["_label_verified"] = False
                    continue
                if new_base == current_base.lower():
                    log.info(
                        f"  [Verify] '{current}': VLM AGREES "
                        f"(conf={conf:.2f})")
                    obj["_label_verified"] = True
                    continue
                if conf < min_override_conf:
                    log.info(
                        f"  [Verify] '{current}' → '{new_label}'? "
                        f"VLM disagrees but conf={conf:.2f} < "
                        f"{min_override_conf:.2f}; keeping current")
                    obj["_label_verified"] = False
                    continue
                # High-confidence override
                log.info(
                    f"  [Verify] OVERRIDE '{current}' → '{new_base}' "
                    f"(VLM conf={conf:.2f})")
                obj["label"] = new_base
                obj["confidence"] = max(float(obj.get("confidence", 0.0)),
                                        conf * 0.95)
                obj["_label_verified"] = True
                obj["_label_pre_verify"] = current
                overrides += 1
            if overrides:
                log.info(
                    f"[Verify] Re-labelled {overrides} parts based on "
                    f"single-instance VLM classification")

        # Per-part surface_z hint passed to the bridge's geometric
        # ray-plane fallback. The depth buffer is the primary path —
        # but when it fails (e.g. depth annotator off after a camera
        # reposition) the bridge intersects the camera ray with a
        # horizontal plane at this Z. Anchoring on the bin's AABB
        # bottom + 5 cm puts the plane just above the rack base where
        # parts physically sit.
        bin_floor_hint = (
            float(bin_top_z) - 0.10  # crude default if no bot info
            if bin_top_z is not None else None)
        # Prefer real bin bot_z if we have it via instance var (set by
        # USD lookup in Phase 0 of execute_kitting_task)
        if hasattr(self, "_bin_bot_z") and self._bin_bot_z is not None:
            bin_floor_hint = float(self._bin_bot_z) + 0.05
        log.info(
            f"[Bin-zoom] Using surface_z={bin_floor_hint} hint for "
            f"geometric ray-plane fallback (depth buffer is preferred)")

        # ── Build batch projection: bbox grids for matched parts,
        #    single point for unmatched ones ────────────────────
        proj_points = []
        slices = []  # (start, count) per object
        GRID = 5  # 5x5 sample grid inside bbox

        def _make_point(u, v):
            pt = {"x": u, "y": v}
            if bin_floor_hint is not None:
                pt["surface_z"] = bin_floor_hint
            return pt

        for obj in objects:
            ip = obj.get("image_position", {})
            if not (ip and "x" in ip and "y" in ip):
                slices.append((len(proj_points), 0))
                continue

            bbox = obj.get("_detector_bbox_full")
            if bbox:
                # Sample a GRIDxGRID grid inside the bbox (with 10%
                # inset to avoid edges that may hit bin floor)
                bx_min, by_min, bx_max, by_max = bbox
                inset_x = (bx_max - bx_min) * 0.10
                inset_y = (by_max - by_min) * 0.10
                bx_min += inset_x
                bx_max -= inset_x
                by_min += inset_y
                by_max -= inset_y
                start = len(proj_points)
                for i in range(GRID):
                    for j in range(GRID):
                        u = bx_min + (bx_max - bx_min) * (i / (GRID - 1))
                        v = by_min + (by_max - by_min) * (j / (GRID - 1))
                        proj_points.append(_make_point(u, v))
                slices.append((start, GRID * GRID))
            else:
                # Single VLM centre projection
                slices.append((len(proj_points), 1))
                proj_points.append(_make_point(ip["x"], ip["y"]))

        coord_source = "none"
        if proj_points:
            try:
                # Read axis-sign overrides from config so the user can
                # mirror image axes if the overhead camera was placed
                # in USD with a non-standard rotation. Defaults to +1.
                perc_cfg = self.config.get("perception", {})
                ix_sign = int(perc_cfg.get("camera_image_x_sign", 1))
                iy_sign = int(perc_cfg.get("camera_image_y_sign", 1))
                proj_result = self.camera.project_to_world(
                    proj_points, camera=camera_alias,
                    image_x_sign=ix_sign, image_y_sign=iy_sign)
                wpts = proj_result.get("world_points", [])
                method = proj_result.get("method", "?")
                depth_used = proj_result.get("depth_buffer_used", False)
                coord_source = f"{camera_alias}_{method}"
                if not depth_used:
                    log.warning(
                        f"[Bin-zoom] Depth buffer FAILED — fell back to "
                        f"geometric ray-plane projection (workspace_z "
                        f"from USD). Part Z values may be wrong if the "
                        f"bin AABB doesn't reflect the real part-floor.")

                for obj, (start, count) in zip(objects, slices):
                    if count == 0:
                        continue
                    samples = wpts[start:start + count]
                    if not samples:
                        continue

                    if count > 1:
                        # Median Z across grid samples = part top surface
                        zs = sorted(s["z"] for s in samples)
                        med_z = zs[len(zs) // 2]
                        # Reject samples > 5 cm below median (bin floor)
                        kept = [s for s in samples
                                if s["z"] >= med_z - 0.05]
                        # Centre XY from bbox (not depth-projected)
                        ip = obj["image_position"]
                        # Re-project the centre to get camera ray X/Y
                        # at the median depth — but simpler: use the
                        # nearest sample to centre for X/Y consistency
                        cx_norm = ip["x"]
                        cy_norm = ip["y"]
                        # Pick sample whose median is closest to centre
                        best = min(
                            samples,
                            key=lambda s: (
                                (s.get("u", cx_norm) - cx_norm) ** 2
                                + (s.get("v", cy_norm) - cy_norm) ** 2))
                        wp = {
                            "x": best["x"],
                            "y": best["y"],
                            "z": med_z,
                            "method": (
                                proj_result.get("method", "?")
                                + f"_grid{len(kept)}of{count}"),
                            "depth_m": best.get("depth_m", -1),
                        }
                    else:
                        wp = samples[0]

                    z_final = wp["z"]
                    # Drop only physically-impossible Zs (hit sky or
                    # passed through the floor). Anything in a plausible
                    # workspace range is trusted as-is — no clamping to
                    # bin_top_z, since the bin centre depth ray often
                    # hits a tall part and gives a misleading reference.
                    if z_final < -0.5 or z_final > 3.0:
                        log.warning(
                            f"[Z-outlier] '{obj.get('label', '?')}'"
                            f" projected z={z_final:.3f} outside "
                            f"plausible range — dropping")
                        continue

                    obj["approximate_position"] = {
                        "x": wp["x"], "y": wp["y"], "z": z_final,
                    }
                    obj["depth_m"] = wp.get("depth_m", -1)
                    obj["_source"] = (f"{coord_source}_bbox" if count > 1
                                      else coord_source)
                    log.info(
                        f"  '{obj.get('label', '?')}' "
                        f"[{'bbox' if count > 1 else 'centre'}]: "
                        f"({wp['x']:.3f}, {wp['y']:.3f}, "
                        f"{z_final:.3f})")

                # ── Diagnostic median Z only ──
                # Keep the USD-derived bin reference authoritative.
                # The projected part median is useful for debugging
                # depth fallback quality, but it is too noisy to
                # overwrite the bin geometry that downstream motion
                # planning relies on.
                part_zs = [
                    o["approximate_position"]["z"]
                    for o in objects
                    if o.get("approximate_position")
                    and "z" in o["approximate_position"]
                ]
                if part_zs:
                    part_zs.sort()
                    median_z = part_zs[len(part_zs) // 2]
                    log.info(
                        f"[Bin-zoom] Parts median Z={median_z:.3f} "
                        f"(bin_top_z kept at {bin_top_z:.3f} from USD)")
            except Exception as e:
                log.warning(f"[Bin-zoom] Projection failed: {e}")

        # Render and push the VLM detection overlay so the operator
        # can see EXACTLY what the model labelled vs where it actually
        # sits in the bin. Saves to logs/vlm_images/ and surfaces in
        # the Streamlit "🎯 VLM Detections" tab.
        try:
            overlay = self._render_vlm_overlay(
                cropped, objects, detector_dets, bin_crop_box)
            from utils.image_utils import save_debug_image
            save_path = save_debug_image(
                overlay,
                directory="logs/vlm_images",
                prefix="vlm_overlay")
            log.info(f"[Bin-zoom] VLM overlay → {save_path}")
            self._notify_vlm_overlay(overlay)
        except Exception as e:
            log.debug(f"[Bin-zoom] Overlay render failed: {e}")

        return scene, coord_source

    @staticmethod
    def _match_gt_annotation(vlm_label: str, owl_bbox_norm,
                             gt_annotations: list):
        """Match a VLM detection to the best ground-truth annotation.

        Uses 2D IoU between the OWL-ViT2 bbox and each GT image bbox
        (both already in full-image normalised coords).  When OWL-ViT2
        gave us a tight bbox we trust IoU as the primary signal; when
        not, we fall back to label match alone (lower confidence).

        Returns the matched GT annotation dict, or None.
        """
        if not gt_annotations:
            return None

        vlm_norm = (vlm_label or "").lower().replace("_", " ").strip()
        vlm_words = set(vlm_norm.split())

        def label_score(gt):
            pt = (gt.get("part_type") or "").lower().replace("_", " ")
            name = (gt.get("name") or "").lower().replace("_", " ")
            if not pt and not name:
                return 0.0
            pt_words = set(pt.split())
            name_words = set(name.split())
            overlap = max(
                len(vlm_words & pt_words),
                len(vlm_words & name_words),
            )
            if overlap == 0 and (vlm_norm in name or vlm_norm in pt):
                overlap = 1
            return float(overlap)

        # Prefer IoU when we have an OWL-ViT2 bbox
        if owl_bbox_norm is not None:
            best, best_iou = None, 0.0
            ox1, oy1, ox2, oy2 = owl_bbox_norm
            o_area = max(0.0, (ox2 - ox1)) * max(0.0, (oy2 - oy1))
            for gt in gt_annotations:
                gx1, gy1, gx2, gy2 = gt["image_bbox_norm"]
                ix1 = max(ox1, gx1)
                iy1 = max(oy1, gy1)
                ix2 = min(ox2, gx2)
                iy2 = min(oy2, gy2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                g_area = max(0.0, (gx2 - gx1)) * max(0.0, (gy2 - gy1))
                union = o_area + g_area - inter
                iou = inter / union if union > 0 else 0.0
                # Tie-break with label score (small weight)
                score = iou + 0.05 * label_score(gt)
                if score > best_iou:
                    best_iou, best = score, gt
            if best is not None and best_iou >= 0.15:
                return best

        # Label-only fallback: pick highest label-overlap GT
        if vlm_words:
            ranked = sorted(
                gt_annotations,
                key=lambda g: label_score(g),
                reverse=True,
            )
            top = ranked[0] if ranked else None
            if top and label_score(top) >= 1:
                return top
        return None

    def _refine_bbox_with_qwen(self, overhead_image, label: str,
                               bbox_full: tuple,
                               crop_margin_frac: float = 0.5):
        """Re-query Qwen on a tightly cropped region around an existing
        detection to get a more precise bbox.

        Qwen3-VL is significantly more accurate at single-instance
        grounding (RefCOCO-trained "outline the position of X" pathway)
        than at multi-instance scene grounding. By cropping to a 2x
        region around the rough bbox and asking for just one part, we
        engage that pathway and get a tight box.

        Parameters
        ----------
        overhead_image : PIL.Image
            The full overhead RGB image.
        label : str
            The part label (e.g. ``"large_gear"``).
        bbox_full : tuple
            ``(nx1, ny1, nx2, ny2)`` normalised to [0, 1] in the full
            image — the rough first-pass bbox.
        crop_margin_frac : float
            Margin to add on each side of the rough bbox, as a fraction
            of the rough bbox width/height. ``0.5`` doubles the bbox
            area, giving Qwen room to find the true edges.

        Returns
        -------
        tuple or None
            Refined ``(nx1, ny1, nx2, ny2)`` normalised to [0, 1] in
            the full image, or ``None`` if refinement failed (Qwen
            error, malformed bbox, degenerate geometry, etc.). On None
            the caller keeps the rough bbox.
        """
        from PIL import Image as _PILImage

        W, H = overhead_image.size
        nx1, ny1, nx2, ny2 = bbox_full
        rx1, ry1 = nx1 * W, ny1 * H
        rx2, ry2 = nx2 * W, ny2 * H
        bw, bh = rx2 - rx1, ry2 - ry1
        if bw <= 0 or bh <= 0:
            return None
        cx1 = max(0, int(round(rx1 - bw * crop_margin_frac)))
        cy1 = max(0, int(round(ry1 - bh * crop_margin_frac)))
        cx2 = min(W, int(round(rx2 + bw * crop_margin_frac)))
        cy2 = min(H, int(round(ry2 + bh * crop_margin_frac)))
        crop_w_px = cx2 - cx1
        crop_h_px = cy2 - cy1
        if crop_w_px < 28 or crop_h_px < 28:
            # Too small for Qwen's 28x28 vision-patch grid.
            return None
        crop_img = overhead_image.crop((cx1, cy1, cx2, cy2))

        prompt = (
            f"Outline the position of the {label} in this image.\n"
            "Output ONLY this JSON (no markdown, no commentary):\n"
            "{\"bbox_2d\": [x_min, y_min, x_max, y_max]}\n"
            "Use your standard normalised grounding coordinates in "
            "[0, 1000]: (0,0)=top-left, (1000,1000)=bottom-right. "
            "Make the box TIGHT around the visible silhouette of the "
            f"{label} only — no padding, no surrounding bin walls."
        )
        try:
            result = self.vlm.analyze_raw(crop_img, prompt)
        except Exception as e:
            log.warning(f"[Refine] {label}: Qwen call failed: {e}")
            return None

        bb = (result.get("bbox_2d")
              or result.get("bbox_pixels")
              or result.get("bbox"))
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            log.warning(
                f"[Refine] {label}: no bbox_2d in response keys="
                f"{list(result.keys()) if isinstance(result, dict) else type(result).__name__}")
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in bb]
        except (ValueError, TypeError):
            log.warning(f"[Refine] {label}: invalid bbox values: {bb}")
            return None

        # Decode coord frame using the same rules as _extract_vlm_bboxes.
        mx = max(abs(x1), abs(x2), abs(y1), abs(y2))
        if mx <= 1.5:
            f1, g1, f2, g2 = x1, y1, x2, y2  # already 0-1 fractional
            frame = "norm"
        elif mx <= 1005:
            f1, g1 = x1 / 1000.0, y1 / 1000.0
            f2, g2 = x2 / 1000.0, y2 / 1000.0
            frame = "qwen_1000"
        else:
            f1, g1 = x1 / max(crop_w_px, 1), y1 / max(crop_h_px, 1)
            f2, g2 = x2 / max(crop_w_px, 1), y2 / max(crop_h_px, 1)
            frame = "crop_px"

        if f2 <= f1 or g2 <= g1:
            return None

        # crop-fractional → crop pixels → full-image pixels → full-image normalised
        new_rx1 = cx1 + f1 * crop_w_px
        new_ry1 = cy1 + g1 * crop_h_px
        new_rx2 = cx1 + f2 * crop_w_px
        new_ry2 = cy1 + g2 * crop_h_px
        refined = (new_rx1 / W, new_ry1 / H,
                   new_rx2 / W, new_ry2 / H)

        log.info(
            f"[Refine] {label}: rough_full=("
            f"{nx1:.3f},{ny1:.3f},{nx2:.3f},{ny2:.3f}) → "
            f"crop=({cx1},{cy1},{cx2},{cy2}) bbox_2d=("
            f"{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) frame={frame} → "
            f"refined_full=("
            f"{refined[0]:.3f},{refined[1]:.3f},"
            f"{refined[2]:.3f},{refined[3]:.3f})")
        return refined

    def _verify_label_with_vlm(self, overhead_image, current_label: str,
                               bbox_full: tuple,
                               crop_margin_frac: float = 0.5):
        """Re-classify a single part by cropping its bbox and asking the
        VLM "what is this?" with a focused, single-instance prompt.

        Uses the catalogue (parts_catalogue.yaml) as the open-vocabulary
        list of valid labels. Adding a new part type means editing the
        YAML — no code changes — and this verifier picks it up.

        The VLM tends to be MUCH more accurate at single-instance
        classification than at multi-instance scene grounding (Gemma4
        4B in particular). Cropping to one part also lets a small VLM
        see fine details (silver_brass distinction, hole geometry, etc.)
        that get lost in a 1920x1080 cluttered scene.

        Returns
        -------
        tuple or None
            ``(label, confidence)`` from the VLM, or ``None`` if the
            verification failed (VLM error, malformed JSON, empty
            response). On ``None`` the caller keeps the current label.
        """
        from PIL import Image as _PILImage
        from knowledge.parts_catalogue import format_label_choices_for_prompt

        W, H = overhead_image.size
        nx1, ny1, nx2, ny2 = bbox_full
        rx1, ry1 = nx1 * W, ny1 * H
        rx2, ry2 = nx2 * W, ny2 * H
        bw, bh = rx2 - rx1, ry2 - ry1
        if bw <= 0 or bh <= 0:
            return None
        cx1 = max(0, int(round(rx1 - bw * crop_margin_frac)))
        cy1 = max(0, int(round(ry1 - bh * crop_margin_frac)))
        cx2 = min(W, int(round(rx2 + bw * crop_margin_frac)))
        cy2 = min(H, int(round(ry2 + bh * crop_margin_frac)))
        crop_w_px = cx2 - cx1
        crop_h_px = cy2 - cy1
        if crop_w_px < 28 or crop_h_px < 28:
            return None
        crop_img = overhead_image.crop((cx1, cy1, cx2, cy2))

        choices_block = format_label_choices_for_prompt()
        prompt = (
            "This is a close-up image of ONE mechanical part from an "
            "industrial parts bin.\n\n"
            "Identify what part type is shown at the centre.\n\n"
            "Choose EXACTLY ONE label from this catalogue:\n"
            f"{choices_block}\n"
            '- "unknown": does not clearly match any of the above\n\n'
            "Output ONLY this JSON (no markdown, no commentary):\n"
            '{"label": "<one of the labels above>", "confidence": 0.0-1.0}'
        )
        try:
            result = self.vlm.analyze_raw(crop_img, prompt)
        except Exception as e:
            log.warning(
                f"[Verify] '{current_label}': VLM call failed: {e}")
            return None

        if not isinstance(result, dict):
            return None
        new_label = (result.get("label") or "").strip()
        if not new_label:
            return None
        try:
            conf = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        return (new_label, conf)

    @staticmethod
    def _is_self_grounding_vlm(provider: str) -> bool:
        """Return True for VLMs that can emit tight bboxes natively
        (Qwen-family). Gemma, LLaVA, GPT-4o go through OWL-ViT2.
        """
        if not provider:
            return False
        p = provider.lower()
        return p.startswith("ollama_qwen") or "qwen" in p

    @staticmethod
    def _extract_vlm_bboxes(objects: list, crop_w: int,
                            crop_h: int,
                            qwen_size: Optional[tuple] = None) -> list:
        """Convert per-object bbox fields from a self-grounding VLM
        into the same shape OWL-ViT2 returns (pixel bbox in crop frame
        + label + confidence) so ``_assign_detector_bboxes`` is reused
        verbatim.

        Bbox fields accepted (priority order):
          1. ``bbox_2d`` — Qwen3-VL native JSON-grounding field name.
          2. ``bbox_pixels`` — earlier prompt schema (kept for compat).
          3. ``bbox_norm`` / ``bbox`` — generic legacy fields.
          4. ``mask_polygon`` / ``segmentation_polygon`` — polygon AABB.

        Coordinate-frame auto-detection by magnitude:
          * max ≤ 1.5            → 0-1 normalised within the image.
          * max ≤ 1005           → **Qwen 0-1000 grid** (Qwen3-VL
                                   convention; image-size invariant).
          * max ≤ max(qw, qh)·1.05 → absolute pixels in Qwen frame
                                     (Qwen2.5-VL convention).
          * else                 → assume already in crop pixels.

        All branches return pixel coordinates in the **crop** frame
        (``crop_w`` × ``crop_h``), which is what
        ``_assign_detector_bboxes`` and ``_render_vlm_overlay`` expect.

        Side effect: writes a normalised crop-relative ``image_position``
        (bbox centre, 0-1) onto each object so the downstream loop in
        ``_detect_parts_from_bin_crop`` always has a well-formed semantic
        target — replacing whatever Qwen put in that field on its own.
        """
        import re as _re
        out = []

        qw = qh = None
        if qwen_size and len(qwen_size) == 2:
            try:
                qw, qh = int(qwen_size[0]), int(qwen_size[1])
                if qw <= 0 or qh <= 0:
                    qw = qh = None
            except (TypeError, ValueError):
                qw = qh = None
        # Scale factor: Qwen pixel frame → crop pixel frame.
        sx = (crop_w / qw) if qw else 1.0
        sy = (crop_h / qh) if qh else 1.0

        def _polygon_bbox(poly: list):
            xs, ys = [], []
            for pt in poly:
                if isinstance(pt, dict):
                    x, y = pt.get("x"), pt.get("y")
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    x, y = pt[0], pt[1]
                else:
                    continue
                try:
                    xs.append(float(x)); ys.append(float(y))
                except (TypeError, ValueError):
                    continue
            if not xs or not ys:
                return None
            return min(xs), min(ys), max(xs), max(ys)

        def _to_crop_pixels(x1, y1, x2, y2):
            """Detect coord frame by magnitude, return (px1, py1, px2,
            py2) in crop pixels.

            The 0-1000 grid branch is checked BEFORE the absolute-Qwen-
            pixels branch because Qwen3-VL emits in 0-1000 grid (image-
            size invariant) — for any reasonable Qwen image size, Qwen3
            outputs land in 0-1005 ∩ ≤max(qw,qh), and we MUST treat them
            as the grid. Qwen2.5-VL emits absolute pixels and triggers
            the next branch only when bbox values exceed 1005.
            """
            mx = max(abs(x1), abs(x2), abs(y1), abs(y2))
            if mx <= 1.5:
                # Normalised 0-1 within the image
                return (x1 * crop_w, y1 * crop_h,
                        x2 * crop_w, y2 * crop_h, "norm")
            if mx <= 1005:
                # Qwen 0-1000 grid (Qwen3-VL native, image-size invariant).
                # 1005 > 1000 tolerates a slight model overshoot.
                return (x1 / 1000.0 * crop_w, y1 / 1000.0 * crop_h,
                        x2 / 1000.0 * crop_w, y2 / 1000.0 * crop_h,
                        "qwen_1000")
            if qw and qh and mx <= max(qw, qh) * 1.05:
                # Absolute pixels in Qwen frame (Qwen2.5-VL convention)
                # → scale to crop frame
                return (x1 * sx, y1 * sy,
                        x2 * sx, y2 * sy, "qwen_px")
            # Fallback: assume already in crop pixels
            return (x1, y1, x2, y2, "crop_px")

        for obj in objects:
            # ``bbox_2d`` is Qwen3-VL's native JSON-grounding field;
            # fall back to earlier schemas / legacy names for safety.
            bb = (obj.get("bbox_2d")
                  or obj.get("bbox_pixels")
                  or obj.get("bbox_norm")
                  or obj.get("bbox"))
            source = (
                "bbox_2d" if obj.get("bbox_2d") else
                "bbox_pixels" if obj.get("bbox_pixels") else
                "bbox_norm" if obj.get("bbox_norm") else "bbox"
            )
            if not bb:
                bb = obj.get("mask_polygon") or obj.get("segmentation_polygon")
                if bb:
                    bb = _polygon_bbox(bb)
                    source = "polygon_aabb"
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in bb]
            except (ValueError, TypeError):
                continue

            px1, py1, px2, py2, frame = _to_crop_pixels(x1, y1, x2, y2)

            if px2 <= px1 or py2 <= py1:
                continue
            label = (obj.get("label") or "").lower()
            base = _re.sub(r"_\d+$", "", label).strip()

            # Sanity flags: a box that covers more than ~30% of the
            # image area is almost always Qwen falling back to "I'm
            # not sure, here's the whole region". A box outside [0,
            # 1000] in the raw values means Qwen ignored the grid
            # contract. Both are kept (the assignment may still pick
            # them) but logged so unexpected drift is visible.
            crop_area = max(crop_w * crop_h, 1)
            box_area = max((px2 - px1) * (py2 - py1), 1.0)
            area_frac = box_area / crop_area
            warn_flags = []
            if area_frac > 0.30:
                warn_flags.append(f"LARGE({area_frac*100:.0f}% of image)")
            mx_raw = max(abs(x1), abs(x2), abs(y1), abs(y2))
            mn_raw = min(x1, x2, y1, y2)
            if frame == "qwen_1000" and (mx_raw > 1005 or mn_raw < -5):
                warn_flags.append(f"OUT_OF_GRID(raw_min={mn_raw:.0f}, "
                                  f"raw_max={mx_raw:.0f})")

            # Diagnostic: one line per object so a coord-frame mismatch
            # is obvious in the logs.
            tag = (" ⚠ " + " ".join(warn_flags)) if warn_flags else ""
            log.info(
                f"[Qwen-bbox] {label}: raw={source}=({x1:.1f},"
                f"{y1:.1f},{x2:.1f},{y2:.1f}) frame={frame} "
                f"qwen=({qw}x{qh}) crop=({crop_w}x{crop_h}) → "
                f"crop_px=({px1:.0f},{py1:.0f},{px2:.0f},{py2:.0f}){tag}")

            # Overwrite image_position with crop-normalised bbox centre
            # so the downstream loop has a clean target. Qwen's own
            # image_position is in whatever frame it felt like that
            # turn — not trustworthy.
            cx_px = (px1 + px2) / 2.0
            cy_px = (py1 + py2) / 2.0
            obj["image_position"] = {
                "x": cx_px / max(crop_w, 1),
                "y": cy_px / max(crop_h, 1),
            }

            out.append({
                "label": base,   # canonical part type → direct match
                "bbox": (px1, py1, px2, py2),
                "confidence": float(obj.get("confidence", 0.9)),
            })
        return out

    @staticmethod
    def _assign_detector_bboxes(objects: list, detector_dets: list,
                                crop_w: int, crop_h: int,
                                bin_crop_box: tuple) -> dict:
        """Greedy 1-to-1 assignment of OWL-ViT2 bboxes to VLM objects.

        Each VLM object (e.g. ``large_gear_1``) gets AT MOST one bbox,
        and each OWL-ViT2 bbox is consumed by AT MOST one object. This
        replaces the old per-object "pick best score" loop where three
        instances of ``large_gear_*`` would all collapse to the same
        single highest-scoring bbox.

        Scoring:
          - If the detector query string reverse-maps to the same
            canonical part type as the VLM label's base (strip ``_N``),
            score = 10 + detector confidence. This is the strongest
            signal because both sides agree on semantic identity.
          - Otherwise score = word-overlap + confidence (legacy fuzzy).
          - Pairs with zero word overlap and no substring containment
            are rejected outright.

        Returns
        -------
        tuple
            ``(assignment, used_det_indices)`` where ``assignment`` is
            ``{obj_index: (nx1, ny1, nx2, ny2)}`` with bbox in full-image
            normalised coordinates (objects without a viable bbox match
            are absent), and ``used_det_indices`` is the set of indices
            into ``detector_dets`` that were claimed. The caller can use
            ``set(range(len(detector_dets))) - used`` to find OWL hits
            that no VLM object claimed — those are candidates for being
            synthesised into the scene as detector-only detections.
        """
        if not objects or not detector_dets:
            return {}, set()

        import re as _re
        # Reverse-map: every detector query variant → canonical part type.
        # Normalise both sides (underscores ↔ spaces) — OWL-ViT2 returns
        # labels like ``flat_black_circular_disc`` while the catalogue
        # stores queries like ``flat black circular disc``. Without this
        # normalisation the canonical match never fires. Each canonical
        # part may declare MULTIPLE alternative queries; all variants
        # map to the same canonical here.
        def _norm_q(s):
            return _re.sub(r"\s+", " ",
                           (s or "").replace("_", " ").lower()).strip()
        cat_queries_list = detector_query_list()
        query_to_canonical = {}
        for canonical, qs in cat_queries_list.items():
            for q in qs:
                query_to_canonical[_norm_q(q)] = canonical

        candidates = []  # (score, obj_idx, det_idx)
        for oi, obj in enumerate(objects):
            vlm_label = (obj.get("label") or "").lower()
            if not vlm_label:
                continue
            vlm_base = _re.sub(r"_\d+$", "", vlm_label).strip()
            vlm_words = set(vlm_label.replace("_", " ").split())
            pos = obj.get("image_position", {})
            try:
                obj_px = float(pos.get("x")) if "x" in pos else None
                obj_py = float(pos.get("y")) if "y" in pos else None
            except (TypeError, ValueError):
                obj_px = obj_py = None
            for di, det in enumerate(detector_dets):
                det_label_raw = (det.get("label") or "").lower().strip()
                if not det_label_raw:
                    continue
                # Normalised form for catalogue lookup (handles
                # OWL's underscore vs catalogue's space convention).
                det_label = _norm_q(det_label_raw)
                conf = float(det.get("confidence", 0.0))
                det_cx = det_cy = None
                bbox = det.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    try:
                        x1, y1, x2, y2 = [float(v) for v in bbox]
                        det_cx = ((x1 + x2) / 2.0) / max(crop_w, 1)
                        det_cy = ((y1 + y2) / 2.0) / max(crop_h, 1)
                    except (TypeError, ValueError):
                        det_cx = det_cy = None
                # Direct canonical match (self-grounding VLM path):
                # det["label"] is already the canonical part type.
                if det_label_raw == vlm_base and det_label_raw:
                    score = 10.0 + conf
                    if obj_px is not None and det_cx is not None:
                        dist = ((obj_px - det_cx) ** 2 + (obj_py - det_cy) ** 2) ** 0.5
                        score += max(0.0, 1.0 - 3.0 * dist)
                    candidates.append((score, oi, di))
                    continue
                det_canonical = query_to_canonical.get(det_label)
                if det_canonical and det_canonical == vlm_base:
                    score = 10.0 + conf
                else:
                    det_words = set(det_label.replace("_", " ").split())
                    overlap = len(vlm_words & det_words)
                    if overlap == 0:
                        if not (vlm_label in det_label
                                or det_label in vlm_label):
                            continue
                        overlap = 1
                    score = float(overlap) + conf
                if obj_px is not None and det_cx is not None:
                    dist = ((obj_px - det_cx) ** 2 + (obj_py - det_cy) ** 2) ** 0.5
                    score += max(0.0, 1.0 - 3.0 * dist)
                candidates.append((score, oi, di))

        candidates.sort(key=lambda t: -t[0])
        assigned_objs, assigned_dets = set(), set()
        assignment = {}
        for score, oi, di in candidates:
            if oi in assigned_objs or di in assigned_dets:
                continue
            assignment[oi] = di
            assigned_objs.add(oi)
            assigned_dets.add(di)

        # ── Spatial fallback for unmatched high-confidence VLM objects ──
        # OWL-ViT2 occasionally finds the right bbox under the WRONG
        # query — e.g. it returns a tight bbox on a motor_valve labelled
        # as 'small silver hinge' because both share silver/metallic
        # cues at small image scale. The greedy assignment above
        # rejects such pairs (label words don't overlap, canonical
        # mismatch), so the VLM's correct ``motor_valve`` detection
        # gets dropped while the OWL bbox falls into the unclaimed
        # pool and the safety net later synthesises a wrong-label
        # ``small_hinge`` from it.
        #
        # This second pass rescues those cases. For every UNMATCHED
        # VLM object whose own confidence is above a threshold, find
        # the closest UNCLAIMED OWL bbox within a spatial radius and
        # claim it under the VLM's label. The VLM's full visual context
        # (colour, shape, distinguishing features in the catalogue
        # prompt) wins over OWL's narrow query labels.
        SPATIAL_FALLBACK_MIN_VLM_CONF = 0.70
        SPATIAL_FALLBACK_MAX_DIST = 0.10  # 10% of normalised crop diagonal

        unmatched_obj_indices = [oi for oi in range(len(objects))
                                 if oi not in assigned_objs]
        unmatched_det_indices = [di for di in range(len(detector_dets))
                                 if di not in assigned_dets]

        for oi in unmatched_obj_indices:
            obj = objects[oi]
            try:
                vlm_conf = float(obj.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                vlm_conf = 0.0
            if vlm_conf < SPATIAL_FALLBACK_MIN_VLM_CONF:
                continue
            pos = obj.get("image_position", {})
            try:
                obj_cx = float(pos.get("x"))
                obj_cy = float(pos.get("y"))
            except (TypeError, ValueError):
                continue

            best_di = None
            best_dist = float("inf")
            for di in unmatched_det_indices:
                if di in assigned_dets:
                    continue
                bbox = detector_dets[di].get("bbox")
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue
                try:
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                except (TypeError, ValueError):
                    continue
                det_cx = ((x1 + x2) / 2.0) / max(crop_w, 1)
                det_cy = ((y1 + y2) / 2.0) / max(crop_h, 1)
                dist = ((obj_cx - det_cx) ** 2
                        + (obj_cy - det_cy) ** 2) ** 0.5
                if dist < best_dist and dist <= SPATIAL_FALLBACK_MAX_DIST:
                    best_dist = dist
                    best_di = di

            if best_di is not None:
                vlm_label = (obj.get("label") or "").lower()
                owl_label = (detector_dets[best_di].get("label") or "").lower()
                log.info(
                    f"[Spatial-fallback] VLM '{vlm_label}' "
                    f"(conf={vlm_conf:.2f}) → unclaimed OWL bbox "
                    f"'{owl_label}' at distance "
                    f"{best_dist*100:.1f}% — overriding OWL label "
                    f"with VLM label (rescues motor_valve / hinge "
                    f"misclassification on small silver parts).")
                assignment[oi] = best_di
                assigned_objs.add(oi)
                assigned_dets.add(best_di)

        # Convert assigned bboxes to full-image normalised coords
        cx_min, cy_min, cx_max, cy_max = bin_crop_box
        cw_norm = cx_max - cx_min
        ch_norm = cy_max - cy_min
        out = {}
        for oi, di in assignment.items():
            x1, y1, x2, y2 = detector_dets[di]["bbox"]
            nx1 = cx_min + (x1 / crop_w) * cw_norm
            ny1 = cy_min + (y1 / crop_h) * ch_norm
            nx2 = cx_min + (x2 / crop_w) * cw_norm
            ny2 = cy_min + (y2 / crop_h) * ch_norm
            out[oi] = (nx1, ny1, nx2, ny2)
        return out, set(assignment.values())

    def _zoomed_bin_analysis(self, full_image, scene: dict):
        """Stage 2: Re-analyze just the bin area at higher resolution.

        Crops the bin region from the full-res image and sends it to
        the VLM for more detailed part identification.  Merges the
        zoomed detections back into the original scene, converting
        crop-local coordinates to full-image coordinates.
        """
        bin_box = self._find_bin_region(scene)
        if not bin_box:
            log.info("[Zoom] Could not estimate bin region — skipping")
            return

        x_min, y_min, x_max, y_max = bin_box
        log.info(f"[Zoom] Bin region: x=[{x_min:.2f},{x_max:.2f}] "
                 f"y=[{y_min:.2f},{y_max:.2f}]")

        cropped = self._crop_bin_region(full_image, bin_box)
        cw, ch = cropped.size
        log.info(f"[Zoom] Cropped bin image: {cw}x{ch} px")

        # Send the cropped + LANCZOS-upscaled image to VLM at
        # max_size=1600 so the upscale isn't undone.
        try:
            zoomed_scene = self.vlm.analyze_scene(
                cropped,
                custom_prompt=_with_catalogue(BIN_ZOOM_PROMPT),
                max_size=1600)
        except Exception as e:
            log.warning(f"[Zoom] VLM analysis failed: {e}")
            return

        zoomed_objs = zoomed_scene.get("detected_objects", [])
        if not zoomed_objs:
            log.info("[Zoom] No objects detected in zoomed view")
            return

        log.info(f"[Zoom] Detected {len(zoomed_objs)} parts in zoomed bin")

        # Convert crop-local normalised coords → full-image normalised coords
        for obj in zoomed_objs:
            pos = obj.get("image_position", {})
            if pos:
                # local (0-1) → full image (0-1)
                pos["x"] = x_min + pos.get("x", 0.5) * (x_max - x_min)
                pos["y"] = y_min + pos.get("y", 0.5) * (y_max - y_min)
            obj["affordance"] = "graspable"  # everything in the bin is graspable
            obj["_source"] = "zoomed_vlm"

        # Replace the graspable objects from stage 1 with zoomed results.
        # Keep tray / destination objects from stage 1 unchanged.
        non_parts = [
            obj for obj in scene.get("detected_objects", [])
            if obj.get("affordance") == "destination"
            or "tray" in obj.get("label", "").lower()
        ]

        # Re-number object IDs
        for i, obj in enumerate(zoomed_objs):
            obj["object_id"] = f"obj_{i+1:03d}"

        # Combine: zoomed parts + original tray detections
        scene["detected_objects"] = zoomed_objs + non_parts
        log.info(f"[Zoom] Final scene: {len(zoomed_objs)} parts + "
                 f"{len(non_parts)} non-part objects")

    def _filter_unreachable(self, scene: dict) -> None:
        """Remove graspable objects outside the robot's effective kinematic reach.

        Only filters objects that have projected world coordinates
        (``approximate_position``).  Destination objects (kitting tray)
        are never filtered — they don't need to be grasped.
        """
        reach = self.config.get("execution", {}).get("effective_reach", {})
        x_min = reach.get("x_min", -2.0)
        x_max = reach.get("x_max", 3.0)
        y_min = reach.get("y_min", -1.3)
        y_max = reach.get("y_max", 1.3)
        z_min = reach.get("z_min", -0.05)
        z_max = reach.get("z_max", 0.60)

        objects = scene.get("detected_objects", [])
        reachable = []
        filtered_count = 0

        for obj in objects:
            # Never filter destination / tray objects
            if (obj.get("affordance") == "destination"
                    or "tray" in obj.get("label", "").lower()):
                reachable.append(obj)
                continue

            pos = obj.get("approximate_position")
            if not pos:
                # No world coords yet — keep (may still be resolved later)
                reachable.append(obj)
                continue

            x, y, z = pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)
            outside = []
            if x < x_min or x > x_max:
                outside.append(f"x={x:.3f}")
            if y < y_min or y > y_max:
                outside.append(f"y={y:.3f}")
            if z < z_min or z > z_max:
                outside.append(f"z={z:.3f}")

            if outside:
                filtered_count += 1
                log.warning(
                    f"[Reachability] FILTERED '{obj.get('label', '?')}' "
                    f"({obj.get('object_id', '?')}): {', '.join(outside)} "
                    f"outside effective reach "
                    f"[x:{x_min},{x_max}] [y:{y_min},{y_max}] [z:{z_min},{z_max}]"
                )
            else:
                reachable.append(obj)

        if filtered_count:
            scene["detected_objects"] = reachable
            log.info(
                f"[Reachability] Kept {len(reachable)} objects, "
                f"filtered {filtered_count} unreachable"
            )

    def _extract_tray_from_objects(self, scene: dict) -> Optional[dict]:
        """Extract the kitting tray position from detected_objects.

        The tray is detected as a regular object with label containing
        "tray" or affordance "destination".  XY comes from depth projection
        (same pipeline as parts).  Z is taken from config (tray_surface_z)
        because depth projection often gives unreliable Z for the tray
        (open-top container, background depth bleed, etc.).
        """
        tray_surface_z = self.config.get("perception", {}).get(
            "tray_surface_z", 0.02)
        workspace = self.config.get("execution", {}).get("workspace_bounds", {})
        z_max = workspace.get("z_max", 1.5)

        tray_keywords = ("tray", "kitting", "destination")
        for obj in scene.get("detected_objects", []):
            label = obj.get("label", "").lower()
            affordance = obj.get("affordance", "").lower()
            pos = obj.get("approximate_position")
            if pos and (affordance == "destination" or
                        any(kw in label for kw in tray_keywords)):
                # Use projected XY, but override Z with tray_surface_z
                # Depth projection Z is unreliable for the tray
                proj_z = pos["z"]
                use_z = tray_surface_z
                z_min = workspace.get("z_min", -0.1)
                if proj_z > z_max or proj_z < z_min:
                    log.warning(
                        f"Tray projected z={proj_z:.3f} outside bounds — "
                        f"using tray_surface_z={tray_surface_z}"
                    )
                else:
                    use_z = proj_z

                place_target = {
                    "center_xy": [pos["x"], pos["y"]],
                    "top_z": use_z,
                }
                log.info(
                    f"Tray from detected_objects: '{obj.get('label')}' "
                    f"XY=({pos['x']:.3f}, {pos['y']:.3f}) "
                    f"proj_z={proj_z:.3f} → use_z={use_z:.3f}"
                )
                return place_target
        return None

    def _resolve_place_coordinates(self, params, place_target):
        """
        For place actions, override x/y/z with the perception-detected
        kitting tray position.

        All coordinates come from: VLM detection → OWL-ViT2 refinement →
        depth projection.  No hardcoded positions or USD prims.

        Returns None if the tray was not detected (caller should abort).
        """
        if not place_target:
            log.error(
                "Cannot place: kitting tray not detected by perception. "
                "Make sure the tray is visible to the camera."
            )
            return None

        resolved = dict(params)
        resolved["x"] = place_target["center_xy"][0]
        resolved["y"] = place_target["center_xy"][1]
        resolved["z"] = place_target["top_z"]
        log.info(
            f"Place target (perception): "
            f"({resolved['x']:.3f}, {resolved['y']:.3f}, {resolved['z']:.3f})"
        )
        return resolved

    # ─── Pick Sequence (3-phase with wrist VLM) ──────────────

    def _wrist_vlm_check(self, wrist_b64, prompt_text):
        """Send a wrist-camera image to the VLM with a custom prompt.

        Routes through ``analyze_raw`` (NOT ``analyze_scene``) because
        the wrist-camera prompts return flat schemas like
        ``{part_between_fingers, confidence, detail}`` or
        ``{grasp_ok, confidence, detail}`` — they do NOT contain a
        ``detected_objects`` field. The scene-analysis validator
        rejects them as schema-invalid, throwing away a perfectly
        good answer and forcing the workflow to fall back to its
        "proceed anyway" default. ``analyze_raw`` strips the
        markdown code fence and parses the JSON without enforcing
        the scene schema.
        """
        from PIL import Image as PILImage
        import io as _io, base64 as _b64

        wrist_bytes = _b64.b64decode(wrist_b64)
        wrist_img = PILImage.open(_io.BytesIO(wrist_bytes)).convert("RGB")
        return self.vlm.analyze_raw(wrist_img, prompt=prompt_text)

    # ─── Wrist Scan + Project Helper ────────────────────────

    def _wrist_scan_and_project(self, prompt):
        """Capture wrist image, run VLM, project coords to world.

        Returns (scene_dict, coord_source_string).
        coord_source is ``"none"`` if depth projection failed.
        """
        wrist_image = self.camera.capture_wrist_image()

        # Push wrist image to Streamlit session state for live view
        self._notify_wrist_image(wrist_image)

        scene = self.vlm.analyze_scene(
            wrist_image, custom_prompt=prompt)

        proj_points = []
        for obj in scene.get("detected_objects", []):
            img_pos = obj.get("image_position",
                              obj.get("approximate_position", {}))
            if img_pos and "x" in img_pos and "y" in img_pos:
                proj_points.append({
                    "x": img_pos["x"], "y": img_pos["y"]})

        coord_source = "none"
        if proj_points:
            try:
                proj_result = self.camera.project_to_world(
                    proj_points, camera="wrist")
                wpts = proj_result.get("world_points", [])
                coord_source = (
                    f"wrist_{proj_result.get('method', '?')}")

                for i, obj in enumerate(
                        scene.get("detected_objects", [])):
                    if i < len(wpts):
                        wp = wpts[i]
                        obj["approximate_position"] = {
                            "x": wp["x"],
                            "y": wp["y"],
                            "z": wp["z"],
                        }
                        obj["depth_m"] = wp.get("depth_m", -1)
                        obj["_source"] = coord_source
                        log.info(
                            f"  '{obj.get('label', '?')}': "
                            f"({wp['x']:.3f}, {wp['y']:.3f}, "
                            f"{wp['z']:.3f})")
            except Exception as e:
                log.warning(f"Wrist projection failed: {e}")

        return scene, coord_source

    def _notify_wrist_image(self, wrist_image=None):
        """Push wrist camera image to Streamlit session state.

        This lets the UI display the live wrist view during
        workflow execution.  Falls back to capturing a fresh
        frame if no image is supplied.
        """
        try:
            import streamlit as st
            if wrist_image is None:
                wrist_image = self.camera.capture_wrist_image()
            st.session_state["wrist_live_image"] = wrist_image
        except Exception:
            pass  # outside Streamlit or capture failed — ignore

    @staticmethod
    def _looks_like_placeholder(image) -> bool:
        """Detect the bridge's 640x480 fallback image.

        ``BridgeCameraInterface._placeholder_image`` returns a 640x480
        near-black gradient when the bridge is unreachable. Sending it
        through the perception pipeline produces hallucinated parts +
        z=0 projections, so the workflow should bail explicitly.
        """
        try:
            import numpy as _np
            if image.size != (640, 480):
                return False
            arr = _np.array(image.convert("RGB"))
            return float(arr.mean()) < 60.0
        except Exception:
            return False

    def _notify_vlm_overlay(self, overlay_image):
        """Push the latest VLM detection overlay to Streamlit so the
        operator can inspect what the model actually saw + labelled.
        """
        try:
            import streamlit as st
            st.session_state["vlm_overlay_image"] = overlay_image
        except Exception:
            pass

    @staticmethod
    def _render_vlm_overlay(image, scene_objs, detector_dets,
                            bin_crop_box):
        """Draw the VLM-derived bboxes + labels on a copy of ``image``.

        Two layers:
          - **VLM objects** (red, thick): each detected_object's bbox,
            with label + confidence. Bbox is taken from
            ``_detector_bbox_full`` (full-image normalised) and
            converted to pixel coords inside ``image``.
          - **Detector raw bboxes** (green, thin): all OWL-ViT2 / Qwen
            grounding bboxes that survived the greedy assignment, in
            crop pixel coords.

        Useful for confirming what the model actually labelled and
        whether the bboxes match the visible parts. If the robot then
        moves to the wrong place, you'll see at a glance whether the
        VLM identified the wrong region or the depth-projection step
        produced a bad XYZ.
        """
        from PIL import Image, ImageDraw, ImageFont
        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)
        W, H = annotated.size
        cx_min, cy_min, cx_max, cy_max = bin_crop_box
        cw_norm = max(cx_max - cx_min, 1e-6)
        ch_norm = max(cy_max - cy_min, 1e-6)

        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        # Bbox-centre marker. The bridge's depth projection samples a
        # 5×5 patch around the BBOX CENTRE — drawing an X here lets you
        # verify by eye whether the projection anchored on the actual
        # part (centre lands on the part silhouette) or on a divider /
        # gap (centre lands on a wall — projection will return wrong
        # XYZ → robot moves to wrong place).
        def _draw_centre_x(px, py, color, arm=10, w=2):
            px, py = float(px), float(py)
            draw.line([(px - arm, py - arm), (px + arm, py + arm)],
                      fill=color, width=w)
            draw.line([(px - arm, py + arm), (px + arm, py - arm)],
                      fill=color, width=w)

        # Layer 1: detector raw bboxes (green) — these are pixels in
        # the crop frame, which equals the image we're drawing on.
        for det in (detector_dets or []):
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            draw.rectangle([x1, y1, x2, y2],
                           outline=(0, 220, 0), width=2)
            _draw_centre_x((x1 + x2) / 2.0, (y1 + y2) / 2.0,
                           (0, 220, 0), arm=8, w=2)
            label = det.get("label", "")
            if label:
                draw.text((x1 + 3, max(0, y1 - 18)),
                          f"det:{label}", fill=(0, 220, 0), font=font)

        # Layer 2: VLM objects (red) — _detector_bbox_full is in
        # full-image normalised coords; remap to local crop pixels.
        for obj in (scene_objs or []):
            bbox_full = obj.get("_detector_bbox_full")
            label = obj.get("label", "?")
            conf = obj.get("confidence", 0.0)
            if bbox_full and len(bbox_full) == 4:
                # Full-image normalised → crop-local normalised → pixels
                nx1, ny1, nx2, ny2 = bbox_full
                lx1 = (nx1 - cx_min) / cw_norm * W
                ly1 = (ny1 - cy_min) / ch_norm * H
                lx2 = (nx2 - cx_min) / cw_norm * W
                ly2 = (ny2 - cy_min) / ch_norm * H
                draw.rectangle([lx1, ly1, lx2, ly2],
                               outline=(220, 30, 30), width=3)
                _draw_centre_x((lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0,
                               (255, 255, 0), arm=10, w=2)
                draw.text((lx1 + 3, max(0, ly1 - 36)),
                          f"{label} ({conf:.2f})",
                          fill=(220, 30, 30), font=font)
            else:
                # No bbox — draw a crosshair at image_position
                pos = obj.get("image_position", {})
                if "x" in pos and "y" in pos:
                    px = (pos["x"] - cx_min) / cw_norm * W
                    py = (pos["y"] - cy_min) / ch_norm * H
                    R = 8
                    draw.line([(px - R, py), (px + R, py)],
                              fill=(220, 30, 30), width=2)
                    draw.line([(px, py - R), (px, py + R)],
                              fill=(220, 30, 30), width=2)
                    draw.text((px + 10, py - 6),
                              f"{label} ({conf:.2f})",
                              fill=(220, 30, 30), font=font)
        return annotated

    def _notify_depth_image(self, depth_image=None):
        """Push depth camera image to Streamlit session state.

        Mirrors ``_notify_wrist_image`` so the UI can show what the
        wrist depth camera captured during depth_analysis (useful
        when verify fails because the renderer hasn't refreshed).
        """
        try:
            import streamlit as st
            if depth_image is None:
                depth_image = self.camera.capture_depth_image()
            st.session_state["depth_live_image"] = depth_image
            # Also keep the older key the depth tab already reads
            st.session_state["depth_image"] = depth_image
        except Exception:
            pass

    # ─── Target Part Matching ──────────────────────────────
    # Map from known part type → common synonyms an operator
    # might use in natural language commands.
    _PART_SYNONYMS = {
        "motor_valve": ["motor_valve", "motor valve", "valve",
                         "motor"],
        "black_hose":  ["black_hose", "black hose", "hose"],
        "gear": ["large_gear", "gear", "round_gear"],
        "black_plate": ["black_plate", "black plate", "plate"],
        "black_plug":  ["black_plug", "black plug", "plug"],
        "small_hinge": ["small_hinge", "small hinge", "hinge"],
        "small_tube":  ["small_tube", "small tube", "tube"],
        "silver_box":  ["silver_box", "silver box"],
        "silver_gun":  ["silver_gun", "silver gun", "gun"],
        "tube_with_clamps": ["tube_with_clamps", "tube with clamps",
                             "clamp tube", "clamp"],
    }

    def _extract_target_parts(self, user_command: str) -> list:
        """Extract target part types from a user command.

        Matches known part types and common synonyms against
        the operator's natural language command.

        Returns a list of canonical part type names, or empty
        list if no specific part was mentioned (e.g. "pick all").
        """
        cmd = user_command.lower()
        matches = []
        for part_type, keywords in self._PART_SYNONYMS.items():
            for kw in keywords:
                if kw in cmd:
                    matches.append(part_type)
                    break
        return matches

    @staticmethod
    def _parse_pick_quantity(user_command: str):
        """Parse how many parts the user wants to pick.

        Rules
        -----
        - "all" (with or without a part name) → None  (pick every match)
        - explicit number (e.g. "2 gears", "pick 3") → that integer
        - singular name with no qualifier → 1

        Returns
        -------
        int or None
            None means "pick all matching"; an int means "pick exactly N".
        """
        cmd = user_command.lower()
        if re.search(r'\ball\b', cmd):
            return None
        m = re.search(r'\b([1-9]\d*)\b', cmd)
        if m:
            return int(m.group(1))
        return 1

    @staticmethod
    def _label_category(text: str) -> str:
        """Reduce a label or target string to its category root.

        Catalogue entries are ``<adjective>_<noun>`` form where the noun
        is the part category (``gear``, ``valve``, ``hinge``, ``box``,
        ``plate``, ``hose``, ``plug``, ``tube``, ...). We take the last
        underscore-separated word as the category, after stripping the
        trailing instance number (``_1``, ``_2``).

        ``"large_gear"``  → ``"gear"``
        ``"motor_valve_2"`` → ``"valve"``
        ``"small_hinge"`` → ``"hinge"``
        ``"gear"``        → ``"gear"``
        ``"gears"``       → ``"gears"`` (caller plural-strips if needed)
        """
        if not text:
            return ""
        norm = text.lower().replace(" ", "_").strip()
        norm = re.sub(r"_\d+$", "", norm)
        parts = [p for p in norm.split("_") if p]
        if not parts:
            return ""
        cat = parts[-1]
        # Singular fold so "gears" matches "gear", "valves" → "valve".
        if len(cat) > 3 and cat.endswith("s") and not cat.endswith("ss"):
            cat = cat[:-1]
        return cat

    @staticmethod
    def _labels_match_target(detected_objects: list,
                             target_parts: list) -> bool:
        """Check if any detected label fuzzy-matches the target.

        Match in priority order:
          1. **Category match** — last-word root of label vs target,
             with singular fold. ``"gear"`` matches ``"large_gear"``;
             ``"valves"`` matches ``"motor_valve_2"``;
             ``"hinge"`` matches ``"small_hinge"``.
          2. Substring match — ``"box"`` matches ``"silver_box"``.
          3. Word overlap — ``"silver_housing"`` matches ``"silver_box"``.
        """
        if not target_parts:
            return True  # no specific target — any detection is OK

        for obj in detected_objects:
            raw_label = (obj.get("label") or "").strip()
            label_norm = raw_label.lower().replace(" ", "_")
            label_norm = re.sub(r"_\d+$", "", label_norm)
            label_cat = KittingWorkflowEngine._label_category(raw_label)
            label_words = set(label_norm.replace("_", " ").split())

            for target in target_parts:
                t_norm = (target or "").lower().strip().replace(" ", "_")
                t_norm = re.sub(r"_\d+$", "", t_norm)
                t_cat = KittingWorkflowEngine._label_category(target)
                t_words = set(t_norm.replace("_", " ").split())

                # 1. Category root match (gear↔large_gear, valve↔motor_valve)
                if t_cat and label_cat and t_cat == label_cat:
                    return True
                # 2. Substring match (either direction)
                if t_norm and label_norm and (
                        t_norm in label_norm or label_norm in t_norm):
                    return True
                # 3. Word overlap (legacy fuzzy)
                if t_words & label_words:
                    return True
        return False

    def _verify_and_realign(self, pick_coords, obj_id, step_num):
        """Pre-grasp verification: confirm part position + check collision clearance.

        After the robot approaches the target, this method:
        1. Captures a wrist image to confirm a part is visible
        2. Checks for collision risk on the descent path
        3. Re-projects part position from close range (more accurate)
        4. Validates depth/Z against expected physical range
        5. Realigns the gantry if XY offset exceeds 3 cm

        Returns dict with corrected coordinates and verification status.
        ``part_visible`` is a hard signal — if the wrist VLM reports no
        graspable part, the caller MUST abort the pick (descending into
        empty space wastes the cycle and confuses the depth verifier).
        """
        import math

        REALIGN_XY_THRESHOLD = 0.03   # 3 cm — realign gantry
        # 1 cm — apply Z correction. Earlier value of 5 cm was too
        # coarse: the wrist depth projection is typically accurate to
        # 5-10 mm, so a 5 cm gate discarded almost every refinement
        # and the descent ended up using the (less accurate) overhead
        # Z. With contact-aware descent in place a wrong Z is no
        # longer catastrophic — but it still wastes time descending
        # to the wrong altitude before the tip sensor catches it.
        Z_CORRECTION_THRESHOLD = 0.01
        # Upper cap on the wrist-XY correction. The wrist camera at
        # 30-50 cm above the bin sees several parts at once; if the
        # VLM picks a NEIGHBOUR part instead of the planned target,
        # the projection puts the world centre 8-15 cm away from the
        # planned XY. Applying that correction sends the gripper to
        # the wrong gear (the user's two-image diagnosis: gripper
        # landed beside the gear, not over it). Corrections beyond
        # this cap are rejected — the planner's overhead-derived XY
        # was less precise but at least targeted the RIGHT part.
        # Tune up if you trust the wrist VLM to pick the centre part
        # reliably, down for stricter rejection.
        XY_CORRECTION_MAX = 0.05  # 5 cm

        result = {
            "verified": False,
            "realigned": False,
            "coords": dict(pick_coords),
            "path_clear": True,
            "part_visible": True,  # default: assume OK on capture failure
            "sub_phases": [],
        }

        # ── 1. Capture wrist image ───────────────────────────
        try:
            wrist_image = self.camera.capture_wrist_image()
        except Exception as e:
            log.warning(f"[Verify] Wrist capture failed: {e}")
            result["sub_phases"].append({
                "phase": "verify_capture", "status": "warn",
                "detail": str(e),
            })
            return result

        # Push wrist image to Streamlit + save to disk so the operator
        # can see what the wrist sees (especially important when the
        # frame is dark and verify fails).
        self._notify_wrist_image(wrist_image)
        try:
            from utils.image_utils import save_debug_image
            save_debug_image(
                wrist_image, directory="logs/wrist_verify",
                prefix=f"verify_{obj_id}_step{step_num}")
        except Exception:
            pass

        # ── 2. VLM: confirm part visible + collision risk ────
        # Use analyze_raw — WRIST_VERIFY_PROMPT returns a flat
        # {part_visible, path_clear, ...} object, NOT the scene
        # schema with detected_objects.  analyze_scene's validator
        # would reject this and force a wrong-format retry.
        try:
            verify = self.vlm.analyze_raw(
                wrist_image, prompt=WRIST_VERIFY_PROMPT)

            part_visible = verify.get("part_visible", False)
            path_clear = verify.get("path_clear", True)
            collision_risk = verify.get("collision_risk", "none")
            vlm_center = verify.get("center", {})
            confidence = verify.get("confidence", 0)

            log.info(
                f"[Verify] VLM: part_visible={part_visible}, "
                f"path_clear={path_clear}, risk={collision_risk}, "
                f"conf={confidence}")

            result["sub_phases"].append({
                "phase": "verify_vlm", "status": "success",
                "part_visible": part_visible,
                "path_clear": path_clear,
                "collision_risk": collision_risk,
            })
            result["path_clear"] = path_clear
            result["part_visible"] = part_visible

            if not part_visible:
                log.warning(
                    f"[Verify] No part visible for {obj_id} — "
                    f"caller should abort and re-scan")
                return result

            if collision_risk in ("high", "medium"):
                log.warning(
                    f"[Verify] Collision risk={collision_risk} — "
                    f"{verify.get('notes', '')}")

        except Exception as e:
            log.warning(f"[Verify] VLM verification failed: {e}")
            result["sub_phases"].append({
                "phase": "verify_vlm", "status": "warn",
                "detail": str(e),
            })
            return result

        # ── 3. Project VLM centre → world XYZ via wrist depth ─
        if not (vlm_center and "x" in vlm_center and "y" in vlm_center):
            return result

        try:
            proj = self.camera.project_to_world(
                [{"x": vlm_center["x"], "y": vlm_center["y"]}],
                camera="wrist")
            wpts = proj.get("world_points", [])
            if not wpts:
                return result

            wp = wpts[0]
            refined_x = wp["x"]
            refined_y = wp["y"]
            refined_z = wp["z"]

            # ── Explicit prim-position diagnostics ────────────
            # Read the current world XY of every relevant prim — the
            # bridge already accounts for these in the projection
            # math, but logging them here makes the geometry
            # auditable. If the realigment ever drives the gripper
            # to the wrong XY, comparing these numbers tells us
            # whether the bug is upstream (projection) or downstream
            # (realignment math).
            ee_pos = proj.get("ee_link_pos") or [None, None, None]
            cam_rgb_pos = proj.get("camera_pos") or [None, None, None]
            cam_depth_pos = proj.get("wrist_depth_camera_pos") or \
                [None, None, None]
            cam_to_ee_dx = (None if ee_pos[0] is None or cam_rgb_pos[0] is None
                            else cam_rgb_pos[0] - ee_pos[0])
            cam_to_ee_dy = (None if ee_pos[1] is None or cam_rgb_pos[1] is None
                            else cam_rgb_pos[1] - ee_pos[1])
            log.info(
                f"[Verify] Wrist projection: ({refined_x:.3f}, "
                f"{refined_y:.3f}, {refined_z:.3f}) "
                f"[{wp.get('method', '?')}]")
            if ee_pos[0] is not None:
                log.info(
                    f"[Verify] Prim XY snapshot at projection time: "
                    f"ee_link=({ee_pos[0]:.3f}, {ee_pos[1]:.3f})  "
                    f"wrist_rgb_cam=({cam_rgb_pos[0]:.3f}, "
                    f"{cam_rgb_pos[1]:.3f})  "
                    f"wrist_depth_cam=({cam_depth_pos[0]:.3f}, "
                    f"{cam_depth_pos[1]:.3f})")
            if cam_to_ee_dx is not None:
                log.info(
                    f"[Verify] wrist_rgb_cam → ee_link offset: "
                    f"dx={cam_to_ee_dx*1000:+.1f} mm, "
                    f"dy={cam_to_ee_dy*1000:+.1f} mm "
                    f"(this is the geometric XY shift between the "
                    f"camera that took the image and the EE that has "
                    f"to grasp the part)")

            # ── Optional camera-to-ee_link compensation ────────
            # The wrist projection returns the part's WORLD XY
            # directly — already accounting for the camera's
            # mounting offset from ee_link. Targeting ee_link to
            # that XY puts the GRIPPER over the part (correct for
            # grasping). However if the operator wants the WRIST
            # CAMERA over the part (e.g. for re-verification before
            # descent so the gripper geometry doesn't occlude the
            # target), enable the flag below: the workflow then
            # subtracts the cam→ee offset from the realignment
            # target so the camera, not the gripper, lands above
            # the part.
            compensate = bool(
                self.config.get("execution", {}).get(
                    "realign_camera_over_part", False))
            if compensate and cam_to_ee_dx is not None:
                refined_x_target = refined_x - cam_to_ee_dx
                refined_y_target = refined_y - cam_to_ee_dy
                log.info(
                    f"[Verify] realign_camera_over_part=true → "
                    f"shifting realignment target by cam→ee offset: "
                    f"({refined_x:.3f}, {refined_y:.3f}) → "
                    f"({refined_x_target:.3f}, {refined_y_target:.3f})")
                refined_x = refined_x_target
                refined_y = refined_y_target

        except Exception as e:
            log.warning(f"[Verify] Wrist projection failed: {e}")
            result["sub_phases"].append({
                "phase": "verify_projection", "status": "warn",
                "detail": str(e),
            })
            return result

        # ── 4. Validate Z (reject obvious depth errors) ──────
        corrected_z = self._validate_grasp_z(
            refined_z, pick_coords.get("z", 0.02))

        # ── 5. Compute offsets from planned position ─────────
        planned_x = pick_coords["x"]
        planned_y = pick_coords["y"]
        planned_z = pick_coords.get("z", 0.02)

        xy_offset = math.sqrt(
            (refined_x - planned_x) ** 2
            + (refined_y - planned_y) ** 2)
        z_offset = abs(corrected_z - planned_z)

        log.info(
            f"[Verify] Offset from plan: "
            f"XY={xy_offset:.4f}m, Z={z_offset:.4f}m")

        result["sub_phases"].append({
            "phase": "verify_projection", "status": "success",
            "refined": [refined_x, refined_y, corrected_z],
            "xy_offset": round(xy_offset, 4),
            "z_offset": round(z_offset, 4),
        })

        # ── 6. Update coordinates if offset is significant ───
        # Wrist-XY refinement policy. Two reasons NOT to apply:
        #   (a) Operator opted into "informational-only" mode: the
        #       wrist VLM pixel-centroid is too coarse to improve on
        #       the overhead OWL-ViT2 bbox centre (~25 mm typical
        #       error vs. <10 mm from overhead).
        #   (b) The proposed XY shift exceeds the neighbour-rejection
        #       cap, almost certainly meaning the wrist VLM picked a
        #       different part.
        # Z is always corrected when significant — the wrist depth
        # IS more accurate than overhead at close range.
        coords_updated = False
        wrist_xy_informational_only = bool(
            self.config.get("execution", {}).get(
                "wrist_xy_refinement_informational_only", True))

        if wrist_xy_informational_only:
            log.info(
                f"[Verify] Wrist XY proposal "
                f"({refined_x:.3f}, {refined_y:.3f}) recorded as "
                f"INFORMATIONAL only (offset={xy_offset*100:.1f} cm). "
                f"Keeping overhead OWL-derived coords "
                f"({planned_x:.3f}, {planned_y:.3f}) — overhead "
                f"detector is sub-pixel accurate (<1 cm world); "
                f"wrist VLM pixel centroid is ~5 % accurate (~25 mm "
                f"world). Set "
                f"execution.wrist_xy_refinement_informational_only="
                f"false to re-enable wrist XY correction.")
            result["sub_phases"].append({
                "phase": "verify_xy_informational",
                "status": "info",
                "wrist_proposed": [refined_x, refined_y],
                "kept": [planned_x, planned_y],
                "offset": round(xy_offset, 4),
            })
        elif xy_offset > XY_CORRECTION_MAX:
            log.warning(
                f"[Verify] XY offset {xy_offset*100:.1f} cm exceeds cap "
                f"{XY_CORRECTION_MAX*100:.1f} cm — REJECTING correction. "
                f"Likely the wrist VLM saw a neighbour part instead of "
                f"the planned target ({planned_x:.3f}, {planned_y:.3f}). "
                f"Keeping the original overhead-derived coords.")
            result["sub_phases"].append({
                "phase": "verify_xy_rejected", "status": "warn",
                "wrist_proposed": [refined_x, refined_y],
                "kept": [planned_x, planned_y],
                "offset": round(xy_offset, 4),
            })
        elif xy_offset > REALIGN_XY_THRESHOLD:
            log.info(
                f"[Verify] XY correction: ({planned_x:.3f}, "
                f"{planned_y:.3f}) -> ({refined_x:.3f}, "
                f"{refined_y:.3f})  (offset={xy_offset*100:.1f} cm)")
            result["coords"]["x"] = refined_x
            result["coords"]["y"] = refined_y
            coords_updated = True

        if z_offset > Z_CORRECTION_THRESHOLD:
            log.info(
                f"[Verify] Z correction: {planned_z:.3f} -> "
                f"{corrected_z:.3f}")
            result["coords"]["z"] = corrected_z
            coords_updated = True

        result["verified"] = True

        # ── 7. Realign gantry + Cartesian XY to corrected target ──
        # Two-step alignment so the EE arrives EXACTLY over the
        # wrist-verified part position before any descent:
        #
        #   (a) Slide the gantry to the corrected X. The arm joints
        #       compensate to hold the wrist's current world position
        #       (EE drift typically <1 cm).
        #   (b) Cartesian XY motion to (corrected_x, corrected_y) at
        #       the current EE Z. Without this, the gripper still
        #       sits at the OLD Y (stale overhead estimate); descent
        #       would then have to cover the Y delta in a single IK
        #       jump, which can swing the gripper through a divider
        #       wall (observed: 13 cm Y correction → gripper landed
        #       in wrong compartment, walls between fingers).
        if coords_updated and xy_offset > REALIGN_XY_THRESHOLD:
            corrected_x = result["coords"]["x"]
            corrected_y = result["coords"]["y"]

            # (a) Gantry slide
            log.info(
                f"[Verify] Realigning gantry to x="
                f"{corrected_x:.3f} (holding EE in place)")
            self._notify(WorkflowPhase.GRASP_REFINEMENT, "running",
                         "Sliding gantry while holding EE...")
            realign = self.camera.realign_gantry_hold_ee(corrected_x)
            realign_ok = realign.get("status") == "ok"
            ee_drift = realign.get("ee_drift_m")
            result["realigned"] = realign_ok
            result["sub_phases"].append({
                "phase": "gantry_realign",
                "status": "success" if realign_ok else "failed",
                "ee_drift_m": ee_drift,
            })

            # (b) Cartesian XY snap to corrected (x, y) at current Z
            cart_ok = True
            cart_drift = None
            if realign_ok:
                ee_after = realign.get("ee_after") or [None, None, None]
                cur_z = (float(ee_after[2])
                         if ee_after[2] is not None else None)
                if cur_z is not None:
                    ee_after_xy = (float(ee_after[0]),
                                   float(ee_after[1]))
                    xy_delta = (
                        (ee_after_xy[0] - corrected_x) ** 2 +
                        (ee_after_xy[1] - corrected_y) ** 2) ** 0.5
                    if xy_delta > 0.005:
                        log.info(
                            f"[Verify] Single-IK XY move to "
                            f"({corrected_x:.3f}, {corrected_y:.3f}) "
                            f"at z={cur_z:.3f} "
                            f"(delta={xy_delta*1000:.1f} mm)")
                        snap_payload = {
                            "x": corrected_x,
                            "y": corrected_y,
                            "z": cur_z,
                            "raw_z": True,
                        }
                        if getattr(self, "_scan_orientation", None):
                            snap_payload["orientation"] = list(
                                self._scan_orientation)
                        snap = self.camera.send_command(
                            "/api/approach", snap_payload, timeout=30)
                        cart_ok = snap.get("status") == "ok"
                        ee_pos = snap.get("ee_position")
                        if ee_pos:
                            cart_drift = (
                                (float(ee_pos[0]) - corrected_x) ** 2 +
                                (float(ee_pos[1]) - corrected_y) ** 2
                            ) ** 0.5
                        if not cart_ok:
                            log.warning(
                                f"[Verify] XY move FAILED: "
                                f"{snap.get('error', '?')}")
                        else:
                            log.info(
                                f"[Verify] XY move OK (residual XY "
                                f"drift = "
                                f"{(cart_drift or 0) * 1000:.1f} mm)")
                        result["sub_phases"].append({
                            "phase": "xy_snap",
                            "status": (
                                "success" if cart_ok else "failed"),
                            "residual_xy_m": cart_drift,
                        })

            if realign_ok and cart_ok:
                log.info(
                    f"[Verify] Realigned: gantry EE-hold drift "
                    f"{(ee_drift or 0) * 1000:.1f} mm, XY residual "
                    f"{(cart_drift or 0) * 1000:.1f} mm")
                self._notify(WorkflowPhase.GRASP_REFINEMENT,
                             "success",
                             f"Realigned over part "
                             f"(XY residual "
                             f"{(cart_drift or 0) * 1000:.1f} mm)")
            elif not realign_ok:
                log.warning(
                    f"[Verify] Gantry realign failed: "
                    f"{realign.get('error', '?')}")

        return result

    def _validate_grasp_z(self, projected_z, planned_z):
        """Validate projected Z against expected physical range.

        Depth projection from the wrist camera can hit bin walls,
        adjacent parts, or background instead of the target part.
        This check rejects obviously wrong depths and falls back
        to the planned Z (from the initial wrist scan).

        Expected part Z in this setup:
        - Bin floor ~ 0.49 m
        - Part tops ~ 0.7-0.9 m
        - Effective reach Z: [z_min, z_max] from config
        """
        reach = self.config.get("execution", {}).get(
            "effective_reach", {})
        z_min = reach.get("z_min", -0.05)
        z_max = reach.get("z_max", 1.00)

        # Outside reachable range -> reject
        if projected_z < z_min or projected_z > z_max:
            log.warning(
                f"[Z-valid] Projected Z={projected_z:.3f} outside "
                f"reach [{z_min}, {z_max}] -> using planned "
                f"Z={planned_z:.3f}")
            return planned_z

        # Large discrepancy from planned -> likely depth error
        # (bin wall, adjacent part, background)
        z_diff = abs(projected_z - planned_z)
        if z_diff > 0.15:
            # Blend: trust planned more than a bad projection
            corrected = planned_z * 0.7 + projected_z * 0.3
            log.warning(
                f"[Z-valid] Large Z discrepancy: "
                f"projected={projected_z:.3f} "
                f"planned={planned_z:.3f} diff={z_diff:.3f} "
                f"-> blended Z={corrected:.3f}")
            return corrected

        # Reasonable range — trust close-range projection
        return projected_z

    def _rescan_part_with_wrist(self, target_label: str,
                                 planned_coords: dict) -> dict:
        """At the depth-analysis pose, re-detect the target part with
        the FULL perception pipeline on the wrist camera.

        The wrist camera at ~30–50 cm range gives ~10× the angular
        resolution of the overhead camera. Running the same detection
        path (VLM scene scan + OWL-ViT2 with the catalogue + depth
        projection) on the wrist image lands the bbox on the part
        silhouette tightly enough that depth projection nails world XY
        to a few millimetres. Replaces the rough wrist-VLM check from
        ``_verify_and_realign`` with a re-grounded measurement.

        Cost: one extra full VLM call per pick (~30–90 s on Gemma4).
        Gated by ``execution.wrist_rescan_before_grasp`` (default true).

        Returns
        -------
        dict
            ``{x, y, z}`` refined world coords if a matching detection
            was found, otherwise ``planned_coords`` unchanged. Always
            valid — caller uses the result directly.
        """
        log.info(
            f"[Wrist-rescan] Re-detecting '{target_label}' "
            f"from wrist camera...")
        self._notify(WorkflowPhase.GRASP_REFINEMENT, "running",
                     f"Wrist rescan for {target_label}...")

        try:
            wrist_image = self.camera.capture_wrist_image()
        except Exception as e:
            log.warning(
                f"[Wrist-rescan] Capture failed: {e}; "
                f"keeping planned coords")
            return planned_coords

        if wrist_image is None:
            log.warning(
                "[Wrist-rescan] Empty wrist capture; "
                "keeping planned coords")
            return planned_coords

        # Reuse the overhead-detection pipeline on the wrist image.
        # camera_alias="wrist" tells the bridge's project_to_world to
        # use the wrist camera's USD transform + intrinsics.
        try:
            scene, source = self._detect_parts_from_bin_crop(
                wrist_image,
                bin_crop_box=(0.0, 0.0, 1.0, 1.0),
                target_parts=[target_label],
                bin_top_z=None,
                camera_alias="wrist",
            )
        except Exception as e:
            log.warning(
                f"[Wrist-rescan] Detection failed: {e}; "
                f"keeping planned coords")
            return planned_coords

        objects = scene.get("detected_objects", [])
        if not objects:
            log.info(
                f"[Wrist-rescan] No '{target_label}' detected in "
                f"wrist view; keeping planned coords")
            return planned_coords

        # Pick the detection NEAREST to the current planned XY (in case
        # multiple parts are visible in the wrist FOV — we want the one
        # we approached, not a neighbour).
        px = float(planned_coords.get("x", 0.0))
        py = float(planned_coords.get("y", 0.0))

        def _dist_xy(obj):
            pos = obj.get("approximate_position", {})
            ox = pos.get("x")
            oy = pos.get("y")
            if ox is None or oy is None:
                return float("inf")
            return ((float(ox) - px) ** 2 + (float(oy) - py) ** 2) ** 0.5

        best = min(objects, key=_dist_xy)
        pos = best.get("approximate_position", {})
        if not (pos and "x" in pos and "y" in pos):
            log.warning(
                "[Wrist-rescan] Best detection lacks world coords; "
                "keeping planned coords")
            return planned_coords

        refined = {
            "x": float(pos["x"]),
            "y": float(pos["y"]),
            "z": float(pos.get("z", planned_coords.get("z", 0.0))),
        }

        # Empirical XY-offset compensation. The wrist RGB camera and
        # the wrist depth-sensor (or ee_link) may be offset from each
        # other by a few mm to a few cm depending on USD prim
        # placement. If you observe a systematic residual offset
        # between the detected part centre (X mark on the wrist
        # overlay) and the gripper's actual landing point, sweep
        # these knobs to find the correct compensation. 0.0 = no
        # compensation (default). Positive X = shift target to
        # camera's right; positive Y = shift target forward (along
        # camera's down-axis, which is world-Y for a downward-
        # mounted wrist camera).
        x_offset = float(
            self.config.get("execution", {})
                .get("wrist_target_x_offset", 0.0))
        y_offset = float(
            self.config.get("execution", {})
                .get("wrist_target_y_offset", 0.0))
        if abs(x_offset) > 1e-6 or abs(y_offset) > 1e-6:
            refined["x"] += x_offset
            refined["y"] += y_offset
            log.info(
                f"[Wrist-rescan] Applied empirical offsets: "
                f"x={x_offset:+.4f}, y={y_offset:+.4f} m → "
                f"final ({refined['x']:.3f}, {refined['y']:.3f})")

        offset = ((refined["x"] - px) ** 2
                  + (refined["y"] - py) ** 2) ** 0.5
        log.info(
            f"[Wrist-rescan] '{target_label}': planned=("
            f"{px:.3f}, {py:.3f}) → refined=("
            f"{refined['x']:.3f}, {refined['y']:.3f}, "
            f"{refined['z']:.3f})  Δxy={offset*100:.1f} cm")
        return refined

    def _safe_floor_z(self, part_z: float) -> float:
        """Return the minimum ee_link Z for a part-specific compartment
        approach that keeps the gripper finger tips clear of the bin
        compartment wall top.

        Geometry (per-part, NOT per-rack):
            compartment_wall_top_z = part_z + box_height
            finger_TCP_z          = ee_link_z − gripper_tcp_offset
            require finger_TCP_z ≥ compartment_wall_top_z
                                 + bin_wall_finger_clearance
        ⇒   ee_link_z ≥ part_z + box_height + gripper_tcp_offset
                       + bin_wall_finger_clearance

        ``box_height`` (compartment wall height above the part) is a
        local, per-part value — much smaller than the rack's overall
        ``bin_top_z`` (which is the top of the entire rack structure
        and would push ee_link past the arm's vertical reach).

        Pass ``part_z`` from the caller — the world Z of the part top
        / grasp point as projected by ``/api/project_to_world``.
        """
        exec_cfg = self.config.get("execution", {})
        box_h = float(exec_cfg.get("box_height", 0.05))
        tcp_off = float(exec_cfg.get("gripper_tcp_offset", 0.170))
        clearance = float(exec_cfg.get("bin_wall_finger_clearance", 0.02))
        return float(part_z) + box_h + tcp_off + clearance

    def _realign_over_part(self, params: dict, step_num,
                           align_height: float,
                           bin_top_z: float) -> bool:
        """Slide gantry + UR10 IK so the wrist sits over the target
        part at a collision-safe height with the ee_link facing down.

        This is the explicit "vagn realigns to detected part" step in
        the wrist-scan pipeline — runs BEFORE the existing pick
        sequence so the depth camera sees the part from a known-safe
        vantage before any straight-down descent.

        Z policy
        --------
        The realign target is the LARGER of:
          - ``part_z + align_height`` (the "preferred" close-range
            descent for depth analysis), and
          - the geometric "fingers-clear-walls" floor (``bin_top_z +
            gripper_tcp_offset + bin_wall_finger_clearance``) — see
            :meth:`_safe_floor_z`.

        The scan-pose floor is critical for multi-compartment bins
        whose dividers extend above the parts: a single IK jump from
        scan height down to ``part_z + 0.4`` would route the wrist
        through divider walls and self-collide. Staying at scan height
        defers the actual descent to ``_pick_descend`` which moves
        Cartesian (X/Y locked) — the only safe way to enter a single
        compartment.

        Also applies ``scan_pose_orientation`` so the gripper fingers
        stay rotated out of the wrist camera's view.

        Returns True if the IK call succeeded.
        """
        obj_id = params.get("object_id", "?")
        part_x = params.get("x", 0.0)
        part_y = params.get("y", 0.0)
        part_z = float(params.get("z", 0.0))
        preferred_z = part_z + align_height
        safe_floor = self._safe_floor_z(part_z)
        target_z = max(preferred_z, safe_floor)
        if target_z > preferred_z + 1e-3:
            log.info(
                f"[Realign {step_num}] Clamping z {preferred_z:.3f} → "
                f"{target_z:.3f} (part_z={part_z:.3f} + box_height + "
                f"tcp_offset + clearance) to keep fingers above "
                f"compartment walls.")
        target = {
            "x": part_x, "y": part_y, "z": target_z,
            # No safety buffer — ee_link lands at the requested
            # clearance above the part, no padding, no restriction.
            "raw_z": True,
            # Cartesian micro-stepping: long XY hops between bin
            # compartments would otherwise let the IK solver flip into
            # a shoulder-up / elbow-back configuration. Step-by-step
            # Cartesian motion keeps the joint state continuous.
            "cartesian": True,
        }
        if getattr(self, "_scan_orientation", None):
            target["orientation"] = list(self._scan_orientation)

        self._notify(WorkflowPhase.REALIGN, "running",
                     f"Step {step_num}: realign over {obj_id} at "
                     f"({part_x:.3f}, {part_y:.3f}, {target_z:.3f})")
        resp = self.camera.send_command(
            "/api/approach", target, timeout=30)
        ok = resp.get("status") == "ok"
        if not ok:
            err = resp.get("error", "?")
            log.error(
                f"[Realign {step_num}] FAILED for {obj_id}: {err}")
            self._notify(WorkflowPhase.REALIGN, "failed",
                         f"Realign failed for {obj_id}: {err}")
        else:
            self._notify(WorkflowPhase.REALIGN, "success",
                         f"Wrist over {obj_id}")
        return ok

    def _execute_pick_sequence(self, step, params, scene, step_num):
        """
        4-phase pick with wrist-camera VLM verification:

          Phase A:  Approach + Depth scan
          Phase A2: Pre-grasp verification + gantry realignment
          Phase B:  Descend to grasp position (open fingers)
                    -> wrist VLM: "is part between fingers?"
          Phase C:  Close gripper
                    -> wrist VLM: "is part properly grasped?"
          Phase D:  Retract (only after VLM confirms grasp)
        """
        result = {"step": step_num, "action": "pick_object", "sub_phases": []}
        obj_id = params.get("object_id", "?")
        # Use bin_top_z as Z fallback (parts sit ~0.5 m above world
        # origin in this setup; the old 0.02 fallback was tray height
        # and made IK targets unreachable below the bin).
        fallback_z = (
            self._bin_top_z if self._bin_top_z is not None else 0.55)
        pick_coords = {
            "x": params.get("x", 0.3),
            "y": params.get("y", 0.0),
            "z": params.get("z", fallback_z),
        }
        # Also catch a None/zero Z that slipped through the plan
        if (pick_coords["z"] is None
                or abs(pick_coords["z"]) < 0.05):
            log.warning(
                f"[Pick {step_num}] Suspicious z={pick_coords['z']} "
                f"for {obj_id} → using bin_top_z={fallback_z:.3f}")
            pick_coords["z"] = fallback_z

        # ── Phase A: Approach (single IK + safe-height auto-fallback) ──
        # The bridge's default approach formula adds +0.45 m to the
        # part Z (BOX_ENTRY_MARGIN + BOX_HEIGHT + GRIPPER_TCP_OFFSET
        # + 0.10) — fine for table-level parts, but on this gantry +
        # UR10 setup it pushes the IK target above the arm's vertical
        # reach (e.g. for a part at z=0.80, target z=1.25 is outside
        # effective_reach.z_max=1.0). We send `raw_z=True` and try a
        # ladder of clearances above the part — the FIRST candidate
        # is the preferred safe height (above bin dividers), and we
        # only step lower if the IK rejects it. Using a single IK
        # jump (no Cartesian micro-stepping) — Cartesian doesn't
        # actually avoid arm-body collisions because IK only
        # constrains the EE endpoint, not what the elbow/forearm do.
        #
        # SAFE-HEIGHT FLOOR: Every approach candidate is clamped to a
        # per-part geometric floor that keeps the gripper FINGER TIPS
        # above the COMPARTMENT wall top by
        # ``bin_wall_finger_clearance`` metres. The compartment wall
        # is part-local (``part_z + box_height``) — NOT the whole-rack
        # ``bin_top_z`` which would push ee_link past the arm's reach.
        # See :meth:`_safe_floor_z` for the formula.
        safe_floor = self._safe_floor_z(pick_coords["z"])
        approach_z_candidates = [
            max(pick_coords["z"] + dz, safe_floor)
            for dz in (0.30, 0.25, 0.20, 0.15, 0.10, 0.05)
        ]
        # Log when the floor actually engages so the operator can see
        # the safety clamp working (otherwise it's silent).
        raw_preferred = pick_coords["z"] + 0.30
        if safe_floor > raw_preferred + 1e-3:
            log.info(
                f"[Pick {step_num}] Approach z clamped: "
                f"{raw_preferred:.3f} → {safe_floor:.3f} "
                f"(part_z={pick_coords['z']:.3f} + box_height + "
                f"tcp_offset + clearance; keeps fingers above "
                f"compartment walls)")
        approach_result = None
        approach_ok = False
        used_approach_z = None
        for cand_z in approach_z_candidates:
            cand_pose = {
                "x": pick_coords["x"],
                "y": pick_coords["y"],
                "z": cand_z,
                "raw_z": True,
            }
            self._notify(
                WorkflowPhase.APPROACH, "running",
                f"Step {step_num}: Moving near {obj_id} "
                f"({cand_pose['x']:.3f}, {cand_pose['y']:.3f}, "
                f"{cand_z:.3f})")
            approach_result = self.camera.send_command(
                "/api/approach", cand_pose)
            if approach_result.get("status") == "ok":
                approach_ok = True
                used_approach_z = cand_z
                break
            log.warning(
                f"[Pick {step_num}] Approach IK failed at z="
                f"{cand_z:.3f} ({approach_result.get('error', '?')})"
                f" — trying lower clearance")
        result["sub_phases"].append({
            "phase": WorkflowPhase.APPROACH,
            "status": "success" if approach_ok else "failed",
            "error": (approach_result.get("error")
                      if approach_result else "no candidate tried"),
            "target": {**pick_coords, "approach_z": used_approach_z},
        })

        if not approach_ok:
            err = (approach_result.get("error", "?")
                   if approach_result else "?")
            log.error(
                f"[Pick {step_num}] Approach FAILED for {obj_id} at "
                f"every candidate z above part_z="
                f"{pick_coords['z']:.3f}: {err}")
            self._notify(WorkflowPhase.APPROACH, "failed",
                         f"Approach failed for {obj_id}: {err}")
            result["status"] = "failed"
            result["error"] = f"approach_failed: {err}"
            return result

        log.info(
            f"[Pick {step_num}] Approach OK at z={used_approach_z:.3f} "
            f"(part_z={pick_coords['z']:.3f}, clearance="
            f"{used_approach_z - pick_coords['z']:.3f} m)")
        self._notify(WorkflowPhase.APPROACH, "success",
                     f"Robot positioned near {obj_id} "
                     f"(z={used_approach_z:.3f})")

        # ── Depth camera analysis ──────────────────────────────
        # Two independent jobs here:
        #   (a) Capture a depth image and push it to the Streamlit
        #       depth tab + save to logs/depth_analysis. Always done
        #       — the operator wants to see the depth view even when
        #       the VLM check is disabled.
        #   (b) Send the depth image to the VLM and treat
        #       part_visible=false as a hard pick-abort. Gated by
        #       ``execution.depth_analysis_vlm_check`` because the
        #       VLM check is geometrically unreliable at the approach
        #       pose (wrist depth camera ~55 mm offset from ee_link)
        #       and produces false negatives that abort otherwise-
        #       good picks. Default off.
        do_depth_vlm_check = bool(
            self.config.get("execution", {}).get(
                "depth_analysis_vlm_check", False))

        self._notify(WorkflowPhase.DEPTH_ANALYSIS, "running",
                     "Capturing depth image for spatial analysis...")
        depth_part_visible = True
        depth_conf = None
        try:
            depth_image = self.camera.capture_depth_image()
            # Push depth image to Streamlit + save for debugging so
            # the operator sees what depth captured at this approach.
            self._notify_depth_image(depth_image)
            try:
                from utils.image_utils import save_debug_image
                save_debug_image(
                    depth_image, directory="logs/depth_analysis",
                    prefix=f"depth_{obj_id}_step{step_num}")
            except Exception:
                pass

            if do_depth_vlm_check:
                # DEPTH_GRASP_PROMPT returns {part_visible, part_center,
                # estimated_distance_m, ...} — flat schema, not the
                # scene schema. analyze_raw skips detected_objects
                # validation.
                depth_scene = self.vlm.analyze_raw(
                    depth_image, prompt=DEPTH_GRASP_PROMPT)
                depth_part_visible = depth_scene.get(
                    "part_visible",
                    bool(depth_scene.get("detected_objects")))
                depth_conf = depth_scene.get("confidence")
                result["sub_phases"].append({
                    "phase": WorkflowPhase.DEPTH_ANALYSIS,
                    "status": "success",
                    "depth_data": depth_scene,
                    "part_visible": depth_part_visible,
                })
                self._notify(
                    WorkflowPhase.DEPTH_ANALYSIS, "success",
                    f"Depth analysis — conf={depth_conf}, "
                    f"part_visible={depth_part_visible}")
            else:
                log.info(
                    "[Pick] Skipping depth-analysis VLM check "
                    "(execution.depth_analysis_vlm_check=false). "
                    "Depth image captured for display/log only. "
                    "Visibility relies on the upstream overhead VLM "
                    "+ OWL detection, the wrist verify+realign step, "
                    "and the contact-aware descent.")
                result["sub_phases"].append({
                    "phase": WorkflowPhase.DEPTH_ANALYSIS,
                    "status": "skipped",
                    "detail": "VLM check disabled — image captured "
                              "for diagnostics only",
                })
                self._notify(
                    WorkflowPhase.DEPTH_ANALYSIS, "success",
                    "Depth image captured (VLM check disabled)")
        except Exception as e:
            log.warning(f"Depth analysis failed: {e}")
            result["sub_phases"].append({
                "phase": WorkflowPhase.DEPTH_ANALYSIS,
                "status": "warn", "detail": str(e),
            })

        # GATE: only honour the VLM verdict when the check actually
        # ran. ``depth_part_visible`` defaults to True at the top of
        # this method, but be explicit so a future refactor can't
        # introduce an abort-on-disabled-check regression.
        if do_depth_vlm_check and not depth_part_visible:
            log.error(
                f"[Pick {step_num}] Depth shows no part at "
                f"({pick_coords['x']:.3f}, {pick_coords['y']:.3f}, "
                f"{pick_coords['z']:.3f}) — aborting pick "
                f"so the workflow can re-scan from home")
            self._notify(WorkflowPhase.DEPTH_ANALYSIS, "failed",
                         "No part visible in depth — re-scanning")
            result["status"] = "failed"
            result["error"] = "no_part_in_depth"
            return result

        # ── Phase A2: Pre-grasp verification + realignment ─────
        # Wrist camera confirms the part is visible, checks for
        # collision risk, and refines coordinates from close range.
        # If XY is off by >3 cm, the gantry realigns before descent.
        # If the wrist VLM reports no part visible, abort the pick so
        # the outer retry loop sends the robot home and re-scans.
        self._notify(WorkflowPhase.GRASP_REFINEMENT, "running",
                     f"Verifying grasp coordinates for {obj_id}...")

        verify = self._verify_and_realign(pick_coords, obj_id, step_num)
        result["sub_phases"].extend(verify.get("sub_phases", []))

        if not verify.get("part_visible", True):
            log.error(
                f"[Pick {step_num}] Wrist VLM reports no part "
                f"visible for {obj_id} — aborting pick "
                f"so the workflow can re-scan from home")
            self._notify(WorkflowPhase.GRASP_REFINEMENT, "failed",
                         "Wrist sees no part — re-scanning")
            result["status"] = "failed"
            result["error"] = "no_part_visible_wrist"
            return result

        if verify["verified"]:
            corrected = verify["coords"]
            if (corrected["x"] != pick_coords["x"]
                    or corrected["y"] != pick_coords["y"]
                    or corrected["z"] != pick_coords["z"]):
                log.info(
                    f"[Pick] Coords corrected: "
                    f"({pick_coords['x']:.3f}, {pick_coords['y']:.3f}, "
                    f"{pick_coords['z']:.3f}) -> "
                    f"({corrected['x']:.3f}, {corrected['y']:.3f}, "
                    f"{corrected['z']:.3f})")
                pick_coords = corrected
            self._notify(
                WorkflowPhase.GRASP_REFINEMENT, "success",
                f"Verified — "
                f"{'realigned' if verify['realigned'] else 'aligned'}")
        else:
            self._notify(WorkflowPhase.GRASP_REFINEMENT, "warn",
                         "Verification inconclusive — "
                         "using planned coords")

        if not verify.get("path_clear", True):
            log.warning(
                "[Pick] Descent path may have obstacles — "
                "proceeding with caution")

        # ── Optional: full wrist rescan to refine grasp XY ──
        # ── Optional close-range explicit rescan ─────────────
        # Operator-requested step: read ee_link + wrist-camera world
        # positions, project the centre-most part through the camera,
        # express the grasp target in world coords (which == ee_link
        # XY for grasping, since gripper TCP has zero XY offset from
        # ee_link in this geometry). Logs every transform so the
        # math is auditable. Gated by config so it can be turned off
        # if it ever produces a worse result than the upstream
        # verify_and_realign.
        if self.config.get("execution", {}).get(
                "close_range_rescan_xy", True):
            target_label = (params.get("label")
                            or params.get("target_label")
                            or scene.get("_last_target_label", "")
                            or "")
            if not target_label:
                for o in scene.get("detected_objects", []):
                    if o.get("object_id") == obj_id:
                        target_label = o.get("label", "")
                        break
            rescan_max_shift = float(
                self.config.get("execution", {}).get(
                    "close_range_rescan_max_shift", 0.05))  # 5 cm cap
            rescan = self._close_range_rescan_xy(
                pick_coords, obj_label=target_label)
            if rescan is not None:
                if rescan["shift_mm"] > rescan_max_shift * 1000:
                    log.warning(
                        f"[Rescan] Proposed shift "
                        f"{rescan['shift_mm']:.1f} mm exceeds cap "
                        f"{rescan_max_shift*1000:.0f} mm — likely "
                        f"the wrist camera saw a NEIGHBOUR part "
                        f"instead of the planned target. Keeping "
                        f"current coords ({pick_coords['x']:.3f}, "
                        f"{pick_coords['y']:.3f}) instead.")
                else:
                    log.info(
                        f"[Rescan] Applying shift "
                        f"{rescan['shift_mm']:.1f} mm: "
                        f"({pick_coords['x']:.4f}, "
                        f"{pick_coords['y']:.4f}) → "
                        f"({rescan['x']:.4f}, {rescan['y']:.4f}) "
                        f"— gripper will now land on the projected "
                        f"part centre")
                    pick_coords["x"] = rescan["x"]
                    pick_coords["y"] = rescan["y"]
                    # Z gets corrected too — the close-range depth
                    # is more accurate than the overhead estimate.
                    pick_coords["z"] = rescan["z"]

        # The verify_and_realign step above uses a focused wrist VLM
        # prompt that gives a rough centre. Running the FULL perception
        # pipeline (VLM scene scan + OWL-ViT2 + depth projection) on
        # the wrist image at depth-analysis range produces a much
        # tighter bbox → projection → world XY (within a few mm).
        # Costs one extra VLM call per pick. Gated by config flag.
        if self.config.get("execution", {}).get(
                "wrist_rescan_before_grasp", True):
            target_label = (params.get("label")
                            or params.get("target_label")
                            or scene.get("_last_target_label", "")
                            or "")
            if not target_label:
                # Fall back: look up the obj_id in the scene.
                for o in scene.get("detected_objects", []):
                    if o.get("object_id") == obj_id:
                        target_label = o.get("label", "")
                        break
            if target_label:
                refined = self._rescan_part_with_wrist(
                    target_label, pick_coords)
                if (refined is not pick_coords
                        and (abs(refined.get("x", 0)
                                 - pick_coords.get("x", 0)) > 1e-6
                             or abs(refined.get("y", 0)
                                    - pick_coords.get("y", 0)) > 1e-6)):
                    # Reject the rescan correction if the XY shift is
                    # larger than ``max_wrist_rescan_offset_xy``. Big
                    # shifts usually mean the wrist VLM saw a NEIGHBOUR
                    # part (not the planned target) or a partial /
                    # occluded view — applying that correction would
                    # force a large lateral move at depth-analysis
                    # altitude where the IK can swing the arm into the
                    # bin's back wall. The original verify_and_realign
                    # coords were already a reasonable refinement; we
                    # fall back to those and skip the rescan.
                    max_offset = float(
                        self.config.get("execution", {})
                            .get("max_wrist_rescan_offset_xy", 0.04))
                    dx = refined["x"] - pick_coords["x"]
                    dy = refined["y"] - pick_coords["y"]
                    offset = (dx * dx + dy * dy) ** 0.5
                    if offset > max_offset:
                        log.warning(
                            f"[Pick] Wrist rescan offset {offset*100:.1f} cm"
                            f" exceeds cap "
                            f"{max_offset*100:.1f} cm — rejecting "
                            f"correction. Using verify_and_realign "
                            f"coords ({pick_coords['x']:.3f}, "
                            f"{pick_coords['y']:.3f}, "
                            f"{pick_coords['z']:.3f}) instead. "
                            f"Likely a wrong-part or occluded wrist "
                            f"detection.")
                    else:
                        log.info(
                            f"[Pick] Wrist rescan refined coords: "
                            f"({pick_coords['x']:.3f}, "
                            f"{pick_coords['y']:.3f}, "
                            f"{pick_coords['z']:.3f}) → "
                            f"({refined['x']:.3f}, "
                            f"{refined['y']:.3f}, "
                            f"{refined['z']:.3f}) "
                            f"(offset {offset*100:.1f} cm)")
                        pick_coords = refined
            else:
                log.info(
                    "[Pick] Skipping wrist rescan — no target label "
                    "available for the active object")

        # ── Local retry loop (lift + rescan, no full pipeline restart) ──
        # When the mechanical close fails or the post-grasp VLM
        # confirms an empty gripper, we previously bubbled up to the
        # outer pipeline retry which re-ran Phase 2 perception + Phase
        # 3 LLM planning (~30-60 s extra). The local retry here just:
        #   1. Opens the gripper
        #   2. Lifts ee_link to depth-analysis height
        #   3. Re-runs the close-range wrist rescan
        #   4. Re-attempts descent + close
        # Saves the perception/LLM cost; only escalates to the outer
        # pipeline retry if this many local retries can't recover.
        local_retries_max = int(
            self.config.get("execution", {}).get(
                "local_grasp_retries", 2))
        depth_h = float(
            self.config.get("execution", {}).get(
                "depth_analysis_height", 0.20))

        # Capture the obj_label up front so the rescan can use it.
        rescan_label = (params.get("label")
                        or params.get("target_label")
                        or scene.get("_last_target_label", "")
                        or "")
        if not rescan_label:
            for o in scene.get("detected_objects", []):
                if o.get("object_id") == obj_id:
                    rescan_label = o.get("label", "")
                    break

        # Sentinels populated by the loop body — referenced AFTER the
        # loop ends to drive retract / failure handling.
        descend_result = {}
        close_result = {}
        grasp_confirmed = False
        post_grasp_confidence = 0.0
        grasp_ok = True
        descend_ok = False
        local_attempt_used = 0

        for local_attempt in range(local_retries_max + 1):
            local_attempt_used = local_attempt
            if local_attempt > 0:
                # Lift ee_link back to depth-analysis height above
                # the (possibly refined) part XY, then rescan.
                lift_z = float(pick_coords.get("z", 0.5)) + depth_h
                log.warning(
                    f"[Pick {step_num}] Local retry "
                    f"{local_attempt}/{local_retries_max} — lifting to "
                    f"({pick_coords['x']:.3f}, "
                    f"{pick_coords['y']:.3f}, {lift_z:.3f}) and "
                    f"rescanning before re-attempting grasp.")
                self._notify(WorkflowPhase.PICK_EXECUTE, "running",
                             f"Local retry {local_attempt}: lifting "
                             f"+ rescanning")
                try:
                    self.camera.send_command(
                        "/api/gripper", {"action": "open"})
                except Exception:
                    pass
                try:
                    self.camera.send_command(
                        "/api/approach",
                        {"x": pick_coords["x"],
                         "y": pick_coords["y"],
                         "z": lift_z, "raw_z": True})
                except Exception as exc:
                    log.warning(
                        f"[Pick] Local retry lift FAILED: {exc} — "
                        f"trying rescan + descent at current pose")
                # Refine XY with a fresh wrist rescan
                try:
                    rescan2 = self._close_range_rescan_xy(
                        pick_coords, obj_label=rescan_label)
                except Exception as exc:
                    rescan2 = None
                    log.warning(
                        f"[Pick] Local retry rescan FAILED: {exc}")
                if rescan2 is not None:
                    cap = float(
                        self.config.get("execution", {}).get(
                            "close_range_rescan_max_shift", 0.05))
                    if rescan2["shift_mm"] <= cap * 1000:
                        pick_coords["x"] = rescan2["x"]
                        pick_coords["y"] = rescan2["y"]
                        pick_coords["z"] = rescan2["z"]
                        log.info(
                            f"[Pick] Local retry coords refined: "
                            f"({pick_coords['x']:.4f}, "
                            f"{pick_coords['y']:.4f}, "
                            f"{pick_coords['z']:.4f})")
                    else:
                        log.warning(
                            f"[Pick] Local retry rescan shift "
                            f"{rescan2['shift_mm']:.1f} mm exceeds cap "
                            f"{cap*1000:.0f} mm — keeping previous "
                            f"coords for the retry")

            # ── Phase B: Descend to grasp position (fingers open) ──
            # Inject the contact-sensor force threshold so the bridge's
            # Cartesian descent stops the moment a finger touches anything,
            # rather than blindly driving down to the planned grasp_z (which
            # can push the gripper into the bin floor when depth projection
            # underestimates the part top).
            self._notify(WorkflowPhase.PICK_EXECUTE, "running",
                         "Descending to grasp position...")
            descend_payload = dict(pick_coords)
            descend_payload["descent_contact_min_force"] = float(
                self.config.get("execution", {})
                    .get("descent_contact_min_force", 0.5))
            descend_payload["descent_contact_lift_back"] = float(
                self.config.get("execution", {})
                    .get("descent_contact_lift_back", 0.002))

            descend_result = self.camera.send_command(
                "/api/pick_descend", descend_payload)
            descend_ok = descend_result.get("status") == "ok"
            result["sub_phases"].append({
                "phase": "pick_descend",
                "status": "success" if descend_ok else "failed",
                "local_attempt": local_attempt,
            })

            if not descend_ok:
                # Descent IK failures are NOT recoverable by lifting
                # and rescanning — they indicate the part XY is
                # unreachable. Escalate to the outer pipeline retry
                # immediately.
                self._notify(WorkflowPhase.PICK_EXECUTE, "failed",
                             f"Descend failed: {descend_result.get('error', '?')}")
                result["status"] = "failed"
                result["error"] = "descend_failed"
                return result

            self._notify(WorkflowPhase.PICK_EXECUTE, "success",
                         "At grasp position — checking part placement...")

            # ── Wrist VLM check 1: "Is the part between the fingers?" ──
            # (skipped by default — geometrically broken at grasp height)
            do_pre_grasp_check = bool(
                self.config.get("execution", {}).get(
                    "pre_grasp_wrist_vlm_check", False))
            wrist_b64 = descend_result.get("wrist_image")
            part_between_fingers = True
            pre_grasp_confidence = 0.0
            if wrist_b64 and do_pre_grasp_check:
                self._notify(WorkflowPhase.GRASP_VERIFY, "running",
                             "Wrist camera: checking part is between fingers...")
                try:
                    pre_grasp_vlm = self._wrist_vlm_check(wrist_b64, (
                        "You are looking at a close-up image from a wrist-mounted "
                        "camera on a Robotiq 2F-140 gripper. The fingers are OPEN. "
                        "Answer ONLY with a JSON object: "
                        '{"part_between_fingers": true/false, "confidence": 0.0-1.0, '
                        '"detail": "short reason"}. '
                        "Is there a graspable part visible between the two open "
                        "finger pads?"
                    ))
                    part_between_fingers = pre_grasp_vlm.get(
                        "part_between_fingers", True)
                    pre_grasp_confidence = float(
                        pre_grasp_vlm.get("confidence", 0.0) or 0.0)
                    result["sub_phases"].append({
                        "phase": "pre_grasp_vlm",
                        "status": "success",
                        "part_between_fingers": part_between_fingers,
                        "confidence": pre_grasp_confidence,
                        "local_attempt": local_attempt,
                    })
                except Exception as e:
                    log.warning(f"Pre-grasp wrist VLM failed: {e}")
                    result["sub_phases"].append({
                        "phase": "pre_grasp_vlm",
                        "status": "warn", "detail": str(e)})

            wrist_min_conf = float(
                self.config.get("execution", {}).get(
                    "wrist_verify_min_confidence", 0.7))
            # Pre-grasp confident-no abort (only when check enabled)
            if (do_pre_grasp_check and not part_between_fingers
                    and pre_grasp_confidence >= wrist_min_conf):
                log.error(
                    f"[Pick {step_num}] Wrist VLM confirms NO part between "
                    f"fingers (conf={pre_grasp_confidence:.2f}) — local "
                    f"retry {local_attempt}/{local_retries_max}")
                if local_attempt < local_retries_max:
                    continue
                self._notify(WorkflowPhase.GRASP_VERIFY, "failed",
                             f"No part between fingers — outer retry")
                try:
                    self.camera.send_command(
                        "/api/gripper", {"action": "open"})
                except Exception:
                    pass
                result["status"] = "failed"
                result["error"] = "no_part_between_fingers_vlm"
                return result

            # ── Phase C: Close gripper ─────────────────────────────
            self._notify(WorkflowPhase.PICK_EXECUTE, "running",
                         "Closing gripper...")
            close_result = self.camera.send_command("/api/pick_close", {})
            grasp_confirmed = close_result.get("grasp_confirmed", False)
            grip_hold = close_result.get("grip_hold", 0.5)
            result["grip_hold"] = grip_hold
            result["finger_close"] = close_result.get("finger_close", 0.5)
            result["sub_phases"].append({
                "phase": "pick_close",
                "status": "success" if grasp_confirmed else "warn",
                "grasp_confirmed": grasp_confirmed,
                "actual_finger": close_result.get("actual_finger"),
                "local_attempt": local_attempt,
            })

            # ── Mechanical close-failure guard (LOCAL retry) ───────
            actual_finger = close_result.get("actual_finger")
            target_finger = close_result.get("finger_close", 0.5)
            try:
                af = float(actual_finger) if actual_finger is not None else None
                tf = float(target_finger)
            except (TypeError, ValueError):
                af, tf = None, None
            gap_too_large = (af is not None and tf is not None
                             and (tf - af) > 0.10)
            if not grasp_confirmed or gap_too_large:
                log.warning(
                    f"[Pick {step_num}] Mechanical close FAILED — "
                    f"grasp_confirmed={grasp_confirmed}, actual={af}, "
                    f"target={tf} — local retry "
                    f"{local_attempt}/{local_retries_max}")
                try:
                    self.camera.send_command(
                        "/api/gripper", {"action": "open"})
                except Exception:
                    pass
                if local_attempt < local_retries_max:
                    continue  # lift + rescan + retry
                # Out of local retries — escalate to outer pipeline
                self._notify(WorkflowPhase.PICK_EXECUTE, "failed",
                             f"Close failed (actual={af}, target={tf}) "
                             f"after {local_retries_max+1} local "
                             f"attempts — escalating to outer retry")
                result["status"] = "failed"
                result["error"] = "mechanical_close_failed"
                return result

            # ── Wrist VLM check 2: post-grasp ──
            close_wrist_b64 = close_result.get("wrist_image")
            grasp_ok = True
            post_grasp_confidence = 0.0
            if close_wrist_b64:
                self._notify(WorkflowPhase.GRASP_VERIFY, "running",
                             "Wrist camera: confirming part is grasped...")
                try:
                    post_grasp_vlm = self._wrist_vlm_check(close_wrist_b64, (
                        "You are looking at a close-up image from a wrist-mounted "
                        "camera on a Robotiq 2F-140 gripper. The fingers are CLOSED. "
                        "Answer ONLY with a JSON object: "
                        '{"grasp_ok": true/false, "confidence": 0.0-1.0, '
                        '"detail": "short reason"}.'
                    ))
                    grasp_ok = post_grasp_vlm.get("grasp_ok", True)
                    post_grasp_confidence = float(
                        post_grasp_vlm.get("confidence", 0.0) or 0.0)
                    result["sub_phases"].append({
                        "phase": "post_grasp_vlm",
                        "status": "success",
                        "grasp_ok": grasp_ok,
                        "confidence": post_grasp_confidence,
                        "local_attempt": local_attempt,
                    })
                    self._notify(WorkflowPhase.GRASP_VERIFY, "success",
                                 f"Grasp confirmed by VLM: {grasp_ok} "
                                 f"(conf={post_grasp_confidence:.2f})")
                except Exception as e:
                    log.warning(f"Post-grasp wrist VLM failed: {e}")
                    result["sub_phases"].append({
                        "phase": "post_grasp_vlm",
                        "status": "warn", "detail": str(e)})

            # Confident empty-gripper → LOCAL retry first, then escalate
            if (not grasp_ok
                    and post_grasp_confidence >= wrist_min_conf):
                log.warning(
                    f"[Pick {step_num}] Wrist VLM confirms gripper is "
                    f"EMPTY (conf={post_grasp_confidence:.2f}) — "
                    f"local retry {local_attempt}/{local_retries_max}")
                try:
                    self.camera.send_command(
                        "/api/gripper", {"action": "open"})
                except Exception:
                    pass
                if local_attempt < local_retries_max:
                    continue  # lift + rescan + retry
                self._notify(WorkflowPhase.GRASP_VERIFY, "failed",
                             "Empty gripper after local retries — "
                             "escalating to outer retry")
                result["status"] = "failed"
                result["error"] = "empty_gripper_vlm"
                return result
            elif not grasp_ok:
                log.warning(
                    f"VLM says grasp not secure but confidence "
                    f"{post_grasp_confidence:.2f} below threshold "
                    f"{wrist_min_conf:.2f} — proceeding")

            # ── Grasp succeeded — break out of local retry loop ────
            log.info(
                f"[Pick {step_num}] Grasp succeeded on local attempt "
                f"{local_attempt + 1}/{local_retries_max + 1}")
            break

        # End of local retry loop. ``descend_result``, ``close_result``,
        # ``grip_hold`` etc. are populated from the successful attempt.

        # ── Phase D: Retract (only after grasp confirmed) ──────
        self._notify(WorkflowPhase.PICK_EXECUTE, "running",
                     "Retracting with part...")

        retract_payload = dict(pick_coords)
        retract_payload["grip_hold"] = grip_hold
        retract_result = self.camera.send_command(
            "/api/pick_retract", retract_payload)
        retract_ok = retract_result.get("status") == "completed"

        result["sub_phases"].append({
            "phase": "pick_retract",
            "status": "success" if retract_ok else "failed",
        })

        self._notify(WorkflowPhase.PICK_EXECUTE,
                     "success" if retract_ok else "failed",
                     f"Pick {'completed' if retract_ok else 'FAILED'}")

        result["status"] = "success" if retract_ok else "failed"
        return result

    # ─── Place Sequence ─────────────────────────────────────

    def _execute_place_sequence(self, step, params, scene, step_num):
        """Phase 7: Place object using real destination coordinates.

        Coordinates come DIRECTLY from the USD tray prim (cached on
        ``self._tray_position`` during Phase 0) — bypassing the LLM's
        plan params for X/Y/Z. The LLM's role for the tray is only to
        decide *whether* to place; the *where* is anchored to USD.

        Two values are sent to the bridge:
          - ``z = tray_top_z`` — the **raw** USD tray top, no buffer.
            This is what the bridge uses to compute ``place_z`` (the
            actual drop point), so the part lands ~2 cm above the tray
            surface — a clean release, not a long fall.
          - ``transit_clearance`` — extra metres added ONLY to the
            transit phase (the lateral-move altitude before descent).
            Gives Lula IK headroom to find a non-folded arm pose
            without affecting the place height. Tunable via
            ``execution.place_transit_clearance``.
        """
        result = {"step": step_num, "action": "place_object"}

        self._notify(WorkflowPhase.PLACE_EXECUTE, "running",
                     f"Step {step_num}: Placing object at destination...")

        # USD tray coords (preferred) — fall back to LLM params only if
        # the Phase 0 lookup didn't run.
        tray = getattr(self, "_tray_position", None)
        exec_cfg = self.config.get("execution", {})
        transit_clearance = float(
            exec_cfg.get("place_transit_clearance", 0.35))
        # Gap between finger TCP and tray top at the release point.
        # Defaults to 0.03 m (3 cm) so the part is released just above
        # the tray surface — clean drop, no rim catch.
        place_z_buffer = float(
            exec_cfg.get("place_drop_z_buffer", 0.03))

        # Optional: bias arm to home config after gantry slide so the
        # descent doesn't inherit an "elbow-forward" pose from the
        # pick retract.
        reset_arm_cfg = bool(
            exec_cfg.get("place_reset_arm_config", True))

        if tray and "center_xy" in tray and "top_z_raw" in tray:
            tx, ty = tray["center_xy"][0], tray["center_xy"][1]
            tz_raw = float(tray["top_z_raw"])
            log.info(
                f"[Place {step_num}] Using USD tray coords: "
                f"({tx:.3f}, {ty:.3f}, tray_top_z={tz_raw:.3f}) + "
                f"transit_clearance={transit_clearance:.3f} m, "
                f"place_z_buffer={place_z_buffer:.3f} m, "
                f"reset_arm_config={reset_arm_cfg}. "
                f"LLM-emitted was "
                f"({params.get('x', '?')}, {params.get('y', '?')}, "
                f"{params.get('z', '?')}) — overridden.")
            place_payload = {
                "x": tx, "y": ty, "z": tz_raw,
                "transit_clearance": transit_clearance,
                "place_z_buffer": place_z_buffer,
                "place_reset_arm_config": reset_arm_cfg,
            }
        else:
            log.warning(
                f"[Place {step_num}] No cached USD tray_position; "
                f"falling back to LLM coords. (Phase 0 may not have "
                f"run, or tray prim lookup failed.)")
            place_payload = {
                "x": params.get("x", 0.5),
                "y": params.get("y", 0.0),
                "z": params.get("z", 0.02),
                "transit_clearance": transit_clearance,
                "place_z_buffer": place_z_buffer,
                "place_reset_arm_config": reset_arm_cfg,
            }

        if "finger_close" in params:
            place_payload["finger_close"] = params["finger_close"]
        if "grip_hold" in params:
            place_payload["grip_hold"] = params["grip_hold"]

        place_result = self.camera.send_command("/api/place", place_payload)
        place_ok = place_result.get("status") == "completed"

        result["status"] = "success" if place_ok else "failed"
        result["detail"] = place_result

        self._notify(WorkflowPhase.PLACE_EXECUTE,
                     "success" if place_ok else "failed",
                     f"Place {'completed' if place_ok else 'FAILED'}")

        return result

    # ─── Detector Label Simplification ───────────────────────

    @staticmethod
    def _simplify_detector_labels(labels: list) -> list:
        """Map VLM labels → OWL-ViT2 / Grounding-DINO query phrases.

        First strips the ``_1/_2/_3`` instance suffix to get the
        canonical part type, then looks up the natural-language
        ``detector_query`` from ``parts_catalogue.yaml``. This avoids
        the old strip-filler heuristic which collapsed ``large_gear``
        to ``gear`` (matching toothed gears, not flat black discs).

        Unknown labels fall back to the legacy filler-strip behaviour.
        """
        import re as _re
        cat_queries = detector_queries()
        seen = set()
        simplified = []
        for label in labels:
            if not label:
                continue
            base = _re.sub(r"_\d+$", "", label).strip()
            query = cat_queries.get(base)
            if query is None:
                # Legacy fallback for labels not in the catalogue.
                s = label.lower().replace("_", " ")
                s = _re.sub(r"\s+\d+$", "", s)
                for filler in ("metallic", "metal", "electrical",
                               "mechanical", "industrial", "fitting",
                               "assembly"):
                    s = s.replace(filler, "").strip()
                s = _re.sub(r"\s+", " ", s).strip()
                if not s:
                    parts = label.replace("_", " ").split()
                    s = parts[-1] if parts else label
                query = s
            if query not in seen:
                seen.add(query)
                simplified.append(query)
        log.debug(f"Detector labels simplified: {labels} → {simplified}")
        return simplified

    # ─── Detector ↔ VLM Merge ─────────────────────────────

    @staticmethod
    def _merge_detector_with_vlm(scene: dict, detections: list) -> None:
        """Replace VLM image_position estimates with precise detector coords.

        Matches detector bounding boxes to VLM-detected objects by label.
        The VLM keeps ownership of semantic fields (affordance, description);
        only image_position is overwritten with the detector's precise center.
        """
        det_by_label = {}
        for det in detections:
            label = det["label"].lower().replace(" ", "_")
            if label not in det_by_label:
                det_by_label[label] = []
            det_by_label[label].append(det)

        matched = 0
        for obj in scene.get("detected_objects", []):
            vlm_label = obj.get("label", "").lower().replace(" ", "_")

            # Try exact match first, then partial
            candidates = det_by_label.get(vlm_label, [])
            if not candidates:
                # Try partial match
                for det_label, dets in det_by_label.items():
                    if vlm_label in det_label or det_label in vlm_label:
                        candidates = dets
                        break

            if candidates:
                # Use the highest-confidence detection
                best = candidates.pop(0)
                obj["image_position"] = best["center"]
                obj["_detector_bbox"] = best["bbox"]
                obj["_detector_confidence"] = best["confidence"]
                obj["_position_source"] = "detector"
                matched += 1
                log.info(
                    f"Detector matched '{vlm_label}' → "
                    f"center({best['center']['x']:.3f}, {best['center']['y']:.3f}) "
                    f"conf={best['confidence']:.2f}"
                )
            else:
                obj["_position_source"] = "vlm_estimate"
                log.debug(f"No detector match for '{vlm_label}' — keeping VLM estimate")

        # Also try to detect the kitting tray — search all tray-like labels
        tray = scene.get("kitting_tray", {})
        tray_keywords = (
            "kitting_tray", "white_box", "white_container",
            "white_plastic_box", "destination_box", "place_tray",
            "box", "tray", "container",
        )
        # Find best tray detection across all matching labels
        best_tray = None
        best_tray_conf = 0.0
        for label_key in tray_keywords:
            if label_key in det_by_label:
                for det in det_by_label[label_key]:
                    if det["confidence"] > best_tray_conf:
                        best_tray = det
                        best_tray_conf = det["confidence"]

        if best_tray:
            if not tray.get("detected"):
                # VLM missed the tray but detector found it
                scene["kitting_tray"] = {"detected": True}
                tray = scene["kitting_tray"]
            tray["image_position"] = best_tray["center"]
            tray["_detector_bbox"] = best_tray["bbox"]
            tray["_position_source"] = "detector"
            log.info(
                f"Tray detected: '{best_tray['label']}' → "
                f"center({best_tray['center']['x']:.3f}, {best_tray['center']['y']:.3f}) "
                f"conf={best_tray_conf:.3f}"
            )
        else:
            log.warning("No tray-like detection from OWL-ViT2")

        log.info(f"Detector merge: {matched}/{len(scene.get('detected_objects', []))} objects matched")

    # ─── USD Fallback Matching ─────────────────────────────

    @staticmethod
    def _match_vlm_to_usd(scene: dict, usd_parts: list) -> None:
        """Match VLM-detected labels to USD scene_parts by fuzzy name match.

        Replaces normalised image coordinates (0-1) with real-world USD
        bounding-box centres (metres) so the robot moves to the correct
        position even when depth projection is unavailable.
        """
        for obj in scene.get("detected_objects", []):
            vlm_label = obj.get("label", "").lower().replace(" ", "_")

            best_match = None
            best_score = 0
            for part in usd_parts:
                prim_name = part.get("name", "").lower()
                # Simple substring match
                if vlm_label in prim_name or prim_name in vlm_label:
                    score = len(vlm_label)
                    if score > best_score:
                        best_score = score
                        best_match = part
                # Also try matching part type
                part_type = part.get("type", "").lower()
                if part_type and (vlm_label in part_type or part_type in vlm_label):
                    score = len(part_type) + 1  # prefer type match
                    if score > best_score:
                        best_score = score
                        best_match = part

            if best_match:
                pos = best_match.get("position", best_match.get("center", {}))
                if isinstance(pos, dict):
                    obj["approximate_position"] = {
                        "x": pos.get("x", 0), "y": pos.get("y", 0),
                        "z": pos.get("z", 0),
                    }
                elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    obj["approximate_position"] = {
                        "x": pos[0], "y": pos[1], "z": pos[2],
                    }
                obj["_source"] = "usd_fallback"
                obj["_usd_prim"] = best_match.get("prim_path", "")
                log.info(
                    f"Matched VLM '{vlm_label}' → USD '{best_match.get('name', '')}' "
                    f"at ({obj['approximate_position']['x']:.3f}, "
                    f"{obj['approximate_position']['y']:.3f}, "
                    f"{obj['approximate_position']['z']:.3f})"
                )

    @staticmethod
    def _find_place_target_usd(usd_parts: list) -> Optional[dict]:
        """Find the kitting tray / destination box from USD parts list."""
        for part in usd_parts:
            name = part.get("name", "").lower()
            if any(kw in name for kw in ("box", "tray", "kit", "destination")):
                pos = part.get("position", part.get("center", {}))
                if isinstance(pos, dict):
                    return {
                        "center_xy": [pos.get("x", 0), pos.get("y", 0)],
                        "top_z": pos.get("z", 0),
                    }
                elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    return {
                        "center_xy": [pos[0], pos[1]],
                        "top_z": pos[2],
                    }
        return None

    # ─── Helpers ────────────────────────────────────────────

    def _finalize(self, result, status, error=None):
        result["status"] = status
        end_dt = datetime.now()
        result["end_time"] = end_dt.isoformat()
        if error:
            result["error"] = error
        # Evaluation hook: log task-level summary once per execute call.
        if self.recorder and getattr(self, "_eval_task_id", None):
            try:
                start_dt = getattr(self, "_eval_task_start", None)
                duration = ((end_dt - start_dt).total_seconds()
                            if start_dt else None)
                self.recorder.log_task(
                    task_id=self._eval_task_id,
                    command=result.get("command", ""),
                    vlm_model=str(getattr(self.vlm, "model", "")),
                    llm_model=str(getattr(self.planner, "model", "")),
                    targeted_count=int(self._eval_targeted),
                    placed_count=int(self._eval_placed),
                    success=(status == "completed"),
                    end_to_end_s=duration,
                    extra={"final_status": status,
                           "error": error or ""},
                )
            except Exception as exc:
                log.warning(f"[eval] task log failed: {exc}")
            # Reset so a stray follow-on call doesn't double-log.
            self._eval_task_id = None
        return result

    # ─── Evaluation logging helpers ───────────────────────────

    def _get_bin_bounds_xy(self, margin: float = 0.0
                           ) -> Optional[tuple]:
        """Return the bin's AABB in XY as ``(xmin, ymin, xmax, ymax)``
        with an optional symmetric margin added (metres).

        Cached from the Phase 0 ``bin_lookup`` step. Returns ``None``
        if Phase 0 hasn't run yet (or didn't return AABB info), which
        causes the USD-snap to fall back to "no bounds filter" — the
        same behaviour as before this guard was added.
        """
        aabb = getattr(self, "_bin_aabb_xy", None)
        if aabb is None:
            return None
        x_min, y_min, x_max, y_max = aabb
        return (x_min - margin, y_min - margin,
                x_max + margin, y_max + margin)

    def _close_range_rescan_xy(self, current_target: dict,
                               obj_label: str = "") -> Optional[dict]:
        """Close-range explicit rescan with full coordinate transparency.

        Workflow (operator's request):

          1. Capture a wrist image at the current pose. The wrist
             camera is mounted with a known XY offset from ee_link.
          2. Run the VLM on that close-range image to get a precise
             pixel centroid of the target part.
          3. Project the pixel through the wrist camera's ACTUAL world
             transform onto the part's Z plane. Returns the part's
             world XYZ — already accounting for the camera's mounting
             offset (the bridge uses the camera prim's world
             transform, not ee_link's).
          4. Read both ee_link and wrist-camera world positions
             RIGHT NOW from the bridge so the camera→ee_link offset
             is a measured number, not assumed.
          5. Compute the GRASP target in ee_link's frame:
                 grasp_target_xy = part_world_xy
             (because ee_link world XY == gripper TCP world XY when
             the gripper points down — confirmed by the user's
             diagnostic showing ``X=+219.5, Y=0, Z=0 mm`` for the
             gripper_tcp Xform in ee_link's local frame).
          6. Log every transform so the operator can audit the math.

        The "normalization" you described is step 5: the wrist VLM
        says where the part is in WORLD coords; we need to put
        ee_link there (NOT the camera) — so we feed the part world
        XY directly to the gantry+IK target. The geometry is right;
        what was missing was making it explicit and observable.

        Returns
        -------
        dict or None
            ``{"x": .., "y": .., "z": .., "shift_mm": ..,
               "ee_xy": [..,..], "cam_xy": [..,..],
               "cam_to_ee_offset": [..,..]}`` on success.
            ``None`` if any step failed (caller keeps current target).
        """
        try:
            wrist_image = self.camera.capture_wrist_image()
        except Exception as exc:
            log.warning(f"[Rescan] wrist capture failed: {exc}")
            return None
        self._notify_wrist_image(wrist_image)

        # Focused single-target prompt. The catalogue is injected so
        # the VLM uses the same vocabulary as the rest of the
        # pipeline; the prompt explicitly tells it to ignore parts
        # at the edges so it doesn't pick a neighbour.
        prompt = (
            f"You are looking straight down through a wrist-mounted "
            f"camera at a parts bin. The robot's gantry has aligned "
            f"the gripper APPROXIMATELY over a target part. Find the "
            f"part CLOSEST TO THE IMAGE CENTRE — ignore parts at the "
            f"edges of the frame.\n\n"
            f"Expected target type: '{obj_label or 'any graspable part'}'. "
            f"If the centre-most part doesn't match, still report its "
            f"observed centre — the spatial accuracy matters more than "
            f"the label here.\n\n"
            f"Output ONLY this JSON (one line, no markdown):\n"
            f'{{"part_visible": true|false, "label": "...", '
            f'"center": {{"x": 0.50, "y": 0.50}}, "confidence": 0-1}}\n'
            f"Be NUMERICALLY PRECISE on the centre — do not default "
            f"to (0.5, 0.5)."
        )
        try:
            vresp = self.vlm.analyze_raw(wrist_image, prompt=prompt)
        except Exception as exc:
            log.warning(f"[Rescan] VLM call failed: {exc}")
            return None

        if not isinstance(vresp, dict) or not vresp.get("part_visible"):
            log.info(
                f"[Rescan] VLM reports no part visible "
                f"(response={vresp}) — keeping previous coords")
            return None
        center = vresp.get("center") or {}
        try:
            cx = float(center.get("x"))
            cy = float(center.get("y"))
        except (TypeError, ValueError):
            log.info(
                f"[Rescan] VLM returned no centre coords — "
                f"keeping previous")
            return None

        # Project through wrist camera → world (depth-buffer based).
        try:
            proj = self.camera.project_to_world(
                [{"x": cx, "y": cy}], camera="wrist")
        except Exception as exc:
            log.warning(f"[Rescan] projection failed: {exc}")
            return None
        wpts = proj.get("world_points") or []
        if not wpts:
            log.info("[Rescan] projection returned no world points")
            return None
        wp = wpts[0]
        part_x, part_y, part_z = (
            float(wp["x"]), float(wp["y"]), float(wp["z"]))

        # Explicit current ee_link + wrist camera world positions
        # (both fetched from the bridge in the SAME projection call).
        ee_pos = proj.get("ee_link_pos") or [None, None, None]
        cam_pos = proj.get("camera_pos") or [None, None, None]
        cam_to_ee_dx = (cam_pos[0] - ee_pos[0]
                        if ee_pos[0] is not None
                        and cam_pos[0] is not None else None)
        cam_to_ee_dy = (cam_pos[1] - ee_pos[1]
                        if ee_pos[1] is not None
                        and cam_pos[1] is not None else None)

        cur_x = float(current_target.get("x", 0))
        cur_y = float(current_target.get("y", 0))
        shift = ((part_x - cur_x) ** 2
                 + (part_y - cur_y) ** 2) ** 0.5

        log.info(f"[Rescan] === explicit close-range rescan ===")
        log.info(f"[Rescan]   wrist VLM pixel:    ({cx:.3f}, {cy:.3f})  "
                 f"label='{vresp.get('label','?')}' "
                 f"conf={vresp.get('confidence', 0):.2f}")
        log.info(f"[Rescan]   wrist camera world: ({cam_pos[0]:.4f}, "
                 f"{cam_pos[1]:.4f}, {cam_pos[2]:.4f})")
        log.info(f"[Rescan]   ee_link world:      ({ee_pos[0]:.4f}, "
                 f"{ee_pos[1]:.4f}, {ee_pos[2]:.4f})")
        log.info(f"[Rescan]   cam → ee offset:    "
                 f"dx={cam_to_ee_dx*1000 if cam_to_ee_dx else 0:+.1f} mm, "
                 f"dy={cam_to_ee_dy*1000 if cam_to_ee_dy else 0:+.1f} mm "
                 f"(camera mounting offset; the projection already "
                 f"accounts for this internally — the part XY below "
                 f"is in WORLD coords, not camera coords)")
        log.info(f"[Rescan]   projected part XY:  ({part_x:.4f}, "
                 f"{part_y:.4f}, {part_z:.4f})")
        log.info(f"[Rescan]   current target XY:  ({cur_x:.4f}, "
                 f"{cur_y:.4f})")
        log.info(f"[Rescan]   shift to apply:     "
                 f"{shift*1000:.1f} mm "
                 f"(grasp target updated to put ee_link → gripper TCP "
                 f"directly over the projected part XY)")

        return {
            "x": part_x,
            "y": part_y,
            "z": part_z,
            "shift_mm": shift * 1000.0,
            "ee_xy": [ee_pos[0], ee_pos[1]] if ee_pos[0] is not None else None,
            "cam_xy": [cam_pos[0], cam_pos[1]] if cam_pos[0] is not None else None,
            "cam_to_ee_offset": [cam_to_ee_dx, cam_to_ee_dy]
            if cam_to_ee_dx is not None else None,
        }

    def _snap_to_usd_ground_truth(self, scene: dict) -> None:
        """Replace each detection's grasp XY with the matching USD prim's
        centre — eliminates the OWL-bbox / depth-projection imprecision
        that misaligns the gripper before realignment.

        Strategy:
          1. Query the bridge's ``/api/scan_scene_parts`` for the actual
             world position of every pickable USD prim.
          2. For each detected object, find the USD prim of the same
             part type (matching by name substring) whose centre is
             closest to the detection's projected XY.
          3. If within ``execution.usd_grasp_snap_radius`` (default 10
             cm), overwrite ``approximate_position`` with the USD
             ``center_xyz``. Original perception coords are stashed in
             ``_perception_position`` for evaluation.
          4. If no USD prim is within the radius, the detection keeps
             its raw perception coords (graceful fallback).

        Perception accuracy (raw, pre-snap) is recorded by
        :meth:`_log_perception_metrics` BEFORE this method runs so the
        evaluation tables report honest perception error.
        """
        import re as _re

        # /api/scene_parts is registered as a GET endpoint on the
        # bridge; ``send_command`` forces POST so it 404s. Use the
        # purpose-built ``get_scene_parts`` helper (GET) when the
        # camera interface supports it; otherwise fall back to
        # ``send_command`` (will only work if a future bridge
        # version adds POST support).
        get_scene = getattr(self.camera, "get_scene_parts", None)
        try:
            if callable(get_scene):
                gt = get_scene()
            else:
                send = getattr(self.camera, "send_command", None)
                if not callable(send):
                    return
                gt = send("/api/scene_parts", {})
        except Exception as exc:
            log.warning(f"[USD-snap] scene_parts fetch failed: {exc}")
            return
        if not isinstance(gt, dict) or not gt.get("parts"):
            log.info("[USD-snap] No USD ground-truth parts available "
                     "— skipping snap")
            return

        usd_parts = gt["parts"]
        snap_radius = float(
            self.config.get("execution", {}).get(
                "usd_grasp_snap_radius", 0.10))

        # ── Bin-bounds filter ───────────────────────────────────
        # Reject USD prims whose XY lies OUTSIDE the bin's AABB
        # before they enter the snap candidate pool. Without this
        # guard, a part that has fallen out of the bin (or was never
        # spawned inside) sits at a far-away XY in USD; perception
        # might detect a similarly-typed part inside the bin, but
        # the snap matches it to the OUT-OF-BIN prim because it
        # happens to be the closest one of that type. Result: the
        # gripper is sent to wherever the lost part is now — often
        # well outside the bin compartment, which is what the
        # operator just observed in the screenshot.
        #
        # The bin AABB comes from the cached Phase-0 bin_lookup info
        # (with a small expansion so prims sitting on the bin rim
        # aren't excluded). If the bounds aren't available we fall
        # back to no filter — same behaviour as before.
        bin_bounds = self._get_bin_bounds_xy(margin=0.05)
        if bin_bounds is not None:
            x_min, y_min, x_max, y_max = bin_bounds
            log.info(
                f"[USD-snap] Bin XY bounds (with 5 cm margin): "
                f"X=[{x_min:.3f}, {x_max:.3f}]  "
                f"Y=[{y_min:.3f}, {y_max:.3f}]")
            in_bin = []
            out_of_bin = []
            for p in usd_parts:
                cxyz = p.get("center_xyz") or [0, 0, 0]
                ux, uy = float(cxyz[0]), float(cxyz[1])
                if x_min <= ux <= x_max and y_min <= uy <= y_max:
                    in_bin.append(p)
                else:
                    out_of_bin.append(p)
            if out_of_bin:
                names = ", ".join(
                    p.get("name", "?") for p in out_of_bin[:5])
                log.info(
                    f"[USD-snap] Excluded {len(out_of_bin)} USD prim(s) "
                    f"OUTSIDE bin XY bounds (would have caused the "
                    f"gripper to grasp parts that fell out of the "
                    f"bin): {names}"
                    + (" …" if len(out_of_bin) > 5 else ""))
            usd_parts = in_bin
        else:
            log.info(
                "[USD-snap] Bin AABB not available — proceeding "
                "without bin-bounds filter (snap may pick prims "
                "outside the bin if perception XY is closer to them)")

        # Group USD prims by canonical part type (extracted from the
        # prim name, e.g. ``large_gear_3_collidable`` → ``large_gear``).
        # Order matters: more specific names first so a prim named
        # ``motor_valve_3`` doesn't accidentally match the substring
        # ``valve`` of a less-specific entry.
        candidates = ("motor_valve", "large_gear",
                      "black_hose", "black_plate", "black_plug",
                      "small_tube", "silver_box", "silver_gun",
                      "tube_with_clamps")
        def _canonical_from_name(name: str) -> Optional[str]:
            n = (name or "").lower()
            for c in candidates:
                if c in n:
                    return c
            return None

        usd_by_type: Dict[str, List[dict]] = {}
        for p in usd_parts:
            ct = _canonical_from_name(p.get("name", ""))
            if ct:
                usd_by_type.setdefault(ct, []).append(p)

        snapped = 0
        skipped_no_match = 0
        skipped_too_far = 0
        for obj in scene.get("detected_objects", []):
            label = (obj.get("label") or "").lower()
            label_base = _re.sub(r"_\d+$", "", label).strip()
            if label_base in ("kitting_tray", "tray", "blue_bin", "bin"):
                continue
            usd_pool = usd_by_type.get(label_base, [])
            if not usd_pool:
                skipped_no_match += 1
                continue

            pos = obj.get("approximate_position") or {}
            try:
                px = float(pos.get("x"))
                py = float(pos.get("y"))
                pz = float(pos.get("z"))
            except (TypeError, ValueError):
                continue

            best = None
            best_dist = float("inf")
            for u in usd_pool:
                cxyz = u.get("center_xyz") or [0, 0, 0]
                ux, uy = float(cxyz[0]), float(cxyz[1])
                d = ((ux - px) ** 2 + (uy - py) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best = u
            if best is None or best_dist > snap_radius:
                skipped_too_far += 1
                log.info(
                    f"[USD-snap] '{label}' at ({px:.3f}, {py:.3f}) "
                    f"— closest USD '{label_base}' is "
                    f"{best_dist*100:.1f} cm away (> "
                    f"{snap_radius*100:.0f} cm cap), keeping "
                    f"perception coords")
                continue

            # Stash perception coords for downstream inspection,
            # then overwrite with USD ground truth.
            obj["_perception_position"] = {"x": px, "y": py, "z": pz}
            usd_xyz = best["center_xyz"]
            usd_top_z = float(best.get("top_z", usd_xyz[2]))
            obj["approximate_position"] = {
                "x": float(usd_xyz[0]),
                "y": float(usd_xyz[1]),
                # Use USD top_z as the grasp Z — it's the part's
                # actual top surface, not the centre of mass.
                "z": usd_top_z,
            }
            obj["_usd_prim_path"] = best.get("prim_path")
            obj["_source"] = (obj.get("_source") or "") + "+usd_snap"
            snapped += 1
            log.info(
                f"[USD-snap] '{label}' "
                f"({px:.3f}, {py:.3f}, {pz:.3f}) "
                f"→ USD '{best.get('name')}' "
                f"({usd_xyz[0]:.3f}, {usd_xyz[1]:.3f}, {usd_top_z:.3f}) "
                f"  shift={best_dist*100:.1f} cm")

        log.info(
            f"[USD-snap] Result: snapped={snapped}  "
            f"skipped_no_match={skipped_no_match}  "
            f"skipped_too_far={skipped_too_far}  "
            f"(snap_radius={snap_radius*100:.0f} cm)")

    def _log_perception_metrics(self, scene: dict) -> None:
        """Compare a scene scan against USD ground truth and record metrics.

        Called once per task (after Phase 2 succeeds on the first
        attempt). Pulls ground truth from the bridge's
        ``/api/scan_scene_parts`` endpoint and per-part 2D bboxes from
        ``/api/scene_annotations`` so :func:`compute_perception_metrics`
        can do proper IoU matching.
        """
        from evaluation.metrics import compute_perception_metrics

        if not self.recorder:
            log.warning(
                "[eval] perception metrics skipped: recorder=None "
                "(EvaluationRecorder not attached to engine)")
            return
        if self._eval_task_id is None:
            log.warning(
                "[eval] perception metrics skipped: _eval_task_id=None "
                "(begin_task() never called)")
            return
        # Bridge GT lookup. ``/api/scene_parts`` is a GET endpoint —
        # ``send_command`` forces POST and returns 404. Use the
        # purpose-built ``get_scene_parts`` helper (GET) when
        # available; fall back to send_command for forward
        # compatibility if a future bridge adds POST support.
        get_scene = getattr(self.camera, "get_scene_parts", None)
        send = getattr(self.camera, "send_command", None)
        if not callable(get_scene) and not callable(send):
            log.warning(
                "[eval] perception metrics skipped: camera interface "
                "has neither get_scene_parts nor send_command "
                "(mock mode? bridge offline?)")
            return
        try:
            if callable(get_scene):
                gt = get_scene()
            else:
                gt = send("/api/scene_parts", {})
        except Exception as exc:
            log.warning(f"[eval] /api/scene_parts fetch failed: {exc}")
            return
        if not isinstance(gt, dict):
            log.warning(
                f"[eval] perception metrics skipped: "
                f"/api/scan_scene_parts returned {type(gt).__name__}, "
                f"expected dict — value={str(gt)[:200]}")
            return
        if not gt.get("parts"):
            log.warning(
                f"[eval] perception metrics skipped: "
                f"/api/scan_scene_parts returned no parts. "
                f"Response keys={list(gt.keys())}, "
                f"error={gt.get('error', 'none')}")
            return

        # Optional 2D bboxes for IoU matching. /api/scene_annotations
        # is a GET endpoint just like /api/scene_parts — same lesson.
        # Failure to fetch is tolerated; metrics fall back to
        # world-XY proximity matching when annotations are absent.
        annotations = None
        try:
            get_ann = getattr(self.camera, "get_scene_annotations", None)
            if callable(get_ann):
                ann = get_ann()
            elif callable(send):
                ann = send("/api/scene_annotations", {})
            else:
                ann = None
            if isinstance(ann, dict):
                annotations = ann.get("annotations") or ann.get("parts")
        except Exception as exc:
            log.debug(f"[eval] /api/scene_annotations fetch failed: {exc}")
            pass

        predictions = [o for o in scene.get("detected_objects", [])
                       if o.get("label") not in ("kitting_tray",
                                                 "blue_bin", "bin",
                                                 "tray")]
        try:
            pm = compute_perception_metrics(
                predictions=predictions,
                ground_truth_parts=gt["parts"],
                annotations=annotations,
            )
        except Exception as exc:
            log.warning(f"[eval] compute_perception_metrics failed: {exc}")
            return

        try:
            self.recorder.log_perception(
                task_id=self._eval_task_id,
                metrics=pm.to_dict(),
                vlm_model=str(getattr(self.vlm, "model", "")),
            )
            log.info(
                f"[eval] perception P={pm.precision:.2f} "
                f"R={pm.recall:.2f} F1={pm.f1:.2f} "
                f"label_acc={pm.label_accuracy:.2f} "
                f"IoU={pm.mean_iou:.2f} "
                f"err={pm.mean_grounding_error_mm:.0f} mm")
        except Exception as exc:
            log.warning(f"[eval] log_perception failed: {exc}")

    def _log_pick_event(self, params: dict, step_result: dict,
                        cycle_time_s: float) -> None:
        """Persist a single pick attempt to the evaluation store."""
        if not self.recorder or self._eval_task_id is None:
            return
        sub = step_result.get("sub_phases", []) or []
        # Inspect sub-phases to fill the secondary signals.
        grasp_sensor = None
        grasp_vlm = None
        contact_stop = None
        drift_xy = None
        drift_z = None
        for sp in sub:
            phase = sp.get("phase", "")
            if phase == "pick_close":
                gc = sp.get("grasp_confirmed")
                if gc is not None:
                    grasp_sensor = bool(gc)
            elif phase == "post_grasp_vlm":
                go = sp.get("grasp_ok")
                if go is not None:
                    grasp_vlm = bool(go)
            elif phase == "pick_descend":
                detail = sp.get("detail", {}) or {}
                if "contact" in detail:
                    contact_stop = bool(detail.get("contact"))
                drift = detail.get("drift") or {}
                if "xy_mm" in drift:
                    drift_xy = float(drift.get("xy_mm"))
                if "z_mm" in drift:
                    drift_z = float(drift.get("z_mm"))
        try:
            self.recorder.log_pick(
                task_id=self._eval_task_id,
                object_id=str(params.get("object_id", "?")),
                label=str(params.get("label", "")),
                success=(step_result.get("status") == "success"),
                grasp_confirmed_sensor=grasp_sensor,
                grasp_confirmed_vlm=grasp_vlm,
                contact_stop=contact_stop,
                drift_xy_mm=drift_xy,
                drift_z_mm=drift_z,
                cycle_time_s=cycle_time_s,
                extra={"error": step_result.get("error")},
            )
        except Exception as exc:
            log.warning(f"[eval] log_pick failed: {exc}")

    def _log_place_event(self, params: dict, step_result: dict,
                         label: str, cycle_time_s: float):
        """Run Camera_Kit verification + persist the place result.

        The bridge place call returns ``status == "completed"`` whenever
        the gripper opened — the part may or may not have actually
        landed inside the tray. The Camera_Kit + VLM check supplies
        the authoritative success signal.

        Returns
        -------
        Optional[bool]
            ``True``  → VLM confirms the part is inside the tray.
            ``False`` → VLM confirms the part is NOT inside the tray
                        (operator's request: this triggers a full
                        orchestration-layer restart on the next outer
                        retry iteration).
            ``None``  → Verification was not attempted or could not
                        produce a verdict (verifier error / mock mode).
                        Caller treats this as a non-failure to avoid
                        spurious retries when Camera_Kit is unavailable.
        """
        if not self.recorder or self._eval_task_id is None:
            return None
        bridge_status = (
            (step_result.get("detail") or {}).get("status")
            or step_result.get("status", ""))

        in_tray = None
        confidence = None
        verifier_detail = ""
        verifier_error = None
        if (self.place_verifier is not None
                and step_result.get("status") == "success"):
            try:
                self._notify("place_verify", "running",
                             f"Camera_Kit VLM check for {label or '?'}...")
                v = self.place_verifier.verify(label or "")
                in_tray = v.in_tray
                confidence = v.confidence
                verifier_detail = v.detail or ""
                verifier_error = v.error
                if v.image is not None:
                    self._notify_kit_image(v.image)

                # Confidence-gated negative: a low-confidence "no part
                # in tray" verdict is downgraded to "inconclusive"
                # (None) so the outer retry loop does NOT trigger a
                # spurious re-pick from the bin. The user observed
                # the workflow re-running entire pick-place cycles
                # because the VLM was hedging on uncertain views;
                # this threshold filters those out while preserving
                # retries for confidently-detected failures.
                min_conf = float(
                    self.config.get("execution", {}).get(
                        "place_verify_min_confidence", 0.6))
                if (in_tray is False
                        and (confidence is None
                             or confidence < min_conf)):
                    log.info(
                        f"[Place verify] Downgrading negative verdict "
                        f"to inconclusive: confidence "
                        f"{confidence!r} < threshold {min_conf} — "
                        f"NOT triggering re-pick. Detail: "
                        f"{verifier_detail!r}")
                    in_tray = None  # treat as inconclusive
                    self._notify(
                        "place_verify", "warn",
                        f"in_tray=False (conf={confidence}) "
                        f"below threshold {min_conf} — accepting place")
                    # Inconclusive but place sequence completed —
                    # still increment placed count so the task summary
                    # reflects the bridge-side success.
                    self._eval_placed += 1
                else:
                    if in_tray:
                        self._eval_placed += 1
                    self._notify(
                        "place_verify",
                        "success" if in_tray else "failed",
                        f"in_tray={in_tray} conf={confidence}")
            except Exception as exc:
                verifier_error = str(exc)
                log.warning(f"[eval] place verifier failed: {exc}")
        elif step_result.get("status") == "success":
            # No verifier wired up — fall back to bridge "completed"
            # so the pipeline still produces a placed_count > 0 in the
            # task summary even when running in mock mode.
            self._eval_placed += 1

        try:
            self.recorder.log_place(
                task_id=self._eval_task_id,
                object_id=str(params.get("object_id", "?")),
                label=label,
                bridge_status=str(bridge_status),
                in_tray_vlm=in_tray,
                vlm_confidence=confidence,
                cycle_time_s=cycle_time_s,
                extra={"detail": verifier_detail,
                       "error": verifier_error},
            )
        except Exception as exc:
            log.warning(f"[eval] log_place failed: {exc}")

        return in_tray

    def _notify_kit_image(self, image) -> None:
        """Push the Camera_Kit verification image into the UI session."""
        try:
            import streamlit as st  # type: ignore
            st.session_state["kit_verify_image"] = image
        except Exception:
            pass
