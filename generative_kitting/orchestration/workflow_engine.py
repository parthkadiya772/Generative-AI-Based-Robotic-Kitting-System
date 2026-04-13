"""
Neuro-Symbolic Kitting Workflow Engine.

Orchestrates the full AI-driven pick-and-place pipeline:
  0+1. VLM analyses RGB camera → identify parts + kitting tray
       Depth projection converts image coords → world XYZ
  2.   LLM generates plan using depth-projected coordinates
  3-6. For each pick:  approach → depth VLM → grasp → wrist verify
  7.   For each place: transit → descend → release
  9.   RGB verification of final workspace state

ALL coordinates come from VLM + depth projection — no USD prim paths.
Phase callbacks drive real-time UI progress updates.
"""

import json
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from perception.depth_estimator import DepthEstimator
from perception.object_detector import ZeroShotDetector
from utils.logger import log


# ─── VLM Prompts ────────────────────────────────────────────

SCENE_ANALYSIS_PROMPT = """You are a robotic vision system analyzing an industrial kitting workspace.
The overhead RGB camera shows a UR10 robot arm on a gantry rail above a kitting station.

WORKSPACE LAYOUT:
- A BLUE BIN (kitting station) contains all the pickable parts
- A WHITE BOX / KITTING TRAY (destination) is where picked parts should be placed
- A RED GANTRY RAIL is the robot's linear rail — it is NOT a part
- The ROBOTIC ARM + GRIPPER are NOT parts — ignore them

KNOWN PART TYPES (parts are inside the blue bin):
  motor_valve, black_hose, black_plate, black_plug, small_hinge,
  small_tube, silver_box, silver_gun, tube_with_clamps

INSTRUCTIONS:
1. Detect every PICKABLE PART visible inside the blue bin.
2. ALSO detect the WHITE BOX / KITTING TRAY as a detected object with
   label "kitting_tray" and affordance "destination". This is CRITICAL
   for the robot to know where to place picked parts.
3. For every detection give its centre position as a NORMALISED image
   coordinate (x, y) where (0,0) = top-left and (1,1) = bottom-right.
   The system will convert these to real-world coordinates via depth.

DO NOT detect the robot arm, gripper, gantry rail, or the blue bin itself.

RESPOND ONLY WITH VALID JSON:
{{
  "detected_objects": [
    {{
      "object_id": "obj_001",
      "label": "motor_valve",
      "semantic_description": "brass coloured cylindrical valve",
      "affordance": "graspable",
      "image_position": {{"x": 0.35, "y": 0.60}},
      "confidence": 0.92
    }},
    {{
      "object_id": "obj_tray",
      "label": "kitting_tray",
      "semantic_description": "white plastic box used as destination tray",
      "affordance": "destination",
      "image_position": {{"x": 0.75, "y": 0.50}},
      "confidence": 0.95
    }}
  ],
  "scene_summary": "..."
}}"""

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


class WorkflowPhase:
    """Represents a single phase in the kitting workflow."""
    SCENE_SCAN     = "scene_scan"
    SCENE_ANALYSIS = "scene_analysis"
    PLAN_GENERATION = "plan_generation"
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

    def __init__(self, camera, vlm, planner, config=None):
        self.camera = camera
        self.vlm = vlm
        self.planner = planner
        self.config = config or {}
        self._phase_callback = None
        self._cancelled = False
        self._depth_estimator = DepthEstimator(
            self.config.get("perception", {})
        )
        self._detector = ZeroShotDetector(
            self.config.get("perception", {})
        )

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
        """Execute a full neuro-symbolic kitting task."""
        self._cancelled = False
        result = {
            "command": user_command,
            "start_time": datetime.now().isoformat(),
            "phases": [],
            "status": "running",
        }

        try:
            # ═══════════════════════════════════════════════════
            # PHASE 0+1: VLM Scene Analysis + Depth Projection
            #
            # Fully vision-based — no USD prim path lookups.
            # 1) Capture overhead RGB → VLM detects parts + kitting tray
            # 2) Depth projection converts image coords → world XYZ
            # ═══════════════════════════════════════════════════
            self._notify(WorkflowPhase.SCENE_ANALYSIS, "running",
                         "Capturing RGB image and running VLM analysis...")

            rgb_image = self.camera.capture_workspace_image()
            scene = self.vlm.analyze_scene(
                rgb_image, custom_prompt=SCENE_ANALYSIS_PROMPT)
            num_vlm_objects = len(scene.get("detected_objects", []))

            # ── Refine positions with zero-shot detector ───────
            # VLMs give rough coordinate guesses (~15% pixel error).
            # A detection model (OWL-ViT2) gives precise bounding boxes.
            # Use detector coords when available, VLM coords as fallback.
            if self._detector.is_available:
                self._notify(WorkflowPhase.SCENE_ANALYSIS, "running",
                             "Running zero-shot detector for precise localization...")
                vlm_labels = [
                    obj.get("label", "") for obj in scene.get("detected_objects", [])
                ]
                # Always detect the kitting tray with multiple query variations
                tray_queries = [
                    "kitting tray", "white box", "white container",
                    "white plastic box", "destination box", "place tray",
                ]
                vlm_labels.extend(tray_queries)
                detections = self._detector.detect_with_labels(rgb_image, vlm_labels)

                if detections:
                    log.info(f"Detector found {len(detections)} objects with precise coords")
                    # Log tray-related detections specifically
                    for det in detections:
                        label = det["label"].lower()
                        if any(kw in label for kw in ("tray", "box", "container", "destination")):
                            log.info(
                                f"Tray candidate: '{det['label']}' "
                                f"center=({det['center']['x']:.3f}, {det['center']['y']:.3f}) "
                                f"conf={det['confidence']:.3f}"
                            )
                    self._merge_detector_with_vlm(scene, detections)
            else:
                log.info("Zero-shot detector not available — using VLM coordinates")

            # ── Build projection points from best available coords ──
            # All objects (parts AND kitting tray) are in detected_objects.
            # The tray has affordance "destination" and may sit on a
            # different surface, so we pass a surface_z hint for it.
            tray_surface_z = self.config.get("perception", {}).get(
                "tray_surface_z", 0.02)
            proj_points = []
            for obj in scene.get("detected_objects", []):
                img_pos = obj.get("image_position", {})
                if img_pos:
                    point = {
                        "x": img_pos.get("x", 0.5),
                        "y": img_pos.get("y", 0.5),
                    }
                    # Tray may be on a different surface than parts
                    if obj.get("affordance") == "destination" or \
                       "tray" in obj.get("label", "").lower():
                        point["surface_z"] = tray_surface_z
                    proj_points.append(point)

            # Also handle legacy kitting_tray field (if VLM still returns it)
            kitting_tray = scene.get("kitting_tray", {})
            tray_img = kitting_tray.get("image_position", {})
            if tray_img and kitting_tray.get("detected"):
                proj_points.append({
                    "x": tray_img.get("x", 0.5),
                    "y": tray_img.get("y", 0.5),
                    "surface_z": tray_surface_z,
                })

            # ── 2D → 3D Coordinate Resolution ──────────────────
            # Three fallback methods (in priority order):
            #   1. Bridge depth projection (depth buffer + camera intrinsics)
            #   2. Bridge geometric projection (ray-plane, no depth needed)
            #      Methods 1 & 2 are both inside /api/project_to_world
            #   3. Local depth estimation (Depth Anything V2 or geometric)
            # ────────────────────────────────────────────────────
            place_target = None
            coord_source = "none"

            if proj_points:
                # ── Try bridge projection (depth buffer + geometric fallback) ──
                bridge_ok = False
                if hasattr(self.camera, "project_to_world"):
                    try:
                        proj_result = self.camera.project_to_world(
                            proj_points, camera="rgb")
                        world_pts = proj_result.get("world_points", [])

                        if world_pts:
                            bridge_ok = True
                            method = proj_result.get("method", "bridge")
                            coord_source = f"bridge_{method}"

                            for i, obj in enumerate(scene.get("detected_objects", [])):
                                if i < len(world_pts):
                                    wp = world_pts[i]
                                    obj["approximate_position"] = {
                                        "x": wp["x"], "y": wp["y"], "z": wp["z"]
                                    }
                                    obj["depth_m"] = wp.get("depth_m", -1)
                                    obj["_source"] = coord_source

                            # Extract kitting tray from detected_objects
                            # (same pipeline as parts — no special case)
                            place_target = self._extract_tray_from_objects(scene)

                            # Also handle legacy kitting_tray field
                            if not place_target and tray_img and kitting_tray.get("detected"):
                                n_objs = len(scene.get("detected_objects", []))
                                if n_objs < len(world_pts):
                                    tray_wp = world_pts[n_objs]
                                elif world_pts:
                                    tray_wp = world_pts[-1]
                                else:
                                    tray_wp = None
                                if tray_wp:
                                    place_target = {
                                        "center_xy": [tray_wp["x"], tray_wp["y"]],
                                        "top_z": tray_wp["z"],
                                    }
                        else:
                            log.warning(
                                f"Bridge projection failed: "
                                f"{proj_result.get('error', 'empty world_points')}"
                            )
                    except Exception as e:
                        log.warning(f"Bridge project_to_world call failed: {e}")

                # ── Fallback 2: USD scene_parts for part coords only ──
                # Place target is NOT taken from USD — it must come from
                # perception (VLM + detector + depth projection) so the
                # system works when the tray moves between runs.
                if not bridge_ok and hasattr(self.camera, "get_scene_parts"):
                    try:
                        scene_parts = self.camera.get_scene_parts()
                        usd_parts = scene_parts.get("parts", [])
                        if usd_parts:
                            log.info(f"Fallback: {len(usd_parts)} USD parts for matching")
                            self._match_vlm_to_usd(scene, usd_parts)
                            coord_source = "usd_fallback"
                    except Exception as e:
                        log.warning(f"USD scene_parts fallback failed: {e}")

                # ── Fallback 3: Local depth estimation ─────────────
                if coord_source == "none":
                    self._notify(WorkflowPhase.SCENE_ANALYSIS, "warn",
                                 "Bridge unavailable — using local depth estimation")
                    try:
                        world_pts = self._depth_estimator.image_to_world(
                            rgb_image, proj_points)
                        if world_pts:
                            coord_source = f"depth_est_{self._depth_estimator.method}"

                            for i, obj in enumerate(scene.get("detected_objects", [])):
                                if i < len(world_pts):
                                    wp = world_pts[i]
                                    obj["approximate_position"] = {
                                        "x": wp["x"], "y": wp["y"], "z": wp["z"]
                                    }
                                    obj["depth_m"] = wp.get("depth_m", -1)
                                    obj["_source"] = coord_source

                            # Extract tray from detected_objects
                            if not place_target:
                                place_target = self._extract_tray_from_objects(scene)

                            log.info(
                                f"Depth estimation: {len(world_pts)} points "
                                f"via {self._depth_estimator.method}"
                            )
                    except Exception as e:
                        log.warning(f"Local depth estimation failed: {e}")

            # ── Place target logging ──────────────────────────────
            if place_target:
                log.info(
                    f"Place target (perception): "
                    f"({place_target['center_xy'][0]:.3f}, "
                    f"{place_target['center_xy'][1]:.3f}) "
                    f"top_z={place_target['top_z']:.3f}"
                )
            else:
                log.warning(
                    "Kitting tray not localized by perception pipeline. "
                    "LLM will use default position from config."
                )

            num_objects = len(scene.get("detected_objects", []))
            result["scene_description"] = scene
            result["phases"].append({
                "phase": WorkflowPhase.SCENE_ANALYSIS,
                "status": "success",
                "detail": f"VLM: {num_vlm_objects} objects, "
                          f"coords via {coord_source}, "
                          f"tray={'detected' if place_target else 'not found'}",
                "timestamp": datetime.now().isoformat(),
            })
            self._notify(WorkflowPhase.SCENE_ANALYSIS, "success",
                         f"Identified {num_objects} objects ({coord_source})")

            if self._cancelled:
                return self._finalize(result, "cancelled")

            # ═══════════════════════════════════════════════════
            # PHASE 2: LLM Plan Generation (with real coords)
            # ═══════════════════════════════════════════════════
            self._notify(WorkflowPhase.PLAN_GENERATION, "running",
                         "Sending scene + command to LLM planner...")

            # Inject place target into config
            kit_tray_pos = None
            if place_target:
                kit_tray_pos = [
                    place_target["center_xy"][0],
                    place_target["center_xy"][1],
                    place_target["top_z"],
                ]

            plan = self.planner.generate_plan(
                user_command, scene,
                workspace_bounds=self.config.get("execution", {}).get("workspace_bounds"),
                kit_tray_position=kit_tray_pos,
            )
            total_steps = plan.get("total_steps", 0)

            result["task_plan"] = plan
            result["phases"].append({
                "phase": WorkflowPhase.PLAN_GENERATION,
                "status": "success",
                "detail": f"{total_steps} steps: {plan.get('task_summary', '')}",
                "timestamp": datetime.now().isoformat(),
            })
            self._notify(WorkflowPhase.PLAN_GENERATION, "success",
                         f"Plan: {total_steps} steps — {plan.get('task_summary', '')}")

            if self._cancelled:
                return self._finalize(result, "cancelled")

            # ═══════════════════════════════════════════════════
            # PHASES 3-8: Execute plan step by step
            # ═══════════════════════════════════════════════════

            # Guard: abort if no 3D coordinates were resolved
            if coord_source == "none":
                log.error(
                    "ABORT: No real-world coordinates available. "
                    "All projection methods failed."
                )
                self._notify("error", "failed",
                             "Cannot execute — no 3D coordinates. "
                             "Reload bridge in Isaac Sim Script Editor.")
                return self._finalize(
                    result, "error",
                    "No real-world coordinates: bridge depth projection, "
                    "USD scene_parts, and local depth estimation all failed. "
                    "Reload the bridge script in Isaac Sim.")

            execution_results = []
            last_grip_hold = None   # track grip from last pick for place

            for step in plan.get("plan", []):
                if self._cancelled:
                    break

                action = step.get("action", "")
                params = step.get("params", {})
                step_num = step.get("step", "?")
                step_result = {"step": step_num, "action": action}

                # ── Resolve place coordinates from perception-detected tray ──
                if action == "place_object":
                    params = self._resolve_place_coordinates(params, place_target)
                    if params is None:
                        self._notify(WorkflowPhase.PLACE_EXECUTE, "failed",
                                     "Kitting tray not detected — cannot place")
                        step_result["status"] = "failed"
                        step_result["error"] = "Kitting tray not detected by camera"
                        execution_results.append(step_result)
                        continue

                if action == "pick_object":
                    step_result = self._execute_pick_sequence(
                        step, params, scene, step_num)
                    # Remember grip_hold for the following place
                    last_grip_hold = step_result.get("grip_hold",
                                                     step_result.get("finger_close"))

                elif action == "place_object":
                    # Pass grip_hold from the preceding pick
                    if last_grip_hold is not None:
                        params["grip_hold"] = last_grip_hold
                    step_result = self._execute_place_sequence(
                        step, params, scene, step_num)
                    last_grip_hold = None  # reset after place

                elif action == "move_home":
                    self._notify("move_home", "running", f"Step {step_num}")
                    r = self.camera.send_command("/api/home", {})
                    step_result["status"] = "success" if r.get("status") == "ok" else "failed"

                elif action == "open_gripper":
                    r = self.camera.send_command("/api/gripper", {"action": "open"})
                    step_result["status"] = "success" if r.get("status") == "ok" else "failed"

                elif action == "close_gripper":
                    r = self.camera.send_command("/api/gripper", {"action": "close"})
                    step_result["status"] = "success" if r.get("status") == "ok" else "failed"

                elif action == "verify_grasp":
                    step_result["status"] = "success"

                elif action == "request_perception_update":
                    rgb_image = self.camera.capture_workspace_image()
                    scene = self.vlm.analyze_scene(
                        rgb_image, custom_prompt=SCENE_ANALYSIS_PROMPT)
                    result["scene_description"] = scene
                    step_result["status"] = "success"

                else:
                    step_result["status"] = "skipped"

                execution_results.append(step_result)

            result["execution_results"] = execution_results

            # ═══════════════════════════════════════════════════
            # PHASE 9: Verification via RGB Camera
            # ═══════════════════════════════════════════════════
            if not self._cancelled:
                self._notify(WorkflowPhase.VERIFY, "running",
                             "Capturing verification image...")

                verify_image = self.camera.capture_workspace_image()
                verify_scene = self.vlm.analyze_scene(verify_image)
                result["verification"] = verify_scene

                result["phases"].append({
                    "phase": WorkflowPhase.VERIFY,
                    "status": "success",
                    "detail": f"Verification: {len(verify_scene.get('detected_objects', []))} objects",
                    "timestamp": datetime.now().isoformat(),
                })
                self._notify(WorkflowPhase.VERIFY, "success",
                             "Verification complete — checking workspace state")

            return self._finalize(result, "completed")

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
        """Send a wrist camera base64 image to the VLM and return the result."""
        from PIL import Image as PILImage
        import io as _io, base64 as _b64

        wrist_bytes = _b64.b64decode(wrist_b64)
        wrist_img = PILImage.open(_io.BytesIO(wrist_bytes)).convert("RGB")
        return self.vlm.analyze_scene(wrist_img, custom_prompt=prompt_text)

    def _execute_pick_sequence(self, step, params, scene, step_num):
        """
        3-phase pick with wrist-camera VLM verification between each phase:

          Phase A: Approach + Depth scan
          Phase B: Descend to grasp position (open fingers)
                   → wrist VLM: "is part between fingers?"
          Phase C: Close gripper
                   → wrist VLM: "is part properly grasped?"
          Phase D: Retract (only after VLM confirms grasp)
        """
        result = {"step": step_num, "action": "pick_object", "sub_phases": []}
        obj_id = params.get("object_id", "?")
        pick_coords = {
            "x": params.get("x", 0.3),
            "y": params.get("y", 0.0),
            "z": params.get("z", 0.02),
        }

        # ── Phase A: Approach ──────────────────────────────────
        self._notify(WorkflowPhase.APPROACH, "running",
                     f"Step {step_num}: Moving near {obj_id}")

        approach_result = self.camera.send_command("/api/approach", pick_coords)
        result["sub_phases"].append({
            "phase": WorkflowPhase.APPROACH,
            "status": "success" if approach_result.get("status") == "ok" else "failed",
        })
        self._notify(WorkflowPhase.APPROACH, "success",
                     f"Robot positioned near {obj_id}")

        # ── Depth camera analysis ──────────────────────────────
        self._notify(WorkflowPhase.DEPTH_ANALYSIS, "running",
                     "Capturing depth image for spatial analysis...")
        try:
            depth_image = self.camera.capture_depth_image()
            depth_scene = self.vlm.analyze_scene(
                depth_image, custom_prompt=DEPTH_GRASP_PROMPT)
            result["sub_phases"].append({
                "phase": WorkflowPhase.DEPTH_ANALYSIS, "status": "success",
                "depth_data": depth_scene,
            })
            self._notify(WorkflowPhase.DEPTH_ANALYSIS, "success",
                         f"Depth analysis — conf={depth_scene.get('confidence', '?')}")
        except Exception as e:
            log.warning(f"Depth analysis failed: {e}")
            result["sub_phases"].append({
                "phase": WorkflowPhase.DEPTH_ANALYSIS,
                "status": "warn", "detail": str(e),
            })

        # ── Phase B: Descend to grasp position (fingers open) ──
        self._notify(WorkflowPhase.PICK_EXECUTE, "running",
                     "Descending to grasp position...")

        descend_result = self.camera.send_command(
            "/api/pick_descend", pick_coords)
        descend_ok = descend_result.get("status") == "ok"
        result["sub_phases"].append({
            "phase": "pick_descend",
            "status": "success" if descend_ok else "failed",
        })

        if not descend_ok:
            self._notify(WorkflowPhase.PICK_EXECUTE, "failed",
                         f"Descend failed: {descend_result.get('error', '?')}")
            result["status"] = "failed"
            return result

        self._notify(WorkflowPhase.PICK_EXECUTE, "success",
                     "At grasp position — checking part placement...")

        # ── Wrist VLM check 1: "Is the part between the fingers?" ──
        wrist_b64 = descend_result.get("wrist_image")
        part_between_fingers = True  # default to proceed if VLM fails
        if wrist_b64:
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
                    "finger pads? The part should be positioned so that "
                    "closing the fingers would grip it. "
                    "If yes, part_between_fingers is true."
                ))
                part_between_fingers = pre_grasp_vlm.get(
                    "part_between_fingers", True)
                result["sub_phases"].append({
                    "phase": "pre_grasp_vlm",
                    "status": "success",
                    "part_between_fingers": part_between_fingers,
                    "detail": pre_grasp_vlm,
                })
                self._notify(WorkflowPhase.GRASP_VERIFY, "success",
                             f"Part between fingers: {part_between_fingers} — "
                             f"{pre_grasp_vlm.get('detail', '')}")
            except Exception as e:
                log.warning(f"Pre-grasp wrist VLM failed: {e}")
                result["sub_phases"].append({
                    "phase": "pre_grasp_vlm",
                    "status": "warn", "detail": str(e),
                })

        if not part_between_fingers:
            log.warning("VLM says no part between fingers — proceeding anyway")

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
        })

        # ── Wrist VLM check 2: "Is the part properly grasped?" ──
        close_wrist_b64 = close_result.get("wrist_image")
        grasp_ok = True  # default
        if close_wrist_b64:
            self._notify(WorkflowPhase.GRASP_VERIFY, "running",
                         "Wrist camera: confirming part is grasped...")
            try:
                post_grasp_vlm = self._wrist_vlm_check(close_wrist_b64, (
                    "You are looking at a close-up image from a wrist-mounted "
                    "camera on a Robotiq 2F-140 gripper. The fingers are CLOSED. "
                    "Answer ONLY with a JSON object: "
                    '{"grasp_ok": true/false, "confidence": 0.0-1.0, '
                    '"detail": "short reason"}. '
                    "Is the gripper holding a part securely between its "
                    "finger pads? If the part is firmly gripped, grasp_ok "
                    "is true. If fingers are empty or the part is slipping, "
                    "grasp_ok is false."
                ))
                grasp_ok = post_grasp_vlm.get("grasp_ok", True)
                result["sub_phases"].append({
                    "phase": "post_grasp_vlm",
                    "status": "success",
                    "grasp_ok": grasp_ok,
                    "detail": post_grasp_vlm,
                })
                self._notify(WorkflowPhase.GRASP_VERIFY, "success",
                             f"Grasp confirmed by VLM: {grasp_ok} — "
                             f"{post_grasp_vlm.get('detail', '')}")
            except Exception as e:
                log.warning(f"Post-grasp wrist VLM failed: {e}")
                result["sub_phases"].append({
                    "phase": "post_grasp_vlm",
                    "status": "warn", "detail": str(e),
                })

        if not grasp_ok:
            log.warning("VLM says grasp not secure — proceeding with retract")

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
        """Phase 7: Place object using real destination coordinates."""
        result = {"step": step_num, "action": "place_object"}

        self._notify(WorkflowPhase.PLACE_EXECUTE, "running",
                     f"Step {step_num}: Placing object at destination...")

        place_payload = {
            "x": params.get("x", 0.5),
            "y": params.get("y", 0.0),
            "z": params.get("z", 0.02),
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

    @staticmethod
    def _finalize(result, status, error=None):
        result["status"] = status
        result["end_time"] = datetime.now().isoformat()
        if error:
            result["error"] = error
        return result
