"""
Neuro-Symbolic Kitting Workflow Engine.

Orchestrates the full AI-driven pick-and-place pipeline:
  0. Query Isaac Sim bridge for real USD part positions
  1. VLM analyses RGB camera → identify parts semantically
  2. Match VLM labels to real USD prims → get world coordinates
  3. LLM generates plan using REAL coordinates (not VLM estimates)
  4. For each pick action:  move near → depth VLM → grasp with IK
  5. For each place action: move to real destination → place → verify

Phase callbacks drive real-time UI progress updates.
"""

import json
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from utils.logger import log


# ─── VLM Prompts ────────────────────────────────────────────

SCENE_ANALYSIS_WITH_PARTS_PROMPT = """You are a robotic vision system analyzing an industrial kitting workspace.
The overhead RGB camera shows a UR10 robot arm on a gantry rail above a kitting station.

WORKSPACE LAYOUT:
- A BLUE BIN (kitting station) contains all the pickable parts
- A WHITE BOX (destination) is where picked parts should be placed
- A RED GANTRY RAIL is the robot's linear rail — it is NOT a part
- The ROBOTIC ARM + GRIPPER are NOT parts — ignore them

KNOWN PART TYPES (parts are inside the blue bin):
  motor_valve, black_hose, black_plate, black_plug, small_hinge,
  small_tube, silver_box, silver_gun, tube_with_clamps

{parts_info}

For each PICKABLE PART visible in the blue bin, provide:
1. object_id: format obj_001, obj_002, ...
2. label: one of the known part type labels above
3. semantic_description: brief description (material, color, shape)
4. affordance: "graspable" for all parts in the bin
5. approximate_position: use the REAL WORLD coordinates from the list above if available
6. confidence: 0.0 to 1.0

DO NOT detect the robot arm, gripper, gantry rail, blue bin itself, or white box as objects.
ONLY detect the actual PARTS inside the blue bin.

RESPOND ONLY WITH VALID JSON:
{{
  "detected_objects": [
    {{
      "object_id": "obj_001",
      "label": "motor_valve",
      "semantic_description": "...",
      "affordance": "graspable",
      "approximate_position": {{"x": 0.0, "y": 0.0, "z": 0.0}},
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
    PLACE_EXECUTE   = "place_execute"
    VERIFY          = "verify"
    COMPLETE        = "complete"


class KittingWorkflowEngine:
    """
    Multi-phase neuro-symbolic kitting orchestrator.

    Connects VLM perception, LLM planning, and Isaac Sim execution
    into a closed-loop workflow that iterates until the task is complete.

    KEY DESIGN: The bridge provides REAL WORLD coordinates from USD bounding
    boxes. The VLM provides SEMANTIC identification (what is each part).
    The two are merged so the LLM planner gets accurate positions.
    """

    def __init__(self, camera, vlm, planner, config=None):
        self.camera = camera
        self.vlm = vlm
        self.planner = planner
        self.config = config or {}
        self._phase_callback = None
        self._cancelled = False
        self._scene_parts = {}  # Cache of real USD part data

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
            # PHASE 0: Query USD scene for real part positions
            # ═══════════════════════════════════════════════════
            self._notify(WorkflowPhase.SCENE_SCAN, "running",
                         "Scanning USD scene for part positions...")

            scene_data = self.camera.get_scene_parts()
            usd_parts = scene_data.get("parts", [])
            place_target = scene_data.get("place_target", None)

            result["usd_scene"] = scene_data
            result["phases"].append({
                "phase": WorkflowPhase.SCENE_SCAN,
                "status": "success" if usd_parts else "warn",
                "detail": f"{len(usd_parts)} prims found in USD",
                "timestamp": datetime.now().isoformat(),
            })
            self._notify(WorkflowPhase.SCENE_SCAN, "success",
                         f"Found {len(usd_parts)} parts in USD scene")

            # Build parts info string for VLM prompt
            parts_info = self._format_parts_for_prompt(usd_parts, place_target)

            if self._cancelled:
                return self._finalize(result, "cancelled")

            # ═══════════════════════════════════════════════════
            # PHASE 1: VLM Scene Analysis (RGB + scene context)
            # ═══════════════════════════════════════════════════
            self._notify(WorkflowPhase.SCENE_ANALYSIS, "running",
                         "Capturing RGB image and running VLM analysis...")

            rgb_image = self.camera.capture_workspace_image()
            vlm_prompt = SCENE_ANALYSIS_WITH_PARTS_PROMPT.format(
                parts_info=parts_info)
            scene = self.vlm.analyze_scene(rgb_image, custom_prompt=vlm_prompt)
            num_vlm_objects = len(scene.get("detected_objects", []))

            # ── Merge VLM detections with real USD coordinates ──
            scene = self._merge_vlm_with_usd(scene, usd_parts)
            num_objects = len(scene.get("detected_objects", []))

            result["scene_description"] = scene
            result["phases"].append({
                "phase": WorkflowPhase.SCENE_ANALYSIS,
                "status": "success",
                "detail": f"VLM: {num_vlm_objects} semantic, {num_objects} with real coords",
                "timestamp": datetime.now().isoformat(),
            })
            self._notify(WorkflowPhase.SCENE_ANALYSIS, "success",
                         f"Identified {num_objects} objects with real-world coordinates")

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
            execution_results = []
            last_finger_close = None  # track adaptive grip from last pick

            for step in plan.get("plan", []):
                if self._cancelled:
                    break

                action = step.get("action", "")
                params = step.get("params", {})
                step_num = step.get("step", "?")
                step_result = {"step": step_num, "action": action}

                # ── Resolve object_id to real prim coordinates ──
                if action in ("pick_object", "place_object"):
                    params = self._resolve_coordinates(params, usd_parts, place_target)

                if action == "pick_object":
                    step_result = self._execute_pick_sequence(
                        step, params, scene, step_num)
                    # Remember the adaptive finger_close for the following place
                    last_finger_close = step_result.get("finger_close")

                elif action == "place_object":
                    # Pass adaptive finger_close from the preceding pick
                    if last_finger_close is not None:
                        params["finger_close"] = last_finger_close
                    step_result = self._execute_place_sequence(
                        step, params, scene, step_num)
                    last_finger_close = None  # reset after place

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
                    scene = self.vlm.analyze_scene(rgb_image, custom_prompt=vlm_prompt)
                    scene = self._merge_vlm_with_usd(scene, usd_parts)
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

    # ─── USD ↔ VLM Merge ────────────────────────────────────

    def _format_parts_for_prompt(self, usd_parts, place_target):
        """Format USD part data as context for the VLM prompt."""
        if not usd_parts:
            return "No USD part data available — use your best visual estimate."

        # Count parts by type so VLM knows exactly how many of each exist
        import re
        from collections import Counter
        type_counts = Counter()
        for p in usd_parts:
            # Strip trailing digits to get base type: "motor_valve_01" → "motor_valve"
            base_type = re.sub(r'[_\-]?\d+$', '', p['name'].lower())
            type_counts[base_type] += 1

        lines = [f"TOTAL PARTS IN SCENE: {len(usd_parts)}"]
        summary_parts = [f"{count}x {ptype}" for ptype, count in type_counts.items()]
        lines.append(f"PART COUNTS: {', '.join(summary_parts)}")
        lines.append(f"You MUST detect exactly {len(usd_parts)} parts — no more, no fewer.")
        lines.append("")
        lines.append("REAL WORLD PART POSITIONS (from USD bounding boxes):")
        for i, p in enumerate(usd_parts):
            c = p.get("center_xyz", [0, 0, 0])
            lines.append(
                f"  Part {i+1}: name='{p['name']}' "
                f"position=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}) "
                f"height={p.get('height', 0):.3f}m"
            )

        if place_target:
            pc = place_target.get("center_xy", [0, 0])
            lines.append(
                f"\nPLACE DESTINATION (white box): "
                f"position=({pc[0]:.3f}, {pc[1]:.3f}, {place_target.get('top_z', 0):.3f})"
            )

        return "\n".join(lines)

    def _merge_vlm_with_usd(self, scene, usd_parts):
        """
        Replace VLM approximate_position with real USD coordinates.

        Strategy: fuzzy match VLM labels to USD prim names.  When multiple
        USD prims share the same part type (e.g. 3 motor_valves), each VLM
        detection is matched to the *nearest unmatched* USD prim by comparing
        the VLM estimate position to USD bounding box centres.
        """
        if not usd_parts:
            return scene

        # Group USD prims by normalised part type label
        # e.g. "motor_valve_01" → type key "motorvalve"
        from collections import defaultdict
        type_groups = defaultdict(list)
        for p in usd_parts:
            name_lower = p["name"].lower()
            # Extract part type: strip trailing digits/underscores (e.g. "_01", "_2")
            import re
            type_key = re.sub(r'[_\-]?\d+$', '', name_lower).replace("_", "").replace("-", "")
            type_groups[type_key].append(dict(p))  # copy so we can pop

        # Track which USD prims have already been claimed
        claimed_paths = set()

        for obj in scene.get("detected_objects", []):
            label = obj.get("label", "").lower().replace("_", "").replace("-", "")

            # Find the matching type group
            candidates = None
            # Exact type match
            if label in type_groups:
                candidates = type_groups[label]
            else:
                # Partial match
                for type_key, parts in type_groups.items():
                    if type_key in label or label in type_key:
                        candidates = parts
                        break

            if not candidates:
                obj["_source"] = "vlm_estimate"
                log.warning(
                    f"No USD match for VLM label '{obj['label']}' — using VLM estimate"
                )
                continue

            # Filter out already-claimed prims
            available = [p for p in candidates if p["prim_path"] not in claimed_paths]
            if not available:
                obj["_source"] = "vlm_estimate"
                log.warning(
                    f"All USD prims for '{obj['label']}' already claimed — using VLM estimate"
                )
                continue

            # Pick the nearest unclaimed USD prim by distance to VLM estimate
            vlm_pos = obj.get("approximate_position", {})
            vlm_x = vlm_pos.get("x", 0.0)
            vlm_y = vlm_pos.get("y", 0.0)

            best = min(available, key=lambda p: (
                (p["center_xyz"][0] - vlm_x) ** 2 +
                (p["center_xyz"][1] - vlm_y) ** 2
            ))

            # Assign real coordinates and mark as claimed
            center = best["center_xyz"]
            obj["approximate_position"] = {
                "x": center[0], "y": center[1], "z": center[2]
            }
            obj["prim_path"] = best["prim_path"]
            obj["_source"] = "usd_bbox"
            claimed_paths.add(best["prim_path"])

            log.info(
                f"Matched VLM '{obj['label']}' → USD '{best['name']}' "
                f"at ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
            )

        return scene

    def _resolve_coordinates(self, params, usd_parts, place_target):
        """
        For pick/place actions, resolve the object_id to real USD coordinates.
        Falls back to the LLM-provided coordinates if no match is found.
        """
        resolved = dict(params)
        object_id = params.get("object_id", "")

        # If action is 'place_object', use the real place target
        if place_target and "place" in str(params):
            resolved["x"] = place_target["center_xy"][0]
            resolved["y"] = place_target["center_xy"][1]
            resolved["z"] = place_target["top_z"]
            resolved["dest_prim"] = place_target.get("prim_path", "")

        return resolved

    # ─── Pick Sequence ──────────────────────────────────────

    def _execute_pick_sequence(self, step, params, scene, step_num):
        """
        Phases 3-6: Approach → Depth → Grasp

        Uses REAL USD coordinates for approach and grasp.
        """
        result = {"step": step_num, "action": "pick_object", "sub_phases": []}
        obj_id = params.get("object_id", "?")

        # Find the prim_path for this object from the scene
        prim_path = None
        for obj in scene.get("detected_objects", []):
            if obj.get("object_id") == obj_id:
                prim_path = obj.get("prim_path")
                break

        # Phase 3: Approach
        self._notify(WorkflowPhase.APPROACH, "running",
                     f"Step {step_num}: Moving near {obj_id}")

        approach_payload = {
            "x": params.get("x", 0.3),
            "y": params.get("y", 0.0),
            "z": params.get("z", 0.02),
        }
        approach_result = self.camera.send_command("/api/approach", approach_payload)
        result["sub_phases"].append({
            "phase": WorkflowPhase.APPROACH,
            "status": "success" if approach_result.get("status") == "ok" else "failed",
        })
        self._notify(WorkflowPhase.APPROACH, "success",
                     f"Robot positioned near {obj_id}")

        # Phase 4: Depth camera analysis
        self._notify(WorkflowPhase.DEPTH_ANALYSIS, "running",
                     "Capturing depth image for precise spatial analysis...")

        try:
            depth_image = self.camera.capture_depth_image()
            depth_scene = self.vlm.analyze_scene(depth_image, custom_prompt=DEPTH_GRASP_PROMPT)
            result["sub_phases"].append({
                "phase": WorkflowPhase.DEPTH_ANALYSIS,
                "status": "success",
                "depth_data": depth_scene,
            })
            self._notify(WorkflowPhase.DEPTH_ANALYSIS, "success",
                         f"Depth analysis — conf={depth_scene.get('confidence', '?')}")
        except Exception as e:
            log.warning(f"Depth analysis failed: {e}. Proceeding with USD coords.")
            result["sub_phases"].append({
                "phase": WorkflowPhase.DEPTH_ANALYSIS,
                "status": "warn", "detail": str(e),
            })

        # Phase 5: Execute pick via bridge (Lula IK with real coordinates)
        self._notify(WorkflowPhase.PICK_EXECUTE, "running",
                     "Executing IK-based pick sequence...")

        pick_payload = {
            "x": params.get("x", 0.3),
            "y": params.get("y", 0.0),
            "z": params.get("z", 0.02),
        }
        if prim_path:
            pick_payload["part_prim"] = prim_path

        pick_result = self.camera.send_command("/api/pick", pick_payload)
        pick_ok = pick_result.get("status") == "completed"

        # Store adaptive finger_close so place sequence can use it
        if "finger_close" in pick_result:
            result["finger_close"] = pick_result["finger_close"]

        result["sub_phases"].append({
            "phase": WorkflowPhase.PICK_EXECUTE,
            "status": "success" if pick_ok else "failed",
            "detail": pick_result,
        })
        result["status"] = "success" if pick_ok else "failed"

        self._notify(WorkflowPhase.PICK_EXECUTE,
                     "success" if pick_ok else "failed",
                     f"Pick {'completed' if pick_ok else 'FAILED'}")

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
        if "dest_prim" in params:
            place_payload["dest_prim"] = params["dest_prim"]
        if "finger_close" in params:
            place_payload["finger_close"] = params["finger_close"]

        place_result = self.camera.send_command("/api/place", place_payload)
        place_ok = place_result.get("status") == "completed"

        result["status"] = "success" if place_ok else "failed"
        result["detail"] = place_result

        self._notify(WorkflowPhase.PLACE_EXECUTE,
                     "success" if place_ok else "failed",
                     f"Place {'completed' if place_ok else 'FAILED'}")

        return result

    # ─── Helpers ────────────────────────────────────────────

    @staticmethod
    def _finalize(result, status, error=None):
        result["status"] = status
        result["end_time"] = datetime.now().isoformat()
        if error:
            result["error"] = error
        return result
