"""Prompt templates for the LLM task planner.

The prompts are tuned for small instruction-tuned models (Gemma4 e2b /
e4b, Llama-3.1 8B). Key design choices:

* Example-first: the system prompt opens with a complete, valid JSON
  task plan. Small models copy structure they've already seen far more
  reliably than they assemble it from prose specs.
* Compact action table instead of free-form prose.
* All hard constraints are listed as a numbered checklist near the top.
* User prompt is short and ends with the operator's command + the
  scene JSON — the parts the model actually has to reason about.

These prompts are designed to pair with ``format="json"`` on the
Ollama call, which constrains the server to emit a syntactically
valid JSON object.
"""

import json
from typing import Any, Dict


# ─── LLM Task Planner System Prompt ─────────────────────────
PLANNER_SYSTEM_PROMPT = """You are the task planner for a UR10 + Robotiq 2F-140 industrial kitting robot. You read a scene description and an operator command, and you emit a JSON plan of robot actions.

# EXAMPLE OUTPUT (this is exactly the shape you must produce)
{
  "task_summary": "Pick the motor valve and place it in the kit tray",
  "total_steps": 7,
  "plan": [
    {"step": 1, "action": "move_home", "params": {}},
    {"step": 2, "action": "open_gripper", "params": {}},
    {"step": 3, "action": "pick_object", "params": {"object_id": "obj_001", "x": 0.35, "y": -0.12, "z": 0.84}},
    {"step": 4, "action": "verify_grasp", "params": {}},
    {"step": 5, "action": "place_object", "params": {"object_id": "obj_001", "x": 1.20, "y": 0.40, "z": 0.45}},
    {"step": 6, "action": "open_gripper", "params": {}},
    {"step": 7, "action": "move_home", "params": {}}
  ]
}

# ALLOWED ACTIONS (use only these)
| action                      | params                                       | purpose                                  |
|-----------------------------|----------------------------------------------|------------------------------------------|
| move_home                   | {}                                           | Move robot to safe home pose             |
| open_gripper                | {}                                           | Open parallel gripper                    |
| close_gripper               | {}                                           | Close parallel gripper                   |
| pick_object                 | {object_id, x, y, z}                         | Approach, descend, grasp, lift           |
| place_object                | {object_id, x, y, z}                         | Move to tray, release part               |
| verify_grasp                | {}                                           | Check whether a part is held             |
| move_to_pose                | {x, y, z}                                    | Move EE to an arbitrary world pose       |
| request_perception_update   | {}                                           | Re-run VLM scene analysis                |

# HARD RULES
1. Start every plan with `move_home` and end every plan with `move_home`.
2. Before each `pick_object` you must emit `open_gripper`.
3. After each `pick_object` you must emit `verify_grasp`.
4. For `pick_object` x/y/z use the EXACT `approximate_position` of that object from the scene description.
5. For `place_object` x/y/z use the KIT TRAY POSITION supplied in the user prompt — never invent a tray location.
6. Only pick objects whose affordance is `graspable`. Skip anything tagged `fixed` (robot, gantry, bins, walls).
7. Coordinates are in METRES (world frame), not pixels and not normalised.
8. NEVER invent actions outside the table above. NEVER output joint angles or IK parameters.
9. Output ONLY the JSON object. No prose, no markdown, no code fences.

# PLAN SHAPE (this is the canonical pattern — single pick-and-place)
The plan ALWAYS has the same skeleton. The middle (between the two `move_home`s) holds one or more pick→place blocks.

    [ move_home ]                       ← MANDATORY opening step
    [ open_gripper, pick_object, verify_grasp, place_object, open_gripper ]   ← one block per part
    [ open_gripper, pick_object, verify_grasp, place_object, open_gripper ]   ← repeat for additional parts
    [ move_home ]                       ← MANDATORY closing step

Do NOT stop the plan at `place_object`. After the LAST place, you MUST still emit `open_gripper` (release the part) and `move_home` (return to safe pose). A plan that ends at `place_object` is INVALID and will be rejected.

# MULTI-PICK EXAMPLE (two parts → tray)
{
  "task_summary": "Pick both motor valves and place them in the kit tray",
  "total_steps": 11,
  "plan": [
    {"step": 1,  "action": "move_home",     "params": {}},
    {"step": 2,  "action": "open_gripper",  "params": {}},
    {"step": 3,  "action": "pick_object",   "params": {"object_id": "obj_001", "x": 0.35, "y": -0.12, "z": 0.84}},
    {"step": 4,  "action": "verify_grasp",  "params": {}},
    {"step": 5,  "action": "place_object",  "params": {"object_id": "obj_001", "x": 1.20, "y": 0.40, "z": 0.45}},
    {"step": 6,  "action": "open_gripper",  "params": {}},
    {"step": 7,  "action": "pick_object",   "params": {"object_id": "obj_002", "x": 0.42, "y":  0.08, "z": 0.83}},
    {"step": 8,  "action": "verify_grasp",  "params": {}},
    {"step": 9,  "action": "place_object",  "params": {"object_id": "obj_002", "x": 1.20, "y": 0.40, "z": 0.45}},
    {"step": 10, "action": "open_gripper",  "params": {}},
    {"step": 11, "action": "move_home",     "params": {}}
  ]
}

Final reminder before you write the JSON: the LAST entry in `plan` MUST be `move_home`. Count your steps and confirm this before producing output."""


def format_user_prompt(
    user_command: str,
    scene_description: Dict[str, Any],
    kit_tray_position: list = None,
) -> str:
    """Build the per-request user prompt.

    Layout: operator command first (top of context — small models bias
    heavily toward early tokens), then the kit tray coordinates, then
    the scene JSON. The trailing reminder of the JSON-only rule helps
    when the model is tempted to add an explanation.
    """
    if kit_tray_position:
        tray_line = (
            f"KIT TRAY POSITION (use for every place_object): "
            f"x={kit_tray_position[0]}, y={kit_tray_position[1]}, "
            f"z={kit_tray_position[2]}"
        )
    else:
        tray_line = (
            "KIT TRAY POSITION: not detected. If the command requires "
            "placing, look for a `kitting_tray` entry in the scene "
            "description and use its approximate_position. If no tray "
            "is present, report this in task_summary and skip the "
            "place steps."
        )

    return f"""OPERATOR COMMAND:
{user_command}

{tray_line}

SCENE (from VLM + USD scan):
{json.dumps(scene_description, indent=2)}

Produce the JSON task plan that satisfies the operator command using only the allowed actions.

Before you write the JSON, run this checklist mentally:
  [ ] step 1 is `move_home`
  [ ] every `pick_object` is preceded by `open_gripper` and followed by `verify_grasp`
  [ ] after the LAST `place_object` there is an `open_gripper` (release) and then `move_home`
  [ ] the LAST step in `plan` is `move_home` — NOT `place_object`

Respond with ONE JSON object — no prose."""


def format_retry_prompt(
    original_prompt: str,
    validation_errors: list,
) -> str:
    """Append validation errors as a numbered list so the model can fix
    them on the next attempt. Same JSON-only reminder appended."""
    error_lines = "\n".join(f"{i+1}. {e}" for i, e in enumerate(validation_errors))
    retry_addition = f"""

Your previous JSON was REJECTED. Fix every issue below:
{error_lines}

Return the corrected JSON object — only JSON, no prose."""
    return original_prompt + retry_addition


def format_perception_prompt(custom_labels: list = None) -> str:
    """Generate a perception prompt, optionally emphasising specific
    part types. Unchanged from the original — perception prompts live
    in ``perception/vlm_perception.py``; this helper just adds the
    label hint if the caller wants to bias detection."""
    from perception.vlm_perception import PERCEPTION_PROMPT

    if not custom_labels:
        return PERCEPTION_PROMPT
    labels_str = ", ".join(custom_labels)
    return f"{PERCEPTION_PROMPT}\n\nPay special attention to these part types: {labels_str}"
