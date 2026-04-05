"""
Prompt Templates for the LLM Task Planner.

Contains the system prompt, perception prompt, and user prompt
formatting functions for the orchestration layer.
"""

import json
from typing import Any, Dict


# ─── LLM Task Planner System Prompt ─────────────────────────
PLANNER_SYSTEM_PROMPT = """You are a robotic task planner for an industrial kitting workstation.
You control a UR10 robotic arm on a gantry rail with a parallel gripper (Robotiq 2F-140).

WORKSPACE LAYOUT:
- A BLUE BIN contains all the PICKABLE parts (the kitting station)
- A WHITE BOX is the DESTINATION where parts should be placed (kit tray)
- The RED GANTRY RAIL is the robot's X-axis rail — NOT a part
- The ROBOT ARM and GRIPPER are fixtures — NOT parts

AVAILABLE ACTION PRIMITIVES (you may ONLY use these):
- move_home(): Move robot to safe home position above the workspace
- move_to_pose(x, y, z, rx, ry, rz): Move end-effector to target pose
- open_gripper(): Open the parallel gripper
- close_gripper(): Close the parallel gripper to grasp an object
- pick_object(object_id, x, y, z): Move above object, descend, grasp, and lift
- place_object(object_id, x, y, z): Move to target location and release object
- verify_grasp(): Check if gripper is holding an object (returns boolean)
- request_perception_update(): Trigger a new VLM scene analysis

CRITICAL RULES:
- ALWAYS start with move_home()
- ALWAYS call open_gripper() before pick_object()
- ALWAYS call verify_grasp() after pick_object()
- If verify_grasp() returns false, retry pick once, then skip and log error
- ALWAYS end with move_home()
- NEVER generate raw joint angles or inverse kinematics parameters
- NEVER invent action primitives not listed above
- For pick_object: use the EXACT x, y, z coordinates from the detected_objects list
- For place_object: use the KIT TRAY POSITION provided in the prompt
- Coordinates are in METRES (real world), NOT normalized image coordinates

OUTPUT FORMAT:
Respond ONLY with a valid JSON task plan. No markdown, no explanation, no backticks.

Required JSON structure:
{
  "task_summary": "Brief description of the task",
  "total_steps": <integer>,
  "plan": [
    {"step": 1, "action": "move_home", "params": {}},
    {"step": 2, "action": "open_gripper", "params": {}},
    {"step": 3, "action": "pick_object", "params": {"object_id": "obj_001", "x": 0.35, "y": -0.12, "z": 0.02}},
    {"step": 4, "action": "verify_grasp", "params": {}},
    {"step": 5, "action": "place_object", "params": {"object_id": "obj_001", "x": 0.5, "y": 0.0, "z": 0.02}},
    {"step": 6, "action": "open_gripper", "params": {}},
    {"step": 7, "action": "move_home", "params": {}}
  ]
}"""


def format_user_prompt(
    user_command: str,
    scene_description: Dict[str, Any],
    kit_tray_position: list = None,
) -> str:
    """
    Combine the user's natural language command with the VLM's structured
    scene description into a prompt for the LLM planner.

    Parameters
    ----------
    user_command : str
        The operator's natural language command
        (e.g., "Pick all bolts and place them in the kit tray").
    scene_description : dict
        Structured output from the VLM perception layer containing
        ``detected_objects`` and ``scene_summary``.
    kit_tray_position : list, optional
        [x, y, z] position of the kit tray. Defaults to [0.5, 0.0, 0.02].

    Returns
    -------
    str
        Formatted prompt string for the LLM.
    """
    tray_pos = kit_tray_position or [0.5, 0.0, 0.02]

    prompt = f"""OPERATOR COMMAND: {user_command}

CURRENT WORKSPACE STATE (from VLM + USD scene scan):
{json.dumps(scene_description, indent=2)}

KIT TRAY / DESTINATION POSITION: x={tray_pos[0]}, y={tray_pos[1]}, z={tray_pos[2]}

Generate a complete task plan to fulfill the operator's command using ONLY the available action primitives.

IMPORTANT:
- Start and end with move_home()
- open_gripper() before each pick_object()
- verify_grasp() after each pick_object()
- For pick_object: copy the EXACT x, y, z from the corresponding object's approximate_position
- For place_object: use the KIT TRAY POSITION coordinates shown above
- ONLY pick objects with affordance "graspable" — ignore "fixed" items (robot, gantry, bins)
- Coordinates are in METRES (world coordinates from USD), NOT normalized pixels

Respond with ONLY valid JSON. No markdown, no explanation."""

    return prompt


def format_retry_prompt(
    original_prompt: str,
    validation_errors: list,
) -> str:
    """
    Append validation error feedback to the original prompt for retry.

    Parameters
    ----------
    original_prompt : str
        The full prompt that was sent to the LLM.
    validation_errors : list of str
        List of specific validation errors from the previous attempt.

    Returns
    -------
    str
        Updated prompt with error feedback.
    """
    error_text = "\n".join(f"  - {e}" for e in validation_errors)

    retry_addition = f"""

YOUR PREVIOUS PLAN WAS INVALID. The following errors were found:
{error_text}

Please fix these errors and generate a corrected plan.
Remember: respond with ONLY valid JSON, no markdown, no backticks."""

    return original_prompt + retry_addition


def format_perception_prompt(custom_labels: list = None) -> str:
    """
    Generate a perception prompt, optionally specifying expected labels.

    Parameters
    ----------
    custom_labels : list of str, optional
        If provided, the prompt will emphasise looking for these
        specific part types.

    Returns
    -------
    str
        Formatted perception prompt for the VLM.
    """
    from perception.vlm_perception import PERCEPTION_PROMPT

    if not custom_labels:
        return PERCEPTION_PROMPT

    labels_str = ", ".join(custom_labels)
    addition = f"\n\nPay special attention to these part types: {labels_str}"
    return PERCEPTION_PROMPT + addition
