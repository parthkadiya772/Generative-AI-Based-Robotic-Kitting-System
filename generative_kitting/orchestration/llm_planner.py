"""
LLM Task Planner for the Generative Kitting System.

Receives a human operator's natural language command + VLM scene
description, and generates a sequential task plan using only
predefined action primitives.

Supports:
  - OpenAI GPT-4o (cloud)
  - Meta Llama-3.1 via Ollama (local / VPN)
"""
import os
import json
import re
import time
from typing import Any, Dict, List, Optional

from utils.logger import log
from orchestration.prompt_templates import (
    PLANNER_SYSTEM_PROMPT,
    format_user_prompt,
    format_retry_prompt,
)
from orchestration.validators import validate_task_plan, PlanValidationError


class LLMPlanner:
    """
    LLM-based task planner that converts natural language commands
    into structured, validated action plans.
    """

    def __init__(self, config: dict):
        """
        Initialise the LLM planner.

        Parameters
        ----------
        config : dict
            The ``orchestration`` section of config.yaml.
        """
        self.provider = config.get("llm_provider", "ollama_llama")
        self.model = config.get("llm_model", "llama3.1:8b")
        self.base_url = config.get(
            "llm_base_url",
            os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        self.max_retries = config.get("max_retries", 3)
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 4096)
        self.max_plan_steps = config.get("max_plan_steps", 50)

        # OpenAI API key (only for cloud provider)
        self.api_key = os.environ.get(
            config.get("llm_api_key_env", "OPENAI_API_KEY"), ""
        )

        log.info(
            f"LLMPlanner initialised — provider={self.provider}, "
            f"model={self.model}, base_url={self.base_url}"
        )

    # ─── Public API ──────────────────────────────────────────

    def generate_plan(
        self,
        user_command: str,
        scene_description: Dict[str, Any],
        workspace_bounds: Optional[Dict[str, float]] = None,
        kit_tray_position: Optional[list] = None,
        max_picks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate and validate a task plan from a natural language command.

        Parameters
        ----------
        user_command : str
            The operator's command (e.g., "Pick all motor valves").
        scene_description : dict
            The VLM perception output with ``detected_objects`` and ``scene_summary``.
        workspace_bounds : dict, optional
            Workspace bounds for plan validation.
        kit_tray_position : list, optional
            [x, y, z] of the kit tray.
        max_picks : int or None
            Maximum number of pick_object steps to include.  None means
            pick all matching parts.

        Returns
        -------
        dict
            Validated task plan with ``task_summary``, ``total_steps``, and ``plan``.

        Raises
        ------
        RuntimeError
            If all retries are exhausted.
        """
        user_prompt = format_user_prompt(
            user_command, scene_description, kit_tray_position,
            max_picks=max_picks,
        )

        # Extract scene object IDs for validation
        scene_objects = [
            obj["object_id"]
            for obj in scene_description.get("detected_objects", [])
        ]

        last_error = None
        current_prompt = user_prompt

        for attempt in range(1, self.max_retries + 2):
            log.info(f"LLM planning attempt {attempt}/{self.max_retries + 1}")
            start_time = time.time()

            try:
                raw_response = self._call_llm(current_prompt)
                elapsed = time.time() - start_time
                log.debug(f"LLM response ({elapsed:.1f}s, {len(raw_response)} chars)")

                # Parse JSON
                plan = self._parse_plan_response(raw_response)

                # Validate
                validated_plan = validate_task_plan(
                    plan,
                    scene_objects=scene_objects,
                    workspace_bounds=workspace_bounds,
                    max_steps=self.max_plan_steps,
                )

                # Enforce quantity limit — safety net if the LLM ignored
                # the constraint in the prompt.
                if max_picks is not None:
                    validated_plan = self._enforce_pick_limit(
                        validated_plan, max_picks)

                log.info(
                    f"Task plan generated: {validated_plan['total_steps']} steps — "
                    f"'{validated_plan['task_summary']}'"
                )
                return validated_plan

            except PlanValidationError as e:
                last_error = e
                log.warning(f"Plan validation failed (attempt {attempt}): {e}")
                if attempt <= self.max_retries:
                    current_prompt = format_retry_prompt(
                        user_prompt, e.errors
                    )

            except Exception as e:
                last_error = e
                log.warning(f"LLM attempt {attempt} failed: {e}")
                if attempt <= self.max_retries:
                    current_prompt = format_retry_prompt(
                        user_prompt, [str(e)]
                    )

        raise RuntimeError(
            f"LLM planning failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    # ─── Pick-limit enforcement ───────────────────────────────

    @staticmethod
    def _enforce_pick_limit(plan: Dict[str, Any], max_picks: int) -> Dict[str, Any]:
        """Trim the plan to at most *max_picks* pick_object steps.

        Removes complete pick cycles (open_gripper → pick_object →
        verify_grasp → place_object) beyond the limit and ensures the
        plan still ends with move_home.
        """
        steps = plan.get("plan", [])
        kept: List[Any] = []
        pick_count = 0

        for step in steps:
            action = step.get("action", "")

            if action == "pick_object":
                pick_count += 1
                if pick_count > max_picks:
                    # Drop the open_gripper we already buffered for this cycle.
                    while kept and kept[-1].get("action") == "open_gripper":
                        kept.pop()
                    break  # discard this pick and everything after it
                kept.append(step)

            elif action == "move_home":
                # Keep initial move_home; skip duplicates mid-plan.
                if not kept or kept[-1].get("action") != "move_home":
                    kept.append(step)

            else:
                kept.append(step)

        # Guarantee a final move_home.
        if not kept or kept[-1].get("action") != "move_home":
            last_num = kept[-1].get("step", 0) if kept else 0
            kept.append({"step": last_num + 1, "action": "move_home", "params": {}})

        result = dict(plan)
        result["plan"] = kept
        result["total_steps"] = len(kept)
        return result

    # ─── Response Parsing ────────────────────────────────────

    @staticmethod
    def _parse_plan_response(raw_response: str) -> Dict[str, Any]:
        """
        Parse the raw LLM text into a task plan dict.

        Handles markdown code fences and surrounding text.
        """
        text = raw_response.strip()

        # Strip markdown code fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()

        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Extract JSON object
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Failed to parse LLM response as JSON. "
            f"Response starts with: {text[:200]}"
        )

    # ─── Provider Backends ───────────────────────────────────

    def _call_llm(self, user_prompt: str) -> str:
        """Route to the appropriate LLM backend."""
        if self.provider == "openai":
            return self._call_openai(user_prompt)
        elif self.provider in ("ollama_llama", "ollama"):
            return self._call_ollama(user_prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _call_openai(self, user_prompt: str) -> str:
        """Send prompt to OpenAI GPT-4o API."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def _call_ollama(self, user_prompt: str) -> str:
        """Send prompt to Ollama (Llama-3.1 or compatible)."""
        import ollama

        client = ollama.Client(host=self.base_url)

        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )
        return response["message"]["content"]
