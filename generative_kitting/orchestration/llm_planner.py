"""
LLM Task Planner for the Generative Kitting System.

Receives a human operator's natural language command + VLM scene
description, and generates a sequential task plan using only
predefined action primitives.

Supports:
  - OpenAI GPT-4o (cloud)
  - Meta Llama-3.1 via Ollama (local / VPN)
"""

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
        self.base_url = config.get("llm_base_url", "http://localhost:11434")
        self.max_retries = config.get("max_retries", 3)
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 4096)
        self.max_plan_steps = config.get("max_plan_steps", 50)

        # OpenAI API key (only for cloud provider)
        import os
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
            user_command, scene_description, kit_tray_position
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
        """Route to the appropriate LLM backend.

        Any provider starting with ``ollama`` (``ollama_llama``,
        ``ollama_gemma``, ``ollama_qwen``, ...) is treated as an
        Ollama chat call. The provider tag only selects the prompt /
        client path; the actual model is given by ``self.model``,
        so the same code path serves Gemma4, Llama-3, Qwen, etc.
        """
        if self.provider == "openai":
            return self._call_openai(user_prompt)
        if self.provider == "vllm":
            return self._call_vllm(user_prompt)
        if self.provider.startswith("ollama"):
            return self._call_ollama(user_prompt)
        raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _call_openai(self, user_prompt: str) -> str:
        """Send prompt to OpenAI's hosted API."""
        return self._call_openai_compatible(user_prompt)

    def _call_vllm(self, user_prompt: str) -> str:
        """Send prompt to a self-hosted vLLM server.

        vLLM serves the OpenAI chat-completions schema, so only the base
        URL and the placeholder key differ (vLLM ignores the key unless
        started with ``--api-key``).
        """
        return self._call_openai_compatible(
            user_prompt,
            base_url=self.base_url.rstrip("/"),
            api_key=self.api_key or "EMPTY",
        )

    def _call_openai_compatible(self, user_prompt: str,
                                base_url: str = None,
                                api_key: str = None) -> str:
        """Chat-completions call. ``base_url=None`` targets api.openai.com.

        JSON mode is requested for the same reason the Ollama path sets
        ``format="json"``: smaller models wrap plans in prose otherwise.
        """
        from openai import OpenAI

        client = OpenAI(api_key=api_key or self.api_key, base_url=base_url)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    def _call_ollama(self, user_prompt: str) -> str:
        """Send the prompt to an Ollama-served model.

        ``format="json"`` is set on every call. Smaller models
        (Gemma4 e2b / e4b in particular) frequently wrap their JSON
        in chatter or markdown without it; enabling JSON-mode lets
        the server enforce schema-valid output and saves a retry.
        """
        import ollama

        client = ollama.Client(host=self.base_url)

        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )
        return response["message"]["content"]
