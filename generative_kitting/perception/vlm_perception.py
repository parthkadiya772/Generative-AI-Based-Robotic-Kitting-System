"""
VLM Perception Module for the Generative Kitting System.

Sends RGB images to a Vision-Language Model for zero-shot object
detection and spatial grounding.  Supports multiple VLM backends:
  - OpenAI GPT-4o (cloud)
  - Qwen2.5-VL via Ollama (local / VPN)
  - LLaVA-OV via Ollama (local / VPN)
  - HuggingFace Transformers (local)
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

from PIL import Image

from utils.image_utils import encode_image_base64, resize_for_vlm
from utils.logger import log
from perception.validators import validate_scene_response


# ─── Default Perception Prompt ───────────────────────────────
PERCEPTION_PROMPT = """You are a robotic vision system analyzing an industrial kitting workspace.
Analyze this RGB image and identify ALL mechanical parts visible.
For each object, provide:
1. A unique object_id (format: obj_001, obj_002, ...)
2. A semantic label (e.g., motor_valve, black_hose, black_plate, black_plug, small_hinge, small_tube, silver_box, silver_gun, tube_with_clamps)
3. A brief semantic description including material, color, and approximate size
4. An affordance property: "graspable", "stackable", "fragile", or "fixed"
5. Approximate spatial position as normalized coordinates (x, y, z) relative to the workspace center
6. Your confidence score (0.0 to 1.0)

Also provide a scene_summary describing the overall workspace state, noting any occlusions or clutter.

RESPOND ONLY WITH VALID JSON. No markdown, no explanation.

Required JSON structure:
{
  "detected_objects": [
    {
      "object_id": "obj_001",
      "label": "part_name",
      "semantic_description": "...",
      "affordance": "graspable",
      "approximate_position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "confidence": 0.95
    }
  ],
  "scene_summary": "..."
}"""


class VLMPerception:
    """
    Vision-Language Model perception interface.

    Captures or accepts an RGB image from the Isaac Sim workspace,
    sends it to a configured VLM backend, and returns a structured
    scene description (detected objects + spatial positions).
    """

    def __init__(self, config: dict):
        """
        Initialise the VLM perception module.

        Parameters
        ----------
        config : dict
            The ``perception`` section of config.yaml.
        """
        self.provider = config.get("vlm_provider", "ollama_qwen")
        self.model = config.get("vlm_model", "qwen2.5-vl:7b")
        self.base_url = config.get("vlm_base_url", "http://localhost:11434")
        self.max_retries = config.get("max_retries", 2)
        self.temperature = config.get("temperature", 0.1)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)

        # OpenAI API key (only for cloud provider)
        import os
        self.api_key = os.environ.get(
            config.get("vlm_api_key_env", "OPENAI_API_KEY"), ""
        )

        log.info(
            f"VLMPerception initialised — provider={self.provider}, "
            f"model={self.model}, base_url={self.base_url}"
        )

    # ─── Public API ──────────────────────────────────────────

    def analyze_scene(
        self,
        image: Image.Image,
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send an image to the VLM and return a structured scene description.

        Parameters
        ----------
        image : PIL.Image.Image
            RGB workspace image from the Isaac Sim camera.
        custom_prompt : str, optional
            Override the default perception prompt.

        Returns
        -------
        dict
            Parsed and validated scene description with keys:
            ``detected_objects`` (list) and ``scene_summary`` (str).

        Raises
        ------
        RuntimeError
            If all retries are exhausted without a valid response.
        """
        prompt = custom_prompt or PERCEPTION_PROMPT
        image = resize_for_vlm(image, max_size=1024)

        last_error = None
        for attempt in range(1, self.max_retries + 2):
            log.info(f"VLM perception attempt {attempt}/{self.max_retries + 1}")

            try:
                raw_response = self._call_vlm(image, prompt)
                log.debug(f"VLM raw response ({len(raw_response)} chars): {raw_response[:500]}")

                parsed = self.parse_vlm_response(raw_response)
                validated = validate_scene_response(
                    parsed,
                    confidence_threshold=self.confidence_threshold,
                )
                log.info(
                    f"VLM returned {len(validated['detected_objects'])} objects "
                    f"(scene: {validated['scene_summary'][:80]}...)"
                )
                return validated

            except Exception as e:
                last_error = e
                log.warning(f"VLM attempt {attempt} failed: {e}")
                if attempt <= self.max_retries:
                    # Retry with a stricter prompt
                    prompt = (
                        PERCEPTION_PROMPT
                        + f"\n\nPREVIOUS ATTEMPT FAILED: {e}\n"
                        "Be more careful with JSON formatting. "
                        "Ensure all keys and values are correct."
                    )

        raise RuntimeError(
            f"VLM perception failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    # ─── Response Parsing ────────────────────────────────────

    @staticmethod
    def parse_vlm_response(raw_response: str) -> Dict[str, Any]:
        """
        Parse the raw VLM text response into a structured dict.

        Handles common VLM output quirks:
        - Strips markdown code fences (```json ... ```)
        - Strips leading/trailing whitespace
        - Attempts JSON recovery from partial responses

        Parameters
        ----------
        raw_response : str
            The raw text returned by the VLM.

        Returns
        -------
        dict
            Parsed JSON with ``detected_objects`` and ``scene_summary``.

        Raises
        ------
        ValueError
            If the response cannot be parsed as valid JSON.
        """
        text = raw_response.strip()

        # Strip markdown code fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()

        # Try direct JSON parse
        try:
            data = json.loads(text)
            return data
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object from surrounding text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group())
                return data
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Failed to parse VLM response as JSON. "
            f"Response starts with: {text[:200]}"
        )

    # ─── Provider Backends ───────────────────────────────────

    def _call_vlm(self, image: Image.Image, prompt: str) -> str:
        """Route to the appropriate VLM backend."""
        if self.provider == "openai":
            return self._call_openai(image, prompt)
        elif self.provider in ("ollama_qwen", "ollama_llava", "ollama_gemma",
                               "ollama_llama"):
            return self._call_ollama(image, prompt)
        elif self.provider == "huggingface":
            return self._call_huggingface(image, prompt)
        else:
            raise ValueError(f"Unknown VLM provider: {self.provider}")

    def _call_openai(self, image: Image.Image, prompt: str) -> str:
        """Send image + prompt to OpenAI GPT-4o Vision API."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        b64_image = encode_image_base64(image, fmt="PNG")

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=self.temperature,
            max_tokens=4096,
        )
        return response.choices[0].message.content

    def _call_ollama(self, image: Image.Image, prompt: str) -> str:
        """Send image + prompt to a local/VPN Ollama instance.

        Handles both old ollama library (dict response) and new library
        (ChatResponse object) to avoid silent empty-string returns when
        the library version doesn't match what the dict-access code expects.
        """
        import ollama

        client = ollama.Client(host=self.base_url)
        b64_image = encode_image_base64(image, fmt="PNG")

        response = client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64_image],   # base64 string, not raw bytes
                }
            ],
            options={"temperature": self.temperature},
            stream=False,   # force complete response, not streaming tokens
        )

        # ollama <0.4: response is a dict  {"message": {"content": "..."}}
        # ollama ≥0.4: response is a ChatResponse object  response.message.content
        if isinstance(response, dict):
            content = response.get("message", {}).get("content", "")
        else:
            msg = getattr(response, "message", None)
            content = getattr(msg, "content", "") if msg is not None else ""

        if not content:
            raise ValueError(
                f"Ollama returned an empty response for model '{self.model}'. "
                f"Verify the model is installed on {self.base_url} "
                f"(run: ollama list) and the server is reachable."
            )
        return content

    def _call_huggingface(self, image: Image.Image, prompt: str) -> str:
        """Run VLM inference locally via HuggingFace Transformers."""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch
        except ImportError:
            raise ImportError(
                "HuggingFace Transformers and PyTorch are required for "
                "the 'huggingface' provider. Install with: "
                "pip install transformers torch"
            )

        # Lazy-load model (cached after first call)
        if not hasattr(self, "_hf_model"):
            log.info(f"Loading HuggingFace model: {self.model}")
            self._hf_processor = AutoProcessor.from_pretrained(self.model)
            self._hf_model = AutoModelForVision2Seq.from_pretrained(
                self.model,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            log.info("HuggingFace model loaded successfully")

        inputs = self._hf_processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self._hf_model.device)

        with torch.no_grad():
            generated_ids = self._hf_model.generate(
                **inputs, max_new_tokens=4096,
                temperature=self.temperature,
            )

        response = self._hf_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return response
