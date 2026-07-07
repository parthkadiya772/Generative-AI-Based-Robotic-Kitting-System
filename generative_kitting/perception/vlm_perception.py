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

from knowledge.parts_catalogue import format_catalogue_for_prompt
from utils.image_utils import encode_image_base64, resize_for_vlm
from utils.logger import log
from perception.validators import validate_scene_response


# ─── Default Perception Prompt ───────────────────────────────
# The catalogue (loaded from knowledge/parts_catalogue.yaml) is
# prepended at call time to anchor the VLM's labels to the known
# vocabulary. Without it, Gemma4 invents strings like
# "circular_mechanical_part" or "metal_flange_component".
PERCEPTION_PROMPT = """You are a robotic vision system analyzing an industrial kitting workspace.
Analyze this RGB image and identify ALL mechanical parts visible.

For each object, provide:
1. A unique object_id (format: obj_001, obj_002, ...)
2. A ``label`` — MUST be one of the EXACT label strings listed in the
   KNOWN PART TYPES catalogue above. Do NOT invent descriptive labels.
   If a part is clearly visible but does not match any catalogue entry,
   use ``"unknown"``.
3. A brief ``semantic_description`` (material, colour, approximate size)
4. An ``affordance``: "graspable", "stackable", "fragile", or "fixed"
5. ``approximate_position`` as normalised coordinates (x, y, z) relative
   to the workspace centre
6. A ``confidence`` score (0.0 to 1.0)

Do NOT guess parts you cannot clearly see. Only report what is visible.

Also provide a ``scene_summary`` describing the overall workspace state,
noting any occlusions or clutter.

RESPOND ONLY WITH VALID JSON. No markdown, no explanation.

Required JSON structure:
{
  "detected_objects": [
    {
      "object_id": "obj_001",
      "label": "motor_valve",
      "semantic_description": "...",
      "affordance": "graspable",
      "approximate_position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "confidence": 0.95
    }
  ],
  "scene_summary": "..."
}"""


def _default_prompt_with_catalogue() -> str:
    """Return the default PERCEPTION_PROMPT with the parts catalogue
    prepended. Cached implicitly via the catalogue loader's lru_cache.
    If the catalogue file is missing the base prompt is returned."""
    block = format_catalogue_for_prompt()
    if not block:
        return PERCEPTION_PROMPT
    return f"{block}\n\n{PERCEPTION_PROMPT}"


# ─── Qwen-VL coordinate-frame helper ─────────────────────────
# Qwen2.5-VL / Qwen3-VL emit bbox coordinates in absolute pixels of the
# input image, *after* the preprocessor's smart_resize snaps both axes to
# multiples of 28 (the vision-patch size). If we send an image whose dims
# are NOT 28-multiples, smart_resize fires and the bbox we get back is in
# a coordinate frame we never observed — causing constant pixel offsets.
# Pre-resizing here makes smart_resize a no-op so the bbox lives in the
# exact (W, H) we sent.
def qwen_target_size(width: int, height: int,
                     long_edge_cap: int = 1260,
                     patch: int = 28) -> tuple:
    """Compute the (W, H) Qwen-VL will see for an image of size (width, height).

    long_edge_cap=1260 (45 * patch) keeps the pixel budget under
    ~1 MP for typical 16:9 aspect ratios (1260×700 = 882k px), staying
    within Qwen's default token budget while giving ~28% more pixels
    than the conservative 1120 cap. More pixels = finer patch density
    over each part = tighter grounded bboxes.
    """
    w, h = int(width), int(height)
    long_edge = max(w, h)
    if long_edge > long_edge_cap:
        scale = long_edge_cap / float(long_edge)
        w = int(round(w * scale))
        h = int(round(h * scale))
    w = max(patch, int(round(w / patch)) * patch)
    h = max(patch, int(round(h / patch)) * patch)
    return (w, h)


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
        self.config = config
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

        # Set by `_prepare_image_for_qwen` so callers (e.g.
        # workflow_engine._extract_vlm_bboxes) can scale Qwen's pixel
        # bbox output back to crop-relative coords.
        self._last_qwen_image_size: Optional[tuple] = None

        log.info(
            f"VLMPerception initialised — provider={self.provider}, "
            f"model={self.model}, base_url={self.base_url}"
        )

    # ─── Public API ──────────────────────────────────────────

    def analyze_scene(
        self,
        image: Image.Image,
        custom_prompt: Optional[str] = None,
        max_size: int = 1024,
    ) -> Dict[str, Any]:
        """
        Send an image to the VLM and return a structured scene description.

        Parameters
        ----------
        image : PIL.Image.Image
            RGB workspace image from the Isaac Sim camera.
        custom_prompt : str, optional
            Override the default perception prompt.
        max_size : int
            Long-edge cap applied by :func:`resize_for_vlm`. Default 1024
            keeps wide overhead shots cheap; bin-crop callers should
            pass a larger value (e.g. 1600) so LANCZOS-upscaled crops
            reach the VLM at full sharpened resolution.

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
        # Use caller's prompt as-is (callers inject catalogue where
        # needed). Fall back to the default prompt + catalogue block
        # so labels stay grounded to the known vocabulary.
        base_prompt = custom_prompt or _default_prompt_with_catalogue()
        prompt = base_prompt
        # Skip resize_for_vlm for Qwen — _prepare_image_for_qwen in
        # _call_ollama handles sizing AND snaps to 28-multiples (its
        # vision-patch grid). Downsampling here first would force an
        # upscale-back-up later, introducing interpolation artifacts
        # that make Qwen emit garbage like "this this this this".
        if self.provider != "ollama_qwen":
            image = resize_for_vlm(image, max_size=max_size)

        last_error = None
        for attempt in range(1, self.max_retries + 2):
            log.info(f"VLM perception attempt {attempt}/{self.max_retries + 1}")

            try:
                raw_response = self._call_vlm(image, prompt)
                log.debug(f"VLM raw response ({len(raw_response)} chars): {raw_response[:500]}")

                parsed = self.parse_vlm_response(raw_response)
                self._normalize_positions(parsed)
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
                # The OpenAI SDK wraps httpx errors as APIConnectionError
                # with a generic "Connection error." string. The actual
                # network cause lives in __cause__; surface it so the
                # operator can tell connection-refused from DNS-error
                # from TLS-error etc.
                cause_chain = []
                cur = e
                seen = set()
                while cur is not None and id(cur) not in seen:
                    seen.add(id(cur))
                    cause_chain.append(
                        f"{type(cur).__name__}: {cur}".strip())
                    cur = getattr(cur, "__cause__", None)
                detail = " ← ".join(cause_chain) if len(cause_chain) > 1 else str(e)
                log.warning(f"VLM attempt {attempt} failed: {detail}")
                if attempt <= self.max_retries:
                    # Retry keeps the original prompt (so catalogue +
                    # any custom guidance survive) and only appends
                    # stricter JSON-formatting hints.
                    prompt = (
                        base_prompt
                        + f"\n\nPREVIOUS ATTEMPT FAILED: {e}\n"
                        "Be more careful with JSON formatting. "
                        "Ensure all keys and values are correct."
                    )

        raise RuntimeError(
            f"VLM perception failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    def analyze_raw(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Send an image + prompt and return parsed JSON without
        scene-schema validation.

        Use this for prompts whose response shape is NOT the standard
        scene-analysis format (no ``detected_objects`` key) — e.g. the
        wrist-verify and depth-analysis prompts which return fields
        like ``part_visible``, ``path_clear``, ``part_center``, etc.

        Returns the parsed dict directly, retrying only on JSON parse
        failure (not on schema mismatch).
        """
        image = resize_for_vlm(image, max_size=1024)
        last_error = None
        for attempt in range(1, self.max_retries + 2):
            log.info(f"VLM raw attempt {attempt}/{self.max_retries + 1}")
            try:
                raw_response = self._call_vlm(image, prompt)
                log.debug(
                    f"VLM raw response ({len(raw_response)} chars): "
                    f"{raw_response[:500]}")
                return self.parse_vlm_response(raw_response)
            except Exception as e:
                last_error = e
                log.warning(f"VLM raw attempt {attempt} failed: {e}")
        raise RuntimeError(
            f"VLM raw call failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_error}")

    # ─── Field Normalization ────────────────────────────────

    @staticmethod
    def _normalize_positions(parsed: Dict[str, Any]) -> None:
        """Normalize VLM position fields so the validator always sees
        ``approximate_position``.

        The scene-analysis prompt asks for ``image_position`` (normalised
        2D coords) while the validator requires ``approximate_position``.
        This bridges the two: when an object has a tighter geometry field
        such as ``image_position``, ``bbox_norm`` or ``mask_polygon`` but no
        ``approximate_position``, one is created with z=0 as a placeholder.
        The workflow engine overwrites z with depth-projected values later.
        """

        def _center_from_polygon(poly: Any) -> Optional[Dict[str, float]]:
            xs = []
            ys = []
            if not isinstance(poly, list):
                return None
            for pt in poly:
                if isinstance(pt, dict):
                    x = pt.get("x")
                    y = pt.get("y")
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    x, y = pt[0], pt[1]
                else:
                    continue
                try:
                    xs.append(float(x))
                    ys.append(float(y))
                except (TypeError, ValueError):
                    continue
            if not xs or not ys:
                return None
            return {
                "x": sum(xs) / len(xs),
                "y": sum(ys) / len(ys),
            }

        for obj in parsed.get("detected_objects", []):
            if "image_position" not in obj:
                bbox = obj.get("bbox_norm") or obj.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    try:
                        x1, y1, x2, y2 = [float(v) for v in bbox]
                        obj["image_position"] = {
                            "x": (x1 + x2) / 2.0,
                            "y": (y1 + y2) / 2.0,
                        }
                    except (TypeError, ValueError):
                        pass
                if "image_position" not in obj:
                    poly_center = _center_from_polygon(
                        obj.get("mask_polygon") or obj.get("segmentation_polygon")
                    )
                    if poly_center:
                        obj["image_position"] = poly_center

            if "approximate_position" not in obj and "image_position" in obj:
                img = obj["image_position"]
                obj["approximate_position"] = {
                    "x": img.get("x", 0.5),
                    "y": img.get("y", 0.5),
                    "z": 0.0,
                }

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

    def _prepare_image_for_qwen(self, image: Image.Image) -> Image.Image:
        """Resize to Qwen's vision-patch grid so its bbox pixels live in
        a known (W, H) frame.

        Qwen-VL's preprocessor smart-resizes inputs to multiples of 28
        and emits bbox coordinates in those resized pixels. By
        pre-resizing here to a 28-multiple shape inside Qwen's pixel
        budget, smart_resize is a no-op and the bbox coords we get back
        are in *exactly* the dimensions we sent. The size is stashed on
        ``self._last_qwen_image_size`` so the caller can scale bbox
        pixels → crop-normalised coords.
        """
        target = qwen_target_size(*image.size)
        if target != image.size:
            image = image.resize(target, Image.LANCZOS)
        self._last_qwen_image_size = target
        return image

    def _call_ollama(self, image: Image.Image, prompt: str) -> str:
        """Send image + prompt to a local/VPN Ollama instance.

        Handles both old ollama library (dict response) and new library
        (ChatResponse object) to avoid silent empty-string returns when
        the library version doesn't match what the dict-access code expects.

        Provider-specific handling:
          - ``ollama_qwen``: Qwen2.5-VL uses 28×28 vision patches and
            degenerates to "O O O O..." output when the image dimensions
            aren't multiples of 28, or when the ~4k default context is
            exhausted by the image tokens + prompt. We cap the long
            edge at 1280 px, snap both sides to multiples of 28, and
            bump ``num_ctx``/``num_predict`` so the full JSON response
            fits in the context window.
          - Other providers: default Ollama options, unchanged behaviour.
        """
        import ollama

        client = ollama.Client(host=self.base_url)

        img_for_send = image
        if self.provider == "ollama_qwen":
            img_for_send = self._prepare_image_for_qwen(image)

        # JPEG for Qwen (smaller payload, fewer Ollama-side decode
        # quirks); PNG for everything else (lossless, Gemma/LLaVA-friendly).
        img_fmt = "JPEG" if self.provider == "ollama_qwen" else "PNG"
        b64_image = encode_image_base64(img_for_send, fmt=img_fmt)

        # Qwen2.5-VL on Ollama produces empty / "this this this..."
        # garbage at strict greedy decoding (temp=0). Floor at 0.05
        # so the sampler has room to escape degenerate loops while
        # still being close to deterministic. Other models keep the
        # configured temperature.
        temp = float(self.temperature)
        if self.provider == "ollama_qwen" and temp < 0.05:
            temp = 0.05
        options = {"temperature": temp}
        # Bigger context for any Ollama VLM call — the catalogue-grounded
        # prompt plus image tokens routinely exceeds the 4k default.
        # 16384 covers a 1904×1064 image (~2584 vision tokens) plus
        # the catalogue + grounding prompt (~3000 tokens) plus a JSON
        # response budget of ~1500 tokens.
        options["num_ctx"] = int(self.config.get("vlm_num_ctx", 32768))
        # Cap response budget — if a model is going to emit garbage
        # whitespace under format=json, fail fast (~1500 tokens) rather
        # than wasting 2048 tokens of newlines. Real scene-analysis
        # JSON for ~10 parts is well under 1000 tokens.
        options["num_predict"] = int(self.config.get("vlm_num_predict", 1500))

        # Optional JSON mode — Ollama enforces JSON-shaped decoding,
        # which is usually the most reliable cure for "this this this..."
        # or empty Qwen output. Newer Ollama releases can also return an
        # empty response for some vision models when JSON mode is too
        # strict, so we keep a fallback call without ``format=json``.
        chat_kwargs = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [b64_image],   # base64 string, not raw bytes
            }],
            "options": options,
            "stream": False,
        }

        response = None
        content = ""
        attempted_json_mode = False
        for include_json_mode in (self.provider == "ollama_qwen", False):
            if include_json_mode:
                chat_kwargs["format"] = "json"
                attempted_json_mode = True
            else:
                chat_kwargs.pop("format", None)

            response = client.chat(**chat_kwargs)

            # ollama <0.4: response is a dict  {"message": {"content": "..."}}
            # ollama ≥0.4: response is a ChatResponse object  response.message.content
            if isinstance(response, dict):
                content = response.get("message", {}).get("content", "")
            else:
                msg = getattr(response, "message", None)
                content = getattr(msg, "content", "") if msg is not None else ""

            if content or not attempted_json_mode:
                break

            log.warning(
                f"[Ollama empty response] Retrying model='{self.model}' "
                f"without format=json because JSON mode returned empty content.")

        if not content:
            # Empty content from a 200 OK is almost always a server-side
            # OOM / model-load failure. Dump the full response so the
            # operator can see done_reason, eval counts, durations, etc.
            # — these fields tell us what Ollama actually did.
            diag_fields = (
                "model", "created_at", "done", "done_reason",
                "total_duration", "load_duration",
                "prompt_eval_count", "prompt_eval_duration",
                "eval_count", "eval_duration",
            )
            if isinstance(response, dict):
                diag = {k: response.get(k) for k in diag_fields
                        if k in response}
            else:
                diag = {k: getattr(response, k, None) for k in diag_fields
                        if getattr(response, k, None) is not None}
            log.warning(
                f"[Ollama empty response] model='{self.model}' "
                f"diag={diag}")
            raise ValueError(
                f"Ollama returned an empty response for model "
                f"'{self.model}'. This usually means the model failed "
                f"to load or was evicted from VRAM. Check the Ollama "
                f"server (running other large models alongside this "
                f"one can cause OOM). On the server: `ollama ps` to "
                f"see what's loaded, `nvidia-smi` for VRAM usage. "
                f"Diagnostic fields above show done_reason etc."
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
