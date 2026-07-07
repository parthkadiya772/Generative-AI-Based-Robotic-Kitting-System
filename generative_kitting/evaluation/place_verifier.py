"""VLM-based place verification using the dedicated tray camera.

The Robotiq contact sensor reports ``grasp_confirmed`` whenever the
fingers stop closing — but that fires whether or not a part is between
them (closed-on-empty looks identical to closed-on-part to a force
sensor at low thresholds). For evaluation we need a higher-fidelity
signal that asks: *did the part actually end up in the kitting tray?*

This module:

  1. Captures a frame from ``/World/Camera_Kit`` (top-down view of the
     kitting tray, independent of robot pose).
  2. Sends it to the VLM with a yes/no prompt for the specific part.
  3. Returns a structured result the recorder logs and the Streamlit
     dashboard displays.
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from PIL import Image


_PROMPT_TEMPLATE = (
    "You are inspecting a top-down photograph of a kitting tray. "
    "The robot just placed a part labelled '{label}' into the tray. "
    "Answer with a single JSON object on one line, no extra text:\n"
    '{{"part_in_tray": true|false, '
    '"confidence": 0.0-1.0, '
    '"detail": "short reason"}}\n\n'
    "Decision rule:\n"
    "- part_in_tray = true   if ANY object resembling '{label}' is "
    "visible anywhere inside the tray rim (even partially).\n"
    "- part_in_tray = false  ONLY IF the tray is clearly empty or "
    "the part is clearly OUTSIDE the tray (e.g. fell next to the "
    "tray, on the floor, or back into the bin).\n\n"
    "Calibrate confidence honestly:\n"
    "- 0.9-1.0 = unmistakable placement / clearly empty tray\n"
    "- 0.5-0.8 = visible but partially occluded / unclear viewing\n"
    "- below 0.5 = uncertain — image is blurry / cropped / lighting "
    "is poor / multiple parts overlap. Use a low confidence rather "
    "than guessing.\n"
    "When in doubt, prefer part_in_tray=true with a moderate "
    "confidence over a false negative — a wrong negative will trigger "
    "an unnecessary robot re-pick."
)


@dataclass
class PlaceVerificationResult:
    """Outcome of one Camera_Kit verification call."""

    in_tray: Optional[bool] = None
    confidence: Optional[float] = None
    detail: str = ""
    image: Optional[Image.Image] = None
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_log_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("image", None)
        d.pop("raw_response", None)
        return d


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1", "y"):
            return True
        if v in ("false", "no", "0", "n"):
            return False
    return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from a VLM response."""
    if not text:
        return None
    if isinstance(text, dict):
        return text
    # Try direct parse first.
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        pass
    # Pull the first balanced ``{ ... }`` block out of the response.
    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except ValueError:
        return None


class PlaceVerifier:
    """Camera_Kit + VLM driver that judges post-place success."""

    def __init__(self, camera_interface, vlm):
        """
        Parameters
        ----------
        camera_interface
            Object exposing ``capture_kit_image()`` (the Bridge interface
            in this project — Mock interfaces are silently degraded).
        vlm
            Object exposing ``analyze_scene(image, custom_prompt=...)``
            or ``analyze_raw(image, prompt=...)``. The standard
            :class:`VLMPerception` instance fits both interfaces.
        """
        self.camera = camera_interface
        self.vlm = vlm

    def verify(self, label: str) -> PlaceVerificationResult:
        """Capture Camera_Kit, query the VLM, return a structured result.

        Failures (camera unavailable, VLM error, JSON malformed) are
        captured in the ``error`` field rather than raised — the caller
        records the verification as ``in_tray=None`` so it shows up as
        "unknown" in the dashboard rather than crashing the pick loop.
        """
        if not hasattr(self.camera, "capture_kit_image"):
            return PlaceVerificationResult(
                error="camera_interface lacks capture_kit_image; "
                      "Camera_Kit only available in bridge mode")
        try:
            image = self.camera.capture_kit_image()
        except Exception as exc:
            return PlaceVerificationResult(
                error=f"capture_kit_image failed: {exc}")

        prompt = _PROMPT_TEMPLATE.format(label=label or "the placed part")

        try:
            if hasattr(self.vlm, "analyze_raw"):
                response = self.vlm.analyze_raw(image, prompt=prompt)
            else:
                response = self.vlm.analyze_scene(
                    image, custom_prompt=prompt)
        except Exception as exc:
            return PlaceVerificationResult(
                image=image,
                error=f"VLM call failed: {exc}")

        parsed: Optional[Dict[str, Any]]
        if isinstance(response, dict):
            parsed = response
        else:
            parsed = _extract_json(str(response))

        if not parsed:
            return PlaceVerificationResult(
                image=image, raw_response={"raw": response},
                error="Unable to parse VLM response as JSON")

        in_tray = _coerce_bool(parsed.get("part_in_tray"))
        if in_tray is None:
            # Some VLMs use "is_present" / "answer" keys.
            in_tray = _coerce_bool(parsed.get("is_present"))
        if in_tray is None:
            in_tray = _coerce_bool(parsed.get("answer"))
        confidence = _coerce_float(parsed.get("confidence"))
        detail = str(parsed.get("detail")
                     or parsed.get("reason")
                     or "")

        return PlaceVerificationResult(
            in_tray=in_tray,
            confidence=confidence,
            detail=detail,
            image=image,
            raw_response=parsed,
        )
