"""
Parts Catalogue Loader.

Reads ``parts_catalogue.yaml`` and formats it into a prompt block that
the VLM sees alongside the scene image. The goal is to anchor the VLM's
labels to a fixed vocabulary with explicit visual descriptions, so it
stops inventing labels like ``circular_mechanical_parts``.

Usage:
    from knowledge.parts_catalogue import format_catalogue_for_prompt
    prompt_block = format_catalogue_for_prompt()
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import yaml

from utils.logger import log


CATALOGUE_PATH = os.path.join(
    os.path.dirname(__file__), "parts_catalogue.yaml")


@lru_cache(maxsize=1)
def load_catalogue(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the YAML catalogue once and cache the result."""
    p = path or CATALOGUE_PATH
    if not os.path.exists(p):
        log.warning(f"Parts catalogue not found at {p}")
        return {"parts": {}, "labelling_rules": []}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {
        "parts": data.get("parts", {}) or {},
        "labelling_rules": data.get("labelling_rules", []) or [],
    }


def known_part_types() -> List[str]:
    """Return the list of canonical part type keys (motor_valve, ...)."""
    return list(load_catalogue()["parts"].keys())


def detector_queries() -> Dict[str, str]:
    """Return a ``{label: detector_query}`` map for OWL-ViT2 / Grounding-DINO.

    Each part may declare a ``detector_query`` field with a natural-language
    phrase that the zero-shot detector understands better than the canonical
    label key. Falls back to the label itself (with ``_`` → space) if the
    field is missing.
    """
    out: Dict[str, str] = {}
    for label, info in load_catalogue()["parts"].items():
        q = (info or {}).get("detector_query")
        out[label] = q if q else label.replace("_", " ")
    return out


def format_catalogue_for_prompt(include_rules: bool = True) -> str:
    """Render the catalogue as a text block for prepending to a VLM prompt.

    Output is intentionally compact — VLMs lose track of long preambles.
    Each part gets one short paragraph: label, colour/shape, distinguishing
    features, what NOT to confuse it with.
    """
    cat = load_catalogue()
    if not cat["parts"]:
        return ""

    lines: List[str] = ["KNOWN PART TYPES (use these EXACT label strings):"]
    for label, info in cat["parts"].items():
        color = info.get("color", "")
        shape = info.get("shape", "")
        size = info.get("size", "")
        feats = info.get("distinguishing_features", []) or []
        avoid = info.get("not_to_confuse_with", []) or []

        lines.append(f"\n• {label}")
        if color or shape or size:
            descr = "; ".join(x for x in (color, shape, size) if x)
            lines.append(f"    Looks like: {descr}.")
        if feats:
            lines.append("    Key features:")
            for f in feats:
                lines.append(f"      - {f}")
        if avoid:
            lines.append("    Distinguish from:")
            for a in avoid:
                lines.append(f"      - {a}")

    if include_rules and cat["labelling_rules"]:
        lines.append("\nLABELLING RULES:")
        for rule in cat["labelling_rules"]:
            lines.append(f"  - {rule}")

    return "\n".join(lines)
