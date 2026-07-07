"""
Perception Visualiser — Camera Feed Annotation & Overlay.

Draws VLM detection results (bounding boxes, labels, confidence scores)
on top of the camera feed image for real-time tracking visualisation.

Supports multiple perception overlay modes:
  - Raw Feed:        Unmodified camera image
  - VLM Detections:  Bounding boxes + labels from the VLM
  - Confidence Map:  Colour-coded by detection confidence
  - Affordance View: Colour-coded by object affordance type
  - Scene Grid:      Workspace grid overlay with object positions
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.logger import log


# ─── Colour Palettes ────────────────────────────────────────

# Per-label colours (RGBA for blending)
LABEL_COLOURS = {
    "motor_valve":       (59, 130, 246, 200),   # Blue
    "black_hose":        (16, 185, 129, 200),   # Emerald
    "black_plate":       (139, 92, 246, 200),   # Violet
    "black_plug":        (245, 158, 11, 200),   # Amber
    "small_hinge":       (236, 72, 153, 200),   # Pink
    "small_tube":        (20, 184, 166, 200),   # Teal
    "silver_box":        (148, 163, 184, 200),  # Slate
    "silver_gun":        (251, 146, 60, 200),   # Orange
    "tube_with_clamps":  (34, 197, 94, 200),    # Green
}

DEFAULT_COLOUR = (167, 139, 250, 200)  # Lavender fallback

# Affordance colours
AFFORDANCE_COLOURS = {
    "graspable":  (34, 197, 94, 180),    # Green
    "stackable":  (59, 130, 246, 180),   # Blue
    "fragile":    (239, 68, 68, 180),    # Red
    "fixed":      (148, 163, 184, 180),  # Grey
}

# Confidence colour gradient (low → high)
CONFIDENCE_GRADIENT = [
    (239, 68, 68),     # 0.0 — Red
    (245, 158, 11),    # 0.5 — Amber
    (34, 197, 94),     # 1.0 — Green
]


def _get_label_colour(label: str) -> Tuple[int, ...]:
    """Get the colour for a given object label."""
    return LABEL_COLOURS.get(label, DEFAULT_COLOUR)


def _get_confidence_colour(confidence: float) -> Tuple[int, int, int]:
    """Interpolate a colour from the confidence gradient."""
    c = max(0.0, min(1.0, confidence))
    if c < 0.5:
        t = c / 0.5
        r = int(CONFIDENCE_GRADIENT[0][0] * (1 - t) + CONFIDENCE_GRADIENT[1][0] * t)
        g = int(CONFIDENCE_GRADIENT[0][1] * (1 - t) + CONFIDENCE_GRADIENT[1][1] * t)
        b = int(CONFIDENCE_GRADIENT[0][2] * (1 - t) + CONFIDENCE_GRADIENT[1][2] * t)
    else:
        t = (c - 0.5) / 0.5
        r = int(CONFIDENCE_GRADIENT[1][0] * (1 - t) + CONFIDENCE_GRADIENT[2][0] * t)
        g = int(CONFIDENCE_GRADIENT[1][1] * (1 - t) + CONFIDENCE_GRADIENT[2][1] * t)
        b = int(CONFIDENCE_GRADIENT[1][2] * (1 - t) + CONFIDENCE_GRADIENT[2][2] * t)
    return (r, g, b)


def _try_load_font(size: int = 14):
    """Try to load a clean font, fall back to default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


# ═════════════════════════════════════════════════════════════
# OVERLAY RENDERERS
# ═════════════════════════════════════════════════════════════

def draw_vlm_detections(
    image: Image.Image,
    detected_objects: List[Dict[str, Any]],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    box_size_px: int = 60,
) -> Image.Image:
    """
    Draw VLM detection overlays on a camera image.

    Objects are shown as labelled markers at their approximate
    positions (normalised coordinates mapped to pixel space).

    Parameters
    ----------
    image : PIL.Image.Image
        Base camera image.
    detected_objects : list of dict
        VLM detection results with approximate_position, label, confidence.
    image_width, image_height : int, optional
        Override image dimensions for coordinate mapping.
    box_size_px : int
        Size of the detection marker box in pixels.

    Returns
    -------
    PIL.Image.Image
        Annotated image (RGBA).
    """
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _try_load_font(13)
    font_small = _try_load_font(10)

    w = image_width or img.size[0]
    h = image_height or img.size[1]

    for obj in detected_objects:
        pos = obj.get("approximate_position", {})
        label = obj.get("label", "unknown")
        confidence = obj.get("confidence", 0.0)
        obj_id = obj.get("object_id", "")
        affordance = obj.get("affordance", "")

        # Map normalised coordinates to pixel space
        # VLM returns x,y,z relative to workspace centre
        # We map x → horizontal, y → vertical (with inversion)
        px = int((pos.get("x", 0.5) + 0.2) / 1.0 * w)   # x_min=-0.2, range=1.0
        py = int((0.5 - pos.get("y", 0.0)) / 1.0 * h)    # y inverted

        # Clamp to image bounds
        px = max(box_size_px, min(w - box_size_px, px))
        py = max(box_size_px, min(h - box_size_px, py))

        colour = _get_label_colour(label)
        half = box_size_px // 2

        # Draw detection box
        draw.rectangle(
            [px - half, py - half, px + half, py + half],
            outline=colour[:3],
            width=2,
        )

        # Draw crosshair
        cross_len = half // 2
        draw.line([px - cross_len, py, px + cross_len, py], fill=colour[:3], width=1)
        draw.line([px, py - cross_len, px, py + cross_len], fill=colour[:3], width=1)

        # Draw corner brackets
        bracket = 8
        for dx, dy in [(-half, -half), (half, -half), (-half, half), (half, half)]:
            x0, y0 = px + dx, py + dy
            sx = 1 if dx < 0 else -1
            sy = 1 if dy < 0 else -1
            draw.line([x0, y0, x0 + sx * bracket, y0], fill=colour[:3], width=2)
            draw.line([x0, y0, x0, y0 + sy * bracket], fill=colour[:3], width=2)

        # Draw label background
        label_text = f"{label}"
        conf_text = f"{confidence:.0%}"
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        label_x = px - half
        label_y = py - half - text_h - 8

        # Background pill
        draw.rounded_rectangle(
            [label_x - 4, label_y - 2, label_x + text_w + 30, label_y + text_h + 4],
            radius=4,
            fill=(0, 0, 0, 160),
        )

        # Label text
        draw.text((label_x, label_y), label_text, fill=colour[:3], font=font)

        # Confidence badge
        conf_colour = _get_confidence_colour(confidence)
        draw.text(
            (label_x + text_w + 8, label_y + 1),
            conf_text,
            fill=conf_colour,
            font=font_small,
        )

    # Composite overlay onto base image
    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


def draw_confidence_map(
    image: Image.Image,
    detected_objects: List[Dict[str, Any]],
    box_size_px: int = 70,
) -> Image.Image:
    """
    Draw objects colour-coded by confidence score.
    Red = low confidence, Green = high confidence.
    """
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _try_load_font(12)

    w, h = img.size
    half = box_size_px // 2

    for obj in detected_objects:
        pos = obj.get("approximate_position", {})
        confidence = obj.get("confidence", 0.0)
        label = obj.get("label", "?")

        px = int((pos.get("x", 0.5) + 0.2) / 1.0 * w)
        py = int((0.5 - pos.get("y", 0.0)) / 1.0 * h)
        px = max(half, min(w - half, px))
        py = max(half, min(h - half, py))

        colour = _get_confidence_colour(confidence)

        # Filled circle with transparency
        draw.ellipse(
            [px - half, py - half, px + half, py + half],
            fill=(*colour, 80),
            outline=(*colour, 220),
            width=2,
        )

        # Confidence text in centre
        text = f"{confidence:.0%}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((px - tw // 2, py - th // 2), text, fill=(255, 255, 255), font=font)

        # Label below
        draw.text((px - half, py + half + 4), label, fill=colour, font=font)

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


def draw_affordance_view(
    image: Image.Image,
    detected_objects: List[Dict[str, Any]],
    box_size_px: int = 60,
) -> Image.Image:
    """
    Draw objects colour-coded by affordance type.
    Green = graspable, Blue = stackable, Red = fragile, Grey = fixed.
    """
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _try_load_font(11)

    w, h = img.size
    half = box_size_px // 2

    for obj in detected_objects:
        pos = obj.get("approximate_position", {})
        affordance = obj.get("affordance", "graspable")
        label = obj.get("label", "?")

        px = int((pos.get("x", 0.5) + 0.2) / 1.0 * w)
        py = int((0.5 - pos.get("y", 0.0)) / 1.0 * h)
        px = max(half, min(w - half, px))
        py = max(half, min(h - half, py))

        colour = AFFORDANCE_COLOURS.get(affordance, (148, 163, 184, 180))

        # Marker shape varies by affordance
        if affordance == "fragile":
            # Diamond shape for fragile
            draw.polygon(
                [(px, py - half), (px + half, py), (px, py + half), (px - half, py)],
                fill=(*colour[:3], 60),
                outline=colour[:3],
            )
        elif affordance == "stackable":
            # Rounded rectangle for stackable
            draw.rounded_rectangle(
                [px - half, py - half, px + half, py + half],
                radius=8,
                fill=(*colour[:3], 60),
                outline=colour[:3],
                width=2,
            )
        else:
            # Standard rectangle
            draw.rectangle(
                [px - half, py - half, px + half, py + half],
                fill=(*colour[:3], 40),
                outline=colour[:3],
                width=2,
            )

        # Label + affordance
        text = f"{label}\n[{affordance}]"
        draw.text((px - half, py + half + 4), text, fill=colour[:3], font=font)

    # Legend
    legend_y = h - 25
    legend_x = 10
    for aff, col in AFFORDANCE_COLOURS.items():
        draw.rectangle(
            [legend_x, legend_y, legend_x + 12, legend_y + 12],
            fill=col[:3],
        )
        draw.text((legend_x + 16, legend_y - 1), aff, fill=(220, 220, 220), font=font)
        legend_x += 100

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


def draw_scene_grid(
    image: Image.Image,
    detected_objects: List[Dict[str, Any]],
    grid_divisions: int = 8,
) -> Image.Image:
    """
    Draw a workspace grid overlay with object position markers.
    Useful for understanding spatial distribution.
    """
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _try_load_font(10)

    w, h = img.size
    cell_w = w // grid_divisions
    cell_h = h // grid_divisions

    # Draw grid lines
    for i in range(1, grid_divisions):
        # Vertical
        draw.line(
            [i * cell_w, 0, i * cell_w, h],
            fill=(100, 100, 200, 40),
            width=1,
        )
        # Horizontal
        draw.line(
            [0, i * cell_h, w, i * cell_h],
            fill=(100, 100, 200, 40),
            width=1,
        )

    # Draw axes labels
    for i in range(grid_divisions):
        x_val = -0.2 + (i / grid_divisions) * 1.0
        y_val = 0.5 - (i / grid_divisions) * 1.0
        draw.text((i * cell_w + 2, 2), f"x={x_val:.1f}", fill=(100, 100, 200, 120), font=font)
        draw.text((2, i * cell_h + 2), f"y={y_val:.1f}", fill=(100, 100, 200, 120), font=font)

    # Draw object markers
    for obj in detected_objects:
        pos = obj.get("approximate_position", {})
        label = obj.get("label", "?")
        colour = _get_label_colour(label)

        px = int((pos.get("x", 0.5) + 0.2) / 1.0 * w)
        py = int((0.5 - pos.get("y", 0.0)) / 1.0 * h)
        px = max(8, min(w - 8, px))
        py = max(8, min(h - 8, py))

        # Pulsing dot marker
        draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=colour[:3], outline=(255, 255, 255, 200))
        draw.text((px + 10, py - 6), label, fill=colour[:3], font=font)

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


# ═════════════════════════════════════════════════════════════
# PERCEPTION MODE DISPATCHER
# ═════════════════════════════════════════════════════════════

PERCEPTION_MODES = {
    "📷 Raw Feed":          "raw",
    "🎯 VLM Detections":    "vlm_detections",
    "🔥 Confidence Map":    "confidence_map",
    "🏷️ Affordance View":   "affordance_view",
    "📐 Scene Grid":        "scene_grid",
}


def render_perception_overlay(
    image: Image.Image,
    detected_objects: List[Dict[str, Any]],
    mode: str = "vlm_detections",
) -> Image.Image:
    """
    Render the appropriate perception overlay based on the selected mode.

    Parameters
    ----------
    image : PIL.Image.Image
        Raw camera feed image.
    detected_objects : list of dict
        VLM detection results.
    mode : str
        One of: "raw", "vlm_detections", "confidence_map",
        "affordance_view", "scene_grid".

    Returns
    -------
    PIL.Image.Image
        The image with the selected overlay applied.
    """
    if mode == "raw" or not detected_objects:
        return image

    if mode == "vlm_detections":
        return draw_vlm_detections(image, detected_objects)
    elif mode == "confidence_map":
        return draw_confidence_map(image, detected_objects)
    elif mode == "affordance_view":
        return draw_affordance_view(image, detected_objects)
    elif mode == "scene_grid":
        return draw_scene_grid(image, detected_objects)
    else:
        log.warning(f"Unknown perception mode: {mode}")
        return image
