"""Metric calculators for VLM-based robotic kitting evaluation.

Three Tier-1 metric families covered here:

1. **Perception accuracy** — vs USD ground truth from
   ``/api/scan_scene_parts`` (precision, recall, label accuracy,
   mean IoU, mean grounding error in mm).
2. **Bounding-box IoU** — for matching predicted detections to GT
   parts using a greedy 1-to-1 max-IoU assignment.
3. **Latency** — per-stage timing summary (VLM, detector, LLM, IK,
   total cycle).

Pick / place / end-to-end success counters live on the
:class:`evaluation.recorder.EvaluationRecorder` since they are derived
from workflow phase results, not computed numerically here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _strip_instance_suffix(label: str) -> str:
    """``large_gear_3`` → ``large_gear``. Keeps non-suffixed labels."""
    return re.sub(r"_\d+$", "", (label or "").strip().lower())


def bbox_iou(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> float:
    """Intersection-over-union for axis-aligned boxes ``(x1, y1, x2, y2)``.

    Coordinates may be in any consistent frame (pixels or normalised);
    only relative areas matter.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def _xy_distance(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    dx, dy = p[0] - q[0], p[1] - q[1]
    return (dx * dx + dy * dy) ** 0.5


def _xyz_distance(p: Tuple[float, float, float],
                  q: Tuple[float, float, float]) -> float:
    dx, dy, dz = p[0] - q[0], p[1] - q[1], p[2] - q[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


# ─────────────────────────────────────────────────────────────
# Per-scan perception metrics
# ─────────────────────────────────────────────────────────────

@dataclass
class PerceptionMetrics:
    """Single-scan perception evaluation result."""

    n_ground_truth: int = 0
    n_predicted: int = 0
    n_matched: int = 0
    n_label_correct: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    label_accuracy: float = 0.0
    mean_iou: float = 0.0
    mean_grounding_error_mm: float = 0.0
    median_grounding_error_mm: float = 0.0
    max_grounding_error_mm: float = 0.0
    per_match: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _gt_label_from_prim_name(name: str) -> str:
    """Extract the canonical part type from a USD prim name.

    USD prim names look like ``large_gear_03_collidable`` or
    ``motor_valve_2``. We match against the known part-type vocabulary
    to recover ``large_gear`` / ``motor_valve``.
    """
    n = (name or "").lower()
    candidates = (
        "motor_valve", "small_hinge", "large_gear",
        "black_hose", "black_plate", "black_plug",
        "small_tube", "silver_box", "silver_gun",
        "tube_with_clamps",
    )
    for c in candidates:
        if c in n:
            return c
    # Fallback: strip trailing digits and ``_collidable`` suffix.
    n = re.sub(r"(_collidable|_collider)$", "", n)
    n = re.sub(r"_\d+$", "", n)
    return n


def compute_perception_metrics(
    predictions: List[dict],
    ground_truth_parts: List[dict],
    iou_threshold: float = 0.10,
    image_size: Optional[Tuple[int, int]] = None,
    annotations: Optional[List[dict]] = None,
) -> PerceptionMetrics:
    """Evaluate one VLM/detector scan against USD ground truth.

    Parameters
    ----------
    predictions
        Detected objects as produced by the workflow engine
        (``scene["detected_objects"]``). Each entry should carry:
            * ``label``               — predicted part type
            * ``approximate_position`` — predicted world XYZ in metres
            * one of ``bbox_norm`` / ``bbox`` / ``bbox_2d`` /
              ``image_position`` for IoU matching against GT 2D bbox
    ground_truth_parts
        Output of bridge ``/api/scan_scene_parts`` — list with
        ``name``, ``center_xyz``, ``bbox_min``, ``bbox_max``.
    iou_threshold
        Minimum 2D-IoU required to declare a prediction–GT match. The
        default of 0.10 is intentionally low because predicted bbox
        and projected GT bbox come from different camera frames /
        cropping windows — strict IoU thresholds (≥0.5) would punish
        the system for crop/zoom geometry it never saw.
    image_size
        ``(W, H)`` of the image the predictions were made on; used to
        normalise pixel bboxes when ``annotations`` is given in pixels.
    annotations
        Optional output of bridge ``/api/scene_annotations`` — gives
        per-GT-part 2D bbox in the camera frame, enabling proper IoU
        matching. When omitted, matching falls back to nearest world-XY
        within 8 cm.
    """
    if not ground_truth_parts:
        return PerceptionMetrics(n_predicted=len(predictions or []))
    if predictions is None:
        predictions = []

    # ── Build per-GT 2D bbox lookup if annotations given ──
    gt_bboxes_by_name: Dict[str, Tuple[float, float, float, float]] = {}
    if annotations:
        for a in annotations:
            name = a.get("name") or a.get("prim_path", "").rsplit("/", 1)[-1]
            bb = a.get("bbox_norm") or a.get("normalized_bbox") or a.get("bbox_pixel")
            if not name or not bb or len(bb) < 4:
                continue
            x1, y1, x2, y2 = (float(bb[0]), float(bb[1]),
                              float(bb[2]), float(bb[3]))
            # If pixel-frame, normalise so we can compare to predictions
            if image_size and (x2 > 1.5 or y2 > 1.5):
                W, H = image_size
                x1, x2 = x1 / W, x2 / W
                y1, y2 = y1 / H, y2 / H
            gt_bboxes_by_name[name] = (x1, y1, x2, y2)

    def _pred_bbox(pred: dict) -> Optional[Tuple[float, float, float, float]]:
        bb = pred.get("bbox_norm") or pred.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            try:
                return (float(bb[0]), float(bb[1]),
                        float(bb[2]), float(bb[3]))
            except (TypeError, ValueError):
                return None
        return None

    # ── Greedy max-IoU 1-to-1 matching ──
    candidates: List[Tuple[float, int, int]] = []  # (score, gi, pi)
    for gi, gt in enumerate(ground_truth_parts):
        gt_name = gt.get("name", "")
        gt_label = _gt_label_from_prim_name(gt_name)
        gt_bbox = gt_bboxes_by_name.get(gt_name)
        gt_xy = (gt.get("center_xyz", [0, 0, 0])[0],
                 gt.get("center_xyz", [0, 0, 0])[1])
        for pi, pred in enumerate(predictions):
            pred_label = _strip_instance_suffix(pred.get("label", ""))
            pred_bb = _pred_bbox(pred)
            iou = bbox_iou(pred_bb, gt_bbox) if (pred_bb and gt_bbox) else 0.0
            pred_pos = pred.get("approximate_position") or {}
            try:
                pred_xy = (float(pred_pos.get("x")),
                           float(pred_pos.get("y")))
                xy_dist = _xy_distance(pred_xy, gt_xy)
            except (TypeError, ValueError):
                xy_dist = float("inf")
            # Score: prefer IoU when available, fall back to spatial
            # proximity. A label match boosts the score so a correctly
            # labelled detection wins over a slightly tighter wrong one.
            label_match = (pred_label == gt_label)
            if iou >= iou_threshold:
                score = 1.0 + iou + (0.5 if label_match else 0.0)
            elif xy_dist <= 0.08:  # 8 cm world-XY fallback
                score = 0.5 + max(0.0, 0.08 - xy_dist) * 5 + (
                    0.5 if label_match else 0.0)
            else:
                continue
            candidates.append((score, gi, pi))

    candidates.sort(key=lambda t: -t[0])
    matched_gt: set = set()
    matched_pred: set = set()
    matches: List[dict] = []
    for _score, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        gt = ground_truth_parts[gi]
        pred = predictions[pi]
        gt_label = _gt_label_from_prim_name(gt.get("name", ""))
        pred_label = _strip_instance_suffix(pred.get("label", ""))
        gt_xyz = tuple(gt.get("center_xyz", [0, 0, 0]))
        pred_pos = pred.get("approximate_position") or {}
        try:
            pred_xyz = (float(pred_pos.get("x")),
                        float(pred_pos.get("y")),
                        float(pred_pos.get("z")))
            grounding_err_m = _xyz_distance(pred_xyz, gt_xyz)
        except (TypeError, ValueError):
            grounding_err_m = -1.0
        pred_bb = _pred_bbox(pred)
        gt_bb = gt_bboxes_by_name.get(gt.get("name", ""))
        iou = bbox_iou(pred_bb, gt_bb) if (pred_bb and gt_bb) else 0.0
        matches.append({
            "gt_name": gt.get("name", ""),
            "gt_label": gt_label,
            "pred_label": pred_label,
            "iou": iou,
            "grounding_error_mm": (grounding_err_m * 1000.0
                                   if grounding_err_m >= 0 else None),
            "label_correct": (pred_label == gt_label),
        })
        matched_gt.add(gi)
        matched_pred.add(pi)

    n_gt = len(ground_truth_parts)
    n_pred = len(predictions)
    n_match = len(matches)
    n_label_ok = sum(1 for m in matches if m["label_correct"])
    precision = n_match / n_pred if n_pred else 0.0
    recall = n_match / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    label_acc = n_label_ok / n_match if n_match else 0.0
    valid_ious = [m["iou"] for m in matches if m["iou"] > 0]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    valid_errs = [m["grounding_error_mm"] for m in matches
                  if m["grounding_error_mm"] is not None]
    mean_err = sum(valid_errs) / len(valid_errs) if valid_errs else 0.0
    median_err = (sorted(valid_errs)[len(valid_errs) // 2]
                  if valid_errs else 0.0)
    max_err = max(valid_errs) if valid_errs else 0.0

    return PerceptionMetrics(
        n_ground_truth=n_gt,
        n_predicted=n_pred,
        n_matched=n_match,
        n_label_correct=n_label_ok,
        precision=precision,
        recall=recall,
        f1=f1,
        label_accuracy=label_acc,
        mean_iou=mean_iou,
        mean_grounding_error_mm=mean_err,
        median_grounding_error_mm=median_err,
        max_grounding_error_mm=max_err,
        per_match=matches,
    )


# ─────────────────────────────────────────────────────────────
# Latency aggregator
# ─────────────────────────────────────────────────────────────

@dataclass
class LatencyMetrics:
    """Cumulative timings across the pipeline stages (seconds)."""

    vlm_total_s: float = 0.0
    vlm_calls: int = 0
    detector_total_s: float = 0.0
    detector_calls: int = 0
    llm_total_s: float = 0.0
    llm_calls: int = 0
    ik_total_s: float = 0.0
    ik_calls: int = 0
    cycle_total_s: float = 0.0
    cycle_count: int = 0

    def add(self, stage: str, seconds: float) -> None:
        if seconds is None or seconds < 0:
            return
        if stage == "vlm":
            self.vlm_total_s += seconds; self.vlm_calls += 1
        elif stage == "detector":
            self.detector_total_s += seconds; self.detector_calls += 1
        elif stage == "llm":
            self.llm_total_s += seconds; self.llm_calls += 1
        elif stage == "ik":
            self.ik_total_s += seconds; self.ik_calls += 1
        elif stage == "cycle":
            self.cycle_total_s += seconds; self.cycle_count += 1

    @staticmethod
    def _avg(total: float, n: int) -> float:
        return float(total / n) if n else 0.0

    def to_dict(self) -> dict:
        return {
            "vlm_avg_s": self._avg(self.vlm_total_s, self.vlm_calls),
            "vlm_calls": self.vlm_calls,
            "detector_avg_s": self._avg(self.detector_total_s, self.detector_calls),
            "detector_calls": self.detector_calls,
            "llm_avg_s": self._avg(self.llm_total_s, self.llm_calls),
            "llm_calls": self.llm_calls,
            "ik_avg_s": self._avg(self.ik_total_s, self.ik_calls),
            "ik_calls": self.ik_calls,
            "cycle_avg_s": self._avg(self.cycle_total_s, self.cycle_count),
            "cycle_count": self.cycle_count,
        }
