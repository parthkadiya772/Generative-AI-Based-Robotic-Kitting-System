"""Quantitative evaluation for the generative kitting framework.

Provides:
  * :mod:`metrics`        — per-stage metric calculators (perception
    accuracy vs USD ground truth, grounding error, latency).
  * :mod:`recorder`       — JSONL session log + SQLite cumulative store
    with CSV export for the thesis report.
  * :mod:`place_verifier` — VLM-based post-place check using the
    independent ``/World/Camera_Kit`` overhead view of the tray.

The Streamlit "Evaluation" tab consumes these modules to build the
live + cumulative dashboards.
"""

from .metrics import (
    PerceptionMetrics,
    LatencyMetrics,
    compute_perception_metrics,
    bbox_iou,
)
from .recorder import EvaluationRecorder
from .place_verifier import PlaceVerifier

__all__ = [
    "PerceptionMetrics",
    "LatencyMetrics",
    "compute_perception_metrics",
    "bbox_iou",
    "EvaluationRecorder",
    "PlaceVerifier",
]
