"""Persistence for evaluation events.

Two stores side-by-side:

* **JSONL session log** (``logs/evaluation/session_<id>.jsonl``) — one
  line per event, easy to tail / replay / diff in the thesis appendix.
* **SQLite cumulative store** (``logs/evaluation/eval.db``) — all runs
  ever, queried by the Streamlit "Cumulative" view to build aggregate
  charts and the CSV export the thesis tables come from.

Event schema (kept narrow on purpose so the thesis tables map 1-to-1
to columns):

    {
      "event":        "perception" | "pick" | "place" | "task",
      "timestamp":    ISO-8601 string,
      "session_id":   short-hex tag for the Streamlit session,
      "task_id":      monotonic int per "Execute" press,
      "command":      operator natural-language instruction (task only),
      "vlm_model":    active VLM (perception only)
      "llm_model":    active LLM (task only)
      …event-specific payload…
    }
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS perception_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    task_id         INTEGER,
    vlm_model       TEXT,
    n_ground_truth  INTEGER,
    n_predicted     INTEGER,
    n_matched       INTEGER,
    precision_      REAL,
    recall          REAL,
    f1              REAL,
    label_accuracy  REAL,
    mean_iou        REAL,
    mean_grounding_error_mm   REAL,
    median_grounding_error_mm REAL,
    max_grounding_error_mm    REAL,
    vlm_latency_s   REAL,
    detector_latency_s REAL,
    detail_json     TEXT
);
CREATE TABLE IF NOT EXISTS pick_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    task_id         INTEGER,
    object_id       TEXT,
    label           TEXT,
    success         INTEGER NOT NULL,
    grasp_confirmed_sensor INTEGER,
    grasp_confirmed_vlm    INTEGER,
    contact_stop    INTEGER,
    drift_xy_mm     REAL,
    drift_z_mm      REAL,
    cycle_time_s    REAL,
    detail_json     TEXT
);
CREATE TABLE IF NOT EXISTS place_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    task_id         INTEGER,
    object_id       TEXT,
    label           TEXT,
    bridge_status   TEXT,
    in_tray_vlm     INTEGER,
    vlm_confidence  REAL,
    cycle_time_s    REAL,
    detail_json     TEXT
);
CREATE TABLE IF NOT EXISTS task_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    task_id         INTEGER,
    command         TEXT,
    vlm_model       TEXT,
    llm_model       TEXT,
    targeted_count  INTEGER,
    placed_count    INTEGER,
    success         INTEGER,
    end_to_end_s    REAL,
    detail_json     TEXT
);
"""


class EvaluationRecorder:
    """Thread-safe append-only event store.

    A single recorder instance is held on the Streamlit session state;
    the workflow engine calls :meth:`log_*` from its background pick /
    place phases. Concurrent writes are serialised with an internal
    lock so the SQLite + JSONL files stay consistent.
    """

    def __init__(self, log_dir: str, session_id: str):
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.session_id = session_id
        self._lock = threading.Lock()

        self.jsonl_path = os.path.join(
            self.log_dir, f"session_{session_id}.jsonl")
        self.db_path = os.path.join(self.log_dir, "eval.db")

        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        # Monotonic task counter — starts at the highest existing one
        # for this session so re-opening Streamlit does not clobber.
        self.task_id = self._max_task_id_for_session() + 1

    # ── ID management ───────────────────────────────────────
    def _max_task_id_for_session(self) -> int:
        cur = self._conn.cursor()
        max_seen = 0
        for table in ("perception_events", "pick_events",
                      "place_events", "task_events"):
            cur.execute(
                f"SELECT COALESCE(MAX(task_id), 0) FROM {table} "
                f"WHERE session_id = ?", (self.session_id,))
            row = cur.fetchone()
            if row and row[0]:
                max_seen = max(max_seen, int(row[0]))
        return max_seen

    def begin_task(self) -> int:
        """Allocate the task_id for a new operator command."""
        with self._lock:
            tid = self.task_id
            self.task_id += 1
            return tid

    # ── JSONL writer ────────────────────────────────────────
    def _write_jsonl(self, payload: Dict[str, Any]) -> None:
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception:
            # Don't let logging break the workflow.
            pass

    # ── Public log methods ──────────────────────────────────
    def log_perception(
        self,
        task_id: int,
        metrics: Dict[str, Any],
        vlm_model: str = "",
        vlm_latency_s: Optional[float] = None,
        detector_latency_s: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ts = datetime.now().isoformat()
        detail = {**(extra or {}), "per_match": metrics.get("per_match", [])}
        with self._lock:
            self._conn.execute(
                "INSERT INTO perception_events ("
                "timestamp, session_id, task_id, vlm_model, "
                "n_ground_truth, n_predicted, n_matched, "
                "precision_, recall, f1, label_accuracy, mean_iou, "
                "mean_grounding_error_mm, median_grounding_error_mm, "
                "max_grounding_error_mm, vlm_latency_s, "
                "detector_latency_s, detail_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, self.session_id, task_id, vlm_model,
                 metrics.get("n_ground_truth"), metrics.get("n_predicted"),
                 metrics.get("n_matched"), metrics.get("precision"),
                 metrics.get("recall"), metrics.get("f1"),
                 metrics.get("label_accuracy"), metrics.get("mean_iou"),
                 metrics.get("mean_grounding_error_mm"),
                 metrics.get("median_grounding_error_mm"),
                 metrics.get("max_grounding_error_mm"),
                 vlm_latency_s, detector_latency_s,
                 json.dumps(detail, default=str)))
            self._conn.commit()
        self._write_jsonl({
            "event": "perception", "timestamp": ts,
            "session_id": self.session_id, "task_id": task_id,
            "vlm_model": vlm_model,
            "vlm_latency_s": vlm_latency_s,
            "detector_latency_s": detector_latency_s,
            **{k: v for k, v in metrics.items() if k != "per_match"},
            "per_match": metrics.get("per_match", []),
        })

    def log_pick(
        self,
        task_id: int,
        object_id: str,
        label: str,
        success: bool,
        grasp_confirmed_sensor: Optional[bool] = None,
        grasp_confirmed_vlm: Optional[bool] = None,
        contact_stop: Optional[bool] = None,
        drift_xy_mm: Optional[float] = None,
        drift_z_mm: Optional[float] = None,
        cycle_time_s: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ts = datetime.now().isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO pick_events ("
                "timestamp, session_id, task_id, object_id, label, "
                "success, grasp_confirmed_sensor, grasp_confirmed_vlm, "
                "contact_stop, drift_xy_mm, drift_z_mm, cycle_time_s, "
                "detail_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, self.session_id, task_id, object_id, label,
                 1 if success else 0,
                 None if grasp_confirmed_sensor is None
                       else (1 if grasp_confirmed_sensor else 0),
                 None if grasp_confirmed_vlm is None
                       else (1 if grasp_confirmed_vlm else 0),
                 None if contact_stop is None
                       else (1 if contact_stop else 0),
                 drift_xy_mm, drift_z_mm, cycle_time_s,
                 json.dumps(extra or {}, default=str)))
            self._conn.commit()
        self._write_jsonl({
            "event": "pick", "timestamp": ts,
            "session_id": self.session_id, "task_id": task_id,
            "object_id": object_id, "label": label,
            "success": bool(success),
            "grasp_confirmed_sensor": grasp_confirmed_sensor,
            "grasp_confirmed_vlm": grasp_confirmed_vlm,
            "contact_stop": contact_stop,
            "drift_xy_mm": drift_xy_mm, "drift_z_mm": drift_z_mm,
            "cycle_time_s": cycle_time_s,
        })

    def log_place(
        self,
        task_id: int,
        object_id: str,
        label: str,
        bridge_status: str,
        in_tray_vlm: Optional[bool],
        vlm_confidence: Optional[float],
        cycle_time_s: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ts = datetime.now().isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO place_events ("
                "timestamp, session_id, task_id, object_id, label, "
                "bridge_status, in_tray_vlm, vlm_confidence, cycle_time_s, "
                "detail_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, self.session_id, task_id, object_id, label,
                 bridge_status,
                 None if in_tray_vlm is None
                       else (1 if in_tray_vlm else 0),
                 vlm_confidence, cycle_time_s,
                 json.dumps(extra or {}, default=str)))
            self._conn.commit()
        self._write_jsonl({
            "event": "place", "timestamp": ts,
            "session_id": self.session_id, "task_id": task_id,
            "object_id": object_id, "label": label,
            "bridge_status": bridge_status,
            "in_tray_vlm": in_tray_vlm,
            "vlm_confidence": vlm_confidence,
            "cycle_time_s": cycle_time_s,
        })

    def log_task(
        self,
        task_id: int,
        command: str,
        vlm_model: str,
        llm_model: str,
        targeted_count: int,
        placed_count: int,
        success: bool,
        end_to_end_s: Optional[float],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ts = datetime.now().isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO task_events ("
                "timestamp, session_id, task_id, command, vlm_model, "
                "llm_model, targeted_count, placed_count, success, "
                "end_to_end_s, detail_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, self.session_id, task_id, command, vlm_model,
                 llm_model, targeted_count, placed_count,
                 1 if success else 0, end_to_end_s,
                 json.dumps(extra or {}, default=str)))
            self._conn.commit()
        self._write_jsonl({
            "event": "task", "timestamp": ts,
            "session_id": self.session_id, "task_id": task_id,
            "command": command, "vlm_model": vlm_model,
            "llm_model": llm_model,
            "targeted_count": targeted_count,
            "placed_count": placed_count,
            "success": bool(success),
            "end_to_end_s": end_to_end_s,
        })

    # ── Read API for the Streamlit cumulative view ─────────
    def fetch_all(self, table: str) -> List[Dict[str, Any]]:
        if table not in {"perception_events", "pick_events",
                         "place_events", "task_events"}:
            return []
        cur = self._conn.cursor()
        cur.execute(f"SELECT * FROM {table} ORDER BY id")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def export_csv(self, table: str, out_path: str) -> Optional[str]:
        rows = self.fetch_all(table)
        if not rows:
            return None
        import csv
        cols = list(rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return out_path

    def summary(self) -> Dict[str, Any]:
        """Aggregate KPI dictionary for the Streamlit headline metrics.

        Note: SQLite ``SUM`` returns ``NULL`` when every input is NULL
        (e.g. ``in_tray_vlm`` rows where the Camera_Kit verifier was
        unavailable). Every value pulled from ``cur.fetchone()`` is
        coerced through helpers so a missing aggregate becomes ``0`` /
        ``0.0`` rather than crashing the dashboard.
        """
        def _i(v) -> int:
            try:
                return int(v) if v is not None else 0
            except (TypeError, ValueError):
                return 0

        def _f(v) -> float:
            try:
                return float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        def _rate(num, den) -> float:
            return (_i(num) / _i(den)) if _i(den) else 0.0

        cur = self._conn.cursor()
        out: Dict[str, Any] = {}
        cur.execute(
            "SELECT COUNT(*), AVG(precision_), AVG(recall), AVG(f1), "
            "AVG(label_accuracy), AVG(mean_iou), "
            "AVG(mean_grounding_error_mm), AVG(vlm_latency_s) "
            "FROM perception_events")
        n, p, r, f1, la, iou, ge, vl = cur.fetchone()
        out["perception"] = {
            "n_scans": _i(n),
            "precision": _f(p), "recall": _f(r),
            "f1": _f(f1), "label_accuracy": _f(la),
            "mean_iou": _f(iou),
            "mean_grounding_error_mm": _f(ge),
            "vlm_latency_s": _f(vl),
        }
        cur.execute(
            "SELECT COUNT(*), SUM(success), AVG(cycle_time_s), "
            "AVG(drift_xy_mm), AVG(drift_z_mm) FROM pick_events")
        n, s, ct, dxy, dz = cur.fetchone()
        out["pick"] = {
            "n": _i(n), "success": _i(s),
            "rate": _rate(s, n),
            "avg_cycle_s": _f(ct),
            "avg_drift_xy_mm": _f(dxy),
            "avg_drift_z_mm": _f(dz),
        }
        # ``SUM(in_tray_vlm)`` is NULL when no place row carries a
        # verified-in-tray flag (Camera_Kit verifier unavailable). Count
        # only the rows where the flag is present so the rate has a
        # meaningful denominator.
        cur.execute(
            "SELECT COUNT(*), SUM(in_tray_vlm), AVG(vlm_confidence), "
            "AVG(cycle_time_s), "
            "SUM(CASE WHEN in_tray_vlm IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM place_events")
        n, s, c, ct, n_with_flag = cur.fetchone()
        out["place"] = {
            "n": _i(n),
            "n_verified": _i(n_with_flag),
            "success_vlm": _i(s),
            "rate_vlm": _rate(s, n_with_flag),
            "avg_vlm_confidence": _f(c),
            "avg_cycle_s": _f(ct),
        }
        cur.execute(
            "SELECT COUNT(*), SUM(success), AVG(end_to_end_s), "
            "AVG(CAST(placed_count AS REAL)/NULLIF(targeted_count,0)) "
            "FROM task_events")
        n, s, e, completion = cur.fetchone()
        out["task"] = {
            "n": _i(n), "success": _i(s),
            "rate": _rate(s, n),
            "avg_end_to_end_s": _f(e),
            "avg_completion_ratio": _f(completion),
        }
        return out

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
