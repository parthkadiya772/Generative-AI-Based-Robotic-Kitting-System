"""Evaluation Report Generator — produces thesis-ready aggregate
statistics + CSV exports + a Markdown summary from ``eval.db``.

Run it from the ``generative_kitting`` directory:

    python -m evaluation.report

Output files (written next to ``eval.db`` in ``logs/evaluation/``):

* ``perception_events.csv``  — per-scan rows
* ``pick_events.csv``        — per-pick rows
* ``place_events.csv``       — per-place rows
* ``task_events.csv``        — per-task rows
* ``failure_modes.csv``      — pick-failure breakdown by error code
* ``thesis_summary.md``      — drop-in Results-section paragraph
                               with mean / std / N for every metric

The Markdown summary is structured to map 1:1 to thesis Result tables
— each section corresponds to a paragraph in the Results & Discussion
chapter (Perception accuracy / Grasping success / Place verification /
End-to-end / Failure mode breakdown).
"""

from __future__ import annotations

import json
import os
import sqlite3
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional


def _connect(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"eval.db not found at {db_path}. Run a kitting task in "
            f"Streamlit first to populate the evaluation tables.")
    return sqlite3.connect(db_path)


def _fetch(conn: sqlite3.Connection, table: str) -> List[dict]:
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table} ORDER BY id")
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _stats(values: List[float]) -> Dict[str, Any]:
    """Return mean / median / stdev / min / max / N for a list of floats."""
    clean = [float(v) for v in values
             if v is not None and isinstance(v, (int, float))]
    if not clean:
        return {"n": 0, "mean": 0.0, "median": 0.0, "stdev": 0.0,
                "min": 0.0, "max": 0.0}
    return {
        "n": len(clean),
        "mean": statistics.mean(clean),
        "median": statistics.median(clean),
        "stdev": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def _export_csv(rows: List[dict], out_path: str) -> int:
    if not rows:
        return 0
    import csv
    cols = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            # Stringify nested JSON columns so the CSV is uniform.
            for k, v in list(r.items()):
                if isinstance(v, (dict, list)):
                    r[k] = json.dumps(v)
            w.writerow(r)
    return len(rows)


def _failure_breakdown(picks: List[dict]) -> List[dict]:
    """Tally pick failures by error code from detail_json.

    Returns rows ``[{error: '...', count: N, fraction: 0.x}, ...]``
    sorted by count descending. Useful for the "why does grasping
    fail?" paragraph in the thesis Discussion.
    """
    counts = Counter()
    failures = [p for p in picks if not p.get("success")]
    for p in failures:
        try:
            detail = json.loads(p.get("detail_json") or "{}")
        except (TypeError, ValueError):
            detail = {}
        err = detail.get("error") or "unspecified"
        counts[err] += 1
    total = sum(counts.values()) or 1
    return [
        {"error": err, "count": c, "fraction": c / total}
        for err, c in counts.most_common()
    ]


def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    line1 = "| " + " | ".join(str(h) for h in headers) + " |"
    line2 = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join(
        "| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join([line1, line2, body])


def generate_report(db_path: Optional[str] = None,
                    out_dir: Optional[str] = None) -> str:
    """Produce all CSVs + the Markdown summary. Returns the path to
    the Markdown file."""
    here = os.path.abspath(os.path.dirname(__file__))
    repo = os.path.abspath(os.path.join(here, ".."))
    db_path = db_path or os.path.join(
        repo, "logs", "evaluation", "eval.db")
    out_dir = out_dir or os.path.dirname(db_path)
    os.makedirs(out_dir, exist_ok=True)

    conn = _connect(db_path)
    try:
        perception = _fetch(conn, "perception_events")
        picks = _fetch(conn, "pick_events")
        places = _fetch(conn, "place_events")
        tasks = _fetch(conn, "task_events")
    finally:
        conn.close()

    # ── CSV exports ─────────────────────────────────────────
    csv_counts = {
        "perception_events.csv": _export_csv(
            perception, os.path.join(out_dir, "perception_events.csv")),
        "pick_events.csv": _export_csv(
            picks, os.path.join(out_dir, "pick_events.csv")),
        "place_events.csv": _export_csv(
            places, os.path.join(out_dir, "place_events.csv")),
        "task_events.csv": _export_csv(
            tasks, os.path.join(out_dir, "task_events.csv")),
        "failure_modes.csv": _export_csv(
            _failure_breakdown(picks),
            os.path.join(out_dir, "failure_modes.csv")),
    }

    # ── Aggregate stats ────────────────────────────────────
    p_precision = _stats([r["precision_"] for r in perception])
    p_recall = _stats([r["recall"] for r in perception])
    p_f1 = _stats([r["f1"] for r in perception])
    p_label_acc = _stats([r["label_accuracy"] for r in perception])
    p_iou = _stats([r["mean_iou"] for r in perception])
    p_err = _stats([r["mean_grounding_error_mm"] for r in perception])
    p_lat = _stats([r["vlm_latency_s"] for r in perception])

    pick_success_count = sum(1 for r in picks if r.get("success"))
    pick_total = len(picks)
    pick_drift_xy = _stats([r["drift_xy_mm"] for r in picks])
    pick_drift_z = _stats([r["drift_z_mm"] for r in picks])
    pick_cycle = _stats([r["cycle_time_s"] for r in picks])
    pick_contact = sum(1 for r in picks if r.get("contact_stop"))

    place_total = len(places)
    place_verified = sum(1 for r in places
                         if r.get("in_tray_vlm") is not None)
    place_success = sum(1 for r in places
                        if r.get("in_tray_vlm") == 1)
    place_conf = _stats([r["vlm_confidence"] for r in places])
    place_cycle = _stats([r["cycle_time_s"] for r in places])

    task_total = len(tasks)
    task_success = sum(1 for r in tasks if r.get("success"))
    task_completion = _stats([
        (r["placed_count"] / r["targeted_count"])
        if r.get("targeted_count") else 0.0
        for r in tasks])
    task_e2e = _stats([r["end_to_end_s"] for r in tasks])

    failure_rows = _failure_breakdown(picks)

    # ── Markdown summary ───────────────────────────────────
    md_lines: List[str] = []
    md_lines.append(
        "# Evaluation Report — Generative AI-Based Robotic Kitting\n")
    md_lines.append(
        "Aggregate statistics generated from `eval.db`. "
        "Drop these tables into the *Results & Discussion* chapter "
        "as-is; cross-reference each section with the corresponding "
        "raw `*.csv` file for the source data.\n")

    # 1. Perception
    md_lines.append("## 1. Perception accuracy (vs USD ground truth)\n")
    md_lines.append(
        f"Sample size: **{p_precision['n']} scan(s)** logged in "
        f"`perception_events.csv`.\n")
    md_lines.append(_md_table(
        ["Metric", "N", "Mean", "Median", "Stdev", "Min", "Max"],
        [
            ["Precision", p_precision["n"],
             f"{p_precision['mean']:.3f}",
             f"{p_precision['median']:.3f}",
             f"{p_precision['stdev']:.3f}",
             f"{p_precision['min']:.3f}",
             f"{p_precision['max']:.3f}"],
            ["Recall", p_recall["n"],
             f"{p_recall['mean']:.3f}",
             f"{p_recall['median']:.3f}",
             f"{p_recall['stdev']:.3f}",
             f"{p_recall['min']:.3f}",
             f"{p_recall['max']:.3f}"],
            ["F1", p_f1["n"], f"{p_f1['mean']:.3f}",
             f"{p_f1['median']:.3f}", f"{p_f1['stdev']:.3f}",
             f"{p_f1['min']:.3f}", f"{p_f1['max']:.3f}"],
            ["Label accuracy", p_label_acc["n"],
             f"{p_label_acc['mean']:.3f}",
             f"{p_label_acc['median']:.3f}",
             f"{p_label_acc['stdev']:.3f}",
             f"{p_label_acc['min']:.3f}",
             f"{p_label_acc['max']:.3f}"],
            ["Mean IoU", p_iou["n"], f"{p_iou['mean']:.3f}",
             f"{p_iou['median']:.3f}", f"{p_iou['stdev']:.3f}",
             f"{p_iou['min']:.3f}", f"{p_iou['max']:.3f}"],
            ["Grounding error (mm)", p_err["n"],
             f"{p_err['mean']:.1f}", f"{p_err['median']:.1f}",
             f"{p_err['stdev']:.1f}", f"{p_err['min']:.1f}",
             f"{p_err['max']:.1f}"],
            ["VLM latency (s)", p_lat["n"], f"{p_lat['mean']:.2f}",
             f"{p_lat['median']:.2f}", f"{p_lat['stdev']:.2f}",
             f"{p_lat['min']:.2f}", f"{p_lat['max']:.2f}"],
        ]))
    md_lines.append("\n")

    # 2. Grasping
    md_lines.append("## 2. Grasping success and alignment\n")
    pick_rate = (pick_success_count / pick_total) if pick_total else 0.0
    contact_rate = (pick_contact / pick_total) if pick_total else 0.0
    md_lines.append(
        f"Sample size: **{pick_total} pick attempt(s)** logged in "
        f"`pick_events.csv`. "
        f"Pick success rate: **{pick_rate*100:.1f} %** "
        f"({pick_success_count} of {pick_total}). "
        f"Tip-sensor contact-stop fired in "
        f"**{pick_contact} ({contact_rate*100:.1f} %)** of attempts.\n")
    md_lines.append(_md_table(
        ["Metric", "N", "Mean", "Median", "Stdev", "Min", "Max"],
        [
            ["End-of-descent XY drift (mm)", pick_drift_xy["n"],
             f"{pick_drift_xy['mean']:.1f}",
             f"{pick_drift_xy['median']:.1f}",
             f"{pick_drift_xy['stdev']:.1f}",
             f"{pick_drift_xy['min']:.1f}",
             f"{pick_drift_xy['max']:.1f}"],
            ["End-of-descent Z drift (mm)", pick_drift_z["n"],
             f"{pick_drift_z['mean']:.1f}",
             f"{pick_drift_z['median']:.1f}",
             f"{pick_drift_z['stdev']:.1f}",
             f"{pick_drift_z['min']:.1f}",
             f"{pick_drift_z['max']:.1f}"],
            ["Pick cycle time (s)", pick_cycle["n"],
             f"{pick_cycle['mean']:.2f}",
             f"{pick_cycle['median']:.2f}",
             f"{pick_cycle['stdev']:.2f}",
             f"{pick_cycle['min']:.2f}",
             f"{pick_cycle['max']:.2f}"],
        ]))
    md_lines.append("\n")

    # 3. Place verification
    md_lines.append("## 3. Place verification (Camera_Kit VLM)\n")
    place_rate_overall = (place_success / place_total) if place_total else 0.0
    place_rate_verified = (place_success / place_verified) if place_verified else 0.0
    md_lines.append(
        f"Sample size: **{place_total} place attempt(s)** "
        f"({place_verified} verified by Camera_Kit VLM). "
        f"Verified-success rate: **{place_rate_verified*100:.1f} %** "
        f"({place_success} of {place_verified} verified). "
        f"Overall (treating un-verified as success): "
        f"**{place_rate_overall*100:.1f} %**.\n")
    md_lines.append(_md_table(
        ["Metric", "N", "Mean", "Median", "Stdev", "Min", "Max"],
        [
            ["VLM confidence", place_conf["n"],
             f"{place_conf['mean']:.2f}",
             f"{place_conf['median']:.2f}",
             f"{place_conf['stdev']:.2f}",
             f"{place_conf['min']:.2f}",
             f"{place_conf['max']:.2f}"],
            ["Place cycle time (s)", place_cycle["n"],
             f"{place_cycle['mean']:.2f}",
             f"{place_cycle['median']:.2f}",
             f"{place_cycle['stdev']:.2f}",
             f"{place_cycle['min']:.2f}",
             f"{place_cycle['max']:.2f}"],
        ]))
    md_lines.append("\n")

    # 4. End-to-end
    md_lines.append("## 4. End-to-end task performance\n")
    task_rate = (task_success / task_total) if task_total else 0.0
    md_lines.append(
        f"Sample size: **{task_total} operator command(s)**. "
        f"End-to-end success rate: **{task_rate*100:.1f} %** "
        f"({task_success} of {task_total} fully completed).\n")
    md_lines.append(_md_table(
        ["Metric", "N", "Mean", "Median", "Stdev", "Min", "Max"],
        [
            ["Per-task completion ratio", task_completion["n"],
             f"{task_completion['mean']:.3f}",
             f"{task_completion['median']:.3f}",
             f"{task_completion['stdev']:.3f}",
             f"{task_completion['min']:.3f}",
             f"{task_completion['max']:.3f}"],
            ["End-to-end time (s)", task_e2e["n"],
             f"{task_e2e['mean']:.1f}",
             f"{task_e2e['median']:.1f}",
             f"{task_e2e['stdev']:.1f}",
             f"{task_e2e['min']:.1f}",
             f"{task_e2e['max']:.1f}"],
        ]))
    md_lines.append("\n")

    # 5. Failure mode breakdown
    md_lines.append("## 5. Failure-mode breakdown\n")
    if failure_rows:
        md_lines.append(
            f"Of **{pick_total - pick_success_count} failed pick(s)**, "
            f"the breakdown by failure cause is:\n")
        md_lines.append(_md_table(
            ["Error code", "Count", "Fraction"],
            [[r["error"], r["count"], f"{r['fraction']*100:.1f} %"]
             for r in failure_rows]))
        md_lines.append(
            "\nError codes correspond to specific abort points in the "
            "pick pipeline:\n"
            "- `mechanical_close_failed` — gripper closed but the "
            "finger angle did not reach target. Indicates the descent "
            "stopped against the bin floor / divider before the part.\n"
            "- `empty_gripper_vlm` — post-grasp wrist VLM confirmed "
            "empty fingers (high-confidence visual signal).\n"
            "- `descend_failed` — Cartesian descent IK rejected the "
            "target pose. Indicates the planned XY is unreachable.\n"
            "- `no_part_in_depth` — the depth-analysis VLM check "
            "(disabled by default) rejected the approach.\n"
            "- `place_not_verified` — Camera_Kit VLM did not see the "
            "part inside the kitting tray after place.\n")
    else:
        md_lines.append(
            "No failed picks recorded — all attempts succeeded.\n")
    md_lines.append("\n")

    # 6. CSV files written
    md_lines.append("## 6. Raw CSV exports\n")
    md_lines.append(
        "All per-event raw data has been exported alongside this "
        "report — drop into Excel / pandas for deeper analysis or "
        "thesis-figure plots:\n")
    md_lines.append(_md_table(
        ["File", "Rows"],
        [[k, v] for k, v in csv_counts.items()]))
    md_lines.append("\n")

    md_path = os.path.join(out_dir, "thesis_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    return md_path


def main() -> None:
    md_path = generate_report()
    print(f"Wrote thesis summary → {md_path}")
    out_dir = os.path.dirname(md_path)
    print(f"\nAll outputs in {out_dir}:")
    for fn in sorted(os.listdir(out_dir)):
        full = os.path.join(out_dir, fn)
        size = os.path.getsize(full)
        print(f"  {fn:40s}  {size:>10,} bytes")


if __name__ == "__main__":
    main()
