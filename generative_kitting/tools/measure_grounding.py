"""Grounding-error measurement against simulation ground truth.

Answers "where is the part?" stage by stage, against the USD ground truth the
bridge exposes (/api/scene_annotations: world centre, top z, image box per
part, per camera). Runs the PRODUCTION code: KittingWorkflowEngine._detect_parts
(VLM -> box decoding -> depth projection -> _top_surface_point) and the wrist
verify's patch (_wrist_patch_points -> _top_surface_point).

Stages
  camera     GT image box -> production grid + estimator -> 3D. The camera model,
             depth and estimator with a perfect detector (no VLM).
  centre     GT centre pixel -> 3D (the old single-pixel method), for reference.
  detection  VLM boxes vs GT boxes (pixel error, IoU, per-axis affine fit) and
             the resulting 3D target vs GT (end-to-end, overhead camera).
  wrist      (--wrist) robot at the 0.25 m standoff above each part: GT wrist box
             -> estimator, and the wrist VLM centre -> 7x7 patch -> estimator.

VLM sources
  --vlm live   config.yaml perception VLM (needs the vLLM server)
  --vlm fake   GT boxes distorted by --fake-scale-x/-y, --fake-offset-x/-y [px],
               --fake-noise [px]: tests this tool and the decoding path offline
  --vlm none   camera / centre (and wrist camera) stages only

usage (repo root; Isaac Sim bridge running, scene at rest):
  generative_kitting/.aikido/bin/python generative_kitting/tools/measure_grounding.py --vlm live --wrist
Writes generative_kitting/logs/grounding/<timestamp>/report.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from orchestration.workflow_engine import (  # noqa: E402
    WRIST_VERIFY_PROMPT,
    KittingWorkflowEngine,
    _bbox_grid_points,
    _top_surface_point,
    _wrist_patch_points,
)
from perception.camera_interface import BridgeCameraInterface  # noqa: E402
from utils.config_loader import load_config, package_dir  # noqa: E402


def _label(name):
    """GT prim name -> catalogue label: large_gear_2 -> large_gear."""
    return re.sub(r"_\d+$", "", name)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _stats(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    return {"n": int(v.size), "mean": round(float(v.mean()), 2), "std": round(float(v.std()), 2),
            "p95": round(float(np.percentile(v, 95)), 2), "max": round(float(v.max()), 2)}


class FakeVLM:
    """Ground-truth boxes, distorted on purpose. Same provider/model as the live
    Qwen path so _detect_parts takes the self-grounding branch."""

    def __init__(self, camera, scale_x=1.0, scale_y=1.0, off_x=0.0, off_y=0.0, noise=0.0, seed=0):
        self.provider, self.model = "vllm", "Qwen/Qwen3-VL"
        self.camera, self.alias = camera, "rgb"
        self.sx, self.sy, self.ox, self.oy, self.noise = scale_x, scale_y, off_x, off_y, noise
        self.rng = random.Random(seed)

    def _distort(self, x, y, affine=True):
        sx, sy, ox, oy = (self.sx, self.sy, self.ox, self.oy) if affine else (1.0, 1.0, 0.0, 0.0)
        return (sx * x + ox + self.rng.gauss(0, self.noise),
                sy * y + oy + self.rng.gauss(0, self.noise))

    def analyze_scene(self, image, custom_prompt=None):
        ann = self.camera.get_scene_annotations(camera=self.alias)
        W, H = ann["image_width"], ann["image_height"]
        objs = []
        for a in ann.get("annotations", []):
            if not a.get("visible", True):
                continue
            x1, y1, x2, y2 = a["image_bbox_px"]
            x1, y1 = self._distort(x1, y1)
            x2, y2 = self._distort(x2, y2)
            # Qwen3-VL style: bbox_2d on a 0-1000 grid
            objs.append({"object_id": a["name"], "label": _label(a["name"]), "confidence": 0.95,
                         "bbox_2d": [round(1000 * x1 / W), round(1000 * y1 / H),
                                     round(1000 * x2 / W), round(1000 * y2 / H)]})
        return {"detected_objects": objs, "scene_summary": "fake VLM (distorted ground truth)"}

    def analyze_raw(self, image, prompt=None):
        ann = self.camera.get_scene_annotations(camera=self.alias)
        W, H = ann["image_width"], ann["image_height"]
        vis = [a for a in ann.get("annotations", []) if a.get("visible", True)]
        if not vis:
            return {"part_visible": False}
        a = min(vis, key=lambda a: math.dist(a["image_center_px"], (W / 2, H / 2)))
        # wrist pointing: noise only (the overhead affine is a property of
        # the overhead frame and must not be transferred to a 1280x720 view)
        x, y = self._distort(*a["image_center_px"], affine=False)
        return {"part_visible": True, "path_clear": True, "collision_risk": "none",
                "center": {"x": x / W, "y": y / H}, "confidence": 0.9, "label": _label(a["name"])}


def _project(cam, points, alias, fresh):
    res = cam.project_to_world([{"x": u, "y": v} for u, v in points], camera=alias, fresh_depth=fresh)
    if "error" in res:
        raise RuntimeError(f"project_to_world ({alias}): {res['error']}")
    return res.get("world_points", [])


def _err(est, a):
    """(xy error mm, z - top mm) of an estimate (x, y, z, ...) against a GT annotation."""
    if est is None:
        return None, None
    return (1000 * math.dist(est[:2], a["world_center"][:2]), 1000 * (est[2] - a["world_top_z"]))


def stage_camera(cam, ann, alias):
    W, H = ann["image_width"], ann["image_height"]
    rows, fresh = [], True
    for a in ann["annotations"]:
        x1, y1, x2, y2 = a["image_bbox_px"]
        grid = _bbox_grid_points((x1 / W, y1 / H, x2 / W, y2 / H))
        wp = _project(cam, grid + [(a["image_center_norm"][0], a["image_center_norm"][1])], alias, fresh)
        fresh = False
        cam_xy, cam_dz = _err(_top_surface_point(wp[:-1]), a)
        c = wp[-1]
        ctr_xy, ctr_dz = _err((c["x"], c["y"], c["z"]) if "error" not in c else None, a)
        rows.append({"part": a["name"], "camera_xy_mm": cam_xy, "camera_dz_mm": cam_dz,
                     "centre_xy_mm": ctr_xy, "centre_dz_mm": ctr_dz})
    return rows


def stage_detection(engine, cam, ann, alias):
    W, H = ann["image_width"], ann["image_height"]
    image = cam.capture_workspace_image(alias) if alias != "rgb" else cam.capture_workspace_image()
    t0 = time.time()
    scene, source = engine._detect_parts(image, target_parts=None, bin_top_z=None, camera_alias=alias)
    latency = time.time() - t0
    preds = [o for o in scene.get("detected_objects", []) if o.get("_detector_bbox_full")]
    gts, used, rows = list(ann["annotations"]), set(), []
    for o in preds:
        nb = [float(v) for v in o["_detector_bbox_full"]]
        if max(nb) > 1.5:                      # raw 0-1000 grid fell through (require_detector_bbox false)
            o["_raw_grid_bbox"] = True
            nb = [v / 1000.0 for v in nb]
        pb = (nb[0] * W, nb[1] * H, nb[2] * W, nb[3] * H)
        pc = ((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2)
        cand = [a for a in gts if a["name"] not in used]
        if not cand:
            rows.append({"pred_label": o.get("label"), "match": None})
            continue
        a = min(cand, key=lambda a: math.dist(a["image_center_px"], pc))
        if math.dist(a["image_center_px"], pc) > 120:   # > ~15 cm: not this part
            rows.append({"pred_label": o.get("label"), "match": None})
            continue
        used.add(a["name"])
        pos = o.get("approximate_position") or {}
        est = (pos["x"], pos["y"], pos["z"]) if {"x", "y", "z"} <= set(pos) else None
        xy, dz = _err(est, a)
        rows.append({"part": a["name"], "pred_label": o.get("label"), "label_ok": _label(a["name"]) == o.get("label"),
                     "gt_cx": a["image_center_px"][0], "gt_cy": a["image_center_px"][1],
                     "pred_cx": pc[0], "pred_cy": pc[1], "dx_px": pc[0] - a["image_center_px"][0],
                     "dy_px": pc[1] - a["image_center_px"][1], "iou": _iou(pb, a["image_bbox_px"]),
                     "xy_mm": xy, "dz_mm": dz, "raw_grid_bbox": bool(o.get("_raw_grid_bbox")),
                     "source": o.get("_source")})
    matched = [r for r in rows if r.get("part")]
    fit = {}
    if len(matched) >= 3:
        for ax in ("x", "y"):
            g = np.array([r[f"gt_c{ax}"] for r in matched]); p = np.array([r[f"pred_c{ax}"] for r in matched])
            A = np.vstack([g, np.ones_like(g)]).T
            (s, off), *_ = np.linalg.lstsq(A, p, rcond=None)
            fit[ax] = {"scale": round(float(s), 4), "offset_px": round(float(off), 1),
                       "raw_rms_px": round(float(np.sqrt(np.mean((p - g) ** 2))), 1),
                       "residual_rms_px": round(float(np.sqrt(np.mean((A @ [s, off] - p) ** 2))), 1)}
    summary = {"n_gt": len(gts), "n_pred": len(preds), "n_matched": len(matched),
               "precision": round(len(matched) / len(preds), 3) if preds else None,
               "recall": round(len(matched) / len(gts), 3) if gts else None,
               "latency_s": round(latency, 1), "coord_source": source, "affine_fit": fit}
    return rows, summary


def stage_wrist(engine, cam, vlm, overhead_ann, use_vlm):
    rows = []
    for a in overhead_ann["annotations"]:
        gx, gy = a["world_center"][:2]
        cam.send_command("/api/home", {})
        r = cam.send_command("/api/approach", {"x": gx, "y": gy, "z": a["world_top_z"] + 0.25})
        if "error" in r:
            rows.append({"part": a["name"], "error": f"approach: {r['error']}"}); continue
        wann = cam.get_scene_annotations(camera="wrist")
        mine = [w for w in wann.get("annotations", []) if w["name"] == a["name"]]
        row = {"part": a["name"]}
        if mine:
            w, W, H = mine[0], wann["image_width"], wann["image_height"]
            x1, y1, x2, y2 = w["image_bbox_px"]
            est = _top_surface_point(_project(cam, _bbox_grid_points((x1 / W, y1 / H, x2 / W, y2 / H)), "wrist", True))
            row["wrist_camera_xy_mm"], row["wrist_camera_dz_mm"] = _err(est, a)
        if use_vlm:
            if isinstance(vlm, FakeVLM):
                vlm.alias = "wrist"
            v = vlm.analyze_raw(cam.capture_wrist_image(), prompt=WRIST_VERIFY_PROMPT)
            if isinstance(vlm, FakeVLM):
                vlm.alias = "rgb"
            c = v.get("center") or {}
            if v.get("part_visible") and {"x", "y"} <= set(c):
                est = _top_surface_point(_project(cam, _wrist_patch_points(float(c["x"]), float(c["y"])), "wrist", False))
                row["wrist_vlm_xy_mm"], row["wrist_vlm_dz_mm"] = _err(est, a)
            else:
                row["wrist_vlm_xy_mm"] = None
        rows.append(row)
    cam.send_command("/api/home", {})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlm", choices=("live", "fake", "none"), default="none")
    ap.add_argument("--wrist", action="store_true", help="also measure the wrist camera (moves the robot)")
    ap.add_argument("--bridge", default=f"http://{os.environ.get('BRIDGE_HOST', '127.0.0.1')}:{os.environ.get('BRIDGE_PORT', '8600')}")
    ap.add_argument("--repeats", type=int, default=1, help="detection repeats (VLM variance)")
    ap.add_argument("--fake-scale-x", type=float, default=1.0)
    ap.add_argument("--fake-scale-y", type=float, default=1.0)
    ap.add_argument("--fake-offset-x", type=float, default=0.0)
    ap.add_argument("--fake-offset-y", type=float, default=0.0)
    ap.add_argument("--fake-noise", type=float, default=0.0)
    args = ap.parse_args()

    cfg = load_config(os.path.join(PKG, "config.yaml"))
    cam = BridgeCameraInterface(args.bridge)
    cam.send_command("/api/home", {})
    ann = cam.get_scene_annotations(camera="rgb")
    if ann.get("status") != "ok":
        sys.exit(f"scene_annotations failed: {ann.get('error')}")

    if args.vlm == "live":
        from perception.vlm_perception import VLMPerception
        vlm = VLMPerception(cfg.get("perception", {}))
    elif args.vlm == "fake":
        vlm = FakeVLM(cam, args.fake_scale_x, args.fake_scale_y, args.fake_offset_x, args.fake_offset_y, args.fake_noise)
    else:
        vlm = None
    engine = KittingWorkflowEngine(cam, vlm, planner=None, config=cfg)

    report = {"timestamp": datetime.now().isoformat(), "vlm": args.vlm,
              "vlm_model": getattr(vlm, "model", None), "camera": ann["camera_path"],
              "image": [ann["image_width"], ann["image_height"]], "args": vars(args)}
    report["camera_stage"] = stage_camera(cam, ann, "rgb")
    if vlm is not None:
        report["detection_runs"] = [dict(zip(("rows", "summary"), stage_detection(engine, cam, ann, "rgb")))
                                    for _ in range(args.repeats)]
    if args.wrist:
        report["wrist_stage"] = stage_wrist(engine, cam, vlm, ann, vlm is not None)

    s = {"camera_xy_mm": _stats(r["camera_xy_mm"] for r in report["camera_stage"]),
         "camera_dz_mm": _stats(abs(r["camera_dz_mm"]) for r in report["camera_stage"] if r["camera_dz_mm"] is not None),
         "centre_xy_mm": _stats(r["centre_xy_mm"] for r in report["camera_stage"]),
         "centre_dz_mm": _stats(abs(r["centre_dz_mm"]) for r in report["camera_stage"] if r["centre_dz_mm"] is not None)}
    if vlm is not None:
        det = [r for run in report["detection_runs"] for r in run["rows"] if r.get("part")]
        s["detection_px"] = _stats(math.hypot(r["dx_px"], r["dy_px"]) for r in det)
        s["detection_iou"] = _stats(r["iou"] for r in det)
        s["end_to_end_xy_mm"] = _stats(r["xy_mm"] for r in det)
        s["end_to_end_dz_mm"] = _stats(abs(r["dz_mm"]) for r in det if r["dz_mm"] is not None)
    if args.wrist:
        w = report["wrist_stage"]
        s["wrist_camera_xy_mm"] = _stats(r.get("wrist_camera_xy_mm") for r in w)
        s["wrist_camera_dz_mm"] = _stats(abs(r["wrist_camera_dz_mm"]) for r in w if r.get("wrist_camera_dz_mm") is not None)
        if vlm is not None:
            s["wrist_vlm_xy_mm"] = _stats(r.get("wrist_vlm_xy_mm") for r in w)
            s["wrist_vlm_dz_mm"] = _stats(abs(r["wrist_vlm_dz_mm"]) for r in w if r.get("wrist_vlm_dz_mm") is not None)
    # per part type (gear vs valve differ in kind, not only in size)
    for lab in sorted({_label(r["part"]) for r in report["camera_stage"]}):
        s[f"{lab}:camera_xy_mm"] = _stats(r["camera_xy_mm"] for r in report["camera_stage"] if _label(r["part"]) == lab)
        if vlm is not None:
            s[f"{lab}:end_to_end_xy_mm"] = _stats(r["xy_mm"] for r in det if _label(r["part"]) == lab)
        if args.wrist:
            s[f"{lab}:wrist_camera_xy_mm"] = _stats(r.get("wrist_camera_xy_mm") for r in report["wrist_stage"] if _label(r["part"]) == lab)
            if vlm is not None:
                s[f"{lab}:wrist_vlm_xy_mm"] = _stats(r.get("wrist_vlm_xy_mm") for r in report["wrist_stage"] if _label(r["part"]) == lab)
    report["summary"] = s

    out = package_dir() / "logs" / "grounding" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    lines = [f"# Grounding error — {report['timestamp']}", "",
             f"VLM: {args.vlm} ({report['vlm_model']}); overhead {report['camera']} {report['image'][0]}x{report['image'][1]}", "",
             "| metric | n | mean | std | p95 | max |", "|---|---|---|---|---|---|"]
    for k, v in s.items():
        if v:
            lines.append(f"| {k} | {v['n']} | {v['mean']} | {v['std']} | {v['p95']} | {v['max']} |")
    for i, run in enumerate(report.get("detection_runs", [])):
        sm = run["summary"]
        lines += ["", f"Detection run {i + 1}: {sm['n_matched']}/{sm['n_gt']} matched, {sm['n_pred']} predicted, "
                      f"latency {sm['latency_s']} s, affine fit {json.dumps(sm['affine_fit'])}"]
        if any(r.get("raw_grid_bbox") for r in run["rows"]):
            lines.append("WARNING: a raw 0-1000 bbox reached the depth grid (require_detector_bbox is false).")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
