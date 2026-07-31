#!/usr/bin/env python3
"""Isolate projection bias vs. detection bias in the overhead camera path.

Background
----------
Evaluation shows a systematic ~57 mm (large_gear) to ~75 mm (motor_valve)
grounding error between predicted world XYZ and USD ground truth. This script
removes the VLM/detector from the loop entirely: it takes each part's
GROUND-TRUTH image-centre pixel (forward-projected from USD by the bridge) and
runs it back through ``/api/project_to_world`` (the same inverse projection the
robot targets with). The residual is therefore pure projection error.

Interpretation
---------------
* If projecting the GT centre pixel still lands ~tens of mm off the GT world
  centre  → the bug is in the INVERSE PROJECTION (intrinsics/extrinsics).
  Compare the "usd" vs "api" columns: if "api" is accurate and "usd" is not,
  set ``perception.intrinsics_source: api`` in config.yaml.
* If projecting the GT centre pixel is accurate (a few mm)  → projection is
  fine and the eval error is DETECTION BIAS (detector bbox centre is offset
  from the true part centre).

Usage
-----
Run from any machine that can reach the Isaac Sim bridge (Isaac Sim must be
running with the bridge loaded)::

    python diagnose_projection_bias.py
    python diagnose_projection_bias.py --bridge http://10.7.0.35:8600
"""
import argparse
import json
import math
import urllib.request


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _project(base, point, source):
    """Project one normalised pixel through the bridge with a given
    intrinsics source. Returns the first world point or None."""
    res = _post(f"{base}/api/project_to_world", {
        "points": [point],
        "camera": "rgb",
        "intrinsics_source": source,
    })
    wpts = res.get("world_points") or []
    return wpts[0] if wpts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default="http://localhost:8600",
                    help="Bridge base URL (default http://localhost:8600)")
    args = ap.parse_args()
    base = args.bridge.rstrip("/")

    print(f"Querying GT scene annotations from {base} ...")
    ann = _get(f"{base}/api/scene_annotations?camera=rgb")
    parts = ann.get("annotations", [])
    if not parts:
        print("No annotations returned. Is the scene playing and populated?")
        print(json.dumps(ann, indent=2)[:1000])
        return

    print(f"\nGot {len(parts)} parts. Projecting each GT image-centre back "
          f"through /api/project_to_world.\n")
    header = (f"{'part':<16}{'src':<5}"
              f"{'dx_mm':>9}{'dy_mm':>9}{'dz_mm':>9}"
              f"{'|xy|_mm':>10}{'|xyz|_mm':>10}")
    print(header)
    print("-" * len(header))

    # Accumulate per-source XY vectors to reveal a constant translation.
    agg = {"usd": [], "api": []}

    for p in parts:
        name = p.get("name", "?")
        gt = p.get("world_center") or [0, 0, 0]
        center_norm = p.get("image_center_norm")
        if not (center_norm and len(center_norm) == 2):
            continue
        point = {"x": float(center_norm[0]), "y": float(center_norm[1])}

        for source in ("usd", "api"):
            try:
                wp = _project(base, point, source)
            except Exception as e:
                print(f"{name:<16}{source:<5}  project failed: {e}")
                continue
            if wp is None:
                print(f"{name:<16}{source:<5}  no world point returned")
                continue
            dx = (wp["x"] - gt[0]) * 1000.0
            dy = (wp["y"] - gt[1]) * 1000.0
            dz = (wp["z"] - gt[2]) * 1000.0
            dxy = math.hypot(dx, dy)
            dxyz = math.sqrt(dx * dx + dy * dy + dz * dz)
            agg[source].append((dx, dy))
            print(f"{name:<16}{source:<5}"
                  f"{dx:>9.1f}{dy:>9.1f}{dz:>9.1f}{dxy:>10.1f}{dxyz:>10.1f}")
        print()

    # ── Summary: is the XY error a constant translation? ──
    print("Mean XY error vector per intrinsics source "
          "(constant vector ⇒ principal-point/extrinsic bias):")
    for source, vecs in agg.items():
        if not vecs:
            continue
        n = len(vecs)
        mx = sum(v[0] for v in vecs) / n
        my = sum(v[1] for v in vecs) / n
        # Spread tells constant-translation (low spread) from
        # radial/scale error (grows with distance from centre).
        spread = math.sqrt(
            sum((v[0] - mx) ** 2 + (v[1] - my) ** 2 for v in vecs) / n)
        print(f"  {source}: mean=({mx:+.1f}, {my:+.1f}) mm  "
              f"|mean|={math.hypot(mx, my):.1f} mm  spread={spread:.1f} mm")

    print("\nDecision:")
    print("  * 'api' accurate (|mean| < ~10 mm) but 'usd' not  → "
          "set perception.intrinsics_source: api")
    print("  * both still off by ~tens of mm                   → "
          "audit extrinsics / axis signs (see plan Step 2a)")
    print("  * both accurate                                   → "
          "projection is fine; error is detection bias (plan Step 2b)")


if __name__ == "__main__":
    main()
