"""Isolate why the robot is not moving. Run from a normal terminal.

    python generative_kitting/diagnose_motion.py

Talks to the Isaac Sim bridge over HTTP, so it runs OUTSIDE Isaac Sim
while the bridge runs inside it. Nothing here uses the planner except
the final step, so a failure early on localises the problem.

Steps:
  1. /api/status         — is the bridge up, is cuMotion ready
  2. /api/jog            — move ONE joint via apply_action only. No
                           planner, no IK, no trajectory. If this fails
                           nothing else can work.
  3. /api/verify_planner — plan to the tool's own current pose; a large
                           joint delta means ROBOT_ROOT_PATH is wrong

Read the Isaac Sim console alongside this — the bridge prints the
detailed [jog] / [verify] lines there.
"""

import json
import sys
import urllib.error
import urllib.request

BRIDGE = "http://localhost:8600"
TIMEOUT = 90


def call(path, payload=None):
    """GET when payload is None, else POST. Returns a dict."""
    url = f"{BRIDGE}{path}"
    try:
        if payload is None:
            req = urllib.request.Request(url, method="GET")
        else:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        return {"error": f"{type(e).__name__}: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def head(title):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def main():
    head("1. BRIDGE STATUS")
    st = call("/api/status")
    if "error" in st:
        print(f"  Bridge unreachable at {BRIDGE}: {st['error']}")
        print("  Start isaac_sim_bridge.py in Isaac Sim's Script Editor.")
        return 1
    for k in ("ik_ready", "planner", "cumotion_error", "collision_spheres"):
        if k in st:
            print(f"  {k:20s} = {st[k]}")

    # ── 2. Jog: the decisive test ────────────────────────────────
    # index 1 = shoulder_pan (index 0 is the gantry, a prismatic axis
    # whose motion is harder to see). 0.2 rad is ~11 degrees.
    head("2. JOG — apply_action only, no planner")
    jog = call("/api/jog", {"index": 1, "delta": 0.2, "frames": 120})
    if "error" in jog:
        print(f"  FAILED: {jog['error']}")
        return 1

    print(f"  timeline_playing = {jog['timeline_playing']}")
    print(f"  joint[{jog['joint_index']}] '{jog['joint_name']}'")
    print(f"     commanded {jog['delta_commanded']:+.3f}")
    print(f"     before    {jog['before']:+.4f}")
    print(f"     after     {jog['after']:+.4f}")
    print(f"     actual    {jog['delta_actual']:+.4f}")

    if not jog["moved"]:
        head("VERDICT: the articulation is not controllable")
        if not jog["timeline_playing"]:
            print("  The timeline is STOPPED. apply_action only sets drive")
            print("  targets — physics has to be stepping for the joints to")
            print("  follow. Press PLAY in Isaac Sim and re-run.")
        else:
            print("  The timeline is playing, so the drive is not responding.")
            print("  Open the Physics Inspector and check the ARM joints")
            print("  (shoulder_pan, shoulder_lift, elbow, wrist_1..3):")
            print("    - Stiffness must be non-zero (the gripper uses 2000)")
            print("    - Max Force must not be 0")
            print("  Nothing in the planning stack can move the robot until")
            print("  a plain apply_action does.")
        return 1

    print("\n  -> Articulation IS controllable. The planner stack is next.")

    # ── 3. Planner frame check ───────────────────────────────────
    head("3. VERIFY PLANNER — is ROBOT_ROOT_PATH correct?")
    v = call("/api/verify_planner", {})
    if "error" in v:
        print(f"  {v['error']}")
        print("\n  A 'NO PATH to its own pose' here means the world->robot-root")
        print("  transform is wrong. Try ROBOT_ROOT_PATH = '/World/gantry'")
        print("  (the URDF's root link is 'gantry').")
        return 1

    for path, pos in (v.get("root_candidates") or {}).items():
        mark = "  <- ROBOT_ROOT_PATH" if path == v.get("robot_root") else ""
        print(f"  {path:34s} ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}){mark}")
    err = v.get("position_error_m")
    print()
    if err is not None:
        print(f"  tool position error = {err * 1000:.2f} mm"
              f"  -> {'PASS' if v['passed'] else 'FAIL'}")
    else:
        print(f"  max joint delta = {v['max_joint_delta']:.4f}"
              f"  -> {'PASS' if v['passed'] else 'FAIL'}")
        print("  (no FK binding, so this is a weak check — the arm is")
        print("   redundant and many configs reach the same tool pose)")
        print(f"  kinematics API: {v.get('kinematics_api')}")
    if not v["passed"]:
        print("  The planned goal does not reach the requested tool pose,")
        print("  so the world->robot-root transform is offset. Compare the")
        print("  candidates above against the URDF root link 'gantry'.")
        return 1

    # ── 4. Is the point cloud blocking the planner? ──────────────
    head("4. PLAN TEST — with vs without the collision world")
    pt = call("/api/plan_test", {"lift": 0.15})
    if "error" in pt:
        print(f"  {pt['error']}")
        return 1
    for r in pt.get("reachability") or []:
        t = r["target"]
        print(f"  {r['label']:34s} z={t[2]:+.3f}  "
              f"{'PATH FOUND' if r['path_found'] else 'NO PATH'}")
    print()
    tgt = pt["target"]
    print(f"  obstacle comparison target "
          f"({tgt[0]:+.3f}, {tgt[1]:+.3f}, {tgt[2]:+.3f})")
    print(f"  with    {pt['collision_spheres']:>5d} spheres : "
          f"{'PATH FOUND' if pt['with_obstacles'] else 'NO PATH'}")
    print(f"  with        0 spheres : "
          f"{'PATH FOUND' if pt['without_obstacles'] else 'NO PATH'}")
    print()
    print(f"  {pt['verdict']}")
    if not pt["with_obstacles"]:
        return 1

    head("ALL CHECKS PASSED")
    print("  Articulation moves, the planner agrees with the scene, and")
    print("  planning succeeds against the collision world.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
