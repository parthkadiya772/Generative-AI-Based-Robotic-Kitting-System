"""
Isaac Sim Script Editor — Kitting System Launcher.

Paste this entire script into Isaac Sim's Script Editor and click Run.
It works within Isaac Sim's own Python environment without external dependencies.

Prerequisites:
  - AIKIDO.usd is already loaded in the stage
  - OLLAMA_BASE_URL env var points at your model server
    (defaults to http://localhost:11434 — see .env.example)
"""

import sys
import os
import json
import asyncio
import traceback
import numpy as np

# ─── Configuration ───────────────────────────────────────────

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
VLM_MODEL = "qwen2.5vl:32b"
LLM_MODEL = "llama3.1:8b"

# Robot prim paths (from your AIKIDO.usd)
ROBOT_PRIM = "/World/gantry"
UR10_BASE = "/World/gantry/gantry_home/ur10_flattened/ur10_instanceable/base_link"
EE_LINK   = "/World/gantry/gantry_home/ur10_flattened/ur10_instanceable/ee_link"
CAMERA    = "/World/Camera"

# Home joint positions (from robot_control.py)
HOME_JOINTS = [0.0, 0, -1.5708, 1.5708, -1.5708, -1.5708, 0]


# ─── Simple Logger ───────────────────────────────────────────
class SimpleLog:
    def info(self, msg):    print(f"[INFO]  {msg}")
    def warning(self, msg): print(f"[WARN]  {msg}")
    def error(self, msg):   print(f"[ERROR] {msg}")
    def debug(self, msg):   print(f"[DEBUG] {msg}")
    def success(self, msg): print(f"[OK]    {msg}")

log = SimpleLog()


# ─── Test 1: Verify Scene ───────────────────────────────────
def test_scene():
    """Check that the AIKIDO scene is loaded and key prims exist."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if not stage:
        log.error("No stage found! Load AIKIDO.usd first.")
        return False

    log.info(f"Stage loaded: {stage.GetRootLayer().identifier}")

    all_found = True
    for path in [ROBOT_PRIM, UR10_BASE, EE_LINK]:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            log.info(f"  ✓ Found: {path}")
        else:
            log.error(f"  ✗ Missing: {path}")
            all_found = False

    cam = stage.GetPrimAtPath(CAMERA)
    if cam.IsValid():
        log.info(f"  ✓ Camera: {CAMERA}")
    else:
        log.warning(f"  ⚠ Camera not at {CAMERA}")

    return all_found


# ─── Test 2: Ollama Connectivity ────────────────────────────
def test_ollama():
    """Check that the Ollama API is reachable via VPN."""
    try:
        import urllib.request

        url = f"{OLLAMA_BASE_URL}/api/tags"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Content-Type", "application/json")

        log.info(f"Testing Ollama at {OLLAMA_BASE_URL}...")

        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            log.info(f"Available: {models}")

            for name, label in [(VLM_MODEL, "VLM"), (LLM_MODEL, "LLM")]:
                if name in models:
                    log.success(f"  ✓ {label}: {name}")
                else:
                    log.warning(f"  ⚠ {label} '{name}' not found")
        return True

    except Exception as e:
        log.error(f"Cannot reach Ollama: {e}")
        return False


# ─── Test 3: Robot Init + Motion ─────────────────────────────
async def test_robot_and_motion():
    """
    Initialise the robot articulation via World, then move to home.
    This is the correct sequence: World → add robot → play timeline
    → initialize articulation → set joints → step.
    """
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        import omni.timeline

        # ── Step A: Create or get the World ──────────────────
        world = World.instance()
        if world:
            log.info("Using existing World instance")
        else:
            world = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
            log.info("Created new World instance")

        # ── Step B: Add robot articulation ───────────────────
        robot = world.scene.get_object("kitting_test")
        if not robot:
            robot = SingleArticulation(prim_path=ROBOT_PRIM, name="kitting_test")
            world.scene.add(robot)
            log.info("Added robot to World scene")
        else:
            log.info("Robot already in World scene")

        # ── Step C: Start the timeline to initialise physics ─
        log.info("Starting simulation timeline...")
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        # Wait a few frames for physics to initialise
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()

        # ── Step D: Initialise the articulation ──────────────
        log.info("Initialising articulation...")
        try:
            robot.initialize()
            log.success("Articulation initialised!")
        except Exception as e:
            log.warning(f"robot.initialize() failed: {e}")
            log.info("Trying post_reset()...")
            try:
                robot.post_reset()
                log.success("post_reset() succeeded")
            except Exception as e2:
                log.error(f"post_reset() also failed: {e2}")
                traceback.print_exc()
                return False, False

        # ── Step E: Read articulation info ────────────────────
        num_dof = robot.num_dof
        dof_names = robot.dof_names
        log.info(f"DOFs: {num_dof}")
        log.info(f"Joint names: {dof_names}")

        if num_dof is None or num_dof == 0:
            log.error("Articulation has no DOFs — check ROBOT_PRIM path")
            return True, False  # robot found but motion fails

        current = robot.get_joint_positions()
        log.info(f"Current joints: {current}")

        # ROBOT CHECK PASSED
        robot_ok = True

        # ── Step F: Move to home position ─────────────────────
        log.info("Setting home position...")

        # All 15 DOFs: [gantry, 6x arm, 8x gripper fingers (open)]
        # Gripper joints at 0.0 = fully open
        home_all = np.zeros(num_dof, dtype=np.float32)

        # Set arm joints (indices 0-6: gantry + 6 UR10 joints)
        arm_home = HOME_JOINTS  # [gantry, shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3]
        for i, val in enumerate(arm_home):
            if i < num_dof:
                home_all[i] = val

        # Gripper joints (indices 7-14) stay at 0.0 (open)
        log.info(f"Target ({num_dof} joints): {home_all}")

        robot.set_joint_positions(home_all)

        # Step simulation for 2 seconds
        log.info("Stepping simulation (120 frames)...")
        for i in range(120):
            await omni.kit.app.get_app().next_update_async()

        final = robot.get_joint_positions()
        log.info(f"Final joints: {final}")
        log.success("Motion test completed!")

        return True, True  # robot_ok, motion_ok

    except ImportError as e:
        log.error(f"Import failed: {e}")
        traceback.print_exc()
        return False, False
    except Exception as e:
        log.error(f"Robot/motion test failed: {e}")
        traceback.print_exc()
        return False, False


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("  GENERATIVE KITTING SYSTEM — Diagnostics")
    print("=" * 60)

    # Test 1: Scene
    print("\n[1/3] Checking scene...")
    scene_ok = test_scene()

    # Test 2: Ollama
    print("\n[2/3] Checking Ollama API...")
    ollama_ok = test_ollama()

    # Test 3: Robot + Motion (combined — needs async init)
    print("\n[3/3] Initialising robot & testing motion...")
    if scene_ok:
        robot_ok, motion_ok = await test_robot_and_motion()
    else:
        log.warning("Skipping robot test — scene not loaded")
        robot_ok, motion_ok = False, False

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Scene loaded:     {'✓' if scene_ok else '✗'}")
    print(f"  Ollama reachable: {'✓' if ollama_ok else '✗'}")
    print(f"  Robot ready:      {'✓' if robot_ok else '✗'}")
    print(f"  Motion works:     {'✓' if motion_ok else '✗'}")
    print("=" * 60)

    if all([scene_ok, ollama_ok, robot_ok, motion_ok]):
        print("\n✅ ALL SYSTEMS GO — Ready for kitting operations!")
    else:
        print("\n⚠️ Some checks failed — review the output above.")

asyncio.ensure_future(main())
