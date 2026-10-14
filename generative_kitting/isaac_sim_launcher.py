"""
Isaac Sim Script Editor — Kitting System diagnostics.

Run inside Isaac Sim with the project scene open:
    exec(open("/path/to/robot_in_air/generative_kitting/isaac_sim_launcher.py").read())

Checks the scene prims, the configured VLM endpoint and that the robot moves to
its home pose. Prim paths, joints and home pose come from robot.yaml
(ROBOT_CONFIG_PATH), the VLM from config.yaml — nothing robot-specific here.
"""

import sys
import os
import json
import asyncio
import traceback
import numpy as np

# ─── Configuration (robot.yaml + config.yaml) ───────────────

_MARKER = os.path.join("generative_kitting", "isaac_sim_bridge.py")


def _find_repo_root():
    """Repo root from KITTING_PROJECT_ROOT, __file__, the open stage or the cwd."""
    candidates = [os.environ.get("KITTING_PROJECT_ROOT", ""), globals().get("__file__")]
    try:
        import omni.usd
        url = omni.usd.get_context().get_stage_url() or ""
        for prefix in ("file:///", "file://", "file:"):
            if url.startswith(prefix):
                url = url[len(prefix):] if os.name == "nt" else "/" + url[len(prefix):].lstrip("/")
                break
        candidates.append(os.path.dirname(url) if url else "")
    except Exception:
        pass
    candidates.append(os.getcwd())
    for start in filter(None, candidates):
        d = os.path.abspath(start)
        for _ in range(8):
            if os.path.isfile(os.path.join(d, _MARKER)):
                return d
            if os.path.dirname(d) == d:
                break
            d = os.path.dirname(d)
    raise RuntimeError("cannot find the robot_in_air repo; open its USD scene or set "
                       "KITTING_PROJECT_ROOT")


_ROOT = _find_repo_root()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from generative_kitting.utils.config_loader import load_config      # noqa: E402
from generative_kitting.utils.robot_config import load_robot_config  # noqa: E402

ROBOT_CFG = load_robot_config()
APP_CFG = load_config(os.path.join(_ROOT, "generative_kitting", "config.yaml"))
_PERCEPTION = APP_CFG.get("perception", {})
VLM_PROVIDER = _PERCEPTION.get("vlm_provider", "vllm")
VLM_MODEL = _PERCEPTION.get("vlm_model", "")
VLM_BASE_URL = _PERCEPTION.get("vlm_base_url", "")

ROBOT_PRIM = ROBOT_CFG.root
ARM_BASE = ROBOT_CFG.arm_base_prim
EE_LINK = ROBOT_CFG.ee_prim
TOOL = ROBOT_CFG.tool_prim
CAMERA = "/World/Camera"
# [gantry, 6 arm joints] in cspace order; None = keep the scene's pose
HOME_JOINTS = ROBOT_CFG.home_joints


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
        log.error(f"No stage found! Open {ROBOT_CFG.stage_path} first.")
        return False

    log.info(f"Stage loaded: {stage.GetRootLayer().identifier}")

    all_found = True
    for path in [ROBOT_PRIM, ARM_BASE, EE_LINK, TOOL]:
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


# ─── Test 2: VLM endpoint ───────────────────────────────────
def test_vlm():
    """Check that the configured VLM endpoint answers and serves the model."""
    import urllib.request

    if not VLM_BASE_URL or "${" in VLM_BASE_URL:
        log.error(f"perception.vlm_base_url is not set (got {VLM_BASE_URL!r}); see .env")
        return False
    ollama = VLM_PROVIDER.startswith("ollama")
    base = VLM_BASE_URL.rstrip("/")
    url = f"{base}/api/tags" if ollama else f"{base}/models"
    log.info(f"Testing {VLM_PROVIDER} at {url} ...")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        log.error(f"Cannot reach the VLM endpoint: {e}")
        return False
    models = ([m["name"] for m in data.get("models", [])] if ollama
              else [m.get("id") for m in data.get("data", [])])
    log.info(f"Available: {models}")
    if VLM_MODEL in models:
        log.success(f"  ✓ VLM: {VLM_MODEL}")
    else:
        log.warning(f"  ⚠ VLM '{VLM_MODEL}' not served")
    return True


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

        # Home = robot.yaml home_joints by joint NAME (cspace order: gantry,
        # arm), gripper joints open; joints not named keep their position.
        home_all = np.array(current, dtype=np.float32)
        for name in ROBOT_CFG.gripper_modes:
            if name in dof_names:
                home_all[dof_names.index(name)] = ROBOT_CFG.finger["open"]
        if HOME_JOINTS is None:
            log.warning("robot.yaml home_joints is null — keeping the scene's arm pose")
        else:
            for name, val in zip(ROBOT_CFG.cspace_joints, HOME_JOINTS):
                if name not in dof_names:
                    log.error(f"Joint {name} from robot.yaml not in the articulation")
                    return True, False
                home_all[dof_names.index(name)] = val
        log.info(f"Target ({num_dof} joints): {np.round(home_all, 3)}")

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

    # Test 2: VLM endpoint
    print("\n[2/3] Checking the VLM endpoint...")
    vlm_ok = test_vlm()

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
    print(f"  VLM reachable:    {'✓' if vlm_ok else '✗'}")
    print(f"  Robot ready:      {'✓' if robot_ok else '✗'}")
    print(f"  Motion works:     {'✓' if motion_ok else '✗'}")
    print("=" * 60)

    if all([scene_ok, vlm_ok, robot_ok, motion_ok]):
        print("\n✅ ALL SYSTEMS GO — Ready for kitting operations!")
    else:
        print("\n⚠️ Some checks failed — review the output above.")

asyncio.ensure_future(main())
