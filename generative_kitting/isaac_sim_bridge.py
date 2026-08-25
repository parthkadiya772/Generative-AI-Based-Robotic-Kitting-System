"""
Isaac Sim ↔ Streamlit Bridge Server  (v2 — Neuro-Symbolic Kitting).

Run this script inside Isaac Sim's Script Editor.  It exposes an HTTP
API on port 8600 that the Streamlit dashboard consumes for:

  GET  /api/ping          → health check + capabilities list
  GET  /api/status        → robot joint positions, DOF info, sim state
  GET  /api/camera        → live camera frame (rgb|depth|wrist|kit) as
                            base64 JPEG; caches the depth buffer from the
                            same rendered frame for /api/project_to_world
  GET  /api/scene_annotations → USD ground truth, EVALUATION ONLY —
                            never used for grasp coordinates
  GET  /api/prim_center?prim=<path> → USD bbox centre/top/height for a
                            static fixture (bin, tray). Not for parts.
  POST /api/plan_test      → plan one pose with AND without the collision
                            world, to prove whether the cloud blocks it
  POST /api/jog            → nudge one joint via apply_action only (no
                            planner, no IK) and report if it moved
  POST /api/verify_planner → self-test: plan to the tool's own current
                            pose; a large joint delta means the
                            world->robot-root transform is wrong
  POST /api/build_collision_world → rebuild the cuMotion collision world
                            from the 4 corner cameras (structure only)
  POST /api/wrist_obstacles → add neighbour-part obstacles from the wrist
                            camera, carving out the target grasp point
  POST /api/execute       → execute a list of action primitives
  POST /api/joints        → set joint positions directly
  POST /api/home          → move robot to home position
  POST /api/gripper       → open / close gripper
  POST /api/approach      → move near a world XYZ (cuMotion plan)
  POST /api/realign_gantry → slide gantry to target X while the
                             arm compensates to hold the ee_link at
                             its current world position (no swing)
  POST /api/pick          → full pick sequence at XYZ
  POST /api/place         → full place sequence at XYZ

Usage in Isaac Sim Script Editor:
  exec(open("/path/to/robot_in_air/generative_kitting/kitting_bridge_server.py").read())
  # kitting_bridge_server.py auto-discovers KITTING_PROJECT_ROOT from your .env
  # and then exec()s this file — do not run isaac_sim_bridge.py directly.
"""

import sys, os, json, asyncio, threading, traceback, io, base64, time, tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

# ═════════════════════════════════════════════════════════════
# ENVIRONMENT
#
#   This file runs inside Isaac Sim's Script Editor — a DIFFERENT
#   process from Streamlit, which is what normally loads .env via
#   utils.config_loader. Without this, every os.environ.get below
#   returns "" and anything configured through .env (notably
#   ROBOT_CONFIG_PATH for cuMotion) silently comes up unset.
#
#   exec(open(...).read()) leaves __file__ undefined, so the project
#   root is discovered by walking up from whatever anchors exist.
# ═════════════════════════════════════════════════════════════

def _load_env_file():
    """Load <project_root>/.env into os.environ. Returns the path or None.

    Existing environment variables are NOT overwritten — a real shell
    export wins over the file, matching dotenv precedence and
    utils.config_loader's behaviour.
    """
    candidates = []
    try:
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass  # exec'd from the Script Editor — no __file__

    # The open USD scene (AIKIDO.usd) lives in the project root, which
    # makes it the most reliable anchor: Isaac Sim's cwd is usually its
    # own install directory, nowhere near this project.
    try:
        import omni.usd
        stage_url = omni.usd.get_context().get_stage_url() or ""
        if stage_url:
            local = stage_url.replace("file:///", "").replace("file://", "")
            candidates.append(os.path.dirname(os.path.abspath(local)))
    except Exception:
        pass

    candidates.append(os.environ.get("KITTING_PROJECT_ROOT", ""))
    candidates.append(os.getcwd())

    for start in candidates:
        if not start:
            continue
        d = os.path.abspath(start)
        for _ in range(6):                      # walk up a few levels
            env_path = os.path.join(d, ".env")
            if os.path.isfile(env_path):
                for raw in open(env_path, encoding="utf-8").read().splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip('"\''))
                return env_path
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    return None


_ENV_FILE = _load_env_file()
if _ENV_FILE:
    print(f"[env] loaded {_ENV_FILE}")
else:
    print("[env] WARNING: no .env found — ROBOT_CONFIG_PATH and friends "
          "must be set as real environment variables, or cuMotion will "
          "not initialise")


# ═════════════════════════════════════════════════════════════
# CONFIGURATION (mirrors the constants in robot_control.py)
# ═════════════════════════════════════════════════════════════

# Bridge host/port come from env so the operator can pin the listener
# to loopback (default) or open it to the LAN deliberately. The HTTP
# API has no authentication; binding to 0.0.0.0 lets anyone on the
# subnet call /api/execute, so the safe default is 127.0.0.1.
BRIDGE_HOST = os.environ.get("BRIDGE_HOST", "127.0.0.1")
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", "8600"))
ROBOT_PRIM  = "/World/gantry"

# Dual cameras
CAMERA_RGB_PRIM   = "/World/Camera"
CAMERA_DEPTH_PRIM = "/World/rsd555/Depth"
CAMERA_WRIST_PRIM = "/World/rsd555/Camera_OmniVision_OV9782_Color"
# Tray-overlook camera — used after place to verify the part actually
# landed in the kitting tray. Independent of the robot pose so the VLM
# always gets a clean top-down view of the destination.
CAMERA_KIT_PRIM   = "/World/Camera_Kit"
IMU_PRIM          = "/World/rsd555/Imu_Sensor"

# Robot geometry
UR10_BASE_PATH = "/World/gantry/gantry_home/ur10_flattened/ur10_instanceable/base_link"
EE_PATH        = "/World/gantry/gantry_home/ur10_flattened/ur10_instanceable/ee_link"
PLACE_BOX_PATH = "/World/box_840"

# Contact sensors — TWO of them, used for different phases.
#
#   1. Inner-finger PAD sensor — sits on the ``Finger4`` collider mesh
#      (the rectangular inner pad). Fires when the gripper closes
#      around a part. Used by ``_adaptive_close`` to terminate the
#      close motion with a firm grip.
#   2. Fingertip BOTTOM sensor — sits on the ``Fingertip`` collider
#      mesh (the curved bottom edge). Fires when the finger tip
#      touches the bin floor / part top during the descent. Used by
#      ``_descend_with_contact_stop`` to halt the Cartesian descent
#      before the gripper crashes into the surface, then lift back a
#      few millimetres so the fingers can close cleanly.
#
# IMPORTANT: an Isaac Sim Contact Sensor MUST be a direct child of a
# prim with ``UsdPhysics.CollisionAPI`` applied. The intermediate
# Xforms ``left_inner_finger`` and ``Fingertip_01`` only carry
# ``MeshCollisionAPI`` (a setting, not a collider), so a sensor
# parented under them fails to initialise with
#   "Contact Sensor needs to be created under another prim that has
#    collision api enabled on."
# The collider with ``CollisionAPI`` lives one level deeper on the
# actual ``Mesh`` prims. Both sensor paths below point INTO those
# mesh prims so the sensor finds its CollisionAPI parent.
CONTACT_SENSOR_PRIM     = "/World/gantry/gantry_home/ur10_flattened/robotiq_fixed_physics/Robotiq_2F_140_physics_edit/left_inner_finger/Fingertip_01/Fingertip/Contact_Sensor"
CONTACT_SENSOR_TIP_PRIM = CONTACT_SENSOR_PRIM

# HOME_JOINTS is captured from the USD scene at startup (see _init_robot).
# The robot is ceiling-mounted on the gantry — the correct rest pose depends
# on how the scene was authored, not on a hardcoded floor-mounted guess.
HOME_JOINTS = None  # set by _init_robot()

# Gantry
GANTRY_X_JOINT  = "/World/gantry/gantry_home/vagn/gantry_vagn_joint"
GANTRY_X_OFFSET = 1.27

# Gripper / finger joints
# Robotiq 2F-140: 0.0 rad = fully open (140mm span), 0.695 rad = fully closed (0mm)
# In simulation the finger pads are thick — the joint angle must overshoot
# well past the part width so the pads actually squeeze the surface.
FINGER_OPEN  = 0.0
FINGER_CLOSE = 0.5       # matches robot_control.py — physics stops fingers on part contact
FINGER_CLOSE_MIN = 0.55  # minimum grip (large parts ~80mm)
FINGER_CLOSE_MAX = 0.69  # near-closed grip (small parts ~15mm)
ROBOTIQ_MAX_STROKE = 0.140  # 140mm max opening
ROBOTIQ_MAX_RAD = 0.695     # joint rad at fully closed
GRIP_OVERSHOOT = 0.060   # 60mm extra closing past part width for firm contact
ENTRY_CLEARANCE = 0.020  # 20mm extra gap over part width for bin-entry pre-shape

# Gripper geometry
# 219.5 mm = measured ee_link → gripper_tcp distance for the
# Robotiq 2F-140 in this scene. Set by reading the world transform
# of the ``gripper_tcp`` Xform the operator added at the gripper's
# longest physical extension (the fingertip plane). The earlier
# value (0.150) corresponded to the FINGER PAD only — left a 60-70
# mm gap that wasn't modelled in IK, so Lula planned ee_link
# positions that put the actual fingertips below the bin floor and
# the gripper jammed under reaction force.
GRIPPER_TCP_OFFSET   = 0.19357
GRIPPER_BODY_WIDTH   = 0.160
GRIPPER_HALF_WIDTH   = GRIPPER_BODY_WIDTH / 2.0
HOVER_CLEARANCE      = 0.020
GRASP_LIFT           = 0.000  # Descent target: finger TCP exactly at part top.
                              # The earlier value (+0.040) was tuned for tall
                              # parts (~50 mm motor valves) and parked the
                              # finger PADS 40 mm ABOVE the part top — fine
                              # when the part body extends down past the
                              # pads, useless for flat parts (~5 mm gears)
                              # where the gripper closes on empty air.
                              #
                              # With GRASP_LIFT = 0 and the contact-aware
                              # descent enabled, the fingertip sensor stops
                              # the Cartesian Z motion the moment the tip
                              # touches the part (or the bin floor, for
                              # parts whose Z was under-projected). The
                              # descent then lifts back ``contact_lift_back``
                              # (default 5 mm) so the fingers have room to
                              # close cleanly around the part edge — works
                              # uniformly for thin and tall parts without
                              # per-shape tuning.
                              #
                              # For parts that NEED to be gripped lower
                              # than their visible top (e.g. tall valves
                              # where you want to clamp mid-body, not the
                              # cap), pass ``grasp_lift_override=-0.040``
                              # in the bridge ``/api/pick_descend`` payload
                              # — the workflow can read this from the
                              # parts catalogue per part type.
GRASP_DEPTH_FRACTION = 0.45   # only used by compute_grasp_geometry (legacy/debug)
BOX_HEIGHT           = 0.05
BOX_ENTRY_MARGIN     = 0.15
PLACE_DROP_HEIGHT    = 0.02
PLACE_RETRACT_HEIGHT = 0.25
TRANSIT_SAFE_HEIGHT  = 0.14  # just 10cm above safe_z — enough to clear bin rim without
                             # going so high that Lula IK folds the ceiling-mounted arm
DOWNWARD_ORIENTATION = np.array([1.0, 0.0, 1.0, 0.0])

# Interpolation parameters for smooth motion
INTERP_MAX_STEP  = 0.15   # rad per streaming waypoint (smaller = smoother path)
INTERP_THRESHOLD = 0.5    # rad — only interpolate if any joint jumps more than this
STREAM_FRAMES    = 3      # physics frames per streaming waypoint (low = continuous motion)
SETTLE_FRAMES    = 60     # physics frames to settle at final position


# ═════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════

def normalize_quat(q):
    mag = np.linalg.norm(q)
    if mag < 1e-9:
        raise ValueError(f"Quaternion {q} is near-zero")
    return q / mag


def get_world_pos(prim_path):
    import omni.usd
    from isaacsim.core.utils.prims import get_prim_at_path
    return np.array(omni.usd.get_world_transform_matrix(
        get_prim_at_path(prim_path)).ExtractTranslation())


def compute_bbox_geometry(stage, prim_path, label="PRIM"):
    from pxr import Gf, UsdGeom, Usd
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: '{prim_path}'")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True)
    world_bound = bbox_cache.ComputeWorldBound(prim)
    aligned_box = world_bound.ComputeAlignedRange()
    min_pt = np.array(aligned_box.GetMin())
    max_pt = np.array(aligned_box.GetMax())
    height = float(max_pt[2] - min_pt[2])
    center_xy = [float((min_pt[0]+max_pt[0])/2), float((min_pt[1]+max_pt[1])/2)]
    return {
        "center_xy": center_xy,
        "pivot": get_world_pos(prim_path),
        "top_z": float(max_pt[2]),
        "bot_z": float(min_pt[2]),
        "height": height,
        "min_pt": min_pt, "max_pt": max_pt,
    }


def compute_grasp_geometry(stage, part_path, depth_frac=0.5):
    bbox = compute_bbox_geometry(stage, part_path, "PART")
    grasp_z = bbox["top_z"] - depth_frac * bbox["height"]
    # Compute the narrowest XY dimension for adaptive grip
    dims = bbox["max_pt"] - bbox["min_pt"]
    part_width = float(min(dims[0], dims[1]))  # narrowest side for grip
    return {
        "part_center_xy": bbox["center_xy"],
        "part_top_z": bbox["top_z"],
        "part_bot_z": bbox["bot_z"],
        "part_height": bbox["height"],
        "part_width": part_width,
        "grasp_z": grasp_z,
    }


def compute_adaptive_finger_close(part_width):
    """Compute gripper close value adapted to part size.

    The Robotiq 2F-140 finger pads are thick — the joint angle must
    overshoot well past the part width so the pads actually squeeze
    firmly against the surface.  GRIP_OVERSHOOT (40mm) accounts for
    pad thickness + compression needed for a secure hold.

    Motor valve (~80mm) -> 0.50 rad (pads squeeze at ~40mm gap)
    Small tube  (~30mm) -> 0.63 rad (pads squeeze at ~10mm gap)
    Small hinge (~15mm) -> 0.68 rad (near-closed, tight grip)
    """
    # Target opening = part width minus overshoot (pads need to compress past surface)
    desired_opening = max(0.0, part_width - GRIP_OVERSHOOT)
    # Convert opening (metres) to joint angle (radians)
    # 0.0 rad = 140mm open, 0.695 rad = 0mm closed
    finger_rad = ROBOTIQ_MAX_RAD * (1.0 - desired_opening / ROBOTIQ_MAX_STROKE)
    clamped = float(np.clip(finger_rad, FINGER_CLOSE_MIN, FINGER_CLOSE_MAX))
    print(f"  [GRIP] part_width={part_width*1000:.1f}mm overshoot={GRIP_OVERSHOOT*1000:.0f}mm -> finger_close={clamped:.3f} rad")
    return clamped


def compute_entry_finger_shape(part_width):
    """Finger angle for bin entry: wide enough to pass over part, narrow enough for bin walls.

    Pre-shapes the gripper BEFORE descending into the bin so the open (140mm)
    fingers cannot clip the bin walls.  The result is ENTRY_CLEARANCE (20mm)
    wider than the part → fingers clear the part during the descent, then a
    slow final close makes contact from the sides rather than top-down.

    Returns a value between FINGER_OPEN and FINGER_CLOSE_MIN.
    """
    desired_opening = part_width + ENTRY_CLEARANCE
    desired_opening = min(desired_opening, ROBOTIQ_MAX_STROKE)
    finger_rad = ROBOTIQ_MAX_RAD * (1.0 - desired_opening / ROBOTIQ_MAX_STROKE)
    # Must be wider than minimum grip (less closed) — we are still entering
    return float(np.clip(finger_rad, FINGER_OPEN, FINGER_CLOSE_MIN - 0.05))


def apply_arm_joints(robot, dof_names, arm_names, ik_result):
    targets = robot.get_joint_positions()
    for i, lula_name in enumerate(arm_names):
        for idx, dof_name in enumerate(dof_names):
            if lula_name in dof_name:
                targets[idx] = ik_result[i]
                break
    return targets


def set_finger_joints(robot, dof_names, value, base_targets=None):
    """Set gripper joint targets for encompassing (adaptive) grip.

    Robotiq 2F-140 joint geometry:
      finger_joint = +value  (outer phalanx rotates inward)
      inner_finger_joint = 0 (inner pad stays at 0 RELATIVE TO outer phalanx)
        → inner pad absolute angle = outer_angle + 0 = outer_angle
        → inner pad points inward/downward with the outer phalanx
        → creates an encompassing undercut that prevents the part from
          falling straight down during vertical lift (adaptive/encompassing mode)

    The old behaviour was inner_finger_joint = -value, which counter-rotates
    the inner pad to keep it vertical (absolute 0 rad = flat/parallel contact).
    That grip has NO undercut — only side friction holds the part during lift.
    """
    targets = base_targets if base_targets is not None else robot.get_joint_positions()
    for idx, name in enumerate(dof_names):
        if "inner_finger_joint" in name:
            targets[idx] = 0.0  # encompassing: inner pad follows outer phalanx direction
        elif "finger_joint" in name or "knuckle_joint" in name:
            targets[idx] = value
    return targets


# ═════════════════════════════════════════════════════════════
# COLLISION WORLD  (4-camera point cloud → cuMotion sphere colliders)
#
#   Four corner cameras render Replicator "pointcloud" annotators. The
#   merged cloud is voxel-downsampled and fed to cuMotion as spheres,
#   one per voxel, radius = voxel/2 so the set is watertight.
#
#   Points near known grasp targets are carved out: an obstacle sitting
#   on the grasp point makes the pick unplannable by construction. The
#   targets come from the perception pipeline (camera + depth), so this
#   path involves no USD lookup at all.
#
#   Local awareness of neighbouring parts comes later, from the wrist
#   camera at standoff height — see _add_wrist_obstacles().
# ═════════════════════════════════════════════════════════════

POINTCLOUD_CAMERAS = {
    "/World/pointcloud_view/cam1": (480, 640),
    "/World/pointcloud_view/cam2": (480, 640),
    "/World/pointcloud_view/cam3": (480, 640),
    "/World/pointcloud_view/cam4": (480, 640),
}

COLLISION_VOXEL_SIZE = 0.02       # 2 cm — trades fidelity for sphere count
COLLISION_MAX_SPHERES = 6000      # hard cap; planner cost grows with count
COLLISION_SAFETY_MARGIN = 0.005   # 5 mm inflation on every obstacle
# Radius of the XY cylinder cleared around each known grasp target so
# the part being picked never becomes its own obstacle.
TARGET_CARVE_RADIUS = 0.06

_pc_annotators = {}               # cam_path -> replicator annotator


def _voxel_downsample(points, voxel_size):
    """Keep one point per occupied voxel."""
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


async def _capture_pointcloud(cameras=None, ticks=5):
    """Render the corner cameras and return merged world-frame points (N,3).

    Replicator's ``pointcloud`` annotator already returns WORLD-frame
    points, so no camera transform is applied here.
    """
    import omni.replicator.core as rep
    import omni.usd

    cameras = cameras or POINTCLOUD_CAMERAS
    stage = omni.usd.get_context().get_stage()

    for cam_path, res in cameras.items():
        if cam_path in _pc_annotators:
            continue
        if not stage.GetPrimAtPath(cam_path).IsValid():
            print(f"  [pcd] missing camera prim: {cam_path}")
            continue
        rp = rep.create.render_product(cam_path, res)
        annot = rep.AnnotatorRegistry.get_annotator(
            "pointcloud", init_params={"includeUnlabelled": True})
        annot.attach(rp)
        _pc_annotators[cam_path] = annot
        print(f"  [pcd] attached annotator to {cam_path} @ {res}")

    if not _pc_annotators:
        return np.empty((0, 3))

    for _ in range(ticks):
        await rep.orchestrator.step_async()

    clouds = []
    for cam_path, annot in _pc_annotators.items():
        data = annot.get_data()
        if not data or "data" not in data or len(data["data"]) == 0:
            print(f"  [pcd] {cam_path}: empty buffer")
            continue
        pts = np.asarray(data["data"]).reshape(-1, 3)
        # Replicator emits (0,0,0) for rays that hit nothing (skybox).
        pts = pts[np.linalg.norm(pts, axis=1) > 0.01]
        if len(pts):
            clouds.append(pts)
            print(f"  [pcd] {cam_path}: {len(pts)} points")

    if not clouds:
        return np.empty((0, 3))
    merged = np.vstack(clouds)
    lo, hi = merged.min(axis=0), merged.max(axis=0)
    print(f"  [pcd] merged world extent: "
          f"x[{lo[0]:+.2f}, {hi[0]:+.2f}]  "
          f"y[{lo[1]:+.2f}, {hi[1]:+.2f}]  "
          f"z[{lo[2]:+.2f}, {hi[2]:+.2f}]  ({len(merged)} pts)")
    return merged


def _carve_around_points(points, targets, radius):
    """Remove cloud points near known grasp targets.

    Returns ``(kept, n_removed)``.

    The parts we intend to pick must NOT become collision geometry — a
    sphere sitting on the grasp point makes the pick unplannable by
    construction. ``targets`` are the world XYZ the perception pipeline
    already produced from camera + depth, so this needs no USD lookup
    and no bin bounding box.

    Carving is by XY cylinder rather than a sphere: the gripper descends
    vertically, so the whole column above a part has to stay clear.
    """
    if len(points) == 0 or not targets:
        return points, 0
    tgt = np.asarray(targets, dtype=np.float64)[:, :2]
    d = np.linalg.norm(points[:, None, :2] - tgt[None, :, :], axis=2)
    near = (d <= radius).any(axis=1)
    return points[~near], int(near.sum())


def _spheres_from_points(points, prefix, radius, margin=COLLISION_SAFETY_MARGIN):
    """Push one sphere collider per point into the cuMotion world."""
    import warp as wp

    n = len(points)
    if n == 0:
        return 0

    prim_paths = [f"{prefix}/{i}" for i in range(n)]
    with wp.ScopedDevice(STATE.cumotion_device):
        radii = wp.array(np.full(n, radius, dtype=np.float32), dtype=wp.float32)
        scales = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3)
        tols = wp.array(np.full(n, margin, dtype=np.float32), dtype=wp.float32)
        positions = wp.array(points.astype(np.float32), dtype=wp.vec3)
        quats = wp.array(
            np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32), dtype=wp.vec4)
        enabled = wp.array(np.ones(n, dtype=np.int32), dtype=wp.int32)

    STATE.cumotion_world.add_spheres(
        prim_paths=prim_paths, radii=radii, scales=scales,
        safety_tolerances=tols, poses=(positions, quats),
        enabled_array=enabled)
    return n


async def _plan_test(params=None):
    """Plan the same pose twice: with the collision world, and without.

    Answers one question directly — is the point cloud what is blocking
    the planner? A temporary EMPTY world + planner is built for the
    second attempt, so the live collision world is left untouched.

    The usual culprit is the robot itself: the corner cameras see the
    arm, so its geometry ends up in the cloud, and the robot's own XRDF
    spheres then overlap those obstacles. cuMotion reports the START
    configuration as in-collision and every plan fails, including
    trivial ones.

    Params:
        position / orientation: target pose. Defaults to the tool's own
            current pose lifted by ``lift`` metres.
        lift: default 0.15
    """
    import omni.usd
    from isaacsim.robot_motion.cumotion import (CumotionWorldInterface,
                                                GraphBasedMotionPlanner)

    params = params or {}
    if not STATE.cumotion_ready:
        return {"error": _planner_unavailable_reason()}

    stage = omni.usd.get_context().get_stage()
    tool_mat = omni.usd.get_world_transform_matrix(
        stage.GetPrimAtPath(TOOL_FRAME_PRIM))
    t = tool_mat.ExtractTranslation()
    lift = float(params.get("lift", 0.15))

    ori = params.get("orientation") or normalize_quat(
        DOWNWARD_ORIENTATION).tolist()
    ori = np.array(ori, dtype=np.float64)
    q0 = _current_cspace()
    tool = np.array([float(t[0]), float(t[1]), float(t[2])])

    # A reachability sweep, not a single guess. This arm is ceiling
    # mounted, so "up" heads toward the mount and runs out of envelope;
    # the useful workspace is BELOW. Probing several offsets shows where
    # the boundary is instead of leaving one failure ambiguous.
    if params.get("position"):
        probes = [("requested", np.array(params["position"], dtype=np.float64))]
    else:
        probes = [
            ("current pose (no move)", tool.copy()),
            (f"+{lift:.2f} m Z (up, toward mount)", tool + [0, 0, lift]),
            (f"-{lift:.2f} m Z (down, into cell)", tool + [0, 0, -lift]),
            ("-0.50 m Z (well below)", tool + [0, 0, -0.50]),
            ("-1.00 m Z (bin height)", tool + [0, 0, -1.00]),
        ]

    print(f"  [plan-test] tool now ({tool[0]:.3f}, {tool[1]:.3f}, {tool[2]:.3f})")
    print(f"  [plan-test] q_start = [{', '.join(f'{v:+.3f}' for v in q0)}]")
    if STATE.collision_sphere_count == 0:
        print("  [plan-test] NOTE: the collision world is EMPTY, so the "
              "with/without comparison is degenerate. Call "
              "/api/build_collision_world first to test obstacles.")

    reach = []
    for label, tgt in probes:
        found = STATE.cumotion_planner.plan_to_pose_target(
            q_initial=q0, position=tgt, orientation=ori) is not None
        reach.append({"label": label,
                      "target": [float(v) for v in tgt],
                      "path_found": bool(found)})
        print(f"  [plan-test]   {label:34s} z={tgt[2]:+.3f}  "
              f"{'PATH FOUND' if found else 'NO PATH'}")

    pos = np.array(probes[1][1] if len(probes) > 1 else probes[0][1],
                   dtype=np.float64)

    # A — the live world, however many spheres it holds.
    with_world = STATE.cumotion_planner.plan_to_pose_target(
        q_initial=q0, position=pos, orientation=ori) is not None
    print(f"  [plan-test] WITH obstacles    "
          f"({STATE.collision_sphere_count} spheres): "
          f"{'PATH FOUND' if with_world else 'NO PATH'}")

    # B — a throwaway empty world, so the live one is not disturbed.
    empty_world = CumotionWorldInterface(
        world_to_robot_base=_robot_base_pose_arrays(),
        device=STATE.cumotion_device)
    empty_planner = GraphBasedMotionPlanner(
        cumotion_robot=STATE.cumotion_robot,
        cumotion_world_interface=empty_world,
        tool_frame=CUMOTION_TOOL_FRAME)
    empty_world.world_view.update()
    without_world = empty_planner.plan_to_pose_target(
        q_initial=q0, position=pos, orientation=ori) is not None
    print(f"  [plan-test] WITHOUT obstacles (0 spheres): "
          f"{'PATH FOUND' if without_world else 'NO PATH'}")

    if without_world and not with_world:
        verdict = ("The collision world is blocking the planner. Almost "
                   "certainly the robot's own body is in the point cloud "
                   "(the corner cameras see the arm), so the START "
                   "configuration is in collision.")
    elif not without_world and not with_world:
        verdict = ("Blocked even with NO obstacles — the pose is "
                   "unreachable or the start config is invalid. Not a "
                   "point-cloud problem.")
    elif with_world:
        verdict = "Planning succeeds with obstacles; the cloud is not the blocker."
    else:
        verdict = "Plans with obstacles but not without — unexpected."
    print(f"  [plan-test] {verdict}")

    return {"status": "ok",
            "target": [float(v) for v in pos],
            "reachability": reach,
            "with_obstacles": with_world,
            "without_obstacles": without_world,
            "collision_spheres": STATE.collision_sphere_count,
            "verdict": verdict}


async def _jog_joint(params):
    """Nudge one joint and report whether it actually moved.

    Deliberately bypasses cuMotion, trajectories and IK: it reads the
    joint positions, adds a delta to one index, calls apply_action, ticks
    physics, and reads back. If this does not move the robot then nothing
    else will, and the problem is in the articulation/physics setup
    rather than anywhere in the planning stack.
    """
    import omni.kit.app
    import omni.timeline
    from isaacsim.core.utils.types import ArticulationAction

    if STATE.robot is None:
        return {"error": "robot not initialised"}

    index = int(params.get("index", 1))          # default: shoulder_pan
    delta = float(params.get("delta", 0.2))      # rad (or m for gantry)
    frames = int(params.get("frames", 120))

    timeline = omni.timeline.get_timeline_interface()
    playing = bool(timeline.is_playing())

    q0 = np.array(STATE.robot.get_joint_positions(), dtype=np.float64)
    if not (0 <= index < len(q0)):
        return {"error": f"index {index} out of range (0..{len(q0)-1})"}

    target = q0.copy()
    target[index] = q0[index] + delta

    name = (STATE.dof_names[index]
            if STATE.dof_names and index < len(STATE.dof_names) else "?")
    print(f"  [jog] timeline playing = {playing}")
    print(f"  [jog] joint[{index}] '{name}': {q0[index]:+.4f} -> "
          f"{target[index]:+.4f} (delta {delta:+.3f})")

    STATE.robot.apply_action(ArticulationAction(joint_positions=target))
    for _ in range(frames):
        await omni.kit.app.get_app().next_update_async()

    q1 = np.array(STATE.robot.get_joint_positions(), dtype=np.float64)
    moved = float(q1[index] - q0[index])
    ok = abs(moved) > abs(delta) * 0.1

    print(f"  [jog] result: {q1[index]:+.4f}  (moved {moved:+.4f})  "
          f"-> {'OK' if ok else 'DID NOT MOVE'}")
    if not ok:
        if not playing:
            print("  [jog] TIMELINE IS STOPPED — press Play. apply_action "
                  "only sets drive targets; physics must be stepping.")
        else:
            print("  [jog] Timeline is playing, so the drive is not "
                  "responding: check this joint's stiffness / max force "
                  "in the Physics Inspector (0 stiffness = no motion).")

    return {"status": "ok", "moved": ok,
            "timeline_playing": playing,
            "joint_index": index, "joint_name": name,
            "before": float(q0[index]), "after": float(q1[index]),
            "delta_commanded": delta, "delta_actual": moved,
            "all_joints_before": [float(v) for v in q0],
            "all_joints_after": [float(v) for v in q1]}


async def _verify_planner(params=None):
    """Plan to the tool's CURRENT pose and report how far the goal moved.

    This is the one check that validates ROBOT_ROOT_PATH. The planner
    works in robot-root coordinates, so if that transform is wrong every
    world-frame target is offset by the same error — silently, since
    planning still succeeds.

    Asking for the pose the tool is ALREADY at must return a goal
    configuration equal to the current one. A large delta means the
    world->robot-root transform is wrong; check ROBOT_ROOT_PATH against
    the URDF's root link (``gantry``).
    """
    import omni.usd

    if not STATE.cumotion_ready:
        return {"error": _planner_unavailable_reason()}

    stage = omni.usd.get_context().get_stage()
    for path in (ROBOT_ROOT_PATH, TOOL_FRAME_PRIM):
        if not stage.GetPrimAtPath(path).IsValid():
            return {"error": f"prim not found: {path}"}

    # Report every plausible root so an offset between them is visible.
    # The URDF's root link is `gantry`, so ROBOT_ROOT_PATH should be the
    # prim matching THAT, not an intermediate grouping Xform.
    candidates = {}
    for cand in ("/World/gantry", "/World/gantry/gantry_home",
                 ROBOT_ROOT_PATH):
        prim = stage.GetPrimAtPath(cand)
        if prim.IsValid():
            ct = omni.usd.get_world_transform_matrix(prim).ExtractTranslation()
            candidates[cand] = [float(ct[0]), float(ct[1]), float(ct[2])]
    print("  [verify] candidate root prims:")
    for k, v in candidates.items():
        mark = "  <- ROBOT_ROOT_PATH" if k == ROBOT_ROOT_PATH else ""
        print(f"             {k}  ({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}){mark}")

    root_mat = omni.usd.get_world_transform_matrix(
        stage.GetPrimAtPath(ROBOT_ROOT_PATH))
    root_t = root_mat.ExtractTranslation()

    tool_mat = omni.usd.get_world_transform_matrix(
        stage.GetPrimAtPath(TOOL_FRAME_PRIM))
    t = tool_mat.ExtractTranslation()
    q = tool_mat.ExtractRotationQuat()
    imag = q.GetImaginary()
    tool_pos = [float(t[0]), float(t[1]), float(t[2])]
    tool_quat = [float(q.GetReal()), float(imag[0]),
                 float(imag[1]), float(imag[2])]

    q_now = _current_cspace()

    # Plan straight to the tool frame (no TCP offset — we want the pose
    # cuMotion itself reasons about).
    path_obj = STATE.cumotion_planner.plan_to_pose_target(
        q_initial=q_now, position=np.array(tool_pos),
        orientation=np.array(tool_quat))
    if path_obj is None:
        return {"error": "planner found NO PATH to the tool's own current "
                         "pose — ROBOT_ROOT_PATH is almost certainly wrong",
                "robot_root": ROBOT_ROOT_PATH,
                "robot_root_world_pos": [float(v) for v in root_t],
                "tool_world_pos": tool_pos}

    traj = _path_to_trajectory(path_obj)
    state = traj.get_target_state(float(traj.duration))
    goal = np.asarray(state.joints.positions).flatten()
    indices = state.joints.position_indices
    if indices is not None and STATE.cumotion_dof_indices:
        by_dof = {int(d): goal[s]
                  for s, d in enumerate(np.asarray(indices).flatten())}
        goal = np.array([by_dof.get(d, 0.0)
                         for d in STATE.cumotion_dof_indices])

    delta = np.abs(goal - q_now)
    max_delta = float(delta.max())

    # Joint delta is NOT a valid pass/fail metric for this robot. With
    # the gantry plus 6 arm joints the arm is redundant: a whole family
    # of configurations reaches the same tool pose, so a sampling-based
    # planner can legitimately return a different one. What matters is
    # whether the ACHIEVED tool pose matches the requested one, which
    # needs forward kinematics on the goal configuration.
    kin = getattr(STATE.cumotion_robot, "kinematics", None)
    fk_methods = sorted(m for m in dir(kin) if not m.startswith("_")) if kin else []
    print(f"  [verify] kinematics API: {fk_methods}")

    fk_pos = None
    for name in ("compute_forward_kinematics", "forward_kinematics",
                 "compute_link_pose", "link_pose", "pose", "forward"):
        fn = getattr(kin, name, None)
        if not callable(fn):
            continue
        for args in ((goal,), (goal, CUMOTION_TOOL_FRAME),
                     (CUMOTION_TOOL_FRAME, goal)):
            try:
                out = fn(*args)
            except Exception:
                continue
            try:
                arr = np.asarray(
                    out.translation if hasattr(out, "translation") else out,
                    dtype=np.float64).flatten()
                if arr.size >= 3:
                    fk_pos = arr[:3]
                    print(f"  [verify] FK via kinematics.{name}")
                    break
            except Exception:
                continue
        if fk_pos is not None:
            break

    if fk_pos is not None:
        # FK is expressed in the ROBOT-BASE frame. Rather than lifting it
        # to world by hand (which needs the base ROTATION, not just its
        # translation — this arm is ceiling-mounted, so ignoring rotation
        # produces a large bogus error), convert the REQUESTED pose into
        # the base frame using the very function the planner uses, and
        # compare there. Same frame on both sides, no hand-rolled maths.
        from isaacsim.robot_motion.cumotion.impl.utils import (
            isaac_sim_to_cumotion_pose)

        pos_w2b, quat_w2b = (
            STATE.cumotion_world.get_world_to_robot_base_transform())
        pose_base_target = isaac_sim_to_cumotion_pose(
            position_world_to_target=np.array(tool_pos, dtype=np.float32),
            orientation_world_to_target=np.array(tool_quat, dtype=np.float32),
            position_world_to_base=pos_w2b,
            orientation_world_to_base=quat_w2b)

        requested_base = None
        for attr in ("translation", "position", "t"):
            val = getattr(pose_base_target, attr, None)
            if val is not None:
                try:
                    arr = np.asarray(
                        val.numpy() if hasattr(val, "numpy") else val,
                        dtype=np.float64).flatten()
                    if arr.size >= 3:
                        requested_base = arr[:3]
                        break
                except Exception:
                    continue

        if requested_base is None:
            pos_err = None
            ok = max_delta < 0.35
            print(f"  [verify] could not read translation from "
                  f"{type(pose_base_target).__name__} "
                  f"(attrs: {[a for a in dir(pose_base_target) if not a.startswith('_')]}) "
                  f"— falling back to joint delta")
        else:
            # We do not know for certain whether the FK binding reports in
            # the base frame or already in world. Measure against both and
            # take the smaller — the frame it is actually in will match to
            # sub-millimetre, the other will be off by the base offset.
            err_base = float(np.linalg.norm(fk_pos - requested_base))
            err_world = float(np.linalg.norm(fk_pos - np.array(tool_pos)))
            frame = "base" if err_base <= err_world else "world"
            pos_err = min(err_base, err_world)
            ok = pos_err < 0.005

            print(f"  [verify] requested (base frame)  ({requested_base[0]:+.4f}, "
                  f"{requested_base[1]:+.4f}, {requested_base[2]:+.4f})")
            print(f"  [verify] requested (world frame) ({tool_pos[0]:+.4f}, "
                  f"{tool_pos[1]:+.4f}, {tool_pos[2]:+.4f})")
            print(f"  [verify] FK of goal config       ({fk_pos[0]:+.4f}, "
                  f"{fk_pos[1]:+.4f}, {fk_pos[2]:+.4f})")
            print(f"  [verify] error vs base = {err_base*1000:8.2f} mm")
            print(f"  [verify] error vs world= {err_world*1000:8.2f} mm")
            print(f"  [verify] FK appears to report in the {frame.upper()} frame")
            print(f"  [verify] POSITION ERROR = {pos_err*1000:.2f} mm -> "
                  f"{'PASS' if ok else 'FAIL'}")
    else:
        pos_err = None
        ok = max_delta < 0.35
        print("  [verify] no usable FK binding; falling back to joint delta. "
              "Redundancy makes this only a rough check.")

    print(f"  [verify] robot_root={ROBOT_ROOT_PATH} at "
          f"({root_t[0]:.3f}, {root_t[1]:.3f}, {root_t[2]:.3f})")
    print(f"  [verify] tool world pos ({tool_pos[0]:.3f}, "
          f"{tool_pos[1]:.3f}, {tool_pos[2]:.3f})")
    print(f"  [verify] max |dq| = {max_delta:.4f} (informational: the arm is "
          f"redundant, so a different config for the same pose is normal)")

    return {"status": "ok", "passed": ok,
            "position_error_m": pos_err,
            "kinematics_api": fk_methods,
            "max_joint_delta": max_delta,
            "joint_delta": [float(v) for v in delta],
            "robot_root": ROBOT_ROOT_PATH,
            "robot_root_world_pos": [float(v) for v in root_t],
            "root_candidates": candidates,
            "tool_world_pos": tool_pos,
            "collision_spheres": STATE.collision_sphere_count}


async def _build_collision_world(params=None):
    """Rebuild the collision world from the 4 corner cameras.

    Params:
        voxel_size: grid size for downsampling (default 2 cm)
        exclude_points: [[x, y, z], ...] world coords of parts to be
            picked. An XY cylinder of ``carve_radius`` is cleared around
            each so the target never becomes its own obstacle.
        carve_radius: cylinder radius in metres (default 6 cm)

    Call once per scan cycle, after perception, before planning the pick.
    """
    params = params or {}
    if STATE.cumotion_world is None and not _init_cumotion():
        return {"error": "cuMotion not initialised"}

    voxel = float(params.get("voxel_size", COLLISION_VOXEL_SIZE))

    raw = await _capture_pointcloud()
    if len(raw) == 0:
        return {"error": "no point cloud data from any camera"}

    targets = params.get("exclude_points") or []
    carve_r = float(params.get("carve_radius", TARGET_CARVE_RADIUS))
    kept, n_carved = _carve_around_points(raw, targets, carve_r)
    voxels = _voxel_downsample(kept, voxel)

    if len(voxels) > COLLISION_MAX_SPHERES:
        keep = np.random.default_rng(0).choice(
            len(voxels), COLLISION_MAX_SPHERES, replace=False)
        print(f"  [pcd] capping {len(voxels)} → {COLLISION_MAX_SPHERES} spheres")
        voxels = voxels[keep]

    # Fresh world each rebuild — cuMotion has no per-prefix removal.
    _reset_cumotion_world()
    n = _spheres_from_points(voxels, "/collision/structure", voxel / 2.0)
    STATE.cumotion_world.world_view.update()
    STATE.collision_sphere_count = n

    print(f"  [pcd] collision world: {len(raw)} raw → {n_carved} carved around "
          f"{len(targets)} target(s) @ {carve_r*100:.0f}cm "
          f"→ {len(voxels)} voxels @ {voxel*100:.0f}cm → {n} spheres")
    if not targets:
        print("  [pcd] NOTE: no exclude_points given — every part in the bin "
              "is an obstacle, so grasp poses will not plan. Pass the "
              "detected part coordinates.")
    return {"status": "ok", "raw_points": int(len(raw)),
            "carved": n_carved, "targets": len(targets),
            "carve_radius": carve_r, "spheres": n,
            "voxel_size": voxel}


async def _add_wrist_obstacles(params):
    """Add neighbour-part obstacles from the wrist camera at standoff.

    Captured at ``pick_align_height`` rather than grasp height: at grasp
    height the fingers occlude the target and the camera's offset from
    the TCP makes the view geometrically poor.

    A cylinder of ``carve_radius`` around the grasp XY is EXCLUDED, so
    the target itself never becomes an obstacle — otherwise the descent
    to grasp is unplannable. Everything else the wrist sees (neighbouring
    parts, bin dividers) becomes a collider.
    """
    if STATE.cumotion_world is None:
        return {"error": "cuMotion not initialised"}

    grasp_xy = params.get("grasp_xy")
    if not (isinstance(grasp_xy, (list, tuple)) and len(grasp_xy) >= 2):
        return {"error": "grasp_xy [x, y] required"}
    carve_r = float(params.get("carve_radius", 0.06))
    voxel = float(params.get("voxel_size", 0.01))   # finer: wrist is close

    raw = await _capture_pointcloud(
        cameras={CAMERA_WRIST_PRIM: (480, 640)}, ticks=4)
    if len(raw) == 0:
        return {"error": "wrist point cloud empty"}

    d_xy = np.linalg.norm(raw[:, :2] - np.array(grasp_xy[:2]), axis=1)
    neighbours = raw[d_xy > carve_r]
    n_carved = int((d_xy <= carve_r).sum())

    voxels = _voxel_downsample(neighbours, voxel)
    n = _spheres_from_points(voxels, "/collision/wrist_local", voxel / 2.0)
    STATE.cumotion_world.world_view.update()

    print(f"  [pcd-wrist] {len(raw)} raw → {n_carved} carved around target "
          f"→ {n} neighbour spheres")
    return {"status": "ok", "spheres": n, "carved": n_carved,
            "carve_radius": carve_r}


# ═════════════════════════════════════════════════════════════
# CUMOTION MOTION PLANNING
#
#   Replaces Lula IK. Every motion is planned against the collision
#   world above, so the arm routes AROUND the gantry and bin walls
#   instead of through them.
#
#   ``ik_solve`` keeps its original ``(action, ok)`` signature so all
#   existing call sites work unchanged — but it now plans, and stashes
#   the resulting trajectory on ``STATE.pending_trajectory``. The very
#   next ``_apply_interpolated`` consumes that trajectory and follows
#   the planned path instead of interpolating straight to the goal.
#   That coupling is what makes ~20 call sites collision-aware without
#   rewriting each one; it is deliberate, and single-use so a stale plan
#   can never be replayed.
# ═════════════════════════════════════════════════════════════

ROBOT_CONFIG_PATH = os.environ.get("ROBOT_CONFIG_PATH", "")
# Exported from Isaac Sim. Mirrors simulation.robot_{urdf,xrdf}_filename
# in config.yaml — the bridge runs standalone and cannot read that file.
ROBOT_URDF_FILENAME = "AIKIDO.urdf"
ROBOT_XRDF_FILENAME = "AIKIDO.xrdf"
CUMOTION_TOOL_FRAME = "robotiq_base_link"
# Transfer the ee_link-derived DOWNWARD_ORIENTATION into the tool frame
# using the fixed rotation between the two, measured from USD. Set False
# to send the raw ee_link quaternion (the old, incorrect behaviour).
USE_MEASURED_TOOL_ORIENTATION = True
# A measured tool->TCP distance below this is not believable: the prim
# transforms are coincident and the gripper offset lives in mesh data,
# so the configured GRIPPER_TCP_OFFSET is kept instead.
MIN_CREDIBLE_TCP_OFFSET = 0.05


def quat_mul(a, b):
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    h = angle / 2.0
    return np.array([np.cos(h), *(axis * np.sin(h))])


def derive_down_quat(tool_p, tool_q, tcp_p):
    """Tool orientation that aims the gripper's approach axis at world -Z.

    Derived from the scene rather than from a constant. The approach axis
    is the direction from the tool frame to the authored TCP; rotating
    that onto world -Z and applying the same rotation to the tool's
    current orientation gives an orientation that is correct by
    construction, whatever the gripper mount happens to be.

    This replaces transferring the Lula-era DOWNWARD_ORIENTATION, which
    described ee_link and has not been valid since the frame change.
    """
    approach = np.asarray(tcp_p, dtype=np.float64) - np.asarray(tool_p, dtype=np.float64)
    norm = np.linalg.norm(approach)
    if norm < 1e-9:
        return None, None
    approach = approach / norm
    target = np.array([0.0, 0.0, -1.0])

    axis = np.cross(approach, target)
    dot = float(np.clip(np.dot(approach, target), -1.0, 1.0))
    if np.linalg.norm(axis) < 1e-8:
        # Parallel or anti-parallel: no unique axis, so pick any
        # perpendicular one. Anti-parallel needs a half turn.
        q_align = (np.array([1.0, 0.0, 0.0, 0.0]) if dot > 0
                   else quat_from_axis_angle([1.0, 0.0, 0.0], np.pi))
    else:
        q_align = quat_from_axis_angle(axis, np.arccos(dot))

    q = quat_mul(q_align, np.asarray(tool_q, dtype=np.float64))
    return q / np.linalg.norm(q), approach

# Fixed rail mount — the URDF root, above gantry_vagn_joint. See
# _robot_base_pose_arrays for why this must not be base_link.
# The URDF's root link is `gantry`, and this must be the prim that
# matches it: the FIXED rail base. /World/gantry/gantry_home is the
# moving carriage — verify_planner showed it sitting at x=+1.270
# (== GANTRY_X_OFFSET) while the rail base is at x=-2.000. Using the
# carriage double-counts gantry travel, offsetting every planned pose
# by the current gantry position.
ROBOT_ROOT_PATH = "/World/gantry"

# USD prim for CUMOTION_TOOL_FRAME. Used only by /api/verify_planner to
# compare cuMotion's idea of the tool pose against the scene's.
TOOL_FRAME_PRIM = ("/World/gantry/gantry_home/ur10_flattened/"
                   "robotiq_fixed_physics/Robotiq_2F_140_physics_edit/"
                   "robotiq_base_link")

# Per-joint limits for time-parameterising a planned path.
# Order follows the XRDF <cspace>: gantry, pan, lift, elbow, w1, w2, w3.
CUMOTION_MAX_VEL = np.array([0.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0])
CUMOTION_MAX_ACC = np.array([1.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5])

TRAJECTORY_DT = 1.0 / 60.0        # sampling step when streaming


def _reset_cumotion_world():
    """Drop and recreate the collision world (no per-obstacle removal API)."""
    from isaacsim.robot_motion.cumotion import CumotionWorldInterface
    import warp as wp

    base_pos, base_quat = _robot_base_pose_arrays()
    STATE.cumotion_world = CumotionWorldInterface(
        world_to_robot_base=(base_pos, base_quat),
        device=STATE.cumotion_device)
    STATE.collision_sphere_count = 0
    # The planner holds a reference to the world interface, so rebuild it.
    if STATE.cumotion_robot is not None:
        from isaacsim.robot_motion.cumotion import GraphBasedMotionPlanner
        STATE.cumotion_planner = GraphBasedMotionPlanner(
            cumotion_robot=STATE.cumotion_robot,
            cumotion_world_interface=STATE.cumotion_world,
            tool_frame=CUMOTION_TOOL_FRAME)


def _robot_base_pose_arrays():
    """World→robot-root transform as warp arrays.

    This must be the FIXED rail mount, not ``base_link``. The XRDF cspace
    lists ``gantry_vagn_joint`` as joint 0, so cuMotion treats the gantry
    as part of the kinematic chain and the URDF root sits ABOVE it —
    stationary in the world. Passing the sliding ``base_link`` here would
    double-count the gantry travel: once in the joint value, once in the
    base transform.
    """
    import omni.usd
    import warp as wp

    stage = omni.usd.get_context().get_stage()
    mat = omni.usd.get_world_transform_matrix(
        stage.GetPrimAtPath(ROBOT_ROOT_PATH))
    t = mat.ExtractTranslation()
    q = mat.ExtractRotationQuat()
    imag = q.GetImaginary()
    with wp.ScopedDevice(STATE.cumotion_device):
        pos = wp.array(
            np.array([[t[0], t[1], t[2]]], dtype=np.float32), dtype=wp.vec3)
        quat = wp.array(
            np.array([[q.GetReal(), imag[0], imag[1], imag[2]]],
                     dtype=np.float32), dtype=wp.vec4)
    return pos, quat


# Purpose-made TCP Xform authored in the USD — the finger contact point.
# Preferred over inferring from mesh prims, whose transforms are all
# coincident with robotiq_base_link (the geometry offset is baked into
# the meshes, not the prim hierarchy).
GRIPPER_TCP_PRIM = ("/World/gantry/gantry_home/ur10_flattened/"
                    "robotiq_fixed_physics/Robotiq_2F_140_physics_edit/"
                    "gripper_tcp")
FINGER_PAD_PRIM = ("/World/gantry/gantry_home/ur10_flattened/"
                   "robotiq_fixed_physics/Robotiq_2F_140_physics_edit/"
                   "left_inner_finger/Fingertip_01/Fingertip")


def get_tcp_pos():
    """World position of the finger contact pad — the actual TCP.

    Motion targets are TCP points. ``ee_link`` is NOT the TCP: the pad
    hangs roughly a gripper-length below it. Deriving a "TCP" target from
    ``get_world_pos(EE_PATH)`` and then adding the TCP offset counts that
    gap twice, which on this ceiling-mounted arm pushes the goal out of
    reach. Falls back to ee_link only if the pad prim is missing.
    """
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    for path in (GRIPPER_TCP_PRIM, FINGER_PAD_PRIM):
        if stage.GetPrimAtPath(path).IsValid():
            return get_world_pos(path)
    return get_world_pos(EE_PATH)


def _measure_tool_frames():
    """Measure the tool frame from USD and correct the Lula-era constants.

    ``GRIPPER_TCP_OFFSET`` and ``DOWNWARD_ORIENTATION`` were both derived
    for ``ee_link`` when Lula planned in that frame. cuMotion plans for
    the XRDF tool frame (``robotiq_base_link``), so both are re-measured
    here and cached on STATE.
    """
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    def pose(path):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None, None
        m = omni.usd.get_world_transform_matrix(prim)
        t = m.ExtractTranslation()
        q = m.ExtractRotationQuat()
        im = q.GetImaginary()
        return (np.array([t[0], t[1], t[2]]),
                np.array([q.GetReal(), im[0], im[1], im[2]]))

    ee_p, ee_q = pose(EE_PATH)
    tool_p, tool_q = pose(TOOL_FRAME_PRIM)
    tcp_p, tcp_q = pose(GRIPPER_TCP_PRIM)
    pad_p, _ = pose(FINGER_PAD_PRIM)

    print("[frames] measuring tool geometry from USD:")
    for name, pp in (("ee_link", ee_p), (CUMOTION_TOOL_FRAME, tool_p),
                     ("gripper_tcp", tcp_p), ("finger pad", pad_p)):
        print(f"[frames]   {name:22s} "
              f"{'NOT FOUND' if pp is None else f'({pp[0]:+.4f}, {pp[1]:+.4f}, {pp[2]:+.4f})'}")

    if ee_q is not None and tool_q is not None:
        d = abs(float(np.dot(ee_q, tool_q)))
        print(f"[frames]   |dot(q_ee, q_tool)| = {d:.4f} -> "
              f"{'aligned' if d > 0.999 else 'ROTATED: the ee_link-derived DOWNWARD_ORIENTATION is invalid for this tool frame'}")

    src_p = tcp_p if tcp_p is not None else pad_p
    src_name = "gripper_tcp" if tcp_p is not None else "finger pad"
    if tool_p is not None and src_p is not None:
        # Use the full 3D distance, not dz. The offset is a fixed length
        # along the gripper's approach axis; dz only equals it when the
        # tool happens to point straight down, which it need not at boot.
        dist = float(np.linalg.norm(src_p - tool_p))
        dz = float(abs(src_p[2] - tool_p[2]))
        if dist < MIN_CREDIBLE_TCP_OFFSET:
            STATE.tcp_offset = None
            print(f"[frames]   tool->{src_name} = {dist:.5f} m — NOT CREDIBLE "
                  f"(prim transforms coincide; the offset is baked into the "
                  f"meshes). Keeping GRIPPER_TCP_OFFSET = "
                  f"{GRIPPER_TCP_OFFSET:.5f}")
        else:
            STATE.tcp_offset = dist
            print(f"[frames]   tool->{src_name} = {dist:.5f} m along the tool "
                  f"axis (dz {dz:.5f}); configured was "
                  f"{GRIPPER_TCP_OFFSET:.5f}")
            if abs(dist - dz) > 0.01:
                print(f"[frames]   note: dist != dz, so the gripper is not "
                      f"pointing straight down right now — expected, and the "
                      f"axial distance is the correct value to use")

    # Derive "down" from the scene geometry, not from the ee_link constant.
    if tool_p is not None and tool_q is not None and src_p is not None:
        down_q, approach = derive_down_quat(tool_p, tool_q, src_p)
        if down_q is not None:
            STATE.tool_down_quat = down_q
            facing = ("UP" if approach[2] > 0.7 else
                      "DOWN" if approach[2] < -0.7 else "SIDEWAYS")
            print(f"[frames]   gripper approach axis now = "
                  f"({approach[0]:+.3f}, {approach[1]:+.3f}, "
                  f"{approach[2]:+.3f})  -> pointing {facing}")
            print(f"[frames]   DOWNWARD for tool (derived) = "
                  f"({down_q[0]:+.4f}, {down_q[1]:+.4f}, "
                  f"{down_q[2]:+.4f}, {down_q[3]:+.4f})")
            print(f"[frames]   (stale ee_link constant was "
                  f"{tuple(np.round(normalize_quat(DOWNWARD_ORIENTATION), 4))})")


def _planner_unavailable_reason():
    """Actionable message for callers when no motion can be planned."""
    return (f"cuMotion planner unavailable: {STATE.cumotion_error}. "
            f"Check the Isaac Sim console for [cuMotion] lines at startup.")


def _init_cumotion():
    """Load the cuMotion robot, collision world and planner.

    Returns True on success. Raises nothing — the bridge reports the
    failure through /api/status so a misconfigured path is visible.
    """
    import warp as wp
    from isaacsim.robot_motion.cumotion import load_cumotion_robot

    if not ROBOT_CONFIG_PATH:
        STATE.cumotion_error = (
            "ROBOT_CONFIG_PATH is not set. The bridge runs in Isaac Sim's "
            "process, so it needs .env next to the project root (see the "
            "[env] line above) or a real environment variable.")
        print(f"[cuMotion] {STATE.cumotion_error}")
        return False
    if not os.path.isdir(ROBOT_CONFIG_PATH):
        STATE.cumotion_error = (
            f"ROBOT_CONFIG_PATH does not exist: {ROBOT_CONFIG_PATH}")
        print(f"[cuMotion] {STATE.cumotion_error}")
        return False

    try:
        STATE.cumotion_device = wp.get_device("cuda:0")
    except Exception as e:
        STATE.cumotion_error = f"CUDA device unavailable: {e}"
        print(f"[cuMotion] {STATE.cumotion_error}")
        return False

    try:
        STATE.cumotion_robot = load_cumotion_robot(
            directory=ROBOT_CONFIG_PATH,
            urdf_filename=ROBOT_URDF_FILENAME,
            xrdf_filename=ROBOT_XRDF_FILENAME)
    except Exception as e:
        STATE.cumotion_error = f"robot load failed: {e}"
        print(f"[cuMotion] {STATE.cumotion_error}")
        return False

    joints = list(STATE.cumotion_robot.controlled_joint_names)
    print(f"[cuMotion] loaded robot from {ROBOT_CONFIG_PATH}")
    print(f"[cuMotion] planned joints ({len(joints)}): {joints}")

    _reset_cumotion_world()
    STATE.cumotion_ready = STATE.cumotion_planner is not None

    # Map XRDF cspace joints → articulation DOF indices, once.
    STATE.cumotion_dof_indices = []
    for name in joints:
        try:
            STATE.cumotion_dof_indices.append(STATE.dof_names.index(name))
        except (ValueError, AttributeError):
            print(f"[cuMotion] WARNING: joint '{name}' not in articulation DOFs")
            STATE.cumotion_dof_indices = []
            break

    if STATE.cumotion_ready:
        STATE.cumotion_error = None
    elif STATE.cumotion_error == "not initialised yet":
        STATE.cumotion_error = "planner construction returned None"
    try:
        _measure_tool_frames()
    except Exception as e:
        print(f"[frames] measurement failed: {e}")

    print(f"[cuMotion] ready={STATE.cumotion_ready} "
          f"dof_indices={STATE.cumotion_dof_indices}")
    return STATE.cumotion_ready


def _current_cspace():
    """Current joint values in XRDF cspace order."""
    q = STATE.robot.get_joint_positions()
    if STATE.cumotion_dof_indices:
        return np.array([float(q[i]) for i in STATE.cumotion_dof_indices],
                        dtype=np.float64)
    return np.array(q[:7], dtype=np.float64)


def _tool_pose_from_tcp(tcp_pos, orientation):
    """Convert a TCP target to the tool-frame pose cuMotion plans for.

    The pipeline's targets are finger-contact (TCP) points, but the XRDF
    tool frame is ``robotiq_base_link``, which sits GRIPPER_TCP_OFFSET
    back along the tool axis. For the top-down grasps this pipeline uses,
    the tool axis is world -Z, so the tool frame is that far ABOVE the
    TCP. Planning to the raw TCP would drive the gripper body into the
    part by the offset.
    """
    pos = np.array(tcp_pos, dtype=np.float64).copy()
    pos[2] += getattr(STATE, "tcp_offset", None) or GRIPPER_TCP_OFFSET
    ori = np.array(orientation, dtype=np.float64)
    # The hardcoded downward quaternion describes ee_link, not this tool
    # frame. When the measured tool orientation is available prefer it —
    # it is achievable by construction.
    # Only substitute when the CALLER asked for the ee_link "down"; a
    # deliberate custom orientation is passed through untouched.
    down = getattr(STATE, "tool_down_quat", None)
    if down is not None and USE_MEASURED_TOOL_ORIENTATION:
        if np.allclose(ori, normalize_quat(DOWNWARD_ORIENTATION), atol=1e-6):
            ori = np.array(down, dtype=np.float64)
    return pos, ori


def _plan_pose(tcp_pos, orientation):
    """Plan a collision-free path to a TCP pose. Returns a trajectory or None.

    Logs both frames. The caller supplies a TCP (finger-contact) target;
    cuMotion plans for the tool frame, which sits GRIPPER_TCP_OFFSET
    above it for a top-down grasp. Printing only the tool pose made the
    numbers look untraceable — a part detected at z=0.80 became a plan
    for z=0.99 with nothing connecting the two.
    """
    tool_pos, quat = _tool_pose_from_tcp(tcp_pos, orientation)
    q0 = _current_cspace()
    tcp_now = get_tcp_pos()
    offset = getattr(STATE, "tcp_offset", None) or GRIPPER_TCP_OFFSET

    print(f"  [plan] TCP target  ({tcp_pos[0]:+.3f}, {tcp_pos[1]:+.3f}, "
          f"{tcp_pos[2]:+.3f})   from TCP now "
          f"({tcp_now[0]:+.3f}, {tcp_now[1]:+.3f}, {tcp_now[2]:+.3f})")
    print(f"  [plan] tool target ({tool_pos[0]:+.3f}, {tool_pos[1]:+.3f}, "
          f"{tool_pos[2]:+.3f})   (+{offset:.3f} tool offset)  "
          f"quat ({quat[0]:+.3f}, {quat[1]:+.3f}, {quat[2]:+.3f}, {quat[3]:+.3f})")

    path = STATE.cumotion_planner.plan_to_pose_target(
        q_initial=q0, position=tool_pos, orientation=quat)
    if path is None:
        print(f"  [plan] NO PATH — target unreachable, in collision, or "
              f"start in collision ({STATE.collision_sphere_count} spheres)")
        return None
    print("  [plan] path found")
    return _path_to_trajectory(path)


def _plan_cspace(q_target):
    """Plan a collision-free path to a joint configuration."""
    q0 = _current_cspace()
    path = STATE.cumotion_planner.plan_to_cspace_target(
        q_initial=q0, q_target=np.array(q_target, dtype=np.float64))
    if path is None:
        print(f"  [plan] NO PATH to cspace target")
        return None
    return _path_to_trajectory(path)


def _path_to_trajectory(path):
    """Time-parameterise a planned path against the joint limits."""
    n = len(STATE.cumotion_robot.controlled_joint_names)
    return path.to_minimal_time_joint_trajectory(
        max_velocities=CUMOTION_MAX_VEL[:n],
        max_accelerations=CUMOTION_MAX_ACC[:n],
        robot_joint_space=STATE.dof_names,
        active_joints=STATE.cumotion_robot.controlled_joint_names)


async def _execute_trajectory(traj, finger_value=None,
                              settle_frames=SETTLE_FRAMES):
    """Stream a cuMotion trajectory to the articulation.

    Samples at TRAJECTORY_DT and issues one apply_action per sample, so
    the arm follows the planned (collision-free) path rather than cutting
    the corner a straight joint interpolation would take.
    """
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction

    if traj is None:
        return False

    duration = float(traj.duration)
    q_before = _current_cspace()
    goal_state = traj.get_target_state(duration)
    goal_str = "?"
    if goal_state is not None and goal_state.joints.positions is not None:
        gp = np.asarray(goal_state.joints.positions).flatten()
        goal_str = ", ".join(f"{v:+.3f}" for v in gp[:7])
    print(f"  [traj] executing {duration:.2f}s trajectory")
    print(f"  [traj]   q_start = [{', '.join(f'{v:+.3f}' for v in q_before)}]")
    print(f"  [traj]   q_goal  = [{goal_str}]")

    t = 0.0
    last = None
    while t < duration:
        state = traj.get_target_state(t)
        if state is not None and state.joints.positions is not None:
            targets = STATE.robot.get_joint_positions().copy()
            positions = np.asarray(state.joints.positions).flatten()
            indices = state.joints.position_indices
            if indices is not None:
                for slot, dof_idx in enumerate(np.asarray(indices).flatten()):
                    targets[int(dof_idx)] = positions[slot]
            else:
                targets[:len(positions)] = positions
            if finger_value is not None:
                targets = set_finger_joints(
                    STATE.robot, STATE.dof_names, finger_value, targets)
            STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
            last = targets
        await omni.kit.app.get_app().next_update_async()
        t += TRAJECTORY_DT

    if last is not None:
        STATE.robot.apply_action(ArticulationAction(joint_positions=last))
    for _ in range(settle_frames):
        await omni.kit.app.get_app().next_update_async()

    # Did the articulation actually follow? If q_end == q_start the plan
    # is fine but the joints are not tracking — physics paused, drives
    # disabled, or apply_action not reaching this articulation.
    q_after = _current_cspace()
    moved = float(np.abs(q_after - q_before).max())
    print(f"  [traj]   q_end   = [{', '.join(f'{v:+.3f}' for v in q_after)}]")
    print(f"  [traj]   max joint change = {moved:.4f}")
    if moved < 1e-3:
        print("  [traj]   *** ROBOT DID NOT MOVE — trajectory streamed but the "
              "joints did not follow. Check the timeline is PLAYING and the "
              "arm drives have non-zero stiffness. ***")
    return True


def ik_solve(pos, ori, warm=None):
    """Plan a collision-free path to a TCP pose with cuMotion.

    Keeps the name and ``(action, ok)`` return of the Lula helper it
    replaces so the ~25 call sites read the same. ``warm`` is accepted
    and ignored — a sampling-based planner has no warm start.

    Returns ``(action, ok)`` where ``action`` is the goal joint
    configuration in XRDF cspace order, matching what Lula used to
    return so ``apply_arm_joints`` still works.

    Side effect: the planned trajectory is stashed on
    ``STATE.pending_trajectory`` for the next ``_apply_interpolated`` to
    follow. Without that the caller would jump straight to the goal and
    throw the collision-free path away.
    """
    if not STATE.cumotion_ready:
        print("  [plan] cuMotion not ready")
        return None, False

    traj = _plan_pose(pos, ori)
    if traj is None:
        STATE.pending_trajectory = None
        return None, False

    state = traj.get_target_state(float(traj.duration))
    if state is None or state.joints.positions is None:
        STATE.pending_trajectory = None
        return None, False

    STATE.pending_trajectory = traj
    goal = np.asarray(state.joints.positions).flatten()
    indices = state.joints.position_indices
    if indices is not None and STATE.cumotion_dof_indices:
        by_dof = {int(d): goal[s]
                  for s, d in enumerate(np.asarray(indices).flatten())}
        goal = np.array([by_dof.get(d, 0.0)
                         for d in STATE.cumotion_dof_indices])
    # NO shoulder-pan clamp here. That guard existed because Lula had no
    # collision awareness of the gantry rail, and it clamped index 0 —
    # which was shoulder_pan in Lula's 6-joint result but is
    # gantry_vagn_joint in cuMotion's 7-joint cspace. Clamping it
    # silently truncated gantry travel. cuMotion enforces the URDF joint
    # limits itself, and rail avoidance is the collision world's job.
    return goal, True



def _check_self_collision_risk(current_joints, target_joints):
    """Check if a direct move risks self-collision (large joint angle jumps)."""
    if current_joints is None or target_joints is None:
        return True
    n = min(7, len(current_joints), len(target_joints))
    arm_current = np.array(current_joints[:n])
    arm_target = np.array(target_joints[:n])
    max_delta = np.max(np.abs(arm_target - arm_current))
    return max_delta > INTERP_THRESHOLD


def _interpolate_joints(start, end, max_step=INTERP_MAX_STEP):
    """Generate interpolated joint waypoints between start and end configurations."""
    start = np.array(start, dtype=np.float64)
    end = np.array(end, dtype=np.float64)
    diff = end - start
    max_delta = np.max(np.abs(diff))
    if max_delta <= max_step:
        return [end]
    n_steps = int(np.ceil(max_delta / max_step))
    waypoints = []
    for i in range(1, n_steps + 1):
        frac = i / n_steps
        waypoints.append(start + frac * diff)
    return waypoints


# ── URDF / YAML Patching ────────────────────────────────────

# ═════════════════════════════════════════════════════════════
# SHARED STATE
# ═════════════════════════════════════════════════════════════

class BridgeState:
    def __init__(self):
        self.robot = None
        self.world = None
        self.cameras = {}   # alias -> Camera
        self.frames = {}    # alias -> {"depth": ndarray, "resolution": (w, h)}
        # ── cuMotion planning ──
        self.cumotion_robot = None        # CumotionRobot (URDF + XRDF)
        self.cumotion_world = None        # CumotionWorldInterface
        self.cumotion_planner = None      # GraphBasedMotionPlanner
        self.cumotion_device = None       # warp device (needs CUDA)
        self.cumotion_dof_indices = []    # XRDF cspace order -> articulation DOF
        self.cumotion_ready = False
        self.cumotion_error = "not initialised yet"
        self.tcp_offset = None            # measured tool -> finger pad
        self.tool_quat = None             # measured tool orientation
        self.tool_down_quat = None        # ee_link "down" in the tool frame
        self.collision_sphere_count = 0
        # Single-use handoff: set by ik_solve, consumed by the next
        # _apply_interpolated so the planned path is actually followed.
        self.pending_trajectory = None
        self.target_frame = "ee_link"
        self.arm_names = []
        self.dof_names = []
        self.num_dof = 15
        self.is_ready = False
        self.ik_ready = False
        self.is_executing = False
        self.contact_sensor = None        # Inner pad — for grip close
        self.contact_sensor_tip = None    # Fingertip bottom — for descent stop
        self.last_error = None
        self.execution_log = []
        self._command_queue = []
        self._result_queue = []
        self._lock = threading.Lock()

    def push_command(self, cmd):
        with self._lock:
            self._command_queue.append(cmd)

    def pop_command(self):
        with self._lock:
            return self._command_queue.pop(0) if self._command_queue else None

    def push_result(self, result):
        with self._lock:
            self._result_queue.append(result)

    def pop_result(self, timeout=60):
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if self._result_queue:
                    return self._result_queue.pop(0)
            time.sleep(0.05)
        return {"error": "Timeout waiting for Isaac Sim response"}


STATE = BridgeState()


# ═════════════════════════════════════════════════════════════
# HTTP REQUEST HANDLER
# ═════════════════════════════════════════════════════════════

class BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default logging

    def _send_json(self, data, status=200):
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass  # client disconnected before response was fully sent

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length > 0 else {}

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/status":
            self._handle_status()
        elif path == "/api/camera":
            cam_type = "rgb"
            if "?" in self.path:
                params = dict(p.split("=") for p in self.path.split("?")[1].split("&") if "=" in p)
                cam_type = params.get("type", "rgb")
            self._handle_camera(cam_type)
        elif path == "/api/ping":
            self._send_json({
                "status": "ok", "bridge": "isaac_sim", "port": BRIDGE_PORT,
                "cameras": ["rgb", "depth", "wrist", "kit"], "ik_ready": STATE.ik_ready,
            })
        elif path == "/api/scene_parts":
            self._handle_scene_parts()
        elif path == "/api/scene_annotations":
            cam_type = "rgb"
            if "?" in self.path:
                params = dict(p.split("=") for p in self.path.split("?")[1].split("&") if "=" in p)
                cam_type = params.get("camera", "rgb")
            self._handle_scene_annotations(cam_type)
        elif path == "/api/prim_center":
            prim_path = ""
            if "?" in self.path:
                params = dict(p.split("=", 1) for p in self.path.split("?", 1)[1].split("&") if "=" in p)
                prim_path = params.get("prim", "")
                # URL-decode (prim paths contain '/')
                import urllib.parse as _u
                prim_path = _u.unquote(prim_path)
            self._handle_prim_center(prim_path)
        else:
            self._send_json({"error": f"Unknown endpoint: {path}"}, 404)

    def do_POST(self):
        body = self._read_body()
        path = self.path.split("?")[0]
        if path == "/api/execute":
            self._handle_execute(body)
        elif path == "/api/joints":
            self._handle_set_joints(body)
        elif path == "/api/home":
            self._handle_home()
        elif path == "/api/gripper":
            self._handle_gripper(body)
        elif path == "/api/approach":
            self._handle_approach(body)
        elif path == "/api/plan_test":
            STATE.push_command({"action": "plan_test", "params": body})
            self._send_json(STATE.pop_result(timeout=180))
        elif path == "/api/jog":
            STATE.push_command({"action": "jog", "params": body})
            self._send_json(STATE.pop_result(timeout=60))
        elif path == "/api/verify_planner":
            STATE.push_command({"action": "verify_planner", "params": body})
            self._send_json(STATE.pop_result(timeout=120))
        elif path == "/api/build_collision_world":
            STATE.push_command({"action": "build_collision_world",
                                "params": body})
            self._send_json(STATE.pop_result(timeout=180))
        elif path == "/api/wrist_obstacles":
            STATE.push_command({"action": "wrist_obstacles", "params": body})
            self._send_json(STATE.pop_result(timeout=60))
        elif path == "/api/realign_gantry":
            self._handle_realign_gantry(body)
        elif path == "/api/pick":
            self._handle_pick(body)
        elif path == "/api/pick_descend":
            self._handle_pick_descend(body)
        elif path == "/api/pick_close":
            self._handle_pick_close(body)
        elif path == "/api/pick_retract":
            self._handle_pick_retract(body)
        elif path == "/api/place":
            self._handle_place(body)
        elif path == "/api/project_to_world":
            self._handle_project(body)
        else:
            self._send_json({"error": f"Unknown endpoint: {path}"}, 404)

    # ── GET handlers ─────────────────────────────────────────

    def _handle_status(self):
        if not STATE.is_ready:
            self._send_json({"status": "not_ready"})
            return
        try:
            joints = STATE.robot.get_joint_positions()
            # Sensor health — the operator uses these flags to verify
            # that a fresh bridge build is actually loaded after an
            # ``exec(open(...))`` reload in the Script Editor. If
            # ``contact_sensor_tip_ready`` is False the descent runs
            # blind and will jam the gripper into the bin floor.
            self._send_json({
                "status": "ready" if not STATE.is_executing else "executing",
                "num_dof": STATE.num_dof,
                "joint_names": list(STATE.dof_names),
                "joint_positions": joints.tolist() if joints is not None else [],
                "is_executing": STATE.is_executing,
                "ik_ready": STATE.ik_ready,
                "planner": "cumotion",
                "cumotion_error": STATE.cumotion_error,
                "collision_spheres": STATE.collision_sphere_count,
                "contact_sensor_pad_ready":
                    STATE.contact_sensor is not None,
                "contact_sensor_tip_ready":
                    STATE.contact_sensor_tip is not None,
                "cameras": ["rgb", "depth", "wrist", "kit"],
                "last_error": STATE.last_error,
            })
        except Exception as e:
            self._send_json({"status": "error", "error": str(e)})

    def _handle_camera(self, cam_type="rgb"):
        if not STATE.is_ready:
            self._send_json({"error": "Not ready"}, 503)
            return
        STATE.push_command({"action": "camera_capture", "camera": cam_type})
        result = STATE.pop_result(timeout=10)
        self._send_json(result)

    def _handle_scene_parts(self):
        """Scan the USD stage for pickable parts with real-world coordinates."""
        if not STATE.is_ready:
            self._send_json({"error": "Not ready"}, 503)
            return
        STATE.push_command({"action": "scan_scene_parts"})
        result = STATE.pop_result(timeout=15)
        self._send_json(result)

    def _handle_scene_annotations(self, cam_type="rgb"):
        """GT per-part annotations as seen from the requested camera."""
        if not STATE.is_ready:
            self._send_json({"error": "Not ready"}, 503)
            return
        STATE.push_command({"action": "scene_annotations", "camera": cam_type})
        result = STATE.pop_result(timeout=20)
        self._send_json(result)

    def _handle_prim_center(self, prim_path: str):
        """USD bbox query: returns ``{center_xy, top_z, bot_z, height}``
        for the requested prim. Used by the wrist-scan workflow to
        position the robot above the bin and to compute tray placement
        without relying on overhead VLM landmark detection.
        """
        if not STATE.is_ready:
            self._send_json({"error": "Not ready"}, 503)
            return
        if not prim_path:
            self._send_json({"error": "Missing 'prim' query parameter"}, 400)
            return
        STATE.push_command({"action": "compute_prim_center",
                            "prim_path": prim_path})
        result = STATE.pop_result(timeout=10)
        self._send_json(result)

    # ── POST handlers ────────────────────────────────────────

    def _handle_execute(self, body):
        plan = body.get("plan", [])
        if not plan:
            self._send_json({"error": "No plan provided"}, 400)
            return
        if STATE.is_executing:
            self._send_json({"error": "Already executing"}, 409)
            return
        STATE.push_command({"action": "execute_plan", "plan": plan})
        result = STATE.pop_result(timeout=300)
        self._send_json(result)

    def _handle_set_joints(self, body):
        positions = body.get("positions", [])
        if not positions:
            self._send_json({"error": "No positions"}, 400)
            return
        STATE.push_command({"action": "set_joints", "positions": positions})
        self._send_json(STATE.pop_result(timeout=15))

    def _handle_home(self):
        STATE.push_command({"action": "move_home"})
        self._send_json(STATE.pop_result(timeout=15))

    def _handle_gripper(self, body):
        action = body.get("action", "open")
        STATE.push_command({"action": f"gripper_{action}"})
        self._send_json(STATE.pop_result(timeout=10))

    def _handle_approach(self, body):
        """Move near a target XYZ using a cuMotion-planned trajectory."""
        STATE.push_command({"action": "approach", "params": body})
        self._send_json(STATE.pop_result(timeout=30))

    def _handle_realign_gantry(self, body):
        """Slide gantry to target X while holding ee_link in place.

        Body: ``{"x": float}`` — world X to slide the gantry to. The
        arm joints micro-step their IK each frame so the wrist's
        world (X, Y, Z) stays constant during the slide. Used after
        depth verification refines the part XY: the gantry shifts
        underneath the EE, but the wrist itself doesn't drift, so
        the subsequent Cartesian descent enters the compartment from
        the same vantage we just confirmed with the depth camera.
        """
        STATE.push_command({"action": "realign_gantry_hold_ee",
                            "params": body})
        self._send_json(STATE.pop_result(timeout=30))

    def _handle_pick(self, body):
        """Full pick sequence at XYZ."""
        STATE.push_command({"action": "pick_object_ik", "params": body})
        self._send_json(STATE.pop_result(timeout=60))

    def _handle_pick_descend(self, body):
        """Descend to grasp with open fingers, return wrist image."""
        STATE.push_command({"action": "pick_descend", "params": body})
        self._send_json(STATE.pop_result(timeout=60))

    def _handle_pick_close(self, body):
        """Close gripper and confirm grasp, return wrist image."""
        STATE.push_command({"action": "pick_close", "params": body})
        self._send_json(STATE.pop_result(timeout=60))

    def _handle_pick_retract(self, body):
        """Retract with grasped part."""
        STATE.push_command({"action": "pick_retract", "params": body})
        self._send_json(STATE.pop_result(timeout=60))

    def _handle_place(self, body):
        """Full place sequence at XYZ."""
        STATE.push_command({"action": "place_object_ik", "params": body})
        self._send_json(STATE.pop_result(timeout=60))

    def _handle_project(self, body):
        """Project normalised image coords to world XYZ via depth."""
        STATE.push_command({"action": "project_to_world", "params": body})
        self._send_json(STATE.pop_result(timeout=15))


# (Async command processor is now _process_commands_loop in the lifecycle section)


# ═════════════════════════════════════════════════════════════
# CAMERA CAPTURE  (Isaac Sim 6.0.1 RTX sensor API)
#
#   RGB and depth are read from the SAME rendered frame with no app
#   tick between them, then the depth buffer is cached per alias so
#   /api/project_to_world projects geometry belonging to the exact
#   image the VLM was shown.
#
#   CameraSensor.__init__ calls enforce_square_pixels(), which syncs
#   verticalAperture to the resolution aspect ratio without warning.
# ═════════════════════════════════════════════════════════════

# NOTE: the RTX API uses OpenCV/NumPy resolution order — (height, width).
CAMERA_ALIASES = {
    "rgb":   (CAMERA_RGB_PRIM,   (1080, 1920)),
    "depth": (CAMERA_DEPTH_PRIM, (720, 1280)),
    "wrist": (CAMERA_WRIST_PRIM, (720, 1280)),
    "kit":   (CAMERA_KIT_PRIM,   (720, 1280)),
}
CAMERA_ANNOTATORS = ["rgb", "distance_to_image_plane"]
# The D555 depth prim is a depth-only sensor: attaching an rgb annotator
# to it yields malformed buffers rather than a black image.
CAMERA_ANNOTATORS_BY_ALIAS = {"depth": ["distance_to_image_plane"]}

# A newly created render product needs many app updates before its
# annotators return anything; poll rather than guess a tick count.
_WARMUP_MAX_TICKS = 400


def _has_annotator(sensor, name):
    """True if this sensor was configured with the named annotator."""
    try:
        return name in getattr(sensor, "_annotators", {})
    except Exception:
        return False


def _read_annotator(sensor, name):
    """Fetch one annotator as a host numpy array, or None if not ready.

    ``get_data`` returns warp arrays on the GPU; ``.numpy()`` pulls them
    to host. Depth arrives as (h, w, 1) and is squeezed to (h, w).
    """
    data, _ = sensor.get_data(name)
    if data is None:
        return None
    arr = data.numpy()
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


async def _get_camera(alias):
    """Return (sensor, prim_path) with rgb + depth annotators attached."""
    from isaacsim.sensors.experimental.rtx import CameraSensor
    import omni.kit.app

    stage = omni.usd.get_context().get_stage()

    prim_path, resolution = CAMERA_ALIASES.get(alias, (alias, (720, 1280)))

    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid() and not prim.HasAPI("OmniSensorAPI"):
        prim.ApplyAPI("OmniSensorAPI")

    sensor = STATE.cameras.get(alias)
    if sensor is None:
        annots = CAMERA_ANNOTATORS_BY_ALIAS.get(alias, CAMERA_ANNOTATORS)
        sensor = CameraSensor(prim_path, resolution=resolution,
                              annotators=annots)
        STATE.cameras[alias] = sensor

        app = omni.kit.app.get_app()
        for tick in range(1, _WARMUP_MAX_TICKS + 1):
            await app.next_update_async()
            if tick % 20 == 0 \
                    and _read_annotator(sensor, "rgb") is not None \
                    and _read_annotator(sensor, "distance_to_image_plane") is not None:
                break
        h_res, w_res = sensor.resolution
        print(f"  [{alias}] camera ready after {tick} ticks "
              f"({prim_path} @ {w_res}x{h_res})")

    return sensor, prim_path


async def _render_frame(sensor, alias):
    """Tick until the RGB frame is non-black, then snapshot rgb + depth.

    The wrist RGB renderer lags the depth annotator after the robot
    moves, so it gets more retries.
    """
    import omni.kit.app
    app = omni.kit.app.get_app()

    max_attempts, ticks = (8, 8) if alias == "wrist" else (3, 3)
    rgb, mean = None, -1.0

    for _ in range(max_attempts):
        for _ in range(ticks):
            await app.next_update_async()

        rgb = _read_annotator(sensor, "rgb") if _has_annotator(sensor, "rgb") else None
        if rgb is not None and rgb.ndim == 3 and rgb.size > 0:
            mean = float(np.mean(rgb[:, :, :3]))
            if mean > 3:
                break

    # Same frame, no tick in between — this pairing is the whole point.
    depth = _read_annotator(sensor, "distance_to_image_plane")

    # The D555 depth prim is a depth-only sensor: it has no colour output,
    # so an absent/black RGB frame there is normal, not a failure.
    if rgb is None or rgb.ndim != 3:
        rgb = None
    elif mean <= 3:
        print(f"  [{alias}] WARNING: frame still dark (mean={mean:.1f})")

    return rgb, depth


async def _capture_camera(cam_type="rgb"):
    try:
        sensor, _ = await _get_camera(cam_type)
        rgb, depth = await _render_frame(sensor, cam_type)

        STATE.frames[cam_type] = {
            "depth": depth,
            "resolution": tuple(sensor.resolution),
        }

        from PIL import Image
        if rgb is not None and rgb.size:
            img = Image.fromarray(rgb[:, :, :3])
        elif depth is not None and depth.size:
            # Depth-only sensor: return depth as a viewable greyscale
            # image, normalised over its valid range.
            valid = (depth > DEPTH_MIN_M) & (depth < DEPTH_MAX_M)
            if not valid.any():
                return {"error": f"{cam_type} depth frame has no valid pixels"}
            lo = float(depth[valid].min())
            hi = float(depth[valid].max())
            span = max(hi - lo, 1e-6)
            grey = np.zeros(depth.shape, dtype=np.uint8)
            grey[valid] = (255.0 * (depth[valid] - lo) / span).astype(np.uint8)
            img = Image.fromarray(grey).convert("RGB")
        else:
            return {"error": f"{cam_type} camera returned no rgb and no depth"}
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return {
            "image_base64": base64.b64encode(buf.getvalue()).decode(),
            "width": img.width, "height": img.height,
            "format": "jpeg", "camera": cam_type,
            "depth_available": depth is not None,
        }

    except Exception as e:
        import traceback
        print(f"  [{cam_type}] capture failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"error": f"{cam_type} camera failed: {type(e).__name__}: {e}"}


# ═════════════════════════════════════════════════════════════
# CAMERA MODEL
#
#   The RTX sensor API ships no projection helpers, so the pinhole
#   model lives here. It mirrors what the deprecated
#   isaacsim.sensors.camera.Camera did:
#
#     fx = w * focal / horizontal_aperture,  cx = w / 2
#     view_ros = R_U_TRANSFORM @ inverse(camera_to_world)
#
#   R_U_TRANSFORM flips Y and Z to convert the USD camera convention
#   (+Y up, -Z forward) into the ROS one (+Y down, +Z forward), which
#   is what the intrinsics matrix expects. Because that flip lives in
#   the matrix, no per-camera axis-sign overrides are needed.
# ═════════════════════════════════════════════════════════════

# USD camera frame -> ROS camera frame
R_U_TRANSFORM = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])


def _intrinsics_matrix(prim_path, resolution):
    """Pinhole K read from the USD camera prim.

    Focal length and apertures share the same tenths-of-a-unit scaling,
    so it cancels in the ratio — no unit conversion needed.
    """
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    usd_cam = UsdGeom.Camera(stage.GetPrimAtPath(prim_path))
    if not usd_cam:
        raise RuntimeError(f"not a UsdGeom.Camera: {prim_path}")

    focal = float(usd_cam.GetFocalLengthAttr().Get() or 0.0)
    h_ap = float(usd_cam.GetHorizontalApertureAttr().Get() or 0.0)
    v_ap = float(usd_cam.GetVerticalApertureAttr().Get() or 0.0)
    if focal <= 0 or h_ap <= 0 or v_ap <= 0:
        raise RuntimeError(
            f"{prim_path} has invalid optics "
            f"(focal={focal}, h_aperture={h_ap}, v_aperture={v_ap})")

    h_res, w_res = resolution
    return np.array([
        [w_res * focal / h_ap, 0.0, w_res / 2.0],
        [0.0, h_res * focal / v_ap, h_res / 2.0],
        [0.0, 0.0, 1.0],
    ])


def _view_matrix_ros(prim_path):
    """World -> ROS camera frame, from the prim's live world transform."""
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    local_to_world = UsdGeom.Imageable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())
    # USD matrices are row-vector major; transpose for column-vector math.
    camera_to_world = np.array(local_to_world, dtype=np.float64).T
    return R_U_TRANSFORM @ np.linalg.inv(camera_to_world)


def _world_points_from_pixels(prim_path, resolution, pixels, depths):
    """Unproject pixels + depth to world XYZ. pixels (n,2), depths (n,)."""
    K = _intrinsics_matrix(prim_path, resolution)
    view = _view_matrix_ros(prim_path)

    uv1 = np.hstack([np.asarray(pixels, dtype=np.float64),
                     np.ones((len(pixels), 1))])
    cam = np.linalg.inv(K) @ (uv1.T * np.asarray(depths, dtype=np.float64))
    cam_h = np.vstack([cam, np.ones((1, cam.shape[1]))])
    return (np.linalg.inv(view) @ cam_h)[:3].T


def _pixels_from_world_points(prim_path, resolution, points):
    """Project world XYZ to pixel coords. points (n,3) -> (n,2)."""
    K = _intrinsics_matrix(prim_path, resolution)
    view = _view_matrix_ros(prim_path)

    pts = np.asarray(points, dtype=np.float64)
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
    projected = (K @ view[:3, :]) @ homogeneous.T
    projected[:2, :] /= projected[2, :]
    return projected[:2, :].T


def _camera_frame_z(prim_path, point):
    """Depth along the optical axis. Non-positive means behind the lens."""
    view = _view_matrix_ros(prim_path)
    p = np.asarray([*point, 1.0], dtype=np.float64)
    return float((view @ p)[2])


# ═════════════════════════════════════════════════════════════
# WORLD PROJECTION
#
#   Pixel + depth -> world XYZ. Coordinates come purely from vision;
#   no USD prim lookups for parts.
#
#   Depth is the cached buffer from the frame /api/camera returned. If
#   no depth is available the request fails rather than guessing a
#   surface plane.
# ═════════════════════════════════════════════════════════════

DEPTH_MIN_M = 0.01
DEPTH_MAX_M = 100.0
DEPTH_PATCH_RADIUS = 2          # 5x5 median, robust to edge noise


def _sample_depth(depth_buf, px, py):
    """Median of valid depths in a small patch around (px, py)."""
    h, w = depth_buf.shape[:2]
    ix = int(np.clip(round(px), 0, w - 1))
    iy = int(np.clip(round(py), 0, h - 1))
    r = DEPTH_PATCH_RADIUS
    patch = depth_buf[max(0, iy - r):min(h, iy + r + 1),
                      max(0, ix - r):min(w, ix + r + 1)]
    valid = patch[(patch > DEPTH_MIN_M) & (patch < DEPTH_MAX_M)]
    return float(np.median(valid)) if valid.size else None


async def _project_to_world(params):
    """Project normalised image coordinates to world XYZ.

    Args (params dict):
        points: list of {x, y} with values in [0, 1]
        camera: alias ("rgb", "wrist", "kit", "depth") or a prim path
        fresh_depth: True to re-render instead of using the cached frame

    Returns:
        {status, world_points: [{x, y, z, depth_m}, ...], camera, ...}
        or {error} when no usable depth exists.
    """
    import omni.usd

    points = params.get("points", [])
    alias = params.get("camera", "rgb")

    sensor, cam_path = await _get_camera(alias)
    resolution = tuple(sensor.resolution)
    h_res, w_res = resolution

    # Prefer the depth captured alongside the RGB frame the caller saw.
    cached = STATE.frames.get(alias) or {}
    depth_buf = None if params.get("fresh_depth") else cached.get("depth")
    depth_source = "cached frame"
    if depth_buf is None:
        _, depth_buf = await _render_frame(sensor, alias)
        depth_source = "fresh render"

    if depth_buf is None or depth_buf.size == 0:
        return {"error": f"{alias} depth buffer unavailable — cannot project",
                "camera": cam_path}

    valid_mask = (depth_buf > DEPTH_MIN_M) & (depth_buf < DEPTH_MAX_M)
    n_valid = int(valid_mask.sum())
    if n_valid < depth_buf.size * 0.01:
        return {"error": f"{alias} depth buffer has {n_valid} valid pixels "
                         f"of {depth_buf.size} — cannot project",
                "camera": cam_path}

    h_depth, w_depth = depth_buf.shape[:2]
    print(f"  [project] {alias} @ {w_res}x{h_res} | depth {w_depth}x{h_depth} "
          f"({depth_source}, {100*n_valid/depth_buf.size:.1f}% valid) | "
          f"{len(points)} points")
    if (h_depth, w_depth) != resolution:
        print(f"  [project] WARNING: depth resolution != camera resolution")

    # ── Pixel coords: nx*w, matching the cx = w/2 intrinsics convention
    pixels, depths, misses = [], [], []
    for i, pt in enumerate(points):
        nx = float(pt.get("x", 0.5))
        ny = float(pt.get("y", 0.5))
        px = min(nx * w_depth, w_depth - 1e-3)
        py = min(ny * h_depth, h_depth - 1e-3)

        d = _sample_depth(depth_buf, px, py)
        if d is None:
            misses.append(i)
            continue
        pixels.append([px, py])
        depths.append(d)

    if not pixels:
        return {"error": f"no valid depth at any of the {len(points)} "
                         f"requested pixels", "camera": cam_path}

    world = _world_points_from_pixels(cam_path, (h_depth, w_depth),
                                      pixels, depths)

    world_points, k = [], 0
    for i in range(len(points)):
        if i in misses:
            world_points.append({"error": "no valid depth at pixel"})
            continue
        wx, wy, wz = (float(v) for v in world[k])
        world_points.append({"x": wx, "y": wy, "z": wz,
                             "depth_m": depths[k]})
        print(f"  [{i}] px({pixels[k][0]:.1f},{pixels[k][1]:.1f}) "
              f"d={depths[k]:.3f} -> world({wx:.3f}, {wy:.3f}, {wz:.3f})")
        k += 1

    if misses:
        print(f"  [project] {len(misses)} point(s) had no valid depth: {misses}")

    # Camera + ee_link world positions let the caller sanity-check the
    # geometry that backed the projection.
    extra = {}
    cam_pos = omni.usd.get_world_transform_matrix(
        omni.usd.get_context().get_stage().GetPrimAtPath(cam_path)).GetRow(3)
    extra["camera_pos"] = [float(cam_pos[0]), float(cam_pos[1]),
                           float(cam_pos[2])]
    try:
        ee = get_world_pos(EE_PATH)
        extra["ee_link_pos"] = [float(ee[0]), float(ee[1]), float(ee[2])]
    except Exception:
        pass

    return {"status": "ok", "world_points": world_points,
            "camera": cam_path, "method": "depth",
            "depth_source": depth_source, **extra}


# ═════════════════════════════════════════════════════════════
# SCENE PARTS SCANNER
# ═════════════════════════════════════════════════════════════

# Parts container prim -- all pickable parts live under this
PARTS_CONTAINER = "/World/robot_facade_full"

# Known part type labels -- only prims whose names contain one of these
# are treated as pickable parts.  Everything else (bin walls, separators,
# fixtures) is filtered out by the scene scanner.
KNOWN_PART_TYPES = [
    "motor_valve", "black_hose", "black_plate", "black_plug",
    "small_hinge", "small_tube", "silver_box", "silver_gun",
    "tube_with_clamps", "gear",
]

def _is_pickable_part(prim):
    """Check if a USD prim is a pickable part (has rigid body physics applied)."""
    from pxr import UsdPhysics
    # Direct rigid body check
    if UsdPhysics.RigidBodyAPI.Get(prim.GetStage(), prim.GetPath()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return True
    # Check immediate children for rigid body (some parts wrap mesh in a child)
    for child in prim.GetChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            return True
    return False


async def _compute_prim_center(prim_path: str):
    """Return ``{center_xy, top_z, bot_z, height, pivot, ok}`` for a prim.

    The wrist-scan workflow uses this to fetch the bin / tray geometry
    AND to read scan-pose marker positions (e.g. ``/World/robo_eye``).
    For markers, the AABB centre may differ from the prim's actual
    transform — ``pivot`` is the prim's world translation and is the
    correct value to use as an IK target.
    """
    import omni.usd
    try:
        if not prim_path:
            return {"ok": False, "error": "Empty prim path"}
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return {"ok": False, "error": f"Prim not found: {prim_path}"}
        bbox = compute_bbox_geometry(stage, prim_path, label="QUERY")
        pivot = bbox["pivot"]
        min_pt = bbox["min_pt"]
        max_pt = bbox["max_pt"]
        return {
            "ok": True,
            "prim_path": prim_path,
            "center_xy": bbox["center_xy"],
            "top_z":     bbox["top_z"],
            "bot_z":     bbox["bot_z"],
            "height":    bbox["height"],
            "pivot":     [float(pivot[0]), float(pivot[1]),
                          float(pivot[2])],
            "min_pt":    [float(min_pt[0]), float(min_pt[1]),
                          float(min_pt[2])],
            "max_pt":    [float(max_pt[0]), float(max_pt[1]),
                          float(max_pt[2])],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _scan_scene_parts():
    """Scan the USD stage for pickable parts with real-world bounding box coordinates.

    Auto-detects parts by checking for UsdPhysics.RigidBodyAPI (physics-enabled
    prims are graspable parts; static geometry like bin walls is skipped).
    Falls back to name-matching against KNOWN_PART_TYPES if no rigid bodies found.
    """
    import omni.usd
    from pxr import UsdGeom, Usd

    try:
        stage = omni.usd.get_context().get_stage()
        container = stage.GetPrimAtPath(PARTS_CONTAINER)

        if not container.IsValid():
            return {"error": f"Parts container not found: {PARTS_CONTAINER}",
                    "parts": [], "place_target": None}

        parts = []
        skipped = []
        for child in container.GetChildren():
            child_path = str(child.GetPath())
            child_name = child.GetName()

            # Auto-detect: physics-enabled prims are pickable parts
            is_rigid = _is_pickable_part(child)
            # Fallback: match prim name against known part types
            name_lower = child_name.lower()
            name_match = any(pt in name_lower for pt in KNOWN_PART_TYPES)

            if not is_rigid and not name_match:
                skipped.append(child_name)
                continue

            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True)
            world_bound = bbox_cache.ComputeWorldBound(child)
            aligned = world_bound.ComputeAlignedRange()
            min_pt = np.array(aligned.GetMin())
            max_pt = np.array(aligned.GetMax())

            # Skip zero-volume prims
            dims = max_pt - min_pt
            if np.all(dims < 1e-6):
                skipped.append(child_name)
                continue

            center = (min_pt + max_pt) / 2.0
            height = float(dims[2])
            top_z = float(max_pt[2])

            parts.append({
                "prim_path": child_path,
                "name": child_name,
                "center_xyz": [float(center[0]), float(center[1]), float(center[2])],
                "top_z": top_z,
                "height": height,
                "bbox_min": min_pt.tolist(),
                "bbox_max": max_pt.tolist(),
            })

        if skipped:
            print(f"  [SCAN] Skipped {len(skipped)} non-part prims: {skipped}")

        # Also get place destination geometry
        place_info = None
        try:
            dest_geom = compute_bbox_geometry(stage, PLACE_BOX_PATH, "DESTINATION")
            place_info = {
                "prim_path": PLACE_BOX_PATH,
                "center_xy": dest_geom["center_xy"],
                "top_z": dest_geom["top_z"],
            }
        except Exception:
            pass

        return {
            "status": "ok",
            "parts_container": PARTS_CONTAINER,
            "num_parts": len(parts),
            "parts": parts,
            "place_target": place_info,
        }

    except Exception as e:
        return {"error": f"Scene scan failed: {e}", "parts": []}


# ═════════════════════════════════════════════════════════════
# SCENE ANNOTATIONS — ground-truth coords + per-camera 2D bbox
#
# Combines USD ground-truth bboxes (_scan_scene_parts) with the
# inverse projection (world XYZ → image pixel) for the requested
# camera, so the workflow can match VLM/OWL-ViT2 detections to
# authoritative world coordinates instead of guessing via depth.
# ═════════════════════════════════════════════════════════════

async def _get_scene_annotations(camera="rgb"):
    """Return per-part GT annotations as seen from the given camera.

    For every pickable part under PARTS_CONTAINER:
      * world bbox + centre (from USD, exact)
      * 2D pixel bbox in the camera image (projected from 3D corners)
      * normalised image bbox + centre
      * visibility flag (depth buffer matches projected Z within 5cm)
      * occluded flag (centre pixel hidden by other geometry)

    EVALUATION ONLY. The grasp pipeline never reads world coords from
    here — it uses camera + depth via /api/project_to_world. These
    annotations exist so perception error can be measured against a
    known reference.
    """
    sensor, cam_path = await _get_camera(camera)
    resolution = tuple(sensor.resolution)
    h_res, w_res = resolution

    _, depth_buf = await _render_frame(sensor, camera)

    # Pull GT parts from existing scanner
    scan = await _scan_scene_parts()
    if "error" in scan:
        return {"error": scan["error"], "annotations": []}
    parts = scan.get("parts", [])

    def _project_world_to_pixel(wx, wy, wz):
        """world XYZ → (px, py, depth_to_image_plane) or None if behind cam."""
        z_cam = _camera_frame_z(cam_path, (wx, wy, wz))
        if z_cam <= 1e-3:
            return None
        px, py = _pixels_from_world_points(cam_path, resolution,
                                           [(wx, wy, wz)])[0]
        return float(px), float(py), z_cam

    annotations = []
    for part in parts:
        bmin = part["bbox_min"]
        bmax = part["bbox_max"]
        center = part["center_xyz"]

        # Project all 8 bbox corners into the image
        corners = [
            (bmin[0], bmin[1], bmin[2]), (bmax[0], bmin[1], bmin[2]),
            (bmin[0], bmax[1], bmin[2]), (bmax[0], bmax[1], bmin[2]),
            (bmin[0], bmin[1], bmax[2]), (bmax[0], bmin[1], bmax[2]),
            (bmin[0], bmax[1], bmax[2]), (bmax[0], bmax[1], bmax[2]),
        ]
        proj = [_project_world_to_pixel(*c) for c in corners]
        proj_valid = [p for p in proj if p is not None]
        if len(proj_valid) < 4:
            # Mostly behind the camera — skip
            continue

        xs = [p[0] for p in proj_valid]
        ys = [p[1] for p in proj_valid]
        x1 = max(0.0, min(xs))
        y1 = max(0.0, min(ys))
        x2 = min(float(w_res - 1), max(xs))
        y2 = min(float(h_res - 1), max(ys))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        # Project centre point separately for accurate centre pixel
        c_proj = _project_world_to_pixel(*center)
        if c_proj is not None:
            cpx, cpy, c_depth = c_proj
        else:
            cpx, cpy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            c_depth = -1.0

        # Visibility: compare depth buffer at centre pixel to projected depth
        visible = True
        occlusion_pct = 0.0
        if depth_buf is not None and c_depth > 0:
            h_d, w_d = depth_buf.shape[:2]
            measured_d = _sample_depth(depth_buf,
                                       cpx * w_d / w_res,
                                       cpy * h_d / h_res)
            # Measured depth much closer than the part's projected depth
            # means something is in front of it → occluded.
            if measured_d is not None and measured_d < c_depth - 0.05:
                visible = False
                occlusion_pct = max(0.0, min(1.0,
                                            (c_depth - measured_d) / c_depth))

        # Identify part type by name match
        name_lower = part["name"].lower()
        part_type = next(
            (pt for pt in KNOWN_PART_TYPES if pt in name_lower), None)

        annotations.append({
            "name": part["name"],
            "prim_path": part["prim_path"],
            "part_type": part_type,
            "world_center": center,
            "world_top_z": part["top_z"],
            "world_bbox_min": bmin,
            "world_bbox_max": bmax,
            "image_bbox_px": [float(x1), float(y1), float(x2), float(y2)],
            "image_bbox_norm": [
                float(x1) / w_res, float(y1) / h_res,
                float(x2) / w_res, float(y2) / h_res,
            ],
            "image_center_px": [float(cpx), float(cpy)],
            "image_center_norm": [float(cpx) / w_res, float(cpy) / h_res],
            "projected_depth_m": float(c_depth),
            "visible": bool(visible),
            "occlusion": float(occlusion_pct),
        })

    print(f"  [annotations] {len(annotations)} parts visible from {camera}")
    return {
        "status": "ok",
        "camera": camera,
        "camera_path": cam_path,
        "image_width": w_res,
        "image_height": h_res,
        "annotations": annotations,
        "place_target": scan.get("place_target"),
    }


# ═════════════════════════════════════════════════════════════
# BASIC MOTION PRIMITIVES
# ═════════════════════════════════════════════════════════════

async def _move_home():
    """Move robot to home position using interpolation to prevent self-collision.

    First retracts to transit safe height (if IK is available), then
    interpolates joint-by-joint to the home configuration.
    """
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction

    global HOME_JOINTS
    if HOME_JOINTS is None:
        HOME_JOINTS = STATE.robot.get_joint_positions().tolist()[:7]
        print(f"  [HOME] Captured home joints from current pose: {[f'{j:.3f}' for j in HOME_JOINTS]}")

    # Build the full home target
    current = STATE.robot.get_joint_positions().copy()
    home_all = current.copy()
    for i, val in enumerate(HOME_JOINTS):
        if i < len(home_all):
            home_all[i] = val
    # Open gripper for home
    home_all = set_finger_joints(STATE.robot, STATE.dof_names, FINGER_OPEN, home_all)

    # If IK is ready, first retract to a safe height to avoid collisions
    if STATE.ik_ready:
        # Retract from the TCP, not ee_link. Passing an ee_link height as
        # a TCP target made _tool_pose_from_tcp add the gripper length a
        # second time, putting the goal ~0.38 m too high — unreachable on
        # a ceiling-mounted arm that is already parked near the top.
        tcp_pos = get_tcp_pos()
        safe_z = tcp_pos[2] + TRANSIT_SAFE_HEIGHT
        # Only worth retracting if the tool is actually low. The point of
        # this step is to lift clear of the bin before travelling; when
        # the arm is already parked high, adding more height just asks for
        # a pose above the ceiling mount that cannot be reached.
        base_z = get_world_pos(ROBOT_ROOT_PATH)[2]
        if tcp_pos[2] >= base_z:
            print(f"  [home] TCP z={tcp_pos[2]:.3f} already at/above the "
                  f"mount z={base_z:.3f} — skipping the retract")
            ok = False
        else:
            locked_ori = normalize_quat(DOWNWARD_ORIENTATION)
            action, ok = ik_solve(np.array([tcp_pos[0], tcp_pos[1], safe_z]),
                                  locked_ori)
        if ok:
            targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, action)
            await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES)

    # Interpolate to home position to prevent self-collision
    await _apply_interpolated(home_all, settle_frames=SETTLE_FRAMES)
    return {"status": "ok", "action": "move_home",
            "joint_positions": STATE.robot.get_joint_positions().tolist()}


async def _set_joints(positions):
    import omni.kit.app
    pos = np.array(positions, dtype=np.float32)
    if len(pos) == 7:
        current = STATE.robot.get_joint_positions()
        full = current.copy(); full[:7] = pos; pos = full
    elif len(pos) != STATE.num_dof:
        return {"error": f"Expected {STATE.num_dof} or 7 values, got {len(pos)}"}
    STATE.robot.set_joint_positions(pos)
    for _ in range(60):
        await omni.kit.app.get_app().next_update_async()
    return {"status": "ok", "joint_positions": STATE.robot.get_joint_positions().tolist()}


async def _set_gripper(finger_value):
    """Open or close the gripper. Waits long enough for physics to settle."""
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction
    targets = set_finger_joints(STATE.robot, STATE.dof_names, finger_value)
    STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
    # 90 frames for gripper to fully close/open (with boosted stiffness, faster settle)
    for _ in range(90):
        await omni.kit.app.get_app().next_update_async()
    return {"status": "ok",
            "action": f"gripper_{'open' if finger_value < 0.1 else 'close'}",
            "finger_value": finger_value}


async def _close_gripper_slow(target_value, n_steps=12):
    """Slowly close (or open) the gripper by interpolating in n_steps.

    Each step applies a new joint target and waits ~12 physics frames, giving
    a smooth progressive closure the physics engine can track.  This prevents
    the abrupt snap of a single apply_action that can knock parts or break
    finger contact geometry.

    Typical usage: close from entry pre-shape → adaptive finger_close at
    grasp depth so the fingers wrap around the part rather than hammering it.
    """
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction

    # Read the current finger angle from the first non-inner, non-knuckle joint
    current_joints = STATE.robot.get_joint_positions()
    current_finger = 0.0
    for idx, name in enumerate(STATE.dof_names):
        if "finger_joint" in name and "inner" not in name and "knuckle" not in name:
            current_finger = float(current_joints[idx])
            break

    frames_per_step = 12  # ~0.2 s per step at 60 Hz → full close in ~2.4 s
    for step in range(1, n_steps + 1):
        frac = step / n_steps
        interp = current_finger + (target_value - current_finger) * frac
        targets = set_finger_joints(STATE.robot, STATE.dof_names, interp)
        STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(frames_per_step):
            await omni.kit.app.get_app().next_update_async()

    return {"status": "ok", "finger_value": target_value}


def _force_grip(finger_value):
    """Lock gripper on part with maximum holding force.

    Three-step grip lock:
      1. set_joint_positions  — teleport fingers to closed position (instant)
      2. set_joint_velocities — zero out any opening velocity from contacts
      3. apply_action with FINGER_CLOSE_MAX — sets PD target BEYOND the
         part surface so the controller continuously pushes fingers inward
         (like a real gripper commanding full close against a part)

    Call this AFTER every apply_action that moves the arm while holding a part.
    """
    from isaacsim.core.utils.types import ArticulationAction

    current = STATE.robot.get_joint_positions()

    # Step 1: teleport fingers to the requested position
    gripped = set_finger_joints(STATE.robot, STATE.dof_names, finger_value, current.copy())
    STATE.robot.set_joint_positions(gripped)

    # Step 2: zero out all joint velocities so no residual drift
    zero_vel = np.zeros_like(current)
    STATE.robot.set_joint_velocities(zero_vel)

    # Step 3: command PD target at FINGER_CLOSE_MAX (beyond part surface)
    #   PD error = FINGER_CLOSE_MAX - actual_finger_pos > 0
    #   → controller applies continuous closing torque → firm grip
    grip_targets = set_finger_joints(STATE.robot, STATE.dof_names, FINGER_CLOSE_MAX, gripped.copy())
    STATE.robot.apply_action(ArticulationAction(joint_positions=grip_targets))


async def _verify_grasp():
    finger_pos = STATE.robot.get_joint_positions()[7]
    is_grasping = abs(finger_pos) > 0.05
    return {"status": "ok", "is_grasping": bool(is_grasping), "finger_pos": float(finger_pos)}


# ═════════════════════════════════════════════════════════════
# LULA IK MOTION PRIMITIVES
# ═════════════════════════════════════════════════════════════

def _get_warm_start():
    """Legacy warm-start builder. cuMotion is sampling-based and ignores
    it; kept only because call sites still pass one through."""
    current = STATE.robot.get_joint_positions()
    warm = np.zeros(len(STATE.arm_names))
    for i, lula_name in enumerate(STATE.arm_names):
        for idx, dof_name in enumerate(STATE.dof_names):
            if lula_name in dof_name:
                warm[i] = current[idx]
                break
    return warm


def _get_home_warm_start():
    """Build a Lula-compatible warm-start from the HOME pose joint values.

    Used as a config-bias for IK calls where we want the solver to
    return a pose CLOSE TO HOME (elbow up, wrist down) rather than
    whatever the chained warm-start sequence has drifted into. The
    place flow uses this to break out of an "elbow-forward" config
    inherited from the pick retract before descending into the tray.

    Falls back to the current-joints warm start if HOME_JOINTS hasn't
    been captured yet.
    """
    if HOME_JOINTS is None:
        return _get_warm_start()
    warm = np.zeros(len(STATE.arm_names))
    for i, lula_name in enumerate(STATE.arm_names):
        for idx, dof_name in enumerate(STATE.dof_names):
            if lula_name in dof_name and idx < len(HOME_JOINTS):
                warm[i] = HOME_JOINTS[idx]
                break
    return warm


def _update_base_pose():
    """Refresh cuMotion's world→robot-root transform.

    Rarely needed: the root is the fixed rail mount and the gantry is a
    planned joint, so this only matters if the whole rig is repositioned
    in the scene.
    """
    if STATE.cumotion_world is None:
        return None
    pos, quat = _robot_base_pose_arrays()
    STATE.cumotion_world.update_world_to_robot_root_transforms(poses=(pos, quat))
    return pos.numpy()[0]


async def _apply_interpolated(target_joints, settle_frames=SETTLE_FRAMES, finger_value=None):
    """Move arm to target with smooth continuous motion.

    For large joint jumps (> INTERP_THRESHOLD): generates intermediate
    waypoints and STREAMS them at STREAM_FRAMES per waypoint — creating
    smooth continuous motion instead of step-pause jerks.

    For small motions: single apply_action with settle.

    In both cases, only the FINAL position gets a full settle.
    """
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction

    # If a cuMotion plan is waiting, follow it — the planned path routes
    # around obstacles, whereas the joint interpolation below would cut
    # straight to the goal and potentially through them. Consumed once so
    # a stale plan can never be replayed on a later, unrelated move.
    traj = STATE.pending_trajectory
    STATE.pending_trajectory = None
    if traj is not None:
        if await _execute_trajectory(traj, finger_value=finger_value,
                                    settle_frames=settle_frames):
            return

    current_joints = STATE.robot.get_joint_positions()

    # Build final target with finger values baked in
    final_target = target_joints if finger_value is None else \
        set_finger_joints(STATE.robot, STATE.dof_names, finger_value, target_joints.copy())

    # Smooth streaming for large motions
    if _check_self_collision_risk(current_joints, target_joints):
        waypoints = _interpolate_joints(current_joints, final_target)
        for wp in waypoints:
            STATE.robot.apply_action(ArticulationAction(joint_positions=wp))
            for _ in range(STREAM_FRAMES):
                await omni.kit.app.get_app().next_update_async()

    # Final settle at exact target
    STATE.robot.apply_action(ArticulationAction(joint_positions=final_target))
    for _ in range(settle_frames):
        await omni.kit.app.get_app().next_update_async()


async def _ik_move_to(target_xyz, orientation=None, settle_frames=SETTLE_FRAMES):
    """Move end-effector to XYZ along a cuMotion-planned trajectory."""
    if not STATE.ik_ready:
        return {"error": _planner_unavailable_reason()}

    target_pos = np.array(target_xyz, dtype=np.float64)
    ori = orientation if orientation is not None else normalize_quat(DOWNWARD_ORIENTATION)

    warm = _get_warm_start()
    action, ok = ik_solve(target_pos, ori, warm)
    if not ok:
        return {"error": f"IK failed for target {target_xyz}"}

    target_joints = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, action)
    tcp_before = get_tcp_pos()
    await _apply_interpolated(target_joints, settle_frames=settle_frames)

    # Verify the move actually happened. A plan can succeed and the
    # trajectory can stream while the joints never follow — reporting
    # "ok" then sends the rest of the pipeline (wrist VLM, descend) to
    # look for a part the gripper is nowhere near.
    tcp_after = get_tcp_pos()
    moved = float(np.linalg.norm(tcp_after - tcp_before))
    err = float(np.linalg.norm(tcp_after - target_pos))
    print(f"  [move] TCP {np.round(tcp_before, 3).tolist()} -> "
          f"{np.round(tcp_after, 3).tolist()}")
    print(f"  [move] travelled {moved:.4f} m; {err:.4f} m from the "
          f"requested target")
    if moved < 1e-3:
        print("  [move] *** TCP DID NOT MOVE — the plan was executed but the "
              "joints did not follow ***")
    elif err > 0.05:
        print(f"  [move] *** TCP ended {err:.3f} m from target — the arm "
              f"moved, but not to where it was asked ***")

    ee_pos = get_world_pos(EE_PATH)
    return {"status": "ok", "target": list(target_xyz),
            "ee_position": ee_pos.tolist(),
            "tcp_position": tcp_after.tolist(),
            "tcp_travelled": moved, "tcp_error": err}


async def _cartesian_pose_move(target_pos, target_ori, step_size=0.04,
                                stream_frames=None):
    """Move EE to (x, y, z) target via Cartesian micro-steps in full 3D.

    Generalises ``_retract_cartesian_up`` for arbitrary XYZ moves.
    Each step is warm-started from the previous IK solution so the
    solver stays in the same joint-space configuration family —
    prevents shoulder-flip / elbow-up contortions on long XY hops.
    Used by /api/approach when ``cartesian: True`` is set in payload
    (scan-pose move + per-pick realign).

    Returns ``(ok, ee_position)``.
    """
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction

    frames_per_step = stream_frames if stream_frames is not None else STREAM_FRAMES
    target_pos = np.asarray(target_pos, dtype=np.float64)
    current_ee = get_world_pos(EE_PATH)
    delta = target_pos - current_ee
    dist = float(np.linalg.norm(delta))
    if dist < 0.005:
        return True, current_ee.tolist()

    n_steps = max(2, int(np.ceil(dist / step_size)))
    warm = _get_warm_start()
    last_action = None

    for step in range(1, n_steps + 1):
        frac = step / n_steps
        step_pos = current_ee + delta * frac
        action, ok = ik_solve(step_pos, target_ori, warm)
        if not ok:
            print(f"  [CART_POSE] IK failed at step {step}/{n_steps} "
                  f"pos={step_pos.tolist()} — stopping here")
            return False, get_world_pos(EE_PATH).tolist()
        targets = apply_arm_joints(STATE.robot, STATE.dof_names,
                                    STATE.arm_names, action)
        STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(frames_per_step):
            await omni.kit.app.get_app().next_update_async()
        warm = action
        last_action = action

    # Final settle at exact target
    final_action, final_ok = ik_solve(target_pos, target_ori, warm)
    if final_ok:
        targets = apply_arm_joints(STATE.robot, STATE.dof_names,
                                    STATE.arm_names, final_action)
        STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
    for _ in range(SETTLE_FRAMES):
        await omni.kit.app.get_app().next_update_async()
    return final_ok, get_world_pos(EE_PATH).tolist()


async def _descend_with_contact_stop(target_z, x, y, locked_ori,
                                      warm_start, finger_value=None,
                                      step_size=0.01,
                                      contact_min_force=0.5,
                                      contact_lift_back=0.002,
                                      stream_frames=None):
    """Cartesian descent with per-step contact-sensor monitoring.

    Identical to ``_retract_cartesian_up`` for descent, except it polls
    the gripper's contact sensor after each Cartesian micro-step and
    STOPS THE DESCENT the moment a finger touches anything (a part, the
    bin floor, a divider). Prevents the "gripper pushed into bin
    bottom" failure mode where the planned grasp_z is below the part's
    actual top, leaving the gripper jammed against the floor.

    Args:
        target_z:           planned final Z (used as the "go no further"
                            limit if no contact is detected first).
        x, y:               world XY to hold constant during descent.
        locked_ori:         normalised TCP quaternion.
        warm_start:         IK warm-start joints (use the hover IK
                            result for continuity).
        finger_value:       if not None, bake this finger angle into
                            every waypoint (keeps fingers open during
                            the approach descent).
        step_size:          Cartesian Z step (m) per IK solve. 1 cm
                            default — small enough that contact is
                            detected before significant overshoot.
        contact_min_force:  Newtons. Threshold above which we declare
                            "in contact" and stop. Mirrors the
                            ``_adaptive_close`` threshold so behaviour
                            is consistent with the gripper-close phase.
        stream_frames:      physics frames per step (defaults to
                            ``STREAM_FRAMES``).

    Returns:
        ``{"ok": bool, "stopped_z": float, "contact": bool,
           "force": float, "steps_taken": int}``
    """
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction

    frames_per_step = stream_frames if stream_frames is not None else STREAM_FRAMES
    current_ee = get_world_pos(EE_PATH)
    current_z = float(current_ee[2])

    if current_z - target_z < 0.005:
        return {"ok": True, "stopped_z": current_z, "contact": False,
                "force": 0.0, "steps_taken": 0}

    # Confirmation line so the operator can see in the Isaac Sim console
    # that the contact-aware code path is actually running (vs. the
    # bridge being on a stale build that uses the old single-jump IK).
    # Use the FINGERTIP-BOTTOM sensor for descent stop detection.
    # The inner-finger PAD sensor (used by the gripper-close motion)
    # rarely registers when the tip strikes the bin floor because its
    # normal points inward toward the part, not downward.
    sensor_attached = STATE.contact_sensor_tip is not None
    print(f"  [DESCEND-CONTACT] starting from z={current_z:.3f} → "
          f"target z={target_z:.3f}, threshold={contact_min_force:.2f}N, "
          f"step={step_size*1000:.0f}mm, tip_sensor_attached={sensor_attached}")
    if not sensor_attached:
        print("  [DESCEND-CONTACT] WARNING: contact_sensor_tip is None — "
              "descent will run to planned target without floor-touch "
              "feedback. Check the tip-sensor init log line.")

    n_steps = max(2, int(np.ceil((current_z - target_z) / step_size)))
    warm = warm_start
    in_contact = False
    contact_force = 0.0
    last_z = current_z

    for step in range(1, n_steps + 1):
        frac = step / n_steps
        z_step = current_z + (target_z - current_z) * frac
        step_pos = np.array([x, y, z_step])

        action, ok = ik_solve(step_pos, locked_ori, warm)
        if not ok:
            print(f"  [DESCEND-CONTACT] IK failed at step {step}/{n_steps} "
                  f"z={z_step:.3f} — stopping here")
            break

        targets = apply_arm_joints(STATE.robot, STATE.dof_names,
                                   STATE.arm_names, action)
        if finger_value is not None:
            targets = set_finger_joints(STATE.robot, STATE.dof_names,
                                        finger_value, targets)
        STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(frames_per_step):
            await omni.kit.app.get_app().next_update_async()

        warm = action
        last_z = float(get_world_pos(EE_PATH)[2])

        # Per-step contact check — uses the FINGERTIP-BOTTOM sensor
        # (NOT the inner-pad sensor used by the gripper close). If the
        # finger tip touches the bin floor or the top of a part during
        # descent, stop immediately. Per-step force ALWAYS logged (even
        # when below threshold) so silent zeros are visible.
        touching, force_mag = _read_tip_contact_sensor()
        print(f"  [DESCEND-CONTACT] step {step}/{n_steps} "
              f"z={last_z:.3f} force={force_mag:.3f}N "
              f"touching={touching}")
        if touching and force_mag >= contact_min_force:
            in_contact = True
            contact_force = force_mag
            print(f"  [DESCEND-CONTACT] Contact at z={last_z:.3f}, "
                  f"force={force_mag:.2f}N (step {step}/{n_steps}) "
                  f"— stopping descent above planned target "
                  f"{target_z:.3f}")

            # Lift back a couple of mm so the finger pads aren't
            # pressed against the surface — gives the fingers room
            # to close around the part instead of jamming against
            # the bin floor / part top.
            if contact_lift_back > 0:
                lift_z = last_z + contact_lift_back
                lift_pos = np.array([x, y, lift_z])
                lift_action, lift_ok = ik_solve(lift_pos, locked_ori, warm)
                if lift_ok:
                    targets = apply_arm_joints(
                        STATE.robot, STATE.dof_names,
                        STATE.arm_names, lift_action)
                    if finger_value is not None:
                        targets = set_finger_joints(
                            STATE.robot, STATE.dof_names,
                            finger_value, targets)
                    STATE.robot.apply_action(
                        ArticulationAction(joint_positions=targets))
                    for _ in range(SETTLE_FRAMES):
                        await omni.kit.app.get_app().next_update_async()
                    after_lift_z = float(get_world_pos(EE_PATH)[2])
                    print(f"  [DESCEND-CONTACT] Lifted back "
                          f"{contact_lift_back*1000:.1f} mm "
                          f"to z={after_lift_z:.3f} for clean close")
                    last_z = after_lift_z
                else:
                    print(f"  [DESCEND-CONTACT] WARNING: lift-back IK "
                          f"failed (target z={lift_z:.3f}); fingers may "
                          f"jam during close")
            break

    # Settle at whatever Z we reached (either contact-stopped or target).
    # NOTE: an XY snap-back step was tried here but reverted — the
    # extra IK solve at end-of-descent occasionally found a different
    # joint configuration that swung the EE outside the bin entirely.
    # The per-step Cartesian micro-stepping already constrains XY
    # tightly enough; residual drift (a few mm) is preferable to a
    # fall-back IK solve that can pick a wildly different arm pose.
    for _ in range(SETTLE_FRAMES):
        await omni.kit.app.get_app().next_update_async()

    return {"ok": True, "stopped_z": last_z, "contact": in_contact,
            "force": contact_force, "steps_taken": step
            if 'step' in locals() else 0}


async def _cartesian_lateral_move(target_x, target_y, z, locked_ori,
                                   finger_value=None, step_size=0.05,
                                   stream_frames=None):
    """Move the end-effector laterally to (target_x, target_y) at
    constant Z via Cartesian micro-steps.

    A single IK solve to a far-away target lets the joint-space PD
    controller sweep through whatever curved Cartesian path the joint
    interpolation produces — and that path can DIP below the start
    altitude during the swing (because the elbow rotates down/forward).
    This is what was clipping the kitting-tray rim during the place
    transit.

    Cartesian micro-stepping forces the EE to trace a straight line in
    world space at constant Z: each step is a fresh IK solve at a
    progressively-different (x, y) but the SAME z, warm-started from
    the previous result so the joint configuration stays in the same
    family. No dipping, no clipping.

    Args:
        target_x, target_y: world destination XY (metres).
        z:                  world Z to hold constant during the move.
        locked_ori:         normalised TCP quaternion.
        finger_value:       finger angle to bake into every waypoint
                            (e.g. ``grip_hold`` to keep the part).
        step_size:          max Cartesian XY distance (m) per step.
                            Default 5 cm — small enough to stay
                            under joint limits, large enough to
                            avoid making 50 IK calls for short moves.
        stream_frames:      physics frames per step (defaults to
                            ``STREAM_FRAMES``).

    Returns:
        ``True`` on success, ``False`` if any IK step fails mid-way
        (caller decides whether to abort or continue).
    """
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction

    frames_per_step = stream_frames if stream_frames is not None else STREAM_FRAMES
    current_ee = get_world_pos(EE_PATH)
    current_x = float(current_ee[0])
    current_y = float(current_ee[1])

    dx = target_x - current_x
    dy = target_y - current_y
    distance = (dx * dx + dy * dy) ** 0.5
    if distance < 0.01:
        return True  # already at target — nothing to do

    n_steps = max(2, int(np.ceil(distance / step_size)))
    warm = _get_warm_start()

    for step in range(1, n_steps + 1):
        frac = step / n_steps
        x_step = current_x + dx * frac
        y_step = current_y + dy * frac
        step_pos = np.array([x_step, y_step, z])

        action, ok = ik_solve(step_pos, locked_ori, warm)
        if not ok:
            print(f"  [CARTESIAN_LATERAL] IK failed at step {step}/"
                  f"{n_steps} ({x_step:.3f}, {y_step:.3f}, {z:.3f}) "
                  f"— stopping mid-move")
            return False

        targets = apply_arm_joints(STATE.robot, STATE.dof_names,
                                   STATE.arm_names, action)
        if finger_value is not None:
            targets = set_finger_joints(STATE.robot, STATE.dof_names,
                                        finger_value, targets)
        STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(frames_per_step):
            await omni.kit.app.get_app().next_update_async()

        warm = action

    # Final settle at the exact (target_x, target_y, z).
    final_pos = np.array([target_x, target_y, z])
    final_action, final_ok = ik_solve(final_pos, locked_ori, warm)
    if final_ok:
        targets = apply_arm_joints(STATE.robot, STATE.dof_names,
                                   STATE.arm_names, final_action)
        if finger_value is not None:
            targets = set_finger_joints(STATE.robot, STATE.dof_names,
                                        finger_value, targets)
        STATE.robot.apply_action(ArticulationAction(joint_positions=targets))

    for _ in range(SETTLE_FRAMES):
        await omni.kit.app.get_app().next_update_async()

    return final_ok


async def _retract_cartesian_up(target_z, x, y, locked_ori,
                                 finger_value=None, step_size=0.04,
                                 stream_frames=None):
    """Move the end-effector to target_z via Cartesian micro-steps (up or down).

    Works for both retract (up) and slow descent (down) — direction is derived
    from current EE Z vs. target_z.

    A single large IK jump lets the solver find a distant joint-space solution
    that requires shoulder/elbow to flip.  Micro-stepping prevents this: each
    step is warm-started from the previous result so the solver stays in the
    same configuration family the entire way.

    Args:
        target_z:      world Z to reach (metres).
        x, y:          world XY to hold constant.
        locked_ori:    normalised TCP quaternion.
        finger_value:  if not None, bake this finger target into every waypoint.
        step_size:     max Cartesian distance (m) per IK solve (default 4 cm).
        stream_frames: physics frames per waypoint; defaults to global STREAM_FRAMES.
    Returns:
        True on success, False if an IK step fails mid-way.
    """
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction

    frames_per_step = stream_frames if stream_frames is not None else STREAM_FRAMES

    current_ee = get_world_pos(EE_PATH)
    current_z  = current_ee[2]

    if abs(current_z - target_z) < 0.01:
        return True  # already close enough — nothing to do

    n_steps = max(2, int(np.ceil(abs(target_z - current_z) / step_size)))
    warm    = _get_warm_start()

    for step in range(1, n_steps + 1):
        frac     = step / n_steps
        z_step   = current_z + (target_z - current_z) * frac
        step_pos = np.array([x, y, z_step])

        action, ok = ik_solve(step_pos, locked_ori, warm)
        if not ok:
            direction = "up" if target_z > current_z else "down"
            print(f"  [CARTESIAN_{direction.upper()}] IK failed at step {step}/{n_steps} z={z_step:.3f} — stopping here")
            break

        targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, action)
        if finger_value is not None:
            targets = set_finger_joints(STATE.robot, STATE.dof_names, finger_value, targets)

        STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(frames_per_step):
            await omni.kit.app.get_app().next_update_async()

        warm = action   # chain: each step seeds the next

    # Final settle at the exact target position
    final_pos    = np.array([x, y, target_z])
    final_action, final_ok = ik_solve(final_pos, locked_ori, warm)
    if final_ok:
        targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, final_action)
        if finger_value is not None:
            targets = set_finger_joints(STATE.robot, STATE.dof_names, finger_value, targets)
        STATE.robot.apply_action(ArticulationAction(joint_positions=targets))

    for _ in range(SETTLE_FRAMES):
        await omni.kit.app.get_app().next_update_async()

    return final_ok


async def _smooth_joint_retract(safe_pos, locked_ori, finger_value=None):
    """Lift arm to safe height in ONE smooth joint-space motion after grasping.

    Strategy
    --------
    1. IK-solve for safe_pos using the current (grasp) joints as warm-start.
       Because the warm-start is the actual grasping configuration, the solver
       stays in the same joint-space neighbourhood — no shoulder/elbow flip.
    2. Check: if any joint change > 0.8 rad the solver still flipped →
       fall back to Cartesian micro-steps which are flip-proof.
    3. If the solution is sane, joint-interpolate with fine steps (0.08 rad max
       per waypoint) and 10 physics frames per waypoint — the arm appears to
       perform a single, deliberate upward motion rather than jittery micro-IK.

    This replaces the post-grasp _retract_cartesian_up call so the robot lifts
    naturally with minimal joint changes (primarily shoulder_lift/elbow).
    """
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction

    grasp_warm = _get_warm_start()

    target_action, ok = ik_solve(safe_pos, locked_ori, grasp_warm)
    if not ok:
        print("  [SMOOTH_RETRACT] IK failed — falling back to micro-steps")
        return await _retract_cartesian_up(
            safe_pos[2], safe_pos[0], safe_pos[1], locked_ori,
            finger_value=finger_value, stream_frames=10)

    # Detect configuration flip: any arm joint jumping > 0.8 rad = bad solution
    max_delta = float(np.max(np.abs(np.array(target_action) - np.array(grasp_warm))))
    if max_delta > 0.8:
        print(f"  [SMOOTH_RETRACT] IK flip detected (Δ={max_delta:.2f} rad) — micro-steps")
        return await _retract_cartesian_up(
            safe_pos[2], safe_pos[0], safe_pos[1], locked_ori,
            finger_value=finger_value, stream_frames=10)

    # Build full joint target
    current_full = STATE.robot.get_joint_positions()
    target_full  = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, target_action)
    if finger_value is not None:
        target_full = set_finger_joints(STATE.robot, STATE.dof_names, finger_value, target_full.copy())

    # Fine joint-space interpolation (0.08 rad/waypoint) with 10 frames each
    waypoints = _interpolate_joints(current_full, target_full, max_step=0.08)
    for wp in waypoints:
        if finger_value is not None:
            wp = set_finger_joints(STATE.robot, STATE.dof_names, finger_value, wp.copy())
        STATE.robot.apply_action(ArticulationAction(joint_positions=wp))
        for _ in range(10):   # 10 frames per step ≈ 0.17 s per 0.08 rad — visible, smooth
            await omni.kit.app.get_app().next_update_async()

    # Final settle at exact target
    STATE.robot.apply_action(ArticulationAction(joint_positions=target_full))
    for _ in range(SETTLE_FRAMES):
        await omni.kit.app.get_app().next_update_async()

    return True


async def _move_gantry_x(target_x, finger_hold=None, hold_ee_pos=None):
    """Move the X-gantry to align with a target X coordinate.

    Args:
        target_x: world X coordinate the gantry should reach.
        finger_hold: if not None, set finger PD targets to this value
                     during the move so the gripper maintains holding force.
        hold_ee_pos: if not None, a 3-element world position [x,y,z] that the
                     end-effector should stay locked onto during the gantry slide.
                     Like a human pointing a finger at a spot while walking —
                     the arm joints compensate each step to keep the EE stationary.
    """
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction

    targets = STATE.robot.get_joint_positions()
    gantry_idx = None
    for idx, name in enumerate(STATE.dof_names):
        if GANTRY_X_JOINT in name:
            gantry_idx = idx
            break

    if gantry_idx is None:
        return {"error": "Gantry joint not found"}

    gantry_target = target_x - GANTRY_X_OFFSET
    gantry_current = float(targets[gantry_idx])
    gantry_delta = gantry_target - gantry_current

    # ── EE-hold mode: interpolate gantry + compensate arm each step ──
    # Pre-computes arm compensation BEFORE moving the gantry so both are
    # applied in a single apply_action — no "drift then correct" gap.
    # The gantry is a prismatic joint along X, so the base_link shifts
    # by exactly the gantry delta in X.  We predict the new base pose
    # analytically and solve IK against it, then apply gantry + arm
    # simultaneously.  X,Y are locked tight; Z has slight flexibility
    # so IK has room to find valid solutions.
    if hold_ee_pos is not None and STATE.ik_ready and abs(gantry_delta) > 0.005:
        # The gantry is joint 0 of the XRDF cspace, so ONE pose plan
        # covers "slide the rail while the tool stays put" — cuMotion
        # coordinates gantry and arm itself. The old code had to predict
        # the base pose per step and re-solve because Lula treated the
        # gantry as an external base shift.
        ee_lock = np.array(hold_ee_pos, dtype=np.float64)
        locked_ori = normalize_quat(DOWNWARD_ORIENTATION)
        print(f"  [GANTRY] EE-hold slide: {gantry_current:.3f} → "
              f"{gantry_target:.3f}, holding EE at "
              f"({ee_lock[0]:.3f}, {ee_lock[1]:.3f}, {ee_lock[2]:.3f})")

        q_goal = _current_cspace().copy()
        q_goal[0] = gantry_target          # cspace joint 0 == gantry
        traj = _plan_cspace(q_goal)
        if traj is not None:
            await _execute_trajectory(traj, finger_value=finger_hold)
        else:
            print("  [GANTRY] plan failed — falling back to direct joint move")
            targets = STATE.robot.get_joint_positions().copy()
            targets[gantry_idx] = gantry_target
            if finger_hold is not None:
                targets = set_finger_joints(
                    STATE.robot, STATE.dof_names, finger_hold, targets)
            STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
            for _ in range(SETTLE_FRAMES):
                await omni.kit.app.get_app().next_update_async()

        ee_final = get_world_pos(EE_PATH)
        drift_xy = np.linalg.norm(ee_final[:2] - ee_lock[:2])
        drift_z = abs(ee_final[2] - ee_lock[2])
        print(f"  [GANTRY] EE-hold done. EE=({ee_final[0]:.3f}, {ee_final[1]:.3f}, {ee_final[2]:.3f}) "
              f"XY-drift={drift_xy:.4f}m  Z-drift={drift_z:.4f}m")
        return {"status": "ok", "gantry_x": float(gantry_target),
                "ee_drift_xy": float(drift_xy), "ee_drift_z": float(drift_z)}

    # ── Simple mode: just move gantry without arm compensation ──
    targets[gantry_idx] = gantry_target
    if finger_hold is not None:
        targets = set_finger_joints(STATE.robot, STATE.dof_names, finger_hold, targets)

    STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
    for _ in range(SETTLE_FRAMES):
        await omni.kit.app.get_app().next_update_async()

    return {"status": "ok", "gantry_x": float(gantry_target)}


async def _approach_target(params):
    """Move arm near target XYZ.

    Two modes (selected by ``raw_z`` in the payload):

    * ``raw_z=False`` (default, pick-approach): adds a safety buffer
      ``BOX_ENTRY_MARGIN + BOX_HEIGHT + GRIPPER_TCP_OFFSET + 0.10``
      to the requested z so the gripper hovers safely above a bin wall
      before the descent sequence takes over.

    * ``raw_z=True`` (pose move): the requested z is sent to IK
      verbatim. Used by the wrist-scan pipeline to park the ee_link
      at an exact world height for camera framing or gantry
      re-alignment, with no buffering and no restriction.

    All coordinates come from the workflow engine — either USD prim
    bbox lookups (bin/tray) or VLM + depth projection (parts).
    """
    x = params.get("x", 0.3)
    y = params.get("y", 0.0)
    z = params.get("z", 0.5)
    raw_z = bool(params.get("raw_z", False))
    cartesian = bool(params.get("cartesian", False))
    orientation = params.get("orientation")  # optional [w, x, y, z]

    if raw_z:
        target_z = z
    else:
        target_z = z + BOX_ENTRY_MARGIN + BOX_HEIGHT + GRIPPER_TCP_OFFSET + 0.10

    # Move gantry to EXACT target X first, then update base pose
    await _move_gantry_x(x)
    _update_base_pose()

    if orientation is not None:
        try:
            ori = normalize_quat(orientation)
        except Exception as e:
            return {"error": f"Invalid orientation {orientation}: {e}"}
    else:
        ori = normalize_quat(DOWNWARD_ORIENTATION)

    if cartesian:
        # Step-by-step Cartesian motion — keeps the arm in the same
        # joint-space configuration family across long XY hops, so
        # the IK solver doesn't flip into a shoulder-up / elbow-back
        # pose halfway through.
        ok, ee_pos = await _cartesian_pose_move(
            [x, y, target_z], ori)
        if not ok:
            return {"error": f"Cartesian IK failed for target "
                             f"[{x}, {y}, {target_z}]"}
        return {"status": "ok", "action": "approach",
                "target": [x, y, target_z], "ee_position": ee_pos,
                "mode": "cartesian"}

    result = await _ik_move_to([x, y, target_z], orientation=ori)
    if "error" in result:
        return result

    return {"status": "ok", "action": "approach",
            "target": [x, y, target_z], "ee_position": result["ee_position"]}


async def _realign_gantry_hold_ee(params):
    """Slide the gantry to ``params['x']`` while holding the
    end-effector locked at its current world position.

    Used by the workflow's depth-verify-and-realign step: depth
    refines the part XY by a few cm, the gantry needs to nudge
    along X to put the wrist exactly above the corrected target,
    but we don't want the EE to swing up or sideways during the
    nudge — the wrist should stay parked on its current vantage
    so the next Cartesian descent enters the bin compartment from
    the same pose we just confirmed visually.

    Implements this with `_move_gantry_x(target_x, hold_ee_pos=ee_now)`
    which interpolates the gantry in micro-steps while solving IK
    against a base pose that shifts by exactly the gantry delta —
    arm joints compensate analytically per step.
    """
    if not STATE.is_ready:
        return {"error": "Not ready"}
    if not STATE.ik_ready:
        return {"error": _planner_unavailable_reason()}

    target_x = params.get("x")
    if target_x is None:
        return {"error": "Missing 'x' in payload"}

    ee_before = get_world_pos(EE_PATH).copy()
    await _move_gantry_x(float(target_x), hold_ee_pos=ee_before)
    _update_base_pose()
    ee_after = get_world_pos(EE_PATH)

    drift = float(np.linalg.norm(ee_after - ee_before))
    print(f"  [REALIGN] gantry → x={target_x:.3f}; "
          f"EE drift = {drift*1000:.1f} mm")
    return {
        "status": "ok",
        "action": "realign_gantry_hold_ee",
        "target_x": float(target_x),
        "ee_before": ee_before.tolist(),
        "ee_after":  ee_after.tolist(),
        "ee_drift_m": drift,
    }


def _pick_z_positions(z, grasp_lift_override=None):
    """Shared Z waypoint calculations for the 3-phase pick sequence.

    ``grasp_lift_override`` (optional) — caller can override the global
    ``GRASP_LIFT`` constant per pick. Negative values descend BELOW the
    part top to grip mid-body for tall parts; zero parks the finger
    TCP exactly at part top (default), which lets the contact-aware
    descent stop on first touch and lift back a clean 5 mm.
    """
    tcp_z_offset = 0.0 if STATE.target_frame == "gripper_tcp" else GRIPPER_TCP_OFFSET
    grasp_lift = GRASP_LIFT
    if grasp_lift_override is not None:
        try:
            grasp_lift = float(grasp_lift_override)
        except (TypeError, ValueError):
            pass
    part_top_z = z
    grasp_z = z + grasp_lift       # TCP at part top + lift (lift can be 0 or negative)
    safe_z = part_top_z + BOX_HEIGHT + BOX_ENTRY_MARGIN + tcp_z_offset
    entry_z = part_top_z + BOX_HEIGHT + tcp_z_offset
    hover_z = part_top_z + HOVER_CLEARANCE + tcp_z_offset
    grasp_target_z = grasp_z + tcp_z_offset
    transit_z = safe_z + TRANSIT_SAFE_HEIGHT
    return dict(safe_z=safe_z, entry_z=entry_z, hover_z=hover_z,
                grasp_target_z=grasp_target_z, transit_z=transit_z,
                tcp_z_offset=tcp_z_offset, grasp_lift=grasp_lift)


# ─── PICK PHASE 1: Descend to grasp position ───────────────

async def _pick_descend(params):
    """Descend to grasp position with open fingers.

    Gantry aligns → arm descends to grasp height → captures wrist image.
    The gripper stays OPEN — the caller must verify part placement via the
    returned wrist image before requesting gripper close.

    Returns wrist_image (base64) so the workflow can run VLM verification.
    """
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction

    if not STATE.ik_ready:
        return {"error": _planner_unavailable_reason()}

    x = params.get("x", 0.3)
    y = params.get("y", 0.0)
    z = params.get("z", 0.02)

    STATE.is_executing = True
    steps = []
    try:
        locked_ori = normalize_quat(DOWNWARD_ORIENTATION)
        # Optional per-part grasp-depth override from the workflow.
        # Workflow looks this up in the parts catalogue (negative values
        # descend below part top to grip mid-body of tall parts).
        grasp_lift_override = params.get("grasp_lift_override")
        zp = _pick_z_positions(z, grasp_lift_override=grasp_lift_override)
        safe_z = zp["safe_z"]

        safe_pos = np.array([x, y, safe_z])
        entry_pos = np.array([x, y, zp["entry_z"]])
        hover_pos = np.array([x, y, zp["hover_z"]])
        grasp_pos = np.array([x, y, zp["grasp_target_z"]])

        print(f"  [DESCEND] target=({x:.3f}, {y:.3f}, {z:.3f}) "
              f"grasp_lift={zp['grasp_lift']:+.3f} "
              f"safe_z={safe_z:.3f} hover_z={zp['hover_z']:.3f} "
              f"grasp_z={zp['grasp_target_z']:.3f}")

        # 1. Open gripper
        await _set_gripper(FINGER_OPEN)
        steps.append({"phase": "open_gripper", "status": "ok"})

        # 2. Gantry alignment — slide gantry to part X while keeping
        #    EE locked at its current world position (depth-scan pose).
        #    Like a human pointing a finger at something while walking:
        #    the arm compensates continuously so the fingertip stays put.
        ee_before = get_world_pos(EE_PATH).copy()
        await _move_gantry_x(x, hold_ee_pos=ee_before)
        steps.append({"phase": "gantry", "status": "ok"})

        # 2b. XY correction — after gantry slide the EE may have drifted
        #     slightly in X.  Before descending, re-solve IK to put the
        #     EE exactly above the part [x, y] at the current height.
        #     This is the critical "snap to target" before the grasp descent.
        current_ee = get_world_pos(EE_PATH)
        xy_error = np.linalg.norm(current_ee[:2] - np.array([x, y]))
        if xy_error > 0.002:  # >2mm drift → correct
            correct_pos = np.array([x, y, float(current_ee[2])])
            print(f"  [DESCEND] XY correction: drift={xy_error:.4f}m, "
                  f"snapping EE to ({x:.3f}, {y:.3f}, {current_ee[2]:.3f})")
            correct_warm = _get_warm_start()
            correct_action, correct_ok = ik_solve(correct_pos, locked_ori, correct_warm)
            if correct_ok:
                targets = apply_arm_joints(
                    STATE.robot, STATE.dof_names, STATE.arm_names, correct_action)
                await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES)
                corrected_ee = get_world_pos(EE_PATH)
                print(f"  [DESCEND] XY corrected → ({corrected_ee[0]:.3f}, "
                      f"{corrected_ee[1]:.3f}, {corrected_ee[2]:.3f})")
            else:
                print("  [DESCEND] WARNING: XY correction IK failed")
            steps.append({"phase": "xy_correction", "status": "ok" if correct_ok else "warn"})

        # 3. Safe height (vertical-first to avoid bin wall sweeps)
        current_ee = get_world_pos(EE_PATH)
        warm = _get_warm_start()
        if current_ee[2] < safe_z - 0.05:
            v_ok = await _retract_cartesian_up(
                safe_z, current_ee[0], current_ee[1], locked_ori, step_size=0.04)
            if not v_ok:
                return {"error": "Cartesian retract failed", "steps": steps}
            steps.append({"phase": "safe_retract", "status": "ok"})
            warm = _get_warm_start()

        p1_action, p1_ok = ik_solve(safe_pos, locked_ori, warm)
        if not p1_ok:
            return {"error": f"IK safe failed {safe_pos.tolist()}", "steps": steps}
        targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, p1_action)
        await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES)
        steps.append({"phase": "safe_height", "status": "ok"})

        # 4. Re-solve seed
        settled_warm = _get_warm_start()
        p2_action, p2_ok = ik_solve(safe_pos, locked_ori, settled_warm)
        if not p2_ok:
            p2_action = p1_action

        # 5. Entry — just inside box rim
        entry_action, entry_ok = ik_solve(entry_pos, locked_ori, p2_action)
        if entry_ok:
            targets = apply_arm_joints(
                STATE.robot, STATE.dof_names, STATE.arm_names, entry_action)
            await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES)
            steps.append({"phase": "entry", "status": "ok"})
            p2_action = entry_action

        # 6. Hover — above part
        hover_action, hover_ok = ik_solve(hover_pos, locked_ori, p2_action)
        if not hover_ok:
            return {"error": f"IK hover failed {hover_pos.tolist()}", "steps": steps}
        targets = apply_arm_joints(
            STATE.robot, STATE.dof_names, STATE.arm_names, hover_action)
        await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES)
        steps.append({"phase": "hover", "status": "ok"})

        # 6b. Verify EE is directly above the part at hover height.
        #     Each IK solve at a different Z can drift X,Y slightly.
        #     This is the last correction before grasp — the grasp step
        #     is just a tiny Z drop, so nailing X,Y here guarantees a
        #     precise grasp.
        hover_ee = get_world_pos(EE_PATH)
        hover_xy_err = np.linalg.norm(hover_ee[:2] - np.array([x, y]))
        print(f"  [DESCEND] Hover EE=({hover_ee[0]:.3f}, {hover_ee[1]:.3f}, "
              f"{hover_ee[2]:.3f})  XY error={hover_xy_err:.4f}m")
        if hover_xy_err > 0.002:  # >2mm → re-solve at hover height
            repose_target = np.array([x, y, float(hover_ee[2])])
            print(f"  [DESCEND] Re-posing EE to ({x:.3f}, {y:.3f}, {hover_ee[2]:.3f})")
            repose_warm = _get_warm_start()
            repose_action, repose_ok = ik_solve(repose_target, locked_ori, repose_warm)
            if repose_ok:
                targets = apply_arm_joints(
                    STATE.robot, STATE.dof_names, STATE.arm_names, repose_action)
                await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES)
                reposed_ee = get_world_pos(EE_PATH)
                print(f"  [DESCEND] Re-posed → ({reposed_ee[0]:.3f}, "
                      f"{reposed_ee[1]:.3f}, {reposed_ee[2]:.3f})")
                hover_action = repose_action  # use corrected as warm start for grasp
            else:
                print("  [DESCEND] WARNING: hover re-pose IK failed")
            steps.append({"phase": "hover_repose",
                           "status": "ok" if repose_ok else "warn"})

        # 7. Grasp descent — Cartesian micro-step with contact-sensor
        #    stop. The PLANNED grasp_z (from _pick_z_positions) assumes
        #    the part top Z is exactly correct; if depth projection is
        #    off by 1-2 cm the gripper would overshoot into the bin
        #    floor or jam against the part. Instead we step down a few
        #    millimetres at a time and abort the moment a finger
        #    touches anything (part / floor / divider). Same contact
        #    sensor + threshold as the gripper-close phase, so the
        #    behaviour is consistent across the pipeline.
        contact_min_force = float(
            params.get("descent_contact_min_force", 0.5))
        contact_lift_back = float(
            params.get("descent_contact_lift_back", 0.002))
        # 5 mm step instead of 10 mm — each per-step IK call introduces
        # a tiny XY drift (~0.5 mm) because the solver picks a slightly
        # different joint configuration at each Z slice. With 1 cm
        # steps over a 22 cm descent that compounds to ~12 mm of XY
        # drift visible at the gripper. Halving the step roughly
        # halves the drift; the cost is twice as many IK solves
        # (still <2 s extra wall-clock per pick).
        descent_step = float(
            params.get("descent_step_size", 0.005))
        descent_result = await _descend_with_contact_stop(
            target_z=zp["grasp_target_z"],
            x=x, y=y,
            locked_ori=locked_ori,
            warm_start=hover_action,
            finger_value=FINGER_OPEN,  # keep fingers open during approach
            step_size=descent_step,
            contact_min_force=contact_min_force,
            contact_lift_back=contact_lift_back)
        if not descent_result.get("ok", False):
            return {"error": "Cartesian descent IK failed",
                    "steps": steps}
        steps.append({
            "phase": "grasp_descend",
            "status": "ok",
            "contact": descent_result.get("contact", False),
            "stopped_z": descent_result.get("stopped_z"),
            "contact_force": descent_result.get("force", 0.0),
        })
        if descent_result.get("contact"):
            print(f"  [PICK] Descent stopped on contact at z="
                  f"{descent_result.get('stopped_z'):.3f} "
                  f"(planned grasp_z={zp['grasp_target_z']:.3f}); "
                  f"force={descent_result.get('force'):.2f}N")

        # 7b. End-of-descent telemetry — show the actual EE world
        #     position vs. the planned target so any residual XY drift
        #     is visible. If this consistently shows a non-zero
        #     drift, either:
        #       (a) IK accumulation over the descent steps — bump
        #           descent step_size down or add per-step XY snap;
        #       (b) Camera-EE offset — apply
        #           execution.wrist_target_(x|y)_offset to compensate;
        #       (c) Bbox-centre vs grasp-centre mismatch — re-tune
        #           OWL-ViT2 query phrasing or tighten bbox heuristics.
        actual_ee = get_world_pos(EE_PATH)
        dx_drift = float(actual_ee[0]) - x
        dy_drift = float(actual_ee[1]) - y
        dz_drift = float(actual_ee[2]) - zp["grasp_target_z"]
        xy_drift = (dx_drift ** 2 + dy_drift ** 2) ** 0.5
        print(f"  [DESCEND] End-of-descent EE vs planned target:")
        print(f"             planned (x, y, z) = ({x:.4f}, {y:.4f}, "
              f"{zp['grasp_target_z']:.4f})")
        print(f"             actual  (x, y, z) = ({actual_ee[0]:.4f}, "
              f"{actual_ee[1]:.4f}, {actual_ee[2]:.4f})")
        print(f"             drift   (dx, dy, dz) = ({dx_drift*1000:+.1f}, "
              f"{dy_drift*1000:+.1f}, {dz_drift*1000:+.1f}) mm  "
              f"|XY|={xy_drift*1000:.1f} mm")

        # 7c. Post-descent XY snap — single IK solve at the actual
        #     stopped Z that targets EXACTLY (x, y) again. The
        #     Cartesian micro-stepping accumulates joint-family drift
        #     that shows up as residual XY at the bottom of the
        #     descent (~12 mm in past runs). One IK solve at the
        #     stopped Z, warm-started from the current joint state,
        #     pulls the EE back onto the planned XY without the
        #     "swing-out-of-bin" failure mode the previous attempt
        #     had — that was caused by the OLD URDF having
        #     gripper_tcp on the wrong axis (+Z instead of +X), so
        #     the solver had to pick a wildly different config to
        #     hit the target. With the corrected URDF the post-
        #     descent IK stays in the same shoulder/elbow family.
        #
        #     Gated on ``post_descent_xy_snap`` (default true) so
        #     the operator can disable it if it ever causes problems.
        do_post_snap = bool(params.get("post_descent_xy_snap", True))
        snap_threshold = float(
            params.get("post_descent_xy_snap_threshold", 0.003))  # 3 mm
        if do_post_snap and xy_drift > snap_threshold:
            print(f"  [DESCEND] Post-descent XY snap: drift "
                  f"{xy_drift*1000:.1f} mm > "
                  f"{snap_threshold*1000:.1f} mm → re-IKing to "
                  f"({x:.4f}, {y:.4f}) at z={float(actual_ee[2]):.4f}")
            snap_pos = np.array([x, y, float(actual_ee[2])])
            snap_warm = STATE.robot.get_joint_positions()[:6]
            snap_action, snap_ok = ik_solve(snap_pos, locked_ori, snap_warm)
            if snap_ok:
                snap_targets = apply_arm_joints(
                    STATE.robot, STATE.dof_names,
                    STATE.arm_names, snap_action)
                snap_targets = set_finger_joints(
                    STATE.robot, STATE.dof_names,
                    FINGER_OPEN, snap_targets)
                STATE.robot.apply_action(
                    ArticulationAction(joint_positions=snap_targets))
                for _ in range(SETTLE_FRAMES):
                    await omni.kit.app.get_app().next_update_async()
                snapped_ee = get_world_pos(EE_PATH)
                snap_residual = (
                    (float(snapped_ee[0]) - x) ** 2
                    + (float(snapped_ee[1]) - y) ** 2) ** 0.5
                print(f"  [DESCEND] Post-snap EE = ({snapped_ee[0]:.4f}, "
                      f"{snapped_ee[1]:.4f}, {snapped_ee[2]:.4f}); "
                      f"residual XY = {snap_residual*1000:.1f} mm")
                steps.append({
                    "phase": "post_descent_xy_snap",
                    "status": "ok",
                    "residual_mm": snap_residual * 1000.0,
                })
            else:
                print(f"  [DESCEND] Post-snap IK FAILED — leaving EE "
                      f"at drifted position; close may grasp off-centre")
                steps.append({
                    "phase": "post_descent_xy_snap",
                    "status": "warn",
                    "detail": "IK rejected; left as-is",
                })

        # 8. Capture wrist camera — caller uses this for VLM "part between fingers?" check
        wrist_frame = await _capture_camera("wrist")
        wrist_b64 = wrist_frame.get("image_base64")
        steps.append({"phase": "wrist_capture",
                       "status": "ok" if wrist_b64 else "warn"})

        return {"status": "ok", "action": "pick_descend",
                "position": [x, y, z], "steps": steps,
                "wrist_image": wrist_b64,
                "ee_drift_mm": {
                    "dx": dx_drift * 1000,
                    "dy": dy_drift * 1000,
                    "dz": dz_drift * 1000,
                    "xy_total": xy_drift * 1000,
                }}

    except Exception as e:
        return {"status": "error", "error": str(e), "steps": steps}
    finally:
        STATE.is_executing = False


# ─── PICK PHASE 2: Close gripper and confirm ───────────────

def _read_contact_sensor():
    """Read the contact sensor on the left inner finger PAD.

    This sensor's normal points INWARD toward the part — it fires
    when the gripper closes around something. Used by the adaptive
    gripper-close motion to declare "in contact with part."

    Returns (in_contact: bool, force_magnitude: float).
    """
    if STATE.contact_sensor is None:
        return False, 0.0
    try:
        frame = STATE.contact_sensor.get_current_frame()
        in_contact = bool(frame.get("in_contact", False))
        force_vec = frame.get("force", [0, 0, 0])
        force_mag = float(np.linalg.norm(force_vec))
        return in_contact, force_mag
    except Exception:
        return False, 0.0


def _read_tip_contact_sensor():
    """Read the contact sensor on the FINGERTIP BOTTOM.

    This sensor's normal is aligned with the descent axis — it fires
    when the finger tip touches the bin floor or the top of a part
    during the Cartesian Z descent. Used by the contact-aware descent
    to halt before the gripper jams into the surface.

    Returns (in_contact: bool, force_magnitude: float).
    """
    if STATE.contact_sensor_tip is None:
        return False, 0.0
    try:
        frame = STATE.contact_sensor_tip.get_current_frame()
        in_contact = bool(frame.get("in_contact", False))
        force_vec = frame.get("force", [0, 0, 0])
        force_mag = float(np.linalg.norm(force_vec))
        return in_contact, force_mag
    except Exception:
        return False, 0.0


async def _adaptive_close(min_force=0.5, step_rad=0.02, max_steps=30,
                           overshoot_rad=0.04):
    """Close gripper incrementally using contact sensor feedback.

    Closes in small steps, checking the contact sensor after each.
    Stops when contact force exceeds min_force, then applies a small
    overshoot for firm hold.

    Returns (final_angle, contact_detected, force).
    """
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction

    current_angle = 0.0
    contact_detected = False
    contact_force = 0.0

    for step in range(max_steps):
        current_angle += step_rad
        if current_angle > ROBOTIQ_MAX_RAD:
            current_angle = ROBOTIQ_MAX_RAD

        targets = STATE.robot.get_joint_positions()
        targets = set_finger_joints(
            STATE.robot, STATE.dof_names, current_angle, targets)
        STATE.robot.apply_action(ArticulationAction(joint_positions=targets))

        # Let physics settle (4 frames per step)
        for _ in range(4):
            await omni.kit.app.get_app().next_update_async()

        in_contact, force_mag = _read_contact_sensor()
        if in_contact and force_mag >= min_force:
            contact_detected = True
            contact_force = force_mag
            print(f"  [ADAPTIVE] Contact at step {step+1}, "
                  f"angle={current_angle:.3f} rad, force={force_mag:.2f}N")
            break

        if current_angle >= ROBOTIQ_MAX_RAD:
            print(f"  [ADAPTIVE] Max angle reached without contact")
            break

    # Apply overshoot for firm hold
    if contact_detected:
        firm_angle = min(current_angle + overshoot_rad, ROBOTIQ_MAX_RAD)
        targets = STATE.robot.get_joint_positions()
        targets = set_finger_joints(
            STATE.robot, STATE.dof_names, firm_angle, targets)
        STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()
        current_angle = firm_angle
        _, contact_force = _read_contact_sensor()
        print(f"  [ADAPTIVE] Firm hold at {firm_angle:.3f} rad, "
              f"force={contact_force:.2f}N")

    return current_angle, contact_detected, contact_force


async def _pick_close(params):
    """Close the gripper and confirm the part is grasped.

    Two modes:
      - Adaptive (sensor available): incremental close with contact feedback
      - Preset (no sensor): fixed angle with retry loop

    Returns a wrist camera image so the caller can run VLM verification
    that the part is properly held before requesting retract.
    """
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction

    finger_close = params.get("finger_close", FINGER_CLOSE)
    GRASP_CONFIRM_THRESHOLD = 0.05  # rad

    STATE.is_executing = True
    try:
        grasp_confirmed = False
        actual_finger = 0.0
        grip_attempt = 0
        adaptive_used = False

        # ── Adaptive mode: contact sensor driven ─────────────
        if STATE.contact_sensor is not None:
            adaptive_used = True
            print(f"  [CLOSE] Adaptive mode (contact sensor)")
            final_angle, contact_ok, force = await _adaptive_close()
            actual_finger = final_angle
            grasp_confirmed = contact_ok
            finger_close = final_angle
            grip_attempt = 1

            if not contact_ok:
                # Fallback: close to preset angle
                print(f"  [CLOSE] No contact detected — "
                      f"falling back to preset {FINGER_CLOSE:.3f}")
                finger_close = FINGER_CLOSE
                targets = STATE.robot.get_joint_positions()
                targets = set_finger_joints(
                    STATE.robot, STATE.dof_names, finger_close, targets)
                STATE.robot.apply_action(
                    ArticulationAction(joint_positions=targets))
                for _ in range(120):
                    await omni.kit.app.get_app().next_update_async()
                actual_finger = finger_close
                grasp_confirmed = True  # assume preset works

        # ── Preset mode: fixed angle with retries ────────────
        else:
            for grip_attempt in range(1, 4):
                print(f"  [CLOSE] attempt {grip_attempt}/3  target={finger_close:.3f}")
                targets = STATE.robot.get_joint_positions()
                targets = set_finger_joints(
                    STATE.robot, STATE.dof_names, finger_close, targets)
                STATE.robot.apply_action(ArticulationAction(joint_positions=targets))

                settle = 120 if grip_attempt == 1 else 180
                for _ in range(settle):
                    await omni.kit.app.get_app().next_update_async()

                actual_joints = STATE.robot.get_joint_positions()
                for idx, name in enumerate(STATE.dof_names):
                    if ("finger_joint" in name and "inner" not in name
                            and "knuckle" not in name):
                        actual_finger = float(actual_joints[idx])
                        break

                print(f"  [CLOSE] actual={actual_finger:.4f}  "
                      f"threshold={GRASP_CONFIRM_THRESHOLD}")

                if actual_finger >= GRASP_CONFIRM_THRESHOLD:
                    grasp_confirmed = True
                    print(f"  [CLOSE] ✓ Grasp confirmed (attempt {grip_attempt})")
                    break
                print(f"  [CLOSE] ✗ Retrying...")

        grip_hold = finger_close
        grasp_check = await _verify_grasp()

        # Capture wrist image for "is part properly grasped?" VLM check
        wrist_frame = await _capture_camera("wrist")
        wrist_b64 = wrist_frame.get("image_base64")

        return {"status": "ok", "action": "pick_close",
                "grasp_confirmed": grasp_confirmed,
                "finger_close": finger_close,
                "actual_finger": round(actual_finger, 4),
                "grip_hold": grip_hold,
                "attempts": grip_attempt,
                "adaptive": adaptive_used,
                "grasp_check": grasp_check,
                "wrist_image": wrist_b64}

    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        STATE.is_executing = False


# ─── PICK PHASE 3: Retract with part ───────────────────────

async def _pick_retract(params):
    """Retract the arm to safe + transit height while holding the part.

    Must be called after _pick_close — the gripper is already closed.
    """
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction

    if not STATE.ik_ready:
        return {"error": _planner_unavailable_reason()}

    x = params.get("x", 0.3)
    y = params.get("y", 0.0)
    z = params.get("z", 0.02)
    grip_hold = params.get("grip_hold", FINGER_CLOSE)

    STATE.is_executing = True
    steps = []
    try:
        locked_ori = normalize_quat(DOWNWARD_ORIENTATION)
        zp = _pick_z_positions(z)
        safe_pos = np.array([x, y, zp["safe_z"]])
        transit_pos = np.array([x, y, zp["transit_z"]])

        # Re-enforce grip before any retract motion — extra settle ensures
        # fingers are fully closed and physics contacts are established
        print(f"  [RETRACT] Re-enforcing grip (hold={grip_hold:.3f}) before lift")
        targets = STATE.robot.get_joint_positions()
        targets = set_finger_joints(
            STATE.robot, STATE.dof_names, grip_hold, targets)
        STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(90):  # extra 90 frames (~1.5s) to let fingers fully seat
            await omni.kit.app.get_app().next_update_async()

        # Retract to safe height — SINGLE IK (NOT Cartesian).
        # Cartesian micro-stepping was tried here but it preserved the
        # grasp-pose's folded joint configuration during the climb,
        # producing the weird elbow-over-bin retract. A fresh IK from
        # current joints lets the solver pick a clean upward
        # configuration. The grip is re-enforced before AND after each
        # IK call so finger force is preserved across the jumps.
        print(f"  [RETRACT] Single-IK lift to safe_z={safe_pos[2]:.3f}")
        retract_warm = _get_warm_start()
        retract_action, retract_ok = ik_solve(safe_pos, locked_ori, retract_warm)
        if retract_ok:
            targets = apply_arm_joints(STATE.robot, STATE.dof_names,
                                       STATE.arm_names, retract_action)
            targets = set_finger_joints(STATE.robot, STATE.dof_names,
                                        grip_hold, targets)
            STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
            for _ in range(150):
                await omni.kit.app.get_app().next_update_async()
        steps.append({"phase": "retract",
                       "status": "ok" if retract_ok else "failed"})
        if not retract_ok:
            return {"error": f"IK failed for retract safe {safe_pos.tolist()}",
                    "steps": steps}

        # Transit to clear-height — also single IK with grip re-enforced.
        print(f"  [RETRACT] Single-IK transit to z={transit_pos[2]:.3f}")
        transit_warm = _get_warm_start()
        transit_action, transit_ok = ik_solve(transit_pos, locked_ori, transit_warm)
        if transit_ok:
            targets = apply_arm_joints(STATE.robot, STATE.dof_names,
                                       STATE.arm_names, transit_action)
            targets = set_finger_joints(STATE.robot, STATE.dof_names,
                                        grip_hold, targets)
            STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
            for _ in range(120):
                await omni.kit.app.get_app().next_update_async()
        steps.append({"phase": "transit_safe",
                       "status": "ok" if transit_ok else "skipped"})

        return {"status": "completed", "action": "pick_retract",
                "position": [x, y, z], "steps": steps,
                "grip_hold": grip_hold}

    except Exception as e:
        return {"status": "error", "error": str(e), "steps": steps}
    finally:
        STATE.is_executing = False


# ─── Combined pick (backward-compatible) ───────────────────

async def _pick_object_ik(params):
    """Full pick sequence — calls descend → close → retract in one shot.

    Kept for /api/pick backward compatibility. The 3-step workflow with
    wrist-camera VLM checks between phases uses the individual endpoints
    /api/pick_descend, /api/pick_close, /api/pick_retract.
    """
    # Phase 1: Descend
    descend_result = await _pick_descend(params)
    if descend_result.get("status") == "error":
        return descend_result

    # Phase 2: Close
    close_result = await _pick_close(params)
    if close_result.get("status") == "error":
        return close_result

    # Phase 3: Retract
    retract_params = dict(params)
    retract_params["grip_hold"] = close_result.get("grip_hold", FINGER_CLOSE)
    retract_result = await _pick_retract(retract_params)

    # Merge results
    all_steps = (descend_result.get("steps", []) +
                 [{"phase": "close_gripper",
                   "grasp_confirmed": close_result.get("grasp_confirmed"),
                   "actual_finger": close_result.get("actual_finger")}] +
                 retract_result.get("steps", []))

    return {"status": retract_result.get("status", "error"),
            "action": "pick_object",
            "position": descend_result.get("position"),
            "steps": all_steps,
            "finger_close": close_result.get("finger_close", FINGER_CLOSE),
            "grip_hold": close_result.get("grip_hold", FINGER_CLOSE),
            "wrist_image": close_result.get("wrist_image")}


async def _place_object_ik(params):
    """Full IK-based place sequence with chained warm-starts.

    All coordinates come from VLM + depth projection — no USD prim paths.
    The workflow engine passes x/y/z from the depth-projected kitting tray
    position detected by the VLM in Phase 0+1.

    Same multi-phase pattern as pick — each IK solution chains
    as warm-start for the next to prevent self-collision.
    """
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction

    if not STATE.ik_ready:
        return {"error": _planner_unavailable_reason()}

    x = params.get("x", 0.5)
    y = params.get("y", 0.0)
    z = params.get("z", 0.02)
    # Optional EXTRA transit clearance (metres) — added ONLY to the
    # transit phase (the lateral move at altitude before the descent).
    # Caller passes this when the default ~14 cm transit headroom isn't
    # enough for the IK to find a non-folded arm pose (e.g. forearm
    # clipping the kitting tray rim). Does NOT affect the place /
    # release / retract heights — the actual drop stays at
    # PLACE_DROP_HEIGHT above the tray, so parts land cleanly.
    transit_clearance = float(params.get("transit_clearance", 0.0))
    # Optional override of PLACE_DROP_HEIGHT — the gap between the
    # finger TCP and the tray surface at the release point. Caller
    # passes this when the default 2 cm is too tight (parts catch on
    # the tray rim) or too loose (parts bounce). Falls back to
    # PLACE_DROP_HEIGHT if not supplied.
    place_z_buffer = params.get("place_z_buffer", None)
    if place_z_buffer is not None:
        place_z_buffer = float(place_z_buffer)
    # Use finger_close and contact-aware grip_hold from pick result
    finger_close = params.get("finger_close", FINGER_CLOSE)
    grip_hold = params.get("grip_hold", finger_close)

    STATE.is_executing = True
    steps = []
    try:
        locked_ori = normalize_quat(DOWNWARD_ORIENTATION)
        tcp_z_offset = 0.0 if STATE.target_frame == "gripper_tcp" else GRIPPER_TCP_OFFSET

        # Use the passed coordinates directly (from VLM + depth projection)
        dest_x, dest_y, dest_top_z = x, y, z

        # place_z_buffer overrides PLACE_DROP_HEIGHT if the caller sent it.
        drop_height = (place_z_buffer if place_z_buffer is not None
                        else PLACE_DROP_HEIGHT)
        place_z = dest_top_z + drop_height + tcp_z_offset
        place_safe_z = dest_top_z + BOX_ENTRY_MARGIN + tcp_z_offset
        retract_z = dest_top_z + PLACE_RETRACT_HEIGHT + tcp_z_offset
        if place_z_buffer is not None:
            print(f"  [PLACE] place_z_buffer={place_z_buffer:.3f} m "
                  f"(overriding PLACE_DROP_HEIGHT={PLACE_DROP_HEIGHT:.3f}); "
                  f"finger TCP releases {drop_height*100:.1f} cm above tray")

        # Transit altitude = safe_z + base TRANSIT_SAFE_HEIGHT + caller-
        # supplied extra clearance. Lateral moves only — descent below
        # uses place_z which is unchanged.
        transit_pos = np.array([dest_x, dest_y,
                                place_safe_z + TRANSIT_SAFE_HEIGHT
                                + transit_clearance])
        if transit_clearance > 0:
            print(f"  [PLACE] transit_clearance={transit_clearance:.3f} m "
                  f"applied → transit_z={transit_pos[2]:.3f} "
                  f"(place_z={place_z:.3f} unchanged)")
        safe_pos = np.array([dest_x, dest_y, place_safe_z])
        place_pos = np.array([dest_x, dest_y, place_z])
        retract_pos = np.array([dest_x, dest_y, retract_z])

        # 1. Transit -- move arm to safe height before lateral gantry move.
        #    Single IK solve (same pattern as pick retract).
        current_ee = get_world_pos(EE_PATH)
        transit_up_z = current_ee[2] + TRANSIT_SAFE_HEIGHT + transit_clearance
        transit_up_pos = np.array([current_ee[0], current_ee[1], transit_up_z])
        transit_up_warm = _get_warm_start()
        transit_up_action, transit_up_ok = ik_solve(transit_up_pos, locked_ori, transit_up_warm)
        if transit_up_ok:
            targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, transit_up_action)
            targets = set_finger_joints(STATE.robot, STATE.dof_names, grip_hold, targets)
            STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
            for _ in range(150):
                await omni.kit.app.get_app().next_update_async()
        steps.append({"phase": "transit_up", "status": "ok" if transit_up_ok else "warn"})

        # 2. Move gantry to destination X — simple mode (no hold_ee_pos).
        #    The EE travels WITH the gantry, carrying the part to the
        #    new destination.  grip_hold keeps finger contact during slide.
        await _move_gantry_x(dest_x, finger_hold=grip_hold)
        steps.append({"phase": "gantry_move", "status": "ok"})

        # 2b. Reset arm config to a home-biased pose (elbow up, wrist
        #     down). The pick retract + gantry slide leaves the arm
        #     in whatever joint configuration Lula picked for the
        #     pick-retract IK — sometimes "elbow forward" with the
        #     forearm pointing into the workspace. That config gets
        #     PRESERVED through the lateral move (Cartesian, chained
        #     warm-starts) and the descent (also Cartesian), planting
        #     the arm body inside the tray rim.
        #
        #     Re-IK to the SAME current EE position but with the home
        #     joints as warm-start. Lula returns a config close to
        #     home (elbow up); the joint controller drives the arm
        #     there over a few frames, then we proceed.
        if params.get("place_reset_arm_config", True):
            current_ee = get_world_pos(EE_PATH)
            reset_pos = np.array([
                float(current_ee[0]), float(current_ee[1]),
                float(current_ee[2])])
            reset_warm = _get_home_warm_start()
            reset_action, reset_ok = ik_solve(reset_pos, locked_ori, reset_warm)
            if reset_ok:
                targets = apply_arm_joints(
                    STATE.robot, STATE.dof_names, STATE.arm_names,
                    reset_action)
                targets = set_finger_joints(
                    STATE.robot, STATE.dof_names, grip_hold, targets)
                STATE.robot.apply_action(
                    ArticulationAction(joint_positions=targets))
                for _ in range(90):  # ~1.5 s settle
                    await omni.kit.app.get_app().next_update_async()
                print(f"  [PLACE] Arm config reset to home-bias at "
                      f"({reset_pos[0]:.3f}, {reset_pos[1]:.3f}, "
                      f"{reset_pos[2]:.3f})")
            else:
                print("  [PLACE] WARNING: home-bias reset IK failed; "
                      "proceeding with current config")
            steps.append({
                "phase": "reset_arm_config",
                "status": "ok" if reset_ok else "skipped"})

        # 3. Transit at destination — Cartesian XY micro-step at
        #    constant Z. The single-IK approach used to dip the EE
        #    below the planned transit altitude during the joint
        #    swing (shoulder + elbow rotating to reach dest_y),
        #    clipping the kitting tray rim. The Cartesian lateral
        #    move forces a straight-line world-space path at the
        #    constant transit altitude — no dipping.
        transit_dest_ok = await _cartesian_lateral_move(
            target_x=dest_x, target_y=dest_y,
            z=float(transit_pos[2]),
            locked_ori=locked_ori,
            finger_value=grip_hold,
            step_size=0.05)
        steps.append({"phase": "transit_at_dest",
                       "status": "ok" if transit_dest_ok else "warn"})

        # 4. Verify EE is EXACTLY over the tray centre at transit
        #    altitude before any descent begins. The Cartesian lateral
        #    move ends with a final IK to (dest_x, dest_y, transit_z)
        #    but small drift can remain. If we don't snap that drift
        #    out before descending, the next IK can pick a config that
        #    swings the elbow forward over the tray, dropping the
        #    wrist into the tray walls.
        #
        #    User spec: "maintain safe height until exactly over the
        #    kitting tray, THEN place the part."
        ee_at_transit = get_world_pos(EE_PATH)
        xy_drift = float(np.linalg.norm(
            ee_at_transit[:2] - np.array([dest_x, dest_y])))
        print(f"  [PLACE] EE at transit altitude: ("
              f"{ee_at_transit[0]:.3f}, {ee_at_transit[1]:.3f}, "
              f"{ee_at_transit[2]:.3f})  XY drift from tray centre: "
              f"{xy_drift*1000:.1f} mm")
        if xy_drift > 0.005:  # >5 mm
            snap_pos = np.array([dest_x, dest_y, float(ee_at_transit[2])])
            snap_warm = _get_warm_start()
            snap_action, snap_ok = ik_solve(snap_pos, locked_ori, snap_warm)
            if snap_ok:
                targets = apply_arm_joints(STATE.robot, STATE.dof_names,
                                           STATE.arm_names, snap_action)
                targets = set_finger_joints(STATE.robot, STATE.dof_names,
                                            grip_hold, targets)
                STATE.robot.apply_action(
                    ArticulationAction(joint_positions=targets))
                for _ in range(60):
                    await omni.kit.app.get_app().next_update_async()
                ee_after = get_world_pos(EE_PATH)
                after_drift = float(np.linalg.norm(
                    ee_after[:2] - np.array([dest_x, dest_y])))
                print(f"  [PLACE] EE snap-to-tray-centre: "
                      f"{xy_drift*1000:.1f} mm → "
                      f"{after_drift*1000:.1f} mm at transit altitude")
            else:
                print(f"  [PLACE] WARNING: snap-to-tray IK failed; "
                      f"descending with {xy_drift*1000:.1f} mm drift")
        steps.append({"phase": "transit_verify_at_tray", "status": "ok"})

        # 5. SINGLE Cartesian Z descent from transit altitude all the
        #    way to place_z. X/Y is locked at (dest_x, dest_y) so the
        #    EE traces a straight vertical line through tray centre.
        #    No intermediate "safe_pos" IK that could reconfigure the
        #    arm into an elbow-forward pose; the Cartesian descent
        #    inherits the verified-correct config from step 4 and
        #    preserves it the whole way down. Combines the previous
        #    place_safe + place_descend phases — simpler, safer.
        place_ok = await _retract_cartesian_up(
            place_z, dest_x, dest_y, locked_ori,
            finger_value=grip_hold, step_size=0.03, stream_frames=8)
        steps.append({"phase": "place_descend",
                       "status": "ok" if place_ok else "warn"})

        # 8. Open gripper — release part
        await _set_gripper(FINGER_OPEN)
        steps.append({"phase": "release", "status": "ok"})

        # 9. Retract — interpolated (no finger hold, part released)
        retract_warm = _get_warm_start()
        retract_action, retract_ok = ik_solve(retract_pos, locked_ori, retract_warm)
        if retract_ok:
            targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, retract_action)
            await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES)
        steps.append({"phase": "retract", "status": "ok"})

        # 10. Transit safe height after retract (clear kitting tray before going home)
        retract_transit_warm = _get_warm_start()
        retract_transit_z = retract_z + TRANSIT_SAFE_HEIGHT
        retract_transit_pos = np.array([dest_x, dest_y, retract_transit_z])
        rt_action, rt_ok = ik_solve(retract_transit_pos, locked_ori, retract_transit_warm)
        if rt_ok:
            targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, rt_action)
            await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES)
        steps.append({"phase": "transit_safe_post_place", "status": "ok"})

        return {"status": "completed", "action": "place_object",
                "position": [dest_x, dest_y, dest_top_z], "steps": steps}

    except Exception as e:
        return {"status": "error", "error": str(e), "steps": steps}
    finally:
        STATE.is_executing = False


# ═════════════════════════════════════════════════════════════
# EXECUTE PLAN (action primitive dispatcher)
# ═════════════════════════════════════════════════════════════

async def _execute_plan(plan_steps):
    import omni.kit.app
    STATE.is_executing = True
    results = []
    try:
        for step in plan_steps:
            step_num = step.get("step", "?")
            action = step.get("action", "")
            params = step.get("params", {})
            step_result = {"step": step_num, "action": action}
            t0 = time.time()
            try:
                if action == "move_home":
                    r = await _move_home()
                elif action == "open_gripper":
                    r = await _set_gripper(FINGER_OPEN)
                elif action == "close_gripper":
                    r = await _set_gripper(FINGER_CLOSE)
                elif action == "pick_object":
                    r = await _pick_object_ik(params)
                elif action == "place_object":
                    r = await _place_object_ik(params)
                elif action == "move_to_pose":
                    pos = [params.get("x", 0), params.get("y", 0), params.get("z", 0)]
                    r = await _ik_move_to(pos)
                elif action == "verify_grasp":
                    r = await _verify_grasp()
                elif action == "request_perception_update":
                    r = {"status": "ok", "note": "Perception handled by Streamlit"}
                else:
                    r = {"status": "skipped", "reason": f"Unknown: {action}"}
                step_result["status"] = "success"
                step_result["detail"] = r
            except Exception as e:
                step_result["status"] = "failed"
                step_result["error"] = str(e)
            step_result["duration"] = round(time.time() - t0, 2)
            results.append(step_result)
    finally:
        STATE.is_executing = False

    return {
        "status": "completed",
        "total_steps": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "steps": results,
    }


# ═════════════════════════════════════════════════════════════
# BRIDGE LIFECYCLE MANAGEMENT
#
# The bridge starts when this script runs in Isaac Sim's Script
# Editor and stops automatically when the simulation is stopped.
# This releases port 8600 so Streamlit can detect disconnection.
# ═════════════════════════════════════════════════════════════

_bridge_server = None          # HTTPServer instance (set on start, cleared on stop)
_bridge_server_thread = None   # server thread
_timeline_sub = None           # timeline event subscription (for cleanup)
_bridge_running = False        # flag for the async command loop


def _shutdown_bridge(reason="simulation stopped"):
    """Shut down the HTTP server and reset bridge state.

    Called when the simulation is stopped or paused in Isaac Sim.
    Releases port 8600 so Streamlit sees "bridge not reachable".
    """
    global _bridge_server, _bridge_server_thread, _bridge_running

    if not _bridge_running and _bridge_server is None:
        return  # already shut down

    _bridge_running = False

    print(f"\n{'=' * 60}")
    print(f"  KITTING BRIDGE — Shutting down ({reason})")
    print(f"{'=' * 60}")

    # Stop HTTP server — releases port 8600
    if _bridge_server is not None:
        try:
            _bridge_server.shutdown()
            _bridge_server.server_close()
            print("[OK] HTTP server stopped — port 8600 released")
        except Exception as e:
            print(f"[WARN] HTTP server shutdown error: {e}")
        _bridge_server = None

    _bridge_server_thread = None

    # Reset shared state so a fresh start_bridge() re-initialises everything
    STATE.is_ready = False
    STATE.ik_ready = False
    STATE.is_executing = False
    STATE.cameras.clear()
    STATE.frames.clear()
    STATE.cumotion_robot = None
    STATE.cumotion_world = None
    STATE.cumotion_planner = None
    STATE.cumotion_ready = False
    STATE.pending_trajectory = None
    STATE.collision_sphere_count = 0

    # Drain any pending commands/results so they don't leak into next session
    with STATE._lock:
        STATE._command_queue.clear()
        STATE._result_queue.clear()

    print("[OK] Bridge state reset — Streamlit will see 'disconnected'")
    print(f"{'=' * 60}\n")


def _on_timeline_event(event):
    """Callback fired by Isaac Sim when the timeline state changes.

    Shuts down the bridge when the simulation is stopped or paused.
    """
    import omni.timeline
    if event.type == int(omni.timeline.TimelineEventType.STOP):
        _shutdown_bridge("simulation stopped")
    elif event.type == int(omni.timeline.TimelineEventType.PAUSE):
        _shutdown_bridge("simulation paused")


async def start_bridge():
    global _bridge_server, _bridge_server_thread, _timeline_sub, _bridge_running

    import omni.kit.app, omni.timeline
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation

    # ── Clean up any previous session ────────────────────────
    _shutdown_bridge("restarting")

    print("=" * 60)
    print("  KITTING BRIDGE SERVER v2 — Starting...")
    print("=" * 60)

    # ── World & Robot ────────────────────────────────────────
    world = World.instance()
    if not world:
        world = World(physics_dt=1/60, rendering_dt=1/60)

    robot = world.scene.get_object("kitting_bridge")
    if not robot:
        robot = SingleArticulation(prim_path=ROBOT_PRIM, name="kitting_bridge")
        world.scene.add(robot)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(10):
        await omni.kit.app.get_app().next_update_async()

    robot.initialize()

    STATE.robot = robot
    STATE.world = world
    STATE.num_dof = robot.num_dof
    STATE.dof_names = robot.dof_names
    STATE.is_ready = True
    """
    # Moderately boost finger joint stiffness so the PD controller holds
    # parts during arm motion without generating destructive contact forces.
    # 2000 N·m/rad is ~5-20x the URDF default — firm grip, not explosive.
    try:
        controller = robot.get_articulation_controller()
        kps, kds = controller.get_gains()
        for idx, name in enumerate(STATE.dof_names):
            if "finger_joint" in name or "knuckle_joint" in name or "inner_finger_joint" in name:
                kps[idx] = 2000.0   # firm, not destructive
                kds[idx] = 200.0
        controller.set_gains(kps, kds)
        print("[OK] Finger joint stiffness set (Kp=2000, Kd=200)")
    except Exception as e:
        print(f"[WARN] Could not set finger gains: {e}")
    """
    # Hold current pose at startup — tell the PD controller "stay here"
    # BEFORE physics advances, so the boosted Kp=2000 fingers don't snap
    # closed and the arm doesn't jerk to default zero targets.
    from isaacsim.core.utils.types import ArticulationAction
    startup_joints = robot.get_joint_positions()
    startup_joints = set_finger_joints(robot, STATE.dof_names, FINGER_OPEN, startup_joints.copy())
    robot.apply_action(ArticulationAction(joint_positions=startup_joints))
    for _ in range(30):
        await omni.kit.app.get_app().next_update_async()

    # Capture the robot's initial rest pose as HOME_JOINTS
    global HOME_JOINTS
    initial_joints = robot.get_joint_positions()
    HOME_JOINTS = initial_joints.tolist()[:7]
    print(f"[OK] Robot ready — {STATE.num_dof} DOFs")
    print(f"[OK] Home joints captured: {[f'{j:.3f}' for j in HOME_JOINTS]}")

    # ── cuMotion motion planning ─────────────────────────────
    # The XRDF cspace includes gantry_vagn_joint, so the gantry is a
    # planned joint rather than a base shift — no set_robot_base_pose
    # bookkeeping, and the planner can trade gantry travel against arm
    # reach to route around obstacles.
    try:
        if _init_cumotion():
            STATE.arm_names = list(
                STATE.cumotion_robot.controlled_joint_names)
            STATE.target_frame = CUMOTION_TOOL_FRAME
            STATE.ik_ready = True
            print(f"[OK] cuMotion ready — tool_frame='{CUMOTION_TOOL_FRAME}', "
                  f"joints={STATE.arm_names}")
            print("[..] Collision world is EMPTY until "
                  "/api/build_collision_world is called")
        else:
            STATE.ik_ready = False
            print("[WARN] cuMotion init failed — motion will use set_joints")
    except Exception as e:
        print(f"[WARN] cuMotion init failed: {e}")
        STATE.ik_ready = False

    # ── Subscribe to timeline events (stop/pause → shutdown) ─
    if _timeline_sub is not None:
        try:
            _timeline_sub.unsubscribe()
        except Exception:
            pass

    stream = timeline.get_timeline_event_stream()
    _timeline_sub = stream.create_subscription_to_pop(_on_timeline_event)
    print("[OK] Timeline subscription active — bridge will auto-stop on sim stop")

    # ── Contact Sensors ──────────────────────────────────────    
    from isaacsim.sensors.experimental.physics import ContactSensor
    if ContactSensor is None:
        STATE.contact_sensor = None
        STATE.contact_sensor_tip = None
        print("[WARN] ContactSensor class not found in either "
              "'isaacsim.sensors.contact_sensor' or "
              "'omni.isaac.sensor'. Descent will run blind and the "
              "adaptive gripper close will use the open-loop fallback.")
    else:
        print(f"[OK] ContactSensor class loaded")
        try:
            cs_pad = ContactSensor(CONTACT_SENSOR_PRIM)
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()
            STATE.contact_sensor = cs_pad
            print(f"[OK] Contact sensor (PAD) ready: {CONTACT_SENSOR_PRIM}")
        except Exception as e:
            STATE.contact_sensor = None
            print(f"[WARN] Contact sensor (PAD) init failed "
                  f"(preset gripper mode): {e}")

        try:
            cs_tip = ContactSensor(CONTACT_SENSOR_TIP_PRIM)
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()
            STATE.contact_sensor_tip = cs_tip
            print(f"[OK] Contact sensor (TIP) ready: {CONTACT_SENSOR_TIP_PRIM}")
        except Exception as e:
            STATE.contact_sensor_tip = None
            print(f"[WARN] Contact sensor (TIP) init failed — descent "
                  f"will run to planned target without floor-touch "
                  f"feedback: {e}")

    # ── HTTP Server ──────────────────────────────────────────
    _bridge_server = HTTPServer((BRIDGE_HOST, BRIDGE_PORT), BridgeHandler)
    _bridge_server.timeout = 1  # so shutdown() isn't blocked forever

    def serve():
        print(f"[OK] Bridge v2 on http://{BRIDGE_HOST}:{BRIDGE_PORT}")
        print(f"     Planner: {'cuMotion ({})'.format(STATE.target_frame) if STATE.ik_ready else 'DISABLED — ' + str(STATE.cumotion_error)}")
        print(f"     Cameras: RGB={CAMERA_RGB_PRIM}")
        print(f"              Depth={CAMERA_DEPTH_PRIM}")
        print(f"              Wrist={CAMERA_WRIST_PRIM}")
        print("=" * 60)
        _bridge_server.serve_forever()
        print("[OK] HTTP server thread exited")

    _bridge_server_thread = threading.Thread(target=serve, daemon=True)
    _bridge_server_thread.start()

    # ── Async command loop (exits when _bridge_running = False) ──
    _bridge_running = True
    await _process_commands_loop()


async def _process_commands_loop():
    """Command loop that exits cleanly when the bridge is shut down."""
    import omni.kit.app
    while _bridge_running:
        cmd = STATE.pop_command()
        if cmd is None:
            await omni.kit.app.get_app().next_update_async()
            continue

        action = cmd.get("action", "")
        try:
            if action == "camera_capture":
                result = await _capture_camera(cmd.get("camera", "rgb"))
            elif action == "move_home":
                result = await _move_home()
            elif action == "set_joints":
                result = await _set_joints(cmd["positions"])
            elif action == "gripper_open":
                result = await _set_gripper(FINGER_OPEN)
            elif action == "gripper_close":
                result = await _set_gripper(FINGER_CLOSE)
            elif action == "execute_plan":
                result = await _execute_plan(cmd["plan"])
            elif action == "scan_scene_parts":
                result = await _scan_scene_parts()
            elif action == "scene_annotations":
                result = await _get_scene_annotations(cmd.get("camera", "rgb"))
            elif action == "compute_prim_center":
                result = await _compute_prim_center(cmd.get("prim_path", ""))
            elif action == "plan_test":
                result = await _plan_test(cmd.get("params", {}))
            elif action == "jog":
                result = await _jog_joint(cmd.get("params", {}))
            elif action == "verify_planner":
                result = await _verify_planner(cmd.get("params", {}))
            elif action == "build_collision_world":
                result = await _build_collision_world(cmd.get("params", {}))
            elif action == "wrist_obstacles":
                result = await _add_wrist_obstacles(cmd.get("params", {}))
            elif action == "approach":
                result = await _approach_target(cmd.get("params", {}))
            elif action == "realign_gantry_hold_ee":
                result = await _realign_gantry_hold_ee(cmd.get("params", {}))
            elif action == "pick_object_ik":
                result = await _pick_object_ik(cmd.get("params", {}))
            elif action == "pick_descend":
                result = await _pick_descend(cmd.get("params", {}))
            elif action == "pick_close":
                result = await _pick_close(cmd.get("params", {}))
            elif action == "pick_retract":
                result = await _pick_retract(cmd.get("params", {}))
            elif action == "place_object_ik":
                result = await _place_object_ik(cmd.get("params", {}))
            elif action == "project_to_world":
                result = await _project_to_world(cmd.get("params", {}))
            else:
                result = {"error": f"Unknown action: {action}"}
        except Exception as e:
            result = {"error": str(e), "traceback": traceback.format_exc()}
            STATE.last_error = str(e)

        STATE.push_result(result)

    print("[OK] Command loop exited")


asyncio.ensure_future(start_bridge())
