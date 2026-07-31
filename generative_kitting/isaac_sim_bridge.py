"""
Isaac Sim ↔ Streamlit Bridge Server  (v2 — Neuro-Symbolic Kitting).

Run this script inside Isaac Sim's Script Editor.  It exposes an HTTP
API on port 8600 that the Streamlit dashboard consumes for:

  GET  /api/ping          → health check + capabilities list
  GET  /api/status        → robot joint positions, DOF info, sim state
  GET  /api/camera        → live camera frame (rgb|depth) as base64 JPEG
  GET  /api/scene_annotations → GT per-part 3D + 2D bbox seen by camera
  GET  /api/prim_center?prim=<path> → USD bbox centre/top/height for a prim
  POST /api/execute       → execute a list of action primitives
  POST /api/joints        → set joint positions directly
  POST /api/home          → move robot to home position
  POST /api/gripper       → open / close gripper
  POST /api/approach      → move near a world XYZ (Lula IK)
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
# CONFIGURATION (mirrors the constants in robot_control.py)
# ═════════════════════════════════════════════════════════════

# Bridge host/port come from env so the operator can pin the listener
# to loopback (default) or open it to the LAN deliberately. The HTTP
# API has no authentication; binding to 0.0.0.0 lets anyone on the
# subnet call /api/execute, so the safe default is 127.0.0.1.
BRIDGE_HOST = os.environ.get("BRIDGE_HOST", "127.0.0.1")
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", "8600"))
ROBOT_PRIM  = "/World"

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
UR10_BASE_PATH = "/World/gantry_home/ur10_flattened/ur10_instanceable/base_link"
EE_PATH        = "/World/gantry_home/ur10_flattened/ur10_instanceable/ee_link"
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
CONTACT_SENSOR_PRIM     = "/World/gantry_home/ur10_flattened/robotiq_fixed_physics/Robotiq_2F_140_physics_edit/left_inner_finger/Fingertip_01/Fingertip/Contact_Sensor"
CONTACT_SENSOR_TIP_PRIM = CONTACT_SENSOR_PRIM

# URDF / YAML for Lula IK. Resolved from the ISAACSIM_PATH env var so
# the bridge runs on any operator's machine (Windows / Linux / Mac)
# without editing this file. Set ISAACSIM_PATH to the root of your
# Isaac Sim install (the directory containing `exts/`). os.path.join
# keeps the separator OS-correct.
_ISAACSIM_PATH = os.environ.get("ISAACSIM_PATH", "")


def _resolve_ext_dir(ext_name):
    """Return the ext directory for ``ext_name``, checking both ``exts``
    and ``extsDeprecated``. Isaac Sim 6.0 moved several bundled exts (e.g.
    ``isaacsim.robot_motion.motion_generation``) into ``extsDeprecated``."""
    for _sub in ("exts", "extsDeprecated"):
        _cand = os.path.join(_ISAACSIM_PATH, _sub, ext_name)
        if os.path.isdir(_cand):
            return _cand
    return os.path.join(_ISAACSIM_PATH, "exts", ext_name)


URDF_PATH = os.path.join(
    _resolve_ext_dir("isaacsim.asset.importer.urdf"),
    "data", "urdf", "robots", "ur10", "urdf", "ur10.urdf")
YAML_PATH = os.path.join(
    _resolve_ext_dir("isaacsim.robot_motion.motion_generation"),
    "motion_policy_configs", "universal_robots", "ur10", "rmpflow",
    "ur10_robot_description.yaml")

# HOME_JOINTS is captured from the USD scene at startup (see _init_robot).
# The robot is ceiling-mounted on the gantry — the correct rest pose depends
# on how the scene was authored, not on a hardcoded floor-mounted guess.
HOME_JOINTS = None  # set by _init_robot()

# Gantry
GANTRY_X_JOINT  = "gantry_vagn_joint"
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


def ik_solve(lula_solver, frame, pos, ori, warm):
    action, ok = lula_solver.compute_inverse_kinematics(
        frame_name=frame, target_position=pos,
        target_orientation=ori, warm_start=warm)
    if ok:
        action = _clamp_shoulder_pan(action)
    return action, ok


# ── Shoulder-Pan Safety Clamp (asymmetric) ─────────────────
# The robot is ceiling-mounted on a gantry rail (along X). Lula
# IK has no collision awareness of the gantry structure, so it
# can find solutions where the arm swings THROUGH the red gantry
# rail. We constrain shoulder_pan (joint 0) with an ASYMMETRIC
# range: tight on the gantry-rail side, loose on the open side.
# This preserves enough workspace for far-corner picks (which
# need ~1.9 rad on the open side) while preventing rotation past
# the rail on the gantry side.
#
#   shoulder_pan = 0 rad → arm hangs straight down (home)
#   shoulder_pan > 0    → arm rotates one way
#   shoulder_pan < 0    → arm rotates the other way
#
# >>> WHICH SIGN IS THE GANTRY SIDE? <<<
# If the arm collides with the rail when reaching some target:
#   1. Look at the [IK SAFETY] log line on the run that collided.
#      If shoulder_pan was POSITIVE near +2.3562, the gantry is on
#      the + side → swap the signs of the two constants below.
#      If shoulder_pan was NEGATIVE near -2.3562, the gantry is on
#      the - side → leave as-is.
#   2. Re-run; the clamp now fires on the gantry side.
# A wrong guess locks IK on the safe side instead — easy to spot
# (every approach fails IK), trivial to flip.
#
# Initial guess: gantry on the NEGATIVE side. Tight = -π/2 (1.57 rad);
# loose = +3π/4 (2.36 rad). Last week's working config was symmetric
# ±π/2; the symmetric +3π/4 broke the guard. Asymmetric splits the
# difference: full reach on the open side, original guard on the
# gantry side.

SHOULDER_PAN_MIN = -1.5708   # tight: -π/2 (assumed gantry-side limit)
SHOULDER_PAN_MAX =  2.3562   # loose: +3π/4 (open-side limit)


def _clamp_shoulder_pan(ik_result):
    """Clamp shoulder_pan joint in IK result to prevent gantry collision.

    Logs whenever the clamp fires and records which side hit the limit
    so the operator can confirm which sign is the gantry side.
    """
    clamped = np.array(ik_result, dtype=np.float64)
    # Joint 0 in Lula's 6-joint result is shoulder_pan
    if len(clamped) > 0:
        original = clamped[0]
        clamped[0] = np.clip(clamped[0], SHOULDER_PAN_MIN, SHOULDER_PAN_MAX)
        if abs(original - clamped[0]) > 0.01:
            side = "MIN(gantry?)" if original < SHOULDER_PAN_MIN else "MAX(open?)"
            print(f"  [IK SAFETY] shoulder_pan clamped: {original:.3f} → "
                  f"{clamped[0]:.3f} rad  hit={side}  "
                  f"limits=[{SHOULDER_PAN_MIN:+.3f}, {SHOULDER_PAN_MAX:+.3f}]")
    return clamped


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

GRIPPER_TCP_LINK = """
  <link name="gripper_tcp">
    <inertial>
      <mass value="0.001"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
    <collision>
      <origin xyz="0 0 {half_tcp}" rpy="0 0 0"/>
      <geometry><box size="{width} {width} {tcp_offset}"/></geometry>
    </collision>
  </link>
  <joint name="gripper_tcp_joint" type="fixed">
    <parent link="ee_link"/>
    <child link="gripper_tcp"/>
    <origin xyz="0 0 {tcp_offset}" rpy="0 0 0"/>
  </joint>
""".format(tcp_offset=GRIPPER_TCP_OFFSET, half_tcp=GRIPPER_TCP_OFFSET/2, width=GRIPPER_BODY_WIDTH)
# NOTE on axis: in the standard UR10 URDF (Universal Robots'
# convention), ``ee_link`` has its tool axis along LOCAL +Z. Lula
# reads the URDF — not the USD — so the patched joint follows
# the URDF convention. The earlier USD diagnostic the operator
# ran reported the gripper_tcp Xform at LOCAL +X 219.5 mm of
# ee_link, but that's the USD-side view: Isaac Sim's URDF
# importer rotates the link frame so that URDF's +Z aligns with
# USD's +X. The two views describe the same physical position;
# the URDF patch must use URDF coords, hence ``xyz="0 0 0.220"``
# (along URDF +Z). We tried placing the joint along +X earlier
# to match the USD measurement directly — that broke IK because
# Lula was then planning with a virtual gripper rotated 90° from
# the actual physical scene, sending ee_link off-target by the
# gripper's full length.


def patch_urdf_and_yaml():
    temp_dir = tempfile.gettempdir()
    fixed_urdf = os.path.join(temp_dir, "fixed_ur10_with_gripper.urdf")
    fixed_yaml = os.path.join(temp_dir, "fixed_ur10_with_gripper.yaml")

    with open(URDF_PATH, 'r') as f:
        urdf = f.read()
    urdf = urdf.replace("</inertial>",
        '<inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/></inertial>')
    urdf = urdf.replace("0.0027000046)", "0.0027000046")
    if "gripper_tcp" not in urdf:
        urdf = urdf.replace("</robot>", GRIPPER_TCP_LINK + "\n</robot>")
    with open(fixed_urdf, 'w') as f:
        f.write(urdf)

    with open(YAML_PATH, 'r') as f:
        yaml_txt = f.read()
    yaml_txt = yaml_txt.replace("root_link: world", "root_link: base_link")
    if "gripper_tcp_joint" not in yaml_txt:
        yaml_txt = yaml_txt.rstrip() + "\nee_fixed_joints:\n  - gripper_tcp_joint\n"
    with open(fixed_yaml, 'w') as f:
        f.write(yaml_txt)

    return fixed_urdf, fixed_yaml


# ═════════════════════════════════════════════════════════════
# SHARED STATE
# ═════════════════════════════════════════════════════════════

class BridgeState:
    def __init__(self):
        self.robot = None
        self.world = None
        self.camera_rgb = None
        self.camera_depth = None
        self.camera_wrist = None
        self.camera_kit = None
        self.lula_solver = None
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
        """Move near a target XYZ using Lula IK."""
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
# CAMERA CAPTURE (with black-frame fix)
# ═════════════════════════════════════════════════════════════

async def _capture_camera(cam_type="rgb"):
    try:
        from isaacsim.sensors.camera import Camera
        import omni.kit.app

        if cam_type == "depth":
            if STATE.camera_depth is None:
                STATE.camera_depth = Camera(prim_path=CAMERA_DEPTH_PRIM, resolution=(1280, 720))
                STATE.camera_depth.initialize()
                for _ in range(10):
                    await omni.kit.app.get_app().next_update_async()
            cam = STATE.camera_depth
        elif cam_type == "wrist":
            if STATE.camera_wrist is None:
                STATE.camera_wrist = Camera(prim_path=CAMERA_WRIST_PRIM, resolution=(1280, 720))
                STATE.camera_wrist.initialize()
                for _ in range(10):
                    await omni.kit.app.get_app().next_update_async()
            cam = STATE.camera_wrist
        elif cam_type == "kit":
            if STATE.camera_kit is None:
                STATE.camera_kit = Camera(prim_path=CAMERA_KIT_PRIM, resolution=(1280, 720))
                STATE.camera_kit.initialize()
                for _ in range(10):
                    await omni.kit.app.get_app().next_update_async()
            cam = STATE.camera_kit
        else:
            if STATE.camera_rgb is None:
                STATE.camera_rgb = Camera(prim_path=CAMERA_RGB_PRIM, resolution=(1920, 1080))
                STATE.camera_rgb.initialize()
                for _ in range(10):
                    await omni.kit.app.get_app().next_update_async()
            cam = STATE.camera_rgb

        # ── Black-frame fix ──────────────────────────────────
        # The wrist RGB renderer is slower to refresh after the robot
        # moves than the depth annotator at the same prim hierarchy
        # (different render paths). Give it more ticks + more retries
        # to avoid the workflow seeing a stale black frame.
        if cam_type == "wrist":
            max_attempts, ticks_per_attempt = 8, 8
        else:
            max_attempts, ticks_per_attempt = 3, 3

        rgba = None
        last_mean = -1.0
        for attempt in range(max_attempts):
            cam.get_current_frame()
            for _ in range(ticks_per_attempt):
                await omni.kit.app.get_app().next_update_async()

            rgba = cam.get_rgba()
            if rgba is not None and rgba.size > 0:
                last_mean = float(np.mean(rgba[:, :, :3]))
                # Frame is "valid" if it has non-trivial brightness
                if last_mean > 3:
                    if attempt > 0:
                        print(f"  [{cam_type}] valid frame on attempt "
                              f"{attempt + 1}/{max_attempts} "
                              f"(mean={last_mean:.1f})")
                    break

        if rgba is None or rgba.size == 0:
            return {"error": f"{cam_type} camera returned empty frame"}

        if last_mean <= 3:
            print(f"  [{cam_type}] WARNING: frame still dark after "
                  f"{max_attempts} attempts (mean={last_mean:.1f}) — "
                  f"renderer may not have caught up")

        from PIL import Image
        img = Image.fromarray(rgba[:, :, :3])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {
            "image_base64": b64, "width": img.width, "height": img.height,
            "format": "jpeg", "camera": cam_type,
        }

    except Exception as e:
        return {"error": f"{cam_type} camera failed: {e}"}


# ═════════════════════════════════════════════════════════════
# DEPTH-BASED WORLD PROJECTION
#
#   Converts VLM normalised image coordinates (0-1) into real-world
#   XYZ by reading the depth buffer from the same camera that took
#   the RGB frame.  This replaces USD prim-path lookups so that
#   coordinates come purely from vision.
#
#   Pipeline:  VLM (nx,ny) → pixel (px,py) → depth d →
#              camera intrinsics → camera-frame 3D →
#              camera extrinsics → world-frame XYZ
# ═════════════════════════════════════════════════════════════

def _get_workspace_surface_z():
    """Get Z height of workspace surface where parts sit (geometric
    projection fallback).

    Reads the bin AABB and returns ``bot_z + 0.05`` — i.e. just above
    the rack base, which is approximately where parts rest on the
    internal compartment floor.

    Earlier the function used the bin prim's PIVOT translation Z,
    which is often 0 (e.g. when the bin's parent transform places it
    at the world origin). That made every geometric projection emit
    ``z = 0`` and the IK targeted points below the floor.
    """
    import omni.usd
    try:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(PARTS_CONTAINER)
        if prim.IsValid():
            bbox = compute_bbox_geometry(stage, PARTS_CONTAINER,
                                         label="WORKSPACE_Z")
            surface_z = float(bbox["bot_z"]) + 0.05
            print(f"  [workspace_z] Bin AABB bot_z={bbox['bot_z']:.4f}, "
                  f"top_z={bbox['top_z']:.4f} → using surface_z="
                  f"{surface_z:.4f}")
            return surface_z
    except Exception as e:
        print(f"  [workspace_z] USD read failed: {e}")
    return 0.02  # last-resort default (table-level)


async def _project_to_world(params):
    """Project normalised image coordinates to world XYZ.

    Two methods (automatic fallback):
      1. **Depth buffer** — per-pixel depth from Isaac Sim camera.
         Most accurate; gives true Z for each object.
      2. **Geometric ray-plane** — intersect camera ray with the
         workspace surface plane (Z = bin surface).  No depth buffer
         needed; assumes objects sit on a flat surface.

    Args (in params dict):
        points: list of {x, y} with values in [0, 1].
                Each point may include an optional "surface_z" to override
                the default workspace Z for geometric fallback (useful for
                objects on different surfaces, e.g. kitting tray vs. bin).
        camera: prim path or alias ("rgb", "depth") — default "rgb"

    Returns:
        {status, world_points: [{x, y, z, depth_m, method}, ...]}
    """
    import omni.kit.app, omni.usd
    from isaacsim.sensors.camera import Camera
    from pxr import UsdGeom, Gf
    import math

    points = params.get("points", [])
    cam_alias = params.get("camera", "rgb")
    force_method = params.get("method", None)  # "geometric" to skip depth buffer
    # Optional axis-sign overrides — useful when the camera was placed
    # in USD with a non-standard rotation that mirrors detected coords.
    img_x_sign = float(params.get("image_x_sign", 1))
    img_y_sign = float(params.get("image_y_sign", 1))
    if img_x_sign != 1 or img_y_sign != 1:
        print(f"  [project] Axis sign overrides: "
              f"x={img_x_sign:+.0f}, y={img_y_sign:+.0f}")

    # Resolve alias → prim path
    if cam_alias == "rgb":
        cam_path = CAMERA_RGB_PRIM
        res = (1920, 1080)
    elif cam_alias == "depth":
        cam_path = CAMERA_DEPTH_PRIM
        res = (1280, 720)
    elif cam_alias == "wrist":
        cam_path = CAMERA_WRIST_PRIM
        res = (1280, 720)
    elif cam_alias == "kit":
        cam_path = CAMERA_KIT_PRIM
        res = (1280, 720)
    else:
        cam_path = cam_alias
        res = (1280, 720)

    # ── Initialise camera ──────────────────────────────────────
    if cam_alias == "rgb" and STATE.camera_rgb is not None:
        cam = STATE.camera_rgb
    elif cam_alias == "depth" and STATE.camera_depth is not None:
        cam = STATE.camera_depth
    elif cam_alias == "wrist" and STATE.camera_wrist is not None:
        cam = STATE.camera_wrist
    elif cam_alias == "kit" and STATE.camera_kit is not None:
        cam = STATE.camera_kit
    else:
        cam = Camera(prim_path=cam_path, resolution=res)
        cam.initialize()
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()

    # Attach depth annotator if not already present — required for get_depth()
    try:
        cam.add_distance_to_image_plane_to_frame()
        print(f"  [project] Depth annotator attached to {cam_path}")
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
    except Exception:
        pass  # may already be attached or unsupported

    # ── Camera intrinsics ──────────────────────────────────────
    # Pinhole model from USD camera attributes:
    #     fx = w_res * focal_length / horizontal_aperture
    #     fy = h_res * focal_length / vertical_aperture
    #     cx, cy = principal point (image centre for an unshifted lens)
    # Reading focal_length + aperture directly from the USD prim is
    # more reliable than `cam.get_intrinsics_matrix()` which can be
    # stale or unset after camera repositioning. We fall back to the
    # API method, then to a 60° HFOV estimate.
    w_res, h_res = res
    fx = fy = cx = cy = None
    intr_source = "?"
    try:
        from pxr import UsdGeom
        intrinsics_stage = omni.usd.get_context().get_stage()
        usd_cam = UsdGeom.Camera(intrinsics_stage.GetPrimAtPath(cam_path))
        if usd_cam:
            focal_mm = float(usd_cam.GetFocalLengthAttr().Get() or 0.0)
            h_aperture = float(
                usd_cam.GetHorizontalApertureAttr().Get() or 0.0)
            v_aperture = float(
                usd_cam.GetVerticalApertureAttr().Get() or 0.0)
            focus_distance = None
            try:
                focus_distance = float(
                    usd_cam.GetFocusDistanceAttr().Get() or 0.0)
            except Exception:
                pass
            if focal_mm > 0 and h_aperture > 0 and v_aperture > 0:
                fx = w_res * focal_mm / h_aperture
                fy = h_res * focal_mm / v_aperture
                cx, cy = w_res / 2.0, h_res / 2.0
                intr_source = (
                    f"USD attrs (focal={focal_mm:.2f}mm, "
                    f"H={h_aperture:.2f}mm, V={v_aperture:.2f}mm"
                    f"{', focus=' + format(focus_distance, '.2f') + 'm' if focus_distance else ''})")
    except Exception as e:
        print(f"  [project] USD camera attr read failed: {e}")

    if fx is None:
        try:
            intrinsics = cam.get_intrinsics_matrix()
            fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
            cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
            intr_source = "Camera.get_intrinsics_matrix()"
        except Exception:
            hfov_rad = math.radians(60)
            fx = fy = (w_res / 2.0) / math.tan(hfov_rad / 2.0)
            cx, cy = w_res / 2.0, h_res / 2.0
            intr_source = "60° HFOV estimate"
    print(f"  [project] Intrinsics: fx={fx:.1f} fy={fy:.1f} "
          f"cx={cx:.1f} cy={cy:.1f}  ← {intr_source}")

    # ── Camera extrinsics (world transform) ────────────────────
    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath(cam_path)
    cam_world_mat = omni.usd.get_world_transform_matrix(cam_prim)

    cam_pos = cam_world_mat.GetRow(3)
    # Camera's local axes mapped to world frame — diagnoses sign /
    # axis-flip issues when projection puts parts on the wrong side.
    # In OpenGL convention the camera looks down its -Z, +X is image
    # right, +Y is image up. After applying the camera's USD world
    # transform we expect:
    #   right_world  ≈ horizontal (any direction in the world XY plane)
    #   up_world     ≈ pointing roughly opposite to the look direction's
    #                  vertical drop (image-up = world-up if camera
    #                  is roll-free)
    #   fwd_world    ≈ from camera toward what it's pointed at
    right_world = cam_world_mat.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    up_world    = cam_world_mat.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))
    fwd_world   = cam_world_mat.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
    print(f"  [project] Camera world pos: ({cam_pos[0]:.3f}, "
          f"{cam_pos[1]:.3f}, {cam_pos[2]:.3f})")
    print(f"  [project] Camera basis in world:")
    print(f"             right (+X_cam) = ({right_world[0]:+.3f}, "
          f"{right_world[1]:+.3f}, {right_world[2]:+.3f})")
    print(f"             up    (+Y_cam) = ({up_world[0]:+.3f}, "
          f"{up_world[1]:+.3f}, {up_world[2]:+.3f})")
    print(f"             fwd   (-Z_cam) = ({fwd_world[0]:+.3f}, "
          f"{fwd_world[1]:+.3f}, {fwd_world[2]:+.3f})")

    # ── Try depth buffer ───────────────────────────────────────
    has_depth = False
    depth_buf = None

    # Warm up camera for depth
    for _ in range(8):
        cam.get_current_frame()
        await omni.kit.app.get_app().next_update_async()

    try:
        depth_buf = cam.get_depth()
        if depth_buf is not None and depth_buf.size > 0:
            valid_mask = (depth_buf > 0.01) & (depth_buf < 100.0)
            valid_count = int(valid_mask.sum())
            total_pixels = depth_buf.size
            print(f"  [project] Depth buffer: {depth_buf.shape}, "
                  f"{valid_count}/{total_pixels} valid pixels "
                  f"({100*valid_count/total_pixels:.1f}%), "
                  f"range=[{depth_buf[valid_mask].min():.3f}, {depth_buf[valid_mask].max():.3f}]"
                  if valid_count > 0 else f"  [project] Depth buffer: {depth_buf.shape}, 0 valid pixels")
            if valid_count > total_pixels * 0.01:  # at least 1% valid
                has_depth = True
                h_depth, w_depth = depth_buf.shape[:2]
            else:
                print(f"  [project] WARNING: depth buffer has too few valid pixels — using geometric fallback")
        else:
            print(f"  [project] Depth buffer empty or None")
    except Exception as e:
        print(f"  [project] Depth buffer read failed: {e}")

    # ── Workspace surface Z for geometric fallback ─────────────
    workspace_z = _get_workspace_surface_z()

    if force_method == "geometric":
        method_used = "geometric (forced)"
    else:
        method_used = "depth" if has_depth else "geometric"
    print(f"  [project] Method: {method_used} | {len(points)} points to project")

    # ── Inverse projection: world point → image pixel ──────────
    # Uses the SAME camera intrinsics + extrinsics we already read
    # above (fx, fy, cx, cy, cam_world_mat, img_*_sign). No external
    # references — purely the camera prim's own data. Used for the
    # round-trip self-consistency log below.
    def _world_to_pixel(wx, wy, wz):
        # World → camera frame: invert the camera world transform.
        cam_inv = cam_world_mat.GetInverse()
        cam_pt = cam_inv.Transform(Gf.Vec3d(float(wx), float(wy), float(wz)))
        cx_f, cy_f, cz_f = cam_pt[0], cam_pt[1], cam_pt[2]
        # OpenGL convention: camera looks down -Z, points in front have
        # cz_f < 0. Distance from camera = -cz_f.
        if cz_f >= -1e-6:
            return None  # behind the camera or on the lens plane
        # Pinhole projection (inverse of the unprojection formulas above).
        # Solve cam_x = sign_x * (px - cx) * d / fx for px, with d = -cz_f.
        d = -cz_f
        px_f = (cx_f * fx) / (img_x_sign * d) + cx
        py_f = -(cy_f * fy) / (img_y_sign * d) + cy
        return (px_f, py_f)

    # ── Project each point ─────────────────────────────────────
    world_points = []
    for i, pt in enumerate(points):
        nx = float(pt.get("x", 0.5))
        ny = float(pt.get("y", 0.5))

        point_method = None

        # ── Method 1: Depth buffer projection ──────────────────
        if has_depth and force_method != "geometric":
            px = min(int(nx * (w_depth - 1)), w_depth - 1)
            py = min(int(ny * (h_depth - 1)), h_depth - 1)

            # Sample depth in 5×5 neighbourhood (robust to noise/edges)
            r = 2
            y0, y1 = max(0, py - r), min(h_depth, py + r + 1)
            x0, x1 = max(0, px - r), min(w_depth, px + r + 1)
            patch = depth_buf[y0:y1, x0:x1]
            valid = patch[(patch > 0.01) & (patch < 100.0)]

            if valid.size > 0:
                d = float(np.median(valid))

                # Unproject pixel → camera frame (OpenGL: +X right, +Y up, -Z forward)
                cam_x = img_x_sign * (px - cx) * d / fx
                cam_y = img_y_sign * -(py - cy) * d / fy
                cam_z = -d

                # Transform camera-frame point to world using USD canonical method
                # (GetTranspose() * Vec4d is ambiguous in pxr bindings — Transform() is correct)
                world_pt = cam_world_mat.Transform(Gf.Vec3d(cam_x, cam_y, cam_z))

                world_points.append({
                    "x": float(world_pt[0]),
                    "y": float(world_pt[1]),
                    "z": float(world_pt[2]),
                    "depth_m": d,
                    "method": "depth",
                })
                point_method = "depth"

        # ── Method 2: Geometric ray-plane intersection ─────────
        if point_method is None:
            px = min(int(nx * (w_res - 1)), w_res - 1)
            py = min(int(ny * (h_res - 1)), h_res - 1)

            # Per-point surface Z override (e.g. tray on different surface)
            point_z = float(pt.get("surface_z", workspace_z))

            # Ray direction in camera frame → world frame
            # TransformDir() transforms directions (ignores translation),
            # unlike Transform() which transforms points.
            ray_world = cam_world_mat.TransformDir(Gf.Vec3d(
                img_x_sign * (px - cx) / fx,
                img_y_sign * -(py - cy) / fy,
                -1.0,
            ))

            dz = ray_world[2]
            if abs(dz) < 1e-6:
                world_points.append({
                    "x": 0, "y": 0, "z": point_z, "depth_m": -1,
                    "error": "ray parallel to plane", "method": "geometric",
                })
                continue

            # Intersect ray with Z = point_z plane
            t = (point_z - cam_pos[2]) / dz
            world_x = cam_pos[0] + t * ray_world[0]
            world_y = cam_pos[1] + t * ray_world[1]

            world_points.append({
                "x": float(world_x),
                "y": float(world_y),
                "z": float(point_z),
                "depth_m": float(abs(cam_pos[2] - point_z)),
                "method": "geometric",
            })
            point_method = "geometric"

        wp = world_points[-1]
        # ── Round-trip self-consistency check ──
        # Project the resulting world point BACK through the camera to
        # a pixel using the same intrinsics + extrinsics. If the result
        # matches the input pixel within ~1-2 px, the camera math is
        # internally consistent (any remaining error is downstream:
        # bbox-centring / depth-sample location). If it doesn't match,
        # the camera math itself is broken (most often an axis-sign
        # mismatch in camera_image_x_sign / camera_image_y_sign).
        # NOTE the input pixel uses w_res×h_res (RGB resolution) for
        # the geometric branch and w_depth×h_depth for the depth branch
        # — we compare against whichever resolution this point was
        # projected from.
        if point_method == "depth":
            in_w, in_h = w_depth, h_depth
        else:
            in_w, in_h = w_res, h_res
        in_px = nx * (in_w - 1)
        in_py = ny * (in_h - 1)
        rt = _world_to_pixel(wp.get("x", 0), wp.get("y", 0), wp.get("z", 0))
        if rt is None:
            rt_str = "BEHIND-CAMERA"
        else:
            rt_px, rt_py = rt
            dpx = rt_px - in_px
            dpy = rt_py - in_py
            err = (dpx * dpx + dpy * dpy) ** 0.5
            tag = "OK" if err < 2.0 else ("DRIFT" if err < 10.0 else "BROKEN")
            rt_str = (f"round-trip pixel=({rt_px:.1f},{rt_py:.1f}) "
                      f"err=({dpx:+.1f},{dpy:+.1f}) |err|={err:.1f}px [{tag}]")
        print(f"  [{i}] image({nx:.2f},{ny:.2f})→({in_px:.0f},{in_py:.0f}) "
              f"→ world({wp['x']:.3f}, {wp['y']:.3f}, {wp['z']:.3f}) "
              f"[{point_method}]  {rt_str}")

    # Augment the response with the actual world positions of the
    # camera that took the image AND the robot's ee_link, so the
    # caller can see the geometry that backed the projection.
    # The wrist RealSense has TWO sensor prims (RGB + pseudo-depth)
    # mounted with a small physical offset; if the caller wants to
    # verify how their assumed-aligned coords actually relate to
    # ee_link, they need both numbers.
    extra: dict = {}
    try:
        ee_pos = get_world_pos(EE_PATH)
        extra["ee_link_pos"] = [float(ee_pos[0]),
                                float(ee_pos[1]),
                                float(ee_pos[2])]
    except Exception:
        pass
    try:
        extra["camera_pos"] = [float(cam_pos[0]),
                                float(cam_pos[1]),
                                float(cam_pos[2])]
    except Exception:
        pass
    # When projecting from the wrist colour camera, ALSO expose the
    # depth camera's world transform — they are sibling prims on the
    # RSD455 mount with different translations, and any caller doing
    # depth alignment math needs both.
    if cam_alias == "wrist":
        try:
            stage_d = omni.usd.get_context().get_stage()
            d_prim = stage_d.GetPrimAtPath(CAMERA_DEPTH_PRIM)
            if d_prim and d_prim.IsValid():
                d_mat = omni.usd.get_world_transform_matrix(d_prim)
                d_pos = d_mat.GetRow(3)
                extra["wrist_depth_camera_pos"] = [
                    float(d_pos[0]), float(d_pos[1]), float(d_pos[2])]
        except Exception:
            pass
    return {"status": "ok", "world_points": world_points,
            "camera": cam_path, "method": method_used,
            "depth_buffer_used": has_depth,
            **extra}


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

    The workflow uses the world coords as ground truth and the 2D
    bbox to match VLM / OWL-ViT2 detections to the right entry.
    """
    import omni.usd
    from pxr import Gf
    from isaacsim.sensors.camera import Camera
    import omni.kit.app

    # Map camera alias → prim path + resolution (mirror _capture_camera)
    if camera == "rgb":
        cam_path, res = CAMERA_RGB_PRIM, (1920, 1080)
    elif camera == "depth":
        cam_path, res = CAMERA_DEPTH_PRIM, (1280, 720)
    elif camera == "wrist":
        cam_path, res = CAMERA_WRIST_PRIM, (1280, 720)
    elif camera == "kit":
        cam_path, res = CAMERA_KIT_PRIM, (1280, 720)
    else:
        cam_path, res = camera, (1280, 720)

    # Reuse cached cameras to avoid re-init + black frame on first call
    if camera == "rgb" and STATE.camera_rgb is not None:
        cam = STATE.camera_rgb
    elif camera == "depth" and STATE.camera_depth is not None:
        cam = STATE.camera_depth
    elif camera == "wrist" and STATE.camera_wrist is not None:
        cam = STATE.camera_wrist
    elif camera == "kit" and STATE.camera_kit is not None:
        cam = STATE.camera_kit
    else:
        cam = Camera(prim_path=cam_path, resolution=res)
        cam.initialize()
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()

    # Make sure the depth annotator is attached for visibility check
    try:
        cam.add_distance_to_image_plane_to_frame()
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()
    except Exception:
        pass

    # Intrinsics
    w_res, h_res = res
    try:
        K = cam.get_intrinsics_matrix()
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
    except Exception:
        import math
        hfov = math.radians(60)
        fx = fy = (w_res / 2.0) / math.tan(hfov / 2.0)
        cx, cy = w_res / 2.0, h_res / 2.0

    # Extrinsics: cam→world matrix and its inverse (world→cam)
    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath(cam_path)
    cam_world_mat = omni.usd.get_world_transform_matrix(cam_prim)
    world_to_cam = cam_world_mat.GetInverse()

    # Optional depth buffer for visibility check (best-effort)
    depth_buf = None
    try:
        for _ in range(5):
            cam.get_current_frame()
            await omni.kit.app.get_app().next_update_async()
        d = cam.get_depth()
        if d is not None and d.size > 0:
            depth_buf = d
    except Exception:
        depth_buf = None

    # Pull GT parts from existing scanner
    scan = await _scan_scene_parts()
    if "error" in scan:
        return {"error": scan["error"], "annotations": []}
    parts = scan.get("parts", [])

    def _project_world_to_pixel(wx, wy, wz):
        """world XYZ → (px, py, depth_to_cam_plane) or None if behind cam."""
        cam_pt = world_to_cam.Transform(Gf.Vec3d(float(wx), float(wy), float(wz)))
        cx_, cy_, cz_ = float(cam_pt[0]), float(cam_pt[1]), float(cam_pt[2])
        # OpenGL camera: forward = -Z. Point in front of camera ⇒ cz_ < 0.
        if cz_ >= -1e-3:
            return None
        d = -cz_
        px = cx_ * fx / d + cx
        py = -cy_ * fy / d + cy
        return px, py, d

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
            ipx = int(min(max(cpx, 0), w_res - 1) * (w_d - 1) / max(w_res - 1, 1))
            ipy = int(min(max(cpy, 0), h_res - 1) * (h_d - 1) / max(h_res - 1, 1))
            r = 2
            patch = depth_buf[max(0, ipy - r):ipy + r + 1,
                              max(0, ipx - r):ipx + r + 1]
            valid = patch[(patch > 0.01) & (patch < 100.0)]
            if valid.size > 0:
                measured_d = float(np.median(valid))
                # If measured depth is much closer than the part's projected
                # depth, something is in front of it → occluded
                if measured_d < c_depth - 0.05:
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
        ee_pos = get_world_pos(EE_PATH)
        safe_z = ee_pos[2] + TRANSIT_SAFE_HEIGHT
        locked_ori = normalize_quat(DOWNWARD_ORIENTATION)
        warm = _get_warm_start()
        action, ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                              np.array([ee_pos[0], ee_pos[1], safe_z]),
                              locked_ori, warm)
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
    """Build a Lula-compatible warm-start from current joint positions."""
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
    """Update Lula solver with current arm base transform."""
    import omni.usd
    from isaacsim.core.utils.prims import get_prim_at_path
    base_prim = get_prim_at_path(UR10_BASE_PATH)
    base_matrix = omni.usd.get_world_transform_matrix(base_prim)
    base_pos = np.array(base_matrix.ExtractTranslation())
    rot = base_matrix.ExtractRotation().GetQuat()
    base_rot = np.array([rot.real, rot.imaginary[0], rot.imaginary[1], rot.imaginary[2]])
    STATE.lula_solver.set_robot_base_pose(base_pos, base_rot)
    return base_pos


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
    """Move end-effector to XYZ using Lula IK with smooth streaming motion."""
    if not STATE.ik_ready:
        return {"error": "IK solver not initialised"}

    target_pos = np.array(target_xyz, dtype=np.float64)
    ori = orientation if orientation is not None else normalize_quat(DOWNWARD_ORIENTATION)

    warm = _get_warm_start()
    action, ok = ik_solve(STATE.lula_solver, STATE.target_frame, target_pos, ori, warm)
    if not ok:
        return {"error": f"IK failed for target {target_xyz}"}

    target_joints = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, action)
    await _apply_interpolated(target_joints, settle_frames=settle_frames)

    ee_pos = get_world_pos(EE_PATH)
    return {"status": "ok", "target": list(target_xyz), "ee_position": ee_pos.tolist()}


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
        action, ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                              step_pos, target_ori, warm)
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
    final_action, final_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                       target_pos, target_ori, warm)
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

        action, ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                              step_pos, locked_ori, warm)
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
                lift_action, lift_ok = ik_solve(
                    STATE.lula_solver, STATE.target_frame,
                    lift_pos, locked_ori, warm)
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

        action, ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                              step_pos, locked_ori, warm)
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
    final_action, final_ok = ik_solve(
        STATE.lula_solver, STATE.target_frame,
        final_pos, locked_ori, warm)
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

        action, ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                               step_pos, locked_ori, warm)
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
    final_action, final_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                       final_pos, locked_ori, warm)
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

    target_action, ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                  safe_pos, locked_ori, grasp_warm)
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
        import omni.usd
        from isaacsim.core.utils.prims import get_prim_at_path

        ee_lock = np.array(hold_ee_pos, dtype=np.float64)
        locked_ori = normalize_quat(DOWNWARD_ORIENTATION)
        n_steps = max(4, int(abs(gantry_delta) / 0.02))  # ~0.02m per step

        # Read initial base pose (position + rotation) once — rotation
        # doesn't change during a prismatic slide, only X translates.
        base_prim = get_prim_at_path(UR10_BASE_PATH)
        base_matrix = omni.usd.get_world_transform_matrix(base_prim)
        initial_base_pos = np.array(base_matrix.ExtractTranslation())
        rot = base_matrix.ExtractRotation().GetQuat()
        base_rot = np.array([rot.real, rot.imaginary[0],
                             rot.imaginary[1], rot.imaginary[2]])

        print(f"  [GANTRY] EE-hold slide: {gantry_current:.3f} → {gantry_target:.3f} "
              f"({n_steps} steps), holding EE at "
              f"({ee_lock[0]:.3f}, {ee_lock[1]:.3f}, {ee_lock[2]:.3f})")

        for step in range(1, n_steps + 1):
            frac = step / n_steps
            intermediate = gantry_current + frac * gantry_delta
            delta_from_start = intermediate - gantry_current

            # 1. Predict where base_link WILL be after gantry moves
            predicted_base = initial_base_pos.copy()
            predicted_base[0] += delta_from_start  # prismatic along X

            # 2. Tell Lula the predicted base pose
            STATE.lula_solver.set_robot_base_pose(predicted_base, base_rot)

            # 3. Solve IK for locked EE X,Y — allow Z to flex slightly
            #    so the solver has more room for valid configurations
            warm = _get_warm_start()
            action, ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                  ee_lock, locked_ori, warm)

            # 4. Apply gantry + arm joints SIMULTANEOUSLY — no drift gap
            if ok:
                targets = apply_arm_joints(
                    STATE.robot, STATE.dof_names, STATE.arm_names, action)
            else:
                targets = STATE.robot.get_joint_positions()
            targets[gantry_idx] = intermediate
            if finger_hold is not None:
                targets = set_finger_joints(
                    STATE.robot, STATE.dof_names, finger_hold, targets)
            STATE.robot.apply_action(ArticulationAction(joint_positions=targets))

            # 5. Step physics — both gantry + arm move together
            for _ in range(4):
                await omni.kit.app.get_app().next_update_async()

        # Final settle
        for _ in range(SETTLE_FRAMES):
            await omni.kit.app.get_app().next_update_async()

        # Sync Lula base pose with actual USD state after the full slide
        _update_base_pose()
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

    _update_base_pose()
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
        return {"error": "IK solver not initialised"}

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
        return {"error": "IK solver not initialised"}

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
            correct_action, correct_ok = ik_solve(
                STATE.lula_solver, STATE.target_frame,
                correct_pos, locked_ori, correct_warm)
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

        p1_action, p1_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                     safe_pos, locked_ori, warm)
        if not p1_ok:
            return {"error": f"IK safe failed {safe_pos.tolist()}", "steps": steps}
        targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, p1_action)
        await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES)
        steps.append({"phase": "safe_height", "status": "ok"})

        # 4. Re-solve seed
        settled_warm = _get_warm_start()
        p2_action, p2_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                     safe_pos, locked_ori, settled_warm)
        if not p2_ok:
            p2_action = p1_action

        # 5. Entry — just inside box rim
        entry_action, entry_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                           entry_pos, locked_ori, p2_action)
        if entry_ok:
            targets = apply_arm_joints(
                STATE.robot, STATE.dof_names, STATE.arm_names, entry_action)
            await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES)
            steps.append({"phase": "entry", "status": "ok"})
            p2_action = entry_action

        # 6. Hover — above part
        hover_action, hover_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                           hover_pos, locked_ori, p2_action)
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
            repose_action, repose_ok = ik_solve(
                STATE.lula_solver, STATE.target_frame,
                repose_target, locked_ori, repose_warm)
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
            snap_action, snap_ok = ik_solve(
                STATE.lula_solver, STATE.target_frame,
                snap_pos, locked_ori, snap_warm)
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
        return {"error": "IK solver not initialised"}

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
        retract_action, retract_ok = ik_solve(
            STATE.lula_solver, STATE.target_frame,
            safe_pos, locked_ori, retract_warm)
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
        transit_action, transit_ok = ik_solve(
            STATE.lula_solver, STATE.target_frame,
            transit_pos, locked_ori, transit_warm)
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
        return {"error": "IK solver not initialised"}

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
        transit_up_action, transit_up_ok = ik_solve(
            STATE.lula_solver, STATE.target_frame,
            transit_up_pos, locked_ori, transit_up_warm)
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
            reset_action, reset_ok = ik_solve(
                STATE.lula_solver, STATE.target_frame,
                reset_pos, locked_ori, reset_warm)
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
            snap_action, snap_ok = ik_solve(
                STATE.lula_solver, STATE.target_frame,
                snap_pos, locked_ori, snap_warm)
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
        retract_action, retract_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                               retract_pos, locked_ori, retract_warm)
        if retract_ok:
            targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, retract_action)
            await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES)
        steps.append({"phase": "retract", "status": "ok"})

        # 10. Transit safe height after retract (clear kitting tray before going home)
        retract_transit_warm = _get_warm_start()
        retract_transit_z = retract_z + TRANSIT_SAFE_HEIGHT
        retract_transit_pos = np.array([dest_x, dest_y, retract_transit_z])
        rt_action, rt_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                     retract_transit_pos, locked_ori, retract_transit_warm)
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
    STATE.camera_rgb = None
    STATE.camera_depth = None
    STATE.camera_wrist = None
    STATE.lula_solver = None

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

    # ── Lula IK Solver ───────────────────────────────────────
    try:
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

        fixed_urdf, fixed_yaml = patch_urdf_and_yaml()
        print(f"[OK] URDF patched: {fixed_urdf}")

        lula_solver = LulaKinematicsSolver(
            robot_description_path=fixed_yaml, urdf_path=fixed_urdf)

        # Determine target frame
        target_frame = "ee_link"
        try:
            test_pos = np.array([0.0, -0.5, 1.2])
            test_ori = normalize_quat(DOWNWARD_ORIENTATION)
            _, test_ok = lula_solver.compute_inverse_kinematics(
                frame_name="gripper_tcp", target_position=test_pos,
                target_orientation=test_ori, warm_start=np.zeros(6))
            if test_ok:
                target_frame = "gripper_tcp"
        except Exception:
            pass

        STATE.lula_solver = lula_solver
        STATE.target_frame = target_frame
        STATE.arm_names = lula_solver.get_joint_names()
        STATE.ik_ready = True

        # Set initial base pose
        _update_base_pose()

        print(f"[OK] Lula IK ready — frame='{target_frame}', joints={STATE.arm_names}")
    except Exception as e:
        print(f"[WARN] Lula IK init failed (motion will use set_joints): {e}")
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
    # Two independent sensors — one on the inner finger pad (used by
    # the adaptive gripper-close motion) and one on the fingertip
    # bottom (used by the contact-aware descent to detect floor / part
    # top before the gripper jams into the surface).
    #
    # ``ContactSensor`` lives in two different namespaces depending on
    # Isaac Sim version:
    #   * Newer (4.5+):  isaacsim.sensors.contact_sensor.ContactSensor
    #   * Older (≤ 4.4): omni.isaac.sensor.ContactSensor
    # Try the new path first, fall back to the legacy path so the
    # bridge works across both. The ``Camera`` class in this file
    # already uses the legacy ``omni.isaac.sensor`` path successfully,
    # which tells us the legacy namespace is available on this build.
    # Isaac Sim 6.0.1: ContactSensor lives in ``isaacsim.sensors.physics``.
    # Older names (``isaacsim.sensors.contact_sensor`` on 4.5-5.x,
    # ``omni.isaac.sensor`` on <=4.4) are kept as fallbacks for older installs.
    ContactSensor = None
    _cs_namespace = None
    for _cs_mod in ("isaacsim.sensors.physics",
                    "isaacsim.sensors.contact_sensor",
                    "omni.isaac.sensor"):
        try:
            import importlib as _importlib
            ContactSensor = getattr(
                _importlib.import_module(_cs_mod), "ContactSensor")
            _cs_namespace = _cs_mod
            break
        except (ImportError, AttributeError):
            continue

    if ContactSensor is None:
        STATE.contact_sensor = None
        STATE.contact_sensor_tip = None
        print("[WARN] ContactSensor class not found in either "
              "'isaacsim.sensors.contact_sensor' or "
              "'omni.isaac.sensor'. Descent will run blind and the "
              "adaptive gripper close will use the open-loop fallback.")
    else:
        print(f"[OK] ContactSensor class loaded from "
              f"'{_cs_namespace}'")
        try:
            cs_pad = ContactSensor(prim_path=CONTACT_SENSOR_PRIM)
            cs_pad.initialize()
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()
            STATE.contact_sensor = cs_pad
            print(f"[OK] Contact sensor (PAD) ready: {CONTACT_SENSOR_PRIM}")
        except Exception as e:
            STATE.contact_sensor = None
            print(f"[WARN] Contact sensor (PAD) init failed "
                  f"(preset gripper mode): {e}")

        try:
            cs_tip = ContactSensor(prim_path=CONTACT_SENSOR_TIP_PRIM)
            cs_tip.initialize()
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
        print(f"     IK: {'Lula ({})'.format(STATE.target_frame) if STATE.ik_ready else 'DISABLED'}")
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
