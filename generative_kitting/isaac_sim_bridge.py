"""
Isaac Sim ↔ Streamlit Bridge Server  (v2 — Neuro-Symbolic Kitting).

Run this script inside Isaac Sim's Script Editor.  It exposes an HTTP
API on port 8600 that the Streamlit dashboard consumes for:

  GET  /api/ping          → health check + capabilities list
  GET  /api/status        → robot joint positions, DOF info, sim state
  GET  /api/camera        → live camera frame (rgb|depth) as base64 JPEG
  POST /api/execute       → execute a list of action primitives
  POST /api/joints        → set joint positions directly
  POST /api/home          → move robot to home position
  POST /api/gripper       → open / close gripper
  POST /api/approach      → move near a world XYZ (Lula IK)
  POST /api/pick          → full pick sequence at XYZ
  POST /api/place         → full place sequence at XYZ

Usage in Isaac Sim Script Editor:
  exec(open("c:/KP/AI_and_Automation/Sem_4/Thesis/robot_in_air/generative_kitting/isaac_sim_bridge.py").read())
"""

import sys, os, json, asyncio, threading, traceback, io, base64, time, tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

# ═════════════════════════════════════════════════════════════
# CONFIGURATION (mirrors the constants in robot_control.py)
# ═════════════════════════════════════════════════════════════

BRIDGE_PORT = 8600
ROBOT_PRIM  = "/World"

# Dual cameras
CAMERA_RGB_PRIM   = "/World/Camera"
CAMERA_DEPTH_PRIM = "/World/Realsense/RSD455/Camera_Pseudo_Depth"
IMU_PRIM          = "/World/Realsense/RSD455/Imu_Sensor"

# Robot geometry
UR10_BASE_PATH = "/World/gantry_home/ur10_flattened/ur10_instanceable/base_link"
EE_PATH        = "/World/gantry_home/ur10_flattened/ur10_instanceable/ee_link"
PLACE_BOX_PATH = "/World/box_840"

# URDF / YAML for Lula IK
URDF_PATH = r"c:/kp/ai_and_automation/sem_4/thesis/isaacsim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/ur10/urdf/ur10.urdf"
YAML_PATH = r"c:/kp/ai_and_automation/sem_4/thesis/isaacsim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur10/rmpflow/ur10_robot_description.yaml"

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
GRIPPER_TCP_OFFSET   = 0.150
GRIPPER_BODY_WIDTH   = 0.160
GRIPPER_HALF_WIDTH   = GRIPPER_BODY_WIDTH / 2.0
HOVER_CLEARANCE      = 0.020
GRASP_DEPTH_FRACTION = 0.45  # 0.45 = below mid-height, longer grip span, avoids bin bottom
BOX_HEIGHT           = 0.05
BOX_ENTRY_MARGIN     = 0.15
PLACE_DROP_HEIGHT    = 0.02
PLACE_RETRACT_HEIGHT = 0.25
TRANSIT_SAFE_HEIGHT  = 0.10  # just 10cm above safe_z — enough to clear bin rim without
                             # going so high that Lula IK folds the ceiling-mounted arm
DOWNWARD_ORIENTATION = np.array([1.0, 0.0, 1.0, 0.0])

# Interpolation parameters for smooth motion
INTERP_MAX_STEP  = 0.15   # rad per streaming waypoint (smaller = smoother path)
INTERP_THRESHOLD = 0.5    # rad — only interpolate if any joint jumps more than this
STREAM_FRAMES    = 3      # physics frames per streaming waypoint (low = continuous motion)
SETTLE_FRAMES    = 60     # physics frames to settle at final position


# ═════════════════════════════════════════════════════════════
# HELPERS (ported from robot_control.py)
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
    return action, ok


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
        self.lula_solver = None
        self.target_frame = "ee_link"
        self.arm_names = []
        self.dof_names = []
        self.num_dof = 15
        self.is_ready = False
        self.ik_ready = False
        self.is_executing = False
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
                "cameras": ["rgb", "depth"], "ik_ready": STATE.ik_ready,
            })
        elif path == "/api/scene_parts":
            self._handle_scene_parts()
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
        elif path == "/api/pick":
            self._handle_pick(body)
        elif path == "/api/place":
            self._handle_place(body)
        else:
            self._send_json({"error": f"Unknown endpoint: {path}"}, 404)

    # ── GET handlers ─────────────────────────────────────────

    def _handle_status(self):
        if not STATE.is_ready:
            self._send_json({"status": "not_ready"})
            return
        try:
            joints = STATE.robot.get_joint_positions()
            self._send_json({
                "status": "ready" if not STATE.is_executing else "executing",
                "num_dof": STATE.num_dof,
                "joint_names": list(STATE.dof_names),
                "joint_positions": joints.tolist() if joints is not None else [],
                "is_executing": STATE.is_executing,
                "ik_ready": STATE.ik_ready,
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

    def _handle_pick(self, body):
        """Full pick sequence at XYZ."""
        STATE.push_command({"action": "pick_object_ik", "params": body})
        self._send_json(STATE.pop_result(timeout=60))

    def _handle_place(self, body):
        """Full place sequence at XYZ."""
        STATE.push_command({"action": "place_object_ik", "params": body})
        self._send_json(STATE.pop_result(timeout=60))


# ═════════════════════════════════════════════════════════════
# ASYNC COMMAND PROCESSOR
# ═════════════════════════════════════════════════════════════

async def process_commands():
    import omni.kit.app
    while True:
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
            elif action == "approach":
                result = await _approach_target(cmd.get("params", {}))
            elif action == "pick_object_ik":
                result = await _pick_object_ik(cmd.get("params", {}))
            elif action == "place_object_ik":
                result = await _place_object_ik(cmd.get("params", {}))
            else:
                result = {"error": f"Unknown action: {action}"}
        except Exception as e:
            result = {"error": str(e), "traceback": traceback.format_exc()}
            STATE.last_error = str(e)

        STATE.push_result(result)


# ═════════════════════════════════════════════════════════════
# CAMERA CAPTURE (with black-frame fix)
# ═════════════════════════════════════════════════════════════

async def _capture_camera(cam_type="rgb"):
    try:
        from omni.isaac.sensor import Camera
        import omni.kit.app

        if cam_type == "depth":
            if STATE.camera_depth is None:
                STATE.camera_depth = Camera(prim_path=CAMERA_DEPTH_PRIM, resolution=(1280, 720))
                STATE.camera_depth.initialize()
                for _ in range(10):
                    await omni.kit.app.get_app().next_update_async()
            cam = STATE.camera_depth
        else:
            if STATE.camera_rgb is None:
                STATE.camera_rgb = Camera(prim_path=CAMERA_RGB_PRIM, resolution=(1920, 1080))
                STATE.camera_rgb.initialize()
                for _ in range(10):
                    await omni.kit.app.get_app().next_update_async()
            cam = STATE.camera_rgb

        # ── Black-frame fix ──────────────────────────────────
        # Request two consecutive frames to avoid stale/black data
        for attempt in range(3):
            cam.get_current_frame()
            for _ in range(3):
                await omni.kit.app.get_app().next_update_async()

            rgba = cam.get_rgba()
            if rgba is not None and rgba.size > 0:
                # Check if frame is mostly black (mean < 5 across RGB)
                if np.mean(rgba[:, :, :3]) > 3:
                    break  # valid frame
            # else retry

        if rgba is None or rgba.size == 0:
            return {"error": f"{cam_type} camera returned empty frame"}

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
    "tube_with_clamps",
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


async def _move_gantry_x(target_x, finger_hold=None):
    """Move the X-gantry to align with a target X coordinate.

    Args:
        target_x: world X coordinate the gantry should reach.
        finger_hold: if not None, set finger PD targets to this value
                     during the move so the gripper maintains holding force.
    """
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction

    targets = STATE.robot.get_joint_positions()
    gantry_idx = None
    for idx, name in enumerate(STATE.dof_names):
        if GANTRY_X_JOINT in name:
            gantry_idx = idx
            targets[idx] = target_x - GANTRY_X_OFFSET
            break

    if gantry_idx is None:
        return {"error": "Gantry joint not found"}

    # If holding a part, set finger PD targets to max so the controller
    # continuously pushes fingers inward during the gantry move
    if finger_hold is not None:
        targets = set_finger_joints(STATE.robot, STATE.dof_names, finger_hold, targets)

    STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
    for _ in range(SETTLE_FRAMES):
        await omni.kit.app.get_app().next_update_async()

    _update_base_pose()
    return {"status": "ok", "gantry_x": float(targets[gantry_idx])}


async def _approach_target(params):
    """Move arm near target XYZ at safe height (for depth camera inspection).

    Uses the real USD bbox centre X for gantry alignment so the robot is
    FULLY positioned over the part before the depth scan — no lateral
    correction is needed after scanning.
    """
    x = params.get("x", 0.3)
    y = params.get("y", 0.0)
    z = params.get("z", 0.5)
    part_prim = params.get("part_prim", None)

    # If a USD prim path is given, use the real bbox centre for gantry X
    # so there is no second gantry slide after the depth scan.
    if part_prim:
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            geom = compute_bbox_geometry(stage, part_prim, "APPROACH")
            x = geom["center_xy"][0]
            y = geom["center_xy"][1]
            z = geom["top_z"]
        except Exception as e:
            print(f"  [APPROACH] bbox failed ({e}), using param coords")

    # Stay high: BOX_ENTRY_MARGIN above bin rim, not inside it
    safe_z = z + BOX_ENTRY_MARGIN + BOX_HEIGHT + GRIPPER_TCP_OFFSET + 0.10

    # Move gantry to EXACT part X first, then update base pose
    await _move_gantry_x(x)
    _update_base_pose()

    # Move arm to safe height above target XY
    result = await _ik_move_to([x, y, safe_z])
    if "error" in result:
        return result

    return {"status": "ok", "action": "approach",
            "target": [x, y, z], "ee_position": result["ee_position"]}


async def _pick_object_ik(params):
    """Full IK-based pick sequence with chained warm-starts.

    Ported from the proven robot_control.py multi-phase approach:
      Phase 1: IK solve for safe height above target
      Phase 2: Re-solve from settled joints (same target, better seed)
      Phase 3a: Hover — warm-start from Phase 2 solution
      Phase 3b: Grasp — warm-start from hover solution
      Phase 4: Retract — warm-start from settled post-grasp joints

    Each phase chains the IK solution as warm-start for the next,
    keeping the solver in the same joint-space neighbourhood and
    preventing self-collision on this ceiling-mounted gantry UR10.
    """
    import omni.kit.app, omni.usd
    from isaacsim.core.utils.types import ArticulationAction

    if not STATE.ik_ready:
        return {"error": "IK solver not initialised"}

    x = params.get("x", 0.3)
    y = params.get("y", 0.0)
    z = params.get("z", 0.02)
    part_path = params.get("part_prim", None)

    STATE.is_executing = True
    steps = []
    try:
        locked_ori = normalize_quat(DOWNWARD_ORIENTATION)
        tcp_z_offset = 0.0 if STATE.target_frame == "gripper_tcp" else GRIPPER_TCP_OFFSET

        # Use bbox geometry if part_prim is provided
        finger_close = FINGER_CLOSE  # fixed value — physics stops fingers on contact
        if part_path:
            stage = omni.usd.get_context().get_stage()
            try:
                geom = compute_grasp_geometry(stage, part_path, GRASP_DEPTH_FRACTION)
                x = geom["part_center_xy"][0]
                y = geom["part_center_xy"][1]
                part_top_z = geom["part_top_z"]
                grasp_z = geom["grasp_z"]
            except Exception:
                part_top_z = z
                grasp_z = z
        else:
            part_top_z = z
            grasp_z = z

        safe_z = part_top_z + BOX_HEIGHT + BOX_ENTRY_MARGIN + tcp_z_offset
        entry_z = part_top_z + BOX_HEIGHT + tcp_z_offset  # just inside box rim
        hover_z = part_top_z + HOVER_CLEARANCE + tcp_z_offset
        grasp_target_z = grasp_z + tcp_z_offset
        transit_z = safe_z + TRANSIT_SAFE_HEIGHT

        safe_pos = np.array([x, y, safe_z])
        entry_pos = np.array([x, y, entry_z])
        hover_pos = np.array([x, y, hover_z])
        grasp_pos = np.array([x, y, grasp_target_z])
        transit_pos = np.array([x, y, transit_z])

        print(f"  [PICK] target=({x:.3f}, {y:.3f}, {z:.3f}) safe_z={safe_z:.3f} hover_z={hover_z:.3f} grasp_z={grasp_target_z:.3f} finger={finger_close:.3f}")

        # 1. Move gantry ONLY if not already aligned (approach already positioned it).
        #    Skipping prevents the visible X-slide after depth scan.
        gantry_aligned = False
        current_joints = STATE.robot.get_joint_positions()
        for idx, name in enumerate(STATE.dof_names):
            if GANTRY_X_JOINT in name:
                if abs(current_joints[idx] - (x - GANTRY_X_OFFSET)) < 0.01:
                    gantry_aligned = True
                break
        if not gantry_aligned:
            await _move_gantry_x(x)
        _update_base_pose()
        steps.append({"phase": "gantry", "status": "ok"})

        # 2. Open gripper before any arm movement
        await _set_gripper(FINGER_OPEN)
        steps.append({"phase": "open_gripper", "status": "ok"})

        # 3. Move to safe height — VERTICAL-FIRST to avoid sweeping through bin walls.
        #
        #    Two-phase approach prevents lateral arm swing during bin exit:
        #      3a) If arm is below safe_z (inside / near bin): solve IK at CURRENT
        #          EE X,Y but target safe_z — pulls straight up with no lateral motion.
        #      3b) Lateral align — from above the bin, move to exact [x, y, safe_z].
        #          IK from safe height → safe height is a small, collision-free move.
        #
        #    Without this, IK re-solves for safe_pos (new bbox XY) from a position
        #    that is offset in X and/or Z, and the solver finds a different arm
        #    configuration that swings fingers through the bin walls on the way up.
        current_ee = get_world_pos(EE_PATH)
        warm = _get_warm_start()

        # Phase 3a: vertical retract via Cartesian micro-steps (skip if already above safe_z).
        #   Each 4 cm IK step is warm-started from the previous result so the solver
        #   cannot flip to a distant configuration — arm rises straight up, no swinging.
        if current_ee[2] < safe_z - 0.05:
            v_ok = await _retract_cartesian_up(
                safe_z, current_ee[0], current_ee[1], locked_ori, step_size=0.04)
            if not v_ok:
                return {"error": f"Cartesian retract (Phase 3a) failed", "steps": steps}
            steps.append({"phase": "safe_height_retract", "status": "ok"})
            warm = _get_warm_start()  # fresh seed for lateral align

        # Phase 3b: lateral align to exact [x, y, safe_z] (from above — always safe).
        #   Always apply — skipping with a "close enough" threshold was the root
        #   cause of bin-wall collisions: the arm descended with a 2-4 cm offset
        #   that put the open fingers into the bin wall.
        p1_action, p1_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                     safe_pos, locked_ori, warm)
        if not p1_ok:
            return {"error": f"IK Phase 3b failed for safe_pos {safe_pos.tolist()}", "steps": steps}
        targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, p1_action)
        await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES)
        steps.append({"phase": "safe_height", "status": "ok"})

        # Gripper stays fully open during descent — physics stops fingers
        # on part contact naturally (same as robot_control.py).

        # 4. Re-solve from current settled pose — this is the seed for the descent
        settled_warm = _get_warm_start()
        p2_action, p2_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                     safe_pos, locked_ori, settled_warm)
        if not p2_ok:
            p2_action = p1_action  # fallback

        # 5. Entry — descend vertically to just inside the box rim
        #    This forces a pure vertical path through the rim opening,
        #    preventing the IK solver from swinging the arm horizontally.
        entry_action, entry_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                           entry_pos, locked_ori, p2_action)
        if entry_ok:
            targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, entry_action)
            await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES)
            steps.append({"phase": "entry", "status": "ok"})
            # Use entry solution as warm-start for hover
            p2_action = entry_action

        # 6. Hover — descend inside box, above part
        hover_action, hover_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                           hover_pos, locked_ori, p2_action)
        if not hover_ok:
            return {"error": f"IK hover failed for {hover_pos.tolist()}", "steps": steps}

        targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, hover_action)
        await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES)
        steps.append({"phase": "hover", "status": "ok"})

        # 6. PHASE 3b -- Plunge to grasp height
        #    Warm-start from hover solution
        grasp_action, grasp_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                           grasp_pos, locked_ori, hover_action)
        if not grasp_ok:
            return {"error": f"IK grasp failed for {grasp_pos.tolist()}", "steps": steps}

        targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, grasp_action)
        await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES)
        steps.append({"phase": "grasp_descend", "status": "ok"})

        # 7. Close gripper — direct joint target, physics stops fingers on
        #    part contact naturally (matches robot_control.py exactly).
        #    120 settle frames lets the PD controller reach the part surface
        #    smoothly without snapping.
        targets = STATE.robot.get_joint_positions()
        targets = set_finger_joints(STATE.robot, STATE.dof_names, finger_close, targets)
        STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(120):
            await omni.kit.app.get_app().next_update_async()
        # Lock grip: by frame 120 fingers are nearly there — this snaps
        # the last few degrees and sets PD target to FINGER_CLOSE_MAX so
        # the controller holds firm during retract (prevents late-close).
        _force_grip(finger_close)
        for _ in range(SETTLE_FRAMES):
            await omni.kit.app.get_app().next_update_async()
        steps.append({"phase": "close_gripper", "status": "ok", "finger_close": finger_close})

        # 9. Verify grasp
        grasp_check = await _verify_grasp()
        steps.append({"phase": "verify_grasp", "status": "ok", "detail": grasp_check})

        # PHASE 4 -- Retract to safe height above bin.
        #   Single IK solve warm-started from current settled joints so the
        #   solver stays in the same configuration (no flip) — matches
        #   robot_control.py Phase 4.  Isaac Sim physics makes the motion
        #   naturally smooth over 150 frames; no manual interpolation needed.
        settled_warm = _get_warm_start()
        retract_action, retract_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                               safe_pos, locked_ori, settled_warm)
        if not retract_ok:
            return {"error": "IK retract (Phase 4) failed", "steps": steps}
        targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, retract_action)
        targets = set_finger_joints(STATE.robot, STATE.dof_names, FINGER_CLOSE_MAX, targets)
        STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(150):
            await omni.kit.app.get_app().next_update_async()
        _force_grip(finger_close)
        steps.append({"phase": "retract", "status": "ok"})

        # PHASE 4b -- Rise to transit clear-height (same XY, higher Z).
        #   Second single IK solve — robot_control.py Phase 4b pattern.
        settled_transit_warm = _get_warm_start()
        transit_action, transit_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                               transit_pos, locked_ori, settled_transit_warm)
        if transit_ok:
            targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, transit_action)
            targets = set_finger_joints(STATE.robot, STATE.dof_names, FINGER_CLOSE_MAX, targets)
            STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
            for _ in range(150):
                await omni.kit.app.get_app().next_update_async()
            _force_grip(finger_close)
        steps.append({"phase": "transit_safe", "status": "ok" if transit_ok else "skipped"})

        return {"status": "completed", "action": "pick_object",
                "position": [x, y, z], "steps": steps,
                "finger_close": finger_close}

    except Exception as e:
        return {"status": "error", "error": str(e), "steps": steps}
    finally:
        STATE.is_executing = False


async def _place_object_ik(params):
    """Full IK-based place sequence with chained warm-starts.

    Same multi-phase pattern as pick — each IK solution chains
    as warm-start for the next to prevent self-collision.
    """
    import omni.kit.app, omni.usd
    from isaacsim.core.utils.types import ArticulationAction

    if not STATE.ik_ready:
        return {"error": "IK solver not initialised"}

    x = params.get("x", 0.5)
    y = params.get("y", 0.0)
    z = params.get("z", 0.02)
    dest_prim = params.get("dest_prim", PLACE_BOX_PATH)
    # Use adaptive finger_close from the pick result if provided
    finger_close = params.get("finger_close", FINGER_CLOSE)

    STATE.is_executing = True
    steps = []
    try:
        locked_ori = normalize_quat(DOWNWARD_ORIENTATION)
        tcp_z_offset = 0.0 if STATE.target_frame == "gripper_tcp" else GRIPPER_TCP_OFFSET

        # Compute destination geometry
        stage = omni.usd.get_context().get_stage()
        try:
            dest_geom = compute_bbox_geometry(stage, dest_prim, "DESTINATION")
            dest_x = dest_geom["center_xy"][0]
            dest_y = dest_geom["center_xy"][1]
            dest_top_z = dest_geom["top_z"]
        except Exception:
            dest_x, dest_y, dest_top_z = x, y, z

        place_z = dest_top_z + PLACE_DROP_HEIGHT + tcp_z_offset
        place_safe_z = dest_top_z + BOX_ENTRY_MARGIN + tcp_z_offset
        retract_z = dest_top_z + PLACE_RETRACT_HEIGHT + tcp_z_offset

        transit_pos = np.array([dest_x, dest_y, place_safe_z + TRANSIT_SAFE_HEIGHT])
        safe_pos = np.array([dest_x, dest_y, place_safe_z])
        place_pos = np.array([dest_x, dest_y, place_z])
        retract_pos = np.array([dest_x, dest_y, retract_z])

        # 1. Transit -- raise arm high before lateral gantry move via micro-steps.
        #    Same Cartesian stepping used for pick retract: prevents configuration
        #    flips while ascending with a part in the gripper.
        current_ee    = get_world_pos(EE_PATH)
        transit_up_z  = current_ee[2] + TRANSIT_SAFE_HEIGHT
        transit_up_ok = await _retract_cartesian_up(
            transit_up_z, current_ee[0], current_ee[1], locked_ori,
            finger_value=FINGER_CLOSE_MAX, step_size=0.04)
        _force_grip(finger_close)
        steps.append({"phase": "transit_up", "status": "ok" if transit_up_ok else "warn"})

        # 2. Force-lock grip before gantry move
        _force_grip(finger_close)

        # 3. Move gantry to destination X — finger_hold=FINGER_CLOSE_MAX keeps
        #    PD target at max during the 120-frame gantry settle
        await _move_gantry_x(dest_x, finger_hold=FINGER_CLOSE_MAX)
        _update_base_pose()
        steps.append({"phase": "gantry_move", "status": "ok"})

        # 4. Force-lock grip after gantry move
        _force_grip(finger_close)

        # 5. Transit height at destination — stay high above tray before descending
        #    This prevents the arm from swinging through the kitting tray
        #    when the IK solver reconfigures for the new XY position.
        transit_dest_warm = _get_warm_start()
        transit_dest_action, transit_dest_ok = ik_solve(
            STATE.lula_solver, STATE.target_frame,
            transit_pos, locked_ori, transit_dest_warm)
        if transit_dest_ok:
            targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, transit_dest_action)
            await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES, finger_value=FINGER_CLOSE_MAX)
            _force_grip(finger_close)
        steps.append({"phase": "transit_at_dest", "status": "ok" if transit_dest_ok else "warn"})

        # 6. Safe height above destination — descend from transit to just above tray rim
        settled_warm = _get_warm_start()
        safe_action, safe_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                         safe_pos, locked_ori, settled_warm)
        if not safe_ok:
            return {"error": f"IK failed for place safe {safe_pos.tolist()}", "steps": steps}

        targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, safe_action)
        await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES, finger_value=FINGER_CLOSE_MAX)
        _force_grip(finger_close)
        steps.append({"phase": "place_safe", "status": "ok"})

        # 7. Descend slowly to place height via Cartesian micro-steps.
        #    3 cm steps with 8 frames each creates a clearly visible, controlled
        #    descent — the arm "lowers" the part rather than snapping to place height.
        place_ok = await _retract_cartesian_up(
            place_z, dest_x, dest_y, locked_ori,
            finger_value=FINGER_CLOSE_MAX, step_size=0.03, stream_frames=8)
        _force_grip(finger_close)
        steps.append({"phase": "place_descend", "status": "ok" if place_ok else "warn"})

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
# STARTUP
# ═════════════════════════════════════════════════════════════

async def start_bridge():
    import omni.kit.app, omni.timeline
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation

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

    # Capture the robot's initial rest pose as HOME_JOINTS
    global HOME_JOINTS
    initial_joints = robot.get_joint_positions()
    HOME_JOINTS = initial_joints.tolist()[:7]
    print(f"[OK] Robot ready — {STATE.num_dof} DOFs")
    print(f"[OK] Home joints captured: {[f'{j:.3f}' for j in HOME_JOINTS]}")

    # ── Lula IK Solver ───────────────────────────────────────
    try:
        from omni.isaac.motion_generation import LulaKinematicsSolver

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

    # ── HTTP Server ──────────────────────────────────────────
    server = HTTPServer(("0.0.0.0", BRIDGE_PORT), BridgeHandler)

    def serve():
        print(f"[OK] Bridge v2 on http://localhost:{BRIDGE_PORT}")
        print(f"     IK: {'Lula ({})'.format(STATE.target_frame) if STATE.ik_ready else 'DISABLED'}")
        print(f"     Cameras: RGB={CAMERA_RGB_PRIM}")
        print(f"              Depth={CAMERA_DEPTH_PRIM}")
        print("=" * 60)
        server.serve_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    await process_commands()


asyncio.ensure_future(start_bridge())
