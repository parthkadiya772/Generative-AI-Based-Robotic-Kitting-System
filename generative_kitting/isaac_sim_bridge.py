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
CAMERA_WRIST_PRIM = "/World/Realsense/RSD455/Camera_OmniVision_OV9782_Color"
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
GRASP_LIFT           = 0.040  # 40mm above surface Z — finger pads grip at part mid-height
                              # (parts are ~20-50mm tall, so 40mm puts pads at ~mid-body)
                              # Must keep clearance from bin bottom so fingers can close
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


# ── Shoulder-Pan Safety Clamp ──────────────────────────────
# The robot is ceiling-mounted on a gantry rail (along X).
# Lula IK has no collision awareness of the gantry structure,
# so it can find solutions where the arm swings THROUGH the
# red gantry rail.  We constrain shoulder_pan (joint 0) to a
# safe range that keeps the arm reaching downward into the
# workspace, never backward through the gantry.
#
# shoulder_pan = 0 rad → arm hangs straight down (home)
# Safe range: roughly -π/2 to +π/2 (arm stays in front)

SHOULDER_PAN_MIN = -1.5708   # -π/2: arm reaches to one side
SHOULDER_PAN_MAX =  1.5708   # +π/2: arm reaches to other side


def _clamp_shoulder_pan(ik_result):
    """Clamp shoulder_pan joint in IK result to prevent gantry collision."""
    clamped = np.array(ik_result, dtype=np.float64)
    # Joint 0 in Lula's 6-joint result is shoulder_pan
    if len(clamped) > 0:
        original = clamped[0]
        clamped[0] = np.clip(clamped[0], SHOULDER_PAN_MIN, SHOULDER_PAN_MAX)
        if abs(original - clamped[0]) > 0.01:
            print(f"  [IK SAFETY] shoulder_pan clamped: {original:.3f} → {clamped[0]:.3f} rad "
                  f"(preventing gantry collision)")
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
                "cameras": ["rgb", "depth", "wrist"], "ik_ready": STATE.ik_ready,
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
        from omni.isaac.sensor import Camera
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
    """Get Z height of workspace surface where parts sit.

    Reads the parts container prim position from USD.  Falls back to
    a reasonable default if the prim isn't available.
    """
    import omni.usd
    try:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(PARTS_CONTAINER)
        if prim.IsValid():
            mat = omni.usd.get_world_transform_matrix(prim)
            z = float(mat.GetRow(3)[2])
            print(f"  [workspace_z] Parts container Z = {z:.4f}")
            return z
    except Exception as e:
        print(f"  [workspace_z] USD read failed: {e}")
    return 0.02  # reasonable default for bin surface


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
    from omni.isaac.sensor import Camera
    from pxr import UsdGeom, Gf
    import math

    points = params.get("points", [])
    cam_alias = params.get("camera", "rgb")

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
    w_res, h_res = res
    try:
        intrinsics = cam.get_intrinsics_matrix()  # 3×3 numpy
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        print(f"  [project] Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    except Exception:
        hfov_rad = math.radians(60)
        fx = fy = (w_res / 2.0) / math.tan(hfov_rad / 2.0)
        cx, cy = w_res / 2.0, h_res / 2.0
        print(f"  [project] Intrinsics (estimated): fx={fx:.1f} fy={fy:.1f}")

    # ── Camera extrinsics (world transform) ────────────────────
    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath(cam_path)
    cam_world_mat = omni.usd.get_world_transform_matrix(cam_prim)

    cam_pos = cam_world_mat.GetRow(3)
    print(f"  [project] Camera world pos: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f})")

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

    method_used = "depth" if has_depth else "geometric"
    print(f"  [project] Method: {method_used} | {len(points)} points to project")

    # ── Project each point ─────────────────────────────────────
    world_points = []
    for i, pt in enumerate(points):
        nx = float(pt.get("x", 0.5))
        ny = float(pt.get("y", 0.5))

        point_method = None

        # ── Method 1: Depth buffer projection ──────────────────
        if has_depth:
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
                cam_x =  (px - cx) * d / fx
                cam_y = -(py - cy) * d / fy
                cam_z = -d

                cam_pt = Gf.Vec4d(cam_x, cam_y, cam_z, 1.0)
                world_pt = cam_world_mat.GetTranspose() * cam_pt

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

            # Ray direction in camera frame
            ray_cam = Gf.Vec4d(
                (px - cx) / fx,
                -(py - cy) / fy,
                -1.0,
                0.0,   # direction vector (w=0)
            )
            # Transform ray direction to world frame
            ray_world = cam_world_mat.GetTranspose() * ray_cam

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
        print(f"  [{i}] image({nx:.2f},{ny:.2f}) → world({wp['x']:.3f}, {wp['y']:.3f}, {wp['z']:.3f}) [{point_method}]")

    return {"status": "ok", "world_points": world_points,
            "camera": cam_path, "method": method_used}


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

    All coordinates come from VLM + depth projection — no USD prim paths.
    The workflow engine already resolved image coords to world XYZ via
    the /api/project_to_world endpoint before calling this.
    """
    x = params.get("x", 0.3)
    y = params.get("y", 0.0)
    z = params.get("z", 0.5)

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


def _pick_z_positions(z):
    """Shared Z waypoint calculations for the 3-phase pick sequence."""
    tcp_z_offset = 0.0 if STATE.target_frame == "gripper_tcp" else GRIPPER_TCP_OFFSET
    part_top_z = z
    grasp_z = z + GRASP_LIFT       # TCP stays ABOVE part top, pads wrap below
    safe_z = part_top_z + BOX_HEIGHT + BOX_ENTRY_MARGIN + tcp_z_offset
    entry_z = part_top_z + BOX_HEIGHT + tcp_z_offset
    hover_z = part_top_z + HOVER_CLEARANCE + tcp_z_offset
    grasp_target_z = grasp_z + tcp_z_offset
    transit_z = safe_z + TRANSIT_SAFE_HEIGHT
    return dict(safe_z=safe_z, entry_z=entry_z, hover_z=hover_z,
                grasp_target_z=grasp_target_z, transit_z=transit_z,
                tcp_z_offset=tcp_z_offset)


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
        zp = _pick_z_positions(z)
        safe_z = zp["safe_z"]

        safe_pos = np.array([x, y, safe_z])
        entry_pos = np.array([x, y, zp["entry_z"]])
        hover_pos = np.array([x, y, zp["hover_z"]])
        grasp_pos = np.array([x, y, zp["grasp_target_z"]])

        print(f"  [DESCEND] target=({x:.3f}, {y:.3f}, {z:.3f}) "
              f"safe_z={safe_z:.3f} hover_z={zp['hover_z']:.3f} "
              f"grasp_z={zp['grasp_target_z']:.3f}")

        # 1. Open gripper
        await _set_gripper(FINGER_OPEN)
        steps.append({"phase": "open_gripper", "status": "ok"})

        # 2. Gantry alignment — slide gantry to part X while keeping
        #    EE at its current world position (depth-scan pose).
        #    The arm joints adjust to compensate for the gantry slide,
        #    so the gripper stays looking at the same spot.
        gantry_aligned = False
        current_joints = STATE.robot.get_joint_positions()
        for idx, name in enumerate(STATE.dof_names):
            if GANTRY_X_JOINT in name:
                if abs(current_joints[idx] - (x - GANTRY_X_OFFSET)) < 0.01:
                    gantry_aligned = True
                break
        if not gantry_aligned:
            ee_before = get_world_pos(EE_PATH).copy()
            print(f"  [DESCEND] Gantry re-align to x={x:.3f}, "
                  f"holding EE at ({ee_before[0]:.3f}, {ee_before[1]:.3f}, {ee_before[2]:.3f})")
            await _move_gantry_x(x)
            _update_base_pose()
            # Solve IK to keep EE at same world position after gantry moved
            restore_warm = _get_warm_start()
            restore_action, restore_ok = ik_solve(
                STATE.lula_solver, STATE.target_frame,
                ee_before, locked_ori, restore_warm)
            if restore_ok:
                targets = apply_arm_joints(
                    STATE.robot, STATE.dof_names, STATE.arm_names, restore_action)
                await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES)
                ee_after = get_world_pos(EE_PATH)
                print(f"  [DESCEND] EE held at ({ee_after[0]:.3f}, {ee_after[1]:.3f}, {ee_after[2]:.3f}) "
                      f"drift={np.linalg.norm(ee_after - ee_before):.4f}m")
            else:
                print("  [DESCEND] WARNING: EE restore IK failed — EE may have shifted")
        else:
            _update_base_pose()
        steps.append({"phase": "gantry", "status": "ok"})

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

        # 7. Grasp position — gentle 5mm drop from hover (GRASP_LIFT above part top)
        grasp_action, grasp_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                           grasp_pos, locked_ori, hover_action)
        if not grasp_ok:
            return {"error": f"IK grasp failed {grasp_pos.tolist()}", "steps": steps}
        targets = apply_arm_joints(
            STATE.robot, STATE.dof_names, STATE.arm_names, grasp_action)
        await _apply_interpolated(targets, settle_frames=SETTLE_FRAMES)
        steps.append({"phase": "grasp_descend", "status": "ok"})

        # 8. Capture wrist camera — caller uses this for VLM "part between fingers?" check
        wrist_frame = await _capture_camera("wrist")
        wrist_b64 = wrist_frame.get("image_base64")
        steps.append({"phase": "wrist_capture",
                       "status": "ok" if wrist_b64 else "warn"})

        return {"status": "ok", "action": "pick_descend",
                "position": [x, y, z], "steps": steps,
                "wrist_image": wrist_b64}

    except Exception as e:
        return {"status": "error", "error": str(e), "steps": steps}
    finally:
        STATE.is_executing = False


# ─── PICK PHASE 2: Close gripper and confirm ───────────────

async def _pick_close(params):
    """Close the gripper and confirm the part is grasped.

    Retries up to 3 times with increasing settle time.
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

        # Retract to safe height using Cartesian micro-steps (slow lift)
        # Micro-steps keep the solver in the same joint neighbourhood AND
        # continuously enforce finger_value at every waypoint — so the grip
        # is maintained throughout the lift instead of relaxing mid-jump.
        print(f"  [RETRACT] Lifting to safe_z={safe_pos[2]:.3f} via micro-steps")
        retract_ok = await _retract_cartesian_up(
            safe_pos[2], x, y, locked_ori,
            finger_value=grip_hold, step_size=0.03, stream_frames=8)
        steps.append({"phase": "retract",
                       "status": "ok" if retract_ok else "failed"})
        if not retract_ok:
            return {"error": "Cartesian retract failed", "steps": steps}

        # Transit to clear-height (also micro-stepped to maintain grip)
        print(f"  [RETRACT] Transit to z={transit_pos[2]:.3f}")
        transit_ok = await _retract_cartesian_up(
            transit_pos[2], x, y, locked_ori,
            finger_value=grip_hold, step_size=0.04, stream_frames=6)
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

        place_z = dest_top_z + PLACE_DROP_HEIGHT + tcp_z_offset
        place_safe_z = dest_top_z + BOX_ENTRY_MARGIN + tcp_z_offset
        retract_z = dest_top_z + PLACE_RETRACT_HEIGHT + tcp_z_offset

        transit_pos = np.array([dest_x, dest_y, place_safe_z + TRANSIT_SAFE_HEIGHT])
        safe_pos = np.array([dest_x, dest_y, place_safe_z])
        place_pos = np.array([dest_x, dest_y, place_z])
        retract_pos = np.array([dest_x, dest_y, retract_z])

        # 1. Transit -- move arm to safe height before lateral gantry move.
        #    Single IK solve (same pattern as pick retract).
        current_ee = get_world_pos(EE_PATH)
        transit_up_z = current_ee[2] + TRANSIT_SAFE_HEIGHT
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

        # 2. Move gantry to destination X — grip_hold keeps soft contact
        await _move_gantry_x(dest_x, finger_hold=grip_hold)
        _update_base_pose()
        steps.append({"phase": "gantry_move", "status": "ok"})

        # 3. Transit height at destination
        transit_dest_warm = _get_warm_start()
        transit_dest_action, transit_dest_ok = ik_solve(
            STATE.lula_solver, STATE.target_frame,
            transit_pos, locked_ori, transit_dest_warm)
        if transit_dest_ok:
            targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, transit_dest_action)
            targets = set_finger_joints(STATE.robot, STATE.dof_names, grip_hold, targets)
            STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
            for _ in range(150):
                await omni.kit.app.get_app().next_update_async()
        steps.append({"phase": "transit_at_dest", "status": "ok" if transit_dest_ok else "warn"})

        # 4. Safe height above destination
        settled_warm = _get_warm_start()
        safe_action, safe_ok = ik_solve(STATE.lula_solver, STATE.target_frame,
                                         safe_pos, locked_ori, settled_warm)
        if not safe_ok:
            return {"error": f"IK failed for place safe {safe_pos.tolist()}", "steps": steps}

        targets = apply_arm_joints(STATE.robot, STATE.dof_names, STATE.arm_names, safe_action)
        targets = set_finger_joints(STATE.robot, STATE.dof_names, grip_hold, targets)
        STATE.robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(150):
            await omni.kit.app.get_app().next_update_async()
        steps.append({"phase": "place_safe", "status": "ok"})

        # 5. Descend slowly to place height via Cartesian micro-steps.
        place_ok = await _retract_cartesian_up(
            place_z, dest_x, dest_y, locked_ori,
            finger_value=grip_hold, step_size=0.03, stream_frames=8)
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

    # ── Subscribe to timeline events (stop/pause → shutdown) ─
    if _timeline_sub is not None:
        try:
            _timeline_sub.unsubscribe()
        except Exception:
            pass

    stream = timeline.get_timeline_event_stream()
    _timeline_sub = stream.create_subscription_to_pop(_on_timeline_event)
    print("[OK] Timeline subscription active — bridge will auto-stop on sim stop")

    # ── HTTP Server ──────────────────────────────────────────
    _bridge_server = HTTPServer(("0.0.0.0", BRIDGE_PORT), BridgeHandler)
    _bridge_server.timeout = 1  # so shutdown() isn't blocked forever

    def serve():
        print(f"[OK] Bridge v2 on http://localhost:{BRIDGE_PORT}")
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
            elif action == "approach":
                result = await _approach_target(cmd.get("params", {}))
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
