"""Standalone cuMotion pick-and-place. Run in Isaac Sim's Script Editor.

    exec(open("c:/KP/AI_and_Automation/Sem_4/Thesis/robot_in_air/generative_kitting/robot_control_cumotion.py").read())

No Streamlit, no HTTP bridge, no command queue, no VLM. Just: build a
collision world from the 4 corner cameras, take the part you selected in
the Stage, and pick-and-place it with cuMotion.

HOW TO USE
    1. Press PLAY (physics must be stepping).
    2. Select the part prim you want to pick, in the Stage tree.
    3. exec(...) this file.

WHY IT EXISTS
    The bridge reports trajectories executing while the robot stays put,
    yet a plain apply_action moves it fine. Everything here runs inline
    in the Script Editor, so the async command loop and the HTTP layer
    are out of the picture. EXEC_MODE switches between the two ways of
    driving the arm so they can be compared directly:

        "stream" — apply every trajectory waypoint (what the bridge does)
        "direct" — apply only the final configuration and wait
        "both"   — try stream, and if the joints did not move, try direct

    Whichever moves the robot tells us where the bridge bug is.
"""

import asyncio
import os

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.core.utils.types import ArticulationAction

# ═════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════

ROBOT_PRIM = "/World/gantry"          # articulation root
ROBOT_ROOT_PATH = "/World/gantry"     # URDF root link `gantry` (verified)
ROBOT_CONFIG_PATH = os.environ.get(
    "ROBOT_CONFIG_PATH",
    "C:/KP/AI_and_Automation/Sem_4/Thesis/robot_in_air/generative_kitting/robot_config")
URDF_NAME, XRDF_NAME = "AIKIDO.urdf", "AIKIDO.xrdf"
TOOL_FRAME = "robotiq_base_link"

POINTCLOUD_CAMERAS = {
    "/World/pointcloud_view/cam1": (480, 640),
    "/World/pointcloud_view/cam2": (480, 640),
    "/World/pointcloud_view/cam3": (480, 640),
    "/World/pointcloud_view/cam4": (480, 640),
}

# ── Choosing the part ────────────────────────────────────────
# PREFERRED: explicit world XYZ of the grasp point, copied from the
# perception log, e.g.
#     [scan] motor_valve -> x=-1.208 y=-0.600 z=0.821
# This project derives part coordinates from camera + depth precisely
# because USD prim lookups are unreliable in this scene:
# ComputeWorldBound returns prototype-space values for the instanceable
# part prims — the same bug behind bin top_z=1321 and tray top_z=200.
# TARGET_PRIM is kept only as a fallback and its result is checked below.
TARGET_XYZ = (-0.5699191459599902, -0.7131680933408455, 0.8557111150372856)                    # e.g. (-1.208, -0.600, 0.821)
TARGET_PRIM = "/World/robot_facade_full/motor_valve_1"                      # fallback; USD bbox may be wrong
PARTS_CONTAINER = "/World/robot_facade_full"

# A grasp point outside these bounds is not a real part coordinate.
WORKSPACE_BOUNDS = ((-3.5, 3.5), (-2.5, 2.5), (-0.2, 2.5))

# Where to drop the part. Set from your tray.
PLACE_XYZ = (0.683, -0.753, 1.05)

EXEC_MODE = "both"                    # "stream" | "direct" | "both"

APPROACH_HEIGHT = 0.25                # metres above the part to hover
GRASP_CLEARANCE = 0.005               # stop this far above the part top
LIFT_HEIGHT = 0.25                    # lift after grasping
GRIPPER_TCP_OFFSET = 0.19357          # tool frame -> finger contact pad
# (w, x, y, z). This value was derived for ee_link under Lula. cuMotion
# plans for robotiq_base_link, which may be a differently-oriented frame,
# so asking for this quaternion can demand a pose the arm must contort to
# reach. With USE_CURRENT_ORIENTATION the script instead reads the tool's
# actual orientation from USD — frame-agnostic and always achievable.
DOWNWARD_QUAT = (1.0, 0.0, 1.0, 0.0)
USE_CURRENT_ORIENTATION = True

# Frames to compare in the report below.
EE_LINK_PRIM = ("/World/gantry/gantry_home/ur10_flattened/"
                "ur10_instanceable/ee_link")
TOOL_PRIM = ("/World/gantry/gantry_home/ur10_flattened/"
             "robotiq_fixed_physics/Robotiq_2F_140_physics_edit/"
             "robotiq_base_link")
# Authored TCP Xform — the finger contact point. Preferred over mesh
# prims, whose transforms all coincide with robotiq_base_link.
GRIPPER_TCP_PRIM = ("/World/gantry/gantry_home/ur10_flattened/"
                    "robotiq_fixed_physics/Robotiq_2F_140_physics_edit/"
                    "gripper_tcp")
FINGER_PAD_PRIM = ("/World/gantry/gantry_home/ur10_flattened/"
                   "robotiq_fixed_physics/Robotiq_2F_140_physics_edit/"
                   "left_inner_finger/Fingertip_01/Fingertip")
MIN_CREDIBLE_TCP_OFFSET = 0.05

FINGER_OPEN, FINGER_CLOSE = 0.0, 0.5
SETTLE_FRAMES = 60
TRAJ_DT = 1.0 / 60.0

VOXEL_SIZE = 0.02
MAX_SPHERES = 6000
CARVE_RADIUS = 0.08                   # clear the cloud around the target
SAFETY_MARGIN = 0.005

MAX_VEL = np.array([0.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0])
MAX_ACC = np.array([1.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5])

# ═════════════════════════════════════════════════════════════

STATE = {}


def log(msg):
    print(f"[pick] {msg}")


def normalize_quat(q):
    q = np.array(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    h = angle / 2.0
    return np.array([np.cos(h), *(axis * np.sin(h))])


def derive_down_quat(tool_p, tool_q, tcp_p):
    """Tool orientation aiming the gripper's approach axis at world -Z.

    Measured from the scene: the approach axis is tool -> TCP. On this
    robot it currently points UP, so a top-down grasp needs a half turn.
    Derived rather than hardcoded, so it stays correct if the gripper
    mount or the gripper_tcp Xform changes.
    """
    approach = np.asarray(tcp_p, float) - np.asarray(tool_p, float)
    norm = np.linalg.norm(approach)
    if norm < 1e-9:
        return None, None
    approach = approach / norm
    target = np.array([0.0, 0.0, -1.0])
    axis = np.cross(approach, target)
    dot = float(np.clip(np.dot(approach, target), -1.0, 1.0))
    if np.linalg.norm(axis) < 1e-8:
        q_align = (np.array([1.0, 0.0, 0.0, 0.0]) if dot > 0
                   else quat_from_axis_angle([1.0, 0.0, 0.0], np.pi))
    else:
        q_align = quat_from_axis_angle(axis, np.arccos(dot))
    q = quat_mul(q_align, np.asarray(tool_q, float))
    return q / np.linalg.norm(q), approach


# ─── Setup ───────────────────────────────────────────────────

def setup():
    """Load the articulation, cuMotion robot, world and planner."""
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.robot_motion.cumotion import (CumotionWorldInterface,
                                                GraphBasedMotionPlanner,
                                                load_cumotion_robot)
    import warp as wp

    robot = SingleArticulation(prim_path=ROBOT_PRIM, name="pick_robot")
    robot.initialize()
    STATE["robot"] = robot
    STATE["dof_names"] = list(robot.dof_names)
    log(f"articulation {ROBOT_PRIM}: {len(STATE['dof_names'])} DOFs")

    STATE["device"] = wp.get_device("cuda:0")
    cumotion_robot = load_cumotion_robot(
        directory=ROBOT_CONFIG_PATH,
        urdf_filename=URDF_NAME, xrdf_filename=XRDF_NAME)
    STATE["cumotion_robot"] = cumotion_robot

    joints = list(cumotion_robot.controlled_joint_names)
    STATE["cspace_joints"] = joints
    STATE["dof_idx"] = [STATE["dof_names"].index(j) for j in joints]
    log(f"planned joints: {joints}")
    log(f"dof indices   : {STATE['dof_idx']}")

    rebuild_world([])
    return True


def robot_base_pose():
    """World -> robot-root transform as warp arrays (the FIXED rail)."""
    import warp as wp
    stage = omni.usd.get_context().get_stage()
    mat = omni.usd.get_world_transform_matrix(
        stage.GetPrimAtPath(ROBOT_ROOT_PATH))
    t = mat.ExtractTranslation()
    q = mat.ExtractRotationQuat()
    im = q.GetImaginary()
    with wp.ScopedDevice(STATE["device"]):
        pos = wp.array(np.array([[t[0], t[1], t[2]]], dtype=np.float32),
                       dtype=wp.vec3)
        quat = wp.array(np.array([[q.GetReal(), im[0], im[1], im[2]]],
                                 dtype=np.float32), dtype=wp.vec4)
    return pos, quat


def rebuild_world(sphere_points):
    """Recreate the collision world (cuMotion has no per-obstacle removal)."""
    from isaacsim.robot_motion.cumotion import (CumotionWorldInterface,
                                                GraphBasedMotionPlanner)
    import warp as wp

    world = CumotionWorldInterface(world_to_robot_base=robot_base_pose(),
                                   device=STATE["device"])
    n = 0
    if len(sphere_points):
        n = len(sphere_points)
        paths = [f"/collision/pcd/{i}" for i in range(n)]
        with wp.ScopedDevice(STATE["device"]):
            radii = wp.array(np.full(n, VOXEL_SIZE / 2, dtype=np.float32),
                             dtype=wp.float32)
            scales = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3)
            tols = wp.array(np.full(n, SAFETY_MARGIN, dtype=np.float32),
                            dtype=wp.float32)
            pos = wp.array(sphere_points.astype(np.float32), dtype=wp.vec3)
            quat = wp.array(np.tile([1.0, 0, 0, 0], (n, 1)).astype(np.float32),
                            dtype=wp.vec4)
            en = wp.array(np.ones(n, dtype=np.int32), dtype=wp.int32)
        world.add_spheres(prim_paths=paths, radii=radii, scales=scales,
                          safety_tolerances=tols, poses=(pos, quat),
                          enabled_array=en)
    world.world_view.update()

    STATE["world"] = world
    STATE["spheres"] = n
    STATE["planner"] = GraphBasedMotionPlanner(
        cumotion_robot=STATE["cumotion_robot"],
        cumotion_world_interface=world, tool_frame=TOOL_FRAME)
    return n


def dof_limits():
    """(lower, upper) per articulation DOF, or (None, None) if unavailable.

    These are the PHYSICS limits from the USD, which are not necessarily
    the URDF limits cuMotion planned against. A goal outside them can be
    clamped or rejected by PhysX, producing a command that never moves.
    """
    robot = STATE["robot"]
    for name in ("get_dof_limits", "get_joint_limits", "dof_limits"):
        fn = getattr(robot, name, None)
        try:
            lim = fn() if callable(fn) else fn
            if lim is None:
                continue
            arr = np.asarray(lim, dtype=np.float64)
            arr = arr.reshape(-1, 2) if arr.ndim == 3 else arr
            if arr.ndim == 2 and arr.shape[1] == 2:
                return arr[:, 0], arr[:, 1]
        except Exception:
            continue
    return None, None


async def self_test_jog():
    """Nudge one joint exactly the way the bridge's /api/jog does.

    If THIS does not move the robot, the articulation handle is the
    problem — most likely a second SingleArticulation on the same prim
    while the bridge still holds one. Stop the bridge and retry.
    """
    idx = STATE["dof_idx"][1]              # shoulder_pan
    q0 = np.array(STATE["robot"].get_joint_positions(), dtype=np.float64)
    tgt = q0.copy()
    tgt[idx] += 0.2
    STATE["robot"].apply_action(ArticulationAction(joint_positions=tgt))
    await tick(120)
    q1 = np.array(STATE["robot"].get_joint_positions(), dtype=np.float64)
    moved = float(abs(q1[idx] - q0[idx]))
    log(f"self-test jog: joint[{idx}] {q0[idx]:+.4f} -> {q1[idx]:+.4f} "
        f"(moved {moved:.4f}) -> {'OK' if moved > 0.02 else 'DID NOT MOVE'}")
    if moved <= 0.02:
        log("  The articulation is not accepting commands from THIS script.")
        log("  If isaac_sim_bridge.py is running it holds its own")
        log("  SingleArticulation on the same prim — stop it and re-run.")
        return False
    # put it back
    STATE["robot"].apply_action(ArticulationAction(joint_positions=q0))
    await tick(60)
    return True


def check_goal_limits(goal):
    """Warn about any planned joint outside the physics limits."""
    bbox_sanity_report()
    tool_q = frames_report()
    tl_p0, _ = prim_pose(TOOL_PRIM)
    tcp_p0, _ = prim_pose(GRIPPER_TCP_PRIM)
    if tool_q is not None and tl_p0 is not None and tcp_p0 is not None:
        down_q, approach = derive_down_quat(tl_p0, tool_q, tcp_p0)
        if down_q is not None:
            facing = ("UP" if approach[2] > 0.7 else
                      "DOWN" if approach[2] < -0.7 else "SIDEWAYS")
            log(f"gripper approach axis = ({approach[0]:+.3f}, "
                f"{approach[1]:+.3f}, {approach[2]:+.3f}) -> pointing {facing}")
            log(f"derived DOWNWARD quat = ({down_q[0]:+.4f}, {down_q[1]:+.4f}, "
                f"{down_q[2]:+.4f}, {down_q[3]:+.4f})")
            STATE["tool_quat"] = down_q
    tl_p, _ = prim_pose(TOOL_PRIM)
    tcp_p, _ = prim_pose(GRIPPER_TCP_PRIM)
    src_p = tcp_p if tcp_p is not None else prim_pose(FINGER_PAD_PRIM)[0]
    if tl_p is not None and src_p is not None:
        dist = float(np.linalg.norm(src_p - tl_p))
        if dist >= MIN_CREDIBLE_TCP_OFFSET:
            STATE["tcp_offset"] = dist
            log(f"using measured TCP offset {dist:.5f} m")
        else:
            log(f"measured offset {dist:.5f} m not credible — keeping "
                f"configured {GRIPPER_TCP_OFFSET:.5f}")

    lo, hi = dof_limits()
    if lo is None:
        log("  (could not read physics DOF limits)")
        return True
    ok = True
    for slot, dof in enumerate(STATE["dof_idx"]):
        v = goal[slot]
        if v < lo[dof] - 1e-6 or v > hi[dof] + 1e-6:
            ok = False
            log(f"  *** joint[{dof}] '{STATE['dof_names'][dof]}' goal "
                f"{v:+.3f} is OUTSIDE physics limits "
                f"[{lo[dof]:+.3f}, {hi[dof]:+.3f}] — PhysX will clamp or "
                f"reject this command")
    return ok


def prim_pose(path):
    """(position, quaternion wxyz) of a prim in world, or (None, None)."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None, None
    m = omni.usd.get_world_transform_matrix(prim)
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()
    im = q.GetImaginary()
    return (np.array([t[0], t[1], t[2]], dtype=np.float64),
            np.array([q.GetReal(), im[0], im[1], im[2]], dtype=np.float64))


def frames_report():
    """Compare ee_link / robotiq_base_link / finger pad.

    The Lula-era constants (DOWNWARD_ORIENTATION, GRIPPER_TCP_OFFSET)
    were both measured from ee_link. cuMotion plans for the XRDF tool
    frame instead, so this prints what those numbers should actually be.
    Returns the tool's current world quaternion.
    """
    log("frame report:")
    ee_p, ee_q = prim_pose(EE_LINK_PRIM)
    tl_p, tl_q = prim_pose(TOOL_PRIM)
    tcp_p, _ = prim_pose(GRIPPER_TCP_PRIM)
    pad_p, _ = prim_pose(FINGER_PAD_PRIM)

    for name, pos, quat in (("ee_link", ee_p, ee_q),
                            (TOOL_FRAME, tl_p, tl_q),
                            ("gripper_tcp", tcp_p, None),
                            ("finger pad", pad_p, None)):
        if pos is None:
            log(f"  {name:20s} PRIM NOT FOUND")
            continue
        line = f"  {name:20s} pos ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})"
        if quat is not None:
            line += (f"  quat ({quat[0]:+.4f}, {quat[1]:+.4f}, "
                     f"{quat[2]:+.4f}, {quat[3]:+.4f})")
        log(line)

    if ee_p is not None and tl_p is not None:
        log(f"  ee_link -> {TOOL_FRAME}: offset {np.linalg.norm(tl_p-ee_p):.4f} m")
        # Are the two frames aligned? |dot| of the quaternions is 1 if so.
        d = abs(float(np.dot(ee_q, tl_q)))
        log(f"  frame alignment |dot(q_ee, q_tool)| = {d:.4f} "
            f"({'ALIGNED' if d > 0.999 else 'ROTATED — the ee_link-derived '
               'DOWNWARD_ORIENTATION is NOT valid for this tool frame'})")

    src_p = tcp_p if tcp_p is not None else pad_p
    src_name = "gripper_tcp" if tcp_p is not None else "finger pad"
    if tl_p is not None and src_p is not None:
        # Axial distance, not dz — dz only equals the offset when the
        # gripper points straight down, which it need not right now.
        dist = float(np.linalg.norm(src_p - tl_p))
        log(f"  {TOOL_FRAME} -> {src_name} = {dist:.5f} m "
            f"(dz {abs(src_p[2]-tl_p[2]):.5f})")
        log(f"    configured GRIPPER_TCP_OFFSET = {GRIPPER_TCP_OFFSET:.5f}")
        if dist < MIN_CREDIBLE_TCP_OFFSET:
            log("    NOT CREDIBLE — keeping the configured value")

    if tl_q is not None:
        log(f"  tool orientation NOW = ({tl_q[0]:+.4f}, {tl_q[1]:+.4f}, "
            f"{tl_q[2]:+.4f}, {tl_q[3]:+.4f})")
        log(f"  hardcoded DOWNWARD_QUAT = {tuple(normalize_quat(DOWNWARD_QUAT).round(4))}")
    return tl_q


# ─── Point cloud ─────────────────────────────────────────────

async def capture_pointcloud():
    """Merged world-frame points from the 4 corner cameras."""
    import omni.replicator.core as rep
    stage = omni.usd.get_context().get_stage()

    if "annots" not in STATE:
        STATE["annots"] = {}
        for path, res in POINTCLOUD_CAMERAS.items():
            if not stage.GetPrimAtPath(path).IsValid():
                log(f"  missing camera {path}")
                continue
            rp = rep.create.render_product(path, res)
            a = rep.AnnotatorRegistry.get_annotator(
                "pointcloud", init_params={"includeUnlabelled": True})
            a.attach(rp)
            STATE["annots"][path] = a
            log(f"  attached {path}")

    for _ in range(5):
        await rep.orchestrator.step_async()

    clouds = []
    for path, a in STATE["annots"].items():
        d = a.get_data()
        if not d or "data" not in d or len(d["data"]) == 0:
            continue
        pts = np.asarray(d["data"]).reshape(-1, 3)
        pts = pts[np.linalg.norm(pts, axis=1) > 0.01]
        if len(pts):
            clouds.append(pts)
    return np.vstack(clouds) if clouds else np.empty((0, 3))


def voxelise(points, size):
    if not len(points):
        return points
    keys = np.floor(points / size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


async def build_collision_world(target_xyz):
    """Capture, carve around the target, voxelise, load into cuMotion."""
    raw = await capture_pointcloud()
    if not len(raw):
        log("no point cloud — planning with an EMPTY world")
        rebuild_world(np.empty((0, 3)))
        return

    lo, hi = raw.min(axis=0), raw.max(axis=0)
    log(f"cloud {len(raw)} pts  x[{lo[0]:+.2f},{hi[0]:+.2f}] "
        f"y[{lo[1]:+.2f},{hi[1]:+.2f}] z[{lo[2]:+.2f},{hi[2]:+.2f}]")

    # The part being picked must not be its own obstacle.
    d = np.linalg.norm(raw[:, :2] - np.array(target_xyz[:2]), axis=1)
    kept = raw[d > CARVE_RADIUS]
    log(f"carved {int((d <= CARVE_RADIUS).sum())} pts around the target")

    vox = voxelise(kept, VOXEL_SIZE)
    if len(vox) > MAX_SPHERES:
        log(f"NOTE: {len(vox)} voxels exceeds MAX_SPHERES={MAX_SPHERES}; "
            f"raise VOXEL_SIZE instead of sampling — random subsets leave "
            f"holes the arm can plan straight through")
        vox = vox[np.random.default_rng(0).choice(
            len(vox), MAX_SPHERES, replace=False)]
    n = rebuild_world(vox)
    log(f"collision world: {n} spheres @ {VOXEL_SIZE * 100:.0f} cm")


# ─── Motion ──────────────────────────────────────────────────

def current_q():
    q = STATE["robot"].get_joint_positions()
    return np.array([float(q[i]) for i in STATE["dof_idx"]], dtype=np.float64)


def plan_to(tcp_xyz, quat=None):
    """Plan to a TCP pose. Returns a trajectory or None."""
    if quat is None:
        quat = STATE.get("tool_quat") if USE_CURRENT_ORIENTATION else None
        quat = quat if quat is not None else DOWNWARD_QUAT
    pos = np.array(tcp_xyz, dtype=np.float64).copy()
    pos[2] += STATE.get("tcp_offset", GRIPPER_TCP_OFFSET)
    path = STATE["planner"].plan_to_pose_target(
        q_initial=current_q(), position=pos,
        orientation=normalize_quat(quat))
    if path is None:
        log(f"  NO PATH to tool ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) "
            f"[{STATE['spheres']} spheres]")
        return None
    n = len(STATE["cspace_joints"])
    return path.to_minimal_time_joint_trajectory(
        max_velocities=MAX_VEL[:n], max_accelerations=MAX_ACC[:n],
        robot_joint_space=STATE["dof_names"],
        active_joints=STATE["cspace_joints"])


def full_targets(cspace_q, finger=None):
    """Expand a cspace config into a full articulation target vector."""
    t = STATE["robot"].get_joint_positions().copy()
    for slot, dof in enumerate(STATE["dof_idx"]):
        t[dof] = cspace_q[slot]
    if finger is not None:
        for i, name in enumerate(STATE["dof_names"]):
            if "inner_finger_joint" in name:
                t[i] = 0.0
            elif "finger_joint" in name or "knuckle_joint" in name:
                t[i] = finger
    return t


def goal_of(traj):
    st = traj.get_target_state(float(traj.duration))
    p = np.asarray(st.joints.positions).flatten()
    idx = st.joints.position_indices
    if idx is not None:
        by_dof = {int(d): p[s] for s, d in enumerate(np.asarray(idx).flatten())}
        return np.array([by_dof.get(d, 0.0) for d in STATE["dof_idx"]])
    return p[:len(STATE["dof_idx"])]


async def tick(n):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


async def run_trajectory(traj, finger=None, mode=None):
    """Execute a trajectory. Returns the max joint change achieved."""
    mode = mode or EXEC_MODE
    q0 = current_q()
    goal = goal_of(traj)
    log(f"  duration {float(traj.duration):.2f}s")
    log(f"    q_start {np.round(q0, 3).tolist()}")
    log(f"    q_goal  {np.round(goal, 3).tolist()}")
    check_goal_limits(goal)

    async def stream():
        t = 0.0
        while t < float(traj.duration):
            st = traj.get_target_state(t)
            if st is not None and st.joints.positions is not None:
                p = np.asarray(st.joints.positions).flatten()
                idx = st.joints.position_indices
                tgt = STATE["robot"].get_joint_positions().copy()
                if idx is not None:
                    for s_, d_ in enumerate(np.asarray(idx).flatten()):
                        tgt[int(d_)] = p[s_]
                else:
                    tgt[:len(p)] = p
                if finger is not None:
                    tgt = full_targets(
                        [tgt[d] for d in STATE["dof_idx"]], finger)
                STATE["robot"].apply_action(
                    ArticulationAction(joint_positions=tgt))
            await omni.kit.app.get_app().next_update_async()
            t += TRAJ_DT
        await tick(SETTLE_FRAMES)

    async def direct():
        STATE["robot"].apply_action(
            ArticulationAction(joint_positions=full_targets(goal, finger)))
        await tick(max(SETTLE_FRAMES, int(float(traj.duration) * 60)))

    if mode in ("stream", "both"):
        await stream()
        moved = float(np.abs(current_q() - q0).max())
        log(f"    stream -> max joint change {moved:.4f}")
        if moved > 1e-3 or mode == "stream":
            return moved

        log("    stream moved nothing; trying DIRECT")

    await direct()
    moved = float(np.abs(current_q() - q0).max())
    log(f"    direct -> max joint change {moved:.4f}")
    return moved


def probe_reachability(target):
    """Ask cuMotion which heights above the target it can actually reach.

    Scale-dependent constants (approach heights, bin clearances, workspace
    bounds) were tuned before the environment was rescaled, so rather than
    trusting any of them, this asks the planner directly. It answers the
    only question that matters: can the arm get to this part, and at what
    height does it stop being able to?
    """
    log("reachability probe at the target XY:")
    quat = STATE.get("tool_quat")
    if quat is None:
        quat = normalize_quat(DOWNWARD_QUAT)
    q0 = current_q()
    offset = STATE.get("tcp_offset", GRIPPER_TCP_OFFSET)

    reachable = []
    for dz in (0.0, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60):
        tcp = np.array([target[0], target[1], target[2] + dz])
        tool = tcp.copy()
        tool[2] += offset
        found = STATE["planner"].plan_to_pose_target(
            q_initial=q0, position=tool, orientation=quat) is not None
        reachable.append((dz, found))
        log(f"    +{dz:.2f} m above part  (tool z={tool[2]:+.3f})  "
            f"{'REACHABLE' if found else 'no path'}")

    ok = [dz for dz, f in reachable if f]
    if not ok:
        log("  NOTHING reachable above this part. Either the part is outside")
        log("  the arm's envelope, or the collision world is blocking every")
        log("  approach. Re-run with MAX_SPHERES=0 to separate the two.")
    else:
        log(f"  reachable heights: {[f'+{d:.2f}' for d in ok]}")
        log(f"  -> use an approach height in that range; the configured "
            f"APPROACH_HEIGHT is {APPROACH_HEIGHT:.2f}")
    return ok


async def move_to(tcp_xyz, finger=None, label=""):
    log(f"{label} -> ({tcp_xyz[0]:+.3f}, {tcp_xyz[1]:+.3f}, {tcp_xyz[2]:+.3f})")
    traj = plan_to(tcp_xyz)
    if traj is None:
        return False
    return await run_trajectory(traj, finger=finger) > 1e-3


async def set_gripper(value, label=""):
    log(f"{label} gripper -> {value}")
    STATE["robot"].apply_action(
        ArticulationAction(joint_positions=full_targets(current_q(), value)))
    await tick(SETTLE_FRAMES)


# ─── Target selection ────────────────────────────────────────

def _bbox_top_centre(prim):
    """(x, y, z_top) of a prim's world bbox, or None if it has no extent."""
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                             [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if box.IsEmpty():
        return None, None
    lo, hi = box.GetMin(), box.GetMax()
    # Grasp the TOP surface — that is what the gripper descends onto.
    return ((float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2),
             float(hi[2])),
            (float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2])))


def list_candidate_parts():
    """Print pickable prims under PARTS_CONTAINER with their grasp points."""
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    container = stage.GetPrimAtPath(PARTS_CONTAINER)
    if not container.IsValid():
        log(f"PARTS_CONTAINER not found: {PARTS_CONTAINER}")
        return
    log(f"candidate parts under {PARTS_CONTAINER}:")
    for child in container.GetChildren():
        if not child.IsA(UsdGeom.Imageable):
            continue
        xyz, size = _bbox_top_centre(child)
        if xyz is None:
            continue
        log(f"  {str(child.GetPath())}")
        log(f"      grasp ({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f})  "
            f"size {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f}")


def bbox_sanity_report(paths=None):
    """Compare prim TRANSLATION against ComputeWorldBound for key prims.

    After the environment rescale, bounding boxes come back in the wrong
    space (sizes 1000x too large), while transforms stay correct. This
    prints both so the split is visible: anything derived from a bbox
    is suspect, anything derived from a translation is trustworthy.
    """
    from pxr import Usd, UsdGeom
    stage = omni.usd.get_context().get_stage()
    paths = paths or [PARTS_CONTAINER, "/World/box_840", TOOL_PRIM]

    log("bbox sanity (translation vs ComputeWorldBound):")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                             [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            log(f"  {path}: NOT FOUND")
            continue
        t = omni.usd.get_world_transform_matrix(prim).ExtractTranslation()
        log(f"  {path}")
        log(f"      translate ({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})")
        try:
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if box.IsEmpty():
                log("      bbox EMPTY")
            else:
                lo, hi = box.GetMin(), box.GetMax()
                size = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
                flag = "  <-- IMPLAUSIBLE" if max(size) > 10.0 else ""
                log(f"      bbox top_z {hi[2]:+.3f}  size "
                    f"{size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f}{flag}")
        except Exception as e:
            log(f"      bbox failed: {e}")
    log("  -> use translations; treat bbox-derived heights as unusable")


def _plausible(xyz):
    """True if a grasp point lies inside the physical workspace."""
    return all(lo <= v <= hi for v, (lo, hi) in zip(xyz, WORKSPACE_BOUNDS))


def selected_target():
    """Grasp point: TARGET_XYZ, else TARGET_PRIM, else the Stage selection."""
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()

    if TARGET_XYZ is not None:
        xyz = tuple(float(v) for v in TARGET_XYZ)
        log(f"target from TARGET_XYZ: ({xyz[0]:+.3f}, {xyz[1]:+.3f}, "
            f"{xyz[2]:+.3f})")
        if not _plausible(xyz):
            log("  WARNING: outside the workspace bounds — check the value")
        return xyz, "TARGET_XYZ"

    path = TARGET_PRIM.strip()
    source = "TARGET_PRIM"
    if not path:
        paths = ctx.get_selection().get_selected_prim_paths()
        path = paths[0] if paths else ""
        source = "Stage selection"

    if not path:
        log("No part chosen. Either set TARGET_PRIM at the top of this "
            "file, or select a part prim in the Stage tree.")
        list_candidate_parts()
        return None, None

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        log(f"prim not found: {path}")
        list_candidate_parts()
        return None, None

    xyz, size = _bbox_top_centre(prim)
    if xyz is None:
        log(f"prim has no bounding box: {path}")
        return None, None

    log(f"target from {source}: {path}")
    log(f"  grasp point ({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f})  "
        f"size {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f}")

    if not _plausible(xyz) or max(size) > 1.0:
        log("  *** NOT A USABLE GRASP POINT ***")
        log("  ComputeWorldBound returns prototype-space values for the")
        log("  instanceable part prims in this scene, so USD lookups give")
        log("  coordinates and sizes that are not world metres. Set")
        log("  TARGET_XYZ instead, copying the coordinates from the")
        log("  perception log ([scan] lines).")
        return None, None
    return xyz, path


# ─── Sequence ────────────────────────────────────────────────

async def main():
    print("\n" + "=" * 60)
    print("  cuMotion pick-and-place (standalone)")
    print("=" * 60)

    if not omni.timeline.get_timeline_interface().is_playing():
        log("TIMELINE IS STOPPED — press Play, then re-run. apply_action "
            "only sets drive targets; physics must be stepping.")
        return

    target, _ = selected_target()
    if target is None:
        return

    if "robot" not in STATE:
        setup()

    lo, hi = dof_limits()
    if lo is not None:
        log("physics DOF limits for the planned joints:")
        for slot, dof in enumerate(STATE["dof_idx"]):
            log(f"  [{dof}] {STATE['dof_names'][dof]:24s} "
                f"[{lo[dof]:+.3f}, {hi[dof]:+.3f}]")

    if not await self_test_jog():
        return

    log("building collision world from 4 cameras...")
    await build_collision_world(target)

    ok_heights = probe_reachability(target)
    if not ok_heights:
        log("aborting: no reachable approach height for this target")
        return

    # Prefer the configured height when the planner says it works;
    # otherwise take the lowest height that does.
    approach_dz = (APPROACH_HEIGHT if APPROACH_HEIGHT in ok_heights
                   else min(ok_heights, key=lambda d: abs(d - APPROACH_HEIGHT)))
    if approach_dz != APPROACH_HEIGHT:
        log(f"APPROACH_HEIGHT {APPROACH_HEIGHT:.2f} not reachable — "
            f"using +{approach_dz:.2f} instead")

    above = (target[0], target[1], target[2] + approach_dz)
    grasp = (target[0], target[1], target[2] + GRASP_CLEARANCE)
    lifted = (target[0], target[1], target[2] + LIFT_HEIGHT)
    place_above = (PLACE_XYZ[0], PLACE_XYZ[1], PLACE_XYZ[2] + APPROACH_HEIGHT)

    await set_gripper(FINGER_OPEN, "1. open")
    if not await move_to(above, FINGER_OPEN, "2. approach"):
        log("FAILED at approach"); return
    if not await move_to(grasp, FINGER_OPEN, "3. descend"):
        log("FAILED at descend"); return
    await set_gripper(FINGER_CLOSE, "4. close")
    if not await move_to(lifted, FINGER_CLOSE, "5. lift"):
        log("FAILED at lift"); return
    if not await move_to(place_above, FINGER_CLOSE, "6. transit"):
        log("FAILED at transit"); return
    if not await move_to(PLACE_XYZ, FINGER_CLOSE, "7. place"):
        log("FAILED at place"); return
    await set_gripper(FINGER_OPEN, "8. release")
    await move_to(place_above, FINGER_OPEN, "9. retract")

    print("=" * 60)
    log("DONE")


asyncio.ensure_future(main())
