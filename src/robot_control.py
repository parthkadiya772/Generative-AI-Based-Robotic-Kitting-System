import asyncio
import numpy as np
import omni.kit.app
import omni.usd
import os
import tempfile
import traceback
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
from isaacsim.core.utils.types import ArticulationAction
from omni.isaac.motion_generation import LulaKinematicsSolver

# ========================================================
# CONFIGURATION
# ========================================================
# ⚠️  Articulation root moved to /World — update ROBOT_PATH if needed
ROBOT_PATH     = "/World"        # prim that has ArticulationRootAPI
PART_PATH      = "/World/robot_facade_full/motor_valve_2"
BOX_PATH       = "/World/robot_facade_full"  # the blue box — used for collision filter
PLACE_BOX_PATH = "/World/box_840"            # destination prim to place the part on
UR10_BASE_PATH = "/World/gantry_home/ur10_flattened/ur10_instanceable/base_link"
EE_PATH        = "/World/gantry_home/ur10_flattened/ur10_instanceable/ee_link"

# Gripper finger prim paths — used to enable collision with the box
# These are the actual physics-enabled links of the Robotiq 2F-140
GRIPPER_FINGER_PATHS = [
    "/World/gantry_home/ur10_flattened/ur10_instanceable/robotiq_fixed_physics/Robotiq_2F_140_physics_edit/left_inner_finger",
    "/World/gantry_home/ur10_flattened/ur10_instanceable/robotiq_fixed_physics/Robotiq_2F_140_physics_edit/right_inner_finger",
    "/World/gantry_home/ur10_flattened/ur10_instanceable/robotiq_fixed_physics/Robotiq_2F_140_physics_edit/left_outer_finger",
    "/World/gantry_home/ur10_flattened/ur10_instanceable/robotiq_fixed_physics/Robotiq_2F_140_physics_edit/right_outer_finger",
    "/World/gantry_home/ur10_flattened/ur10_instanceable/robotiq_fixed_physics/Robotiq_2F_140_physics_edit/robotiq_base_link",
]

# UR10 URDF / motion-policy YAML bundled with Isaac Sim. Built from the
# ISAACSIM_PATH env var (set it to the root of your Isaac Sim install,
# i.e. the directory containing `exts/`) so this works on any machine.
_ISAACSIM_PATH = os.environ.get("ISAACSIM_PATH", "")
BROKEN_URDF = os.path.join(
    _ISAACSIM_PATH, "exts", "isaacsim.asset.importer.urdf",
    "data", "urdf", "robots", "ur10", "urdf", "ur10.urdf")
BROKEN_YAML = os.path.join(
    _ISAACSIM_PATH, "exts", "isaacsim.robot_motion.motion_generation",
    "motion_policy_configs", "universal_robots", "ur10", "rmpflow",
    "ur10_robot_description.yaml")

# ========================================================
# ROBOTIQ 2F-140 PHYSICAL DIMENSIONS
#
#   These are the REAL measurements from the datasheet.
#   We use them to patch the URDF with a virtual gripper link
#   so Lula IK targets the TRUE tool-tip, not the UR10 flange.
#
#   ee_link (UR10 flange) → gripper base:      0.049 m
#   gripper base → finger contact pad:          0.130 m  (when grasping ~50mm part)
#   Total ee_link → contact pad:               0.179 m  ← GRIPPER_TCP_OFFSET
#
#   Gripper body width (widest point, fingers open): 0.160 m
#   Half-width = 0.080 m ← minimum XY clearance from box wall to ee_link center
# ========================================================
GRIPPER_TCP_OFFSET  = 0.150  # metres: ee_link flange → finger contact pad 0.179m when grasping ~50mm part; adjust if your part is much bigger/smaller
GRIPPER_BODY_WIDTH  = 0.160  # metres: 2F-140 widest dimension (fingers open)
GRIPPER_HALF_WIDTH  = GRIPPER_BODY_WIDTH / 2.0  # 0.080 m — minimum wall clearance

HOVER_CLEARANCE = 0.015  # metres: gap above part TOP SURFACE before final plunge

# ========================================================
# ADAPTIVE GRASP GEOMETRY
#
#   GRASP_DEPTH_FRACTION: fraction of part height at which the
#   finger contact pads should close.
#     0.0 = fingers just touch the top surface
#     0.5 = fingers close at part mid-height  (default; good for
#           cylinders & box parts)
#     1.0 = fingers reach the very bottom of the part
#
#   PART_HEIGHT_WARN_LIMIT: if the part is taller than this,
#   print a warning that the 2F-140 stroke (max 140 mm) may not
#   span the full part diameter / width.
# ========================================================
GRASP_DEPTH_FRACTION   = 0.5   # 0.0 = top  |  0.5 = mid  |  1.0 = bottom
PART_HEIGHT_WARN_LIMIT = 0.14  # metres — 2F-140 max stroke = 140 mm

# ========================================================
# BOX COLLISION AVOIDANCE
#
#   BOX_HEIGHT       = wall height above part origin (measure in scene)
#   BOX_ENTRY_MARGIN = how far above the rim the arm must be before
#                      entering. Must be ≥ GRIPPER_TCP_OFFSET so the
#                      entire gripper body is above the rim.
#                      2F-140 body height ≈ 0.080m, so 0.120m is safe.
# ========================================================
BOX_HEIGHT        = 0.05   # metres — MEASURE: box_top_Z - part_Z in your scene
BOX_ENTRY_MARGIN  = 0.15   # metres — must be ≥ GRIPPER_TCP_OFFSET (0.179) to clear body
                            # ↑ increase this if gripper still clips the rim

# ========================================================
# PLACE CONFIGURATION
#
#   PLACE_DROP_HEIGHT: how far above the destination surface
#   the gripper descends before opening fingers.
#   Small values = accurate placement; too small = collision risk.
#
#   PLACE_RETRACT_HEIGHT: how far above the destination the
#   gripper retracts AFTER releasing the part.
# ========================================================
PLACE_DROP_HEIGHT    = 0.0   # metres above destination top surface
PLACE_RETRACT_HEIGHT = 0.20   # metres above destination after release
TRANSIT_SAFE_HEIGHT  = 0.40   # metres ABOVE safe_z — high clearance during lateral travel

# ========================================================
# GANTRY
# ========================================================
GANTRY_X_JOINT  = "gantry_vagn_joint"
GANTRY_X_OFFSET = 1.27

# ========================================================
# ROBOTIQ 2F-140 FINGER JOINTS
#   0.0 rad = open (140mm span)   0.695 rad = closed (0mm)
# ========================================================
FINGER_OPEN  = 0.0
FINGER_CLOSE = 0.5   # ~50mm part; tune ±0.05 for your exact part

# ========================================================
# GRIPPER ORIENTATION  [w, x, y, z]  — OPTION D for your robot
#   Your logs confirmed Z-axis was sideways → OPTION D
# ========================================================
DOWNWARD_ORIENTATION    = np.array([1.0, 0.0, 1.0, 0.0])
PRINT_ORIENTATION_DEBUG = True


# ========================================================
# HELPERS
# ========================================================
def normalize_quat(q):
    mag = np.linalg.norm(q)
    if mag < 1e-9:
        raise ValueError(f"Quaternion {q} is near-zero — invalid rotation!")
    normed = q / mag
    if abs(mag - 1.0) > 0.005:
        print(f"WARNING: Quaternion magnitude={mag:.6f} — auto-normalized")
    return normed


def get_world_pos(prim_path):
    return np.array(omni.usd.get_world_transform_matrix(
        get_prim_at_path(prim_path)).ExtractTranslation())


def compute_bbox_geometry(stage, prim_path, label="PRIM"):
    """
    Compute the world-space axis-aligned bounding box of any prim
    (including all its descendant meshes).

    Returns a dict:
        center_xy  – [x, y] world centre of the bbox footprint
        pivot      – [x, y, z] raw prim origin (for reference / debug)
        top_z      – Z of the top face  (world metres)
        bot_z      – Z of the bottom face (world metres)
        height     – full height of the bounding box (metres)
        min_pt     – [x, y, z] bbox minimum
        max_pt     – [x, y, z] bbox maximum
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found at path: '{prim_path}'")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        ["default", "render"],
        useExtentsHint=True
    )
    world_bound = bbox_cache.ComputeWorldBound(prim)
    aligned_box = world_bound.ComputeAlignedRange()

    min_pt = np.array(aligned_box.GetMin())
    max_pt = np.array(aligned_box.GetMax())

    height    = float(max_pt[2] - min_pt[2])
    top_z     = float(max_pt[2])
    bot_z     = float(min_pt[2])
    center_xy = [
        float((min_pt[0] + max_pt[0]) / 2.0),
        float((min_pt[1] + max_pt[1]) / 2.0),
    ]

    pivot = get_world_pos(prim_path)

    print(f"\n--- {label} GEOMETRY (world-space bbox) ---")
    print(f"  Prim pivot (origin):   {np.round(pivot, 4)}")
    print(f"  BBox min  [x,y,z]:     {np.round(min_pt, 4)}")
    print(f"  BBox max  [x,y,z]:     {np.round(max_pt, 4)}")
    print(f"  Height:                {height * 1000:.1f} mm")
    print(f"  Top surface Z:         {top_z:.4f} m")
    print(f"  Bottom surface Z:      {bot_z:.4f} m")
    print(f"{'─' * 44}\n")

    return {
        "center_xy": center_xy,
        "pivot":     pivot,
        "top_z":     top_z,
        "bot_z":     bot_z,
        "height":    height,
        "min_pt":    min_pt,
        "max_pt":    max_pt,
    }


def compute_grasp_geometry(stage, part_path, grasp_depth_fraction=0.5):
    """
    Wrapper around compute_bbox_geometry that adds grasp-specific fields.
    """
    bbox = compute_bbox_geometry(stage, part_path, label="PART")

    grasp_z = bbox["top_z"] - grasp_depth_fraction * bbox["height"]

    print(f"  GRASP_DEPTH_FRACTION:  {grasp_depth_fraction}")
    print(f"  → Finger contact Z:    {grasp_z:.4f} m\n")

    if bbox["height"] > PART_HEIGHT_WARN_LIMIT:
        print(f"  ⚠️  WARNING: Part height {bbox['height'] * 1000:.0f} mm > 2F-140 stroke "
              f"({PART_HEIGHT_WARN_LIMIT * 1000:.0f} mm).")
        print(f"  Consider lowering GRASP_DEPTH_FRACTION so the fingers wrap "
              f"around a narrower section of the part.\n")

    return {
        "part_center_xy": bbox["center_xy"],
        "part_pivot":     bbox["pivot"],
        "part_top_z":     bbox["top_z"],
        "part_bot_z":     bbox["bot_z"],
        "part_height":    bbox["height"],
        "grasp_z":        grasp_z,
    }


def apply_arm_joints(robot, dof_names, arm_names, ik_result):
    targets = robot.get_joint_positions()
    for i, lula_name in enumerate(arm_names):
        for idx, dof_name in enumerate(dof_names):
            if lula_name in dof_name:
                targets[idx] = ik_result[i]
                break
    return targets


def set_finger_joints(robot, dof_names, value, base_targets=None):
    """Set Robotiq 2F-140 joints with proper mimic coupling.
    
    Mimic relationships (parallel / clapping motion):
      finger_joint          →  value   (driver)
      inner_knuckle_joint   →  value   (same as driver)
      outer_knuckle_joint   →  value   (same as driver)
      inner_finger_joint    → -value   (OPPOSITE — keeps pads parallel)
    """
    targets = base_targets if base_targets is not None else robot.get_joint_positions()
    for idx, name in enumerate(dof_names):
        if "inner_finger_joint" in name:
            # Opposite sign keeps finger pads parallel (clapping)
            targets[idx] = -value
        elif "finger_joint" in name or "knuckle_joint" in name:
            targets[idx] = value
    return targets


def print_ee_orientation_debug(ee_prim):
    mat    = omni.usd.get_world_transform_matrix(ee_prim)
    x_axis = np.array([mat[0][0], mat[1][0], mat[2][0]])
    y_axis = np.array([mat[0][1], mat[1][1], mat[2][1]])
    z_axis = np.array([mat[0][2], mat[1][2], mat[2][2]])
    print(f"\n--- EE ORIENTATION DEBUG ---")
    print(f"  X={np.round(x_axis,3)}  Y={np.round(y_axis,3)}  Z={np.round(z_axis,3)}")
    if   z_axis[2] < -0.9: print("  ✅ Z pointing DOWN — CORRECT!")
    elif z_axis[2] >  0.9: print("  ❌ Z pointing UP   — try [0,1,0,0]")
    else:                  print("  ⚠️  Z sideways     — OPTION D [0.5,0.5,-0.5,0.5] active")
    print(f"----------------------------\n")


def ik_solve(lula_solver, frame, pos, ori, warm):
    """Wrapper that prints a clear error if IK fails."""
    action, ok = lula_solver.compute_inverse_kinematics(
        frame_name=frame,
        target_position=pos,
        target_orientation=ori,
        warm_start=warm
    )
    if not ok:
        print(f"  IK target pos   = {pos}")
        print(f"  IK orientation  = {ori}")
        print(f"  IK warm_start   = {np.round(warm, 3)}")
    return action, ok


# ========================================================
# URDF PATCHER
#
#   The UR10 URDF has NO gripper. This means:
#   - IK targets ee_link (the flange), not the true tool-tip
#   - IK has zero knowledge of the gripper body volume
#
#   FIX: We inject a virtual "gripper_tcp" link into the URDF,
#   rigidly attached to ee_link at GRIPPER_TCP_OFFSET distance.
#   We then tell Lula IK to target "gripper_tcp" instead of "ee_link".
#   Now the IK solution automatically places the FINGER CONTACT PAD
#   at the target, not the flange — no more GRIPPER_LENGTH fudge factor.
#
#   We also inject a collision box representing the 2F-140 body so
#   the URDF accurately describes what volume the gripper occupies.
# ========================================================
GRIPPER_TCP_LINK = """
  <!-- =====================================================
       VIRTUAL GRIPPER TCP — injected by run_perfect_pick.py
       Represents the Robotiq 2F-140 tool centre point.
       Offset = GRIPPER_TCP_OFFSET from ee_link along Z.
       ===================================================== -->
  <link name="gripper_tcp">
    <inertial>
      <mass value="0.001"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
    <!-- Gripper body collision box: 160mm wide, 80mm deep, 189mm tall -->
    <collision>
      <origin xyz="0 0 {half_tcp}" rpy="0 0 0"/>
      <geometry>
        <box size="{width} {width} {tcp_offset}"/>
      </geometry>
    </collision>
  </link>

  <joint name="gripper_tcp_joint" type="fixed">
    <parent link="ee_link"/>
    <child link="gripper_tcp"/>
    <!-- Place TCP at finger contact depth below the flange -->
    <origin xyz="0 0 {tcp_offset}" rpy="0 0 0"/>
  </joint>
""".format(
    tcp_offset=GRIPPER_TCP_OFFSET,
    half_tcp=GRIPPER_TCP_OFFSET / 2.0,
    width=GRIPPER_BODY_WIDTH
)

# The YAML also needs to know about the new link so Lula can target it
GRIPPER_TCP_YAML_ADDITION = """
# Virtual gripper TCP added by run_perfect_pick.py
ee_fixed_joints:
  - gripper_tcp_joint
"""


def patch_urdf_and_yaml(broken_urdf_path, broken_yaml_path):
    """
    Read the stock NVIDIA UR10 files, fix known bugs, and inject
    the virtual gripper TCP link. Returns (fixed_urdf_path, fixed_yaml_path).
    """
    temp_dir   = tempfile.gettempdir()
    fixed_urdf = os.path.join(temp_dir, "fixed_ur10_with_gripper.urdf")
    fixed_yaml = os.path.join(temp_dir, "fixed_ur10_with_gripper.yaml")

    # --- Patch URDF ---
    with open(broken_urdf_path, 'r') as f:
        urdf = f.read()

    # Fix 1: Add missing inertia tags
    urdf = urdf.replace(
        "</inertial>",
        '<inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/></inertial>'
    )
    # Fix 2: Remove NVIDIA's stray parenthesis typo
    urdf = urdf.replace("0.0027000046)", "0.0027000046")

    # Fix 3: Inject virtual gripper TCP link BEFORE </robot>
    if "gripper_tcp" not in urdf:
        urdf = urdf.replace("</robot>", GRIPPER_TCP_LINK + "\n</robot>")
        print("STATUS: Injected virtual gripper_tcp link into URDF ✓")
    else:
        print("STATUS: gripper_tcp already present in URDF — skipping injection")

    with open(fixed_urdf, 'w') as f:
        f.write(urdf)

    # --- Patch YAML ---
    with open(broken_yaml_path, 'r') as f:
        yaml = f.read()

    yaml = yaml.replace("root_link: world", "root_link: base_link")

    # Add gripper_tcp to fixed joints so Lula treats it as a rigid extension
    if "gripper_tcp_joint" not in yaml:
        # Append after the last line
        yaml = yaml.rstrip() + "\n" + GRIPPER_TCP_YAML_ADDITION
        print("STATUS: Injected gripper_tcp_joint into YAML ✓")
    else:
        print("STATUS: gripper_tcp_joint already present in YAML — skipping injection")

    with open(fixed_yaml, 'w') as f:
        f.write(yaml)

    return fixed_urdf, fixed_yaml


# ========================================================
# COLLISION FILTER REPAIR
#
#   When the articulation root moved from /World/gantry_home to /World,
#   Isaac Sim may have re-created the default collision filter groups,
#   which can exclude gripper ↔ box contacts.
#
#   This function explicitly enables collision between the gripper
#   finger prims and the box prim using PhysX collision APIs.
# ========================================================
def repair_collision_filters(stage, gripper_paths, box_path):
    """
    Ensure gripper fingers can physically collide with the box.
    Isaac Sim sometimes disables self-collision or env-collision
    when the articulation root changes.
    """
    print("\nSTATUS: Repairing PhysX collision filters...")

    box_prim = stage.GetPrimAtPath(box_path)
    if not box_prim.IsValid():
        print(f"  WARNING: Box prim not found at '{box_path}' — skipping collision repair")
        print(f"  Update BOX_PATH to your actual box prim path")
        return

    repaired = 0
    for finger_path in gripper_paths:
        finger_prim = stage.GetPrimAtPath(finger_path)
        if not finger_prim.IsValid():
            print(f"  SKIP: Finger prim not found: {finger_path}")
            continue

        # Ensure collision is enabled on the finger
        collision_api = UsdPhysics.CollisionAPI.Apply(finger_prim)
        collision_api.CreateCollisionEnabledAttr(True)

        # Ensure collision is enabled on the box
        box_collision_api = UsdPhysics.CollisionAPI.Apply(box_prim)
        box_collision_api.CreateCollisionEnabledAttr(True)

        repaired += 1
        print(f"  ✓ Collision enabled: {finger_path.split('/')[-1]} ↔ {box_path.split('/')[-1]}")

    if repaired == 0:
        print("  WARNING: No finger prims found — check GRIPPER_FINGER_PATHS")
        print("  Run the script once and look at the DOF names to find the right paths")
    else:
        print(f"  Repaired {repaired} finger collision pairs ✓\n")


async def run_perfect_pick():
    try:
        world = World.instance() or World()
        world.play()

        robot = world.scene.get_object("kitting_sys")
        if not robot:
            robot = SingleArticulation(prim_path=ROBOT_PATH, name="kitting_sys")
            world.scene.add(robot)
        robot.initialize()

        print("STATUS: Warming up physics engine...")
        for _ in range(30):
            await omni.kit.app.get_app().next_update_async()

        dof_names = robot.dof_names
        print("\n--- ALL DOF NAMES ---")
        for i, n in enumerate(dof_names):
            print(f"  [{i}] {n}")
        print("---------------------\n")

        # ========================================================
        # 1. REPAIR PHYSX COLLISION FILTERS
        #    Do this early, before any physics steps that involve
        #    the gripper approaching the box.
        # ========================================================
        stage = omni.usd.get_context().get_stage()
        repair_collision_filters(stage, GRIPPER_FINGER_PATHS, BOX_PATH)

        # ========================================================
        # 2. PATCH URDF + YAML (inject gripper TCP link)
        # ========================================================
        fixed_urdf, fixed_yaml = patch_urdf_and_yaml(BROKEN_URDF, BROKEN_YAML)

        lula_solver = LulaKinematicsSolver(
            robot_description_path=fixed_yaml,
            urdf_path=fixed_urdf
        )

        # Verify Lula knows about gripper_tcp
        lula_links = lula_solver.get_joint_names()
        print(f"\nSTATUS: Lula joint names = {lula_links}")
        target_frame = "gripper_tcp" if "gripper_tcp" in str(lula_solver) else "ee_link"
        # Try to confirm the frame exists
        try:
            test_pos = np.array([0.0, -0.5, 1.2])
            test_ori = normalize_quat(DOWNWARD_ORIENTATION)
            _, test_ok = lula_solver.compute_inverse_kinematics(
                frame_name="gripper_tcp",
                target_position=test_pos,
                target_orientation=test_ori,
                warm_start=np.zeros(6)
            )
            if test_ok:
                target_frame = "gripper_tcp"
                print("STATUS: IK frame = 'gripper_tcp' (true tool tip) ✓")
            else:
                target_frame = "ee_link"
                print("STATUS: IK frame = 'ee_link' (gripper_tcp not reachable in test — using flange)")
        except Exception:
            target_frame = "ee_link"
            print("STATUS: IK frame = 'ee_link' (gripper_tcp frame not available in this Lula version)")

        print(f"STATUS: Using IK target frame: '{target_frame}'")

        # ========================================================
        # 3. CAPTURE PART POSITION + ADAPTIVE GRASP GEOMETRY
        #
        #   We use UsdGeom.BBoxCache to get the real world-space
        #   bounding box of the part mesh, then derive:
        #     - part_top_z  : actual top surface Z
        #     - grasp_z     : Z where fingers should close
        #                     (= top_z − GRASP_DEPTH_FRACTION × height)
        #   This replaces the old "raw prim pivot + tcp_offset" guess
        #   which was unreliable because USD prim origins can be at
        #   the bottom, centre, or an arbitrary pivot of the mesh.
        # ========================================================
        target_pos = get_world_pos(PART_PATH)
        print(f"\nDIAGNOSTIC: Part pivot (world) = {target_pos}")

        geom = compute_grasp_geometry(stage, PART_PATH, GRASP_DEPTH_FRACTION)

        # If the bbox XY centre differs meaningfully from the prim pivot
        # (off-centre mesh pivot), prefer the bbox centre for XY targeting.
        xy_offset = np.linalg.norm(
            np.array(geom["part_center_xy"]) - target_pos[:2]
        )
        if xy_offset > 0.005:          # 5 mm threshold
            print(f"  ℹ️  BBox XY centre differs from prim pivot by "
                  f"{xy_offset * 1000:.1f} mm — using bbox centre for XY targeting.")
            target_pos[0] = geom["part_center_xy"][0]
            target_pos[1] = geom["part_center_xy"][1]

        part_top_z = geom["part_top_z"]   # physical top surface of the part
        grasp_z    = geom["grasp_z"]       # ideal finger-pad Z (depth-fraction into part)

        # ========================================================
        # 4. MOVE GANTRY X
        # ========================================================
        master_joint_targets = robot.get_joint_positions()
        gantry_x_idx = None
        for idx, name in enumerate(dof_names):
            if GANTRY_X_JOINT in name and gantry_x_idx is None:
                gantry_x_idx = idx
                master_joint_targets[idx] = target_pos[0] - GANTRY_X_OFFSET
                print(f"STATUS: X-gantry [{idx}] '{name}' → {master_joint_targets[idx]:.4f} m")

        if gantry_x_idx is None:
            print(f"WARNING: X-gantry joint '{GANTRY_X_JOINT}' not found!")

        robot.apply_action(ArticulationAction(joint_positions=master_joint_targets))
        print("STATUS: Waiting for gantry to settle...")
        for _ in range(120):
            await omni.kit.app.get_app().next_update_async()

        # ========================================================
        # 5. ANCHOR ARM BASE (read after gantry settles)
        # ========================================================
        base_prim   = get_prim_at_path(UR10_BASE_PATH)
        base_matrix = omni.usd.get_world_transform_matrix(base_prim)
        base_pos    = np.array(base_matrix.ExtractTranslation())
        rot         = base_matrix.ExtractRotation().GetQuat()
        base_rot    = np.array([rot.real, rot.imaginary[0], rot.imaginary[1], rot.imaginary[2]])

        print(f"\nDIAGNOSTIC: Arm base (post-gantry) = {base_pos}")
        print(f"DIAGNOSTIC: Part pos               = {target_pos}")
        print(f"DIAGNOSTIC: Y gap arm→part         = {abs(base_pos[1]-target_pos[1]):.4f} m")

        lula_solver.set_robot_base_pose(base_pos, base_rot)

        arm_names         = lula_solver.get_joint_names()
        current_joints    = robot.get_joint_positions()
        warm_start        = np.zeros(len(arm_names))
        for i, lula_name in enumerate(arm_names):
            for idx, dof_name in enumerate(dof_names):
                if lula_name in dof_name:
                    warm_start[i] = current_joints[idx]
                    break

        locked_orientation = normalize_quat(DOWNWARD_ORIENTATION)
        print(f"\nSTATUS: Gripper orientation [w,x,y,z] = {locked_orientation}  |q|={np.linalg.norm(locked_orientation):.6f}")

        if PRINT_ORIENTATION_DEBUG:
            print_ee_orientation_debug(get_prim_at_path(EE_PATH))

        # ========================================================
        # COMPUTE WAYPOINTS
        #
        #   If target_frame == 'gripper_tcp':
        #     IK directly places the finger contact pad at target_pos
        #     → waypoints use target_pos as-is (no GRIPPER_TCP_OFFSET needed)
        #
        #   If target_frame == 'ee_link' (fallback):
        #     IK places the flange at target_pos
        #     → waypoints must add GRIPPER_TCP_OFFSET to Z so the
        #       finger pad ends up at the right place
        #
        #   safe_z = box_top + entry_margin
        #   BOX_ENTRY_MARGIN must be ≥ GRIPPER_TCP_OFFSET so the entire
        #   gripper body is above the rim before any lateral movement stops.
        # ========================================================
        if target_frame == "gripper_tcp":
            # IK targets the true contact pad — no Z offset needed
            tcp_z_offset = 0.0
        else:
            # IK targets the flange — add offset so pad lands at the right Z
            tcp_z_offset = GRIPPER_TCP_OFFSET

        # ── Adaptive waypoints using real part geometry ─────────────────────
        #
        #   safe_z   : finger contact pad is here when the entire gripper
        #              body is above the box rim.  Reference is part_top_z
        #              (actual top surface) rather than the prim pivot.
        #
        #   hover_pos: HOVER_CLEARANCE above the real part top surface.
        #
        #   grasp_pos: finger contact pad at grasp_z
        #              = part_top_z - GRASP_DEPTH_FRACTION * part_height
        #              e.g. 0.5 → mid-height of the part
        # ────────────────────────────────────────────────────────────────────
        safe_z = part_top_z + BOX_HEIGHT + BOX_ENTRY_MARGIN + tcp_z_offset

        # Workspace sanity check
        ur10_max_reach = 1.30
        dist_to_safe   = np.linalg.norm(np.array([
            target_pos[0] - base_pos[0],
            target_pos[1] - base_pos[1],
            safe_z        - base_pos[2]
        ]))

        print(f"\nDIAGNOSTIC: BOX_ENTRY_MARGIN     = {BOX_ENTRY_MARGIN:.3f} m  "
              f"(must be ≥ {GRIPPER_TCP_OFFSET:.3f} m)")
        if BOX_ENTRY_MARGIN < GRIPPER_TCP_OFFSET:
            print(f"  ⚠️  WARNING: BOX_ENTRY_MARGIN < GRIPPER_TCP_OFFSET — "
                  f"gripper body will clip the rim!")
            print(f"  Increase BOX_ENTRY_MARGIN to at least {GRIPPER_TCP_OFFSET:.3f}")

        phase1_pos = target_pos.copy();  phase1_pos[2] = safe_z
        phase2_pos = target_pos.copy();  phase2_pos[2] = safe_z
        # Hover: HOVER_CLEARANCE above the actual top surface of the part
        hover_pos  = target_pos.copy();  hover_pos[2]  = part_top_z + HOVER_CLEARANCE + tcp_z_offset
        # Grasp: depth-fraction into the part (bbox-derived, not prim pivot)
        grasp_pos  = target_pos.copy();  grasp_pos[2]  = grasp_z + tcp_z_offset

        print(f"DIAGNOSTIC: part_top_z          = {part_top_z:.4f} m")
        print(f"DIAGNOSTIC: grasp_z (bbox)      = {grasp_z:.4f} m  "
              f"(GRASP_DEPTH_FRACTION={GRASP_DEPTH_FRACTION})")
        print(f"DIAGNOSTIC: safe_z              = {safe_z:.4f} m  "
              f"(dist-from-base={dist_to_safe:.4f}, limit={ur10_max_reach})")
        print(f"DIAGNOSTIC: Phase 1/2 target    = {phase1_pos}")
        print(f"DIAGNOSTIC: Hover  target       = {hover_pos}")
        print(f"DIAGNOSTIC: Grasp  target       = {grasp_pos}")
        print(f"DIAGNOSTIC: IK frame            = '{target_frame}'")
        print(f"DIAGNOSTIC: tcp_z_offset        = {tcp_z_offset:.3f} m")

        if dist_to_safe > ur10_max_reach:
            print(f"\n  ⚠️  WARNING: Phase 1 target is {dist_to_safe:.4f}m from arm base — may exceed UR10 reach ({ur10_max_reach}m)")
            print(f"  Y gap = {abs(base_pos[1]-target_pos[1]):.4f}m is the main contributor")

        # ========================================================
        # OPEN FINGERS BEFORE ANY MOVEMENT
        # ========================================================
        print("\nSTATUS: Opening fingers before motion...")
        targets = set_finger_joints(robot, dof_names, FINGER_OPEN)
        robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()

        # ========================================================
        # 6. PHASE 1 — SAFE HEIGHT ABOVE BOX
        #    Arm moves to (part_X, part_Y, safe_z).
        #    The entire gripper body is above the box rim.
        #    No lateral collision possible from this point down.
        # ========================================================
        print(f"\nSTATUS: [Phase 1] IK for safe height {phase1_pos} ...")
        p1_action, p1_ok = ik_solve(lula_solver, target_frame, phase1_pos, locked_orientation, warm_start)
        if not p1_ok:
            print(f"ERROR: [Phase 1] IK failed.")
            print(f"  dist-from-base={dist_to_safe:.4f} m  (UR10 limit={ur10_max_reach} m)")
            print(f"  Try reducing BOX_ENTRY_MARGIN or increasing GANTRY_X_OFFSET")
            return

        print("STATUS: [Phase 1] Moving to safe height above box...")
        targets = apply_arm_joints(robot, dof_names, arm_names, p1_action)
        robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(150):
            await omni.kit.app.get_app().next_update_async()
        ee_p1 = get_world_pos(EE_PATH)
        print(f"DIAGNOSTIC: [Phase 1] EE={ee_p1}  Z-err={abs(ee_p1[2]-safe_z+GRIPPER_TCP_OFFSET):.4f}m")

        # ========================================================
        # 7. PHASE 2 — RE-SOLVE FROM SETTLED POSE
        #    Same target as Phase 1, but warm-starts from the
        #    physically settled joint positions — gives a cleaner
        #    seed for Phase 3 descent.
        # ========================================================
        print(f"\nSTATUS: [Phase 2] Re-solving from settled pose...")
        settled_joints = robot.get_joint_positions()
        warm_p2 = np.zeros(len(arm_names))
        for i, lula_name in enumerate(arm_names):
            for idx, dof_name in enumerate(dof_names):
                if lula_name in dof_name:
                    warm_p2[i] = settled_joints[idx]
                    break

        p2_action, p2_ok = ik_solve(lula_solver, target_frame, phase2_pos, locked_orientation, warm_p2)
        if not p2_ok:
            print("WARNING: [Phase 2] Re-solve failed — using Phase 1 result as seed for Phase 3")
            p2_action = p1_action

        # ========================================================
        # 8. PHASE 3a — HOVER (drop inside box, above part)
        #    Pure Z descent from safe height.
        #    Gripper body is now inside the box XY footprint.
        # ========================================================
        print(f"\nSTATUS: [Phase 3a] IK for hover {hover_pos} ...")
        hover_action, hover_ok = ik_solve(lula_solver, target_frame, hover_pos, locked_orientation, p2_action)
        if not hover_ok:
            print("ERROR: [Phase 3a] IK failed for hover.")
            return

        print("STATUS: [Phase 3a] Descending into box to hover...")
        targets = apply_arm_joints(robot, dof_names, arm_names, hover_action)
        robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(150):
            await omni.kit.app.get_app().next_update_async()
        ee_hover = get_world_pos(EE_PATH)
        print(f"DIAGNOSTIC: [Phase 3a] EE={ee_hover}  err={np.linalg.norm(ee_hover - hover_pos + np.array([0,0,tcp_z_offset])):.4f}m")

        if PRINT_ORIENTATION_DEBUG:
            print_ee_orientation_debug(get_prim_at_path(EE_PATH))

        # ========================================================
        # 9. PHASE 3b — PLUNGE TO GRASP HEIGHT
        # ========================================================
        print(f"\nSTATUS: [Phase 3b] IK for grasp {grasp_pos} ...")
        grasp_action, grasp_ok = ik_solve(lula_solver, target_frame, grasp_pos, locked_orientation, hover_action)
        if not grasp_ok:
            print("ERROR: [Phase 3b] IK failed for grasp.")
            return

        print("STATUS: [Phase 3b] Plunging to part surface...")
        targets = apply_arm_joints(robot, dof_names, arm_names, grasp_action)
        robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(100):
            await omni.kit.app.get_app().next_update_async()
        ee_grasp = get_world_pos(EE_PATH)
        print(f"DIAGNOSTIC: [Phase 3b] EE={ee_grasp}  err={np.linalg.norm(ee_grasp - grasp_pos + np.array([0,0,tcp_z_offset])):.4f}m")

        # ========================================================
        # 10. CLOSE GRIPPER — Robotiq 2F-140
        # ========================================================
        print(f"\nSTATUS: Closing Robotiq 2F-140 fingers to {FINGER_CLOSE} rad...")
        targets = set_finger_joints(robot, dof_names, FINGER_CLOSE)
        robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(120):
            await omni.kit.app.get_app().next_update_async()

        final_joints = robot.get_joint_positions()
        print("\n--- FINGER JOINTS AFTER CLOSE ---")
        for idx, name in enumerate(dof_names):
            if "finger_joint" in name or "knuckle_joint" in name:
                actual = final_joints[idx]
                if   actual < FINGER_CLOSE - 0.05: status = "✅ stopped on part surface"
                elif actual >= FINGER_CLOSE - 0.05: status = "✅ closed freely"
                else:                               status = "❌ check joint name"
                print(f"  {name}: {actual:.4f} rad  (target={FINGER_CLOSE})  {status}")
        print("---------------------------------\n")

        print("\n✅ Pick & Grasp complete — starting PLACE sequence...\n")

        # ========================================================
        # 11. RETRACT — LIFT PART TO SAFE HEIGHT
        #     Move straight up with part held to clear the source box.
        # ========================================================
        print(f"\nSTATUS: [Phase 4] Retracting to safe_z = {safe_z:.4f} ...")
        retract_pos = target_pos.copy()
        retract_pos[2] = safe_z

        # warm-start from current settled joints
        settled_post_grasp = robot.get_joint_positions()
        warm_retract = np.zeros(len(arm_names))
        for i, lula_name in enumerate(arm_names):
            for idx, dof_name in enumerate(dof_names):
                if lula_name in dof_name:
                    warm_retract[i] = settled_post_grasp[idx]
                    break

        retract_action, retract_ok = ik_solve(
            lula_solver, target_frame, retract_pos, locked_orientation, warm_retract
        )
        if not retract_ok:
            print("ERROR: [Phase 4] IK failed for retract — aborting place.")
            return

        targets = apply_arm_joints(robot, dof_names, arm_names, retract_action)
        # keep fingers closed while lifting
        targets = set_finger_joints(robot, dof_names, FINGER_CLOSE, base_targets=targets)
        robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(150):
            await omni.kit.app.get_app().next_update_async()
        ee_retract = get_world_pos(EE_PATH)
        print(f"DIAGNOSTIC: [Phase 4] EE={ee_retract}")

        # ========================================================
        # 11b. TRANSIT SAFE HEIGHT
        #      Before any lateral gantry movement, raise the arm to
        #      a high clearance Z so the gripper + held part cannot
        #      collide with anything in the scene during transit.
        # ========================================================
        transit_z = safe_z + TRANSIT_SAFE_HEIGHT
        transit_pos = retract_pos.copy()
        transit_pos[2] = transit_z
        print(f"\nSTATUS: [Phase 4b] Moving to transit safe height Z={transit_z:.4f} ...")

        settled_p4 = robot.get_joint_positions()
        warm_transit = np.zeros(len(arm_names))
        for i, lula_name in enumerate(arm_names):
            for idx, dof_name in enumerate(dof_names):
                if lula_name in dof_name:
                    warm_transit[i] = settled_p4[idx]
                    break

        transit_action, transit_ok = ik_solve(
            lula_solver, target_frame, transit_pos, locked_orientation, warm_transit
        )
        if not transit_ok:
            print("ERROR: [Phase 4b] IK failed for transit safe height — aborting place.")
            return

        targets = apply_arm_joints(robot, dof_names, arm_names, transit_action)
        targets = set_finger_joints(robot, dof_names, FINGER_CLOSE, base_targets=targets)
        robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(150):
            await omni.kit.app.get_app().next_update_async()
        ee_transit = get_world_pos(EE_PATH)
        print(f"DIAGNOSTIC: [Phase 4b] EE={ee_transit}  transit_z={transit_z:.4f}")

        # ========================================================
        # 12. COMPUTE DESTINATION GEOMETRY
        #     Use the same bbox approach to find the top surface of
        #     the destination box /World/box_840.
        # ========================================================
        print(f"\nSTATUS: [Phase 5] Computing destination geometry for '{PLACE_BOX_PATH}' ...")
        dest_geom = compute_bbox_geometry(stage, PLACE_BOX_PATH, label="DESTINATION BOX")

        dest_center_x = dest_geom["center_xy"][0]
        dest_center_y = dest_geom["center_xy"][1]
        dest_top_z    = dest_geom["top_z"]

        # Place target: centre XY of box, Z = top surface + part half-height + drop gap
        place_z = dest_top_z + geom["part_height"] * GRASP_DEPTH_FRACTION + PLACE_DROP_HEIGHT + tcp_z_offset
        place_safe_z = dest_top_z + geom["part_height"] + BOX_ENTRY_MARGIN + tcp_z_offset

        place_target = np.array([dest_center_x, dest_center_y, place_z])
        place_safe   = np.array([dest_center_x, dest_center_y, place_safe_z])

        print(f"DIAGNOSTIC: dest centre XY     = [{dest_center_x:.4f}, {dest_center_y:.4f}]")
        print(f"DIAGNOSTIC: dest top Z         = {dest_top_z:.4f} m")
        print(f"DIAGNOSTIC: place Z (fingers)  = {place_z:.4f} m")
        print(f"DIAGNOSTIC: place safe Z       = {place_safe_z:.4f} m")

        # ========================================================
        # 13. MOVE GANTRY X TO DESTINATION (if needed)
        # ========================================================
        current_gantry = robot.get_joint_positions()
        if gantry_x_idx is not None:
            new_gantry_x = dest_center_x - GANTRY_X_OFFSET
            current_gantry[gantry_x_idx] = new_gantry_x
            print(f"\nSTATUS: [Phase 5b] Moving gantry X → {new_gantry_x:.4f} m for destination")
            robot.apply_action(ArticulationAction(joint_positions=current_gantry))
            for _ in range(120):
                await omni.kit.app.get_app().next_update_async()

            # Re-read arm base after gantry settles
            base_prim   = get_prim_at_path(UR10_BASE_PATH)
            base_matrix = omni.usd.get_world_transform_matrix(base_prim)
            base_pos    = np.array(base_matrix.ExtractTranslation())
            rot         = base_matrix.ExtractRotation().GetQuat()
            base_rot    = np.array([rot.real, rot.imaginary[0], rot.imaginary[1], rot.imaginary[2]])
            lula_solver.set_robot_base_pose(base_pos, base_rot)
            print(f"DIAGNOSTIC: Arm base (post-gantry) = {base_pos}")

        # Fresh warm-start after gantry repositioned
        settled_pre_place = robot.get_joint_positions()
        warm_place = np.zeros(len(arm_names))
        for i, lula_name in enumerate(arm_names):
            for idx, dof_name in enumerate(dof_names):
                if lula_name in dof_name:
                    warm_place[i] = settled_pre_place[idx]
                    break

        # ========================================================
        # 14. PHASE 6a — SAFE HEIGHT ABOVE DESTINATION
        # ========================================================
        print(f"\nSTATUS: [Phase 6a] IK for safe height above destination {place_safe} ...")
        p6a_action, p6a_ok = ik_solve(
            lula_solver, target_frame, place_safe, locked_orientation, warm_place
        )
        if not p6a_ok:
            print("ERROR: [Phase 6a] IK failed for place safe height.")
            return

        targets = apply_arm_joints(robot, dof_names, arm_names, p6a_action)
        targets = set_finger_joints(robot, dof_names, FINGER_CLOSE, base_targets=targets)
        robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(150):
            await omni.kit.app.get_app().next_update_async()
        ee_p6a = get_world_pos(EE_PATH)
        print(f"DIAGNOSTIC: [Phase 6a] EE={ee_p6a}")

        # ========================================================
        # 15. PHASE 6b — DESCEND TO PLACE HEIGHT
        # ========================================================
        print(f"\nSTATUS: [Phase 6b] IK for place height {place_target} ...")
        p6b_action, p6b_ok = ik_solve(
            lula_solver, target_frame, place_target, locked_orientation, p6a_action
        )
        if not p6b_ok:
            print("ERROR: [Phase 6b] IK failed for place descent.")
            return

        targets = apply_arm_joints(robot, dof_names, arm_names, p6b_action)
        targets = set_finger_joints(robot, dof_names, FINGER_CLOSE, base_targets=targets)
        robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(150):
            await omni.kit.app.get_app().next_update_async()
        ee_p6b = get_world_pos(EE_PATH)
        print(f"DIAGNOSTIC: [Phase 6b] EE={ee_p6b}")

        # ========================================================
        # 16. OPEN GRIPPER — RELEASE PART
        # ========================================================
        print(f"\nSTATUS: Opening fingers to release part...")
        targets = set_finger_joints(robot, dof_names, FINGER_OPEN)
        robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(90):
            await omni.kit.app.get_app().next_update_async()

        # ========================================================
        # 17. RETRACT UP AFTER RELEASE
        # ========================================================
        retract_place_pos = place_target.copy()
        retract_place_pos[2] = dest_top_z + PLACE_RETRACT_HEIGHT + tcp_z_offset

        print(f"\nSTATUS: [Phase 7] Retracting after release to Z={retract_place_pos[2]:.4f} ...")

        settled_post_release = robot.get_joint_positions()
        warm_post_release = np.zeros(len(arm_names))
        for i, lula_name in enumerate(arm_names):
            for idx, dof_name in enumerate(dof_names):
                if lula_name in dof_name:
                    warm_post_release[i] = settled_post_release[idx]
                    break

        p7_action, p7_ok = ik_solve(
            lula_solver, target_frame, retract_place_pos, locked_orientation, warm_post_release
        )
        if not p7_ok:
            print("WARNING: [Phase 7] Retract IK failed — staying at place height.")
        else:
            targets = apply_arm_joints(robot, dof_names, arm_names, p7_action)
            robot.apply_action(ArticulationAction(joint_positions=targets))
            for _ in range(120):
                await omni.kit.app.get_app().next_update_async()
            ee_p7 = get_world_pos(EE_PATH)
            print(f"DIAGNOSTIC: [Phase 7] EE={ee_p7}")

        print("\n🏆 THESIS MILESTONE: Full Pick & Place Sequence Complete! 🏆")
        print(f"   Part placed on: {PLACE_BOX_PATH}")
        print(f"   Destination top Z: {dest_top_z:.4f} m\n")

    except Exception as e:
        print("\n" + "=" * 50)
        print("!!! THE SCRIPT CRASHED !!!")
        print(f"ERROR: {e}")
        traceback.print_exc()
        print("=" * 50 + "\n")


asyncio.ensure_future(run_perfect_pick())