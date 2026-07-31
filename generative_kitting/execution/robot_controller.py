"""
Robot Controller — Execution Layer (Layer 3) for the Generative Kitting System.

Translates validated task plans into deterministic Isaac Sim Python API calls.
Wraps the proven pick-and-place logic from robot_control.py into a modular
action-primitive interface.

The LLM NEVER touches this layer — it only receives action names + parameters.
All inverse kinematics, trajectory computation, and physics happen here.

Refactored from: src/robot_control.py
"""

import asyncio
import os
import tempfile
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import log
from execution.safety import SafetyMonitor, emergency_stop, ActionTimeout


# ═══════════════════════════════════════════════════════════════
# HELPERS — Ported from robot_control.py
# ═══════════════════════════════════════════════════════════════

def normalize_quat(q):
    """Normalise a quaternion [w, x, y, z] to unit length."""
    q = np.array(q, dtype=np.float64)
    mag = np.linalg.norm(q)
    if mag < 1e-9:
        raise ValueError(f"Quaternion {q} is near-zero — invalid rotation!")
    normed = q / mag
    if abs(mag - 1.0) > 0.005:
        log.warning(f"Quaternion magnitude={mag:.6f} — auto-normalized")
    return normed


def get_world_pos(prim_path):
    """Get world-space position of a prim."""
    import omni.usd
    from isaacsim.core.utils.prims import get_prim_at_path
    return np.array(
        omni.usd.get_world_transform_matrix(
            get_prim_at_path(prim_path)
        ).ExtractTranslation()
    )


def compute_bbox_geometry(stage, prim_path, label="PRIM"):
    """
    Compute the world-space axis-aligned bounding box of any prim.

    Returns dict: center_xy, pivot, top_z, bot_z, height, min_pt, max_pt
    """
    from pxr import Gf, UsdGeom, Usd

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found at path: '{prim_path}'")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        ["default", "render"],
        useExtentsHint=True,
    )
    world_bound = bbox_cache.ComputeWorldBound(prim)
    aligned_box = world_bound.ComputeAlignedRange()

    min_pt = np.array(aligned_box.GetMin())
    max_pt = np.array(aligned_box.GetMax())

    height = float(max_pt[2] - min_pt[2])
    top_z = float(max_pt[2])
    bot_z = float(min_pt[2])
    center_xy = [
        float((min_pt[0] + max_pt[0]) / 2.0),
        float((min_pt[1] + max_pt[1]) / 2.0),
    ]

    pivot = get_world_pos(prim_path)

    log.debug(
        f"BBox [{label}]: min={np.round(min_pt, 4)}, max={np.round(max_pt, 4)}, "
        f"height={height*1000:.1f}mm"
    )

    return {
        "center_xy": center_xy,
        "pivot": pivot,
        "top_z": top_z,
        "bot_z": bot_z,
        "height": height,
        "min_pt": min_pt,
        "max_pt": max_pt,
    }


GRIPPER_TCP_LINK_TEMPLATE = '''
  <link name="gripper_tcp">
    <inertial>
      <mass value="0.001"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
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
    <origin xyz="0 0 {tcp_offset}" rpy="0 0 0"/>
  </joint>
'''


def patch_urdf_and_yaml(urdf_path, yaml_path, tcp_offset, body_width):
    """
    Patch the stock NVIDIA UR10 URDF and YAML files:
    - Fix missing inertia tags
    - Remove stray parenthesis typo
    - Inject virtual gripper_tcp link
    - Fix root_link reference

    Returns (fixed_urdf_path, fixed_yaml_path)
    """
    temp_dir = tempfile.gettempdir()
    fixed_urdf = os.path.join(temp_dir, "fixed_ur10_with_gripper.urdf")
    fixed_yaml = os.path.join(temp_dir, "fixed_ur10_with_gripper.yaml")

    # --- Patch URDF ---
    with open(urdf_path, "r") as f:
        urdf = f.read()

    urdf = urdf.replace(
        "</inertial>",
        '<inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/></inertial>',
    )
    urdf = urdf.replace("0.0027000046)", "0.0027000046")

    if "gripper_tcp" not in urdf:
        tcp_link = GRIPPER_TCP_LINK_TEMPLATE.format(
            tcp_offset=tcp_offset,
            half_tcp=tcp_offset / 2.0,
            width=body_width,
        )
        urdf = urdf.replace("</robot>", tcp_link + "\n</robot>")
        log.info("Injected virtual gripper_tcp link into URDF ✓")

    with open(fixed_urdf, "w") as f:
        f.write(urdf)

    # --- Patch YAML ---
    with open(yaml_path, "r") as f:
        yaml_content = f.read()

    yaml_content = yaml_content.replace("root_link: world", "root_link: base_link")

    if "gripper_tcp_joint" not in yaml_content:
        yaml_content = (
            yaml_content.rstrip()
            + "\n\n# Virtual gripper TCP added by generative_kitting\n"
            "ee_fixed_joints:\n  - gripper_tcp_joint\n"
        )
        log.info("Injected gripper_tcp_joint into YAML ✓")

    with open(fixed_yaml, "w") as f:
        f.write(yaml_content)

    return fixed_urdf, fixed_yaml


def repair_collision_filters(stage, gripper_paths, box_path):
    """Ensure gripper fingers can physically collide with the box."""
    from pxr import UsdPhysics

    box_prim = stage.GetPrimAtPath(box_path)
    if not box_prim.IsValid():
        log.warning(f"Box prim not found at '{box_path}' — skipping collision repair")
        return

    repaired = 0
    for finger_path in gripper_paths:
        finger_prim = stage.GetPrimAtPath(finger_path)
        if not finger_prim.IsValid():
            continue
        UsdPhysics.CollisionAPI.Apply(finger_prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(box_prim).CreateCollisionEnabledAttr(True)
        repaired += 1

    log.info(f"Repaired {repaired} gripper-box collision pairs")


# ═══════════════════════════════════════════════════════════════
# ROBOT CONTROLLER CLASS
# ═══════════════════════════════════════════════════════════════

class RobotController:
    """
    Isaac Sim UR10 robot controller implementing the action primitive interface.

    Each method corresponds to exactly one action primitive that the LLM
    can reference in task plans. All physics, IK, and trajectory logic
    is encapsulated here — the LLM never sees it.
    """

    def __init__(self, config: dict, world=None, robot=None, stage=None):
        """
        Initialise the robot controller.

        Parameters
        ----------
        config : dict
            The full parsed config.yaml dict.
        world : World, optional
            Isaac Sim World instance (created if None).
        robot : SingleArticulation, optional
            Robot articulation (created if None).
        stage : Usd.Stage, optional
            USD stage (obtained from context if None).
        """
        self.config = config
        self.exec_cfg = config.get("execution", {})
        self.sim_cfg = config.get("simulation", {})

        # Safety monitor
        self.safety = SafetyMonitor(self.exec_cfg)

        # Gripper parameters
        self.tcp_offset = self.exec_cfg.get("gripper_tcp_offset", 0.150)
        self.body_width = self.exec_cfg.get("gripper_body_width", 0.160)
        self.half_width = self.body_width / 2.0
        self.finger_open = self.exec_cfg.get("finger_open_rad", 0.0)
        self.finger_close = self.exec_cfg.get("finger_close_rad", 0.5)
        self.hover_clearance = self.exec_cfg.get("hover_clearance", 0.015)
        self.box_height = self.exec_cfg.get("box_height", 0.05)
        self.box_entry_margin = self.exec_cfg.get("box_entry_margin", 0.15)
        self.place_drop_height = self.exec_cfg.get("place_drop_height", 0.0)
        self.place_retract_height = self.exec_cfg.get("place_retract_height", 0.20)
        self.transit_safe_height = self.exec_cfg.get("transit_safe_height", 0.40)
        self.grasp_depth_fraction = self.exec_cfg.get("grasp_depth_fraction", 0.5)
        self.downward_orientation = normalize_quat(
            self.exec_cfg.get("downward_orientation", [1.0, 0.0, 1.0, 0.0])
        )
        # Kit tray position is detected dynamically by the perception
        # pipeline (VLM + detector + depth projection), not hardcoded.
        self.kit_tray_position = self.exec_cfg.get("kit_tray_position")

        # Gantry parameters
        self.gantry_joint = self.sim_cfg.get("gantry_x_joint", "gantry_vagn_joint")
        self.gantry_offset = self.sim_cfg.get("gantry_x_offset", 1.27)

        # Prim paths
        self.robot_path = self.sim_cfg.get("robot_prim_path", "/World")
        self.ur10_base_path = self.sim_cfg.get(
            "ur10_base_path",
            "/World/gantry_home/ur10_flattened/ur10_instanceable/base_link",
        )
        self.ee_path = self.sim_cfg.get(
            "ee_path",
            "/World/gantry_home/ur10_flattened/ur10_instanceable/ee_link",
        )
        self.gripper_finger_paths = self.sim_cfg.get("gripper_finger_paths", [])
        self.box_prim_path = self.sim_cfg.get("box_prim_path", "/World/robot_facade_full")

        # State (initialised by init_sim)
        self.world = world
        self.robot = robot
        self.stage = stage
        self.lula_solver = None
        self.target_frame = "ee_link"
        self.dof_names = []
        self.arm_names = []
        self.gantry_x_idx = None

        # Perception callback (set by main app)
        self._perception_callback = None

        log.info("RobotController created (awaiting init_sim)")

    async def init_sim(self):
        """
        Initialise the Isaac Sim environment:
        - Create World + robot articulation
        - Patch URDF/YAML and create Lula IK solver
        - Repair collision filters
        """
        import omni.kit.app
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.prims import get_prim_at_path
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

        # World
        if self.world is None:
            self.world = World.instance() or World()

        self.world.play()

        # Robot
        if self.robot is None:
            existing = self.world.scene.get_object("kitting_sys")
            if existing:
                self.robot = existing
            else:
                self.robot = SingleArticulation(
                    prim_path=self.robot_path, name="kitting_sys"
                )
                self.world.scene.add(self.robot)
        self.robot.initialize()

        # Warm up physics
        log.info("Warming up physics engine...")
        for _ in range(30):
            await omni.kit.app.get_app().next_update_async()

        self.dof_names = list(self.robot.dof_names)
        log.info(f"Robot DOFs: {len(self.dof_names)}")

        # Stage
        if self.stage is None:
            self.stage = omni.usd.get_context().get_stage()

        # Collision filters
        repair_collision_filters(
            self.stage, self.gripper_finger_paths, self.box_prim_path
        )

        # Patch URDF + YAML → Lula IK solver
        urdf_path = self.sim_cfg.get("urdf_path", "")
        yaml_path = self.sim_cfg.get("yaml_path", "")

        if urdf_path and yaml_path:
            fixed_urdf, fixed_yaml = patch_urdf_and_yaml(
                urdf_path, yaml_path, self.tcp_offset, self.body_width
            )
            self.lula_solver = LulaKinematicsSolver(
                robot_description_path=fixed_yaml,
                urdf_path=fixed_urdf,
            )
            log.info(f"Lula IK solver initialised")

            # Check if gripper_tcp frame is available
            try:
                test_pos = np.array([0.0, -0.5, 1.2])
                test_ori = self.downward_orientation
                _, test_ok = self.lula_solver.compute_inverse_kinematics(
                    frame_name="gripper_tcp",
                    target_position=test_pos,
                    target_orientation=test_ori,
                    warm_start=np.zeros(6),
                )
                if test_ok:
                    self.target_frame = "gripper_tcp"
                    log.info("IK frame = 'gripper_tcp' (true tool tip) ✓")
                else:
                    self.target_frame = "ee_link"
                    log.info("IK frame = 'ee_link' (fallback)")
            except Exception:
                self.target_frame = "ee_link"
                log.info("IK frame = 'ee_link' (gripper_tcp not available)")

            self.arm_names = list(self.lula_solver.get_joint_names())
            log.info(f"Lula arm joints: {self.arm_names}")

        # Find gantry index
        for idx, name in enumerate(self.dof_names):
            if self.gantry_joint in name:
                self.gantry_x_idx = idx
                break

        log.info(f"RobotController fully initialised (gantry_idx={self.gantry_x_idx})")

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────

    def _get_tcp_z_offset(self) -> float:
        """Return the Z offset based on the IK frame."""
        return 0.0 if self.target_frame == "gripper_tcp" else self.tcp_offset

    def _get_warm_start(self) -> np.ndarray:
        """Build warm-start joint array from current robot state."""
        current = self.robot.get_joint_positions()
        warm = np.zeros(len(self.arm_names))
        for i, lula_name in enumerate(self.arm_names):
            for idx, dof_name in enumerate(self.dof_names):
                if lula_name in dof_name:
                    warm[i] = current[idx]
                    break
        return warm

    def _get_current_ee_pos(self) -> np.ndarray:
        """Return the current world-space EE position."""
        return get_world_pos(self.ee_path)

    def _apply_arm_joints(self, ik_result):
        """Apply IK solution to robot joint targets."""
        targets = self.robot.get_joint_positions()
        for i, lula_name in enumerate(self.arm_names):
            for idx, dof_name in enumerate(self.dof_names):
                if lula_name in dof_name:
                    targets[idx] = ik_result[i]
                    break
        return targets

    def _set_finger_joints(self, value, base_targets=None):
        """Set Robotiq 2F-140 finger joints with mimic coupling."""
        targets = (
            base_targets if base_targets is not None
            else self.robot.get_joint_positions()
        )
        for idx, name in enumerate(self.dof_names):
            if "inner_finger_joint" in name:
                targets[idx] = -value
            elif "finger_joint" in name or "knuckle_joint" in name:
                targets[idx] = value
        return targets

    def _ik_solve(self, pos, ori=None, warm_start=None):
        """Compute IK and return (joint_values, success)."""
        if ori is None:
            ori = self.downward_orientation
        if warm_start is None:
            warm_start = self._get_warm_start()

        action, ok = self.lula_solver.compute_inverse_kinematics(
            frame_name=self.target_frame,
            target_position=np.array(pos),
            target_orientation=ori,
            warm_start=warm_start,
        )
        if not ok:
            log.error(f"IK failed for pos={pos}")
        return action, ok

    async def _move_and_wait(self, targets, steps=150):
        """Apply joint targets and wait for motion to settle."""
        import omni.kit.app
        from isaacsim.core.utils.types import ArticulationAction

        self.robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(steps):
            await omni.kit.app.get_app().next_update_async()

    async def _update_base_pose(self):
        """Re-read arm base pose after gantry movement and update Lula."""
        import omni.usd
        from isaacsim.core.utils.prims import get_prim_at_path

        base_prim = get_prim_at_path(self.ur10_base_path)
        base_matrix = omni.usd.get_world_transform_matrix(base_prim)
        base_pos = np.array(base_matrix.ExtractTranslation())
        rot = base_matrix.ExtractRotation().GetQuat()
        base_rot = np.array([rot.real, rot.imaginary[0], rot.imaginary[1], rot.imaginary[2]])
        self.lula_solver.set_robot_base_pose(base_pos, base_rot)
        return base_pos

    async def _move_gantry_to(self, target_x, hold_ee_pos=None):
        """Move gantry X to align with a target X position.

        Args:
            target_x: world X coordinate the gantry should reach.
            hold_ee_pos: if not None, a 3-element world position [x,y,z] that
                         the EE should stay locked onto during the gantry slide.
                         The arm joints compensate each step so the EE stays
                         stationary — like pointing a finger while walking.
        """
        import omni.kit.app
        from isaacsim.core.utils.types import ArticulationAction

        if self.gantry_x_idx is None:
            log.warning("No gantry X joint — skipping gantry move")
            return

        gantry_target = target_x - self.gantry_offset
        targets = self.robot.get_joint_positions()
        gantry_current = float(targets[self.gantry_x_idx])
        gantry_delta = gantry_target - gantry_current

        # ── EE-hold mode: pre-compute arm compensation, apply simultaneously ──
        # The gantry is a prismatic joint along X — the base_link shifts
        # by exactly the gantry delta in X.  We predict the new base pose
        # analytically and solve IK against it, then apply gantry + arm
        # in one action so the EE never drifts.
        if hold_ee_pos is not None and self.lula_solver and abs(gantry_delta) > 0.005:
            import omni.usd
            from isaacsim.core.utils.prims import get_prim_at_path

            ee_lock = np.array(hold_ee_pos, dtype=np.float64)
            n_steps = max(4, int(abs(gantry_delta) / 0.02))

            # Read initial base pose once — rotation is constant for prismatic
            base_prim = get_prim_at_path(self.ur10_base_path)
            base_matrix = omni.usd.get_world_transform_matrix(base_prim)
            initial_base_pos = np.array(base_matrix.ExtractTranslation())
            rot = base_matrix.ExtractRotation().GetQuat()
            base_rot = np.array([rot.real, rot.imaginary[0],
                                 rot.imaginary[1], rot.imaginary[2]])

            log.info(f"Gantry EE-hold slide: {gantry_current:.3f} → {gantry_target:.3f} "
                     f"({n_steps} steps)")

            for step in range(1, n_steps + 1):
                frac = step / n_steps
                intermediate = gantry_current + frac * gantry_delta
                delta_from_start = intermediate - gantry_current

                # 1. Predict base_link position after gantry moves
                predicted_base = initial_base_pos.copy()
                predicted_base[0] += delta_from_start

                # 2. Tell Lula the predicted base pose
                self.lula_solver.set_robot_base_pose(predicted_base, base_rot)

                # 3. Solve IK for locked EE position
                action, ok = self._ik_solve(ee_lock)

                # 4. Apply gantry + arm SIMULTANEOUSLY
                if ok:
                    targets = self._apply_arm_joints(action)
                else:
                    targets = self.robot.get_joint_positions()
                targets[self.gantry_x_idx] = intermediate
                self.robot.apply_action(ArticulationAction(joint_positions=targets))

                for _ in range(4):
                    await omni.kit.app.get_app().next_update_async()

            # Final settle + sync Lula with actual USD state
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()
            await self._update_base_pose()
            return

        # ── Simple mode: just move gantry without arm compensation ──
        targets[self.gantry_x_idx] = gantry_target
        self.robot.apply_action(ArticulationAction(joint_positions=targets))
        log.info(f"Moving gantry X → {gantry_target:.4f}m")

        for _ in range(120):
            await omni.kit.app.get_app().next_update_async()

        await self._update_base_pose()

    # ═══════════════════════════════════════════════════════════
    # ACTION PRIMITIVES — The only interface the LLM can call
    # ═══════════════════════════════════════════════════════════

    async def move_home(self) -> bool:
        """Move robot to predefined safe home joint configuration."""
        log.info("ACTION: move_home()")
        try:
            home_joints = self.exec_cfg.get(
                "home_joint_positions",
                [0.0, 0, -1.5708, 1.5708, -1.5708, -1.5708, 0],
            )

            targets = self.robot.get_joint_positions()
            # Map home joints to the first N DOFs
            for i, val in enumerate(home_joints):
                if i < len(targets):
                    targets[i] = val

            await self._move_and_wait(targets, steps=150)
            log.info("move_home() → SUCCESS")
            return True
        except Exception as e:
            log.error(f"move_home() FAILED: {e}")
            return False

    async def move_to_pose(
        self, x: float, y: float, z: float,
        rx: float = 0, ry: float = 0, rz: float = 0,
    ) -> bool:
        """Move end-effector to target pose via IK."""
        log.info(f"ACTION: move_to_pose({x:.4f}, {y:.4f}, {z:.4f})")

        if not self.safety.validate_pose(x, y, z):
            log.error(f"move_to_pose REJECTED — out of workspace bounds")
            return False

        try:
            # Move gantry if needed
            await self._move_gantry_to(x)

            ik_result, ok = self._ik_solve([x, y, z])
            if not ok:
                log.error("move_to_pose: IK solver returned no solution")
                return False

            targets = self._apply_arm_joints(ik_result)
            await self._move_and_wait(targets)

            log.info("move_to_pose() → SUCCESS")
            return True
        except Exception as e:
            log.error(f"move_to_pose() FAILED: {e}")
            return False

    async def open_gripper(self) -> bool:
        """Open the parallel gripper to max width."""
        log.info("ACTION: open_gripper()")
        try:
            targets = self._set_finger_joints(self.finger_open)
            await self._move_and_wait(targets, steps=60)
            log.info("open_gripper() → SUCCESS")
            return True
        except Exception as e:
            log.error(f"open_gripper() FAILED: {e}")
            return False

    async def close_gripper(self) -> bool:
        """Close the gripper with configured grip force."""
        log.info("ACTION: close_gripper()")
        try:
            targets = self._set_finger_joints(self.finger_close)
            await self._move_and_wait(targets, steps=120)
            log.info("close_gripper() → SUCCESS")
            return True
        except Exception as e:
            log.error(f"close_gripper() FAILED: {e}")
            return False

    async def pick_object(
        self, object_id: str, x: float, y: float, z: float,
    ) -> bool:
        """
        Full pick sequence: approach → hover → descend → grasp → lift.
        Replicates Phases 1–3b + close from robot_control.py.
        """
        log.info(f"ACTION: pick_object('{object_id}', {x:.4f}, {y:.4f}, {z:.4f})")

        if not self.safety.validate_pose(x, y, z):
            log.error("pick_object REJECTED — out of workspace bounds")
            return False

        try:
            tcp_z = self._get_tcp_z_offset()

            # Lift to a transit pose before any lateral gantry motion so
            # the arm clears the rail and nearby bin structures.
            current_ee = self._get_current_ee_pos()
            transit_z = max(float(current_ee[2]) + self.transit_safe_height,
                            z + self.transit_safe_height)
            lift_ik, lift_ok = self._ik_solve([current_ee[0], current_ee[1], transit_z])
            if lift_ok:
                lift_targets = self._apply_arm_joints(lift_ik)
                await self._move_and_wait(lift_targets)

            # Move gantry
            await self._move_gantry_to(x, hold_ee_pos=[current_ee[0], current_ee[1], transit_z])

            # Safe height above potential box
            safe_z = z + self.box_height + self.box_entry_margin + tcp_z

            # Phase 1: Safe height above target
            ik1, ok1 = self._ik_solve([x, y, safe_z])
            if not ok1:
                log.error("pick_object: IK failed for safe height")
                return False
            targets = self._apply_arm_joints(ik1)
            await self._move_and_wait(targets)

            # Phase 2: Hover above part
            hover_z = z + self.hover_clearance + tcp_z
            ik2, ok2 = self._ik_solve([x, y, hover_z], warm_start=ik1)
            if not ok2:
                log.error("pick_object: IK failed for hover")
                return False
            targets = self._apply_arm_joints(ik2)
            await self._move_and_wait(targets)

            # Phase 3: Descend to grasp
            grasp_z = z + tcp_z + 0.005  # 5mm above surface
            ik3, ok3 = self._ik_solve([x, y, grasp_z], warm_start=ik2)
            if not ok3:
                log.error("pick_object: IK failed for grasp descent")
                return False
            targets = self._apply_arm_joints(ik3)
            await self._move_and_wait(targets, steps=100)

            # Close gripper
            targets = self._set_finger_joints(self.finger_close)
            await self._move_and_wait(targets, steps=120)

            # Lift back to safe height
            ik4, ok4 = self._ik_solve([x, y, safe_z], warm_start=ik3)
            if ok4:
                targets = self._apply_arm_joints(ik4)
                targets = self._set_finger_joints(self.finger_close, targets)
                await self._move_and_wait(targets)

            log.info(f"pick_object('{object_id}') → SUCCESS")
            return True

        except Exception as e:
            log.error(f"pick_object('{object_id}') FAILED: {e}")
            traceback.print_exc()
            return False

    async def place_object(
        self, object_id: str, x: float, y: float, z: float,
    ) -> bool:
        """
        Full place sequence: transit → approach → descend → release → retract.
        Replicates Phases 4–7 from robot_control.py.
        """
        log.info(f"ACTION: place_object('{object_id}', {x:.4f}, {y:.4f}, {z:.4f})")

        if not self.safety.validate_pose(x, y, z):
            log.error("place_object REJECTED — out of workspace bounds")
            return False

        try:
            tcp_z = self._get_tcp_z_offset()

            # Lift to a transit pose before any lateral gantry motion so
            # the arm clears the rail and the bin rim during long moves.
            current_ee = self._get_current_ee_pos()
            transit_z = max(float(current_ee[2]) + self.transit_safe_height,
                            z + self.transit_safe_height)
            lift_ik, lift_ok = self._ik_solve([current_ee[0], current_ee[1], transit_z])
            if lift_ok:
                lift_targets = self._apply_arm_joints(lift_ik)
                await self._move_and_wait(lift_targets)

            # Move gantry to destination
            await self._move_gantry_to(x, hold_ee_pos=[current_ee[0], current_ee[1], transit_z])

            # Safe height above destination
            safe_z = z + self.box_entry_margin + tcp_z

            # Phase 1: Safe height above destination
            ik1, ok1 = self._ik_solve([x, y, safe_z])
            if not ok1:
                log.error("place_object: IK failed for safe height")
                return False
            targets = self._apply_arm_joints(ik1)
            targets = self._set_finger_joints(self.finger_close, targets)
            await self._move_and_wait(targets)

            # Phase 2: Descend to place height
            place_z = z + self.place_drop_height + tcp_z
            ik2, ok2 = self._ik_solve([x, y, place_z], warm_start=ik1)
            if not ok2:
                log.error("place_object: IK failed for place descent")
                return False
            targets = self._apply_arm_joints(ik2)
            targets = self._set_finger_joints(self.finger_close, targets)
            await self._move_and_wait(targets)

            # Open gripper to release
            targets = self._set_finger_joints(self.finger_open)
            await self._move_and_wait(targets, steps=90)

            # Retract
            retract_z = z + self.place_retract_height + tcp_z
            ik3, ok3 = self._ik_solve([x, y, retract_z], warm_start=ik2)
            if ok3:
                targets = self._apply_arm_joints(ik3)
                await self._move_and_wait(targets, steps=120)

            log.info(f"place_object('{object_id}') → SUCCESS")
            return True

        except Exception as e:
            log.error(f"place_object('{object_id}') FAILED: {e}")
            traceback.print_exc()
            return False

    async def verify_grasp(self) -> bool:
        """
        Check if the gripper is currently holding an object.
        Inspects finger joint positions — if closed beyond threshold,
        the fingers were stopped by a part (successful grasp).
        """
        log.info("ACTION: verify_grasp()")
        try:
            joint_positions = self.robot.get_joint_positions()
            grasp_detected = False

            for idx, name in enumerate(self.dof_names):
                if "finger_joint" in name and "inner_finger" not in name:
                    actual = joint_positions[idx]
                    # If finger didn't reach full close, it's contacting an object
                    if actual < self.finger_close - 0.05:
                        grasp_detected = True
                        log.debug(
                            f"  {name}: actual={actual:.4f} < target={self.finger_close} "
                            "→ object detected"
                        )

            if grasp_detected:
                log.info("verify_grasp() → TRUE (object held)")
            else:
                log.warning("verify_grasp() → FALSE (no object detected)")

            return grasp_detected

        except Exception as e:
            log.error(f"verify_grasp() FAILED: {e}")
            return False

    async def request_perception_update(self) -> Dict[str, Any]:
        """Trigger a new VLM scene analysis via the perception callback."""
        log.info("ACTION: request_perception_update()")

        if self._perception_callback:
            try:
                result = await self._perception_callback()
                log.info("request_perception_update() → SUCCESS")
                return result
            except Exception as e:
                log.error(f"request_perception_update() FAILED: {e}")
                return {"detected_objects": [], "scene_summary": f"Perception error: {e}"}
        else:
            log.warning("No perception callback registered")
            return {"detected_objects": [], "scene_summary": "No perception callback"}

    def set_perception_callback(self, callback):
        """Register a callback for request_perception_update()."""
        self._perception_callback = callback

    # ═══════════════════════════════════════════════════════════
    # PLAN EXECUTOR
    # ═══════════════════════════════════════════════════════════

    async def execute_plan(self, task_plan: Dict[str, Any], session_id: str = "") -> Dict[str, Any]:
        """
        Execute a validated task plan step by step.

        Parameters
        ----------
        task_plan : dict
            Validated plan with ``plan`` list of steps.
        session_id : str
            Session identifier for logging.

        Returns
        -------
        dict
            Execution report with per-step results.
        """
        steps = task_plan.get("plan", [])
        log.info(f"Executing plan: {len(steps)} steps — '{task_plan.get('task_summary', '')}'")

        report = {
            "task_summary": task_plan.get("task_summary", ""),
            "total_steps": len(steps),
            "completed_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "skipped_steps": 0,
            "step_results": [],
            "total_time_seconds": 0.0,
        }

        start_time = time.time()

        for step in steps:
            if self.safety.estop_requested:
                log.critical("Emergency stop — aborting plan execution")
                await self.move_home()
                break

            step_num = step.get("step", 0)
            action = step.get("action", "")
            params = step.get("params", {})

            log.info(f"─── Step {step_num}: {action}({params}) ───")
            step_start = time.time()

            try:
                result = await self._dispatch_action(action, params)
                step_duration = time.time() - step_start

                step_result = {
                    "step": step_num,
                    "action": action,
                    "params": params,
                    "success": result,
                    "duration_seconds": round(step_duration, 3),
                    "error": "",
                }

                if result:
                    report["successful_steps"] += 1
                else:
                    report["failed_steps"] += 1
                    step_result["error"] = "Action returned False"

                    # Retry logic for pick_object after failed verify_grasp
                    if action == "verify_grasp" and not result:
                        log.warning("Grasp verification failed — will retry if configured")

            except Exception as e:
                step_duration = time.time() - step_start
                step_result = {
                    "step": step_num,
                    "action": action,
                    "params": params,
                    "success": False,
                    "duration_seconds": round(step_duration, 3),
                    "error": str(e),
                }
                report["failed_steps"] += 1
                log.error(f"Step {step_num} ({action}) raised exception: {e}")

            report["completed_steps"] += 1
            report["step_results"].append(step_result)

        report["total_time_seconds"] = round(time.time() - start_time, 2)

        log.info(
            f"Plan execution complete: "
            f"{report['successful_steps']}/{report['total_steps']} succeeded, "
            f"{report['failed_steps']} failed, "
            f"{report['total_time_seconds']}s total"
        )

        return report

    async def _dispatch_action(self, action: str, params: dict) -> bool:
        """Dispatch an action name to the corresponding method."""
        if action == "move_home":
            return await self.move_home()
        elif action == "move_to_pose":
            return await self.move_to_pose(
                params.get("x", 0), params.get("y", 0), params.get("z", 0),
                params.get("rx", 0), params.get("ry", 0), params.get("rz", 0),
            )
        elif action == "open_gripper":
            return await self.open_gripper()
        elif action == "close_gripper":
            return await self.close_gripper()
        elif action == "pick_object":
            return await self.pick_object(
                params.get("object_id", ""),
                params.get("x", 0), params.get("y", 0), params.get("z", 0),
            )
        elif action == "place_object":
            return await self.place_object(
                params.get("object_id", ""),
                params.get("x", 0), params.get("y", 0), params.get("z", 0),
            )
        elif action == "verify_grasp":
            return await self.verify_grasp()
        elif action == "request_perception_update":
            result = await self.request_perception_update()
            return bool(result.get("detected_objects"))
        else:
            log.error(f"Unknown action: {action}")
            return False
