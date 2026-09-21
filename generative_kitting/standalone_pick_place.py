import os
import asyncio
import numpy as np
import warp as wp
from pxr import Gf

import omni.timeline
import omni.usd
import omni.kit.app

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.cumotion import (
    load_cumotion_robot,
    CumotionWorldInterface,
    GraphBasedMotionPlanner,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION (NO POINT CLOUDS / FREE SPACE PLANNING)
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_PART_PRIM = "/World/robot_facade_full/large_gear_1"
TRAY_PRIM = "/World/box_840"

ROBOT_PRIM = "/World/gantry"
ROBOT_ROOT_PATH = "/World/gantry"
TOOL_FRAME = "robotiq_base_link"
GRIPPER_TCP_OFFSET = 0.170  # metres (robotiq_base_link -> fingertips)

PROJECT_ROOT = "c:/KP/AI_and_Automation/Sem_4/Thesis/robot_in_air"
ROBOT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "generative_kitting", "robot_config")
ROBOT_URDF = "AIKIDO.urdf"
ROBOT_XRDF = "AIKIDO.xrdf"

DOWNWARD_ORIENTATION = np.array([0.0, 0.70710678, 0.70710678, 0.0], dtype=np.float64)
FINGER_OPEN = 0.0
FINGER_CLOSE = 0.58


def get_prim_translation(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim not found: {prim_path}")
    mat = omni.usd.get_world_transform_matrix(prim)
    t = mat.ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])])


# ─── CUMOTION CONTROLLER (FREE SPACE) ─────────────────────────────────────────

class CuMotionFreeSpaceController:
    def __init__(self):
        self.device = wp.get_device("cuda:0")
        self.stage = omni.usd.get_context().get_stage()
        
        # 1. Load cuMotion Model
        self.cumotion_robot = load_cumotion_robot(
            directory=ROBOT_CONFIG_PATH,
            urdf_filename=ROBOT_URDF,
            xrdf_filename=ROBOT_XRDF
        )
        self.controlled_joints = list(self.cumotion_robot.controlled_joint_names)
        
        # 2. Bind Articulation
        self.robot = SingleArticulation(prim_path=ROBOT_PRIM, name="cumotion_robot")
        self.robot.initialize()
        
        self.dof_names = list(self.robot.dof_names)
        self.cspace_indices = [self.dof_names.index(j) for j in self.controlled_joints]
        
        # Gripper master drive joints
        self.gripper_dofs = [
            i for i, name in enumerate(self.dof_names)
            if name in ["finger_joint", "right_outer_knuckle_joint"]
        ]
        
        # Track commanded target state
        self.current_targets = np.array(self.robot.get_joint_positions(), dtype=np.float64)
        
        # 3. Initialize Empty Collision World (Free Space)
        self.world_interface = self._init_empty_world()
        self.planner = GraphBasedMotionPlanner(
            cumotion_robot=self.cumotion_robot,
            cumotion_world_interface=self.world_interface,
            tool_frame=TOOL_FRAME
        )

    def _init_empty_world(self):
        root_prim = self.stage.GetPrimAtPath(ROBOT_ROOT_PATH)
        mat = omni.usd.get_world_transform_matrix(root_prim)
        t = mat.ExtractTranslation()
        q = mat.ExtractRotationQuat()
        imag = q.GetImaginary()
        pos = np.array([t[0], t[1], t[2]], dtype=np.float64)
        quat = np.array([q.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float64)

        with wp.ScopedDevice(self.device):
            pos_wp = wp.array(pos.reshape(1, 3).astype(np.float32), dtype=wp.vec3)
            quat_wp = wp.array(quat.reshape(1, 4).astype(np.float32), dtype=wp.vec4)
            
        world = CumotionWorldInterface(world_to_robot_base=(pos_wp, quat_wp), device=self.device)
        world.world_view.update()
        return world

    def get_current_cspace(self):
        q = self.current_targets if self.current_targets is not None else self.robot.get_joint_positions()
        q_cspace = np.array([q[i] for i in self.cspace_indices], dtype=np.float64)
        q_cspace[0] = np.clip(q_cspace[0], -3.09999, -1e-6)
        return q_cspace

    async def self_test_jog(self):
        """Test physical joint responsiveness by commanding a small shoulder nudge."""
        print("  [Self-Test] Testing physical joint actuation...")
        q0 = np.array(self.robot.get_joint_positions(), dtype=np.float64)
        tgt = q0.copy()
        test_idx = self.cspace_indices[1]  # shoulder_pan
        tgt[test_idx] += 0.2

        tgt_2d = np.ascontiguousarray(tgt, dtype=np.float32).reshape(1, -1)
        if hasattr(self.robot, "_articulation_view") and self.robot._articulation_view is not None:
            self.robot._articulation_view.set_joint_position_targets(tgt_2d)
        self.robot.apply_action(ArticulationAction(joint_positions=tgt))
        
        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()

        q1 = np.array(self.robot.get_joint_positions(), dtype=np.float64)
        moved = abs(q1[test_idx] - q0[test_idx])
        print(f"  [Self-Test] Joint[{test_idx}] moved {moved:.4f} rad -> {'OK (Physics Active)' if moved > 0.02 else 'FAILED (Frozen Articulation)'}")
        
        # Reset back
        tgt_2d_reset = np.ascontiguousarray(q0, dtype=np.float32).reshape(1, -1)
        if hasattr(self.robot, "_articulation_view") and self.robot._articulation_view is not None:
            self.robot._articulation_view.set_joint_position_targets(tgt_2d_reset)
        self.robot.apply_action(ArticulationAction(joint_positions=q0))
        for _ in range(30):
            await omni.kit.app.get_app().next_update_async()
            
        self.current_targets = q0.copy()
        return moved > 0.02

    def plan_trajectory(self, target_pos_world, target_quat_world=DOWNWARD_ORIENTATION, q_start=None):
        q0 = q_start if q_start is not None else self.get_current_cspace()
        
        path = self.planner.plan_to_pose_target(
            q_initial=q0,
            position=np.asarray(target_pos_world, dtype=np.float64),
            orientation=np.asarray(target_quat_world, dtype=np.float64)
        )
        if path is None:
            print(f"  [cuMotion Free-Space] No path found to target: {target_pos_world}")
            return None

        n = len(self.controlled_joints)
        return path.to_minimal_time_joint_trajectory(
            max_velocities=np.full(n, 1.2, dtype=np.float32),
            max_accelerations=np.full(n, 2.4, dtype=np.float32),
            robot_joint_space=self.dof_names,
            active_joints=self.controlled_joints
        )

    def get_trajectory_goal_q(self, trajectory):
        if trajectory is None:
            return self.get_current_cspace()
        st = trajectory.get_target_state(float(trajectory.duration))
        p = np.asarray(st.joints.positions).flatten()
        idx = getattr(st.joints, "position_indices", None)
        if idx is not None:
            by_dof = {int(d): p[s] for s, d in enumerate(np.asarray(idx).flatten())}
            return np.array([by_dof.get(d, 0.0) for d in self.cspace_indices])
        return p[:len(self.cspace_indices)]

    async def execute_trajectory(self, trajectory, finger_pos=None):
        if trajectory is None:
            return False

        duration = float(trajectory.duration)
        dt = 1.0 / 60.0
        t = 0.0
        
        q_start = self.current_targets.copy()
        current_stream = q_start.copy()
        frame_count = 0
        
        while t < duration:
            st = trajectory.get_target_state(t)
            if st is not None and st.joints.positions is not None:
                p = np.asarray(st.joints.positions).flatten()
                indices = getattr(st.joints, "position_indices", None)
                
                if indices is not None:
                    idx_arr = np.asarray(indices).flatten()
                    for s, d_idx in enumerate(idx_arr):
                        if d_idx < len(current_stream):
                            current_stream[int(d_idx)] = p[s]
                else:
                    for s, dof_idx in enumerate(self.cspace_indices):
                        current_stream[dof_idx] = p[s]
                
                if finger_pos is not None:
                    for g_idx in self.gripper_dofs:
                        current_stream[g_idx] = finger_pos

                # Push directly to 2D Fabric targets (shape: [1, num_dof])
                tgt_2d = np.ascontiguousarray(current_stream, dtype=np.float32).reshape(1, -1)
                if hasattr(self.robot, "_articulation_view") and self.robot._articulation_view is not None:
                    self.robot._articulation_view.set_joint_position_targets(tgt_2d)

                self.robot.apply_action(ArticulationAction(joint_positions=current_stream))

            await omni.kit.app.get_app().next_update_async()
            t += dt
            frame_count += 1

        for _ in range(15):
            await omni.kit.app.get_app().next_update_async()
            
        self.current_targets = current_stream.copy()
        moved = float(np.max(np.abs(self.current_targets - q_start)))
        print(f"  [Trajectory Finished] {frame_count} frames streamed | Commanded delta = {moved:.4f} rad")
        return True

    async def set_gripper(self, position):
        current_stream = self.current_targets.copy()
        for g_idx in self.gripper_dofs:
            current_stream[g_idx] = position
            
        tgt_2d = np.ascontiguousarray(current_stream, dtype=np.float32).reshape(1, -1)
        if hasattr(self.robot, "_articulation_view") and self.robot._articulation_view is not None:
            self.robot._articulation_view.set_joint_position_targets(tgt_2d)
            
        self.robot.apply_action(ArticulationAction(joint_positions=current_stream))
        self.current_targets = current_stream.copy()
        
        for _ in range(30):
            await omni.kit.app.get_app().next_update_async()


# ─── MAIN EXECUTION SEQUENCE ──────────────────────────────────────────────────

async def run_free_space_pipeline():
    print("\n" + "=" * 75)
    print("  cuMotion FREE-SPACE Motion Pipeline (No Point Clouds)")
    print("=" * 75)

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()
        for _ in range(20):
            await omni.kit.app.get_app().next_update_async()

    controller = CuMotionFreeSpaceController()
    stage = controller.stage

    # 1. Verify PhysX Joint Actuation
    await controller.self_test_jog()

    # 2. Extract Coordinates from USD
    part_pos = get_prim_translation(stage, TARGET_PART_PRIM)
    tray_pos = get_prim_translation(stage, TRAY_PRIM)
    print(f"  Target Part: ({part_pos[0]:.3f}, {part_pos[1]:.3f}, {part_pos[2]:.3f})")
    print(f"  Kitting Tray: ({tray_pos[0]:.3f}, {tray_pos[1]:.3f}, {tray_pos[2]:.3f})")

    # 3. Waypoint definition
    approach_pos = [part_pos[0], part_pos[1], part_pos[2] + 0.20 + GRIPPER_TCP_OFFSET]
    grasp_pos    = [part_pos[0], part_pos[1], part_pos[2] + 0.02 + GRIPPER_TCP_OFFSET]
    lift_pos     = [part_pos[0], part_pos[1], part_pos[2] + 0.25 + GRIPPER_TCP_OFFSET]
    tray_app_pos = [tray_pos[0], tray_pos[1], tray_pos[2] + 0.25 + GRIPPER_TCP_OFFSET]
    tray_plc_pos = [tray_pos[0], tray_pos[1], tray_pos[2] + 0.05 + GRIPPER_TCP_OFFSET]

    # 4. Sequential Planning & Execution in Free Space
    print("\n  Beginning Free-Space Pick & Place Sequence...")
    
    # Open
    print("  -> Opening Gripper")
    await controller.set_gripper(FINGER_OPEN)

    # Approach
    print("  -> Planning & Executing Approach")
    traj_approach = controller.plan_trajectory(approach_pos)
    await controller.execute_trajectory(traj_approach, finger_pos=FINGER_OPEN)

    # Descend
    print("  -> Planning & Executing Descent to Grasp")
    q_curr = controller.get_trajectory_goal_q(traj_approach)
    traj_grasp = controller.plan_trajectory(grasp_pos, q_start=q_curr)
    await controller.execute_trajectory(traj_grasp, finger_pos=FINGER_OPEN)

    # Close
    print("  -> Closing Gripper")
    await controller.set_gripper(FINGER_CLOSE)

    # Lift
    print("  -> Planning & Executing Lift")
    q_curr = controller.get_trajectory_goal_q(traj_grasp)
    traj_lift = controller.plan_trajectory(lift_pos, q_start=q_curr)
    await controller.execute_trajectory(traj_lift, finger_pos=FINGER_CLOSE)

    # Transit
    print("  -> Planning & Executing Transit to Tray")
    q_curr = controller.get_trajectory_goal_q(traj_lift)
    traj_transit = controller.plan_trajectory(tray_app_pos, q_start=q_curr)
    await controller.execute_trajectory(traj_transit, finger_pos=FINGER_CLOSE)

    # Place
    print("  -> Planning & Executing Lowering into Tray")
    q_curr = controller.get_trajectory_goal_q(traj_transit)
    traj_place = controller.plan_trajectory(tray_plc_pos, q_start=q_curr)
    await controller.execute_trajectory(traj_place, finger_pos=FINGER_CLOSE)

    # Release
    print("  -> Releasing Part")
    await controller.set_gripper(FINGER_OPEN)

    print("\n" + "=" * 75)
    print("  [SUCCESS] Free-Space Sequence Completed.")
    print("=" * 75)


asyncio.ensure_future(run_free_space_pipeline())