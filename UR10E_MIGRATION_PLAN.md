# UR10e migration plan

Companion to `UR10E_MIGRATION_INVENTORY.md` (line-level list of every old setting).
Drafted 2026-10-12.

**Decisions**
1. **One source of truth.** Robot settings live in `generative_kitting/robot_config/<robot>/robot.yaml`. No robot constants are hard-coded in control code.
2. **cuMotion URDF/XRDF** are produced only with Isaac Sim tools: the URDF Exporter extension, and the Robot Description Editor for the XRDF and collision spheres. The robot's own USD colliders are set up properly with Isaac's robot setup tools first.
3. **The pre-existing bugs (B1–B4)** are fixed as part of this migration.

**Target layout**
```
generative_kitting/robot_config/
  ur10/    AIKIDO.urdf, AIKIDO.xrdf, AIKIDO.yaml, meshes/, robot.yaml   ← frozen old robot (reproducibility)
  ur10e/   robot.yaml, AIKIDO.urdf, AIKIDO.xrdf, meshes/                ← new
.env:  ROBOT_CONFIG_PATH=.../robot_config/ur10e     ← selects the robot
```
> **Note:** the UR10 files have already been moved into `robot_config/ur10/`, but `.env` still points `ROBOT_CONFIG_PATH` at `robot_config/`. Until Step 1 lands, the bridge cannot find `AIKIDO.urdf`, so cuMotion init fails.

Every step ends with a **live check** in Isaac Sim, run through the MCP tools and the bridge's HTTP endpoints, before the next step starts.

---

## Step 0: Baseline ✅ done (commit dad5b16)
- The working tree has many uncommitted changes, including `isaac_sim_bridge.py`, `config.yaml` and `robot_controller.py`. Commit them, or snapshot them, on a branch (e.g. `ur10e-migration`) so each later step is one reviewable diff.
- **Check:** `git status` is clean on the new branch.

## Step 1: Define `robot.yaml` and its loader ✅ done (commit f4516cd)
- **Schema** (proposed sections):
  - `robot`: name, stage USD path
  - `articulation`: root prim, DOF count
  - `joints`: gantry joint, arm joint names in cspace order, gripper driver joint, mimic/passive rules
  - `links`: base, ee, tool frame for cuMotion, all with relative prim paths
  - `tcp`: offset vector, frame
  - `orientation`: tool-down quaternion, scan quaternion
  - `gantry`: x offset, limits
  - `home`: joint positions
  - `limits`: max velocity, max acceleration
  - `gripper`: type, stroke, max rad, open, close min/max, overshoot, entry clearance, contact sensor prims, settle frames, adaptive-close parameters
  - `camera_mount`: prims, mount link, self-filter radius
  - `cumotion`: URDF/XRDF filenames, link-name → prim map for self-filtering
- **Loader:** `generative_kitting/utils/robot_config.py`. Plain Python + PyYAML (6.0.3 is available inside Isaac). It must work in two places: the Script Editor (bridge) and Streamlit/CLI. It returns a typed object, resolves prim paths against the articulation root, and fails loudly if a required key is missing.
- **Write two files:** `ur10/robot.yaml` holds today's values, so the old robot stays reproducible. `ur10e/robot.yaml` uses the measured stage values, with placeholders where Step 3 still has to measure.
- **Repoint `.env`:** `ROBOT_CONFIG_PATH` → `robot_config/ur10e`.
- **Check:** unit tests for the loader. Every prim path in `ur10e/robot.yaml` must resolve on the live stage (one MCP script).

## Step 2: Bridge reads `robot.yaml`, plus fix B1 ✅ done (commit 060a70f)
- Import the loader as `generative_kitting.utils.robot_config`, with the repo root on `sys.path`. Inside Isaac, a bare `utils` resolves to OpenCV's `cv2/utils`.
- Replace the hard-coded constants in `isaac_sim_bridge.py` (L135–186 and L1447–1473) with loader values. The module-level names stay, so the ~20 call sites don't change in this step.
- Remove the robot keys from `config.yaml` that are now unused, or make them point at `robot.yaml`.
- **B1:** the gantry joint lookup compares the full prim path with the short DOF name, so it never matches. Fix it to use the short joint name from the YAML.
- **Check:** the bridge starts on the UR10e stage. The joint report lists 15 DOFs, the contact sensor initialises, and a home move and a gantry jog both work.

## Step 3: Measure the new geometry live ✅ done
- **TCP:** the offset from `robotiq_base_link` to the finger-pad midpoint, and the axis it lies along. This replaces the five conflicting values.
- **Tool-down quaternion** for the new tool frame (it replaces `[1,0,1,0]`). Check the scan orientation too.
- **Fingers:** the closed angle and the width-to-angle curve of `finger_joint` on the new drives.
- **Wrist camera → TCP offset:** `wrist_target_x/y_offset`.
- Write the results into `ur10e/robot.yaml`.
- **Check:** with the arm at home, place a debug marker at the TCP computed from the YAML. It should sit between the finger pads in the viewport and in a capture.

## Step 4: Rebuild cuMotion inputs with Isaac Sim's own tools (no hand-written URDF or spheres)
- **Robot colliders:** inspect and fix the USD physics colliders on every robot link (gantry, vagn, UR10e links, 2F-140) using Isaac Sim's robot setup tools: `isaacsim.robot_setup.collision_detector` and the physics collider debug view. Every link should have a proper collider, and there should be no self-collisions at home pose.
- **URDF:** export it with the **URDF Exporter** extension (`isaacsim.asset.exporter.urdf` / `.ui`) from `/World/gantry`, giving the chain gantry → vagn → ur10e → robotiq. Review the exporter output; any cleanup is done by changing the exporter settings or the stage, not by hand-editing the file.
- **XRDF:** build it in the **Robot Description Editor** (`isaacsim.robot_setup.xrdf_editor`):
  - Select the 7 cspace joints and set the gripper joints as fixed.
  - Generate collision spheres with the editor's sphere generator for every link, including `vagn`, `base_link` and the gantry carriage.
  - Generate the self-collision ignore list in the editor.
  - Export the XRDF into `robot_config/ur10e/`.
- **Progress:**
  - Self-collision check: 0 pairs at the spawn pose (jointed links are skipped by PhysX).
  - URDF exported. The exporter had a bug: nested links dumped their meshes into the parent link. Patched locally in Isaac, `isaacsim.asset.exporter.urdf/.../converter/link_reader.py` (`it.PruneChildren()`, backup `.orig`).
  - XRDF editor cannot select the articulation's own top link (`gantry`, the whole portal frame). Patched locally: `isaacsim.robot_setup.xrdf_editor/.../sphere_generation.py` (backup `.orig`).
  - `gantry ↔ vagn` is in the editor's auto ignore list (joint-connected), so the sliding vagn never self-collides with the frame. Because the frame is a robot link, the self-filter removes it from the depth data, so nvblox never puts it in the ESDF.
  - The editor needs a fresh Play; a stale physics view leaves its panels empty. Fix: Stop → Play, then select the articulation.
  - Exporter mimic bug: it copied PhysX `gearing` straight into the URDF multiplier, but PhysX means `q + gearing·q_ref + offset = 0`. Patched in `joint_reader.py` (multiplier = −gearing; backup `.orig`).
  - XRDF editor writes default positions for continuous joints (inner fingers), and cuMotion's loader then fails with "bad optional access". Patched `xrdf_io.py` and `editor_state.py` to skip them (backups `.orig`). Also set `finger_joint` default to 0, because −0.0005 from the sim is outside its limits.
  - Export the URDF only while no editor spheres are on the stage; otherwise they are exported as visuals. Editing editor files hot-reloads it, which drops its spheres. Restore them with Import XRDF.
  - Re-export into an **empty** folder. The exporter suffixes mesh names instead of overwriting them.
  - The gantry rail has no physics collider (user chose to skip it). The URDF has no velocity limits; check this at cuMotion init.
- **Sphere keys:** the XRDF uses URDF link names. The self-filter maps link names to prims through `robot.yaml`, which removes the old USD-path key problem (B5).
- **Check:**
  - Forward kinematics from the URDF should match the stage link poses within 1 mm, at about 5 joint configurations.
  - cuMotion init reports no unmatched cspace names, and the start state is not in collision.
  - `_plan_test` succeeds.
  - Self-filtering removes the robot from the point cloud.

## Step 5: Frames and motion bugs (B2, B3, B4)
- **One TCP convention:** targets are always the TCP. A single function converts TCP to tool frame, and feedback reads the same frame. This fixes B2 (TCP offset added twice) and B3 (`ee_link` vs TCP mixing).
- **B4:** `zp` is undefined in `_pick_descend`.
- Re-check the pick/place height maths: approach, descent, transit and drop heights.
- **Check:** step-by-step live run. Move to approach, then descend with contact stop: the fingertips should stop at the expected height above a known part. Then retract and transit.

## Step 6: Gripper re-tune
- **`set_finger_joints`:** rewrite it for the new drive layout. The outer finger joints are now driven (k=2000); the inner fingers are springs.
- The inner finger joints are weak springs (k=0.002), so `zero` mode barely holds them; measured about −0.34 rad when closed.
- With the gripper pointing down, `open` (0.0) only reached 0.054 rad (gap 123 mm instead of 141 mm).
- Replace the linear width→angle mapping with `gripper.width_table`. Re-check `close_min` (0.55 rad ≈ 42 mm, not the 80 mm its old comment claims).
- Re-tune adaptive close, the grasp-confirm threshold, the overshoot and the settle frames. All of these move into the YAML.
- Make grasp verification read the driver joint by name, not a hard-coded index.
- **Check:** grasp at least 3 part sizes (small tube, motor valve, large gear). Record contact force and finger angle each time.

## Step 7: Secondary consumers
- **`execution/robot_controller.py`:**
  - Load `robot.yaml`.
  - Drop the stock-UR10 URDF patching.
  - Fix its gantry lookup.
  - Fix or remove the Lula path.
- **`isaac_sim_launcher.py` and `diagnose_motion.py`:** read paths, home pose and joint indices from the YAML.
- **`workflow_engine.py`:**
  - Remove duplicated robot defaults, such as the TCP offset and finger close; read them from config instead.
  - Clean up `effective_reach` and the other defaults that differ from `config.yaml`.
- **Workspace bounds:** keep one copy. Merge `validators.py` ×2 and the `safety.py` fallback.
- **LLM prompts:** change "UR10" → "UR10e" and "RSD455" → "D555". Re-check the camera-height text.
- **Check:** `pytest tests/` passes; mock-mode CLI runs.

## Step 8: Docs and cleanup
- Update `CLAUDE.md`, `AGENTS.md` and `GEMINI.md`: robot description and the scene reference table.
- Update the header of `robot_self_filtering_spec.md`.
- `PIPELINE_UPDATES_PENDING.md` §4–5: re-measure on the new stage which prims move with the robot, and the label placement.
- Archive `src/` (legacy) and `robot_config/ur10/AIKIDO.yaml` (dead Lula config).
- Fix `gripper_max_width` (0.085 → 0.140) or remove it.

## Step 9: End-to-end validation
- Full UI run: "Pick all motor valves and place them into tray compartment 1". Record success rate and failure modes using the evaluation recorder, and compare against the UR10 baseline in `THESIS_EVALUATION_RESULTS.md`.

---

**Out of scope** (tracked in `PIPELINE_UPDATES_PENDING.md`): the depth-only annotator fix (§1–2), the cuMotion obstacle pool (§3), the semantic-ID issue (§6), and the `.env` host paths in Docker.
