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

## Step 4: Rebuild cuMotion inputs with Isaac Sim's own tools (no hand-written URDF or spheres) ✅ done
- **Result:**
  - URDF + XRDF load in cuMotion with 7 cspace joints. FK matches the stage within 0.77 mm.
  - No self-collision at spawn or tool-down.
  - `plan_test` from tool-down finds paths to all 5 targets, with and without the 6000-sphere camera world.
  - Self-filter: 786 spheres from `AIKIDO.editor.xrdf` via `cumotion.link_prims`. With the arm below z 1.35, the start state is clear of the camera world (min 5 mm).
  - Also fixed: `_plan_cspace` passed `q_target=`, but the API takes `q_final=`.
- **Follow-ups:**
  - About 29 robot surface points (forearm, wrist_3) survive the self-filter. Fix by raising `sphere_margin` (0.02) or adding forearm spheres.
  - ✅ Planning speed: the XRDF `world_collision` group now leaves out gantry and vagn (`rebase_xrdf_spheres(..., world_collision_exclude=("gantry", "vagn"))`). A full `plan_test` takes 30 s instead of about 7 min (per target 0.9–1.2 s, bin height 11 s, was 20–26 s / 5 min). Self-collision still uses all 786 spheres.
  - `plan_test` default targets are unreachable from the straight-arm spawn pose; start it from a tool-down pose.
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

## Step 5: Frames and motion bugs (B2, B3, B4) ✅ done
- **Result:**
  - One TCP convention: `get_tcp_pos()` is the tool frame plus `tcp.offset` along the tool's +Z. `_tool_pose_from_tcp` is its inverse for any orientation. All targets are TCP heights.
  - B2: removed the extra `+ _tcp_offset()` in `_pick_z_positions`, `_pick_descend` and `_place_object_ik`.
  - B3: all 19 motion feedback reads use the TCP. `ee_link` does not even move with the arm on this stage.
  - B4: `_pick_descend` uses `_pick_z_positions`.
  - Cartesian micro-steps use `ik_step`, damped least squares on cuMotion's Jacobian seeded from the current joints, instead of a planner call per step. It rejects joint jumps over 0.5 rad.
  - `_pick_descend` makes a planned XY align instead of a gantry-only slide (that slide dragged the TCP 866 mm).
  - Also fixed:
    - `_update_base_pose` array shape.
    - HTTP results are matched to their command id; a timed-out result used to be returned to the next request.
  - Live, on `large_gear_1`:
    - approach 1.1 mm off target, tool vertical
    - descent stopped 0.4 mm from target z, 1.3 mm XY drift
    - close + retract lifted the gear 0.33 m; transit TCP within 1 mm
- **Open items, resolved 2026-10-12:**
  - ✅ Home: `robot.yaml home_joints` is a non-singular tool-down pose, TCP (−0.05, −0.30, 1.08), between rack and tray. Smallest singular value 0.23 (the Step 3 pose: 0.00014), 26 cm from the scene. `/api/home` follows a planned cspace path: from spawn through the camera world in 18 s. A straight 20 cm descent and slide from home are accurate to 1–2 mm. Never start Cartesian moves from the Step 3 pose; it is a wrist singularity.
  - ✅ `_move_gantry_x(hold_ee_pos)` steps the gantry and re-solves the arm with the gantry locked (`ik_step(fixed_joints=(0,))`). `realign_gantry`: 0.35 m slide, 1.1 mm TCP drift. A plain gantry slide uses a planned cspace path.
  - ✅ Retract/transit/place moves go through `_apply_interpolated`, so they follow the planned trajectory instead of jumping to the goal joints.
  - ✅ Editor sphere prims removed from the stage via the editor's own clear (re-import `AIKIDO.editor.xrdf` to edit again).
  - ✅ Wrong `extent` attributes (mm values, 1000× too big) fixed in 8 asset files (`UR10e_Container_Assets/SubUSDs/{floor,robot_facade_full,motor_valve,box_table,box_840,pallet,gantry,vagn}.usd`). USD bboxes (`scene_parts`, bin/tray lookups) are correct again.
  - ✅ HTTP timeouts for planned motion endpoints raised (home 120 s, approach 180 s, pick/place 180–300 s).
  - Remaining for Step 6:
    - The pad contact sensor did not fire on the gear.
    - Grasp depth: the open fingertips reach only 19 mm below the TCP.
    - The place "arm config reset" was a Lula warm-start trick and just re-solves the same pose with cuMotion; revisit.
  - Do not save the live stage over `AIKIDO_ur10e_swapped.usd`: its root layer holds simulation state and the bridge's runtime contact sensors.

## Step 6: Gripper re-tune 🟡 in progress (items 1–3 done 2026-10-13)
- **Done:**
  - **Passive four-bar + armature** (2026-10-13, replaces the earlier parallel-pad drives). The asset leaves the inner finger and pad joints unactuated, and with the outer finger joints `free` each side's loop had 5 revolutes (2 DOF), so the pads flopped and re-opened only to 122 mm. Now:
    - Modes: outer finger = `zero` (rigid, as on the real 2F-140); inner and pad = `free`. No inner/pad drives: they held the linkage at the contact angle (`STATE.pad_hold`) and ate the squeeze.
    - `gripper.armature: 0.001` on all 8 gripper joints (bridge start, `set_armatures`). See item 3 below for why.
    - Free air: pads parallel to 0.0002 rad, re-open repeatable to finger 0.0. Holding: pads parallel within 0.03 rad (<2°), mimic lag ≤ 0.009 rad.
  - `gripper.width_table` re-measured with parallel pads: `[finger rad, pad gap m, pad-face centre depth m]`. Loader has `gap_for_finger()` / `finger_for_gap()` (linear fallback for UR10). Bridge mappings use them.
  - `finger_joint` torque capped at 10 N·m (`joint_gains`; 30–60 N per pad over the stroke, see item 3). The asset's 1000 N·m was unrealistic.
  - **Contact sensor fixed.** The bridge called `get_current_frame()`, which does not exist in Isaac 6.1's experimental ContactSensor, so it always read 0. It now uses `get_data()`.
  - Adaptive close:
    - starts from the current angle
    - contact = sensor ≥ `min_force` or finger stall (`stall_rad`)
    - squeezes `overshoot_rad` past contact
    - parameters come from `robot.yaml`
  - `_verify_grasp`: held = sensor force ≥ `min_force` or stalled, and not fully closed. `pick_close` no longer reports "confirmed" by assumption.
  - `pick_retract`: straight vertical lift with `ik_step` (grip held), then a hold check. A planned, non-vertical path dragged the part out of the fingers.
  - Live pick test script: `generative_kitting/tools/pick_test.sh X Y TOP_Z [GRASP_LIFT]`.
- **Results:**

  | Part | Result |
  |---|---|
  | `large_gear_1` (95 mm, 42 mm tall, 0.15 kg) | Picks and lifts reliably. Contact (sensor) at 0.319 rad, hold 48 N (47 N after item 3), hold check after lift OK. Run: `pick_test.sh -1.5206 -0.7157 0.8345` |
  | `motor_valve_1` (0.33 kg) | Picks and lifts (2026-10-13, after item 3). Contact (sensor) at 0.297 rad, hold 49.5 N, valve lifted ~0.4 m with the gripper. Run: `pick_test.sh -0.632 -0.732 0.8655 -0.0245` |

- **Motor valve notes:**
  - The flat band across the pads is only z 0.807–0.831 (about 102 mm wide; centre x −0.580, y −0.717 at spawn). Above and below it the valve is narrower or rounded.
  - The valve is yawed about 4° and tilted about 4.6°, so the pads meet the rectangular flange at corners or edges (point contact, little friction torque) and it pivots out.
  - The last attempt (`pick_test.sh -0.5804 -0.717 0.8655 -0.0465`): contact at a 133 mm pad gap, 40 N, lost on lift. The pads probably touch something wider than the band at that height. Check which links and vertices touch.
  - **Measured 2026-10-13 (item 1 done):**
    - The pads close along world Y. At x −0.58 the valve is 117–125 mm wide in Y, not 102 mm. At the old attempt the right pad touched the mesh on only a 3×5 mm patch at its −x edge, and the left pad was 6.4 mm from the mesh.
    - **Root cause of the "133 mm, 40 N" grip:** at `grasp_lift −0.0465` the pad bottoms start at z ≈ 0.795. The bin floor is at 0.7935. Closing lowers the pads about 16 mm (width_table depth 0.1756 → 0.192 at a 95 mm gap), so the **left pad hit the bin floor** (overlap query: `robot_facade_full`) before it reached the valve. The pad contact sensor counts any contact, so it reported 43 N and `grasp_confirmed: True` while neither pad touched the valve.
    - The physics collider (convexDecomposition, raycast) matches the visual mesh at the grasp area, so the hulls are not the problem.
    - Better grasp point: the body block at x ≈ −0.632, y ≈ −0.732 (spawn), flat on both sides, 90–97 mm wide over z 0.80–0.862. With `pick_test.sh -0.632 -0.732 0.8655 -0.0245`, both pads sit flat on the valve (patches ~17–22 × 40 mm each). The valve moved only 6 mm and turned 0.9° during the close. Contact at 0.297 rad (94 mm), holding 0.329 rad (89 mm).
    - **Still lost on lift:** the pad force is only 1.5–4.8 N (sensor), against 48 N on the gear at a similar angle. That makes item 3 the blocker for the valve now.
    - Item 3 leads: `finger_joint` held 0.035 rad short of its target (k = 114 592 should saturate the 10 N·m cap), yet the projected joint effort reads only 0.32 N·m. `right_outer_knuckle_joint` lags `finger_joint` by 0.043 rad. It is a **soft PhysX mimic** (`naturalFrequency 1000`, `dampingRatio 1`, gearing −1). The outer finger joints also carry k = 114 592 drives (max 1000 N·m) that lock the four-bar. Find where the 10 N·m goes: the mimic compliance, or the over-constrained loop (outer finger drives + inner/pad drives). Measure pad force by contact report on both pads, not the left sensor alone.
    - Follow-ups: (a) the grasp check needs to ignore non-part contacts (filter the sensor to the target body, or require both pads in contact); (b) the descent should keep the closing pad arc above the floor (`grasp_z` ≥ floor + pad drop for the expected width + margin).
- **Resume here (next session):**
  1. ~~Valve: measure contacts at the last attempt.~~ Done 2026-10-13, see above. The valve now grasps flat at (−0.632, −0.732, top 0.8655, `grasp_lift −0.0245`) but slips because the pad force is too low.
  2. ~~Grasp yaw.~~ Done 2026-10-13.
     - `approach` and `pick_descend` take an optional `yaw` [rad] about world +Z (0 = pads close along world Y), folded into ±90° because the gripper is symmetric. `pick_descend` re-aligns with a planned move if XY or orientation is off. `pick_retract` lifts at the current tool orientation, so it never twists a held part. `pick_test.sh` takes `YAW_DEG` as a 5th argument. The workflow engine passes `pick_coords` through, so a perception `yaw` will reach the bridge (perception does not produce one yet).
     - `_tool_quat()` reads the orientation from cuMotion FK of the XRDF tool frame, the frame the IK targets.
     - `_ik_move_to` now fails if the arm ends > 20 mm or > 5° from the target. It used to return "ok" from anywhere.
     - Valve at 4°: by hand it picks (0.04° yaw error, 49 N, hold OK). `pick_test.sh` from a fresh start failed 3/3. The approach's gantry pre-slide always fails ("NO PATH to cspace target", and the error is ignored). The pose planner then picks the gantry and arm configuration itself: at yaw 0 it reliably lands on gantry −1.86, but with any yaw it often picks gantry ≈ −0.85 with wrist_1 ≈ +2.85, and the arm physically stops ~1.4 rad short. Not fixed here; the collision world moves to nvblox in Step 7 (see there).
  3. ~~Pad force across the stroke.~~ Done 2026-10-13.
     - **Root cause: solver, not drive layout.** The driven `left_outer_knuckle` is 9 g with I ≈ 2e-7 kg·m², inside a closed loop. PhysX passed only ~1 N·m of the drive to the pads: pad force was identical at caps of 1, 10, 30 and 100 N·m, and a 40 N push opened the finger fully. Not the cause: joint limits, joint friction, self-collision, sleeping, 8 velocity iterations, 240 Hz.
     - **Fix: joint armature 1e-3 kg·m² on the gripper joints.** Same 10 N·m cap: 94 mm block 15.6 → 58.7 N per pad, holds an 80 N side push (0.6 mm). Armature 1e-4 lets the pads tilt (0.05–0.07 rad); 1e-2 halves the force. 128 position iterations raise the force but tilt the pads 0.3–0.6 rad.
     - With armature the inner/pad drives are not needed for parallel pads, and they were acting as a brake: with frozen `pad_hold` targets, commanding the finger open to 0.25 rad left it at 0.300 rad.
     - Pad force against finger angle (floating 0.3 kg block, 10 N·m cap, contact report on both pads):

       | Block | Finger rad | Pad force L / R |
       |---|---|---|
       | 130 mm | 0.104 | 29.5 / 29.5 N |
       | 110 mm | 0.223 | 33.8 / 33.8 N |
       | 94 mm | 0.314 | 38.7 / 38.7 N |
       | 75 mm | 0.418 | 49.8 / 49.8 N |
       | 57 mm | 0.515 | 59.8 / 60.0 N |
       | 37 mm | 0.615 | 55.6 / 56.0 N |

       All inside the real 2F-140's 20–125 N, so the cap stays at 10 N·m.
     - Left unused: the `pad` / `mirror` modes and `STATE.pad_hold` in the bridge. Remove them, or keep them for another gripper.
  4. Third part size. The scene has only gears and valves (no small tube). Use a second gear orientation or add a small part.
  5. Re-run the gear test as a regression check, then commit and mark Step 6 done. Stale `close_min` / `close_max` / `close` values (0.55 / 0.69 / 0.5 rad) should be re-checked against the new width_table (`close` = 57 mm gap).
- **How to start a session:**
  - Open `UR10e_Container_Assets/AIKIDO_ur10e_swapped.usd` in Isaac Sim 6.1. Enable the isaac-sim MCP extension (port 8766; `python_server` stays on 8226).
  - In the Script Editor: `exec(open(".../generative_kitting/kitting_bridge_server.py").read())`. The bridge plays the timeline; Stop resets the scene and shuts the bridge down.
  - Do not save the live stage: its root layer collects simulation state and runtime contact sensors.
- **`set_finger_joints`:** rewrite it for the new drive layout. The outer finger joints are now driven (k=2000); the inner fingers are springs.
- The inner finger joints are weak springs (k=0.002), so `zero` mode barely holds them; measured about −0.34 rad when closed.
- With the gripper pointing down, `open` (0.0) only reached 0.054 rad (gap 123 mm instead of 141 mm).
- Replace the linear width→angle mapping with `gripper.width_table`. Re-check `close_min` (0.55 rad ≈ 42 mm, not the 80 mm its old comment claims).
- Re-tune adaptive close, the grasp-confirm threshold, the overshoot and the settle frames. All of these move into the YAML.
- Make grasp verification read the driver joint by name, not a hard-coded index.
- **Check:** grasp at least 3 part sizes (small tube, motor valve, large gear). Record contact force and finger angle each time.

## Step 7: nvblox collision world for cuMotion ✅ done (2026-10-14, 7.5 awaiting review)
Target pipeline: 4 corner depth cameras → robot self-masking → nvblox (TSDF → ESDF) → one cuMotion SDF obstacle → planner. It replaces the point-cloud → sphere world: the 6000-sphere cap (93 831 voxels were cut to 6000 on 2026-10-13), the target carve-out, and the static-world scan.
- **Constraints found 2026-10-13:**
  - cuMotion 1.1.0a2 (bundled with Isaac 6.1) supports SDF obstacles: `create_obstacle(Obstacle.Type.SDF)` with an `Obstacle.Grid(nx, ny, nz, voxel_size, host_precision, device_precision)`, filled by `World.set_sdf_grid_values_from_host(values)` (z fastest) and checked with `World.inspect_sdf`.
  - nvblox: `nvblox_torch 0.0.10+cu12ubuntu24` (GitHub release, matches Ubuntu 24.04 / CUDA 12.8). It requires `torch<=2.9.1` and `numpy<1.27`, and `libpy_nvblox.so` links libtorch directly. Isaac 6.1 runs torch 2.11 / numpy 2.x, so nvblox cannot load inside Isaac's process.
- **Design:** nvblox runs as a separate local service in its own venv (torch 2.9.1 + cu128). Per world update the bridge sends, for each depth camera: depth image, intrinsics, world pose and a robot mask. The service integrates them and returns a dense ESDF over the workspace box, and the bridge writes it into the cuMotion SDF obstacle.
- **Steps:**
  1. ~~Venv + service skeleton.~~ Done 2026-10-13: `generative_kitting/nvblox_service/` (`requirements.txt`, `setup_venv.sh`, `run.sh`, venv in `.venv/`, git-ignored). torch 2.9.1+cu128, nvblox_torch 0.0.10.
     - Gotchas: the wheel's metadata version (0.0.10.dev1) does not match its filename, so `uv` needs `UV_SKIP_WHEEL_FILENAME_CHECK=1`. `libnvblox_lib.so` needs NPP (`nvidia-npp-cu12`) and has no RPATH to pip's `nvidia/` folders, so start python through `run.sh` (sets `LD_LIBRARY_PATH`).
     - nvblox tests with Isaac running: the 16 files we need pass (frame integration, masking, TSDF layer, ESDF update/query/gradients). 3 fail on CUDA out-of-memory (dummy maps of ~191k blocks; dense-layer export of 206 MB); 2 crash in mesh saving (not used).
     - **GPU budget (measured 2026-10-13, RTX 4070 Laptop 8 GB):** Isaac Sim with the scene, bridge, cameras and cuMotion: 5.4 GB. nvblox process, synthetic 4-camera cell (floor + bin-sized box), dense export box 3 × 2 × 1.5 m:

       | Voxel | After mapper init | After ESDF + dense export | ESDF update |
       |---|---|---|---|
       | 5 cm | 504 MiB | 524 MiB | 29 ms |
       | 2 cm | 504 MiB | 740 MiB | 89 ms |
       | 1 cm | 504 MiB | 1508 MiB | 278 ms |

       The real cell has more surface than the synthetic one (walls, gantry, tray, parts), so expect ~1.0–1.3 GB at 2 cm. Use 2 cm: Isaac + nvblox ≈ 6.5–6.7 GB of 8 GB. 1 cm does not fit beside Isaac. Limit integration to the workspace (max integration distance / crop) and never build whole-scene dense layers. The VLM runs on a remote server and does not use this GPU.
  2. ~~Bridge: capture depth + intrinsics + pose per camera, plus a robot mask from the XRDF spheres.~~ Done 2026-10-14 (awaiting review). `POST /api/nvblox_update` renders `distance_to_image_plane` on the pointcloud cameras' render products, masks pixels inside the XRDF spheres at the live joint state or outside `nvblox.workspace_min/max`, and posts depth + mask + K + optical pose to `nvblox_service/server.py` (`/integrate`, `/query`, `/clear`, `/status`, port 8610). Settings in `robot.yaml` `nvblox:`. Start the service with `nvblox_service/run.sh generative_kitting/nvblox_service/server.py` (from the repo root).
     - **Camera-model bug fixed:** `_intrinsics_matrix` used the USD `verticalAperture`, but the renderer keeps the horizontal aperture and draws square pixels. cam1–4 are authored with 15.29 instead of 15.72 at 480×640, which put their floors 10–40 cm apart. Now fy = fx, and the cameras agree to 5–8 mm. No change for rgb/depth/wrist/kit (CameraSensor already syncs the aperture).
     - **Cameras:** cam1–4 cannot see into the rack: the bin parts are 71–100 % occluded. `/World/Camera` and `/World/Camera_Kit` were added (6 cameras).
     - **Robot mask (checked by hiding the robot):** margin 4 cm, vagn 5 cm. 0 moving-robot pixels leak at spawn and home. The static frame's spheres sit up to 8.8 cm inside its surface; that rim stays in the map as real structure (cuMotion skips world collision for gantry/vagn). A 10 cm frame margin erased the rack end at x −2.35, so it is not used. At home, the nearest mapped surface is 26 cm from the arm spheres.
     - **ESDF accuracy:** p90 error ≤ 2.5 cm against exact distances to the integrated points (0.1–2 m from surfaces). `esdf_max_site_distance_vox: 2.0` (the default 1.0 overestimated by 4+ cm near small parts). The ESDF still reads ~+1–2 cm too far near surfaces, so **Step 7.3 needs ≥ 2–3 cm of safety margin**.
     - **Map updates:** incremental (`clear` is false by default). `Mapper.clear()` does not return blocks to the pool, which doubled (16k → 131k blocks over 8 rebuilds) and ran the GPU out of memory once. `tsdf_decay_factor: 0.5`: a removed part leaves the map in 2 updates; space nothing sees (e.g. under the arm) fades in ~10.
     - **Measured (RTX 4070 Laptop 8 GB, cuMotion not yet initialised):** nvblox 714 MiB flat over 10 updates (10.4k blocks in a 5.2 × 3.2 × 2.15 m box). Isaac 3.7–4.3 GB, total ≤ 5.0 GB; with the 7.1 cuMotion figure (Isaac 5.4 GB) expect ≈ 6.1 GB. Per update: capture ~0.8–1.1 s (6 cameras, 5 render ticks), mask 0.06–0.2 s, nvblox integrate ~15 ms + ESDF 6–37 ms.
     - **For 7.4:** a part held by the gripper is not part of the robot spheres, so it would be mapped as an obstacle in the hand; mask it (attached-object spheres) along with the grasp target.
  3. ~~ESDF → cuMotion SDF obstacle; `inspect_sdf` clean; voxel size and box from `robot.yaml`.~~ Done 2026-10-14 (awaiting review). `POST /api/nvblox_world` (params as `nvblox_update`, plus `check`) updates nvblox, fetches `GET /sdf` from the service, subtracts `nvblox.sdf_safety_margin` (2 cm) and loads it as one cuMotion SDF obstacle (260 × 160 × 108 voxels at 2 cm, corner at `workspace_min`). The obstacle is reused while the grid shape stays the same and dropped when `_reset_cumotion_world` recreates the world.
     - **Not nvblox's ESDF:** it jumps by more than one voxel between 2–3 % of neighbour pairs (up to 0.27 m), which `inspect_sdf` flags and which is simply wrong. The service builds an exact signed EDT (scipy) from the TSDF occupancy (observed and distance ≤ 0); unobserved voxels count as free. Against the integrated points: median −0.4 to −0.7 cm (reads closer), p90 +0.6 cm; 99.5 % of surface points lie within 2 cm of an occupied voxel, the tray 99.6 % within 4 cm (149 points of a thin feature 5.3 cm off). About 0.8 s per export (CPU).
     - **cuMotion conventions (measured):** values at voxel centres, z fastest, the pose places the grid corner, poses in the robot base frame. `distance_to` reads √3/2 × voxel (1.73 cm) short at voxel centres and 0.7–1.7 cm short between them (conservative).
     - **Checks:** `inspect_sdf` 0 errors / 0 flat / 0 jumps. cuMotion distance − grid value = −0.0173 m at all 500 sampled voxel centres (the grid is placed exactly). Planning from home: no start-state collision. Targets inside the table body, the table top and the rack top: no path with the SDF, path found without it. 20 cm above the tray and above `motor_valve_2`: found. The grasp pose at `motor_valve_2`: no path (the target is an obstacle, see 7.4).
     - **Time per refresh:** capture ~1.1 s + mask 0.2 s + nvblox 0.1 s + SDF 0.9 s + load 0.01 s ≈ 2.3 s.
  4. ~~Replace the sphere world in `build_collision_world`; handle the target and the held part.~~ Done 2026-10-14 (awaiting review). User decision: **carve the target, mask the held part.** `build_collision_world` (same API: `exclude_points` = part tops, `carve_radius`) now resets the cuMotion world and loads the nvblox SDF; the point-cloud → sphere code is no longer called from it (removed in 5).
     - **Why not mask the target:** the descent and the retract are Cartesian (`_descend_with_contact_stop`, `_retract_cartesian_up`) and never check the SDF; only the transits are planned. Masked pixels are only "not seen", so the target would fade out by decay (~10 updates), and a precise mask needs a segmentation outline.
     - **Target carve:** the service (`POST /sdf` `{"carve": [[x, y, z_min, z_max, r]]}`) marks a vertical column free before the EDT, from `top − nvblox.target_carve_below` (0.10 m) to the box top. `target_carve_radius: 0.10`: valve corners reach 9.6 cm from its centre, and at 8 cm the open finger sat next to one. The field stays clean (`inspect_sdf` 0 errors).
     - **Held part:** `_pick_close` sets `STATE.held_part` when the grasp is confirmed (`nvblox.held_part_radius` 0.11 m, or `held_radius`), and opening the gripper clears it. The robot mask then includes a sphere at the TCP. Holding `large_gear_1` 25 cm up: the cameras saw 167 depth points on the gear, 0 integrated; plans from the holding pose (no move, ±0.10 m) succeed.
     - **Planning checks** (`plan_test` takes **tool-frame** positions, `robotiq_base_link`, TCP + 0.189 m): the grasp at the carved valve_2 is found; the neighbouring valve_1 (0.23 m away) stays blocked. The gear_1 grasp (yaw 0) is refused: the open right finger is ~1 cm from the bin side wall (y ≈ −0.84), inside the 2 cm margin + cuMotion's 1.7 cm. It does not affect today's pick (descent not planned); a planned descent near walls needs a grasp yaw that closes away from the wall.
     - **End-to-end (`tools/pick_test.sh`, fresh scene each):** `large_gear_1` (−1.5206, −0.7157, 0.8345): approach, descent, close 50.5 N, hold OK after lift. `motor_valve_1` (−0.632, −0.732, 0.8655, −0.0245): close 49.6 N, hold OK. A centre grasp of `motor_valve_2` (yaw 0) was lost on lift: the known valve-geometry issue from Step 6, not the collision world.
     - `workflow_engine.py` logs the new result fields; `camera_interface.build_collision_world` docs updated; `pick_test.sh` prints the carve and `inspect_sdf`.
  5. ~~Remove the dead paths: sphere cap, static-world scan, carve-out; and the approach's separate gantry pre-slide whose failure is ignored.~~ Done 2026-10-14 (awaiting review). Removed from the bridge: `COLLISION_*` / `TARGET_CARVE_RADIUS` / `STATIC_WORLD_*`, `_voxel_downsample`, `_carve_around_points`, `_spheres_from_points`, the static scan (`_save/_load/_build_static_world`, `/api/build_static_world`) and the wrist-camera spheres (`_add_wrist_obstacles`, `/api/wrist_obstacles`, unused by the workflow). Also removed: the matching `camera_interface` methods and `collision_sphere_count`; `/api/status`, `plan_test` and the plan logs now report `collision_world` ("nvblox SDF 260x160x108 @ 2 cm" or "empty"). `_capture_pointcloud` / `_filter_robot_self_points` stay (surface_z, diagnostics).
     - **Gantry pre-slide:** a gantry-only cspace move from home drags the hanging arm through the rack, so it never planned ("NO PATH to cspace target" on every approach, error ignored). Removed; the pose planner moves gantry and arm together (gantry = cspace joint 0).
- **Check:** ESDF distance at known points (bin wall, floor, tray) within one voxel; planned approaches at yaw 0 and ±30° reach the target (`_ik_move_to` check) in 5/5 fresh runs; gear and valve picks still pass.
- **Check result (2026-10-14): passed.** Distances: the SDF matches exact distances to the integrated points within one voxel (7.2/7.3). Approaches to 0.25 m above `large_gear_1` from home, world rebuilt each run, bridge restarted per yaw: yaw 0 5/5 (1.4 mm), +30° 5/5 (1.3 mm), −30° 5/5 (1.4 mm); no cspace NO PATH. `pick_test.sh` on fresh scenes: `large_gear_1` 50.5 N, hold OK; `motor_valve_1` 49.6 N, hold OK.
- **Later (user, 2026-10-13):** calibrate all cameras (intrinsics + extrinsics) so detected part coordinates and the nvblox map share one frame; then test the VLM pipeline end to end (part detection → 3D grasp target → planned pick). Camera poses come from USD until calibration replaces them.

## Step 8: Secondary consumers
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

## Step 9: Docs and cleanup
- Update `CLAUDE.md`, `AGENTS.md` and `GEMINI.md`: robot description and the scene reference table.
- Update the header of `robot_self_filtering_spec.md`.
- `PIPELINE_UPDATES_PENDING.md` §4–5: re-measure on the new stage which prims move with the robot, and the label placement.
- Archive `src/` (legacy) and `robot_config/ur10/AIKIDO.yaml` (dead Lula config).
- Fix `gripper_max_width` (0.085 → 0.140) or remove it.

## Step 10: End-to-end validation
- Full UI run: "Pick all motor valves and place them into tray compartment 1". Record success rate and failure modes using the evaluation recorder, and compare against the UR10 baseline in `THESIS_EVALUATION_RESULTS.md`.

---

**Out of scope** (tracked in `PIPELINE_UPDATES_PENDING.md`): the depth-only annotator fix (§1–2), the cuMotion obstacle pool (§3), the semantic-ID issue (§6), and the `.env` host paths in Docker.
