# UR10 → UR10e migration — inventory of old robot/gripper settings

Audit date: 2026-10-12. Read-only audit; nothing has been changed yet.
Ground truth = live stage `UR10e_Container_Assets/AIKIDO_ur10e_swapped.usd` in Isaac Sim 6.1 (queried via MCP).

---

## 0. New stage ground truth (measured live)

| Item | Old (UR10 stage) | New (UR10e stage) |
|---|---|---|
| Articulation root | `/World/gantry` | `/World/gantry` (unchanged) |
| Arm prefix | `/World/gantry/gantry_home/ur10_flattened/ur10_instanceable/` | `/World/gantry/gantry_home/ur10e/` |
| Gripper prefix | `.../ur10_flattened/robotiq_fixed_physics/Robotiq_2F_140_physics_edit/` | `/World/gantry/gantry_home/ur10e_robotiq_2f140/` |
| Arm links | base_link … wrist_3_link, ee_link | same names; `ee_link` still exists |
| Gripper mount | `ee_link` → `robotiq_base_link` (+0.0922 m) | `wrist_3_link` → `robotiq_base_link` via `ur10e_gripper_fixed_joint`; world positions coincide (≈0 offset) |
| Gantry joint | `gantry_vagn_joint`, [-3.1, 0] m | same; prim at `/World/gantry/gantry_home/vagn/...`; base_link world X ≈ 1.269 → `GANTRY_X_OFFSET 1.27` still valid |
| DOF count / order | 15: gantry, 6 arm, 8 gripper | 15: `gantry_vagn_joint, shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, finger_joint(7), right_outer_knuckle, left/right_outer_finger, left/right_inner_finger, left/right_inner_finger_pad` |
| Loop-closure joints | — | `left/right_inner_knuckle_joint` excluded from articulation |
| Elbow limit | ±2π | **±π** |
| finger_joint limit | [0, 0.785] | [0, 0.785] (unchanged) |
| Arm drives | — | pan/lift/elbow k=3271 d=13.1 (fmax 330/330/150); wrists k=1268 d=5.07 fmax 56 |
| Gripper drives | — | finger_joint & outer_finger k=2000 d=200; inner_finger k=0.002 (spring) |
| Contact sensor | `.../left_inner_finger/Fingertip_01/Fingertip/Contact_Sensor` | `/World/gantry/gantry_home/ur10e_robotiq_2f140/left_inner_finger/Contact_Sensor` (left finger only) |
| Wrist camera | `/World/rsd555` (top-level) | `/World/rsd555`, `FixedJoint_cam` to `ur10e/wrist_3_link`, localPos0 (0, -0.0582, 0.0465) |
| Gripper base orientation | — | robotiq_base_link world rot ≈ identity; ee_link/vagn rot = 90° about X — **tool-down quaternion must be re-derived** |

---

## 1. Live pipeline map (what actually matters)

```
streamlit_app / main.py → config.yaml → workflow_engine.py ──HTTP:8600──► isaac_sim_bridge.py (Script Editor)
                                                                           └─ reads $ROBOT_CONFIG_PATH/AIKIDO.urdf + AIKIDO.xrdf (cuMotion + self-filter)
```
- **The bridge does not read robot data from config.yaml** — it hard-codes its own copies. It is the main thing to fix.
- `execution/robot_controller.py` is the secondary `main.py --sim` path (Lula IK, currently never initialised).
- Dead/legacy: `src/` (robot_control.py, kitting_lula_config.yaml, kitting_*.urdf), `robot_config/AIKIDO.yaml` (Lula format, unreferenced).

---

## 2. `generative_kitting/isaac_sim_bridge.py` (LIVE, highest priority)

### Prim paths
| Line | Symbol | Current value | New value |
|---|---|---|---|
| 135 | `ROBOT_PRIM` | `/World/gantry` | unchanged |
| 148 | `UR10_BASE_PATH` | `.../ur10_flattened/ur10_instanceable/base_link` (unused) | `/World/gantry/gantry_home/ur10e/base_link` |
| 149 | `EE_PATH` | `.../ur10_flattened/ur10_instanceable/ee_link` (~20 call sites) | `/World/gantry/gantry_home/ur10e/ee_link` |
| 153–154 | `CONTACT_SENSOR_PRIM` / `_TIP_PRIM` | `.../Robotiq_2F_140_physics_edit/left_inner_finger/Fingertip_01/Fingertip/Contact_Sensor` | `.../ur10e_robotiq_2f140/left_inner_finger/Contact_Sensor` |
| 160 | `GANTRY_X_JOINT` | full prim path | short name `gantry_vagn_joint` (see bug B1) |
| 1462 | `ROBOT_ROOT_PATH` | `/World/gantry` | unchanged |
| 1466–1468 | `TOOL_FRAME_PRIM` | `.../Robotiq_2F_140_physics_edit/robotiq_base_link` | `.../ur10e_robotiq_2f140/robotiq_base_link` |
| 139/140/145 | camera/IMU prims | `/World/rsd555/{Depth, Camera_OmniVision_OV9782_Color, Imu_Sensor}` | unchanged (still valid) |
| 612–617 | self-filter camera housing | `/World/rsd555`, 0.10 m radius | path OK; re-check radius |

### cuMotion / kinematics
| Line | Symbol | Current value | Note |
|---|---|---|---|
| 1450–1451 | URDF/XRDF filenames | `AIKIDO.urdf` / `AIKIDO.xrdf` | files describe UR10 → regenerate (§4) |
| 1452 | `CUMOTION_TOOL_FRAME` | `robotiq_base_link` | name still valid; verify in new URDF |
| 1471–1473 | `CUMOTION_MAX_VEL` / `MAX_ACC` | `[0.5,1.5,1.5,1.5,2.0,2.0,2.0]` / `[1.0,2.0,2.0,2.0,2.5,2.5,2.5]` | review vs UR10e limits |
| 1602–1605 | `_current_cspace` | `q[:7]`, gantry clip [-3.09999, -1e-6] | DOF order still holds |
| 186 | `DOWNWARD_ORIENTATION` | `[1,0,1,0]` | re-derive for new tool frame |

### TCP / gripper
| Line | Symbol | Current value | Note |
|---|---|---|---|
| 175 | `GRIPPER_TCP_OFFSET` | `0.19357` | robotiq_base_link → finger contact; re-measure on new asset |
| 176–177 | `GRIPPER_BODY_WIDTH` | 0.160 (unused) | 2F-140, fine |
| 1835 | `BridgeState.target_frame` | `"ee_link"` | |
| 166–169 | `FINGER_OPEN/CLOSE/CLOSE_MIN/CLOSE_MAX` | 0.0 / 0.5 / 0.55 / 0.69 | re-tune on new drives |
| 170–171 | `ROBOTIQ_MAX_STROKE` / `ROBOTIQ_MAX_RAD` | 0.140 / 0.695 | verify closed angle on new asset |
| 172–173 | `GRIP_OVERSHOOT` / `ENTRY_CLEARANCE` | 0.060 / 0.020 | docstring L255 says 40 mm (mismatch) |
| 261, 279 | width→angle | linear `ROBOTIQ_MAX_RAD*(1-w/stroke)` | verify mapping |
| 311–316 | `set_finger_joints` | substring rules on `outer_finger_joint` / `inner_finger_joint` / `finger_joint` / `knuckle_joint` | new asset drives outer_finger (k=2000) — rules must be revisited |
| 3054, 4252 | `name == "finger_joint"` | | still valid |
| 3102–3103 | `_verify_grasp` | DOF index 7, >0.05 rad | index 7 still finger_joint |
| 4203 | `GRASP_CONFIRM_THRESHOLD` | 0.05 rad | |
| 4129–4130 | `_adaptive_close` | min_force 0.5, step 0.02, steps 30, overshoot 0.04 | re-tune |
| 3949–3950 | descent contact | 0.5 N, 0.002 m | |
| various | settle frames | 90/12/120/180 | re-tune |

### Gantry / home / DOF assumptions
- 161–163 `GANTRY_X_OFFSET 1.27`, `GANTRY_MIN/MAX_X -3.10/0.00` — still valid.
- 157/4826 `HOME_JOINTS` captured from stage `[:7]`; 1838 `num_dof=15`; 3011 `_set_joints` 7-prefix; 1783, 4028, 3139 — assumptions still hold for new DOF order.
- 794 `_jog_joint` default index 1 = shoulder_pan — holds.

### Workspace constants tied to old reach
- 549 self-filter `max_z=1.35`, margin 0.02; 572–574 prefilter bounds.
- 2992–2993 `_move_home` ee_link z thresholds 1.6 / 1.8.
- 704–710 `_plan_test` probe offsets −0.50/−1.00.

### Pre-existing bugs the swap will surface
- **B1** L3704 `if GANTRY_X_JOINT in name` — full path vs short DOF name, never matches → gantry moves fail.
- **B2** TCP offset applied twice: `_pick_descend` L3935, `_pick_z_positions` L3900–3903, place L4464–4466 add it, then `_tool_pose_from_tcp` L1620 adds again.
- **B3** Mixed frames: feedback from `ee_link`, targets in TCP/tool frame (L3265, 3347, 3367).
- **B4** L3992/3996 `zp` undefined in `_pick_descend` → NameError swallowed.
- **B5** XRDF sphere keys use old USD paths → self-filter silently filters nothing.
- Comments: L80 (AIKIDO.usd), L253–307 (2F-140 docs), L359, L1612–1617, Lula-era comments (L1769–1774, 3108, 3125, 4509–4518), L4010–4015.

---

## 3. `generative_kitting/config.yaml`

| Line | Key | Current | Status |
|---|---|---|---|
| 6 | `usd_scene_path` | `${PROJECT_ROOT}/AIKIDO.usd` | → `UR10e_Container_Assets/AIKIDO_ur10e_swapped.usd` (unused by code) |
| 8 | `ur10_base_path` | old ur10_flattened path | → `ur10e/base_link` (consider renaming key `arm_base_path`) |
| 9 | `ee_path` | old path | → `ur10e/ee_link` |
| 17–18 | `gantry_x_joint` / `gantry_x_offset` | full path / 1.27 | short name; offset OK |
| 27–32 | `robot_config_path`, urdf/xrdf filenames, `cumotion_tool_frame` | | unused — bridge hard-codes (candidate: make bridge read these) |
| 35–40 | `gripper_finger_paths` | 5 old Robotiq paths | → `ur10e_robotiq_2f140/...` |
| 68 | `scan_pose_orientation` | `[0.5,0.5,0.5,-0.5]` | re-derive; unused |
| 207–249 | comments | "RSD455", "UR10 ee_link ~55 mm" | stale |
| 331 | `home_joint_positions` | `[0,0,-1.5708,1.5708,-1.5708,-1.5708,0]` | re-check on UR10e |
| 333–350 | `workspace_bounds` / `effective_reach` | | UR10e reach also 1.30 m — likely OK |
| 353 | `gripper_tcp_offset` | 0.170 | re-measure (bridge uses 0.19357, controller default 0.150!) |
| 355 | `gripper_max_width` | 0.085 | **wrong** — 2F-85 value; 2F-140 = 0.140 |
| 357–358 | `finger_open_rad` / `finger_close_rad` | 0.0 / 0.5 | re-tune |
| 548 | `downward_orientation` | `[1,0,1,0]` | re-derive |

---

## 4. `generative_kitting/robot_config/` (cuMotion inputs — LIVE)

- **AIKIDO.urdf** — UR10 (CB3) kinematics: d1 0.1273, a2 0.612, a3 0.5723, d4 0.1639, d5 0.1157, d6 0.0922 (UR10e: 0.1807 / 0.6127 / 0.57155 / 0.17415 / 0.11985 / 0.11655). Elbow ±2π (UR10e ±π). Gantry velocity 0.02618 (suspicious). Gripper hung from `ee_link` (+0.0922); new asset hangs it from `wrist_3_link`. Root link `gantry` also carries every arm/gripper mesh (bug). 76 UR10 meshes in `meshes/`.
- **AIKIDO.xrdf** — cspace 7 joints (names still valid), accel 10, jerk 10000. 42 spheres keyed by **old USD-relative paths** (`gantry_home/ur10_flattened/...`), fitted to UR10 lengths; none on gantry/vagn/base_link/ee_link. Self-collision ignore list by link name.
- **AIKIDO.yaml** — Lula format, dead.
- **Required:** regenerate URDF + XRDF (+ meshes) for UR10e; resolve sphere-key convention (self-filter wants USD paths, cuMotion wants link names); add vagn/gantry spheres.

---

## 5. Execution / orchestration

### `execution/robot_controller.py` (secondary `--sim` path)
- 103–148 `GRIPPER_TCP_LINK_TEMPLATE` / `patch_urdf_and_yaml` — patches stock NVIDIA UR10 URDF, `gripper_tcp` on `ee_link` +Z.
- 239 `tcp_offset` default 0.150; 240–253 gripper/height defaults (several drift from config: box_height 0.05, box_entry_margin 0.15, transit_safe 0.40).
- 259–271 gantry/robot/base/ee defaults — stale paths without `/gantry`.
- 280, 368–372 `target_frame "ee_link"`.
- 341–342 reads `urdf_path`/`yaml_path` keys absent from config → Lula never initialises.
- 356–363 IK self-test pose; 379 gantry match bug (same as B1).
- 418–431 `_set_finger_joints` 2F-140 joint rules; 567–570 home pose duplicate; 797–801 verify_grasp.

### `execution/safety.py`
- 30–35 tabletop default bounds (overridden by config); 202 `gripper_force` 40 N; no joint-limit enforcement exists.

### `execution/motion_library.py` — test-only; additive world-Z TCP offset assumption.

### `orchestration/workflow_engine.py`
- LLM prompts naming UR10 / RSD455: L292, 332, 349, 356–357 (2F-140 prompts at 3792, 3897 are still correct).
- Stale/drifting defaults: 2202–2207 reach z_max 0.60 (config 999); 3053–3060 z_max 1.00; 3214–3239 `_safe_floor_z` tcp 0.170 (dead); 3657 `depth_analysis_height` 0.20 (config 1.10); 3749–3754 contact defaults; 4009–4021 place_transit 0.35 (config 0.20).
- Hard-coded: 3343 `approach_z = z + 0.20`; 3845–3858 finger close 0.5; 3865 0.10 rad gap; 2659–2679 realign thresholds.
- Geometry comments: 3390 (55 mm camera offset), 3528 ("zero XY offset"), 4240–4262 (gripper_tcp X=+219.5 mm) — contradict each other; re-measure.
- 3192–3196 `wrist_target_x/y_offset` — camera→EE calibration, redo for new mount.

### Other
- `orchestration/prompt_templates.py:24` "UR10 + Robotiq 2F-140".
- `orchestration/validators.py:39–44` & `perception/validators.py:46` duplicate workspace bounds.
- `diagnose_motion.py` 93–198 joint indices, "USD limits [-3.1, 0.0]", stiffness text.
- `isaac_sim_launcher.py` 8, 27–33, 179–189 — hard-coded old paths, home pose, DOF layout; reads no config.
- `evaluation/figures.py:543` "~55 mm offset from ee_link"; `.env.example:43` "bundled UR10 URDF".
- `robot_self_filtering_spec.md` header + §3/§5A; `PIPELINE_UPDATES_PENDING.md` §4–5 (re-measure movers/labels on new stage), §1–2 depth-annotator fix.
- `CLAUDE.md` (and likely `AGENTS.md` / `GEMINI.md`): project overview says UR10 + RSD455; "Isaac Sim Scene Reference (`AIKIDO.usd`)" table lists old base/EE/contact-sensor paths.
- `docker-compose.yml`: `.env` host paths won't resolve in container (unrelated to swap).

---

## 6. Duplicated sources of truth (root cause of drift)

| Value | Copies |
|---|---|
| TCP offset | bridge 0.19357 · config 0.170 · controller default 0.150 · workflow_engine default 0.170 · comment +219.5 mm X |
| Home pose | config · robot_controller · isaac_sim_launcher · bridge (captured) |
| Base/EE paths | config · bridge · robot_controller · launcher |
| Finger close 0.5 | bridge · config · controller · workflow_engine (×3) |
| Workspace bounds | config · validators ×2 · safety fallback · workflow_engine |
| cuMotion files / tool frame | config (unused) · bridge (used) |
