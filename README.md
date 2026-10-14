# Generative AI-based Robotic Kitting System

Natural-language kitting ("pick all motor valves and place them into the tray")
on a simulated **UR10e + Robotiq 2F-140** hanging from a gantry rail in
**NVIDIA Isaac Sim 6.1**. A vision-language model (Qwen3-VL) finds the parts
in camera images and plans the task; the robot is driven only by validated
action primitives through a deterministic execution layer, with
**cuMotion** planning against a collision world that **nvblox** builds from
six depth cameras.

The VLM never commands joints. It emits labelled boxes and a plan made of a
fixed action vocabulary (`pick_object`, `place_object`, `move_home`, …); part
positions come from those boxes plus depth, never from the simulator's scene
graph.

---

## Contents

1. [System overview](#1-system-overview)
2. [System requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Running](#5-running)
6. [Running on another machine (portability)](#6-running-on-another-machine-portability)
7. [Method](#7-method)
8. [Evaluation and measured results](#8-evaluation-and-measured-results)
9. [Testing and diagnostics](#9-testing-and-diagnostics)
10. [Project layout](#10-project-layout)
11. [Known limitations](#11-known-limitations)
12. [Troubleshooting](#12-troubleshooting)
13. [Security notes](#13-security-notes)
14. [References](#14-references)
15. [License and citation](#15-license-and-citation)

---

## 1. System overview

```
 Operator command (Streamlit UI or CLI)
        │
        ▼
 ┌──────────────────── Perception ────────────────────┐
 │ overhead RGB-D ─► Qwen3-VL (labels + 2-D boxes)     │
 │      box ─► 9×9 depth grid ─► top-surface estimator │──► part XYZ (world)
 └─────────────────────────────────────────────────────┘
        │
        ▼
 ┌──────────────────── Orchestration ─────────────────┐
 │ Qwen3-VL planner ─► schema / whitelist validation   │──► action plan
 └─────────────────────────────────────────────────────┘
        │  HTTP (localhost:8600)
        ▼
 ┌──────────── Execution: Isaac Sim bridge ───────────┐      ┌─ nvblox service ─┐
 │ 6 depth cams ─► robot mask ─────────────────────────┼─────►│ TSDF ─► EDT SDF  │
 │ cuMotion SDF obstacle ◄─────────────────────────────┼──────│ (localhost:8610) │
 │ plan standoff ─► wrist verify ─► Cartesian descent  │      └──────────────────┘
 │ ─► adaptive close ─► lift ─► place                  │
 └─────────────────────────────────────────────────────┘
```

Three processes run side by side:

| Process | Runs in | Port | Role |
|---|---|---|---|
| Isaac Sim + bridge (`isaac_sim_bridge.py`) | Isaac Sim's Python | 8600 | cameras, robot, cuMotion, HTTP API |
| nvblox service (`nvblox_service/server.py`) | its own venv (torch 2.9.1) | 8610 | depth integration, signed distance field |
| Operator app (`main.py --ui`) | project venv `.aikido` | 8501 | VLM perception, planning, workflow |

The VLM runs on a separate GPU server (vLLM, OpenAI-compatible API).

---

## 2. System requirements

Measured on the development machine (Ubuntu 24.04, RTX 4070 Laptop 8 GB,
driver 580.178).

| Component | Requirement | Notes |
|---|---|---|
| OS | **Linux, Ubuntu 24.04** | the nvblox wheel is built for Ubuntu 24 + CUDA 12; the collision-world service does not run on Windows |
| GPU (robot machine) | NVIDIA RTX, **≥ 8 GB** | measured: Isaac Sim + bridge + cuMotion ≈ 5.4 GB, nvblox ≈ 0.7 GB (2 cm voxels) |
| NVIDIA driver | **≥ 580** | `requirements.txt` installs `torch 2.11+cu130` (CUDA 13). With an older driver install torch from a cu12x index instead |
| Isaac Sim | **6.1** (workstation install) | first start needs internet: some assets come from NVIDIA's asset server |
| Python | 3.12 | both venvs |
| `uv` | any recent | creates the nvblox venv (`pip install uv` or the uv installer) |
| VLM server | vLLM serving Qwen3-VL | a separate GPU host; the 8B model at 8 bit is ≈ 9 GB of weights plus KV cache; reached over the network / VPN |
| Disk | ~1 GB repo + ~6 GB venvs | the repo pack is ~150 MB; USD assets up to 89 MB per file |

Without Isaac Sim you can still run the planner and perception on a mock image
(section 5.4).

---

## 3. Installation

### 3.1 Clone

```bash
git clone https://github.com/parthkadiya772/Generative-AI-Based-Robotic-Kitting-System.git robot_in_air
cd robot_in_air
```

The clone can live anywhere: no path inside the repo is machine-specific
(section 6).

### 3.2 Environment file

```bash
cp .env.example .env
```

Edit only the values that describe *your* machine and network:

```env
VLLM_BASE_URL=http://<vlm-server>:8000/v1   # must include the port and /v1
OLLAMA_BASE_URL=http://localhost:11434      # only if you use Ollama models
BRIDGE_HOST=127.0.0.1                       # keep loopback unless you know why not
BRIDGE_PORT=8600
ROBOT_CONFIG_PATH=${PROJECT_ROOT}/generative_kitting/robot_config/ur10e   # leave as is
```

`PROJECT_ROOT` is filled in automatically; never write an absolute path for it.

### 3.3 Isaac Sim 6.1

Install the Isaac Sim 6.1 workstation build (see NVIDIA's installation guide)
and check it starts (`./isaac-sim.sh`). Isaac Sim is **not** a Python
dependency of this repo: the bridge is executed *inside* it.

### 3.4 nvblox service venv (collision world)

```bash
generative_kitting/nvblox_service/setup_venv.sh
```

Creates `generative_kitting/nvblox_service/.venv` with `torch 2.9.1+cu128` and
`nvblox_torch 0.0.10` and runs a GPU smoke test. nvblox needs its own venv
because `nvblox_torch` links libtorch ≤ 2.9.1 while Isaac Sim 6.1 ships torch
2.11, so it cannot load inside Isaac Sim's process.

### 3.5 Operator app venv

```bash
generative_kitting/run_ui.sh          # creates generative_kitting/.aikido on first run
```

or manually:

```bash
uv venv --python 3.12 generative_kitting/.aikido
uv pip install --python generative_kitting/.aikido/bin/python -r generative_kitting/requirements.txt
```

### 3.6 VLM server (vLLM)

On the GPU server:

```bash
vllm serve cyankiwi/Qwen3-VL-8B-Instruct-AWQ-8bit \
     --served-model-name Qwen/Qwen3-VL --port 8000
```

The served model name must match `perception.vlm_model` and
`orchestration.llm_model` in `generative_kitting/config.yaml`
(`Qwen/Qwen3-VL`). Check it from the robot machine:
`curl $VLLM_BASE_URL/models`.

---

## 4. Configuration

| File | Holds | Rule |
|---|---|---|
| `.env` (git-ignored) | network endpoints, ports, keys | the only per-machine file |
| `generative_kitting/robot_config/ur10e/robot.yaml` | **all robot facts**: prims, joints, gantry, home pose, TCP, gripper, cuMotion URDF/XRDF, nvblox settings | single source of truth; no robot constants in code |
| `generative_kitting/config.yaml` | perception, planning, workflow, workspace bounds, logging | `${VAR}` placeholders resolve from `.env` |
| `generative_kitting/knowledge/parts_catalogue.yaml` | part vocabulary given to the VLM | add a part here, no code change |

`ROBOT_CONFIG_PATH` selects the robot; `robot_config/ur10/` is the frozen UR10
baseline kept for reproducibility.

Key `robot.yaml → nvblox` settings (values measured, see section 8):

| Key | Value | Meaning |
|---|---|---|
| `voxel_size` | 0.02 m | 1 cm does not fit beside Isaac Sim in 8 GB |
| `cameras` | 4 corner + `/World/Camera` + `/World/Camera_Kit` | the rack hides the bin from the corner cameras |
| `workspace_min/max` | [−2.6, −1.6, −0.05] … [2.6, 1.6, 2.1] m | the integrated and exported box |
| `robot_mask_margin` | 0.04 m (vagn 0.05) | sphere inflation for masking the robot out of depth |
| `sdf_safety_margin` | 0.02 m | obstacle inflation seen by cuMotion |
| `target_carve_radius` / `_below` | 0.10 / 0.10 m | column cleared around a grasp target |
| `held_part_radius` | 0.11 m | depth mask around a grasped part |
| `tsdf_decay_factor` | 0.5 | a removed part leaves the map in 2 updates |

---

## 5. Running

### 5.1 Full pipeline (Isaac Sim + robot)

1. **Isaac Sim:** open `UR10e_Container_Assets/AIKIDO_ur10e_swapped.usd`
   (= `robot.yaml robot.stage`).
2. **nvblox service** (terminal 1, repo root):
   ```bash
   generative_kitting/nvblox_service/run.sh generative_kitting/nvblox_service/server.py
   ```
3. **Bridge** (Isaac Sim → `Window → Script Editor`):
   ```python
   exec(open("/path/to/robot_in_air/generative_kitting/kitting_bridge_server.py").read())
   ```
   Wait for `[OK] Bridge v2 on http://127.0.0.1:8600`.
4. **Operator UI** (terminal 2): `generative_kitting/run_ui.sh`, then open
   <http://localhost:8501> and type a command.

### 5.2 CLI (plans only)

```bash
generative_kitting/.aikido/bin/python generative_kitting/main.py -c "Pick all motor valves"
generative_kitting/.aikido/bin/python generative_kitting/main.py      # interactive
```

The CLI perceives and plans; robot execution runs through the UI and the
bridge.

### 5.3 Isaac Sim diagnostics

```python
exec(open("/path/to/robot_in_air/generative_kitting/isaac_sim_launcher.py").read())
```
checks the robot prims from `robot.yaml`, the configured VLM endpoint and a
move to the home pose. From a terminal, with the bridge running:
```bash
generative_kitting/.aikido/bin/python generative_kitting/diagnose_motion.py
```

### 5.4 Without Isaac Sim

```bash
generative_kitting/.aikido/bin/python generative_kitting/main.py --mock-image path/to/workspace.png -c "Pick all gears"
```

### 5.5 Bridge HTTP API (selection)

| Endpoint | Purpose |
|---|---|
| `GET /api/status` | joints, TCP position, planner and collision-world state |
| `POST /api/build_collision_world` | depth → nvblox → SDF → cuMotion; `exclude_points` = grasp targets to carve |
| `POST /api/approach`, `/api/pick_descend`, `/api/pick_close`, `/api/pick_retract`, `/api/place` | staged pick and place |
| `POST /api/project_to_world` | normalised image points → world XYZ through the camera's depth |
| `GET /api/scene_annotations?camera=rgb` | ground-truth boxes from USD (**evaluation only**) |
| `POST /api/plan_test` | plan a tool-frame pose with vs without the collision world |

---

## 6. Running on another machine (portability)

**Is execution tied to one machine's paths? No.** Nothing in the code,
`config.yaml` or `robot.yaml` contains an absolute path. Specifically:

| What | How it is located |
|---|---|
| Repo root (bridge inside Isaac Sim) | `kitting_bridge_server.py` searches for the marker file `generative_kitting/isaac_sim_bridge.py`, in order: `KITTING_PROJECT_ROOT` (optional), its own `__file__`, the folder of the **open USD stage** and its parents, the current directory, then `$HOME` (6 levels) |
| Repo root (Python apps) | from the source file location (`utils/config_loader.project_root`) |
| Paths in `.env` / YAML | `${PROJECT_ROOT}/…`, expanded at load time (set before `.env` is parsed, so python-dotenv expands it too) |
| Logs, debug images, parts DB | anchored at `generative_kitting/`, independent of the current directory |
| USD scene | all 41 layers reference each other with relative paths |
| cuMotion URDF / XRDF | relative mesh paths (`./meshes/…`) |
| nvblox venv | `run.sh` finds `.venv/lib/python3*/…` next to itself |

To set the project up elsewhere you change **only** `.env` (VLM server URL,
optional Ollama URL, bridge host/port). The one path you type yourself is the
`exec(open(".../kitting_bridge_server.py"))` line in Isaac Sim's Script
Editor, because Isaac Sim cannot know where you cloned the repo; it can be any
location. To force a specific checkout (e.g. two clones), put
`KITTING_PROJECT_ROOT = "/path/to/clone"` before that `exec` line.

What does depend on the machine is the **software stack**, not paths: Ubuntu
24 + CUDA 12 for nvblox, a driver ≥ 580 for the CUDA 13 torch build, and
Isaac Sim 6.1. A Docker image to pin these is planned (work in progress, not
yet part of the repository).

---

## 7. Method

### 7.1 Perception: from a VLM box to a grasp point

**VLM grounding.** Qwen3-VL receives the full overhead frame (1920×1080) with
the part catalogue and returns, per part, a label and a box (`bbox_2d`). The
boxes are decoded as a 0–1000 relative grid; the measured boxes are consistent
with that (all six landed on their parts, with a 1.5 % scale error in x).
`_extract_vlm_bboxes` chooses the convention by value range, which should be
pinned per model.

**Depth projection.** Image points are back-projected through the camera's
pinhole model using the rendered `distance_to_image_plane` depth:
**x_cam = d · K⁻¹ [u, v, 1]ᵀ**, **x_world = T_world←cam · x_cam**.
The renderer keeps the horizontal aperture and draws square pixels at any
resolution, so **f_y = f_x = W·f / h_aperture** (using the USD vertical
aperture instead put the corner cameras' floors 10–40 cm apart).

**Top-surface estimator.** A single pixel at the box centre is a poor
estimate: on these parts it looks through the bore to the bin floor (40 mm
too low on gears, 85 mm on valves). Instead a 9×9 grid inside the box (10 %
inset) is projected and reduced to the part's top surface:

- z_top = 90th percentile of the sample heights,
- (x, y) = mean of the samples with z ≥ z_top − 1 cm.

Samples that miss the part land on the floor and drop out of the band, so the
estimate is robust to box misplacement (section 8.2). The same estimator runs
on a 7×7 patch (±4 % of the image) around the wrist VLM's pointed centre.

**Eye-in-hand verification.** At the 0.25 m standoff the wrist camera (0.39 m
above the part) re-measures the part; its z is always applied and its xy is
applied within a 5 cm cap (rejects a neighbouring part). The wrist view is the
most precise source (section 8.2), consistent with the eye-in-hand advantage
known from visual servoing [Chaumette & Hutchinson 2006].

### 7.2 Collision world: depth cameras → nvblox → signed distance field

1. **Capture.** Six depth cameras (4 corner, 2 overhead) render depth on the
   same render products as their point-cloud annotators.
2. **Robot self-masking.** Every pixel is back-projected; pixels inside the
   XRDF collision spheres at the *live* joint state (forward kinematics from
   the USD link poses, +4 cm, carriage +5 cm), outside the workspace box, or
   inside a sphere around a held part are masked (`mask = 0`). Validated
   against ground truth by rendering with and without the robot: 0 moving-robot
   pixels leak at the spawn and home poses.
3. **Integration.** nvblox [Millane et al. 2024] fuses the masked depth into a
   2 cm TSDF, allocating blocks only inside the workspace box. Updates are
   incremental with weight decay 0.5 per update (a removed part disappears in
   2 updates; space nothing sees fades in ~10).
4. **Exact SDF.** nvblox's own ESDF jumped by more than one voxel between 2–3 %
   of neighbouring voxels (up to 0.27 m), which cuMotion's `inspect_sdf`
   rejects. The service therefore builds the field itself: occupancy =
   observed TSDF voxels with distance ≤ 0; signed field = Euclidean distance
   transform outside minus inside (exact EDT [Maurer et al. 2003], as
   implemented by `scipy.ndimage.distance_transform_edt`), which is
   1-Lipschitz by construction. Error against the integrated points: median −0.4 to −0.7 cm
   (conservative), p90 +0.6 cm.
5. **Planner world.** The field minus a 2 cm safety margin is loaded as one
   cuMotion SDF obstacle (260×160×108 voxels). Grasp targets are carved as
   vertical columns (r = 0.10 m) before the EDT so the field stays exact.

### 7.3 Motion

cuMotion plans collision-free transits (gantry + 6 arm joints as one 7-DOF
c-space) to a standoff 0.25 m above the part. Descent and lift are Cartesian
micro-steps with a contact stop; the adaptive close steps the finger until the
pad sensor or a finger stall reports contact, then squeezes a further 0.06 rad.
A grasp is confirmed only if the fingers stop short of their target **and**
the pad gap is ≤ 0.13 m (wider means the fingers stopped on the floor or a
wall). If the start state lies only inside the safety margin (e.g. after a
failed grasp next to a wall), the planner retries once with the margin
removed: the current configuration is physically realised, so a "collision"
there is inflation, not contact.

### 7.4 Calibration

In simulation the camera model is exact: the true centre pixel back-projects
to within 3.6–5 mm of the true part position, and the six cameras agree to
5–8 mm. The pick errors observed during development were therefore **not**
calibration errors but estimator and decoding errors (section 8.1). For a
physical cell the planned procedure is: intrinsics with a ChArUco board
[Zhang 2000; Garrido-Jurado et al. 2014], wrist hand-eye calibration AX = XB
[Tsai & Lenz 1989; Park & Martin 1994], fixed cameras by robot-world/hand-eye
AX = ZB [Shah 2013], TCP by pivoting — first validated in simulation against
the known USD transforms.

---

## 8. Evaluation and measured results

### 8.1 Error budget method

`generative_kitting/tools/measure_grounding.py` measures each stage against the
simulator's ground truth (`/api/scene_annotations`: world centre, top height
and image box per part, per camera), running the production code path:

| Stage | Input | Isolates |
|---|---|---|
| camera | ground-truth image box | camera model + depth + estimator |
| centre | ground-truth centre pixel | the old single-pixel method |
| detection | VLM boxes | 2-D error, IoU, per-axis affine fit, end-to-end 3-D error |
| wrist | robot at the standoff | wrist box (truth) and wrist VLM centre |

Reports (mean, std, p95, max; per part type) go to
`generative_kitting/logs/grounding/<timestamp>/report.md`.

```bash
generative_kitting/.aikido/bin/python generative_kitting/tools/measure_grounding.py --vlm live --wrist --repeats 3
```

### 8.2 Results (simulation, 3 gears + 3 valves)

| XY error | Gear | Valve |
|---|---|---|
| Old method (box-grid corner sample) | 57–100 mm | 76–111 mm |
| Overhead, ground-truth box | 2.2 mm | 21.6 mm |
| Overhead, VLM-like box error (~20 px) | 5.8 mm | 20.8 mm |
| Wrist, ground-truth box | **0.2 mm** | 22.0 mm |
| Wrist, VLM centre with 6 px noise | **2.8 mm** | 11.1 mm |

Height: the top-surface estimate is within 2–4 mm on gears (the single centre
pixel was 40 mm low). Gripper tolerance: the open 2F-140 (144 mm) leaves about
±24 mm around a 95 mm gear. Valves are asymmetric: the centroid of their top
surface is not their geometric centre, so they need a defined grasp point
(mask or grasp-point model) rather than a better detector.

Other checks: approaches at yaw 0 / ±30°: 15/15 within 1.4 mm;
`large_gear_1` and `motor_valve_1` picks hold after the lift (50 / 42–50 N).

---

## 9. Testing and diagnostics

```bash
cd generative_kitting
.aikido/bin/python -m pytest tests/ -q        # 135 tests, no Isaac Sim needed
```

| Tool | Use |
|---|---|
| `tools/measure_grounding.py` | error budget against ground truth (`--vlm live|fake|none`, `--wrist`) |
| `tools/pick_test.sh X Y TOP_Z [GRASP_LIFT] [YAW_DEG]` | one staged pick through the bridge |
| `diagnose_motion.py` | bridge status, joint jog, planner frame check, plan test |
| `isaac_sim_launcher.py` | in-Isaac scene / VLM / home check |

---

## 10. Project layout

```
robot_in_air/
├── .env.example                  # template; copy to .env
├── UR10e_Container_Assets/       # the Isaac Sim scene (AIKIDO_ur10e_swapped.usd)
├── AIKIDO.usd                    # UR10 baseline scene
├── UR10E_MIGRATION_PLAN.md       # step-by-step migration log with measurements
├── PIPELINE_UPDATES_PENDING.md   # verified Isaac Sim findings
├── archive/                      # legacy UR10 script + Lula config (not used)
└── generative_kitting/
    ├── config.yaml               # perception / planning / workflow settings
    ├── main.py                   # CLI and UI entry point
    ├── kitting_bridge_server.py  # Script Editor launcher (finds the repo)
    ├── isaac_sim_bridge.py       # bridge inside Isaac Sim (cameras, cuMotion, nvblox client)
    ├── nvblox_service/           # collision-world service (own venv)
    ├── robot_config/ur10e/       # robot.yaml, URDF, XRDF, meshes
    ├── perception/               # VLM, camera clients, validators
    ├── orchestration/            # planner, prompts, workflow engine
    ├── execution/                # safety, motion helpers
    ├── knowledge/                # parts catalogue and database
    ├── ui/streamlit_app.py       # operator dashboard
    ├── utils/                    # config / robot-config loaders, workspace bounds
    ├── tools/                    # measure_grounding.py, pick_test.sh
    └── tests/
```

---

## 11. Known limitations

- **Bin and tray positions** still come from USD prim lookups
  (`/World/robot_facade_full`, `/World/box_840`); part positions are camera +
  VLM. Moving these to perception is the next step.
- **Valves:** ~20 mm grasp-point offset (asymmetric part); a grasp-point /
  segmentation stage is planned.
- **Held part:** masked out of the map, but not attached to cuMotion's robot
  model, so transit plans do not account for its volume.
- **Narrow grasps:** a planned move straight to a grasp next to a bin wall can
  be refused by the 2 cm + ~1.7 cm margins; the descent itself is Cartesian
  and unaffected.
- **Minimum grasp height:** the pad centre must be ≥ ~5 cm above the surface a
  part stands on (finger bottom 47 mm below the pad centre; closing lowers the
  pads 16–24 mm).
- Isaac Sim: do not delete physics prims while the timeline plays; it
  invalidates the physics views (recover: stop, `World.clear_instance()`,
  play + stop, restart the bridge).

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Could not find the robot_in_air repo` when starting the bridge | the open stage is not from this clone; open `UR10e_Container_Assets/AIKIDO_ur10e_swapped.usd` or set `KITTING_PROJECT_ROOT` before `exec` |
| `nvblox service not reachable at http://127.0.0.1:8610` | start `nvblox_service/run.sh … server.py` |
| `nvblox venv missing` | run `nvblox_service/setup_venv.sh` |
| `ROBOT_CONFIG_PATH is not set` / robot config not found | `.env` missing: `cp .env.example .env` |
| VLM timeouts | `VLLM_BASE_URL` unreachable (VPN?) or missing `/v1`; `curl $VLLM_BASE_URL/models` |
| Model not found on the server | `--served-model-name` must equal `perception.vlm_model` |
| `NO PATH` for every plan | start inside the map: check `/api/plan_test`; rebuild the world with `/api/build_collision_world` |
| `torch` CUDA errors in the app venv | driver older than 580: reinstall torch from a cu12x index |
| CUDA out of memory in nvblox | keep `voxel_size: 0.02` and the workspace box; close other GPU apps |

---

## 13. Security notes

- The bridge (8600) and the nvblox service (8610) bind to `127.0.0.1` and have
  **no authentication**: they move the robot. Do not expose them to a network
  you do not control.
- Server addresses and keys live in `.env`, which is git-ignored. Do not
  commit `.env`, `generative_kitting/logs/` captures or `eval.db` if they
  contain sensitive imagery.
- Set your git identity before committing (`git config user.email
  <id>+<user>@users.noreply.github.com`) so commits do not carry your local
  username and hostname.

---

## 14. References

- Chaumette, F., Hutchinson, S. (2006). Visual servo control, Part I: Basic
  approaches. *IEEE Robotics & Automation Magazine*, 13(4), 82–90.
- Maurer, C. R., Qi, R., Raghavan, V. (2003). A linear time algorithm for
  computing exact Euclidean distance transforms of binary images in arbitrary
  dimensions. *IEEE Trans. Pattern Analysis and Machine Intelligence*, 25(2),
  265–270.
- Garrido-Jurado, S., Muñoz-Salinas, R., Madrid-Cuevas, F. J., Marín-Jiménez,
  M. J. (2014). Automatic generation and detection of highly reliable fiducial
  markers under occlusion. *Pattern Recognition*, 47(6), 2280–2292.
- Millane, A., Oleynikova, H., Wirbel, E., Steiner, R., Ramasamy, V., Tingdahl,
  D., Siegwart, R. (2024). nvblox: GPU-accelerated incremental signed distance
  field mapping. *IEEE ICRA*.
- Park, F. C., Martin, B. J. (1994). Robot sensor calibration: solving AX = XB
  on the Euclidean group. *IEEE Trans. Robotics and Automation*, 10(5), 717–721.
- Shah, M. (2013). Solving the robot-world/hand-eye calibration problem using
  the Kronecker product. *Journal of Mechanisms and Robotics*, 5(3).
- Tsai, R. Y., Lenz, R. K. (1989). A new technique for fully autonomous and
  efficient 3D robotics hand/eye calibration. *IEEE Trans. Robotics and
  Automation*, 5(3), 345–358.
- Zhang, Z. (2000). A flexible new technique for camera calibration. *IEEE
  Trans. Pattern Analysis and Machine Intelligence*, 22(11), 1330–1334.
- Qwen Team (2025). Qwen2.5-VL technical report. arXiv:2502.13923.

---

## 15. License and citation

This codebase accompanies the Master's thesis "Generative AI-based Robotic
Kitting System" (Kadiya, 2026). See `THESIS_FRAMEWORK_SUMMARY.md` and
`THESIS_EVALUATION_RESULTS.md` for the academic context.
