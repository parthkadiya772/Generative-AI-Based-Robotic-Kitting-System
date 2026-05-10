# Project Structure — Generative AI-Based Robotic Kitting System

The project repository is split below into three sections so each part
fits cleanly in a single screenshot for the thesis report.

---

## Part 1 — Repository Root + USD Scenes + Bridge Entry Points

```
robot_in_air/                          ← repository root
├── AIKIDO.usd                         Master USD scene (UR10 + gantry +
│                                      bin + tray + cameras)
├── AIKIDO_USDA.usda                   Text-format USD export
├── robot_in_air.usd                   Robot-on-gantry sub-scene
├── robotiq_fixed_physics.usd          Robotiq 2F-140 gripper (PhysX)
├── ur10_flattened.usd                 UR10 arm flattened variant
├── ur10_gripper_camera.usd            UR10 + gripper + wrist cameras
│
├── meshes/  usd_meshes/               Per-part collision meshes
├── transforms/  transforms_simple/    USD transform layers
├── kitting_table/  ur10_setup/        USD scene fragments
│
├── configuration/                     Standalone robot-config files
│   ├── kitting_lula_config.yaml       Lula IK collision spheres
│   ├── kitting_robot.urdf             URDF used by bridge after patching
│   └── robot_control.py               Manual-control script (legacy)
│
├── CLAUDE.md                          Project instructions
├── THESIS_FRAMEWORK_SUMMARY.md        Full architecture write-up (v5)
│
└── generative_kitting/                ← MAIN SOURCE PACKAGE
    ├── main.py                        CLI — supports --sim / --ui
    ├── config.yaml                    Master configuration
    ├── requirements.txt               Python dependencies
    ├── run_ui.bat / run_ui.sh         Streamlit launchers
    ├── isaac_sim_bridge.py            HTTP REST bridge (port 8600)
    │                                  runs INSIDE Isaac Sim Script Editor
    ├── isaac_sim_launcher.py          Headless Isaac Sim boot script
    ├── kitting_bridge_server.py       Lighter-weight bridge variant
    ├── diagnose_qwen.py               Qwen-VL endpoint smoke test
    └── diagnose_vlm_perception.py     VLM-only diagnostic
```

---

## Part 2 — Neural & Deterministic Layers + Evaluation

```
generative_kitting/
├── perception/                        ── PERCEPTION (neural)
│   ├── vlm_perception.py              VLM client — Qwen3-VL / Qwen2.5-VL /
│   │                                  Gemma4 / GPT-4o
│   ├── object_detector.py             OWL-ViT2 / Grounding-DINO detector
│   ├── detector_server.py             Optional remote detector server
│   ├── depth_estimator.py             Depth-Anything-V2 fallback
│   ├── camera_interface.py            Bridge + Mock camera adapters
│   │                                  (rgb / depth / wrist / kit)
│   ├── validators.py                  Scene-response JSON validation
│   └── visualiser.py                  Bbox / confidence overlays
│
├── orchestration/                     ── ORCHESTRATION (neural)
│   ├── workflow_engine.py             Master pipeline driver (Phase 0→7)
│   │                                  Hybrid VLM + detector,
│   │                                  wrist re-localisation,
│   │                                  evaluation hooks,
│   │                                  Camera_Kit place verification
│   ├── llm_planner.py                 LLM client (Llama 3.1 via Ollama)
│   ├── prompt_templates.py            Scene + planning prompts
│   └── validators.py                  Action-primitive whitelist
│                                      Plan-length guard (≤ 50)
│
├── execution/                         ── EXECUTION (deterministic)
│   ├── robot_controller.py            Plans → IK + gripper actions
│   ├── motion_library.py              Waypoint generators
│   └── safety.py                      Workspace bounds, force limits,
│                                      e-stop, action timeouts
│
└── evaluation/                        ── QUANTITATIVE EVALUATION
    ├── metrics.py                     Precision / recall / F1 / IoU /
    │                                  grounding error in mm
    ├── recorder.py                    JSONL session log + SQLite store +
    │                                  CSV export
    └── place_verifier.py              Camera_Kit + VLM "did the part
                                       land in tray?" check
```

---

## Part 3 — Knowledge, Simulation, UI, Tests, Logs

```
generative_kitting/
├── knowledge/                         ── KNOWLEDGE / VOCABULARY
│   ├── parts_catalogue.yaml           Single source of truth —
│   │                                  visual descriptions +
│   │                                  multi-query detector phrases
│   ├── parts_catalogue.py             Loader + helpers
│   ├── parts_database.py              SQLite ORM-light
│   ├── seed_data.py                   Seed parts + kits into kitting.db
│   └── kitting.db                     SQLite store
│
├── simulation/                        ── ISAAC SIM SCENE HELPERS
│   ├── scene_setup.py                 Workspace + reference geometry
│   └── spawn_parts.py                 Random part scattering for tests
│
├── ui/                                ── OPERATOR DASHBOARD
│   └── streamlit_app.py               Streamlit UI — camera tabs
│                                      (RGB / Depth / Wrist /
│                                      Tray-Kit / VLM Detections),
│                                      chat command, perception overlays,
│                                      execution log,
│                                      Quantitative Evaluation panel
│                                      (live + cumulative + CSV export)
│
├── utils/                             ── SHARED UTILITIES
│   ├── logger.py                      Centralised Python logging
│   └── image_utils.py                 PIL helpers, debug-image saver
│
├── tests/                             ── TEST SUITE  (68 tests, pytest)
│   ├── test_planner.py                Plan validation, action whitelist
│   ├── test_controller.py             IK / motion library / safety
│   ├── test_perception.py             VLM client, validator, catalogue
│   └── test_integration.py            End-to-end (mock bridge)
│
└── logs/                              ── RUNTIME OUTPUT (gitignored)
    ├── depth_analysis/                Saved wrist-depth images per pick
    ├── vlm_images/                    Saved VLM overlay frames
    ├── wrist_verify/                  Pre-grasp wrist verification frames
    └── evaluation/                    JSONL session logs + eval.db +
                                       CSV exports for thesis tables
```

---

## Architectural Highlights

**Three-layer pipeline.** Perception → Orchestration → Execution. Neural
components (VLM scene analysis, LLM task planning) are isolated from the
deterministic execution layer (IK, gripper control, safety).

**Bridge is the simulator boundary.** `isaac_sim_bridge.py` is the only
file that runs inside Isaac Sim's Script Editor. The Streamlit dashboard,
workflow engine, and VLM / LLM clients all communicate with it over HTTP
on port 8600 — the same protocol that would talk to a physical-robot
controller.

**Catalogue-driven vocabulary.** `parts_catalogue.yaml` is the single
source of truth for known part types — visual descriptions feed the VLM
prompt, multi-query phrases feed the OWL-ViT2 detector, and size hints
feed the geometric safety calculations. Adding a new part type is a YAML
edit, no code changes.

**Quantitative evaluation built in.** Every scan, pick, place, and task
is recorded into JSONL + SQLite. The Streamlit *Quantitative Evaluation*
panel surfaces live + cumulative metrics (precision / recall / F1 / IoU /
grounding error / pick & place success rate / per-stage latency) and
exports CSVs that map one-to-one to the tables in the Results chapter.
