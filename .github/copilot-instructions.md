# AI Agent Instructions for Robot Kitting Project

This file helps AI agents understand the codebase structure and be immediately productive.

## Project Overview

Generative AI-based Robotic Kitting System — a neuro-symbolic framework with a three-layer pipeline:
- **Perception** (VLM): Zero-shot scene understanding → detect parts & positions
- **Orchestration** (LLM): Natural language commands → validated action plans  
- **Execution**: Deterministic robot control → UR10 + gripper in NVIDIA Isaac Sim

Operator says "pick all motor valves" → VLM sees parts → LLM plans sequence → robot executes.

## Code Structure Quick Reference

| Layer | Path | Purpose |
|-------|------|---------|
| **Perception** | `generative_kitting/perception/` | VLM calls, camera interface, schema validation |
| **Orchestration** | `generative_kitting/orchestration/` | LLM planning, prompt templates, 10-phase workflow engine |
| **Execution** | `generative_kitting/execution/` | Deterministic Isaac Sim API calls, motion library, safety |
| **Knowledge** | `generative_kitting/knowledge/` | SQLite parts database, seed data |
| **UI** | `generative_kitting/ui/` | Streamlit dashboard with live video & chat |
| **Tests** | `generative_kitting/tests/` | pytest suite (65 tests, must all pass) |
| **Legacy** | `src/robot_control.py` | Original monolithic control script (reference only) |

## Key Files by Task

### Configuration
- **[config.yaml](generative_kitting/config.yaml)** — ALL tunable parameters: prim paths, model endpoints, workspace bounds, gripper geometry, motion params

### Entry Points
- **[main.py](generative_kitting/main.py)** — CLI entry point (interactive, single-command, `--ui` Streamlit mode)
- **[isaac_sim_bridge.py](generative_kitting/isaac_sim_bridge.py)** — HTTP server (runs inside Isaac Sim Script Editor), exposes `/api/pick`, `/api/place`, `/api/camera`, `/api/scene_parts`
- **[streamlit_app.py](generative_kitting/ui/streamlit_app.py)** — Web dashboard

### Core Logic by Layer
- **[vlm_perception.py](generative_kitting/perception/vlm_perception.py)** — VLM API calls
- **[llm_planner.py](generative_kitting/orchestration/llm_planner.py)** — LLM API calls
- **[workflow_engine.py](generative_kitting/orchestration/workflow_engine.py)** — 10-phase pipeline orchestrator (VLM→USD match→LLM→execute)
- **[robot_controller.py](generative_kitting/execution/robot_controller.py)** — Plan→Isaac Sim API translation
- **[motion_library.py](generative_kitting/execution/motion_library.py)** — Waypoint generation for pick/place/transit

### Validation (Critical)
- **[orchestration/validators.py](generative_kitting/orchestration/validators.py)** — Plan validation, action primitives, workspace bounds
- **[perception/validators.py](generative_kitting/perception/validators.py)** — VLM response schema validation

## Design Principles

### LLM Isolation & Action Primitives
- **LLM outputs**: Only validated action primitives (`move_home`, `open_gripper`, `close_gripper`, `pick_object`, `place_object`, `verify_grasp`, etc.)
- **LLM cannot directly control robot** — execution layer is 100% deterministic
- **Plan validation** enforces: max 50 steps, starts with `move_home`, respects workspace bounds

### Coordinate Flow (VLM→USD→LLM)
1. VLM returns normalized image coordinates (0-1)
2. Isaac Sim bridge scans USD stage → real bounding boxes
3. VLM labels fuzzy-matched to USD prim names → coordinates replaced with real metres
4. LLM receives scene with ground-truth coordinates, not pixel estimates

### Configuration Centralization
All tunable parameters in **[config.yaml](generative_kitting/config.yaml)**:
- Prim paths: UR10, gripper, cameras, bin, tray, markers
- VLM/LLM endpoints (Ollama VPN IP, OpenAI, local Ollama)
- Workspace bounds: X [-2.5, 3.5], Y [-2.0, 2.0], Z [-0.1, 2.0]
- Offsets: Gantry X offset = 1.27, Gripper TCP = 0.150m from ee_link

## Development Essentials

### Running Application
```bash
# CLI interactive mode (no Isaac Sim required)
python generative_kitting/main.py

# Single command
python generative_kitting/main.py -c "Pick all motor valves"

# Streamlit web UI
python generative_kitting/main.py --ui

# Re-seed parts database
python generative_kitting/main.py --seed-db

# Isaac Sim bridge (inside Isaac Sim Script Editor):
exec(open("generative_kitting/isaac_sim_bridge.py").read())
```

### Running Tests
```bash
cd generative_kitting
python -m pytest tests/ -v                    # All 65 tests
python -m pytest tests/test_planner.py -v     # Single file
pytest tests/test_planner.py::TestPlanValidation::test_valid_plan -v  # Single test
```
⚠️ **All tests MUST pass before merging.**

### Virtual Environment
- Root: `.aikido/Scripts/activate` (Windows)
- generative_kitting has its own `.aikido` venv

## Critical Constraints & Gotchas

| Constraint | Details |
|-----------|---------|
| **VPN Required** | Ollama models at `10.7.0.35:11434` — VPN-only access |
| **Isaac Sim Bridge Process** | Runs inside **Script Editor**, NOT terminal — separate Python environment |
| **7-DOF Joints** | `[gantry_x, shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3]` — not standard 6-DOF |
| **Gripper TCP** | 0.150m from ee_link to finger pad (Robotiq 2F-140) — critical for IK accuracy |
| **URDF Patching** | Lula IK solver needs runtime TCP link injection in `isaac_sim_bridge.py` |
| **Black Frames** | Bridge has 3-retry black-frame recovery (brightness check) — camera startup safety |
| **Workspace Bounds** | X [-2.5, 3.5], Y [-2.0, 2.0], Z [-0.1, 2.0] — exceeds standard UR10 reach due to gantry |
| **Gantry Offset** | Always add 1.27 to X when converting USD coords to robot coords |

## Code Review & Debugging Workflow

### Use Code-Review-Graph Tools FIRST
This project has a knowledge graph. Always use these MCP tools before Grep/Read:
- **`detect_changes`** — Reviewing code diffs (gives risk scores)
- **`get_review_context`** — Get source snippets for review (token-efficient)
- **`get_impact_radius`** — Understand blast radius of changes
- **`get_affected_flows`** — Find which execution paths are impacted
- **`query_graph`** — Trace callers, callees, imports, dependencies, tests
- **`semantic_search_nodes`** — Find functions/classes by keyword
- **`get_architecture_overview`** — Understand high-level structure

### Common Debugging Checklist
- **VLM failures** → Check schema in `perception/validators.py`, confirm VLM prompt in `prompt_templates.py`
- **Plan failures** → Check `orchestration/validators.py` (action primitives, bounds, grasp sequences)
- **Robot motion errors** → Check workspace bounds in config.yaml, verify gantry offset (1.27)
- **Isaac Sim bridge errors** → Check Isaac Sim Script Editor console (port 8600 HTTP errors)
- **Test failures** → Run `pytest tests/ -v` locally, check test isolation (mock images, mock API responses)

Logs written to `logs/kitting_session.log` (path in config.yaml).

## Coding Standards

### Style & Naming
- **Functions/variables**: `snake_case`
- **Classes**: `CamelCase`
- **Logging**: Use `loguru` — `log.info()`, `log.error()`, `log.debug()`
- **Error handling**: Raise `PlanValidationError` with `errors` list + message in orchestration layer

### Validation Pattern
All external inputs (VLM response, LLM plan, user command) must be validated:
- VLM response → `perception/validators.py`
- LLM plan → `orchestration/validators.py`
- Always check before downstream use (fail-fast principle)

### Module Organization
Each layer has:
- `*.py` — core logic
- `validators.py` — input/output validation
- `prompt_templates.py` (orchestration only) — system prompts with scene context

## Part Types Reference
```
motor_valve, black_hose, black_plate, black_plug, small_hinge, small_tube, 
silver_box, silver_gun, tube_with_clamps
```

## USD Scene Structure
Main scene file: **[AIKIDO.usd](AIKIDO.usd)**

Key prim paths:
- Robot: `/World/gantry_home/ur10_flattened/ur10_instanceable/base_link`
- End effector: `/World/gantry_home/ur10_flattened/ur10_instanceable/ee_link`
- RGB Camera: `/World/Camera`
- Depth Camera: `/World/Realsense/RSD455/Camera_Pseudo_Depth`
- Parts bin: `/World/robot_facade_full`
- Place tray: `/World/box_840`

## Additional Resources

- **[CLAUDE.md](CLAUDE.md)** — Detailed architecture, 10-phase workflow, neuro-symbolic coordination
- **[AGENTS.md](AGENTS.md)** — Code-review-graph MCP tools documentation
- **[implementation_plan.md](generative_kitting/implementation_plan.md)** — Project timeline & milestones
- **[walkthrough.md](generative_kitting/walkthrough.md)** — Step-by-step execution flow

---

**Last Updated**: 2026-04-27  
**For Questions**: Refer to CLAUDE.md for deep dives, use code-review-graph tools for code exploration.
