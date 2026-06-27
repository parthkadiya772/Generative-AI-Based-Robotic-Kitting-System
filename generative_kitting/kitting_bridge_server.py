"""One-line launcher for the Isaac Sim bridge.

Open this file in the Script Editor and click Run. It exec()s
isaac_sim_bridge.py from the location below.

WHY __file__ DOES NOT WORK HERE:
Isaac Sim's Script Editor copies the script body to a Temp file
(``%TEMP%\\xpa0.0\\script_*.py``) before running it. That means
``__file__`` inside the executed script points at that temp path,
not at your repo. So the bridge location must be supplied
explicitly — either:

  (a) Set the ``KITTING_PROJECT_ROOT`` environment variable to the
      absolute path of your repo root BEFORE launching Isaac Sim.
      Example (PowerShell):
          $env:KITTING_PROJECT_ROOT = "C:/path/to/robot_in_air"
          isaac-sim.bat

  (b) OR edit the ``PROJECT_ROOT`` line below to the absolute path
      of your local clone. Forward slashes are fine on all OSes
      (pathlib handles the separator).

The path must point at the directory that contains
``generative_kitting/`` (i.e. the repo root, NOT the package).
"""

import os
import pathlib

# ============================================================
# Edit this line if you don't set KITTING_PROJECT_ROOT in your
# environment. Use either forward slashes (`/`) or escape your
# backslashes (`\\`). The `r"..."` prefix lets you paste a raw
# Windows path with single backslashes too.
# ============================================================
PROJECT_ROOT = r"C:/KP/AI_and_Automation/Sem_4/Thesis/robot_in_air"
# ============================================================

_root = os.environ.get("KITTING_PROJECT_ROOT") or PROJECT_ROOT
if not _root:
    raise RuntimeError(
        "kitting_bridge_server: PROJECT_ROOT is empty. Either set "
        "KITTING_PROJECT_ROOT in your environment before launching "
        "Isaac Sim, OR edit PROJECT_ROOT in this file to the "
        "absolute path of your robot_in_air repo root.")

_bridge = pathlib.Path(_root) / "generative_kitting" / "isaac_sim_bridge.py"
if not _bridge.is_file():
    raise FileNotFoundError(
        f"isaac_sim_bridge.py not found at:\n  {_bridge}\n"
        f"Check that PROJECT_ROOT (or KITTING_PROJECT_ROOT) points "
        f"at the directory that contains generative_kitting/.")

# Load <PROJECT_ROOT>/.env so values like ISAACSIM_PATH, BRIDGE_HOST,
# and OLLAMA_BASE_URL are picked up by the bridge — Isaac Sim's GUI
# launch doesn't inherit shell exports, so a .env file is the only
# place the bridge can pull config from without code edits.
# Plain inline parser to avoid depending on python-dotenv being
# installed in Isaac Sim's bundled Python.
_env_file = pathlib.Path(_root) / ".env"
if _env_file.is_file():
    for _raw in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _raw.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _value = _line.partition("=")
        _key = _key.strip()
        _value = _value.strip().strip('"').strip("'")
        os.environ.setdefault(_key, _value)
    print(f"[OK] .env loaded from {_env_file}")

exec(_bridge.read_text(encoding="utf-8"))
