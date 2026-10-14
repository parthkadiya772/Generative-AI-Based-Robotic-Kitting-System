"""Workspace bounds have one copy: config.yaml execution.workspace_bounds."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.safety import WorkspaceBoundary
from orchestration import validators as plan_validators
from perception import validators as scene_validators
from utils.config_loader import load_config, project_root
from utils.workspace import BOUND_KEYS, default_workspace_bounds


def test_default_bounds_come_from_config():
    cfg = load_config(project_root() / "generative_kitting" / "config.yaml")
    expected = {k: float(cfg["execution"]["workspace_bounds"][k]) for k in BOUND_KEYS}
    assert default_workspace_bounds() == expected


def test_no_module_keeps_its_own_copy():
    for mod in (plan_validators, scene_validators):
        assert not hasattr(mod, "DEFAULT_WORKSPACE_BOUNDS")


def test_safety_boundary_fills_missing_keys_from_config():
    b = WorkspaceBoundary({"z_max": 1.0})
    d = default_workspace_bounds()
    assert (b.x_min, b.x_max, b.z_max) == (d["x_min"], d["x_max"], 1.0)


def test_relative_log_dir_is_next_to_config(tmp_path, monkeypatch):
    # A relative logging.log_dir must not depend on the process cwd.
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("logging:\n  log_dir: logs\n")
    monkeypatch.chdir("/")
    assert load_config(cfg_file)["logging"]["log_dir"] == str(tmp_path / "logs")
