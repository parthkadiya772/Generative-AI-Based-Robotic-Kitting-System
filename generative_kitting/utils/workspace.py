"""Workspace bounds: the one copy is config.yaml ``execution.workspace_bounds``.

The validators and the safety boundary fall back to this when a caller
passes no bounds, instead of keeping their own (drifting) defaults.
"""

from __future__ import annotations

from functools import lru_cache

from .config_loader import load_config, project_root

BOUND_KEYS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


@lru_cache(maxsize=1)
def default_workspace_bounds() -> dict:
    """World-frame target bounds [m] from config.yaml; raises if a key is missing."""
    cfg = load_config(project_root() / "generative_kitting" / "config.yaml")
    bounds = (cfg.get("execution") or {}).get("workspace_bounds") or {}
    missing = [k for k in BOUND_KEYS if k not in bounds]
    if missing:
        raise ValueError(f"config.yaml execution.workspace_bounds: missing {missing}")
    return {k: float(bounds[k]) for k in BOUND_KEYS}
