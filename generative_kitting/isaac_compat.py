"""Isaac Sim 6.0.1 compatibility adapter.

Isaac Sim 6.0 moved the classic single-robot control API (``World``,
``SingleArticulation``, ``ArticulationAction``) into ``extsDeprecated`` and
replaced it with the batched-tensor ``isaacsim.core.experimental`` API. This
module wraps the experimental ``Articulation`` behind the small classic-style
interface the execution layer already uses (``apply_action`` /
``get_joint_positions``), so call sites don't have to change.

Used by importable modules (e.g. ``execution/robot_controller.py``). The
``exec()``-launched scripts (``isaac_sim_bridge.py``, ``src/robot_control.py``,
``isaac_sim_launcher.py``) inline an equivalent adapter because they have no
reliable ``__file__`` / ``sys.path`` for a local import.

All ``isaacsim`` imports are lazy (inside methods) so this module imports
cleanly outside Isaac Sim (keeps the test suite green).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class ArticulationAction:
    """Drop-in for the deprecated
    ``isaacsim.core.utils.types.ArticulationAction``. Only
    ``joint_positions`` is used; ``joint_velocities`` kept for parity."""
    joint_positions: Optional[Sequence[float]] = None
    joint_velocities: Optional[Sequence[float]] = None


def _tensor_to_np(arr) -> np.ndarray:
    """warp / torch / list → 1-D numpy (row 0 of an (N, D) batch)."""
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[0]
    return arr


class RobotArticulation:
    """Classic-style wrapper over
    ``isaacsim.core.experimental.prims.Articulation``."""

    def __init__(self, prim_path: str):
        from isaacsim.core.experimental.prims import Articulation
        self._art = Articulation(paths=prim_path)
        self._dof_names_cache = None

    @property
    def num_dof(self) -> int:
        return int(self._art.num_dofs)

    @property
    def dof_names(self):
        if self._dof_names_cache is None:
            self._dof_names_cache = list(self._art.dof_names)
        return self._dof_names_cache

    def initialize(self):
        # Force the physics tensor view to resolve (lazy after play()).
        _ = self._art.num_dofs
        _ = self.dof_names
        return self

    def get_joint_positions(self) -> np.ndarray:
        return _tensor_to_np(self._art.get_dof_positions())

    def apply_action(self, action) -> None:
        if getattr(action, "joint_positions", None) is None:
            return
        targets = np.asarray(
            action.joint_positions, dtype=np.float32).reshape(1, -1)
        self._art.set_dof_position_targets(targets)

    def set_joint_positions(self, positions) -> None:
        """Instantaneously set DOF positions (teleport)."""
        arr = np.asarray(positions, dtype=np.float32).reshape(1, -1)
        self._art.set_dof_positions(arr)

    def set_finger_gains(self, substrings, kp, kd):
        """Set PD gains only on DOFs whose name contains any of ``substrings``."""
        idxs = [i for i, n in enumerate(self.dof_names)
                if any(s in n for s in substrings)]
        if not idxs:
            return []
        self._art.set_dof_gains(
            stiffnesses=np.full((1, len(idxs)), float(kp), dtype=np.float32),
            dampings=np.full((1, len(idxs)), float(kd), dtype=np.float32),
            dof_indices=idxs)
        return idxs

    @property
    def raw(self):
        return self._art


def get_prim_at_path(prim_path: str):
    """USD-native replacement for
    ``isaacsim.core.utils.prims.get_prim_at_path`` (version-proof)."""
    import omni.usd
    return omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)


def is_prim_path_valid(prim_path: str) -> bool:
    prim = get_prim_at_path(prim_path)
    return bool(prim) and prim.IsValid()
