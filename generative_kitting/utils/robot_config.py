"""Robot hardware config loader (``robot_config/<robot>/robot.yaml``).

Single source of truth for robot-specific settings: prim paths, joint
names, TCP, gripper, wrist camera mount and cuMotion files. Used by the
Isaac Sim bridge, the controller and the orchestration layer.

The robot folder is chosen by ``ROBOT_CONFIG_PATH`` (set in ``.env``) or
passed explicitly. Like ``config_loader``, this needs only PyYAML, so it
runs unchanged inside Isaac Sim's Script Editor.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .config_loader import (
    _ensure_project_root_env,
    _expand,
    _expand_string,
    _load_dotenv_if_present,
)

ROBOT_YAML = "robot.yaml"
GRIPPER_MODES = ("drive", "pad", "mirror", "zero", "free")

# Every key must be present; values may be null until measured.
_REQUIRED_KEYS = (
    "robot.name", "robot.stage",
    "articulation.root", "articulation.num_dof",
    "prims.arm_base", "prims.ee", "prims.tool", "prims.gripper_root",
    "joints.gantry", "joints.arm", "joints.gripper_drive", "joints.gripper_modes",
    "gantry.x_offset", "gantry.limits",
    "home_joints",
    "tcp.offset", "tcp.frame",
    "orientation.tool_down", "orientation.scan",
    "gripper.stroke", "gripper.body_width", "gripper.force_limit",
    "gripper.finger.open", "gripper.finger.close",
    "gripper.finger.close_min", "gripper.finger.close_max", "gripper.finger.max_rad",
    "gripper.grip_overshoot", "gripper.entry_clearance", "gripper.grasp_confirm_rad",
    "gripper.adaptive_close.min_force", "gripper.adaptive_close.step_rad",
    "gripper.adaptive_close.max_steps", "gripper.adaptive_close.overshoot_rad",
    "gripper.contact_sensor", "gripper.finger_links",
    "wrist_camera.root", "wrist_camera.depth", "wrist_camera.rgb",
    "wrist_camera.imu", "wrist_camera.mount_link",
    "wrist_camera.self_filter_radius", "wrist_camera.ee_offset_xy",
    "cumotion.urdf", "cumotion.xrdf", "cumotion.self_filter_xrdf",
    "cumotion.root_link", "cumotion.tool_frame",
    "cumotion.max_vel", "cumotion.max_acc", "cumotion.link_prims",
)

# null here is a valid setting, not an unmeasured value.
_NULL_ALLOWED = frozenset({"home_joints"})


class RobotConfigError(ValueError):
    """robot.yaml is missing, malformed or incomplete."""


class RobotConfig:
    """Typed view over a loaded robot.yaml."""

    def __init__(self, data: dict, config_dir: Path):
        self.raw = data
        self.config_dir = config_dir
        self._validate()

    # ── generic access ──────────────────────────────────────────

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def _has(self, dotted: str) -> bool:
        sentinel = object()
        return self.get(dotted, sentinel) is not sentinel

    def missing(self) -> list[str]:
        """Keys still null (not measured yet)."""
        return [k for k in _REQUIRED_KEYS
                if k not in _NULL_ALLOWED and self.get(k) is None]

    def require_complete(self) -> None:
        """Raise if any value is still null. Call before driving the robot."""
        missing = self.missing()
        if missing:
            raise RobotConfigError(
                f"{self.config_dir / ROBOT_YAML}: values not set: {', '.join(missing)}")

    def prim(self, path: str) -> str:
        """Resolve a path relative to the articulation root; absolute paths pass through, "." is the root."""
        if path.startswith("/"):
            return path
        if path == ".":
            return self.root
        return f"{self.root.rstrip('/')}/{path}"

    # ── robot / articulation ────────────────────────────────────

    @property
    def name(self) -> str:
        return self.raw["robot"]["name"]

    @property
    def stage_path(self) -> str:
        return self.raw["robot"]["stage"]

    @property
    def root(self) -> str:
        return self.raw["articulation"]["root"]

    @property
    def num_dof(self) -> int:
        return int(self.raw["articulation"]["num_dof"])

    @property
    def arm_base_prim(self) -> str:
        return self.prim(self.raw["prims"]["arm_base"])

    @property
    def ee_prim(self) -> str:
        return self.prim(self.raw["prims"]["ee"])

    @property
    def tool_prim(self) -> str:
        return self.prim(self.raw["prims"]["tool"])

    @property
    def gripper_root_prim(self) -> str:
        return self.prim(self.raw["prims"]["gripper_root"])

    # ── joints ──────────────────────────────────────────────────

    @property
    def gantry_joint(self) -> str:
        return self.raw["joints"]["gantry"]

    @property
    def arm_joints(self) -> list[str]:
        return list(self.raw["joints"]["arm"])

    @property
    def cspace_joints(self) -> list[str]:
        """Planning joints in cuMotion order: gantry, then arm."""
        return [self.gantry_joint, *self.arm_joints]

    @property
    def gripper_drive_joint(self) -> str:
        return self.raw["joints"]["gripper_drive"]

    @property
    def gripper_modes(self) -> dict[str, str]:
        return dict(self.raw["joints"]["gripper_modes"])

    # ── gantry / home / tcp / orientation ───────────────────────

    @property
    def gantry_x_offset(self) -> float:
        return float(self.raw["gantry"]["x_offset"])

    @property
    def gantry_limits(self) -> tuple[float, float]:
        lo, hi = self.raw["gantry"]["limits"]
        return float(lo), float(hi)

    @property
    def home_joints(self) -> list[float] | None:
        home = self.raw["home_joints"]
        return None if home is None else [float(v) for v in home]

    @property
    def tcp_offset(self) -> float | None:
        return self.raw["tcp"]["offset"]

    @property
    def tool_down(self) -> list[float] | None:
        return self.raw["orientation"]["tool_down"]

    @property
    def scan_orientation(self) -> list[float] | None:
        return self.raw["orientation"]["scan"]

    # ── gripper ─────────────────────────────────────────────────

    @property
    def gripper(self) -> dict:
        return self.raw["gripper"]

    @property
    def finger(self) -> dict:
        return self.raw["gripper"]["finger"]

    @property
    def width_table(self) -> list[list[float]] | None:
        """Rows of [finger rad, pad gap m, pad-face centre depth m], or None."""
        return self.raw["gripper"].get("width_table")

    @property
    def gripper_joint_gains(self) -> dict[str, list[float]]:
        """Joint -> [stiffness, damping, max force] applied at bridge start."""
        return self.raw["gripper"].get("joint_gains") or {}

    @property
    def gripper_armature(self) -> float:
        """Joint armature [kg m^2] for every gripper_modes joint (0 = leave the asset's)."""
        return float(self.raw["gripper"].get("armature") or 0.0)

    def gap_for_finger(self, finger_rad: float) -> float:
        """Pad gap [m] at a finger_joint angle (width_table, else linear in stroke)."""
        table = self.width_table
        if table:
            rad, gap = [r[0] for r in table], [r[1] for r in table]
            return float(_interp(finger_rad, rad, gap))
        frac = min(max(finger_rad / self.finger["max_rad"], 0.0), 1.0)
        return float(self.gripper["stroke"] * (1.0 - frac))

    def finger_for_gap(self, gap: float) -> float:
        """finger_joint angle [rad] for a pad gap [m]; clamped to the table range."""
        table = self.width_table
        if table:
            rad, gaps = [r[0] for r in table][::-1], [r[1] for r in table][::-1]
            return float(_interp(gap, gaps, rad))
        frac = min(max(gap / self.gripper["stroke"], 0.0), 1.0)
        return float(self.finger["max_rad"] * (1.0 - frac))

    @property
    def contact_sensor_prim(self) -> str:
        return self.prim(self.raw["gripper"]["contact_sensor"])

    @property
    def finger_link_prims(self) -> list[str]:
        return [self.prim(p) for p in self.raw["gripper"]["finger_links"]]

    # ── wrist camera ────────────────────────────────────────────

    @property
    def wrist_camera(self) -> dict:
        return self.raw["wrist_camera"]

    @property
    def camera_mount_prim(self) -> str:
        return self.prim(self.raw["wrist_camera"]["mount_link"])

    # ── cuMotion ────────────────────────────────────────────────

    @property
    def urdf_path(self) -> Path:
        return self.config_dir / self.raw["cumotion"]["urdf"]

    @property
    def xrdf_path(self) -> Path:
        return self.config_dir / self.raw["cumotion"]["xrdf"]

    @property
    def self_filter_xrdf_path(self) -> Path:
        return self.config_dir / self.raw["cumotion"]["self_filter_xrdf"]

    @property
    def cumotion_tool_frame(self) -> str:
        return self.raw["cumotion"]["tool_frame"]

    @property
    def cumotion_root_link(self) -> str:
        return self.raw["cumotion"]["root_link"]

    @property
    def max_vel(self) -> list[float]:
        return [float(v) for v in self.raw["cumotion"]["max_vel"]]

    @property
    def max_acc(self) -> list[float]:
        return [float(v) for v in self.raw["cumotion"]["max_acc"]]

    @property
    def link_prims(self) -> dict[str, str]:
        return {k: self.prim(v) for k, v in (self.raw["cumotion"]["link_prims"] or {}).items()}

    # ── nvblox collision world ──────────────────────────────────

    @property
    def nvblox(self) -> dict | None:
        """nvblox service settings, or None when this robot has no nvblox section."""
        return self.raw.get("nvblox")

    # ── validation ──────────────────────────────────────────────

    def _validate(self) -> None:
        src = self.config_dir / ROBOT_YAML
        absent = [k for k in _REQUIRED_KEYS if not self._has(k)]
        if absent:
            raise RobotConfigError(f"{src}: missing keys: {', '.join(absent)}")

        errors = []
        modes = self.raw["joints"]["gripper_modes"] or {}
        bad = {j: m for j, m in modes.items() if m not in GRIPPER_MODES}
        if bad:
            errors.append(f"joints.gripper_modes: invalid modes {bad}, use {GRIPPER_MODES}")
        if modes.get(self.gripper_drive_joint) != "drive":
            errors.append("joints.gripper_drive must have mode 'drive' in gripper_modes")

        table = self.width_table
        if table is not None:
            if any(len(r) != 3 for r in table):
                errors.append("gripper.width_table: rows need [finger_rad, gap_m, centre_depth_m]")
            elif not (all(a[0] < b[0] for a, b in zip(table, table[1:]))
                      and all(a[1] > b[1] for a, b in zip(table, table[1:]))):
                errors.append("gripper.width_table: finger_rad must increase and gap decrease")
        for joint, gains in self.gripper_joint_gains.items():
            if joint not in modes or len(gains) != 3:
                errors.append(f"gripper.joint_gains.{joint}: needs a gripper_modes joint "
                              "and [stiffness, damping, max_force]")
        armature = self.raw["gripper"].get("armature")
        if armature is not None and (not isinstance(armature, (int, float)) or armature < 0):
            errors.append("gripper.armature: needs a number >= 0 [kg m^2]")

        n_cspace = len(self.cspace_joints)
        for key in ("cumotion.max_vel", "cumotion.max_acc"):
            if len(self.get(key)) != n_cspace:
                errors.append(f"{key}: expected {n_cspace} values (gantry + arm)")

        home = self.raw["home_joints"]
        if home is not None and len(home) != n_cspace:
            errors.append(f"home_joints: expected {n_cspace} values or null")

        lo, hi = self.gantry_limits
        if lo >= hi:
            errors.append("gantry.limits: lower must be below upper")

        for key in ("orientation.tool_down", "orientation.scan"):
            q = self.get(key)
            if q is not None and len(q) != 4:
                errors.append(f"{key}: quaternion needs 4 values [w, x, y, z]")

        xy = self.get("wrist_camera.ee_offset_xy")
        if xy is not None and len(xy) != 2:
            errors.append("wrist_camera.ee_offset_xy: needs 2 values [x, y]")

        if not self.root.startswith("/"):
            errors.append("articulation.root must be an absolute prim path")

        nv = self.nvblox
        if nv is not None:
            for key in ("url", "voxel_size", "cameras", "workspace_min", "workspace_max",
                        "robot_mask_margin", "max_integration_distance",
                        "esdf_max_site_distance_vox"):
                if nv.get(key) is None:
                    errors.append(f"nvblox.{key}: not set")
            lo, hi = nv.get("workspace_min") or [], nv.get("workspace_max") or []
            if len(lo) != 3 or len(hi) != 3 or any(a >= b for a, b in zip(lo, hi)):
                errors.append("nvblox.workspace_min/max: need [x, y, z] with min < max")
            if (nv.get("voxel_size") or 0) <= 0:
                errors.append("nvblox.voxel_size: needs a number > 0 [m]")
            link_margins = nv.get("robot_mask_link_margins") or {}
            unknown = set(link_margins) - set(self.raw["cumotion"]["link_prims"] or {})
            if unknown:
                errors.append(f"nvblox.robot_mask_link_margins: not in cumotion.link_prims: "
                              f"{sorted(unknown)}")

        if errors:
            raise RobotConfigError(f"{src}:\n  " + "\n  ".join(errors))


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    """Piecewise-linear interpolation with xs increasing; clamps outside the range."""
    if x <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def load_robot_config(config_dir: str | os.PathLike | None = None) -> RobotConfig:
    """Load ``robot.yaml`` from *config_dir*, or from ``$ROBOT_CONFIG_PATH``."""
    _load_dotenv_if_present()
    _ensure_project_root_env()
    if config_dir is None:
        config_dir = _expand_string(os.environ.get("ROBOT_CONFIG_PATH", ""))
        if not config_dir:
            raise RobotConfigError("ROBOT_CONFIG_PATH is not set (see .env)")
    config_dir = Path(config_dir).expanduser().resolve()
    path = config_dir / ROBOT_YAML
    if not path.is_file():
        raise RobotConfigError(f"robot config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return RobotConfig(_expand(data), config_dir)
