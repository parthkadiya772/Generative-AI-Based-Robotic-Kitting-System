"""
Unit tests for the robot hardware config loader (robot_config/<robot>/robot.yaml).
"""

import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.robot_config import RobotConfigError, load_robot_config

ROBOT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "robot_config"
UR10_DIR = ROBOT_CONFIG_DIR / "ur10"
UR10E_DIR = ROBOT_CONFIG_DIR / "ur10e"


def _write_variant(tmp_path, base_dir, edit):
    """Copy a robot.yaml into tmp_path with *edit(data)* applied."""
    data = yaml.safe_load((base_dir / "robot.yaml").read_text())
    edit(data)
    (tmp_path / "robot.yaml").write_text(yaml.safe_dump(data))
    return tmp_path


# ═════════════════════════════════════════════════════════════
# SHIPPED CONFIGS
# ═════════════════════════════════════════════════════════════

class TestShippedConfigs:

    def test_ur10_is_complete(self):
        cfg = load_robot_config(UR10_DIR)
        assert cfg.missing() == []
        cfg.require_complete()

    def test_ur10e_is_complete(self):
        cfg = load_robot_config(UR10E_DIR)
        assert cfg.missing() == []
        cfg.require_complete()

    def test_null_value_blocks_require_complete(self, tmp_path):
        d = _write_variant(tmp_path, UR10E_DIR, lambda c: c["tcp"].update(offset=None))
        cfg = load_robot_config(d)
        assert cfg.missing() == ["tcp.offset"]
        with pytest.raises(RobotConfigError, match="tcp.offset"):
            cfg.require_complete()

    def test_ur10e_prim_paths(self):
        cfg = load_robot_config(UR10E_DIR)
        assert cfg.root == "/World/gantry"
        assert cfg.ee_prim == "/World/gantry/gantry_home/ur10e/ee_link"
        assert cfg.tool_prim == "/World/gantry/gantry_home/ur10e_robotiq_2f140/robotiq_base_link"
        assert cfg.contact_sensor_prim == (
            "/World/gantry/gantry_home/ur10e_robotiq_2f140/left_inner_finger/Contact_Sensor")
        assert cfg.camera_mount_prim == "/World/gantry/gantry_home/ur10e/wrist_3_link"
        assert cfg.wrist_camera["depth"] == "/World/rsd555/Depth"

    def test_ur10e_joints(self):
        cfg = load_robot_config(UR10E_DIR)
        assert cfg.gantry_joint == "gantry_vagn_joint"
        assert cfg.cspace_joints[0] == "gantry_vagn_joint"
        assert len(cfg.cspace_joints) == 7
        assert cfg.gripper_drive_joint == "finger_joint"
        assert cfg.gripper_modes["finger_joint"] == "drive"

    def test_ur10_matches_old_bridge_constants(self):
        cfg = load_robot_config(UR10_DIR)
        assert cfg.ee_prim == "/World/gantry/gantry_home/ur10_flattened/ur10_instanceable/ee_link"
        assert cfg.tcp_offset == pytest.approx(0.19357)
        assert cfg.tool_down == [1.0, 0.0, 1.0, 0.0]
        assert cfg.gantry_limits == (-3.10, 0.0)
        assert cfg.finger["close"] == 0.5
        assert cfg.max_vel == [0.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0]

    def test_cumotion_files_resolve_to_robot_folder(self):
        cfg = load_robot_config(UR10_DIR)
        assert cfg.urdf_path == UR10_DIR / "AIKIDO.urdf"
        assert cfg.urdf_path.is_file()
        assert cfg.xrdf_path.is_file()

    def test_ur10e_link_prims(self):
        cfg = load_robot_config(UR10E_DIR)
        assert cfg.link_prims["gantry"] == "/World/gantry"
        assert cfg.link_prims["left_inner_finger"] == (
            "/World/gantry/gantry_home/ur10e_robotiq_2f140/left_inner_finger")
        assert cfg.self_filter_xrdf_path == UR10E_DIR / "AIKIDO.editor.xrdf"

    def test_ur10e_xrdf_sphere_links_have_prims(self):
        cfg = load_robot_config(UR10E_DIR)
        for path in (cfg.xrdf_path, cfg.self_filter_xrdf_path):
            data = yaml.safe_load(path.read_text())
            links = set(data["geometry"]["auto_generated_collision_sphere_group"]["spheres"])
            assert links <= set(cfg.link_prims), f"{path.name}: {links - set(cfg.link_prims)}"

    def test_ur10e_world_collision_skips_static_links(self):
        data = yaml.safe_load(load_robot_config(UR10E_DIR).xrdf_path.read_text())
        world = data["geometry"][data["world_collision"]["geometry"]]["spheres"]
        self_ = data["geometry"][data["self_collision"]["geometry"]]["spheres"]
        assert "gantry" not in world and "vagn" not in world
        assert {"gantry", "vagn"} <= set(self_)
        assert set(world) == set(self_) - {"gantry", "vagn"}

    def test_ur10e_parallel_pad_modes(self):
        cfg = load_robot_config(UR10E_DIR)
        assert cfg.gripper_modes["left_inner_finger_joint"] == "mirror"
        assert cfg.gripper_modes["left_inner_finger_pad_joint"] == "pad"
        assert set(cfg.gripper_joint_gains) == {
            "finger_joint", "left_inner_finger_joint", "right_inner_finger_joint",
            "left_inner_finger_pad_joint", "right_inner_finger_pad_joint"}

    def test_width_table_mapping(self):
        cfg = load_robot_config(UR10E_DIR)
        assert cfg.gap_for_finger(0.0) == pytest.approx(0.1438)
        assert cfg.gap_for_finger(0.45) == pytest.approx((0.0757 + 0.0566) / 2)
        assert cfg.finger_for_gap(0.0757) == pytest.approx(0.4)
        assert cfg.finger_for_gap(0.5) == pytest.approx(0.0)       # wider than open
        assert cfg.finger_for_gap(0.0) == pytest.approx(0.76)      # clamps at closed
        for rad in (0.05, 0.33, 0.71):
            assert cfg.finger_for_gap(cfg.gap_for_finger(rad)) == pytest.approx(rad)

    def test_linear_fallback_without_width_table(self):
        cfg = load_robot_config(UR10_DIR)
        assert cfg.width_table is None
        assert cfg.gap_for_finger(0.0) == pytest.approx(0.140)
        assert cfg.finger_for_gap(0.0) == pytest.approx(cfg.finger["max_rad"])

    def test_project_root_expanded(self):
        cfg = load_robot_config(UR10E_DIR)
        assert "${" not in cfg.stage_path
        assert cfg.stage_path.endswith("UR10e_Container_Assets/AIKIDO_ur10e_swapped.usd")


# ═════════════════════════════════════════════════════════════
# DIRECTORY SELECTION
# ═════════════════════════════════════════════════════════════

class TestDirectorySelection:

    def test_env_var_selects_robot(self, monkeypatch):
        monkeypatch.setenv("ROBOT_CONFIG_PATH", str(UR10E_DIR))
        assert load_robot_config().name == "ur10e_2f140_gantry"

    def test_missing_folder(self, tmp_path):
        with pytest.raises(RobotConfigError, match="not found"):
            load_robot_config(tmp_path)


# ═════════════════════════════════════════════════════════════
# VALIDATION
# ═════════════════════════════════════════════════════════════

class TestValidation:

    def test_missing_key(self, tmp_path):
        d = _write_variant(tmp_path, UR10_DIR, lambda c: c["tcp"].pop("offset"))
        with pytest.raises(RobotConfigError, match="missing keys: tcp.offset"):
            load_robot_config(d)

    def test_invalid_gripper_mode(self, tmp_path):
        def edit(c):
            c["joints"]["gripper_modes"]["left_inner_finger_joint"] = "spring"
        with pytest.raises(RobotConfigError, match="invalid modes"):
            load_robot_config(_write_variant(tmp_path, UR10_DIR, edit))

    def test_drive_joint_must_be_driven(self, tmp_path):
        def edit(c):
            c["joints"]["gripper_modes"]["finger_joint"] = "free"
        with pytest.raises(RobotConfigError, match="gripper_drive"):
            load_robot_config(_write_variant(tmp_path, UR10_DIR, edit))

    def test_limit_length_must_match_cspace(self, tmp_path):
        def edit(c):
            c["cumotion"]["max_vel"] = [1.0] * 6
        with pytest.raises(RobotConfigError, match="max_vel"):
            load_robot_config(_write_variant(tmp_path, UR10_DIR, edit))

    def test_bad_quaternion(self, tmp_path):
        def edit(c):
            c["orientation"]["tool_down"] = [1.0, 0.0, 0.0]
        with pytest.raises(RobotConfigError, match="quaternion"):
            load_robot_config(_write_variant(tmp_path, UR10_DIR, edit))

    def test_home_joints_length(self, tmp_path):
        def edit(c):
            c["home_joints"] = [0.0] * 6
        with pytest.raises(RobotConfigError, match="home_joints"):
            load_robot_config(_write_variant(tmp_path, UR10_DIR, edit))

    def test_width_table_must_be_monotonic(self, tmp_path):
        def edit(c):
            c["gripper"]["width_table"][2][1] = 0.2
        with pytest.raises(RobotConfigError, match="width_table"):
            load_robot_config(_write_variant(tmp_path, UR10E_DIR, edit))

    def test_joint_gains_need_known_joint(self, tmp_path):
        def edit(c):
            c["gripper"]["joint_gains"]["no_such_joint"] = [1.0, 0.1, 1.0]
        with pytest.raises(RobotConfigError, match="joint_gains"):
            load_robot_config(_write_variant(tmp_path, UR10E_DIR, edit))

    def test_absolute_paths_pass_through(self):
        cfg = load_robot_config(UR10E_DIR)
        assert cfg.prim("/World/rsd555") == "/World/rsd555"


def test_env_var_with_project_root_placeholder(monkeypatch):
    monkeypatch.setenv("ROBOT_CONFIG_PATH", "${PROJECT_ROOT}/generative_kitting/robot_config/ur10e")
    assert load_robot_config().config_dir == UR10E_DIR
