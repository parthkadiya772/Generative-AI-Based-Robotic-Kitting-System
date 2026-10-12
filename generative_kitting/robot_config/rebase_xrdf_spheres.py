"""
Re-express XRDF collision spheres from USD link frames in URDF link frames.

The Robot Description Editor writes sphere centres in each link prim's local
frame. The URDF Exporter puts a child link's frame at its joint
(localPos1/localRot1, plus an axis-flip correction), so when the two differ
(the 2F-140 finger links), cuMotion would place the spheres in the wrong spot.

Optionally also writes a world-collision sphere group without static links
(the gantry frame and vagn only need self-collision checks against the arm;
checking their 650 spheres against the camera world made planning ~20x slower).

Run inside Isaac Sim with the robot stage open:

    exec(open(".../robot_config/rebase_xrdf_spheres.py").read())
    rebase_xrdf_spheres(".../ur10e/AIKIDO.editor.xrdf", ".../ur10e/AIKIDO.xrdf",
                        world_collision_exclude=("gantry", "vagn"))
"""

import copy

import numpy as np
import omni.usd
import yaml
from pxr import Gf, UsdGeom, UsdPhysics

from isaacsim.asset.exporter.urdf.converter.robot_finder import find_robot
from isaacsim.asset.exporter.urdf.converter.urdf_frames import (
    _detect_axis_flip,
    _get_axis_token,
    _get_joint_body_path,
    _make_axis_flip_correction,
)


def _urdf_frame_in_link(joint_prim):
    """Child-side joint frame (= URDF link frame) in the child link prim's frame, row-vector convention."""
    joint = UsdPhysics.Joint(joint_prim)
    pos = joint.GetLocalPos1Attr().Get() or Gf.Vec3f(0.0)
    rot = joint.GetLocalRot1Attr().Get() or Gf.Quatf(1.0)
    frame = Gf.Matrix4d(1.0)
    frame.SetTransform(Gf.Rotation(Gf.Quatd(rot)), Gf.Vec3d(pos))
    if _detect_axis_flip(joint_prim):
        frame = _make_axis_flip_correction(_get_axis_token(joint_prim)) * frame
    return frame


def link_frame_offsets(root_prim_path="/World/gantry"):
    """Map URDF link name -> 4x4 (row-vector) transform taking USD-link coords to URDF-link coords."""
    stage = omni.usd.get_context().get_stage()
    desc = find_robot(stage, root_prim_path)
    offsets = {}
    for joint_prim in desc.ordered_joints:
        child_path = _get_joint_body_path(joint_prim, 1)
        if not child_path:
            continue
        child = stage.GetPrimAtPath(child_path)
        scale = Gf.Transform(UsdGeom.Xformable(child).GetLocalTransformation()).GetScale()
        if not all(abs(s - 1.0) < 1e-6 for s in scale):
            raise RuntimeError(f"{child_path} has non-unit scale {scale}; offsets would be wrong")
        offsets[child.GetName()] = np.array(_urdf_frame_in_link(joint_prim).GetInverse(), dtype=float)
    return offsets


WORLD_COLLISION_GROUP = "world_collision_sphere_group"


def rebase_xrdf_spheres(editor_xrdf, out_xrdf, root_prim_path="/World/gantry",
                        world_collision_exclude=(), tol=1e-6):
    """Write *out_xrdf* with every sphere centre moved into its URDF link frame. Returns moved link names."""
    with open(editor_xrdf) as f:
        data = yaml.safe_load(f)
    offsets = link_frame_offsets(root_prim_path)
    moved = []
    for group in data.get("geometry", {}).values():
        for link, spheres in group.get("spheres", {}).items():
            T = offsets.get(link)
            if T is None or np.allclose(T, np.eye(4), atol=tol):
                continue
            for s in spheres:
                c = np.append(np.asarray(s["center"], dtype=float), 1.0) @ T
                s["center"] = [round(float(v), 6) for v in c[:3]]
            moved.append(link)
    if world_collision_exclude:
        self_group = data["self_collision"]["geometry"]
        spheres = data["geometry"][self_group]["spheres"]
        missing = set(world_collision_exclude) - set(spheres)
        if missing:
            raise KeyError(f"world_collision_exclude: no spheres for {sorted(missing)}")
        data["geometry"][WORLD_COLLISION_GROUP] = {
            "spheres": copy.deepcopy({k: v for k, v in spheres.items() if k not in world_collision_exclude})}
        data["world_collision"] = {"geometry": WORLD_COLLISION_GROUP}
    with open(out_xrdf, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return moved
