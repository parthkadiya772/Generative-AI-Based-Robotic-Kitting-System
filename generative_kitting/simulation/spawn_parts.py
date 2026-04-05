"""
Part Spawning for the Kitting Workspace.

Randomly scatters industrial parts across the workspace table
for training, testing, and demonstration purposes.
"""

import os
import json
import random
from typing import List, Optional, Tuple

from utils.logger import log


# Default part pool with relative probabilities
DEFAULT_PART_POOL = [
    {"label": "motor_valve",       "usd": "motor_valve.usd",       "weight": 1},
    {"label": "black_hose",        "usd": "black_hose.usd",        "weight": 2},
    {"label": "black_plate",       "usd": "black_plate.usd",       "weight": 2},
    {"label": "black_plug",        "usd": "black_plug.usd",        "weight": 3},
    {"label": "small_hinge",       "usd": "small_hinge.usd",       "weight": 3},
    {"label": "small_tube",        "usd": "small_tube.usd",        "weight": 2},
    {"label": "silver_box",        "usd": "silver_box.usd",        "weight": 1},
    {"label": "silver_gun",        "usd": "silver_gun.usd",        "weight": 1},
    {"label": "tube_with_clamps",  "usd": "tube_with_clamps.usd",  "weight": 1},
]


def spawn_random_parts(
    stage,
    meshes_dir: str = "../usd_meshes",
    count: int = 6,
    workspace_bounds: dict = None,
    part_pool: list = None,
    seed: Optional[int] = None,
) -> List[dict]:
    """
    Spawn random parts on the workspace table.

    Parameters
    ----------
    stage : Usd.Stage
        The USD stage.
    meshes_dir : str
        Directory containing the USD mesh files.
    count : int
        Number of parts to spawn.
    workspace_bounds : dict
        Keys: x_min, x_max, y_min, y_max, z_min.
    part_pool : list, optional
        Custom part definitions. Each dict: {label, usd, weight}.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list of dict
        Spawned part info: [{prim_path, label, position}, ...]
    """
    try:
        from pxr import Gf, UsdGeom
        from omni.isaac.core.utils.stage import add_reference_to_stage
    except ImportError:
        log.warning("Isaac Sim not available — cannot spawn parts")
        return []

    if seed is not None:
        random.seed(seed)

    pool = part_pool or DEFAULT_PART_POOL
    bounds = workspace_bounds or {
        "x_min": 0.0, "x_max": 0.6,
        "y_min": -0.3, "y_max": 0.3,
        "z_min": 0.02,
    }

    # Weighted random selection
    labels = [p["label"] for p in pool]
    weights = [p["weight"] for p in pool]
    usd_map = {p["label"]: p["usd"] for p in pool}

    spawned = []
    for i in range(count):
        label = random.choices(labels, weights=weights, k=1)[0]
        usd_file = usd_map[label]
        usd_path = os.path.join(meshes_dir, usd_file)

        if not os.path.exists(usd_path):
            log.warning(f"USD file not found: {usd_path} — skipping")
            continue

        prim_path = f"/World/SpawnedParts/{label}_{i:03d}"

        # Random position within bounds
        x = random.uniform(bounds["x_min"], bounds["x_max"])
        y = random.uniform(bounds["y_min"], bounds["y_max"])
        z = bounds.get("z_min", 0.02)

        # Random Z rotation
        rz = random.uniform(0, 360)

        add_reference_to_stage(usd_path, prim_path)

        # Set pose
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))

            # Apply random rotation around Z
            rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), rz)
            quat = rotation.GetQuat()
            xform.AddOrientOp().Set(Gf.Quatf(quat))

            # Apply scale (parts are in mm, scene in m)
            xform.AddScaleOp().Set(Gf.Vec3d(0.001, 0.001, 0.001))

        spawned.append({
            "prim_path": prim_path,
            "label": label,
            "position": {"x": x, "y": y, "z": z},
            "rotation_z_deg": rz,
        })

        log.debug(f"Spawned {label} at ({x:.3f}, {y:.3f}, {z:.3f}) rot_z={rz:.1f}°")

    log.info(f"Spawned {len(spawned)} random parts on workspace")
    return spawned


def spawn_kit_tray(
    stage,
    position: list = None,
    meshes_dir: str = "../usd_meshes",
    tray_usd: str = "box_840.usd",
) -> Optional[str]:
    """
    Spawn the kit tray at the designated position.

    Parameters
    ----------
    stage : Usd.Stage
    position : list
        [x, y, z] position for the tray.
    meshes_dir : str
        Directory containing USD meshes.
    tray_usd : str
        USD file for the tray mesh.

    Returns
    -------
    str or None
        Prim path of the spawned tray.
    """
    try:
        from pxr import Gf, UsdGeom
        from omni.isaac.core.utils.stage import add_reference_to_stage
    except ImportError:
        log.warning("Isaac Sim not available — cannot spawn kit tray")
        return None

    pos = position or [0.5, 0.0, 0.02]
    usd_path = os.path.join(meshes_dir, tray_usd)
    prim_path = "/World/KitTray"

    if not os.path.exists(usd_path):
        log.warning(f"Tray USD not found: {usd_path}")
        return None

    add_reference_to_stage(usd_path, prim_path)

    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))

    log.info(f"Kit tray spawned at {pos}")
    return prim_path


def clear_spawned_parts(stage) -> int:
    """Remove all previously spawned parts."""
    try:
        from omni.isaac.core.utils.prims import delete_prim
    except ImportError:
        return 0

    prim = stage.GetPrimAtPath("/World/SpawnedParts")
    if prim.IsValid():
        children = list(prim.GetChildren())
        count = len(children)
        delete_prim("/World/SpawnedParts")
        log.info(f"Cleared {count} spawned parts")
        return count
    return 0
