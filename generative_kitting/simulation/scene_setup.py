"""
Scene Setup — Isaac Sim Scene Initialisation.

Loads the pre-built AIKIDO.usd kitting workspace, configures
physics, sets up the camera, and initialises the robot articulation.
"""

import os
from typing import Any, Dict, Optional, Tuple

from utils.logger import log


def load_kitting_scene(config: dict):
    """
    Load the AIKIDO.usd kitting workspace scene into Isaac Sim.

    This function handles:
    1. Opening the pre-built USD scene file (AIKIDO.usd)
    2. Configuring physics parameters (timestep, gravity)

    Parameters
    ----------
    config : dict
        The full parsed config.yaml dict.

    Returns
    -------
    stage
        The USD stage with the loaded scene.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdPhysics
    except ImportError:
        log.warning("Isaac Sim modules not available — skipping scene setup")
        return None

    sim_cfg = config.get("simulation", {})
    scene_path = sim_cfg.get("usd_scene_path", "")

    if not scene_path:
        log.error("No usd_scene_path specified in config.yaml")
        return None

    # Normalise path separators for Isaac Sim (expects forward slashes)
    scene_path = scene_path.replace("\\", "/")

    if not os.path.exists(scene_path.replace("/", os.sep)):
        log.error(f"Scene file not found: {scene_path}")
        return None

    # Open the pre-built USD scene directly
    context = omni.usd.get_context()
    result, error = context.open_stage(scene_path)

    if not result:
        log.error(f"Failed to open AIKIDO scene: {error}")
        return None

    stage = context.get_stage()
    log.info(f"Loaded AIKIDO scene: {scene_path}")

    # Configure physics
    physics_dt = sim_cfg.get("physics_dt", 1.0 / 60.0)
    rendering_dt = sim_cfg.get("rendering_dt", 1.0 / 60.0)
    log.info(f"Physics dt={physics_dt:.6f}s, Rendering dt={rendering_dt:.6f}s")

    return stage


def _set_prim_pose(stage, prim_path, position, rotation, rotation_type="quat"):
    """Set position and rotation on a USD prim."""
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()

    # Position
    if isinstance(position, (list, tuple)):
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    elif isinstance(position, dict):
        xform.AddTranslateOp().Set(Gf.Vec3d(
            position.get("x", 0), position.get("y", 0), position.get("z", 0)
        ))

    # Rotation
    if rotation_type == "quat" and isinstance(rotation, dict):
        quat = Gf.Quatf(
            rotation.get("w", 1.0),
            rotation.get("x", 0.0),
            rotation.get("y", 0.0),
            rotation.get("z", 0.0),
        )
        xform.AddOrientOp().Set(quat)
    elif rotation_type == "euler_deg" and isinstance(rotation, (list, tuple)):
        quat_x = Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation[0]).GetQuat()
        quat_y = Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation[1]).GetQuat()
        quat_z = Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation[2]).GetQuat()
        quat = quat_x * quat_y * quat_z
        xform.AddOrientOp().Set(quat)


def _set_prim_scale(stage, prim_path, sx, sy, sz):
    """Set scale on a USD prim."""
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

    xform = UsdGeom.Xformable(prim)
    xform.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))


def setup_camera(
    stage,
    camera_prim_path: str = "/World/Camera_Top",
    position: tuple = (0.3, 0.0, 1.5),
    target: tuple = (0.3, 0.0, 0.0),
    resolution: tuple = (1280, 720),
):
    """
    Create or configure an overhead camera in the scene.

    Parameters
    ----------
    stage : Usd.Stage
        The USD stage.
    camera_prim_path : str
        Where to create the camera prim.
    position : tuple
        Camera world position (x, y, z).
    target : tuple
        Point the camera looks at.
    resolution : tuple
        Image resolution (width, height).

    Returns
    -------
    camera or None
        Isaac Sim Camera object, or None if not available.
    """
    try:
        from omni.isaac.sensor import Camera
        from pxr import UsdGeom, Gf

        # Create camera prim if it doesn't exist
        camera_prim = stage.GetPrimAtPath(camera_prim_path)
        if not camera_prim.IsValid():
            camera_prim = UsdGeom.Camera.Define(stage, camera_prim_path)
            log.info(f"Created camera prim at {camera_prim_path}")

        camera = Camera(prim_path=camera_prim_path, resolution=resolution)
        camera.set_world_pose(position=list(position))

        log.info(
            f"Camera configured at {camera_prim_path} — "
            f"pos={position}, resolution={resolution}"
        )
        return camera

    except ImportError:
        log.warning("Isaac Sim Camera not available — skipping camera setup")
        return None


def setup_robot(config: dict):
    """
    Initialise the UR10 robot articulation and Lula IK solver.

    Parameters
    ----------
    config : dict
        The full parsed config.yaml dict.

    Returns
    -------
    tuple of (robot, lula_solver) or (None, None)
    """
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from omni.isaac.motion_generation import LulaKinematicsSolver
    except ImportError:
        log.warning("Isaac Sim not available — returning None for robot setup")
        return None, None

    sim_cfg = config.get("simulation", {})
    robot_path = sim_cfg.get("robot_prim_path", "/World")

    world = World.instance() or World()
    robot = world.scene.get_object("kitting_sys")
    if not robot:
        robot = SingleArticulation(prim_path=robot_path, name="kitting_sys")
        world.scene.add(robot)

    log.info(f"Robot articulation at {robot_path}")
    return robot, None  # Lula solver created by RobotController.init_sim()
