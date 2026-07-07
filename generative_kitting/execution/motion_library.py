"""
Motion Library — Pre-computed Motion Sequences.

Pure functions that generate waypoint lists for common motion patterns.
These are consumed by RobotController — no sim dependency.
"""

from typing import Dict, List, Tuple
from utils.logger import log


def compute_pre_grasp_approach(
    target_x: float,
    target_y: float,
    target_z: float,
    safe_z: float,
    hover_clearance: float = 0.015,
    grasp_depth_fraction: float = 0.5,
    part_height: float = 0.05,
    tcp_z_offset: float = 0.0,
) -> List[Dict]:
    """
    Compute waypoints for a pre-grasp approach sequence.

    Generates the motion path:
        safe_height → hover_above_part → grasp_height

    Parameters
    ----------
    target_x, target_y, target_z : float
        Part position (world coordinates).
    safe_z : float
        Safe height above any obstacles.
    hover_clearance : float
        Gap above the part top surface before final plunge.
    grasp_depth_fraction : float
        How deep into the part the fingers should close (0 = top, 1 = bottom).
    part_height : float
        Height of the part bounding box.
    tcp_z_offset : float
        Offset for IK frame (0 if gripper_tcp, GRIPPER_TCP_OFFSET if ee_link).

    Returns
    -------
    list of dict
        Waypoints with keys: name, x, y, z, description.
    """
    part_top_z = target_z + part_height / 2.0    # approximate top surface
    grasp_z = part_top_z - grasp_depth_fraction * part_height

    waypoints = [
        {
            "name": "safe_height",
            "x": target_x,
            "y": target_y,
            "z": safe_z + tcp_z_offset,
            "description": "Safe height above workspace (clear all obstacles)",
        },
        {
            "name": "hover",
            "x": target_x,
            "y": target_y,
            "z": part_top_z + hover_clearance + tcp_z_offset,
            "description": f"Hover {hover_clearance*1000:.0f}mm above part top surface",
        },
        {
            "name": "grasp",
            "x": target_x,
            "y": target_y,
            "z": grasp_z + tcp_z_offset,
            "description": f"Grasp at {grasp_depth_fraction*100:.0f}% depth into part",
        },
    ]

    log.debug(
        f"Pre-grasp approach: {len(waypoints)} waypoints, "
        f"safe_z={safe_z:.4f}, grasp_z={grasp_z:.4f}"
    )
    return waypoints


def compute_place_sequence(
    target_x: float,
    target_y: float,
    target_z: float,
    safe_z: float,
    drop_height: float = 0.0,
    retract_height: float = 0.20,
    tcp_z_offset: float = 0.0,
) -> List[Dict]:
    """
    Compute waypoints for a place sequence.

    Generates the motion path:
        safe_height_above_dest → place_height → [release] → retract

    Parameters
    ----------
    target_x, target_y, target_z : float
        Destination placement position.
    safe_z : float
        Safe height above the destination.
    drop_height : float
        Height above target surface for release.
    retract_height : float
        Height above destination after release.
    tcp_z_offset : float
        Offset for IK frame.

    Returns
    -------
    list of dict
        Waypoints with keys: name, x, y, z, description.
    """
    waypoints = [
        {
            "name": "safe_above_dest",
            "x": target_x,
            "y": target_y,
            "z": safe_z + tcp_z_offset,
            "description": "Safe height above destination",
        },
        {
            "name": "place_descend",
            "x": target_x,
            "y": target_y,
            "z": target_z + drop_height + tcp_z_offset,
            "description": f"Descend to place height ({drop_height*1000:.0f}mm above surface)",
        },
        # [RELEASE HAPPENS HERE]
        {
            "name": "retract",
            "x": target_x,
            "y": target_y,
            "z": target_z + retract_height + tcp_z_offset,
            "description": f"Retract {retract_height*1000:.0f}mm after release",
        },
    ]

    log.debug(
        f"Place sequence: {len(waypoints)} waypoints, "
        f"place_z={target_z + drop_height:.4f}"
    )
    return waypoints


def compute_transit_path(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    transit_z: float,
    tcp_z_offset: float = 0.0,
) -> List[Dict]:
    """
    Compute a safe transit path between two positions at high altitude.

    The robot lifts to transit_z, moves laterally, then is ready
    for the next sequence (approach or place).

    Parameters
    ----------
    start_x, start_y : float
        Starting XY position.
    end_x, end_y : float
        Target XY position.
    transit_z : float
        Safe transit altitude.
    tcp_z_offset : float
        Offset for IK frame.

    Returns
    -------
    list of dict
        Waypoints with keys: name, x, y, z, description.
    """
    z = transit_z + tcp_z_offset

    waypoints = [
        {
            "name": "lift_to_transit",
            "x": start_x,
            "y": start_y,
            "z": z,
            "description": f"Lift to transit height Z={transit_z:.4f}m",
        },
        {
            "name": "lateral_move",
            "x": end_x,
            "y": end_y,
            "z": z,
            "description": f"Lateral move to ({end_x:.3f}, {end_y:.3f})",
        },
    ]

    log.debug(
        f"Transit path: ({start_x:.3f},{start_y:.3f}) → "
        f"({end_x:.3f},{end_y:.3f}) at Z={transit_z:.4f}"
    )
    return waypoints


def compute_full_pick_place_sequence(
    pick_x: float,
    pick_y: float,
    pick_z: float,
    place_x: float,
    place_y: float,
    place_z: float,
    safe_z: float,
    transit_z: float,
    hover_clearance: float = 0.015,
    grasp_depth_fraction: float = 0.5,
    part_height: float = 0.05,
    drop_height: float = 0.0,
    retract_height: float = 0.20,
    tcp_z_offset: float = 0.0,
) -> Dict[str, List[Dict]]:
    """
    Compute the complete pick-and-place motion sequence.

    Returns a dict with named phases, each containing a list of waypoints.

    Returns
    -------
    dict
        Keys: "approach", "transit", "place" — each a list of waypoint dicts.
    """
    approach = compute_pre_grasp_approach(
        pick_x, pick_y, pick_z, safe_z,
        hover_clearance, grasp_depth_fraction, part_height, tcp_z_offset,
    )

    transit = compute_transit_path(
        pick_x, pick_y, place_x, place_y, transit_z, tcp_z_offset,
    )

    place = compute_place_sequence(
        place_x, place_y, place_z, safe_z,
        drop_height, retract_height, tcp_z_offset,
    )

    return {
        "approach": approach,
        "transit": transit,
        "place": place,
    }
