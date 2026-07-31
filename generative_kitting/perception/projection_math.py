"""Pure pinhole re-projection — mirrors the depth branch of
``isaac_sim_bridge._project_to_world``.

The projection-calibration tuner needs to re-project a part's *stored*
centre pixel + depth under operator-overridden camera parameters, live,
without re-running the bridge or the VLM. The centre pixel and its depth
do not depend on the camera model, so only this small amount of math has
to be repeated per parameter change.

Keep this in lock-step with the bridge depth-branch formulas
(``cam_x = sign_x * (px - cx) * d / fx`` etc.) so the tuner's numbers
match what the robot would actually receive.
"""
from __future__ import annotations

from typing import Sequence, Tuple

Matrix4 = Sequence[Sequence[float]]


def reproject_depth_point(
    nx: float, ny: float, depth: float, *,
    cam_world_matrix: Matrix4,
    w_res: int, h_res: int, w_depth: int, h_depth: int,
    fx: float, fy: float, cx: float, cy: float,
    img_x_sign: float = 1.0, img_y_sign: float = 1.0,
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[float, float, float]:
    """Re-project one normalised pixel + depth to a world XYZ.

    Parameters mirror the bridge. ``cam_world_matrix`` is the camera's
    4x4 world transform, row-major, in pxr's row-vector convention
    (``world = [cam_x, cam_y, cam_z, 1] · M``; translation in row 3).
    ``fx/fy/cx/cy`` are in RGB (``w_res``) resolution and are rescaled to
    depth resolution internally, exactly as the bridge does. ``offset`` is
    a world-space translation added after projection (a knob for testing a
    constant frame bias).
    """
    # Pixel indices in depth-buffer resolution (matches bridge int() trunc).
    px = min(int(nx * (w_depth - 1)), w_depth - 1)
    py = min(int(ny * (h_depth - 1)), h_depth - 1)

    sx = w_depth / w_res
    sy = h_depth / h_res
    fx_d, fy_d = fx * sx, fy * sy
    cx_d, cy_d = cx * sx, cy * sy

    cam_x = img_x_sign * (px - cx_d) * depth / fx_d
    cam_y = img_y_sign * -(py - cy_d) * depth / fy_d
    cam_z = -depth

    m = cam_world_matrix
    wx = cam_x * m[0][0] + cam_y * m[1][0] + cam_z * m[2][0] + m[3][0]
    wy = cam_x * m[0][1] + cam_y * m[1][1] + cam_z * m[2][1] + m[3][1]
    wz = cam_x * m[0][2] + cam_y * m[1][2] + cam_z * m[2][2] + m[3][2]
    return (wx + offset[0], wy + offset[1], wz + offset[2])
