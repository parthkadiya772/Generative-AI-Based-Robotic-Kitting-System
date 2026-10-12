"""Unit tests for the pure re-projection helper used by the tuner."""
import math

from perception.projection_math import reproject_depth_point


# Overhead camera at (0, 0, H) looking straight down, axes aligned with
# world axes. In pxr row-vector convention the world matrix has the camera
# basis vectors in rows 0-2 and the translation in row 3.
H = 5.0
CAM_M = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, H,   1.0],
]
INTR = dict(w_res=1920, h_res=1080, w_depth=1920, h_depth=1080,
            fx=2000.0, fy=2000.0, cx=960.0, cy=540.0)


def _forward(world_x, world_y):
    """Forward-project an aligned-overhead world point to (nx, ny, depth)."""
    # cam frame (axes aligned): cam_pt = world - cam_pos
    cam_x, cam_y, cam_z = world_x, world_y, 0.0 - H
    d = -cam_z
    px = INTR["cx"] + cam_x * INTR["fx"] / d
    py = INTR["cy"] - cam_y * INTR["fy"] / d
    nx = px / (INTR["w_depth"] - 1)
    ny = py / (INTR["h_depth"] - 1)
    return nx, ny, d


def test_round_trip_recovers_world_point():
    wx, wy = 0.30, -0.20
    nx, ny, d = _forward(wx, wy)
    x, y, z = reproject_depth_point(
        nx, ny, d, cam_world_matrix=CAM_M, img_x_sign=1, img_y_sign=1,
        **INTR)
    # 1 px ≈ d/fx = 2.5 mm; int() truncation can cost ~1 px.
    assert math.isclose(x, wx, abs_tol=0.003)
    assert math.isclose(y, wy, abs_tol=0.003)
    assert math.isclose(z, 0.0, abs_tol=1e-6)


def test_world_offset_applies():
    nx, ny, d = _forward(0.0, 0.0)
    base = reproject_depth_point(nx, ny, d, cam_world_matrix=CAM_M, **INTR)
    shifted = reproject_depth_point(
        nx, ny, d, cam_world_matrix=CAM_M, offset=(0.057, -0.039, 0.0),
        **INTR)
    assert math.isclose(shifted[0] - base[0], 0.057, abs_tol=1e-9)
    assert math.isclose(shifted[1] - base[1], -0.039, abs_tol=1e-9)


def test_principal_point_shift_moves_world_x():
    """A cx change of N px shifts world X by N * d / fx (the constant-bias knob)."""
    nx, ny, d = _forward(0.0, 0.0)
    base = reproject_depth_point(nx, ny, d, cam_world_matrix=CAM_M, **INTR)
    intr2 = dict(INTR)
    intr2["cx"] = INTR["cx"] + 20.0  # +20 px principal-point error
    moved = reproject_depth_point(nx, ny, d, cam_world_matrix=CAM_M, **intr2)
    expected_dx = -20.0 * d / INTR["fx"]  # cam_x = (px - cx)*d/fx
    assert math.isclose(moved[0] - base[0], expected_dx, abs_tol=1e-6)


def test_x_sign_flip_mirrors():
    nx, ny, d = _forward(0.25, 0.0)
    pos = reproject_depth_point(nx, ny, d, cam_world_matrix=CAM_M,
                                img_x_sign=1, **INTR)
    neg = reproject_depth_point(nx, ny, d, cam_world_matrix=CAM_M,
                                img_x_sign=-1, **INTR)
    assert math.isclose(pos[0], -neg[0], abs_tol=1e-9)
