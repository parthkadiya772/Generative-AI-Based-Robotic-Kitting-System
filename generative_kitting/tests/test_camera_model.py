"""Verify the bridge's hand-ported camera model.

Isaac Sim 6.0.1's RTX sensor API ships no projection helpers, so
isaac_sim_bridge.py implements the pinhole model itself. These tests
pin that implementation to the algebra the deprecated
isaacsim.sensors.camera.Camera used, using pure numpy — no Isaac Sim.

Reference formulas (isaacsim/sensors/camera/camera.py, 6.0.1):
    get_intrinsics_matrix        fx = w*focal/h_ap, cx = w/2
                                 fy = fx (square pixels; the bridge
                                 ignores verticalAperture)
    get_view_matrix_ros          R_U_TRANSFORM @ inv(local_to_world.T)
    get_camera_points_from_image_coords
                                 inv(K) @ (uv1.T * depth)
    get_world_points_from_image_coords
                                 inv(view) @ pad(cam_pts, 1)
    get_image_coords_from_world_points
                                 (K @ view[:3,:]) @ pad(pts, 1), then /w
"""

import numpy as np
import pytest

R_U_TRANSFORM = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])


def intrinsics(resolution, focal, h_ap):
    """Mirror of the bridge's _intrinsics_matrix (resolution is (h, w)).

    Square pixels: the renderer keeps the horizontal aperture at any
    resolution, so verticalAperture is not an input.
    """
    h_res, w_res = resolution
    f_px = w_res * focal / h_ap
    return np.array([
        [f_px, 0.0, w_res / 2.0],
        [0.0, f_px, h_res / 2.0],
        [0.0, 0.0, 1.0],
    ])


def view_matrix_ros(camera_to_world):
    """Mirror of the bridge's _view_matrix_ros, given cam->world already
    transposed out of USD row-vector layout."""
    return R_U_TRANSFORM @ np.linalg.inv(camera_to_world)


def world_from_pixels(K, view, pixels, depths):
    uv1 = np.hstack([np.asarray(pixels, float), np.ones((len(pixels), 1))])
    cam = np.linalg.inv(K) @ (uv1.T * np.asarray(depths, float))
    cam_h = np.vstack([cam, np.ones((1, cam.shape[1]))])
    return (np.linalg.inv(view) @ cam_h)[:3].T


def pixels_from_world(K, view, points):
    pts = np.asarray(points, float)
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
    projected = (K @ view[:3, :]) @ homogeneous.T
    projected[:2, :] /= projected[2, :]
    return projected[:2, :].T


# ── Fixtures: an overhead camera at Z=5.10 looking straight down ──
# Matches the real /World/Camera: 1920x1080, focal 15.9, aperture 20.955.
FOCAL, H_AP = 15.9, 20.955
RESOLUTION = (1080, 1920)                       # (h, w)


@pytest.fixture
def overhead():
    """Camera at (0,0,5.10), -Z_cam pointing at world -Z (straight down)."""
    # USD camera looks down its own -Z with +Y up. Pointing it at the
    # floor means -Z_cam = world -Z, so the camera basis is identity
    # except for the translation.
    camera_to_world = np.eye(4)
    camera_to_world[:3, 3] = [0.0, 0.0, 5.10]
    K = intrinsics(RESOLUTION, FOCAL, H_AP)
    return K, view_matrix_ros(camera_to_world)


class TestIntrinsics:
    def test_square_pixels_give_equal_focal_lengths(self):
        K = intrinsics(RESOLUTION, FOCAL, H_AP)
        assert K[0, 0] == pytest.approx(K[1, 1])

    def test_matches_measured_values_from_live_scene(self):
        # Diagnostic run against /World/Camera reported fx = fy = 1456.836
        K = intrinsics(RESOLUTION, FOCAL, H_AP)
        assert K[0, 0] == pytest.approx(1456.836, abs=1e-3)
        assert K[1, 1] == pytest.approx(1456.836, abs=1e-3)

    def test_principal_point_is_image_centre(self):
        K = intrinsics(RESOLUTION, FOCAL, H_AP)
        assert (K[0, 2], K[1, 2]) == (960.0, 540.0)

    def test_corner_camera_at_480x640(self):
        # cam1-4: focal 18.148, h_ap 20.955, authored v_ap 15.29 (not 4:3).
        # Their depth only lines up across cameras with fy = fx.
        K = intrinsics((480, 640), 18.14756, H_AP)
        assert K[0, 0] == pytest.approx(K[1, 1])
        assert (K[0, 2], K[1, 2]) == (320.0, 240.0)


class TestViewMatrix:
    def test_flips_y_and_z_into_ros_convention(self, overhead):
        _, view = overhead
        # World +Z (up) must map to ROS camera -Z (behind the lens) for a
        # downward-looking camera.
        d = view[:3, :3] @ np.array([0.0, 0.0, 1.0])
        assert d == pytest.approx([0.0, 0.0, -1.0])

    def test_camera_origin_maps_to_zero(self, overhead):
        _, view = overhead
        origin = view @ np.array([0.0, 0.0, 5.10, 1.0])
        assert origin[:3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)

    def test_point_below_camera_has_positive_depth(self, overhead):
        _, view = overhead
        # ROS convention: +Z is forward, so the floor is at positive Z.
        assert (view @ np.array([0.0, 0.0, 0.0, 1.0]))[2] == pytest.approx(5.10)

    def test_point_above_camera_has_negative_depth(self, overhead):
        _, view = overhead
        assert (view @ np.array([0.0, 0.0, 6.0, 1.0]))[2] < 0


class TestUnprojection:
    def test_image_centre_maps_to_point_below_camera(self, overhead):
        K, view = overhead
        world = world_from_pixels(K, view, [[960.0, 540.0]], [5.08])[0]
        assert world == pytest.approx([0.0, 0.0, 0.02], abs=1e-9)

    def test_round_trip_pixel_to_world_to_pixel(self, overhead):
        K, view = overhead
        pixels = [[100.0, 200.0], [960.0, 540.0], [1800.0, 1000.0]]
        depths = [5.0, 5.08, 4.9]
        world = world_from_pixels(K, view, pixels, depths)
        back = pixels_from_world(K, view, world)
        assert back == pytest.approx(np.array(pixels), abs=1e-6)

    def test_round_trip_world_to_pixel_to_world(self, overhead):
        K, view = overhead
        points = [[0.5, -0.3, 0.02], [-1.2, 0.8, 0.05], [0.0, 0.0, 0.0]]
        pixels = pixels_from_world(K, view, points)
        depths = [5.10 - p[2] for p in points]
        assert world_from_pixels(K, view, pixels, depths) == pytest.approx(
            np.array(points), abs=1e-9)

    def test_pixel_right_of_centre_maps_to_world_plus_x(self, overhead):
        K, view = overhead
        world = world_from_pixels(K, view, [[1460.0, 540.0]], [5.08])[0]
        assert world[0] > 0
        assert world[1] == pytest.approx(0.0, abs=1e-9)

    def test_pixel_below_centre_maps_to_world_minus_y(self, overhead):
        # Image +Y is down; ROS camera +Y is down; for a camera whose
        # +Y_cam aligns with world +Y, image-down is world -Y.
        K, view = overhead
        world = world_from_pixels(K, view, [[960.0, 1040.0]], [5.08])[0]
        assert world[0] == pytest.approx(0.0, abs=1e-9)
        assert world[1] < 0

    def test_scale_is_depth_over_focal_length(self, overhead):
        K, view = overhead
        depth = 5.08
        offset_px = 100.0
        world = world_from_pixels(K, view, [[960.0 + offset_px, 540.0]],
                                  [depth])[0]
        assert world[0] == pytest.approx(offset_px * depth / K[0, 0])

    def test_deeper_points_span_more_world_per_pixel(self, overhead):
        K, view = overhead
        near = world_from_pixels(K, view, [[1060.0, 540.0]], [1.0])[0]
        far = world_from_pixels(K, view, [[1060.0, 540.0]], [5.0])[0]
        assert abs(far[0]) > abs(near[0])


class TestRotatedCamera:
    """A rotated camera must need no sign overrides — the view matrix
    carries the orientation. This is what the removed
    camera_image_x_sign / camera_image_y_sign knobs were papering over.
    """

    @staticmethod
    def _yaw(degrees, height=5.10):
        t = np.radians(degrees)
        m = np.eye(4)
        m[:3, :3] = [[np.cos(t), -np.sin(t), 0.0],
                     [np.sin(t), np.cos(t), 0.0],
                     [0.0, 0.0, 1.0]]
        m[:3, 3] = [0.0, 0.0, height]
        return m

    @pytest.mark.parametrize("yaw", [0, 45, 90, 180, 270])
    def test_round_trip_holds_at_any_yaw(self, yaw):
        K = intrinsics(RESOLUTION, FOCAL, H_AP)
        view = view_matrix_ros(self._yaw(yaw))
        points = [[0.5, -0.3, 0.02], [-1.2, 0.8, 0.05]]
        pixels = pixels_from_world(K, view, points)
        depths = [5.10 - p[2] for p in points]
        assert world_from_pixels(K, view, pixels, depths) == pytest.approx(
            np.array(points), abs=1e-9)

    def test_180_yaw_mirrors_both_axes(self):
        K = intrinsics(RESOLUTION, FOCAL, H_AP)
        upright = view_matrix_ros(self._yaw(0))
        flipped = view_matrix_ros(self._yaw(180))
        point = [[0.4, 0.25, 0.02]]
        a = pixels_from_world(K, upright, point)[0]
        b = pixels_from_world(K, flipped, point)[0]
        # Mirrored about the principal point in both axes.
        assert b[0] - 960.0 == pytest.approx(-(a[0] - 960.0))
        assert b[1] - 540.0 == pytest.approx(-(a[1] - 540.0))


class TestBridgeImplementationMatches:
    """The functions above must stay byte-for-byte equivalent to the
    bridge's. Importing isaac_sim_bridge needs Isaac Sim, so compare the
    source text of the arithmetic instead.
    """

    def test_bridge_uses_the_same_r_u_transform(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent
               / "isaac_sim_bridge.py").read_text(encoding="utf-8")
        assert "R_U_TRANSFORM @ np.linalg.inv(camera_to_world)" in src
        assert "[0.0, -1.0, 0.0, 0.0]" in src
        assert "[0.0, 0.0, -1.0, 0.0]" in src

    def test_bridge_transposes_the_usd_matrix(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent
               / "isaac_sim_bridge.py").read_text(encoding="utf-8")
        # Dropping the .T silently transposes the rotation — the classic
        # USD row-vector-major mistake.
        assert "np.array(local_to_world, dtype=np.float64).T" in src

    def test_bridge_has_no_axis_sign_overrides(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent
               / "isaac_sim_bridge.py").read_text(encoding="utf-8")
        assert "image_x_sign" not in src
        assert "image_y_sign" not in src
