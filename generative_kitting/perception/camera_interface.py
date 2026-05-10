"""
Camera Interface for Isaac Sim.

Captures RGB images from the simulation viewport camera for VLM analysis.
Provides a mock interface for standalone testing outside Isaac Sim.
"""

import io
import os
from typing import Optional

import numpy as np
from PIL import Image

from utils.image_utils import numpy_to_pil, save_debug_image
from utils.logger import log


class CameraInterface:
    """
    Interface to the Isaac Sim viewport camera.

    Captures RGB frames from a USD camera prim using the
    ``omni.isaac.sensor`` extension.
    """

    def __init__(self, camera_prim_path: str = "/World/Camera", resolution: tuple = (1280, 720)):
        """
        Attach to an Isaac Sim camera prim.

        Parameters
        ----------
        camera_prim_path : str
            USD prim path of the camera in the scene.
        resolution : tuple
            (width, height) of the captured images.
        """
        self.camera_prim_path = camera_prim_path
        self.resolution = resolution
        self._camera = None

        try:
            from omni.isaac.sensor import Camera
            self._camera = Camera(
                prim_path=camera_prim_path,
                resolution=resolution,
            )
            self._camera.initialize()
            log.info(
                f"CameraInterface attached to {camera_prim_path} "
                f"at {resolution[0]}x{resolution[1]}"
            )
        except ImportError:
            log.warning(
                "omni.isaac.sensor not available — using MockCameraInterface. "
                "This is expected when running outside Isaac Sim."
            )
        except Exception as e:
            log.warning(f"Failed to initialise Isaac Sim camera: {e}")

    def capture_workspace_image(self) -> Image.Image:
        """
        Capture an RGB image from the Isaac Sim camera.

        Returns
        -------
        PIL.Image.Image
            RGB image of the workspace.

        Raises
        ------
        RuntimeError
            If the camera is not initialised.
        """
        if self._camera is None:
            raise RuntimeError(
                "Camera not initialised. Are you running inside Isaac Sim?"
            )

        # Get RGBA frame from Isaac Sim
        rgba = self._camera.get_rgba()
        if rgba is None:
            raise RuntimeError("Camera returned None — scene may not be playing")

        image = numpy_to_pil(rgba)
        log.debug(f"Captured {image.size[0]}x{image.size[1]} image from {self.camera_prim_path}")
        return image

    def capture_and_save(
        self,
        directory: str = "logs/vlm_images",
        prefix: str = "workspace",
    ) -> tuple:
        """
        Capture an image and save it to disk.

        Returns
        -------
        tuple of (PIL.Image.Image, str)
            The captured image and the path it was saved to.
        """
        image = self.capture_workspace_image()
        path = save_debug_image(image, directory=directory, prefix=prefix)
        log.info(f"Saved workspace image to {path}")
        return image, path

    @property
    def is_available(self) -> bool:
        """Check if the camera is initialised and ready."""
        return self._camera is not None


class MockCameraInterface:
    """
    Mock camera for standalone testing outside Isaac Sim.

    Loads images from disk instead of capturing from the simulator.
    """

    def __init__(
        self,
        image_path: Optional[str] = None,
        image_dir: str = "logs/vlm_images",
    ):
        """
        Parameters
        ----------
        image_path : str, optional
            Path to a single image to always return.
        image_dir : str
            Directory containing sample images. If ``image_path``
            is not set, the most recent image in this directory is used.
        """
        self.image_path = image_path
        self.image_dir = image_dir
        self.camera_prim_path = "/Mock/Camera"
        log.info(f"MockCameraInterface initialised (image_path={image_path})")

    def capture_workspace_image(self) -> Image.Image:
        """
        Return a mock workspace image loaded from disk.

        If no image path is configured, generates a placeholder
        gradient image.
        """
        if self.image_path and os.path.exists(self.image_path):
            image = Image.open(self.image_path).convert("RGB")
            log.debug(f"Mock camera loaded {self.image_path}")
            return image

        # Try to find the most recent image in the image directory
        if os.path.isdir(self.image_dir):
            files = sorted(
                [
                    os.path.join(self.image_dir, f)
                    for f in os.listdir(self.image_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ],
                key=os.path.getmtime,
                reverse=True,
            )
            if files:
                image = Image.open(files[0]).convert("RGB")
                log.debug(f"Mock camera loaded latest image: {files[0]}")
                return image

        # Generate a placeholder gradient image
        log.warning("No mock image available — generating placeholder")
        arr = np.zeros((720, 1280, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(40, 80, 720)[:, None]    # R gradient
        arr[:, :, 1] = np.linspace(40, 60, 1280)[None, :]   # G gradient
        arr[:, :, 2] = 50                                      # B constant
        return Image.fromarray(arr)

    def capture_and_save(
        self,
        directory: str = "logs/vlm_images",
        prefix: str = "mock_workspace",
    ) -> tuple:
        """Capture and save mock image."""
        image = self.capture_workspace_image()
        path = save_debug_image(image, directory=directory, prefix=prefix)
        return image, path

    @property
    def is_available(self) -> bool:
        return True


class BridgeCameraInterface:
    """
    Camera interface that connects to the Isaac Sim bridge server.

    Fetches live camera frames over HTTP from the bridge running
    inside Isaac Sim's Script Editor (``isaac_sim_bridge.py``).
    """

    def __init__(self, bridge_url: str = "http://localhost:8600"):
        """
        Parameters
        ----------
        bridge_url : str
            URL of the Isaac Sim bridge server.
        """
        self.bridge_url = bridge_url.rstrip("/")
        self.camera_prim_path = "/World/Camera"
        self._connected = False
        self._check_connection()

    def _check_connection(self):
        """Ping the bridge to see if it's available."""
        import urllib.request
        try:
            req = urllib.request.Request(f"{self.bridge_url}/api/ping", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = resp.read().decode()
                self._connected = True
                log.info(f"BridgeCameraInterface connected to {self.bridge_url}")
        except Exception:
            self._connected = False
            log.warning(
                f"Isaac Sim bridge not reachable at {self.bridge_url}. "
                f"Start isaac_sim_bridge.py in Isaac Sim's Script Editor."
            )

    def capture_workspace_image(self, cam_type: str = "rgb") -> Image.Image:
        """
        Fetch a live camera frame from the Isaac Sim bridge.

        Parameters
        ----------
        cam_type : str
            "rgb" for the overhead workspace camera,
            "depth" for the arm-mounted RealSense depth camera.

        Returns
        -------
        PIL.Image.Image
            Image from the specified simulation camera.
        """
        import urllib.request
        import base64
        import json

        try:
            # Use plain /api/camera for RGB, add ?type= for depth/wrist/kit
            if cam_type in ("depth", "wrist", "kit"):
                url = f"{self.bridge_url}/api/camera?type={cam_type}"
            else:
                url = f"{self.bridge_url}/api/camera"

            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            # If bridge returned "Unknown endpoint" (old bridge without query parsing)
            # fallback to plain /api/camera
            if "error" in data and "Unknown endpoint" in data.get("error", ""):
                log.debug(f"Retrying without query param: {data['error']}")
                url = f"{self.bridge_url}/api/camera"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())

            if "error" in data:
                log.warning(f"Bridge camera error: {data['error']}")
                return self._placeholder_image()

            b64 = data.get("image_base64", "")
            if not b64:
                return self._placeholder_image()

            img_bytes = base64.b64decode(b64)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            self._connected = True
            log.debug(f"Bridge {cam_type} frame: {image.size[0]}x{image.size[1]}")
            return image

        except Exception as e:
            self._connected = False
            log.warning(f"Bridge camera fetch failed: {e}")
            return self._placeholder_image()

    def capture_depth_image(self) -> Image.Image:
        """Fetch a depth camera frame from the arm-mounted RealSense."""
        return self.capture_workspace_image(cam_type="depth")

    def capture_wrist_image(self) -> Image.Image:
        """Fetch an RGB frame from the wrist-mounted RealSense."""
        return self.capture_workspace_image(cam_type="wrist")

    def capture_kit_image(self) -> Image.Image:
        """Fetch an RGB frame from the tray-overlook camera (/World/Camera_Kit).

        Used after the place phase to verify (via VLM) that the part
        actually landed in the kitting tray, independent of the gripper
        contact-sensor signal which can false-positive on empty closes.
        """
        return self.capture_workspace_image(cam_type="kit")

    def project_to_world(self, points: list, camera: str = "rgb",
                         method: str = None,
                         image_x_sign: int = 1,
                         image_y_sign: int = 1) -> dict:
        """Project normalised image coordinates to world XYZ via depth.

        Parameters
        ----------
        points : list of dict
            Each dict has ``x`` and ``y`` in [0, 1] (normalised image coords).
        camera : str
            Camera alias: "rgb", "depth", or "wrist".
        method : str, optional
            Force projection method: "geometric" skips the depth buffer
            and uses ray-plane intersection (useful for overhead landmark
            detection where depth buffer hits bin walls instead of floor).
        image_x_sign, image_y_sign : int
            Sign overrides for the image axes — set to ``-1`` to mirror
            an axis when the camera was placed in USD with a non-standard
            rotation. Default is +1 (OpenGL convention).

        Returns
        -------
        dict
            ``world_points`` list with ``x``, ``y``, ``z``, ``depth_m`` per point.
        """
        payload = {"points": points, "camera": camera,
                   "image_x_sign": int(image_x_sign),
                   "image_y_sign": int(image_y_sign)}
        if method:
            payload["method"] = method
        return self.send_command(
            "/api/project_to_world",
            payload,
            timeout=15,
        )

    def _placeholder_image(self) -> Image.Image:
        """Generate a placeholder when the bridge is unreachable."""
        arr = np.zeros((480, 640, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(30, 70, 480)[:, None]
        arr[:, :, 1] = np.linspace(30, 50, 640)[None, :]
        arr[:, :, 2] = 40
        return Image.fromarray(arr)

    def capture_and_save(
        self,
        directory: str = "logs/vlm_images",
        prefix: str = "bridge_workspace",
    ) -> tuple:
        """Capture and save bridge image."""
        image = self.capture_workspace_image()
        path = save_debug_image(image, directory=directory, prefix=prefix)
        return image, path

    def get_robot_status(self) -> dict:
        """Fetch robot status from the bridge."""
        import urllib.request
        import json

        try:
            req = urllib.request.Request(
                f"{self.bridge_url}/api/status", method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"status": "disconnected", "error": str(e)}

    def get_scene_parts(self) -> dict:
        """Fetch real-world coordinates of all pickable parts from USD stage."""
        import urllib.request
        import json

        try:
            req = urllib.request.Request(
                f"{self.bridge_url}/api/scene_parts", method="GET"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"parts": [], "error": str(e)}

    def get_prim_center(self, prim_path: str) -> dict:
        """Fetch USD bbox centre + top_z for a given prim path.

        Used by the wrist-scan workflow to look up the blue parts bin
        and the kitting tray without an overhead VLM landmark step.

        Returns ``{ok, center_xy, top_z, bot_z, height}`` on success
        or ``{ok: False, error: ...}`` if the bridge is unreachable
        or the prim doesn't exist.
        """
        import urllib.request
        import urllib.parse
        import json

        try:
            q = urllib.parse.urlencode({"prim": prim_path})
            url = f"{self.bridge_url}/api/prim_center?{q}"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_scene_annotations(self, camera: str = "rgb") -> dict:
        """Fetch GT per-part annotations as seen from the given camera.

        Each annotation contains world XYZ centre, world bbox, image bbox
        (px + normalised), centre pixel, projected depth, and visibility/
        occlusion flags. Coordinates are ground-truth from the USD stage
        — the workflow uses these to bypass VLM coordinate hallucination.

        Returns ``{"annotations": [], "error": ...}`` if the bridge isn't
        reachable or the endpoint isn't available (older bridge).
        """
        import urllib.request
        import json

        try:
            url = f"{self.bridge_url}/api/scene_annotations?camera={camera}"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"annotations": [], "error": str(e)}

    def execute_plan(self, plan: list) -> dict:
        """Send an action plan to Isaac Sim for execution."""
        import urllib.request
        import json

        try:
            payload = json.dumps({"plan": plan}).encode()
            req = urllib.request.Request(
                f"{self.bridge_url}/api/execute",
                data=payload,
                method="POST",
            )
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def send_home(self) -> dict:
        """Send the robot to home position."""
        return self.send_command("/api/home", {})

    def send_command(self, endpoint: str, payload: dict, timeout: int = 120) -> dict:
        """
        Send a generic POST command to the Isaac Sim bridge.

        Parameters
        ----------
        endpoint : str
            API path, e.g. "/api/pick", "/api/approach".
        payload : dict
            JSON body to send.
        timeout : int
            Request timeout in seconds.

        Returns
        -------
        dict
            Bridge response.
        """
        import urllib.request
        import json

        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.bridge_url}{endpoint}",
                data=data,
                method="POST",
            )
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            log.warning(f"Bridge command {endpoint} failed: {e}")
            return {"status": "error", "error": str(e)}

    def send_approach(self, params: dict) -> dict:
        """Move robot near a target XYZ position."""
        return self.send_command("/api/approach", params, timeout=30)

    def realign_gantry_hold_ee(self, target_x: float) -> dict:
        """Slide the gantry to ``target_x`` while keeping ee_link
        locked at its current world position.

        Used by the depth-verify-and-realign step so the wrist
        doesn't swing during a small post-depth XY refinement.
        Returns the bridge response containing ``ee_drift_m`` —
        useful for confirming the EE actually held still.
        """
        return self.send_command(
            "/api/realign_gantry", {"x": float(target_x)}, timeout=30)

    def execute_grasp(self, params: dict) -> dict:
        """Execute a full IK-based pick sequence."""
        return self.send_command("/api/pick", params, timeout=60)

    def execute_place(self, params: dict) -> dict:
        """Execute a full IK-based place sequence."""
        return self.send_command("/api/place", params, timeout=60)

    def check_connection(self) -> bool:
        """Re-ping the bridge and update the connection flag."""
        self._check_connection()
        return self._connected

    @property
    def is_available(self) -> bool:
        return self._connected

