"""
Monocular Depth Estimation for the Generative Kitting System.

Provides depth maps from single RGB images to convert VLM 2D detections
into approximate 3D world coordinates.  This bridges the gap between
VLM perception (2D image) and robot IK (3D world).

Methods:
  1. **Depth Anything V2** — state-of-the-art monocular depth via
     HuggingFace Transformers.  Gives relative depth from a single RGB
     image, scaled to metric using known camera height.
  2. **Geometric** — assumes flat workspace at a known Z.  Uses camera
     intrinsics to project image coords to the workspace plane.
     No ML model needed.

Usage:
    estimator = DepthEstimator(config)
    world_coords = estimator.image_to_world(rgb_image, vlm_points)
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from utils.logger import log


class DepthEstimator:
    """Monocular depth estimation from RGB images.

    Converts VLM normalised image coordinates (0-1) into approximate
    world coordinates (metres) using either a learned depth model or
    known camera geometry.
    """

    def __init__(self, config: dict = None):
        """
        Parameters
        ----------
        config : dict
            ``perception`` section of config.yaml, plus optional keys:
            - ``depth_method``: ``"depth_anything"`` or ``"geometric"``
            - ``camera_height_m``: camera height above workspace (metres)
            - ``camera_fov_h_deg``: horizontal field of view (degrees)
            - ``camera_resolution``: [width, height]
        """
        cfg = config or {}
        self.method = cfg.get("depth_method", "geometric")

        # Camera parameters (from config or Isaac Sim defaults)
        self.cam_height = cfg.get("camera_height_m", 1.8)
        self.cam_fov_h = cfg.get("camera_fov_h_deg", 60.0)
        self.cam_resolution = cfg.get("camera_resolution", [1920, 1080])
        self.workspace_z = cfg.get("workspace_surface_z", 0.02)

        # Camera position in world (top-down camera centre XY)
        self.cam_world_x = cfg.get("camera_world_x", 0.3)
        self.cam_world_y = cfg.get("camera_world_y", 0.0)

        # Depth model (lazy-loaded)
        self._model = None

        log.info(
            f"DepthEstimator init: method={self.method}, "
            f"cam_height={self.cam_height}m, fov={self.cam_fov_h}°"
        )

    # ─── Public API ─────────────────────────────────────────

    def estimate_depth_map(self, image: Image.Image) -> np.ndarray:
        """Estimate per-pixel depth from an RGB image.

        Returns
        -------
        np.ndarray
            Depth map (H, W) in approximate metres.
        """
        if self.method == "depth_anything":
            return self._depth_anything(image)
        return self._geometric_depth(image)

    def image_to_world(
        self,
        image: Image.Image,
        points: List[Dict[str, float]],
        workspace_z: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Convert normalised image coordinates to world XYZ.

        Combines the depth map (or geometric model) with known camera
        parameters to produce approximate 3D world coordinates.

        Parameters
        ----------
        image : PIL.Image.Image
            The RGB image the VLM analysed.
        points : list of dict
            Each dict has ``x`` and ``y`` in [0, 1] (normalised image).
        workspace_z : float, optional
            Override workspace surface Z.  Defaults to config value.

        Returns
        -------
        list of dict
            ``{x, y, z, depth_m, method}`` per point in world metres.
        """
        wz = workspace_z if workspace_z is not None else self.workspace_z
        w, h = image.size

        # Camera intrinsics from FOV
        fx = (w / 2.0) / math.tan(math.radians(self.cam_fov_h) / 2.0)
        fy = fx  # square pixels
        cx, cy = w / 2.0, h / 2.0

        # Get depth map
        depth_map = self.estimate_depth_map(image)

        world_points = []
        for pt in points:
            nx = float(pt.get("x", 0.5))
            ny = float(pt.get("y", 0.5))
            px = min(int(nx * (w - 1)), w - 1)
            py = min(int(ny * (h - 1)), h - 1)

            if self.method == "depth_anything" and self._model is not None:
                # Sample depth from the estimated depth map
                r = 3
                y0, y1 = max(0, py - r), min(h, py + r + 1)
                x0, x1 = max(0, px - r), min(w, px + r + 1)
                patch = depth_map[y0:y1, x0:x1]
                valid = patch[patch > 0.01]
                d = float(np.median(valid)) if valid.size > 0 else self.cam_height
            else:
                d = self.cam_height

            # Project pixel to world XY using pinhole model
            # Assumes camera is looking straight down at the workspace
            world_x = self.cam_world_x + (px - cx) * d / fx
            world_y = self.cam_world_y - (py - cy) * d / fy

            world_points.append({
                "x": float(world_x),
                "y": float(world_y),
                "z": float(wz),
                "depth_m": float(d),
                "method": f"depth_est_{self.method}",
            })

            log.debug(
                f"DepthEst: image({nx:.2f},{ny:.2f}) → "
                f"world({world_x:.3f}, {world_y:.3f}, {wz:.3f}) "
                f"[d={d:.3f}m, {self.method}]"
            )

        return world_points

    # ─── Depth Anything V2 ──────────────────────────────────

    def _depth_anything(self, image: Image.Image) -> np.ndarray:
        """Estimate depth using Depth Anything V2 (HuggingFace)."""
        try:
            if self._model is None:
                from transformers import pipeline

                self._model = pipeline(
                    "depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Small-hf",
                    device="cpu",
                )
                log.info("Depth Anything V2 (Small) model loaded")

            result = self._model(image)
            depth = np.array(result["depth"], dtype=np.float32)

            # Scale relative depth to metric using known camera height
            if depth.max() > 0:
                depth = depth / depth.max() * self.cam_height

            return depth

        except ImportError:
            log.warning(
                "transformers not available for Depth Anything V2 — "
                "install with: pip install transformers torch. "
                "Falling back to geometric depth."
            )
            self.method = "geometric"
            return self._geometric_depth(image)

        except Exception as e:
            log.warning(f"Depth Anything V2 failed: {e} — falling back to geometric")
            return self._geometric_depth(image)

    # ─── Geometric (flat plane) ─────────────────────────────

    def _geometric_depth(self, image: Image.Image) -> np.ndarray:
        """Constant depth map assuming flat workspace at known camera height."""
        w, h = image.size
        return np.full((h, w), self.cam_height, dtype=np.float32)

    # ─── Properties ─────────────────────────────────────────

    @property
    def is_ml_available(self) -> bool:
        """Check if ML-based depth estimation is available."""
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False
