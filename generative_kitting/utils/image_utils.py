"""
Image utility functions for the Generative Kitting System.

Handles encoding images for API calls (base64), converting between
numpy arrays and PIL Images, and saving debug snapshots.
"""

import base64
import io
import os
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """
    Convert a numpy array (H, W, C) from Isaac Sim camera to a PIL Image.

    Handles both uint8 [0,255] and float32 [0.0, 1.0] arrays.
    Strips alpha channel if present (RGBA → RGB).

    Parameters
    ----------
    arr : np.ndarray
        Image array of shape (H, W, 3) or (H, W, 4).

    Returns
    -------
    PIL.Image.Image
        RGB PIL Image.
    """
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)

    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]  # RGBA → RGB

    return Image.fromarray(arr, mode="RGB")


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a numpy array (H, W, 3) uint8.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image (any mode).

    Returns
    -------
    np.ndarray
        RGB array of shape (H, W, 3), dtype uint8.
    """
    return np.array(image.convert("RGB"))


def encode_image_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """
    Encode a PIL Image as a base64 string for API transmission.

    Parameters
    ----------
    image : PIL.Image.Image
        Image to encode.
    fmt : str
        Image format (PNG, JPEG, WEBP).

    Returns
    -------
    str
        Base64-encoded string of the image bytes.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_image_base64(b64_string: str) -> Image.Image:
    """
    Decode a base64 string back to a PIL Image.

    Parameters
    ----------
    b64_string : str
        Base64-encoded image data.

    Returns
    -------
    PIL.Image.Image
        Decoded PIL Image.
    """
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data))


def save_debug_image(
    image: Image.Image,
    directory: str = "logs/vlm_images",
    prefix: str = "capture",
    fmt: str = "png",
) -> str:
    """
    Save a timestamped debug image to disk.

    Parameters
    ----------
    image : PIL.Image.Image
        Image to save.
    directory : str
        Target directory.
    prefix : str
        Filename prefix.
    fmt : str
        Image format extension.

    Returns
    -------
    str
        Absolute path to the saved file.
    """
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.{fmt}"
    filepath = os.path.join(directory, filename)
    image.save(filepath)
    return os.path.abspath(filepath)


def load_image(path: str) -> Image.Image:
    """
    Load an image from disk as a PIL Image.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    PIL.Image.Image
        Loaded RGB image.
    """
    return Image.open(path).convert("RGB")


def resize_for_vlm(
    image: Image.Image,
    max_size: int = 1024,
) -> Image.Image:
    """
    Resize an image so its longest edge is at most `max_size` pixels.
    Preserves aspect ratio.  Useful for keeping VLM API costs down.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.
    max_size : int
        Maximum pixel dimension for the longest edge.

    Returns
    -------
    PIL.Image.Image
        Resized image (or original if already small enough).
    """
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)
