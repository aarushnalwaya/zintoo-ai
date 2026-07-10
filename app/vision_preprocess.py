"""
Image preprocessing — the ONE definition shared by training and serving.

Train/serve skew is the classic silent killer in vision deployments: the model
scores 92% offline and 60% in production because the server resized with a
different interpolation or forgot to convert BGR->RGB. To make that class of
bug impossible here:

  1. This module is the single source of truth for decode -> resize -> crop.
  2. Mean/std normalisation is **baked into the exported ONNX graph** (a Sub/Div
     pair at the front), so the server never applies it and cannot get it wrong.
     `ml/export_onnx.py` inserts it; `ml/train.py` asserts the same constants.

Contract (must not drift):
  * RGB, 8-bit
  * Resize shorter side to RESIZE_SHORT (bilinear, antialias)
  * Center-crop CROP_SIZE x CROP_SIZE
  * float32, scaled to [0, 1]
  * CHW layout, batched to NCHW

Anything downstream of that (mean/std) lives inside the graph.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps

# ─── The contract. Changing any of these invalidates an exported model. ───
RESIZE_SHORT = 256
CROP_SIZE = 224
# ImageNet statistics — baked into the ONNX graph, kept here so the training
# script and the export script provably agree on the same numbers.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB — reject before decoding
MAX_PIXELS = 40_000_000             # guard against decompression bombs


class ImageDecodeError(ValueError):
    """Raised for anything the client got wrong (bad bytes, too large, etc)."""


def decode_image(data: bytes) -> Image.Image:
    """Safely decode untrusted bytes into an RGB PIL image."""
    if not data:
        raise ImageDecodeError("Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageDecodeError(
            f"Image too large ({len(data) / 1e6:.1f} MB); limit is {MAX_UPLOAD_BYTES // 1024 // 1024} MB"
        )
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()               # cheap structural check, consumes the file
        img = Image.open(io.BytesIO(data))  # must reopen after verify()
    except ImageDecodeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ImageDecodeError(f"Unreadable image: {exc}") from exc

    w, h = img.size
    if w * h > MAX_PIXELS:
        raise ImageDecodeError("Image resolution too large")
    if w < 8 or h < 8:
        raise ImageDecodeError("Image too small")

    # Honour EXIF rotation (phone uploads), then force 3-channel RGB.
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def resize_and_crop(img: Image.Image) -> Image.Image:
    """Resize shorter side to RESIZE_SHORT, then center-crop CROP_SIZE."""
    w, h = img.size
    scale = RESIZE_SHORT / min(w, h)
    new_w, new_h = max(CROP_SIZE, round(w * scale)), max(CROP_SIZE, round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - CROP_SIZE) // 2
    top = (new_h - CROP_SIZE) // 2
    return img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))


def to_tensor(img: Image.Image) -> np.ndarray:
    """PIL RGB -> float32 NCHW in [0, 1], shape (1, 3, CROP_SIZE, CROP_SIZE)."""
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ImageDecodeError("Expected a 3-channel RGB image")
    arr = arr.astype(np.float32) / 255.0     # HWC [0,1]
    arr = np.transpose(arr, (2, 0, 1))       # CHW
    return np.expand_dims(arr, 0)            # NCHW


def preprocess(data: bytes) -> np.ndarray:
    """bytes -> model-ready NCHW float32 tensor. Raises ImageDecodeError."""
    return to_tensor(resize_and_crop(decode_image(data)))


def normalize_reference(x: np.ndarray) -> np.ndarray:
    """Reference mean/std normalisation.

    The server does NOT call this — it is baked into the ONNX graph. It exists
    so tests can prove the graph's built-in normalisation matches this exactly.
    """
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return (x - mean) / std
