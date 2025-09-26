"""Visualization helpers for adsorbate tracking."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from .adsorbates import AdsorbateDetection


def draw_lattice_overlay(
    frame_shape: Tuple[int, int],
    detection: AdsorbateDetection,
    atom_diameter_pixels: float,
) -> NDArray[np.uint8]:
    """Return an RGB image with an artificial lattice overlay on a blank canvas."""

    height, width = frame_shape
    base = np.zeros((height, width, 3), dtype=np.uint8)
    radius = max(int(round(atom_diameter_pixels / 2.0)), 1)
    for point, present in zip(detection.points, detection.present):
        color = (160, 160, 160) if not present else (0, 255, 255)
        cv2.circle(base, (int(round(point[0])), int(round(point[1]))), radius, color, -1)
    return base


def combine_frames(original_bgr: NDArray[np.uint8], overlay: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Create a side-by-side comparison frame."""
    return np.concatenate([original_bgr, overlay], axis=1)


def write_video(
    output_path: Path,
    frames: Iterable[NDArray[np.uint8]],
    fps: float,
) -> None:
    """Write frames to an mp4 video."""
    frames = list(frames)
    if not frames:
        raise ValueError("No frames to write")
    height, width, _ = frames[0].shape
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()
