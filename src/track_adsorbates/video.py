"""Video loading utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


@dataclass
class VideoData:
    """Container for video frames and metadata."""

    path: Path
    frames: List[np.ndarray]
    fps: float

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def frame_shape(self) -> Tuple[int, int]:
        if not self.frames:
            raise ValueError("Video has no frames")
        frame = self.frames[0]
        return frame.shape[:2]


def load_video(path: str | Path, max_frames: int | None = None) -> VideoData:
    """Load an mp4 video as grayscale frames.

    Parameters
    ----------
    path:
        Path to the mp4 file.
    max_frames:
        Optional limit to the number of frames to load.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    frames: List[np.ndarray] = []
    success = True
    while success:
        success, frame = capture.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32))
        if max_frames is not None and len(frames) >= max_frames:
            break

    capture.release()
    if not frames:
        raise ValueError("Video does not contain any frames")

    return VideoData(path=Path(path), frames=frames, fps=fps)


def iter_frames(video: VideoData) -> Iterable[np.ndarray]:
    """Yield frames from the video data."""
    for frame in video.frames:
        yield frame
