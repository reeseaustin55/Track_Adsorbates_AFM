"""Lattice estimation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.feature import peak_local_max


@dataclass
class LatticeParameters:
    """Represents a 2D lattice basis."""

    a_length: float
    b_length: float
    angle_deg: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "a_length": float(self.a_length),
            "b_length": float(self.b_length),
            "angle_deg": float(self.angle_deg),
        }

    def wiggle(self, percent: float) -> "LatticeParameters":
        scale = 1.0 + percent / 100.0
        return LatticeParameters(
            a_length=self.a_length * scale,
            b_length=self.b_length * scale,
            angle_deg=self.angle_deg,
        )


@dataclass
class FrameLatticeFit:
    """Stores lattice fit for a frame."""

    parameters: LatticeParameters
    translation: Tuple[float, float]
    quality: float


def _fft_peaks(frame: NDArray[np.float32], num_peaks: int = 30) -> NDArray[np.int_]:
    shifted = np.fft.fftshift(np.fft.fft2(frame - np.mean(frame)))
    power = np.abs(shifted)
    power = ndimage.gaussian_filter(power, sigma=3)
    peaks = peak_local_max(power, num_peaks=num_peaks, exclude_border=False)
    return peaks


def _frequency_vectors(peaks: NDArray[np.int_], shape: Tuple[int, int]) -> NDArray[np.float32]:
    center = np.array(shape) / 2.0
    vectors = peaks - center
    vectors = np.fliplr(vectors)  # convert to (x, y)
    freq = vectors / np.array(shape[::-1])  # cycles per pixel
    return freq.astype(np.float32)


def _direct_length(freq_vec: NDArray[np.float32]) -> float:
    magnitude = np.linalg.norm(freq_vec)
    if magnitude == 0:
        return np.inf
    return 1.0 / magnitude


def estimate_lattice_from_frame(
    frame: NDArray[np.float32],
    approx_params: LatticeParameters,
    pixel_size_nm: float,
    tolerance: float = 0.35,
) -> LatticeParameters | None:
    """Estimate lattice parameters for a single frame using FFT peak picking."""

    peaks = _fft_peaks(frame)
    freq_vectors = _frequency_vectors(peaks, frame.shape)
    # Filter out zero frequency
    freq_vectors = freq_vectors[np.linalg.norm(freq_vectors, axis=1) > 0]

    # Convert expected lengths to pixels
    expected_a = approx_params.a_length * 0.1 / pixel_size_nm
    expected_b = approx_params.b_length * 0.1 / pixel_size_nm
    expected_angle = np.deg2rad(approx_params.angle_deg)

    best_score = np.inf
    best_params = None

    for i in range(len(freq_vectors)):
        for j in range(i + 1, len(freq_vectors)):
            g1 = freq_vectors[i]
            g2 = freq_vectors[j]
            length1 = _direct_length(g1)
            length2 = _direct_length(g2)
            if not (expected_a * (1 - tolerance) <= length1 <= expected_a * (1 + tolerance)):
                continue
            if not (expected_b * (1 - tolerance) <= length2 <= expected_b * (1 + tolerance)):
                continue
            angle = np.arccos(
                np.clip(
                    np.dot(g1, g2)
                    / (np.linalg.norm(g1) * np.linalg.norm(g2)),
                    -1.0,
                    1.0,
                )
            )
            direct_angle = np.pi - angle
            score = (
                abs(length1 - expected_a)
                + abs(length2 - expected_b)
                + abs(direct_angle - expected_angle)
            )
            if score < best_score:
                best_score = score
                best_params = LatticeParameters(
                    a_length=length1 * pixel_size_nm / 0.1,
                    b_length=length2 * pixel_size_nm / 0.1,
                    angle_deg=np.degrees(direct_angle),
                )

    return best_params


def consolidate_lattice(parameters: Iterable[LatticeParameters], threshold: float = 2.5) -> LatticeParameters:
    """Consolidate per-frame lattice estimates into a robust set of parameters."""
    a_values = np.array([p.a_length for p in parameters])
    b_values = np.array([p.b_length for p in parameters])
    angles = np.array([p.angle_deg for p in parameters])

    def robust_mean(values: NDArray[np.float64]) -> float:
        median = np.median(values)
        mad = np.median(np.abs(values - median)) + 1e-6
        mask = np.abs(values - median) <= threshold * mad
        if np.any(mask):
            return float(np.mean(values[mask]))
        return float(median)

    return LatticeParameters(
        a_length=robust_mean(a_values),
        b_length=robust_mean(b_values),
        angle_deg=robust_mean(angles),
    )


def lattice_vectors_in_pixels(params: LatticeParameters, pixel_size_nm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return lattice basis vectors in pixel units."""
    a_len_pixels = params.a_length * 0.1 / pixel_size_nm
    b_len_pixels = params.b_length * 0.1 / pixel_size_nm
    angle_rad = np.deg2rad(params.angle_deg)

    a_vec = np.array([a_len_pixels, 0.0], dtype=np.float32)
    b_vec = np.array(
        [
            b_len_pixels * np.cos(angle_rad),
            b_len_pixels * np.sin(angle_rad),
        ],
        dtype=np.float32,
    )
    return a_vec, b_vec


def refine_translation(
    frame: NDArray[np.float32],
    a_vec: NDArray[np.float32],
    b_vec: NDArray[np.float32],
) -> Tuple[float, float]:
    """Estimate the best translation for the lattice within the frame."""
    height, width = frame.shape
    coords = []
    # sample grid over two unit cells
    for i in range(-1, 2):
        for j in range(-1, 2):
            coords.append(i * a_vec + j * b_vec)
    coords = np.array(coords)
    xs = coords[:, 0]
    ys = coords[:, 1]
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)
    intensities = frame[ys.astype(int), xs.astype(int)]
    # Use centroid of bright points as translation
    weights = intensities / (np.sum(intensities) + 1e-6)
    tx = float(np.sum(xs * weights))
    ty = float(np.sum(ys * weights))
    return tx, ty


def build_lattice_points(
    frame_shape: Tuple[int, int],
    a_vec: NDArray[np.float32],
    b_vec: NDArray[np.float32],
    translation: Tuple[float, float],
    padding: float = 2.0,
) -> NDArray[np.float32]:
    """Generate lattice points that lie within the frame."""
    height, width = frame_shape
    tx, ty = translation
    points: List[np.ndarray] = []
    max_i = int(np.ceil((width + padding) / max(np.linalg.norm(a_vec), 1e-3))) + 2
    max_j = int(np.ceil((height + padding) / max(np.linalg.norm(b_vec), 1e-3))) + 2
    for i in range(-max_i, max_i + 1):
        for j in range(-max_j, max_j + 1):
            point = tx + i * a_vec[0] + j * b_vec[0], ty + i * a_vec[1] + j * b_vec[1]
            if -padding <= point[0] < width + padding and -padding <= point[1] < height + padding:
                points.append(point)
    return np.array(points, dtype=np.float32)
