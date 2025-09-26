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
    orientation_deg: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "a_length": float(self.a_length),
            "b_length": float(self.b_length),
            "angle_deg": float(self.angle_deg),
            "orientation_deg": float(self.orientation_deg % 360.0),
        }

    def wiggle(self, percent: float) -> "LatticeParameters":
        scale = 1.0 + percent / 100.0
        return LatticeParameters(
            a_length=self.a_length * scale,
            b_length=self.b_length * scale,
            angle_deg=self.angle_deg,
            orientation_deg=self.orientation_deg,
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
            for g_a, g_b in [
                (g1, g2),
                (g2, g1),
            ]:
                det = g_a[0] * g_b[1] - g_a[1] * g_b[0]
                if abs(det) < 1e-8:
                    continue

                reciprocal = np.stack([g_a, g_b]).astype(np.float64)
                direct_basis = np.linalg.inv(reciprocal)
                direct_a = direct_basis[:, 0]
                direct_b = direct_basis[:, 1]

                len_a = np.linalg.norm(direct_a)
                len_b = np.linalg.norm(direct_b)

                if not (
                    expected_a * (1 - tolerance)
                    <= len_a
                    <= expected_a * (1 + tolerance)
                ):
                    continue
                if not (
                    expected_b * (1 - tolerance)
                    <= len_b
                    <= expected_b * (1 + tolerance)
                ):
                    continue

                cos_angle = np.clip(
                    np.dot(direct_a, direct_b) / (len_a * len_b + 1e-12),
                    -1.0,
                    1.0,
                )
                direct_angle = np.arccos(cos_angle)

                score = (
                    abs(len_a - expected_a)
                    + abs(len_b - expected_b)
                    + abs(direct_angle - expected_angle)
                )
                if score < best_score:
                    best_score = score
                    orientation = np.degrees(np.arctan2(direct_a[1], direct_a[0])) % 360.0
                    best_params = LatticeParameters(
                        a_length=len_a * pixel_size_nm * 10.0,
                        b_length=len_b * pixel_size_nm * 10.0,
                        angle_deg=np.degrees(direct_angle),
                        orientation_deg=orientation,
                    )

    return best_params


def consolidate_lattice(
    parameters: Iterable[LatticeParameters], threshold: float = 2.5
) -> LatticeParameters:
    """Consolidate per-frame lattice estimates into a robust set of parameters."""
    a_values = np.array([p.a_length for p in parameters])
    b_values = np.array([p.b_length for p in parameters])
    angles = np.array([p.angle_deg for p in parameters])
    orientations = np.array([p.orientation_deg for p in parameters])

    def robust_mean(values: NDArray[np.float64]) -> float:
        median = np.median(values)
        mad = np.median(np.abs(values - median)) + 1e-6
        mask = np.abs(values - median) <= threshold * mad
        if np.any(mask):
            return float(np.mean(values[mask]))
        return float(median)

    def robust_angle(values: NDArray[np.float64]) -> float:
        if not len(values):
            return 0.0
        median = np.median(values)
        wrapped = ((values - median + 180.0) % 360.0) - 180.0
        mad = np.median(np.abs(wrapped)) + 1e-6
        if np.any(np.abs(wrapped) <= threshold * mad):
            mask = np.abs(wrapped) <= threshold * mad
            selected = np.deg2rad(values[mask])
        else:
            selected = np.deg2rad(values)
        sin_mean = np.mean(np.sin(selected))
        cos_mean = np.mean(np.cos(selected))
        angle = np.degrees(np.arctan2(sin_mean, cos_mean))
        return float((angle + 360.0) % 360.0)

    return LatticeParameters(
        a_length=robust_mean(a_values),
        b_length=robust_mean(b_values),
        angle_deg=robust_mean(angles),
        orientation_deg=robust_angle(orientations),
    )


def lattice_vectors_in_pixels(
    params: LatticeParameters, pixel_size_nm: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Return lattice basis vectors in pixel units."""
    a_len_pixels = params.a_length * 0.1 / pixel_size_nm
    b_len_pixels = params.b_length * 0.1 / pixel_size_nm
    angle_rad = np.deg2rad(params.angle_deg)

    orientation_rad = np.deg2rad(params.orientation_deg)

    a_vec = np.array(
        [
            a_len_pixels * np.cos(orientation_rad),
            a_len_pixels * np.sin(orientation_rad),
        ],
        dtype=np.float32,
    )
    b_vec = np.array(
        [
            b_len_pixels * np.cos(orientation_rad + angle_rad),
            b_len_pixels * np.sin(orientation_rad + angle_rad),
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
