"""Lattice estimation and geometry helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.feature import peak_local_max


BASE_OFFSET = np.array([0.5, 0.5], dtype=np.float32)


@dataclass
class LatticeParameters:
    """Represents the metric of a 2D lattice in direct space."""

    a_length: float  # Å
    b_length: float  # Å
    angle_deg: float  # between a and b
    orientation_deg: float  # orientation of a relative to +x

    def as_dict(self) -> dict[str, float]:
        return {
            "a_length": float(self.a_length),
            "b_length": float(self.b_length),
            "angle_deg": float(self.angle_deg),
            "orientation_deg": float(self.orientation_deg % 360.0),
        }


@dataclass
class FrameLatticeFit:
    """Stores the lattice estimate and translation for a single frame."""

    parameters: LatticeParameters
    a_vec: NDArray[np.float32]
    b_vec: NDArray[np.float32]
    translation: NDArray[np.float32]
    phases: NDArray[np.float32]
    quality: float


def enhance_frame(frame: NDArray[np.float32], sigma_lo: float = 0.8, sigma_hi: float = 3.0) -> NDArray[np.float32]:
    """Difference-of-Gaussians enhancement used prior to FFT analysis."""

    low = ndimage.gaussian_filter(frame, sigma_lo)
    high = ndimage.gaussian_filter(frame, sigma_hi)
    dog = np.clip(low - high, 0, None)
    if dog.max() > 0:
        dog /= dog.max()
    return dog.astype(np.float32)


def estimate_lattice_from_frame(
    frame: NDArray[np.float32],
    approx_params: LatticeParameters,
    pixel_size_nm: float,
    tolerance: float = 0.35,
    search_steps: int = 18,
) -> FrameLatticeFit | None:
    """Estimate lattice basis and translation for a single frame.

    Parameters
    ----------
    frame:
        Raw grayscale frame.
    approx_params:
        Approximate lattice parameters supplied by the user or locked from
        the consolidation step.
    pixel_size_nm:
        Pixel size in nanometres.
    tolerance:
        Relative tolerance on the lattice lengths (fraction).  Angle
        tolerance is derived from this window.
    search_steps:
        Number of discrete phase samples in each direction when refining the
        translation.  Higher values give finer translations but increase the
        computational cost.
    """

    enhanced = enhance_frame(frame)
    peaks = _fft_peaks(enhanced)
    if peaks.size == 0:
        return None

    freq_vectors = _frequency_vectors(peaks, frame.shape)
    freq_vectors = freq_vectors[np.linalg.norm(freq_vectors, axis=1) > 0]
    if not len(freq_vectors):
        return None

    expected_a = approx_params.a_length * 0.1 / pixel_size_nm
    expected_b = approx_params.b_length * 0.1 / pixel_size_nm
    expected_angle = np.deg2rad(approx_params.angle_deg)

    angle_tolerance = max(np.deg2rad(2.0), tolerance * expected_angle)

    best_score = -np.inf
    best_basis: tuple[np.ndarray, np.ndarray] | None = None
    best_params: LatticeParameters | None = None

    for i in range(len(freq_vectors)):
        for j in range(i + 1, len(freq_vectors)):
            for g_a, g_b in ((freq_vectors[i], freq_vectors[j]), (freq_vectors[j], freq_vectors[i])):
                reciprocal = np.stack([g_a, g_b]).astype(np.float64)
                if abs(np.linalg.det(reciprocal)) < 1e-10:
                    continue
                direct_basis = np.linalg.inv(reciprocal)
                direct_a = direct_basis[:, 0]
                direct_b = direct_basis[:, 1]

                len_a = np.linalg.norm(direct_a)
                len_b = np.linalg.norm(direct_b)

                if not (
                    expected_a * (1 - tolerance) <= len_a <= expected_a * (1 + tolerance)
                ):
                    continue
                if not (
                    expected_b * (1 - tolerance) <= len_b <= expected_b * (1 + tolerance)
                ):
                    continue

                cos_angle = np.clip(
                    np.dot(direct_a, direct_b) / (len_a * len_b + 1e-12),
                    -1.0,
                    1.0,
                )
                angle = np.arccos(cos_angle)
                if abs(angle - expected_angle) > angle_tolerance:
                    continue

                orientation = np.degrees(np.arctan2(direct_a[1], direct_a[0])) % 360.0
                quality = _pair_quality(enhanced, g_a, g_b)
                if quality > best_score:
                    best_score = quality
                    best_basis = (direct_a.astype(np.float64), direct_b.astype(np.float64))
                    best_params = LatticeParameters(
                        a_length=float(len_a * pixel_size_nm * 10.0),
                        b_length=float(len_b * pixel_size_nm * 10.0),
                        angle_deg=float(np.degrees(angle)),
                        orientation_deg=float(orientation),
                    )

    if best_basis is None or best_params is None:
        return None

    a_vec = best_basis[0].astype(np.float32)
    b_vec = best_basis[1].astype(np.float32)
    translation, phases = refine_translation(enhanced, a_vec, b_vec, search_steps)

    return FrameLatticeFit(
        parameters=best_params,
        a_vec=a_vec,
        b_vec=b_vec,
        translation=translation,
        phases=phases,
        quality=float(best_score),
    )


def consolidate_lattice(fits: Iterable[FrameLatticeFit], threshold: float = 2.5) -> LatticeParameters:
    """Consolidate per-frame fits into a robust lattice parameter set."""

    params = [fit.parameters for fit in fits]
    if not params:
        raise ValueError("No lattice fits supplied for consolidation")

    a_values = np.array([p.a_length for p in params], dtype=np.float64)
    b_values = np.array([p.b_length for p in params], dtype=np.float64)
    angles = np.array([p.angle_deg for p in params], dtype=np.float64)
    orientations = np.array([p.orientation_deg for p in params], dtype=np.float64)

    def robust_mean(values: NDArray[np.float64]) -> float:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)) + 1e-6)
        mask = np.abs(values - median) <= threshold * mad
        if np.any(mask):
            return float(np.mean(values[mask]))
        return median

    def robust_angle(values: NDArray[np.float64]) -> float:
        if not len(values):
            return 0.0
        median = np.median(values)
        wrapped = ((values - median + 180.0) % 360.0) - 180.0
        mad = np.median(np.abs(wrapped)) + 1e-6
        mask = np.abs(wrapped) <= threshold * mad
        selected = np.deg2rad(values[mask]) if np.any(mask) else np.deg2rad(values)
        sin_mean = np.mean(np.sin(selected))
        cos_mean = np.mean(np.cos(selected))
        return float((np.degrees(np.arctan2(sin_mean, cos_mean)) + 360.0) % 360.0)

    return LatticeParameters(
        a_length=robust_mean(a_values),
        b_length=robust_mean(b_values),
        angle_deg=robust_mean(angles),
        orientation_deg=robust_angle(orientations),
    )


def lattice_vectors_in_pixels(
    params: LatticeParameters, pixel_size_nm: float
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
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
    enhanced_frame: NDArray[np.float32],
    a_vec: NDArray[np.float32],
    b_vec: NDArray[np.float32],
    search_steps: int = 18,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Search for the best translation using a discrete phase grid."""

    offsets = np.linspace(0.0, 1.0, num=search_steps, endpoint=False)
    sample_offsets = _sample_offsets(a_vec, b_vec)
    height, width = enhanced_frame.shape

    best_score = -np.inf
    best_phi = np.zeros(2, dtype=np.float32)

    A = np.column_stack([a_vec, b_vec])

    for phi_a in offsets:
        for phi_b in offsets:
            coeffs = sample_offsets + np.array([phi_a, phi_b])
            points = BASE_OFFSET + coeffs @ A.T
            xs = np.clip(points[:, 0], 0, width - 1)
            ys = np.clip(points[:, 1], 0, height - 1)
            score = float(np.sum(enhanced_frame[ys.astype(int), xs.astype(int)]))
            if score > best_score:
                best_score = score
                best_phi = np.array([phi_a, phi_b], dtype=np.float32)

    translation = BASE_OFFSET + A @ best_phi
    return translation.astype(np.float32), best_phi


def cover_index_window(
    frame_shape: Tuple[int, int],
    base: NDArray[np.float32],
    a_vec: NDArray[np.float32],
    b_vec: NDArray[np.float32],
    margin: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return inclusive index ranges that cover the frame for the lattice."""

    height, width = frame_shape
    A = np.column_stack([a_vec, b_vec]).astype(np.float64)
    if np.linalg.matrix_rank(A) < 2:
        return np.arange(-1, 2, dtype=int), np.arange(-1, 2, dtype=int)

    corners = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [0.0, height - 1.0],
            [width - 1.0, height - 1.0],
        ]
    ).T
    rel = np.linalg.solve(A, corners - base[:, None])
    n_min = int(np.floor(np.min(rel[0]))) - margin
    n_max = int(np.ceil(np.max(rel[0]))) + margin
    m_min = int(np.floor(np.min(rel[1]))) - margin
    m_max = int(np.ceil(np.max(rel[1]))) + margin
    return np.arange(n_min, n_max + 1, dtype=int), np.arange(m_min, m_max + 1, dtype=int)


def lattice_points_from_indices(
    indices: NDArray[np.int32],
    phases: Sequence[float],
    a_vec: NDArray[np.float32],
    b_vec: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute lattice points for the given integer indices and phases."""

    coeffs = indices.astype(np.float32) + np.asarray(phases, dtype=np.float32)
    A = np.column_stack([a_vec, b_vec])
    return BASE_OFFSET + coeffs @ A.T


def _fft_peaks(frame: NDArray[np.float32], num_peaks: int = 40) -> NDArray[np.int_]:
    shifted = np.fft.fftshift(np.fft.fft2(frame - np.mean(frame)))
    power = np.abs(shifted)
    power = ndimage.gaussian_filter(power, sigma=2.0)
    peaks = peak_local_max(power, num_peaks=num_peaks, exclude_border=False)
    return peaks


def _frequency_vectors(peaks: NDArray[np.int_], shape: Tuple[int, int]) -> NDArray[np.float32]:
    center = (np.array(shape[::-1]) - 1) / 2.0
    vectors = peaks[:, ::-1] - center  # convert to (x, y)
    freq = vectors / np.array(shape[::-1], dtype=np.float64)
    return freq.astype(np.float32)


def _pair_quality(frame: NDArray[np.float32], g_a: NDArray[np.float32], g_b: NDArray[np.float32]) -> float:
    """Score a reciprocal pair by sampling their Fourier amplitudes."""

    height, width = frame.shape
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    coords = np.array([g_a, -g_a, g_b, -g_b])
    xs = np.clip(np.round(coords[:, 0] * width + cx).astype(int), 0, width - 1)
    ys = np.clip(np.round(coords[:, 1] * height + cy).astype(int), 0, height - 1)
    return float(np.sum(frame[ys, xs]))


def _sample_offsets(
    a_vec: NDArray[np.float32], b_vec: NDArray[np.float32], extent: int = 2
) -> NDArray[np.float32]:
    offsets: List[Tuple[float, float]] = []
    for n in range(-extent, extent + 1):
        for m in range(-extent, extent + 1):
            offsets.append((n, m))
    return np.array(offsets, dtype=np.float32)
