"""Shared helper utilities for adsorbate tracking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter, maximum_filter


Array2D = NDArray[np.float64]
BoolArray2D = NDArray[np.bool_]


@dataclass
class NeighborShift:
    dn: int
    dm: int


def nm_cover_image(
    size_hw: Tuple[int, int],
    base: NDArray[np.float64],
    a1: NDArray[np.float64],
    a2: NDArray[np.float64],
    margin: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ranges of integer lattice indices that cover an image."""
    h, w = size_hw
    A = np.column_stack([a1, a2])
    if np.linalg.matrix_rank(A) < 2:
        return np.arange(-1, 2), np.arange(-1, 2)
    corners = np.array([[1, 1, w, w], [1, h, 1, h]], dtype=float)
    q = np.linalg.lstsq(A, corners - base[:, None], rcond=None)[0]
    nmin = int(np.floor(np.min(q[0])) - margin)
    nmax = int(np.ceil(np.max(q[0])) + margin)
    mmin = int(np.floor(np.min(q[1])) - margin)
    mmax = int(np.ceil(np.max(q[1])) + margin)
    return np.arange(nmin, nmax + 1), np.arange(mmin, mmax + 1)


def reorder_by_nearest(
    stored: NDArray[np.float64],
    recomputed: NDArray[np.float64],
    occ_vec: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reorder occupancy values by nearest-neighbour matching."""
    if stored.size == 0 or recomputed.size == 0:
        return np.zeros(len(recomputed), dtype=float)
    used = np.zeros(len(stored), dtype=bool)
    out = np.zeros(len(recomputed), dtype=float)
    for i, target in enumerate(recomputed):
        d2 = np.sum((stored - target) ** 2, axis=1)
        d2[used] = np.inf
        j = int(np.argmin(d2))
        used[j] = True
        out[i] = occ_vec[j] if j < len(occ_vec) else 0
    return out


def binary_corr(A: BoolArray2D, B: BoolArray2D, dn: int, dm: int) -> int:
    """Binary correlation score between lattice-aligned frames."""
    if dn == 0 and dm == 0:
        overlap_a = A
        overlap_b = B
    else:
        slc_a, slc_b = _shift_overlap_slices(A.shape, dn, dm)
        if slc_a is None:
            return 0
        overlap_a = A[slc_a]
        overlap_b = B[slc_b]
    return int(np.sum(overlap_a & overlap_b))


def integer_shift(B: BoolArray2D, dn: int, dm: int) -> BoolArray2D:
    """Shift a boolean matrix by integer lattice steps."""
    h, w = B.shape
    out = np.zeros_like(B)
    slc_src, slc_dst = _shift_slices(B.shape, dn, dm)
    if slc_src is not None:
        out[slc_dst] = B[slc_src]
    return out


def logical_and_overlap(A: BoolArray2D, B: BoolArray2D) -> BoolArray2D:
    """Return element-wise AND on the overlapping region of two arrays."""
    h = min(A.shape[0], B.shape[0])
    w = min(A.shape[1], B.shape[1])
    return A[:h, :w] & B[:h, :w]


def remove_matches(B: BoolArray2D, mask_on_a: BoolArray2D, dn: int, dm: int) -> BoolArray2D:
    """Remove matched entries from ``B`` corresponding to shift ``(dn, dm)``."""
    shifted = integer_shift(mask_on_a, -dn, -dm)
    shifted = shifted[: B.shape[0], : B.shape[1]]
    out = B.copy()
    out[shifted] = False
    return out


def enhance_dog(image: Array2D, sigma_lo: float, sigma_hi: float) -> Array2D:
    """Enhance high-frequency features using a difference-of-Gaussians filter."""
    low = gaussian_filter(image, sigma_lo)
    high = gaussian_filter(image, sigma_hi)
    diff = np.clip(low - high, a_min=0, a_max=None)
    return rescale(diff)


def rescale(image: Array2D) -> Array2D:
    """Linearly rescale image to [0, 1]."""
    mn = np.min(image)
    mx = np.max(image)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(image)
    return (image - mn) / (mx - mn)


def sample_at_points(image: Array2D, pts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Bilinear sampling of ``image`` at floating-point coordinates ``pts``."""
    if len(pts) == 0:
        return np.empty(0)
    h, w = image.shape
    x = np.clip(pts[:, 0], 1, w) - 1
    y = np.clip(pts[:, 1], 1, h) - 1
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    a = x - x0
    b = y - y0
    vals = (
        (1 - a) * (1 - b) * image[y0, x0]
        + a * (1 - b) * image[y0, x1]
        + (1 - a) * b * image[y1, x0]
        + a * b * image[y1, x1]
    )
    return vals


def draw_spots(image: Array2D, pts: NDArray[np.float64], radius: int, value: float) -> Array2D:
    out = image.copy()
    h, w = out.shape
    r = max(1, int(round(radius)))
    for cx, cy in pts:
        xmin = max(0, int(np.floor(cx - r)))
        xmax = min(w - 1, int(np.ceil(cx + r)))
        ymin = max(0, int(np.floor(cy - r)))
        ymax = min(h - 1, int(np.ceil(cy + r)))
        if xmin > xmax or ymin > ymax:
            continue
        xs = np.arange(xmin, xmax + 1)
        ys = np.arange(ymin, ymax + 1)
        xx, yy = np.meshgrid(xs, ys)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        block = out[ymin : ymax + 1, xmin : xmax + 1]
        block[mask] = value
        out[ymin : ymax + 1, xmin : xmax + 1] = block
    return out


def angle_between_deg(u: NDArray[np.float64], v: NDArray[np.float64]) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return np.nan
    c = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def angle_diff_deg(a: float, b: float) -> float:
    a = np.mod(a, 180.0)
    b = np.mod(b, 180.0)
    d = abs(a - b)
    return float(min(d, 180.0 - d))


def angle_mode_deg(vals: Sequence[float], bin_size_deg: float) -> float:
    valid = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    if valid.size == 0:
        return float("nan")
    vals_mod = np.mod(valid, 180.0)
    edges = np.arange(0, 180 + bin_size_deg, bin_size_deg)
    counts, edges = np.histogram(vals_mod, edges)
    idx = int(np.argmax(counts))
    mask = (vals_mod >= edges[idx]) & (vals_mod < edges[idx + 1])
    return circ_mean_deg(vals_mod[mask])


def circ_mean_deg(angles_deg: ArrayLike) -> float:
    if len(angles_deg) == 0:
        return float("nan")
    angles_rad = np.deg2rad(np.mod(angles_deg, 180.0))
    z = np.mean(np.exp(1j * 2 * angles_rad))
    return float(np.mod(np.degrees(0.5 * np.angle(z)), 180.0))


def length_mode_px(lengths: Sequence[float]) -> float:
    valid = np.array([v for v in lengths if np.isfinite(v) and v > 0], dtype=float)
    if valid.size == 0:
        return float("nan")
    if valid.size == 1:
        return float(valid[0])
    bw = max(0.5, 2 * (np.percentile(valid, 75) - np.percentile(valid, 25)) / (len(valid) ** (1 / 3)))
    edges = np.arange(valid.min() - bw, valid.max() + 2 * bw, bw)
    counts, edges = np.histogram(valid, edges)
    idx = int(np.argmax(counts))
    mask = (valid >= edges[idx]) & (valid < edges[idx + 1])
    return float(np.mean(valid[mask]))


def local_maxima_mask(magnitude: Array2D) -> BoolArray2D:
    """Binary mask of local maxima using a 3x3 footprint."""
    filtered = maximum_filter(magnitude, size=3, mode="nearest")
    return (magnitude == filtered) & (magnitude > 0)


def kmeans_1d(data: NDArray[np.float64], k: int, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    if data.size < k:
        raise ValueError("Not enough data for k-means clustering")
    # initialise centroids using quantiles
    percentiles = np.linspace(0, 100, k + 2)[1:-1]
    centroids = np.percentile(data, percentiles)
    labels = np.zeros(data.shape, dtype=int)
    for _ in range(max_iter):
        distances = np.abs(data[:, None] - centroids[None, :])
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for idx in range(k):
            mask = labels == idx
            if np.any(mask):
                centroids[idx] = np.mean(data[mask])
    return labels, centroids


def gaussian_fft_peaks(
    image: Array2D,
    r1: float,
    r2: float,
    slack: float,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Harvest prominent peaks from the FFT magnitude image."""
    h, w = image.shape
    fft = np.fft.fftshift(np.fft.fft2(image - np.mean(image)))
    magnitude = np.abs(fft)
    yy, xx = np.indices((h, w))
    cx = (w - 1) / 2
    cy = (h - 1) / 2
    gx = (xx - cx) / w
    gy = (yy - cy) / h
    radius = np.hypot(gx, gy)
    mask = radius <= 2.0 / min(h, w)
    magnitude[mask] = 0
    peaks = local_maxima_mask(magnitude)
    band_min = min(r1 * (1 - slack), r2 * (1 - slack))
    band_max = max(r1 * (1 + slack), r2 * (1 + slack))
    candidate_mask = peaks & (radius >= band_min) & (radius <= band_max)
    coords = np.column_stack(np.nonzero(candidate_mask))
    values = magnitude[candidate_mask]
    if values.size == 0:
        return magnitude, np.empty((0, 2)), np.empty(0)
    order = np.argsort(values)[::-1][:top_k]
    coords = coords[order]
    values = values[order]
    freqs = np.column_stack(((coords[:, 1] - cx) / w, (coords[:, 0] - cy) / h))
    return magnitude, freqs, values


def _shift_overlap_slices(shape: Tuple[int, int], dn: int, dm: int):
    h, w = shape
    c1 = max(0, -dn)
    c2 = min(w, w - dn)
    r1 = max(0, -dm)
    r2 = min(h, h - dm)
    if c1 >= c2 or r1 >= r2:
        return None, None
    slc_a = np.s_[r1:r2, c1:c2]
    slc_b = np.s_[r1 + dm : r2 + dm, c1 + dn : c2 + dn]
    return slc_a, slc_b


def _shift_slices(shape: Tuple[int, int], dn: int, dm: int):
    h, w = shape
    src_c1 = max(0, dn)
    src_c2 = min(w, w + dn)
    src_r1 = max(0, dm)
    src_r2 = min(h, h + dm)
    dst_c1 = max(0, -dn)
    dst_c2 = min(w, w - dn)
    dst_r1 = max(0, -dm)
    dst_r2 = min(h, h - dm)
    if src_c1 >= src_c2 or src_r1 >= src_r2:
        return None, None
    src_slice = np.s_[src_r1:src_r2, src_c1:src_c2]
    dst_slice = np.s_[dst_r1:dst_r2, dst_c1:dst_c2]
    return src_slice, dst_slice
