"""Adsorbate detection and diffusion analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


@dataclass
class AdsorbateDetection:
    """Stores lattice positions and occupancy for a single frame."""

    points: NDArray[np.float32]
    present: NDArray[np.bool_]
    indices: NDArray[np.int32]
    phases: NDArray[np.float32]

    def to_dict(self) -> Dict[str, NDArray]:
        return {
            "points": self.points,
            "present": self.present,
            "indices": self.indices,
            "phases": self.phases,
        }


@dataclass
class Track:
    id: int
    positions: List[Tuple[float, float]]
    frames: List[int]

    def add(self, frame_index: int, position: Tuple[float, float]) -> None:
        self.frames.append(frame_index)
        self.positions.append(position)

    def displacement_vectors(self) -> List[np.ndarray]:
        vectors: List[np.ndarray] = []
        for i in range(1, len(self.positions)):
            prev = np.array(self.positions[i - 1])
            curr = np.array(self.positions[i])
            vectors.append(curr - prev)
        return vectors


@dataclass
class DiffusionResults:
    """Container for diffusion statistics."""

    frame_values: NDArray[np.float64]
    average: float
    slope: float


def detect_adsorbates(
    frame: NDArray[np.float32],
    lattice_points: NDArray[np.float32],
    lattice_indices: NDArray[np.int32],
    phases: Sequence[float],
    atom_diameter_pixels: float,
) -> AdsorbateDetection:
    """Detect adsorbates by sampling intensities under lattice nodes.

    The samples are normalised per-frame so the classification adapts to
    contrast variations.  A two-tier decision is used: if the spread in
    sampled values is significant we threshold near the 70th percentile;
    otherwise we rescue the few brightest sites so that sparse occupancies
    are still captured.
    """

    height, width = frame.shape
    radius = max(float(atom_diameter_pixels) / 2.0, 1.0)
    y_coords = np.clip(lattice_points[:, 1], 0, height - 1)
    x_coords = np.clip(lattice_points[:, 0], 0, width - 1)

    samples = nd_gaussian_sample(frame, x_coords, y_coords, radius)
    if samples.size == 0:
        present = np.zeros(0, dtype=bool)
    else:
        samples = samples.astype(np.float32)
        sample_min = float(np.min(samples))
        sample_max = float(np.max(samples))
        span = sample_max - sample_min
        if span < 1e-6:
            present = np.zeros_like(samples, dtype=bool)
        else:
            norm = (samples - sample_min) / span
            high = np.percentile(norm, 70.0)
            if high < 0.35:
                high = 0.35
            present = norm >= high
            # ensure we keep at least a few of the brightest spots
            if np.count_nonzero(present) < max(3, int(0.03 * len(norm))):
                order = np.argsort(norm)[::-1]
                keep = order[: max(3, min(len(norm), int(np.ceil(0.05 * len(norm)))))]
                present = np.zeros_like(norm, dtype=bool)
                present[keep] = True

    return AdsorbateDetection(
        points=lattice_points.astype(np.float32, copy=False),
        present=present.astype(bool, copy=False),
        indices=lattice_indices.astype(np.int32, copy=False),
        phases=np.asarray(phases, dtype=np.float32),
    )


def nd_gaussian_sample(
    frame: NDArray[np.float32],
    x_coords: NDArray[np.float32],
    y_coords: NDArray[np.float32],
    radius: float,
) -> NDArray[np.float32]:
    """Sample frame intensities using a Gaussian weighting around given coordinates."""

    if x_coords.size == 0:
        return np.zeros(0, dtype=np.float32)

    sigma = max(radius / 2.0, 0.5)
    smoothed = ndimage.gaussian_filter(frame, sigma=sigma, mode="reflect")
    return _bilinear_sample(smoothed, x_coords, y_coords)


def _bilinear_sample(
    frame: NDArray[np.float32],
    x_coords: NDArray[np.float32],
    y_coords: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Bilinear interpolation for arbitrary sampling points."""

    height, width = frame.shape
    x0 = np.floor(x_coords).astype(int)
    y0 = np.floor(y_coords).astype(int)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    x0 = np.clip(x0, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)

    fx = x_coords - x0
    fy = y_coords - y0

    top_left = frame[y0, x0]
    top_right = frame[y0, x1]
    bottom_left = frame[y1, x0]
    bottom_right = frame[y1, x1]

    top = top_left * (1 - fx) + top_right * fx
    bottom = bottom_left * (1 - fx) + bottom_right * fx
    return (top * (1 - fy) + bottom * fy).astype(np.float32)


def assign_tracks(
    detections: List[AdsorbateDetection],
    max_distance: float,
) -> List[Track]:
    """Link adsorbate detections across frames using nearest-neighbour assignment."""

    tracks: List[Track] = []
    active_tracks: Dict[int, Track] = {}
    next_id = 0

    for frame_idx, detection in enumerate(detections):
        present_points = detection.points[detection.present]
        if not len(present_points):
            # No detections; deactivate all active tracks
            active_tracks.clear()
            continue

        if not active_tracks:
            for point in present_points:
                track = Track(id=next_id, positions=[tuple(point)], frames=[frame_idx])
                tracks.append(track)
                active_tracks[next_id] = track
                next_id += 1
            continue

        track_ids = list(active_tracks.keys())
        prev_points = np.array([active_tracks[tid].positions[-1] for tid in track_ids])
        cost = np.linalg.norm(prev_points[:, None, :] - present_points[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_points = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= max_distance:
                track_id = track_ids[r]
                position = tuple(present_points[c])
                active_tracks[track_id].add(frame_idx, position)
                assigned_tracks.add(track_id)
                assigned_points.add(c)

        # Unassigned tracks are dropped
        for track_id in list(active_tracks.keys()):
            if track_id not in assigned_tracks:
                del active_tracks[track_id]

        # Start new tracks for unassigned points
        for idx, point in enumerate(present_points):
            if idx not in assigned_points:
                track = Track(id=next_id, positions=[tuple(point)], frames=[frame_idx])
                tracks.append(track)
                active_tracks[next_id] = track
                next_id += 1

    return tracks


def compute_diffusion(
    tracks: Iterable[Track],
    pixel_size_nm: float,
    fps: float,
) -> DiffusionResults:
    """Compute diffusion coefficient statistics from tracks."""

    displacements: List[float] = []
    frame_values: List[float] = []
    dt = 1.0 / max(fps, 1e-6)

    for track in tracks:
        positions = [np.array(p) for p in track.positions]
        for i in range(1, len(positions)):
            disp = positions[i] - positions[i - 1]
            disp_nm = np.linalg.norm(disp) * pixel_size_nm
            displacements.append(disp_nm)
            D = (disp_nm**2) / (4.0 * dt)
            frame_values.append(D)

    if not displacements:
        return DiffusionResults(
            frame_values=np.zeros(0, dtype=np.float64),
            average=0.0,
            slope=0.0,
        )

    frame_values_array = np.array(frame_values, dtype=np.float64)
    average = float(np.mean(frame_values_array))

    # Build MSD curve
    max_lag = max(len(track.positions) for track in tracks)
    msd = []
    times = []
    for lag in range(1, max_lag):
        lag_sq = []
        for track in tracks:
            positions = [np.array(p) for p in track.positions]
            for i in range(lag, len(positions)):
                disp = positions[i] - positions[i - lag]
                lag_sq.append((np.linalg.norm(disp) * pixel_size_nm) ** 2)
        if lag_sq:
            msd.append(np.mean(lag_sq))
            times.append(lag * dt)
    if len(msd) >= 2:
        slope, _ = np.polyfit(times, msd, 1)
        diffusion = slope / 4.0
    else:
        diffusion = average
        slope = 4.0 * diffusion
    return DiffusionResults(
        frame_values=frame_values_array,
        average=diffusion,
        slope=slope,
    )
