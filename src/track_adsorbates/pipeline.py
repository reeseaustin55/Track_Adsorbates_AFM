"""Core processing pipeline for adsorbate tracking."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .adsorbates import (
    AdsorbateDetection,
    DiffusionResults,
    assign_tracks,
    compute_diffusion,
    detect_adsorbates,
)
from .io_utils import export_diffusion, export_frame_diffusion, export_lattice
from .lattice import (
    BASE_OFFSET,
    FrameLatticeFit,
    LatticeParameters,
    consolidate_lattice,
    cover_index_window,
    estimate_lattice_from_frame,
    lattice_points_from_indices,
    lattice_vectors_in_pixels,
)
from .video import VideoData
from .visualization import combine_frames, draw_lattice_overlay, write_video


@dataclass
class PipelineConfig:
    approx_a_angstrom: float
    approx_b_angstrom: float
    approx_angle_deg: float
    frame_width_nm: float
    wiggle_percent: float = 10.0
    atom_diameter_angstrom: float = 1.5
    drift_allowance_atoms: float = 2.0


@dataclass
class PipelineResult:
    lattice: LatticeParameters
    diffusion: DiffusionResults
    detections: List[AdsorbateDetection]
    overlay_video_path: Path
    diffusion_json: Path
    diffusion_csv: Path
    lattice_json: Path


PHI_REFERENCE = np.array([0.5, 0.5], dtype=np.float32)


def run_pipeline(video: VideoData, config: PipelineConfig) -> PipelineResult:
    output_dir = video.path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    approx_lattice = LatticeParameters(
        a_length=config.approx_a_angstrom,
        b_length=config.approx_b_angstrom,
        angle_deg=config.approx_angle_deg,
        orientation_deg=0.0,
    )

    width_pixels = video.frame_shape[1]
    pixel_size_nm = config.frame_width_nm / max(width_pixels, 1)

    first_pass = _fit_frames(video.frames_gray, approx_lattice, pixel_size_nm, tolerance=0.35)
    true_lattice = consolidate_lattice(first_pass)
    lattice_path = output_dir / f"{video.path.stem}_true_lattice.json"
    export_lattice(lattice_path, true_lattice)

    wiggle = max(config.wiggle_percent / 100.0, 0.02)
    second_pass = _fit_frames(video.frames_gray, true_lattice, pixel_size_nm, tolerance=wiggle, search_steps=24)

    a_vec_true, b_vec_true = lattice_vectors_in_pixels(true_lattice, pixel_size_nm)
    A_true = np.column_stack([a_vec_true, b_vec_true])
    base_ref = BASE_OFFSET + A_true @ PHI_REFERENCE
    n_range, m_range = cover_index_window(video.frame_shape, base_ref, a_vec_true, b_vec_true, margin=4)
    grid_indices = _grid_indices(n_range, m_range)

    detections: List[AdsorbateDetection] = []
    atom_diameter_pixels = config.atom_diameter_angstrom * 0.1 / max(pixel_size_nm, 1e-6)

    for frame, fit in zip(video.frames_gray, second_pass):
        phi_refined = _phases_from_fit(fit)
        phi_locked = np.mod(phi_refined, 1.0)
        points_all = lattice_points_from_indices(grid_indices, phi_locked, a_vec_true, b_vec_true)
        inside_mask = _points_within_frame(points_all, frame.shape)
        if not np.any(inside_mask):
            detections.append(
                AdsorbateDetection(
                    points=np.empty((0, 2), dtype=np.float32),
                    present=np.zeros(0, dtype=bool),
                    indices=np.empty((0, 2), dtype=np.int32),
                    phases=phi_locked,
                )
            )
            continue

        frame_norm = _normalise_frame(frame)
        detection = detect_adsorbates(
            frame_norm,
            points_all[inside_mask],
            grid_indices[inside_mask],
            phi_locked,
            atom_diameter_pixels=atom_diameter_pixels,
        )
        detections.append(detection)

    aligned_detections = _align_detections(
        detections,
        drift_allowance_atoms=config.drift_allowance_atoms,
        a_vec=a_vec_true,
        b_vec=b_vec_true,
        n_range=n_range,
        m_range=m_range,
    )

    grid_spacing = max(np.linalg.norm(a_vec_true), np.linalg.norm(b_vec_true))
    max_distance = max(grid_spacing * max(config.drift_allowance_atoms, 0.5), grid_spacing * 0.75)
    tracks = assign_tracks(aligned_detections, max_distance=max_distance)
    diffusion = compute_diffusion(tracks, pixel_size_nm=pixel_size_nm, fps=video.fps)

    overlay_frames = []
    for color_frame, detection in zip(video.frames_color, aligned_detections):
        overlay = draw_lattice_overlay(color_frame.shape[:2], detection, atom_diameter_pixels)
        overlay_frames.append(combine_frames(color_frame, overlay))

    overlay_path = output_dir / f"{video.path.stem}_overlay.mp4"
    write_video(overlay_path, overlay_frames, fps=video.fps)

    diffusion_json = output_dir / f"{video.path.stem}_diffusion_summary.json"
    diffusion_csv = output_dir / f"{video.path.stem}_diffusion_frame_values.csv"
    export_diffusion(diffusion_json, diffusion, fps=video.fps)
    export_frame_diffusion(diffusion_csv, diffusion.frame_values)

    return PipelineResult(
        lattice=true_lattice,
        diffusion=diffusion,
        detections=aligned_detections,
        overlay_video_path=overlay_path,
        diffusion_json=diffusion_json,
        diffusion_csv=diffusion_csv,
        lattice_json=lattice_path,
    )


def _fit_frames(
    frames: Sequence[np.ndarray],
    approx: LatticeParameters,
    pixel_size_nm: float,
    tolerance: float,
    search_steps: int = 18,
) -> List[FrameLatticeFit]:
    fits: List[FrameLatticeFit] = []
    fallback = None
    for frame in frames:
        fit = estimate_lattice_from_frame(
            frame,
            approx,
            pixel_size_nm,
            tolerance=tolerance,
            search_steps=search_steps,
        )
        if fit is None:
            fit = fallback
        if fit is None:
            # build a synthetic fallback using the approximate lattice
            a_vec, b_vec = lattice_vectors_in_pixels(approx, pixel_size_nm)
            translation = BASE_OFFSET + np.zeros(2, dtype=np.float32)
            fit = FrameLatticeFit(
                parameters=approx,
                a_vec=a_vec,
                b_vec=b_vec,
                translation=translation.astype(np.float32),
                phases=np.zeros(2, dtype=np.float32),
                quality=0.0,
            )
        fits.append(fit)
        fallback = fit
    return fits


def _phases_from_fit(fit: FrameLatticeFit) -> np.ndarray:
    A = np.column_stack([fit.a_vec, fit.b_vec])
    phases = np.linalg.solve(A, (fit.translation - BASE_OFFSET).astype(np.float64))
    return phases.astype(np.float32)


def _grid_indices(n_range: np.ndarray, m_range: np.ndarray) -> np.ndarray:
    nn, mm = np.meshgrid(n_range, m_range)
    stacked = np.stack([nn.ravel(), mm.ravel()], axis=1)
    return stacked.astype(np.int32)


def _points_within_frame(points: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    height, width = shape
    xs = points[:, 0]
    ys = points[:, 1]
    return (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)


def _normalise_frame(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32)
    min_val = float(frame.min())
    max_val = float(frame.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(frame)
    return (frame - min_val) / (max_val - min_val)


def _align_detections(
    detections: Sequence[AdsorbateDetection],
    drift_allowance_atoms: float,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    n_range: np.ndarray,
    m_range: np.ndarray,
) -> List[AdsorbateDetection]:
    if not detections:
        return []

    allowance = int(max(1, round(drift_allowance_atoms)))
    aligned: List[AdsorbateDetection] = []
    prev_matrix = None
    for idx, detection in enumerate(detections):
        if idx == 0:
            aligned.append(detection)
            prev_matrix = _occupancy_matrix(detection, n_range, m_range)
            continue

        current_matrix = _occupancy_matrix(detection, n_range, m_range)
        best_score = -np.inf
        best_shift = (0, 0)
        for dn in range(-allowance, allowance + 1):
            for dm in range(-allowance, allowance + 1):
                shifted = _shift_matrix(current_matrix, dn, dm)
                score = float(np.sum(prev_matrix & shifted)) if prev_matrix is not None else float(np.sum(shifted))
                if score > best_score:
                    best_score = score
                    best_shift = (dn, dm)

        shift_vec = best_shift[0] * a_vec + best_shift[1] * b_vec
        new_detection = AdsorbateDetection(
            points=detection.points + shift_vec,
            present=detection.present,
            indices=detection.indices + np.array(best_shift, dtype=np.int32),
            phases=detection.phases,
        )
        aligned.append(new_detection)
        prev_matrix = _occupancy_matrix(new_detection, n_range, m_range)

    return aligned


def _occupancy_matrix(
    detection: AdsorbateDetection,
    n_range: np.ndarray,
    m_range: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((len(m_range), len(n_range)), dtype=bool)
    if detection.points.size == 0:
        return matrix

    n0 = int(n_range[0])
    m0 = int(m_range[0])
    n_idx = detection.indices[:, 0]
    m_idx = detection.indices[:, 1]
    mask = (
        (n_idx >= n_range[0])
        & (n_idx <= n_range[-1])
        & (m_idx >= m_range[0])
        & (m_idx <= m_range[-1])
    )
    n_idx = n_idx[mask] - n0
    m_idx = m_idx[mask] - m0
    present = detection.present[mask]
    matrix[m_idx, n_idx] = present
    return matrix


def _shift_matrix(matrix: np.ndarray, dn: int, dm: int) -> np.ndarray:
    if dn == 0 and dm == 0:
        return matrix
    shifted = np.zeros_like(matrix)
    rows, cols = matrix.shape

    src_row_start = max(0, dm)
    src_row_end = min(rows, rows + dm)
    dst_row_start = max(0, -dm)
    dst_row_end = dst_row_start + (src_row_end - src_row_start)

    src_col_start = max(0, dn)
    src_col_end = min(cols, cols + dn)
    dst_col_start = max(0, -dn)
    dst_col_end = dst_col_start + (src_col_end - src_col_start)

    if src_row_start < src_row_end and src_col_start < src_col_end:
        shifted[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = matrix[
            src_row_start:src_row_end, src_col_start:src_col_end
        ]
    return shifted
