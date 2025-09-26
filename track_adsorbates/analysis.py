"""Diffusion analysis pipeline for adsorbate tracking."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import savemat

from .helpers import (
    binary_corr,
    integer_shift,
    nm_cover_image,
    remove_matches,
    reorder_by_nearest,
)
from .simple_lattice import FrameBundle, simple_lattice_occupancy


@dataclass
class DiffusionResults:
    D: float
    D_CI95: Tuple[float, float]
    mean_d2_nm2: float
    delta_t: float
    N_steps: int
    step_hist_counts: np.ndarray
    step_order: np.ndarray
    drift_nm: np.ndarray
    drift_pix: np.ndarray
    locked_basis_a1: np.ndarray
    locked_basis_a2: np.ndarray
    locked_params: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        a1 = data.pop("locked_basis_a1")
        a2 = data.pop("locked_basis_a2")
        data["locked_basis"] = {"a1": a1, "a2": a2}
        data["locked_params"] = self.locked_params
        data["step_hist_counts"] = {
            "dn": self.step_order[:, 0],
            "dm": self.step_order[:, 1],
            "count": self.step_hist_counts,
        }
        return data


def analyze_adsorbate_diffusion(
    vid_path: str | Path,
    vid_width_nm: float,
    ax_a: float,
    ay_a: float,
    gamma_deg_guess: float,
    *,
    probe_diam_a: float | None = None,
    max_frames: int | None = None,
) -> DiffusionResults:
    """Run the full adsorbate diffusion analysis pipeline."""
    bundle = simple_lattice_occupancy(
        vid_path,
        vid_width_nm,
        ax_a,
        ay_a,
        gamma_deg_guess,
        probe_diam_a=probe_diam_a,
        max_frames=max_frames,
    )

    vid_path = Path(vid_path)
    folder = vid_path.parent if vid_path.parent != Path("") else Path.cwd()
    base = vid_path.stem

    a1 = bundle.ax_star * np.array([np.cos(np.radians(bundle.theta_star)), np.sin(np.radians(bundle.theta_star))])
    a2 = bundle.ay_star * np.array(
        [
            np.cos(np.radians(bundle.theta_star + bundle.gamma_star)),
            np.sin(np.radians(bundle.theta_star + bundle.gamma_star)),
        ]
    )

    phi0 = np.array([0.5, 0.5])
    base0 = np.array([1.0, 1.0]) + phi0[0] * a1 + phi0[1] * a2
    n_range, m_range = nm_cover_image(bundle.frame_shape, base0, a1, a2, bundle.params.cover_margin)
    nn_glob, mm_glob = np.meshgrid(n_range, m_range)
    nn_vec = nn_glob.ravel()
    mm_vec = mm_glob.ravel()
    nmin = n_range[0]
    mmin = m_range[0]
    Ln = len(n_range)
    Lm = len(m_range)

    occ_mats: List[np.ndarray] = []
    for frame in bundle.occ_frames:
        occ_matrix = _reconstruct_occupancy(
            frame,
            nn_vec,
            mm_vec,
            nmin,
            mmin,
            Ln,
            Lm,
            a1,
            a2,
            bundle.frame_shape,
        )
        occ_mats.append(occ_matrix)

    neighbor_shifts = np.array(
        [
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, -1],
            [0, 0],
            [0, 1],
            [1, -1],
            [1, 0],
            [1, 1],
        ]
    )

    pair_shift = np.zeros((len(occ_mats) - 1, 2), dtype=int)
    for k in range(len(occ_mats) - 1):
        O1 = occ_mats[k]
        O2 = occ_mats[k + 1]
        best_score = -np.inf
        best_shift = np.array([0, 0])
        for dn, dm in neighbor_shifts:
            score = binary_corr(O1, O2, dn, dm)
            if score > best_score:
                best_score = score
                best_shift = np.array([dn, dm])
        pair_shift[k] = best_shift

    drift_nm = np.zeros((len(occ_mats), 2), dtype=int)
    for k in range(1, len(occ_mats)):
        drift_nm[k] = drift_nm[k - 1] + pair_shift[k - 1]
    drift_pix = np.column_stack(
        [
            drift_nm[:, 0] * a1[0] + drift_nm[:, 1] * a2[0],
            drift_nm[:, 0] * a1[1] + drift_nm[:, 1] * a2[1],
        ]
    )

    occ_aligned = []
    for k, occ in enumerate(occ_mats):
        dn, dm = -drift_nm[k]
        occ_aligned.append(integer_shift(occ, dn, dm))

    step_order = np.array(
        [
            [0, 0],
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1],
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1],
        ]
    )
    step_counts = np.zeros(len(step_order), dtype=int)

    S1 = 0.0
    S2 = 0.0
    N_steps = 0

    for k in range(len(occ_aligned) - 1):
        A = occ_aligned[k].copy()
        B = occ_aligned[k + 1].copy()
        for idx, (dn, dm) in enumerate(step_order):
            shifted = integer_shift(B, dn, dm)
            overlap = A & shifted
            c = int(np.sum(overlap))
            if c == 0:
                continue
            A[overlap] = False
            B = remove_matches(B, overlap, dn, dm)
            dvec = dn * a1 + dm * a2
            d2 = float(np.dot(dvec, dvec))
            S1 += c * d2
            S2 += c * (d2**2)
            N_steps += c
            step_counts[idx] += c

    delta_t = 1.0 / bundle.fps
    if N_steps > 1:
        mean_d2 = S1 / N_steps
        var_d2 = max(0.0, S2 / N_steps - mean_d2**2)
        se_d2 = np.sqrt(var_d2 / N_steps)
        nm_per_px = bundle.nm_per_px
        mean_d2_nm2 = (nm_per_px**2) * mean_d2
        se_d2_nm2 = (nm_per_px**2) * se_d2
        D = mean_d2_nm2 / (4 * delta_t)
        D_lo = (mean_d2_nm2 - 1.96 * se_d2_nm2) / (4 * delta_t)
        D_hi = (mean_d2_nm2 + 1.96 * se_d2_nm2) / (4 * delta_t)
    else:
        mean_d2_nm2 = float("nan")
        D = float("nan")
        D_lo = float("nan")
        D_hi = float("nan")

    results = DiffusionResults(
        D=D,
        D_CI95=(D_lo, D_hi),
        mean_d2_nm2=mean_d2_nm2,
        delta_t=delta_t,
        N_steps=N_steps,
        step_hist_counts=step_counts,
        step_order=step_order,
        drift_nm=drift_nm,
        drift_pix=drift_pix,
        locked_basis_a1=a1,
        locked_basis_a2=a2,
        locked_params={
            "ax_px": bundle.ax_star,
            "ay_px": bundle.ay_star,
            "gamma_deg": bundle.gamma_star,
            "theta_deg": bundle.theta_star,
        },
    )

    out_path = folder / f"{base}_diffusion_results.mat"
    savemat(out_path, {"results": results.to_dict()})

    return results


def _reconstruct_occupancy(
    frame: FrameBundle,
    nn_vec: np.ndarray,
    mm_vec: np.ndarray,
    nmin: int,
    mmin: int,
    Ln: int,
    Lm: int,
    a1: np.ndarray,
    a2: np.ndarray,
    frame_shape: Tuple[int, int],
) -> np.ndarray:
    phi = np.array(frame.phi_locked)
    base = np.array([1.0, 1.0]) + phi[0] * a1 + phi[1] * a2
    xy = np.column_stack(
        [
            base[0] + nn_vec * a1[0] + mm_vec * a2[0],
            base[1] + nn_vec * a1[1] + mm_vec * a2[1],
        ]
    )
    width = frame_shape[1]
    height = frame_shape[0]
    in_img = (
        (xy[:, 0] >= 1)
        & (xy[:, 0] <= width)
        & (xy[:, 1] >= 1)
        & (xy[:, 1] <= height)
    )
    occ_vec = frame.occ
    if len(occ_vec) != np.sum(in_img):
        occ_vec = reorder_by_nearest(frame.nodes_px_locked, xy[in_img], occ_vec)
    n_idx = nn_vec[in_img] - nmin
    m_idx = mm_vec[in_img] - mmin
    occ_matrix = np.zeros((Lm, Ln), dtype=bool)
    for val, ni, mi in zip(occ_vec > 0, n_idx, m_idx):
        if 0 <= mi < Lm and 0 <= ni < Ln:
            occ_matrix[mi, ni] = val
    return occ_matrix
