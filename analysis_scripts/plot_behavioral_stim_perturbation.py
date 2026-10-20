"""
plot_behavioral_stim_perturbation.py
====================================

Spatiotemporal Neural-Behavioral Perturbation Mapping across Cerebellum (NPRW)
and Motor Cortex (Utah Array).

Scientific Objective:
    Map single-trial neural firing dynamics across electrodes and time bins onto the 3D
    behavioral reach perturbation components (Lateral ΔX, Forward ΔY, Height ΔZ) and
    movement termination / plateau timing.

Outputs per Condition:
    1. behavioral_perturbation_summary.png & .svg:
       - Panel A: Time-resolved directional reach deviations ΔX(t), ΔY(t), ΔZ(t) (Mean ± SEM).
       - Panel B: Overhead reach endpoint perturbation vectors (ΔX vs ΔY) with confidence ellipse.
       - Panel C: 3D endpoint dispersion (RMS distance) comparing Stim vs Control.
       - Panel D: Plateau onset timing comparison (Stim vs Control jittered trials + Δt test).
    2. spatiotemporal_projection_NPRW.png & .svg:
       - 3-row map (Lateral ΔX, Forward ΔY, Height ΔZ) showing single-trial neural-behavior
         projection r(c, t) across 128 cerebellar channels and time bins (20 ms each).
    3. spatiotemporal_projection_UA.png & .svg:
       - 3-row spatiotemporal map for Motor Cortex Utah Array.
    4. spatiotemporal_neural_behavior_summary.csv:
       - Channel-by-channel tuning statistics (peak correlation, latency, depth, and significance).
"""

from __future__ import annotations

import os
import re
import sys
import copy
import json
import math
import argparse
import warnings
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Ellipse
from scipy.ndimage import gaussian_filter1d
from scipy import stats

# Suppress benign numerical runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Degrees of freedom <= 0 for slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN slice encountered.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Add analysis_scripts and root to sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import plot_3d_kinematics as p3d
    HAS_3D_KINEMATICS = True
except Exception:
    HAS_3D_KINEMATICS = False
    p3d = None

import RCP_analysis as rcp
try:
    from RCP_analysis.python.functions.config_loading import *
    CONFIG_LOADED = True
except Exception:
    CONFIG_LOADED = False
    PARAMS = None
    OUT_BASE = None
    PERI_ROOT = None
    SESSION_LOC = None
    METADATA_CSV = None

from RCP_analysis.python.functions.kinematics_utils import extract_xy_coordinates

# =============================================================================
# Global Analysis & Display Parameters
# =============================================================================

WIN_PLOT_MS: Tuple[float, float] = (0.0, 500.0)
BIN_WIDTH_MS: float = 20.0
SMOOTH_SIGMA_MS: float = 20.0
BEHAVIOR_WINDOW_MS: Tuple[float, float] = (0.0, 500.0)
CONTROL_REF_STAT: str = "median"  # 'median' (recommended for reach kinematics) or 'mean'

# Plot output format selection (default PNG only)
SAVE_PNG: bool = True
SAVE_SVG: bool = False

# 3D Kinematics Settings
DEFAULT_KEYPOINT: str = "middle"
SQUARE_SIZE_MM: float = 1.0  # Calibrated mm/unit
ALIGN_TO_TASK_FRAME: bool = True

# Stimulation Artifact Blanking Parameters
NPRW_BLANK_PRE_MS: float = 20.0
NPRW_BLANK_POST_MS: float = 20.0
BLANK_UA_STIM_PERIOD: bool = False

P_VAL_THRESH: float = 0.05
FDR_ALPHA: float = 0.05


def _log(msg: str):
    print(f"[spatiotemporal_perturbation] {msg}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_get_npz(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key in data.files:
        val = data[key]
        if hasattr(val, "shape") and val.ndim == 0:
            return val.item()
        return val
    # Case-insensitive search
    k_lower = key.lower()
    for f in data.files:
        if f.lower() == k_lower:
            val = data[f]
            if hasattr(val, "shape") and val.ndim == 0:
                return val.item()
            return val
    return default


def scalarize(x, default=None):
    if x is None:
        return default
    if isinstance(x, (int, float, str, bool)):
        return x
    if hasattr(x, "__len__") and len(x) == 1:
        elem = x[0]
        return scalarize(elem, default)
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return default


def resolve_session_name_from_path(p: Path) -> str:
    path_str = str(p)
    m = re.search(r"(NRR_[A-Za-z0-9]+)", path_str)
    if m:
        return m.group(1)
    if p.is_dir():
        return p.name
    return p.parent.name


def extract_br_index(path: Optional[Path]) -> int:
    if path is None:
        return -1
    m = re.search(r"BR_(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def get_stim_duration_ms(data: np.lib.npyio.NpzFile) -> float:
    for k in ["stim_dur_nominal_ms", "stim_dur_measured_ms", "stim_duration_ms", "stim_ms"]:
        val = safe_get_npz(data, k, None)
        if val is not None:
            s = scalarize(val, None)
            if s is not None and float(s) > 0.0:
                return float(s)
    dur_s = safe_get_npz(data, "stim_duration_s", None)
    if dur_s is not None:
        val = scalarize(dur_s, None)
        if val is not None and float(val) > 0.0:
            return float(val) * 1000.0
    return 100.0


# =============================================================================
# Reference Condition Mapping (Stim -> Best Control)
# =============================================================================

def load_reference_mapping(all_files: List[Path]) -> Dict[Path, Path]:
    """
    Pairs each STIM peristim reach condition with the best target-matched CONTROL condition.
    Selects the candidate control file with the maximum number of valid reaches.
    """
    ref_map: Dict[Path, Path] = {}
    ctrl_files_by_target: Dict[str, List[Path]] = {"target_a": [], "target_b": []}

    for f in all_files:
        p_str = str(f).lower()
        if "at_rest" in p_str:
            continue
        is_ctrl = "control_reaches" in p_str or "ctrl" in f.stem.lower() or "control" in f.stem.lower()
        if is_ctrl:
            target = "target_b" if "target_b" in p_str else "target_a"
            ctrl_files_by_target[target].append(f)

    for f in all_files:
        p_str = str(f).lower()
        if "at_rest" in p_str:
            continue
        is_ctrl = "control_reaches" in p_str or "ctrl" in f.stem.lower() or "control" in f.stem.lower()
        if is_ctrl:
            continue
        is_stim = "stim_reaches" in p_str or "stim" in f.stem.lower() or "target_" in f.stem.lower()
        if not is_stim:
            continue

        target = "target_b" if "target_b" in p_str else "target_a"
        candidates = ctrl_files_by_target.get(target, [])
        if not candidates:
            continue

        valid_candidates = []
        for cand in candidates:
            try:
                with np.load(cand, allow_pickle=True) as cd:
                    c_ev = safe_get_npz(cd, "event_ms")
                    if c_ev is None:
                        c_ev = safe_get_npz(cd, "event_timestamps_ms")
                    reach_cnt = len(c_ev) if c_ev is not None else 0
                    valid_candidates.append((reach_cnt, cand))
            except Exception:
                valid_candidates.append((0, cand))

        if valid_candidates:
            best_cand = max(valid_candidates, key=lambda x: x[0])[1]
            ref_map[f] = best_cand

    return ref_map


# =============================================================================
# Electrode Map & Array Utilities
# =============================================================================

def load_bad_channels(session_loc: Optional[Path] = None) -> Dict[str, set]:
    bad_dict = {"NPRW": set(), "UA": set()}
    if session_loc is None:
        return bad_dict
    bad_f = session_loc / "bad_channels.json"
    if bad_f.exists():
        try:
            with open(bad_f, "r") as fp:
                data = json.load(fp)
            for k in ["NPRW", "UA"]:
                if k in data:
                    bad_dict[k] = set(data[k])
        except Exception as e:
            _log(f"Error reading bad_channels.json: {e}")
    return bad_dict


def load_electrode_mapping(monkey: Optional[str] = None) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, np.ndarray]]:
    mapping: Dict[int, Dict[str, Any]] = {}
    banks: Dict[str, List[int]] = {"SMA": [], "PMd": [], "M1i": [], "M1s": []}

    csv_candidates = [
        _REPO_ROOT / "config" / f"electrode_mapping_{monkey}.csv" if monkey else None,
        _REPO_ROOT / "config" / "electrode_mapping.csv",
        _SCRIPT_DIR / "electrode_mapping.csv",
    ]
    csv_path = None
    for c in csv_candidates:
        if c and c.exists():
            csv_path = c
            break

    if csv_path:
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                ch = int(row.get("channel", row.get("Channel", 0)))
                bank = str(row.get("bank", row.get("Bank", ""))).strip()
                depth = float(row.get("depth", row.get("Depth", ch * 20.0)))
                mapping[ch] = {"bank": bank, "depth": depth}
                if bank in banks:
                    banks[bank].append(ch)
        except Exception as e:
            _log(f"Failed to read electrode mapping CSV: {e}")

    bank_arrays = {k: np.array(v) for k, v in banks.items()}
    return mapping, bank_arrays


# =============================================================================
# Neural Rate Processing
# =============================================================================

def prepare_counts_3d(counts_raw: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Standardizes neural count matrices to shape (n_channels, n_trials, n_time).
    Transposes from (n_trials, n_channels, n_time) when appropriate.
    """
    if counts_raw is None or counts_raw.size == 0:
        return None
    arr = np.asarray(counts_raw, dtype=float)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim == 3:
        # PeriStim format is (n_trials, n_channels, n_time)
        if arr.shape[1] == 128 or (arr.shape[0] != 128 and arr.shape[1] > arr.shape[0]):
            return np.transpose(arr, (1, 0, 2))
        return arr
    return None


def nan_gaussian_filter1d(arr_3d: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0.0:
        return arr_3d.copy()
    out = np.empty_like(arr_3d, dtype=float)
    n_ch, n_trials, _ = arr_3d.shape
    for c in range(n_ch):
        for tr in range(n_trials):
            row = arr_3d[c, tr, :]
            nan_mask = np.isnan(row)
            if np.all(nan_mask):
                out[c, tr, :] = np.nan
                continue
            if not np.any(nan_mask):
                out[c, tr, :] = gaussian_filter1d(row, sigma=sigma_bins, mode="nearest")
                continue
            filled = row.copy()
            valid_idx = np.where(~nan_mask)[0]
            filled[nan_mask] = np.interp(np.where(nan_mask)[0], valid_idx, row[valid_idx])
            smoothed = gaussian_filter1d(filled, sigma=sigma_bins, mode="nearest")
            smoothed[nan_mask] = np.nan
            out[c, tr, :] = smoothed
    return out


def rebin_and_smooth(
    counts_3d: np.ndarray,
    centers_ms: np.ndarray,
    target_bin_w: float = BIN_WIDTH_MS,
    smooth_sigma_ms: float = SMOOTH_SIGMA_MS,
    win_ms: Tuple[float, float] = WIN_PLOT_MS,
    blank_win: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(counts_3d, dtype=float)
    centers = np.asarray(centers_ms, dtype=float)

    t_min, t_max = win_ms
    start_edge = np.floor(t_min / target_bin_w) * target_bin_w
    end_edge = np.ceil(t_max / target_bin_w) * target_bin_w
    out_edges = np.arange(start_edge, end_edge + target_bin_w * 0.5, target_bin_w)
    out_centers = 0.5 * (out_edges[:-1] + out_edges[1:])

    n_ch, n_trials, _ = counts.shape
    n_out = len(out_centers)

    bin_idx = np.digitize(centers, out_edges) - 1
    rebinned = np.zeros((n_ch, n_trials, n_out), dtype=float)

    for b in range(n_out):
        mask = bin_idx == b
        if np.any(mask):
            rebinned[:, :, b] = np.sum(counts[:, :, mask], axis=2)

    rate_scale = 1000.0 / target_bin_w
    rates_hz = rebinned * rate_scale

    if blank_win is not None:
        b0, b1 = blank_win
        blank_mask = (out_centers >= b0) & (out_centers <= b1)
        rates_hz[:, :, blank_mask] = np.nan

    sigma_bins = smooth_sigma_ms / target_bin_w
    rates_smoothed = nan_gaussian_filter1d(rates_hz, sigma_bins)
    return rates_smoothed, out_centers


# =============================================================================
# 3D Kinematics Ingestion & Plateau Integration
# =============================================================================

def load_plateau_analysis_table(session: str, sess_dir: Path) -> Optional[pd.DataFrame]:
    candidates = [
        sess_dir / "results" / "figures" / "plateau_analysis" / "tables" / "plateau_detection_results_all_conditions.csv",
        sess_dir / "results" / "figures" / "plateau_analysis" / "tables" / f"plateau_results_{session}.csv",
    ]
    for cand in candidates:
        if cand.exists():
            try:
                df = pd.read_csv(cand)
                if "br_idx" in df.columns and "onset_ms" in df.columns:
                    _log(f"Loaded plateau analysis table: {cand.name}")
                    return df
            except Exception:
                pass
    return None


def extract_plateau_times_from_table(
    df_plat: Optional[pd.DataFrame],
    br_idx: int,
    target: str,
    n_trials: int,
) -> Optional[np.ndarray]:
    if df_plat is None or df_plat.empty or br_idx < 0:
        return None
    try:
        t_clean = f"target{target.strip().lower()}"
        mask = (df_plat["br_idx"] == br_idx)
        if "condition" in df_plat.columns:
            mask = mask & df_plat["condition"].str.replace(" ", "").str.replace("_", "").str.lower().str.contains(t_clean)
        sub = df_plat[mask]
        if sub.empty:
            return None
        out = np.full(n_trials, np.nan, dtype=float)
        for _, row in sub.iterrows():
            t_idx = int(row["trial_idx"])
            if 0 <= t_idx < n_trials and bool(row.get("plateau_found", True)):
                out[t_idx] = float(row["onset_ms"])
        if np.any(np.isfinite(out)):
            return out
    except Exception:
        pass
    return None


def compute_trial_endpoints_3d(
    coords: np.ndarray,
    beh_t: np.ndarray,
    search_window_ms: Tuple[float, float] = BEHAVIOR_WINDOW_MS,
    override_plateau_times: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes single-trial 3D reach endpoints and plateau onset times.
    Samples endpoint coordinates specifically at movement termination / plateau onset.
    """
    n_trials, n_coords, n_time = coords.shape
    endpoints = np.full((n_trials, n_coords), np.nan, dtype=float)
    plateau_times = np.full(n_trials, np.nan, dtype=float)

    if n_time < 3 or len(beh_t) != n_time:
        return endpoints, plateau_times

    dt_s = (beh_t[1] - beh_t[0]) / 1000.0 if len(beh_t) > 1 else 0.01
    t0, t1 = search_window_ms
    win_mask = (beh_t >= t0) & (beh_t <= t1)
    if not np.any(win_mask):
        win_mask = (beh_t >= 0.0) & (beh_t <= 500.0)
    win_indices = np.where(win_mask)[0]

    for i in range(n_trials):
        tr_coords = coords[i]
        if np.all(np.isnan(tr_coords)):
            continue

        if override_plateau_times is not None and i < len(override_plateau_times) and np.isfinite(override_plateau_times[i]):
            t_plat = float(override_plateau_times[i])
            k_end = int(np.argmin(np.abs(beh_t - t_plat)))
            endpoints[i, :] = tr_coords[:, k_end]
            plateau_times[i] = t_plat
            continue

        # Deceleration fallback from continuous 3D speed into reach apex
        vel = np.gradient(tr_coords, dt_s, axis=1)
        speed = np.sqrt(np.nansum(vel ** 2, axis=0))

        fwd_axis = 1 if n_coords >= 2 else 0
        fwd_trace = tr_coords[fwd_axis]
        win_fwd = fwd_trace[win_indices]
        if np.all(np.isnan(win_fwd)):
            continue

        k_local_peak_y = int(np.nanargmax(win_fwd))
        k_peak_y = win_indices[k_local_peak_y]
        win_speed = speed[win_indices[: k_local_peak_y + 1]]
        v_max = float(np.nanmax(win_speed)) if len(win_speed) > 0 and np.any(np.isfinite(win_speed)) else 20.0
        v_thresh = max(15.0, v_max * 0.20)

        k_end = k_peak_y
        if len(win_speed) > 3:
            k_local_peak_v = int(np.nanargmax(win_speed))
            k_peak_v = win_indices[k_local_peak_v]
            for k in range(k_peak_v, min(k_peak_y + 4, n_time - 2)):
                if speed[k] <= v_thresh:
                    k_end = k
                    break

        endpoints[i, :] = tr_coords[:, k_end]
        plateau_times[i] = beh_t[k_end]

    return endpoints, plateau_times


def load_condition_3d_task_kinematics(
    npz_path: Path,
    session_name: str,
    dlc_3d_dir: Optional[Path] = None,
    session_landmarks: Optional[Dict[str, np.ndarray]] = None,
    meta_csv: Optional[Path] = None,
    aligned_ckpt_dir: Optional[Path] = None,
    behv_ckpt_dir: Optional[Path] = None,
    raw_trial_indices: Optional[Sequence[int]] = None,
    keypoint: str = DEFAULT_KEYPOINT,
    square_size_mm: float = SQUARE_SIZE_MM,
    align_task_frame: bool = ALIGN_TO_TASK_FRAME,
) -> Optional[Dict[str, Any]]:
    """Loads calibrated 3D task-frame reach coordinates [X_lateral, Y_forward, Z_height] in mm."""
    if not HAS_3D_KINEMATICS or p3d is None:
        return None

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            event_ms = safe_get_npz(data, "event_ms")
            if event_ms is None:
                event_ms = safe_get_npz(data, "event_timestamps_ms")
            if event_ms is None or len(event_ms) == 0:
                return None
            peri_meta = safe_get_npz(data, "meta", None)
            br_idx = int(safe_get_npz(data, "br_idx", extract_br_index(npz_path)))

        if dlc_3d_dir is None or not dlc_3d_dir.exists():
            return None

        vid_idx = p3d.resolve_video_index_for_br(br_idx, meta_csv, peri_meta) if meta_csv else br_idx
        if vid_idx is None:
            vid_idx = br_idx

        dlc_csv_path = p3d.find_3d_dlc_csv(dlc_3d_dir, session_name, vid_idx, br_idx)
        if dlc_csv_path is None or not dlc_csv_path.exists():
            return None

        df_dlc, bodyparts = p3d.load_dlc_3d_file(dlc_csv_path, square_size_mm=square_size_mm)
        if df_dlc is None:
            return None

        frame_t_ms = p3d.load_frame_timeline_ms(
            br_idx=br_idx,
            vid_idx=vid_idx,
            session=session_name,
            aligned_ckpt_root=aligned_ckpt_dir,
            behv_ckpt_root=behv_ckpt_dir,
        )
        if frame_t_ms is None:
            frame_t_ms = np.arange(len(df_dlc)) * 10.0

        rel_t, x_all, y_all, z_all, valid_mask = p3d.segment_condition_3d_reaches(
            dlc_df=df_dlc,
            frame_t_ms=frame_t_ms,
            event_ms=event_ms,
            keypoint=keypoint,
            pre_zero_ms=getattr(p3d, "PRE_ZERO_MS", -800.0),
            post_zero_ms=getattr(p3d, "POST_ZERO_MS", 600.0),
            step_ms=getattr(p3d, "STEP_MS", 10.0),
        )

        if align_task_frame:
            if session_landmarks is None:
                session_landmarks = p3d.scan_session_landmarks(dlc_3d_dir, square_size_mm=square_size_mm)
            R_task, p_start = p3d.compute_task_rotation_matrix(
                dlc_df=df_dlc,
                session_landmarks=session_landmarks,
                square_size_mm=square_size_mm,
            )
            pts_cam = np.stack([x_all, y_all, z_all], axis=0) - p_start.reshape(3, 1, 1)
            pts_task = np.einsum("ji,jkl->ikl", R_task, pts_cam)
            coords_3d = np.transpose(pts_task, (1, 0, 2))
        else:
            coords_3d = np.stack([x_all, y_all, z_all], axis=1)

        raw_tids = np.asarray(raw_trial_indices) if raw_trial_indices is not None else np.arange(len(event_ms))

        return {
            "beh_t": rel_t,
            "coords": coords_3d,
            "valid_mask": valid_mask,
            "raw_trial_indices": raw_tids,
            "is_3d": True,
            "coord_names": ["X_lateral", "Y_forward", "Z_height"],
        }
    except Exception as e:
        _log(f"3D kinematics loading failed for {npz_path.name}: {e}")
        return None


# =============================================================================
# Confidence Ellipse Drawing Utility
# =============================================================================

def draw_confidence_ellipse(ax: plt.Axes, x: np.ndarray, y: np.ndarray, n_std: float = 1.5, **kwargs):
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return
    xv, yv = x[valid], y[valid]
    q_x = np.nanpercentile(xv, [2, 98])
    q_y = np.nanpercentile(yv, [2, 98])
    in_box = (xv >= q_x[0]) & (xv <= q_x[1]) & (yv >= q_y[0]) & (yv <= q_y[1])
    if np.sum(in_box) >= 3:
        xv_e, yv_e = xv[in_box], yv[in_box]
    else:
        xv_e, yv_e = xv, yv

    cov = np.cov(xv_e, yv_e)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    if np.any(np.isnan(vals)):
        return
    vals = np.maximum(vals, 1e-4)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=(float(np.nanmedian(xv_e)), float(np.nanmedian(yv_e))), width=w, height=h, angle=theta, **kwargs)
    ax.add_patch(ell)


# =============================================================================
# Visualizations
# =============================================================================

def plot_behavioral_perturbation_summary(
    beh_t: np.ndarray,
    delta_traces: np.ndarray,       # (n_trials, 3, n_time)
    endpoints_s: np.ndarray,        # (n_s, 3)
    endpoints_c: np.ndarray,        # (n_c, 3)
    plateau_t_s: np.ndarray,        # (n_s,)
    plateau_t_c: np.ndarray,        # (n_c,)
    stim_pulse_ms: float = 100.0,
    condition_title: str = "",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Consolidated publication summary of 3D reach perturbation kinematics.
        Panel A: Time-resolved directional reach deviations ΔX(t), ΔY(t), ΔZ(t) (Mean ± SEM).
        Panel B: Overhead reach endpoint perturbation vectors (ΔX vs ΔY) with confidence ellipse.
        Panel C: 3D endpoint dispersion (RMS distance) comparing Stim vs Control.
        Panel D: Plateau onset timing comparison (Stim vs Control jittered trials + Δt test).
    """
    fig = plt.figure(figsize=(14.0, 10.0), dpi=300)
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.28)

    # 1. Panel A: Time-Resolved Directional Deviations
    ax_a = fig.add_subplot(gs[0, 0])
    t_mask = (beh_t >= 0.0) & (beh_t <= 500.0)
    t_sub = beh_t[t_mask]
    colors = {"Lateral (ΔX)": "#2980b9", "Forward (ΔY)": "#e74c3c", "Height (ΔZ)": "#27ae60"}

    for dim_idx, (dim_name, color) in enumerate(colors.items()):
        traces = delta_traces[:, dim_idx, t_mask]
        m = np.nanmean(traces, axis=0)
        sem = stats.sem(traces, axis=0, nan_policy="omit")
        ax_a.plot(t_sub, m, color=color, linewidth=2.0, label=dim_name)
        ax_a.fill_between(t_sub, m - sem, m + sem, color=color, alpha=0.18)

    ax_a.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    ax_a.axvspan(0.0, stim_pulse_ms, color="#f39c12", alpha=0.15, label=f"Stim ({stim_pulse_ms:.0f} ms)")
    ax_a.set_xlabel("Time Relative to Stim (ms)", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Perturbation Displacement (mm)", fontsize=11, fontweight="bold")
    ax_a.set_title("A. Time-Resolved Directional Deviations (Mean ± SEM)", fontsize=12, fontweight="bold")
    ax_a.legend(loc="upper left", frameon=True, fontsize=9)
    ax_a.grid(True, linestyle="--", alpha=0.4)

    # 2. Panel B: Overhead Reach Endpoint Perturbations (ΔX vs ΔY)
    ax_b = fig.add_subplot(gs[0, 1])
    c_ref = np.nanmedian(endpoints_c, axis=0)
    dx_s = endpoints_s[:, 0] - c_ref[0]
    dy_s = endpoints_s[:, 1] - c_ref[1]
    dx_c = endpoints_c[:, 0] - c_ref[0]
    dy_c = endpoints_c[:, 1] - c_ref[1]

    ax_b.scatter(dx_c, dy_c, color="#34495e", s=45, alpha=0.7, edgecolors="none", label=f"Control (N={len(dx_c)})")
    ax_b.scatter(dx_s, dy_s, color="#e74c3c", s=55, alpha=0.8, edgecolors="none", label=f"Stim (N={len(dx_s)})")
    ax_b.scatter(0, 0, marker="D", s=100, color="#2c3e50", label="Control Median Origin (0,0)")

    m_dx_s, m_dy_s = float(np.nanmedian(dx_s)), float(np.nanmedian(dy_s))
    ax_b.annotate(
        "", xy=(m_dx_s, m_dy_s), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=2.5)
    )
    draw_confidence_ellipse(ax_b, dx_c, dy_c, n_std=1.5, edgecolor="#34495e", facecolor="none", linestyle=":", lw=1.2)
    draw_confidence_ellipse(ax_b, dx_s, dy_s, n_std=1.5, edgecolor="#e74c3c", facecolor="none", linestyle="--", lw=1.5)

    ax_b.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=1.0)
    ax_b.axvline(0.0, color="#7f8c8d", linestyle="--", linewidth=1.0)
    ax_b.set_xlabel("Lateral Endpoint Shift ΔX (mm, left < 0 < right)", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Forward Endpoint Shift ΔY (mm, under < 0 < over)", fontsize=11, fontweight="bold")
    ax_b.set_title("B. Overhead Endpoint Perturbations (ΔX vs ΔY)", fontsize=12, fontweight="bold")
    ax_b.legend(loc="lower left", frameon=True, fontsize=9)
    ax_b.grid(True, linestyle="--", alpha=0.4)

    # 3. Panel C: Endpoint Dispersion (RMS Distance to Condition Median)
    ax_c = fig.add_subplot(gs[1, 0])
    s_med = np.nanmedian(endpoints_s, axis=0)
    disp_s = np.sqrt(np.nansum((endpoints_s - s_med[None, :]) ** 2, axis=1))
    disp_c = np.sqrt(np.nansum((endpoints_c - c_ref[None, :]) ** 2, axis=1))
    rms_s = float(np.sqrt(np.nanmean(disp_s ** 2)))
    rms_c = float(np.sqrt(np.nanmean(disp_c ** 2)))

    bars = ax_c.bar([0, 1], [rms_c, rms_s], color=["#4c5c68", "#e76f51"], width=0.55, edgecolor="#222222")
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels([f"Control (N={len(disp_c)})", f"Stim (N={len(disp_s)})"], fontsize=10, fontweight="bold")
    ax_c.set_ylabel("3D Endpoint Dispersion (RMS mm)", fontsize=11, fontweight="bold")
    ax_c.set_title("C. Endpoint Dispersion & Precision", fontsize=12, fontweight="bold")

    ratio = rms_s / max(rms_c, 1e-4)
    _, p_levene = stats.levene(disp_s[np.isfinite(disp_s)], disp_c[np.isfinite(disp_c)]) if len(disp_s) > 2 else (0, 1.0)
    ax_c.text(0.5, 0.85, f"Dispersion Ratio: {ratio:.2f}x\nLevene p = {p_levene:.3f}",
              transform=ax_c.transAxes, ha="center", va="center",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#bdc3c7", alpha=0.9), fontsize=10)
    ax_c.grid(True, axis="y", linestyle="--", alpha=0.4)

    # 4. Panel D: Plateau Onset Delay Comparison
    ax_d = fig.add_subplot(gs[1, 1])
    np.random.seed(42)
    jit_c = np.random.normal(0, 0.04, size=len(plateau_t_c))
    jit_s = np.random.normal(0, 0.04, size=len(plateau_t_s))

    ax_d.scatter(np.zeros(len(plateau_t_c)) + jit_c, plateau_t_c, color="#34495e", s=45, alpha=0.7, edgecolors="none")
    ax_d.scatter(np.ones(len(plateau_t_s)) + jit_s, plateau_t_s, color="#e74c3c", s=55, alpha=0.8, edgecolors="none")

    m_plat_c = float(np.nanmedian(plateau_t_c))
    m_plat_s = float(np.nanmedian(plateau_t_s))
    sem_plat_c = stats.sem(plateau_t_c, nan_policy="omit")
    sem_plat_s = stats.sem(plateau_t_s, nan_policy="omit")

    ax_d.errorbar(0, m_plat_c, yerr=sem_plat_c, fmt="D", color="#111111", markersize=9, capsize=6, lw=2.0)
    ax_d.errorbar(1, m_plat_s, yerr=sem_plat_s, fmt="D", color="#111111", markersize=9, capsize=6, lw=2.0)

    delta_plat = m_plat_s - m_plat_c
    try:
        _, p_plat = stats.mannwhitneyu(plateau_t_s[np.isfinite(plateau_t_s)], plateau_t_c[np.isfinite(plateau_t_c)])
    except Exception:
        p_plat = 1.0

    star_str = "***" if p_plat < 0.001 else ("**" if p_plat < 0.01 else ("*" if p_plat < 0.05 else "n.s."))
    y_max = max(np.nanmax(plateau_t_s), np.nanmax(plateau_t_c)) + 40
    ax_d.plot([0, 0, 1, 1], [y_max - 10, y_max, y_max, y_max - 10], color="#222222", lw=1.5)
    ax_d.text(0.5, y_max + 8, f"{star_str} (Δt = {delta_plat:+.0f} ms, p = {p_plat:.3f})", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels([f"Control (med={m_plat_c:.0f} ms)", f"Stim (med={m_plat_s:.0f} ms)"], fontsize=10, fontweight="bold")
    ax_d.set_ylabel("Plateau Onset Time from Stim (ms)", fontsize=11, fontweight="bold")
    ax_d.set_title("D. Plateau Timing Delay", fontsize=12, fontweight="bold")
    ax_d.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(f"Behavioral Perturbation Summary: {condition_title}", fontsize=14, fontweight="bold", y=0.98)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if SAVE_PNG:
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
        if SAVE_SVG:
            svg_path = out_path.with_suffix(".svg")
            fig.savefig(svg_path, bbox_inches="tight")
        _log(f"Saved behavioral perturbation figure: {out_path.name}")

    return fig


def compute_stim_vs_ctrl_modulation(
    rates_stim: np.ndarray,      # (n_ch, n_trials_stim, n_bins)
    rates_ctrl: np.ndarray,      # (n_ch, n_trials_ctrl, n_bins)
    time_ms: np.ndarray,         # (n_bins,)
    blank_window: Optional[Tuple[float, float]] = None,
    alpha: float = FDR_ALPHA,
) -> Dict[str, Any]:
    """
    Computes channel-level Welch's two-sample t-test comparing STIM vs target-matched CONTROL
    neural firing rates across all electrodes in the non-blanked analysis window.
    Applies Benjamini-Hochberg FDR control across channels to identify stim-responsive channels.
    Unresponsive channels are masked out in visualization and prediction.
    Computes single-trial stim expression scores S_i using ONLY the responsive channels.
    """
    n_ch, n_s, n_bins = rates_stim.shape
    _, n_c, _ = rates_ctrl.shape

    # Identify non-blanked analysis time bins
    valid_bins = np.ones(n_bins, dtype=bool)
    if blank_window is not None:
        b0, b1 = blank_window
        valid_bins = (time_ms < b0) | (time_ms > b1)

    # For each channel and trial, compute mean firing rate across valid time bins
    s_trial_rates = np.nanmean(rates_stim[:, :, valid_bins], axis=2)
    c_trial_rates = np.nanmean(rates_ctrl[:, :, valid_bins], axis=2)

    ch_tvals = np.zeros(n_ch, dtype=float)
    ch_pvals = np.ones(n_ch, dtype=float)

    for c in range(n_ch):
        s_v = s_trial_rates[c, :]
        c_v = c_trial_rates[c, :]
        val_s = np.isfinite(s_v)
        val_c = np.isfinite(c_v)
        if np.sum(val_s) >= 3 and np.sum(val_c) >= 3:
            if np.std(s_v[val_s]) > 1e-4 or np.std(c_v[val_c]) > 1e-4:
                tv, pv = stats.ttest_ind(s_v[val_s], c_v[val_c], equal_var=False)
                ch_tvals[c] = float(tv) if np.isfinite(tv) else 0.0
                ch_pvals[c] = float(pv) if np.isfinite(pv) else 1.0

    # Multiple testing correction across channels (Benjamini-Hochberg FDR)
    sorted_idx = np.argsort(ch_pvals)
    sorted_p = ch_pvals[sorted_idx]
    fdr_thresh = alpha * np.arange(1, n_ch + 1) / float(n_ch)
    sig_fdr = sorted_p <= fdr_thresh

    if np.any(sig_fdr):
        p_crit = float(sorted_p[np.max(np.where(sig_fdr)[0])])
        resp_channels = (ch_pvals <= p_crit) & (ch_pvals < alpha)
        mask_type = f"FDR q<{alpha:.2f}"
    else:
        p_crit = alpha
        resp_channels = (ch_pvals < alpha)
        mask_type = f"Uncorr p<{alpha:.2f}"

    # Fallback to uncorrected p < 0.05 if FDR yields fewer than 3 channels
    if np.sum(resp_channels) < 3 and mask_type.startswith("FDR"):
        uncorr = ch_pvals < alpha
        if np.sum(uncorr) >= 3:
            resp_channels = uncorr
            mask_type = f"Uncorr p<{alpha:.2f} (FDR<3)"

    n_resp = int(np.sum(resp_channels))

    # Single-Trial Stim Expression Score S_i computed using ONLY responsive channels
    ctrl_ch_mean = np.nanmean(c_trial_rates, axis=1)  # (n_ch,)
    stim_scores = np.zeros(n_s, dtype=float)
    if n_resp > 0:
        weights = np.sign(ch_tvals[resp_channels])
        weights[weights == 0.0] = 1.0
        for i in range(n_s):
            diff = s_trial_rates[resp_channels, i] - ctrl_ch_mean[resp_channels]
            stim_scores[i] = float(np.nansum(diff * weights) / max(np.sum(np.abs(weights)), 1.0))

    return {
        "resp_channels": resp_channels,
        "ch_tvals": ch_tvals,
        "ch_pvals": ch_pvals,
        "mask_type": mask_type,
        "n_resp": n_resp,
        "p_crit": p_crit,
        "stim_scores": stim_scores,
    }


def plot_spatiotemporal_projection_map(
    rates: np.ndarray,                  # (n_ch, n_trials, n_bins) STIM
    rates_ctrl: Optional[np.ndarray],     # (n_ch, n_ctrl_trials, n_bins) CONTROL
    time_ms: np.ndarray,                # (n_bins,)
    delta_x_traj: np.ndarray,           # (n_trials, n_bins)
    delta_y_traj: np.ndarray,           # (n_trials, n_bins)
    delta_z_traj: np.ndarray,           # (n_trials, n_bins)
    dx_s: np.ndarray,                   # (n_trials,)
    dy_s: np.ndarray,                   # (n_trials,)
    dz_s: np.ndarray,                   # (n_trials,)
    dr_s: np.ndarray,                   # (n_trials,)
    dt_s: np.ndarray,                   # (n_trials,)
    channel_ids: Sequence[int],
    channel_depths: Optional[Sequence[float]] = None,
    array_name: str = "NPRW",
    condition_title: str = "",
    stim_pulse_ms: float = 100.0,
    blank_window: Optional[Tuple[float, float]] = None,
    out_path: Optional[Path] = None,
) -> Tuple[plt.Figure, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Spatiotemporal Neural-Behavior Projection Map with Channel-Level Masking of
    Unresponsive Electrodes and Single-Trial Behavioral Prediction.

    Layout:
        3-row x 3-column consolidated figure:
        - Col 0: Spatiotemporal correlation heatmap r(c, t) with unresponsive channels masked out in light gray.
        - Col 1: Marginal tuning profile (horizontal bars for responsive channels showing Mean |r|).
        - Col 2: Single-trial perturbation prediction scatter plot from Stim Expression Score S_i
                 Row 1: S_i vs Lateral Endpoint Shift ΔX
                 Row 2: S_i vs Forward Endpoint Shift ΔY
                 Row 3: S_i vs Plateau Timing Delay Δt
    """
    n_ch, n_s, n_bins = rates.shape

    # 1. Compute Stim vs Control Channel Responsiveness
    if rates_ctrl is not None and rates_ctrl.shape[0] == n_ch and rates_ctrl.shape[2] == n_bins:
        stim_mod = compute_stim_vs_ctrl_modulation(rates, rates_ctrl, time_ms, blank_window=blank_window)
    else:
        stim_mod = {
            "resp_channels": np.ones(n_ch, dtype=bool),
            "ch_tvals": np.zeros(n_ch),
            "ch_pvals": np.ones(n_ch),
            "mask_type": "None",
            "n_resp": n_ch,
            "p_crit": 0.05,
            "stim_scores": np.zeros(n_s),
        }

    resp_channels = stim_mod["resp_channels"]
    stim_scores = stim_mod["stim_scores"]
    mask_type = stim_mod["mask_type"]
    n_resp = stim_mod["n_resp"]
    p_crit = stim_mod["p_crit"]

    components = [
        ("Lateral Perturbation Component ΔX(t)", "Lateral_dX", delta_x_traj, dx_s, "Lateral Shift ΔX (mm)", "#2980b9"),
        ("Forward Perturbation Component ΔY(t)", "Forward_dY", delta_y_traj, dy_s, "Forward Shift ΔY (mm)", "#e74c3c"),
        ("Height Perturbation Component ΔZ(t)", "Height_dZ", delta_z_traj, dt_s, "Plateau Timing Delay Δt (ms)", "#8e44ad"),
    ]

    fig = plt.figure(figsize=(18.0, 12.0), dpi=300)
    gs = gridspec.GridSpec(3, 3, width_ratios=[4.2, 0.9, 1.8], height_ratios=[1.0, 1.0, 1.0], hspace=0.34, wspace=0.22)

    summary_records: List[Dict[str, Any]] = []
    is_ua = "utah" in array_name.lower() or "ua" in array_name.lower()

    cmap = copy.copy(plt.get_cmap("RdBu_r"))
    cmap.set_bad(color="#dfe4ea")  # clean light gray for unresponsive channels

    for row_idx, (comp_title, comp_tag, beh_traj, beh_target, target_label, col_color) in enumerate(components):
        corr_map = np.full((n_ch, n_bins), np.nan, dtype=float)
        pval_map = np.ones((n_ch, n_bins), dtype=float)

        for c in range(n_ch):
            if not resp_channels[c]:
                continue  # Masked out!
            for t in range(n_bins):
                r_c = rates[c, :, t]
                b_t = beh_traj[:, t]
                valid = np.isfinite(r_c) & np.isfinite(b_t)
                if np.sum(valid) >= 4 and np.std(r_c[valid]) > 1e-6 and np.std(b_t[valid]) > 1e-6:
                    r_val, p_val = stats.pearsonr(r_c[valid], b_t[valid])
                    corr_map[c, t] = r_val if np.isfinite(r_val) else 0.0
                    pval_map[c, t] = p_val if np.isfinite(p_val) else 1.0

            ch_abs = np.abs(corr_map[c])
            best_t = int(np.nanargmax(ch_abs)) if np.any(np.isfinite(ch_abs)) else 0
            peak_r = float(corr_map[c, best_t]) if np.isfinite(corr_map[c, best_t]) else 0.0
            peak_lat = float(time_ms[best_t])
            p_peak = float(pval_map[c, best_t]) if np.isfinite(pval_map[c, best_t]) else 1.0
            sig_cnt = int(np.sum(pval_map[c] < P_VAL_THRESH))

            summary_records.append({
                "array": array_name,
                "channel": channel_ids[c] if c < len(channel_ids) else c + 1,
                "component": comp_tag,
                "is_responsive": bool(resp_channels[c]),
                "peak_r": peak_r,
                "peak_latency_ms": peak_lat,
                "p_val_at_peak": p_peak,
                "sig_bins_count": sig_cnt,
            })

        # Col 0: Main Heatmap (Channels x Time Bins)
        ax_map = fig.add_subplot(gs[row_idx, 0])
        vmax = max(0.40, float(np.nanpercentile(np.abs(corr_map[np.isfinite(corr_map)]), 99))) if np.any(np.isfinite(corr_map)) else 0.40
        im = ax_map.imshow(
            corr_map,
            aspect="auto",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            extent=[time_ms[0], time_ms[-1], n_ch + 0.5, 0.5],
            origin="upper",
            interpolation="nearest",
        )

        # Stim shading
        ax_map.axvspan(0.0, stim_pulse_ms, color="#f39c12", alpha=0.18, label="Stim Pulse")
        ax_map.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0)

        if is_ua:
            for boundary in [30.5, 62.5, 93.5]:
                ax_map.axhline(boundary, color="white", linewidth=1.2, linestyle="-")

            ax_map.set_yticks([15.5, 46.5, 78.0, 111.0])
            ax_map.set_yticklabels(["SMA (1-30)", "PMd (31-62)", "M1 inf (63-93)", "M1 sup (94-128)"], fontsize=8.5, fontweight="bold")
            ax_map.set_ylabel("Utah Array Region", fontsize=10, fontweight="bold")
        else:
            ax_map.set_yticks([1, 32, 64, 96, 128])
            ax_map.set_yticklabels([1, 32, 64, 96, 128], fontsize=9)
            ax_map.set_ylabel(f"{array_name} Electrode", fontsize=10, fontweight="bold")

        ax_map.set_title(f"{comp_title} | Responsive Channels (N={n_resp}/{n_ch}, Gray=Masked)", fontsize=11, fontweight="bold")
        if row_idx == 2:
            ax_map.set_xlabel("Time from Stim Onset (ms)", fontsize=11, fontweight="bold")
        else:
            ax_map.set_xticklabels([])

        # Colorbar
        cb = plt.colorbar(im, ax=ax_map, pad=0.018, fraction=0.028)
        cb.set_label("Pearson r", fontsize=9, fontweight="bold")

        # Col 1: Marginal Channel Tuning Profile (Horizontal Bars for Responsive Channels)
        ax_prof = fig.add_subplot(gs[row_idx, 1], sharey=ax_map)
        if is_ua:
            for boundary in [30.5, 62.5, 93.5]:
                ax_prof.axhline(boundary, color="#7f8c8d", linewidth=1.0, linestyle=":")
        mean_abs_r = np.nanmean(np.abs(corr_map), axis=1)
        ch_y = np.arange(1, n_ch + 1)
        if np.any(resp_channels):
            ax_prof.barh(ch_y[resp_channels], mean_abs_r[resp_channels], height=0.75, color="#2c3e50", alpha=0.85, label="Mean |r|")
        ax_prof.axvline(0.0, color="#7f8c8d", linestyle="--", linewidth=0.8)
        max_x = max(0.35, float(np.nanmax(mean_abs_r[resp_channels]) * 1.25)) if np.any(np.isfinite(mean_abs_r[resp_channels])) else 0.35
        ax_prof.set_xlim(-0.02, max_x)
        ax_prof.grid(True, linestyle="--", alpha=0.3)
        if row_idx == 2:
            ax_prof.set_xlabel("Mean |r|", fontsize=10, fontweight="bold")
        else:
            ax_prof.set_xticklabels([])
        ax_prof.tick_params(labelleft=False)
        if row_idx == 0:
            ax_prof.legend(loc="lower right", fontsize=8, frameon=True)

        # Col 2: Single-Trial Behavioral Prediction Scatter Plot (Only Responsive Channels)
        ax_pred = fig.add_subplot(gs[row_idx, 2])
        valid_pred = np.isfinite(stim_scores) & np.isfinite(beh_target)
        x_pts = stim_scores[valid_pred]
        y_pts = beh_target[valid_pred]
        n_pts = len(x_pts)

        if n_pts >= 4 and np.std(x_pts) > 1e-4 and np.std(y_pts) > 1e-4:
            r_pred, p_pred = stats.pearsonr(x_pts, y_pts)
            rho_pred, p_rho = stats.spearmanr(x_pts, y_pts)

            ax_pred.scatter(x_pts, y_pts, color=col_color, s=55, edgecolors="#2c3e50", linewidths=0.9, alpha=0.85, zorder=3)
            # OLS regression fit line & 95% confidence interval
            poly_fit = np.polyfit(x_pts, y_pts, 1)
            x_grid = np.linspace(np.min(x_pts), np.max(x_pts), 100)
            y_fit = np.polyval(poly_fit, x_grid)
            ax_pred.plot(x_grid, y_fit, color="#2c3e50", linewidth=1.8, linestyle="-", zorder=2)

            residuals = y_pts - np.polyval(poly_fit, x_pts)
            s_err = np.sqrt(np.sum(residuals ** 2) / max(n_pts - 2, 1))
            ci = 1.96 * s_err * np.sqrt(1.0 / n_pts + (x_grid - np.mean(x_pts)) ** 2 / max(np.sum((x_pts - np.mean(x_pts)) ** 2), 1e-6))
            ax_pred.fill_between(x_grid, y_fit - ci, y_fit + ci, color=col_color, alpha=0.15, zorder=1)

            star_s = "***" if p_pred < 0.001 else ("**" if p_pred < 0.01 else ("*" if p_pred < 0.05 else "n.s."))
            stat_txt = (
                f"Pearson r = {r_pred:+.2f} ({star_s}, p={p_pred:.3f})\n"
                f"Spearman ρ = {rho_pred:+.2f} (p={p_rho:.3f})\n"
                f"Trials N = {n_pts}\n"
                f"Predictor: Firing Rate of\n{n_resp}/{n_ch} Stim-Responsive Chs\n({mask_type})"
            )
        else:
            r_pred, p_pred, rho_pred, p_rho = 0.0, 1.0, 0.0, 1.0
            stat_txt = f"Insufficient variance / trials\n({n_pts} trials, {n_resp} resp channels)"

        ax_pred.text(
            0.05, 0.94, stat_txt, transform=ax_pred.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#bdc3c7", alpha=0.9),
            fontsize=8.5, fontweight="bold"
        )
        ax_pred.set_ylabel(target_label, fontsize=10, fontweight="bold")
        ax_pred.set_title(f"Single-Trial Prediction ({target_label.split('(')[0].strip()})", fontsize=11, fontweight="bold")
        ax_pred.grid(True, linestyle="--", alpha=0.4)
        if row_idx == 2:
            ax_pred.set_xlabel("Single-Trial Stim Score $S_i$ (a.u.)", fontsize=10, fontweight="bold")
        else:
            ax_pred.set_xticklabels([])

    fig.suptitle(f"{array_name} Channel-Masked Spatiotemporal Map & Behavioral Prediction: {condition_title}",
                 fontsize=14, fontweight="bold", y=0.99)

    # Centered global legend below title
    legend_elements = [
        Patch(facecolor="#dfe4ea", edgecolor="#999999", label=f"Unresponsive Channels (Masked Out, {n_ch - n_resp} ch)"),
        Patch(facecolor="#e74c3c", edgecolor="#2c3e50", label=f"Stim-Responsive Channels ({n_resp}/{n_ch} ch, {mask_type})"),
        Patch(facecolor="#f39c12", alpha=0.3, label=f"Stimulation ({stim_pulse_ms:.0f} ms)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.965),
               frameon=True, fontsize=9.5)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if SAVE_PNG:
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
        if SAVE_SVG:
            svg_path = out_path.with_suffix(".svg")
            fig.savefig(svg_path, bbox_inches="tight")
        _log(f"Saved spatiotemporal projection map: {out_path.name}")

    # Compute overall prediction record
    def _corr_pair(x, y):
        val = np.isfinite(x) & np.isfinite(y)
        if np.sum(val) >= 4 and np.std(x[val]) > 1e-4 and np.std(y[val]) > 1e-4:
            rp, pp = stats.pearsonr(x[val], y[val])
            rs, ps = stats.spearmanr(x[val], y[val])
            return float(rp), float(pp), float(rs), float(ps)
        return 0.0, 1.0, 0.0, 1.0

    r_dx, p_dx, rho_dx, ps_dx = _corr_pair(stim_scores, dx_s)
    r_dy, p_dy, rho_dy, ps_dy = _corr_pair(stim_scores, dy_s)
    r_dz, p_dz, rho_dz, ps_dz = _corr_pair(stim_scores, dz_s)
    r_dr, p_dr, rho_dr, ps_dr = _corr_pair(stim_scores, dr_s)
    r_dt, p_dt, rho_dt, ps_dt = _corr_pair(stim_scores, dt_s)

    pred_metrics = {
        "array": array_name,
        "condition": condition_title,
        "n_stim_trials": n_s,
        "n_ctrl_trials": rates_ctrl.shape[1] if rates_ctrl is not None else 0,
        "mask_type": mask_type,
        "n_resp_channels": n_resp,
        "n_total_channels": n_ch,
        "pct_resp_channels": float(100.0 * n_resp / max(n_ch, 1)),
        "p_crit": p_crit,
        "r_lateral_dx": r_dx,
        "p_lateral_dx": p_dx,
        "rho_lateral_dx": rho_dx,
        "r_forward_dy": r_dy,
        "p_forward_dy": p_dy,
        "rho_forward_dy": rho_dy,
        "r_height_dz": r_dz,
        "p_height_dz": p_dz,
        "r_3d_total_dr": r_dr,
        "p_3d_total_dr": p_dr,
        "r_plateau_dt": r_dt,
        "p_plateau_dt": p_dt,
        "rho_plateau_dt": rho_dt,
    }

    return fig, summary_records, pred_metrics


# =============================================================================
# Main Pipeline Processing Single Session
# =============================================================================

def process_condition(
    stim_path: Path,
    ctrl_path: Path,
    session_name: str,
    out_dir: Path,
    dlc_3d_dir: Optional[Path] = None,
    session_landmarks: Optional[Dict[str, np.ndarray]] = None,
    meta_csv: Optional[Path] = None,
    aligned_ckpt_dir: Optional[Path] = None,
    behv_ckpt_dir: Optional[Path] = None,
    df_plat: Optional[pd.DataFrame] = None,
    elec_info: Optional[Dict[int, Dict[str, Any]]] = None,
    bad_channels: Optional[Dict[str, set]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Processes one STIM condition and pairs it with target-matched CONTROL."""
    _log(f"Processing condition: {stim_path.name}")
    _log(f"  Matched control: {ctrl_path.name}")

    with np.load(stim_path, allow_pickle=True) as sd, np.load(ctrl_path, allow_pickle=True) as cd:
        stim_ev = safe_get_npz(sd, "event_ms")
        if stim_ev is None:
            stim_ev = safe_get_npz(sd, "event_timestamps_ms")
        ctrl_ev = safe_get_npz(cd, "event_ms")
        if ctrl_ev is None:
            ctrl_ev = safe_get_npz(cd, "event_timestamps_ms")
        stim_dur_ms = get_stim_duration_ms(sd)

        # Neural arrays
        s_nprw = safe_get_npz(sd, "NPRW_counts")
        if s_nprw is None:
            s_nprw = safe_get_npz(sd, "nprw_counts")
        c_nprw = safe_get_npz(cd, "NPRW_counts")
        if c_nprw is None:
            c_nprw = safe_get_npz(cd, "nprw_counts")
        s_nprw_t = safe_get_npz(sd, "NPRW_rel_t")
        if s_nprw_t is None:
            s_nprw_t = safe_get_npz(sd, "nprw_t")
        c_nprw_t = safe_get_npz(cd, "NPRW_rel_t")
        if c_nprw_t is None:
            c_nprw_t = safe_get_npz(cd, "nprw_t")

        s_ua = safe_get_npz(sd, "UA_counts")
        c_ua = safe_get_npz(cd, "UA_counts")
        s_ua_t = safe_get_npz(sd, "UA_rel_t")
        if s_ua_t is None:
            s_ua_t = safe_get_npz(sd, "UA_t")
        c_ua_t = safe_get_npz(cd, "UA_rel_t")
        if c_ua_t is None:
            c_ua_t = safe_get_npz(cd, "UA_t")

        stim_tids = safe_get_npz(sd, "raw_trial_indices", safe_get_npz(sd, "stim_trial_ids", np.arange(len(stim_ev)) if stim_ev is not None else np.array([])))
        ctrl_tids = safe_get_npz(cd, "raw_trial_indices", safe_get_npz(cd, "ctrl_trial_ids", np.arange(len(ctrl_ev)) if ctrl_ev is not None else np.array([])))

    target = "Target_B" if "target_b" in stim_path.stem.lower() else "Target_A"
    cond_title = f"{stim_path.stem} ({target})"

    # 1. Ingest 3D Kinematics
    stim_kin = load_condition_3d_task_kinematics(
        npz_path=stim_path,
        session_name=session_name,
        dlc_3d_dir=dlc_3d_dir,
        session_landmarks=session_landmarks,
        meta_csv=meta_csv,
        aligned_ckpt_dir=aligned_ckpt_dir,
        behv_ckpt_dir=behv_ckpt_dir,
        raw_trial_indices=stim_tids,
    )
    ctrl_kin = load_condition_3d_task_kinematics(
        npz_path=ctrl_path,
        session_name=session_name,
        dlc_3d_dir=dlc_3d_dir,
        session_landmarks=session_landmarks,
        meta_csv=meta_csv,
        aligned_ckpt_dir=aligned_ckpt_dir,
        behv_ckpt_dir=behv_ckpt_dir,
        raw_trial_indices=ctrl_tids,
    )

    if stim_kin is None or ctrl_kin is None:
        _log(f"  [Skip] 3D kinematics unavailable for {stim_path.name}")
        return None

    s_valid = stim_kin["valid_mask"]
    c_valid = ctrl_kin["valid_mask"]
    s_coords = stim_kin["coords"]  # (n_s, 3, n_time)
    c_coords = ctrl_kin["coords"]  # (n_c, 3, n_time)
    beh_t = stim_kin["beh_t"]

    stim_br = extract_br_index(stim_path)
    ctrl_br = extract_br_index(ctrl_path)
    target_str = "B" if "target_b" in stim_path.stem.lower() else "A"

    plat_s_table = extract_plateau_times_from_table(df_plat, stim_br, target_str, s_coords.shape[0])
    plat_c_table = extract_plateau_times_from_table(df_plat, ctrl_br, target_str, c_coords.shape[0])

    endpoints_s, plateau_t_s = compute_trial_endpoints_3d(s_coords, beh_t, override_plateau_times=plat_s_table)
    endpoints_c, plateau_t_c = compute_trial_endpoints_3d(c_coords, beh_t, override_plateau_times=plat_c_table)

    # 2. Resample 3D Kinematics onto Uniform 20-ms Neural Grid
    t_min, t_max = WIN_PLOT_MS
    time_ms = np.arange(t_min + BIN_WIDTH_MS * 0.5, t_max, BIN_WIDTH_MS)
    n_bins = len(time_ms)

    s_coords_rebinned = np.full((s_coords.shape[0], 3, n_bins), np.nan, dtype=float)
    c_coords_rebinned = np.full((c_coords.shape[0], 3, n_bins), np.nan, dtype=float)

    for i in range(s_coords.shape[0]):
        for d in range(3):
            val = np.isfinite(s_coords[i, d])
            if np.sum(val) > 1:
                s_coords_rebinned[i, d] = np.interp(time_ms, beh_t[val], s_coords[i, d, val], left=np.nan, right=np.nan)

    for i in range(c_coords.shape[0]):
        for d in range(3):
            val = np.isfinite(c_coords[i, d])
            if np.sum(val) > 1:
                c_coords_rebinned[i, d] = np.interp(time_ms, beh_t[val], c_coords[i, d, val], left=np.nan, right=np.nan)

    # Median Control Baseline Trajectory & Single-Trial Perturbation Shifts
    ctrl_ref_traj = np.nanmedian(c_coords_rebinned, axis=0)  # (3, n_bins)
    delta_s_traj = s_coords_rebinned - ctrl_ref_traj[None, :, :]  # (n_s, 3, n_bins)

    c_ref = np.nanmedian(endpoints_c, axis=0)
    dx_s = endpoints_s[:, 0] - c_ref[0]
    dy_s = endpoints_s[:, 1] - c_ref[1]
    dz_s = endpoints_s[:, 2] - c_ref[2]
    dr_s = np.sqrt(dx_s ** 2 + dy_s ** 2 + dz_s ** 2)
    dt_s = plateau_t_s - np.nanmedian(plateau_t_c)

    # Generate Behavioral Perturbation Summary Figure
    base_name = stim_path.stem
    fig_beh = plot_behavioral_perturbation_summary(
        beh_t=time_ms,
        delta_traces=delta_s_traj,
        endpoints_s=endpoints_s,
        endpoints_c=endpoints_c,
        plateau_t_s=plateau_t_s,
        plateau_t_c=plateau_t_c,
        stim_pulse_ms=stim_dur_ms,
        condition_title=cond_title,
        out_path=out_dir / f"{base_name}_behavioral_perturbation_summary.png",
    )
    plt.close(fig_beh)

    # 3. Spatiotemporal Neural-Behavior Mapping & Prediction
    all_summary_rows = []
    all_pred_rows: List[Dict[str, Any]] = []

    # A. Cerebellum: NPRW (128 Channels)
    if s_nprw is not None and s_nprw_t is not None and c_nprw is not None and c_nprw_t is not None:
        s_nprw_3d = prepare_counts_3d(s_nprw)
        c_nprw_3d = prepare_counts_3d(c_nprw)
        if s_nprw_3d is not None and c_nprw_3d is not None and s_nprw_3d.shape[1] == s_coords.shape[0]:
            rates_nprw_s, _ = rebin_and_smooth(
                s_nprw_3d, s_nprw_t, target_bin_w=BIN_WIDTH_MS, smooth_sigma_ms=SMOOTH_SIGMA_MS,
                win_ms=WIN_PLOT_MS, blank_win=(0.0, stim_dur_ms + NPRW_BLANK_POST_MS)
            )
            rates_nprw_c, _ = rebin_and_smooth(
                c_nprw_3d, c_nprw_t, target_bin_w=BIN_WIDTH_MS, smooth_sigma_ms=SMOOTH_SIGMA_MS,
                win_ms=WIN_PLOT_MS, blank_win=(0.0, stim_dur_ms + NPRW_BLANK_POST_MS)
            )

            fig_nprw, recs_nprw, pred_nprw = plot_spatiotemporal_projection_map(
                rates=rates_nprw_s,
                rates_ctrl=rates_nprw_c,
                time_ms=time_ms,
                delta_x_traj=delta_s_traj[:, 0, :],
                delta_y_traj=delta_s_traj[:, 1, :],
                delta_z_traj=delta_s_traj[:, 2, :],
                dx_s=dx_s,
                dy_s=dy_s,
                dz_s=dz_s,
                dr_s=dr_s,
                dt_s=dt_s,
                channel_ids=np.arange(1, rates_nprw_s.shape[0] + 1),
                array_name="NPRW",
                condition_title=cond_title,
                stim_pulse_ms=stim_dur_ms,
                blank_window=(0.0, stim_dur_ms + NPRW_BLANK_POST_MS),
                out_path=out_dir / f"{base_name}_spatiotemporal_projection_NPRW.png",
            )
            plt.close(fig_nprw)
            all_summary_rows.extend(recs_nprw)
            all_pred_rows.append(pred_nprw)

    # B. Motor Cortex: Utah Array (128 Channels)
    if s_ua is not None and s_ua_t is not None and c_ua is not None and c_ua_t is not None:
        s_ua_3d = prepare_counts_3d(s_ua)
        c_ua_3d = prepare_counts_3d(c_ua)
        if s_ua_3d is not None and c_ua_3d is not None and s_ua_3d.shape[1] == s_coords.shape[0]:
            rates_ua_s, _ = rebin_and_smooth(
                s_ua_3d, s_ua_t, target_bin_w=BIN_WIDTH_MS, smooth_sigma_ms=SMOOTH_SIGMA_MS,
                win_ms=WIN_PLOT_MS, blank_win=(0.0, stim_dur_ms) if BLANK_UA_STIM_PERIOD else None
            )
            rates_ua_c, _ = rebin_and_smooth(
                c_ua_3d, c_ua_t, target_bin_w=BIN_WIDTH_MS, smooth_sigma_ms=SMOOTH_SIGMA_MS,
                win_ms=WIN_PLOT_MS, blank_win=(0.0, stim_dur_ms) if BLANK_UA_STIM_PERIOD else None
            )

            fig_ua, recs_ua, pred_ua = plot_spatiotemporal_projection_map(
                rates=rates_ua_s,
                rates_ctrl=rates_ua_c,
                time_ms=time_ms,
                delta_x_traj=delta_s_traj[:, 0, :],
                delta_y_traj=delta_s_traj[:, 1, :],
                delta_z_traj=delta_s_traj[:, 2, :],
                dx_s=dx_s,
                dy_s=dy_s,
                dz_s=dz_s,
                dr_s=dr_s,
                dt_s=dt_s,
                channel_ids=np.arange(1, rates_ua_s.shape[0] + 1),
                array_name="Utah_Array",
                condition_title=cond_title,
                stim_pulse_ms=stim_dur_ms,
                blank_window=(0.0, stim_dur_ms) if BLANK_UA_STIM_PERIOD else None,
                out_path=out_dir / f"{base_name}_spatiotemporal_projection_UA.png",
            )
            plt.close(fig_ua)
            all_summary_rows.extend(recs_ua)
            all_pred_rows.append(pred_ua)

    # Save summary tables
    if all_summary_rows:
        df_sum = pd.DataFrame(all_summary_rows)
        csv_path = out_dir / f"{base_name}_spatiotemporal_neural_behavior_summary.csv"
        df_sum.to_csv(csv_path, index=False)
        _log(f"Saved spatiotemporal tuning table: {csv_path.name}")

    if all_pred_rows:
        df_pred = pd.DataFrame(all_pred_rows)
        pred_csv = out_dir / f"{base_name}_stim_behavior_prediction_summary.csv"
        df_pred.to_csv(pred_csv, index=False)
        _log(f"Saved stim behavior prediction summary: {pred_csv.name}")

    return all_pred_rows


def run_spatiotemporal_perturbation_pipeline(
    session: str,
    peri_root: Path,
    out_dir: Path,
    sess_dir: Optional[Path] = None,
    dlc_3d_dir: Optional[Path] = None,
    process_only: Optional[List[int]] = None,
):
    ensure_dir(out_dir)
    all_files = sorted(list(peri_root.rglob("*.npz")))
    ref_map = load_reference_mapping(all_files)

    stim_files = [f for f in all_files if f in ref_map]

    if process_only:
        filtered = []
        for f in stim_files:
            br_i = extract_br_index(f)
            if br_i in process_only:
                filtered.append(f)
        stim_files = filtered

    _log(f"Discovered {len(stim_files)} STIM condition(s) for session {session}.")
    if sess_dir is None:
        sess_dir = peri_root.parent.parent

    df_plat = load_plateau_analysis_table(session, sess_dir)
    meta_csv = p3d.find_metadata_csv(session=session, metadata_csv=sess_dir, peri_root=peri_root) if (HAS_3D_KINEMATICS and p3d) else None
    if meta_csv is not None and not Path(meta_csv).is_file():
        meta_csv = None
    aligned_ckpt = sess_dir / "results" / "checkpoints" / "aligned_dlc_br"
    behv_ckpt = sess_dir / "results" / "checkpoints" / "behavior"

    session_landmarks = None
    if dlc_3d_dir and dlc_3d_dir.exists() and HAS_3D_KINEMATICS and p3d:
        try:
            session_landmarks = p3d.scan_session_landmarks(dlc_3d_dir, square_size_mm=SQUARE_SIZE_MM)
        except Exception as e:
            _log(f"Session landmarks scan skipped: {e}")

    session_preds: List[Dict[str, Any]] = []

    for stim_path in stim_files:
        ctrl_path = ref_map[stim_path]
        try:
            cond_preds = process_condition(
                stim_path=stim_path,
                ctrl_path=ctrl_path,
                session_name=session,
                out_dir=out_dir,
                dlc_3d_dir=dlc_3d_dir,
                session_landmarks=session_landmarks,
                meta_csv=meta_csv,
                aligned_ckpt_dir=aligned_ckpt if aligned_ckpt.exists() else None,
                behv_ckpt_dir=behv_ckpt if behv_ckpt.exists() else None,
                df_plat=df_plat,
            )
            if cond_preds:
                session_preds.extend(cond_preds)
        except Exception as e:
            _log(f"Error processing {stim_path.name}: {e}")
            traceback.print_exc()

    if session_preds:
        df_sess_pred = pd.DataFrame(session_preds)
        sess_pred_csv = out_dir / f"{session}_session_stim_behavior_prediction_summary.csv"
        df_sess_pred.to_csv(sess_pred_csv, index=False)
        _log(f"Saved session-level stim prediction summary: {sess_pred_csv.name}")

    _log("Spatiotemporal perturbation pipeline complete.")


# =============================================================================
# CLI Entrypoint
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Spatiotemporal Neural-Behavioral Perturbation Analysis")
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--process-only", nargs="+", type=int, default=None)
    parser.add_argument("--svg", action="store_true", default=False, help="Save SVG vector plots in addition to PNG")
    args = parser.parse_args()

    global SAVE_SVG
    if args.svg:
        SAVE_SVG = True

    session_name = args.session or os.environ.get("RCP_SESSION", "NRR_RW012")
    process_only = args.process_only
    if process_only is None and os.environ.get("RCP_PROCESS_ONLY"):
        try:
            val = os.environ["RCP_PROCESS_ONLY"]
            process_only = json.loads(val) if val.startswith("[") else [int(x) for x in val.split(",")]
        except Exception:
            process_only = None

    # Resolve Session Directory
    sess_dir = p3d.resolve_session_dir(session=session_name) if (HAS_3D_KINEMATICS and p3d) else None
    if sess_dir is None:
        base_nike = Path(r"\\cullenstor.win.ad.jhu.edu\data\Current Project Databases - NHP\2025 Cerebellum prosthesis\Nike")
        matches = [d for d in base_nike.glob(f"*{session_name}*") if d.is_dir() and not d.name.startswith(".")]
        if matches:
            sess_dir = matches[0]
        else:
            sess_dir = base_nike / session_name

    peri_root = sess_dir / "results" / "checkpoints" / "PeriStim"
    if not peri_root.exists() and (sess_dir / "checkpoints" / "PeriStim").exists():
        peri_root = sess_dir / "checkpoints" / "PeriStim"

    dlc_3d_dir = None
    if (sess_dir / "Video" / "DLC_3D_new").exists():
        dlc_3d_dir = sess_dir / "Video" / "DLC_3D_new"
    elif (sess_dir / "Video" / "DLC_3D").exists():
        dlc_3d_dir = sess_dir / "Video" / "DLC_3D"
    elif (sess_dir / "DLC_3D").exists():
        dlc_3d_dir = sess_dir / "DLC_3D"

    out_dir = sess_dir / "results" / "figures" / "channel_behavior_regression"

    _log(f"Starting Spatiotemporal Perturbation Analysis for {session_name}")
    _log(f"  Session Dir: {sess_dir}")
    _log(f"  PeriStim:    {peri_root}")
    _log(f"  DLC 3D Dir:  {dlc_3d_dir}")
    _log(f"  Out Dir:     {out_dir}")
    _log(f"  Filter:      {process_only}")

    run_spatiotemporal_perturbation_pipeline(
        session=session_name,
        peri_root=peri_root,
        out_dir=out_dir,
        sess_dir=sess_dir,
        dlc_3d_dir=dlc_3d_dir,
        process_only=process_only,
    )


if __name__ == "__main__":
    main()
