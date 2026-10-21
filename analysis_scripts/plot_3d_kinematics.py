"""
plot_3d_kinematics.py
=====================

Batch analysis of 3D reaching kinematics and trajectory variability using
triangulated two-camera DeepLabCut (DLC_3D) data.

Scientific Goal:
    Evaluate reach trajectory dynamics, 3D normalized distance profiles, and
    trial-to-trial trajectory variability across stimulation and control conditions
    using true 3D triangulated coordinates (X, Y, Z).

Key Workflow:
    1. Condition Discovery:
       - Discovers conditions from PeriStim NPZ outputs for target A and/or B.
       - Maps Blackrock index (BR_File) to video recording index using session metadata.
    2. Continuous 3D DLC Loading:
       - Loads continuous multi-camera triangulated 3D CSV/H5 files from:
         <SESSION_LOC>/Video/DLC_3D_new/ (or configured 3D folder).
       - Body parts: index, middle, hand, wrist, start, TA, TB (default keypoint: 'middle').
       - Interpolates small NaN gaps (<= 4 consecutive frames).
    3. Trial-by-Trial 3D Reach Segmentation ("Cutting"):
       - Uses continuous frame timing (beh_t_ms from Aligned checkpoints or
         ns5_sample from both_cams_aligned.csv).
       - Extracts per-trial 3D reaching windows around event timestamps (event_ms_sub)
         onto a uniform relative time grid (e.g. -800 to +600 ms, or -500 to +500 ms).
    4. 3D Kinematics & Signal Processing:
       - Applies Butterworth low-pass filter (order 4, cutoff 8.0 Hz) to X, Y, Z coordinates.
       - Computes 3D Euclidean distance from starting baseline rest position (e.g. at -600 ms):
         d(t) = sqrt((x(t)-x0)^2 + (y(t)-y0)^2 + (z(t)-z0)^2).
       - Normalizes distance trajectories (0 to 1) and computes 3D velocity profiles.
    5. 3D Trajectory Variability:
       - Computes time-resolved pairwise 3D Euclidean distances between reaches:
         d_ij(t) = sqrt((xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2).
       - Computes path-progress normalized 3D pairwise variability (resampling across 0-100% path length).
    6. High-Quality Visualizations:
       - Consolidated Variability & Normalized Distance Comparison (2 panels):
         * Panel 1: Time-resolved pairwise 3D variability (mm) with +/- 1 SD shading.
         * Panel 2: Kinematics 3D normalized distance with +/- 1 SD shading.
         * Formatted with Control in orange dashed line, stim conditions in 'bone' gradient.
       - Multi-Projection 3D Trajectory Overlay (4 panels: 3D Trajectories, X-Y Projection, X-Z Projection, Y-Z Projection).
       - Dedicated 2D X-Y Trajectory Overlay (matching classic plot_plateau_analysis layout).
       - Per-Condition Comprehensive Diagnostic Figures.
"""

from __future__ import annotations

import os
import sys
import re
import json
import argparse
import traceback
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any, Sequence

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter
from scipy import stats

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Repo Setup & Configuration Loading
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import RCP_analysis as rcp
from RCP_analysis.python.functions.kinematics_utils import (
    butterworth_filter,
    mean_ci95_trace_summary,
    value_at_ref_time,
)

try:
    from RCP_analysis.python.functions.config_loading import *  # noqa: F401,F403
    CONFIG_LOADED = True
except Exception as e:
    CONFIG_LOADED = False
    PARAMS = None
    SESSION_LOC = None
    VIDEO_ROOT = None
    PERI_ROOT = None
    OUT_BASE = None
    METADATA_CSV = None
    BEHV_CKPT_ROOT = None
    ALIGNED_CKPT_ROOT = None

# =============================================================================
# Analysis & Visualization Parameters
# =============================================================================

# Keypoint to analyze
DEFAULT_KEYPOINT: str = "middle"

# Time window relative to stimulation onset / IR beam crossing (t=0)
PRE_ZERO_MS: float = -800.0  # extraction & filter warm-up buffer
POST_ZERO_MS: float = 600.0
STEP_MS: float = 10.0  # sampling step for reach segments (ms)

# =============================================================================
# Master Plotting Time Window (ms)
# =============================================================================
# Single global master time window used for ALL plots across the entire script:
# 3D trajectories, 2D projections, X-Y overlays, interactive Plotly, normalized distance,
# and trajectory variability summaries.
# e.g. (-300.0, 400.0) plots from -300 ms to +400 ms.
# e.g. (0.0, 400.0) restricts all plots and metrics to post-0 in time (0 to 400 ms).
PLOT_WINDOW_MS: Tuple[float, float] = (-300.0, 400.0)

# Backward-compatibility aliases
TRAJ_PLOT_START_MS: float = PLOT_WINDOW_MS[0] if PLOT_WINDOW_MS[0] is not None else 0.0
TRAJ_PLOT_END_MS: float = PLOT_WINDOW_MS[1] if PLOT_WINDOW_MS[1] is not None else 400.0
TRAJECTORY_WINDOW_MS: Tuple[float, float] = PLOT_WINDOW_MS
ONLY_POST_ZERO: bool = bool(PLOT_WINDOW_MS[0] is not None and PLOT_WINDOW_MS[0] >= 0.0)
POST_ZERO_START_MS: float = 0.0

# Metric Scaling & Task Frame Alignment
SQUARE_SIZE_MM: float = 1.0  # 3D DLC coordinates are already in mm (1.0 mm/unit)
ALIGN_TO_TASK_FRAME: bool = True  # Rotate camera frame into Forward/Lateral/Up

# Starting rest position reference time for distance calculation (matches extract_peri_stim & kinematics_utils)
DISTANCE_REFERENCE_TIME_MS: float = -600.0

# Butterworth low-pass filter settings
BUTTER_CUTOFF_HZ: float = 8.0
BUTTER_ORDER: int = 4

PULSE_WIDTH_MS: float = 2.5

# Quality gating
MAX_NAN_FRACTION_PER_TRIAL: float = 0.35  # drop trial if >35% missing in reach window
MIN_REACHES_PER_CONDITION: int = 3

# Trajectory variability settings (driven by master PLOT_WINDOW_MS)
TRAJECTORY_VARIABILITY_CONFIG: Dict[str, Any] = {
    "min_reaches": 3,
    "time_summary_window_ms": PLOT_WINDOW_MS,
    "n_progress_points": 101,
    "progress_summary_window": (0.05, 0.95),
    "pairwise_agg": "median",  # "median" or "mean"
    "trajectory_alpha": 0.40,
    "trajectory_linewidth": 1.0,
}

# Figure output settings
FIG_FORMAT: str = "svg"
FIG_DPI: int = 300
PLOT_EACH_CONDITION: bool = True
SAVE_FIGS: bool = True

plt.rcParams.update(
    {
        "font.size": 10,
        "figure.figsize": (10, 6),
        "axes.grid": False,
    }
)


# =============================================================================
# Helper Utilities
# =============================================================================

def _log(msg: str) -> None:
    print(f"[3d_kinematics] {msg}")


def slice_trajectory_to_window(
    x: np.ndarray,
    y: np.ndarray,
    z: Optional[np.ndarray],
    time_arr: np.ndarray,
    window: Optional[Tuple[Optional[float], Optional[float]]] = TRAJECTORY_WINDOW_MS,
    row: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Slice trajectory traces to the specified time window (e.g. 0 to 400 ms).
    If window start is negative (e.g. -200 ms) and row has full buffers, automatically draws from full buffer.
    """
    if window is None or time_arr is None or len(time_arr) == 0:
        return x, y, z, time_arr

    t_min, t_max = window

    # If t_min is negative and full-window buffers are available, use full-window data
    if t_min is not None and t_min < 0 and row is not None and "time_full" in row:
        time_arr = row["time_full"]
        x = row.get("x_full", x)
        y = row.get("y_full", y)
        if z is not None and "z_full" in row:
            z = row["z_full"]

    mask = np.ones(len(time_arr), dtype=bool)
    if t_min is not None:
        mask &= (time_arr >= t_min)
    if t_max is not None:
        mask &= (time_arr <= t_max)

    if not np.any(mask):
        return x, y, z, time_arr

    x_s = x[:, mask] if x.ndim == 2 else x[mask]
    y_s = y[:, mask] if y.ndim == 2 else y[mask]
    z_s = (z[:, mask] if z.ndim == 2 else z[mask]) if z is not None else None
    t_s = time_arr[mask]
    return x_s, y_s, z_s, t_s


_SESSION_LANDMARKS_CACHE: Dict[Tuple[Path, float], Dict[str, np.ndarray]] = {}


def scan_session_landmarks(
    dlc_3d_dir: Path,
    square_size_mm: float = SQUARE_SIZE_MM,
) -> Dict[str, np.ndarray]:
    """
    Scan all 3D DLC CSV/H5 files in dlc_3d_dir to extract robust session-wide
    median positions (in millimeters) for static cage landmarks ('start', 'TA', 'TB').
    Caches results per directory to avoid redundant disk I/O.
    """
    dlc_3d_dir = Path(dlc_3d_dir)
    cache_key = (dlc_3d_dir, float(square_size_mm))
    if cache_key in _SESSION_LANDMARKS_CACHE:
        return _SESSION_LANDMARKS_CACHE[cache_key]

    landmarks: Dict[str, List[List[float]]] = {"start": [], "TA": [], "TB": []}
    candidate_files = sorted(list(dlc_3d_dir.glob("*DLC_3D*.csv"))) + sorted(list(dlc_3d_dir.glob("*DLC_3D*.h5")))

    for f in candidate_files:
        try:
            if f.suffix.lower() == ".h5":
                df_sub = pd.read_hdf(str(f))
            else:
                df_sub = pd.read_csv(f, header=[0, 1, 2], index_col=0, nrows=200)

            for col in df_sub.columns:
                bp = str(col[1]).strip() if len(col) >= 3 else str(col[0]).strip()
                c = str(col[2]).strip().lower() if len(col) >= 3 else str(col[1]).strip().lower()
                if bp in landmarks and c == "x":
                    y_col = (col[0], col[1], "y") if len(col) >= 3 else (col[0], "y")
                    z_col = (col[0], col[1], "z") if len(col) >= 3 else (col[0], "z")
                    if y_col in df_sub.columns and z_col in df_sub.columns:
                        x = pd.to_numeric(df_sub[col], errors="coerce").dropna().values
                        y = pd.to_numeric(df_sub[y_col], errors="coerce").dropna().values
                        z = pd.to_numeric(df_sub[z_col], errors="coerce").dropna().values
                        if len(x) > 0 and len(y) > 0 and len(z) > 0:
                            landmarks[bp].append([float(np.median(x)), float(np.median(y)), float(np.median(z))])
        except Exception:
            pass

    scale = square_size_mm if (square_size_mm and square_size_mm > 0) else 1.0
    res: Dict[str, np.ndarray] = {}
    for bp, vals in landmarks.items():
        if len(vals) > 0:
            res[bp] = np.median(vals, axis=0) * scale

    _SESSION_LANDMARKS_CACHE[cache_key] = res
    if res:
        summary_str = ", ".join(f"{k}: [{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}]" for k, v in res.items())
        _log(f"Derived Session Landmarks (mm): {summary_str}")
    return res


def compute_task_rotation_matrix(
    dlc_df: pd.DataFrame,
    session_landmarks: Optional[Dict[str, np.ndarray]] = None,
    square_size_mm: float = SQUARE_SIZE_MM,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derives an orthonormal basis R_task = [u_lat, u_fwd, u_up] and origin position p_start
    in true physical millimeters, rotating coordinates from the oblique camera frame
    into the monkey's task frame.

    Axes:
        X_task (Lateral): Left <-> Right axis (centered at rest bar, bifurcating left for A, right for B).
        Y_task (Forward): Monotonically forward reaching axis (0 mm at rest bar to ~210-250 mm at button).
        Z_task (Vertical Height): Vertical clearance axis (0 mm at rest bar, positive upward).

    Transformation:
        [X_task, Y_task, Z_task]^T = R_task.T @ ([X_cam, Y_cam, Z_cam]^T - p_start)
    """
    p_start = None
    p_ta = None
    p_tb = None

    # 1. Attempt extracting landmark medians from current file (ensuring millimeters)
    if "start_x" in dlc_df.columns:
        s_vals = dlc_df[["start_x", "start_y", "start_z"]].dropna().values
        if len(s_vals) > 0:
            med_s = np.median(s_vals, axis=0)
            if np.linalg.norm(med_s) < 200.0 and square_size_mm > 1.0:
                med_s = med_s * square_size_mm
            p_start = med_s

    if "TA_x" in dlc_df.columns:
        ta_vals = dlc_df[["TA_x", "TA_y", "TA_z"]].dropna().values
        if len(ta_vals) > 0:
            med_ta = np.median(ta_vals, axis=0)
            if np.linalg.norm(med_ta) < 200.0 and square_size_mm > 1.0:
                med_ta = med_ta * square_size_mm
            p_ta = med_ta

    if "TB_x" in dlc_df.columns:
        tb_vals = dlc_df[["TB_x", "TB_y", "TB_z"]].dropna().values
        if len(tb_vals) > 0:
            med_tb = np.median(tb_vals, axis=0)
            if np.linalg.norm(med_tb) < 200.0 and square_size_mm > 1.0:
                med_tb = med_tb * square_size_mm
            p_tb = med_tb

    # 2. Fall back to precomputed session landmarks if occluded/missing in this file
    if session_landmarks:
        if p_start is None and "start" in session_landmarks:
            p_start = session_landmarks["start"].copy()
        if p_ta is None and "TA" in session_landmarks:
            p_ta = session_landmarks["TA"].copy()
        if p_tb is None and "TB" in session_landmarks:
            p_tb = session_landmarks["TB"].copy()

    # Nominal fallback defaults if landmarks are completely absent
    if p_start is None:
        p_start = np.array([1040.0, 110.0, 18.0])
    if p_ta is None:
        p_ta = np.array([780.0, 108.0, 158.0])

    # 3. Forward unit vector (u_fwd)
    # If both TA and TB exist, target center aligns forward axis symmetrically between them
    if p_ta is not None and p_tb is not None:
        p_target = (p_ta + p_tb) / 2.0
    elif p_ta is not None:
        p_target = p_ta
    elif p_tb is not None:
        p_target = p_tb
    else:
        p_target = p_start + np.array([-260.0, 0.0, 140.0])

    diff_fwd = p_target - p_start
    norm_fwd = np.linalg.norm(diff_fwd)
    u_fwd = diff_fwd / norm_fwd if norm_fwd > 1e-6 else np.array([-0.88, 0.0, 0.47])

    # 4. Up unit vector (u_up)
    # Camera +Y points downwards in rectified camera frame, so -Y_cam = [0, -1, 0] is approx Up in world
    neg_y_cam = np.array([0.0, -1.0, 0.0])
    v_up = neg_y_cam - np.dot(neg_y_cam, u_fwd) * u_fwd
    norm_up = np.linalg.norm(v_up)
    if norm_up < 1e-4:
        v_up = np.array([0.0, 0.0, 1.0]) - np.dot(np.array([0.0, 0.0, 1.0]), u_fwd) * u_fwd
        norm_up = np.linalg.norm(v_up)
    u_up = v_up / norm_up if norm_up > 1e-6 else np.array([0.0, -1.0, 0.0])

    # 5. Lateral unit vector (u_lat)
    u_lat = np.cross(u_fwd, u_up)
    norm_lat = np.linalg.norm(u_lat)
    if norm_lat > 1e-6:
        u_lat = u_lat / norm_lat

    # Ensure u_lat points rightward (towards Target B):
    if p_tb is not None and p_ta is not None:
        if np.dot(p_tb - p_ta, u_lat) < 0:
            u_lat = -u_lat
            u_up = np.cross(u_lat, u_fwd)
    elif u_lat[0] < 0:
        u_lat = -u_lat
        u_up = np.cross(u_lat, u_fwd)

    # 6. Orthonormal basis matrix R_task = [u_lat, u_fwd, u_up]
    R_task = np.column_stack([u_lat, u_fwd, u_up])
    if np.linalg.det(R_task) < 0:
        u_up = -u_up
        R_task = np.column_stack([u_lat, u_fwd, u_up])

    return R_task, p_start


def discover_available_targets(peri_root: Path) -> List[str]:
    """
    Scan PeriStim directory for available reach targets ('A', 'B', etc.).
    Returns sorted list of available target identifiers.
    """
    if not peri_root or not Path(peri_root).exists():
        return ["A"]

    peri_root = Path(peri_root)
    found_targets = set()

    for p in peri_root.rglob("*.npz"):
        if p.name.startswith("._"):
            continue
        p_str = str(p).replace("\\", "/").lower()
        if "target_a" in p_str or "targeta" in p_str or "_ta." in p_str:
            found_targets.add("A")
        if "target_b" in p_str or "targetb" in p_str or "_tb." in p_str:
            found_targets.add("B")

    if not found_targets:
        return ["A"]

    return sorted(list(found_targets))


def resolve_session_dir(session: str, peri_root: Optional[Path] = None) -> Optional[Path]:
    """
    Robustly locate the root session directory (e.g. '20260116_NRR_RW012').
    Checks:
      1. peri_root parent hierarchy (if peri_root is provided).
      2. SESSION_LOC if matching session name.
      3. Global data roots (Nike, Ada, Bert) matching *{session}*.
    """
    if peri_root:
        p = Path(peri_root)
        for parent in [
            p.parents[2] if len(p.parents) > 2 else None,
            p.parents[1] if len(p.parents) > 1 else None,
            p.parent,
        ]:
            if parent is not None and parent.exists():
                if (parent / "Video").exists() or (parent / "results").exists() or (parent / "Metadata").exists() or (parent / "metadata").exists():
                    return parent

    if SESSION_LOC and session and session in str(SESSION_LOC) and Path(SESSION_LOC).exists():
        return Path(SESSION_LOC)

    roots_to_check: List[Path] = []
    if PARAMS and hasattr(PARAMS, "data_root"):
        roots_to_check.append(Path(PARAMS.data_root))
    roots_to_check.extend([
        Path(r"Y:\Current Project Databases - NHP\2025 Cerebellum prosthesis"),
        Path(r"\\cullenstor.win.ad.jhu.edu\data\Current Project Databases - NHP\2025 Cerebellum prosthesis"),
    ])

    for root in roots_to_check:
        if root.exists():
            for m in ["Nike", "Ada", "Bert"]:
                mdir = root / m
                if mdir.exists():
                    matches = [d for d in mdir.glob(f"*{session}*") if d.is_dir() and not d.name.startswith(".")]
                    if matches:
                        return matches[0]
    return None


def find_metadata_csv(
    session: str,
    metadata_csv: Optional[Path] = None,
    peri_root: Optional[Path] = None,
) -> Optional[Path]:
    """Locate session metadata CSV file from provided path, peri_root hierarchy, or standard config paths."""
    if metadata_csv:
        p_meta = Path(metadata_csv)
        if p_meta.is_file() and (not session or session in p_meta.name):
            return p_meta
        if p_meta.is_dir() and not peri_root:
            for sub in [p_meta / "Metadata", p_meta / "metadata", p_meta]:
                if sub.exists() and sub.is_dir():
                    for pat in [f"*{session}*metadata*.csv", "*metadata*.csv"]:
                        matches = [f for f in sub.glob(pat) if not f.name.startswith("._") and "shift" not in f.name.lower()]
                        if matches:
                            return matches[0]

    if METADATA_CSV:
        p_m = Path(METADATA_CSV)
        if p_m.is_file() and (not session or session in p_m.name):
            return p_m

    sess_dir = resolve_session_dir(session=session, peri_root=peri_root)

    candidates: List[Optional[Path]] = []
    if sess_dir:
        candidates.extend([sess_dir / "Metadata", sess_dir / "metadata"])
    if peri_root:
        p = Path(peri_root)
        candidates.extend([
            p.parents[2] / "metadata" if len(p.parents) > 2 else None,
            p.parents[2] / "Metadata" if len(p.parents) > 2 else None,
            p.parents[1] / "metadata" if len(p.parents) > 1 else None,
            p.parents[1] / "Metadata" if len(p.parents) > 1 else None,
            p.parent / "metadata",
            p.parent / "Metadata",
        ])
    if SESSION_LOC:
        candidates.extend([Path(SESSION_LOC) / "Metadata", Path(SESSION_LOC) / "metadata"])
    if PARAMS and hasattr(PARAMS, "data_root"):
        for m in ["Nike", "Ada", "Bert"]:
            candidates.append(Path(PARAMS.data_root) / m / f"20260311_{session}" / "metadata")
            candidates.append(Path(PARAMS.data_root) / m / session / "metadata")

    for c in candidates:
        if c is not None and c.exists():
            for pat in [f"*{session}*metadata*.csv", "*metadata*.csv"]:
                matches = [f for f in c.glob(pat) if not f.name.startswith("._") and "shift" not in f.name.lower()]
                if matches:
                    return matches[0]
    return None


def load_session_metadata(meta_csv: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    """Parse session metadata CSV into condition-indexed dictionary with notes, stim params, etc."""
    if meta_csv is None or not Path(meta_csv).exists():
        return {}

    df = None
    for enc in ("utf-8-sig", "latin-1", "cp1252", "utf-8"):
        try:
            df = pd.read_csv(meta_csv, encoding=enc, engine="python", on_bad_lines="skip")
            break
        except Exception:
            pass

    if df is None or df.empty:
        return {}

    norm_cols = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    br_col = None
    for cand in ("br_file", "br", "br_idx", "condition", "cond"):
        if cand in norm_cols:
            br_col = norm_cols[cand]
            break

    notes_col = None
    for cand in ("notes", "note", "comments", "comment", "description"):
        if cand in norm_cols:
            notes_col = norm_cols[cand]
            break

    freq_col = None
    for cand in ("stim_frequency_hz", "stim_freq_hz", "frequency_hz", "freq_hz"):
        if cand in norm_cols:
            freq_col = norm_cols[cand]
            break

    dur_col = None
    for cand in ("stim_duration_ms", "stim_dur_ms", "duration_ms", "stim_duration_nominal_ms"):
        if cand in norm_cols:
            dur_col = norm_cols[cand]
            break

    curr_col = None
    for cand in ("current_ua", "current_u_a", "current"):
        if cand in norm_cols:
            curr_col = norm_cols[cand]
            break

    mapping: Dict[int, Dict[str, Any]] = {}
    if br_col is not None:
        for _, r in df.iterrows():
            try:
                b_raw = r[br_col]
                if pd.isna(b_raw):
                    continue
                b_str = str(b_raw).strip()
                try:
                    b_int = int(float(b_str))
                except ValueError:
                    nums = re.findall(r"\d+", b_str)
                    if not nums:
                        continue
                    b_int = int(nums[-1])

                notes_val = str(r[notes_col]).strip() if (notes_col and pd.notna(r[notes_col])) else ""
                freq_val = None
                if freq_col and pd.notna(r[freq_col]):
                    try:
                        freq_val = float(r[freq_col])
                    except Exception:
                        pass

                dur_val = str(r[dur_col]).strip() if (dur_col and pd.notna(r[dur_col])) else ""
                curr_val = None
                if curr_col and pd.notna(r[curr_col]):
                    try:
                        curr_val = float(r[curr_col])
                    except Exception:
                        pass

                mapping[b_int] = {
                    "notes": notes_val,
                    "stim_freq_hz": freq_val,
                    "stim_duration_ms": dur_val,
                    "current_ua": curr_val,
                }
            except Exception:
                continue

    return mapping


def interpolate_small_gaps(arr: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """Linearly interpolate NaN gaps up to max_gap consecutive frames."""
    arr = np.asarray(arr, dtype=float).copy()
    if arr.ndim == 1:
        arr = arr[:, None]
    n_rows, n_cols = arr.shape
    for col in range(n_cols):
        x = arr[:, col]
        is_nan = ~np.isfinite(x)
        if not is_nan.any():
            continue
        starts = np.where(is_nan & ~np.r_[False, is_nan[:-1]])[0]
        ends = np.where(is_nan & ~np.r_[is_nan[1:], False])[0]
        for s, e in zip(starts, ends):
            gap_len = e - s + 1
            left_idx, right_idx = s - 1, e + 1
            if gap_len <= max_gap and left_idx >= 0 and right_idx < n_rows:
                y0, y1 = x[left_idx], x[right_idx]
                if np.isfinite(y0) and np.isfinite(y1):
                    x[s : e + 1] = np.linspace(y0, y1, gap_len + 2)[1:-1]
    return arr.squeeze() if arr.shape[1] == 1 else arr


def filter_signal_butterworth(
    trace: np.ndarray,
    dt: float,
    cutoff_hz: float = BUTTER_CUTOFF_HZ,
    order: int = BUTTER_ORDER,
    median_kernel: int = 5,
) -> np.ndarray:
    """
    Filter 1D coordinate trace using a 5-frame median despike followed by
    a Butterworth low-pass filter.
    
    1. Handles NaN values via boundary clamp and linear interpolation so the
       signal is continuous.
    2. Applies a 5-frame (50 ms at 100 Hz) median filter to strip tracking swaps
       and single/double-frame dropouts, eliminating impulse-induced ringing artifacts.
    3. Smooths using zero-phase Butterworth filter (filtfilt).
    """
    trace = np.asarray(trace, dtype=float).copy()
    valid = np.isfinite(trace)
    if not np.any(valid):
        return trace
    if not np.all(valid):
        idx = np.arange(len(trace))
        trace[~valid] = np.interp(idx[~valid], idx[valid], trace[valid])

    # 5-frame median filter despiking to eliminate ringing before filtfilt
    if median_kernel and median_kernel >= 3 and len(trace) >= median_kernel:
        k = median_kernel if (median_kernel % 2 == 1) else (median_kernel + 1)
        trace = median_filter(trace, size=k, mode="nearest")

    if dt < 0.9:
        dt *= 1000.0  # convert seconds to ms
    fs = 1000.0 / dt
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        cutoff_hz = nyq * 0.9

    try:
        b, a = butter(order, cutoff_hz / nyq, btype="low")
        return filtfilt(b, a, trace)
    except Exception:
        return trace


def normalize_trace(trace: np.ndarray, method: str = "max") -> Tuple[np.ndarray, float]:
    """Normalize reach distance trace to [0, 1]."""
    if method == "percentile":
        scale = float(np.nanpercentile(trace, 95))
    else:
        scale = float(np.nanmax(trace))
    scale = max(scale, 1e-6)
    return trace / scale, scale


# =============================================================================
# 3D DLC CSV Loader
# =============================================================================

def load_dlc_3d_file(
    dlc_path: Path,
    square_size_mm: float = SQUARE_SIZE_MM,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Load continuous 3D DLC file (CSV or H5) with multi-level header [scorer, bodyparts, coords].
    Flattens columns to '{bodypart}_{coord}' format and scales coordinates by square_size_mm (mm).
    """
    if not dlc_path.exists():
        _log(f"3D DLC file not found: {dlc_path}")
        return None, []

    try:
        if dlc_path.suffix.lower() == ".h5":
            df_raw = pd.read_hdf(str(dlc_path))
        else:
            df_raw = pd.read_csv(dlc_path, header=[0, 1, 2], index_col=0)

        # Handle column MultiIndex flattening
        flattened_cols = []
        bodyparts_set = set()
        for col in df_raw.columns:
            if len(col) >= 3:
                bp = str(col[1]).strip()
                coord = str(col[2]).strip().lower()
            else:
                bp = str(col[0]).strip()
                coord = str(col[1]).strip().lower()
            flattened_cols.append(f"{bp}_{coord}")
            bodyparts_set.add(bp)

        df = df_raw.copy()
        df.columns = flattened_cols
        df = df.apply(pd.to_numeric, errors="coerce")

        # Metric scaling: convert checkerboard square units to physical millimeters
        if square_size_mm and square_size_mm != 1.0:
            df = df * float(square_size_mm)

        # Interpolate small NaN gaps (<= 4 frames) per column
        for col in df.columns:
            df[col] = interpolate_small_gaps(df[col].values, max_gap=4)

        return df, sorted(list(bodyparts_set))

    except Exception as e:
        _log(f"Error reading 3D DLC file {dlc_path}: {e}")
        return None, []


# =============================================================================
# Frame Timeline & Video Mapping
# =============================================================================

def resolve_video_index_for_br(
    br_idx: int,
    metadata_csv: Optional[Path] = None,
    peri_meta: Optional[Dict] = None,
) -> Optional[int]:
    """Resolve video recording index (e.g. 1 -> 001) for a given Blackrock index."""
    if peri_meta and "video_file" in peri_meta:
        try:
            return int(peri_meta["video_file"])
        except Exception:
            pass

    if metadata_csv and Path(metadata_csv).is_file():
        try:
            br_to_video = rcp.get_metadata_mapping(Path(metadata_csv), "BR_File", "Video_File")
            if br_idx in br_to_video and br_to_video[br_idx] is not None:
                return int(br_to_video[br_idx])
        except Exception as e:
            _log(f"Metadata mapping failed for BR {br_idx}: {e}")

    return int(br_idx)


def find_3d_dlc_csv(
    dlc_3d_root: Path,
    session: str,
    vid_idx: int,
    br_idx: int,
) -> Optional[Path]:
    """Find matching 3D DLC CSV or H5 file in dlc_3d_root."""
    if not dlc_3d_root.exists():
        return None

    candidate_patterns = [
        f"{session}_{vid_idx:03d}*DLC_3D*.csv",
        f"*{vid_idx:03d}*DLC_3D*.csv",
        f"*BR_{br_idx:03d}*DLC_3D*.csv",
        f"*BR_{br_idx}*DLC_3D*.csv",
        f"{session}_{vid_idx:03d}*DLC_3D*.h5",
        f"*{vid_idx:03d}*DLC_3D*.h5",
    ]

    for pat in candidate_patterns:
        matches = sorted(list(dlc_3d_root.glob(pat)))
        if matches:
            return matches[0]

    return None


def load_frame_timeline_ms(
    br_idx: int,
    vid_idx: int,
    session: str,
    aligned_ckpt_root: Optional[Path] = None,
    behv_ckpt_root: Optional[Path] = None,
    fps: float = 100.0,
    fs_neural: float = 30000.0,
) -> Optional[np.ndarray]:
    """
    Extract continuous frame timestamps in milliseconds on the Blackrock timeline.
    Attempts:
        1. Aligned checkpoint NPZ (aligned__*BR_<br_idx>*.npz) -> 'beh_t_ms'
        2. Behavior checkpoint CSV (*both_cams_aligned.csv) -> 'ns5_sample' * 1000 / fs
        3. Fallback: synthetic timeline based on camera FPS.
    """
    if aligned_ckpt_root and Path(aligned_ckpt_root).exists():
        # Search recursively so subdirectories like control_reaches and stim_reaches are included
        cands = sorted(list(Path(aligned_ckpt_root).rglob(f"*BR_{br_idx:03d}*.npz")))
        if not cands:
            cands = sorted(list(Path(aligned_ckpt_root).rglob(f"*BR_{br_idx}*.npz")))
        if cands:
            try:
                npz = np.load(cands[0], allow_pickle=True)
                if "beh_t_ms" in npz and npz["beh_t_ms"].size > 0:
                    return np.asarray(npz["beh_t_ms"], dtype=float)
            except Exception as e:
                _log(f"Could not load beh_t_ms from {cands[0].name}: {e}")

    if behv_ckpt_root and Path(behv_ckpt_root).exists():
        cands = sorted(list(Path(behv_ckpt_root).rglob(f"*{vid_idx:03d}*both_cams_aligned.csv")))
        if not cands:
            cands = sorted(list(Path(behv_ckpt_root).rglob(f"*BR_{br_idx:03d}*both_cams_aligned.csv")))
        if cands:
            try:
                df_align = pd.read_csv(cands[0], header=[0, 1], low_memory=False)
                ns5_col = None
                for col in df_align.columns:
                    if "ns5_sample" in str(col).lower():
                        ns5_col = col
                        break
                if ns5_col is not None:
                    samps = pd.to_numeric(df_align[ns5_col], errors="coerce").values
                    # If row 0 is unaligned NaN header row, strip it to match 0-indexed video frames
                    if len(samps) > 0 and np.isnan(samps[0]):
                        samps = samps[1:]
                    return samps * 1000.0 / fs_neural
            except Exception as e:
                _log(f"Could not load ns5_sample from {cands[0].name}: {e}")

    return None


# =============================================================================
# 3D Reach Segmentation ("Cutting")
# =============================================================================

def segment_condition_3d_reaches(
    dlc_df: pd.DataFrame,
    frame_t_ms: np.ndarray,
    event_ms: np.ndarray,
    keypoint: str = DEFAULT_KEYPOINT,
    pre_zero_ms: float = PRE_ZERO_MS,
    post_zero_ms: float = POST_ZERO_MS,
    step_ms: float = STEP_MS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cut individual reaching trials from continuous 3D DLC positions around event times.

    Returns:
        rel_t : 1D array of relative time points (ms), shape (T,)
        x_all : 2D array, shape (n_trials, T)
        y_all : 2D array, shape (n_trials, T)
        z_all : 2D array, shape (n_trials, T)
        valid_mask : 1D boolean array, shape (n_trials,)
    """
    rel_t = np.arange(pre_zero_ms, post_zero_ms + 1e-6, step_ms)
    n_trials = len(event_ms)
    n_times = len(rel_t)

    col_x = f"{keypoint}_x"
    col_y = f"{keypoint}_y"
    col_z = f"{keypoint}_z"

    for c in [col_x, col_y, col_z]:
        if c not in dlc_df.columns:
            raise ValueError(f"Keypoint coordinate column '{c}' not found in 3D DLC columns: {list(dlc_df.columns)[:8]}...")

    x_series = dlc_df[col_x].values
    y_series = dlc_df[col_y].values
    z_series = dlc_df[col_z].values

    n_pts = min(len(frame_t_ms), len(x_series))
    frame_t_ms = frame_t_ms[:n_pts]
    x_series = x_series[:n_pts]
    y_series = y_series[:n_pts]
    z_series = z_series[:n_pts]

    good_t = np.isfinite(frame_t_ms)
    if not np.any(good_t):
        raise ValueError("Frame timeline contains no finite timestamps.")

    x_all = np.full((n_trials, n_times), np.nan, dtype=float)
    y_all = np.full((n_trials, n_times), np.nan, dtype=float)
    z_all = np.full((n_trials, n_times), np.nan, dtype=float)
    valid_mask = np.ones(n_trials, dtype=bool)

    for i, t_ev in enumerate(event_ms):
        if not np.isfinite(t_ev):
            valid_mask[i] = False
            continue

        t_desired = t_ev + rel_t

        for s_data, out_arr in [(x_series, x_all), (y_series, y_all), (z_series, z_all)]:
            valid_pts = good_t & np.isfinite(s_data)
            if np.sum(valid_pts) >= 2:
                # Nearest boundary clamp prevents artificial NaNs at edges of reach window
                out_arr[i, :] = np.interp(
                    t_desired,
                    frame_t_ms[valid_pts],
                    s_data[valid_pts],
                    left=s_data[valid_pts][0],
                    right=s_data[valid_pts][-1],
                )
                # Fill any interior gaps linearly
                row_valid = np.isfinite(out_arr[i, :])
                if np.any(row_valid) and not np.all(row_valid):
                    idx_grid = np.arange(n_times)
                    out_arr[i, ~row_valid] = np.interp(idx_grid[~row_valid], idx_grid[row_valid], out_arr[i, row_valid])
            else:
                out_arr[i, :] = np.nan
                valid_mask[i] = False

    return rel_t, x_all, y_all, z_all, valid_mask


# =============================================================================
# 3D Kinematics & Signal Processing
# =============================================================================

def process_single_trial_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    t_axis: np.ndarray,
    ref_time_ms: float = DISTANCE_REFERENCE_TIME_MS,
    cutoff_hz: float = BUTTER_CUTOFF_HZ,
    order: int = BUTTER_ORDER,
) -> Dict[str, Any]:
    """
    Smooth X, Y, Z traces using Butterworth filtering, compute 3D Euclidean distance
    from the pre-movement resting baseline position (at ref_time_ms = -600.0 ms),
    normalized distance, and 3D velocity.
    """
    dt = float(t_axis[1] - t_axis[0]) if len(t_axis) > 1 else 10.0
    if dt < 0.9:
        dt *= 1000.0  # convert seconds to ms

    # Butterworth smoothing
    x_smooth = filter_signal_butterworth(x, dt, cutoff_hz=cutoff_hz, order=order)
    y_smooth = filter_signal_butterworth(y, dt, cutoff_hz=cutoff_hz, order=order)
    z_smooth = filter_signal_butterworth(z, dt, cutoff_hz=cutoff_hz, order=order)

    # Reference baseline rest position (robust median across quiet resting window, e.g. [-700, -500] ms)
    ref_mask = (t_axis >= ref_time_ms - 100.0) & (t_axis <= ref_time_ms + 100.0)
    if not np.any(ref_mask) or not np.any(np.isfinite(x[ref_mask])):
        ref_mask = (t_axis >= ref_time_ms - 25.0) & (t_axis <= ref_time_ms + 25.0)

    if np.any(ref_mask) and np.any(np.isfinite(x[ref_mask])):
        x0_raw = float(np.nanmedian(x[ref_mask]))
        y0_raw = float(np.nanmedian(y[ref_mask]))
        z0_raw = float(np.nanmedian(z[ref_mask]))
        x0_s = float(np.nanmedian(x_smooth[ref_mask]))
        y0_s = float(np.nanmedian(y_smooth[ref_mask]))
        z0_s = float(np.nanmedian(z_smooth[ref_mask]))
    else:
        # Fallback to value at ref time or first sample
        x0_raw = value_at_ref_time(x, t_axis, ref_time_ms)
        y0_raw = value_at_ref_time(y, t_axis, ref_time_ms)
        z0_raw = value_at_ref_time(z, t_axis, ref_time_ms)
        x0_s = value_at_ref_time(x_smooth, t_axis, ref_time_ms)
        y0_s = value_at_ref_time(y_smooth, t_axis, ref_time_ms)
        z0_s = value_at_ref_time(z_smooth, t_axis, ref_time_ms)

    # Raw 3D distance from starting baseline rest position
    dist_raw = np.sqrt((x - x0_raw) ** 2 + (y - y0_raw) ** 2 + (z - z0_raw) ** 2)

    # Smoothed 3D distance from starting baseline rest position
    dist_smooth = np.sqrt((x_smooth - x0_s) ** 2 + (y_smooth - y0_s) ** 2 + (z_smooth - z0_s) ** 2)

    # 3D Velocity (mm/ms or units/ms)
    velocity = np.gradient(dist_smooth, dt)

    return {
        "x_raw": x,
        "y_raw": y,
        "z_raw": z,
        "x_smooth": x_smooth,
        "y_smooth": y_smooth,
        "z_smooth": z_smooth,
        "dist_raw": dist_raw,
        "dist_smooth": dist_smooth,
        "velocity": velocity,
        "dt": dt,
    }


def process_condition_3d(
    x_all: np.ndarray,
    y_all: np.ndarray,
    z_all: np.ndarray,
    t_axis: np.ndarray,
    valid_mask: np.ndarray,
    metadata: Dict[str, Any],
    only_post_zero: Optional[bool] = None,
    plot_window: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> Dict[str, Any]:
    """
    Process all segmented 3D reaching trials for one condition.
    Filtering and rest-baseline anchoring use the full extraction window (e.g. -800 to +600 ms).
    Output traces are sliced to `plot_window` (defaulting to PLOT_WINDOW_MS).
    If `only_post_zero` is True, window start is restricted to t >= 0 ms.
    """
    n_trials, n_times = x_all.shape
    dt = float(t_axis[1] - t_axis[0]) if len(t_axis) > 1 else 10.0

    x_smooth_all = np.full((n_trials, n_times), np.nan, dtype=float)
    y_smooth_all = np.full((n_trials, n_times), np.nan, dtype=float)
    z_smooth_all = np.full((n_trials, n_times), np.nan, dtype=float)
    dist_raw_all = np.full((n_trials, n_times), np.nan, dtype=float)
    dist_smooth_all = np.full((n_trials, n_times), np.nan, dtype=float)
    dist_norm = np.full((n_trials, n_times), np.nan, dtype=float)
    scale_factors = np.full(n_trials, np.nan, dtype=float)

    for i in range(n_trials):
        if not valid_mask[i]:
            continue
        try:
            res = process_single_trial_3d(x_all[i], y_all[i], z_all[i], t_axis)
            x_smooth_all[i] = res["x_smooth"]
            y_smooth_all[i] = res["y_smooth"]
            z_smooth_all[i] = res["z_smooth"]
            dist_raw_all[i] = res["dist_raw"]
            dist_smooth_all[i] = res["dist_smooth"]
            dist_norm[i], scale_factors[i] = normalize_trace(dist_smooth_all[i])
        except Exception as e:
            _log(f"Warning: Trial {i} failed in process_single_trial_3d: {e}")
            valid_mask[i] = False

    n_valid = int(np.sum(valid_mask))
    valid_dist = dist_norm[valid_mask] if n_valid > 0 else np.empty((0, n_times))

    if n_valid > 0:
        dist_median = np.nanmedian(valid_dist, axis=0)
        dist_mean = np.nanmean(valid_dist, axis=0)
        dist_std = np.nanstd(valid_dist, axis=0, ddof=1) if n_valid > 1 else np.zeros(n_times)
        n_pts_valid = np.sum(np.isfinite(valid_dist), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            dist_sem = dist_std / np.sqrt(np.maximum(1, n_pts_valid))
        ci_summary = mean_ci95_trace_summary(valid_dist, time=t_axis)
        dist_ci_lower = ci_summary["ci_lower"] if ci_summary else dist_mean - 1.96 * dist_sem
        dist_ci_upper = ci_summary["ci_upper"] if ci_summary else dist_mean + 1.96 * dist_sem
    else:
        dist_median = np.full(n_times, np.nan)
        dist_mean = np.full(n_times, np.nan)
        dist_std = np.full(n_times, np.nan)
        dist_sem = np.full(n_times, np.nan)
        dist_ci_lower = np.full(n_times, np.nan)
        dist_ci_upper = np.full(n_times, np.nan)

    # Slice output traces according to master plot_window (defaulting to PLOT_WINDOW_MS)
    # Note: Pre-0 buffer was retained above for filter warm-up and rest-bar baseline anchoring.
    if plot_window is None:
        plot_window = PLOT_WINDOW_MS

    if only_post_zero is True:
        w_start = max(0.0, plot_window[0] if plot_window[0] is not None else 0.0)
        w_end = plot_window[1] if plot_window[1] is not None else np.inf
    elif plot_window is not None:
        w_start = plot_window[0] if plot_window[0] is not None else -np.inf
        w_end = plot_window[1] if plot_window[1] is not None else np.inf
    else:
        w_start = -np.inf
        w_end = np.inf

    window_mask = (t_axis >= w_start) & (t_axis <= w_end)
    if np.any(window_mask):
        t_axis_out = t_axis[window_mask]
        x_raw_out = x_all[:, window_mask]
        y_raw_out = y_all[:, window_mask]
        z_raw_out = z_all[:, window_mask]
        x_smooth_out = x_smooth_all[:, window_mask]
        y_smooth_out = y_smooth_all[:, window_mask]
        z_smooth_out = z_smooth_all[:, window_mask]
        dist_raw_out = dist_raw_all[:, window_mask]
        dist_smooth_out = dist_smooth_all[:, window_mask]
        dist_norm_out = dist_norm[:, window_mask]
        dist_median_out = dist_median[window_mask]
        dist_mean_out = dist_mean[window_mask]
        dist_std_out = dist_std[window_mask]
        dist_sem_out = dist_sem[window_mask]
        dist_ci_lower_out = dist_ci_lower[window_mask]
        dist_ci_upper_out = dist_ci_upper[window_mask]
    else:
        t_axis_out = t_axis
        x_raw_out = x_all
        y_raw_out = y_all
        z_raw_out = z_all
        x_smooth_out = x_smooth_all
        y_smooth_out = y_smooth_all
        z_smooth_out = z_smooth_all
        dist_raw_out = dist_raw_all
        dist_smooth_out = dist_smooth_all
        dist_norm_out = dist_norm
        dist_median_out = dist_median
        dist_mean_out = dist_mean
        dist_std_out = dist_std
        dist_sem_out = dist_sem
        dist_ci_lower_out = dist_ci_lower
        dist_ci_upper_out = dist_ci_upper

    out = {
        "aligned_time": t_axis_out,
        "x_raw_all": x_raw_out,
        "y_raw_all": y_raw_out,
        "z_raw_all": z_raw_out,
        "x_smooth_all": x_smooth_out,
        "y_smooth_all": y_smooth_out,
        "z_smooth_all": z_smooth_out,
        "dist_raw_all": dist_raw_out,
        "dist_smooth_all": dist_smooth_out,
        "dist_norm": dist_norm_out,
        "scale_factors": scale_factors,
        "dist_median": dist_median_out,
        "dist_mean": dist_mean_out,
        "dist_std": dist_std_out,
        "dist_sem": dist_sem_out,
        "dist_ci_lower": dist_ci_lower_out,
        "dist_ci_upper": dist_ci_upper_out,
        "n_valid": n_valid,
        "n_total": n_trials,
        "valid_mask": valid_mask,
        "dt": dt,
        "only_post_zero": only_post_zero,
        # Full extraction traces preserved for reference/diagnostics
        "full_time": t_axis,
        "x_smooth_all_full": x_smooth_all,
        "y_smooth_all_full": y_smooth_all,
        "z_smooth_all_full": z_smooth_all,
        "dist_norm_full": dist_norm,
    }
    out.update(metadata)
    return out


# =============================================================================
# 3D Trajectory Variability Computation
# =============================================================================

def _pairwise_distances_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Compute pairwise 3D Euclidean distances between trials at each sample."""
    n_trials, n_samples = x.shape
    if n_trials < 2:
        return np.empty((0, n_samples))
    pairwise = []
    for i in range(n_trials - 1):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        dz = z[i + 1 :] - z[i]
        d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        pairwise.append(d)
    return np.vstack(pairwise)


def _aggregate_pairwise(pairwise_dists: np.ndarray, agg: str = "median") -> np.ndarray:
    """Aggregate pairwise distances across pairs at each sample."""
    if pairwise_dists.size == 0:
        return np.array([])
    if agg == "mean":
        return np.nanmean(pairwise_dists, axis=0)
    return np.nanmedian(pairwise_dists, axis=0)


def _per_trial_distance_to_peers_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sample_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute average 3D distance of each trial to all other peer trials."""
    n_trials, n_samples = x.shape
    if sample_mask is None:
        sample_mask = np.ones(n_samples, dtype=bool)
    else:
        sample_mask = np.asarray(sample_mask).astype(bool)

    scores = np.full(n_trials, np.nan)
    if n_trials < 2 or not np.any(sample_mask):
        return scores

    for i in range(n_trials):
        dx = x[np.arange(n_trials) != i][:, sample_mask] - x[i, sample_mask]
        dy = y[np.arange(n_trials) != i][:, sample_mask] - y[i, sample_mask]
        dz = z[np.arange(n_trials) != i][:, sample_mask] - z[i, sample_mask]
        d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        scores[i] = np.nanmean(d)

    return scores


def _compute_cumulative_path_progress_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Compute cumulative 3D path progress from 0.0 to 1.0 for one trajectory."""
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    step = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    cumdist = np.concatenate([[0.0], np.cumsum(step)])
    total = cumdist[-1]
    if not np.isfinite(total) or total <= 0:
        return np.linspace(0.0, 1.0, len(x))
    return cumdist / total


def _interp_trajectory_by_progress_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    progress_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample one 3D trajectory onto normalized path-progress grid."""
    progress = _compute_cumulative_path_progress_3d(x, y, z)
    unique_progress, unique_idx = np.unique(progress, return_index=True)
    if len(unique_progress) < 2:
        return (
            np.full_like(progress_grid, np.nan, dtype=float),
            np.full_like(progress_grid, np.nan, dtype=float),
            np.full_like(progress_grid, np.nan, dtype=float),
        )
    xi = np.interp(progress_grid, unique_progress, x[unique_idx])
    yi = np.interp(progress_grid, unique_progress, y[unique_idx])
    zi = np.interp(progress_grid, unique_progress, z[unique_idx])
    return xi, yi, zi


def _resample_condition_by_progress_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_progress_points: int = 101,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample all valid 3D trajectories of a condition by path progress."""
    progress_grid = np.linspace(0.0, 1.0, n_progress_points)
    n_trials = x.shape[0]
    x_prog = np.full((n_trials, n_progress_points), np.nan)
    y_prog = np.full((n_trials, n_progress_points), np.nan)
    z_prog = np.full((n_trials, n_progress_points), np.nan)

    for i in range(n_trials):
        x_prog[i], y_prog[i], z_prog[i] = _interp_trajectory_by_progress_3d(
            x[i], y[i], z[i], progress_grid
        )

    finite = (
        np.all(np.isfinite(x_prog), axis=1)
        & np.all(np.isfinite(y_prog), axis=1)
        & np.all(np.isfinite(z_prog), axis=1)
    )
    return progress_grid, x_prog[finite], y_prog[finite], z_prog[finite]


def compute_variability_metrics_3d(
    cond_data: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute time-resolved and path-progress 3D trajectory variability."""
    time = cond_data["time"]
    x = cond_data["x"]
    y = cond_data["y"]
    z = cond_data["z"]

    # 1. Time-resolved 3D variability
    pairwise_time = _pairwise_distances_3d(x, y, z)
    curve_time = _aggregate_pairwise(pairwise_time, cfg.get("pairwise_agg", "median"))
    std_time = (
        np.nanstd(pairwise_time, axis=0, ddof=1)
        if (pairwise_time.size > 0 and pairwise_time.shape[0] > 1)
        else np.zeros_like(curve_time)
    )

    t0, t1 = cfg.get("time_summary_window_ms", (-500.0, 500.0))
    time_mask = (time >= t0) & (time <= t1)
    summary_time = float(np.nanmean(curve_time[time_mask])) if np.any(time_mask) and curve_time.size > 0 else np.nan
    per_trial_time = _per_trial_distance_to_peers_3d(x, y, z, time_mask)

    # 2. Path-progress 3D variability
    p_grid, x_p, y_p, z_p = _resample_condition_by_progress_3d(
        x, y, z, n_progress_points=cfg.get("n_progress_points", 101)
    )
    pairwise_prog = _pairwise_distances_3d(x_p, y_p, z_p)
    curve_prog = _aggregate_pairwise(pairwise_prog, cfg.get("pairwise_agg", "median"))
    std_prog = (
        np.nanstd(pairwise_prog, axis=0, ddof=1)
        if (pairwise_prog.size > 0 and pairwise_prog.shape[0] > 1)
        else np.zeros_like(curve_prog)
    )

    p0, p1 = cfg.get("progress_summary_window", (0.05, 0.95))
    prog_mask = (p_grid >= p0) & (p_grid <= p1)
    summary_prog = float(np.nanmean(curve_prog[prog_mask])) if np.any(prog_mask) and curve_prog.size > 0 else np.nan
    per_trial_prog = _per_trial_distance_to_peers_3d(x_p, y_p, z_p, prog_mask)

    dist_mean = cond_data.get("dist_mean", np.full_like(time, np.nan))
    dist_std = cond_data.get("dist_std", np.zeros_like(time))
    dist_sem = cond_data.get("dist_sem", np.zeros_like(time))

    return {
        "cond_num": cond_data.get("cond_num"),
        "cond_key": cond_data.get("cond_key"),
        "br_idx": cond_data.get("br_idx"),
        "is_control": cond_data.get("is_control", False),
        "notes": cond_data.get("notes", ""),
        "pulse_count": cond_data.get("pulse_count"),
        "stim_freq_hz": cond_data.get("stim_freq_hz"),
        "stim_dur_nominal_ms": cond_data.get("stim_dur_nominal_ms"),
        "n_valid": cond_data.get("n_valid", x.shape[0]),
        "time_resolved": {
            "x_axis": time,
            "curve": curve_time,
            "std": std_time,
            "summary_score": summary_time,
            "per_trial_scores": per_trial_time,
        },
        "path_progress": {
            "x_axis": p_grid,
            "curve": curve_prog,
            "std": std_prog,
            "summary_score": summary_prog,
            "per_trial_scores": per_trial_prog,
        },
        "norm_distance": {
            "time": time,
            "mean": dist_mean,
            "std": dist_std,
            "sem": dist_sem,
        },
    }


# =============================================================================
# Condition Labeling & Colors
# =============================================================================

def format_condition_label(
    cond_num: int,
    cond_info: Optional[Dict] = None,
    control_br_indices: Optional[List[int]] = None,
    pulse_width_ms: float = PULSE_WIDTH_MS,
    include_target: bool = False,
    show_indices: bool = True,
) -> str:
    """
    Format publication-quality condition label based on metadata and BR index.
    
    Examples:
        - Control: 'BR 1: Control', 'Control'
        - Stim: 'BR 17: 40 pulses, 400 Hz', '40 pulses, 400 Hz'
        - Current Steering: 'BR 25: 40 pulses, 400 Hz, Current Steering'
        - Multi-target: 'BR 1: Control (Target A)', 'BR 1: Control (Target B)'
    """
    if cond_info is None:
        cond_info = {}

    br_idx = cond_info.get("br_idx", cond_num)
    is_control = (
        bool(control_br_indices and (cond_num in control_br_indices or (br_idx is not None and br_idx in control_br_indices)))
        or cond_info.get("is_control", False)
        or (cond_info.get("pulse_count") == 0)
        or (cond_info.get("stim_freq_hz") == 0.0)
    )

    cond_str = str(cond_info.get("cond_str", ""))
    notes = str(cond_info.get("notes", "")).strip()

    if not is_control:
        if notes and notes.lower() in ("baseline", "control", "ctrl"):
            is_control = True
        elif cond_str and ("control" in cond_str.lower() or "baseline" in cond_str.lower()):
            is_control = True

    # Current steering check
    is_curr_steering = False
    for candidate_str in [notes, cond_str]:
        if not candidate_str:
            continue
        cleaned = candidate_str.lower().replace("-", " ").replace("_", " ")
        if "curr steer" in cleaned or "current steer" in cleaned:
            is_curr_steering = True
            break

    comb_brs = cond_info.get("combined_br_indices")
    if comb_brs and len(comb_brs) > 1 and show_indices:
        prefix = f"BR {','.join(map(str, comb_brs))}: "
        prefix_only = f"BR {','.join(map(str, comb_brs))}"
    else:
        prefix = f"BR {cond_num}: " if show_indices else ""
        prefix_only = f"BR {cond_num}" if show_indices else ""

    if is_control:
        base_label = f"{prefix}Control" if prefix else "Control"
    else:
        pulse_count = cond_info.get("pulse_count")
        freq_hz = cond_info.get("stim_freq_hz")
        dur_ms = cond_info.get("stim_dur_nominal_ms")

        if (freq_hz is None or np.isnan(freq_hz)) and cond_str:
            m_f = re.search(r"(?:Freq:\s*|,\s*|\b)([\d.]+)\s*Hz", cond_str, re.IGNORECASE)
            if m_f:
                freq_hz = float(m_f.group(1))

        if (dur_ms is None or np.isnan(dur_ms)) and cond_str:
            m_d = re.search(r"(?:Duration:\s*|,\s*|\b)([\d.]+)\s*ms", cond_str, re.IGNORECASE)
            if m_d:
                dur_ms = float(m_d.group(1))

        if pulse_count is None and cond_str:
            m_p = re.search(r"(\d+)\s*pulses?", cond_str, re.IGNORECASE)
            if m_p:
                pulse_count = int(m_p.group(1))

        if pulse_count is None and dur_ms is not None and not np.isnan(dur_ms) and dur_ms > 0:
            pulse_count = int(round(dur_ms / pulse_width_ms))

        if pulse_count is not None and freq_hz is not None and not np.isnan(freq_hz):
            p_str = f"{pulse_count} pulse" if pulse_count == 1 else f"{pulse_count} pulses"
            f_str = f"{int(freq_hz)} Hz" if float(freq_hz).is_integer() else f"{freq_hz:.1f} Hz"
            base_label = f"{prefix}{p_str}, {f_str}"
        elif pulse_count is not None:
            p_str = f"{pulse_count} pulse" if pulse_count == 1 else f"{pulse_count} pulses"
            base_label = f"{prefix}{p_str}"
        elif freq_hz is not None and not np.isnan(freq_hz):
            f_str = f"{int(freq_hz)} Hz" if float(freq_hz).is_integer() else f"{freq_hz:.1f} Hz"
            base_label = f"{prefix}{f_str}"
        else:
            base_label = prefix_only or "Stim"

    if is_curr_steering:
        base_label = f"{base_label}, Current Steering"
    elif notes and notes.lower() not in ("baseline", "control", "ctrl", "nan", "none", ""):
        base_label = f"{base_label}, {notes}"

    stim_channels = cond_info.get("stim_channels")
    if stim_channels and not is_control:
        ch_str = f"ch {stim_channels}"
        if ch_str.lower() not in base_label.lower():
            base_label = f"{base_label}, {ch_str}"

    if include_target and cond_info:
        tgt = cond_info.get("target")
        if tgt:
            base_label = f"{base_label} (Target {tgt})"

    return base_label


def assign_condition_colors_and_styles(
    keys: List[Any],
    labels: Dict[Any, str],
    control_br_indices: Optional[List[int]] = None,
    cond_info_dict: Optional[Dict[Any, Dict]] = None,
) -> Tuple[Dict[Any, Any], Dict[Any, str]]:
    """
    Assign colors and linestyles:
      - Colors: Consistent across conditions (warm/orange for Control, bone colormap for Stim).
      - Linestyles:
          * In multi-target comparisons: Target A is solid ('-'), Target B is dashed ('--').
          * In single-target comparisons: Control is dashed, Stim is solid.
    """
    if cond_info_dict is None:
        cond_info_dict = {}

    def _get_cnum(k: Any) -> int:
        info = cond_info_dict.get(k, {})
        if isinstance(info, dict) and "cond_num" in info and info["cond_num"] is not None:
            try:
                return int(info["cond_num"])
            except Exception:
                pass
        if isinstance(k, int):
            return k
        m = re.search(r"BR_(\d+)|(\d+)", str(k))
        return int(m.group(1) or m.group(2)) if m else 0

    def _is_c(k: Any) -> bool:
        if "control" in labels.get(k, "").lower():
            return True
        c = _get_cnum(k)
        if control_br_indices and c in control_br_indices:
            return True
        info = cond_info_dict.get(k, {})
        if isinstance(info, dict) and info.get("is_control"):
            return True
        return False

    def _get_tgt(k: Any) -> str:
        info = cond_info_dict.get(k, {})
        if isinstance(info, dict) and info.get("target"):
            return str(info["target"]).strip().upper()
        s = str(k).upper()
        if "TARGET_B" in s or "_B_" in s or s.endswith("_B") or "TARGETB" in s:
            return "B"
        return "A"

    seen = []
    for k in keys:
        c = _get_cnum(k)
        if c not in seen:
            seen.append(c)
    ctrl_cnums = [c for c in seen if any(_is_c(k) for k in keys if _get_cnum(k) == c)]
    stim_cnums = [c for c in seen if c not in ctrl_cnums]

    # If conditions have varying pulse counts or durations, sort them so lower pulses/durations come first (lighter)
    # and higher pulses/durations come last (darkest).
    has_varying_pulses = False
    if cond_info_dict:
        pulses = [
            float(info.get("pulse_count") or info.get("stim_pulses") or 0.0)
            for k, info in cond_info_dict.items()
            if isinstance(info, dict) and not _is_c(k)
        ]
        if len(set(pulses)) > 1:
            has_varying_pulses = True

    if has_varying_pulses:
        def _stim_order_key(c: int) -> Tuple[float, float, int]:
            matching = [info for k, info in cond_info_dict.items() if _get_cnum(k) == c and isinstance(info, dict)]
            if matching:
                p = float(matching[0].get("pulse_count") or matching[0].get("stim_pulses") or 0.0)
                dur = float(matching[0].get("stim_dur_nominal_ms") or matching[0].get("stim_dur_ms") or 0.0)
                return (p, dur, c)
            return (0.0, 0.0, c)
        stim_cnums = sorted(stim_cnums, key=_stim_order_key)
    else:
        stim_cnums = sorted(stim_cnums)

    # Target A Palettes: Gold for Control, Grey (bone) for Stim (lighter for early conditions, darker for later conditions)
    ctrl_palette_a = ["goldenrod", "gold", "darkgoldenrod", "#E6A817"]
    n_stim = max(1, len(stim_cnums))
    stim_light_a = 0.72
    stim_dark_a = 0.25
    if n_stim == 1:
        stim_colors_a = [plt.get_cmap("bone")(0.35)]
    else:
        # bone colormap: 0.0 is black, 1.0 is white -> linspace from light (0.72) to dark (0.25)
        stim_colors_a = list(plt.get_cmap("bone")(np.linspace(stim_light_a, stim_dark_a, n_stim)))

    # Target B Palettes: Orange for Control, Dark Green for Stim (lighter for early conditions, darker for later conditions)
    ctrl_palette_b = ["darkorange", "chocolate", "coral", "orangered"]
    if n_stim == 1:
        stim_colors_b = [plt.get_cmap("Greens")(0.80)]
    else:
        # Greens colormap: 0.0 is white, 1.0 is dark green -> linspace from light (0.48) to dark (0.88)
        stim_colors_b = list(plt.get_cmap("Greens")(np.linspace(0.48, 0.88, n_stim)))

    # Check if multiple targets exist across keys
    targets = set(_get_tgt(k) for k in keys)
    is_multi_target = len(targets) > 1

    colors = {}
    linestyles = {}

    for k in keys:
        c = _get_cnum(k)
        tgt = _get_tgt(k)
        is_control = _is_c(k)

        if is_control:
            if tgt == "B":
                col_name = ctrl_palette_b[ctrl_cnums.index(c) % len(ctrl_palette_b)] if c in ctrl_cnums else "darkorange"
            else:
                col_name = ctrl_palette_a[ctrl_cnums.index(c) % len(ctrl_palette_a)] if c in ctrl_cnums else "goldenrod"
            colors[k] = matplotlib.colors.to_rgba(col_name)
        else:
            if tgt == "B":
                colors[k] = stim_colors_b[stim_cnums.index(c) % len(stim_colors_b)] if c in stim_cnums else matplotlib.colors.to_rgba("darkgreen")
            else:
                colors[k] = stim_colors_a[stim_cnums.index(c) % len(stim_colors_a)] if c in stim_cnums else matplotlib.colors.to_rgba("dimgray")

        # All condition traces are plotted as solid lines for clean visibility
        linestyles[k] = "-"

    return colors, linestyles


# =============================================================================
# Plotting Routines
# =============================================================================

def plot_trajectory_variability_comparison_3d(
    variability_results: Dict[Any, Dict],
    target: str,
    cfg: Dict[str, Any],
    control_br_indices: Optional[List[int]] = None,
) -> Optional[plt.Figure]:
    """
    Consolidated 2-panel figure:
        Top: Time-resolved pairwise 3D Euclidean distance with +/- 1 SD shading.
        Bottom: Kinematics 3D normalized distance with +/- 1 SD shading.
    """
    if not variability_results:
        return None

    def _sort_key(k: Any) -> Tuple[int, str]:
        item = variability_results[k]
        c_num = int(item.get("cond_num", k if isinstance(k, int) else 0))
        tgt = str(item.get("target", ""))
        return (c_num, tgt)

    keys = sorted(variability_results.keys(), key=_sort_key)
    targets_present = sorted(list(set(
        str(variability_results[k].get("target", "")).upper()
        for k in keys if variability_results[k].get("target")
    )))
    is_multi_target = len(targets_present) > 1

    base_labels = {
        k: format_condition_label(
            variability_results[k].get("cond_num", k if isinstance(k, int) else 0),
            variability_results[k],
            control_br_indices,
            include_target=is_multi_target,
        )
        for k in keys
    }
    colors, linestyles = assign_condition_colors_and_styles(keys, base_labels, control_br_indices, variability_results)
    labels = {
        k: f"{base_labels[k]}, n={variability_results[k].get('n_valid', '?')}"
        for k in keys
    }

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 8), sharex=True)
    ax_var = axes[0]
    ax_dist = axes[1]

    # Top Subplot: Time-resolved 3D pairwise variability
    for k in keys:
        var_res = variability_results[k].get("time_resolved", {})
        x_axis = var_res.get("x_axis", None)
        curve = var_res.get("curve", None)
        curve_std = var_res.get("std", None)
        if x_axis is None or curve is None:
            continue

        x_axis = np.asarray(x_axis, dtype=float)
        curve = np.asarray(curve, dtype=float)

        ax_var.plot(
            x_axis,
            curve,
            color=colors[k],
            linestyle=linestyles[k],
            linewidth=2.0,
            label=labels[k],
        )
        if curve_std is not None:
            curve_std = np.asarray(curve_std, dtype=float)
            valid = np.isfinite(curve) & np.isfinite(curve_std)
            lower = np.maximum(0.0, curve - curve_std)
            upper = curve + curve_std
            ax_var.fill_between(
                x_axis,
                lower,
                upper,
                where=valid,
                color=colors[k],
                alpha=0.20,
                linewidth=0,
            )

    t_min_val = PLOT_WINDOW_MS[0] if (PLOT_WINDOW_MS and PLOT_WINDOW_MS[0] is not None) else 0.0
    t_max_val = PLOT_WINDOW_MS[1] if (PLOT_WINDOW_MS and PLOT_WINDOW_MS[1] is not None) else POST_ZERO_MS

    ax_var.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_var.set_xlim(left=t_min_val, right=t_max_val)
    ax_var.set_ylim(bottom=0)
    ax_var.set_ylabel("Pairwise 3D Distance (mm)", fontsize=10)
    ax_var.set_title(
        f"3D Kinematics: Pairwise Trajectory Variability (Target {target}, {t_min_val:.0f} to {t_max_val:.0f} ms)",
        fontsize=11,
        fontweight="bold",
    )
    ax_var.legend(loc="upper left", frameon=False, fontsize=9)
    ax_var.spines["top"].set_visible(False)
    ax_var.spines["right"].set_visible(False)
    ax_var.grid(False)

    # Bottom Subplot: 3D Normalized Distance
    for k in keys:
        dist_res = variability_results[k].get("norm_distance", {})
        t_dist = dist_res.get("time", None)
        dist_mean = dist_res.get("mean", None)
        dist_std = dist_res.get("std", None)
        if t_dist is None or dist_mean is None:
            continue

        t_dist = np.asarray(t_dist, dtype=float)
        dist_mean = np.asarray(dist_mean, dtype=float)

        ax_dist.plot(
            t_dist,
            dist_mean,
            color=colors[k],
            linestyle=linestyles[k],
            linewidth=2.0,
            label=labels[k],
        )
        if dist_std is not None:
            dist_std = np.asarray(dist_std, dtype=float)
            valid = np.isfinite(dist_mean) & np.isfinite(dist_std)
            ax_dist.fill_between(
                t_dist,
                dist_mean - dist_std,
                dist_mean + dist_std,
                where=valid,
                color=colors[k],
                alpha=0.20,
                linewidth=0,
            )

    ax_dist.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_dist.set_xlim(left=t_min_val, right=t_max_val)
    ax_dist.set_ylim(-0.05, 1.15)
    ax_dist.set_xlabel("Time (ms)", fontsize=10)
    ax_dist.set_ylabel("Normalized 3D Distance", fontsize=10)
    ax_dist.set_title(
        f"3D Kinematics: Normalized Distance from Start (Target {target}, {t_min_val:.0f} to {t_max_val:.0f} ms)",
        fontsize=11,
        fontweight="bold",
    )
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)
    ax_dist.grid(False)

    fig.tight_layout()
    return fig


def plot_xy_trajectory_overlay_for_conditions(
    condition_rows: List[Dict],
    target: str,
    cfg: Dict[str, Any],
    title_extra: str = "",
    control_br_indices: Optional[List[int]] = None,
) -> Optional[plt.Figure]:
    """
    Overlay individual reaches in X-Y plane matching the classic 2D plateau analysis layout.
    """
    if len(condition_rows) == 0:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    targets_present = sorted(list(set(str(r.get("target", "")).upper() for r in condition_rows if r.get("target"))))
    is_multi_target = len(targets_present) > 1

    row_keys = [f"{r.get('cond_num', i)}_{r.get('target', '')}_{i}" for i, r in enumerate(condition_rows)]
    cond_dict = dict(zip(row_keys, condition_rows))
    labels = {k: format_condition_label(cond_dict[k]["cond_num"], cond_dict[k], control_br_indices, include_target=is_multi_target) for k in row_keys}
    colors, linestyles = assign_condition_colors_and_styles(row_keys, labels, control_br_indices, cond_dict)

    t_start_val = TRAJECTORY_WINDOW_MS[0] if (TRAJECTORY_WINDOW_MS and TRAJECTORY_WINDOW_MS[0] is not None) else 0.0
    t_end_val = TRAJECTORY_WINDOW_MS[1] if (TRAJECTORY_WINDOW_MS and TRAJECTORY_WINDOW_MS[1] is not None) else 400.0

    for k, row in zip(row_keys, condition_rows):
        t_arr = row.get("time", np.arange(row["x"].shape[1]))
        x, y, _, t_arr = slice_trajectory_to_window(row["x"], row["y"], None, t_arr, window=TRAJECTORY_WINDOW_MS, row=row)
        n_trials = x.shape[0]
        first_labeled = False

        # Locate index closest to t = 0 ms
        idx_0 = int(np.argmin(np.abs(t_arr - 0.0))) if len(t_arr) > 0 else None
        has_t0 = bool(idx_0 is not None and np.abs(t_arr[idx_0] - 0.0) <= 25.0)

        for i in range(n_trials):
            finite = np.isfinite(x[i]) & np.isfinite(y[i])
            if np.sum(finite) < 2:
                continue
            first_idx = int(np.where(finite)[0][0])

            lbl = None
            if not first_labeled:
                lbl = f"{labels[k]}, n={n_trials}"
                first_labeled = True

            ax.plot(
                x[i, finite],
                y[i, finite],
                color=colors[k],
                linestyle=linestyles[k],
                alpha=cfg.get("trajectory_alpha", 0.40),
                linewidth=cfg.get("trajectory_linewidth", 1.0),
                label=lbl,
            )
            # Small starting point marker
            ax.scatter(
                x[i, first_idx],
                y[i, first_idx],
                color=colors[k],
                s=10,
                marker="o",
                alpha=0.6,
                zorder=4,
            )
            # Red dot at t = 0 ms
            if has_t0 and finite[idx_0]:
                ax.scatter(
                    x[i, idx_0],
                    y[i, idx_0],
                    color="red",
                    s=18,
                    marker="o",
                    alpha=0.9,
                    zorder=6,
                )

    # Proxy artist for t = 0 ms in legend
    ax.scatter([], [], color="red", s=22, marker="o", label="t = 0 ms")

    ax.set_title(
        f"Top-Down Reach Trajectories, Target {target}{title_extra}\n"
        f"[Red ●: position at 0 ms | ●: start at {t_start_val:.0f} ms | {t_start_val:.0f} to {t_end_val:.0f} ms trajectories]",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Lateral (mm)", fontsize=10)
    ax.set_ylabel("Forward Reach (mm)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.axis("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def plot_3d_trajectory_multiprojection(
    condition_rows: List[Dict],
    target: str,
    cfg: Dict[str, Any],
    control_br_indices: Optional[List[int]] = None,
) -> Optional[plt.Figure]:
    """
    4-panel multi-projection 3D trajectory figure:
        Panel 1: 3D Trajectories (Full time).
        Panel 2: X-Y Projection (Full time).
        Panel 3: X-Z Projection (Full time).
        Panel 4: Y-Z Projection (Full time).
    Small markers indicate the starting position of each reach.
    """
    if len(condition_rows) == 0:
        return None

    targets_present = sorted(list(set(str(r.get("target", "")).upper() for r in condition_rows if r.get("target"))))
    is_multi_target = len(targets_present) > 1

    row_keys = [f"{r.get('cond_num', i)}_{r.get('target', '')}_{i}" for i, r in enumerate(condition_rows)]
    cond_dict = dict(zip(row_keys, condition_rows))
    labels = {k: format_condition_label(cond_dict[k]["cond_num"], cond_dict[k], control_br_indices, include_target=is_multi_target) for k in row_keys}
    colors, linestyles = assign_condition_colors_and_styles(row_keys, labels, control_br_indices, cond_dict)

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)

    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[1, 0])
    ax_front = fig.add_subplot(gs[1, 1])

    alpha = cfg.get("trajectory_alpha", 0.40)
    lw = cfg.get("trajectory_linewidth", 1.0)

    t_start_val = TRAJECTORY_WINDOW_MS[0] if (TRAJECTORY_WINDOW_MS and TRAJECTORY_WINDOW_MS[0] is not None) else 0.0
    t_end_val = TRAJECTORY_WINDOW_MS[1] if (TRAJECTORY_WINDOW_MS and TRAJECTORY_WINDOW_MS[1] is not None) else 400.0

    for k, row in zip(row_keys, condition_rows):
        t_arr = row.get("time", np.arange(row["x"].shape[1]))
        x, y, z, t_arr = slice_trajectory_to_window(row["x"], row["y"], row["z"], t_arr, window=TRAJECTORY_WINDOW_MS, row=row)
        n_trials = x.shape[0]
        first_labeled = False

        # Locate index closest to t = 0 ms
        idx_0 = int(np.argmin(np.abs(t_arr - 0.0))) if len(t_arr) > 0 else None
        has_t0 = bool(idx_0 is not None and np.abs(t_arr[idx_0] - 0.0) <= 25.0)

        for i in range(n_trials):
            finite = np.isfinite(x[i]) & np.isfinite(y[i]) & np.isfinite(z[i])
            if np.sum(finite) < 2:
                continue
            first_idx = int(np.where(finite)[0][0])

            lbl = None
            if not first_labeled:
                lbl = f"{labels[k]}, n={n_trials}"
                first_labeled = True

            # Panel 1: 3D Trajectories (Lateral, Forward, Height)
            ax_3d.plot(
                x[i, finite],
                y[i, finite],
                z[i, finite],
                color=colors[k],
                linestyle=linestyles[k],
                alpha=alpha,
                linewidth=lw,
                label=lbl,
            )
            ax_3d.scatter(
                x[i, first_idx],
                y[i, first_idx],
                z[i, first_idx],
                color=colors[k],
                s=10,
                marker="o",
                alpha=0.6,
                depthshade=False,
            )

            # Panel 2: Top-Down View (Lateral vs Forward)
            ax_top.plot(x[i, finite], y[i, finite], color=colors[k], linestyle=linestyles[k], alpha=alpha, linewidth=lw)
            ax_top.scatter(x[i, first_idx], y[i, first_idx], color=colors[k], s=10, marker="o", alpha=0.6, zorder=4)

            # Panel 3: Side View / Sagittal (Forward vs Vertical Height)
            ax_side.plot(y[i, finite], z[i, finite], color=colors[k], linestyle=linestyles[k], alpha=alpha, linewidth=lw)
            ax_side.scatter(y[i, first_idx], z[i, first_idx], color=colors[k], s=10, marker="o", alpha=0.6, zorder=4)

            # Panel 4: Front View / Coronal (Lateral vs Vertical Height)
            ax_front.plot(x[i, finite], z[i, finite], color=colors[k], linestyle=linestyles[k], alpha=alpha, linewidth=lw)
            ax_front.scatter(x[i, first_idx], z[i, first_idx], color=colors[k], s=10, marker="o", alpha=0.6, zorder=4)

            # Red dot at t = 0 ms across all projections
            if has_t0 and finite[idx_0]:
                ax_3d.scatter(x[i, idx_0], y[i, idx_0], z[i, idx_0], color="red", s=18, marker="o", alpha=0.95, depthshade=False)
                ax_top.scatter(x[i, idx_0], y[i, idx_0], color="red", s=18, marker="o", alpha=0.9, zorder=6)
                ax_side.scatter(y[i, idx_0], z[i, idx_0], color="red", s=18, marker="o", alpha=0.9, zorder=6)
                ax_front.scatter(x[i, idx_0], z[i, idx_0], color="red", s=18, marker="o", alpha=0.9, zorder=6)

    # Subplot titles and metric axis labels (mm)
    ax_3d.set_title("3D Trajectories", fontsize=11, fontweight="bold")
    ax_3d.set_xlabel("Lateral (mm)", fontsize=9)
    ax_3d.set_ylabel("Forward Reach (mm)", fontsize=9)
    ax_3d.set_zlabel("Vertical Height (mm)", fontsize=9)
    ax_3d.legend(frameon=False, fontsize=8, loc="upper left")

    ax_top.set_title("Top-Down View (Horizontal)", fontsize=11, fontweight="bold")
    ax_top.set_xlabel("Lateral (mm)", fontsize=9)
    ax_top.set_ylabel("Forward Reach (mm)", fontsize=9)
    ax_top.axis("equal")
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    ax_side.set_title("Side View (Sagittal)", fontsize=11, fontweight="bold")
    ax_side.set_xlabel("Forward Reach (mm)", fontsize=9)
    ax_side.set_ylabel("Vertical Height (mm)", fontsize=9)
    ax_side.axis("equal")
    ax_side.spines["top"].set_visible(False)
    ax_side.spines["right"].set_visible(False)

    ax_front.set_title("Front View (Coronal)", fontsize=11, fontweight="bold")
    ax_front.set_xlabel("Lateral (mm)", fontsize=9)
    ax_front.set_ylabel("Vertical Height (mm)", fontsize=9)
    ax_front.axis("equal")
    ax_front.spines["top"].set_visible(False)
    ax_front.spines["right"].set_visible(False)

    fig.suptitle(
        f"3D DLC Reaching Kinematics - Target {target}\n"
        f"[{t_start_val:.0f} to {t_end_val:.0f} ms trajectories | Red ●: position at 0 ms | ●: start at {t_start_val:.0f} ms]",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    return fig


def plot_interactive_3d_trajectories(
    condition_rows: List[Dict],
    target: str,
    cfg: Dict[str, Any],
    control_br_indices: Optional[List[int]] = None,
    save_path: Optional[Path] = None,
) -> Optional[Any]:
    """
    Generate an interactive 3D reaching trajectories figure using Plotly.
    Allows 3D rotating, panning, zooming, and per-trial hover tooltips.
    Features:
      - Full time trajectories for all trials.
      - Small markers indicating the exact starting point of each reach.
      - Color-coded conditions matching Matplotlib figures (Orange for Control, Bone for Stim).
      - Target A (solid) vs Target B (dashed) line distinction when multiple targets are present.
      - Clickable legend to toggle conditions or trials on and off.
    """
    if not PLOTLY_AVAILABLE or len(condition_rows) == 0:
        return None

    targets_present = sorted(list(set(str(r.get("target", "")).upper() for r in condition_rows if r.get("target"))))
    is_multi_target = len(targets_present) > 1

    row_keys = [f"{r.get('cond_num', i)}_{r.get('target', '')}_{i}" for i, r in enumerate(condition_rows)]
    cond_dict = dict(zip(row_keys, condition_rows))
    labels = {k: format_condition_label(cond_dict[k]["cond_num"], cond_dict[k], control_br_indices, include_target=is_multi_target) for k in row_keys}
    colors, linestyles = assign_condition_colors_and_styles(row_keys, labels, control_br_indices, cond_dict)

    fig = go.Figure()

    t_start_val = TRAJECTORY_WINDOW_MS[0] if (TRAJECTORY_WINDOW_MS and TRAJECTORY_WINDOW_MS[0] is not None) else 0.0
    t_end_val = TRAJECTORY_WINDOW_MS[1] if (TRAJECTORY_WINDOW_MS and TRAJECTORY_WINDOW_MS[1] is not None) else 400.0

    for k, row in zip(row_keys, condition_rows):
        t_arr = row.get("time", np.arange(row["x"].shape[1]))
        x, y, z, time_arr = slice_trajectory_to_window(row["x"], row["y"], row["z"], t_arr, window=TRAJECTORY_WINDOW_MS, row=row)
        n_trials = x.shape[0]

        rgba = colors[k]
        col_str =   f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 0.92)"
        start_col_str = f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 1.0)"

        tgt = str(row.get("target", "")).upper()
        dash_style = "solid" if is_multi_target else ("dash" if linestyles.get(k) == "--" else "solid")

        idx_0 = int(np.argmin(np.abs(time_arr - 0.0))) if len(time_arr) > 0 else None
        has_t0 = bool(idx_0 is not None and np.abs(time_arr[idx_0] - 0.0) <= 25.0)

        x_starts, y_starts, z_starts = [], [], []
        x_zeros, y_zeros, z_zeros = [], [], []

        for i in range(n_trials):
            finite = np.isfinite(x[i]) & np.isfinite(y[i]) & np.isfinite(z[i])
            if np.sum(finite) < 2:
                continue

            first_idx = int(np.where(finite)[0][0])
            x_starts.append(float(x[i, first_idx]))
            y_starts.append(float(y[i, first_idx]))
            z_starts.append(float(z[i, first_idx]))

            if has_t0 and finite[idx_0]:
                x_zeros.append(float(x[i, idx_0]))
                y_zeros.append(float(y[i, idx_0]))
                z_zeros.append(float(z[i, idx_0]))

            xi = x[i, finite]
            yi = y[i, finite]
            zi = z[i, finite]
            ti = time_arr[finite]

            hover_text = [
                f"<b>{labels[k]}</b><br>Trial: {i+1}/{n_trials}<br>Time: {t_val:.1f} ms<br>"
                f"Lateral: {xp:.1f} mm<br>Forward: {yp:.1f} mm<br>Height: {zp:.1f} mm"
                for xp, yp, zp, t_val in zip(xi, yi, zi, ti)
            ]

            fig.add_trace(
                go.Scatter3d(
                    x=xi,
                    y=yi,
                    z=zi,
                    mode="lines",
                    line=dict(color=col_str, width=4.0, dash=dash_style),
                    name=f"{labels[k]} (n={n_trials})",
                    legendgroup=k,
                    showlegend=(i == 0),
                    hoverinfo="text",
                    hovertext=hover_text,
                )
            )

        if len(x_starts) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=x_starts,
                    y=y_starts,
                    z=z_starts,
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=start_col_str,
                        symbol="circle",
                        line=dict(color="rgba(30, 30, 30, 0.75)", width=0.7),
                    ),
                    name=f"{labels[k]} [Start]",
                    legendgroup=k,
                    showlegend=False,
                    hoverinfo="text",
                    hovertext=[f"<b>{labels[k]}</b> Start Point" for _ in x_starts],
                )
            )

        if len(x_zeros) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=x_zeros,
                    y=y_zeros,
                    z=z_zeros,
                    mode="markers",
                    marker=dict(
                        size=5,
                        color="red",
                        symbol="circle",
                        line=dict(color="darkred", width=0.8),
                    ),
                    name="t = 0 ms",
                    legendgroup="t0",
                    showlegend=(k == row_keys[0]),
                    hoverinfo="text",
                    hovertext=[f"<b>{labels[k]}</b> at t = 0 ms" for _ in x_zeros],
                )
            )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>3D DLC Reaching Kinematics - Target {target}</b><br>"
                f"<sup>[{t_start_val:.0f} to {t_end_val:.0f} ms trajectories | "
                f"Red ●: position at 0 ms | ●: start at {t_start_val:.0f} ms]</sup>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=15),
        ),
        scene=dict(
            xaxis=dict(title="Lateral (mm)", showbackground=True, backgroundcolor="rgba(245, 245, 245, 0.6)"),
            yaxis=dict(title="Forward Reach (mm)", showbackground=True, backgroundcolor="rgba(245, 245, 245, 0.6)"),
            zaxis=dict(title="Vertical Height (mm)", showbackground=True, backgroundcolor="rgba(245, 245, 245, 0.6)"),
            aspectmode="data",
        ),
        template="plotly_white",
        legend=dict(
            itemsizing="constant",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(200, 200, 200, 0.5)",
            borderwidth=1,
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.02,
        ),
        margin=dict(l=0, r=0, b=0, t=60),
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path), include_plotlyjs=True, full_html=True)
        _log(f"Saved interactive 3D trajectories figure: {save_path.name}")

    return fig


def plot_condition_summary_3d(
    cond_result: Dict[str, Any],
    cond_num: int,
    target: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate comprehensive per-condition diagnostic figure showing X, Y, Z, distance, and 3D paths."""
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.32, wspace=0.25)

    time = cond_result["aligned_time"]
    x = cond_result["x_smooth_all"]
    y = cond_result["y_smooth_all"]
    z = cond_result["z_smooth_all"]
    dist_norm = cond_result["dist_norm"]
    dist_mean = cond_result["dist_mean"]
    dist_std = cond_result["dist_std"]
    valid_mask = cond_result["valid_mask"]

    # 1. Normalized distance
    ax_dist = fig.add_subplot(gs[0, 0])
    for i in range(len(dist_norm)):
        if valid_mask[i]:
            ax_dist.plot(time, dist_norm[i], color="gray", alpha=0.35, linewidth=0.8)

    t_start_val = PLOT_WINDOW_MS[0] if (PLOT_WINDOW_MS and PLOT_WINDOW_MS[0] is not None) else float(time[0])
    t_end_val = PLOT_WINDOW_MS[1] if (PLOT_WINDOW_MS and PLOT_WINDOW_MS[1] is not None) else float(time[-1])

    ax_dist.plot(time, dist_mean, color="navy", linewidth=2.2, label="Mean +/- 1 SD")
    if np.any(np.isfinite(dist_std)):
        ax_dist.fill_between(time, dist_mean - dist_std, dist_mean + dist_std, color="navy", alpha=0.18, linewidth=0)
    ax_dist.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax_dist.set_xlim(left=t_start_val, right=t_end_val)
    ax_dist.set_ylim(-0.05, 1.15)
    ax_dist.set_xlabel(f"Time from Stim / IR (ms)  [{t_start_val:.0f} to {t_end_val:.0f} ms]", fontsize=8)
    ax_dist.set_ylabel("Normalized Distance", fontsize=9)
    ax_dist.set_title(f"3D Normalized Distance ({t_start_val:.0f} to {t_end_val:.0f} ms)", fontsize=10, fontweight="bold")
    ax_dist.legend(loc="upper left", frameon=False, fontsize=8)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)

    # 2. Coordinates X, Y, Z (Mean +/- 1 SD shaded)
    ax_coords = fig.add_subplot(gs[0, 1])
    coord_info = [
        ("X", x, "Lateral (X)", "firebrick"),
        ("Y", y, "Forward (Y)", "forestgreen"),
        ("Z", z, "Height (Z)", "royalblue"),
    ]
    for name, data, lbl_name, col in coord_info:
        if np.any(valid_mask):
            m = np.nanmean(data[valid_mask], axis=0)
            s = np.nanstd(data[valid_mask], axis=0, ddof=1) if np.sum(valid_mask) > 1 else np.zeros_like(m)
            ax_coords.plot(time, m, color=col, linewidth=2.0, label=lbl_name)
            ax_coords.fill_between(time, m - s, m + s, color=col, alpha=0.18, linewidth=0)
    ax_coords.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax_coords.set_xlim(left=t_start_val, right=t_end_val)
    ax_coords.set_xlabel(f"Time from Stim / IR (ms)  [{t_start_val:.0f} to {t_end_val:.0f} ms]", fontsize=8)
    ax_coords.set_ylabel("Position (mm)", fontsize=9)
    ax_coords.set_title(f"Mean Task Coordinates +/- 1 SD (mm, {t_start_val:.0f} to {t_end_val:.0f} ms)", fontsize=10, fontweight="bold")
    ax_coords.legend(loc="upper left", frameon=False, fontsize=8)
    ax_coords.spines["top"].set_visible(False)
    ax_coords.spines["right"].set_visible(False)

    t_start_val = TRAJECTORY_WINDOW_MS[0] if (TRAJECTORY_WINDOW_MS and TRAJECTORY_WINDOW_MS[0] is not None) else 0.0
    t_end_val = TRAJECTORY_WINDOW_MS[1] if (TRAJECTORY_WINDOW_MS and TRAJECTORY_WINDOW_MS[1] is not None) else 400.0

    # 3-6. Projections and 3D View (Trajectory window: e.g. 0 to 400 ms)
    t_traj_raw = cond_result.get("aligned_time", np.arange(x.shape[1]))
    x_traj, y_traj, z_traj, t_traj = slice_trajectory_to_window(
        cond_result["x_smooth_all"],
        cond_result["y_smooth_all"],
        cond_result["z_smooth_all"],
        t_traj_raw,
        window=TRAJECTORY_WINDOW_MS,
        row=cond_result,
    )

    idx_0 = int(np.argmin(np.abs(t_traj - 0.0))) if len(t_traj) > 0 else None
    has_t0 = bool(idx_0 is not None and np.abs(t_traj[idx_0] - 0.0) <= 25.0)

    # 3. Top-Down View: Lateral (X) vs Forward (Y)
    ax_xy = fig.add_subplot(gs[1, 0])
    for i in range(len(x_traj)):
        if valid_mask[i]:
            finite = np.isfinite(x_traj[i]) & np.isfinite(y_traj[i])
            if np.sum(finite) >= 2:
                first_idx = int(np.where(finite)[0][0])
                ax_xy.plot(x_traj[i, finite], y_traj[i, finite], color="gray", alpha=0.40, linewidth=0.9)
                ax_xy.scatter(x_traj[i, first_idx], y_traj[i, first_idx], color="darkslategray", s=10, marker="o", alpha=0.8, zorder=4)
                if has_t0 and finite[idx_0]:
                    ax_xy.scatter(x_traj[i, idx_0], y_traj[i, idx_0], color="red", s=14, marker="o", alpha=0.9, zorder=6)
    ax_xy.set_title("Top-Down View (Horizontal)", fontsize=10, fontweight="bold")
    ax_xy.set_xlabel("Lateral (mm)", fontsize=8)
    ax_xy.set_ylabel("Forward Reach (mm)", fontsize=8)
    ax_xy.axis("equal")
    ax_xy.spines["top"].set_visible(False)
    ax_xy.spines["right"].set_visible(False)

    # 4. Side View: Forward (Y) vs Height (Z)
    ax_side = fig.add_subplot(gs[1, 1])
    for i in range(len(y_traj)):
        if valid_mask[i]:
            finite = np.isfinite(y_traj[i]) & np.isfinite(z_traj[i])
            if np.sum(finite) >= 2:
                first_idx = int(np.where(finite)[0][0])
                ax_side.plot(y_traj[i, finite], z_traj[i, finite], color="gray", alpha=0.40, linewidth=0.9)
                ax_side.scatter(y_traj[i, first_idx], z_traj[i, first_idx], color="darkslategray", s=10, marker="o", alpha=0.8, zorder=4)
                if has_t0 and finite[idx_0]:
                    ax_side.scatter(y_traj[i, idx_0], z_traj[i, idx_0], color="red", s=14, marker="o", alpha=0.9, zorder=6)
    ax_side.set_title("Side View (Sagittal)", fontsize=10, fontweight="bold")
    ax_side.set_xlabel("Forward Reach (mm)", fontsize=8)
    ax_side.set_ylabel("Vertical Height (mm)", fontsize=8)
    ax_side.axis("equal")
    ax_side.spines["top"].set_visible(False)
    ax_side.spines["right"].set_visible(False)

    # 5. Front View: Lateral (X) vs Height (Z)
    ax_front = fig.add_subplot(gs[2, 0])
    for i in range(len(x_traj)):
        if valid_mask[i]:
            finite = np.isfinite(x_traj[i]) & np.isfinite(z_traj[i])
            if np.sum(finite) >= 2:
                first_idx = int(np.where(finite)[0][0])
                ax_front.plot(x_traj[i, finite], z_traj[i, finite], color="gray", alpha=0.40, linewidth=0.9)
                ax_front.scatter(x_traj[i, first_idx], z_traj[i, first_idx], color="darkslategray", s=10, marker="o", alpha=0.8, zorder=4)
                if has_t0 and finite[idx_0]:
                    ax_front.scatter(x_traj[i, idx_0], z_traj[i, idx_0], color="red", s=14, marker="o", alpha=0.9, zorder=6)
    ax_front.set_title("Front View (Coronal)", fontsize=10, fontweight="bold")
    ax_front.set_xlabel("Lateral (mm)", fontsize=8)
    ax_front.set_ylabel("Vertical Height (mm)", fontsize=8)
    ax_front.axis("equal")
    ax_front.spines["top"].set_visible(False)
    ax_front.spines["right"].set_visible(False)

    # 6. 3D View
    ax_3d = fig.add_subplot(gs[2, 1], projection="3d")
    for i in range(len(x_traj)):
        if valid_mask[i]:
            finite = np.isfinite(x_traj[i]) & np.isfinite(y_traj[i]) & np.isfinite(z_traj[i])
            if np.sum(finite) >= 2:
                first_idx = int(np.where(finite)[0][0])
                ax_3d.plot(x_traj[i, finite], y_traj[i, finite], z_traj[i, finite], color="steelblue", alpha=0.45, linewidth=1.0)
                ax_3d.scatter(x_traj[i, first_idx], y_traj[i, first_idx], z_traj[i, first_idx], color="darkslategray", s=10, marker="o", alpha=0.85, depthshade=False)
                if has_t0 and finite[idx_0]:
                    ax_3d.scatter(x_traj[i, idx_0], y_traj[i, idx_0], z_traj[i, idx_0], color="red", s=14, marker="o", alpha=0.95, depthshade=False)
    ax_3d.set_title("3D Trajectories", fontsize=10, fontweight="bold")
    ax_3d.set_xlabel("Lateral (mm)", fontsize=8)
    ax_3d.set_ylabel("Forward (mm)", fontsize=8)
    ax_3d.set_zlabel("Height (mm)", fontsize=8)

    lbl = format_condition_label(cond_num, cond_result, pulse_width_ms=PULSE_WIDTH_MS)
    fig.suptitle(
        f"{lbl} - Target {target} (n={cond_result['n_valid']} reaches)\n"
        f"[Distance & coordinates: t ≥ 0 ms | Trajectories: {t_start_val:.0f} to {t_end_val:.0f} ms | Red ●: position at 0 ms | ●: start]",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    if save_path:
        fig.savefig(save_path, format=FIG_FORMAT, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_trial_inspection_grid_3d(
    cond_proc: Dict[str, Any],
    br_idx: int,
    target: str,
    metadata: Dict[str, Any],
    raw_trial_indices: Optional[Sequence[int]] = None,
    save_dir: Optional[Path] = None,
    n_cols: int = 5,
) -> Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
    """
    Generate multi-trial inspection grids for visual inspection (similar to inspect_kinematics_trajectories.py):
      1. X-Y Trajectory Grid: LineCollection colored by time (viridis), start (green), end (red), t=0 (blue x).
      2. 3D Normalized Distance Grid: Reach distance vs time with baseline reference and pre-stim window.
    
    Subplot titles show: 'Trial {i+1} (Raw: {raw_idx})' to easily locate bad traces for manual_trial_remove.csv.
    """
    t_start_val = TRAJECTORY_WINDOW_MS[0] if (TRAJECTORY_WINDOW_MS and TRAJECTORY_WINDOW_MS[0] is not None) else 0.0
    t_end_val = TRAJECTORY_WINDOW_MS[1] if (TRAJECTORY_WINDOW_MS and TRAJECTORY_WINDOW_MS[1] is not None) else 400.0

    t_axis_traj = cond_proc.get("aligned_time", np.arange(cond_proc["x_smooth_all"].shape[1]))
    x, y, _, t_full = slice_trajectory_to_window(
        cond_proc["x_smooth_all"],
        cond_proc["y_smooth_all"],
        None,
        t_axis_traj,
        window=TRAJECTORY_WINDOW_MS,
        row=cond_proc,
    )
    dist_norm = cond_proc["dist_norm"]
    t_dist = cond_proc["aligned_time"]
    valid_mask = cond_proc["valid_mask"]

    n_trials = len(x)
    if n_trials == 0:
        return None, None

    n_rows = math.ceil(n_trials / n_cols)
    lbl = format_condition_label(br_idx, metadata, pulse_width_ms=PULSE_WIDTH_MS)

    # -------------------------------------------------------------
    # 1. X-Y Trajectory Grid (matching inspect_kinematics_trajectories style)
    # -------------------------------------------------------------
    fig_xy, axes_xy = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2.5 * n_rows),
        constrained_layout=True
    )
    if n_rows == 1:
        axes_xy = np.array([axes_xy])
    if axes_xy.ndim == 1:
        axes_xy = axes_xy.reshape(1, -1)
    axes_xy_flat = axes_xy.flatten()

    norm_t = Normalize(vmin=float(t_full.min()), vmax=float(t_full.max()))

    for i in range(n_trials):
        ax = axes_xy_flat[i]
        xi = x[i]
        yi = y[i]

        raw_idx = int(raw_trial_indices[i]) if (raw_trial_indices is not None and i < len(raw_trial_indices)) else (i + 1)
        sub_title = f"Trial {i+1} (Raw:{raw_idx})"

        finite = np.isfinite(xi) & np.isfinite(yi)
        if np.sum(finite) < 2:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(sub_title, fontsize=7)
            continue

        pts = np.array([xi[finite], yi[finite]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

        lc = LineCollection(segs, cmap="viridis", norm=norm_t, linewidth=1.5, alpha=0.85)
        lc.set_array(t_full[finite][:-1])
        ax.add_collection(lc)

        ax.scatter(xi[finite][0], yi[finite][0], c="green", s=25, zorder=5, label="Start")
        ax.scatter(xi[finite][-1], yi[finite][-1], c="red", s=25, zorder=5, label="End")

        t0_idx = int(np.argmin(np.abs(t_full)))
        if 0 < t0_idx < len(xi) and np.isfinite(xi[t0_idx]) and np.isfinite(yi[t0_idx]):
            ax.scatter(xi[t0_idx], yi[t0_idx], c="blue", s=35, marker="x", zorder=6, linewidths=1.8)

        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(sub_title, fontsize=7)
        ax.tick_params(labelsize=5)

    for j in range(n_trials, len(axes_xy_flat)):
        axes_xy_flat[j].axis("off")

    fig_xy.suptitle(
        f"{lbl} - Target {target} (n={n_trials}) | Top-Down Reach Trajectories ({t_start_val:.0f} to {t_end_val:.0f} ms)\n"
        "[Green: Start | Red: End | Blue x: t=0 | Put Raw:# in manual_trial_remove.csv to curate]",
        fontsize=10,
        fontweight="bold"
    )
    sm_xy = plt.cm.ScalarMappable(cmap="viridis", norm=norm_t)
    sm_xy.set_array([])
    cbar_xy = fig_xy.colorbar(sm_xy, ax=axes_xy_flat[:n_trials], shrink=0.6, pad=0.02, aspect=30)
    cbar_xy.set_label("Time from Stim / IR (ms)", fontsize=8)

    # -------------------------------------------------------------
    # 2. 3D Normalized Distance Grid (inspect baseline stability / bumps)
    # -------------------------------------------------------------
    fig_dist, axes_dist = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2.2 * n_rows),
        constrained_layout=True,
        sharex=True,
        sharey=True
    )
    if n_rows == 1:
        axes_dist = np.array([axes_dist])
    if axes_dist.ndim == 1:
        axes_dist = axes_dist.reshape(1, -1)
    axes_dist_flat = axes_dist.flatten()

    dist_full = cond_proc.get("dist_norm_full")
    time_full = cond_proc.get("full_time")

    for i in range(n_trials):
        ax = axes_dist_flat[i]
        di = dist_norm[i]

        raw_idx = int(raw_trial_indices[i]) if (raw_trial_indices is not None and i < len(raw_trial_indices)) else (i + 1)

        # Check pre-movement deviation in [-500, -100] ms using full un-sliced trace if available
        if dist_full is not None and time_full is not None and i < len(dist_full):
            pre_mask = (time_full >= -500.0) & (time_full <= -100.0)
            dfi = dist_full[i]
            max_pre_dev = float(np.nanmax(dfi[pre_mask])) if (np.any(pre_mask) and np.any(np.isfinite(dfi[pre_mask]))) else 0.0
            is_shaky = max_pre_dev > 0.25
        else:
            is_shaky = False
            max_pre_dev = 0.0

        sub_title = f"Trial {i+1} (Raw:{raw_idx})"
        if is_shaky:
            sub_title += f" [Pre-dev:{max_pre_dev:.2f}]"

        color_line = "firebrick" if is_shaky else "steelblue"
        title_color = "red" if is_shaky else "black"

        ax.plot(t_dist, di, color=color_line, linewidth=1.4, alpha=0.9)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_xlim(left=t_start_val, right=t_end_val)
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(sub_title, fontsize=7, color=title_color)
        ax.tick_params(labelsize=5)

    for j in range(n_trials, len(axes_dist_flat)):
        axes_dist_flat[j].axis("off")

    fig_dist.suptitle(
        f"{lbl} - Target {target} (n={n_trials}) | 3D Normalized Distance Profiles ({t_start_val:.0f} to {t_end_val:.0f} ms)\n"
        "[Distance normalized to pre-stim resting baseline | Put Raw:# in manual_trial_remove.csv to curate]",
        fontsize=10,
        fontweight="bold"
    )

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_xy = save_dir / f"BR_{br_idx:03d}_Target{target}_trials_xy.png"
        out_dist = save_dir / f"BR_{br_idx:03d}_Target{target}_trials_distance.png"
        fig_xy.savefig(out_xy, dpi=150)
        fig_dist.savefig(out_dist, dpi=150)
        _log(f"Saved trial inspection grids for BR {br_idx}: {out_xy.name}, {out_dist.name}")
        plt.close(fig_xy)
        plt.close(fig_dist)

    return fig_xy, fig_dist


def plot_combined_targets_comparisons(
    results_per_target: Dict[str, Dict[str, Any]],
    comp_dir: Path,
    cond_tag: str,
    cfg: Dict[str, Any],
    control_br_indices: Optional[List[int]] = None,
) -> None:
    """
    Generate unified comparison figures containing both targets (Target A and Target B):
      1. 2-Panel Variability & Normalized Distance comparison (Target A: Gold/Grey, Target B: Orange/Dark Green, solid lines).
      2. 4-Panel 3D & planar multi-projection overlay showing both targets in 3D, X-Y, X-Z, Y-Z with red dots at 0 ms.
      3. Classic 2D X-Y overlay showing both targets bifurcating to button A and button B with red dots at 0 ms.
    """
    if len(results_per_target) < 2:
        return

    combined_rows: List[Dict[str, Any]] = []
    combined_variability: Dict[str, Dict[str, Any]] = {}

    for tgt in sorted(results_per_target.keys()):
        tgt_res = results_per_target[tgt]
        rows = tgt_res.get("trajectory_rows", [])
        combined_rows.extend(rows)

        var_dict = tgt_res.get("variability_results", {})
        for c_num, v_data in var_dict.items():
            key = f"BR_{c_num}_Target_{tgt}"
            v_copy = dict(v_data)
            v_copy["target"] = tgt
            combined_variability[key] = v_copy

    if not combined_rows:
        return

    _log("Generating unified multi-target comparison figures (Targets A & B)...")
    target_label = " & ".join(sorted(results_per_target.keys()))

    # A. 2-Panel Variability & Distance Comparison for Both Targets
    fig_var = plot_trajectory_variability_comparison_3d(
        combined_variability,
        target=target_label,
        cfg=cfg,
        control_br_indices=control_br_indices,
    )
    if fig_var is not None and SAVE_FIGS:
        for tag in ["TargetBoth", "TargetAll"]:
            out_var = comp_dir / f"trajectory_variability_3D_{tag}_conditions_{cond_tag}.{FIG_FORMAT}"
            fig_var.savefig(out_var, format=FIG_FORMAT, dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved multi-target variability comparison figure: trajectory_variability_3D_TargetBoth_conditions_{cond_tag}.{FIG_FORMAT}")
        plt.close(fig_var)

    # B. Multi-Projection 3D Trajectory Overlay for Both Targets
    fig_proj = plot_3d_trajectory_multiprojection(
        combined_rows,
        target=target_label,
        cfg=cfg,
        control_br_indices=control_br_indices,
    )
    if fig_proj is not None and SAVE_FIGS:
        for tag in ["TargetBoth", "TargetAll"]:
            out_proj = comp_dir / f"trajectory_3D_multiprojection_{tag}_conditions_{cond_tag}.{FIG_FORMAT}"
            fig_proj.savefig(out_proj, format=FIG_FORMAT, dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved multi-target 3D multi-projection figure: trajectory_3D_multiprojection_TargetBoth_conditions_{cond_tag}.{FIG_FORMAT}")
        plt.close(fig_proj)

    # C. Classic 2D X-Y Overlay for Both Targets
    fig_xy = plot_xy_trajectory_overlay_for_conditions(
        combined_rows,
        target=target_label,
        cfg=cfg,
        title_extra=" (3D DLC)",
        control_br_indices=control_br_indices,
    )
    if fig_xy is not None and SAVE_FIGS:
        for tag in ["TargetBoth", "TargetAll"]:
            out_xy = comp_dir / f"trajectory_xy_overlay_3D_{tag}_conditions_{cond_tag}.{FIG_FORMAT}"
            fig_xy.savefig(out_xy, format=FIG_FORMAT, dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved multi-target X-Y overlay figure: trajectory_xy_overlay_3D_TargetBoth_conditions_{cond_tag}.{FIG_FORMAT}")
        plt.close(fig_xy)

    # D. Interactive 3D Trajectory Overlay for Both Targets (Plotly HTML)
    if SAVE_FIGS and PLOTLY_AVAILABLE:
        for tag in ["TargetBoth", "TargetAll"]:
            out_html = comp_dir / f"trajectory_3D_interactive_{tag}_conditions_{cond_tag}.html"
            plot_interactive_3d_trajectories(
                combined_rows,
                target=target_label,
                cfg=cfg,
                control_br_indices=control_br_indices,
                save_path=out_html,
            )


# =============================================================================
# Batch Analysis Runner
# =============================================================================

def run_3d_kinematics_analysis(
    session: str,
    dlc_3d_dir: Path,
    peri_root: Path,
    target: str = "A",
    keypoint: str = DEFAULT_KEYPOINT,
    compare_conditions: Optional[List[int]] = None,
    control_br_indices: Optional[List[int]] = None,
    save_dir: Optional[Path] = None,
    metadata_csv: Optional[Path] = None,
    aligned_ckpt_root: Optional[Path] = None,
    behv_ckpt_root: Optional[Path] = None,
    only_post_zero: Optional[bool] = None,
    square_size_mm: float = SQUARE_SIZE_MM,
    align_task_frame: bool = ALIGN_TO_TASK_FRAME,
    plot_window: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> Dict[str, Any]:
    """Execute complete 3D distance and trajectory variability analysis."""
    if plot_window is None:
        plot_window = PLOT_WINDOW_MS
    if save_dir is None:
        if OUT_BASE is not None:
            save_dir = OUT_BASE / "figures" / "3d_kinematics"
        else:
            save_dir = Path.cwd() / "results" / "3d_kinematics"

    per_cond_dir = save_dir / "per_condition"
    comp_dir = save_dir / "comparisons"
    trial_inspect_dir = save_dir / "trial_inspection" / f"Target_{target}"

    for d in [save_dir, per_cond_dir, comp_dir, trial_inspect_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load session metadata CSV
    meta_csv_path = find_metadata_csv(session=session, metadata_csv=metadata_csv, peri_root=peri_root)
    session_meta = load_session_metadata(meta_csv_path)

    # Pre-scan session landmarks for task-frame alignment
    session_landmarks = None
    if align_task_frame and dlc_3d_dir and Path(dlc_3d_dir).exists():
        session_landmarks = scan_session_landmarks(dlc_3d_dir, square_size_mm=square_size_mm)

    _log("=" * 70)
    _log(f"Starting 3D Kinematics Analysis for Session: {session}")
    _log(f"  DLC_3D Directory: {dlc_3d_dir}")
    _log(f"  PeriStim Root   : {peri_root}")
    _log(f"  Target          : {target}")
    _log(f"  Keypoint        : {keypoint}")
    _log(f"  Metric Scale    : {square_size_mm:.1f} mm/unit")
    _log(f"  Task-Frame Align: {align_task_frame}")
    if compare_conditions is not None:
        _log(f"  Comparing Conds : {sorted(compare_conditions)}")
    else:
        _log("  Comparing Conds : All discovered conditions")
    if control_br_indices:
        _log(f"  Control BRs     : {sorted(control_br_indices)}")
    else:
        _log("  Control BRs     : Auto-detected from metadata")
    if session_meta and meta_csv_path:
        _log(f"  Session Metadata: Loaded {len(session_meta)} condition entries from {meta_csv_path.name}")
    _log(f"  Figures Dir     : {save_dir}")
    _log("=" * 70)

    # 1. Discover PeriStim files
    peri_search_patterns = [
        peri_root / "stim_reaches" / f"target_{target}" / "*.npz",
        peri_root / "stim_reaches" / f"target_{target.upper()}" / "*.npz",
        peri_root / "stim_reaches" / f"target_{target.lower()}" / "*.npz",
        peri_root / "control_reaches" / f"target_{target}" / "*.npz",
        peri_root / "control_reaches" / f"target_{target.upper()}" / "*.npz",
        peri_root / "control_reaches" / f"target_{target.lower()}" / "*.npz",
        peri_root / f"Target_{target}" / "*.npz",
        peri_root / f"Target_{target.upper()}" / "*.npz",
        peri_root / f"*target_{target.lower()}*.npz",
        peri_root / f"*target_{target.upper()}*.npz",
        peri_root / "*.npz",
    ]

    discovered_peri_files: List[Path] = []
    for pat in peri_search_patterns:
        if pat.parent.exists():
            for f in sorted(list(pat.parent.glob(pat.name))):
                if f not in discovered_peri_files and not f.name.startswith("._"):
                    discovered_peri_files.append(f)

    if not discovered_peri_files:
        _log(f"No PeriStim files found for target {target} in {peri_root}")
        return {}

    _log(f"Discovered {len(discovered_peri_files)} candidate PeriStim file(s).")

    all_condition_results: Dict[str, Any] = {}
    trajectory_rows_for_overlay: List[Dict[str, Any]] = []

    # Cache for loaded 3D DLC CSVs to avoid redundant disk I/O
    dlc_df_cache: Dict[Path, Tuple[pd.DataFrame, List[str]]] = {}

    for peri_path in discovered_peri_files:
        path_str = str(peri_path).replace("\\", "/").lower()
        other_tgt = "b" if target.upper() == "A" else "a"
        if f"target_{other_tgt}" in path_str or f"target{other_tgt}" in path_str:
            if f"target_{target.lower()}" not in path_str and f"target{target.lower()}" not in path_str:
                continue

        try:
            peri = np.load(peri_path, allow_pickle=True)
        except Exception as e:
            _log(f"Could not load {peri_path.name}: {e}")
            continue

        # Extract BR index
        br_idx = -1
        if "br_idx" in peri:
            try:
                br_idx = int(peri["br_idx"])
            except Exception:
                pass
        if br_idx < 0:
            m = re.search(r"BR_(\d+)", peri_path.name, re.IGNORECASE)
            if m:
                br_idx = int(m.group(1))

        if br_idx < 0:
            continue

        # Condition metadata from CSV
        csv_meta = session_meta.get(br_idx, {})
        notes = str(csv_meta.get("notes", "")).strip()

        # Check if condition is control
        is_ctrl_candidate = bool(
            (control_br_indices and br_idx in control_br_indices)
            or "control_reaches" in str(peri_path).replace("\\", "/").lower()
            or "control" in peri_path.stem.lower()
            or "ctrl" in peri_path.stem.lower()
            or (notes and notes.lower() in ("baseline", "control", "ctrl"))
        )

        # If comparing specific conditions, process requested conditions plus control candidates if none in compare_conditions
        if compare_conditions:
            has_ctrl_in_req = any(
                (control_br_indices and c in control_br_indices)
                or (session_meta.get(c, {}).get("notes", "").strip().lower() in ("baseline", "control", "ctrl"))
                for c in compare_conditions
            )
            if has_ctrl_in_req:
                if br_idx not in compare_conditions:
                    continue
            else:
                if (br_idx not in compare_conditions) and not is_ctrl_candidate:
                    continue

        # Extract event timestamps
        event_ms = None
        for k in ["event_ms_sub", "event_ms"]:
            if k in peri:
                event_ms = peri[k]
                break

        if event_ms is None or event_ms.size == 0:
            _log(f"Skipping {peri_path.name}: no reach events.")
            continue

        raw_trial_indices = None
        for k in ["raw_trial_indices", "raw_indices", "trial_indices", "raw_trial_idx"]:
            if k in peri:
                raw_trial_indices = peri[k]
                break

        # Filter by target label if multiple targets coexist in file
        labels = None
        for k in ["labels_sub", "labels", "trial_labels"]:
            if k in peri:
                labels = peri[k]
                break
        if labels is not None and np.size(labels) == np.size(event_ms):
            labels_arr = np.asarray(labels).astype(str)
            t_mask = np.array([lbl.upper() == target.upper() for lbl in labels_arr])
            if not np.any(t_mask):
                continue
            event_ms = event_ms[t_mask]
            if raw_trial_indices is not None and len(raw_trial_indices) == len(t_mask):
                raw_trial_indices = raw_trial_indices[t_mask]

        if event_ms.size == 0:
            continue

        # Parse condition metadata
        peri_meta = {}
        if "meta" in peri and hasattr(peri["meta"], "item"):
            try:
                peri_meta = peri["meta"].item()
            except Exception:
                pass
        elif "align_meta_raw" in peri and hasattr(peri["align_meta_raw"], "item"):
            try:
                raw_m = peri["align_meta_raw"].item()
                peri_meta = json.loads(raw_m) if isinstance(raw_m, str) else raw_m
            except Exception:
                pass

        vid_idx = resolve_video_index_for_br(br_idx, metadata_csv, peri_meta)
        if vid_idx is None:
            vid_idx = br_idx

        # 2. Find and load matching continuous 3D DLC CSV
        dlc_csv_path = find_3d_dlc_csv(dlc_3d_dir, session, vid_idx, br_idx)
        if dlc_csv_path is None or not dlc_csv_path.exists():
            _log(f"3D DLC file not found for BR {br_idx} (Video {vid_idx:03d}) in {dlc_3d_dir}")
            continue

        if dlc_csv_path not in dlc_df_cache:
            df_dlc, bodyparts = load_dlc_3d_file(dlc_csv_path, square_size_mm=square_size_mm)
            if df_dlc is None:
                continue
            dlc_df_cache[dlc_csv_path] = (df_dlc, bodyparts)
        else:
            df_dlc, bodyparts = dlc_df_cache[dlc_csv_path]

        # 3. Load continuous frame timeline
        frame_t_ms = load_frame_timeline_ms(
            br_idx=br_idx,
            vid_idx=vid_idx,
            session=session,
            aligned_ckpt_root=aligned_ckpt_root,
            behv_ckpt_root=behv_ckpt_root,
        )

        if frame_t_ms is None:
            _log(f"Warning: Aligned frame timestamps not found for BR {br_idx}. Estimating from 100 FPS.")
            n_frames = len(df_dlc)
            frame_t_ms = np.arange(n_frames) * (1000.0 / 100.0)

        # 4. Cut 3D reaches around event times
        try:
            rel_t, x_all, y_all, z_all, valid_mask = segment_condition_3d_reaches(
                dlc_df=df_dlc,
                frame_t_ms=frame_t_ms,
                event_ms=event_ms,
                keypoint=keypoint,
                pre_zero_ms=PRE_ZERO_MS,
                post_zero_ms=POST_ZERO_MS,
                step_ms=STEP_MS,
            )
        except Exception as e:
            _log(f"Failed cutting 3D reaches for BR {br_idx}: {e}")
            continue

        # Transform coordinates into monkey task frame if enabled:
        # [X_task (Lateral), Y_task (Forward), Z_task (Vertical Height)]
        if align_task_frame:
            try:
                R_task, p_start = compute_task_rotation_matrix(
                    dlc_df=df_dlc,
                    session_landmarks=session_landmarks,
                    square_size_mm=square_size_mm,
                )
                pts_cam = np.stack([x_all, y_all, z_all], axis=0) - p_start.reshape(3, 1, 1)
                pts_task = np.einsum('ji,jkl->ikl', R_task, pts_cam)
                x_all, y_all, z_all = pts_task[0], pts_task[1], pts_task[2]
            except Exception as e:
                _log(f"Warning: Task-frame alignment failed for BR {br_idx}: {e}")

        n_valid = int(np.sum(valid_mask))
        _log(f"BR {br_idx:03d} (Video {vid_idx:03d}): Segmented {n_valid}/{len(event_ms)} valid 3D reaches.")

        if n_valid < MIN_REACHES_PER_CONDITION:
            _log(f"Skipping BR {br_idx:03d}: fewer than {MIN_REACHES_PER_CONDITION} valid reaches.")
            continue

        # Extract metadata
        cond_str = str(peri.get("overall_title", f"BR_{br_idx}"))
        pulse_count = int(peri["stim_pulse_counts"].flat[0]) if "stim_pulse_counts" in peri and peri["stim_pulse_counts"].size else None
        stim_freq_hz = float(peri["stim_freq_hz"]) if "stim_freq_hz" in peri else csv_meta.get("stim_freq_hz")
        stim_dur_nominal_ms = float(peri["stim_dur_nominal_ms"]) if "stim_dur_nominal_ms" in peri else None
        if stim_dur_nominal_ms is None and csv_meta.get("stim_duration_ms"):
            try:
                stim_dur_nominal_ms = float(csv_meta["stim_duration_ms"])
            except Exception:
                pass

        if pulse_count is None and stim_dur_nominal_ms and stim_dur_nominal_ms > 0:
            pulse_count = int(round(stim_dur_nominal_ms / PULSE_WIDTH_MS))

        is_control = bool(
            (control_br_indices and br_idx in control_br_indices)
            or "control_reaches" in str(peri_path).replace("\\", "/").lower()
            or "control" in peri_path.stem.lower()
            or "ctrl" in peri_path.stem.lower()
            or (notes and notes.lower() in ("baseline", "control", "ctrl"))
            or "control" in cond_str.lower()
            or "baseline" in cond_str.lower()
            or (pulse_count is not None and pulse_count == 0)
            or (stim_freq_hz is not None and stim_freq_hz == 0.0)
        )

        metadata = {
            "br_idx": br_idx,
            "vid_idx": vid_idx,
            "cond_str": cond_str,
            "notes": notes,
            "pulse_count": pulse_count,
            "stim_freq_hz": stim_freq_hz,
            "stim_dur_nominal_ms": stim_dur_nominal_ms,
            "is_control": is_control,
            "target": target,
        }

        # 5. Process condition kinematics
        cond_proc = process_condition_3d(
            x_all=x_all,
            y_all=y_all,
            z_all=z_all,
            t_axis=rel_t,
            valid_mask=valid_mask,
            metadata=metadata,
            only_post_zero=only_post_zero,
            plot_window=plot_window,
        )
        cond_key = f"Cond_{br_idx}_Target{target}"
        all_condition_results[cond_key] = cond_proc

        # Prepare for trajectory overlay - retain all valid reaches from extract_peri_stim
        clean_mask = cond_proc["valid_mask"]
        if np.sum(clean_mask) >= MIN_REACHES_PER_CONDITION:
            trajectory_rows_for_overlay.append(
                {
                    "cond_num": br_idx,
                    "target": target,
                    "time": cond_proc["aligned_time"],
                    "x": cond_proc["x_smooth_all"][clean_mask],
                    "y": cond_proc["y_smooth_all"][clean_mask],
                    "z": cond_proc["z_smooth_all"][clean_mask],
                    "time_full": cond_proc["full_time"],
                    "x_full": cond_proc["x_smooth_all_full"][clean_mask],
                    "y_full": cond_proc["y_smooth_all_full"][clean_mask],
                    "z_full": cond_proc["z_smooth_all_full"][clean_mask],
                    "dist_mean": cond_proc["dist_mean"],
                    "dist_std": cond_proc["dist_std"],
                    "dist_sem": cond_proc["dist_sem"],
                    "is_control": is_control,
                    "notes": notes,
                    "pulse_count": pulse_count,
                    "stim_freq_hz": stim_freq_hz,
                    "stim_dur_nominal_ms": stim_dur_nominal_ms,
                    "cond_str": cond_str,
                }
            )

        # Plot individual condition summary & trial inspection grids
        if PLOT_EACH_CONDITION and SAVE_FIGS:
            fig_cond_path = per_cond_dir / f"condition_{br_idx:03d}_Target{target}_3D_summary.{FIG_FORMAT}"
            plot_condition_summary_3d(cond_proc, br_idx, target, save_path=fig_cond_path)

            plot_trial_inspection_grid_3d(
                cond_proc=cond_proc,
                br_idx=br_idx,
                target=target,
                metadata=metadata,
                raw_trial_indices=raw_trial_indices,
                save_dir=trial_inspect_dir,
            )

    if not trajectory_rows_for_overlay:
        _log(f"No valid trajectory rows available for Target {target}.")
        return {}

    # Combine multiple control conditions for this target into a single consolidated Control row
    ctrl_rows = [r for r in trajectory_rows_for_overlay if r.get("is_control")]
    stim_rows = [r for r in trajectory_rows_for_overlay if not r.get("is_control")]

    if len(ctrl_rows) > 1:
        min_t = min(r["x"].shape[1] for r in ctrl_rows)
        min_t_full = min(r["x_full"].shape[1] for r in ctrl_rows)

        x_comb = np.vstack([r["x"][:, :min_t] for r in ctrl_rows])
        y_comb = np.vstack([r["y"][:, :min_t] for r in ctrl_rows])
        z_comb = np.vstack([r["z"][:, :min_t] for r in ctrl_rows])

        x_full_comb = np.vstack([r["x_full"][:, :min_t_full] for r in ctrl_rows])
        y_full_comb = np.vstack([r["y_full"][:, :min_t_full] for r in ctrl_rows])
        z_full_comb = np.vstack([r["z_full"][:, :min_t_full] for r in ctrl_rows])

        total_trials = sum(r["x"].shape[0] for r in ctrl_rows)
        dist_means = [r.get("dist_mean", np.zeros(min_t))[:min_t] * r["x"].shape[0] for r in ctrl_rows]
        dist_mean_comb = np.sum(dist_means, axis=0) / max(1, total_trials)
        dist_stds = [r.get("dist_std", np.zeros(min_t))[:min_t] for r in ctrl_rows]
        dist_std_comb = np.mean(dist_stds, axis=0)
        dist_sem_comb = dist_std_comb / np.sqrt(max(1, total_trials))

        br_list = sorted(list(set(r["cond_num"] for r in ctrl_rows)))
        primary_cnum = br_list[0]

        comb_row = {
            "cond_num": primary_cnum,
            "target": target,
            "time": ctrl_rows[0]["time"][:min_t],
            "x": x_comb,
            "y": y_comb,
            "z": z_comb,
            "time_full": ctrl_rows[0]["time_full"][:min_t_full],
            "x_full": x_full_comb,
            "y_full": y_full_comb,
            "z_full": z_full_comb,
            "dist_mean": dist_mean_comb,
            "dist_std": dist_std_comb,
            "dist_sem": dist_sem_comb,
            "is_control": True,
            "notes": f"Combined Control (BR {', '.join(map(str, br_list))})",
            "pulse_count": 0,
            "stim_freq_hz": 0.0,
            "stim_dur_nominal_ms": 0.0,
            "cond_str": "Control",
            "combined_br_indices": br_list,
        }
        _log(f"Target {target}: Combined {len(ctrl_rows)} control conditions (BR {br_list}) into 1 unified Control row ({total_trials} total reaches).")
        trajectory_rows_for_overlay = [comb_row] + stim_rows

    trajectory_rows_for_overlay = sorted(trajectory_rows_for_overlay, key=lambda r: (0 if r.get("is_control") else 1, r["cond_num"]))

    # 6. Trajectory Variability Analysis
    variability_results: Dict[int, Dict[str, Any]] = {}
    for r in trajectory_rows_for_overlay:
        var_metrics = compute_variability_metrics_3d(r, TRAJECTORY_VARIABILITY_CONFIG)
        variability_results[r["cond_num"]] = var_metrics

    # 7. Generate Visualizations (No tables)
    cond_tag = "_".join(map(str, sorted(compare_conditions))) if compare_conditions else "all"

    # A. 2-Panel Variability & Normalized Distance Comparison
    fig_var = plot_trajectory_variability_comparison_3d(
        variability_results,
        target=target,
        cfg=TRAJECTORY_VARIABILITY_CONFIG,
        control_br_indices=control_br_indices,
    )
    if fig_var is not None and SAVE_FIGS:
        out_var = comp_dir / f"trajectory_variability_3D_Target{target}_conditions_{cond_tag}.{FIG_FORMAT}"
        fig_var.savefig(out_var, format=FIG_FORMAT, dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved 3D variability comparison figure: {out_var.name}")
        plt.close(fig_var)

    # B. Multi-Projection 3D Trajectory Overlay (4 Panels)
    fig_proj = plot_3d_trajectory_multiprojection(
        trajectory_rows_for_overlay,
        target=target,
        cfg=TRAJECTORY_VARIABILITY_CONFIG,
        control_br_indices=control_br_indices,
    )
    if fig_proj is not None and SAVE_FIGS:
        out_proj = comp_dir / f"trajectory_3D_multiprojection_Target{target}_conditions_{cond_tag}.{FIG_FORMAT}"
        fig_proj.savefig(out_proj, format=FIG_FORMAT, dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved 3D multi-projection figure: {out_proj.name}")
        plt.close(fig_proj)

    # C. Classic 2D X-Y Overlay
    fig_xy = plot_xy_trajectory_overlay_for_conditions(
        trajectory_rows_for_overlay,
        target=target,
        cfg=TRAJECTORY_VARIABILITY_CONFIG,
        title_extra=" (3D DLC)",
        control_br_indices=control_br_indices,
    )
    if fig_xy is not None and SAVE_FIGS:
        out_xy = comp_dir / f"trajectory_xy_overlay_3D_Target{target}_conditions_{cond_tag}.{FIG_FORMAT}"
        fig_xy.savefig(out_xy, format=FIG_FORMAT, dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved X-Y overlay figure: {out_xy.name}")
        plt.close(fig_xy)

    # D. Interactive 3D Trajectory Overlay (Plotly HTML)
    if SAVE_FIGS and PLOTLY_AVAILABLE:
        out_html = comp_dir / f"trajectory_3D_interactive_Target{target}_conditions_{cond_tag}.html"
        plot_interactive_3d_trajectories(
            trajectory_rows_for_overlay,
            target=target,
            cfg=TRAJECTORY_VARIABILITY_CONFIG,
            control_br_indices=control_br_indices,
            save_path=out_html,
        )

    _log(f"3D Kinematics Analysis for Target {target} Complete.")
    return {
        "all_condition_results": all_condition_results,
        "variability_results": variability_results,
        "trajectory_rows": trajectory_rows_for_overlay,
        "comp_dir": comp_dir,
        "target": target,
    }


# =============================================================================
# CLI & Execution Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch analysis of 3D reaching kinematics and trajectory variability using triangulated DLC_3D data."
    )
    parser.add_argument("--session", type=str, default=None, help="Session name (e.g. NRR_RW022).")
    parser.add_argument("--dlc-3d-dir", type=str, default=None, help="Directory containing triangulated 3D DLC CSV files.")
    parser.add_argument("--peri-root", type=str, default=None, help="Directory containing PeriStim NPZ checkpoints.")
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        choices=["auto", "A", "B", "both", "all"],
        help="Target button to analyze ('auto' discovers and analyzes all available targets A and B).",
    )
    parser.add_argument("--keypoint", type=str, default=DEFAULT_KEYPOINT, help="DLC keypoint (e.g. middle, index, wrist).")
    parser.add_argument("--compare-conditions", nargs="+", type=int, default=None, help="Condition / BR indices to compare.")
    parser.add_argument("--control-br", nargs="+", type=int, default=None, help="Control condition BR indices.")
    parser.add_argument("--save-dir", type=str, default=None, help="Output figure directory.")
    parser.add_argument(
        "--only-post-zero",
        dest="only_post_zero",
        action="store_true",
        default=True,
        help="Only show post-0 in time (t >= 0 ms) citing unreliable pre-0 rest-bar tracking (default: True).",
    )
    parser.add_argument(
        "--window",
        "--traj-window",
        dest="window",
        nargs=2,
        type=float,
        default=None,
        metavar=("START_MS", "END_MS"),
        help="Global time window (start_ms end_ms) for all plots (default: from PLOT_WINDOW_MS).",
    )
    parser.add_argument(
        "--square-size-mm",
        type=float,
        default=SQUARE_SIZE_MM,
        help=f"Checkerboard square / metric scaling factor in mm/unit (default: {SQUARE_SIZE_MM}).",
    )
    parser.add_argument(
        "--no-task-align",
        dest="align_task_frame",
        action="store_false",
        default=True,
        help="Disable rotation into monkey task frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global PLOT_WINDOW_MS, TRAJ_PLOT_START_MS, TRAJ_PLOT_END_MS, TRAJECTORY_WINDOW_MS, ONLY_POST_ZERO
    if args.window is not None:
        PLOT_WINDOW_MS = (float(args.window[0]), float(args.window[1]))
        TRAJ_PLOT_START_MS, TRAJ_PLOT_END_MS = PLOT_WINDOW_MS
        TRAJECTORY_WINDOW_MS = PLOT_WINDOW_MS
        ONLY_POST_ZERO = bool(PLOT_WINDOW_MS[0] >= 0.0)

    session = args.session or (PARAMS.session if PARAMS and hasattr(PARAMS, "session") else "NRR_RW022")

    # Resolve session root directory
    peri_root_arg = Path(args.peri_root) if args.peri_root else None
    sess_dir = resolve_session_dir(session=session, peri_root=peri_root_arg)

    # Resolve PeriStim root
    if peri_root_arg and peri_root_arg.exists():
        peri_root = peri_root_arg
    elif sess_dir and (sess_dir / "results" / "checkpoints" / "PeriStim").exists():
        peri_root = sess_dir / "results" / "checkpoints" / "PeriStim"
    elif PERI_ROOT is not None and Path(PERI_ROOT).exists():
        peri_root = Path(PERI_ROOT)
    else:
        peri_root = Path(rf"Y:\Current Project Databases - NHP\2025 Cerebellum prosthesis\Nike\20260311_{session}\results\checkpoints\PeriStim")

    # Resolve DLC 3D directory
    if args.dlc_3d_dir:
        dlc_3d_dir = Path(args.dlc_3d_dir)
    elif sess_dir and (sess_dir / "Video" / "DLC_3D_new").exists():
        dlc_3d_dir = sess_dir / "Video" / "DLC_3D_new"
    elif sess_dir and (sess_dir / "Video" / "DLC_3D").exists():
        dlc_3d_dir = sess_dir / "Video" / "DLC_3D"
    elif VIDEO_ROOT is not None and (VIDEO_ROOT / "DLC_3D_new").exists():
        dlc_3d_dir = VIDEO_ROOT / "DLC_3D_new"
    elif VIDEO_ROOT is not None and (VIDEO_ROOT / "DLC_3D").exists():
        dlc_3d_dir = VIDEO_ROOT / "DLC_3D"
    else:
        dlc_3d_dir = Path(rf"Y:\Current Project Databases - NHP\2025 Cerebellum prosthesis\Nike\20260311_{session}\Video\DLC_3D_new")
        if not dlc_3d_dir.exists():
            dlc_3d_dir = Path(rf"Y:\Current Project Databases - NHP\2025 Cerebellum prosthesis\Nike\{session}\Video\DLC_3D_new")

    # Resolve save_dir (Output Figures)
    if args.save_dir:
        save_dir = Path(args.save_dir)
    elif sess_dir and (sess_dir / "results").exists():
        save_dir = sess_dir / "results" / "figures" / "3d_kinematics"
    elif OUT_BASE is not None:
        save_dir = OUT_BASE / "figures" / "3d_kinematics"
    else:
        save_dir = Path.cwd() / "results" / "3d_kinematics"

    # Resolve Aligned and Behavior checkpoint roots for frame timelines
    aligned_ckpt = (sess_dir / "results" / "checkpoints" / "Aligned") if sess_dir else ALIGNED_CKPT_ROOT
    behv_ckpt = (sess_dir / "results" / "checkpoints" / "Behavior") if sess_dir else BEHV_CKPT_ROOT

    # Check for process_only from environment (VELES) or PARAMS
    env_process_only = os.environ.get("RCP_PROCESS_ONLY")
    active_process_only = None
    if env_process_only:
        try:
            parsed = json.loads(env_process_only)
            if isinstance(parsed, list) and len(parsed) > 0:
                active_process_only = [int(x) for x in parsed if str(x).lstrip("-").isdigit()]
        except Exception:
            pass

    if active_process_only is None and PARAMS and hasattr(PARAMS, "process_only") and PARAMS.process_only:
        try:
            active_process_only = [int(x) for x in PARAMS.process_only if str(x).lstrip("-").isdigit()]
        except Exception:
            pass

    # Priority: CLI argument > VELES/PARAMS process_only > None (meaning process all discovered conditions)
    compare_conds = args.compare_conditions if args.compare_conditions is not None else active_process_only
    control_brs = args.control_br

    # Determine targets to process
    if args.target.lower() in ("auto", "both", "all"):
        targets = discover_available_targets(peri_root)
    else:
        targets = [args.target.upper()]

    _log(f"Targets to analyze for session {session}: {targets}")

    results_per_target: Dict[str, Dict[str, Any]] = {}

    for tgt in targets:
        res = run_3d_kinematics_analysis(
            session=session,
            dlc_3d_dir=dlc_3d_dir,
            peri_root=peri_root,
            target=tgt,
            keypoint=args.keypoint,
            compare_conditions=compare_conds,
            control_br_indices=control_brs,
            save_dir=save_dir,
            metadata_csv=METADATA_CSV,
            aligned_ckpt_root=aligned_ckpt,
            behv_ckpt_root=behv_ckpt,
            only_post_zero=args.only_post_zero,
            square_size_mm=args.square_size_mm,
            align_task_frame=args.align_task_frame,
            plot_window=PLOT_WINDOW_MS,
        )
        if res and res.get("trajectory_rows"):
            results_per_target[tgt] = res

    # If multiple targets (e.g. A and B) are available and analyzed, generate unified multi-target comparison figures!
    if len(results_per_target) > 1:
        comp_dir = None
        for res in results_per_target.values():
            if "comp_dir" in res and res["comp_dir"] is not None:
                comp_dir = res["comp_dir"]
                break
        if comp_dir is None:
            if save_dir:
                comp_dir = Path(save_dir) / "comparisons"
            elif OUT_BASE is not None:
                comp_dir = OUT_BASE / "figures" / "3d_kinematics" / "comparisons"
            else:
                comp_dir = Path.cwd() / "results" / "3d_kinematics" / "comparisons"

        cond_tag = "_".join(map(str, sorted(compare_conds))) if compare_conds else "all"
        plot_combined_targets_comparisons(
            results_per_target=results_per_target,
            comp_dir=comp_dir,
            cond_tag=cond_tag,
            cfg=TRAJECTORY_VARIABILITY_CONFIG,
            control_br_indices=control_brs,
        )


if __name__ == "__main__":
    main()
