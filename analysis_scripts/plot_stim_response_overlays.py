"""
plot_stim_response_overlays.py

Session-wide stimulation-response overlay analysis built in the style of the
older peristim raster/PSTH script.

This script:
  - Finds all peristim__*.npz files under results/checkpoints/PeriStim.
  - Preserves folder organization by condition group and target:
        figures/stim_response_overlays/stim_reaches/Target_A/...
        figures/stim_response_overlays/control_reaches/Target_A/...
        figures/stim_response_overlays/at_rest/...
  - Produces per-channel NPRW overlay PSTH figures.
  - Produces Utah Array overlay PSTH figures arranged by mapped 8x8 grids.
  - Compares stimulation responses against:
        1. matched control
        2. matched at-rest
        3. the stimulation file's own baseline window
  - Saves per-channel/per-electrode metrics and summaries for later analysis.

Expected NPZ fields are based on extract_peri_stim.py generator output format:
  NPRW_counts, NPRW_width_ms, NPRW_rel_t, NPRW_rates_hz, NPRW_rates_zeroed
  UA_counts, UA_width_ms, UA_rel_t, UA_rates_hz, UA_rates_zeroed
  ua_ids_1based, br_idx, sess, overall_title, event_ms, n_trials

Metadata CSV columns used if present:
  BR_File, Stim_Frequency_Hz, Stim_Duration_ms, UA_port

Utah mapping CSV columns expected:
  ElectrodeID, NSP_ID, Array, Port, GridRow, GridCol

Place this file in:
  RCP_analysis/analysis_scripts/plot_stim_response_overlays.py
"""

from __future__ import annotations

import re
import math
import warnings
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.io import loadmat

try:
    from probeinterface import Probe
    from probeinterface.plotting import plot_probe
except Exception:
    Probe = None
    plot_probe = None

try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    delayed = None

import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

try:
    from RCP_analysis.python.functions.impedance_utils import get_session_impedances
except Exception:
    get_session_impedances = None


# =============================================================================
# Options to tweak
# =============================================================================

PROCESS_NPRW = True
PROCESS_UA = True

GENERATE_STANDARD_VIEW = True
GENERATE_ZOOM_VIEW = False

# Which condition types should generate figures?
GENERATE_STIM_FIGURES = True
GENERATE_CTRL_FIGURES = True
GENERATE_REST_FIGURES = True
GENERATE_CONTSTIM_FIGURES = False
GENERATE_OTHER_FIGURES = False

# Reference matching options
REST_MATCH_MODE = "pulse_count"  # Options: "freq_dur", "pulse_count"
PULSE_COUNT_ROUND_DECIMALS = 6
REQUIRE_MATCHED_REST_FOR_STIM = True

# Performance & speedup options
CACHE_REFERENCES = True  # Cache reference NPZ arrays in memory for massive speedup

N_JOBS = 1
PARALLEL_BACKEND = "loky"

DPI_OUTPUT = 100
SKIP_EXISTING = False
VERBOSE = False

# Main peristim/checkpoint roots.
PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim" if OUT_BASE is not None else Path(".")
FIG_ROOT = OUT_BASE / "figures" / "stim_response_overlays" if OUT_BASE is not None else Path(".")

METADATA_PATH = METADATA_CSV if "METADATA_CSV" in globals() else None


# NPRW mapping / geometry
GEOM_PATH = (
    Path(PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None) and str(PARAMS.geom_mat_rel).startswith("/")
    else (REPO_ROOT / PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None)
    else rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
) if (PARAMS is not None and getattr(PARAMS, "geom_mat_rel", None)) else None

# Plot views.
WIN_PLOT_MS = (-400.0, 400.0)
ZOOM_PRE_MS = 100.0
ZOOM_POST_MS = 100.0

PLOT_VIEWS = []
if GENERATE_STANDARD_VIEW:
    PLOT_VIEWS.append(("standard", WIN_PLOT_MS, ""))
if GENERATE_ZOOM_VIEW:
    PLOT_VIEWS.append(("zoom", None, "_zoom"))


# Plot colors.
CURRENT_COLOR = "tab:grey"
CTRL_COLOR = "tab:orange"
REST_COLOR = "tab:green"

STIM_REGION_COLOR = "gold"
STIM_REGION_ALPHA = 0.40

LOW_ACTIVITY_OUTLINE_COLOR = "tab:blue"
LOW_ACTIVITY_OUTLINE_WIDTH = 2.5

BAD_CH_COLOR = "0.85"

FIG_SIZE_NPRW = (54, 24)
FIG_SIZE_UA = (32, 24)

NPRW_DISPLAY_BIN_WIDTH_MS = 20.0  # Display bin width for NPRW overlay PSTHs
UA_DISPLAY_BIN_WIDTH_MS = 50.0    # Display bin width for Utah overlay PSTHs

NPRW_PEAK_WIN_MS = (0.0, 400.0)
NPRW_HIGH_ACTIVITY_THRESH_HZ = 30.0
UA_PEAK_WIN_MS = (0.0, 400.0)

NPRW_PSTH_YLIM = (0, 200)        # Fixed y-axis limits for NPRW subplots
UA_PSTH_YLIM = (0, 500)          # Fixed y-axis limits for Utah array subplots (set to 500)
PSTH_YLIM = NPRW_PSTH_YLIM       # Fallback/backward compatibility
PSTH_YLIM_ZOOM = (0, 350)
OVERLAY_LINEWIDTH = 2.5          # Thicker lines for control (orange) and rest (green)

REGION_ORDER = ["SMA", "PMd", "M1i", "M1s"]


# =============================================================================
# General helpers
# =============================================================================

def _print(msg: str):
    if VERBOSE: print(msg)

def should_generate_figures_for_condition(cond_type: str) -> bool:
    cond_type = str(cond_type).upper()

    if cond_type == "STIM":
        return GENERATE_STIM_FIGURES
    if cond_type == "CTRL":
        return GENERATE_CTRL_FIGURES
    if cond_type == "REST":
        return GENERATE_REST_FIGURES
    if cond_type == "CONTSTIM":
        return GENERATE_CONTSTIM_FIGURES
    if cond_type == "OTHER":
        return GENERATE_OTHER_FIGURES
    return False

def get_n_trials_from_prepared_counts(counts_arr) -> Optional[int]:
    """Return number of trials from channels x trials x bins counts."""
    if counts_arr is None:
        return None

    arr = np.asarray(counts_arr)
    if arr.ndim != 3:
        return None

    return int(arr.shape[1])

def format_trial_label(condition_name: str, n_trials: Optional[int]) -> str:
    """Format a condition label for titles/legends."""
    if n_trials is None:
        return f"{condition_name} (n=NA)"
    return f"{condition_name} (n={n_trials})"

def rebin_counts_and_axis(
    counts_3d: Optional[np.ndarray],
    centers_ms: Optional[np.ndarray],
    orig_width_ms: Optional[float],
    target_width_ms: float = 20.0,
    target_edges_ms: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Rebin 3D counts array, channels x trials x bins, onto fixed display bins.

    If target_edges_ms is supplied, bins are exactly those edges.
    For WIN_PLOT_MS=(-500, 500) and target_width_ms=20, centers become:
        -490, -470, ..., 470, 490

    Output counts are summed within each target bin.
    """
    if counts_3d is None or centers_ms is None:
        return None, None, target_width_ms

    arr = np.asarray(counts_3d, dtype=float)
    centers = np.asarray(centers_ms, dtype=float).reshape(-1)

    if arr.ndim != 3 or centers.size == 0:
        return counts_3d, centers_ms, orig_width_ms or target_width_ms

    w_orig = float(orig_width_ms) if orig_width_ms and np.isfinite(orig_width_ms) and orig_width_ms > 0 else target_width_ms
    w_target = float(target_width_ms)

    n_ch, n_trials, n_bins_orig = arr.shape
    m_bins = min(n_bins_orig, centers.size)
    arr = arr[:, :, :m_bins]
    centers = centers[:m_bins]

    if target_edges_ms is None:
        t_start = centers[0] - 0.5 * w_orig
        t_end = centers[-1] + 0.5 * w_orig

        # Align to target-width multiples.
        t_start = math.floor(t_start / w_target) * w_target
        t_end = math.ceil(t_end / w_target) * w_target

        target_edges = np.arange(t_start, t_end + 0.5 * w_target, w_target, dtype=float)
    else:
        target_edges = np.asarray(target_edges_ms, dtype=float).reshape(-1)

    if target_edges.size < 2:
        return arr, centers, w_orig

    n_target_bins = target_edges.size - 1
    new_counts = np.zeros((n_ch, n_trials, n_target_bins), dtype=float)
    new_centers = 0.5 * (target_edges[:-1] + target_edges[1:])

    # Assign original bin centers into fixed target bins.
    bin_indices = np.digitize(centers, target_edges, right=False) - 1

    for tb_idx in range(n_target_bins):
        mask = bin_indices == tb_idx
        if np.any(mask):
            new_counts[:, :, tb_idx] = np.nansum(arr[:, :, mask], axis=2)

    return new_counts, new_centers, w_target

def compute_pulse_count(freq_hz: Optional[float], dur_ms: Optional[float]) -> Optional[float]:
    if freq_hz is None or dur_ms is None:
        return None
    try:
        fh = float(freq_hz)
        dm = float(dur_ms)
        if not np.isfinite(fh) or not np.isfinite(dm) or fh <= 0 or dm <= 0:
            return None
        return fh * dm / 1000.0
    except Exception:
        return None

def get_rest_match_key(freq_hz: Optional[float], dur_ms: Optional[float]) -> Optional[Any]:
    if freq_hz is None or dur_ms is None:
        return None
    if REST_MATCH_MODE == "pulse_count":
        pc = compute_pulse_count(freq_hz, dur_ms)
        if pc is None:
            return None
        return ("pulse_count", round(float(pc), PULSE_COUNT_ROUND_DECIMALS))
    else:  # "freq_dur"
        return ("freq_dur", float(freq_hz), float(dur_ms))

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def safe_get_npz(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key in data.files:
        return data[key]
    return default

def scalarize(x, default=None):
    if x is None:
        return default
    try:
        arr = np.asarray(x)
        if arr.shape == ():
            return arr.item()
        if arr.size == 1:
            return arr.reshape(-1)[0].item()
    except Exception:
        pass
    return x

def sanitize_name(s: Any) -> str:
    s = str(s)
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s

def get_base_folder(npz_path: Path) -> str:
    path_str = str(npz_path).lower()
    if "control_reaches" in path_str:
        return "control_reaches"
    if "stim_reaches" in path_str:
        return "stim_reaches"
    if "continuous_stim" in path_str:
        return "continuous_stim"
    if "at_rest" in path_str:
        return "at_rest"
    return "other"

def get_cond_type(npz_path: Path) -> str:
    path_str = str(npz_path).lower()
    if "control_reaches" in path_str:
        return "CTRL"
    if "stim_reaches" in path_str:
        return "STIM"
    if "continuous_stim" in path_str:
        return "CONTSTIM"
    if "at_rest" in path_str:
        return "REST"
    return "OTHER"

def get_target_folder(npz_path: Path) -> str:
    name = npz_path.name.lower()
    path_str = str(npz_path).lower()

    if "_target_a" in name or "target_a" in path_str:
        return "Target_A"
    if "_target_b" in name or "target_b" in path_str:
        return "Target_B"
    if "_target_control" in name or "target_control" in path_str:
        return "Target_control"

    return ""

def output_dir_for_file(npz_path: Path) -> Path:
    base_folder = get_base_folder(npz_path)
    target_folder = get_target_folder(npz_path)

    out_dir = FIG_ROOT / base_folder
    if target_folder:
        out_dir = out_dir / target_folder
    ensure_dir(out_dir)
    return out_dir

def check_output_exists(out_path: Path) -> bool:
    return SKIP_EXISTING and out_path.exists()

def get_time_axis_and_width(
    data: np.lib.npyio.NpzFile,
    prefix: str,
    n_bins: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
    """
    Return (centers_ms, width_ms, edges_ms_or_none) for NPRW/UA.
    """
    rel_t = safe_get_npz(data, f"{prefix}_rel_t", None)
    width = safe_get_npz(data, f"{prefix}_width_ms", None)

    if rel_t is not None:
        centers = np.asarray(rel_t, dtype=float).reshape(-1)
        if n_bins is not None:
            n_bins = int(n_bins)
            if centers.size >= n_bins:
                centers = centers[:n_bins]

        width_ms = np.nan
        if width is not None:
            w = np.asarray(width, dtype=float).reshape(-1)
            w = w[np.isfinite(w) & (w > 0)]
            if w.size:
                width_ms = float(np.nanmedian(w))

        if (not np.isfinite(width_ms) or width_ms <= 0) and centers.size > 1:
            diffs = np.diff(centers)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                width_ms = float(np.nanmedian(diffs))

        return centers, width_ms, None

    return None, None, None

def get_channel_psth_rate_for_bar_axis(
    counts_arr: Optional[np.ndarray],
    centers_ms: Optional[np.ndarray],
    width_ms: Optional[float],
    channel_idx: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    Return x centers, trial-averaged PSTH rate in Hz, and bar/step width
    for one channel.

    counts_arr is expected after prepare_counts_array(...):
        channels x trials x bins

    This normalizes by number of trials and bin width:

        mean_counts_per_bin = nanmean(counts over trials)
        rate_hz = mean_counts_per_bin / (bin_width_ms / 1000)

    This is preferred for overlays because STIM, CTRL, and REST files can have
    different numbers of trials.
    """
    if counts_arr is None or centers_ms is None:
        return None, None, None

    arr = np.asarray(counts_arr, dtype=float)
    centers = np.asarray(centers_ms, dtype=float).reshape(-1)

    if arr.ndim != 3:
        return None, None, None

    if channel_idx < 0 or channel_idx >= arr.shape[0]:
        return None, None, None

    trial_bin_counts = arr[channel_idx, :, :]  # trials x bins

    if trial_bin_counts.ndim != 2 or trial_bin_counts.shape[0] == 0:
        return None, None, None

    mean_counts_per_bin = np.nanmean(trial_bin_counts, axis=0)

    n_bins = mean_counts_per_bin.size
    if centers.size >= n_bins:
        centers = centers[:n_bins]
    else:
        return None, None, None

    bar_width = np.nan
    if width_ms is not None and np.isfinite(width_ms) and width_ms > 0:
        bar_width = float(width_ms)
    elif centers.size > 1:
        diffs = np.diff(centers)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            bar_width = float(np.nanmedian(diffs))

    if not np.isfinite(bar_width) or bar_width <= 0:
        return None, None, None

    rate_hz = mean_counts_per_bin / (bar_width / 1000.0)

    good = np.isfinite(centers) & np.isfinite(rate_hz)
    if not np.any(good):
        return None, None, None

    return centers[good], rate_hz[good], float(bar_width)

def compute_peak_time_and_rate_from_psth(
    x_ms, y_hz, peak_win_ms: Tuple[float, float] = NPRW_PEAK_WIN_MS,
    array_type: str = "UA", 
    local_window_ms: float = 150.0,  # ±75ms around max for centroid
) -> Tuple[float, float]:
    """
    Find peak time using local centroid around maximum.
    
    1. Find the bin with maximum rate in peak_win_ms
    2. Compute weighted centroid within ±local_window_ms/2 of that max
    """
    x_ms = np.asarray(x_ms, dtype=float)
    y_hz = np.asarray(y_hz, dtype=float)

    # Window mask
    mask = (
        np.isfinite(x_ms)
        & np.isfinite(y_hz)
        & (x_ms >= peak_win_ms[0])
        & (x_ms <= peak_win_ms[1])
    )

    if not np.any(mask):
        return np.nan, np.nan

    x_win = x_ms[mask]
    y_win = y_hz[mask]

    if y_win.size == 0 or np.all(~np.isfinite(y_win)):
        return np.nan, np.nan

    # For NPRW: apply threshold before finding max
    if array_type.upper() == "NPRW":
        y_for_max = np.where(y_win >= NPRW_HIGH_ACTIVITY_THRESH_HZ, y_win, 0.0)
    else:
        y_for_max = y_win.copy()

    if np.nansum(y_for_max) <= 0:
        return np.nan, np.nan

    # Step 1: Find time of maximum
    max_idx = np.nanargmax(y_for_max)
    max_time = x_win[max_idx]

    # Step 2: Get bins within local window around max
    half_win = local_window_ms / 2.0
    local_mask = (x_win >= max_time - half_win) & (x_win <= max_time + half_win)
    x_local = x_win[local_mask]
    y_local = y_for_max[local_mask]

    if len(x_local) == 0 or np.nansum(y_local) <= 0:
        # Fallback to max bin
        return float(max_time), float(y_win[max_idx])

    # Step 3: Weighted centroid within local window
    total_rate = np.nansum(y_local)
    peak_time = float(np.nansum(x_local * y_local) / total_rate)

    # Get rate at nearest bin to centroid
    nearest_idx = np.argmin(np.abs(x_win - peak_time))
    peak_rate = float(y_win[nearest_idx])

    return peak_time, peak_rate

def compute_nprw_peak_table_for_plot(
    stim_counts, stim_centers_ms, stim_bin_width_ms,
    ctrl_counts, ctrl_centers_ms, ctrl_bin_width_ms,
    rest_counts, rest_centers_ms, rest_bin_width_ms,
    n_ch, stim_meta=None, stim_path=None, ctrl_path=None, rest_path=None,
):
    """
    Compute NPRW per-channel peak timing/rate for current/STIM, matched control,
    and matched at-rest in NPRW_PEAK_WIN_MS.

    Peak values are computed from display-binned trial-averaged firing rates in Hz.
    """
    rows = []

    stim_meta = stim_meta or {}
    stim_file = str(stim_path) if stim_path is not None else ""
    ctrl_file = str(ctrl_path) if ctrl_path is not None else ""
    rest_file = str(rest_path) if rest_path is not None else ""

    # Get trial counts from the counts arrays
    n_trials_stim = stim_counts.shape[1] if stim_counts is not None and stim_counts.ndim == 3 else 0
    n_trials_ctrl = ctrl_counts.shape[1] if ctrl_counts is not None and ctrl_counts.ndim == 3 else 0
    n_trials_rest = rest_counts.shape[1] if rest_counts is not None and rest_counts.ndim == 3 else 0

    for ch in range(n_ch):
        stim_x, stim_y, _ = get_channel_psth_rate_for_bar_axis(
            stim_counts,
            stim_centers_ms,
            stim_bin_width_ms,
            ch,
        )

        if ctrl_counts is not None and ctrl_centers_ms is not None:
            ctrl_x, ctrl_y, _ = get_channel_psth_rate_for_bar_axis(ctrl_counts,ctrl_centers_ms,ctrl_bin_width_ms,ch,)
        else:
            ctrl_x, ctrl_y = np.array([]), np.array([])

        if rest_counts is not None and rest_centers_ms is not None:
            rest_x, rest_y, _ = get_channel_psth_rate_for_bar_axis(rest_counts,rest_centers_ms,rest_bin_width_ms,ch,)
        else:
            rest_x, rest_y = np.array([]), np.array([])

        stim_peak_t_ms, stim_peak_rate_hz = compute_peak_time_and_rate_from_psth(stim_x, stim_y,NPRW_PEAK_WIN_MS,array_type="NPRW")
        ctrl_peak_t_ms, ctrl_peak_rate_hz = compute_peak_time_and_rate_from_psth(ctrl_x, ctrl_y,NPRW_PEAK_WIN_MS,array_type="NPRW")
        rest_peak_t_ms, rest_peak_rate_hz = compute_peak_time_and_rate_from_psth(rest_x, rest_y,NPRW_PEAK_WIN_MS,array_type="NPRW")

        peak_vals = np.array([stim_peak_rate_hz, ctrl_peak_rate_hz, rest_peak_rate_hz,],dtype=float,)
        finite_peak_vals = peak_vals[np.isfinite(peak_vals)]
        max_peak_rate_hz = float(np.max(finite_peak_vals)) if finite_peak_vals.size else np.nan

        is_high_activity = bool(
            np.isfinite(max_peak_rate_hz)
            and max_peak_rate_hz >= NPRW_HIGH_ACTIVITY_THRESH_HZ
        )

        row = {
            "file": stim_file,
            "channel": ch,
            "peak_window_start_ms": NPRW_PEAK_WIN_MS[0],
            "peak_window_stop_ms": NPRW_PEAK_WIN_MS[1],
            "high_activity_threshold_hz": NPRW_HIGH_ACTIVITY_THRESH_HZ,
            "is_high_activity": is_high_activity,
            "stim_peak_time_ms": stim_peak_t_ms,
            "stim_peak_rate_hz": stim_peak_rate_hz,
            "ctrl_peak_time_ms": ctrl_peak_t_ms,
            "ctrl_peak_rate_hz": ctrl_peak_rate_hz,
            "rest_peak_time_ms": rest_peak_t_ms,
            "rest_peak_rate_hz": rest_peak_rate_hz,
            "max_peak_rate_hz": max_peak_rate_hz,
            # Reference file info
            "matched_ctrl_file": ctrl_file,
            "matched_rest_file": rest_file,
            # Trial counts
            "n_trials_stim": n_trials_stim,
            "n_trials_ctrl": n_trials_ctrl,
            "n_trials_rest": n_trials_rest,
        }

        for k, v in stim_meta.items():
            if k not in row:
                row[k] = v

        rows.append(row)
    return pd.DataFrame(rows)

# =============================================================================
# Metadata/reference helpers
# =============================================================================

def load_metadata_csv() -> Optional[pd.DataFrame]:
    if METADATA_PATH is None:
        return None
    path = Path(METADATA_PATH)
    if not path.exists():
        warnings.warn(f"Metadata CSV does not exist: {path}")
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Could not read metadata CSV {path}: {exc}")
        return None

def find_metadata_row(
    data: np.lib.npyio.NpzFile,
    npz_path: Path,
    metadata_df: Optional[pd.DataFrame],
) -> Optional[pd.Series]:
    if metadata_df is None or metadata_df.empty:
        return None
    if "BR_File" not in metadata_df.columns:
        return None

    br_idx = scalarize(safe_get_npz(data, "br_idx", None), None)

    # Preferred: exact numeric match.
    if br_idx is not None:
        try:
            br_int = int(br_idx)
            br_numeric = pd.to_numeric(metadata_df["BR_File"], errors="coerce")
            candidates = metadata_df[br_numeric == br_int]
            if len(candidates) > 0:
                return candidates.iloc[0]
        except Exception:
            pass

        # Exact string match fallback.
        try:
            br_str = str(br_idx).strip()
            candidates = metadata_df[
                metadata_df["BR_File"].astype(str).str.strip() == br_str
            ]
            if len(candidates) > 0:
                return candidates.iloc[0]
        except Exception:
            pass

    # Last fallback: if BR_File entries are filenames, use exact-ish stem/name matching.
    try:
        stem = npz_path.stem.lower()
        name = npz_path.name.lower()
        candidates = metadata_df[
            metadata_df["BR_File"].astype(str).str.lower().apply(
                lambda x: x == stem or x == name or x in name
            )
        ]
        if len(candidates) > 0:
            return candidates.iloc[0]
    except Exception:
        pass

    return None

def _safe_float_or_none(x):
    try:
        if x is None:
            return None
        if pd.isna(x):
            return None
        if str(x).strip() in ("", "-", "nan", "None"):
            return None
        return float(x)
    except Exception:
        return None

def get_freq_dur(data: np.lib.npyio.NpzFile, npz_path: Path, metadata_df: Optional[pd.DataFrame] = None) -> Tuple[Optional[float], Optional[float]]:
    freq = _safe_float_or_none(safe_get_npz(data, "stim_freq_hz", None))
    dur = _safe_float_or_none(safe_get_npz(data, "stim_dur_nominal_ms", None))
    if dur is None:
        dur_meas = safe_get_npz(data, "stim_dur_measured_ms", None)
        if dur_meas is not None:
            try:
                dur = float(np.nanmedian(np.asarray(dur_meas, dtype=float)))
            except Exception:
                pass

    if freq is not None and dur is not None:
        return freq, dur

    return freq, dur

def normalize_port_value(port) -> Optional[str]:
    if port is None:
        return None

    try:
        port = scalarize(port, None)
    except Exception:
        pass

    s = str(port).strip().upper()

    if s in ("", "NAN", "NONE", "-"):
        return None

    if "A" in s:
        return "A"
    if "B" in s:
        return "B"

    return None

def get_ua_port(data: np.lib.npyio.NpzFile, npz_path: Path, metadata_df: Optional[pd.DataFrame] = None) -> Optional[str]:
    port = normalize_port_value(safe_get_npz(data, "ua_port", None))
    if port is not None:
        return port

    row = find_metadata_row(data, npz_path, metadata_df)
    if row is not None and "UA_port" in row.index:
        port = normalize_port_value(row["UA_port"])
        if port is not None:
            return port

    meta = scalarize(safe_get_npz(data, "meta", None), None)
    if isinstance(meta, dict):
        for key in ["UA_port", "ua_port", "Port", "port"]:
            if key in meta:
                port = normalize_port_value(meta[key])
                if port is not None:
                    return port

    return "A"

def extract_metadata(data: np.lib.npyio.NpzFile, npz_path: Path, metadata_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    br_idx = scalarize(safe_get_npz(data, "br_idx", None), None)

    if br_idx is None:
        m = re.search(r"_BR[_\-]?(\d+)", npz_path.name, flags=re.IGNORECASE)
        if m:
            br_idx = int(m.group(1))
        else:
            br_idx = None

    try:
        if br_idx is not None:
            br_idx = int(br_idx)
    except Exception:
        pass

    sess = scalarize(safe_get_npz(data, "sess", None), None)
    overall_title = scalarize(safe_get_npz(data, "overall_title", None), "")

    cond_type = get_cond_type(npz_path)
    base_folder = get_base_folder(npz_path)
    target_folder = get_target_folder(npz_path)

    freq, dur = get_freq_dur(data, npz_path, metadata_df)
    ua_port = get_ua_port(data, npz_path, metadata_df)

    # 1. Check precomputed pulse counts in data directly
    pulse_count = None
    pc_arr = safe_get_npz(data, "stim_pulse_counts", None)
    if pc_arr is not None and np.size(pc_arr):
        try:
            pulse_count = float(np.nanmedian(np.asarray(pc_arr, dtype=float)))
        except Exception:
            pulse_count = None

    if pulse_count is None or not np.isfinite(pulse_count) or pulse_count <= 0:
        pulse_count = compute_pulse_count(freq, dur)

    return {
        "npz_path": str(npz_path),
        "npz_name": npz_path.name,
        "sess": sess,
        "br_idx": br_idx,
        "overall_title": overall_title,
        "cond_type": cond_type,
        "base_folder": base_folder,
        "target_folder": target_folder,
        "stim_freq_hz": freq,
        "stim_dur_ms": dur,
        "pulse_count": pulse_count,
        "ua_port": ua_port,
    }


@lru_cache(maxsize=128)
def count_trials_in_file(npz_path: Path, array_key: str = "NPRW_counts") -> int:
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            # 1. Direct scalar trial count saved by extract_peri_stim
            n_trials = safe_get_npz(data, "n_trials", None)
            if n_trials is not None:
                try:
                    return int(scalarize(n_trials))
                except Exception:
                    pass

            # 2. Number of event timestamps
            event_ms = safe_get_npz(data, "event_ms", None)
            if event_ms is not None:
                return int(np.asarray(event_ms).size)

            # 3. Fallback to counts array shape inspection
            counts = safe_get_npz(data, array_key, None)
            if counts is None:
                counts = safe_get_npz(data, "UA_counts", None)
            if counts is None:
                return 0
            arr = np.asarray(counts)
            if arr.ndim == 3:
                if arr.shape[1] in (32, 64, 96, 128, 256):
                    return int(arr.shape[0])
                if arr.shape[0] in (32, 64, 96, 128, 256):
                    return int(arr.shape[1])
                return int(arr.shape[0])
            elif arr.ndim == 2:
                return int(arr.shape[0])
            return 0
    except Exception:
        return 0

def get_display_bin_edges_ms(
    win_ms: Optional[Tuple[float, float]],
    target_width_ms: float,
    fallback_centers_ms: Optional[np.ndarray] = None,
    fallback_width_ms: Optional[float] = None,
) -> np.ndarray:
    """
    Return clean display-bin edges aligned to the plotted window.

    For standard view with WIN_PLOT_MS=(-500, 500) and target_width_ms=20,
    this returns:
        [-500, -480, -460, ..., 480, 500]

    This makes plotted bars truly 20 ms wide and aligned to the visible axis.
    """
    w = float(target_width_ms)

    if win_ms is not None:
        start = float(win_ms[0])
        stop = float(win_ms[1])

        # Ensure the stop is included despite floating point.
        edges = np.arange(start, stop + 0.5 * w, w, dtype=float)

        # If floating point overshoots substantially, trim.
        edges = edges[edges <= stop + 1e-9]

        # Make sure final edge is exactly stop if close/missing.
        if edges.size == 0 or not np.isclose(edges[-1], stop):
            if edges.size == 0 or edges[-1] < stop:
                edges = np.append(edges, stop)
            else:
                edges[-1] = stop

        return edges

    # Fallback for nonstandard/None view.
    centers = None if fallback_centers_ms is None else np.asarray(fallback_centers_ms, dtype=float).reshape(-1)
    if centers is None or centers.size == 0:
        return np.array([], dtype=float)

    orig_w = float(fallback_width_ms) if fallback_width_ms and np.isfinite(fallback_width_ms) and fallback_width_ms > 0 else w
    start = centers[0] - 0.5 * orig_w
    stop = centers[-1] + 0.5 * orig_w

    # Align to multiples of target width.
    start = math.floor(start / w) * w
    stop = math.ceil(stop / w) * w

    return np.arange(start, stop + 0.5 * w, w, dtype=float)

def load_reference_data(
    all_files: List[Path],
    metadata_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Reference matching follows the older script style.

    Controls:
      select highest-trial control file for each (target folder, port) combination.

    Rest:
      select highest-trial at-rest file for each match key (freq/dur or pulse_count).
    """
    refs = {
        "control_by_target_port": {},  # Changed from control_by_target
        "rest_by_key": {},
    }

    control_files = [f for f in all_files if "control_reaches" in str(f).lower()]
    rest_files = [f for f in all_files if "at_rest" in str(f).lower()]

    # Build control references keyed by (target, port)
    for f in control_files:
        target = get_target_folder(f)
        if not target:
            continue
        
        # Get port from the control file
        try:
            with np.load(f, allow_pickle=True) as data:
                port = get_ua_port(data, f, metadata_df)
        except Exception:
            port = "A"  # Default fallback
        
        port = normalize_port_value(port) or "A"
        key = (target, port)
        
        old = refs["control_by_target_port"].get(key, None)
        if old is None:
            refs["control_by_target_port"][key] = f
        else:
            if count_trials_in_file(f) > count_trials_in_file(old):
                refs["control_by_target_port"][key] = f

    for f in rest_files:
        try:
            with np.load(f, allow_pickle=True) as data:
                freq, dur = get_freq_dur(data, f, metadata_df)
        except Exception:
            continue

        if freq is None or dur is None:
            continue

        key = get_rest_match_key(freq, dur)
        if key is None:
            continue

        old = refs["rest_by_key"].get(key, None)
        if old is None:
            refs["rest_by_key"][key] = f
        else:
            if count_trials_in_file(f) > count_trials_in_file(old):
                refs["rest_by_key"][key] = f

    return refs

def get_matched_control_file(
    npz_path: Path,
    refs: Dict[str, Any],
    data: Optional[np.lib.npyio.NpzFile] = None,
    metadata_df: Optional[pd.DataFrame] = None,
) -> Optional[Path]:
    """
    Get matched control file for a given session.
    
    For stim/control reaches: match by target AND port.
    For at-rest: return None (at-rest doesn't have a matched control in the same sense).
    """
    cond_type = get_cond_type(npz_path)
    
    # At-rest files don't have a "matched control" in the traditional sense
    if cond_type == "REST":
        return None
    
    target = get_target_folder(npz_path)
    if not target:
        return None
    
    # Get port from the current file
    port = None
    if data is not None:
        port = get_ua_port(data, npz_path, metadata_df)
    
    if port is None:
        # Try to load and get port
        try:
            with np.load(npz_path, allow_pickle=True) as d:
                port = get_ua_port(d, npz_path, metadata_df)
        except Exception:
            port = "A"
    
    port = normalize_port_value(port) or "A"
    key = (target, port)
    
    return refs.get("control_by_target_port", {}).get(key, None)

def get_matched_control_file_for_at_rest(
    data: np.lib.npyio.NpzFile,
    npz_path: Path,
    refs: Dict[str, Any],
    metadata_df: Optional[pd.DataFrame],
) -> Optional[Path]:
    """
    For at-rest conditions, match to any control file (either one).
    Returns the first available control file.
    """
    control_files = refs.get("control_by_target_port", {})
    if control_files:
        # Return any control file (prefer one with most trials)
        best = None
        best_trials = -1
        for ctrl_path in control_files.values():
            trials = count_trials_in_file(ctrl_path)
            if trials > best_trials:
                best = ctrl_path
                best_trials = trials
        return best
    return None

def get_matched_rest_file(
    data: np.lib.npyio.NpzFile,
    npz_path: Path,
    refs: Dict[str, Any],
    metadata_df: Optional[pd.DataFrame],
) -> Optional[Path]:
    freq, dur = get_freq_dur(data, npz_path, metadata_df)
    if freq is None or dur is None:
        return None
    key = get_rest_match_key(freq, dur)
    if key is None:
        return None
    return refs.get("rest_by_key", {}).get(key, None)


# =============================================================================
# Bad channel helpers
# =============================================================================

def get_session_location_for_impedances() -> Optional[Path]:
    """
    Return the session root containing the Impedances folder.

    Prefer PARAMS.session_loc if present. Otherwise infer from OUT_BASE:
        <session>/results/... -> <session>
    """
    for attr in ["session_loc", "SESSION_LOC", "session_path", "SESSION_PATH"]:
        try:
            val = getattr(PARAMS, attr)
            if val is not None:
                return Path(val)
        except Exception:
            pass

    try:
        out_base = Path(OUT_BASE)
        parts_lower = [p.lower() for p in out_base.parts]
        if "results" in parts_lower:
            idx = parts_lower.index("results")
            return Path(*out_base.parts[:idx])
        return out_base.parent
    except Exception:
        return None

def load_bad_channels() -> Dict[str, set]:
    """
    Load bad/excluded channels from session impedance files.

    Expected project helper, if available:
        rcp.get_session_impedances(session_loc)

    User-provided helper returns:
        {'utah': set(...), 'nprw': set(...)}

    This function normalizes that to:
        {'UA': set(...), 'NPRW': set(...)}
    """
    bad = {"NPRW": set(), "UA": set()}
    session_loc = get_session_location_for_impedances()

    imp = None

    # Preferred path: functions exposed via RCP_analysis as rcp.
    try:
        if hasattr(rcp, "get_session_impedances"):
            imp = rcp.get_session_impedances(session_loc)
    except Exception as exc:
        warnings.warn(f"Could not load impedances via rcp.get_session_impedances({session_loc}): {exc}")
        imp = None

    # Fallback to directly imported function, if present.
    if imp is None and get_session_impedances is not None:
        try:
            imp = get_session_impedances(session_loc)
        except TypeError:
            try:
                imp = get_session_impedances(PARAMS)
            except Exception as exc:
                warnings.warn(f"Could not load impedances via imported get_session_impedances: {exc}")
                imp = None
        except Exception as exc:
            warnings.warn(f"Could not load impedances via imported get_session_impedances({session_loc}): {exc}")
            imp = None

    if imp is None:
        return bad

    # Dict-style output, including user-provided {'utah': ..., 'nprw': ...}.
    try:
        if isinstance(imp, dict):
            for k in ["NPRW", "nprw", "rw", "intan", "Intan", "INTAN"]:
                if k in imp and imp[k] is not None:
                    bad["NPRW"].update(int(x) for x in imp[k])

            for k in ["UA", "ua", "utah", "Utah", "UTAH"]:
                if k in imp and imp[k] is not None:
                    bad["UA"].update(int(x) for x in imp[k])

            return bad
    except Exception:
        pass

    # DataFrame-style fallback.
    try:
        if isinstance(imp, pd.DataFrame):
            cols = {c.lower(): c for c in imp.columns}
            ch_col = cols.get("channel", cols.get("chan", cols.get("electrode", None)))
            array_col = cols.get("array", cols.get("probe", cols.get("device", None)))
            bad_col = cols.get("is_bad", cols.get("bad", cols.get("excluded", None)))

            if ch_col is not None and bad_col is not None:
                bad_rows = imp[imp[bad_col].astype(bool)]
                for _, row in bad_rows.iterrows():
                    arr_name = "UA"
                    if array_col is not None:
                        val = str(row[array_col]).upper()
                        if "NPRW" in val or "RW" in val or "INTAN" in val:
                            arr_name = "NPRW"
                        elif "UA" in val or "UTAH" in val:
                            arr_name = "UA"
                    try:
                        bad[arr_name].add(int(row[ch_col]))
                    except Exception:
                        pass
    except Exception:
        pass

    return bad


# =============================================================================
# Utah mapping helpers
# =============================================================================

def load_utah_mapping() -> tuple[dict, dict]:
    """Load Utah array electrode mapping. Returns (elec_info, region_grids)."""
    return load_electrode_mapping_cached(PARAMS.monkey)

@lru_cache(maxsize=4)
def get_electrode_mapping_csv(monkey: str = None) -> Path:
    """Get path to electrode mapping CSV for the specified monkey."""
    if monkey is None:
        monkey = PARAMS.monkey
    csv_path = Path(__file__).parent.parent / "config" / f"electrode_port_mapping_{monkey}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Electrode mapping CSV not found: {csv_path}")
    return csv_path

@lru_cache(maxsize=4)
def load_electrode_mapping_cached(monkey: str = None) -> tuple[dict, dict]:
    """
    Load electrode mapping from CSV file.
    
    Returns:
        tuple: (elec_info, region_grids)
            - elec_info: dict mapping electrode_id -> {nsp_id, port, region, row, col}
            - region_grids: dict mapping region_name -> 2D array of electrode IDs (8x8)
    """
    if monkey is None:
        monkey = PARAMS.monkey
    
    csv_path = get_electrode_mapping_csv(monkey)
    df = pd.read_csv(csv_path)
    
    elec_info = {}
    region_grids = {}
    
    for _, row in df.iterrows():
        elec_id = int(row['ElectrodeID'])
        nsp_id = int(row['NSP_ID']) if 'NSP_ID' in df.columns and pd.notna(row.get('NSP_ID')) else elec_id
        
        # Normalize port to 'A' or 'B'
        port_str = str(row['Port']) if 'Port' in df.columns else 'A'
        port = 'A' if 'A' in port_str.upper() else 'B'
        
        # Get region (Array column)
        region = str(row['Array']) if 'Array' in df.columns else 'Unknown'
        
        # Get grid position
        grid_row = int(row['GridRow']) if 'GridRow' in df.columns and pd.notna(row.get('GridRow')) else 0
        grid_col = int(row['GridCol']) if 'GridCol' in df.columns and pd.notna(row.get('GridCol')) else 0
        
        elec_info[elec_id] = {
            'nsp_id': nsp_id,
            'port': port,
            'region': region,
            'row': grid_row,
            'col': grid_col
        }
        
        # Build region grids, matching the old working script: 8 x 8.
        if region not in region_grids:
            region_grids[region] = np.zeros((8, 8), dtype=int)

        if 0 <= grid_row < 8 and 0 <= grid_col < 8:
            region_grids[region][grid_row, grid_col] = elec_id
        else:
            warnings.warn(
                f"Electrode {elec_id} has out-of-range grid position "
                f"({grid_row}, {grid_col}) for 8x8 grid."
            )
    
    return elec_info, region_grids

def build_elec_to_data_idx(ua_ids_1based, elec_info, recording_port='A'):
    """
    Build electrode_id -> channel index mapping.

    For Port A: local data channels correspond to NSP 1-128.
    For Port B: local data channels correspond to NSP 129-256.

    ua_ids_1based may contain:
      - physical electrode IDs
      - local NSP IDs 1-128
      - global NSP IDs 1-256
      - or may be None
    """
    elec_to_idx = {}

    recording_port = str(recording_port).strip().upper() if recording_port else "A"
    if recording_port not in ("A", "B"):
        recording_port = "A"

    port_elecs = {
        elec_id
        for elec_id, info in elec_info.items()
        if str(info.get("port", "")).upper() == recording_port
    }

    ua_ids_1based = None if ua_ids_1based is None else np.asarray(ua_ids_1based).ravel()

    # Determine whether ua_ids are physical electrode IDs.
    is_elec_ids = False
    if ua_ids_1based is not None and len(ua_ids_1based) > 0 and len(port_elecs) > 0:
        valid_elecs_count = sum(1 for x in ua_ids_1based if int(x) in port_elecs)
        is_elec_ids = valid_elecs_count > len(ua_ids_1based) * 0.5

    if is_elec_ids:
        for ch_idx, elec_id_raw in enumerate(ua_ids_1based):
            try:
                elec_id = int(elec_id_raw)
            except Exception:
                continue
            elec_to_idx[elec_id] = ch_idx
        return elec_to_idx

    # Build reverse lookup using global NSP_IDs from CSV.
    nsp_to_elec = {}
    for elec_id, info in elec_info.items():
        try:
            nsp_to_elec[int(info["nsp_id"])] = elec_id
        except Exception:
            pass

    nsp_offset = 0 if recording_port == "A" else 128

    if ua_ids_1based is not None and len(ua_ids_1based) > 0:
        for ch_idx, nsp_id_raw in enumerate(ua_ids_1based):
            try:
                nsp_id = int(nsp_id_raw)
            except Exception:
                continue

            # If local 1-128, convert to global.
            if 1 <= nsp_id <= 128:
                global_nsp_id = nsp_id + nsp_offset
            else:
                global_nsp_id = nsp_id

            elec_id = nsp_to_elec.get(global_nsp_id)
            if elec_id is not None:
                elec_to_idx[elec_id] = ch_idx
    else:
        # No ua_ids_1based: assume sequential 128-channel recording.
        for ch_idx in range(128):
            global_nsp_id = ch_idx + 1 + nsp_offset
            elec_id = nsp_to_elec.get(global_nsp_id)
            if elec_id is not None:
                elec_to_idx[elec_id] = ch_idx

    return elec_to_idx

def get_active_regions(elec_to_idx: dict, elec_info: dict) -> list[str]:
    active_regions = set()

    for elec_id in elec_to_idx.keys():
        info = elec_info.get(elec_id)
        if info is not None:
            active_regions.add(str(info.get("region", "")))

    return [r for r in REGION_ORDER if r in active_regions]

def get_region_grid(region: str, region_grids: dict) -> np.ndarray:
    """
    Get the 8x8 electrode grid for a region.
    
    Args:
        region: Region name (e.g., 'SMA', 'PMd', 'M1i', 'M1s')
        region_grids: Dict from load_electrode_mapping_cached
    
    Returns:
        np.ndarray: 8x8 array of electrode IDs (0 = no electrode)
    """
    if region in region_grids:
        return region_grids[region]
    return np.zeros((8, 8), dtype=int)


# =============================================================================
# Metric computation
# =============================================================================

def get_stim_duration_ms(
    data: np.lib.npyio.NpzFile,
    npz_path: Path,
    metadata_df: Optional[pd.DataFrame],
) -> float:
    """
    Get stimulation duration in ms.
    
    Returns duration for STIM and REST files (both have stimulation).
    Returns 0.0 for CTRL files (no stimulation).
    """
    cond_type = get_cond_type(npz_path)
    
    if cond_type == "CTRL":
        return 0.0

    # 1. Pull directly from top-level NPZ keys
    dur_nom = safe_get_npz(data, "stim_dur_nominal_ms", None)
    if dur_nom is not None:
        try:
            val = float(scalarize(dur_nom))
            if np.isfinite(val) and val > 0:
                return val
        except Exception:
            pass

    dur_meas = safe_get_npz(data, "stim_dur_measured_ms", None)
    if dur_meas is not None:
        try:
            val = float(np.nanmedian(np.asarray(dur_meas, dtype=float)))
            if np.isfinite(val) and val > 0:
                return val
        except Exception:
            pass

    # 2. Fallback to get_freq_dur and metadata
    freq, dur = get_freq_dur(data, npz_path, metadata_df)
    
    if dur is not None and np.isfinite(dur) and dur > 0:
        return float(dur)
    
    if freq is not None and np.isfinite(freq) and freq > 0:
        # Check stim_pulse_counts in data first
        pc_arr = safe_get_npz(data, "stim_pulse_counts", None)
        if pc_arr is not None and np.size(pc_arr):
            try:
                pc = float(np.nanmedian(np.asarray(pc_arr, dtype=float)))
                if pc > 0:
                    return float(pc / freq * 1000.0)
            except Exception:
                pass

        row = find_metadata_row(data, npz_path, metadata_df)
        pulse_count = None
        
        if row is not None:
            for pcol in ["Pulse_Count", "pulse_count", "n_pulses", "num_pulses", "Stim_Pulses"]:
                if pcol in row.index:
                    try:
                        pc = row[pcol]
                        if pd.notna(pc):
                            pulse_count = float(pc)
                            break
                    except Exception:
                        pass
        
        if pulse_count is None:
            meta = scalarize(safe_get_npz(data, "meta", None), None)
            if isinstance(meta, dict):
                for pkey in ["pulse_count", "n_pulses", "num_pulses", "Pulse_Count"]:
                    if pkey in meta:
                        try:
                            pulse_count = float(meta[pkey])
                            break
                        except Exception:
                            pass
        
        if pulse_count is not None and pulse_count > 0:
            return float(pulse_count / freq * 1000.0)
    
    return 0.0


# =============================================================================
# Plotting helpers
# =============================================================================

def prepare_counts_array(counts: np.ndarray) -> np.ndarray:
    """
    Return counts as channels x trials x bins.

    Existing PeriStim files from the working raster/PSTH script are usually:
        trials x channels x bins

    This function converts them to:
        channels x trials x bins

    If a 2D array is supplied, assumes:
        trials x bins
    and returns:
        1 x trials x bins
    """
    arr = np.asarray(counts)

    if arr.ndim == 2:
        # one channel/electrode: trials x bins
        return arr[None, :, :]

    if arr.ndim != 3:
        return arr

    # Heuristic:
    # PeriStim files are usually trials x channels x bins.
    # channels is commonly 128, 64, etc. trials is usually not exactly 128,
    # but can be. Use metadata shape conventions where possible.
    n0, n1, n2 = arr.shape

    # If second dim looks like channels, convert trials x channels x bins
    # to channels x trials x bins.
    if n1 in (32, 64, 96, 128, 256):
        return np.transpose(arr, (1, 0, 2))

    # If first dim looks like channels, leave as channels x trials x bins.
    if n0 in (32, 64, 96, 128, 256):
        return arr

    # Fallback: old files usually trials x channels x bins.
    return np.transpose(arr, (1, 0, 2))

def shade_windows(ax, stim_dur_ms: float = 0.0,):
    """
    Shade only the stimulation period from 0 ms to stim_dur_ms.
    """
    if stim_dur_ms is not None and np.isfinite(stim_dur_ms) and stim_dur_ms > 0:
        ax.axvspan(
            0.0,
            float(stim_dur_ms),
            ymin=0.0,
            ymax=1.0,
            color=STIM_REGION_COLOR,
            alpha=STIM_REGION_ALPHA,
            linewidth=0,
            zorder=4,
            label="_nolegend_",
        )

        ax.axvline(
            float(stim_dur_ms),
            color=STIM_REGION_COLOR,
            linestyle="--",
            linewidth=1.25,
            alpha=0.95,
            zorder=9,
            label="_nolegend_",
        )

    ax.axvline(
        0.0,
        color="k",
        linestyle="--",
        linewidth=1.25,
        alpha=0.90,
        zorder=9,
        label="_nolegend_",
    )

def get_view_window(view_kind: str, win_ms, stim_dur_ms: float):
    """
    Return x-axis plotting window for standard/zoom views.

    For zoom, mimic the old peristim script:
        (-100 ms, stim_dur_ms + 100 ms)

    If stim_dur_ms is unavailable/non-stim, use 100 ms as fallback.
    """
    if view_kind == "standard":
        return win_ms

    if view_kind == "zoom":
        sd = float(stim_dur_ms) if stim_dur_ms and stim_dur_ms > 0 else 100.0
        return (-ZOOM_PRE_MS, sd + ZOOM_POST_MS)

    return win_ms

def set_view_limits(ax, win_ms):
    if win_ms is not None:
        ax.set_xlim(win_ms[0], win_ms[1])


# =============================================================================
# File processing
# =============================================================================

@lru_cache(maxsize=32)
def _load_npz_cached_internal(path_str: str):
    return np.load(path_str, allow_pickle=True)

def load_optional_npz(path: Optional[Path]):
    if path is None:
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    try:
        if CACHE_REFERENCES:
            # Returned object must not be closed by callers.
            return _load_npz_cached_internal(str(path_obj.resolve()))
        return np.load(path_obj, allow_pickle=True)
    except Exception as exc:
        warnings.warn(f"Could not load reference npz {path}: {exc}")
        return None

def make_output_basename(stim_meta: Dict[str, Any], npz_path: Path, array_part: str, view_suffix: str) -> str:
    br_idx = stim_meta.get("br_idx", None)
    cond_type = stim_meta.get("cond_type", get_cond_type(npz_path))

    if br_idx is None:
        br_str = "CondUNK"
    else:
        try:
            br_str = f"Cond{int(br_idx):03d}"
        except Exception:
            br_str = f"Cond{sanitize_name(br_idx)}"

    processed = sanitize_name(Path(npz_path).stem.replace("peristim__", ""))

    return f"{processed}_{br_str}_{cond_type}_{array_part}{view_suffix}.png"

def process_nprw(
    stim_data,
    stim_path: Path,
    stim_meta: Dict[str, Any],
    ctrl_data,
    ctrl_path: Optional[Path],
    rest_data,
    rest_path: Optional[Path],
    bad: Dict[str, set],
    metadata_df,
) -> int:
    n_saved = 0
    if not PROCESS_NPRW:
        _print("  NPRW: skip, PROCESS_NPRW=False")
        return 0

    if "NPRW_counts" not in stim_data.files:
        _print("  NPRW: skip, no NPRW_counts")
        return 0

    if "NPRW_edges_ms" not in stim_data.files and "NPRW_rel_t" not in stim_data.files:
        _print("  NPRW: skip, no NPRW_rel_t or NPRW_edges_ms")
        return 0

    out_dir = output_dir_for_file(stim_path)
    stim_dur_ms = get_stim_duration_ms(stim_data, stim_path, metadata_df)

    for view_kind, win_ms, view_suffix in PLOT_VIEWS:
        plot_win_ms = get_view_window(view_kind, win_ms, stim_dur_ms)

        out_name = make_output_basename(
            stim_meta=stim_meta,
            npz_path=stim_path,
            array_part="nprw_overlay",
            view_suffix=view_suffix,
        )
        out_path = out_dir / out_name

        if check_output_exists(out_path):
            _print(f"  NPRW: skip existing {out_path.name}")
            continue

        _print(f"  NPRW: plotting {view_kind} to {out_path.name}")
        plot_nprw_overlay_grid(
            stim_data=stim_data,
            stim_path=stim_path,
            ctrl_data=ctrl_data,
            ctrl_path=ctrl_path,
            rest_data=rest_data,
            rest_path=rest_path,
            stim_meta=stim_meta,
            bad_channels=bad.get("NPRW", set()),
            metadata_df=metadata_df,
            view_suffix=view_suffix,
            win_ms=plot_win_ms,
            out_path=out_path,
        )
        n_saved += 1

    return n_saved

def process_ua(
    stim_data,
    stim_path: Path,
    stim_meta: Dict[str, Any],
    ctrl_data,
    ctrl_path: Optional[Path],
    rest_data,
    rest_path: Optional[Path],
    mapping: Dict[str, Any],
    bad: Dict[str, set],
    metadata_df,
) -> int:
    n_saved = 0
    if not PROCESS_UA:
        _print("  UA: skip, PROCESS_UA=False")
        return 0

    if "UA_counts" not in stim_data.files:
        _print("  UA: skip, no UA_counts")
        return 0

    if "UA_edges_ms" not in stim_data.files and "UA_rel_t" not in stim_data.files:
        _print("  UA: skip, no UA_rel_t or UA_edges_ms")
        return 0

    ua_ids = safe_get_npz(stim_data, "ua_ids_1based", None)
    if ua_ids is None:
        _print("  UA: skip, no ua_ids_1based")
        return 0

    recording_port = stim_meta.get("ua_port", None)

    elec_info, region_grids = mapping
    elec_to_idx = build_elec_to_data_idx(ua_ids, elec_info, recording_port or "A")
    active_regions = get_active_regions(elec_to_idx, elec_info)

    if not active_regions:
        _print("  UA: skip, no active mapped regions")
        warnings.warn(f"Could not plot UA grid because no active mapped regions were found for {stim_path.name}.")
        return 0

    _print(f"  UA: active regions: {active_regions}")
    out_dir = output_dir_for_file(stim_path)
    stim_dur_ms = get_stim_duration_ms(stim_data, stim_path, metadata_df)

    for region in active_regions:
        for view_kind, win_ms, view_suffix in PLOT_VIEWS:
            plot_win_ms = get_view_window(view_kind, win_ms, stim_dur_ms)

            out_name = make_output_basename(
                stim_meta=stim_meta,
                npz_path=stim_path,
                array_part=f"ua_{sanitize_name(region)}_overlay",
                view_suffix=view_suffix,
            )
            out_path = out_dir / out_name

            if check_output_exists(out_path):
                _print(f"  UA: skip existing {out_path.name}")
                continue

            _print(f"  UA: plotting region {region} {view_kind} to {out_path.name}")
            plot_ua_region_overlay_grid(
                stim_data=stim_data,
                stim_path=stim_path,
                ctrl_data=ctrl_data,
                ctrl_path=ctrl_path,
                rest_data=rest_data,
                rest_path=rest_path,
                stim_meta=stim_meta,
                mapping=mapping,
                region=region,
                recording_port=recording_port,
                bad_channels=bad.get("UA", set()),
                metadata_df=metadata_df,
                view_suffix=view_suffix,
                win_ms=plot_win_ms,
                out_path=out_path,
            )
            n_saved += 1

    return n_saved

def load_nprw_probe_and_mapping():
    """Load NPRW probe geometry as specified in params.yaml."""
    geom_file = None
    candidate_paths = []
    if GEOM_PATH is not None:
        candidate_paths.append(Path(GEOM_PATH))

    if PARAMS is not None:
        if getattr(PARAMS, "geom_mat_rel", None):
            candidate_paths.append(REPO_ROOT / PARAMS.geom_mat_rel)
            candidate_paths.append(Path(PARAMS.geom_mat_rel))
        nprw_cfg = getattr(PARAMS, "probes", {}).get("NPRW", {}) if hasattr(PARAMS, "probes") else {}
        for k in ("geom_mat_rel", "mapping_mat_rel"):
            rel = nprw_cfg.get(k)
            if rel:
                candidate_paths.append(REPO_ROOT / rel)
                candidate_paths.append(REPO_ROOT / "config" / "probes" / Path(rel).name)
                candidate_paths.append(Path(rel))

    for p in candidate_paths:
        if p is not None and p.exists() and p.is_file():
            geom_file = p.resolve()
            break

    if geom_file is None:
        return None, None, None

    try:
        mat_probe = loadmat(geom_file)
        nprw_geom = {
            "x": mat_probe["xcoords"].ravel(),
            "y": mat_probe["ycoords"].ravel(),
        }
        if "chanMap0ind" in mat_probe:
            dev_idx = mat_probe["chanMap0ind"].ravel()
        else:
            dev_idx = np.arange(nprw_geom["x"].size)

        if Probe is not None:
            nprw_probe = Probe(ndim=2)
            nprw_probe.set_contacts(
                positions=np.c_[nprw_geom["x"], nprw_geom["y"]],
                shapes="square",
                shape_params={"width": 12.0},
            )
            nprw_probe.set_device_channel_indices(dev_idx)
            locs = nprw_probe.contact_positions.astype(float)
            return nprw_probe, dev_idx, locs
        else:
            locs = np.c_[nprw_geom["x"], nprw_geom["y"]].astype(float)
            return None, dev_idx, locs
    except Exception as e:
        _print(f"[warn] Failed to load NPRW probe geometry from {geom_file}: {e}")
        return None, None, None

def get_nprw_stim_channel_indices(
    stim_data: np.lib.npyio.NpzFile,
    stim_path: Path,
    stim_meta: Dict[str, Any],
) -> set:
    """
    Find stimulated channel indices (0-based) from NPZ metadata, arrays, or aux stim stream.
    """
    stim_channels = set()

    # 1. Check direct top-level keys in stim_data
    if stim_data is not None:
        for k in ("active_channels_0based", "stim_channels_0based", "active_channels", "stim_channels"):
            if k in stim_data.files:
                arr = stim_data[k]
                if arr is not None and np.size(arr):
                    offset = 1 if "0based" not in k and (np.min(arr) >= 1) else 0
                    stim_channels.update(int(x) - offset for x in np.asarray(arr).ravel())
                if stim_channels:
                    return stim_channels

    # 2. Check nprw_meta and meta inside stim_data (precomputed by Intan analysis / extract_peri_stim)
    if not stim_channels and stim_data is not None:
        for mkey in ("nprw_meta", "meta"):
            if mkey in stim_data.files:
                mdict = scalarize(stim_data[mkey], None)
                if isinstance(mdict, dict):
                    for skey in ("stim_channels", "stim_channel", "active_channels"):
                        val = mdict.get(skey, None)
                        if val is not None and np.size(val):
                            try:
                                stim_channels.update(int(x) for x in np.asarray(val).ravel() if np.isfinite(x))
                            except Exception:
                                pass
                if stim_channels:
                    return stim_channels

    return stim_channels

def plot_nprw_probe_axis(
    ax_probe,
    probe,
    dev_idx,
    locs,
    stim_channels: set,
    low_activity_channels: set,
    probe_title: str = "NPRW Probe Layout",
):
    """
    Draw NPRW probe on ax_probe.
    - Stimulated channels are highlighted with red fill ("tab:red").
    - Low-activity channels have a purple outline ("tab:purple") with thick linewidth.
    - High-activity channels have standard subtle outline.
    """
    if dev_idx is None and locs is not None:
        dev_idx = np.arange(locs.shape[0])
    n_contacts = len(dev_idx) if dev_idx is not None else (locs.shape[0] if locs is not None else 0)
    if n_contacts == 0:
        ax_probe.axis("off")
        return

    contacts_colors = ["none"] * n_contacts
    contacts_edges = ["black"] * n_contacts
    contacts_lws = [0.8] * n_contacts

    for i in range(n_contacts):
        ch = int(dev_idx[i]) if dev_idx is not None else i
        is_stim = (i in stim_channels) or (ch in stim_channels)
        is_low = (ch in low_activity_channels) or (i in low_activity_channels)

        if is_stim:
            contacts_colors[i] = "tab:red"
        if is_low:
            contacts_edges[i] = "tab:purple"
            contacts_lws[i] = 2.4

    if probe is not None and plot_probe is not None:
        try:
            if getattr(probe, "probe_shape", None) is None:
                probe.create_auto_shape()
            poly, poly_contour = plot_probe(
                probe,
                ax=ax_probe,
                with_contact_id=False,
                contacts_colors=contacts_colors,
                probe_shape_kwargs={"facecolor": "none", "edgecolor": "black", "linewidth": 1.0},
                contact_kwargs={"edgecolors": contacts_edges, "linewidths": contacts_lws, "zorder": 3},
            )
            if poly is not None:
                poly.set_edgecolor(contacts_edges)
                poly.set_linewidth(contacts_lws)
        except Exception as e:
            _print(f"[warn] plot_probe failed, fallback to scatter: {e}")
            if locs is not None:
                ax_probe.scatter(
                    locs[:, 0], locs[:, 1],
                    s=28,
                    c=contacts_colors,
                    edgecolors=contacts_edges,
                    linewidths=contacts_lws,
                    zorder=3,
                )
    elif locs is not None:
        ax_probe.scatter(
            locs[:, 0], locs[:, 1],
            s=28,
            c=contacts_colors,
            edgecolors=contacts_edges,
            linewidths=contacts_lws,
            zorder=3,
        )

    ax_probe.set_title(probe_title, fontsize=11, fontweight="bold", pad=8)
    ax_probe.set_aspect("equal")
    ax_probe.margins(x=0.15, y=0.05)
    ax_probe.set_xticks([])
    ax_probe.set_yticks([])
    for sp in ax_probe.spines.values():
        sp.set_visible(False)

    legend_elements = [
        Patch(facecolor="tab:red", edgecolor="black", label="Stimulated (Red)"),
        Patch(facecolor="none", edgecolor="tab:purple", linewidth=2.0, label="Low Activity (Purple outline)"),
        Patch(facecolor="none", edgecolor="black", linewidth=1.0, label="High Activity (Standard)"),
    ]
    ax_probe.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

def plot_peak_timing_bar_plot(
    ax_bar,
    peak_df: Optional[pd.DataFrame],
    cond_type: str,
    current_trial_label: str,
    ctrl_trial_label: str,
    rest_trial_label: str,
):
    """
    Plot bar plot of the means and standard deviations of the vertical lines timings (peak timings)
    for comparison across Low Activity, High Activity, and All Channels.
    """
    if peak_df is None or peak_df.empty:
        ax_bar.text(0.5, 0.5, "No peak timing data available", ha="center", va="center", transform=ax_bar.transAxes)
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        return

    df = peak_df.copy()
    has_ctrl = "ctrl_peak_time_ms" in df.columns and np.isfinite(df["ctrl_peak_time_ms"]).any()
    has_rest = "rest_peak_time_ms" in df.columns and np.isfinite(df["rest_peak_time_ms"]).any()
    has_stim = "stim_peak_time_ms" in df.columns and np.isfinite(df["stim_peak_time_ms"]).any()

    low_df = df[df["is_high_activity"] == False]
    high_df = df[df["is_high_activity"] == True]

    groups = [
        ("Low Activity", low_df),
        ("High Activity", high_df),
        ("All Channels", df),
    ]

    conditions = []
    if has_stim:
        conditions.append(("Current", "stim_peak_time_ms", CURRENT_COLOR, current_trial_label))
    if has_ctrl:
        conditions.append(("Control", "ctrl_peak_time_ms", CTRL_COLOR, ctrl_trial_label))
    if has_rest:
        conditions.append(("Rest", "rest_peak_time_ms", REST_COLOR, rest_trial_label))

    if not conditions:
        ax_bar.text(0.5, 0.5, "No condition peak timings available", ha="center", va="center", transform=ax_bar.transAxes)
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        return

    n_groups = len(groups)
    n_conds = len(conditions)

    bar_width = 0.8 / max(1, n_conds)
    group_x = np.arange(n_groups)

    for cond_idx, (cond_name, col_name, color, label) in enumerate(conditions):
        means = []
        stds = []
        offsets = group_x - (n_conds - 1) * bar_width / 2.0 + cond_idx * bar_width

        for _, grp_df in groups:
            vals = grp_df[col_name].dropna().values.astype(float)
            finite_vals = vals[np.isfinite(vals)]
            if finite_vals.size > 0:
                m = float(np.mean(finite_vals))
                s = float(np.std(finite_vals, ddof=1)) if finite_vals.size > 1 else 0.0
            else:
                m = 0.0
                s = 0.0
            means.append(m)
            stds.append(s)

        bars = ax_bar.bar(
            offsets,
            means,
            yerr=stds,
            width=bar_width,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
            capsize=4,
            label=f"{label} (mean ± SD)",
        )

        for bar, m, s in zip(bars, means, stds):
            if m > 0 or s > 0:
                y_pos = m + s + 2.0
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y_pos,
                    f"{m:.0f}±{s:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

    group_labels = [f"{name}\n(n={len(grp)})" for name, grp in groups]
    ax_bar.set_xticks(group_x)
    ax_bar.set_xticklabels(group_labels, fontsize=9, fontweight="bold")
    ax_bar.set_ylabel("Peak Timing (ms)", fontsize=10, fontweight="bold")
    ax_bar.set_title("Vertical Line Timings (Peak Timings Comparison)", fontsize=11, fontweight="bold", pad=8)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
    ax_bar.legend(loc="upper right", fontsize=8, framealpha=0.9)

    all_times = []
    for _, col_name, _, _ in conditions:
        all_times.extend(df[col_name].dropna().values.tolist())
    finite_all = [t for t in all_times if np.isfinite(t)]
    if finite_all:
        ymin = max(0, min(0, min(finite_all) - 20))
        ymax = max(finite_all) + 60
        ax_bar.set_ylim(ymin, ymax)

def plot_nprw_overlay_grid(
    stim_data: np.lib.npyio.NpzFile,
    stim_path: Path,
    ctrl_data: Optional[np.lib.npyio.NpzFile],
    ctrl_path: Optional[Path],
    rest_data: Optional[np.lib.npyio.NpzFile],
    rest_path: Optional[Path],
    stim_meta: Dict[str, Any],
    bad_channels: set,
    metadata_df: Optional[pd.DataFrame],
    view_suffix: str,
    win_ms,
    out_path: Path,
):
    stim_counts_raw = safe_get_npz(stim_data, "NPRW_counts", None)
    if stim_counts_raw is None:
        _print("  NPRW plot: skip, no counts")
        return

    stim_counts = prepare_counts_array(stim_counts_raw)
    if stim_counts is None or np.asarray(stim_counts).ndim != 3:
        _print("  NPRW plot: skip, bad counts shape")
        return

    stim_centers, stim_width, _ = get_time_axis_and_width(
        stim_data, "NPRW", n_bins=stim_counts.shape[-1]
    )
    if stim_centers is None:
        _print("  NPRW plot: skip, no time axis")
        return

    n_ch = stim_counts.shape[0]

    ctrl_counts = None
    ctrl_centers = None
    ctrl_width = None
    if ctrl_data is not None and "NPRW_counts" in ctrl_data.files:
        ctrl_counts = prepare_counts_array(ctrl_data["NPRW_counts"])
        ctrl_centers, ctrl_width, _ = get_time_axis_and_width(
            ctrl_data, "NPRW", n_bins=ctrl_counts.shape[-1]
        )

    rest_counts = None
    rest_centers = None
    rest_width = None
    if rest_data is not None and "NPRW_counts" in rest_data.files:
        rest_counts = prepare_counts_array(rest_data["NPRW_counts"])
        rest_centers, rest_width, _ = get_time_axis_and_width(
            rest_data, "NPRW", n_bins=rest_counts.shape[-1]
        )

    stim_dur_ms = get_stim_duration_ms(stim_data, stim_path, metadata_df)

    display_bin_width_ms = NPRW_DISPLAY_BIN_WIDTH_MS

    if display_bin_width_ms is not None and display_bin_width_ms > 0:
        display_edges = get_display_bin_edges_ms(
            win_ms=win_ms,
            target_width_ms=display_bin_width_ms,
            fallback_centers_ms=stim_centers,
            fallback_width_ms=stim_width,
        )

        stim_counts, stim_centers, stim_width = rebin_counts_and_axis(
            stim_counts,
            stim_centers,
            stim_width,
            display_bin_width_ms,
            target_edges_ms=display_edges,
        )

        if ctrl_counts is not None:
            ctrl_counts, ctrl_centers, ctrl_width = rebin_counts_and_axis(
                ctrl_counts,
                ctrl_centers,
                ctrl_width,
                display_bin_width_ms,
                target_edges_ms=display_edges,
            )

        if rest_counts is not None:
            rest_counts, rest_centers, rest_width = rebin_counts_and_axis(
                rest_counts,
                rest_centers,
                rest_width,
                display_bin_width_ms,
                target_edges_ms=display_edges,
            )

    cond_type = str(stim_meta.get("cond_type", "CURRENT")).upper()

    n_trials_current = get_n_trials_from_prepared_counts(stim_counts)
    n_trials_ctrl = get_n_trials_from_prepared_counts(ctrl_counts)
    n_trials_rest = get_n_trials_from_prepared_counts(rest_counts)

    current_trial_label = format_trial_label(cond_type, n_trials_current)
    ctrl_trial_label = format_trial_label("CTRL", n_trials_ctrl)
    rest_trial_label = format_trial_label("REST", n_trials_rest)

    # -------------------------------------------------------------------------
    # NPRW-only peak timing table and low/high activity panel separation.
    # -------------------------------------------------------------------------
    peak_df = compute_nprw_peak_table_for_plot(
        stim_counts=stim_counts,
        stim_centers_ms=stim_centers,
        stim_bin_width_ms=stim_width,
        ctrl_counts=ctrl_counts,
        ctrl_centers_ms=ctrl_centers,
        ctrl_bin_width_ms=ctrl_width,
        rest_counts=rest_counts,
        rest_centers_ms=rest_centers,
        rest_bin_width_ms=rest_width,
        n_ch=n_ch,
        stim_meta=stim_meta,
        stim_path=stim_path,
        ctrl_path=ctrl_path,
        rest_path=rest_path,
    )

    # Separate low and high activity channels
    low_channels = []
    high_channels = []
    peak_by_ch = {}

    if peak_df is not None and not peak_df.empty and "channel" in peak_df.columns:
        for _, row in peak_df.iterrows():
            try:
                peak_by_ch[int(row["channel"])] = row
            except Exception:
                pass

        low_df = peak_df[peak_df["is_high_activity"] == False].copy()
        high_df = peak_df[peak_df["is_high_activity"] == True].copy()

        low_df = low_df.sort_values("channel")
        if "ctrl_peak_rate_hz" in high_df.columns:
            high_df = high_df.sort_values(
                ["ctrl_peak_rate_hz", "channel"],
                ascending=[False, True],
                na_position="last",
            )
        else:
            high_df = high_df.sort_values("channel")

        low_channels = [int(c) for c in low_df["channel"].values]
        high_channels = [int(c) for c in high_df["channel"].values]

        accounted = set(low_channels) | set(high_channels)
        missing = [ch for ch in range(n_ch) if ch not in accounted]
        high_channels.extend(missing)
    else:
        high_channels = list(range(n_ch))

    low_activity_ch_set = set(low_channels)

    # -------------------------------------------------------------------------
    # 3 Plot Groups Grid Layout:
    #   - Left: Low-activity channels
    #   - Middle: High-activity channels
    #   - Right: Top: NPRW Probe, Bottom: Timings Bar Plot
    # -------------------------------------------------------------------------
    R = 16
    C_low = max(1, int(math.ceil(max(1, len(low_channels)) / float(R))))
    C_high = max(1, int(math.ceil(max(1, len(high_channels)) / float(R))))
    W_right = 2.0

    fig_w = max(44.0, 5.0 * (C_low + C_high + W_right))
    fig_h = 24.0

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs_outer = gridspec.GridSpec(
        nrows=1,
        ncols=3,
        figure=fig,
        left=0.03,
        right=0.98,
        top=0.91,
        bottom=0.04,
        width_ratios=[C_low, C_high, W_right],
        wspace=0.18,
    )

    gs_low = gridspec.GridSpecFromSubplotSpec(
        nrows=R,
        ncols=C_low,
        subplot_spec=gs_outer[0],
        hspace=0.38,
        wspace=0.22,
    )

    gs_high = gridspec.GridSpecFromSubplotSpec(
        nrows=R,
        ncols=C_high,
        subplot_spec=gs_outer[1],
        hspace=0.38,
        wspace=0.22,
    )

    gs_right = gridspec.GridSpecFromSubplotSpec(
        nrows=2,
        ncols=1,
        subplot_spec=gs_outer[2],
        height_ratios=[2.2, 1.0],
        hspace=0.28,
    )

    current_label = f"current {cond_type.lower()} file"

    def _render_psth_channel(ax, ch: int, is_low: bool):
        x, y, width = get_channel_psth_rate_for_bar_axis(
            stim_counts, stim_centers, stim_width, ch
        )
        if x is not None and y is not None:
            ax.bar(
                x,
                y,
                width=width,
                align="center",
                color=CURRENT_COLOR,
                edgecolor="none",
                linewidth=0,
                alpha=0.85,
                label=current_label,
            )

        xc, yc, wc = get_channel_psth_rate_for_bar_axis(
            ctrl_counts, ctrl_centers, ctrl_width, ch
        )
        if xc is not None and yc is not None:
            ax.step(
                xc,
                yc,
                where="mid",
                color=CTRL_COLOR,
                linewidth=OVERLAY_LINEWIDTH,
                alpha=0.9,
                label="matched control",
            )

        xr, yr, wr = get_channel_psth_rate_for_bar_axis(
            rest_counts, rest_centers, rest_width, ch
        )
        if xr is not None and yr is not None:
            ax.step(
                xr,
                yr,
                where="mid",
                color=REST_COLOR,
                linewidth=OVERLAY_LINEWIDTH,
                alpha=0.9,
                label="matched rest",
            )

        # Condition-specific peak timing vertical lines
        peak_row = peak_by_ch.get(ch, None)
        if peak_row is not None:
            stim_peak_t = peak_row.get("stim_peak_time_ms", np.nan)
            ctrl_peak_t = peak_row.get("ctrl_peak_time_ms", np.nan)
            rest_peak_t = peak_row.get("rest_peak_time_ms", np.nan)

            if np.isfinite(stim_peak_t):
                ax.axvline(
                    stim_peak_t,
                    color=CURRENT_COLOR,
                    linestyle="-",
                    linewidth=1.4,
                    alpha=0.95,
                    zorder=58,
                    label="_nolegend_",
                )

            if np.isfinite(ctrl_peak_t):
                ax.axvline(
                    ctrl_peak_t,
                    color=CTRL_COLOR,
                    linestyle="-",
                    linewidth=1.4,
                    alpha=0.95,
                    zorder=8,
                    label="_nolegend_",
                )

            if np.isfinite(rest_peak_t):
                ax.axvline(
                    rest_peak_t,
                    color=REST_COLOR,
                    linestyle="-",
                    linewidth=1.4,
                    alpha=0.95,
                    zorder=8,
                    label="_nolegend_",
                )

        set_view_limits(ax, win_ms)

        if view_suffix == "_zoom" and PSTH_YLIM_ZOOM is not None:
            ax.set_ylim(*PSTH_YLIM_ZOOM)
        elif NPRW_PSTH_YLIM is not None:
            ax.set_ylim(*NPRW_PSTH_YLIM)

        shade_windows(ax, stim_dur_ms=stim_dur_ms)

        is_bad = ch in bad_channels or (ch + 1) in bad_channels
        if is_bad:
            ax.set_facecolor(BAD_CH_COLOR)

        if is_low:
            for spine in ax.spines.values():
                spine.set_edgecolor(LOW_ACTIVITY_OUTLINE_COLOR)
                spine.set_linewidth(LOW_ACTIVITY_OUTLINE_WIDTH)
            ax.set_title(f"Ch {ch} low", fontsize=8, fontweight="bold", color="purple")
        else:
            ax.set_title(f"Ch {ch} HIGH", fontsize=8, fontweight="bold")

        ax.tick_params(labelsize=6)

    # 1. Render Left Group: Low Activity Channels
    first_plotted_ax = None
    for idx in range(R * C_low):
        r = idx % R
        c = idx // R
        ax = fig.add_subplot(gs_low[r, c])
        if idx < len(low_channels):
            ch = low_channels[idx]
            _render_psth_channel(ax, ch, is_low=True)
            if first_plotted_ax is None:
                first_plotted_ax = ax
        else:
            ax.axis("off")

    # 2. Render Middle Group: High Activity Channels
    for idx in range(R * C_high):
        r = idx % R
        c = idx // R
        ax = fig.add_subplot(gs_high[r, c])
        if idx < len(high_channels):
            ch = high_channels[idx]
            _render_psth_channel(ax, ch, is_low=False)
            if first_plotted_ax is None:
                first_plotted_ax = ax
        else:
            ax.axis("off")

    # 3. Render Right Group: Top = NPRW Probe, Bottom = Timings Bar Plot
    ax_probe = fig.add_subplot(gs_right[0])
    probe_obj, dev_idx, locs = load_nprw_probe_and_mapping()
    stim_channels = get_nprw_stim_channel_indices(stim_data, stim_path, stim_meta)
    plot_nprw_probe_axis(
        ax_probe=ax_probe,
        probe=probe_obj,
        dev_idx=dev_idx,
        locs=locs,
        stim_channels=stim_channels,
        low_activity_channels=low_activity_ch_set,
        probe_title=f"NPRW Probe Layout ({len(stim_channels)} Stimmed, {len(low_channels)} Low Activity)",
    )

    ax_bar = fig.add_subplot(gs_right[1])
    plot_peak_timing_bar_plot(
        ax_bar=ax_bar,
        peak_df=peak_df,
        cond_type=cond_type,
        current_trial_label=current_trial_label,
        ctrl_trial_label=ctrl_trial_label,
        rest_trial_label=rest_trial_label,
    )

    # Main Legend
    if first_plotted_ax is not None:
        handles, labels = first_plotted_ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.98), fontsize=10, framealpha=0.9)

    title_base = str(stim_meta.get("overall_title", ""))
    br_val = stim_meta.get("br_idx", "UNK")
    if not title_base:
        title_base = f"BR_{br_val}"
    target_val = stim_meta.get("target_folder", "")

    ref_info = []
    if ctrl_counts is not None:
        ref_info.append("Control")
    if rest_counts is not None:
        ref_info.append("Rest")
    ref_str = f"Matched refs: {', '.join(ref_info)}" if ref_info else "No matched refs"

    n_high = len(high_channels)
    n_low = len(low_channels)

    fig.suptitle(
        f"{title_base} | NPRW Overlay PSTH (Trial-Averaged Firing Rate, Hz)\n"
        f"Current File ({cond_type}) | Target: {target_val or 'N/A'} | BR: {br_val} | {ref_str}\n"
        f"NPRW peak window: {NPRW_PEAK_WIN_MS[0]:.0f}-{NPRW_PEAK_WIN_MS[1]:.0f} ms | "
        f"Left: Low Activity Panels (n={n_low}) | Middle: High Activity Panels (n={n_high}, sorted by control peak rate) | "
        f"Right: NPRW Probe Layout & Peak Timings Comparison\n"
        f"[Grey Bars = Current File ({current_trial_label}) | "
        f"Orange Line = Matched {ctrl_trial_label} | "
        f"Green Line = Matched {rest_trial_label} | "
        f"Vertical Lines = condition-specific peak times]",
        fontsize=14,
    )

    fig.savefig(out_path, dpi=DPI_OUTPUT)
    plt.close(fig)


def plot_ua_region_overlay_grid(
    stim_data: np.lib.npyio.NpzFile,
    stim_path: Path,
    ctrl_data: Optional[np.lib.npyio.NpzFile],
    ctrl_path: Optional[Path],
    rest_data: Optional[np.lib.npyio.NpzFile],
    rest_path: Optional[Path],
    stim_meta: Dict[str, Any],
    mapping: Dict[str, Any],
    region: str,
    recording_port: Optional[str],
    bad_channels: set,
    metadata_df: Optional[pd.DataFrame],
    view_suffix: str,
    win_ms,
    out_path: Path,
):
    stim_counts_raw = safe_get_npz(stim_data, "UA_counts", None)
    ua_ids = safe_get_npz(stim_data, "ua_ids_1based", None)

    if stim_counts_raw is None or ua_ids is None:
        _print("  UA plot: skip, no counts or ua_ids_1based")
        return

    stim_counts = prepare_counts_array(stim_counts_raw)
    if stim_counts is None or np.asarray(stim_counts).ndim != 3:
        _print("  UA plot: skip, bad counts shape")
        return

    stim_centers, stim_width, _ = get_time_axis_and_width(
        stim_data, "UA", n_bins=stim_counts.shape[-1]
    )
    if stim_centers is None:
        _print("  UA plot: skip, no time axis")
        return

    ua_ids = np.asarray(ua_ids).astype(int).reshape(-1)

    elec_info, region_grids = mapping
    elec_to_idx = build_elec_to_data_idx(ua_ids, elec_info, recording_port or 'A')

    ctrl_counts = None
    ctrl_centers = None
    ctrl_width = None
    ctrl_elec_to_idx = {}
    if ctrl_data is not None and "UA_counts" in ctrl_data.files:
        ctrl_counts = prepare_counts_array(ctrl_data["UA_counts"])
        ctrl_centers, ctrl_width, _ = get_time_axis_and_width(
            ctrl_data, "UA", n_bins=ctrl_counts.shape[-1]
        )

        ctrl_ids = safe_get_npz(ctrl_data, "ua_ids_1based", None)
        if ctrl_ids is not None:
            ctrl_elec_to_idx = build_elec_to_data_idx(
                np.asarray(ctrl_ids).astype(int), elec_info, recording_port or 'A'
            )

    rest_counts = None
    rest_centers = None
    rest_width = None
    rest_elec_to_idx = {}
    if rest_data is not None and "UA_counts" in rest_data.files:
        rest_counts = prepare_counts_array(rest_data["UA_counts"])
        rest_centers, rest_width, _ = get_time_axis_and_width(
            rest_data, "UA", n_bins=rest_counts.shape[-1]
        )

        rest_ids = safe_get_npz(rest_data, "ua_ids_1based", None)
        if rest_ids is not None:
            rest_elec_to_idx = build_elec_to_data_idx(
                np.asarray(rest_ids).astype(int), elec_info, recording_port or 'A'
            )

    grid = get_region_grid(region, region_grids)
    if grid is None:
        warnings.warn(f"No Utah grid for region {region}")
        return

    stim_dur_ms = get_stim_duration_ms(stim_data, stim_path, metadata_df)

    display_bin_width_ms = UA_DISPLAY_BIN_WIDTH_MS

    if display_bin_width_ms is not None and display_bin_width_ms > 0:
        display_edges = get_display_bin_edges_ms(
            win_ms=win_ms,
            target_width_ms=display_bin_width_ms,
            fallback_centers_ms=stim_centers,
            fallback_width_ms=stim_width,
        )

        stim_counts, stim_centers, stim_width = rebin_counts_and_axis(
            stim_counts,
            stim_centers,
            stim_width,
            display_bin_width_ms,
            target_edges_ms=display_edges,
        )

        if ctrl_counts is not None:
            ctrl_counts, ctrl_centers, ctrl_width = rebin_counts_and_axis(
                ctrl_counts,
                ctrl_centers,
                ctrl_width,
                display_bin_width_ms,
                target_edges_ms=display_edges,
            )

        if rest_counts is not None:
            rest_counts, rest_centers, rest_width = rebin_counts_and_axis(
                rest_counts,
                rest_centers,
                rest_width,
                display_bin_width_ms,
                target_edges_ms=display_edges,
            )

    cond_type = str(stim_meta.get("cond_type", "CURRENT")).upper()

    n_trials_current = get_n_trials_from_prepared_counts(stim_counts)
    n_trials_ctrl = get_n_trials_from_prepared_counts(ctrl_counts)
    n_trials_rest = get_n_trials_from_prepared_counts(rest_counts)

    current_trial_label = format_trial_label(cond_type, n_trials_current)
    ctrl_trial_label = format_trial_label("CTRL", n_trials_ctrl)
    rest_trial_label = format_trial_label("REST", n_trials_rest)

    fig, axes = plt.subplots(8, 8, figsize=FIG_SIZE_UA, squeeze=False)

    current_label = f"current {cond_type.lower()} file"

    for rr in range(8):
        for cc in range(8):
            ax = axes[rr, cc]

            elec_val = grid[rr, cc]

            # Empty/non-electrode grid locations are stored as 0 or nan.
            if not np.isfinite(elec_val) or int(elec_val) <= 0:
                ax.axis("off")
                continue

            elec = int(elec_val)
            stim_idx = elec_to_idx.get(elec, None)

            # No mapped data channel or index out of range: fully blank, no shading.
            if stim_idx is None or stim_idx < 0 or stim_idx >= stim_counts.shape[0]:
                ax.axis("off")
                continue

            x, y, width = get_channel_psth_rate_for_bar_axis(
                stim_counts, stim_centers, stim_width, stim_idx
            )

            stim_peak_t = np.nan
            stim_peak_rate = np.nan

            if x is not None and y is not None:
                ax.bar(
                    x,
                    y,
                    width=width,
                    align="center",
                    color=CURRENT_COLOR,
                    edgecolor="none",
                    linewidth=0,
                    alpha=0.85,
                    label=current_label,
                )

                stim_peak_t, stim_peak_rate = compute_peak_time_and_rate_from_psth(
                    x,
                    y,
                    UA_PEAK_WIN_MS,
                    array_type="UA",
                )

                if np.isfinite(stim_peak_t):
                    ax.axvline(
                        stim_peak_t,
                        color=CURRENT_COLOR,
                        linestyle="-",
                        linewidth=1.4,
                        alpha=0.95,
                        zorder=8,
                        label="_nolegend_",
                    )

            ctrl_idx = ctrl_elec_to_idx.get(elec, None)
            if ctrl_idx is not None:
                xc, yc, wc = get_channel_psth_rate_for_bar_axis(
                    ctrl_counts, ctrl_centers, ctrl_width, ctrl_idx
                )
                if xc is not None and yc is not None:
                    ax.step(
                        xc,
                        yc,
                        where="mid",
                        color=CTRL_COLOR,
                        linewidth=OVERLAY_LINEWIDTH,
                        alpha=0.9,
                        label="matched control",
                    )

                    ctrl_peak_t, ctrl_peak_rate = compute_peak_time_and_rate_from_psth(
                        xc,
                        yc,
                        UA_PEAK_WIN_MS,
                        array_type="UA",
                    )

                    if np.isfinite(ctrl_peak_t):
                        ax.axvline(
                            ctrl_peak_t,
                            color=CTRL_COLOR,
                            linestyle="-",
                            linewidth=1.4,
                            alpha=0.95,
                            zorder=8,
                            label="_nolegend_",
                        )

            rest_idx = rest_elec_to_idx.get(elec, None)
            if rest_idx is not None:
                xr, yr, wr = get_channel_psth_rate_for_bar_axis(
                    rest_counts, rest_centers, rest_width, rest_idx
                )
                if xr is not None and yr is not None:
                    ax.step(
                        xr,
                        yr,
                        where="mid",
                        color=REST_COLOR,
                        linewidth=OVERLAY_LINEWIDTH,
                        alpha=0.9,
                        label="matched rest",
                    )

                    rest_peak_t, rest_peak_rate = compute_peak_time_and_rate_from_psth(
                        xr,
                        yr,
                        UA_PEAK_WIN_MS,
                        array_type="UA",
                    )

                    if np.isfinite(rest_peak_t):
                        ax.axvline(
                            rest_peak_t,
                            color=REST_COLOR,
                            linestyle="-",
                            linewidth=1.4,
                            alpha=0.95,
                            zorder=8,
                            label="_nolegend_",
                        )

            set_view_limits(ax, win_ms)

            if view_suffix == "_zoom" and PSTH_YLIM_ZOOM is not None:
                ax.set_ylim(*PSTH_YLIM_ZOOM)
            elif UA_PSTH_YLIM is not None:
                ax.set_ylim(*UA_PSTH_YLIM)

            shade_windows(ax, stim_dur_ms=stim_dur_ms)

            is_bad = elec in bad_channels or stim_idx in bad_channels or (stim_idx + 1) in bad_channels
            if is_bad:
                ax.set_facecolor(BAD_CH_COLOR)

            ax.set_title(f"E{elec}", fontsize=8)
            ax.tick_params(labelsize=6)

    handles, labels = [], []
    for ax in axes.reshape(-1):
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break

    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=10)

    title_base = str(stim_meta.get("overall_title", ""))
    br_val = stim_meta.get("br_idx", "UNK")
    if not title_base:
        title_base = f"BR_{br_val}"
    target_val = stim_meta.get("target_folder", "")

    ref_info = []
    if ctrl_counts is not None:
        ref_info.append("Control")
    if rest_counts is not None:
        ref_info.append("Rest")
    ref_str = f"Matched refs: {', '.join(ref_info)}" if ref_info else "No matched refs"

    fig.suptitle(
        f"{title_base} | Utah Array - {region} (Port {recording_port or 'A'}) | Overlay PSTH (Trial-Averaged Firing Rate, Hz)\n"
        f"Current File ({cond_type}) | Target: {target_val or 'N/A'} | BR: {br_val} | {ref_str}\n"
        f"[Grey Bars = Current File ({current_trial_label}) | "
        f"Orange Line = Matched {ctrl_trial_label} | "
        f"Green Line = Matched {rest_trial_label}]",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 0.98, 0.93])
    fig.savefig(out_path, dpi=DPI_OUTPUT)
    plt.close(fig)


def process_file(
    file_idx: int,
    total_files: int,
    npz_path: Path,
    refs: dict,
    metadata_df: Optional[pd.DataFrame],
    mapping,
    bad: Dict[str, set],
) -> dict: 
    cond_type_for_progress = get_cond_type(npz_path)

    print(
        f"[{file_idx}/{total_files}] Processing {cond_type_for_progress}: {npz_path.name}",
        flush=True,
    )

    _print(f"\n[{file_idx}/{total_files}] Processing {npz_path.name}")

    data = load_optional_npz(npz_path)
    if data is None:
        _print("  Could not load file.")
        print(
            f"[{file_idx}/{total_files}] FAILED {cond_type_for_progress}: {npz_path.name} "
            f"(could not load file)",
            flush=True,
        )
        return {
            "success": False,
            "file": str(npz_path),
            "cond_type": get_cond_type(npz_path),
            "rows": [],
            "nprw_figs": 0,
            "ua_figs": 0,
        }

    ctrl_data = None
    ctrl_file = None
    rest_data = None
    rest_file = None

    try:
        cond_type = get_cond_type(npz_path)
        generate_figures = should_generate_figures_for_condition(cond_type)

        stim_meta = extract_metadata(data, npz_path, metadata_df)
        stim_dur_ms = get_stim_duration_ms(data, npz_path, metadata_df)

        _print(f"  condition type: {cond_type}")
        _print(f"  generate_figures: {generate_figures}")
        if cond_type == "STIM":
            _print(f"  stim duration from metadata: {stim_dur_ms} ms")

        # Matched control:
        #   STIM/CTRL: match by target + port.
        #   REST: use any available control, preferring most trials.
        if cond_type == "REST":
            ctrl_file = get_matched_control_file_for_at_rest(
                data=data,
                npz_path=npz_path,
                refs=refs,
                metadata_df=metadata_df,
            )
        else:
            ctrl_file = get_matched_control_file(
                npz_path=npz_path,
                refs=refs,
                data=data,
                metadata_df=metadata_df,
            )

        if ctrl_file is not None and Path(ctrl_file) != Path(npz_path):
            ctrl_data = load_optional_npz(ctrl_file)
            _print(f"  matched control: {Path(ctrl_file).name}")
        else:
            ctrl_file = None
            _print("  matched control: None")

        # Matched rest by metadata-derived freq/dur key.
        rest_file = get_matched_rest_file(
            data=data,
            npz_path=npz_path,
            refs=refs,
            metadata_df=metadata_df,
        )

        if rest_file is not None and Path(rest_file) != Path(npz_path):
            rest_data = load_optional_npz(rest_file)
            _print(f"  matched rest: {Path(rest_file).name}")
        else:
            rest_file = None
            _print("  matched rest: None")

        nprw_figs = 0
        ua_figs = 0

        # Figures are gated by condition type.
        if generate_figures:
            if PROCESS_NPRW:
                nprw_figs = process_nprw(
                    stim_data=data,
                    stim_path=npz_path,
                    stim_meta=stim_meta,
                    ctrl_data=ctrl_data,
                    ctrl_path=ctrl_file,
                    rest_data=rest_data,
                    rest_path=rest_file,
                    bad=bad,
                    metadata_df=metadata_df,
                )

            if PROCESS_UA:
                ua_figs = process_ua(
                    stim_data=data,
                    stim_path=npz_path,
                    stim_meta=stim_meta,
                    ctrl_data=ctrl_data,
                    ctrl_path=ctrl_file,
                    rest_data=rest_data,
                    rest_path=rest_file,
                    mapping=mapping,
                    bad=bad,
                    metadata_df=metadata_df,
                )
        else:
            _print(f"  figures skipped for condition type {cond_type}")

        print(
            f"[{file_idx}/{total_files}] Done {cond_type}: {npz_path.name} "
            f"(NPRW figs={nprw_figs}, UA figs={ua_figs})",
            flush=True,
        )

        return {
            "success": True,
            "file": str(npz_path),
            "cond_type": cond_type,
            "stim_dur_ms": stim_dur_ms,
            "nprw_figs": nprw_figs,
            "ua_figs": ua_figs,
        }

    except Exception as exc:
        print(
            f"[{file_idx}/{total_files}] FAILED {cond_type_for_progress}: {npz_path.name} — {exc}",
            flush=True,
        )
        traceback.print_exc()
        return {
            "success": False,
            "file": str(npz_path),
            "cond_type": get_cond_type(npz_path),
            "nprw_figs": 0,
            "ua_figs": 0,
        }

    finally:
        # Cached NPZ objects must not be closed.
        if not CACHE_REFERENCES:
            if ctrl_data is not None:
                ctrl_data.close()
            if rest_data is not None:
                rest_data.close()
            if data is not None:
                data.close()


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(FIG_ROOT)

    print("Stimulation response overlay analysis")
    print(f"Session:       {PARAMS.session}")
    print(f"Monkey:        {PARAMS.monkey}")
    print(f"Peristim root: {PERI_ROOT}")
    print(f"Figure root:   {FIG_ROOT}")
    print(f"Metadata CSV:  {METADATA_PATH}")
    print("Comparison modes:")
    print("  self baseline: True")
    print("  control:       True")
    print("  rest:          True")

    if not PERI_ROOT.exists():
        raise FileNotFoundError(f"PERI_ROOT does not exist: {PERI_ROOT}")

    all_files = sorted(PERI_ROOT.rglob("peristim__*.npz"))
    stim_files = [f for f in all_files if "stim_reaches" in str(f).lower()]
    control_files = [f for f in all_files if "control_reaches" in str(f).lower()]
    rest_files = [f for f in all_files if "at_rest" in str(f).lower()]
    continuous_stim_files = [f for f in all_files if "continuous_stim" in str(f).lower()]

    print(f"Found {len(all_files)} peristim files.")
    print(f"Found {len(stim_files)} stim files.")
    print(f"Found {len(control_files)} control files.")
    print(f"Found {len(rest_files)} at-rest files.")
    print(f"Found {len(continuous_stim_files)} continuous stim files.")

    metadata_df = load_metadata_csv()

    refs = load_reference_data(all_files, metadata_df)
    print(f"Loaded control references: {len(refs.get('control_by_target_port', {}))}")
    print(f"Loaded rest references:    {len(refs.get('rest_by_key', {}))} (mode: {REST_MATCH_MODE})")

    bad = load_bad_channels()
    print(f"Loaded bad/excluded channels: NPRW={len(bad.get('NPRW', set()))}, UA={len(bad.get('UA', set()))}")

    mapping = load_utah_mapping()
    elec_info, region_grids = mapping
    mapping_csv = get_electrode_mapping_csv(PARAMS.monkey)
    print(f"Loaded Utah mapping from: {mapping_csv}")
    print(f"Loaded Utah regions: {list(region_grids.keys())}")

    files_to_process = all_files

    # Optional: preserve PROCESS_ONLY behavior from old script if desired.
    try:
        PROCESS_ONLY = PARAMS.preprocessing.get("process_only")
    except Exception:
        PROCESS_ONLY = None

    print(f"PROCESS_ONLY from PARAMS.preprocessing: {PROCESS_ONLY}")

    if PROCESS_ONLY:
        filtered = []
        for f in files_to_process:
            try:
                with np.load(f, allow_pickle=True) as d:
                    br = int(scalarize(safe_get_npz(d, "br_idx", -1), -1))
                if br in PROCESS_ONLY:
                    filtered.append(f)
            except Exception:
                pass
        files_to_process = filtered

    n_skipped_unmatched_rest = 0

    has_any_rest_refs = len(refs.get("rest_by_key", {})) > 0

    if REQUIRE_MATCHED_REST_FOR_STIM and has_any_rest_refs:
        filtered = []
        for f in files_to_process:
            if get_cond_type(f) == "STIM":
                try:
                    with np.load(f, allow_pickle=True) as d:
                        rest_f = get_matched_rest_file(d, f, refs, metadata_df)

                    if rest_f is not None:
                        filtered.append(f)
                    else:
                        n_skipped_unmatched_rest += 1

                except Exception:
                    filtered.append(f)
            else:
                filtered.append(f)

        files_to_process = filtered

    elif REQUIRE_MATCHED_REST_FOR_STIM and not has_any_rest_refs:
        print(
            "No at-rest reference files were loaded for this session; "
            "STIM files will be processed without matched-rest comparisons."
        )

    if len(files_to_process) == 0:
        print("No peristim files to process.")
        return

    print(f"Processing {len(files_to_process)} files.")

    if VERBOSE and N_JOBS != 1:
        print("VERBOSE=True: forcing serial processing (N_JOBS=1) for readable logs.")
        jobs = 1
    elif CACHE_REFERENCES and PARALLEL_BACKEND == "loky" and N_JOBS != 1:
        print("CACHE_REFERENCES=True with loky multiprocessing is unsafe for cached NPZ handles.")
        print("Forcing serial processing (N_JOBS=1).")
        jobs = 1
    else:
        jobs = N_JOBS

    total_files = len(files_to_process)
    if Parallel is not None and jobs is not None and jobs != 1:
        file_results = Parallel(n_jobs=jobs, backend=PARALLEL_BACKEND)(
            delayed(process_file)(idx + 1, total_files, f, refs, metadata_df, mapping, bad)
            for idx, f in enumerate(files_to_process)
        )
    else:
        file_results = [
            process_file(idx + 1, total_files, f, refs, metadata_df, mapping, bad)
            for idx, f in enumerate(files_to_process)
        ]


    print("Done.")


if __name__ == "__main__":
    main()