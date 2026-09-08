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

import os
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
# User-facing options
# =============================================================================

PROCESS_NPRW = True
PROCESS_UA = True

GENERATE_STANDARD_VIEW = True
GENERATE_ZOOM_VIEW = False

# Which condition types should generate figures?
GENERATE_STIM_FIGURES = True
GENERATE_CTRL_FIGURES = True
GENERATE_REST_FIGURES = True

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
PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim"
FIG_ROOT = OUT_BASE / "figures" / "stim_response_overlays"
RESULT_ROOT = OUT_BASE / "checkpoints" / "stim_response_overlays"

METADATA_PATH = METADATA_CSV

# Plot views.
WIN_PLOT_MS = (-400.0, 400.0)
ZOOM_PRE_MS = 100.0
ZOOM_POST_MS = 100.0

PLOT_VIEWS = []
if GENERATE_STANDARD_VIEW:
    PLOT_VIEWS.append(("standard", WIN_PLOT_MS, ""))
if GENERATE_ZOOM_VIEW:
    PLOT_VIEWS.append(("zoom", None, "_zoom"))

# Analysis windows.
BASELINE_WIN_MS = (-950.0, -650.0)

# Directly-after-stim response length.
POST_WIN_LEN_MS = 100.0

# Utah Array additionally analyzes stimulation + post-stim coupled window.
UA_INCLUDE_DURING_PLUS_POST = True

# Blanking/artifact offsets.
NPRW_POST_OFFSET_MS = 20.0
UA_POST_OFFSET_MS = 0.0

# Significance/classification thresholds.
Z_THRESH = 2.0
DELTA_HZ_THRESH = 5.0
THRESHOLD_MODE = "either"  # "z", "delta", "both", "either"


# Plot colors.
CURRENT_COLOR = "tab:grey"
CTRL_COLOR = "tab:orange"
REST_COLOR = "tab:green"
BASELINE_COLOR = "0.35"

STIM_REGION_COLOR = "gold"
STIM_REGION_ALPHA = 0.40
POST_REGION_COLOR = "tab:purple"
POST_REGION_ALPHA = 0.12
BASELINE_REGION_COLOR = "tab:gray"
BASELINE_REGION_ALPHA = 0.10

LOW_ACTIVITY_OUTLINE_COLOR = "red"
LOW_ACTIVITY_OUTLINE_WIDTH = 3.0

BAD_CH_COLOR = "0.85"

FIG_SIZE_NPRW = (48, 24)
FIG_SIZE_UA = (32, 24)

NPRW_DISPLAY_BIN_WIDTH_MS = 20.0  # Display bin width for NPRW overlay PSTHs
UA_DISPLAY_BIN_WIDTH_MS = 50.0    # Display bin width for Utah overlay PSTHs

NPRW_PEAK_WIN_MS = (0.0, 400.0)
NPRW_HIGH_ACTIVITY_THRESH_HZ = 30.0

UA_MEAN_RATE_WIN_MS = (0.0, 400.0)
UA_PEAK_WIN_MS = (0.0, 400.0)

PSTH_YLIM = (0, 200)          # Fixed y-axis limits (0 to 80 spikes) for all subplots
PSTH_YLIM_ZOOM = (0, 350)
OVERLAY_LINEWIDTH = 2.5       # Thicker lines for control (orange) and rest (green)

REGION_ORDER = ["SMA", "PMd", "M1i", "M1s"]


# =============================================================================
# General helpers
# =============================================================================

def _print(msg: str):
    if VERBOSE:
        print(msg)

def should_generate_figures_for_condition(cond_type: str) -> bool:
    cond_type = str(cond_type).upper()

    if cond_type == "STIM":
        return GENERATE_STIM_FIGURES
    if cond_type == "CTRL":
        return GENERATE_CTRL_FIGURES
    if cond_type == "REST":
        return GENERATE_REST_FIGURES

    return True


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
    if "at_rest" in path_str:
        return "at_rest"
    return "other"


def get_cond_type(npz_path: Path) -> str:
    path_str = str(npz_path).lower()
    if "control_reaches" in path_str:
        return "CTRL"
    if "stim_reaches" in path_str:
        return "STIM"
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


def infer_bin_width_ms(edges_ms: np.ndarray) -> float:
    edges_ms = np.asarray(edges_ms, dtype=float)
    if edges_ms.size < 2:
        return np.nan
    diffs = np.diff(edges_ms)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return np.nan
    return float(np.nanmedian(diffs))


def get_time_axis_and_width(
    data: np.lib.npyio.NpzFile,
    prefix: str,
    n_bins: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
    """
    Return (centers_ms, width_ms, edges_ms_or_none) for NPRW/UA.

    Current extract_peri_stim generator saves:
      {prefix}_rel_t
      {prefix}_width_ms

    Older overlay code expected:
      {prefix}_edges_ms

    This helper supports both.
    """
    edges = safe_get_npz(data, f"{prefix}_edges_ms", None)
    rel_t = safe_get_npz(data, f"{prefix}_rel_t", None)
    width = safe_get_npz(data, f"{prefix}_width_ms", None)

    if edges is not None:
        edges_arr = np.asarray(edges, dtype=float).reshape(-1)
        centers = bin_centers_from_edges(edges_arr, n_bins=n_bins)
        width_ms = infer_bin_width_ms(edges_arr)

        if not np.isfinite(width_ms) or width_ms <= 0:
            if centers is not None and centers.size > 1:
                diffs = np.diff(centers)
                diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                if diffs.size:
                    width_ms = float(np.nanmedian(diffs))

        return centers, width_ms, edges_arr

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


def window_mask_from_centers(
    centers_ms: np.ndarray,
    win_ms: Tuple[float, float],
    n_bins: Optional[int] = None,
) -> np.ndarray:
    """
    Build a time-window mask using bin centers.

    This is the correct mask for current PeriStim outputs, which save
    *_rel_t as bin centers rather than *_edges_ms.
    """
    centers = np.asarray(centers_ms, dtype=float).reshape(-1)

    if n_bins is not None:
        n_bins = int(n_bins)
        if centers.size >= n_bins:
            centers = centers[:n_bins]
        else:
            padded = np.full(n_bins, np.nan)
            padded[:centers.size] = centers
            centers = padded

    return (centers >= win_ms[0]) & (centers < win_ms[1])


def window_rate_hz_from_counts_with_axis(
    counts_trial_bin: np.ndarray,
    centers_ms: np.ndarray,
    bin_width_ms: Optional[float],
    win_ms: Tuple[float, float],
) -> Tuple[float, float, int]:
    """
    Compute mean/std trial rate in Hz for a time window using bin centers
    plus a bin width.

    counts_trial_bin shape:
      trials x bins
    or:
      bins
    """
    arr = np.asarray(counts_trial_bin, dtype=float)

    if arr.ndim == 1:
        arr = arr[None, :]

    if arr.ndim != 2:
        return np.nan, np.nan, 0

    n_bins = arr.shape[-1]
    mask = window_mask_from_centers(centers_ms, win_ms, n_bins=n_bins)

    if mask.size != n_bins:
        m = min(mask.size, n_bins)
        mask2 = np.zeros(n_bins, dtype=bool)
        mask2[:m] = mask[:m]
        mask = mask2

    if not np.any(mask):
        return np.nan, np.nan, int(arr.shape[0])

    if bin_width_ms is not None and np.isfinite(bin_width_ms) and bin_width_ms > 0:
        dur_s = float(np.sum(mask)) * float(bin_width_ms) / 1000.0
    else:
        dur_s = (float(win_ms[1]) - float(win_ms[0])) / 1000.0

    if not np.isfinite(dur_s) or dur_s <= 0:
        return np.nan, np.nan, int(arr.shape[0])

    trial_counts = np.nansum(arr[:, mask], axis=1)
    trial_rates = trial_counts / dur_s

    mean_rate = float(np.nanmean(trial_rates)) if trial_rates.size else np.nan
    std_rate = float(np.nanstd(trial_rates, ddof=1)) if trial_rates.size > 1 else np.nan

    return mean_rate, std_rate, int(trial_rates.size)


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
    x_ms,
    y_hz,
    peak_win_ms: Tuple[float, float] = NPRW_PEAK_WIN_MS,
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
    stim_counts,
    stim_centers_ms,
    stim_bin_width_ms,
    ctrl_counts,
    ctrl_centers_ms,
    ctrl_bin_width_ms,
    rest_counts,
    rest_centers_ms,
    rest_bin_width_ms,
    n_ch,
    stim_meta=None,
    stim_path=None,
    ctrl_path=None,
    rest_path=None,
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
            ctrl_x, ctrl_y, _ = get_channel_psth_rate_for_bar_axis(
                ctrl_counts,
                ctrl_centers_ms,
                ctrl_bin_width_ms,
                ch,
            )
        else:
            ctrl_x, ctrl_y = np.array([]), np.array([])

        if rest_counts is not None and rest_centers_ms is not None:
            rest_x, rest_y, _ = get_channel_psth_rate_for_bar_axis(
                rest_counts,
                rest_centers_ms,
                rest_bin_width_ms,
                ch,
            )
        else:
            rest_x, rest_y = np.array([]), np.array([])

        stim_peak_t_ms, stim_peak_rate_hz = compute_peak_time_and_rate_from_psth(
            stim_x,
            stim_y,
            NPRW_PEAK_WIN_MS,
            array_type="NPRW",
        )
        ctrl_peak_t_ms, ctrl_peak_rate_hz = compute_peak_time_and_rate_from_psth(
            ctrl_x,
            ctrl_y,
            NPRW_PEAK_WIN_MS,
            array_type="NPRW",
        )
        rest_peak_t_ms, rest_peak_rate_hz = compute_peak_time_and_rate_from_psth(
            rest_x,
            rest_y,
            NPRW_PEAK_WIN_MS,
            array_type="NPRW",
        )

        peak_vals = np.array(
            [
                stim_peak_rate_hz,
                ctrl_peak_rate_hz,
                rest_peak_rate_hz,
            ],
            dtype=float,
        )
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

def get_nprw_channel_plot_order(peak_df, n_ch):
    """
    NPRW panel order:
      1. low-activity channels first
      2. high-activity channels second
      3. high-activity channels sorted by descending matched-control peak rate
         in the 0-400 ms window
    """
    if peak_df is None or peak_df.empty:
        return list(range(n_ch))

    df = peak_df.copy()

    if "channel" not in df.columns:
        return list(range(n_ch))

    df["channel"] = df["channel"].astype(int)

    low_df = df[df["is_high_activity"] == False].copy()
    high_df = df[df["is_high_activity"] == True].copy()

    low_df = low_df.sort_values("channel")

    if "ctrl_peak_rate_hz" in high_df.columns:
        high_df = high_df.sort_values(
            ["ctrl_peak_rate_hz", "channel"],
            ascending=[False, True],
            na_position="last",
        )
    else:
        high_df = high_df.sort_values("channel")

    ordered = list(low_df["channel"].astype(int).values) + list(
        high_df["channel"].astype(int).values
    )

    missing = [ch for ch in range(n_ch) if ch not in ordered]
    ordered.extend(missing)

    return ordered


def save_nprw_peak_timing_csv(peak_df, out_path):
    """
    Save per-NPRW-figure peak timing CSV next to the figure.

    Example:
      figure.png -> figure.nprw_peak_timings.csv
    """
    if peak_df is None or peak_df.empty:
        return None

    csv_path = Path(out_path).with_suffix(".nprw_peak_timings.csv")
    peak_df.to_csv(csv_path, index=False)
    return csv_path

def compute_channel_mean_rate_from_psth_window(
    counts_arr: Optional[np.ndarray],
    centers_ms: Optional[np.ndarray],
    width_ms: Optional[float],
    channel_idx: int,
    win_ms: Tuple[float, float],
) -> float:
    """
    Compute mean firing rate in Hz for one channel/electrode over a time window.

    Uses the same prepared/rebinned counts arrays used for plotting:
        channels x trials x bins

    The returned value is the mean across trials of the window firing rate:

        trial_rate_hz = sum(counts in window) / window_duration_s
        output = mean(trial_rate_hz)

    This is appropriate for comparing STIM, matched CONTROL, and matched REST
    files even when they have different numbers of trials.
    """
    if counts_arr is None or centers_ms is None:
        return np.nan

    arr = np.asarray(counts_arr, dtype=float)
    centers = np.asarray(centers_ms, dtype=float).reshape(-1)

    if arr.ndim != 3:
        return np.nan

    if channel_idx is None or channel_idx < 0 or channel_idx >= arr.shape[0]:
        return np.nan

    trial_bin_counts = arr[channel_idx, :, :]  # trials x bins
    if trial_bin_counts.ndim != 2 or trial_bin_counts.shape[0] == 0:
        return np.nan

    n_bins = trial_bin_counts.shape[-1]
    mask = window_mask_from_centers(centers, win_ms, n_bins=n_bins)

    if mask.size != n_bins:
        m = min(mask.size, n_bins)
        mask2 = np.zeros(n_bins, dtype=bool)
        mask2[:m] = mask[:m]
        mask = mask2

    if not np.any(mask):
        return np.nan

    if width_ms is not None and np.isfinite(width_ms) and width_ms > 0:
        dur_s = float(np.sum(mask)) * float(width_ms) / 1000.0
    else:
        dur_s = (float(win_ms[1]) - float(win_ms[0])) / 1000.0

    if not np.isfinite(dur_s) or dur_s <= 0:
        return np.nan

    trial_counts = np.nansum(trial_bin_counts[:, mask], axis=1)
    trial_rates = trial_counts / dur_s

    if trial_rates.size == 0 or np.all(~np.isfinite(trial_rates)):
        return np.nan

    return float(np.nanmean(trial_rates))


def compute_ua_mean_rate_table_for_plot(
    stim_counts,
    stim_centers_ms,
    stim_bin_width_ms,
    ctrl_counts,
    ctrl_centers_ms,
    ctrl_bin_width_ms,
    rest_counts,
    rest_centers_ms,
    rest_bin_width_ms,
    elec_to_idx: dict,
    ctrl_elec_to_idx: dict,
    rest_elec_to_idx: dict,
    elec_info: dict,
    region: str,
    grid: np.ndarray,
    stim_meta=None,
    stim_path=None,
    ctrl_path=None,
    rest_path=None,
    mean_rate_win_ms: Tuple[float, float] = UA_MEAN_RATE_WIN_MS,
):
    """
    Compute per-electrode Utah mean firing rates in a fixed post-event window.

    Unlike NPRW peak timing tables, this does not classify low/high activity and
    does not reorder panels. It simply reports the mean firing rate in the
    window for:
        - current file / STIM
        - matched CONTROL
        - matched REST

    One row is emitted for each plotted electrode in the region grid.
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

    if grid is None:
        return pd.DataFrame()

    for rr in range(grid.shape[0]):
        for cc in range(grid.shape[1]):
            elec_val = grid[rr, cc]

            if not np.isfinite(elec_val) or int(elec_val) <= 0:
                continue

            elec = int(elec_val)
            stim_idx = elec_to_idx.get(elec, None)

            # Only include electrodes that are actually plotted from the current file.
            if stim_idx is None:
                continue

            if stim_counts is None:
                continue

            try:
                if stim_idx < 0 or stim_idx >= np.asarray(stim_counts).shape[0]:
                    continue
            except Exception:
                continue

            ctrl_idx = ctrl_elec_to_idx.get(elec, None)
            rest_idx = rest_elec_to_idx.get(elec, None)

            stim_mean_rate_hz = compute_channel_mean_rate_from_psth_window(
                counts_arr=stim_counts,
                centers_ms=stim_centers_ms,
                width_ms=stim_bin_width_ms,
                channel_idx=stim_idx,
                win_ms=mean_rate_win_ms,
            )

            ctrl_mean_rate_hz = compute_channel_mean_rate_from_psth_window(
                counts_arr=ctrl_counts,
                centers_ms=ctrl_centers_ms,
                width_ms=ctrl_bin_width_ms,
                channel_idx=ctrl_idx,
                win_ms=mean_rate_win_ms,
            ) if ctrl_idx is not None else np.nan

            rest_mean_rate_hz = compute_channel_mean_rate_from_psth_window(
                counts_arr=rest_counts,
                centers_ms=rest_centers_ms,
                width_ms=rest_bin_width_ms,
                channel_idx=rest_idx,
                win_ms=mean_rate_win_ms,
            ) if rest_idx is not None else np.nan

            info = elec_info.get(elec, {}) if elec_info is not None else {}

            row = {
                "file": stim_file,
                "region": region,
                "grid_row": rr,
                "grid_col": cc,
                "electrode_id": elec,
                "stim_channel_idx": stim_idx,
                "ctrl_channel_idx": ctrl_idx,
                "rest_channel_idx": rest_idx,
                "mean_rate_window_start_ms": mean_rate_win_ms[0],
                "mean_rate_window_stop_ms": mean_rate_win_ms[1],
                "stim_mean_rate_hz": stim_mean_rate_hz,
                "ctrl_mean_rate_hz": ctrl_mean_rate_hz,
                "rest_mean_rate_hz": rest_mean_rate_hz,
                "delta_stim_minus_ctrl_mean_rate_hz": (
                    stim_mean_rate_hz - ctrl_mean_rate_hz
                    if np.isfinite(stim_mean_rate_hz) and np.isfinite(ctrl_mean_rate_hz)
                    else np.nan
                ),
                "delta_stim_minus_rest_mean_rate_hz": (
                    stim_mean_rate_hz - rest_mean_rate_hz
                    if np.isfinite(stim_mean_rate_hz) and np.isfinite(rest_mean_rate_hz)
                    else np.nan
                ),
                "delta_rest_minus_ctrl_mean_rate_hz": (
                    rest_mean_rate_hz - ctrl_mean_rate_hz
                    if np.isfinite(rest_mean_rate_hz) and np.isfinite(ctrl_mean_rate_hz)
                    else np.nan
                ),
                "mapping_nsp_id": info.get("nsp_id", np.nan),
                "mapping_port": info.get("port", np.nan),
                "mapping_region": info.get("region", np.nan),
                "mapping_grid_row": info.get("row", np.nan),
                "mapping_grid_col": info.get("col", np.nan),
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

def save_ua_mean_rate_csv(rate_df, out_path):
    """
    Save per-Utah-figure mean-rate CSV next to the figure.

    Example:
      figure.png -> figure.ua_mean_rates.csv
    """
    if rate_df is None or rate_df.empty:
        return None

    csv_path = Path(out_path).with_suffix(".ua_mean_rates.csv")
    rate_df.to_csv(csv_path, index=False)
    return csv_path


def parse_br_from_filename(npz_path: Path) -> Optional[int]:
    m = re.search(r"_BR[_\-]?(\d+)", npz_path.name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

def bin_centers_from_edges(edges_ms: np.ndarray, n_bins: Optional[int] = None) -> np.ndarray:
    """
    Return bin centers from an edge-like vector.

    Normal case:
        len(edges_ms) == n_bins + 1

    Some existing peristim files appear to have edge/time vectors that do not
    exactly match the final count-bin dimension. This helper trims safely so
    plotting and window calculations do not crash.
    """
    edges_ms = np.asarray(edges_ms, dtype=float).reshape(-1)

    if edges_ms.size < 2:
        centers = edges_ms.copy()
    else:
        centers = 0.5 * (edges_ms[:-1] + edges_ms[1:])

    if n_bins is not None:
        n_bins = int(n_bins)

        if centers.size >= n_bins:
            centers = centers[:n_bins]
        elif edges_ms.size >= n_bins:
            # Fallback: treat provided vector as already time-bin centers.
            centers = edges_ms[:n_bins]
        else:
            # Last-resort fallback. This should rarely happen, but prevents
            # hard crashes from malformed files.
            out = np.full(n_bins, np.nan)
            out[:centers.size] = centers
            centers = out

    return centers



def compute_zscore(
    test_mean: float,
    ref_mean: float,
    ref_std: float,
) -> float:
    if not np.isfinite(test_mean) or not np.isfinite(ref_mean):
        return np.nan
    if not np.isfinite(ref_std) or ref_std <= 0:
        return np.nan
    return float((test_mean - ref_mean) / ref_std)


def classify_effect(delta_hz: float, z: float) -> str:
    """
    Simple increase/decrease/unchanged classification.
    """
    delta_ok = np.isfinite(delta_hz) and abs(delta_hz) >= DELTA_HZ_THRESH
    z_ok = np.isfinite(z) and abs(z) >= Z_THRESH

    if THRESHOLD_MODE == "z":
        sig = z_ok
    elif THRESHOLD_MODE == "delta":
        sig = delta_ok
    elif THRESHOLD_MODE == "both":
        sig = z_ok and delta_ok
    else:
        sig = z_ok or delta_ok

    if not sig:
        return "unchanged"

    # Prefer z direction if finite, otherwise delta direction.
    direction_value = z if np.isfinite(z) else delta_hz
    if direction_value > 0:
        return "increase"
    if direction_value < 0:
        return "decrease"
    return "unchanged"

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


def get_freq_dur(
    data: np.lib.npyio.NpzFile,
    npz_path: Path,
    metadata_df: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[float], Optional[float]]:
    row = find_metadata_row(data, npz_path, metadata_df)

    freq = None
    dur = None

    if row is not None:
        for fcol in ["Stim_Frequency_Hz", "stim_freq_hz", "frequency_hz", "recording_stim_freq"]:
            if fcol in row.index and freq is None:
                freq = _safe_float_or_none(row[fcol])

        for dcol in ["Stim_Duration_ms", "stim_dur_ms", "duration_ms", "recording_stim_dur"]:
            if dcol in row.index and dur is None:
                dur = _safe_float_or_none(row[dcol])

    # Fallback to meta dict.
    meta = scalarize(safe_get_npz(data, "meta", None), None)
    if isinstance(meta, dict):
        for fkey in [
            "Stim_Frequency_Hz",
            "stim_freq_hz",
            "frequency_hz",
            "freq_hz",
            "recording_stim_freq",
            "stim_freq",
        ]:
            if freq is None and fkey in meta:
                freq = _safe_float_or_none(meta.get(fkey))

        for dkey in [
            "Stim_Duration_ms",
            "stim_dur_ms",
            "duration_ms",
            "dur_ms",
            "recording_stim_dur",
            "stim_dur",
        ]:
            if dur is None and dkey in meta:
                dur = _safe_float_or_none(meta.get(dkey))

    # Fallback to nprw_meta dict, matching old script.
    nprw_meta = scalarize(safe_get_npz(data, "nprw_meta", None), None)
    if isinstance(nprw_meta, dict):
        if freq is None:
            sf = nprw_meta.get("stim_freq", nprw_meta.get("recording_stim_freq", None))
            if sf is not None:
                try:
                    freq = float(np.nanmedian(np.asarray(sf, dtype=float)))
                except Exception:
                    freq = _safe_float_or_none(sf)

        if dur is None:
            sd = nprw_meta.get("stim_dur", nprw_meta.get("recording_stim_dur", None))
            if sd is not None:
                try:
                    dur = float(np.nanmedian(np.asarray(sd, dtype=float)))
                except Exception:
                    dur = _safe_float_or_none(sd)

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


def get_ua_port(
    data: np.lib.npyio.NpzFile,
    npz_path: Path,
    metadata_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
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

def extract_metadata(
    data: np.lib.npyio.NpzFile,
    npz_path: Path,
    metadata_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    br_idx = scalarize(safe_get_npz(data, "br_idx", None), None)

    if br_idx is None:
        br_idx = parse_br_from_filename(npz_path)

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
    
    # Compute pulse count from frequency and duration
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
            counts = safe_get_npz(data, array_key, None)
            if counts is None:
                counts = safe_get_npz(data, "UA_counts", None)
            if counts is None:
                return 0
            arr = np.asarray(counts)
            if arr.ndim == 3:
                # Existing files are usually trials x channels x bins.
                # If second dim looks like channels, first dim is trials.
                if arr.shape[1] in (32, 64, 96, 128, 256):
                    return int(arr.shape[0])
                # If first dim looks like channels, second dim is trials.
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

    freq, dur = get_freq_dur(data, npz_path, metadata_df)
    
    if dur is not None and np.isfinite(dur) and dur > 0:
        return float(dur)
    
    if freq is not None and np.isfinite(freq) and freq > 0:
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

def get_response_windows(
    stim_dur_ms: float,
    array_type: str,
) -> Dict[str, Tuple[float, float]]:
    if array_type.upper() == "NPRW":
        post_start = float(stim_dur_ms) + NPRW_POST_OFFSET_MS
    else:
        post_start = float(stim_dur_ms) + UA_POST_OFFSET_MS

    windows = {
        "baseline": BASELINE_WIN_MS,
        "post": (post_start, post_start + POST_WIN_LEN_MS),
    }

    if array_type.upper() == "UA" and UA_INCLUDE_DURING_PLUS_POST:
        windows["during_plus_post"] = (0.0, post_start + POST_WIN_LEN_MS)

    return windows


def compute_channel_metrics(
    stim_counts_ch: np.ndarray,
    stim_centers_ms: np.ndarray,
    stim_bin_width_ms: Optional[float],
    stim_meta: Dict[str, Any],
    channel_id: int,
    array_type: str,
    comparison_name: str,
    response_win_name: str,
    response_win_ms: Tuple[float, float],
    ref_counts_ch: Optional[np.ndarray] = None,
    ref_centers_ms: Optional[np.ndarray] = None,
    ref_bin_width_ms: Optional[float] = None,
    is_bad_channel: bool = False,
    region: Optional[str] = None,
    electrode_id: Optional[int] = None,
) -> Dict[str, Any]:

    stim_resp_mean, stim_resp_std, stim_n = window_rate_hz_from_counts_with_axis(
        stim_counts_ch, stim_centers_ms, stim_bin_width_ms, response_win_ms
    )

    stim_base_mean, stim_base_std, _ = window_rate_hz_from_counts_with_axis(
        stim_counts_ch, stim_centers_ms, stim_bin_width_ms, BASELINE_WIN_MS
    )

    out = {
        **stim_meta,
        "array_type": array_type,
        "channel_id": channel_id,
        "electrode_id": electrode_id,
        "region": region,
        "comparison": comparison_name,
        "response_window": response_win_name,
        "response_win_start_ms": response_win_ms[0],
        "response_win_end_ms": response_win_ms[1],
        "baseline_win_start_ms": BASELINE_WIN_MS[0],
        "baseline_win_end_ms": BASELINE_WIN_MS[1],
        "stim_rate_hz": stim_resp_mean,
        "stim_rate_std_hz": stim_resp_std,
        "stim_baseline_rate_hz": stim_base_mean,
        "stim_baseline_std_hz": stim_base_std,
        "n_trials_stim": stim_n,
        "ref_rate_hz": np.nan,
        "ref_rate_std_hz": np.nan,
        "n_trials_ref": 0,
        "delta_hz": np.nan,
        "z": np.nan,
        "effect": "uncomputed",
        "is_significant": False,
        "is_increase": False,
        "is_decrease": False,
        "is_bad_channel": bool(is_bad_channel),
    }

    if comparison_name == "self_baseline":
        ref_mean = stim_base_mean
        ref_std = stim_base_std
        ref_n = stim_n
    else:
        if ref_counts_ch is None or ref_centers_ms is None:
            out["effect"] = "missing_reference"
            return out

        ref_mean, ref_std, ref_n = window_rate_hz_from_counts_with_axis(
            ref_counts_ch,
            ref_centers_ms,
            ref_bin_width_ms,
            response_win_ms,
        )

    delta = stim_resp_mean - ref_mean if np.isfinite(stim_resp_mean) and np.isfinite(ref_mean) else np.nan
    z = compute_zscore(stim_resp_mean, ref_mean, ref_std)
    effect = classify_effect(delta, z)

    out.update({
        "ref_rate_hz": ref_mean,
        "ref_rate_std_hz": ref_std,
        "n_trials_ref": ref_n,
        "delta_hz": delta,
        "z": z,
        "effect": effect,
        "is_significant": effect in ["increase", "decrease"],
        "is_increase": effect == "increase",
        "is_decrease": effect == "decrease",
    })

    return out

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


def shade_windows(
    ax,
    stim_dur_ms: float = 0.0,
    baseline_window_ms: Optional[Tuple[float, float]] = None,
    response_window_ms: Optional[Tuple[float, float]] = None,
):
    """
    Shade only the stimulation period from 0 ms to stim_dur_ms.

    baseline_window_ms and response_window_ms are intentionally ignored.
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
    response_windows = get_response_windows(stim_dur_ms, "NPRW")

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

    # -------------------------------------------------------------------------
    # NPRW-only peak timing table and low/high activity panel ordering.
    # Peaks are computed from the same display-binned PSTHs used for plotting.
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

    save_nprw_peak_timing_csv(peak_df, out_path)

    plot_order = get_nprw_channel_plot_order(peak_df, n_ch)

    peak_by_ch = {}
    if peak_df is not None and not peak_df.empty and "channel" in peak_df.columns:
        for _, row in peak_df.iterrows():
            try:
                peak_by_ch[int(row["channel"])] = row
            except Exception:
                pass

    n_cols = 8
    n_rows = int(math.ceil(n_ch / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=FIG_SIZE_NPRW, squeeze=False)
    axes_flat = axes.reshape(-1)

    cond_type = str(stim_meta.get("cond_type", "CURRENT")).upper()
    current_label = f"current {cond_type.lower()} file"

    for panel_idx, ch in enumerate(plot_order):
        ax = axes_flat[panel_idx]

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

        # NPRW-only peak timing lines, separately for current, control, and rest.
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
        elif PSTH_YLIM is not None:
            ax.set_ylim(*PSTH_YLIM)

        shade_windows(ax, stim_dur_ms=stim_dur_ms)

        is_bad = ch in bad_channels or (ch + 1) in bad_channels
        if is_bad:
            ax.set_facecolor(BAD_CH_COLOR)

        activity_label = ""
        is_low_activity = False

        if peak_row is not None:
            try:
                is_high_activity = bool(peak_row.get("is_high_activity", False))
                if is_high_activity:
                    activity_label = " HIGH"
                else:
                    activity_label = " low"
                    is_low_activity = True
            except Exception:
                activity_label = ""
                is_low_activity = False

        if is_low_activity:
            for spine in ax.spines.values():
                spine.set_edgecolor(LOW_ACTIVITY_OUTLINE_COLOR)
                spine.set_linewidth(LOW_ACTIVITY_OUTLINE_WIDTH)

        ax.set_title(f"Ch {ch}{activity_label}", fontsize=8)
        ax.tick_params(labelsize=6)

    for j in range(len(plot_order), len(axes_flat)):
        axes_flat[j].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
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

    n_high = 0
    n_low = 0
    if peak_df is not None and not peak_df.empty and "is_high_activity" in peak_df.columns:
        n_high = int(peak_df["is_high_activity"].astype(bool).sum())
        n_low = int(len(peak_df) - n_high)

    fig.suptitle(
        f"{title_base} | NPRW Overlay PSTH (Trial-Averaged Firing Rate, Hz)\n"
        f"Current File ({cond_type}) | Target: {target_val or 'N/A'} | BR: {br_val} | {ref_str}\n"
        f"NPRW peak window: {NPRW_PEAK_WIN_MS[0]:.0f}-{NPRW_PEAK_WIN_MS[1]:.0f} ms | "
        f"Low activity panels first: n={n_low}; high activity panels second: n={n_high}, sorted by control peak rate\n"
        f"[Grey Bars = Current File ({cond_type}) | Orange Line = Matched CTRL | Green Line = Matched REST | "
        f"Vertical Lines = condition-specific peak times]",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 0.98, 0.92])
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
    response_windows = get_response_windows(stim_dur_ms, "UA")

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

    # -------------------------------------------------------------------------
    # Utah-array per-electrode mean firing-rate table.
    #
    # Save only for non-zoom/standard Utah figures. This avoids duplicate CSVs
    # for zoom views while using the same display-rebinned arrays used for
    # plotting the standard overlay.
    # -------------------------------------------------------------------------
    if view_suffix != "_zoom":
        ua_rate_df = compute_ua_mean_rate_table_for_plot(
            stim_counts=stim_counts,
            stim_centers_ms=stim_centers,
            stim_bin_width_ms=stim_width,
            ctrl_counts=ctrl_counts,
            ctrl_centers_ms=ctrl_centers,
            ctrl_bin_width_ms=ctrl_width,
            rest_counts=rest_counts,
            rest_centers_ms=rest_centers,
            rest_bin_width_ms=rest_width,
            elec_to_idx=elec_to_idx,
            ctrl_elec_to_idx=ctrl_elec_to_idx,
            rest_elec_to_idx=rest_elec_to_idx,
            elec_info=elec_info,
            region=region,
            grid=grid,
            stim_meta=stim_meta,
            stim_path=stim_path,
            ctrl_path=ctrl_path,
            rest_path=rest_path,
            mean_rate_win_ms=UA_MEAN_RATE_WIN_MS,
        )
        save_ua_mean_rate_csv(ua_rate_df, out_path)

    fig, axes = plt.subplots(8, 8, figsize=FIG_SIZE_UA, squeeze=False)

    cond_type = str(stim_meta.get("cond_type", "CURRENT")).upper()
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
            elif PSTH_YLIM is not None:
                ax.set_ylim(*PSTH_YLIM)

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
        f"[Grey Bars = Current File ({cond_type}) | Orange Line = Matched CTRL | Green Line = Matched REST]",
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
        compute_metrics = cond_type == "STIM"
        generate_figures = should_generate_figures_for_condition(cond_type)

        stim_meta = extract_metadata(data, npz_path, metadata_df)
        stim_dur_ms = get_stim_duration_ms(data, npz_path, metadata_df)

        _print(f"  condition type: {cond_type}")
        _print(f"  generate_figures: {generate_figures}")
        _print(f"  compute_metrics: {compute_metrics}")
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

        rows: List[Dict[str, Any]] = []
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

        # Metrics are only for STIM, independent of figure gating.
        if compute_metrics:
            if PROCESS_NPRW:
                rows.extend(compute_nprw_metrics_for_file(
                    stim_data=data,
                    stim_path=npz_path,
                    stim_meta=stim_meta,
                    ctrl_data=ctrl_data,
                    rest_data=rest_data,
                    bad=bad,
                    metadata_df=metadata_df,
                ))

            if PROCESS_UA:
                rows.extend(compute_ua_metrics_for_file(
                    stim_data=data,
                    stim_path=npz_path,
                    stim_meta=stim_meta,
                    ctrl_data=ctrl_data,
                    rest_data=rest_data,
                    mapping=mapping,
                    bad=bad,
                    metadata_df=metadata_df,
                ))

        print(
            f"[{file_idx}/{total_files}] Done {cond_type}: {npz_path.name} "
            f"(NPRW figs={nprw_figs}, UA figs={ua_figs}, metric rows={len(rows)})",
            flush=True,
        )

        return {
            "success": True,
            "file": str(npz_path),
            "cond_type": cond_type,
            "stim_dur_ms": stim_dur_ms,
            "rows": rows,
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
            "rows": [],
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
def compute_nprw_metrics_for_file(
    stim_data,
    stim_path: Path,
    stim_meta: Dict[str, Any],
    ctrl_data,
    rest_data,
    bad: Dict[str, set],
    metadata_df,
) -> List[Dict[str, Any]]:

    rows = []

    stim_counts_raw = safe_get_npz(stim_data, "NPRW_counts", None)
    if stim_counts_raw is None:
        _print("  NPRW metrics: skip, no NPRW_counts")
        return rows

    stim_counts = prepare_counts_array(stim_counts_raw)
    if stim_counts is None or np.asarray(stim_counts).ndim != 3:
        _print("  NPRW metrics: skip, bad counts shape")
        return rows

    stim_centers, stim_bw, _ = get_time_axis_and_width(
        stim_data, "NPRW", n_bins=stim_counts.shape[-1]
    )
    if stim_centers is None:
        _print("  NPRW metrics: skip, no time axis")
        return rows

    n_ch = stim_counts.shape[0]
    stim_dur_ms = get_stim_duration_ms(stim_data, stim_path, metadata_df)
    windows = get_response_windows(stim_dur_ms, "NPRW")

    _print(f"  NPRW metrics: counts shape={stim_counts.shape}")
    _print(f"  NPRW metrics: centers={len(stim_centers)}, bin_width={stim_bw}")
    _print(f"  NPRW metrics: n_ch={n_ch}, windows={list(windows.keys())}")

    ctrl_counts = None
    ctrl_centers = None
    ctrl_bw = None
    if ctrl_data is not None and "NPRW_counts" in ctrl_data.files:
        ctrl_counts = prepare_counts_array(ctrl_data["NPRW_counts"])
        if ctrl_counts is not None and np.asarray(ctrl_counts).ndim == 3:
            ctrl_centers, ctrl_bw, _ = get_time_axis_and_width(
                ctrl_data, "NPRW", n_bins=ctrl_counts.shape[-1]
            )
        else:
            ctrl_counts = None

    rest_counts = None
    rest_centers = None
    rest_bw = None
    if rest_data is not None and "NPRW_counts" in rest_data.files:
        rest_counts = prepare_counts_array(rest_data["NPRW_counts"])
        if rest_counts is not None and np.asarray(rest_counts).ndim == 3:
            rest_centers, rest_bw, _ = get_time_axis_and_width(
                rest_data, "NPRW", n_bins=rest_counts.shape[-1]
            )
        else:
            rest_counts = None

    for ch in range(n_ch):
        is_bad_ch = ch in bad.get("NPRW", set()) or (ch + 1) in bad.get("NPRW", set())

        for win_name, win in windows.items():
            if win_name == "baseline":
                continue

            rows.append(compute_channel_metrics(
                stim_counts_ch=stim_counts[ch],
                stim_centers_ms=stim_centers,
                stim_bin_width_ms=stim_bw,
                stim_meta=stim_meta,
                channel_id=ch,
                array_type="NPRW",
                comparison_name="self_baseline",
                response_win_name=win_name,
                response_win_ms=win,
                is_bad_channel=is_bad_ch,
            ))

            rows.append(compute_channel_metrics(
                stim_counts_ch=stim_counts[ch],
                stim_centers_ms=stim_centers,
                stim_bin_width_ms=stim_bw,
                stim_meta=stim_meta,
                channel_id=ch,
                array_type="NPRW",
                comparison_name="control",
                response_win_name=win_name,
                response_win_ms=win,
                ref_counts_ch=ctrl_counts[ch] if ctrl_counts is not None and ctrl_centers is not None and ch < ctrl_counts.shape[0] else None,
                ref_centers_ms=ctrl_centers,
                ref_bin_width_ms=ctrl_bw,
                is_bad_channel=is_bad_ch,
            ))

            rows.append(compute_channel_metrics(
                stim_counts_ch=stim_counts[ch],
                stim_centers_ms=stim_centers,
                stim_bin_width_ms=stim_bw,
                stim_meta=stim_meta,
                channel_id=ch,
                array_type="NPRW",
                comparison_name="rest",
                response_win_name=win_name,
                response_win_ms=win,
                ref_counts_ch=rest_counts[ch] if rest_counts is not None and rest_centers is not None and ch < rest_counts.shape[0] else None,
                ref_centers_ms=rest_centers,
                ref_bin_width_ms=rest_bw,
                is_bad_channel=is_bad_ch,
            ))

    _print(f"  NPRW metrics: rows added={len(rows)}")
    return rows


def compute_ua_metrics_for_file(
    stim_data,
    stim_path: Path,
    stim_meta: Dict[str, Any],
    ctrl_data,
    rest_data,
    mapping: Dict[str, Any],
    bad: Dict[str, set],
    metadata_df,
) -> List[Dict[str, Any]]:

    rows = []

    stim_counts_raw = safe_get_npz(stim_data, "UA_counts", None)
    ua_ids = safe_get_npz(stim_data, "ua_ids_1based", None)

    if stim_counts_raw is None or ua_ids is None:
        _print("  UA metrics: skip, no UA_counts or ua_ids_1based")
        return rows

    stim_counts = prepare_counts_array(stim_counts_raw)
    if stim_counts is None or np.asarray(stim_counts).ndim != 3:
        _print("  UA metrics: skip, bad counts shape")
        return rows

    stim_centers, stim_bw, _ = get_time_axis_and_width(
        stim_data, "UA", n_bins=stim_counts.shape[-1]
    )
    if stim_centers is None:
        _print("  UA metrics: skip, no time axis")
        return rows

    ua_ids = np.asarray(ua_ids).astype(int).reshape(-1)
    recording_port = stim_meta.get("ua_port", None)

    elec_info, region_grids = mapping
    elec_to_idx = build_elec_to_data_idx(ua_ids, elec_info, recording_port or 'A')
    stim_dur_ms = get_stim_duration_ms(stim_data, stim_path, metadata_df)
    windows = get_response_windows(stim_dur_ms, "UA")

    _print(f"  UA metrics: counts shape={stim_counts.shape}")
    _print(f"  UA metrics: centers={len(stim_centers)}, bin_width={stim_bw}")
    _print(f"  UA metrics: mapped electrodes={len(elec_to_idx)}")
    _print(f"  UA metrics: windows={list(windows.keys())}")

    ctrl_counts = None
    ctrl_centers = None
    ctrl_bw = None
    ctrl_elec_to_idx = {}
    if ctrl_data is not None and "UA_counts" in ctrl_data.files:
        ctrl_counts = prepare_counts_array(ctrl_data["UA_counts"])
        if ctrl_counts is not None and np.asarray(ctrl_counts).ndim == 3:
            ctrl_centers, ctrl_bw, _ = get_time_axis_and_width(
                ctrl_data, "UA", n_bins=ctrl_counts.shape[-1]
            )
        else:
            ctrl_counts = None

        ctrl_ids = safe_get_npz(ctrl_data, "ua_ids_1based", None)
        if ctrl_ids is not None:
            ctrl_elec_to_idx = build_elec_to_data_idx(
                np.asarray(ctrl_ids).astype(int), elec_info, recording_port or 'A'
            )

    rest_counts = None
    rest_centers = None
    rest_bw = None
    rest_elec_to_idx = {}
    if rest_data is not None and "UA_counts" in rest_data.files:
        rest_counts = prepare_counts_array(rest_data["UA_counts"])
        if rest_counts is not None and np.asarray(rest_counts).ndim == 3:
            rest_centers, rest_bw, _ = get_time_axis_and_width(
                rest_data, "UA", n_bins=rest_counts.shape[-1]
            )
        else:
            rest_counts = None

        rest_ids = safe_get_npz(rest_data, "ua_ids_1based", None)
        if rest_ids is not None:
            rest_elec_to_idx = build_elec_to_data_idx(
                np.asarray(rest_ids).astype(int), elec_info, recording_port or 'A'
            )

    elec_to_region = {eid: info['region'] for eid, info in elec_info.items()}

    for elec, idx in elec_to_idx.items():
        if idx < 0 or idx >= stim_counts.shape[0]:
            continue

        region = None
        if elec in elec_info:
            region = str(elec_info[elec].get("region", ""))
        elif elec in elec_to_region:
            region = str(elec_to_region[elec])

        is_bad_ch = (
            elec in bad.get("UA", set())
            or idx in bad.get("UA", set())
            or (idx + 1) in bad.get("UA", set())
        )

        ctrl_idx = ctrl_elec_to_idx.get(elec, None)
        rest_idx = rest_elec_to_idx.get(elec, None)

        for win_name, win in windows.items():
            if win_name == "baseline":
                continue

            rows.append(compute_channel_metrics(
                stim_counts_ch=stim_counts[idx],
                stim_centers_ms=stim_centers,
                stim_bin_width_ms=stim_bw,
                stim_meta=stim_meta,
                channel_id=idx,
                electrode_id=elec,
                region=region,
                array_type="UA",
                comparison_name="self_baseline",
                response_win_name=win_name,
                response_win_ms=win,
                is_bad_channel=is_bad_ch,
            ))

            rows.append(compute_channel_metrics(
                stim_counts_ch=stim_counts[idx],
                stim_centers_ms=stim_centers,
                stim_bin_width_ms=stim_bw,
                stim_meta=stim_meta,
                channel_id=idx,
                electrode_id=elec,
                region=region,
                array_type="UA",
                comparison_name="control",
                response_win_name=win_name,
                response_win_ms=win,
                ref_counts_ch=ctrl_counts[ctrl_idx] if ctrl_counts is not None and ctrl_centers is not None and ctrl_idx is not None and ctrl_idx < ctrl_counts.shape[0] else None,
                ref_centers_ms=ctrl_centers,
                ref_bin_width_ms=ctrl_bw,
                is_bad_channel=is_bad_ch,
            ))

            rows.append(compute_channel_metrics(
                stim_counts_ch=stim_counts[idx],
                stim_centers_ms=stim_centers,
                stim_bin_width_ms=stim_bw,
                stim_meta=stim_meta,
                channel_id=idx,
                electrode_id=elec,
                region=region,
                array_type="UA",
                comparison_name="rest",
                response_win_name=win_name,
                response_win_ms=win,
                ref_counts_ch=rest_counts[rest_idx] if rest_counts is not None and rest_centers is not None and rest_idx is not None and rest_idx < rest_counts.shape[0] else None,
                ref_centers_ms=rest_centers,
                ref_bin_width_ms=rest_bw,
                is_bad_channel=is_bad_ch,
            ))

    _print(f"  UA metrics: rows added={len(rows)}")
    return rows




# =============================================================================
# Summary outputs
# =============================================================================

def nan_stat_or_nan(values, func):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(func(arr))

def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()

    # Exclude bad channels from summary counts.
    if "is_bad_channel" in work.columns:
        work = work[~work["is_bad_channel"].astype(bool)]

    group_cols = [
        "sess",
        "br_idx",
        "cond_type",
        "base_folder",
        "target_folder",
        "stim_freq_hz",
        "stim_dur_ms",
        "ua_port",
        "array_type",
        "region",
        "comparison",
        "response_window",
    ]

    group_cols = [c for c in group_cols if c in work.columns]

    rows = []
    for keys, g in work.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        rec = dict(zip(group_cols, keys))

        valid = g[~g["effect"].isin(["missing_reference", "uncomputed"])]
        sig = valid[valid["is_significant"].astype(bool)]

        rec.update({
            "n_channels_total": int(len(g)),
            "n_channels_valid": int(len(valid)),
            "n_significant": int(len(sig)),
            "n_increase": int(valid["is_increase"].astype(bool).sum()),
            "n_decrease": int(valid["is_decrease"].astype(bool).sum()),
            "mean_delta_hz": nan_stat_or_nan(valid["delta_hz"], np.mean) if len(valid) else np.nan,
            "median_delta_hz": nan_stat_or_nan(valid["delta_hz"], np.median) if len(valid) else np.nan,
            "mean_z": nan_stat_or_nan(valid["z"], np.mean) if len(valid) else np.nan,
            "median_z": nan_stat_or_nan(valid["z"], np.median) if len(valid) else np.nan,
        })
        rows.append(rec)

    return pd.DataFrame(rows)


def save_outputs(
    all_rows: List[Dict[str, Any]],
    stats_summary: Optional[Dict[str, Any]] = None,
):
    ensure_dir(RESULT_ROOT)

    metrics_csv = RESULT_ROOT / "stim_response_channel_metrics_all.csv"
    summary_csv = RESULT_ROOT / "stim_response_summary.csv"
    metrics_npy = RESULT_ROOT / "stim_response_channel_metrics_all.npy"

    if stats_summary is None:
        stats_summary = {}

    attempted = stats_summary.get("attempted", 0)
    successful = stats_summary.get("successful", 0)
    failed = stats_summary.get("failed", 0)
    stim_proc = stats_summary.get("stim_processed", 0)
    ctrl_proc = stats_summary.get("ctrl_processed", 0)
    rest_proc = stats_summary.get("rest_processed", 0)
    nprw_saved = stats_summary.get("nprw_figs_saved", 0)
    ua_saved = stats_summary.get("ua_figs_saved", 0)
    skipped_rest = stats_summary.get("skipped_unmatched_rest", 0)

    print("\n============================================================")
    print("Run summary:")
    print("  Configuration & Speedups:")
    print(f"    REST_MATCH_MODE:               {REST_MATCH_MODE}")
    print(f"    REQUIRE_MATCHED_REST_FOR_STIM: {REQUIRE_MATCHED_REST_FOR_STIM}")
    print(f"    CACHE_REFERENCES:              {CACHE_REFERENCES}")
    print("  File Processing:")
    print(f"    Files attempted:               {attempted}")
    if REQUIRE_MATCHED_REST_FOR_STIM:
        print(f"    STIM skipped (no rest match):  {skipped_rest}")
    print(f"    Files processed successfully:  {successful}")
    print(f"    Files skipped/failed:          {failed}")
    print(f"      STIM files processed:        {stim_proc}")
    print(f"      CTRL files processed:        {ctrl_proc}")
    print(f"      REST files processed:        {rest_proc}")
    print("  Outputs Generated:")
    print(f"    NPRW figures saved:            {nprw_saved}")
    print(f"    UA figures saved:              {ua_saved}")
    print(f"    Metric rows saved:             {len(all_rows)}")
    print(f"    Output directory:              {RESULT_ROOT}")

    if len(all_rows) == 0:
        pd.DataFrame().to_csv(metrics_csv, index=False)
        pd.DataFrame().to_csv(summary_csv, index=False)
        np.save(metrics_npy, np.array([], dtype=object))
        print("\nWARNING: No metric rows generated!")
        print("Likely causes:")
        print("  - No STIM files found or processed")
        print("  - Missing *_counts in STIM files")
        print("  - Missing *_rel_t or *_edges_ms time axis keys in STIM files")
        print("  - All STIM files skipped by PROCESS_ONLY filter or missing rest requirement")
        print("============================================================")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(metrics_csv, index=False)
    np.save(metrics_npy, np.array(all_rows, dtype=object), allow_pickle=True)

    summary = summarize_metrics(df)
    summary.to_csv(summary_csv, index=False)
    print("============================================================")

    print("\nSaved combined results:")
    print(f"  {metrics_csv}")
    print(f"  {summary_csv}")
    print(f"  {metrics_npy}")


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(FIG_ROOT)
    ensure_dir(RESULT_ROOT)

    print("Stimulation response overlay analysis")
    print(f"Session:       {PARAMS.session}")
    print(f"Monkey:        {PARAMS.monkey}")
    print(f"Peristim root: {PERI_ROOT}")
    print(f"Figure root:   {FIG_ROOT}")
    print(f"Result root:   {RESULT_ROOT}")
    print(f"Metadata CSV:  {METADATA_PATH}")
    print("Comparison modes:")
    print("  self baseline: True")
    print("  control:       True")
    print("  rest:          True")
    print(f"Threshold mode: {THRESHOLD_MODE}")
    print(f"Z threshold:    {Z_THRESH}")
    print(f"Delta threshold:{DELTA_HZ_THRESH} Hz")

    if not PERI_ROOT.exists():
        raise FileNotFoundError(f"PERI_ROOT does not exist: {PERI_ROOT}")

    all_files = sorted(PERI_ROOT.rglob("peristim__*.npz"))
    stim_files = [f for f in all_files if "stim_reaches" in str(f).lower()]
    control_files = [f for f in all_files if "control_reaches" in str(f).lower()]
    rest_files = [f for f in all_files if "at_rest" in str(f).lower()]

    print(f"Found {len(all_files)} peristim files.")
    print(f"Found {len(stim_files)} stim files.")
    print(f"Found {len(control_files)} control files.")
    print(f"Found {len(rest_files)} at-rest files.")

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
    if REQUIRE_MATCHED_REST_FOR_STIM:
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

    if len(files_to_process) == 0:
        print("No peristim files to process.")
        save_outputs([], {"attempted": 0, "skipped_unmatched_rest": n_skipped_unmatched_rest})
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

    all_rows = []
    successful_count = 0
    failed_count = 0
    stim_count = 0
    ctrl_count = 0
    rest_count = 0
    nprw_figs_count = 0
    ua_figs_count = 0

    for res in file_results:
        if isinstance(res, dict):
            if res.get("success", False):
                successful_count += 1
            else:
                failed_count += 1

            ctype = res.get("cond_type", "")
            if ctype == "STIM":
                stim_count += 1
            elif ctype == "CTRL":
                ctrl_count += 1
            elif ctype == "REST":
                rest_count += 1

            nprw_figs_count += res.get("nprw_figs", 0)
            ua_figs_count += res.get("ua_figs", 0)

            rows = res.get("rows", [])
            if rows:
                all_rows.extend(rows)
        elif isinstance(res, list):
            if res:
                all_rows.extend(res)
                successful_count += 1

    stats_summary = {
        "attempted": total_files,
        "successful": successful_count,
        "failed": failed_count,
        "stim_processed": stim_count,
        "ctrl_processed": ctrl_count,
        "rest_processed": rest_count,
        "nprw_figs_saved": nprw_figs_count,
        "ua_figs_saved": ua_figs_count,
        "skipped_unmatched_rest": n_skipped_unmatched_rest,
    }

    save_outputs(all_rows, stats_summary)
    print("Done.")


if __name__ == "__main__":
    main()