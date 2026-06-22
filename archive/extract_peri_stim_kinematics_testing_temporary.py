#!/usr/bin/env python3
"""
Test script for kinematic filtering parameter tuning.
Mirrors extract_peri_stim.py structure exactly.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from collections import defaultdict
import warnings

# Mirror exact imports from extract_peri_stim.py
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

# Variables from config that extract_peri_stim.py uses
PROCESS_ONLY = PARAMS.preprocessing.get("process_only")
HAS_KINEMATICS = PARAMS.preprocessing.get("has_kinematics")

# These should already be defined by config_loading, but if not:
if 'KEYPOINTS_ORDER' not in dir():
    KEYPOINTS_ORDER = tuple(PARAMS.kinematics.get("keypoints", []))

# ============================================================================
# EXISTING PARAMETERS (from extract_peri_stim.py)
# ============================================================================
WIN_MS = (-800.0, 600.0)
NORMALIZE_FIRST_MS = 150.0
MIN_TRIALS = 1

# Reference point for "distance from start"
DISTANCE_REFERENCE_TIME_MS = -600.0

DIRECTION_CHECK_WINDOWS_MS = (
    (-400.0, -370.0),  # baseline
    (-30.0, 0.0),      # pre-event
    (0.0, 30.0),       # post-event
)
KEYPOINT_FOR_DIRECTION_CHECK = "middle"
MIN_DISTANCE_INCREASE = 0.05

# ============================================================================
# NEW BASELINE STABILITY FILTERING - TUNABLE PARAMETERS
# ============================================================================
BASELINE_STABILITY_WINDOW_MS = (-500.0, -200.0)  # Window to check for stability
MAX_BASELINE_DEVIATION = 0.15

# ============================================================================
# NEW: RETURN-TO-BASELINE DETECTION (abort detection)
# ============================================================================
RETURN_CHECK_WINDOW_MS = (50.0, 400.0)  # Window to check for aborted reaches
RETURN_CHECK_BIN_MS = 20.0             # Bin size for rolling average
MAX_RETURN_TO_BASELINE = 0.50          # If distance drops within this of baseline mean, reject

# Output directory
DIAG_OUTPUT_ROOT = ALIGNED_CKPT_ROOT / "filtering_diagnostics"
DIAG_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Directory roots (mirroring extract_peri_stim.py)
CONTROL_ROOT = ALIGNED_CKPT_ROOT / "control_reaches"
STIM_ROOT = ALIGNED_CKPT_ROOT / "stim_reaches"
GRASP_ROOT = ALIGNED_CKPT_ROOT / "Grasp"
IMU_ROOT = ALIGNED_CKPT_ROOT / "IMU"
CONTINUOUS_STIM_ROOT = ALIGNED_CKPT_ROOT / "continuous_stim"
AT_REST_ROOT = ALIGNED_CKPT_ROOT / "at_rest"

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (copied from extract_peri_stim.py)
# ─────────────────────────────────────────────────────────────────────────────

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


def _find_keypoint_index(kp_names: list[str], pattern: str) -> int | None:
    """Find index of keypoint matching pattern (case-insensitive)."""
    pattern_lower = pattern.lower()
    for i, name in enumerate(kp_names):
        if pattern_lower in str(name).lower():
            return i
    return None


def _ordered_xy_indices(cam_cols: list[str],
                        keypoints: tuple[str, ...] = KEYPOINTS_ORDER
                        ) -> tuple[list[int], list[str]]:
    idx: list[int] = []
    names: list[str] = []
    cols_lc = [c.lower() for c in cam_cols]

    def _clean_kp(s: str) -> str:
        s = str(s).strip()
        s = s.strip("[]").strip("'\"")
        return s.lower()

    def _find_xy_for(kp: str) -> tuple[int | None, int | None]:
        kp_lc = _clean_kp(kp).lower()
        ix = next((i for i, c in enumerate(cols_lc) if c.endswith("_x") and kp_lc in c), None)
        iy = next((i for i, c in enumerate(cols_lc) if c.endswith("_y") and kp_lc in c), None)
        return ix, iy

    for kp in keypoints:
        ix, iy = _find_xy_for(kp)
        base = _clean_kp(kp)
        if ix is None or iy is None:
            idx.extend([-1, -1])
            names.extend([f"{base}_x(MISSING)", f"{base}_y(MISSING)"])
        else:
            idx.extend([ix, iy])
            names.extend([f"{base}_x", f"{base}_y"])
    return idx, names


def _select_matrix(cam: np.ndarray, cols: list[str],
                   keypoints: tuple[str, ...] = KEYPOINTS_ORDER
                   ) -> tuple[np.ndarray, list[str]]:
    idx, names = _ordered_xy_indices(cols or [], keypoints=keypoints)

    if cam is None or cam.size == 0:
        M = np.zeros((0, len(idx)), float)
        return M, names

    if cam.ndim == 1:
        cam = cam[:, None]

    n_rows, n_cols = cam.shape
    M = np.empty((n_rows, len(idx)), float)
    for k, j in enumerate(idx):
        if j == -1 or j >= n_cols:
            M[:, k] = np.nan
        else:
            M[:, k] = cam[:, j]
    return M, names


def _z_per_column(M: np.ndarray) -> np.ndarray:
    if M is None or M.size == 0:
        return np.asarray(M, dtype=float)
    if M.ndim == 1:
        M = M[:, None]
    out = M.astype(float, copy=True)
    if out.shape[1] == 0:
        return out
    for j in range(out.shape[1]):
        col = out[:, j]
        m = np.nanmean(col) if np.isfinite(col).any() else np.nan
        s = np.nanstd(col) if np.isfinite(col).any() else np.nan
        if np.isfinite(s) and s > 0:
            out[:, j] = (col - m) / s
        else:
            out[:, j] = np.full_like(col, np.nan)
    return out


def _interp_nans_1d_gentle(x: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """Fill only interior NaN runs up to max_gap by linear interp."""
    x = np.asarray(x, float)
    if x.ndim != 1 or x.size == 0:
        return x
    out = x.copy()
    isn = ~np.isfinite(out)
    if not isn.any():
        return out

    starts = np.where(isn & ~np.r_[False, isn[:-1]])[0]
    ends = np.where(isn & ~np.r_[isn[1:], False])[0]
    for s, e in zip(starts, ends):
        gap = e - s + 1
        left, right = s - 1, e + 1
        if (gap <= max_gap and left >= 0 and right < out.size
                and np.isfinite(out[left]) and np.isfinite(out[right])):
            out[s:e+1] = np.interp(np.arange(s, e+1), [left, right], [out[left], out[right]])
    return out


def _interp_nans_2d_by_col(arr: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """Apply _interp_nans_1d_gentle to each column of a 2D array."""
    arr = np.asarray(arr, float)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    out = arr.copy()
    for j in range(out.shape[1]):
        out[:, j] = _interp_nans_1d_gentle(out[:, j], max_gap=max_gap)
    return out


def _compute_distance_from_start(x: np.ndarray, y: np.ndarray, 
                                  rel_t: np.ndarray = None,
                                  ref_time_ms: float = -600.0) -> np.ndarray:
    """Compute Euclidean distance from a reference position.
    
    Args:
        x, y: Position arrays
        rel_t: Relative time array (if None, uses first valid point)
        ref_time_ms: Time point to use as reference position (default -600ms)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return np.full_like(x, np.nan)
    
    # If rel_t provided, use position at ref_time_ms
    if rel_t is not None:
        rel_t = np.asarray(rel_t, float)
        # Find points near the reference time
        ref_mask = (rel_t >= ref_time_ms - 10) & (rel_t <= ref_time_ms + 10) & valid
        if ref_mask.any():
            x0 = np.nanmean(x[ref_mask])
            y0 = np.nanmean(y[ref_mask])
        else:
            # Fallback: interpolate to ref_time
            valid_idx = np.where(valid)[0]
            x0 = np.interp(ref_time_ms, rel_t[valid_idx], x[valid_idx])
            y0 = np.interp(ref_time_ms, rel_t[valid_idx], y[valid_idx])
    else:
        # Original behavior: first valid point
        first_valid = np.where(valid)[0][0]
        x0, y0 = x[first_valid], y[first_valid]
    
    return np.sqrt((x - x0)**2 + (y - y0)**2)


def _median_lines_for_columns(series_on_common: np.ndarray,
                              t_common_ms: np.ndarray,
                              event_ms: np.ndarray
                              ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Return:
      lines      : (D, T) median line per column
      rel_t_out  : (T,) relative time axis
      n_trials   : int, max number of kept trials across columns
      segs_all   : (n_trials, D, T) baseline-zeroed segments per column
    """
    if series_on_common is None or np.size(series_on_common) == 0:
        return np.zeros((0, 0), float), np.array([], float), 0, np.zeros((0, 0, 0), float)
    if series_on_common.ndim == 1:
        series_on_common = series_on_common[:, None]

    T, D = series_on_common.shape
    if D == 0:
        return np.zeros((0, 0), float), np.array([], float), 0, np.zeros((0, 0, 0), float)

    lines: list[np.ndarray] = []
    kept_counts: list[int] = []
    segs_list: list[np.ndarray] = []
    rel_t_out: np.ndarray | None = None

    for d in range(D):
        line, rel_t, n_kept, segs = _median_behavior_line(
            series_on_common[:, d],
            t_common_ms, event_ms,
            WIN_MS, NORMALIZE_FIRST_MS, MIN_TRIALS
        )
        lines.append(line)
        kept_counts.append(int(n_kept))
        segs_list.append(segs)
        if rel_t_out is None:
            rel_t_out = rel_t

    if rel_t_out is None or rel_t_out.size == 0:
        dt = np.nanmedian(np.diff(t_common_ms)) if t_common_ms.size > 1 else 1.0
        rel_t_out = np.arange(WIN_MS[0], WIN_MS[1] + 1e-9, dt, dtype=float)

    if lines:
        lines_arr = np.vstack(lines)
    else:
        lines_arr = np.zeros((0, rel_t_out.size), float)

    max_trials = max((s.shape[0] for s in segs_list), default=0)
    if max_trials == 0:
        segs_all = np.zeros((0, D, rel_t_out.size), float)
    else:
        segs_all = np.full((max_trials, D, rel_t_out.size), np.nan, float)
        for d, segs in enumerate(segs_list):
            if segs.size == 0:
                continue
            n_d, T_d = segs.shape
            segs_all[:n_d, d, :T_d] = segs

    n_trials_used = int(max(kept_counts)) if kept_counts else 0
    return lines_arr, rel_t_out, n_trials_used, segs_all


def _median_behavior_line(series_on_common: np.ndarray,
                          t_common_ms: np.ndarray,
                          event_ms: np.ndarray,
                          win_ms: tuple[float, float],
                          baseline_ms: float,
                          min_trials: int
                          ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Return:
      line : (T,) median peri-stim trace across kept trials
      rel_t: (T,) relative time axis
      n_kept: int, number of kept trials
      segs: (n_kept, T) baseline-zeroed single-trial segments
    """
    s = np.asarray(series_on_common, float)

    try:
        rate_segs, _, rel_t, _ = rcp.extract_peristim_segments(
            rate_hz=s[None, :],
            counts=None,
            t_ms=t_common_ms,
            stim_ms=event_ms,
            win_ms=win_ms,
            min_trials=min_trials,
        )
    except RuntimeError as e:
        if "Only 0 peri-stim segments" in str(e):
            dt = np.nanmedian(np.diff(t_common_ms)) if t_common_ms.size > 1 else 1.0
            rel_t = np.arange(win_ms[0], win_ms[1] + 1e-9, dt, dtype=float)
            return np.full(rel_t.size, np.nan, float), rel_t, 0, np.zeros((0, rel_t.size), float)
        else:
            raise

    if rate_segs.size == 0:
        dt = np.nanmedian(np.diff(t_common_ms)) if t_common_ms.size > 1 else 1.0
        rel_t = np.arange(win_ms[0], win_ms[1] + 1e-9, dt, dtype=float)
        return np.full(rel_t.size, np.nan, float), rel_t, 0, np.zeros((0, rel_t.size), float)

    rate_segs = rate_segs[:, 0, :]

    bl_mask = (rel_t >= rel_t[0]) & (rel_t <= baseline_ms)
    if not bl_mask.any():
        step = max(1, int(round(baseline_ms / (np.nanmedian(np.diff(rel_t)) if rel_t.size > 1 else 1.0))))
        bl_mask = np.zeros_like(rel_t, bool)
        bl_mask[:step] = True

    keep = np.isfinite(rate_segs[:, bl_mask]).any(axis=1)
    n_kept = int(np.count_nonzero(keep))
    if n_kept == 0:
        return np.full(rate_segs.shape[1], np.nan, float), rel_t, 0, np.zeros((0, rate_segs.shape[1]), float)

    rate_segs = rate_segs[keep]
    rate_segs = rate_segs - np.nanmedian(rate_segs[:, bl_mask], axis=1, keepdims=True)
    line = np.nanmedian(rate_segs, axis=0)
    return line, rel_t, n_kept, rate_segs


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION FUNCTIONS (from extract_peri_stim.py + new baseline stability)
# ─────────────────────────────────────────────────────────────────────────────

def _check_trial_position_increasing(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    rel_t: np.ndarray,
    windows_ms: tuple[tuple[float, float], ...] = DIRECTION_CHECK_WINDOWS_MS,
    min_increase: float = MIN_DISTANCE_INCREASE,
) -> bool:
    """Original direction check from extract_peri_stim.py."""
    pos_x = np.asarray(pos_x, float).ravel()
    pos_y = np.asarray(pos_y, float).ravel()
    rel_t = np.asarray(rel_t, float).ravel()
    
    if pos_x.size == 0 or pos_y.size == 0 or rel_t.size == 0:
        return False
    
    distance = _compute_distance_from_start(pos_x, pos_y, rel_t, DISTANCE_REFERENCE_TIME_MS)
    
    avg_distances = []
    for w0, w1 in windows_ms:
        mask = (rel_t >= w0) & (rel_t <= w1)
        if not mask.any():
            return False
        
        dist_window = distance[mask]
        valid = np.isfinite(dist_window)
        if valid.sum() < 2:
            return False
        
        avg_distances.append(np.nanmean(dist_window))
    
    for i in range(1, len(avg_distances)):
        delta = avg_distances[i] - avg_distances[i - 1]
        if delta < min_increase:
            return False
    
    return True


def _check_return_to_baseline(
    distance: np.ndarray,
    rel_t: np.ndarray,
    baseline_mean: float,
    return_window_ms: tuple[float, float] = RETURN_CHECK_WINDOW_MS,
    bin_ms: float = RETURN_CHECK_BIN_MS,
    max_return: float = MAX_RETURN_TO_BASELINE,
) -> tuple[bool, float]:
    """
    Check if the animal returns too close to baseline during the reach window.
    
    Args:
        distance: Distance from start position over time
        rel_t: Relative time array
        baseline_mean: Mean distance during baseline period
        return_window_ms: Window to check for returns (default 0-400ms)
        bin_ms: Bin size for rolling average (default 20ms)
        max_return: If any bin average is within this of baseline_mean, reject
        
    Returns:
        (passed, min_distance): Whether trial passed, and the minimum binned distance found
    """
    distance = np.asarray(distance, float)
    rel_t = np.asarray(rel_t, float)
    
    # Get the return check window
    win_start, win_end = return_window_ms
    win_mask = (rel_t >= win_start) & (rel_t <= win_end)
    
    if win_mask.sum() < 3:
        return True, np.nan  # Not enough data, pass by default
    
    t_win = rel_t[win_mask]
    dist_win = distance[win_mask]
    
    # Calculate rolling averages in bins
    dt = np.nanmedian(np.diff(rel_t)) if rel_t.size > 1 else 1.0
    bin_samples = max(1, int(bin_ms / dt))
    
    min_bin_distance = np.inf
    
    # Slide through the window
    for i in range(len(dist_win) - bin_samples + 1):
        bin_vals = dist_win[i:i + bin_samples]
        valid = np.isfinite(bin_vals)
        if valid.sum() >= bin_samples // 2:  # At least half valid
            bin_mean = np.nanmean(bin_vals)
            min_bin_distance = min(min_bin_distance, bin_mean)
    
    if not np.isfinite(min_bin_distance):
        return True, np.nan
    
    # Check if minimum binned distance is too close to baseline
    # "Too close" means within max_return of baseline_mean
    threshold = baseline_mean + max_return
    passed = min_bin_distance > threshold
    
    return passed, min_bin_distance


def _check_trial_valid_comprehensive(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    rel_t: np.ndarray,
    baseline_stability_window_ms: tuple[float, float] = BASELINE_STABILITY_WINDOW_MS,
    max_baseline_deviation: float = MAX_BASELINE_DEVIATION,
    direction_windows_ms: tuple[tuple[float, float], ...] = DIRECTION_CHECK_WINDOWS_MS,
    min_increase: float = MIN_DISTANCE_INCREASE,
    return_window_ms: tuple[float, float] = RETURN_CHECK_WINDOW_MS,
    return_bin_ms: float = RETURN_CHECK_BIN_MS,
    max_return_to_baseline: float = MAX_RETURN_TO_BASELINE,
) -> tuple[bool, str, dict]:
    """
    Comprehensive trial validation combining:
    1. Baseline stability check (no false starts)
    2. Direction check (proper reach pattern)
    3. Return-to-baseline check (no aborted reaches)
    
    Returns: (is_valid, reason, metrics_dict)
    """
    pos_x = np.asarray(pos_x, float).ravel()
    pos_y = np.asarray(pos_y, float).ravel()
    rel_t = np.asarray(rel_t, float).ravel()
    
    metrics = {
        'max_baseline_dev': np.nan,
        'min_post_distance': np.nan,
        'baseline_mean_distance': np.nan,
    }
    
    if pos_x.size == 0 or pos_y.size == 0:
        return False, "empty_data", metrics
    
    # Compute distance from reference point
    distance = _compute_distance_from_start(pos_x, pos_y, rel_t, DISTANCE_REFERENCE_TIME_MS)
    
    # === CHECK 1: Baseline Stability ===
    bl_start, bl_end = baseline_stability_window_ms
    bl_mask = (rel_t >= bl_start) & (rel_t <= bl_end)
    
    if bl_mask.sum() < 5:
        return False, "insufficient_baseline", metrics
    
    dist_bl = distance[bl_mask]
    valid_bl = np.isfinite(dist_bl)
    
    if valid_bl.sum() < 3:
        return False, "insufficient_valid_baseline", metrics
    
    baseline_mean = np.nanmean(dist_bl[valid_bl])
    metrics['baseline_mean_distance'] = baseline_mean
    
    # Check max deviation from baseline mean
    max_dev = np.nanmax(np.abs(dist_bl[valid_bl] - baseline_mean))
    metrics['max_baseline_dev'] = max_dev
    
    if max_dev > max_baseline_deviation:
        return False, "unstable_baseline", metrics
    
    # === CHECK 2: Direction / Reach Pattern ===
    avg_distances = []
    for w0, w1 in direction_windows_ms:
        mask = (rel_t >= w0) & (rel_t <= w1)
        if not mask.any():
            return False, f"missing_window_{w0}_{w1}", metrics
        
        dist_window = distance[mask]
        valid_w = np.isfinite(dist_window)
        if valid_w.sum() < 2:
            return False, f"insufficient_data_window_{w0}_{w1}", metrics
        
        avg_distances.append(np.nanmean(dist_window))
    
    for i in range(1, len(avg_distances)):
        delta = avg_distances[i] - avg_distances[i - 1]
        if delta < min_increase:
            return False, f"no_increase_window_{i}", metrics
    
    # === CHECK 3: Return-to-Baseline (Aborted Reach) ===
    return_passed, min_post_dist = _check_return_to_baseline(
        distance, rel_t, baseline_mean,
        return_window_ms, return_bin_ms, max_return_to_baseline
    )
    metrics['min_post_distance'] = min_post_dist
    
    if not return_passed:
        return False, "return_to_baseline", metrics
    
    return True, "valid", metrics


def _get_position_valid_mask_comprehensive(
    cam_segs: np.ndarray,
    rel_t: np.ndarray,
    kp_names: list[str],
    keypoint_base: str = "middle",
    baseline_stability_window_ms: tuple[float, float] = BASELINE_STABILITY_WINDOW_MS,
    max_baseline_deviation: float = MAX_BASELINE_DEVIATION,
    direction_windows_ms: tuple[tuple[float, float], ...] = DIRECTION_CHECK_WINDOWS_MS,
    min_increase: float = MIN_DISTANCE_INCREASE,
    return_window_ms: tuple[float, float] = RETURN_CHECK_WINDOW_MS,
    return_bin_ms: float = RETURN_CHECK_BIN_MS,
    max_return_to_baseline: float = MAX_RETURN_TO_BASELINE,
) -> tuple[np.ndarray, dict]:
    """
    Check all trials with comprehensive validation.
    
    Returns:
        mask: (n_trials,) bool
        diagnostics: dict with rejection reasons, metrics, etc.
    """
    if cam_segs is None or cam_segs.size == 0:
        return np.ones(0, dtype=bool), {}
    
    if cam_segs.ndim == 2:
        print("[warn] cam_segs is 2D, skipping position validation")
        return np.ones(cam_segs.shape[0], dtype=bool), {}
    
    n_trials, n_kp, T = cam_segs.shape
    
    idx_x = _find_keypoint_index(kp_names, f"{keypoint_base}_x")
    idx_y = _find_keypoint_index(kp_names, f"{keypoint_base}_y")
    
    if idx_x is None or idx_y is None:
        print(f"[warn] Could not find {keypoint_base}_x and {keypoint_base}_y")
        return np.ones(n_trials, dtype=bool), {}
    
    mask = np.zeros(n_trials, dtype=bool)
    rejection_reasons = {}
    max_deviations = np.zeros(n_trials)
    min_post_distances = np.zeros(n_trials)
    baseline_means = np.zeros(n_trials)
    direction_valid = np.zeros(n_trials, dtype=bool)
    stability_valid = np.zeros(n_trials, dtype=bool)
    return_valid = np.zeros(n_trials, dtype=bool)
    
    for i in range(n_trials):
        x = cam_segs[i, idx_x, :]
        y = cam_segs[i, idx_y, :]
        
        is_valid, reason, metrics = _check_trial_valid_comprehensive(
            x, y, rel_t,
            baseline_stability_window_ms=baseline_stability_window_ms,
            max_baseline_deviation=max_baseline_deviation,
            direction_windows_ms=direction_windows_ms,
            min_increase=min_increase,
            return_window_ms=return_window_ms,
            return_bin_ms=return_bin_ms,
            max_return_to_baseline=max_return_to_baseline,
        )
        
        mask[i] = is_valid
        max_deviations[i] = metrics.get('max_baseline_dev', np.nan)
        min_post_distances[i] = metrics.get('min_post_distance', np.nan)
        baseline_means[i] = metrics.get('baseline_mean_distance', np.nan)
        
        # Check individual components for diagnostics
        direction_valid[i] = _check_trial_position_increasing(x, y, rel_t, direction_windows_ms, min_increase)
        stability_valid[i] = (max_deviations[i] <= max_baseline_deviation) if np.isfinite(max_deviations[i]) else False
        
        # Return check needs baseline mean
        if np.isfinite(baseline_means[i]):
            distance = _compute_distance_from_start(x, y, rel_t, DISTANCE_REFERENCE_TIME_MS)
            ret_pass, _ = _check_return_to_baseline(
                distance, rel_t, baseline_means[i],
                return_window_ms, return_bin_ms, max_return_to_baseline
            )
            return_valid[i] = ret_pass
        else:
            return_valid[i] = True  # Can't check, pass by default
        
        if not is_valid:
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    
    diagnostics = {
        'rejection_reasons': rejection_reasons,
        'max_deviations': max_deviations,
        'min_post_distances': min_post_distances,
        'baseline_means': baseline_means,
        'direction_valid': direction_valid,
        'stability_valid': stability_valid,
        'return_valid': return_valid,
    }
    
    return mask, diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_filtering_diagnostics(
    cam_segs: np.ndarray,
    rel_t: np.ndarray,
    kp_names: list[str],
    mask: np.ndarray,
    diagnostics: dict,
    title_prefix: str = "",
    output_path: Path = None,
):
    """Generate comprehensive diagnostic plots for filtering results."""
    
    # Find keypoint indices
    idx_x = _find_keypoint_index(kp_names, f"{KEYPOINT_FOR_DIRECTION_CHECK}_x")
    idx_y = _find_keypoint_index(kp_names, f"{KEYPOINT_FOR_DIRECTION_CHECK}_y")
    
    if idx_x is None or idx_y is None:
        print(f"[warn] Cannot plot: keypoint not found")
        return
    
    n_trials = cam_segs.shape[0]
    max_deviations = diagnostics.get('max_deviations', np.zeros(n_trials))
    min_post_distances = diagnostics.get('min_post_distances', np.zeros(n_trials))
    baseline_means = diagnostics.get('baseline_means', np.zeros(n_trials))
    direction_valid = diagnostics.get('direction_valid', np.ones(n_trials, dtype=bool))
    stability_valid = diagnostics.get('stability_valid', np.ones(n_trials, dtype=bool))
    return_valid = diagnostics.get('return_valid', np.ones(n_trials, dtype=bool))
    
    # Compute distance from start for all trials
    distances = np.zeros((n_trials, len(rel_t)))
    for i in range(n_trials):
        distances[i] = _compute_distance_from_start(
            cam_segs[i, idx_x, :], cam_segs[i, idx_y, :],
            rel_t, DISTANCE_REFERENCE_TIME_MS
        )
    
    fig = plt.figure(figsize=(18, 14))
    
    n_valid = mask.sum()
    n_stability_fail = (~stability_valid).sum()
    n_direction_fail = (~direction_valid).sum()
    n_return_fail = (~return_valid).sum()
    
    # 1. Pie chart of rejection reasons
    ax1 = fig.add_subplot(2, 4, 1)
    rejection_reasons = diagnostics.get('rejection_reasons', {})
    if rejection_reasons:
        labels = [f'{k}\n({v})' for k, v in rejection_reasons.items()]
        sizes = list(rejection_reasons.values())
        # Add valid
        labels = [f'Valid ({n_valid})'] + labels
        sizes = [n_valid] + sizes
        colors = ['#2ecc71'] + plt.cm.Reds(np.linspace(0.3, 0.8, len(rejection_reasons))).tolist()
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 8})
    else:
        ax1.pie([n_valid], labels=[f'Valid ({n_valid})'], colors=['#2ecc71'], autopct='%1.1f%%')
    ax1.set_title(f'Rejection Breakdown\n(n={n_trials})')
    
    # 2. Histogram of baseline deviations
    ax2 = fig.add_subplot(2, 4, 2)
    finite_devs = max_deviations[np.isfinite(max_deviations)]
    if len(finite_devs) > 0:
        ax2.hist(finite_devs, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(MAX_BASELINE_DEVIATION, color='red', linestyle='--', linewidth=2, 
                    label=f'Threshold={MAX_BASELINE_DEVIATION}')
    ax2.set_xlabel('Max Baseline Deviation')
    ax2.set_ylabel('Count')
    ax2.set_title('Baseline Deviation Distribution')
    ax2.legend()
    
    # 3. Histogram of min post-stim distances
    ax3 = fig.add_subplot(2, 4, 3)
    finite_post = min_post_distances[np.isfinite(min_post_distances)]
    finite_bl = baseline_means[np.isfinite(baseline_means)]
    if len(finite_post) > 0:
        ax3.hist(finite_post, bins=30, alpha=0.7, edgecolor='black', label='Min post distance')
        if len(finite_bl) > 0:
            mean_bl = np.mean(finite_bl)
            ax3.axvline(mean_bl + MAX_RETURN_TO_BASELINE, color='red', linestyle='--', 
                        linewidth=2, label=f'Threshold (bl_mean + {MAX_RETURN_TO_BASELINE})')
    ax3.set_xlabel('Min Distance in Return Window')
    ax3.set_ylabel('Count')
    ax3.set_title('Post-Stim Distance Distribution')
    ax3.legend()
    
    # 4. Threshold sensitivity curve
    ax4 = fig.add_subplot(2, 4, 4)
    thresholds = np.linspace(0.01, 0.3, 50)
    n_passing = []
    for thresh in thresholds:
        stability_pass = max_deviations <= thresh
        combined_pass = stability_pass & direction_valid & return_valid
        n_passing.append(combined_pass.sum())
    ax4.plot(thresholds, n_passing, 'b-', linewidth=2)
    ax4.axvline(MAX_BASELINE_DEVIATION, color='red', linestyle='--', 
                label=f'Current: {MAX_BASELINE_DEVIATION}')
    ax4.set_xlabel('Baseline Deviation Threshold')
    ax4.set_ylabel('Trials Passing')
    ax4.set_title('Threshold Sensitivity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Distance traces - valid trials
    ax5 = fig.add_subplot(2, 4, 5)
    if mask.any():
        for i in np.where(mask)[0][:20]:
            ax5.plot(rel_t, distances[i, :], 'g-', alpha=0.3, linewidth=0.5)
        ax5.plot(rel_t, np.nanmedian(distances[mask, :], axis=0), 'g-', linewidth=2, label='Median')
    ax5.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax5.axvspan(BASELINE_STABILITY_WINDOW_MS[0], BASELINE_STABILITY_WINDOW_MS[1], 
                alpha=0.1, color='blue', label='Stability window')
    ax5.axvspan(RETURN_CHECK_WINDOW_MS[0], RETURN_CHECK_WINDOW_MS[1], 
                alpha=0.1, color='orange', label='Return check window')
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Distance from start')
    ax5.set_title(f'Valid Trials (n={n_valid})')
    ax5.legend(fontsize=8)
    
    # 6. Distance traces - rejected by stability
    ax6 = fig.add_subplot(2, 4, 6)
    stability_rejected = ~stability_valid
    if stability_rejected.any():
        for i in np.where(stability_rejected)[0][:20]:
            ax6.plot(rel_t, distances[i, :], 'r-', alpha=0.3, linewidth=0.5)
        ax6.plot(rel_t, np.nanmedian(distances[stability_rejected, :], axis=0), 'r-', linewidth=2, label='Median')
    ax6.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax6.axvspan(BASELINE_STABILITY_WINDOW_MS[0], BASELINE_STABILITY_WINDOW_MS[1], 
                alpha=0.1, color='blue', label='Stability window')
    ax6.set_xlabel('Time (ms)')
    ax6.set_ylabel('Distance from start')
    ax6.set_title(f'Stability-Rejected (n={stability_rejected.sum()})')
    ax6.legend(fontsize=8)
    
    # 7. Distance traces - rejected by return-to-baseline
    ax7 = fig.add_subplot(2, 4, 7)
    return_rejected = (~return_valid) & stability_valid & direction_valid
    if return_rejected.any():
        for i in np.where(return_rejected)[0][:20]:
            ax7.plot(rel_t, distances[i, :], 'purple', alpha=0.3, linewidth=0.5)
        ax7.plot(rel_t, np.nanmedian(distances[return_rejected, :], axis=0), 
                 'purple', linewidth=2, label='Median')
    ax7.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax7.axvspan(RETURN_CHECK_WINDOW_MS[0], RETURN_CHECK_WINDOW_MS[1], 
                alpha=0.1, color='orange', label='Return check window')
    ax7.set_xlabel('Time (ms)')
    ax7.set_ylabel('Distance from start')
    ax7.set_title(f'Return-to-Baseline Rejected (n={return_rejected.sum()})')
    ax7.legend(fontsize=8)
    
    # 8. Distance traces - rejected by direction only
    ax8 = fig.add_subplot(2, 4, 8)
    direction_only_rejected = (~direction_valid) & stability_valid & return_valid
    if direction_only_rejected.any():
        for i in np.where(direction_only_rejected)[0][:20]:
            ax8.plot(rel_t, distances[i, :], 'orange', alpha=0.3, linewidth=0.5)
        ax8.plot(rel_t, np.nanmedian(distances[direction_only_rejected, :], axis=0), 
                 'orange', linewidth=2, label='Median')
    ax8.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax8.axvspan(BASELINE_STABILITY_WINDOW_MS[0], BASELINE_STABILITY_WINDOW_MS[1], 
                alpha=0.1, color='blue', label='Stability window')
    ax8.set_xlabel('Time (ms)')
    ax8.set_ylabel('Distance from start')
    ax8.set_title(f'Direction-Only Rejected (n={direction_only_rejected.sum()})')
    ax8.legend(fontsize=8)
    
    plt.suptitle(f'{title_prefix}' if title_prefix else 'Kinematic Filtering Diagnostics', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    
    plt.close(fig)


def plot_threshold_tuning_grid(
    cam_segs: np.ndarray,
    rel_t: np.ndarray,
    kp_names: list[str],
    direction_valid: np.ndarray,
    return_valid: np.ndarray,
    title_prefix: str = "",
    output_path: Path = None,
):
    """Grid plot showing effects of different threshold values."""
    
    idx_x = _find_keypoint_index(kp_names, f"{KEYPOINT_FOR_DIRECTION_CHECK}_x")
    idx_y = _find_keypoint_index(kp_names, f"{KEYPOINT_FOR_DIRECTION_CHECK}_y")
    
    if idx_x is None or idx_y is None:
        return
    
    n_trials = cam_segs.shape[0]
    
    # Compute distances and max deviations for all trials
    distances = np.zeros((n_trials, len(rel_t)))
    max_deviations = np.zeros(n_trials)
    
    bl_mask = (rel_t >= BASELINE_STABILITY_WINDOW_MS[0]) & (rel_t <= BASELINE_STABILITY_WINDOW_MS[1])
    
    for i in range(n_trials):
        x = cam_segs[i, idx_x, :]
        y = cam_segs[i, idx_y, :]
        distances[i] = _compute_distance_from_start(x, y, rel_t, DISTANCE_REFERENCE_TIME_MS)
        
        # Compute baseline deviation using distance
        dist_bl = distances[i, bl_mask]
        valid_bl = np.isfinite(dist_bl)
        if valid_bl.sum() > 0:
            bl_mean = np.nanmean(dist_bl[valid_bl])
            max_deviations[i] = np.nanmax(np.abs(dist_bl[valid_bl] - bl_mean))
        else:
            max_deviations[i] = np.nan
    
    thresholds = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for ax, thresh in zip(axes, thresholds):
        stability_valid = max_deviations <= thresh
        combined_valid = stability_valid & direction_valid & return_valid
        
        n_valid = combined_valid.sum()
        
        # Plot valid trials
        if combined_valid.any():
            for i in np.where(combined_valid)[0][:15]:
                ax.plot(rel_t, distances[i, :], 'g-', alpha=0.3, linewidth=0.5)
        
        # Plot rejected trials
        rejected = ~combined_valid
        if rejected.any():
            for i in np.where(rejected)[0][:15]:
                ax.plot(rel_t, distances[i, :], 'r-', alpha=0.2, linewidth=0.5)
        
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.axvspan(BASELINE_STABILITY_WINDOW_MS[0], BASELINE_STABILITY_WINDOW_MS[1], 
                   alpha=0.1, color='blue')
        ax.axvspan(RETURN_CHECK_WINDOW_MS[0], RETURN_CHECK_WINDOW_MS[1], 
                   alpha=0.1, color='orange')
        ax.set_title(f'Threshold={thresh}\n{n_valid}/{n_trials} valid ({100*n_valid/n_trials:.1f}%)')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Distance')
    
    plt.suptitle(f'{title_prefix} - Threshold Tuning' if title_prefix else 'Threshold Tuning Grid', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FILE PROCESSING (mirrors extract_peri_stim.py logic)
# ─────────────────────────────────────────────────────────────────────────────

def process_file_for_diagnostics(
    npz_path: Path,
    use_ir_ms: bool,
    output_dir: Path,
    file_type: str,
):
    """Process a single file and generate diagnostic plots."""
    stem = npz_path.stem.replace("aligned__", "")
    print(f"\nProcessing: {stem} ({file_type})")
    
    # Load data (same as extract_peri_stim.py)
    aligned_npz = np.load(npz_path, allow_pickle=True)
    
    # Get behavior data
    if "beh_cam0" not in aligned_npz.files:
        print(f"  Skipping: no behavior data")
        return
    
    beh_cam0 = aligned_npz["beh_cam0"]
    raw0 = aligned_npz["beh_cam0_cols"]
    beh_cam0_cols = [str(x) for x in _as_list(raw0)]
    behv_t = aligned_npz["beh_t_ms"].astype(float) if "beh_t_ms" in aligned_npz.files else np.arange(0.0)
    
    if behv_t.size == 0 or beh_cam0.size == 0:
        print(f"  Skipping: empty behavior data")
        return
    
    # Get events
    if use_ir_ms:
        event_ms = aligned_npz["ir_ms"]
    else:
        event_ms = aligned_npz["stim_ms"]
    
    if event_ms.size == 0:
        print(f"  Skipping: no events")
        return
    
    # Interpolate NaNs
    if beh_cam0.size:
        beh_cam0 = _interp_nans_2d_by_col(beh_cam0, max_gap=4)
    
    # Select and z-score
    cam0_M, beh_cam0_names = _select_matrix(beh_cam0, beh_cam0_cols)
    cam0_z = _z_per_column(cam0_M) if cam0_M.size else cam0_M
    
    # Extract peri-stim segments
    _, beh_rel_t, n_beh, beh_cam0_segs = _median_lines_for_columns(
        cam0_z, behv_t, event_ms
    )
    
    if beh_cam0_segs.size == 0 or n_beh == 0:
        print(f"  Skipping: no valid segments extracted")
        return
    
    print(f"  Extracted {n_beh} trials, {beh_cam0_segs.shape}")
    
    # Run comprehensive validation
    mask, diagnostics = _get_position_valid_mask_comprehensive(
        cam_segs=beh_cam0_segs,
        rel_t=beh_rel_t,
        kp_names=beh_cam0_names,
        keypoint_base=KEYPOINT_FOR_DIRECTION_CHECK,
        baseline_stability_window_ms=BASELINE_STABILITY_WINDOW_MS,
        max_baseline_deviation=MAX_BASELINE_DEVIATION,
        direction_windows_ms=DIRECTION_CHECK_WINDOWS_MS,
        min_increase=MIN_DISTANCE_INCREASE,
        return_window_ms=RETURN_CHECK_WINDOW_MS,
        return_bin_ms=RETURN_CHECK_BIN_MS,
        max_return_to_baseline=MAX_RETURN_TO_BASELINE,
    )
    
    # Print summary
    n_total = len(mask)
    n_valid = mask.sum()
    direction_valid = diagnostics.get('direction_valid', np.ones(n_total, dtype=bool))
    stability_valid = diagnostics.get('stability_valid', np.ones(n_total, dtype=bool))
    return_valid = diagnostics.get('return_valid', np.ones(n_total, dtype=bool))
    
    print(f"  Total trials: {n_total}")
    print(f"  Valid (all checks): {n_valid} ({100*n_valid/n_total:.1f}%)")
    print(f"  Direction failures: {(~direction_valid).sum()}")
    print(f"  Stability failures: {(~stability_valid).sum()}")
    print(f"  Return-to-baseline failures: {(~return_valid).sum()}")
    print(f"  Rejection reasons: {diagnostics.get('rejection_reasons', {})}")
    
    # Generate diagnostic plots
    file_output_dir = output_dir / file_type
    file_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Main diagnostics plot
    plot_filtering_diagnostics(
        beh_cam0_segs, beh_rel_t, beh_cam0_names,
        mask, diagnostics,
        title_prefix=f"{stem} ({file_type})",
        output_path=file_output_dir / f"{stem}_diagnostics.png"
    )
    
    # Threshold tuning grid
    plot_threshold_tuning_grid(
        beh_cam0_segs, beh_rel_t, beh_cam0_names,
        direction_valid, return_valid,
        title_prefix=f"{stem} ({file_type})",
        output_path=file_output_dir / f"{stem}_threshold_tuning.png"
    )


def process_file_type(
    root_dir: Path,
    file_type: str,
    use_ir_ms: bool,
    output_dir: Path,
):
    """Process all files of a given type."""
    if not root_dir.exists():
        print(f"\nSkipping {file_type}: {root_dir} does not exist")
        return
    
    files = sorted(root_dir.glob("aligned__*.npz"))
    
    # Apply PROCESS_ONLY filter if defined (same as extract_peri_stim.py)
    if PROCESS_ONLY:
        keep = set(int(x) for x in PROCESS_ONLY)
        files = [f for f in files if (m := re.search(r"BR_(\d+)", f.name)) and int(m.group(1)) in keep]
    
    if not files:
        print(f"\nNo files found for {file_type}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {file_type} ({len(files)} files)")
    print(f"{'='*60}")
    
    for npz_path in files:
        try:
            process_file_for_diagnostics(npz_path, use_ir_ms, output_dir, file_type)
        except Exception as e:
            print(f"  ERROR processing {npz_path.name}: {e}")
            import traceback
            traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main function - runs exactly like extract_peri_stim.py."""
    
    print("="*60)
    print("Kinematic Filtering Diagnostics")
    print("="*60)
    print(f"\nParameters:")
    print(f"  Distance reference time: {DISTANCE_REFERENCE_TIME_MS} ms")
    print(f"  Direction check windows: {DIRECTION_CHECK_WINDOWS_MS}")
    print(f"  Min distance increase: {MIN_DISTANCE_INCREASE}")
    print(f"  Baseline stability window: {BASELINE_STABILITY_WINDOW_MS}")
    print(f"  Max baseline deviation: {MAX_BASELINE_DEVIATION}")
    print(f"  Return check window: {RETURN_CHECK_WINDOW_MS}")
    print(f"  Return check bin size: {RETURN_CHECK_BIN_MS} ms")
    print(f"  Max return to baseline: {MAX_RETURN_TO_BASELINE}")
    print(f"  Keypoint for checks: {KEYPOINT_FOR_DIRECTION_CHECK}")
    print(f"\nOutput directory: {DIAG_OUTPUT_ROOT}")
    
    # Process all file types (same order as extract_peri_stim.py)
    process_file_type(CONTROL_ROOT, "control_reaches", use_ir_ms=True, output_dir=DIAG_OUTPUT_ROOT)
    process_file_type(STIM_ROOT, "stim_reaches", use_ir_ms=False, output_dir=DIAG_OUTPUT_ROOT)
    process_file_type(GRASP_ROOT, "grasp", use_ir_ms=False, output_dir=DIAG_OUTPUT_ROOT)
    process_file_type(IMU_ROOT, "imu", use_ir_ms=False, output_dir=DIAG_OUTPUT_ROOT)
    process_file_type(CONTINUOUS_STIM_ROOT, "continuous_stim", use_ir_ms=True, output_dir=DIAG_OUTPUT_ROOT)
    process_file_type(AT_REST_ROOT, "at_rest", use_ir_ms=False, output_dir=DIAG_OUTPUT_ROOT)
    
    print("\n" + "="*60)
    print("DONE - Diagnostic plots saved to:")
    print(f"  {DIAG_OUTPUT_ROOT}")
    print("="*60)


if __name__ == "__main__":
    main()