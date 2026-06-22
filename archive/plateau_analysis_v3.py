"""
plateau_analysis_v3.py

Analyze normalized hand position to detect reach endpoints (plateaus).
Uses MEDIAN TRACE for plateau detection - more robust to outliers.

Key Changes from v2:
  1. Plateau detection is performed on the MEDIAN trace (not per-trial)
  2. Individual traces are smoothed before computing median
  3. Statistical comparisons use the single median-detected plateau per condition
  4. Diagnostic plots show median trace + 95% CI (SEM) + individual traces

Outputs:
  - Per-condition diagnostic plots (all trials + median + 95% CI + plateau marker)
  - Raw trace alignment diagnostic plots
  - Master/subset summary plots (median ± 95% CI)
  - Bar charts comparing plateau values across conditions
  - Summary statistics CSV
"""

import numpy as np
import warnings
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Import the reach onset detection module
from RCP_analysis.python.functions.reach_onset_detection import (
    detect_reach_onset_batch,
    compute_2d_speed,
    align_traces_to_onset,
    ReachOnsetConfig,
    smooth_1d
)

# Import your config loading (adjust path as needed)
from RCP_analysis.python.functions.config_loading import *


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
CAMERA_TO_USE = "cam1"
KEYPOINT_BASE = "middle"

FIG_OUT_DIR = OUT_BASE / "figures" / "plateau_analysis_v3"
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FORMAT = 'png'
plt.rcParams.update({'font.size': 10})

# ---------------------------------------------------------------------
# TIME REFERENCE CONFIGURATION
# ---------------------------------------------------------------------
TIME_ZERO_REFERENCE = 'stim_timing'

# ---------------------------------------------------------------------
# PLOT CONTROL
# ---------------------------------------------------------------------
GENERATE_CONDITION_DIAGNOSTICS = True
GENERATE_RAW_TRACE_DIAGNOSTICS = True
GENERATE_MASTER_PLOT = True
GENERATE_SUBSET_PLOT = True
GENERATE_STATS_SUMMARY = True

# SUBSET_CONDITIONS = [22, 23, 24, 25, 27] #11
# SUBSET_CONDITIONS = [3, 9, 10, 11, 12] #12
SUBSET_CONDITIONS = [1, 14, 15, 16, 17] #22

# ---------------------------------------------------------------------
# CONTROL CONDITION SPECIFICATION
# ---------------------------------------------------------------------
CONTROL_BR_INDICES = [1]

# ---------------------------------------------------------------------
# REACH ONSET DETECTION CONFIG
# ---------------------------------------------------------------------
ONSET_CONFIG = ReachOnsetConfig(
    start_pos_max_thresh=0.2,
    low_speed_thresh_ratio=0.05,
    stable_window_ms=20.0,
    max_reach_duration_ms=800.0,
    min_reach_duration_ms=100.0,
)

PRE_ZERO_MS = 500.0
POST_ZERO_MS = 500.0

# ---------------------------------------------------------------------
# SMOOTHING PARAMETERS
# ---------------------------------------------------------------------
SMOOTHING_METHOD = 'savgol'
SAVGOL_WINDOW_MS = 60.0
SAVGOL_POLYORDER = 4

# ---------------------------------------------------------------------
# PLATEAU DETECTION PARAMETERS (for median trace)
# ---------------------------------------------------------------------
PLATEAU_WINDOW_MS = 100.0
MIN_DISTANCE_THRESHOLD = 0.95
MAX_CUMULATIVE_SLOPE = 0.05

PLATEAU_VALUE_WINDOW_SAMPLES = 5

# ---------------------------------------------------------------------
# SESSION-SPECIFIC BR FILTERS
# ---------------------------------------------------------------------
SESSION_BR_FILTERS = {}

def get_br_filter_for_session():
    for session_key, br_list in SESSION_BR_FILTERS.items():
        if session_key in PARAMS.session:
            return br_list
    return None

BR_FILTER = get_br_filter_for_session()

_session_root = PERI_ROOT.parent.parent.parent
_session_name = PARAMS.session
METADATA_CSV = _session_root / "Metadata" / f"{_session_name}_metadata.csv"

if not METADATA_CSV.exists():
    _alt_paths = [
        _session_root / "Metadata" / f"{_session_name.split('_')[-1]}_metadata.csv",
        _session_root / "Metadata" / f"{_session_name.replace('NRR_', '')}_metadata.csv",
    ]
    for _alt in _alt_paths:
        if _alt.exists():
            METADATA_CSV = _alt
            break


# ---------------------------------------------------------------------
# METADATA PARSING
# ---------------------------------------------------------------------

def load_metadata_csv(csv_path: Path) -> pd.DataFrame:
    """Load and normalize metadata CSV."""
    if not csv_path.exists():
        print(f"[warn] Metadata CSV not found: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def get_condition_info(br_idx: int, metadata_df: pd.DataFrame) -> dict:
    """Extract condition information for a given BR index."""
    if CONTROL_BR_INDICES is not None and br_idx in CONTROL_BR_INDICES:
        return {
            'condition_type': 'control',
            'label': 'Control',
            'n_channels': 0,
            'frequency_hz': 0,
            'duration_ms': 0,
            'n_pulses': 0,
            'br_idx': br_idx,
            'is_at_rest': False,
        }
    
    if metadata_df is None or metadata_df.empty:
        return _default_condition_info(br_idx)
    
    br_col = None
    for col_name in ['br_file', 'br_idx', 'br', 'br_index', 'condition']:
        if col_name in metadata_df.columns:
            br_col = col_name
            break
    
    if br_col is None:
        return _default_condition_info(br_idx)
    
    mask = metadata_df[br_col] == br_idx
    if not mask.any():
        return _default_condition_info(br_idx)
    
    row = metadata_df[mask].iloc[0]
    
    notes = _get_field(row, ['notes', 'note', 'comment', 'type', 'condition_type'])
    is_at_rest = False
    if notes is not None:
        notes_lower = str(notes).lower()
        if 'at rest' in notes_lower or 'at_rest' in notes_lower or 'atrest' in notes_lower:
            is_at_rest = True
    
    channels = _get_field(row, ['channels', 'n_channels', 'num_channels'])
    freq_hz = _get_field(row, ['stim_frequency_hz', 'frequency_hz', 'freq_hz', 'frequency'])
    duration = _get_field(row, ['stim_duration_ms', 'duration_ms', 'duration'])
    
    is_baseline = False
    if notes is not None:
        notes_lower = str(notes).lower()
        if any(x in notes_lower for x in ['baseline', 'control', 'no stim', 'nostim']):
            is_baseline = True
    if _is_empty(channels) and _is_empty(duration):
        is_baseline = True
    if str(channels).strip() in ['-', '0']:
        is_baseline = True
    
    if is_baseline:
        return {
            'condition_type': 'control',
            'label': 'Control',
            'n_channels': 0,
            'frequency_hz': 0,
            'duration_ms': 0,
            'n_pulses': 0,
            'br_idx': br_idx,
            'is_at_rest': is_at_rest,
        }
    
    n_ch = _parse_int(channels, default=0)
    freq = _parse_float(freq_hz, default=0)
    
    dur_str = str(duration).strip().lower() if duration is not None else ''
    if dur_str in ['cont', 'continuous', 'inf', 'ongoing']:
        return {
            'condition_type': 'continuous_stim',
            'label': f'Cont {int(freq)}Hz {n_ch}Ch',
            'n_channels': n_ch,
            'frequency_hz': freq,
            'duration_ms': 'cont',
            'n_pulses': None,
            'br_idx': br_idx,
            'is_at_rest': is_at_rest,
        }
    
    dur_ms = _parse_float(duration, default=0)
    n_pulses = int(dur_ms / 2.5) if dur_ms > 0 else 0
    
    if n_pulses > 0:
        label = f'{n_pulses}p {int(freq)}Hz {n_ch}Ch'
    elif dur_ms > 0:
        label = f'{int(dur_ms)}ms {int(freq)}Hz {n_ch}Ch'
    else:
        label = f'Stim {int(freq)}Hz {n_ch}Ch'
    
    return {
        'condition_type': 'pulsed_stim',
        'label': label,
        'n_channels': n_ch,
        'frequency_hz': freq,
        'duration_ms': dur_ms,
        'n_pulses': n_pulses,
        'br_idx': br_idx,
        'is_at_rest': is_at_rest,
    }


def _default_condition_info(br_idx: int) -> dict:
    if CONTROL_BR_INDICES is not None and br_idx in CONTROL_BR_INDICES:
        return {
            'condition_type': 'control',
            'label': 'Control',
            'n_channels': 0,
            'frequency_hz': 0,
            'duration_ms': 0,
            'n_pulses': 0,
            'br_idx': br_idx,
            'is_at_rest': False,
        }
    return {
        'condition_type': 'pulsed_stim',
        'label': f'BR{br_idx}',
        'n_channels': None,
        'frequency_hz': None,
        'duration_ms': None,
        'n_pulses': None,
        'br_idx': br_idx,
        'is_at_rest': False,
    }


def _get_field(row: pd.Series, possible_names: list):
    for name in possible_names:
        if name in row.index:
            val = row[name]
            if pd.notna(val):
                return val
    return None


def _is_empty(val) -> bool:
    if val is None:
        return True
    if pd.isna(val):
        return True
    if str(val).strip() in ('', '-', 'nan', 'none', 'na'):
        return True
    return False


def _parse_int(val, default=0) -> int:
    if _is_empty(val):
        return default
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return default


def _parse_float(val, default=0.0) -> float:
    if _is_empty(val):
        return default
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def find_keypoint_index(kp_names: list, pattern: str) -> Optional[int]:
    pattern_lower = pattern.lower()
    for i, name in enumerate(kp_names):
        if pattern_lower in str(name).lower():
            return i
    return None


def get_kinematics_id(filename: str) -> str:
    stem = filename.replace('.npz', '')
    if "baseline__" in stem:
        parts = stem.split("_target_")
        if len(parts) == 2:
            session_part = parts[0].split("_Depth")[0] if "_Depth" in parts[0] else parts[0]
            return f"{session_part}_target_{parts[1]}"
        if "_Depth" in stem:
            return stem.split("_Depth")[0]
    return stem


def get_br_from_condition_string(cond_str: str) -> Optional[int]:
    for pattern in [r"Cond_(\d+)", r"BR_(\d+)", r"(\d+)"]:
        match = re.search(pattern, cond_str)
        if match:
            return int(match.group(1))
    return None


def _get_keypoint_names_from_data(data: dict, camera: str) -> list[str]:
    possible_keys = [
        f"beh_{camera}_names", f"{camera}_names", "beh_names", "keypoint_names", "kp_names",
    ]
    for key in possible_keys:
        if key in data:
            names = data[key]
            return names.tolist() if hasattr(names, 'tolist') else list(names)
    return []


def smooth_trace(trace: np.ndarray, dt: float, window_ms: float = SAVGOL_WINDOW_MS, 
                 polyorder: int = SAVGOL_POLYORDER) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to a trace."""
    window_samples = int(window_ms / dt)
    if window_samples < polyorder + 2:
        window_samples = polyorder + 2
    if window_samples % 2 == 0:
        window_samples += 1
    
    # Handle NaNs
    valid_mask = np.isfinite(trace)
    if np.sum(valid_mask) < window_samples:
        return trace.copy()
    
    smoothed = trace.copy()
    if np.all(valid_mask):
        smoothed = savgol_filter(trace, window_samples, polyorder)
    else:
        # Interpolate over NaNs, smooth, then restore NaNs
        trace_interp = trace.copy()
        nans = ~valid_mask
        if np.any(nans) and np.any(valid_mask):
            trace_interp[nans] = np.interp(
                np.flatnonzero(nans), 
                np.flatnonzero(valid_mask), 
                trace[valid_mask]
            )
        smoothed = savgol_filter(trace_interp, window_samples, polyorder)
        smoothed[nans] = np.nan
    
    return smoothed


# ---------------------------------------------------------------------
# 2D METRICS COMPUTATION
# ---------------------------------------------------------------------

def compute_2d_metrics(x: np.ndarray, y: np.ndarray, dt: float) -> dict:
    """Compute 2D metrics from X and Y position."""
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Empty position arrays")
    
    if len(x) != len(y):
        raise ValueError(f"X and Y length mismatch: {len(x)} vs {len(y)}")
    
    x_valid = np.sum(np.isfinite(x))
    y_valid = np.sum(np.isfinite(y))
    if x_valid < 10 or y_valid < 10:
        raise ValueError(f"Insufficient valid data points: X={x_valid}, Y={y_valid}")
    
    x_smooth = smooth_1d(x, dt, window_ms=SAVGOL_WINDOW_MS)
    y_smooth = smooth_1d(y, dt, window_ms=SAVGOL_WINDOW_MS)
    
    vx = np.gradient(x_smooth, dt)
    vy = np.gradient(y_smooth, dt)
    speed_2d = np.sqrt(vx**2 + vy**2)
    
    return {
        'x_smooth': x_smooth,
        'y_smooth': y_smooth,
        'speed_2d': speed_2d,
    }


# ---------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------

def normalize_distance_traces(distance_all: np.ndarray, 
                              method: str = 'max') -> Tuple[np.ndarray, np.ndarray]:
    """Normalize distance traces to [0, 1] range."""
    n_trials, T = distance_all.shape
    normalized = np.zeros_like(distance_all)
    scale_factors = np.zeros(n_trials)
    
    for i in range(n_trials):
        trace = distance_all[i, :]
        
        if method == 'max':
            scale = np.nanmax(trace)
        elif method == 'percentile':
            scale = np.nanpercentile(trace, 95)
        else:
            scale = np.nanmax(trace)
        
        scale = max(scale, 1e-6)
        scale_factors[i] = scale
        normalized[i, :] = trace / scale
    
    return normalized, scale_factors


# ---------------------------------------------------------------------
# PLATEAU DETECTION (MEDIAN-BASED)
# ---------------------------------------------------------------------

def detect_plateau_on_trace(
    dist_trace: np.ndarray, 
    t_axis: np.ndarray,
    onset_idx: int,
    plateau_window_ms: float = PLATEAU_WINDOW_MS,
    min_distance_threshold: float = MIN_DISTANCE_THRESHOLD,
    max_cumulative_slope: float = MAX_CUMULATIVE_SLOPE,
) -> dict:
    """
    Detect plateau by finding windows with minimal accumulated absolute slope.
    """
    dt = t_axis[1] - t_axis[0]
    T = len(dist_trace)
    plateau_samples = int(plateau_window_ms / dt)
    
    if plateau_samples < 2:
        return {'plateau_time': None, 'plateau_idx': None, 'method': 'failed_window_too_small'}
    
    max_dist = np.nanmax(dist_trace[onset_idx:])
    if max_dist < 1e-6:
        return {'plateau_time': None, 'plateau_idx': None, 'method': 'failed_no_movement'}
    
    min_dist_for_plateau = min_distance_threshold * max_dist
    
    diffs = np.diff(dist_trace)
    
    for i in range(onset_idx, T - plateau_samples):
        window = dist_trace[i:i + plateau_samples]
        
        if np.nanmean(window) < min_dist_for_plateau:
            continue
        
        window_diffs = diffs[i:i + plateau_samples - 1]
        cumulative_abs_slope = np.nansum(np.abs(window_diffs))
        
        cumulative_abs_slope_normalized = cumulative_abs_slope / max_dist
        
        if cumulative_abs_slope_normalized < max_cumulative_slope:
            return {
                'plateau_time': t_axis[i],
                'plateau_idx': i,
                'method': 'cumulative_slope',
                'cumulative_slope': cumulative_abs_slope_normalized,
            }
    
    return {'plateau_time': None, 'plateau_idx': None, 'method': 'failed_no_stable_window'}


def detect_plateau_from_median_trace(
    aligned_distance: np.ndarray,
    aligned_speed: np.ndarray, 
    aligned_time: np.ndarray,
    valid_mask: np.ndarray,
    plateau_window_ms: float = PLATEAU_WINDOW_MS,
    min_distance_threshold: float = MIN_DISTANCE_THRESHOLD,
    max_cumulative_slope: float = MAX_CUMULATIVE_SLOPE,
) -> dict:
    """
    Detect plateau using the MEDIAN trace only.
    
    Key: Individual traces are SMOOTHED before computing median.
    """
    n_trials = aligned_distance.shape[0]
    dt = aligned_time[1] - aligned_time[0] if len(aligned_time) > 1 else 10.0
    
    # First smooth each individual trace
    smoothed_distance = np.zeros_like(aligned_distance)
    for i in range(n_trials):
        smoothed_distance[i, :] = smooth_trace(aligned_distance[i, :], dt)
    
    # Normalize the smoothed distance traces
    dist_norm, scale_factors = normalize_distance_traces(smoothed_distance, method='max')
    
    # Normalize speed using same scale factors
    speed_norm = np.zeros_like(aligned_speed)
    for i in range(n_trials):
        sf = scale_factors[i] if scale_factors[i] > 1e-6 else 1.0
        speed_norm[i, :] = aligned_speed[i, :] / sf
    
    # Compute median traces (only from valid trials)
    valid_dist = dist_norm[valid_mask, :]
    valid_speed = speed_norm[valid_mask, :]
    
    if valid_dist.shape[0] == 0:
        return {
            'plateau_time': None,
            'plateau_idx': None,
            'dist_median': np.full(aligned_time.shape, np.nan),
            'speed_median': np.full(aligned_time.shape, np.nan),
            'dist_norm': dist_norm,
            'speed_norm': speed_norm,
            'scale_factors': scale_factors,
            'method': 'no_valid_trials',
            'smoothed_distance': smoothed_distance,
        }
    
    dist_median = np.nanmedian(valid_dist, axis=0)
    speed_median = np.nanmedian(valid_speed, axis=0)
    
    # Find t=0 index in aligned coordinates
    t0_idx_aligned = np.argmin(np.abs(aligned_time))
    
    # Detect plateau on the median trace
    result = detect_plateau_on_trace(
        dist_trace=dist_median,
        t_axis=aligned_time,
        onset_idx=t0_idx_aligned,
        plateau_window_ms=plateau_window_ms,
        min_distance_threshold=min_distance_threshold,
        max_cumulative_slope=max_cumulative_slope,
    )
    
    return {
        'plateau_time': result['plateau_time'],
        'plateau_idx': result['plateau_idx'],
        'dist_median': dist_median,
        'speed_median': speed_median,
        'dist_norm': dist_norm,
        'speed_norm': speed_norm,
        'scale_factors': scale_factors,
        'method': result.get('method', 'unknown'),
        'smoothed_distance': smoothed_distance,
    }


def analyze_condition_with_median_plateau(
    aligned_distance: np.ndarray,
    aligned_speed: np.ndarray, 
    aligned_time: np.ndarray,
    valid_mask: np.ndarray
) -> dict:
    """
    Main analysis function using median-based plateau detection.
    
    Returns the single plateau value from the median trace.
    """
    n_trials = aligned_distance.shape[0]
    n_valid = np.sum(valid_mask)
    
    # Detect plateau from median trace (uses smoothed traces internally)
    median_result = detect_plateau_from_median_trace(
        aligned_distance, aligned_speed, aligned_time, valid_mask
    )
    
    plateau_time = median_result['plateau_time']
    plateau_idx = median_result['plateau_idx']
    dist_norm = median_result['dist_norm']
    speed_norm = median_result['speed_norm']
    dist_median = median_result['dist_median']
    
    # Get the plateau VALUE from the median trace (single value!)
    if plateau_idx is not None:
        # Average over small window for stability
        half_window = PLATEAU_VALUE_WINDOW_SAMPLES // 2
        T = len(dist_median)
        start_idx = max(0, plateau_idx - half_window)
        end_idx = min(T, plateau_idx + half_window + 1)
        plateau_value = np.nanmean(dist_median[start_idx:end_idx])
    else:
        plateau_value = np.nan
    
    # Compute SEM for confidence intervals
    dist_mean = np.nanmean(dist_norm[valid_mask, :], axis=0)
    dist_std = np.nanstd(dist_norm[valid_mask, :], axis=0)
    dist_sem = dist_std / np.sqrt(max(1, n_valid))
    
    # 95% CI = mean ± 1.96 * SEM
    dist_ci_lower = dist_median - 1.96 * dist_sem
    dist_ci_upper = dist_median + 1.96 * dist_sem
    
    summary = {
        # Normalized traces (smoothed)
        'dist_norm': dist_norm,
        'speed_norm': speed_norm,
        'scale_factors': median_result['scale_factors'],
        
        # Median trace (used for plateau detection)
        'dist_median': dist_median,
        'speed_median': median_result['speed_median'],
        
        # Mean and SEM for CI
        'dist_mean': dist_mean,
        'dist_std': dist_std,
        'dist_sem': dist_sem,
        'dist_ci_lower': dist_ci_lower,
        'dist_ci_upper': dist_ci_upper,
        
        # Plateau info (from median trace) - SINGLE VALUE
        'plateau_time': plateau_time,
        'plateau_idx': plateau_idx,
        'plateau_value': plateau_value,  # THE key metric
        'plateau_method': median_result['method'],
        
        # Trial counts
        'n_valid': n_valid,
        'n_total': n_trials,
    }
    
    return summary


# ---------------------------------------------------------------------
# DATA PROCESSING
# ---------------------------------------------------------------------

def process_condition_with_onset_detection(
    pos_segs: np.ndarray,
    t_axis: np.ndarray,
    kp_names: list,
    idx_x: int,
    idx_y: int
) -> dict:
    """
    Process a condition: detect onset, compute metrics, align traces.
    """
    n_trials = pos_segs.shape[0]
    T = pos_segs.shape[2]
    
    dt = t_axis[1] - t_axis[0] if len(t_axis) > 1 else 10.0
    if dt < 0.9:
        dt *= 1000.0
    
    x_all = pos_segs[:, idx_x, :]
    y_all = pos_segs[:, idx_y, :]
    
    speed_all = np.zeros((n_trials, T))
    for i in range(n_trials):
        x = x_all[i, :]
        y = y_all[i, :]
        if np.sum(np.isfinite(x)) >= 10 and np.sum(np.isfinite(y)) >= 10:
            speed_all[i, :] = compute_2d_speed(x, y, dt)
    
    onset_times, onset_indices, onset_results = detect_reach_onset_batch(
        pos_segs=x_all[:, np.newaxis, :],
        speed_segs=speed_all[:, np.newaxis, :],
        time_axis=t_axis,
        config=ONSET_CONFIG,
        keypoint_idx=0
    )
    
    distance_all = np.zeros((n_trials, T))
    for i in range(n_trials):
        x = x_all[i, :]
        y = y_all[i, :]
        
        if onset_indices[i] >= 0:
            ref_idx = onset_indices[i]
        else:
            ref_idx = 0
        
        x0, y0 = x[ref_idx], y[ref_idx]
        
        x_smooth = smooth_1d(x, dt, window_ms=SAVGOL_WINDOW_MS)
        y_smooth = smooth_1d(y, dt, window_ms=SAVGOL_WINDOW_MS)
        
        distance_all[i, :] = np.sqrt((x_smooth - x0)**2 + (y_smooth - y0)**2)
    
    if TIME_ZERO_REFERENCE == 'reach_onset':
        aligned_distance, aligned_time = align_traces_to_onset(
            distance_all, onset_indices, t_axis,
            pre_onset_ms=PRE_ZERO_MS, post_onset_ms=POST_ZERO_MS
        )
        
        aligned_speed, _ = align_traces_to_onset(
            speed_all, onset_indices, t_axis,
            pre_onset_ms=PRE_ZERO_MS, post_onset_ms=POST_ZERO_MS
        )
        
        aligned_x, _ = align_traces_to_onset(
            x_all, onset_indices, t_axis,
            pre_onset_ms=PRE_ZERO_MS, post_onset_ms=POST_ZERO_MS
        )
        
        aligned_y, _ = align_traces_to_onset(
            y_all, onset_indices, t_axis,
            pre_onset_ms=PRE_ZERO_MS, post_onset_ms=POST_ZERO_MS
        )
        
        valid_mask = onset_indices >= 0
        time_zero_label = "Reach Onset"
        
    else:
        t0_idx = np.argmin(np.abs(t_axis - 0))
        pre_samples = int(PRE_ZERO_MS / dt)
        post_samples = int(POST_ZERO_MS / dt)
        
        start_idx = max(0, t0_idx - pre_samples)
        end_idx = min(T, t0_idx + post_samples)
        
        aligned_time = t_axis[start_idx:end_idx] - t_axis[t0_idx]
        aligned_distance = distance_all[:, start_idx:end_idx]
        aligned_speed = speed_all[:, start_idx:end_idx]
        aligned_x = x_all[:, start_idx:end_idx]
        aligned_y = y_all[:, start_idx:end_idx]
        
        valid_mask = np.ones(n_trials, dtype=bool)
        time_zero_label = "Stim Timing / IR Crossing"
    
    return {
        'distance_all': distance_all,
        'speed_all': speed_all,
        'x_all': x_all,
        'y_all': y_all,
        'aligned_distance': aligned_distance,
        'aligned_speed': aligned_speed,
        'aligned_x': aligned_x,
        'aligned_y': aligned_y,
        'aligned_time': aligned_time,
        'onset_times': onset_times,
        'onset_indices': onset_indices,
        'onset_results': onset_results,
        'valid_mask': valid_mask,
        't_original': t_axis,
        'time_zero_label': time_zero_label,
    }


# ---------------------------------------------------------------------
# STATISTICS - UPDATED FOR SINGLE PLATEAU VALUES
# ---------------------------------------------------------------------

def compute_condition_statistics(all_plateau_data: list) -> dict:
    """
    Compute statistics comparing plateau values across conditions.
    
    Since each condition has ONE plateau value (from median trace),
    we use different statistical approaches:
    
    1. Descriptive comparison of plateau values
    2. Bootstrap confidence intervals (if we had raw traces)
    3. Permutation tests (comparing median traces)
    
    For now, we report:
    - Plateau time and value for each condition
    - Difference from control
    - Effect sizes
    """
    results = {
        'comparisons': [],
        'control_data': {},
        'summary_table': [],
    }
    
    for target in ['A', 'B']:
        target_data = [d for d in all_plateau_data if d['target'] == target]
        
        if len(target_data) < 2:
            continue
        
        control_data = [d for d in target_data 
                        if d['condition_info']['condition_type'] == 'control']
        stim_data = [d for d in target_data 
                     if d['condition_info']['condition_type'] != 'control']
        
        if not control_data:
            # No control - just report values
            for d in target_data:
                results['summary_table'].append({
                    'target': target,
                    'br_idx': d['br_idx'],
                    'label': d['condition_info']['label'],
                    'condition_type': d['condition_info']['condition_type'],
                    'plateau_time': d['plateau_time'],
                    'plateau_value': d['plateau_value'],
                    'n_trials': d['n_valid'],
                })
            continue
        
        # Use first control as reference
        control = control_data[0]
        control_plateau_time = control['plateau_time']
        control_plateau_value = control['plateau_value']
        
        results['control_data'][target] = {
            'plateau_time': control_plateau_time,
            'plateau_value': control_plateau_value,
            'n_trials': control['n_valid'],
            'br_idx': control['br_idx'],
        }
        
        # Add control to summary
        results['summary_table'].append({
            'target': target,
            'br_idx': control['br_idx'],
            'label': control['condition_info']['label'],
            'condition_type': 'control',
            'plateau_time': control_plateau_time,
            'plateau_value': control_plateau_value,
            'n_trials': control['n_valid'],
            'time_diff_from_control': 0,
            'value_diff_from_control': 0,
            'value_pct_change': 0,
        })
        
        # Compare each stim condition to control
        for stim_d in stim_data:
            stim_plateau_time = stim_d['plateau_time']
            stim_plateau_value = stim_d['plateau_value']
            
            # Compute differences
            if control_plateau_time is not None and stim_plateau_time is not None:
                time_diff = stim_plateau_time - control_plateau_time
            else:
                time_diff = np.nan
            
            if control_plateau_value is not None and stim_plateau_value is not None:
                value_diff = stim_plateau_value - control_plateau_value
                if control_plateau_value > 0:
                    value_pct_change = (value_diff / control_plateau_value) * 100
                else:
                    value_pct_change = np.nan
            else:
                value_diff = np.nan
                value_pct_change = np.nan
            
            comparison = {
                'target': target,
                'stim_br_idx': stim_d['br_idx'],
                'stim_label': stim_d['condition_info']['label'],
                'condition_type': stim_d['condition_info']['condition_type'],
                'control_plateau_time': control_plateau_time,
                'stim_plateau_time': stim_plateau_time,
                'plateau_time_diff': time_diff,
                'control_plateau_value': control_plateau_value,
                'stim_plateau_value': stim_plateau_value,
                'plateau_value_diff': value_diff,
                'plateau_value_pct_change': value_pct_change,
                'control_n': control['n_valid'],
                'stim_n': stim_d['n_valid'],
            }
            results['comparisons'].append(comparison)
            
            # Add to summary table
            results['summary_table'].append({
                'target': target,
                'br_idx': stim_d['br_idx'],
                'label': stim_d['condition_info']['label'],
                'condition_type': stim_d['condition_info']['condition_type'],
                'plateau_time': stim_plateau_time,
                'plateau_value': stim_plateau_value,
                'n_trials': stim_d['n_valid'],
                'time_diff_from_control': time_diff,
                'value_diff_from_control': value_diff,
                'value_pct_change': value_pct_change,
            })
    
    return results


# ---------------------------------------------------------------------
# PLOTTING: RAW TRACE DIAGNOSTIC
# ---------------------------------------------------------------------

def plot_raw_trace_diagnostic(
    cond_data: dict,
    condition_info: dict,
    target: str
) -> None:
    """Generate diagnostic plot showing raw traces aligned to time zero."""
    aligned_time = cond_data['aligned_time']
    aligned_x = cond_data.get('aligned_x')
    aligned_y = cond_data.get('aligned_y')
    aligned_distance = cond_data['aligned_distance']
    valid_mask = cond_data['valid_mask']
    br_idx = cond_data.get('br_idx', 0)
    time_zero_label = cond_data.get('time_zero_label', 'Time Zero')
    
    n_trials = aligned_distance.shape[0]
    n_valid = np.sum(valid_mask)
    
    if n_valid == 0:
        print(f"    [skip] No valid trials for BR{br_idx} raw trace plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    cmap = plt.colormaps.get_cmap('viridis')
    
    ax1 = axes[0]
    if aligned_x is not None:
        for i in range(n_trials):
            if not valid_mask[i]:
                continue
            color = cmap(i / max(1, n_trials - 1)) if n_trials > 1 else 'blue'
            ax1.plot(aligned_time, aligned_x[i, :], color=color, alpha=0.5, linewidth=0.8)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label=f't=0 ({time_zero_label})')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('X Position')
    ax1.set_title(f'X Position (n={n_valid} trials)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    if aligned_y is not None:
        for i in range(n_trials):
            if not valid_mask[i]:
                continue
            color = cmap(i / max(1, n_trials - 1)) if n_trials > 1 else 'blue'
            ax2.plot(aligned_time, aligned_y[i, :], color=color, alpha=0.5, linewidth=0.8)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label=f't=0 ({time_zero_label})')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Y Position')
    ax2.set_title(f'Y Position (n={n_valid} trials)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    for i in range(n_trials):
        if not valid_mask[i]:
            continue
        color = cmap(i / max(1, n_trials - 1)) if n_trials > 1 else 'blue'
        ax3.plot(aligned_time, aligned_distance[i, :], color=color, alpha=0.5, linewidth=0.8)
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label=f't=0 ({time_zero_label})')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Distance from Start')
    ax3.set_title(f'Distance (n={n_valid} trials)')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle(f"Session: {PARAMS.session} | BR{br_idx}: {condition_info['label']} | Target {target}\n"
                 f"Raw Traces Aligned to {time_zero_label} (TIME_ZERO_REFERENCE='{TIME_ZERO_REFERENCE}')",
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    
    diag_dir = FIG_OUT_DIR / "raw_trace_diagnostics"
    diag_dir.mkdir(exist_ok=True)
    out_filename = f"RawTraces_BR{br_idx:03d}_Target{target}.{OUTPUT_FORMAT}"
    fig.savefig(diag_dir / out_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"    Saved: raw_trace_diagnostics/{out_filename}")


# ---------------------------------------------------------------------
# PLOTTING: PER-CONDITION DIAGNOSTIC (UPDATED)
# ---------------------------------------------------------------------

def plot_condition_diagnostic_v3(
    cond_data: dict,
    condition_info: dict,
    target: str
) -> None:
    """
    Generate diagnostic plot using MEDIAN-BASED plateau detection.
    
    Shows:
    - All individual trial traces (smoothed, normalized)
    - Median trace (thick black)
    - 95% confidence interval (SEM-based shading)
    - Single plateau marker on median trace only
    """
    aligned_time = cond_data['aligned_time']
    aligned_distance = cond_data['aligned_distance']
    aligned_speed = cond_data['aligned_speed']
    valid_mask = cond_data['valid_mask']
    br_idx = cond_data.get('br_idx', 0)
    time_zero_label = cond_data.get('time_zero_label', 'Time Zero')
    
    n_trials = aligned_distance.shape[0]
    n_valid = np.sum(valid_mask)
    
    if n_valid == 0:
        print(f"    [skip] No valid trials for BR{br_idx}")
        return
    
    # Use median-based plateau detection
    summary = analyze_condition_with_median_plateau(
        aligned_distance, aligned_speed, aligned_time, valid_mask
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    cmap = plt.colormaps.get_cmap('tab10')
    
    # Plot individual trials (smoothed, normalized) - thin lines
    for i in range(n_trials):
        if not valid_mask[i]:
            continue
        
        color = cmap(i % 10)
        ax.plot(aligned_time, summary['dist_norm'][i, :], 
                color=color, alpha=0.3, linewidth=0.8)
    
    # 95% CI shading (based on SEM)
    ax.fill_between(aligned_time,
                    summary['dist_ci_lower'],
                    summary['dist_ci_upper'],
                    color='gray', alpha=0.3, label='95% CI (SEM)')
    
    # Median trace (thick black line)
    ax.plot(aligned_time, summary['dist_median'], color='black', linewidth=2.5, 
            label=f'Median (n={n_valid})')
    
    # Plateau marker on median trace ONLY (single point!)
    plateau_time = summary['plateau_time']
    plateau_idx = summary['plateau_idx']
    plateau_value = summary['plateau_value']
    
    if plateau_time is not None and plateau_idx is not None:
        # Vertical line at plateau time
        ax.axvline(plateau_time, color='red', linestyle=':', linewidth=2, alpha=0.7)
        
        # Diamond marker on median trace
        ax.scatter(plateau_time, summary['dist_median'][plateau_idx],
                   color='red', s=200, marker='D', zorder=10,
                   edgecolors='black', linewidths=2,
                   label=f"Plateau: {plateau_time:.0f}ms, val={plateau_value:.2f}")
    
    # t=0 marker
    ax.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.8, 
               label=f't=0 ({time_zero_label})')
    
    ax.set_xlabel('Time from t=0 (ms)', fontsize=12)
    ax.set_ylabel('Normalized Distance [0-1]', fontsize=12)
    
    plateau_str = f'{plateau_time:.0f}ms (val={plateau_value:.2f})' if plateau_time is not None else 'not detected'
    method_str = summary['plateau_method']
    ax.set_title(f'BR{br_idx}: {condition_info["label"]} | Target {target}\n'
                 f'Median Plateau: {plateau_str} (method: {method_str}) | n={n_valid} trials',
                 fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(-0.1, 1.3)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Session: {PARAMS.session} | v3 (Median-Based Detection)", fontsize=10, y=0.98)
    fig.tight_layout()
    
    # Save
    diag_dir = FIG_OUT_DIR / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    out_filename = f"Diagnostic_BR{br_idx:03d}_Target{target}.{OUTPUT_FORMAT}"
    fig.savefig(diag_dir / out_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"    Saved: diagnostics/{out_filename}")


# ---------------------------------------------------------------------
# PLOTTING: MASTER SUMMARY
# ---------------------------------------------------------------------

def plot_master_summary(
    all_conditions_data: list,
    metadata_df: pd.DataFrame,
    target: str,
    conditions_to_plot: list = None,
    title_suffix: str = ""
) -> list:
    """
    Plot median traces for all/selected conditions with 95% CI on same axes.
    """
    if not all_conditions_data:
        return []
    
    if conditions_to_plot:
        filtered_data = [d for d in all_conditions_data
                         if d.get('br_idx') in conditions_to_plot]
        if not filtered_data:
            print(f"[warn] No conditions matched filter {conditions_to_plot}")
            return []
        all_conditions_data = filtered_data
    
    all_conditions_data = sorted(all_conditions_data, key=lambda x: x.get('br_idx', 0) or 0)
    
    n_conds = len(all_conditions_data)
    cmap = plt.colormaps.get_cmap('tab10')
    
    fig = Figure(figsize=(14, 8))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    
    plateau_results = []
    time_zero_label = "Time Zero"
    
    for i, cond_data in enumerate(all_conditions_data):
        aligned_time = cond_data['aligned_time']
        aligned_distance = cond_data['aligned_distance']
        aligned_speed = cond_data['aligned_speed']
        valid_mask = cond_data['valid_mask']
        br_idx = cond_data.get('br_idx', i)
        time_zero_label = cond_data.get('time_zero_label', 'Time Zero')
        
        condition_info = get_condition_info(br_idx, metadata_df)
        color = cmap(i % 10)
        n_valid = np.sum(valid_mask)
        
        if n_valid == 0:
            continue
        
        # Use median-based analysis
        summary = analyze_condition_with_median_plateau(
            aligned_distance, aligned_speed, aligned_time, valid_mask
        )
        
        plateau_time = summary['plateau_time']
        plateau_value = summary['plateau_value']
        
        # Label
        label = f"{condition_info['label']} (n={n_valid})"
        
        # Plot median ± 95% CI
        ax.plot(aligned_time, summary['dist_median'], color=color, linewidth=2, label=label)
        ax.fill_between(aligned_time,
                        summary['dist_ci_lower'],
                        summary['dist_ci_upper'],
                        color=color, alpha=0.15)
        
        # Plateau marker (only on median trace)
        if plateau_time is not None and summary['plateau_idx'] is not None:
            pt_idx = summary['plateau_idx']
            pt_y = summary['dist_median'][pt_idx]
            
            ax.axvline(plateau_time, color=color, linestyle=':', alpha=0.5, linewidth=1.5)
            ax.scatter(plateau_time, pt_y, 
                       color=color, s=120, marker='D',
                       edgecolors='black', linewidths=1.5, zorder=10)
        
        # Store results
        plateau_results.append({
            'br_idx': br_idx,
            'target': target,
            'condition_info': condition_info,
            'plateau_time': plateau_time,
            'plateau_value': plateau_value,
            'plateau_method': summary['plateau_method'],
            'n_valid': n_valid,
            'n_total': aligned_distance.shape[0],
        })
    
    # Formatting
    ax.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.8, 
               label=f't=0 ({time_zero_label})')
    ax.set_xlabel('Time from t=0 (ms)', fontsize=12)
    ax.set_ylabel('Normalized Distance [0-1]', fontsize=12)
    ax.set_title(f'Target {target}: Normalized Distance{title_suffix}\n'
                 f'Median ± 95% CI (SEM) | ◆ = median plateau', fontsize=12)
    ax.legend(fontsize='small', loc='upper left', ncol=2)
    ax.set_ylim(0, 1.3)
    ax.set_xlim(aligned_time.min(), aligned_time.max())
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Session: {PARAMS.session} | v3 (Median-Based) | TIME_ZERO_REFERENCE='{TIME_ZERO_REFERENCE}'", 
                 fontsize=10, y=0.98)
    fig.tight_layout()
    
    # Save
    if conditions_to_plot:
        cond_suffix = "_BR" + "_".join(str(c) for c in sorted(conditions_to_plot))
    else:
        cond_suffix = "_ALL"
    out_filename = f"Target_{target}{cond_suffix}_Summary.{OUTPUT_FORMAT}"
    fig.savefig(FIG_OUT_DIR / out_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {out_filename}")
    
    return plateau_results


# ---------------------------------------------------------------------
# PLOTTING: BAR CHARTS FOR PLATEAU COMPARISON
# ---------------------------------------------------------------------

def plot_plateau_bar_charts(
    all_plateau_data: list,
    stats_results: dict,
    title_suffix: str = ""
) -> None:
    """
    Create bar charts comparing plateau TIME and plateau VALUE across conditions.
    
    Since each condition has ONE plateau value from its median trace,
    bar charts are appropriate.
    """
    if not all_plateau_data:
        return
    
    for target in ['A', 'B']:
        target_data = [d for d in all_plateau_data if d['target'] == target]
        if not target_data:
            continue
        
        # Sort: controls first, then stim by BR index
        controls = [d for d in target_data if d['condition_info']['condition_type'] == 'control']
        stims = [d for d in target_data if d['condition_info']['condition_type'] != 'control']
        stims = sorted(stims, key=lambda x: x['br_idx'])
        sorted_data = controls + stims
        
        n_conditions = len(sorted_data)
        if n_conditions == 0:
            continue
        
        # Create figure with two subplots: plateau time and plateau value
        fig_width = max(10, n_conditions * 1.2 + 2)
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, 6))
        
        labels = []
        times = []
        values = []
        colors = []
        n_trials_list = []
        
        for d in sorted_data:
            cinfo = d['condition_info']
            
            if cinfo['condition_type'] == 'control':
                labels.append(f"Control\n(BR{d['br_idx']})")
                colors.append('lightgreen')
            elif cinfo['condition_type'] == 'continuous_stim':
                labels.append(f"{cinfo['label']}\n(BR{d['br_idx']})")
                colors.append('lightsalmon')
            else:
                labels.append(f"{cinfo['label']}\n(BR{d['br_idx']})")
                colors.append('steelblue')
            
            times.append(d['plateau_time'] if d['plateau_time'] is not None else 0)
            values.append(d['plateau_value'] if d['plateau_value'] is not None else 0)
            n_trials_list.append(d['n_valid'])
        
        x_pos = np.arange(len(labels))
        
        # --- Plot 1: Plateau TIME ---
        ax1 = axes[0]
        bars1 = ax1.bar(x_pos, times, color=colors, edgecolor='black', alpha=0.7)
        
        # Add value labels
        for i, (bar, t, n) in enumerate(zip(bars1, times, n_trials_list)):
            if t > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{t:.0f}ms\n(n={n})', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Plateau Time from t=0 (ms)', fontsize=11)
        ax1.set_title(f'Plateau Time', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(0, color='green', linestyle='--', alpha=0.5)
        
        # --- Plot 2: Plateau VALUE ---
        ax2 = axes[1]
        bars2 = ax2.bar(x_pos, values, color=colors, edgecolor='black', alpha=0.7)
        
        # Add value labels and difference from control
        control_value = values[0] if controls else None
        for i, (bar, v, n) in enumerate(zip(bars2, values, n_trials_list)):
            label_text = f'{v:.2f}\n(n={n})'
            if control_value is not None and i > 0 and v > 0:
                diff = v - control_value
                pct = (diff / control_value * 100) if control_value > 0 else 0
                label_text += f'\n({pct:+.1f}%)'
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    label_text, ha='center', va='bottom', fontsize=8)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Normalized Distance at Plateau', fontsize=11)
        ax2.set_title(f'Plateau Value (from Median Trace)', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Full reach (1.0)')
        ax2.set_ylim(0, min(1.5, max(values) * 1.3) if values else 1.5)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', alpha=0.7, edgecolor='black', label='Control'),
            Patch(facecolor='steelblue', alpha=0.7, edgecolor='black', label='Pulsed Stim'),
            Patch(facecolor='lightsalmon', alpha=0.7, edgecolor='black', label='Continuous Stim'),
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        fig.suptitle(f'Target {target}: Median-Detected Plateau Comparison{title_suffix}', 
                     fontsize=12, fontweight='bold')
        fig.tight_layout()
        
        suffix = title_suffix.replace(' ', '_').replace(':', '')
        out_filename = f"PlateauBars_Target{target}{suffix}.{OUTPUT_FORMAT}"
        fig.savefig(FIG_OUT_DIR / out_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {out_filename}")


def plot_plateau_comparison_combined(
    all_plateau_data: list,
    stats_results: dict,
    title_suffix: str = ""
) -> None:
    """
    Create a combined comparison plot showing all targets and conditions.
    """
    if not all_plateau_data:
        return
    
    # Get unique targets
    targets = sorted(set(d['target'] for d in all_plateau_data))
    
    if len(targets) == 0:
        return
    
    # Create figure
    fig, axes = plt.subplots(len(targets), 2, figsize=(14, 5 * len(targets)))
    if len(targets) == 1:
        axes = axes.reshape(1, -1)
    
    for t_idx, target in enumerate(targets):
        target_data = [d for d in all_plateau_data if d['target'] == target]
        
        # Sort
        controls = [d for d in target_data if d['condition_info']['condition_type'] == 'control']
        stims = [d for d in target_data if d['condition_info']['condition_type'] != 'control']
        stims = sorted(stims, key=lambda x: x['br_idx'])
        sorted_data = controls + stims
        
        if not sorted_data:
            continue
        
        labels = []
        times = []
        values = []
        colors = []
        
        for d in sorted_data:
            cinfo = d['condition_info']
            short_label = f"BR{d['br_idx']}"
            
            if cinfo['condition_type'] == 'control':
                colors.append('lightgreen')
            elif cinfo['condition_type'] == 'continuous_stim':
                colors.append('lightsalmon')
            else:
                colors.append('steelblue')
            
            labels.append(short_label)
            times.append(d['plateau_time'] if d['plateau_time'] is not None else np.nan)
            values.append(d['plateau_value'] if d['plateau_value'] is not None else np.nan)
        
        x_pos = np.arange(len(labels))
        
        # Time subplot
        ax_time = axes[t_idx, 0]
        valid_times = [t if not np.isnan(t) else 0 for t in times]
        bars = ax_time.bar(x_pos, valid_times, color=colors, edgecolor='black', alpha=0.7)
        ax_time.set_xticks(x_pos)
        ax_time.set_xticklabels(labels, fontsize=9)
        ax_time.set_ylabel('Plateau Time (ms)')
        ax_time.set_title(f'Target {target}: Plateau Time')
        ax_time.grid(True, alpha=0.3, axis='y')
        
        # Value subplot
        ax_val = axes[t_idx, 1]
        valid_values = [v if not np.isnan(v) else 0 for v in values]
        bars = ax_val.bar(x_pos, valid_values, color=colors, edgecolor='black', alpha=0.7)
        ax_val.set_xticks(x_pos)
        ax_val.set_xticklabels(labels, fontsize=9)
        ax_val.set_ylabel('Plateau Value')
        ax_val.set_title(f'Target {target}: Plateau Value')
        ax_val.grid(True, alpha=0.3, axis='y')
        ax_val.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax_val.set_ylim(0, 1.3)
    
    fig.suptitle(f'Plateau Comparison (Median-Based){title_suffix}', fontsize=12, fontweight='bold')
    fig.tight_layout()
    
    suffix = title_suffix.replace(' ', '_').replace(':', '')
    out_filename = f"PlateauComparison_Combined{suffix}.{OUTPUT_FORMAT}"
    fig.savefig(FIG_OUT_DIR / out_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {out_filename}")


# ---------------------------------------------------------------------
# SAVE STATISTICS
# ---------------------------------------------------------------------

def save_statistics_csv(all_plateau_data: list, stats_results: dict, title_suffix: str = "") -> None:
    """Save summary statistics to CSV."""
    rows = []
    for d in all_plateau_data:
        cinfo = d['condition_info']
        rows.append({
            'session': PARAMS.session,
            'target': d['target'],
            'br_idx': d['br_idx'],
            'condition_type': cinfo['condition_type'],
            'condition_label': cinfo['label'],
            'n_trials': d['n_valid'],
            'n_trials_total': d['n_total'],
            'plateau_time_ms': d['plateau_time'],
            'plateau_value': d['plateau_value'],
            'plateau_method': d['plateau_method'],
            'time_zero_reference': TIME_ZERO_REFERENCE,
        })
    
    df = pd.DataFrame(rows)
    suffix = title_suffix.replace(' ', '_').replace(':', '')
    csv_filename = FIG_OUT_DIR / f"Plateau_Summary{suffix}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"  Saved: {csv_filename}")
    
    # Save comparison table if available
    if stats_results and stats_results['summary_table']:
        df_summary = pd.DataFrame(stats_results['summary_table'])
        csv_summary_filename = FIG_OUT_DIR / f"Plateau_ComparisonTable{suffix}.csv"
        df_summary.to_csv(csv_summary_filename, index=False)
        print(f"  Saved: {csv_summary_filename}")
    
    if stats_results and stats_results['comparisons']:
        df_comp = pd.DataFrame(stats_results['comparisons'])
        csv_comp_filename = FIG_OUT_DIR / f"Plateau_ControlVsStim{suffix}.csv"
        df_comp.to_csv(csv_comp_filename, index=False)
        print(f"  Saved: {csv_comp_filename}")


# ---------------------------------------------------------------------
# FILE PROCESSING
# ---------------------------------------------------------------------

def process_single_file(p: Path, target: str, target_code: int, metadata_df: pd.DataFrame) -> Optional[dict]:
    """Process a single file and return processed data."""
    try:
        with np.load(p, allow_pickle=True) as data:
            if p.name.startswith("baseline__"):
                cond = "Baseline"
                br_idx = None
            else:
                if "overall_title" in data:
                    raw_title = data["overall_title"]
                    cond = str(raw_title.item()) if raw_title.shape == () else str(raw_title)
                else:
                    parts = p.name.split("_")
                    cond = f"Cond_{parts[1]}" if len(parts) > 1 and parts[1].isdigit() else "Stim"
                
                br_idx = get_br_from_condition_string(cond)
                if br_idx is None:
                    br_match = re.search(r"BR_(\d+)", p.name)
                    if br_match:
                        br_idx = int(br_match.group(1))
            
            if BR_FILTER is not None and br_idx is not None:
                if br_idx not in BR_FILTER:
                    return None
            
            if br_idx is not None:
                condition_info = get_condition_info(br_idx, metadata_df)
                if condition_info.get('is_at_rest', False):
                    print(f"    [skip] BR{br_idx} is 'at rest' condition - skipping")
                    return None
            
            pos_key = None
            for key_pattern in [f"beh_{CAMERA_TO_USE}_segs", f"{CAMERA_TO_USE}_segs", "segs"]:
                if key_pattern in data:
                    pos_key = key_pattern
                    break
            
            if pos_key is None:
                return None
            
            pos_segs = data[pos_key]
            
            t_axis = None
            for t_key in ["beh_rel_t", "rel_t", "t"]:
                if t_key in data:
                    t_axis = data[t_key]
                    break
            
            if t_axis is None:
                return None
            
            kp_names = _get_keypoint_names_from_data(data, CAMERA_TO_USE)
            if not kp_names:
                kp_names = [f"kp_{i}" for i in range(pos_segs.shape[1])]
            
            idx_x = find_keypoint_index(kp_names, f"{KEYPOINT_BASE}_x")
            idx_y = find_keypoint_index(kp_names, f"{KEYPOINT_BASE}_y")
            
            if idx_x is None or idx_y is None:
                return None
            
            result = process_condition_with_onset_detection(
                pos_segs, t_axis, kp_names, idx_x, idx_y
            )
            
            result['file'] = p.name
            result['cond'] = cond
            result['br_idx'] = br_idx
            result['n_trials'] = pos_segs.shape[0]
            
            return result
    
    except Exception as e:
        print(f"Error processing {p.name}: {e}")
        return None


def process_target(target: str, target_code: int, metadata_df: pd.DataFrame) -> list:
    """Process all files for a given target."""
    candidate_subdirs = [
        Path("stim_reaches") / f"target_{target}",
        Path("control_reaches") / f"target_{target}",
        Path("continuous_stim") / f"target_{target}",
        f"Target_{target}",
        f"target_{target}",
        Path(""),
    ]
    
    found_files_raw = []
    found_dirs = []
    
    for sub in candidate_subdirs:
        d = PERI_ROOT / sub
        if d.exists():
            found_dirs.append(d)
            files = sorted(d.glob("*.npz"))
            if sub == Path(""):
                files = sorted(d.glob(f"*target_{target}*.npz"))
                files.extend(sorted(d.glob(f"*Target_{target}*.npz")))
            found_files_raw.extend(files)
    
    if not found_files_raw:
        print(f"Skipping {target}: No matching .npz files found")
        return []
    
    print(f"\nProcessing Target {target}:")
    for d in found_dirs:
        print(f"  - {d}")
    
    all_files_raw = sorted(list(set(found_files_raw)))
    all_files = []
    processed_ids = set()
    
    for p in all_files_raw:
        kid = get_kinematics_id(p.name)
        if kid in processed_ids:
            continue
        processed_ids.add(kid)
        all_files.append(p)
    
    print(f"Found {len(all_files_raw)} total files. Processing {len(all_files)} unique sessions.")
    
    all_conditions_data = []
    
    for p in all_files:
        res = process_single_file(p, target, target_code, metadata_df)
        if res:
            all_conditions_data.append(res)
    
    return all_conditions_data


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print(f"=" * 70)
    print(f"Plateau Detection Analysis - VERSION 3 (Median-Based)")
    print(f"=" * 70)
    print(f"Session: {PARAMS.session}")
    print(f"Output Dir: {FIG_OUT_DIR}")
    print(f"Metadata CSV: {METADATA_CSV}")
    print(f"Keypoint: {KEYPOINT_BASE}")
    print(f"Control BR indices: {CONTROL_BR_INDICES}")
    print(f"TIME_ZERO_REFERENCE: {TIME_ZERO_REFERENCE}")
    print(f"")
    print(f"V3 Key Features:")
    print(f"  - Individual traces smoothed before computing median")
    print(f"  - Plateau detected on MEDIAN trace (not per-trial)")
    print(f"  - Single plateau time and value per condition")
    print(f"  - Diagnostics show: individual traces + median + 95% CI (SEM)")
    print(f"  - Bar charts for comparing plateau values across conditions")
    print(f"")
    
    try:
        metadata_df = load_metadata_csv(METADATA_CSV)
        print(f"Loaded metadata: {len(metadata_df)} rows")
    except Exception as e:
        print(f"[warn] Could not load metadata: {e}")
        metadata_df = pd.DataFrame()
    
    target_A_data = process_target("A", 1, metadata_df)
    target_B_data = process_target("B", 2, metadata_df)
    
    if GENERATE_RAW_TRACE_DIAGNOSTICS:
        print("\nGenerating raw trace diagnostic plots...")
        for cond_data in target_A_data:
            cinfo = get_condition_info(cond_data['br_idx'], metadata_df)
            plot_raw_trace_diagnostic(cond_data, cinfo, "A")
        for cond_data in target_B_data:
            cinfo = get_condition_info(cond_data['br_idx'], metadata_df)
            plot_raw_trace_diagnostic(cond_data, cinfo, "B")
    
    if GENERATE_CONDITION_DIAGNOSTICS:
        print("\nGenerating per-condition diagnostic plots (median-based)...")
        for cond_data in target_A_data:
            cinfo = get_condition_info(cond_data['br_idx'], metadata_df)
            plot_condition_diagnostic_v3(cond_data, cinfo, "A")
        for cond_data in target_B_data:
            cinfo = get_condition_info(cond_data['br_idx'], metadata_df)
            plot_condition_diagnostic_v3(cond_data, cinfo, "B")
    
    # Master plot (all conditions)
    all_plateau_data_master = []
    if GENERATE_MASTER_PLOT:
        print("\nGenerating MASTER plots (all conditions)...")
        if target_A_data:
            plateau_A = plot_master_summary(
                target_A_data, metadata_df, "A",
                conditions_to_plot=None, title_suffix=" (All Conditions)"
            )
            all_plateau_data_master.extend(plateau_A)
        if target_B_data:
            plateau_B = plot_master_summary(
                target_B_data, metadata_df, "B",
                conditions_to_plot=None, title_suffix=" (All Conditions)"
            )
            all_plateau_data_master.extend(plateau_B)
        
        if all_plateau_data_master:
            print("\nComputing statistics (all conditions)...")
            stats_master = compute_condition_statistics(all_plateau_data_master)
            plot_plateau_bar_charts(all_plateau_data_master, stats_master, "_ALL")
            plot_plateau_comparison_combined(all_plateau_data_master, stats_master, "_ALL")
            if GENERATE_STATS_SUMMARY:
                save_statistics_csv(all_plateau_data_master, stats_master, "_ALL")
    
    # Subset plot (selected conditions)
    if GENERATE_SUBSET_PLOT and SUBSET_CONDITIONS:
        print(f"\nGenerating SUBSET plots (conditions: {SUBSET_CONDITIONS})...")
        all_plateau_data_subset = []
        if target_A_data:
            plateau_A = plot_master_summary(
                target_A_data, metadata_df, "A",
                conditions_to_plot=SUBSET_CONDITIONS, title_suffix=" (Subset)"
            )
            all_plateau_data_subset.extend(plateau_A)
        if target_B_data:
            plateau_B = plot_master_summary(
                target_B_data, metadata_df, "B",
                conditions_to_plot=SUBSET_CONDITIONS, title_suffix=" (Subset)"
            )
            all_plateau_data_subset.extend(plateau_B)
        
        if all_plateau_data_subset:
            print("\nComputing statistics (subset)...")
            stats_subset = compute_condition_statistics(all_plateau_data_subset)
            cond_str = "_BR" + "_".join(str(c) for c in sorted(SUBSET_CONDITIONS))
            plot_plateau_bar_charts(all_plateau_data_subset, stats_subset, cond_str)
            plot_plateau_comparison_combined(all_plateau_data_subset, stats_subset, cond_str)
            if GENERATE_STATS_SUMMARY:
                save_statistics_csv(all_plateau_data_subset, stats_subset, cond_str)
    
    print(f"\n{'=' * 70}")
    print(f"Done! Figures saved to {FIG_OUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()