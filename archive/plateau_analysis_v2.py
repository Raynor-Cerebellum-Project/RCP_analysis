"""
plateau_analysis_v2.py

Analyze normalized hand position to detect reach endpoints (plateaus).
Updated to use reach onset detection and generate improved figures.

Key Changes from v1:
  1. Uses reach onset detection (from reach_onset_detection module) as reference point
  2. Per-condition diagnostic plots showing all trials + mean + plateau markers
  3. Master summary plots with SEM shading
  4. Statistical comparison: Control vs each Stim condition

Outputs:
  - Per-condition diagnostic plots (all trials + mean + plateaus)
  - Raw trace alignment diagnostic plots
  - Master/subset summary plots (mean ± SEM)
  - Box plots of plateau times with statistics
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

FIG_OUT_DIR = OUT_BASE / "figures" / "plateau_analysis_v2"
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FORMAT = 'png'
plt.rcParams.update({'font.size': 10})

# ---------------------------------------------------------------------
# TIME REFERENCE CONFIGURATION
# ---------------------------------------------------------------------
# Set this to control what t=0 means in all analyses
# Options: 'reach_onset' or 'stim_timing' (IR crossing / stim delivery)
TIME_ZERO_REFERENCE = 'stim_timing'  # Change to 'stim_timing' to use original IR crossing

# ---------------------------------------------------------------------
# PLOT CONTROL
# ---------------------------------------------------------------------
GENERATE_CONDITION_DIAGNOSTICS = True    # Per-condition: all trials + mean + plateaus
GENERATE_RAW_TRACE_DIAGNOSTICS = False    # Per-condition: raw traces aligned to t=0
GENERATE_MASTER_PLOT = False              # All conditions on one plot
GENERATE_SUBSET_PLOT = False              # Selected conditions only
GENERATE_STATS_SUMMARY = False

# Condition filtering for subset plot
SUBSET_CONDITIONS = [3, 9, 10, 11, 12 ]  # Set to None to skip subset plot

# ---------------------------------------------------------------------
# CONTROL CONDITION SPECIFICATION
# ---------------------------------------------------------------------
CONTROL_BR_INDICES = [3]  # BR indices that are controls

# ---------------------------------------------------------------------
# REACH ONSET DETECTION CONFIG
# ---------------------------------------------------------------------
ONSET_CONFIG = ReachOnsetConfig(
    start_pos_max_thresh=0.2,      # Position must be within ±0.2 of zero
    low_speed_thresh_ratio=0.05,   # 5% of peak speed
    stable_window_ms=20.0,         # Speed must stay low for 20ms
    max_reach_duration_ms=800.0,
    min_reach_duration_ms=100.0,
)

# Time windows relative to time zero reference
PRE_ZERO_MS = 500.0   # Time before t=0 to include in plots
POST_ZERO_MS = 500.0  # Time after t=0 to include

# ---------------------------------------------------------------------
# SMOOTHING PARAMETERS
# ---------------------------------------------------------------------
SMOOTHING_METHOD = 'savgol'
SAVGOL_WINDOW_MS = 60.0
SAVGOL_POLYORDER = 4

# ---------------------------------------------------------------------
# PLATEAU DETECTION PARAMETERS (relative to onset)
# ---------------------------------------------------------------------
PLATEAU_WINDOW_MS = 100.0          # Window to verify stable plateau
SPEED_THRESHOLD_FACTOR = 0.10     # Speed < 15% of peak = plateau candidate
POSITION_CHANGE_THRESHOLD = 0.10  # Position change < 0.15 in window = stable

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

# Metadata path construction (same as original)
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
    
    # Check for "at rest" condition
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
    
    # Compute velocity components
    vx = np.gradient(x_smooth, dt)
    vy = np.gradient(y_smooth, dt)
    speed_2d = np.sqrt(vx**2 + vy**2)
    
    return {
        'x_smooth': x_smooth,
        'y_smooth': y_smooth,
        'speed_2d': speed_2d,
    }


# ---------------------------------------------------------------------
# NORMALIZATION AND PLATEAU DETECTION
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

# def detect_plateau_single_trial(
#     dist_trace: np.ndarray, 
#     speed_trace: np.ndarray, 
#     t_axis: np.ndarray,
#     onset_idx: int,
#     plateau_window_ms: float = PLATEAU_WINDOW_MS,
#     speed_threshold_factor: float = SPEED_THRESHOLD_FACTOR,
#     position_change_threshold: float = POSITION_CHANGE_THRESHOLD
# ) -> dict:
    """
    Detect plateau ONSET time for a single trial.
    
    Strategy: Find peak distance, then scan BACKWARDS to find when 
    the hand first entered the plateau region.
    """
    dt = t_axis[1] - t_axis[0]
    T = len(dist_trace)
    
    # Define search window
    search_start = onset_idx
    search_end = min(T, onset_idx + int(POST_ZERO_MS / dt))
    
    if search_end <= search_start:
        return {'plateau_time': None, 'plateau_idx': None, 'method': 'failed'}
    
    # --- Find peak distance ---
    search_dist = dist_trace[search_start:search_end]
    peak_rel_idx = np.nanargmax(search_dist)
    peak_idx = search_start + peak_rel_idx
    peak_time = t_axis[peak_idx]
    peak_dist = dist_trace[peak_idx]
    
    # --- Calculate thresholds ---
    # Peak speed during reach phase
    reach_speed = speed_trace[onset_idx:peak_idx+1] if peak_idx > onset_idx else speed_trace[search_start:search_end]
    peak_speed = np.nanmax(reach_speed) if len(reach_speed) > 0 else np.nanmax(speed_trace[search_start:search_end])
    speed_threshold = speed_threshold_factor * peak_speed
    
    # Plateau level (e.g., 90% of peak distance)
    plateau_level_threshold = 0.90
    plateau_level = plateau_level_threshold * peak_dist
    
    # Plateau window in samples
    plateau_samples = max(1, int(plateau_window_ms / dt))
    
    plateau_onset_idx = None
    method = None
    
    # =========================================================================
    # METHOD 1: Scan BACKWARDS from peak to find where plateau began
    # =========================================================================
    # Strategy: Find the earliest point where:
    #   - Distance is already at plateau level (>= 90% of peak)
    #   - Speed is low (hand is not still moving fast)
    
    # First, find the "stable plateau region" near the peak
    # This is where speed is consistently low
    plateau_region_start = None
    
    # Scan backwards from peak
    for i in range(peak_idx, onset_idx, -1):
        # Check if we're still in the plateau region
        at_plateau_level = dist_trace[i] >= plateau_level
        low_speed = speed_trace[i] <= speed_threshold
        
        if at_plateau_level and low_speed:
            # This point is part of the plateau - keep going back
            plateau_region_start = i
        elif plateau_region_start is not None:
            # We've exited the plateau region going backwards
            # The plateau started at plateau_region_start
            break
    
    if plateau_region_start is not None:
        plateau_onset_idx = plateau_region_start
        method = 'backward_scan'
    
    # =========================================================================
    # METHOD 2 (fallback): Find where speed first drops below threshold
    # =========================================================================
    if plateau_onset_idx is None:
        # Scan forward but look for speed drop, not position
        for i in range(onset_idx, search_end):
            if speed_trace[i] <= speed_threshold and dist_trace[i] >= plateau_level * 0.80:
                plateau_onset_idx = i
                method = 'speed_drop'
                break
    
    # =========================================================================
    # METHOD 3 (fallback): Find peak deceleration (when hand starts braking hard)
    # =========================================================================
    if plateau_onset_idx is None:
        accel = np.gradient(speed_trace, dt)
        
        # Find where deceleration is strongest (most negative acceleration)
        search_accel = accel[onset_idx:peak_idx+1]
        if len(search_accel) > 0:
            max_decel_rel_idx = np.nanargmin(search_accel)  # Most negative = strongest braking
            max_decel_idx = onset_idx + max_decel_rel_idx
            
            # The plateau "begins" shortly after max deceleration
            # when acceleration returns toward zero
            for i in range(max_decel_idx, peak_idx):
                if accel[i] >= -0.0005 and dist_trace[i] >= plateau_level * 0.85:
                    plateau_onset_idx = i
                    method = 'deceleration_end'
                    break
    
    # =========================================================================
    # METHOD 4 (last resort): Original stable window check
    # =========================================================================
    if plateau_onset_idx is None:
        for i in range(onset_idx, min(T - plateau_samples, search_end)):
            current_speed = speed_trace[i]
            speed_ok = current_speed <= speed_threshold
            
            window_slice = slice(i, i + plateau_samples)
            window_vals = dist_trace[window_slice]
            position_range = np.nanmax(window_vals) - np.nanmin(window_vals)
            position_ok = position_range < position_change_threshold
            
            if speed_ok and position_ok:
                plateau_onset_idx = i
                method = 'stable_window'
                break
    
    # Calculate plateau time
    plateau_time = t_axis[plateau_onset_idx] if plateau_onset_idx is not None else None
    
    return {
        'plateau_time': plateau_time,
        'plateau_from_onset': plateau_time,
        'plateau_idx': plateau_onset_idx,
        'peak_time': peak_time,
        'peak_idx': peak_idx,
        'speed_threshold': speed_threshold,
        'peak_speed': peak_speed,
        'method': method,
    }


def detect_plateau_single_trial(
    dist_trace: np.ndarray, 
    speed_trace: np.ndarray, 
    t_axis: np.ndarray,
    onset_idx: int,
    plateau_window_ms: float = 100.0,
    min_distance_threshold: float = 0.95,  # Must be at 80% of max
    max_cumulative_slope: float = 0.05,  # Accumulated |slopes| < 10% of max dist
) -> dict:
    """
    Detect plateau by finding windows with minimal accumulated absolute slope.
    
    Strategy: 
      - Compute point-to-point slopes within each window
      - Sum their absolute values
      - A true plateau has low cumulative |slope| (flat, maybe tiny wiggles)
      - A moving trace has high cumulative |slope| (consistent direction)
    """
    dt = t_axis[1] - t_axis[0]
    T = len(dist_trace)
    plateau_samples = int(plateau_window_ms / dt)
    
    if plateau_samples < 2:
        return {'plateau_time': None, 'plateau_idx': None, 'method': 'failed'}
    
    # Get max distance for thresholding
    max_dist = np.nanmax(dist_trace[onset_idx:])
    if max_dist < 1e-6:
        return {'plateau_time': None, 'plateau_idx': None, 'method': 'failed'}
    
    min_dist_for_plateau = min_distance_threshold * max_dist
    
    # Compute point-to-point differences (unnormalized slopes)
    diffs = np.diff(dist_trace)  # Length T-1
    
    # Scan forward from onset
    for i in range(onset_idx, T - plateau_samples):
        window = dist_trace[i:i + plateau_samples]
        
        # Check 1: Is the hand far enough from start?
        if np.nanmean(window) < min_dist_for_plateau:
            continue
        
        # Check 2: Cumulative absolute slope within window
        window_diffs = diffs[i:i + plateau_samples - 1]  # slopes within window
        cumulative_abs_slope = np.nansum(np.abs(window_diffs))
        
        # Normalize by max distance to make threshold interpretable
        cumulative_abs_slope_normalized = cumulative_abs_slope / max_dist
        
        if cumulative_abs_slope_normalized < max_cumulative_slope:
            # Found stable plateau!
            return {
                'plateau_time': t_axis[i],
                'plateau_idx': i,
                'method': 'cumulative_slope',
                'cumulative_slope': cumulative_abs_slope_normalized,
            }
    
    return {'plateau_time': None, 'plateau_idx': None, 'method': 'failed'}

def process_condition_with_onset_detection(
    pos_segs: np.ndarray,
    t_axis: np.ndarray,
    kp_names: list,
    idx_x: int,
    idx_y: int
) -> dict:
    """
    Process a condition: detect onset, compute metrics, align traces.
    
    The alignment depends on TIME_ZERO_REFERENCE setting.
    """
    n_trials = pos_segs.shape[0]
    T = pos_segs.shape[2]
    
    dt = t_axis[1] - t_axis[0] if len(t_axis) > 1 else 10.0
    if dt < 0.9:
        dt *= 1000.0
    
    # Extract X and Y traces
    x_all = pos_segs[:, idx_x, :]
    y_all = pos_segs[:, idx_y, :]
    
    # Compute speed for each trial
    speed_all = np.zeros((n_trials, T))
    for i in range(n_trials):
        x = x_all[i, :]
        y = y_all[i, :]
        if np.sum(np.isfinite(x)) >= 10 and np.sum(np.isfinite(y)) >= 10:
            speed_all[i, :] = compute_2d_speed(x, y, dt)
    
    # Detect reach onset for all trials
    onset_times, onset_indices, onset_results = detect_reach_onset_batch(
        pos_segs=x_all[:, np.newaxis, :],
        speed_segs=speed_all[:, np.newaxis, :],
        time_axis=t_axis,
        config=ONSET_CONFIG,
        keypoint_idx=0
    )
    
    # Compute distance from start for each trial
    distance_all = np.zeros((n_trials, T))
    for i in range(n_trials):
        x = x_all[i, :]
        y = y_all[i, :]
        
        # Use onset position as reference if valid, else use first position
        if onset_indices[i] >= 0:
            ref_idx = onset_indices[i]
        else:
            ref_idx = 0
        
        x0, y0 = x[ref_idx], y[ref_idx]
        
        # Smooth and compute distance
        x_smooth = smooth_1d(x, dt, window_ms=SAVGOL_WINDOW_MS)
        y_smooth = smooth_1d(y, dt, window_ms=SAVGOL_WINDOW_MS)
        
        distance_all[i, :] = np.sqrt((x_smooth - x0)**2 + (y_smooth - y0)**2)
    
    # Determine alignment based on TIME_ZERO_REFERENCE
    if TIME_ZERO_REFERENCE == 'reach_onset':
        # Align to detected reach onset
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
        
        # Valid mask based on onset detection
        valid_mask = onset_indices >= 0
        time_zero_label = "Reach Onset"
        
    else:  # 'stim_timing' - use original time axis (t=0 is IR crossing / stim timing)
        # No alignment needed - t=0 in original data is already stim timing
        # Just trim to window of interest
        t0_idx = np.argmin(np.abs(t_axis - 0))
        pre_samples = int(PRE_ZERO_MS / dt)
        post_samples = int(POST_ZERO_MS / dt)
        
        start_idx = max(0, t0_idx - pre_samples)
        end_idx = min(T, t0_idx + post_samples)
        
        aligned_time = t_axis[start_idx:end_idx] - t_axis[t0_idx]  # Center on t=0
        aligned_distance = distance_all[:, start_idx:end_idx]
        aligned_speed = speed_all[:, start_idx:end_idx]
        aligned_x = x_all[:, start_idx:end_idx]
        aligned_y = y_all[:, start_idx:end_idx]
        
        # All trials are valid when using stim timing
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


def detect_plateau_per_trial_aligned(
    aligned_distance: np.ndarray,
    aligned_speed: np.ndarray, 
    aligned_time: np.ndarray,
    valid_mask: np.ndarray
) -> Tuple[np.ndarray, dict]:
    """
    Detect plateau for each trial using aligned traces.
    
    Returns
    -------
    plateau_times : np.ndarray (n_trials,)
        Plateau time relative to t=0 for each trial (NaN if not detected)
    summary : dict
        Statistics and diagnostic info (uses SEM not SD for CI)
    """
    n_trials = aligned_distance.shape[0]
    
    # Normalize distance traces
    dist_norm, scale_factors = normalize_distance_traces(aligned_distance, method='max')
    
    # Normalize speed
    speed_norm = np.zeros_like(aligned_speed)
    for i in range(n_trials):
        sf = scale_factors[i] if scale_factors[i] > 1e-6 else 1.0
        speed_norm[i, :] = aligned_speed[i, :] / sf
    
    # Find t=0 index in aligned coordinates
    dt = aligned_time[1] - aligned_time[0]
    t0_idx_aligned = np.argmin(np.abs(aligned_time))
    
    # Detect plateau for each trial
    plateau_times = np.full(n_trials, np.nan)
    detection_results = []
    
    for i in range(n_trials):
        if not valid_mask[i]:
            detection_results.append({'plateau_time': None, 'method': 'invalid_trial'})
            continue
        
        result = detect_plateau_single_trial(
            dist_norm[i, :],
            speed_norm[i, :],
            aligned_time,
            onset_idx=t0_idx_aligned
        )
        
        if result['plateau_time'] is not None:
            plateau_times[i] = result['plateau_time']
        
        detection_results.append(result)
    
    # Summary statistics - use SEM for confidence intervals
    valid_times = plateau_times[~np.isnan(plateau_times)]
    n_valid = np.sum(valid_mask)
    
    # Compute SEM (standard error of the mean)
    dist_sem = np.nanstd(dist_norm, axis=0) / np.sqrt(max(1, n_valid))
    
    summary = {
        'dist_norm': dist_norm,
        'speed_norm': speed_norm,
        'scale_factors': scale_factors,
        'dist_mean': np.nanmean(dist_norm, axis=0),
        'dist_median': np.nanmedian(dist_norm, axis=0),
        'dist_std': np.nanstd(dist_norm, axis=0),
        'dist_sem': dist_sem,  # Standard Error of Mean
        'speed_mean': np.nanmean(speed_norm, axis=0),
        'plateau_times_all': plateau_times,
        'plateau_mean': np.nanmean(valid_times) if len(valid_times) > 0 else np.nan,
        'plateau_std': np.nanstd(valid_times) if len(valid_times) > 0 else np.nan,
        'plateau_sem': np.nanstd(valid_times) / np.sqrt(len(valid_times)) if len(valid_times) > 0 else np.nan,
        'n_detected': np.sum(~np.isnan(plateau_times)),
        'n_valid': n_valid,
        'n_total': n_trials,
        'detection_rate': np.sum(~np.isnan(plateau_times)) / max(1, n_valid),
        'detection_results': detection_results,
    }
    
    return plateau_times, summary


# ---------------------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------------------

def compute_control_vs_stim_statistics(all_plateau_data: list) -> dict:
    """Compute statistical tests comparing each stim condition vs control."""
    results = {
        'comparisons': [],
        'control_data': {},
    }
    
    for target in ['A', 'B']:
        target_data = [d for d in all_plateau_data if d['target'] == target]
        
        if len(target_data) < 2:
            continue
        
        control_data = [d for d in target_data 
                        if d['condition_info']['condition_type'] == 'control']
        stim_data = [d for d in target_data 
                     if d['condition_info']['condition_type'] != 'control']
        
        if not control_data or not stim_data:
            continue
        
        # Pool control trials
        control_times = np.concatenate([
            d['plateau_times'][~np.isnan(d['plateau_times'])] 
            for d in control_data
        ])
        
        if len(control_times) < 2:
            continue
        
        results['control_data'][target] = {
            'times': control_times,
            'mean': np.mean(control_times),
            'std': np.std(control_times),
            'n': len(control_times),
            'br_indices': [d['br_idx'] for d in control_data],
        }
        
        # Compare each stim to control
        for stim_d in stim_data:
            stim_times = stim_d['plateau_times'][~np.isnan(stim_d['plateau_times'])]
            
            if len(stim_times) < 2:
                continue
            
            # t-test
            try:
                t_stat, t_p = stats.ttest_ind(control_times, stim_times)
            except Exception:
                t_stat, t_p = np.nan, np.nan
            
            # Mann-Whitney U
            try:
                u_stat, u_p = stats.mannwhitneyu(control_times, stim_times, 
                                                  alternative='two-sided')
            except Exception:
                u_stat, u_p = np.nan, np.nan
            
            # Cohen's d
            pooled_std = np.sqrt((np.var(control_times) + np.var(stim_times)) / 2)
            cohens_d = (np.mean(stim_times) - np.mean(control_times)) / pooled_std if pooled_std > 0 else np.nan
            
            comparison = {
                'target': target,
                'stim_br_idx': stim_d['br_idx'],
                'stim_label': stim_d['condition_info']['label'],
                'condition_type': stim_d['condition_info']['condition_type'],
                'control_n': len(control_times),
                'stim_n': len(stim_times),
                'control_mean': np.mean(control_times),
                'stim_mean': np.mean(stim_times),
                'control_std': np.std(control_times),
                'stim_std': np.std(stim_times),
                'diff_ms': np.mean(stim_times) - np.mean(control_times),
                't_stat': t_stat,
                't_p': t_p,
                'u_stat': u_stat,
                'u_p': u_p,
                'cohens_d': cohens_d,
            }
            results['comparisons'].append(comparison)
    
    return results


# ---------------------------------------------------------------------
# PLOTTING: RAW TRACE DIAGNOSTIC (NEW)
# ---------------------------------------------------------------------

def plot_raw_trace_diagnostic(
    cond_data: dict,
    condition_info: dict,
    target: str
) -> None:
    """
    Generate diagnostic plot showing raw traces aligned to time zero.
    
    This is for debugging reach initiation detection - shows individual
    traces with a dashed line at t=0, without any mean/plateau overlays.
    """
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
    
    # Create figure with subplots for X, Y, and distance
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    cmap = plt.colormaps.get_cmap('viridis')
    
    # --- Plot X traces ---
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
    
    # --- Plot Y traces ---
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
    
    # --- Plot Distance traces ---
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
    
    # Save
    diag_dir = FIG_OUT_DIR / "raw_trace_diagnostics"
    diag_dir.mkdir(exist_ok=True)
    out_filename = f"RawTraces_BR{br_idx:03d}_Target{target}.{OUTPUT_FORMAT}"
    fig.savefig(diag_dir / out_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"    Saved: raw_trace_diagnostics/{out_filename}")


# ---------------------------------------------------------------------
# PLOTTING: PER-CONDITION DIAGNOSTIC (SIMPLIFIED)
# ---------------------------------------------------------------------

def plot_condition_diagnostic_v2(
    cond_data: dict,
    condition_info: dict,
    target: str
) -> None:
    """
    Generate simplified diagnostic plot for a single condition showing:
    - All individual trial traces (normalized distance)
    - Mean trace
    - Plateau markers for each trial
    
    ONLY plots traces + mean + plateau points. No extra panels.
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
    
    # Detect plateaus
    plateau_times, summary = detect_plateau_per_trial_aligned(
        aligned_distance, aligned_speed, aligned_time, valid_mask
    )
    
    # Create SINGLE figure - just traces + mean + plateaus
    fig, ax = plt.subplots(figsize=(12, 7))
    
    cmap = plt.colormaps.get_cmap('viridis')
    
    # Plot individual trials
    for i in range(n_trials):
        if not valid_mask[i]:
            continue
        
        color = cmap(i / max(1, n_trials - 1)) if n_trials > 1 else 'blue'
        ax.plot(aligned_time, summary['dist_norm'][i, :], 
                color=color, alpha=0.3, linewidth=0.8)
        
        # Plateau marker for this trial
        if not np.isnan(plateau_times[i]):
            pt_idx = np.argmin(np.abs(aligned_time - plateau_times[i]))
            ax.scatter(plateau_times[i], summary['dist_norm'][i, pt_idx],
                      color=color, s=40, zorder=5, alpha=0.7,
                      edgecolors='black', linewidths=0.5)
    
    # # Mean trace (thick black line)
    # ax.plot(aligned_time, summary['dist_mean'], color='black', linewidth=2.5, 
    #         label=f'Mean (n={n_valid})')
    
    # # Mean plateau marker (large red diamond)
    # if not np.isnan(summary['plateau_mean']):
    #     pt_idx = np.argmin(np.abs(aligned_time - summary['plateau_mean']))
    #     ax.scatter(summary['plateau_mean'], summary['dist_mean'][pt_idx],
    #               color='red', s=200, marker='D', zorder=10,
    #               edgecolors='black', linewidths=2,
    #               label=f"Mean plateau: {summary['plateau_mean']:.0f}ms")

    # Median trace (thick black line)
    ax.plot(aligned_time, summary['dist_median'], color='black', linewidth=2.5, 
            label=f'Median (n={n_valid})')

    # Median plateau marker (large red diamond)
    plateau_median = np.nanmedian(plateau_times[~np.isnan(plateau_times)]) if np.sum(~np.isnan(plateau_times)) > 0 else np.nan
    if not np.isnan(plateau_median):
        pt_idx = np.argmin(np.abs(aligned_time - plateau_median))
        ax.scatter(plateau_median, summary['dist_median'][pt_idx],
                color='red', s=200, marker='D', zorder=10,
                edgecolors='black', linewidths=2,
                label=f"Median plateau: {plateau_median:.0f}ms")
    
    # t=0 marker
    ax.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.8, 
               label=f't=0 ({time_zero_label})')
    
    ax.set_xlabel('Time from t=0 (ms)', fontsize=12)
    ax.set_ylabel('Normalized Distance [0-1]', fontsize=12)
    # ax.set_title(f'BR{br_idx}: {condition_info["label"]} | Target {target}\n'
    #              f'Individual Traces + Mean + Plateau Points (detected: {summary["n_detected"]}/{n_valid})',
    #              fontsize=11)
    ax.set_title(f'BR{br_idx}: {condition_info["label"]} | Target {target}\n'
             f'Individual Traces + Median + Plateau Points (detected: {summary["n_detected"]}/{n_valid})',
             fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(-0.1, 1.3)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Session: {PARAMS.session}", fontsize=10, y=0.98)
    fig.tight_layout()
    
    # Save
    diag_dir = FIG_OUT_DIR / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    out_filename = f"Diagnostic_BR{br_idx:03d}_Target{target}.{OUTPUT_FORMAT}"
    fig.savefig(diag_dir / out_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"    Saved: diagnostics/{out_filename}")


# ---------------------------------------------------------------------
# PLOTTING: MASTER SUMMARY (ALL CONDITIONS)
# ---------------------------------------------------------------------

def plot_master_summary(
    all_conditions_data: list,
    metadata_df: pd.DataFrame,
    target: str,
    conditions_to_plot: list = None,
    title_suffix: str = ""
) -> list:
    """
    Plot mean traces for all/selected conditions with SEM on same axes.
    
    Returns list of plateau data dicts.
    """
    if not all_conditions_data:
        return []
    
    # Filter conditions if specified
    if conditions_to_plot:
        filtered_data = [d for d in all_conditions_data
                         if d.get('br_idx') in conditions_to_plot]
        if not filtered_data:
            print(f"[warn] No conditions matched filter {conditions_to_plot}")
            return []
        all_conditions_data = filtered_data
    
    # Sort by BR index
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
        
        # Detect plateaus
        plateau_times, summary = detect_plateau_per_trial_aligned(
            aligned_distance, aligned_speed, aligned_time, valid_mask
        )
        
        # Label
        label = f"{condition_info['label']} (n={summary['n_detected']}/{n_valid})"
        
        # Plot mean ± SEM (not 95% CI)
        ax.plot(aligned_time, summary['dist_mean'], color=color, linewidth=2, label=label)
        ax.fill_between(aligned_time,
                        summary['dist_mean'] - summary['dist_sem'],
                        summary['dist_mean'] + summary['dist_sem'],
                        color=color, alpha=0.2)
        
        # Mean plateau marker with SEM error bar
        if not np.isnan(summary['plateau_mean']):
            pt_idx = np.argmin(np.abs(aligned_time - summary['plateau_mean']))
            pt_y = summary['dist_mean'][pt_idx]
            
            ax.axvline(summary['plateau_mean'], color=color, linestyle=':', 
                       alpha=0.5, linewidth=1.5)
            ax.errorbar(summary['plateau_mean'], pt_y, 
                       xerr=summary['plateau_sem'],  # SEM, not 1.96*SEM
                       color=color, capsize=5, capthick=2, linewidth=2,
                       marker='o', markersize=10, markeredgecolor='black',
                       markeredgewidth=1.5, zorder=10)
        
        # Store results
        plateau_results.append({
            'br_idx': br_idx,
            'target': target,
            'condition_info': condition_info,
            'plateau_times': plateau_times,
            'plateau_mean': summary['plateau_mean'],
            'plateau_std': summary['plateau_std'],
            'plateau_sem': summary['plateau_sem'],
            'n_detected': summary['n_detected'],
            'n_valid': n_valid,
            'n_total': aligned_distance.shape[0],
            'detection_rate': summary['detection_rate'],
        })
    
    # Formatting
    ax.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.8, 
               label=f't=0 ({time_zero_label})')
    ax.set_xlabel('Time from t=0 (ms)', fontsize=12)
    ax.set_ylabel('Normalized Distance [0-1]', fontsize=12)
    ax.set_title(f'Target {target}: Normalized Distance{title_suffix}\n'
                 f'Mean ± SEM | ● = mean plateau time (± SEM)', fontsize=12)
    ax.legend(fontsize='small', loc='upper left', ncol=2)
    ax.set_ylim(0, 1.3)
    ax.set_xlim(aligned_time.min(), aligned_time.max())
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Session: {PARAMS.session} | TIME_ZERO_REFERENCE='{TIME_ZERO_REFERENCE}'", 
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
# PLOTTING: BOX PLOT
# ---------------------------------------------------------------------

def plot_plateau_boxplot(
    all_plateau_data: list,
    stats_results: dict,
    title_suffix: str = ""
) -> None:
    """Create box plot showing plateau times with control vs stim comparisons."""
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
        
        fig_width = max(10, n_conditions * 0.8 + 2)
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        box_data = []
        labels = []
        colors = []
        br_indices = []
        
        for d in sorted_data:
            valid_times = d['plateau_times'][~np.isnan(d['plateau_times'])]
            box_data.append(valid_times if len(valid_times) > 0 else [np.nan])
            
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
            
            br_indices.append(d['br_idx'])
        
        # Box plot
        bp = ax.boxplot(box_data, positions=range(len(box_data)),
                       widths=0.6, patch_artist=True, manage_ticks=False)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Individual points
        np.random.seed(42)
        for i, (data, color) in enumerate(zip(box_data, colors)):
            if len(data) > 0 and not np.all(np.isnan(data)):
                valid_data = [v for v in data if not np.isnan(v)]
                if valid_data:
                    jitter = np.random.uniform(-0.15, 0.15, len(valid_data))
                    ax.scatter(np.full(len(valid_data), i) + jitter, valid_data,
                              color='black', alpha=0.4, s=20, edgecolor='white', linewidth=0.5)
        
        # Significance markers
        all_valid = [v for d in box_data for v in (d if hasattr(d, '__iter__') else [d])
                     if not np.isnan(v)]
        y_max = max(all_valid) if all_valid else 400
        y_offset = y_max * 0.05
        
        if stats_results and stats_results['comparisons']:
            target_comps = [c for c in stats_results['comparisons'] if c['target'] == target]
            
            for comp in target_comps:
                stim_br = comp['stim_br_idx']
                stim_pos = None
                for j, br in enumerate(br_indices):
                    if br == stim_br:
                        stim_pos = j
                        break
                
                if stim_pos is not None:
                    p_val = comp['t_p']
                    if not np.isnan(p_val) and p_val < 0.05:
                        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                        cond_data = box_data[stim_pos]
                        if len(cond_data) > 0 and not np.all(np.isnan(cond_data)):
                            cond_max = np.nanmax(cond_data)
                            y_pos = cond_max + y_offset
                        else:
                            y_pos = y_max + y_offset
                        
                        ax.annotate(sig, xy=(stim_pos, y_pos),
                                   ha='center', va='bottom', fontsize=14, fontweight='bold',
                                   color='red')
                        ax.annotate(f'p={p_val:.3f}', xy=(stim_pos, y_pos + y_offset * 0.5),
                                   ha='center', va='bottom', fontsize=7, color='darkred')
        
        # Formatting
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Plateau Time from t=0 (ms)', fontsize=12)
        ax.set_title(f'Target {target}: Plateau Time Distribution{title_suffix}\n'
                    f"(Reference: t=0 = {TIME_ZERO_REFERENCE})", fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0, top=y_max * 1.25)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', alpha=0.7, edgecolor='black', label='Control'),
            Patch(facecolor='steelblue', alpha=0.7, edgecolor='black', label='Pulsed Stim'),
            Patch(facecolor='lightsalmon', alpha=0.7, edgecolor='black', label='Continuous Stim'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        fig.tight_layout()
        
        suffix = title_suffix.replace(' ', '_').replace(':', '')
        out_filename = f"Plateau_BoxPlot_Target{target}{suffix}.{OUTPUT_FORMAT}"
        fig.savefig(FIG_OUT_DIR / out_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {out_filename}")


def save_statistics_csv(all_plateau_data: list, stats_results: dict, title_suffix: str = "") -> None:
    """Save summary statistics to CSV."""
    rows = []
    for d in all_plateau_data:
        valid_times = d['plateau_times'][~np.isnan(d['plateau_times'])]
        cinfo = d['condition_info']
        rows.append({
            'session': PARAMS.session,
            'target': d['target'],
            'br_idx': d['br_idx'],
            'condition_type': cinfo['condition_type'],
            'condition_label': cinfo['label'],
            'n_trials_total': d['n_total'],
            'n_valid_onset': d['n_valid'],
            'n_plateau_detected': d['n_detected'],
            'detection_rate': d['detection_rate'],
            'plateau_mean_ms': d['plateau_mean'],
            'plateau_std_ms': d['plateau_std'],
            'plateau_sem_ms': d['plateau_sem'],
            'plateau_median_ms': np.nanmedian(valid_times) if len(valid_times) > 0 else np.nan,
            'time_zero_reference': TIME_ZERO_REFERENCE,
        })
    
    df = pd.DataFrame(rows)
    suffix = title_suffix.replace(' ', '_').replace(':', '')
    csv_filename = FIG_OUT_DIR / f"Plateau_Summary{suffix}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"  Saved: {csv_filename}")
    
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
            # Determine condition
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
            
            # BR Filter
            if BR_FILTER is not None and br_idx is not None:
                if br_idx not in BR_FILTER:
                    return None
            
            # Check for "at rest" condition - SKIP IT
            if br_idx is not None:
                condition_info = get_condition_info(br_idx, metadata_df)
                if condition_info.get('is_at_rest', False):
                    print(f"    [skip] BR{br_idx} is 'at rest' condition - skipping")
                    return None
            
            # Get position data
            pos_key = None
            for key_pattern in [f"beh_{CAMERA_TO_USE}_segs", f"{CAMERA_TO_USE}_segs", "segs"]:
                if key_pattern in data:
                    pos_key = key_pattern
                    break
            
            if pos_key is None:
                return None
            
            pos_segs = data[pos_key]
            
            # Time axis
            t_axis = None
            for t_key in ["beh_rel_t", "rel_t", "t"]:
                if t_key in data:
                    t_axis = data[t_key]
                    break
            
            if t_axis is None:
                return None
            
            # Keypoint names
            kp_names = _get_keypoint_names_from_data(data, CAMERA_TO_USE)
            if not kp_names:
                kp_names = [f"kp_{i}" for i in range(pos_segs.shape[1])]
            
            idx_x = find_keypoint_index(kp_names, f"{KEYPOINT_BASE}_x")
            idx_y = find_keypoint_index(kp_names, f"{KEYPOINT_BASE}_y")
            
            if idx_x is None or idx_y is None:
                return None
            
            # Process with onset detection
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
        # REMOVED: Path("at_rest"), - we skip at rest conditions entirely
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
    print(f"Plateau Detection Analysis")
    print(f"=" * 70)
    print(f"Session: {PARAMS.session}")
    print(f"Output Dir: {FIG_OUT_DIR}")
    print(f"Metadata CSV: {METADATA_CSV}")
    print(f"Keypoint: {KEYPOINT_BASE}")
    print(f"Control BR indices: {CONTROL_BR_INDICES}")
    print(f"TIME_ZERO_REFERENCE: {TIME_ZERO_REFERENCE}")
    print(f"  ('reach_onset' = align to detected reach start)")
    print(f"  ('stim_timing' = use original t=0 from IR crossing/stim)")
    print("")
    
    # Load metadata
    try:
        metadata_df = load_metadata_csv(METADATA_CSV)
        print(f"Loaded metadata: {len(metadata_df)} rows")
    except Exception as e:
        print(f"[warn] Could not load metadata: {e}")
        metadata_df = pd.DataFrame()
    
    # Process each target
    target_A_data = process_target("A", 1, metadata_df)
    target_B_data = process_target("B", 2, metadata_df)
    
    # Generate raw trace diagnostic plots (for debugging reach detection)
    if GENERATE_RAW_TRACE_DIAGNOSTICS:
        print("\nGenerating raw trace diagnostic plots...")
        for cond_data in target_A_data:
            cinfo = get_condition_info(cond_data['br_idx'], metadata_df)
            plot_raw_trace_diagnostic(cond_data, cinfo, "A")
        for cond_data in target_B_data:
            cinfo = get_condition_info(cond_data['br_idx'], metadata_df)
            plot_raw_trace_diagnostic(cond_data, cinfo, "B")
    
    # Generate per-condition diagnostic plots (traces + mean + plateaus only)
    if GENERATE_CONDITION_DIAGNOSTICS:
        print("\nGenerating per-condition diagnostic plots...")
        for cond_data in target_A_data:
            cinfo = get_condition_info(cond_data['br_idx'], metadata_df)
            plot_condition_diagnostic_v2(cond_data, cinfo, "A")
        for cond_data in target_B_data:
            cinfo = get_condition_info(cond_data['br_idx'], metadata_df)
            plot_condition_diagnostic_v2(cond_data, cinfo, "B")
    
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
            stats_master = compute_control_vs_stim_statistics(all_plateau_data_master)
            plot_plateau_boxplot(all_plateau_data_master, stats_master, "_ALL")
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
            stats_subset = compute_control_vs_stim_statistics(all_plateau_data_subset)
            cond_str = "_BR" + "_".join(str(c) for c in sorted(SUBSET_CONDITIONS))
            plot_plateau_boxplot(all_plateau_data_subset, stats_subset, cond_str)
            if GENERATE_STATS_SUMMARY:
                save_statistics_csv(all_plateau_data_subset, stats_subset, cond_str)
    
    print(f"\n{'=' * 70}")
    print(f"Done! Figures saved to {FIG_OUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()