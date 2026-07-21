import numpy as np
import warnings
import re
from pathlib import Path
from typing import Optional
import pandas as pd
from RCP_analysis.python.functions.config_loading import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""
Purpose:
  Analyze normalized hand position to detect reach endpoints (plateaus).
  
  Key features:
  1. Per-TRIAL plateau detection
  2. Statistical comparison: Control vs each Stim condition
  3. Automatic condition labeling from metadata CSV
  4. Master plot (all conditions) + subset plots
  
Outputs:
  - Normalized distance plots with mean plateau markers
  - Box plots of plateau times with statistics
  - Per-condition diagnostic plots
  - Summary statistics CSV
"""

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
CAMERA_TO_USE = "cam1"
KEYPOINT_BASE = "middle"

FIG_OUT_DIR = OUT_BASE / "figures" / "plateau_analysis"
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FORMAT = 'png'
plt.rcParams.update({'font.size': 10})

# ---------------------------------------------------------------------
# PLOT CONTROL
# ---------------------------------------------------------------------
GENERATE_MASTER_PLOT = True          # All conditions
GENERATE_SUBSET_PLOT = True          # Selected conditions only
GENERATE_DIAGNOSTIC_PLOTS = True     # Per-condition diagnostic
GENERATE_STATS_SUMMARY = True

# Condition filtering for subset plot
SUBSET_CONDITIONS = [20, 21, 22]  # Set to None to skip subset plot

# ---------------------------------------------------------------------
# CONTROL CONDITION SPECIFICATION
# ---------------------------------------------------------------------
# Specify which BR indices are control/baseline conditions
# This overrides metadata lookup if specified
CONTROL_BR_INDICES = [20]  # Set to list of BR indices that are controls, or None to use metadata

# ---------------------------------------------------------------------
# SMOOTHING PARAMETERS
# ---------------------------------------------------------------------
SMOOTHING_METHOD = 'savgol'
SAVGOL_WINDOW_MS = 50.0
SAVGOL_POLYORDER = 3
VELOCITY_WINDOW_MS = 75.0

# ---------------------------------------------------------------------
# PLATEAU DETECTION PARAMETERS
# ---------------------------------------------------------------------
BASELINE_WINDOW_START_MS = -400.0  # Start of baseline window
BASELINE_WINDOW_END_MS = -300.0    # End of baseline window
PLATEAU_WINDOW_MS = 100.0
SPEED_THRESHOLD_FACTOR = 0.15
POSITION_CHANGE_THRESHOLD = 0.15

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
_DIAG_PRINTED = set()


# ---------------------------------------------------------------------
# METADATA PARSING
# ---------------------------------------------------------------------

def load_metadata_csv(csv_path: Path) -> pd.DataFrame:
    """Load and normalize metadata CSV."""
    if not csv_path.exists():
        print(f"[warn] Metadata CSV not found: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"  Metadata columns: {list(df.columns)}")
    return df


def get_condition_info(br_idx: int, metadata_df: pd.DataFrame) -> dict:
    """
    Extract condition information for a given BR index.
    
    Returns dict with:
        - condition_type: 'control', 'continuous_stim', 'pulsed_stim'
        - label: Human-readable label for plots
        - n_channels: Number of stimulated channels
        - frequency_hz: Stimulation frequency
        - duration_ms: Stimulation duration (or 'cont')
        - n_pulses: Number of pulses (for pulsed stim)
    """
    # First check if this BR is explicitly marked as control
    if CONTROL_BR_INDICES is not None and br_idx in CONTROL_BR_INDICES:
        return {
            'condition_type': 'control',
            'label': 'Control',
            'n_channels': 0,
            'frequency_hz': 0,
            'duration_ms': 0,
            'n_pulses': 0,
            'br_idx': br_idx,
        }
    
    # If no metadata, return default based on control list
    if metadata_df is None or metadata_df.empty:
        return _default_condition_info(br_idx)
    
    # Find row matching BR_File
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
    
    # Extract fields (handle various column name formats)
    channels = _get_field(row, ['channels', 'n_channels', 'num_channels'])
    freq_hz = _get_field(row, ['stim_frequency_hz', 'frequency_hz', 'freq_hz', 'frequency'])
    duration = _get_field(row, ['stim_duration_ms', 'duration_ms', 'duration'])
    notes = _get_field(row, ['notes', 'note', 'comment', 'type', 'condition_type'])
    
    # Determine condition type
    is_baseline = False
    if notes is not None:
        notes_lower = str(notes).lower()
        if any(x in notes_lower for x in ['baseline', 'control', 'no stim', 'nostim']):
            is_baseline = True
    if _is_empty(channels) and _is_empty(duration):
        is_baseline = True
    if str(channels).strip() == '-' or str(channels).strip() == '0':
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
        }
    
    # Parse values
    n_ch = _parse_int(channels, default=0)
    freq = _parse_float(freq_hz, default=0)
    
    # Check for continuous stim
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
        }
    
    # Pulsed stim
    dur_ms = _parse_float(duration, default=0)
    n_pulses = int(dur_ms / 2.5) if dur_ms > 0 else 0
    
    # Build descriptive label
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
    }


def _default_condition_info(br_idx: int) -> dict:
    """Return default condition info when metadata lookup fails."""
    # Check if explicitly marked as control
    if CONTROL_BR_INDICES is not None and br_idx in CONTROL_BR_INDICES:
        return {
            'condition_type': 'control',
            'label': 'Control',
            'n_channels': 0,
            'frequency_hz': 0,
            'duration_ms': 0,
            'n_pulses': 0,
            'br_idx': br_idx,
        }
    
    return {
        'condition_type': 'pulsed_stim',  # Assume stim if not control
        'label': f'BR{br_idx}',
        'n_channels': None,
        'frequency_hz': None,
        'duration_ms': None,
        'n_pulses': None,
        'br_idx': br_idx,
    }


def _get_field(row: pd.Series, possible_names: list) -> any:
    """Get field value trying multiple possible column names."""
    for name in possible_names:
        if name in row.index:
            val = row[name]
            if pd.notna(val):
                return val
    return None


def _is_empty(val) -> bool:
    """Check if value is empty/null."""
    if val is None:
        return True
    if pd.isna(val):
        return True
    if str(val).strip() in ('', '-', 'nan', 'none', 'na'):
        return True
    return False


def _parse_int(val, default=0) -> int:
    """Parse integer from various formats."""
    if _is_empty(val):
        return default
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return default


def _parse_float(val, default=0.0) -> float:
    """Parse float from various formats."""
    if _is_empty(val):
        return default
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def smooth_1d(arr: np.ndarray, dt: float) -> np.ndarray:
    """Apply smoothing to 1D array with robust error handling."""
    if SMOOTHING_METHOD == 'none':
        return arr.copy()
    
    if len(arr) < 5:
        return arr.copy()
    
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return arr.copy()
    
    finite_vals = arr[finite_mask]
    if np.all(finite_vals == finite_vals[0]):
        return arr.copy()
    
    arr_clean = arr.copy()
    if not np.all(finite_mask):
        nans = ~finite_mask
        if np.sum(finite_mask) >= 2:
            arr_clean[nans] = np.interp(
                np.flatnonzero(nans),
                np.flatnonzero(finite_mask),
                arr[finite_mask]
            )
        else:
            return arr.copy()
    
    if SMOOTHING_METHOD == 'savgol':
        window_length = int(SAVGOL_WINDOW_MS / dt)
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(window_length, 5)
        
        if window_length > len(arr_clean):
            window_length = len(arr_clean)
            if window_length % 2 == 0:
                window_length -= 1
            window_length = max(window_length, 5)
        
        if window_length > len(arr_clean):
            sigma = max(20.0 / dt, 0.5)
            sigma = min(sigma, len(arr_clean) / 4)
            return gaussian_filter1d(arr_clean, sigma)
        
        polyorder = min(SAVGOL_POLYORDER, window_length - 1)
        
        try:
            return savgol_filter(arr_clean, window_length, polyorder)
        except (np.linalg.LinAlgError, ValueError):
            sigma = max(20.0 / dt, 0.5)
            sigma = min(sigma, len(arr_clean) / 4)
            return gaussian_filter1d(arr_clean, sigma)
    
    elif SMOOTHING_METHOD == 'gaussian':
        sigma = max(20.0 / dt, 0.5)
        sigma = min(sigma, len(arr_clean) / 4)
        return gaussian_filter1d(arr_clean, sigma)
    
    return arr_clean


def compute_velocity_1d(position: np.ndarray, dt: float) -> np.ndarray:
    """Compute velocity from position with robust error handling."""
    if len(position) < 7:
        return np.gradient(position, dt)
    
    finite_mask = np.isfinite(position)
    if not np.all(finite_mask):
        position_clean = position.copy()
        nans = ~finite_mask
        if np.sum(finite_mask) >= 2:
            position_clean[nans] = np.interp(
                np.flatnonzero(nans),
                np.flatnonzero(finite_mask),
                position[finite_mask]
            )
        else:
            return np.zeros_like(position)
    else:
        position_clean = position
    
    window_length = int(VELOCITY_WINDOW_MS / dt)
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(window_length, 7)
    
    if window_length > len(position_clean):
        window_length = len(position_clean)
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(window_length, 3)
    
    polyorder = min(3, window_length - 1)
    
    try:
        return savgol_filter(position_clean, window_length, polyorder=polyorder, 
                            deriv=1, delta=dt)
    except (np.linalg.LinAlgError, ValueError):
        return np.gradient(position_clean, dt)


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
    
    x_smooth = smooth_1d(x, dt)
    y_smooth = smooth_1d(y, dt)
    
    x0, y0 = x_smooth[0], y_smooth[0]
    distance_from_start = np.sqrt((x_smooth - x0)**2 + (y_smooth - y0)**2)
    
    vx = compute_velocity_1d(x_smooth, dt)
    vy = compute_velocity_1d(y_smooth, dt)
    speed_2d = np.sqrt(vx**2 + vy**2)
    
    return {
        'x_smooth': x_smooth,
        'y_smooth': y_smooth,
        'distance_from_start': distance_from_start,
        'speed_2d': speed_2d,
    }


# ---------------------------------------------------------------------
# NORMALIZATION AND PLATEAU DETECTION
# ---------------------------------------------------------------------

def normalize_distance_traces(distance_all: np.ndarray, 
                              method: str = 'max') -> tuple[np.ndarray, np.ndarray]:
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


def detect_plateau_single_trial(dist_trace: np.ndarray, 
                                 speed_trace: np.ndarray, 
                                 t_axis: np.ndarray,
                                 baseline_start_ms: float = BASELINE_WINDOW_START_MS,
                                 baseline_end_ms: float = BASELINE_WINDOW_END_MS,
                                 plateau_window_ms: float = PLATEAU_WINDOW_MS,
                                 speed_threshold_factor: float = SPEED_THRESHOLD_FACTOR,
                                 position_change_threshold: float = POSITION_CHANGE_THRESHOLD) -> dict:
    """
    Detect plateau time for a SINGLE trial.
    
    Returns dict with plateau_time (or None if not found) and diagnostic info.
    """
    dt = t_axis[1] - t_axis[0]
    
    # Step 1: Compute baseline speed from -300 to -200ms window
    baseline_mask = (t_axis >= baseline_start_ms) & (t_axis < baseline_end_ms)
    if not np.any(baseline_mask):
        baseline_mask = t_axis < t_axis[0] + 100
    
    baseline_speed = np.nanmean(speed_trace[baseline_mask])
    baseline_speed_std = np.nanstd(speed_trace[baseline_mask])
    
    if baseline_speed_std < 1e-6:
        baseline_speed_std = baseline_speed * 0.1
    
    # Step 2: Set speed threshold
    peak_speed = np.nanmax(speed_trace)
    speed_threshold = (baseline_speed + 
                       2 * baseline_speed_std + 
                       speed_threshold_factor * peak_speed)
    
    # Step 3: Find peak distance
    peak_idx = np.nanargmax(dist_trace)
    peak_time = t_axis[peak_idx]
    
    # Step 4: Plateau window in samples
    plateau_samples = int(plateau_window_ms / dt)
    plateau_samples = max(plateau_samples, 1)
    
    plateau_time = None
    plateau_idx = None
    method = None
    
    # Step 5: Scan from peak onwards
    for i in range(peak_idx, len(t_axis) - plateau_samples):
        current_speed = speed_trace[i]
        speed_ok = current_speed <= speed_threshold
        
        window_slice = slice(i, i + plateau_samples)
        window_vals = dist_trace[window_slice]
        position_range = np.nanmax(window_vals) - np.nanmin(window_vals)
        position_ok = position_range < position_change_threshold
        
        if speed_ok and position_ok:
            plateau_time = t_axis[i]
            plateau_idx = i
            method = 'both'
            break
        elif speed_ok and plateau_time is None:
            plateau_time = t_axis[i]
            plateau_idx = i
            method = 'speed'
    
    return {
        'plateau_time': plateau_time,
        'plateau_idx': plateau_idx,
        'peak_time': peak_time,
        'peak_idx': peak_idx,
        'baseline_speed': baseline_speed,
        'baseline_speed_std': baseline_speed_std,
        'speed_threshold': speed_threshold,
        'peak_speed': peak_speed,
        'method': method,
    }


def detect_plateau_per_trial(distance_all: np.ndarray, 
                              speed_all: np.ndarray,
                              t_axis: np.ndarray,
                              normalization_method: str = 'max') -> tuple[np.ndarray, dict]:
    """
    Detect plateau time for EACH trial individually.
    
    Returns
    -------
    plateau_times : np.ndarray (n_trials,)
    summary : dict with statistics and normalized traces
    """
    n_trials = distance_all.shape[0]
    
    dist_norm, scale_factors = normalize_distance_traces(distance_all, method=normalization_method)
    
    speed_norm = np.zeros_like(speed_all)
    for i in range(n_trials):
        speed_norm[i, :] = speed_all[i, :] / scale_factors[i]
    
    plateau_times = np.full(n_trials, np.nan)
    detection_results = []
    
    for i in range(n_trials):
        result = detect_plateau_single_trial(
            dist_norm[i, :],
            speed_norm[i, :],
            t_axis
        )
        if result['plateau_time'] is not None:
            plateau_times[i] = result['plateau_time']
        detection_results.append(result)
    
    valid_times = plateau_times[~np.isnan(plateau_times)]
    
    summary = {
        'dist_norm': dist_norm,
        'speed_norm': speed_norm,
        'scale_factors': scale_factors,
        'dist_mean': np.nanmean(dist_norm, axis=0),
        'dist_std': np.nanstd(dist_norm, axis=0),
        'dist_ci': 1.96 * np.nanstd(dist_norm, axis=0) / np.sqrt(n_trials),
        'speed_mean': np.nanmean(speed_norm, axis=0),
        'plateau_times_all': plateau_times,
        'plateau_mean': np.nanmean(valid_times) if len(valid_times) > 0 else np.nan,
        'plateau_std': np.nanstd(valid_times) if len(valid_times) > 0 else np.nan,
        'plateau_sem': np.nanstd(valid_times) / np.sqrt(len(valid_times)) if len(valid_times) > 0 else np.nan,
        'n_detected': np.sum(~np.isnan(plateau_times)),
        'n_total': n_trials,
        'detection_rate': np.sum(~np.isnan(plateau_times)) / n_trials,
        'detection_results': detection_results,
    }
    
    return plateau_times, summary


# ---------------------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------------------

def compute_control_vs_stim_statistics(all_plateau_data: list[dict]) -> dict:
    """
    Compute statistical tests comparing each stim condition vs control.
    
    Parameters
    ----------
    all_plateau_data : list of dicts with keys:
        - 'br_idx': condition identifier
        - 'plateau_times': array of per-trial plateau times
        - 'target': 'A' or 'B'
        - 'condition_info': dict from get_condition_info()
    
    Returns
    -------
    dict with statistical test results
    """
    results = {
        'comparisons': [],
        'control_data': {},
    }
    
    # Group by target
    for target in ['A', 'B']:
        target_data = [d for d in all_plateau_data if d['target'] == target]
        
        if len(target_data) < 2:
            print(f"  [stats] Target {target}: Only {len(target_data)} conditions, skipping")
            continue
        
        # Find control condition(s)
        control_data = [d for d in target_data 
                        if d['condition_info']['condition_type'] == 'control']
        stim_data = [d for d in target_data 
                     if d['condition_info']['condition_type'] != 'control']
        
        print(f"  [stats] Target {target}: {len(control_data)} control, {len(stim_data)} stim conditions")
        
        if not control_data:
            print(f"  [warn] Target {target}: No control conditions found!")
            print(f"         Condition types: {[d['condition_info']['condition_type'] for d in target_data]}")
            continue
            
        if not stim_data:
            print(f"  [warn] Target {target}: No stim conditions found")
            continue
        
        # Pool control trials if multiple control conditions
        control_times = np.concatenate([
            d['plateau_times'][~np.isnan(d['plateau_times'])] 
            for d in control_data
        ])
        
        if len(control_times) < 2:
            print(f"  [warn] Target {target}: Insufficient control trials ({len(control_times)})")
            continue
        
        results['control_data'][target] = {
            'times': control_times,
            'mean': np.mean(control_times),
            'std': np.std(control_times),
            'n': len(control_times),
            'br_indices': [d['br_idx'] for d in control_data],
        }
        
        # Compare each stim condition to control
        for stim_d in stim_data:
            stim_times = stim_d['plateau_times'][~np.isnan(stim_d['plateau_times'])]
            
            if len(stim_times) < 2:
                print(f"    [skip] BR{stim_d['br_idx']}: only {len(stim_times)} valid trials")
                continue
            
            # t-test
            try:
                t_stat, t_p = stats.ttest_ind(control_times, stim_times)
            except Exception as e:
                print(f"    [warn] t-test failed for BR{stim_d['br_idx']}: {e}")
                t_stat, t_p = np.nan, np.nan
            
            # Mann-Whitney U
            try:
                u_stat, u_p = stats.mannwhitneyu(control_times, stim_times, 
                                                  alternative='two-sided')
            except Exception as e:
                print(f"    [warn] Mann-Whitney failed for BR{stim_d['br_idx']}: {e}")
                u_stat, u_p = np.nan, np.nan
            
            # Effect size (Cohen's d)
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
            
            # Print significance
            if t_p < 0.05:
                sig = '***' if t_p < 0.001 else '**' if t_p < 0.01 else '*'
                print(f"    BR{stim_d['br_idx']}: p={t_p:.4f} {sig}")
    
    return results


# ---------------------------------------------------------------------
# PLOTTING: DIAGNOSTIC (PER-CONDITION)
# ---------------------------------------------------------------------

def plot_condition_diagnostic(cond_data: dict, 
                               condition_info: dict,
                               target: str) -> None:
    """
    Generate diagnostic plot for a single condition showing:
    - Individual trial traces with detected plateaus
    - Mean trace with plateau marker
    - Speed traces with threshold
    """
    t = cond_data['t']
    distance_all = cond_data['distance_all']
    speed_all = cond_data['speed_all']
    br_idx = cond_data.get('br_idx', 0)
    
    n_trials = distance_all.shape[0]
    if n_trials == 0:
        return
    
    # Run plateau detection
    plateau_times, summary = detect_plateau_per_trial(
        distance_all, speed_all, t, normalization_method='max'
    )
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Color setup
    cmap = plt.colormaps.get_cmap('tab10')
    
    # --- Top Left: Individual normalized distance traces ---
    ax1 = axes[0, 0]
    for i in range(min(n_trials, 20)):  # Limit to 20 traces for visibility
        alpha = 0.3 if n_trials > 10 else 0.5
        ax1.plot(t, summary['dist_norm'][i, :], color='gray', alpha=alpha, linewidth=0.8)
        if not np.isnan(plateau_times[i]):
            pt_idx = np.argmin(np.abs(t - plateau_times[i]))
            ax1.scatter(plateau_times[i], summary['dist_norm'][i, pt_idx], 
                       color='red', s=30, zorder=5, alpha=0.7)
    
    # Mean trace
    ax1.plot(t, summary['dist_mean'], color='blue', linewidth=2, label='Mean')
    ax1.fill_between(t, 
                     summary['dist_mean'] - summary['dist_ci'],
                     summary['dist_mean'] + summary['dist_ci'],
                     color='blue', alpha=0.2)
    
    # Mean plateau marker
    if not np.isnan(summary['plateau_mean']):
        pt_idx = np.argmin(np.abs(t - summary['plateau_mean']))
        ax1.axvline(summary['plateau_mean'], color='red', linestyle='--', 
                   alpha=0.7, label=f"Mean plateau: {summary['plateau_mean']:.0f}ms")
    
    ax1.axvline(0, color='black', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Time from IR crossing (ms)')
    ax1.set_ylabel('Normalized Distance')
    ax1.set_title(f'Distance Traces (n={n_trials}, detected={summary["n_detected"]})')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_ylim(0, 1.3)
    ax1.grid(True, alpha=0.3)
    
    # --- Top Right: Individual speed traces ---
    ax2 = axes[0, 1]
    for i in range(min(n_trials, 20)):
        alpha = 0.3 if n_trials > 10 else 0.5
        ax2.plot(t, summary['speed_norm'][i, :], color='gray', alpha=alpha, linewidth=0.8)
    
    # Mean speed
    ax2.plot(t, summary['speed_mean'], color='green', linewidth=2, label='Mean speed')
    
    # Show threshold from first trial as reference
    if summary['detection_results']:
        first_result = summary['detection_results'][0]
        ax2.axhline(first_result['speed_threshold'], color='red', linestyle='--', 
                   alpha=0.7, label=f"Threshold: {first_result['speed_threshold']:.3f}")
        ax2.axhline(first_result['baseline_speed'], color='orange', linestyle=':', 
                   alpha=0.7, label=f"Baseline: {first_result['baseline_speed']:.3f}")
    
    # Mark baseline window
    ax2.axvspan(BASELINE_WINDOW_START_MS, BASELINE_WINDOW_END_MS, 
               color='yellow', alpha=0.2, label='Baseline window')
    
    ax2.axvline(0, color='black', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time from IR crossing (ms)')
    ax2.set_ylabel('Normalized Speed')
    ax2.set_title('Speed Traces')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # --- Bottom Left: Histogram of plateau times ---
    ax3 = axes[1, 0]
    valid_times = plateau_times[~np.isnan(plateau_times)]
    if len(valid_times) > 0:
        ax3.hist(valid_times, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax3.axvline(summary['plateau_mean'], color='red', linewidth=2, 
                   label=f"Mean: {summary['plateau_mean']:.0f}±{summary['plateau_sem']:.0f}ms")
    ax3.set_xlabel('Plateau Time (ms)')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Plateau Time Distribution (detection rate: {summary["detection_rate"]*100:.0f}%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # --- Bottom Right: Summary text ---
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = [
        f"Condition: BR{br_idx}",
        f"Label: {condition_info['label']}",
        f"Type: {condition_info['condition_type']}",
        f"Target: {target}",
        "",
        f"N trials: {n_trials}",
        f"N detected: {summary['n_detected']} ({summary['detection_rate']*100:.0f}%)",
        "",
        f"Plateau time (mean±SEM): {summary['plateau_mean']:.1f} ± {summary['plateau_sem']:.1f} ms",
        f"Plateau time (std): {summary['plateau_std']:.1f} ms",
        "",
        "Detection parameters:",
        f"  Baseline window: {BASELINE_WINDOW_START_MS:.0f} to {BASELINE_WINDOW_END_MS:.0f} ms",
        f"  Plateau window: {PLATEAU_WINDOW_MS:.0f} ms",
        f"  Speed threshold factor: {SPEED_THRESHOLD_FACTOR}",
        f"  Position change threshold: {POSITION_CHANGE_THRESHOLD}",
    ]
    
    ax4.text(0.1, 0.9, '\n'.join(summary_text), 
             transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(f"Session: {PARAMS.session} | BR{br_idx}: {condition_info['label']} | Target {target}", 
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    
    # Save
    out_filename = f"Diagnostic_BR{br_idx:03d}_Target{target}.{OUTPUT_FORMAT}"
    fig.savefig(FIG_OUT_DIR / out_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"    Saved diagnostic: {out_filename}")


# ---------------------------------------------------------------------
# PLOTTING: SUMMARY PLOTS
# ---------------------------------------------------------------------

def plot_normalized_summary(all_conditions_data: list[dict],
                            metadata_df: pd.DataFrame,
                            target: str,
                            conditions_to_plot: list[int] | None = None,
                            title_suffix: str = "") -> list[dict]:
    """
    Plot normalized distance with per-trial plateau detection.
    Returns list of plateau data dicts.
    """
    if not all_conditions_data:
        print(f"[warn] No data for target {target} summary")
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
    all_conditions_data = sorted(all_conditions_data, 
                                  key=lambda x: x.get('br_idx', 0) or 0)
    
    n_conds = len(all_conditions_data)
    cmap = plt.colormaps.get_cmap('tab10')
    
    fig = Figure(figsize=(14, 8))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    
    plateau_results = []
    
    for i, cond_data in enumerate(all_conditions_data):
        t = cond_data['t']
        distance_all = cond_data['distance_all']
        speed_all = cond_data['speed_all']
        br_idx = cond_data.get('br_idx', i)
        
        # Get condition info from metadata
        condition_info = get_condition_info(br_idx, metadata_df)
        
        color = cmap(i % 10)
        n_trials = distance_all.shape[0]
        if n_trials == 0:
            continue
        
        # Per-trial plateau detection
        plateau_times, summary = detect_plateau_per_trial(
            distance_all, speed_all, t, normalization_method='max'
        )
        
        # Build label with condition info
        label = f"{condition_info['label']} (n={summary['n_detected']}/{n_trials})"
        
        # Plot normalized distance mean ± CI
        ax.plot(t, summary['dist_mean'], color=color, linewidth=2, label=label)
        ax.fill_between(t, 
                        summary['dist_mean'] - summary['dist_ci'],
                        summary['dist_mean'] + summary['dist_ci'], 
                        color=color, alpha=0.2)
        
        # Mark mean plateau time with error bar
        if not np.isnan(summary['plateau_mean']):
            pt_mean = summary['plateau_mean']
            pt_sem = summary['plateau_sem']
            pt_idx = np.argmin(np.abs(t - pt_mean))
            pt_y = summary['dist_mean'][pt_idx]
            
            ax.axvline(pt_mean, color=color, linestyle=':', alpha=0.5, linewidth=1.5)
            ax.errorbar(pt_mean, pt_y, xerr=pt_sem * 1.96, 
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
            'n_total': n_trials,
            'detection_rate': summary['detection_rate'],
        })
    
    # Formatting
    ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.8, label='t=0 (IR cross)')
    ax.set_xlabel('Time from IR crossing (ms)', fontsize=12)
    ax.set_ylabel('Normalized Distance [0-1]', fontsize=12)
    ax.set_title(f'Target {target}: Normalized Distance from Start{title_suffix}\n'
                 f'● = mean plateau time ± 95% CI (per-trial detection)', 
                 fontsize=12)
    ax.legend(fontsize='small', loc='upper left', ncol=2)
    ax.set_ylim(0, 1.3)
    ax.set_xlim(t.min(), t.max())
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Session: {PARAMS.session}", fontsize=10, y=0.98)
    fig.tight_layout()
    
    # Save
    if conditions_to_plot:
        cond_suffix = "_BR" + "_".join(str(c) for c in sorted(conditions_to_plot))
    else:
        cond_suffix = "_ALL"
    out_filename = f"Target_{target}{cond_suffix}_Normalized.{OUTPUT_FORMAT}"
    fig.savefig(FIG_OUT_DIR / out_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {out_filename}")
    
    return plateau_results


def plot_plateau_boxplot(all_plateau_data: list[dict],
                         stats_results: dict,
                         title_suffix: str = "") -> None:
    """
    Create box plot showing plateau times with control vs stim comparisons.
    X-axis labels show condition type and parameters.
    """
    if not all_plateau_data:
        print("[warn] No plateau data to plot")
        return
    
    # Separate by target
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
        
        # Calculate figure width based on number of conditions
        fig_width = max(10, n_conditions * 0.8 + 2)
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        # Prepare data
        box_data = []
        labels = []
        colors = []
        br_indices = []
        
        for d in sorted_data:
            valid_times = d['plateau_times'][~np.isnan(d['plateau_times'])]
            box_data.append(valid_times if len(valid_times) > 0 else [np.nan])
            
            # Label with condition info (single line, no duplicate BR)
            cinfo = d['condition_info']
            if cinfo['condition_type'] == 'control':
                labels.append(f"Control\n(BR{d['br_idx']})")
                colors.append('lightgreen')
            elif cinfo['condition_type'] == 'continuous_stim':
                labels.append(f"{cinfo['label']}\n(BR{d['br_idx']})")
                colors.append('lightsalmon')
            else:
                # Pulsed stim
                labels.append(f"{cinfo['label']}\n(BR{d['br_idx']})")
                colors.append('steelblue')
            
            br_indices.append(d['br_idx'])
        
        # Create box plot
        bp = ax.boxplot(box_data, positions=range(len(box_data)), 
                       widths=0.6, patch_artist=True, manage_ticks=False)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add individual data points
        np.random.seed(42)
        for i, (data, color) in enumerate(zip(box_data, colors)):
            if len(data) > 0 and not np.all(np.isnan(data)):
                valid_data = data[~np.isnan(data)] if hasattr(data, '__len__') else [data]
                if len(valid_data) > 0:
                    jitter = np.random.uniform(-0.15, 0.15, len(valid_data))
                    ax.scatter(np.full(len(valid_data), i) + jitter, valid_data, 
                              color='black', alpha=0.4, s=20, edgecolor='white', linewidth=0.5)
        
        # Find y_max for significance annotation positioning
        all_valid = [v for d in box_data for v in (d if hasattr(d, '__iter__') else [d]) 
                     if not np.isnan(v)]
        y_max = max(all_valid) if all_valid else 400
        y_offset_base = y_max * 0.05
        
        # Add significance markers
        if stats_results and stats_results['comparisons']:
            target_comparisons = [c for c in stats_results['comparisons'] if c['target'] == target]
            
            for comp in target_comparisons:
                stim_br = comp['stim_br_idx']
                # Find position of this stim condition
                stim_pos = None
                for j, br in enumerate(br_indices):
                    if br == stim_br:
                        stim_pos = j
                        break
                
                if stim_pos is not None:
                    p_val = comp['t_p']
                    if not np.isnan(p_val) and p_val < 0.05:
                        # Determine significance level
                        if p_val < 0.001:
                            sig_marker = '***'
                        elif p_val < 0.01:
                            sig_marker = '**'
                        else:
                            sig_marker = '*'
                        
                        # Get max value for this condition to position star above
                        cond_data = box_data[stim_pos]
                        if len(cond_data) > 0 and not np.all(np.isnan(cond_data)):
                            cond_max = np.nanmax(cond_data)
                            y_pos = cond_max + y_offset_base
                        else:
                            y_pos = y_max + y_offset_base
                        
                        ax.annotate(sig_marker, xy=(stim_pos, y_pos),
                                   ha='center', va='bottom', fontsize=14, fontweight='bold',
                                   color='red')
                        
                        # Also add p-value below the star
                        ax.annotate(f'p={p_val:.3f}', xy=(stim_pos, y_pos + y_offset_base * 0.5),
                                   ha='center', va='bottom', fontsize=7, color='darkred')
        
        # Formatting
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Plateau Time from IR crossing (ms)', fontsize=12)
        ax.set_title(f'Target {target}: Plateau Time Distribution{title_suffix}\n'
                    f'(* p<0.05, ** p<0.01, *** p<0.001 vs Control)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0, top=y_max * 1.25)  # Add space for annotations
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', alpha=0.7, edgecolor='black', label='Control'),
            Patch(facecolor='steelblue', alpha=0.7, edgecolor='black', label='Pulsed Stim'),
            Patch(facecolor='lightsalmon', alpha=0.7, edgecolor='black', label='Continuous Stim'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add control stats annotation
        if target in stats_results.get('control_data', {}):
            ctrl = stats_results['control_data'][target]
            ctrl_text = f"Control: {ctrl['mean']:.0f}±{ctrl['std']:.0f}ms (n={ctrl['n']})"
            ax.text(0.02, 0.98, ctrl_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        fig.tight_layout()
        
        # Save
        suffix = title_suffix.replace(' ', '_').replace(':', '')
        out_filename = f"Plateau_BoxPlot_Target{target}{suffix}.{OUTPUT_FORMAT}"
        fig.savefig(FIG_OUT_DIR / out_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {out_filename}")


def save_statistics_csv(all_plateau_data: list[dict], 
                        stats_results: dict,
                        title_suffix: str = "") -> None:
    """Save summary statistics and comparisons to CSV."""
    
    # Summary per condition
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
            'n_channels': cinfo['n_channels'],
            'frequency_hz': cinfo['frequency_hz'],
            'duration_ms': cinfo['duration_ms'],
            'n_pulses': cinfo['n_pulses'],
            'n_trials': d['n_total'],
            'n_detected': d['n_detected'],
            'detection_rate': d['detection_rate'],
            'plateau_mean_ms': d['plateau_mean'],
            'plateau_std_ms': d['plateau_std'],
            'plateau_sem_ms': d['plateau_sem'],
            'plateau_median_ms': np.nanmedian(valid_times) if len(valid_times) > 0 else np.nan,
        })
    
    df = pd.DataFrame(rows)
    suffix = title_suffix.replace(' ', '_').replace(':', '')
    csv_filename = FIG_OUT_DIR / f"Plateau_Summary{suffix}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"  Saved: {csv_filename}")
    
    # Comparisons
    if stats_results and stats_results['comparisons']:
        df_comp = pd.DataFrame(stats_results['comparisons'])
        csv_comp_filename = FIG_OUT_DIR / f"Plateau_ControlVsStim{suffix}.csv"
        df_comp.to_csv(csv_comp_filename, index=False)
        print(f"  Saved: {csv_comp_filename}")


def print_statistics_summary(stats_results: dict) -> None:
    """Print statistical comparison results to console."""
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY: Control vs Stim Comparisons")
    print("="*70)
    
    if not stats_results or not stats_results['comparisons']:
        print("  No comparisons available")
        print("  Check that CONTROL_BR_INDICES is set correctly")
        return
    
    for target in ['A', 'B']:
        target_comps = [c for c in stats_results['comparisons'] if c['target'] == target]
        if not target_comps:
            continue
        
        print(f"\n--- Target {target} ---")
        
        if target in stats_results.get('control_data', {}):
            ctrl = stats_results['control_data'][target]
            print(f"Control: {ctrl['mean']:.1f} ± {ctrl['std']:.1f} ms (n={ctrl['n']}, BR: {ctrl['br_indices']})")
        
        print(f"\n{'Condition':<25} {'Mean±SD':<15} {'Diff':<10} {'t-test p':<12} {'M-W p':<12} {'Cohen d':<10}")
        print("-"*84)
        
        for c in target_comps:
            sig_t = '***' if c['t_p'] < 0.001 else '**' if c['t_p'] < 0.01 else '*' if c['t_p'] < 0.05 else ''
            sig_u = '***' if c['u_p'] < 0.001 else '**' if c['u_p'] < 0.01 else '*' if c['u_p'] < 0.05 else ''
            
            print(f"{c['stim_label']:<25} "
                  f"{c['stim_mean']:.0f}±{c['stim_std']:.0f} ms    "
                  f"{c['diff_ms']:+.0f} ms   "
                  f"{c['t_p']:.4f}{sig_t:<3} "
                  f"{c['u_p']:.4f}{sig_u:<3} "
                  f"{c['cohens_d']:+.2f}")
    
    print("\n" + "="*70)


# ---------------------------------------------------------------------
# FILE PROCESSING
# ---------------------------------------------------------------------

def process_single_file(p: Path, target: str, target_code: int) -> Optional[dict]:
    """Process a single file and return data for summary."""
    global _DIAG_PRINTED
    
    try:
        with np.load(p, allow_pickle=True) as data:
            # Determine Condition
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
            
            # Get keypoint names
            kp_names = _get_keypoint_names_from_data(data, CAMERA_TO_USE)
            if not kp_names:
                kp_names = [f"kp_{i}" for i in range(pos_segs.shape[1])]
            
            idx_x = find_keypoint_index(kp_names, f"{KEYPOINT_BASE}_x")
            idx_y = find_keypoint_index(kp_names, f"{KEYPOINT_BASE}_y")
            
            # Diagnostic print once
            diag_key = f"{PARAMS.session}_{p.name}"
            if diag_key not in _DIAG_PRINTED:
                print(f"  File: {p.name}, BR: {br_idx}")
                _DIAG_PRINTED.add(diag_key)
            
            if idx_x is None or idx_y is None:
                return None
            
            dt = t_axis[1] - t_axis[0] if len(t_axis) > 1 else 10.0
            if dt < 0.9:
                dt *= 1000.0
            
            n_trials = pos_segs.shape[0]
            distance_all = []
            speed_all = []
            
            for i in range(n_trials):
                x = pos_segs[i, idx_x, :]
                y = pos_segs[i, idx_y, :]

                if np.sum(np.isfinite(x)) < 10 or np.sum(np.isfinite(y)) < 10:
                    continue
                
                try:
                    metrics = compute_2d_metrics(x, y, dt)
                    distance_all.append(metrics['distance_from_start'])
                    speed_all.append(metrics['speed_2d'])
                except (ValueError, np.linalg.LinAlgError):
                    continue
            
            if len(distance_all) == 0:
                return None
            
            distance_all = np.array(distance_all)
            speed_all = np.array(speed_all)
            
            return {
                "file": p.name,
                "n_trials": len(distance_all),
                "cond": cond,
                "br_idx": br_idx,
                "t": t_axis,
                "distance_all": distance_all,
                "speed_all": speed_all,
            }
    
    except Exception as e:
        print(f"Error processing {p.name}: {e}")
        return None


def process_target(target: str, target_code: int):
    """Process all files for a given target."""
    candidate_subdirs = [
        Path("stim_reaches") / f"target_{target}",
        Path("control_reaches") / f"target_{target}",
        Path("continuous_stim") / f"target_{target}",
        Path("at_rest"),
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
        res = process_single_file(p, target, target_code)
        if res:
            all_conditions_data.append(res)
    
    return all_conditions_data


def main():
    print(f"=" * 70)
    print(f"Plateau Detection Analysis with Control vs Stim Comparisons")
    print(f"=" * 70)
    print(f"Session: {PARAMS.session}")
    print(f"Output Dir: {FIG_OUT_DIR}")
    print(f"Metadata CSV: {METADATA_CSV}")
    print(f"Keypoint: {KEYPOINT_BASE}")
    print(f"Baseline window: {BASELINE_WINDOW_START_MS:.0f} to {BASELINE_WINDOW_END_MS:.0f} ms")
    print(f"Control BR indices: {CONTROL_BR_INDICES}")
    print("")
    
    # Load metadata
    try:
        metadata_df = load_metadata_csv(METADATA_CSV)
        print(f"Loaded metadata: {len(metadata_df)} rows")
    except Exception as e:
        print(f"[warn] Could not load metadata CSV: {e}")
        metadata_df = pd.DataFrame()
    
    # Process each target
    target_A_data = process_target("A", 1)
    target_B_data = process_target("B", 2)
    
    # Print condition type summary
    print("\n--- Condition Type Summary ---")
    for data_list, tgt in [(target_A_data, 'A'), (target_B_data, 'B')]:
        print(f"Target {tgt}:")
        for d in data_list:
            cinfo = get_condition_info(d['br_idx'], metadata_df)
            print(f"  BR{d['br_idx']}: {cinfo['condition_type']} - {cinfo['label']}")
    
    # Generate diagnostic plots per condition
    if GENERATE_DIAGNOSTIC_PLOTS:
        print("\nGenerating diagnostic plots per condition...")
        for cond_data in target_A_data:
            cinfo = get_condition_info(cond_data['br_idx'], metadata_df)
            plot_condition_diagnostic(cond_data, cinfo, "A")
        for cond_data in target_B_data:
            cinfo = get_condition_info(cond_data['br_idx'], metadata_df)
            plot_condition_diagnostic(cond_data, cinfo, "B")
    
    # ---- MASTER PLOT (all conditions) ----
    all_plateau_data_master = []
    if GENERATE_MASTER_PLOT:
        print("\nGenerating MASTER plots (all conditions)...")
        if target_A_data:
            plateau_A = plot_normalized_summary(
                target_A_data, metadata_df, "A", 
                conditions_to_plot=None, title_suffix=" (All Conditions)"
            )
            all_plateau_data_master.extend(plateau_A)
        if target_B_data:
            plateau_B = plot_normalized_summary(
                target_B_data, metadata_df, "B", 
                conditions_to_plot=None, title_suffix=" (All Conditions)"
            )
            all_plateau_data_master.extend(plateau_B)
        
        # Statistics for master
        if all_plateau_data_master:
            print("\nComputing statistics (all conditions)...")
            stats_master = compute_control_vs_stim_statistics(all_plateau_data_master)
            plot_plateau_boxplot(all_plateau_data_master, stats_master, "_ALL")
            print_statistics_summary(stats_master)
            if GENERATE_STATS_SUMMARY:
                save_statistics_csv(all_plateau_data_master, stats_master, "_ALL")
    
    # ---- SUBSET PLOT (selected conditions) ----
    if GENERATE_SUBSET_PLOT and SUBSET_CONDITIONS:
        print(f"\nGenerating SUBSET plots (conditions: {SUBSET_CONDITIONS})...")
        all_plateau_data_subset = []
        if target_A_data:
            plateau_A = plot_normalized_summary(
                target_A_data, metadata_df, "A", 
                conditions_to_plot=SUBSET_CONDITIONS, title_suffix=" (Subset)"
            )
            all_plateau_data_subset.extend(plateau_A)
        if target_B_data:
            plateau_B = plot_normalized_summary(
                target_B_data, metadata_df, "B", 
                conditions_to_plot=SUBSET_CONDITIONS, title_suffix=" (Subset)"
            )
            all_plateau_data_subset.extend(plateau_B)
        
        if all_plateau_data_subset:
            print("\nComputing statistics (subset)...")
            stats_subset = compute_control_vs_stim_statistics(all_plateau_data_subset)
            cond_str = "_BR" + "_".join(str(c) for c in sorted(SUBSET_CONDITIONS))
            plot_plateau_boxplot(all_plateau_data_subset, stats_subset, cond_str)
            print_statistics_summary(stats_subset)
            if GENERATE_STATS_SUMMARY:
                save_statistics_csv(all_plateau_data_subset, stats_subset, cond_str)
    
    print(f"\n{'=' * 70}")
    print(f"Done! Figures saved to {FIG_OUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()