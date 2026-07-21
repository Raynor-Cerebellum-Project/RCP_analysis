import numpy as np
import warnings
import re
from pathlib import Path
from typing import Optional
from RCP_analysis.python.functions.config_loading import *
import concurrent.futures
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

"""
Purpose:
  Visualize combined X+Y hand position to better detect reach stops.
  
  Shows multiple representations:
  1. Distance from start position over time
  2. 2D speed (velocity magnitude) over time
  3. 2D trajectory (X vs Y, colored by time)
  4. Raw X and Y overlaid

Outputs:
  - Figures saved to `results/figures/position_2d_debug/`
"""

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
CAMERA_TO_USE = "cam1"

# Keypoints to combine - need both X and Y
KEYPOINT_BASE = "middle"  # Will look for middle_x AND middle_y

# Output Configuration
FIG_OUT_DIR = OUT_BASE / "figures" / "position_2d_debug"
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FORMAT = 'png'

plt.rcParams.update({'font.size': 10})

# ---------------------------------------------------------------------
# PLOT CONTROL
# ---------------------------------------------------------------------
GENERATE_DEBUG_PLOTS = False      # Per-trial 2x2 debug panels (slow, verbose)
GENERATE_SUMMARY_PLOTS = True     # Per-condition summary (all trials overlaid)
GENERATE_TARGET_SUMMARY = True    # Cross-condition summary per target (mean ± 95% CI)

# Condition filtering for TARGET SUMMARY plot only
# Set to None or [] to include all conditions
# Set to list of BR indices to include only those, e.g. [3, 12]
TARGET_SUMMARY_CONDITIONS = None  # None = all conditions
TARGET_SUMMARY_CONDITIONS = [22, 23, 24]  # Example: only BR 3 and 12

# ---------------------------------------------------------------------
# SMOOTHING PARAMETERS
# ---------------------------------------------------------------------
SMOOTHING_METHOD = 'savgol'  # 'savgol', 'gaussian', 'none'
SAVGOL_WINDOW_MS = 50.0      # Smoothing window
SAVGOL_POLYORDER = 3

# Velocity computation
VELOCITY_WINDOW_MS = 75.0

# ---------------------------------------------------------------------
# SESSION-SPECIFIC BR FILTERS
# ---------------------------------------------------------------------
SESSION_BR_FILTERS = {
    # "NRR_RW011": [4, 5, 6, 10, 11],
    # "NRR_RW012": [3, 7, 8, 9, 10, 11, 12, 18, 31],
    # "NRR_RW022": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
}

def get_br_filter_for_session():
    for session_key, br_list in SESSION_BR_FILTERS.items():
        if session_key in PARAMS.session:
            return br_list
    return None

BR_FILTER = get_br_filter_for_session()

_DIAG_PRINTED = set()


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def smooth_1d(arr: np.ndarray, dt: float) -> np.ndarray:
    """Apply smoothing to 1D array with robust error handling."""
    if SMOOTHING_METHOD == 'none':
        return arr.copy()
    
    # Check for problematic input
    if len(arr) < 5:
        warnings.warn(f"Array too short for smoothing ({len(arr)} points), returning copy")
        return arr.copy()
    
    # Check for all-NaN or all-constant
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        warnings.warn("Array contains no finite values, returning copy")
        return arr.copy()
    
    finite_vals = arr[finite_mask]
    if np.all(finite_vals == finite_vals[0]):
        # All values are identical - nothing to smooth
        return arr.copy()
    
    # Handle NaN values by interpolation before smoothing
    arr_clean = arr.copy()
    if not np.all(finite_mask):
        # Interpolate NaN values
        nans = ~finite_mask
        if np.sum(finite_mask) >= 2:  # Need at least 2 points to interpolate
            arr_clean[nans] = np.interp(
                np.flatnonzero(nans),
                np.flatnonzero(finite_mask),
                arr[finite_mask]
            )
        else:
            # Not enough points to interpolate
            return arr.copy()
    
    if SMOOTHING_METHOD == 'savgol':
        window_length = int(SAVGOL_WINDOW_MS / dt)
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(window_length, 5)
        
        # Ensure window_length doesn't exceed array length
        if window_length > len(arr_clean):
            window_length = len(arr_clean)
            if window_length % 2 == 0:
                window_length -= 1
            window_length = max(window_length, 5)
        
        # If still too short, fall back to gaussian
        if window_length > len(arr_clean):
            sigma = 20.0 / dt
            sigma = min(sigma, len(arr_clean) / 4)  # Limit sigma
            return gaussian_filter1d(arr_clean, max(sigma, 0.5))
        
        polyorder = min(SAVGOL_POLYORDER, window_length - 1)
        
        try:
            return savgol_filter(arr_clean, window_length, polyorder)
        except (np.linalg.LinAlgError, ValueError) as e:
            # Fallback to gaussian smoothing
            warnings.warn(f"Savgol filter failed ({e}), falling back to gaussian")
            sigma = 20.0 / dt
            sigma = min(sigma, len(arr_clean) / 4)
            return gaussian_filter1d(arr_clean, max(sigma, 0.5))
    
    elif SMOOTHING_METHOD == 'gaussian':
        sigma = 20.0 / dt
        sigma = min(sigma, len(arr_clean) / 4)  # Limit sigma for short arrays
        return gaussian_filter1d(arr_clean, max(sigma, 0.5))
    
    return arr_clean


def compute_velocity_1d(position: np.ndarray, dt: float) -> np.ndarray:
    """Compute velocity from position with robust error handling."""
    # Check for problematic input
    if len(position) < 7:
        # Use simple diff for very short arrays
        vel = np.gradient(position, dt)
        return vel
    
    # Check for NaN values
    finite_mask = np.isfinite(position)
    if not np.all(finite_mask):
        # Interpolate NaN values first
        position_clean = position.copy()
        nans = ~finite_mask
        if np.sum(finite_mask) >= 2:
            position_clean[nans] = np.interp(
                np.flatnonzero(nans),
                np.flatnonzero(finite_mask),
                position[finite_mask]
            )
        else:
            # Not enough points - return zeros
            return np.zeros_like(position)
    else:
        position_clean = position
    
    window_length = int(VELOCITY_WINDOW_MS / dt)
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(window_length, 7)
    
    # Ensure window doesn't exceed array length
    if window_length > len(position_clean):
        window_length = len(position_clean)
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(window_length, 3)
    
    polyorder = min(3, window_length - 1)
    
    try:
        return savgol_filter(position_clean, window_length, polyorder=polyorder, 
                            deriv=1, delta=dt)
    except (np.linalg.LinAlgError, ValueError) as e:
        # Fallback to numpy gradient
        warnings.warn(f"Savgol derivative failed ({e}), using np.gradient")
        return np.gradient(position_clean, dt)
        


def find_keypoint_index(kp_names: list, pattern: str) -> Optional[int]:
    """Find index of keypoint matching pattern (case-insensitive)."""
    pattern_lower = pattern.lower()
    for i, name in enumerate(kp_names):
        if pattern_lower in str(name).lower():
            return i
    return None


def get_kinematics_id(filename: str) -> str:
    """Generate unique ID for deduplication."""
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
    """Extract BR index from condition string."""
    for pattern in [r"Cond_(\d+)", r"BR_(\d+)", r"(\d+)"]:
        match = re.search(pattern, cond_str)
        if match:
            return int(match.group(1))
    return None


def _get_keypoint_names_from_data(data: dict, camera: str) -> list[str]:
    """Extract keypoint names from data file."""
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
    """
    Compute various 2D metrics from X and Y position.
    """
    # Validate inputs
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Empty position arrays")
    
    if len(x) != len(y):
        raise ValueError(f"X and Y length mismatch: {len(x)} vs {len(y)}")
    
    # Check if data is mostly NaN
    x_valid = np.sum(np.isfinite(x))
    y_valid = np.sum(np.isfinite(y))
    if x_valid < 5 or y_valid < 5:
        raise ValueError(f"Insufficient valid data points: X={x_valid}, Y={y_valid}")
    
    # Smooth positions (now with robust handling)
    x_smooth = smooth_1d(x, dt)
    y_smooth = smooth_1d(y, dt)
    
    # Distance from start
    x0, y0 = x_smooth[0], y_smooth[0]
    distance_from_start = np.sqrt((x_smooth - x0)**2 + (y_smooth - y0)**2)
    
    # Velocity components
    vx = compute_velocity_1d(x_smooth, dt)
    vy = compute_velocity_1d(y_smooth, dt)
    
    # 2D Speed (velocity magnitude)
    speed_2d = np.sqrt(vx**2 + vy**2)
    
    # Path length (cumulative)
    dx = np.diff(x_smooth)
    dy = np.diff(y_smooth)
    step_distances = np.sqrt(dx**2 + dy**2)
    path_length = np.concatenate([[0], np.cumsum(step_distances)])
    
    return {
        'x_raw': x,
        'y_raw': y,
        'x_smooth': x_smooth,
        'y_smooth': y_smooth,
        'distance_from_start': distance_from_start,
        'vx': vx,
        'vy': vy,
        'speed_2d': speed_2d,
        'path_length': path_length,
    }


# ---------------------------------------------------------------------
# NORMALIZATION AND PLATEAU DETECTION
# ---------------------------------------------------------------------

def normalize_distance_traces(distance_all: np.ndarray, t_axis: np.ndarray,
                               method: str = 'max') -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize distance traces to [0, 1] range.
    
    Parameters
    ----------
    distance_all : np.ndarray
        Shape (n_trials, T) - raw distance from start
    t_axis : np.ndarray
        Time axis in ms
    method : str
        'max' - divide by max distance per trial
        'endpoint' - divide by final distance per trial
        'percentile' - divide by 95th percentile per trial (robust to overshoots)
        'baseline_max' - divide by max during baseline period (pre-stim)
        
    Returns
    -------
    normalized : np.ndarray
        Shape (n_trials, T) - normalized distances
    scale_factors : np.ndarray
        Shape (n_trials,) - the normalization factors used
    """
    n_trials, T = distance_all.shape
    normalized = np.zeros_like(distance_all)
    scale_factors = np.zeros(n_trials)
    
    for i in range(n_trials):
        trace = distance_all[i, :]
        
        if method == 'max':
            scale = np.nanmax(trace)
        elif method == 'endpoint':
            # Use mean of last 100ms as endpoint
            t_end_mask = t_axis >= (t_axis[-1] - 100)
            scale = np.nanmean(trace[t_end_mask])
        elif method == 'percentile':
            scale = np.nanpercentile(trace, 95)
        elif method == 'baseline_max':
            # Max distance reached after stim onset
            post_stim_mask = t_axis >= 0
            scale = np.nanmax(trace[post_stim_mask])
        else:
            scale = np.nanmax(trace)
        
        # Avoid division by zero
        scale = max(scale, 1e-6)
        scale_factors[i] = scale
        normalized[i, :] = trace / scale
    
    return normalized, scale_factors


def detect_plateau_time(mean_trace: np.ndarray, speed_trace: np.ndarray, 
                        t_axis: np.ndarray, ci_trace: np.ndarray,
                        baseline_window_ms: float = 200.0,
                        plateau_window_ms: float = 150.0,
                        speed_threshold_factor: float = 0.1,
                        position_change_threshold: float = 0.02) -> dict:
    """
    Detect when the reach has plateaued based on your criteria:
    1. Speed is similar to pre-stimulus baseline (-200ms to 0)
    2. Position is relatively unchanging for ~150-200ms
    
    Parameters
    ----------
    mean_trace : np.ndarray
        Mean normalized distance trace (T,)
    speed_trace : np.ndarray
        Mean speed trace (T,)
    t_axis : np.ndarray
        Time axis in ms
    ci_trace : np.ndarray
        95% CI half-width for distance (T,)
    baseline_window_ms : float
        Window before t=0 to compute baseline speed
    plateau_window_ms : float
        Duration position must be stable to count as plateau
    speed_threshold_factor : float
        Speed must be within this factor of baseline speed
    position_change_threshold : float
        Max allowed change in normalized position over plateau window
        
    Returns
    -------
    dict with:
        - plateau_time: time (ms) when plateau is detected, or None
        - plateau_idx: index in t_axis
        - baseline_speed: mean speed during baseline
        - method: which criterion triggered ('speed', 'position', 'both')
        - confidence: quality metric
    """
    dt = t_axis[1] - t_axis[0]
    
    # 1. Compute baseline speed (pre-stimulus)
    baseline_mask = (t_axis >= -baseline_window_ms) & (t_axis < 0)
    if not np.any(baseline_mask):
        # If no baseline period, use first 100ms
        baseline_mask = t_axis < t_axis[0] + 100
    
    baseline_speed = np.nanmean(speed_trace[baseline_mask])
    baseline_speed_std = np.nanstd(speed_trace[baseline_mask])
    
    # Speed threshold: must be close to baseline
    speed_threshold = baseline_speed + 2 * baseline_speed_std + speed_threshold_factor * np.nanmax(speed_trace)
    
    # 2. Compute position stability window size
    plateau_samples = int(plateau_window_ms / dt)
    
    # 3. Scan from peak onwards to find plateau
    # Start search after the peak distance
    peak_idx = np.nanargmax(mean_trace)
    
    plateau_time = None
    plateau_idx = None
    method = None
    
    # Search for plateau after peak
    for i in range(peak_idx, len(t_axis) - plateau_samples):
        # Check speed criterion
        current_speed = speed_trace[i]
        speed_ok = current_speed <= speed_threshold
        
        # Check position stability criterion
        # Position should not change much over the next plateau_window_ms
        window_slice = slice(i, i + plateau_samples)
        position_range = np.nanmax(mean_trace[window_slice]) - np.nanmin(mean_trace[window_slice])
        position_ok = position_range < position_change_threshold
        
        if speed_ok and position_ok:
            plateau_time = t_axis[i]
            plateau_idx = i
            method = 'both'
            break
        elif speed_ok and plateau_time is None:
            # Store first speed-based candidate
            plateau_time = t_axis[i]
            plateau_idx = i
            method = 'speed'
    
    # Compute confidence based on how well criteria are met
    confidence = 0.0
    if plateau_idx is not None:
        # How much below speed threshold?
        speed_margin = (speed_threshold - speed_trace[plateau_idx]) / speed_threshold
        # How stable is position?
        if plateau_idx + plateau_samples < len(mean_trace):
            pos_stability = 1.0 - (np.nanstd(mean_trace[plateau_idx:plateau_idx+plateau_samples]) / 
                                    np.nanmean(mean_trace[plateau_idx:plateau_idx+plateau_samples]))
        else:
            pos_stability = 0.5
        
        confidence = 0.5 * np.clip(speed_margin, 0, 1) + 0.5 * np.clip(pos_stability, 0, 1)
    
    return {
        'plateau_time': plateau_time,
        'plateau_idx': plateau_idx,
        'baseline_speed': baseline_speed,
        'speed_threshold': speed_threshold,
        'method': method,
        'confidence': confidence,
    }


def compute_normalized_summary(distance_all: np.ndarray, speed_all: np.ndarray,
                                t_axis: np.ndarray,
                                normalization_method: str = 'max') -> dict:
    """
    Compute normalized summary statistics for a condition.
    
    Returns dict with normalized traces, mean, CI, and plateau detection.
    """
    # Normalize distance traces
    dist_norm, scale_factors = normalize_distance_traces(
        distance_all, t_axis, method=normalization_method
    )
    
    # Normalize speed by the same factors (to maintain relationship)
    n_trials = speed_all.shape[0]
    speed_norm = np.zeros_like(speed_all)
    for i in range(n_trials):
        speed_norm[i, :] = speed_all[i, :] / scale_factors[i]
    
    # Compute mean and CI
    dist_mean = np.nanmean(dist_norm, axis=0)
    dist_sem = np.nanstd(dist_norm, axis=0) / np.sqrt(n_trials)
    dist_ci = 1.96 * dist_sem
    
    speed_mean = np.nanmean(speed_norm, axis=0)
    speed_sem = np.nanstd(speed_norm, axis=0) / np.sqrt(n_trials)
    speed_ci = 1.96 * speed_sem
    
    # Detect plateau
    plateau_info = detect_plateau_time(
        dist_mean, speed_mean, t_axis, dist_ci,
        baseline_window_ms=200.0,
        plateau_window_ms=150.0,
    )
    
    return {
        'distance_normalized': dist_norm,
        'speed_normalized': speed_norm,
        'scale_factors': scale_factors,
        'dist_mean': dist_mean,
        'dist_ci': dist_ci,
        'speed_mean': speed_mean,
        'speed_ci': speed_ci,
        'plateau': plateau_info,
        't': t_axis,
    }


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def plot_2d_trajectory_colored(ax, x, y, t, title="2D Trajectory"):
    """Plot X vs Y trajectory colored by time."""
    # Create line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Normalize time for colormap
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-10)
    
    # Create colored line collection
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, 1))
    lc.set_array(t_norm[:-1])
    lc.set_linewidth(2)
    
    ax.add_collection(lc)
    ax.autoscale()
    
    # Mark start and end
    ax.plot(x[0], y[0], 'go', markersize=10, label='Start', zorder=5)
    ax.plot(x[-1], y[-1], 'r*', markersize=12, label='End', zorder=5)
    
    # Mark t=0 (stim onset) if in range
    t0_idx = np.argmin(np.abs(t))
    if 0 < t0_idx < len(x) - 1:
        ax.plot(x[t0_idx], y[t0_idx], 'c^', markersize=10, label='t=0 (stim)', zorder=5)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title)
    ax.legend(fontsize='x-small')
    ax.set_aspect('equal', adjustable='box')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(t.min(), t.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Time (ms)', shrink=0.8)


def plot_trial_debug(trial_data: dict, trial_idx: int, fig, axes):
    """
    Plot all 2D metrics for a single trial.
    
    axes should be a 2x2 array of axes:
    [0,0]: X and Y position vs time
    [0,1]: Distance from start vs time
    [1,0]: 2D Speed vs time
    [1,1]: 2D Trajectory (X vs Y)
    """
    t = trial_data['t']
    metrics = trial_data['metrics']
    
    # --- Panel 1: X and Y Position vs Time ---
    ax1 = axes[0, 0]
    ax1.plot(t, metrics['x_raw'], 'b-', alpha=0.3, linewidth=0.8, label='X raw')
    ax1.plot(t, metrics['y_raw'], 'r-', alpha=0.3, linewidth=0.8, label='Y raw')
    ax1.plot(t, metrics['x_smooth'], 'b-', alpha=0.9, linewidth=1.5, label='X smooth')
    ax1.plot(t, metrics['y_smooth'], 'r-', alpha=0.9, linewidth=1.5, label='Y smooth')
    ax1.axvline(0, color='green', linestyle='--', alpha=0.7, label='t=0')
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Position')
    ax1.set_title(f'Trial {trial_idx}: X & Y Position')
    ax1.legend(fontsize='xx-small', loc='upper left')
    
    # --- Panel 2: Distance from Start ---
    ax2 = axes[0, 1]
    ax2.plot(t, metrics['distance_from_start'], 'purple', linewidth=1.5)
    ax2.axvline(0, color='green', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Distance from Start')
    ax2.set_title(f'Trial {trial_idx}: Distance from Start')
    ax2.set_ylim(bottom=0)
    
    # Mark max distance
    idx_max = np.argmax(metrics['distance_from_start'])
    ax2.plot(t[idx_max], metrics['distance_from_start'][idx_max], 'r*', markersize=10)
    ax2.annotate(f'Max: {metrics["distance_from_start"][idx_max]:.2f}\nt={t[idx_max]:.0f}ms',
                 xy=(t[idx_max], metrics['distance_from_start'][idx_max]),
                 xytext=(5, -10), textcoords='offset points', fontsize=8)
    
    # --- Panel 3: 2D Speed ---
    ax3 = axes[1, 0]
    ax3.plot(t, metrics['speed_2d'], 'orange', linewidth=1.5)
    ax3.axvline(0, color='green', linestyle='--', alpha=0.7)
    ax3.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Speed (2D)')
    ax3.set_title(f'Trial {trial_idx}: 2D Speed')
    ax3.set_ylim(bottom=0)
    
    # Mark peak speed
    idx_peak = np.argmax(metrics['speed_2d'])
    ax3.plot(t[idx_peak], metrics['speed_2d'][idx_peak], 'c^', markersize=8)
    
    # Mark where speed drops below threshold (potential "stop")
    peak_speed = metrics['speed_2d'][idx_peak]
    threshold = peak_speed * 0.1  # 10% of peak
    ax3.axhline(threshold, color='gray', linestyle=':', alpha=0.5)
    
    # Find first crossing below threshold after peak
    below_thresh = np.where(metrics['speed_2d'][idx_peak:] < threshold)[0]
    if len(below_thresh) > 0:
        stop_idx = idx_peak + below_thresh[0]
        ax3.axvline(t[stop_idx], color='red', linestyle='--', alpha=0.7, label=f'Stop: {t[stop_idx]:.0f}ms')
        ax3.legend(fontsize='xx-small')
    
    # --- Panel 4: 2D Trajectory ---
    ax4 = axes[1, 1]
    plot_2d_trajectory_colored(ax4, metrics['x_smooth'], metrics['y_smooth'], t, 
                               title=f'Trial {trial_idx}: 2D Trajectory')


def plot_condition_summary(all_trial_data: list, t_axis: np.ndarray, 
                           target: str, cond: str, n_trials: int,
                           file_stem: str) -> None:
    """
    Plot summary figure showing all trials overlaid for one condition.
    """
    fig_summary = Figure(figsize=(20, 10))
    FigureCanvasAgg(fig_summary)
    
    ax_dist = fig_summary.add_subplot(2, 1, 1)
    ax_speed = fig_summary.add_subplot(2, 1, 2)
    
    cmap = cm.get_cmap('tab20', n_trials)
    
    for i, trial_data in enumerate(all_trial_data):
        t = trial_data['t']
        metrics = trial_data['metrics']
        color = cmap(i % 20)
        alpha = 0.7
        
        ax_dist.plot(t, metrics['distance_from_start'], color=color, alpha=alpha, 
                    linewidth=1, label=f'T{i}' if i < 10 else None)
        ax_speed.plot(t, metrics['speed_2d'], color=color, alpha=alpha, 
                     linewidth=1, label=f'T{i}' if i < 10 else None)
    
    ax_dist.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.8, label='t=0')
    ax_dist.set_xlabel('Time (ms)')
    ax_dist.set_ylabel('Distance from Start')
    ax_dist.set_title(f'All Trials: Distance from Start')
    ax_dist.legend(fontsize='xx-small', loc='upper left', ncol=5)
    ax_dist.set_ylim(bottom=0)
    
    ax_speed.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.8, label='t=0')
    ax_speed.set_xlabel('Time (ms)')
    ax_speed.set_ylabel('2D Speed')
    ax_speed.set_title(f'All Trials: 2D Speed')
    ax_speed.legend(fontsize='xx-small', loc='upper left', ncol=5)
    ax_speed.set_ylim(bottom=0)
    
    fig_summary.suptitle(f"2D Summary: {target} - {cond} ({n_trials} trials)\n"
                        f"Keypoint: {KEYPOINT_BASE}", fontsize=14)
    fig_summary.tight_layout()
    
    safe_cond = "".join([c if c.isalnum() else "_" for c in cond])
    fig_summary.savefig(FIG_OUT_DIR / f"2D_Summary_{target}_{safe_cond}_{file_stem}.{OUTPUT_FORMAT}",
                       dpi=150, bbox_inches='tight')
    plt.close(fig_summary)


def plot_target_summary(all_conditions_data: list[dict], target: str,
                        conditions_to_plot: list[int] | None = None) -> None:
    """
    Plot cross-condition summary for one target: mean ± 95% CI for each condition.
    
    Parameters
    ----------
    all_conditions_data : list of dicts with keys:
        - 'cond': condition string
        - 'br_idx': BR index (for sorting/coloring)
        - 't': time axis (T,)
        - 'distance_all': (n_trials, T) distance from start for all trials
        - 'speed_all': (n_trials, T) 2D speed for all trials
    target : str, e.g. "A" or "B"
    conditions_to_plot : list of BR indices to include, or None for all
    """
    if not all_conditions_data:
        print(f"[warn] No data for target {target} summary")
        return
    
    # Filter conditions if specified
    if conditions_to_plot:
        filtered_data = [d for d in all_conditions_data 
                         if d.get('br_idx') in conditions_to_plot]
        if not filtered_data:
            print(f"[warn] No conditions matched filter {conditions_to_plot} for target {target}")
            return
        print(f"  Filtering to conditions: {conditions_to_plot} ({len(filtered_data)}/{len(all_conditions_data)} conditions)")
        all_conditions_data = filtered_data
    
    # Sort by BR index if available
    all_conditions_data = sorted(all_conditions_data, 
                                  key=lambda x: x.get('br_idx', 0) or 0)
    
    n_conds = len(all_conditions_data)
    cmap = cm.get_cmap('tab10', max(n_conds, 10))
    
    fig = Figure(figsize=(16, 12))
    FigureCanvasAgg(fig)
    
    ax_dist = fig.add_subplot(2, 1, 1)
    ax_speed = fig.add_subplot(2, 1, 2)
    
    for i, cond_data in enumerate(all_conditions_data):
        cond = cond_data['cond']
        t = cond_data['t']
        distance_all = cond_data['distance_all']  # (n_trials, T)
        speed_all = cond_data['speed_all']        # (n_trials, T)
        br_idx = cond_data.get('br_idx', i)
        
        color = cmap(i % 10)
        
        n_trials = distance_all.shape[0]
        if n_trials == 0:
            continue
        
        # Compute mean and SEM
        dist_mean = np.nanmean(distance_all, axis=0)
        dist_sem = np.nanstd(distance_all, axis=0) / np.sqrt(n_trials)
        dist_ci = 1.96 * dist_sem
        
        speed_mean = np.nanmean(speed_all, axis=0)
        speed_sem = np.nanstd(speed_all, axis=0) / np.sqrt(n_trials)
        speed_ci = 1.96 * speed_sem
        
        # Create label
        label = f"BR{br_idx} (n={n_trials})" if br_idx is not None else f"{cond[:20]} (n={n_trials})"
        
        # Plot distance
        ax_dist.plot(t, dist_mean, color=color, linewidth=2, label=label)
        ax_dist.fill_between(t, dist_mean - dist_ci, dist_mean + dist_ci, 
                             color=color, alpha=0.2)
        
        # Plot speed
        ax_speed.plot(t, speed_mean, color=color, linewidth=2, label=label)
        ax_speed.fill_between(t, speed_mean - speed_ci, speed_mean + speed_ci,
                              color=color, alpha=0.2)
    
    # Formatting
    ax_dist.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='t=0')
    ax_dist.set_xlabel('Time (ms)', fontsize=12)
    ax_dist.set_ylabel('Distance from Start', fontsize=12)
    ax_dist.set_title(f'Target {target}: Distance from Start (Mean ± 95% CI)', fontsize=14)
    ax_dist.legend(fontsize='small', loc='upper left', ncol=2)
    ax_dist.set_ylim(bottom=0)
    ax_dist.grid(True, alpha=0.3)
    
    ax_speed.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='t=0')
    ax_speed.set_xlabel('Time (ms)', fontsize=12)
    ax_speed.set_ylabel('2D Speed', fontsize=12)
    ax_speed.set_title(f'Target {target}: 2D Speed (Mean ± 95% CI)', fontsize=14)
    ax_speed.legend(fontsize='small', loc='upper left', ncol=2)
    ax_speed.set_ylim(bottom=0)
    ax_speed.grid(True, alpha=0.3)
    
    # Build filename suffix
    if conditions_to_plot:
        cond_suffix = "_BR" + "_".join(str(c) for c in sorted(conditions_to_plot))
    else:
        cond_suffix = "_AllConditions"
    
    fig.suptitle(f"Target {target} - Conditions Summary\n"
                 f"Session: {PARAMS.session} | Keypoint: {KEYPOINT_BASE}", fontsize=16)
    fig.tight_layout()
    
    out_filename = f"Target_{target}{cond_suffix}_Summary.{OUTPUT_FORMAT}"
    fig.savefig(FIG_OUT_DIR / out_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved target summary: {out_filename}")


def plot_target_summary_normalized(all_conditions_data: list[dict], target: str,
                                    conditions_to_plot: list[int] | None = None,
                                    normalization_method: str = 'max') -> None:
    """
    Plot cross-condition summary with normalized traces and plateau detection.
    """
    if not all_conditions_data:
        print(f"[warn] No data for target {target} summary")
        return
    
    # Filter conditions if specified
    if conditions_to_plot:
        filtered_data = [d for d in all_conditions_data 
                         if d.get('br_idx') in conditions_to_plot]
        if not filtered_data:
            print(f"[warn] No conditions matched filter {conditions_to_plot}")
            return
        all_conditions_data = filtered_data
    
    # Sort by BR index
    all_conditions_data = sorted(all_conditions_data, 
                                  key=lambda x: x.get('br_idx', 0) or 0)
    
    n_conds = len(all_conditions_data)
    cmap = cm.get_cmap('tab10', max(n_conds, 10))
    
    fig = Figure(figsize=(18, 14))
    FigureCanvasAgg(fig)
    
    # 4 panels: raw distance, normalized distance, raw speed, normalized speed
    ax_dist_raw = fig.add_subplot(2, 2, 1)
    ax_dist_norm = fig.add_subplot(2, 2, 2)
    ax_speed_raw = fig.add_subplot(2, 2, 3)
    ax_speed_norm = fig.add_subplot(2, 2, 4)
    
    plateau_times = []
    
    for i, cond_data in enumerate(all_conditions_data):
        cond = cond_data['cond']
        t = cond_data['t']
        distance_all = cond_data['distance_all']
        speed_all = cond_data['speed_all']
        br_idx = cond_data.get('br_idx', i)
        
        color = cmap(i % 10)
        n_trials = distance_all.shape[0]
        if n_trials == 0:
            continue
        
        # Compute normalized summary
        summary = compute_normalized_summary(
            distance_all, speed_all, t, 
            normalization_method=normalization_method
        )
        
        label = f"BR{br_idx} (n={n_trials})"
        
        # --- Raw distance ---
        dist_mean_raw = np.nanmean(distance_all, axis=0)
        dist_ci_raw = 1.96 * np.nanstd(distance_all, axis=0) / np.sqrt(n_trials)
        ax_dist_raw.plot(t, dist_mean_raw, color=color, linewidth=2, label=label)
        ax_dist_raw.fill_between(t, dist_mean_raw - dist_ci_raw, 
                                  dist_mean_raw + dist_ci_raw, color=color, alpha=0.2)
        
        # --- Normalized distance ---
        ax_dist_norm.plot(t, summary['dist_mean'], color=color, linewidth=2, label=label)
        ax_dist_norm.fill_between(t, summary['dist_mean'] - summary['dist_ci'],
                                   summary['dist_mean'] + summary['dist_ci'], 
                                   color=color, alpha=0.2)
        
        # Mark plateau time
        if summary['plateau']['plateau_time'] is not None:
            pt = summary['plateau']['plateau_time']
            pi = summary['plateau']['plateau_idx']
            ax_dist_norm.axvline(pt, color=color, linestyle=':', alpha=0.7)
            ax_dist_norm.plot(pt, summary['dist_mean'][pi], 'o', color=color, 
                             markersize=8, markeredgecolor='black')
            plateau_times.append((br_idx, pt, summary['plateau']['confidence']))
        
        # --- Raw speed ---
        speed_mean_raw = np.nanmean(speed_all, axis=0)
        speed_ci_raw = 1.96 * np.nanstd(speed_all, axis=0) / np.sqrt(n_trials)
        ax_speed_raw.plot(t, speed_mean_raw, color=color, linewidth=2, label=label)
        ax_speed_raw.fill_between(t, speed_mean_raw - speed_ci_raw,
                                   speed_mean_raw + speed_ci_raw, color=color, alpha=0.2)
        
        # --- Normalized speed ---
        ax_speed_norm.plot(t, summary['speed_mean'], color=color, linewidth=2, label=label)
        ax_speed_norm.fill_between(t, summary['speed_mean'] - summary['speed_ci'],
                                    summary['speed_mean'] + summary['speed_ci'],
                                    color=color, alpha=0.2)
        
        # Mark speed threshold
        if summary['plateau']['plateau_idx'] is not None:
            ax_speed_norm.axhline(summary['plateau']['speed_threshold'], 
                                  color=color, linestyle='--', alpha=0.3)
    
    # Formatting
    for ax in [ax_dist_raw, ax_dist_norm, ax_speed_raw, ax_speed_norm]:
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.legend(fontsize='small', loc='upper left')
        ax.grid(True, alpha=0.3)
    
    ax_dist_raw.set_ylabel('Distance from Start (raw)', fontsize=11)
    ax_dist_raw.set_title('Raw Distance from Start', fontsize=12)
    ax_dist_raw.set_ylim(bottom=0)
    
    ax_dist_norm.set_ylabel('Normalized Distance [0-1]', fontsize=11)
    ax_dist_norm.set_title(f'Normalized Distance (method: {normalization_method})\n● = detected plateau', fontsize=12)
    ax_dist_norm.set_ylim(0, 1.3)
    
    ax_speed_raw.set_ylabel('Speed (raw)', fontsize=11)
    ax_speed_raw.set_title('Raw 2D Speed', fontsize=12)
    ax_speed_raw.set_ylim(bottom=0)
    
    ax_speed_norm.set_ylabel('Normalized Speed', fontsize=11)
    ax_speed_norm.set_title('Normalized 2D Speed\n-- = speed threshold for plateau', fontsize=12)
    ax_speed_norm.set_ylim(bottom=0)
    
    # Add plateau time summary text
    if plateau_times:
        plateau_text = "Plateau times:\n" + "\n".join(
            [f"BR{br}: {t:.0f}ms (conf:{c:.2f})" for br, t, c in plateau_times]
        )
        fig.text(0.98, 0.02, plateau_text, fontsize=9, ha='right', va='bottom',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(f"Target {target} - Normalized Comparison\n"
                 f"Session: {PARAMS.session} | Normalization: {normalization_method}", 
                 fontsize=14)
    fig.tight_layout()
    
    # Save
    cond_suffix = "_BR" + "_".join(str(c) for c in sorted(conditions_to_plot)) if conditions_to_plot else "_All"
    out_filename = f"Target_{target}{cond_suffix}_Normalized_{normalization_method}.{OUTPUT_FORMAT}"
    fig.savefig(FIG_OUT_DIR / out_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {out_filename}")
    
    return plateau_times


# ---------------------------------------------------------------------
# FILE PROCESSING
# ---------------------------------------------------------------------

def process_single_file(p: Path, target: str, target_code: int) -> Optional[dict]:
    """
    Process a single file and generate plots based on config flags.
    
    Returns dict with condition data for cross-condition summary, or None.
    """
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
                print(f"Skipping {p.name}: missing position data")
                return None
            
            pos_segs = data[pos_key]  # (n_trials, n_keypoints, T)
            
            # Time axis
            t_axis = None
            for t_key in ["beh_rel_t", "rel_t", "t"]:
                if t_key in data:
                    t_axis = data[t_key]
                    break
            
            if t_axis is None:
                print(f"Skipping {p.name}: missing time axis")
                return None
            
            # Get keypoint names
            kp_names = _get_keypoint_names_from_data(data, CAMERA_TO_USE)
            if not kp_names:
                kp_names = [f"kp_{i}" for i in range(pos_segs.shape[1])]
            
            # Find X and Y indices for the base keypoint
            idx_x = find_keypoint_index(kp_names, f"{KEYPOINT_BASE}_x")
            idx_y = find_keypoint_index(kp_names, f"{KEYPOINT_BASE}_y")
            
            # Diagnostic print once
            diag_key = f"{PARAMS.session}_{p.name}"
            if diag_key not in _DIAG_PRINTED:
                print(f"  File: {p.name}")
                print(f"  Keypoint names: {kp_names[:10]}...")
                print(f"  Looking for: {KEYPOINT_BASE}_x and {KEYPOINT_BASE}_y")
                print(f"  Found indices: X={idx_x}, Y={idx_y}")
                _DIAG_PRINTED.add(diag_key)
            
            if idx_x is None or idx_y is None:
                print(f"  [WARN] Could not find both X and Y for '{KEYPOINT_BASE}' in {p.name}")
                print(f"         Available: {kp_names}")
                return None
            
            # Calculate dt
            dt = t_axis[1] - t_axis[0] if len(t_axis) > 1 else 10.0
            if dt < 0.9:
                dt *= 1000.0
            
            n_trials = pos_segs.shape[0]
            
            # Process all trials and collect data
            all_trial_data = []
            distance_all = []
            speed_all = []
            
            for i in range(n_trials):
                x = pos_segs[i, idx_x, :]
                y = pos_segs[i, idx_y, :]

                if np.sum(np.isfinite(x)) < 10 or np.sum(np.isfinite(y)) < 10:
                    warnings.warn(f"Skipping trial {i} - insufficient valid data")
                    continue
                
                try:
                    metrics = compute_2d_metrics(x, y, dt)
                except (ValueError, np.linalg.LinAlgError) as e:
                    warnings.warn(f"Skipping trial {i} due to error: {e}")
                    continue
                
                all_trial_data.append({
                    'idx_trial': i,
                    't': t_axis,
                    'metrics': metrics,
                })
                
                distance_all.append(metrics['distance_from_start'])
                speed_all.append(metrics['speed_2d'])
            
            # Convert to arrays for summary
            distance_all = np.array(distance_all)  # (n_trials, T)
            speed_all = np.array(speed_all)        # (n_trials, T)
            
            # --- Generate Debug Plots (if enabled) ---
            if GENERATE_DEBUG_PLOTS:
                trials_per_page = 4
                n_pages = int(np.ceil(n_trials / trials_per_page))
                
                for page in range(n_pages):
                    start_idx = page * trials_per_page
                    end_idx = min((page + 1) * trials_per_page, n_trials)
                    page_trials = all_trial_data[start_idx:end_idx]
                    n_page_trials = len(page_trials)
                    
                    fig = Figure(figsize=(20, 5 * n_page_trials))
                    FigureCanvasAgg(fig)
                    
                    for trial_offset, trial_data in enumerate(page_trials):
                        axes = np.array([
                            [fig.add_subplot(n_page_trials * 2, 2, trial_offset * 4 + 1),
                             fig.add_subplot(n_page_trials * 2, 2, trial_offset * 4 + 2)],
                            [fig.add_subplot(n_page_trials * 2, 2, trial_offset * 4 + 3),
                             fig.add_subplot(n_page_trials * 2, 2, trial_offset * 4 + 4)]
                        ])
                        
                        plot_trial_debug(trial_data, trial_data['idx_trial'], fig, axes)
                    
                    fig.suptitle(f"2D Position Debug: {target} - {cond} (Page {page+1}/{n_pages})\n"
                               f"Keypoint: {KEYPOINT_BASE}", fontsize=14)
                    fig.tight_layout()
                    
                    safe_cond = "".join([c if c.isalnum() else "_" for c in cond])
                    fig.savefig(FIG_OUT_DIR / f"2D_Debug_{target}_{safe_cond}_{p.stem}_page{page+1}.{OUTPUT_FORMAT}",
                               dpi=150, bbox_inches='tight')
                    plt.close(fig)
            
            # --- Generate Summary Plot (if enabled) ---
            if GENERATE_SUMMARY_PLOTS:
                plot_condition_summary(all_trial_data, t_axis, target, cond, n_trials, p.stem)
            
            # Return data for cross-condition summary
            return {
                "file": p.name,
                "n_trials": n_trials,
                "condition": cond,
                "cond": cond,
                "br_idx": br_idx,
                "t": t_axis,
                "distance_all": distance_all,
                "speed_all": speed_all,
            }
    
    except Exception as e:
        import traceback
        print(f"Error processing {p.name}: {e}")
        traceback.print_exc()
        return None


def process_target(target: str, target_code: int):
    """Process all files for a given target."""
    candidate_subdirs = [
        Path("stim_reaches") / f"target_{target}",
        Path("control_reaches") / f"target_{target}",
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
    
    # Deduplicate
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
    
    # Process files and collect results for target summary
    all_conditions_data = []
    
    for p in all_files:
        res = process_single_file(p, target, target_code)
        if res:
            print(f"  Processed: {res['file']} ({res['n_trials']} trials)")
            all_conditions_data.append(res)
    
    return all_conditions_data


def main():
    print(f"2D Position Analysis for Session: {PARAMS.session}")
    print(f"Output Dir: {FIG_OUT_DIR}")
    print(f"Keypoint Base: {KEYPOINT_BASE}")
    print(f"Smoothing: {SMOOTHING_METHOD} (window={SAVGOL_WINDOW_MS}ms)")
    print(f"")
    print(f"Plot options:")
    print(f"  Debug plots (per-trial):     {GENERATE_DEBUG_PLOTS}")
    print(f"  Summary plots (per-cond):    {GENERATE_SUMMARY_PLOTS}")
    print(f"  Target summary (all conds):  {GENERATE_TARGET_SUMMARY}")
    print(f"  Target summary conditions:   {TARGET_SUMMARY_CONDITIONS or 'ALL'}")
    print("")
    
    # Process each target and collect data
    target_A_data = process_target("A", 1)
    target_B_data = process_target("B", 2)
    
    # Generate cross-condition summaries per target
    if GENERATE_TARGET_SUMMARY:
        print("\nGenerating target summaries...")
        if target_A_data:
            plot_target_summary(target_A_data, "A", TARGET_SUMMARY_CONDITIONS)
            plot_target_summary_normalized(target_A_data, "A", TARGET_SUMMARY_CONDITIONS)
        if target_B_data:
            plot_target_summary(target_B_data, "B", TARGET_SUMMARY_CONDITIONS)
            plot_target_summary_normalized(target_B_data, "B", TARGET_SUMMARY_CONDITIONS)
    
    print(f"\nDone! Figures saved to {FIG_OUT_DIR}")


if __name__ == "__main__":
    main()