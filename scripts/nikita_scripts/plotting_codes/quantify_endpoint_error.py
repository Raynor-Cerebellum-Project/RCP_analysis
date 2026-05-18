"""
compute_reach_endpoint_error.py

Compute endpoint error and trajectory metrics from 3D DLC kinematics.

Metrics computed:
- Endpoint error: distance from finger to target at closest approach
- Path length: total distance traveled during reach
- Peak velocity: maximum velocity during reach
- Trajectory straightness: direct distance / path length
- Velocity dip: magnitude of velocity drop mid-reach (stim effect)
- Time to target: time from event to closest approach

Input:
    - PeriStim checkpoint (.npz) for reach timestamps, condition info, trial labels
    - 3D DLC file ({condition}_DLC_3D.csv) for middle finger and target positions
    - 2D aligned behavior CSV for frame-to-ns5_sample mapping

Output:
    - CSV with per-reach metrics (results/figures/reach_endpoint/)
    - Summary statistics and visualization plots
"""

import argparse
import re
import json
import warnings
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

import RCP_analysis as rcp

# Suppress the specific MKL warning during import
warnings.filterwarnings('ignore', message='.*MKL.*')

# Try to load config, with robust fallback
try:
    from RCP_analysis.python.functions.config_loading import *
except Exception as e:
    print(f"[warn] Config loading import failed: {e}")

# Check if critical variables exist; if not, set them manually
if 'PERI_ROOT' not in dir() or PERI_ROOT is None:
    print(f"[info] Setting up paths manually from params.yaml...")
    
    REPO_ROOT = Path(__file__).resolve().parents[3]
    PARAMS = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
    SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
    
    OUT_BASE       = SESSION_LOC / "results"
    BR_ROOT        = SESSION_LOC / "Blackrock"
    VIDEO_ROOT     = SESSION_LOC / "Video"
    INTAN_ROOT     = SESSION_LOC / "Intan"
    METADATA_ROOT  = SESSION_LOC / "Metadata"
    
    BEHV_CKPT_ROOT = OUT_BASE / "checkpoints" / "Behavior"
    NPRW_CKPT_ROOT = OUT_BASE / "checkpoints" / "NPRW"
    UA_CKPT_ROOT   = OUT_BASE / "checkpoints" / "UA"
    ALIGNED_CKPT_ROOT = OUT_BASE / "checkpoints" / "Aligned"
    PERI_ROOT      = OUT_BASE / "checkpoints" / "PeriStim"
    
    METADATA_CSV   = METADATA_ROOT / f"{PARAMS.session}_metadata.csv"
    
    if PARAMS.kinematics:
        KEYPOINTS_ORDER = tuple(PARAMS.kinematics.get("keypoints", []))
    else:
        KEYPOINTS_ORDER = ()
    
    print(f"[info] SESSION_LOC: {SESSION_LOC}")

# ---------- Config ----------
RESULTS_ROOT = OUT_BASE / "figures" / "reach_endpoint"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# Keypoint names in 3D DLC
FINGER_KEYPOINT = "middle"
TARGET_KEYPOINT_A = "TA"
TARGET_KEYPOINT_B = "TB"

# Analysis parameters
WIN_MS = (-600.0, 600.0)
CAMERA_FPS = 100.0
MAX_INTERP_GAP_FRAMES = 100
MIN_FINGER_VALID_PCT = 50.0
MIN_APPROACH_TIME_MS = 100.0


# Velocity smoothing window (frames)
VELOCITY_SMOOTH_WINDOW = 5


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute reach endpoint errors from 3D DLC data"
    )
    parser.add_argument(
        "--br", type=int, nargs="+", default=None,
        help="BR index(es) to process."
    )
    parser.add_argument(
        "--condition-type", type=str, 
        choices=["control", "stim", "continuous_stim", "all"],
        default="all",
        help="Which condition type(s) to process."
    )
    parser.add_argument(
        "--target", type=str, choices=["A", "B", "both"],
        default="A",
        help="Which target to analyze."
    )
    parser.add_argument(
        "--summary-br", type=int, nargs="+", default=None,
        help="BR indices to include in summary plots."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug output."
    )
    return parser.parse_args()


def get_process_only_list(args) -> List[int]:
    """Determine which BR indices to process."""
    if args.br is not None:
        return list(args.br)
    process_only = PARAMS.preprocessing.get("process_only", [])
    if process_only:
        return [int(x) for x in process_only]
    return []


def load_dlc_3d(dlc_3d_path: Path) -> Tuple[pd.DataFrame, list]:
    """Load 3D DLC CSV with multi-level header."""
    df_raw = pd.read_csv(dlc_3d_path, header=[0, 1, 2], index_col=0)
    df_raw.columns = [f"{bp}_{coord}" for scorer, bp, coord in df_raw.columns]
    bodyparts = list(set(col.rsplit('_', 1)[0] for col in df_raw.columns))
    df = df_raw.apply(pd.to_numeric, errors='coerce')
    return df, bodyparts


def get_xyz(df: pd.DataFrame, keypoint: str, frame_idx: np.ndarray) -> np.ndarray:
    """Extract x, y, z coordinates for a keypoint at given frame indices."""
    cols = [f"{keypoint}_x", f"{keypoint}_y", f"{keypoint}_z"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for keypoint '{keypoint}': {missing}")
    
    n_frames = len(df)
    # Convert frame indices to 0-based positional indices
    # If df.index starts at 1, frame 1 → position 0
    min_idx = df.index.min()
    pos_idx = frame_idx - min_idx  # Adjust for DataFrame's starting index
    
    valid_mask = (pos_idx >= 0) & (pos_idx < n_frames)
    xyz = np.full((len(frame_idx), 3), np.nan)
    
    if valid_mask.any():
        valid_positions = pos_idx[valid_mask].astype(int)
        xyz[valid_mask] = df.iloc[valid_positions][cols].values
    
    return xyz


def interpolate_small_gaps(arr: np.ndarray, max_gap: int = 3) -> np.ndarray:
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
            
            if (gap_len <= max_gap and 
                left_idx >= 0 and right_idx < n_rows and
                np.isfinite(x[left_idx]) and np.isfinite(x[right_idx])):
                x[s:e+1] = np.interp(
                    np.arange(s, e+1),
                    [left_idx, right_idx],
                    [x[left_idx], x[right_idx]]
                )
    
    return arr.squeeze() if arr.shape[1] == 1 else arr


def detect_outliers_plateau(signal: np.ndarray, 
                            min_plateau_frames: int = 2,
                            max_plateau_frames: int = 20) -> np.ndarray:
    """
    Detect plateau outliers: sudden jump to a flat wrong level, then jump back.
    
    Key insight: Real movement has consistent velocity (smoothly changing signal).
    Plateau outliers have: big jump → near-zero velocity → big jump back.
    
    Args:
        signal: 1D array of values (one coordinate axis)
        min_plateau_frames: Minimum frames the plateau must last
        max_plateau_frames: Maximum frames to consider as plateau (longer = real movement)
        
    Returns:
        Boolean mask where True = outlier
    """
    n = len(signal)
    outlier_mask = np.zeros(n, dtype=bool)
    
    if n < 5:
        return outlier_mask
    
    # Compute frame-to-frame differences
    diff = np.zeros(n - 1)
    for i in range(n - 1):
        if np.isfinite(signal[i]) and np.isfinite(signal[i + 1]):
            diff[i] = signal[i + 1] - signal[i]
        else:
            diff[i] = 0
    
    # Get valid differences for statistics
    valid_diff = diff[diff != 0]
    if len(valid_diff) < 5:
        return outlier_mask
    
    # Estimate typical small movement using MAD (robust)
    abs_diff = np.abs(valid_diff)
    typical_movement = np.median(abs_diff)
    
    if typical_movement < 1e-9:
        typical_movement = np.std(valid_diff) if np.std(valid_diff) > 0 else 1.0
    
    # A "jump" should be much larger than typical frame-to-frame movement
    min_jump_size = 6.0 * typical_movement
    
    if min_jump_size < 1e-6:
        return outlier_mask
    
    # Find large jumps (potential plateau starts/ends)
    large_jump_idx = np.where(np.abs(diff) > min_jump_size)[0]
    
    if len(large_jump_idx) < 2:
        return outlier_mask
    
    # Look for plateau pattern: jump UP/DOWN, stay relatively flat, jump back
    used_jumps = set()
    
    for i in range(len(large_jump_idx)):
        if i in used_jumps:
            continue
            
        start_jump_idx = large_jump_idx[i]
        start_jump_mag = diff[start_jump_idx]
        start_jump_sign = np.sign(start_jump_mag)
        
        # Value before the jump
        pre_jump_value = signal[start_jump_idx] if np.isfinite(signal[start_jump_idx]) else None
        if pre_jump_value is None:
            continue
        
        # Look for a return jump
        for j in range(i + 1, len(large_jump_idx)):
            if j in used_jumps:
                continue
                
            end_jump_idx = large_jump_idx[j]
            end_jump_mag = diff[end_jump_idx]
            end_jump_sign = np.sign(end_jump_mag)
            
            plateau_length = end_jump_idx - start_jump_idx
            
            # Must be opposite sign
            if start_jump_sign == end_jump_sign:
                continue
            
            # Plateau length check
            if plateau_length < min_plateau_frames or plateau_length > max_plateau_frames:
                continue
            
            # Check if signal returns close to pre-jump value
            post_jump_idx = end_jump_idx + 1
            if post_jump_idx < n and np.isfinite(signal[post_jump_idx]):
                post_jump_value = signal[post_jump_idx]
                
                # How close does it return? Allow some tolerance
                return_error = abs(post_jump_value - pre_jump_value)
                jump_size = abs(start_jump_mag)
                
                # Should return to within 70% of jump size
                if return_error < 0.7 * jump_size:
                    # Check that plateau region is relatively flat
                    plateau_start = start_jump_idx + 1
                    plateau_end = end_jump_idx + 1
                    plateau_values = signal[plateau_start:plateau_end]
                    valid_plateau = plateau_values[np.isfinite(plateau_values)]
                    
                    if len(valid_plateau) >= min_plateau_frames:
                        plateau_range = np.max(valid_plateau) - np.min(valid_plateau)
                        
                        # Plateau should be flatter than the jump size
                        # (allow up to 50% of jump magnitude as internal variation)
                        if plateau_range < 0.5 * jump_size:
                            # This is a plateau outlier!
                            outlier_mask[plateau_start:plateau_end] = True
                            used_jumps.add(i)
                            used_jumps.add(j)
                            break
    
    return outlier_mask


def clean_finger_signal(xyz: np.ndarray, fps: float, 
                        max_interp_gap: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean finger position signal by detecting and interpolating plateau outliers.
    
    Plateau outliers are: sudden jump → flat region → sudden jump back.
    Real movement (smooth changes) is preserved.
    
    Args:
        xyz: (N, 3) position array
        fps: Frame rate (for reference)
        max_interp_gap: Maximum gap size to interpolate (frames)
        
    Returns:
        cleaned_xyz: Cleaned position array
        outlier_mask: Boolean mask of detected outliers (True = was outlier)
    """
    n = len(xyz)
    combined_outlier_mask = np.zeros(n, dtype=bool)
    xyz_clean = xyz.copy()
    
    # Process each dimension independently
    for dim in range(3):
        signal = xyz[:, dim].copy()
        
        # Detect plateau outliers
        dim_outliers = detect_outliers_plateau(
            signal,
            min_plateau_frames=2,
            max_plateau_frames=20  # ~200ms at 100fps
        )
        
        combined_outlier_mask |= dim_outliers
        
        # Set outliers to NaN for interpolation
        signal[dim_outliers] = np.nan
        xyz_clean[:, dim] = signal
    
    # Interpolate the outlier regions
    xyz_clean = interpolate_small_gaps(xyz_clean, max_gap=max_interp_gap)
    
    return xyz_clean, combined_outlier_mask


def smooth_signal(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Smooth a 1D signal using uniform (moving average) filter.
    Uses scipy.ndimage which is more robust than savgol_filter for edge cases.
    
    Args:
        arr: 1D array to smooth
        window: Smoothing window size (must be odd for symmetry)
        
    Returns:
        Smoothed array
    """
    if len(arr) < window:
        return arr.copy()
    
    # Make window odd if even
    if window % 2 == 0:
        window += 1
    
    # Handle NaNs by interpolating first, smoothing, then restoring NaN positions
    nan_mask = ~np.isfinite(arr)
    
    if nan_mask.all():
        return arr.copy()
    
    arr_filled = arr.copy()
    
    # Simple linear interpolation of NaNs for smoothing
    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) > 1:
            arr_filled = np.interp(
                np.arange(len(arr)),
                valid_idx,
                arr[valid_idx]
            )
    
    # Apply uniform filter (moving average)
    smoothed = uniform_filter1d(arr_filled, size=window, mode='nearest')
    
    # Ensure non-negative (for velocity)
    smoothed = np.maximum(smoothed, 0)
    
    return smoothed


def compute_velocity(xyz: np.ndarray, dt_sec: float, smooth: bool = True) -> np.ndarray:
    """
    Compute instantaneous velocity magnitude from 3D positions.
    
    Args:
        xyz: (N, 3) position array
        dt_sec: Time step in seconds
        smooth: Whether to apply smoothing to velocity
        
    Returns:
        velocity: (N,) array of velocity magnitudes
    """
    n = xyz.shape[0]
    if n < 2:
        return np.zeros(n)
    
    # Central difference for interior, forward/backward for edges
    vel_vec = np.zeros_like(xyz)
    
    # Handle NaNs in position data
    for i in range(n):
        if i == 0:
            if np.all(np.isfinite(xyz[0])) and np.all(np.isfinite(xyz[1])):
                vel_vec[0] = (xyz[1] - xyz[0]) / dt_sec
            else:
                vel_vec[0] = np.nan
        elif i == n - 1:
            if np.all(np.isfinite(xyz[-1])) and np.all(np.isfinite(xyz[-2])):
                vel_vec[-1] = (xyz[-1] - xyz[-2]) / dt_sec
            else:
                vel_vec[-1] = np.nan
        else:
            if np.all(np.isfinite(xyz[i-1])) and np.all(np.isfinite(xyz[i+1])):
                vel_vec[i] = (xyz[i+1] - xyz[i-1]) / (2 * dt_sec)
            else:
                vel_vec[i] = np.nan
    
    velocity = np.linalg.norm(vel_vec, axis=1)
    
    if smooth:
        velocity = smooth_signal(velocity, window=VELOCITY_SMOOTH_WINDOW)
    
    return velocity


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance between two sets of 3D points."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    if b.shape[0] == 1 and a.shape[0] > 1:
        b = np.broadcast_to(b, a.shape)
    return np.linalg.norm(a - b, axis=1)


def get_target_position_for_reach(target_xyz: np.ndarray, method: str = 'median') -> np.ndarray:
    """Compute a representative target position for a reach window."""
    valid_mask = np.all(np.isfinite(target_xyz), axis=1)
    
    if not valid_mask.any():
        return np.array([np.nan, np.nan, np.nan])
    
    valid_xyz = target_xyz[valid_mask]
    
    if method == 'median':
        return np.median(valid_xyz, axis=0)
    elif method == 'mean':
        return np.mean(valid_xyz, axis=0)
    else:
        return np.median(valid_xyz, axis=0)


def compute_path_length(xyz: np.ndarray) -> float:
    """Compute total path length (sum of frame-to-frame distances)."""
    valid_mask = np.all(np.isfinite(xyz), axis=1)
    valid_xyz = xyz[valid_mask]
    
    if len(valid_xyz) < 2:
        return np.nan
    
    distances = np.linalg.norm(np.diff(valid_xyz, axis=0), axis=1)
    return np.sum(distances)


def compute_trajectory_straightness(xyz: np.ndarray) -> float:
    """
    Compute trajectory straightness (direct distance / path length).
    1.0 = perfectly straight, < 1.0 = curved
    """
    valid_mask = np.all(np.isfinite(xyz), axis=1)
    valid_xyz = xyz[valid_mask]
    
    if len(valid_xyz) < 2:
        return np.nan
    
    direct_distance = np.linalg.norm(valid_xyz[-1] - valid_xyz[0])
    path_length = compute_path_length(xyz)
    
    if path_length < 1e-6 or not np.isfinite(path_length):
        return np.nan
    
    return direct_distance / path_length


def compute_velocity_dip(velocity: np.ndarray, rel_time_ms: np.ndarray,
                         search_start_ms: float = 50.0,
                         search_end_ms: float = 300.0) -> dict:
    """
    Detect velocity dip (deceleration) during reach - characteristic of stim effect.
    """
    # Focus on the search window
    mask = (rel_time_ms >= search_start_ms) & (rel_time_ms <= search_end_ms)
    mask &= np.isfinite(velocity)
    
    if mask.sum() < 5:
        return {
            'dip_detected': False,
            'dip_magnitude': np.nan,
            'dip_time_ms': np.nan,
            'dip_duration_ms': np.nan,
            'pre_dip_velocity': np.nan,
            'min_velocity': np.nan,
            'velocity_recovery': np.nan,
        }
    
    vel_window = velocity[mask]
    time_window = rel_time_ms[mask]
    
    # Find peak velocity in first half of window
    half_idx = max(1, len(vel_window) // 2)
    pre_dip_idx = np.argmax(vel_window[:half_idx])
    pre_dip_velocity = vel_window[pre_dip_idx]
    
    # Find minimum velocity after the peak
    if pre_dip_idx >= len(vel_window) - 1:
        return {
            'dip_detected': False,
            'dip_magnitude': np.nan,
            'dip_time_ms': np.nan,
            'dip_duration_ms': np.nan,
            'pre_dip_velocity': pre_dip_velocity,
            'min_velocity': np.nan,
            'velocity_recovery': np.nan,
        }
    
    post_peak_vel = vel_window[pre_dip_idx:]
    post_peak_time = time_window[pre_dip_idx:]
    
    if len(post_peak_vel) < 2:
        return {
            'dip_detected': False,
            'dip_magnitude': np.nan,
            'dip_time_ms': np.nan,
            'dip_duration_ms': np.nan,
            'pre_dip_velocity': pre_dip_velocity,
            'min_velocity': np.nan,
            'velocity_recovery': np.nan,
        }
    
    min_idx = np.argmin(post_peak_vel)
    min_velocity = post_peak_vel[min_idx]
    min_time = post_peak_time[min_idx]
    
    dip_magnitude = pre_dip_velocity - min_velocity
    
    # Check for recovery
    if min_idx < len(post_peak_vel) - 1:
        post_dip_vel = post_peak_vel[min_idx:]
        recovery_velocity = np.max(post_dip_vel)
        velocity_recovery = recovery_velocity - min_velocity
    else:
        velocity_recovery = 0.0
    
    # Detect significant dip
    dip_detected = (pre_dip_velocity > 1e-6 and 
                   dip_magnitude > 0.3 * pre_dip_velocity and 
                   velocity_recovery > 0.2 * max(dip_magnitude, 1e-6))
    
    # Estimate dip duration
    threshold = 0.5 * pre_dip_velocity
    below_threshold = post_peak_vel < threshold
    if below_threshold.any():
        dip_start = np.argmax(below_threshold)
        dip_end = len(below_threshold) - 1 - np.argmax(below_threshold[::-1])
        if dip_end > dip_start:
            dip_duration_ms = post_peak_time[dip_end] - post_peak_time[dip_start]
        else:
            dip_duration_ms = 0.0
    else:
        dip_duration_ms = 0.0
    
    return {
        'dip_detected': dip_detected,
        'dip_magnitude': dip_magnitude,
        'dip_time_ms': min_time,
        'dip_duration_ms': dip_duration_ms,
        'pre_dip_velocity': pre_dip_velocity,
        'min_velocity': min_velocity,
        'velocity_recovery': velocity_recovery,
    }


def compute_jerk_metric(velocity: np.ndarray, dt_sec: float) -> float:
    """
    Compute mean absolute jerk (derivative of acceleration) - measure of movement smoothness.
    """
    valid_mask = np.isfinite(velocity)
    if valid_mask.sum() < 4:
        return np.nan
    
    # Use only valid velocities
    valid_vel = velocity[valid_mask]
    
    if len(valid_vel) < 4:
        return np.nan
    
    # Acceleration = d(velocity)/dt
    accel = np.diff(valid_vel) / dt_sec
    
    if len(accel) < 2:
        return np.nan
    
    # Jerk = d(acceleration)/dt
    jerk = np.diff(accel) / dt_sec
    
    if len(jerk) == 0:
        return np.nan
    
    return np.mean(np.abs(jerk))


def check_reach_direction(distance: np.ndarray, rel_time_ms: np.ndarray,
                          check_window_ms: float = 200.0,
                          approach_required: float = 0.1) -> Tuple[bool, float, str]:
    """Check if the hand is moving TOWARD the target."""
    mask_pre = (rel_time_ms >= -100) & (rel_time_ms <= -20)
    mask_end = (rel_time_ms >= check_window_ms - 30) & (rel_time_ms <= check_window_ms + 30)
    
    valid_pre = mask_pre & np.isfinite(distance)
    valid_end = mask_end & np.isfinite(distance)
    
    if not valid_pre.any():
        mask_start = (rel_time_ms >= 0) & (rel_time_ms <= 30)
        valid_pre = mask_start & np.isfinite(distance)
    
    if not valid_pre.any() or not valid_end.any():
        return True, 0.0, 'insufficient_data'
    
    dist_at_start = np.median(distance[valid_pre])
    dist_at_end = np.median(distance[valid_end])
    
    net_change = dist_at_end - dist_at_start
    threshold = dist_at_start * approach_required
    
    if net_change > threshold:
        return False, net_change, f'moved_away_{net_change:.2f}'
    elif net_change > 0:
        return True, net_change, f'minimal_movement_{net_change:.2f}'
    else:
        return True, net_change, f'approaching_{-net_change:.2f}'


def find_first_closest_approach(distance: np.ndarray, rel_time_ms: np.ndarray,
                                min_approach_time_ms: float = 70.0,
                                first_approach_threshold: float = 0.5) -> Tuple[int, float, str]:
    """Find the first closest approach AFTER the minimum approach time."""
    n = len(distance)
    if n == 0:
        return -1, np.nan, 'empty'
    
    time_mask = rel_time_ms >= min_approach_time_ms
    valid = np.isfinite(distance) & time_mask
    
    if not valid.any():
        return -1, np.nan, 'no_valid_after_min_time'
    
    start_mask = (rel_time_ms >= -50) & (rel_time_ms <= 50) & np.isfinite(distance)
    if start_mask.any():
        start_distance = np.median(distance[start_mask])
    else:
        first_valid_idx = np.where(np.isfinite(distance))[0]
        if len(first_valid_idx) == 0:
            return -1, np.nan, 'no_valid_data'
        start_distance = distance[first_valid_idx[0]]
    
    valid_indices = np.where(valid)[0]
    
    if len(valid_indices) < 3:
        idx = valid_indices[np.argmin(distance[valid_indices])]
        return idx, distance[idx], 'global_min_few_points'
    
    for i, idx in enumerate(valid_indices[1:-1], 1):
        prev_idx = valid_indices[i-1]
        next_idx = valid_indices[i+1]
        
        if (distance[idx] <= distance[prev_idx] and 
            distance[idx] <= distance[next_idx]):
            
            is_close = (distance[idx] < start_distance * first_approach_threshold or 
                       distance[idx] < start_distance * 0.3)
            
            if is_close:
                return idx, distance[idx], f'first_local_min_after_{min_approach_time_ms:.0f}ms'
    
    valid_distances = distance.copy()
    valid_distances[~valid] = np.inf
    idx = np.argmin(valid_distances)
    
    if np.isfinite(distance[idx]):
        return idx, distance[idx], f'global_min_after_{min_approach_time_ms:.0f}ms'
    
    return -1, np.nan, 'no_valid'


def safe_nanmax(arr):
    """Safe nanmax that returns nan for all-nan arrays."""
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.nan
    return np.nanmax(arr[finite_mask])


def safe_nanmean(arr):
    """Safe nanmean that returns nan for all-nan arrays."""
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.nan
    return np.nanmean(arr[finite_mask])


def process_single_reach(
    dlc_df: pd.DataFrame,
    event_frame: int,
    win_frames: Tuple[int, int],
    fps: float,
    finger_keypoint: str = FINGER_KEYPOINT,
    target_keypoint: str = TARGET_KEYPOINT_A,
    debug: bool = False,
) -> dict:
    """Process a single reach to compute endpoint error and trajectory metrics."""
    pre_frames, post_frames = win_frames
    
    start_frame = event_frame + pre_frames
    end_frame = event_frame + post_frames
    frame_indices = np.arange(start_frame, end_frame + 1)
    rel_time_ms = (frame_indices - event_frame) * (1000.0 / fps)
    
    # Get finger and target positions (RAW)
    try:
        finger_xyz_raw = get_xyz(dlc_df, finger_keypoint, frame_indices)
    except ValueError as e:
        return {
            'status': 'error',
            'error_msg': str(e),
            'drop_reason': 'finger_keypoint_missing',
            'endpoint_error': np.nan,
        }
    
    try:
        target_xyz_raw = get_xyz(dlc_df, target_keypoint, frame_indices)
    except ValueError as e:
        return {
            'status': 'error',
            'error_msg': str(e),
            'drop_reason': 'target_keypoint_missing',
            'endpoint_error': np.nan,
        }
    
    # Calculate data quality
    pct_finger_valid_raw = np.mean(np.all(np.isfinite(finger_xyz_raw), axis=1)) * 100
    pct_target_valid_raw = np.mean(np.all(np.isfinite(target_xyz_raw), axis=1)) * 100
    
    # Clean finger positions
    finger_xyz_clean, outlier_mask = clean_finger_signal(
        finger_xyz_raw, fps, 
        max_interp_gap=MAX_INTERP_GAP_FRAMES
    )
    
    n_outliers_detected = outlier_mask.sum()
    pct_finger_valid_clean = np.mean(np.all(np.isfinite(finger_xyz_clean), axis=1)) * 100
    
    # Get target position
    target_pos = get_target_position_for_reach(target_xyz_raw, method='median')
    
    if not np.all(np.isfinite(target_pos)):
        return {
            'status': 'no_target',
            'error_msg': 'Target not visible in window',
            'drop_reason': 'target_not_visible',
            'endpoint_error': np.nan,
            'pct_finger_valid_raw': pct_finger_valid_raw,
            'pct_finger_valid_clean': pct_finger_valid_clean,
            'pct_target_valid_raw': pct_target_valid_raw,
            'n_outliers_detected': n_outliers_detected,
        }
    
    if pct_finger_valid_clean < MIN_FINGER_VALID_PCT:
        return {
            'status': 'insufficient_finger_data',
            'error_msg': f'Only {pct_finger_valid_clean:.1f}% valid finger data',
            'drop_reason': 'finger_tracking_poor',
            'endpoint_error': np.nan,
            'pct_finger_valid_raw': pct_finger_valid_raw,
            'pct_finger_valid_clean': pct_finger_valid_clean,
            'pct_target_valid_raw': pct_target_valid_raw,
            'n_outliers_detected': n_outliers_detected,
            'target_x': target_pos[0],
            'target_y': target_pos[1],
            'target_z': target_pos[2],
        }
    
    # Compute distance
    distance = euclidean_distance(finger_xyz_clean, target_pos)
    
    # Check direction
    is_approaching, net_change, direction_reason = check_reach_direction(
        distance, rel_time_ms, check_window_ms=200.0, approach_required=0.1
    )
    
    if not is_approaching:
        return {
            'status': 'moving_away',
            'error_msg': f'Hand moving away: {direction_reason}',
            'drop_reason': 'direction_away',
            'endpoint_error': np.nan,
            'net_distance_change': net_change,
            'pct_finger_valid_raw': pct_finger_valid_raw,
            'pct_finger_valid_clean': pct_finger_valid_clean,
            'pct_target_valid_raw': pct_target_valid_raw,
            'n_outliers_detected': n_outliers_detected,
            'target_x': target_pos[0],
            'target_y': target_pos[1],
            'target_z': target_pos[2],
        }
    
    # Compute velocity
    dt_sec = 1.0 / fps
    velocity = compute_velocity(finger_xyz_clean, dt_sec, smooth=True)
    
    # Find closest approach
    approach_idx, min_distance, approach_method = find_first_closest_approach(
        distance,
        rel_time_ms=rel_time_ms,
        min_approach_time_ms=MIN_APPROACH_TIME_MS,
        first_approach_threshold=0.3,
    )
    
    if approach_idx < 0:
        return {
            'status': 'no_approach',
            'error_msg': f'Could not find valid approach after {MIN_APPROACH_TIME_MS}ms',
            'drop_reason': 'no_approach_found',
            'endpoint_error': np.nan,
            'pct_finger_valid_raw': pct_finger_valid_raw,
            'pct_finger_valid_clean': pct_finger_valid_clean,
            'pct_target_valid_raw': pct_target_valid_raw,
            'n_outliers_detected': n_outliers_detected,
            'target_x': target_pos[0],
            'target_y': target_pos[1],
            'target_z': target_pos[2],
        }
    
    approach_time_ms = rel_time_ms[approach_idx]
    approach_frame = frame_indices[approach_idx]
    
    if approach_time_ms < MIN_APPROACH_TIME_MS:
        return {
            'status': 'approach_too_early',
            'error_msg': f'Approach at {approach_time_ms:.0f}ms (min: {MIN_APPROACH_TIME_MS}ms)',
            'drop_reason': 'approach_too_early',
            'endpoint_error': np.nan,
            'pct_finger_valid_raw': pct_finger_valid_raw,
            'pct_finger_valid_clean': pct_finger_valid_clean,
            'pct_target_valid_raw': pct_target_valid_raw,
            'n_outliers_detected': n_outliers_detected,
            'target_x': target_pos[0],
            'target_y': target_pos[1],
            'target_z': target_pos[2],
        }
    
    # === COMPUTE TRAJECTORY METRICS ===
    reach_mask = (rel_time_ms >= 0) & (rel_time_ms <= approach_time_ms + 50)
    reach_xyz = finger_xyz_clean[reach_mask]
    reach_velocity = velocity[reach_mask]
    
    path_length = compute_path_length(reach_xyz)
    straightness = compute_trajectory_straightness(reach_xyz)
    peak_velocity = safe_nanmax(reach_velocity)
    mean_velocity = safe_nanmean(reach_velocity)
    
    # Velocity dip analysis
    dip_metrics = compute_velocity_dip(
        velocity, rel_time_ms, 
        search_start_ms=50.0, 
        search_end_ms=min(300.0, approach_time_ms)
    )
    
    # Jerk metric
    mean_abs_jerk = compute_jerk_metric(reach_velocity, dt_sec)
    
    # Finger position at approach
    finger_at_approach = finger_xyz_clean[approach_idx]
    vel_at_approach = velocity[approach_idx] if approach_idx < len(velocity) else np.nan
    
    return {
        'status': 'success',
        'drop_reason': None,
        'endpoint_error': min_distance,
        'approach_time_ms': approach_time_ms,
        'approach_frame': int(approach_frame),
        'approach_method': approach_method,
        'direction_check': direction_reason,
        
        # Position info
        'finger_x': finger_at_approach[0],
        'finger_y': finger_at_approach[1],
        'finger_z': finger_at_approach[2],
        'target_x': target_pos[0],
        'target_y': target_pos[1],
        'target_z': target_pos[2],
        
        # Velocity metrics
        'velocity_at_approach': vel_at_approach,
        'peak_velocity': peak_velocity,
        'mean_velocity': mean_velocity,
        
        # Trajectory metrics
        'path_length': path_length,
        'straightness': straightness,
        'mean_abs_jerk': mean_abs_jerk,
        
        # Velocity dip metrics
        'dip_detected': dip_metrics['dip_detected'],
        'dip_magnitude': dip_metrics['dip_magnitude'],
        'dip_time_ms': dip_metrics['dip_time_ms'],
        'dip_duration_ms': dip_metrics['dip_duration_ms'],
        'pre_dip_velocity': dip_metrics['pre_dip_velocity'],
        'min_dip_velocity': dip_metrics['min_velocity'],
        'velocity_recovery': dip_metrics['velocity_recovery'],
        
        # Data quality
        'net_distance_change_200ms': net_change,
        'pct_finger_valid_raw': pct_finger_valid_raw,
        'pct_finger_valid_clean': pct_finger_valid_clean,
        'pct_target_valid_raw': pct_target_valid_raw,
        'n_outliers_detected': n_outliers_detected,
    }


def find_condition_name_for_br(br_idx: int) -> Optional[str]:
    """Find the condition name for a given BR index."""
    try:
        br_to_video = rcp.get_metadata_mapping(METADATA_CSV, "BR_File", "Video_File")
        if br_idx in br_to_video and br_to_video[br_idx] is not None:
            vid_idx = int(br_to_video[br_idx])
            session_prefix = getattr(PARAMS, 'session', 'NRR_RW022')
            return f"{session_prefix}_{vid_idx:03d}"
    except Exception as e:
        print(f"[warn] Could not map BR {br_idx} to video file: {e}")
    return None


def process_peristim_file(
    peristim_path: Path,
    dlc_3d_root: Path,
    behv_aligned_root: Path,
    condition_type: str,
    target_filter: str = 'A',
    debug: bool = False,
) -> Optional[pd.DataFrame]:
    """Process a single PeriStim file to compute endpoint errors for all reaches."""
    try:
        peri = np.load(peristim_path, allow_pickle=True)
    except Exception as e:
        print(f"[error] Could not load {peristim_path}: {e}")
        return None
    
    br_idx = int(peri['br_idx']) if 'br_idx' in peri.files else -1
    sess = str(peri['sess']) if 'sess' in peri.files else ''
    
    event_ms = peri['event_ms'] if 'event_ms' in peri.files else np.array([])
    if event_ms.size == 0:
        print(f"[skip] {peristim_path.name}: no events")
        return None
    
    trial_labels = peri['trial_labels'] if 'trial_labels' in peri.files else np.full(event_ms.shape, 'A', dtype='U1')
    
    total_reaches_in_file = len(event_ms)
    
    if target_filter.lower() != 'both':
        target_char = target_filter.upper()
        target_mask = np.array([str(lbl).upper() == target_char for lbl in trial_labels])
        
        if not target_mask.any():
            print(f"[skip] {peristim_path.name}: no reaches to target {target_char}")
            return None
        
        event_ms = event_ms[target_mask]
        trial_labels = trial_labels[target_mask]
        reaches_after_target_filter = len(event_ms)
        print(f"  [filter] Target {target_char}: {reaches_after_target_filter}/{total_reaches_in_file} reaches")
    else:
        reaches_after_target_filter = total_reaches_in_file
    
    # Determine condition name
    condition_name = None
    try:
        meta = json.loads(peri['meta_json'].item()) if 'meta_json' in peri.files else {}
        if 'video_file' in meta:
            vid_idx = int(meta['video_file'])
            session_prefix = PARAMS.session if hasattr(PARAMS, 'session') else 'NRR_RW022'
            condition_name = f"{session_prefix}_{vid_idx:03d}"
    except:
        pass
    
    if condition_name is None:
        condition_name = find_condition_name_for_br(br_idx)
    
    if condition_name is None:
        print(f"[skip] {peristim_path.name}: could not determine condition name")
        return None
    
    # Load 3D DLC file
    dlc_3d_path = dlc_3d_root / f"{condition_name}_DLC_3D.csv"
    if not dlc_3d_path.exists():
        print(f"[skip] {peristim_path.name}: 3D DLC file not found: {dlc_3d_path}")
        return None
    
    try:
        dlc_df, bodyparts = load_dlc_3d(dlc_3d_path)
        print(f"[load] {dlc_3d_path.name}: {len(dlc_df)} frames, bodyparts: {bodyparts}")
    except Exception as e:
        print(f"[error] Could not load 3D DLC {dlc_3d_path}: {e}")
        return None
    
    if FINGER_KEYPOINT not in bodyparts:
        print(f"[error] Finger keypoint '{FINGER_KEYPOINT}' not found")
        return None
    
    # Determine target keypoints
    if target_filter.upper() == 'A':
        target_keypoints = [TARGET_KEYPOINT_A]
    elif target_filter.upper() == 'B':
        target_keypoints = [TARGET_KEYPOINT_B]
    else:
        target_keypoints = [TARGET_KEYPOINT_A, TARGET_KEYPOINT_B]
    
    for tkp in target_keypoints:
        if tkp not in bodyparts:
            print(f"[error] Target keypoint '{tkp}' not found")
            return None
    
    try:
        nprw_meta = peri['nprw_meta'].item() if 'nprw_meta' in peri.files else {}
        fs_neural = float(nprw_meta.get('fs', 30000.0))
    except:
        fs_neural = 30000.0
    
    fps = CAMERA_FPS
    
    pre_frames = int(np.floor(WIN_MS[0] * fps / 1000.0))
    post_frames = int(np.ceil(WIN_MS[1] * fps / 1000.0))
    win_frames = (pre_frames, post_frames)
    
    # Convert event times to frame indices
    behv_aligned_csv = behv_aligned_root / f"{condition_name}_both_cams_aligned.csv"
    
    # Initialize variables for validation
    min_valid_sample = None
    max_valid_sample = None
    
    if behv_aligned_csv.exists():
        try:
            df_align = pd.read_csv(behv_aligned_csv, header=[0, 1])
            
            ns5_col = None
            for col in df_align.columns:
                if 'ns5_sample' in str(col):
                    ns5_col = col
                    break
            
            if ns5_col is not None:
                ns5_samples = pd.to_numeric(df_align[ns5_col], errors='coerce').values
                valid_mask = np.isfinite(ns5_samples)
                valid_frames = np.arange(len(ns5_samples))[valid_mask]
                valid_samples = ns5_samples[valid_mask]
                
                # Store range for validation
                min_valid_sample = np.min(valid_samples)
                max_valid_sample = np.max(valid_samples)
                
                event_samples = (event_ms * fs_neural / 1000.0).astype(np.int64)
                
                event_frames = np.zeros(len(event_ms), dtype=int)
                for i, samp in enumerate(event_samples):
                    sample_diffs = np.abs(valid_samples - samp)
                    nearest_idx = np.argmin(sample_diffs)
                    event_frames[i] = valid_frames[nearest_idx]
                
                # === NEW: Validate events are within video range ===
                valid_event_mask = (event_samples >= min_valid_sample) & (event_samples <= max_valid_sample)
                if not valid_event_mask.all():
                    n_invalid = (~valid_event_mask).sum()
                    print(f"  [WARN] {n_invalid}/{len(event_ms)} events outside video range (samples {min_valid_sample:.0f}-{max_valid_sample:.0f}), skipping them")
                    event_ms = event_ms[valid_event_mask]
                    event_frames = event_frames[valid_event_mask]
                    trial_labels = trial_labels[valid_event_mask]
                
                if len(event_ms) == 0:
                    print(f"[skip] {peristim_path.name}: no events within video range")
                    return None
            else:
                print(f"  [WARN] ns5_sample column not found in aligned CSV, using time-based conversion")
                event_frames = (event_ms * fps / 1000.0).astype(int)
        except Exception as e:
            print(f"  [WARN] Error reading aligned CSV: {e}, using time-based conversion")
            event_frames = (event_ms * fps / 1000.0).astype(int)
    else:
        print(f"  [WARN] Aligned CSV not found: {behv_aligned_csv.name}, using time-based conversion")
        event_frames = (event_ms * fps / 1000.0).astype(int)
    
    # Process each reach
    results = []
    
    for i, (event_frame, event_t, label) in enumerate(zip(event_frames, event_ms, trial_labels)):
        if target_filter.lower() == 'both':
            reach_target_kp = TARGET_KEYPOINT_A if str(label).upper() == 'A' else TARGET_KEYPOINT_B
        else:
            reach_target_kp = target_keypoints[0]
        
        reach_result = process_single_reach(
            dlc_df=dlc_df,
            event_frame=event_frame,
            win_frames=win_frames,
            fps=fps,
            finger_keypoint=FINGER_KEYPOINT,
            target_keypoint=reach_target_kp,
            debug=debug and i < 3,
        )
        
        reach_result['reach_idx'] = i
        reach_result['event_ms'] = event_t
        reach_result['event_frame'] = event_frame
        reach_result['trial_label'] = label
        reach_result['target_keypoint'] = reach_target_kp
        reach_result['condition_type'] = condition_type
        reach_result['br_idx'] = br_idx
        reach_result['condition_name'] = condition_name
        reach_result['sess'] = sess
        
        results.append(reach_result)
    
    if not results:
        return None
    
    df = pd.DataFrame(results)
    df['total_reaches_in_file'] = total_reaches_in_file
    df['reaches_after_target_filter'] = reaches_after_target_filter
    
    return df


def print_session_summary(combined_df: pd.DataFrame):
    """Print a detailed summary of reach processing results."""
    
    print("\n" + "="*70)
    print("SESSION SUMMARY")
    print("="*70)
    
    total_reaches = len(combined_df)
    successful = (combined_df['status'] == 'success').sum()
    
    print(f"\nTotal reaches processed: {total_reaches}")
    print(f"Successful reaches:      {successful} ({100*successful/total_reaches:.1f}%)")
    print(f"Dropped reaches:         {total_reaches - successful} ({100*(total_reaches-successful)/total_reaches:.1f}%)")
    
    print("\n--- Drop Reasons ---")
    drop_reasons = combined_df[combined_df['status'] != 'success']['drop_reason'].value_counts()
    for reason, count in drop_reasons.items():
        pct = 100 * count / total_reaches
        print(f"  {reason}: {count} ({pct:.1f}%)")
    
    print("\n--- By Condition Type ---")
    for cond in combined_df['condition_type'].unique():
        subset = combined_df[combined_df['condition_type'] == cond]
        n_total = len(subset)
        n_success = (subset['status'] == 'success').sum()
        print(f"  {cond}: {n_success}/{n_total} successful ({100*n_success/n_total:.1f}%)")
    
    print("\n--- By BR Index ---")
    br_summary = combined_df.groupby('br_idx').agg({
        'status': lambda x: (x == 'success').sum(),
        'reach_idx': 'count'
    }).rename(columns={'status': 'success', 'reach_idx': 'total'})
    br_summary['pct'] = 100 * br_summary['success'] / br_summary['total']
    
    for br_idx, row in br_summary.iterrows():
        print(f"  BR {br_idx:3d}: {row['success']:3.0f}/{row['total']:3.0f} ({row['pct']:.1f}%)")
    
    # Trajectory metrics summary
    success_df = combined_df[combined_df['status'] == 'success']
    if len(success_df) > 0:
        print("\n--- Trajectory Metrics (successful reaches) ---")
        print(f"  Mean endpoint error: {success_df['endpoint_error'].mean():.2f}")
        print(f"  Mean path length: {success_df['path_length'].mean():.2f}")
        print(f"  Mean straightness: {success_df['straightness'].mean():.3f}")
        print(f"  Mean peak velocity: {success_df['peak_velocity'].mean():.2f}")
        
        print("\n--- Velocity Dip Detection ---")
        for cond in success_df['condition_type'].unique():
            subset = success_df[success_df['condition_type'] == cond]
            if 'dip_detected' in subset.columns:
                dip_pct = 100 * subset['dip_detected'].mean()
                print(f"  {cond}: {dip_pct:.1f}% of reaches have velocity dip")
    
    print("="*70 + "\n")


def create_summary_plots(df: pd.DataFrame, output_dir: Path, summary_br_list: Optional[List[int]] = None):
    """Create summary visualization plots."""
    
    if summary_br_list:
        df = df[df['br_idx'].isin(summary_br_list)].copy()
        if len(df) == 0:
            return
    
    df = df[df['status'] == 'success'].copy()
    if len(df) == 0:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    conditions = df['condition_type'].unique()
    
    # 1. Endpoint error distribution
    ax = axes[0, 0]
    for cond in conditions:
        subset = df[df['condition_type'] == cond]['endpoint_error'].dropna()
        if len(subset) > 0:
            ax.hist(subset, bins=20, alpha=0.5, label=f'{cond} (n={len(subset)})', density=True)
    ax.set_xlabel('Endpoint Error (distance)')
    ax.set_ylabel('Density')
    ax.set_title('Endpoint Error Distribution')
    ax.legend()
    
    # 2. Trajectory straightness by condition
    ax = axes[0, 1]
    df.boxplot(column='straightness', by='condition_type', ax=ax)
    ax.set_xlabel('Condition Type')
    ax.set_ylabel('Straightness (1 = perfectly straight)')
    ax.set_title('Trajectory Straightness')
    plt.suptitle('')
    
    # 3. Path length by condition
    ax = axes[0, 2]
    df.boxplot(column='path_length', by='condition_type', ax=ax)
    ax.set_xlabel('Condition Type')
    ax.set_ylabel('Path Length (distance)')
    ax.set_title('Path Length')
    plt.suptitle('')
    
    # 4. Peak velocity by condition
    ax = axes[1, 0]
    df.boxplot(column='peak_velocity', by='condition_type', ax=ax)
    ax.set_xlabel('Condition Type')
    ax.set_ylabel('Peak Velocity')
    ax.set_title('Peak Velocity')
    plt.suptitle('')
    
    # 5. Velocity dip magnitude
    ax = axes[1, 1]
    dip_df = df[df['dip_detected'] == True]
    if len(dip_df) > 0:
        dip_df.boxplot(column='dip_magnitude', by='condition_type', ax=ax)
        ax.set_title('Velocity Dip Magnitude\n(reaches with detected dips)')
    else:
        ax.text(0.5, 0.5, 'No velocity dips detected', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Velocity Dip Magnitude')
    ax.set_xlabel('Condition Type')
    ax.set_ylabel('Dip Magnitude')
    plt.suptitle('')
    
    # 6. Dip detection rate by BR index
    ax = axes[1, 2]
    if 'dip_detected' in df.columns:
        dip_by_br = df.groupby('br_idx')['dip_detected'].mean() * 100
        ax.bar(range(len(dip_by_br)), dip_by_br.values)
        ax.set_xticks(range(len(dip_by_br)))
        ax.set_xticklabels([str(idx) for idx in dip_by_br.index])
    ax.set_xlabel('BR Index')
    ax.set_ylabel('% Reaches with Velocity Dip')
    ax.set_title('Velocity Dip Detection Rate')
    
    plt.tight_layout()
    
    if summary_br_list:
        br_str = '_BR_' + '_'.join(str(b) for b in sorted(summary_br_list))
        plot_path = output_dir / f"trajectory_metrics_summary{br_str}.png"
    else:
        plot_path = output_dir / "trajectory_metrics_summary.png"
    
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[save] Summary plot: {plot_path}")


def plot_single_reach_diagnostic(
    dlc_df: pd.DataFrame,
    event_frame: int,
    win_frames: Tuple[int, int],
    fps: float,
    reach_idx: int,
    condition_name: str,
    output_dir: Path,
    reach_result: dict,
    finger_keypoint: str = FINGER_KEYPOINT,
    target_keypoint: str = TARGET_KEYPOINT_A,
):
    """Create diagnostic plot for a single reach."""
    from mpl_toolkits.mplot3d import Axes3D
    
    pre_frames, post_frames = win_frames
    start_frame = event_frame + pre_frames
    end_frame = event_frame + post_frames
    frame_indices = np.arange(start_frame, end_frame + 1)
    rel_time_ms = (frame_indices - event_frame) * (1000.0 / fps)
    
    try:
        finger_xyz_raw = get_xyz(dlc_df, finger_keypoint, frame_indices)
        target_xyz_raw = get_xyz(dlc_df, target_keypoint, frame_indices)
    except ValueError:
        return
    
    finger_xyz_clean, outlier_mask = clean_finger_signal(
        finger_xyz_raw, fps, max_interp_gap=MAX_INTERP_GAP_FRAMES
    )
    
    target_pos = get_target_position_for_reach(target_xyz_raw, method='median')
    
    if not np.all(np.isfinite(target_pos)):
        target_pos = np.array([np.nan, np.nan, np.nan])
    
    distance = euclidean_distance(finger_xyz_clean, target_pos)
    
    is_approaching, net_change, direction_reason = check_reach_direction(
        distance, rel_time_ms, check_window_ms=200.0, approach_required=0.1
    )
    
    dt_sec = 1.0 / fps
    velocity_raw = compute_velocity(finger_xyz_clean, dt_sec, smooth=False)
    velocity_smooth = compute_velocity(finger_xyz_clean, dt_sec, smooth=True)
    
    approach_idx, min_dist, method = find_first_closest_approach(
        distance,
        rel_time_ms=rel_time_ms,
        min_approach_time_ms=MIN_APPROACH_TIME_MS,
        first_approach_threshold=0.3,
    )
    
    status = reach_result.get('status', 'unknown')
    drop_reason = reach_result.get('drop_reason', '')
    
    fig = plt.figure(figsize=(18, 14))
    
    # 1. Distance over time
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.axvspan(rel_time_ms.min(), MIN_APPROACH_TIME_MS, alpha=0.1, color='gray', 
                label=f'Before min approach ({MIN_APPROACH_TIME_MS}ms)')
    ax1.plot(rel_time_ms, distance, 'b-', lw=1.5, label='Distance to target')
    ax1.axvline(0, color='r', ls='--', lw=2, label='Event (stim/IR)')
    ax1.axvline(MIN_APPROACH_TIME_MS, color='orange', ls=':', lw=2, label=f'Min approach time')
    
    if approach_idx >= 0 and status == 'success':
        approach_time = rel_time_ms[approach_idx]
        ax1.axvline(approach_time, color='g', ls='-', lw=2, label='Closest approach')
        ax1.scatter([approach_time], [min_dist], c='g', s=100, zorder=5)
        ax1.annotate(f'{approach_time:.0f} ms\n{min_dist:.2f}', 
                    xy=(approach_time, min_dist),
                    xytext=(approach_time + 50, min_dist + 0.5),
                    fontsize=9, 
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    ax1.set_xlabel('Time relative to event (ms)')
    ax1.set_ylabel('Distance to target')
    ax1.set_title(f'Distance to Target Over Time\nMethod: {method}')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3)
    
    # 2. Velocity over time
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(rel_time_ms, velocity_raw, 'purple', lw=0.5, alpha=0.3, label='Raw velocity')
    ax2.plot(rel_time_ms, velocity_smooth, 'purple', lw=1.5, label='Smoothed velocity')
    ax2.axvline(0, color='r', ls='--', lw=2)
    ax2.axvline(MIN_APPROACH_TIME_MS, color='orange', ls=':', lw=2)
    if approach_idx >= 0 and status == 'success':
        ax2.axvline(rel_time_ms[approach_idx], color='g', ls='-', lw=2)
    
    dip_detected = reach_result.get('dip_detected', False)
    if dip_detected:
        dip_time = reach_result.get('dip_time_ms', np.nan)
        if np.isfinite(dip_time):
            ax2.axvline(dip_time, color='cyan', ls='--', lw=2, label='Velocity dip')
    
    ax2.set_xlabel('Time relative to event (ms)')
    ax2.set_ylabel('Velocity (distance/s)')
    ax2.set_title('Finger Velocity Over Time')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Finger X, Y, Z positions
    ax3 = fig.add_subplot(3, 2, 3)
    colors = ['red', 'green', 'blue']
    labels = ['X', 'Y', 'Z']
    for dim, (color, label) in enumerate(zip(colors, labels)):
        ax3.plot(rel_time_ms, finger_xyz_raw[:, dim], color=color, lw=0.5, alpha=0.3)
        ax3.plot(rel_time_ms, finger_xyz_clean[:, dim], color=color, lw=1.5, label=f'Finger {label}')
    
    if outlier_mask.any():
        outlier_times = rel_time_ms[outlier_mask]
        for ot in outlier_times:
            ax3.axvline(ot, color='gray', ls=':', alpha=0.3)
        ax3.axvline(np.nan, color='gray', ls=':', alpha=0.3, label=f'Outliers (n={outlier_mask.sum()})')
    
    ax3.axvline(0, color='r', ls='--', lw=2, alpha=0.5)
    ax3.axvline(MIN_APPROACH_TIME_MS, color='orange', ls=':', lw=2, alpha=0.5)
    if approach_idx >= 0 and status == 'success':
        ax3.axvline(rel_time_ms[approach_idx], color='g', ls='-', lw=2, alpha=0.5)
    ax3.set_xlabel('Time relative to event (ms)')
    ax3.set_ylabel('Position')
    ax3.set_title('Finger Position (X, Y, Z) Over Time\n(faint = raw, solid = cleaned)')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Target X, Y, Z positions
    ax4 = fig.add_subplot(3, 2, 4)
    for dim, (color, label) in enumerate(zip(colors, labels)):
        ax4.plot(rel_time_ms, target_xyz_raw[:, dim], color=color, lw=1.5, 
                label=f'Target {label}', marker='.', markersize=2)
        if np.isfinite(target_pos[dim]):
            ax4.axhline(target_pos[dim], color=color, ls='--', lw=1, alpha=0.5)
    ax4.axvline(0, color='r', ls='--', lw=2, alpha=0.5)
    ax4.set_xlabel('Time relative to event (ms)')
    ax4.set_ylabel('Position')
    ax4.set_title(f'Target Position (X, Y, Z) Over Time\n(dashed = median used for distance calc)')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. 3D Trajectory
    ax5 = fig.add_subplot(3, 2, 5, projection='3d')
    
    valid_clean = np.all(np.isfinite(finger_xyz_clean), axis=1)
    if valid_clean.any():
        valid_times = rel_time_ms[valid_clean]
        sc = ax5.scatter(finger_xyz_clean[valid_clean, 0], 
                        finger_xyz_clean[valid_clean, 1],
                        finger_xyz_clean[valid_clean, 2],
                        c=valid_times, cmap='coolwarm', s=15, alpha=0.8)
        fig.colorbar(sc, ax=ax5, label='Time (ms)', shrink=0.6)
    
    if np.all(np.isfinite(target_pos)):
        ax5.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], 
                   c='red', s=300, marker='*', label='Target', zorder=10)
    
    if approach_idx >= 0 and status == 'success' and valid_clean[approach_idx]:
        ax5.scatter([finger_xyz_clean[approach_idx, 0]], 
                   [finger_xyz_clean[approach_idx, 1]],
                   [finger_xyz_clean[approach_idx, 2]], 
                   c='green', s=200, marker='o', edgecolors='black', 
                   linewidths=2, label='Closest approach', zorder=10)
        if np.all(np.isfinite(target_pos)):
            ax5.plot([finger_xyz_clean[approach_idx, 0], target_pos[0]],
                    [finger_xyz_clean[approach_idx, 1], target_pos[1]],
                    [finger_xyz_clean[approach_idx, 2], target_pos[2]],
                    'g--', lw=2, alpha=0.7)
    
    if valid_clean.any():
        first_valid = np.where(valid_clean)[0][0]
        ax5.scatter([finger_xyz_clean[first_valid, 0]], 
                   [finger_xyz_clean[first_valid, 1]],
                   [finger_xyz_clean[first_valid, 2]], 
                   c='blue', s=150, marker='s', edgecolors='black',
                   linewidths=2, label='Start', zorder=10)
    
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.set_title('3D Finger Trajectory')
    ax5.legend(loc='upper left', fontsize=8)
    
    # 6. Top-down view XY
    ax6 = fig.add_subplot(3, 2, 6)
    
    if valid_clean.any():
        valid_xyz = finger_xyz_clean[valid_clean]
        valid_times = rel_time_ms[valid_clean]
        
        for i in range(len(valid_xyz) - 1):
            t_norm = (valid_times[i] - valid_times.min()) / (valid_times.max() - valid_times.min() + 1e-6)
            color = plt.cm.coolwarm(t_norm)
            ax6.plot([valid_xyz[i, 0], valid_xyz[i+1, 0]], 
                    [valid_xyz[i, 1], valid_xyz[i+1, 1]], 
                    color=color, lw=2, alpha=0.7)
        
        sc = ax6.scatter(finger_xyz_clean[valid_clean, 0], finger_xyz_clean[valid_clean, 1], 
                        c=valid_times, cmap='coolwarm', s=20, alpha=0.8, zorder=5)
        plt.colorbar(sc, ax=ax6, label='Time (ms)')
    
    if np.all(np.isfinite(target_pos)):
        ax6.scatter([target_pos[0]], [target_pos[1]], c='red', s=300, marker='*', 
                   label='Target', zorder=10)
    
    if approach_idx >= 0 and status == 'success' and valid_clean[approach_idx]:
        ax6.scatter([finger_xyz_clean[approach_idx, 0]], 
                   [finger_xyz_clean[approach_idx, 1]], 
                   c='green', s=200, marker='o', edgecolors='black', 
                   linewidths=2, label='Closest approach', zorder=10)
        if np.all(np.isfinite(target_pos)):
            ax6.plot([finger_xyz_clean[approach_idx, 0], target_pos[0]],
                    [finger_xyz_clean[approach_idx, 1], target_pos[1]],
                    'g--', lw=2, alpha=0.7)
    
    if valid_clean.any():
        first_valid = np.where(valid_clean)[0][0]
        ax6.scatter([finger_xyz_clean[first_valid, 0]], 
                   [finger_xyz_clean[first_valid, 1]], 
                   c='blue', s=150, marker='s', edgecolors='black',
                   linewidths=2, label='Start', zorder=10)
    
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_title('Trajectory Top-Down (XY)')
    ax6.legend(loc='best', fontsize=8)
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    
    # Main title
    if status == 'success':
        title_color = 'green'
        straightness = reach_result.get('straightness', np.nan)
        dip_str = " | DIP DETECTED" if dip_detected else ""
        status_str = f'SUCCESS - Error: {min_dist:.2f} | Time: {rel_time_ms[approach_idx]:.0f}ms | Straightness: {straightness:.3f}{dip_str}'
    else:
        title_color = 'red'
        status_str = f'DROPPED - {status}: {drop_reason}'
    
    fig.suptitle(f'{condition_name} - Reach {reach_idx}\n{status_str}',
                fontsize=12, fontweight='bold', color=title_color)
    
    plt.tight_layout()
    
    if status == 'success':
        diag_dir = output_dir / "diagnostics" / condition_name / "success"
    else:
        diag_dir = output_dir / "diagnostics" / condition_name / "dropped"
    
    diag_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(diag_dir / f"reach_{reach_idx:03d}.png", dpi=100, bbox_inches='tight')
    plt.close(fig)


def plot_population_diagnostics(df: pd.DataFrame, output_dir: Path, summary_br_list: Optional[List[int]] = None):
    """Create population-level diagnostic plots."""
    
    if summary_br_list:
        df = df[df['br_idx'].isin(summary_br_list)].copy()
    
    success_df = df[df['status'] == 'success'].copy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Approach time histogram
    ax = axes[0, 0]
    if len(success_df) > 0:
        for cond in success_df['condition_type'].unique():
            subset = success_df[success_df['condition_type'] == cond]
            ax.hist(subset['approach_time_ms'], bins=30, alpha=0.5, label=cond)
    ax.axvline(0, color='r', ls='--', lw=2, label='Event time')
    ax.axvline(MIN_APPROACH_TIME_MS, color='orange', ls=':', lw=2, label=f'Min approach ({MIN_APPROACH_TIME_MS}ms)')
    ax.set_xlabel('Approach Time (ms)')
    ax.set_ylabel('Count')
    ax.set_title('When does closest approach occur?')
    ax.legend()
    
    # 2. Error vs straightness
    ax = axes[0, 1]
    if len(success_df) > 0:
        for cond in success_df['condition_type'].unique():
            subset = success_df[success_df['condition_type'] == cond]
            ax.scatter(subset['straightness'], subset['endpoint_error'], 
                      alpha=0.5, label=cond, s=30)
    ax.set_xlabel('Trajectory Straightness')
    ax.set_ylabel('Endpoint Error')
    ax.set_title('Error vs Straightness')
    ax.legend()
    
    # 3. Finger validity
    ax = axes[0, 2]
    if 'pct_finger_valid_clean' in df.columns:
        ax.hist(df['pct_finger_valid_clean'].dropna(), bins=20, alpha=0.7, color='blue')
        ax.axvline(MIN_FINGER_VALID_PCT, color='r', ls='--', lw=2, 
                   label=f'Min required ({MIN_FINGER_VALID_PCT}%)')
        ax.legend()
    ax.set_xlabel('% Frames with Valid Finger Position')
    ax.set_ylabel('Count')
    ax.set_title('Finger Tracking Quality')
    
    # 4. Drop reasons
    ax = axes[1, 0]
    drop_df = df[df['status'] != 'success']
    if len(drop_df) > 0 and 'drop_reason' in drop_df.columns:
        drop_counts = drop_df['drop_reason'].value_counts()
        ax.pie(drop_counts.values, labels=drop_counts.index, autopct='%1.1f%%')
        ax.set_title(f'Drop Reasons (n={len(drop_df)})')
    else:
        ax.text(0.5, 0.5, 'No dropped reaches', ha='center', va='center', transform=ax.transAxes)
    
    # 5. Velocity dip by BR
    ax = axes[1, 1]
    if len(success_df) > 0 and 'dip_detected' in success_df.columns:
        dip_by_br = success_df.groupby('br_idx')['dip_detected'].mean() * 100
        x = range(len(dip_by_br))
        ax.bar(x, dip_by_br.values, alpha=0.7, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels([str(idx) for idx in dip_by_br.index])
        ax.set_xlabel('BR Index')
        ax.set_ylabel('% with Velocity Dip')
        ax.set_title('Velocity Dip Detection by Condition')
    
    # 6. Error by BR
    ax = axes[1, 2]
    if len(success_df) > 0:
        br_means = success_df.groupby('br_idx')['endpoint_error'].mean().sort_index()
        br_stds = success_df.groupby('br_idx')['endpoint_error'].std().sort_index()
        br_counts = success_df.groupby('br_idx')['endpoint_error'].count().sort_index()
        
        x = range(len(br_means))
        ax.bar(x, br_means.values, yerr=br_stds.values, capsize=3, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{idx}\n(n={br_counts[idx]})' for idx in br_means.index], fontsize=8)
        ax.set_xlabel('BR Index')
        ax.set_ylabel('Mean Endpoint Error')
        ax.set_title('Error by Condition')
    
    plt.tight_layout()
    
    if summary_br_list:
        br_str = '_BR_' + '_'.join(str(b) for b in sorted(summary_br_list))
        fig.savefig(output_dir / f"population_diagnostics{br_str}.png", dpi=150, bbox_inches='tight')
    else:
        fig.savefig(output_dir / "population_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[save] Population diagnostics saved")



def get_condition_label_from_peristim(peristim_path: Path) -> str:
    """
    Extract a descriptive condition label from PeriStim file.
    
    Returns labels like:
    - "Control" for control/baseline
    - "2 pulses @ 400Hz" for stim conditions
    """
    try:
        peri = np.load(peristim_path, allow_pickle=True)
        
        # Try to get stim parameters from meta_json
        if 'meta_json' in peri.files:
            meta = json.loads(peri['meta_json'].item())
            
            # Check for stim parameters
            duration_ms = meta.get('stim_duration_ms', meta.get('duration_ms', None))
            frequency_hz = meta.get('stim_frequency_hz', meta.get('frequency_hz', None))
            
            if duration_ms is not None and frequency_hz is not None:
                # Each pulse is 2.5ms
                pulse_duration_ms = 2.5
                n_pulses = int(duration_ms / pulse_duration_ms)
                return f"{n_pulses} pulses @ {int(frequency_hz)}Hz"
        
        # Try nprw_meta
        if 'nprw_meta' in peri.files:
            nprw_meta = peri['nprw_meta'].item()
            if isinstance(nprw_meta, dict):
                duration_ms = nprw_meta.get('stim_duration_ms', nprw_meta.get('duration_ms', None))
                frequency_hz = nprw_meta.get('stim_frequency_hz', nprw_meta.get('frequency_hz', None))
                
                if duration_ms is not None and frequency_hz is not None:
                    pulse_duration_ms = 2.5
                    n_pulses = int(duration_ms / pulse_duration_ms)
                    return f"{n_pulses} pulses @ {int(frequency_hz)}Hz"
        
    except Exception as e:
        pass
    
    return None


def get_condition_label_from_metadata(br_idx: int, condition_type: str) -> str:
    """
    Get condition label from metadata CSV.
    Returns labels like:
    - "Control (BR 4)" for control/baseline
    - "40p @ 400Hz (BR 5)" for stim conditions
    """
    try:
        meta_df = pd.read_csv(METADATA_CSV)
        
        # Convert BR_File to numeric, coercing errors to NaN
        br_file_numeric = pd.to_numeric(meta_df['BR_File'], errors='coerce')
        
        # Find the row for this BR index
        mask = br_file_numeric == float(br_idx)
        row = meta_df[mask]
        
        if len(row) == 0:
            if condition_type == 'control':
                return f"Control (BR {br_idx})"
            return f"Stim (BR {br_idx})"
        
        row = row.iloc[0]
        
        # Check if this is a baseline/control condition from Notes column
        if 'Notes' in row.index:
            notes = str(row['Notes']).lower() if pd.notna(row['Notes']) else ''
            if 'baseline' in notes or 'rest' in notes:
                return f"Control (BR {br_idx})"
        
        if condition_type == 'control':
            return f"Control (BR {br_idx})"
        
        # Get stim parameters
        duration_ms = None
        frequency_hz = None
        
        if 'Stim_Duration_ms' in meta_df.columns:
            val = row['Stim_Duration_ms']
            if pd.notna(val):
                duration_ms = float(val)
        
        if 'Stim_Frequency_Hz' in meta_df.columns:
            val = row['Stim_Frequency_Hz']
            if pd.notna(val):
                frequency_hz = float(val)
        
        # Build label - always include BR index
        if duration_ms is not None and frequency_hz is not None:
            pulse_duration_ms = 2.5
            n_pulses = int(round(duration_ms / pulse_duration_ms))
            return f"{n_pulses}p @ {int(frequency_hz)}Hz (BR {br_idx})"
        elif frequency_hz is not None:
            return f"Stim @ {int(frequency_hz)}Hz (BR {br_idx})"
        elif duration_ms is not None:
            n_pulses = int(round(duration_ms / 2.5))
            return f"{n_pulses}p (BR {br_idx})"
        
        return f"Stim (BR {br_idx})"
        
    except Exception as e:
        print(f"[warn] Error reading metadata for BR {br_idx}: {e}")
        if condition_type == 'control':
            return f"Control (BR {br_idx})"
        return f"Stim (BR {br_idx})"


def create_endpoint_error_by_condition_svg(
    df: pd.DataFrame, 
    output_dir: Path, 
    condition_labels: dict,
    summary_br_list: Optional[List[int]] = None
):
    """
    Create a publication-quality SVG figure of endpoint error by condition.
    """
    import matplotlib
    matplotlib.rcParams['svg.fonttype'] = 'none'
    
    if summary_br_list:
        df = df[df['br_idx'].isin(summary_br_list)].copy()
    
    success_df = df[df['status'] == 'success'].copy()
    
    if len(success_df) == 0:
        print("[warn] No successful reaches for SVG plot")
        return
    
    # Add condition label column
    success_df['condition_label'] = success_df.apply(
        lambda row: condition_labels.get((row['br_idx'], row['condition_type']), 
                                          f"BR {row['br_idx']}"),
        axis=1
    )
    
    # Group by condition label (each BR is separate now)
    grouped_stats = success_df.groupby('condition_label')['endpoint_error'].agg(['mean', 'std', 'sem', 'count'])
    
    # Order conditions: Control first, then by pulses, frequency, BR number
    def sort_key(label):
        # Extract BR number
        br_match = re.search(r'BR\s*(\d+)', label)
        br_num = int(br_match.group(1)) if br_match else 999
        
        if label.startswith("Control"):
            return (0, 0, 0, br_num)
        
        # Try to parse "Np @ XHz" format
        match = re.match(r'(\d+)p?\s*@\s*(\d+)\s*Hz', label, re.IGNORECASE)
        if match:
            n_pulses = int(match.group(1))
            freq = int(match.group(2))
            return (1, n_pulses, freq, br_num)
        
        return (2, 0, 0, br_num)
    
    sorted_labels = sorted(grouped_stats.index.tolist(), key=sort_key)
    stats = grouped_stats.reindex(sorted_labels)
    
    # Create figure
    n_conditions = len(sorted_labels)
    fig_width = max(6, n_conditions * 1.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    
    x = np.arange(n_conditions)
    width = 0.6
    
    # Colors: control in gray, stim conditions in blue
    colors = ['#808080' if label.startswith("Control") else '#2E86AB' for label in sorted_labels]
    
    # Bar plot with error bars (SEM)
    bars = ax.bar(x, stats['mean'].values, width, 
                  yerr=stats['sem'].values, 
                  capsize=4, 
                  color=colors,
                  edgecolor='black',
                  linewidth=1.5,
                  error_kw={'linewidth': 1.5, 'capthick': 1.5})
    
    # Add individual data points (jittered)
    all_max_y = 0
    for i, label in enumerate(sorted_labels):
        subset = success_df[success_df['condition_label'] == label]['endpoint_error']
        jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
        ax.scatter(x[i] + jitter, subset.values, 
                   color='black', alpha=0.4, s=20, zorder=3)
        if len(subset) > 0:
            all_max_y = max(all_max_y, subset.max())
    
    # Set y limits first so we can position n= labels correctly
    y_max = max(all_max_y, stats['mean'].max() + stats['sem'].max()) * 1.2
    ax.set_ylim(bottom=0, top=y_max)
    
    # Add n values above the highest point for each condition
    for i, label in enumerate(sorted_labels):
        n = int(stats.loc[label, 'count'])
        subset = success_df[success_df['condition_label'] == label]['endpoint_error']
        bar_top = stats.loc[label, 'mean'] + (stats.loc[label, 'sem'] if pd.notna(stats.loc[label, 'sem']) else 0)
        max_point = subset.max() if len(subset) > 0 else bar_top
        y_pos = max(bar_top, max_point) + y_max * 0.03
        ax.text(x[i], y_pos, f'n={n}', ha='center', va='bottom', fontsize=9, style='italic')
    
    # Formatting
    ax.set_ylabel('Endpoint Error (mm)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Stimulation Condition', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_labels, rotation=45, ha='right', fontsize=9)
    ax.set_title('Reach Endpoint Error by Stimulation Condition', fontsize=14, fontweight='bold')
    
    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    if summary_br_list:
        br_str = '_BR_' + '_'.join(str(b) for b in sorted(summary_br_list))
        svg_path = output_dir / f"endpoint_error_by_condition{br_str}.svg"
        png_path = output_dir / f"endpoint_error_by_condition{br_str}.png"
    else:
        svg_path = output_dir / "endpoint_error_by_condition.svg"
        png_path = output_dir / "endpoint_error_by_condition.png"
    
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[save] Endpoint error SVG: {svg_path}")
    print(f"[save] Endpoint error PNG: {png_path}")
    
    # Print statistics table
    print("\n--- Endpoint Error Statistics ---")
    print(f"{'Condition':<30} {'N':>6} {'Mean':>10} {'Std':>10} {'SEM':>10}")
    print("-" * 70)
    for label in sorted_labels:
        row = stats.loc[label]
        std_str = f"{row['std']:>10.3f}" if pd.notna(row['std']) else "       N/A"
        sem_str = f"{row['sem']:>10.3f}" if pd.notna(row['sem']) else "       N/A"
        print(f"{label:<30} {int(row['count']):>6} {row['mean']:>10.3f} {std_str} {sem_str}")


def build_condition_labels(combined_df: pd.DataFrame, peri_roots: dict) -> dict:
    """
    Build a mapping from (br_idx, condition_type) to descriptive labels.
    
    Returns:
        dict: {(br_idx, condition_type): "label string"}
    """
    labels = {}
    
    for _, row in combined_df.drop_duplicates(subset=['br_idx', 'condition_type']).iterrows():
        br_idx = row['br_idx']
        condition_type = row['condition_type']
        
        # First try metadata
        label = get_condition_label_from_metadata(br_idx, condition_type)
        
        if label is None:
            # Try to find the peristim file and extract from there
            peri_root = peri_roots.get(condition_type)
            if peri_root and peri_root.exists():
                peri_files = list(peri_root.glob(f"**/peristim__*BR_{br_idx:03d}*.npz"))
                if not peri_files:
                    peri_files = list(peri_root.glob(f"**/peristim__*BR_{br_idx}*.npz"))
                
                if peri_files:
                    label = get_condition_label_from_peristim(peri_files[0])
        
        if label is None:
            # Fallback
            if condition_type == 'control':
                label = "Control"
            else:
                label = f"{condition_type.replace('_', ' ').title()} (BR {br_idx})"
        
        labels[(br_idx, condition_type)] = label
    
    return labels



def main():
    """Main entry point."""
    
    args = parse_args()
    process_only = get_process_only_list(args)
    
    if process_only:
        print(f"[info] Processing BR indices: {process_only}")
    else:
        print(f"[info] Processing all available conditions")
    
    print(f"[info] Output directory: {RESULTS_ROOT}")
    print(f"[info] Target filter: {args.target}")
    print(f"[info] Minimum approach time: {MIN_APPROACH_TIME_MS} ms")
    
    peri_roots = {
        'control': PERI_ROOT / "control_reaches",
        'stim': PERI_ROOT / "stim_reaches",
        'continuous_stim': PERI_ROOT / "continuous_stim",
    }
    
    if args.condition_type != "all":
        peri_roots = {k: v for k, v in peri_roots.items() if k == args.condition_type}
    
    dlc_3d_root = None
    possible_dlc_roots = [
        Path(VIDEO_ROOT) / "DLC_3D",
        Path(VIDEO_ROOT).parent / "DLC_3D", 
    ]
    
    for path in possible_dlc_roots:
        if path is not None and path.exists():
            dlc_3d_root = path
            break
    
    if dlc_3d_root is None:
        dlc_3d_root = Path(VIDEO_ROOT) / "DLC_3D"
    
    print(f"[info] Using DLC_3D root: {dlc_3d_root}")
    
    behv_aligned_root = BEHV_CKPT_ROOT
    
    all_results = []
    
    for condition_type, peri_root in peri_roots.items():
        if not peri_root.exists():
            continue
        
        peri_files = list(peri_root.glob("**/peristim__*.npz"))
        
        if process_only:
            peri_files = [f for f in peri_files 
                         if (m := re.search(r'BR_(\d+)', f.name)) and int(m.group(1)) in process_only]
        
        print(f"\n[{condition_type}] Found {len(peri_files)} PeriStim files")
        
        for peri_path in sorted(peri_files):
            print(f"\n[process] {peri_path.name}")
            
            df = process_peristim_file(
                peristim_path=peri_path,
                dlc_3d_root=dlc_3d_root,
                behv_aligned_root=behv_aligned_root,
                condition_type=condition_type,
                target_filter=args.target,
                debug=args.debug,
            )
            
            if df is not None and len(df) > 0:
                df['target_type'] = df['trial_label'].str.upper()
                all_results.append(df)
                n_success = (df['status'] == 'success').sum()
                print(f"  → {len(df)} reaches, {n_success} successful ({100*n_success/len(df):.1f}%)")
    
    if not all_results:
        print("[warn] No results to save")
        return
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    print_session_summary(combined_df)
    
    # Save results
    output_parts = []
    if process_only and len(process_only) == 1:
        output_parts.append(f"BR{process_only[0]:03d}")
    elif process_only:
        output_parts.append(f"BR{'_'.join(f'{b:03d}' for b in sorted(process_only))}")
    
    if args.target != 'both':
        output_parts.append(f"target{args.target}")
    
    output_suffix = '_' + '_'.join(output_parts) if output_parts else ""
    
    output_csv = RESULTS_ROOT / f"reach_endpoint_errors{output_suffix}.csv"
    combined_df.to_csv(output_csv, index=False)
    print(f"\n[save] Per-reach metrics: {output_csv}")
    
    # Generate diagnostics
    print("\n[info] Generating individual reach diagnostics...")
    
    for _, row in combined_df.iterrows():
        dlc_path = dlc_3d_root / f"{row['condition_name']}_DLC_3D.csv"
        if dlc_path.exists():
            dlc_df, _ = load_dlc_3d(dlc_path)
            pre_frames = int(np.floor(WIN_MS[0] * CAMERA_FPS / 1000.0))
            post_frames = int(np.ceil(WIN_MS[1] * CAMERA_FPS / 1000.0))
            
            target_kp = row.get('target_keypoint', TARGET_KEYPOINT_A)
            
            plot_single_reach_diagnostic(
                dlc_df=dlc_df,
                event_frame=int(row['event_frame']),
                win_frames=(pre_frames, post_frames),
                fps=CAMERA_FPS,
                reach_idx=int(row['reach_idx']),
                condition_name=row['condition_name'],
                output_dir=RESULTS_ROOT,
                reach_result=row.to_dict(),
                finger_keypoint=FINGER_KEYPOINT,
                target_keypoint=target_kp,
            )
    
    print(f"[save] Diagnostics: {RESULTS_ROOT / 'diagnostics'}")
    
    # Generate summary
    success_df = combined_df[combined_df['status'] == 'success'].copy()
    
    if len(success_df) > 0:
        summary = success_df.groupby(['condition_type', 'target_type']).agg({
            'endpoint_error': ['count', 'mean', 'std', 'median'],
            'approach_time_ms': ['mean', 'std'],
            'path_length': ['mean', 'std'],
            'straightness': ['mean', 'std'],
            'peak_velocity': ['mean', 'std'],
            'dip_detected': 'mean',
        }).round(3)
        
        summary_csv = RESULTS_ROOT / f"reach_endpoint_summary{output_suffix}.csv"
        summary.to_csv(summary_csv)
        print(f"[save] Summary: {summary_csv}")
        print(summary)
        
        create_summary_plots(combined_df, RESULTS_ROOT, summary_br_list=args.summary_br)
    
    plot_population_diagnostics(combined_df, RESULTS_ROOT, summary_br_list=args.summary_br)
    
    # === NEW: Create publication-quality SVG figure ===
    print("\n[info] Building condition labels...")
    condition_labels = build_condition_labels(combined_df, peri_roots)
    
    print("\n[info] Creating endpoint error by condition SVG...")
    create_endpoint_error_by_condition_svg(
        combined_df, 
        RESULTS_ROOT, 
        condition_labels,
        summary_br_list=args.summary_br
    )
    
    print(f"\n[done] Results saved to {RESULTS_ROOT}")


if __name__ == "__main__":
    main()