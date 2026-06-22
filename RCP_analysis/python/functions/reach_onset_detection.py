"""
reach_onset_detection.py

Reusable module for detecting reach onset in behavioral kinematics data.

Purpose:
    Identify when each reach BEGINS for each trial based on:
    1. Speed profile analysis (backward scan from peak velocity)
    2. Position constraints (must start near zero position)
    3. Stability criteria (speed must be low and stable)

Usage:
    from reach_onset_detection import detect_reach_onset, detect_reach_onset_batch
    
    # Single trial
    onset_info = detect_reach_onset(position, speed, time_axis, dt)
    
    # Batch processing
    onset_times, onset_indices, metadata = detect_reach_onset_batch(
        pos_segs, speed_segs, time_axis
    )

Key Parameters (configurable):
    - START_POS_MAX_THRESH: Position must be within this distance from zero at onset
    - LOW_SPEED_THRESH_RATIO: Speed threshold as fraction of peak speed
    - STABLE_WINDOW_MS: Duration speed must stay low to confirm onset
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------
# CONFIGURATION DEFAULTS
# ---------------------------------------------------------------------

@dataclass
class ReachOnsetConfig:
    """Configuration parameters for reach onset detection."""
    
    # Position constraints
    start_pos_max_thresh: float = 0.2  # Start must be within ±0.2 of zero
    
    # Speed thresholds
    low_speed_thresh_ratio: float = 0.05  # 5% of peak speed = "near zero"
    
    # Stability window
    stable_window_ms: float = 20.0  # Speed must stay low for this duration
    
    # Search limits
    max_reach_duration_ms: float = 800.0  # Don't search beyond this
    min_reach_duration_ms: float = 100.0  # Minimum valid reach duration
    
    # Smoothing (for velocity computation if needed)
    velocity_smooth_window_ms: float = 75.0
    savgol_polyorder: int = 3


# Default configuration
DEFAULT_CONFIG = ReachOnsetConfig()


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def compute_speed_from_position(position: np.ndarray, dt: float, 
                                 config: ReachOnsetConfig = None) -> np.ndarray:
    """
    Compute speed (magnitude of velocity) from position trace.
    
    Parameters
    ----------
    position : np.ndarray
        1D position trace (T,)
    dt : float
        Time step in milliseconds
    config : ReachOnsetConfig, optional
        Configuration parameters
        
    Returns
    -------
    speed : np.ndarray
        Speed trace (absolute velocity), shape (T,)
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    if len(position) < 7:
        # Too short for Savgol, use simple gradient
        velocity = np.gradient(position, dt)
        return np.abs(velocity)
    
    # Handle NaN values
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
            return np.abs(np.gradient(position, dt))
    else:
        position_clean = position
    
    # Compute velocity using Savgol derivative
    window_length = int(config.velocity_smooth_window_ms / dt)
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(window_length, 7)
    
    if window_length > len(position_clean):
        window_length = len(position_clean)
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(window_length, 3)
    
    polyorder = min(config.savgol_polyorder, window_length - 1)
    
    try:
        velocity = savgol_filter(position_clean, window_length, polyorder=polyorder,
                                  deriv=1, delta=dt)
    except (np.linalg.LinAlgError, ValueError):
        velocity = np.gradient(position_clean, dt)
    
    return np.abs(velocity)


def smooth_1d(arr: np.ndarray, dt: float, window_ms: float = 50.0,
              polyorder: int = 3) -> np.ndarray:
    """Apply Savgol smoothing to 1D array."""
    if len(arr) < 5:
        return arr.copy()
    
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
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
    
    window_length = int(window_ms / dt)
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
        return gaussian_filter1d(arr_clean, sigma)
    
    poly = min(polyorder, window_length - 1)
    
    try:
        return savgol_filter(arr_clean, window_length, poly)
    except (np.linalg.LinAlgError, ValueError):
        sigma = max(20.0 / dt, 0.5)
        return gaussian_filter1d(arr_clean, sigma)


# ---------------------------------------------------------------------
# CORE DETECTION FUNCTIONS
# ---------------------------------------------------------------------

@dataclass
class ReachOnsetResult:
    """Result of reach onset detection for a single trial."""
    
    # Core results
    onset_time: Optional[float]  # Time of reach onset (ms), None if not found
    onset_idx: Optional[int]     # Index of reach onset, None if not found
    
    # Peak velocity info (used as reference)
    peak_speed_time: float       # Time of peak speed
    peak_speed_idx: int          # Index of peak speed
    peak_speed: float            # Value of peak speed
    
    # Detection metadata
    speed_threshold: float       # Threshold used for detection
    method: str                  # 'speed_stable', 'position_constraint', 'fallback', 'failed'
    
    # Validity
    is_valid: bool               # Whether a valid onset was found
    rejection_reason: str        # Reason if invalid


def detect_reach_onset(
    position: np.ndarray,
    speed: np.ndarray,
    time_axis: np.ndarray,
    dt: float,
    config: ReachOnsetConfig = None,
    search_start_idx: int = 0
) -> ReachOnsetResult:
    """
    Detect reach onset for a SINGLE trial.
    
    Algorithm:
    1. Find peak speed in the trial (within max_reach_duration)
    2. Scan BACKWARD from peak speed
    3. Find where speed drops below threshold AND position is near zero
    4. Verify stability (speed stays low for stable_window)
    
    Parameters
    ----------
    position : np.ndarray
        Position trace (T,) - typically X coordinate of tracked keypoint
    speed : np.ndarray
        Speed trace (T,) - absolute velocity
    time_axis : np.ndarray
        Time values (T,) in milliseconds
    dt : float
        Time step in milliseconds
    config : ReachOnsetConfig, optional
        Detection parameters
    search_start_idx : int
        Index to start searching from (default: 0)
        
    Returns
    -------
    ReachOnsetResult
        Dataclass containing onset time, index, and metadata
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    T = len(position)
    
    # Validate inputs
    if T < 10 or len(speed) != T or len(time_axis) != T:
        return ReachOnsetResult(
            onset_time=None, onset_idx=None,
            peak_speed_time=0, peak_speed_idx=0, peak_speed=0,
            speed_threshold=0, method='failed',
            is_valid=False, rejection_reason='Invalid input arrays'
        )
    
    # 1. Find search window based on max_reach_duration
    max_samples = int(config.max_reach_duration_ms / dt)
    search_end_idx = min(T - 1, search_start_idx + max_samples)
    
    # 2. Find peak speed in search window
    search_speed = speed[search_start_idx:search_end_idx + 1]
    if search_speed.size == 0 or np.all(np.isnan(search_speed)):
        return ReachOnsetResult(
            onset_time=None, onset_idx=None,
            peak_speed_time=time_axis[search_start_idx], 
            peak_speed_idx=search_start_idx, peak_speed=0,
            speed_threshold=0, method='failed',
            is_valid=False, rejection_reason='No valid speed data'
        )
    
    peak_rel_idx = np.nanargmax(search_speed)
    peak_idx = search_start_idx + peak_rel_idx
    peak_speed = speed[peak_idx]
    peak_time = time_axis[peak_idx]
    
    # 3. Calculate speed threshold
    speed_threshold = peak_speed * config.low_speed_thresh_ratio
    
    # 4. Calculate stability window in samples
    win_samples = max(2, int(config.stable_window_ms / dt))
    
    # 5. Backward scan from peak to find onset
    onset_idx = None
    onset_time = None
    method = 'failed'
    rejection_reason = ''
    
    for k in range(peak_idx, search_start_idx - 1, -1):
        current_speed = speed[k]
        current_pos = position[k]
        
        # Check speed criterion
        speed_ok = current_speed < speed_threshold
        
        # Check position criterion (near zero)
        position_ok = np.abs(current_pos) <= config.start_pos_max_thresh
        
        if speed_ok and position_ok:
            # Check stability: speed must stay low for the window BEFORE this point
            window_start = max(0, k - win_samples)
            
            if k > window_start:
                window_speed = speed[window_start:k]
                window_mean = np.nanmean(window_speed)
                
                if window_mean < speed_threshold:
                    # Found valid onset!
                    onset_idx = k
                    onset_time = time_axis[k]
                    method = 'speed_stable'
                    break
            else:
                # At start of trial, accept this point
                onset_idx = k
                onset_time = time_axis[k]
                method = 'start_of_trial'
                break
        elif speed_ok and onset_idx is None:
            # Speed is low but position not near zero - might be best we can do
            # Continue searching for better point
            pass
    
    # 6. Fallback: if no onset found with position constraint, try just speed
    if onset_idx is None:
        for k in range(peak_idx, search_start_idx - 1, -1):
            if speed[k] < speed_threshold:
                window_start = max(0, k - win_samples)
                if k > window_start:
                    if np.nanmean(speed[window_start:k]) < speed_threshold:
                        onset_idx = k
                        onset_time = time_axis[k]
                        method = 'speed_only_fallback'
                        break
                else:
                    onset_idx = k
                    onset_time = time_axis[k]
                    method = 'speed_only_fallback'
                    break
    
    # 7. Ultimate fallback: use search_start_idx
    if onset_idx is None:
        onset_idx = search_start_idx
        onset_time = time_axis[search_start_idx]
        method = 'search_start_fallback'
        rejection_reason = 'No valid onset found, using search start'
    
    # 8. Validate: check minimum reach duration
    reach_duration = peak_time - onset_time
    is_valid = reach_duration >= config.min_reach_duration_ms
    
    if not is_valid and not rejection_reason:
        rejection_reason = f'Reach duration {reach_duration:.1f}ms < minimum {config.min_reach_duration_ms}ms'
    
    return ReachOnsetResult(
        onset_time=onset_time,
        onset_idx=onset_idx,
        peak_speed_time=peak_time,
        peak_speed_idx=peak_idx,
        peak_speed=peak_speed,
        speed_threshold=speed_threshold,
        method=method,
        is_valid=is_valid,
        rejection_reason=rejection_reason
    )


def detect_reach_onset_batch(
    pos_segs: np.ndarray,
    speed_segs: np.ndarray,
    time_axis: np.ndarray,
    config: ReachOnsetConfig = None,
    keypoint_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Detect reach onset for multiple trials (batch processing).
    
    Parameters
    ----------
    pos_segs : np.ndarray
        Position segments (n_trials, n_keypoints, T) or (n_trials, T)
    speed_segs : np.ndarray
        Speed segments (n_trials, n_keypoints, T) or (n_trials, T)
    time_axis : np.ndarray
        Time values (T,) in milliseconds
    config : ReachOnsetConfig, optional
        Detection parameters
    keypoint_idx : int
        Which keypoint to use if pos_segs has keypoint dimension
        
    Returns
    -------
    onset_times : np.ndarray
        Array of onset times (n_trials,), NaN where detection failed
    onset_indices : np.ndarray
        Array of onset indices (n_trials,), -1 where detection failed
    results : list[ReachOnsetResult]
        Full results for each trial
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Handle different input shapes
    if pos_segs.ndim == 3:
        # (n_trials, n_keypoints, T)
        n_trials = pos_segs.shape[0]
        position_traces = pos_segs[:, keypoint_idx, :]
        speed_traces = speed_segs[:, keypoint_idx, :] if speed_segs.ndim == 3 else speed_segs
    elif pos_segs.ndim == 2:
        # (n_trials, T)
        n_trials = pos_segs.shape[0]
        position_traces = pos_segs
        speed_traces = speed_segs
    else:
        raise ValueError(f"Unexpected pos_segs shape: {pos_segs.shape}")
    
    # Compute dt
    dt = time_axis[1] - time_axis[0] if len(time_axis) > 1 else 10.0
    if dt < 0.9:
        dt *= 1000.0  # Convert to ms if in seconds
    
    # Process each trial
    onset_times = np.full(n_trials, np.nan)
    onset_indices = np.full(n_trials, -1, dtype=int)
    results = []
    
    for i in range(n_trials):
        position = position_traces[i, :]
        speed = speed_traces[i, :] if speed_traces.ndim == 2 else np.abs(speed_traces[i, keypoint_idx, :])
        
        # Use absolute speed
        speed = np.abs(speed)
        
        result = detect_reach_onset(
            position=position,
            speed=speed,
            time_axis=time_axis,
            dt=dt,
            config=config
        )
        
        results.append(result)
        
        if result.is_valid and result.onset_time is not None:
            onset_times[i] = result.onset_time
            onset_indices[i] = result.onset_idx
    
    return onset_times, onset_indices, results


def compute_2d_speed(x: np.ndarray, y: np.ndarray, dt: float,
                     config: ReachOnsetConfig = None) -> np.ndarray:
    """
    Compute 2D speed from X and Y position traces.
    
    Parameters
    ----------
    x, y : np.ndarray
        Position traces (T,)
    dt : float
        Time step in milliseconds
    config : ReachOnsetConfig, optional
        Configuration for smoothing
        
    Returns
    -------
    speed_2d : np.ndarray
        2D speed (magnitude of velocity vector)
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Smooth positions
    x_smooth = smooth_1d(x, dt, window_ms=50.0)
    y_smooth = smooth_1d(y, dt, window_ms=50.0)
    
    # Compute velocities
    vx = compute_speed_from_position(x_smooth, dt, config)  # This returns |velocity|
    vy = compute_speed_from_position(y_smooth, dt, config)
    
    # For 2D speed, we need actual velocities (with sign), then compute magnitude
    # Recompute with gradient to preserve sign
    vx_signed = np.gradient(x_smooth, dt)
    vy_signed = np.gradient(y_smooth, dt)
    
    speed_2d = np.sqrt(vx_signed**2 + vy_signed**2)
    
    return speed_2d


# ---------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------

def align_traces_to_onset(
    traces: np.ndarray,
    onset_indices: np.ndarray,
    time_axis: np.ndarray,
    pre_onset_ms: float = 200.0,
    post_onset_ms: float = 600.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align traces to reach onset, creating a common time axis.
    
    Parameters
    ----------
    traces : np.ndarray
        Traces to align (n_trials, T)
    onset_indices : np.ndarray
        Onset index for each trial (n_trials,)
    time_axis : np.ndarray
        Original time axis (T,)
    pre_onset_ms : float
        Time before onset to include
    post_onset_ms : float
        Time after onset to include
        
    Returns
    -------
    aligned_traces : np.ndarray
        Aligned traces (n_trials, T_aligned), NaN padded where needed
    aligned_time : np.ndarray
        New time axis relative to onset (T_aligned,)
    """
    dt = time_axis[1] - time_axis[0] if len(time_axis) > 1 else 10.0
    
    # Create new time axis
    n_pre = int(pre_onset_ms / dt)
    n_post = int(post_onset_ms / dt)
    T_aligned = n_pre + n_post
    aligned_time = np.arange(-n_pre, n_post) * dt
    
    n_trials = traces.shape[0]
    aligned_traces = np.full((n_trials, T_aligned), np.nan)
    
    for i in range(n_trials):
        onset_idx = onset_indices[i]
        if onset_idx < 0:
            continue
        
        # Source indices
        src_start = max(0, onset_idx - n_pre)
        src_end = min(traces.shape[1], onset_idx + n_post)
        
        # Destination indices
        dst_start = n_pre - (onset_idx - src_start)
        dst_end = dst_start + (src_end - src_start)
        
        aligned_traces[i, dst_start:dst_end] = traces[i, src_start:src_end]
    
    return aligned_traces, aligned_time


def get_pre_post_regions(
    onset_time: float,
    time_axis: np.ndarray,
    pre_duration_ms: float = 200.0,
    post_duration_ms: float = 600.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get boolean masks for pre-reach and post-reach (during reach) time regions.
    
    Parameters
    ----------
    onset_time : float
        Reach onset time in ms
    time_axis : np.ndarray
        Time axis
    pre_duration_ms : float
        Duration of pre-reach window
    post_duration_ms : float
        Duration of post-reach window
        
    Returns
    -------
    pre_mask : np.ndarray
        Boolean mask for pre-reach period
    post_mask : np.ndarray
        Boolean mask for post-reach (during reach) period
    """
    pre_mask = (time_axis >= onset_time - pre_duration_ms) & (time_axis < onset_time)
    post_mask = (time_axis >= onset_time) & (time_axis < onset_time + post_duration_ms)
    
    return pre_mask, post_mask