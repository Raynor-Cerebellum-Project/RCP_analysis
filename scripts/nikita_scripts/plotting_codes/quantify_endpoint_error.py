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
WIN_MS = (-600.0, 3000.0)
CAMERA_FPS = 100.0
MAX_INTERP_GAP_FRAMES = 100
MIN_FINGER_VALID_PCT = 50.0
MIN_APPROACH_TIME_MS = 100.0


# Velocity smoothing window (frames)
VELOCITY_SMOOTH_WINDOW = 3

# for removing points based on 2D kinematics likelihood
LIKELIHOOD_THRESHOLD = 0.35



def load_2d_dlc_likelihoods(
    dlc_2d_root: Path,
    condition_name: str,  # e.g., "NRR_RW022_007"
    keypoint: str = "middle",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load likelihood values for a keypoint from both camera views.
    
    Args:
        dlc_2d_root: Path to Video/DLC/ directory
        condition_name: e.g., "NRR_RW022_007"
        keypoint: Body part name (e.g., "middle", "TA", "TB")
        
    Returns:
        likelihood_cam0: (N,) array of likelihoods from camera 0
        likelihood_cam1: (N,) array of likelihoods from camera 1
    """
    # Find the CSV files - they have a complex suffix
    cam0_files = list(dlc_2d_root.glob(f"{condition_name}_Cam-0DLC_*.csv"))
    cam1_files = list(dlc_2d_root.glob(f"{condition_name}_Cam-1DLC_*.csv"))
    
    if not cam0_files or not cam1_files:
        raise FileNotFoundError(f"Could not find 2D DLC files for {condition_name}")
    
    cam0_path = cam0_files[0]
    cam1_path = cam1_files[0]
    
    # Load with multi-level header (rows 1 and 2 are bodyparts and coords)
    df_cam0 = pd.read_csv(cam0_path, header=[1, 2], index_col=0)
    df_cam1 = pd.read_csv(cam1_path, header=[1, 2], index_col=0)
    
    # Extract likelihood column for the keypoint
    # Column structure: (bodypart, coord) -> e.g., ('middle', 'likelihood')
    likelihood_cam0 = df_cam0[(keypoint, 'likelihood')].values.astype(float)
    likelihood_cam1 = df_cam1[(keypoint, 'likelihood')].values.astype(float)
    
    return likelihood_cam0, likelihood_cam1

def get_likelihood_mask(
    dlc_2d_root: Path,
    condition_name: str,
    keypoint: str,
    frame_indices: np.ndarray,
    threshold: float = LIKELIHOOD_THRESHOLD,
) -> Tuple[np.ndarray, int, int]:
    """
    Create a mask where True = VALID (both cameras have likelihood >= threshold).
    
    Args:
        dlc_2d_root: Path to Video/DLC/ directory
        condition_name: Condition name (e.g., "NRR_RW022_007")
        keypoint: Body part name
        frame_indices: Frame indices to check
        threshold: Minimum likelihood (default 0.4)
        
    Returns:
        valid_mask: (N,) boolean array, True where data is reliable
        n_cam0_low: Number of frames with low cam0 likelihood
        n_cam1_low: Number of frames with low cam1 likelihood
    """
    try:
        likelihood_cam0, likelihood_cam1 = load_2d_dlc_likelihoods(
            dlc_2d_root, condition_name, keypoint
        )
    except (FileNotFoundError, KeyError) as e:
        print(f"    [warn] Could not load 2D likelihoods for {keypoint}: {e}")
        # Return all True (don't mask anything) if we can't load likelihoods
        return np.ones(len(frame_indices), dtype=bool), 0, 0
    
    n_frames_2d = len(likelihood_cam0)
    
    valid_mask = np.zeros(len(frame_indices), dtype=bool)
    cam0_low = np.zeros(len(frame_indices), dtype=bool)
    cam1_low = np.zeros(len(frame_indices), dtype=bool)
    
    for i, frame_idx in enumerate(frame_indices):
        # Convert to 0-based index for the likelihood arrays
        idx = int(frame_idx) - 1  # Adjust if your indexing differs
        
        if 0 <= idx < n_frames_2d:
            cam0_ok = likelihood_cam0[idx] >= threshold
            cam1_ok = likelihood_cam1[idx] >= threshold
            valid_mask[i] = cam0_ok and cam1_ok
            cam0_low[i] = not cam0_ok
            cam1_low[i] = not cam1_ok
        else:
            valid_mask[i] = False
            cam0_low[i] = True
            cam1_low[i] = True
    
    return valid_mask, int(cam0_low.sum()), int(cam1_low.sum())

def apply_likelihood_mask(
    xyz: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Set xyz values to NaN where likelihood is below threshold.
    
    Args:
        xyz: (N, 3) position array
        valid_mask: (N,) boolean array, True = valid
        
    Returns:
        xyz_masked: (N, 3) array with invalid points set to NaN
    """
    xyz_masked = xyz.copy()
    xyz_masked[~valid_mask] = np.nan
    return xyz_masked


def remove_high_velocity_points(
    xyz: np.ndarray,
    fps: float,
    max_speed: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple and direct: compute velocity, remove any frame where velocity > max_speed.
    
    Args:
        xyz: (N, 3) position array
        fps: Frames per second
        max_speed: Maximum allowed speed (units per second)
        
    Returns:
        xyz_clean: (N, 3) with bad frames set to NaN
        outlier_mask: (N,) boolean, True = removed
    """
    n = len(xyz)
    outlier_mask = np.zeros(n, dtype=bool)
    xyz_clean = xyz.copy()
    
    if n < 2:
        return xyz_clean, outlier_mask
    
    dt = 1.0 / fps
    
    # Compute velocity magnitude at each frame (using central difference where possible)
    velocity = np.full(n, np.nan)
    
    for i in range(n):
        if i == 0:
            # Forward difference
            if np.all(np.isfinite(xyz[0])) and np.all(np.isfinite(xyz[1])):
                velocity[0] = np.linalg.norm(xyz[1] - xyz[0]) / dt
        elif i == n - 1:
            # Backward difference
            if np.all(np.isfinite(xyz[-1])) and np.all(np.isfinite(xyz[-2])):
                velocity[-1] = np.linalg.norm(xyz[-1] - xyz[-2]) / dt
        else:
            # Central difference (velocity at frame i based on frames i-1 and i+1)
            if np.all(np.isfinite(xyz[i-1])) and np.all(np.isfinite(xyz[i+1])):
                velocity[i] = np.linalg.norm(xyz[i+1] - xyz[i-1]) / (2 * dt)
    
    # Mark frames where velocity exceeds threshold
    high_velocity = velocity > max_speed
    
    # For high velocity frames, we need to decide which point is bad.
    # Strategy: if velocity[i] is high, mark frame i as bad
    # (the velocity "belongs" to frame i in central difference)
    outlier_mask = high_velocity & np.isfinite(velocity)
    
    # Also mark frames adjacent to high-velocity regions
    # (because if v[i] is computed from frames i-1 and i+1, 
    #  and it's too high, at least one of i-1, i, i+1 is wrong)
    for i in range(n):
        if outlier_mask[i]:
            # The velocity at i is too high - mark i as bad
            # Also check neighbors
            if i > 0:
                # Check velocity from i-1 to i
                if np.all(np.isfinite(xyz[i-1])) and np.all(np.isfinite(xyz[i])):
                    speed_to_i = np.linalg.norm(xyz[i] - xyz[i-1]) / dt
                    if speed_to_i > max_speed:
                        outlier_mask[i] = True  # Already true, but explicit
            if i < n - 1:
                # Check velocity from i to i+1
                if np.all(np.isfinite(xyz[i])) and np.all(np.isfinite(xyz[i+1])):
                    speed_from_i = np.linalg.norm(xyz[i+1] - xyz[i]) / dt
                    if speed_from_i > max_speed:
                        outlier_mask[i] = True
    
    # Set outlier positions to NaN
    xyz_clean[outlier_mask] = np.nan
    
    return xyz_clean, outlier_mask

def remove_bracketed_outlier_regions(
    xyz: np.ndarray,
    fps: float,
    max_speed: float = 100.0,
    max_region_frames: int = 50,  # Don't remove regions larger than this
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove entire regions that are bracketed by high-velocity jumps.
    
    If we detect: normal -> [high velocity jump] -> garbage -> [high velocity jump] -> normal
    Then the entire garbage region (including the boundary points) gets removed.
    
    Args:
        xyz: (N, 3) position array
        fps: Frames per second
        max_speed: Velocity threshold for detecting jumps
        max_region_frames: Maximum size of region to remove (safety limit)
        
    Returns:
        xyz_clean: (N, 3) with bad regions set to NaN
        outlier_mask: (N,) boolean, True = removed
    """
    n = len(xyz)
    outlier_mask = np.zeros(n, dtype=bool)
    xyz_clean = xyz.copy()
    
    if n < 3:
        return xyz_clean, outlier_mask
    
    dt = 1.0 / fps
    
    # Step 1: Find all high-velocity transitions
    high_velocity_transitions = []  # List of (from_idx, to_idx) pairs
    
    for i in range(n - 1):
        if np.all(np.isfinite(xyz[i])) and np.all(np.isfinite(xyz[i + 1])):
            displacement = np.linalg.norm(xyz[i + 1] - xyz[i])
            speed = displacement / dt
            
            if speed > max_speed:
                high_velocity_transitions.append(i)  # Transition happens between i and i+1
    
    if len(high_velocity_transitions) < 2:
        # Need at least 2 jumps to bracket a region
        # Still remove individual high-velocity points
        for trans_idx in high_velocity_transitions:
            outlier_mask[trans_idx + 1] = True
            xyz_clean[trans_idx + 1] = np.nan
        return xyz_clean, outlier_mask
    
    # Step 2: Find bracketed regions (jump out, then jump back)
    # A bracketed region is where we jump away and then jump back to near the original position
    
    regions_to_remove = []
    used_transitions = set()
    
    for i, start_trans in enumerate(high_velocity_transitions):
        if start_trans in used_transitions:
            continue
            
        start_pos = xyz[start_trans]  # Position before the jump
        
        if not np.all(np.isfinite(start_pos)):
            continue
        
        # Look for a return jump within max_region_frames
        for j, end_trans in enumerate(high_velocity_transitions[i + 1:], i + 1):
            region_size = end_trans - start_trans
            
            if region_size > max_region_frames:
                break  # Too far, stop looking
            
            if end_trans in used_transitions:
                continue
            
            # Position after the return jump
            end_pos = xyz[end_trans + 1] if end_trans + 1 < n else None
            
            if end_pos is None or not np.all(np.isfinite(end_pos)):
                continue
            
            # Check if we returned to near the starting position
            # The return position should be closer to start than the jumped-to position
            jumped_to_pos = xyz[start_trans + 1]
            
            if not np.all(np.isfinite(jumped_to_pos)):
                continue
            
            dist_start_to_jumped = np.linalg.norm(jumped_to_pos - start_pos)
            dist_start_to_end = np.linalg.norm(end_pos - start_pos)
            
            # Criterion: end position is much closer to start than the jumped position
            # This indicates we jumped away and came back
            if dist_start_to_end < dist_start_to_jumped * 0.5:
                # Found a bracketed region!
                regions_to_remove.append((start_trans + 1, end_trans))  # Remove from after first jump to including second jump point
                used_transitions.add(start_trans)
                used_transitions.add(end_trans)
                break
    
    # Step 3: Remove the bracketed regions
    for region_start, region_end in regions_to_remove:
        for idx in range(region_start, region_end + 1):
            outlier_mask[idx] = True
            xyz_clean[idx] = np.nan
    
    # Step 4: Also remove any remaining isolated high-velocity transitions
    for trans_idx in high_velocity_transitions:
        if trans_idx not in used_transitions:
            # This is an isolated jump - remove the point we jumped TO
            outlier_mask[trans_idx + 1] = True
            xyz_clean[trans_idx + 1] = np.nan
    
    return xyz_clean, outlier_mask


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


# def clean_finger_signal_v2(
#     xyz: np.ndarray, 
#     fps: float,
#     rel_time_ms: np.ndarray = None,  # ADD THIS PARAMETER
#     range_threshold_pct: float = 0.25,
#     chunk_threshold_pct: float = 0.35,
#     n_chunks: int = 5,
#     start_end_frames: int = 10,
#     max_interp_gap: int = 25,
#     outlier_iqr_multiplier: float = 2.5,
#     min_range_for_filtering: float = 0.5,
#     reach_end_time_ms: float = 300.0,  # NEW: hardcoded reach end time
#     max_speed: float = 100.0,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Clean finger position signal using robust range-based and chunk-based outlier detection.
    
#     Uses movement start (~0ms) and reach endpoint (~300ms) for range estimation,
#     not the full window which may include the return movement.
#     """
#     n = len(xyz)
#     combined_outlier_mask = np.zeros(n, dtype=bool)
#     xyz_clean = xyz.copy()
    
#     if n < 10:
#         return xyz_clean, combined_outlier_mask

#     jump_outliers = detect_impossible_jumps(xyz_clean, fps, max_speed=max_speed)
#     combined_outlier_mask |= jump_outliers
#     xyz_clean[jump_outliers] = np.nan

    
#     # === Determine start and end indices for range estimation ===
#     if rel_time_ms is not None:
#         # Start: around 0ms (movement onset)
#         start_mask = (rel_time_ms >= -50) & (rel_time_ms <= 50)
#         # End: around reach endpoint (~300ms), not return
#         end_mask = (rel_time_ms >= reach_end_time_ms - 50) & (rel_time_ms <= reach_end_time_ms + 50)
        
#         start_indices_global = np.where(start_mask)[0]
#         end_indices_global = np.where(end_mask)[0]
#     else:
#         # Fallback: use first/last N frames
#         start_indices_global = np.arange(min(start_end_frames, n))
#         end_indices_global = np.arange(max(0, n - start_end_frames), n)
    
#     # === Compute overall movement magnitude across all axes ===
#     overall_ranges = []
#     for dim in range(3):
#         valid_mask = np.isfinite(xyz[:, dim])
#         if valid_mask.sum() > 5:
#             overall_ranges.append(np.ptp(xyz[valid_mask, dim]))
#         else:
#             overall_ranges.append(0)
    
#     max_overall_range = max(overall_ranges) if overall_ranges else 1.0
#     if max_overall_range < 1e-6:
#         max_overall_range = 1.0
    
#     for dim in range(3):
#         signal = xyz[:, dim].copy()
#         dim_outliers = np.zeros(n, dtype=bool)
        
#         valid_mask = np.isfinite(signal)
#         if valid_mask.sum() < 10:
#             continue
        
#         valid_values = signal[valid_mask]
        
#         # === STEP 1: IQR-based extreme outlier detection ===
#         q1 = np.percentile(valid_values, 25)
#         q3 = np.percentile(valid_values, 75)
#         iqr = q3 - q1
        
#         if iqr < 1e-6:
#             iqr = np.std(valid_values) * 1.35
        
#         iqr_lower = q1 - outlier_iqr_multiplier * iqr
#         iqr_upper = q3 + outlier_iqr_multiplier * iqr
        
#         extreme_outliers = valid_mask & ((signal < iqr_lower) | (signal > iqr_upper))
#         dim_outliers |= extreme_outliers
        
#         # === STEP 2: Robust start/end estimation using hardcoded times ===
#         clean_valid_mask = valid_mask & ~extreme_outliers
        
#         # Get start median (around 0ms)
#         start_valid = clean_valid_mask.copy()
#         if len(start_indices_global) > 0:
#             start_valid &= np.isin(np.arange(n), start_indices_global)
#         start_valid_idx = np.where(start_valid)[0]
        
#         if len(start_valid_idx) >= 3:
#             start_median = np.median(signal[start_valid_idx])
#         else:
#             # Fallback: first valid points
#             clean_valid_indices = np.where(clean_valid_mask)[0]
#             if len(clean_valid_indices) >= 3:
#                 start_median = np.median(signal[clean_valid_indices[:start_end_frames]])
#             else:
#                 continue
        
#         # Get end median (around 300ms, not return)
#         end_valid = clean_valid_mask.copy()
#         if len(end_indices_global) > 0:
#             end_valid &= np.isin(np.arange(n), end_indices_global)
#         end_valid_idx = np.where(end_valid)[0]
        
#         if len(end_valid_idx) >= 3:
#             end_median = np.median(signal[end_valid_idx])
#         else:
#             # Fallback: use global median in that time region
#             clean_valid_indices = np.where(clean_valid_mask)[0]
#             if len(clean_valid_indices) >= 3:
#                 end_median = np.median(signal[clean_valid_indices[-start_end_frames:]])
#             else:
#                 continue
        
#         # === STEP 3: Range-based detection ===
#         range_min = min(start_median, end_median)
#         range_max = max(start_median, end_median)
#         range_span = range_max - range_min
        
#         effective_range_span = max(
#             range_span,
#             max_overall_range * 0.3,
#             iqr * 1.5,
#             min_range_for_filtering
#         )
        
#         axis_has_meaningful_movement = range_span > min_range_for_filtering
        
#         if axis_has_meaningful_movement:
#             buffer = effective_range_span * range_threshold_pct
#             min_buffer = iqr * 0.5
#             buffer = max(buffer, min_buffer)
            
#             allowed_min = range_min - buffer
#             allowed_max = range_max + buffer
            
#             range_outliers = clean_valid_mask & ((signal < allowed_min) | (signal > allowed_max))
#             dim_outliers |= range_outliers
        
#         # === STEP 4: Chunk-based detection ===
#         still_clean_mask = valid_mask & ~dim_outliers
#         chunk_size = max(1, n // n_chunks)
        
#         for chunk_idx in range(n_chunks):
#             chunk_start = chunk_idx * chunk_size
#             chunk_end = min((chunk_idx + 1) * chunk_size, n)
            
#             if chunk_idx == n_chunks - 1:
#                 chunk_end = n
            
#             chunk_data = signal[chunk_start:chunk_end]
#             chunk_clean = still_clean_mask[chunk_start:chunk_end]
            
#             if chunk_clean.sum() < 3:
#                 continue
            
#             chunk_median = np.median(chunk_data[chunk_clean])
#             chunk_iqr = np.percentile(chunk_data[chunk_clean], 75) - np.percentile(chunk_data[chunk_clean], 25)
            
#             if chunk_iqr < 1e-6:
#                 chunk_iqr = effective_range_span * 0.2
            
#             for i in range(chunk_start, chunk_end):
#                 if not still_clean_mask[i] or dim_outliers[i]:
#                     continue
                
#                 deviation = abs(signal[i] - chunk_median)
                
#                 threshold = max(
#                     chunk_iqr * (chunk_threshold_pct / 0.2),
#                     effective_range_span * chunk_threshold_pct
#                 )
                
#                 if not axis_has_meaningful_movement:
#                     threshold *= 1.5
                
#                 if deviation > threshold:
#                     dim_outliers[i] = True
        
#         combined_outlier_mask |= dim_outliers
#         signal[dim_outliers] = np.nan
#         xyz_clean[:, dim] = signal
    
#     # Interpolate the outlier regions
#     xyz_clean = interpolate_small_gaps(xyz_clean, max_gap=max_interp_gap)
    
#     return xyz_clean, combined_outlier_mask


def clean_finger_signal_v2(
    xyz: np.ndarray, 
    fps: float,
    rel_time_ms: np.ndarray = None,
    max_speed: float = 100.0,
    max_interp_gap: int = 5,  # Only interpolate very small gaps (natural tracking loss)
    position_outlier_std: float = 3.0,  # Remove points > N std from median
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean finger position signal by:
    1. Finding high-velocity jumps
    2. Marking entire regions between jump-pairs as bad
    3. Removing position outliers
    4. Only interpolating small gaps (not cleaned regions)
    """
    n = len(xyz)
    outlier_mask = np.zeros(n, dtype=bool)
    xyz_clean = xyz.copy()
    
    if n < 5:
        return xyz_clean, outlier_mask
    
    dt = 1.0 / fps
    
    # === STEP 1: Find all frames involved in high-velocity transitions ===
    velocity_outliers = np.zeros(n, dtype=bool)
    
    for i in range(n - 1):
        if np.all(np.isfinite(xyz[i])) and np.all(np.isfinite(xyz[i + 1])):
            speed = np.linalg.norm(xyz[i + 1] - xyz[i]) / dt
            if speed > max_speed:
                # Mark BOTH frames involved in the jump
                velocity_outliers[i] = True
                velocity_outliers[i + 1] = True
    
    # === STEP 2: Find bracketed regions and mark everything inside ===
    # Look for pattern: good -> bad -> ... -> bad -> good
    # Where the "good" points before and after are similar
    
    jump_indices = np.where(velocity_outliers)[0]
    
    if len(jump_indices) >= 2:
        # Find pairs of jumps that bracket a bad region
        i = 0
        while i < len(jump_indices):
            start_jump = jump_indices[i]
            
            # Find the last good point before this jump
            good_before = start_jump - 1
            while good_before >= 0 and (velocity_outliers[good_before] or not np.all(np.isfinite(xyz[good_before]))):
                good_before -= 1
            
            if good_before < 0:
                i += 1
                continue
            
            pos_before = xyz[good_before]
            
            # Look for a return jump
            for j in range(i + 1, len(jump_indices)):
                end_jump = jump_indices[j]
                
                # Don't look too far
                if end_jump - start_jump > 100:
                    break
                
                # Find the first good point after this jump
                good_after = end_jump + 1
                while good_after < n and (velocity_outliers[good_after] or not np.all(np.isfinite(xyz[good_after]))):
                    good_after += 1
                
                if good_after >= n:
                    continue
                
                pos_after = xyz[good_after]
                
                # Check if we returned to roughly the same position
                dist_before_after = np.linalg.norm(pos_after - pos_before)
                
                # The "middle" of the bracketed region
                mid_idx = (start_jump + end_jump) // 2
                if np.all(np.isfinite(xyz[mid_idx])):
                    dist_to_middle = np.linalg.norm(xyz[mid_idx] - pos_before)
                    
                    # If middle is much further from start than end is, it's a garbage island
                    if dist_to_middle > dist_before_after * 2 and dist_to_middle > 5:
                        # Mark ENTIRE region as bad
                        for k in range(start_jump, end_jump + 1):
                            velocity_outliers[k] = True
                        
                        # Skip to after this region
                        i = j
                        break
            
            i += 1
    
    # === STEP 3: Position-based outlier detection (per axis) ===
    position_outliers = np.zeros(n, dtype=bool)
    
    # Only use non-velocity-outlier points to compute statistics
    clean_mask = ~velocity_outliers & np.all(np.isfinite(xyz), axis=1)
    
    if clean_mask.sum() > 10:
        for dim in range(3):
            clean_values = xyz[clean_mask, dim]
            median_val = np.median(clean_values)
            std_val = np.std(clean_values)
            
            if std_val > 1e-6:
                for i in range(n):
                    if np.isfinite(xyz[i, dim]) and not velocity_outliers[i]:
                        if abs(xyz[i, dim] - median_val) > position_outlier_std * std_val:
                            position_outliers[i] = True
    
    # === STEP 4: Combine all outliers ===
    outlier_mask = velocity_outliers | position_outliers
    xyz_clean[outlier_mask] = np.nan
    
    # === STEP 5: Only interpolate SMALL gaps (natural tracking losses, not cleaned regions) ===
    # This is key: we don't want to interpolate across the garbage we just removed
    xyz_clean = interpolate_small_gaps(xyz_clean, max_gap=max_interp_gap)
    
    return xyz_clean, outlier_mask


def create_oscillation_summary_plot(
    df: pd.DataFrame, 
    output_dir: Path,
    summary_br_list: Optional[List[int]] = None
):
    """Plot oscillation metrics by condition."""
    
    if summary_br_list:
        df = df[df['br_idx'].isin(summary_br_list)].copy()
    
    success_df = df[df['status'] == 'success'].copy()
    if len(success_df) == 0:
        print("[warn] No successful reaches for oscillation plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    conditions = sorted(success_df['condition_type'].unique())
    
    # 1. Number of direction changes by condition
    ax = axes[0, 0]
    data_by_cond = [success_df[success_df['condition_type'] == c]['n_direction_changes'].dropna() 
                   for c in conditions]
    bp = ax.boxplot(data_by_cond, labels=conditions, patch_artist=True)
    colors = ['#808080' if 'control' in c.lower() else '#2E86AB' for c in conditions]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel('Condition')
    ax.set_ylabel('# Direction Changes')
    ax.set_title('Movement Reversals by Condition')
    ax.grid(True, alpha=0.3)
    
    # 2. Direction changes vs approach time
    ax = axes[0, 1]
    for cond in conditions:
        subset = success_df[success_df['condition_type'] == cond]
        color = '#808080' if 'control' in cond.lower() else '#2E86AB'
        ax.scatter(subset['approach_time_ms'], subset['n_direction_changes'], 
                   alpha=0.5, label=cond, s=30, c=color)
    ax.set_xlabel('Approach Time (ms)')
    ax.set_ylabel('# Direction Changes')
    ax.set_title('Reversals vs Reach Duration')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 3. Mean oscillation amplitude by condition
    ax = axes[1, 0]
    data_by_cond = [success_df[success_df['condition_type'] == c]['mean_oscillation_amplitude'].dropna() 
                   for c in conditions]
    # Filter out empty arrays
    valid_data = [(d, c) for d, c in zip(data_by_cond, conditions) if len(d) > 0]
    if valid_data:
        data_only = [d for d, c in valid_data]
        labels_only = [c for d, c in valid_data]
        bp = ax.boxplot(data_only, labels=labels_only, patch_artist=True)
        colors = ['#808080' if 'control' in c.lower() else '#2E86AB' for c in labels_only]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Mean Oscillation Amplitude')
    ax.set_title('Oscillation Amplitude by Condition')
    ax.grid(True, alpha=0.3)
    
    # 4. Direction changes by BR index
    ax = axes[1, 1]
    osc_by_br = success_df.groupby('br_idx').agg({
        'n_direction_changes': ['mean', 'std', 'count']
    })
    osc_by_br.columns = ['mean', 'std', 'count']
    osc_by_br = osc_by_br.sort_index()
    
    x = range(len(osc_by_br))
    bars = ax.bar(x, osc_by_br['mean'].values, yerr=osc_by_br['std'].values,
                  capsize=3, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f"BR {idx}\n(n={int(osc_by_br.loc[idx, 'count'])})" 
                       for idx in osc_by_br.index], fontsize=8)
    ax.set_xlabel('BR Index')
    ax.set_ylabel('Mean # Direction Changes')
    ax.set_title('Oscillations by Recording')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if summary_br_list:
        br_str = '_BR_' + '_'.join(str(b) for b in sorted(summary_br_list))
        save_path = output_dir / f"oscillation_metrics_summary{br_str}.png"
    else:
        save_path = output_dir / "oscillation_metrics_summary.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[save] Oscillation summary: {save_path}")

def compute_direction_changes_v2(
    xyz: np.ndarray,
    rel_time_ms: np.ndarray,
    start_time_ms: float = 0.0,
    end_time_ms: Optional[float] = None,
    min_reversal_magnitude: float = 0.3,  # Minimum position change to count
    min_reversal_duration_ms: float = 20.0,  # Minimum time between reversals
    smoothing_window: int = 3,  # Light smoothing
) -> dict:
    """
    Detect direction changes by finding local maxima/minima (peaks/valleys) 
    in the primary movement axis position signal.
    
    A direction change is a peak or valley where:
    - The position change from previous extremum exceeds min_reversal_magnitude
    - Sufficient time has passed since last reversal
    
    Args:
        xyz: (N, 3) position array
        rel_time_ms: (N,) time array relative to event
        start_time_ms: Start of analysis window
        end_time_ms: End of analysis window (None = use all data)
        min_reversal_magnitude: Minimum position change to count as reversal
        min_reversal_duration_ms: Minimum time between consecutive reversals
        smoothing_window: Light smoothing window (frames) - set to 1 for no smoothing
        
    Returns:
        dict with direction change metrics
    """
    from scipy.signal import find_peaks
    
    n = len(xyz)
    
    if end_time_ms is None:
        end_time_ms = rel_time_ms.max()
    
    time_mask = (rel_time_ms >= start_time_ms) & (rel_time_ms <= end_time_ms)
    
    if time_mask.sum() < 5:
        return {
            'n_direction_changes': 0,
            'direction_change_times_ms': [],
            'direction_change_positions': [],
            'direction_change_types': [],  # 'peak' or 'valley'
            'oscillation_amplitudes': [],
            'mean_oscillation_amplitude': 0,
            'total_oscillation_distance': 0,
            'primary_axis': None,
            'primary_axis_index': None,
        }
    
    # Find primary movement axis
    axis_ranges = []
    for dim in range(3):
        dim_data = xyz[time_mask, dim]
        valid = np.isfinite(dim_data)
        if valid.sum() > 2:
            axis_ranges.append(np.ptp(dim_data[valid]))
        else:
            axis_ranges.append(0)
    
    primary_axis_idx = np.argmax(axis_ranges)
    axis_names = ['x', 'y', 'z']
    primary_axis_name = axis_names[primary_axis_idx]
    primary_range = axis_ranges[primary_axis_idx]
    
    # Extract primary axis position within time window
    window_indices = np.where(time_mask)[0]
    pos_raw = xyz[window_indices, primary_axis_idx].copy()
    time_window = rel_time_ms[window_indices]
    
    # Light smoothing (optional)
    if smoothing_window > 1 and len(pos_raw) >= smoothing_window:
        # Handle NaNs
        valid_pos = np.isfinite(pos_raw)
        if valid_pos.sum() > smoothing_window:
            pos_smooth = pos_raw.copy()
            pos_filled = np.interp(
                np.arange(len(pos_raw)),
                np.where(valid_pos)[0],
                pos_raw[valid_pos]
            )
            pos_smooth = uniform_filter1d(pos_filled, size=smoothing_window, mode='nearest')
            pos_smooth[~valid_pos] = np.nan
        else:
            pos_smooth = pos_raw
    else:
        pos_smooth = pos_raw
    
    # Find peaks (local maxima) and valleys (local minima)
    valid_mask = np.isfinite(pos_smooth)
    if valid_mask.sum() < 5:
        return {
            'n_direction_changes': 0,
            'direction_change_times_ms': [],
            'direction_change_positions': [],
            'direction_change_types': [],
            'oscillation_amplitudes': [],
            'mean_oscillation_amplitude': 0,
            'total_oscillation_distance': 0,
            'primary_axis': primary_axis_name,
            'primary_axis_index': primary_axis_idx,
        }
    
    # Calculate minimum distance between peaks in frames
    fps = 1000.0 / np.nanmedian(np.diff(time_window)) if len(time_window) > 1 else 100.0
    min_distance_frames = max(1, int(min_reversal_duration_ms * fps / 1000.0))
    
    # Prominence = minimum height difference to count as a peak
    # Scale by the axis range, but set a minimum
    prominence = max(min_reversal_magnitude, primary_range * 0.05)
    
    # Find peaks (local maxima)
    # Need to handle NaNs - interpolate for peak finding only
    pos_for_peaks = np.interp(
        np.arange(len(pos_smooth)),
        np.where(valid_mask)[0],
        pos_smooth[valid_mask]
    )
    
    peaks, peak_props = find_peaks(
        pos_for_peaks, 
        prominence=prominence,
        distance=min_distance_frames
    )
    
    # Find valleys (local minima) by inverting
    valleys, valley_props = find_peaks(
        -pos_for_peaks,
        prominence=prominence, 
        distance=min_distance_frames
    )
    
    # Combine and sort by time
    extrema = []
    for idx in peaks:
        if valid_mask[idx]:  # Only include if original data was valid
            extrema.append({
                'frame_idx': idx,
                'time_ms': time_window[idx],
                'position': pos_smooth[idx],
                'type': 'peak',
                'global_idx': window_indices[idx],
            })
    
    for idx in valleys:
        if valid_mask[idx]:
            extrema.append({
                'frame_idx': idx,
                'time_ms': time_window[idx],
                'position': pos_smooth[idx],
                'type': 'valley',
                'global_idx': window_indices[idx],
            })
    
    # Sort by time
    extrema = sorted(extrema, key=lambda x: x['time_ms'])
    
    # Filter: alternating peaks and valleys, with minimum amplitude
    filtered_extrema = []
    oscillation_amplitudes = []
    
    for ext in extrema:
        if not filtered_extrema:
            filtered_extrema.append(ext)
            continue
        
        last = filtered_extrema[-1]
        
        # Must alternate between peak and valley
        if ext['type'] == last['type']:
            # Same type - keep the more extreme one
            if ext['type'] == 'peak' and ext['position'] > last['position']:
                filtered_extrema[-1] = ext
            elif ext['type'] == 'valley' and ext['position'] < last['position']:
                filtered_extrema[-1] = ext
            continue
        
        # Calculate amplitude of this reversal
        amplitude = abs(ext['position'] - last['position'])
        
        if amplitude >= min_reversal_magnitude:
            filtered_extrema.append(ext)
            oscillation_amplitudes.append(amplitude)
    
    # Direction changes = number of extrema minus 1 (first one isn't a "change")
    n_direction_changes = max(0, len(filtered_extrema) - 1)
    
    # Total oscillation distance
    total_oscillation_distance = sum(oscillation_amplitudes)
    
    return {
        'n_direction_changes': n_direction_changes,
        'direction_change_times_ms': [e['time_ms'] for e in filtered_extrema],
        'direction_change_positions': [e['position'] for e in filtered_extrema],
        'direction_change_types': [e['type'] for e in filtered_extrema],
        'direction_change_global_indices': [e['global_idx'] for e in filtered_extrema],
        'oscillation_amplitudes': oscillation_amplitudes,
        'mean_oscillation_amplitude': np.mean(oscillation_amplitudes) if oscillation_amplitudes else 0,
        'total_oscillation_distance': total_oscillation_distance,
        'primary_axis': primary_axis_name,
        'primary_axis_index': primary_axis_idx,
        'primary_axis_range': primary_range,
        'extrema_details': filtered_extrema,
    }


def find_return_to_start(
    xyz: np.ndarray,
    rel_time_ms: np.ndarray,
    start_time_ms: float = 0.0,
    return_threshold_pct: float = 0.15,
    min_search_time_ms: float = 200.0,
) -> Tuple[int, float, str]:
    """
    Find when the hand returns close to its starting position.
    
    This extends the analysis window beyond just the endpoint to capture
    the full reach-and-return movement.
    
    Args:
        xyz: (N, 3) position array
        rel_time_ms: (N,) time array
        start_time_ms: Time to use as "start" reference
        return_threshold_pct: How close to start counts as "returned" (% of max excursion)
        min_search_time_ms: Don't look for return before this time
        
    Returns:
        return_idx: Index of return point
        return_time_ms: Time of return
        status: Description of result
    """
    n = len(xyz)
    
    # Get start position
    start_mask = (rel_time_ms >= start_time_ms - 30) & (rel_time_ms <= start_time_ms + 30)
    start_mask &= np.all(np.isfinite(xyz), axis=1)
    
    if not start_mask.any():
        return -1, np.nan, 'no_valid_start'
    
    start_pos = np.median(xyz[start_mask], axis=0)
    
    # Compute distance from start at each time
    dist_from_start = np.linalg.norm(xyz - start_pos, axis=1)
    
    # Find max excursion
    valid_mask = np.isfinite(dist_from_start) & (rel_time_ms >= min_search_time_ms)
    
    if not valid_mask.any():
        return -1, np.nan, 'no_valid_data_after_min_time'
    
    max_excursion = np.nanmax(dist_from_start[valid_mask])
    max_excursion_idx = np.where(valid_mask)[0][np.nanargmax(dist_from_start[valid_mask])]
    
    # Define return threshold
    return_distance = max_excursion * return_threshold_pct
    
    # Search for return AFTER max excursion
    search_mask = (np.arange(n) > max_excursion_idx) & np.isfinite(dist_from_start)
    
    if not search_mask.any():
        return -1, np.nan, 'no_data_after_max_excursion'
    
    # Find first point that comes within return_distance of start
    search_indices = np.where(search_mask)[0]
    
    for idx in search_indices:
        if dist_from_start[idx] <= return_distance:
            return idx, rel_time_ms[idx], 'return_found'
    
    # Didn't fully return - return the closest point
    closest_idx = search_indices[np.argmin(dist_from_start[search_indices])]
    return closest_idx, rel_time_ms[closest_idx], 'partial_return'

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


def compute_velocity(xyz: np.ndarray, dt_sec: float, smooth: bool = True, 
                     smooth_window: int = 3) -> np.ndarray:
    """
    Compute instantaneous velocity magnitude from 3D positions.
    
    Args:
        xyz: (N, 3) position array
        dt_sec: Time step in seconds
        smooth: Whether to apply smoothing
        smooth_window: Smoothing window size (default: 3 for light smoothing)
        
    Returns:
        velocity: (N,) array of velocity magnitudes
    """
    n = xyz.shape[0]
    if n < 2:
        return np.zeros(n)
    
    # Central difference for interior, forward/backward for edges
    vel_vec = np.zeros_like(xyz)
    
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
    
    if smooth and smooth_window > 1:
        velocity = smooth_signal(velocity, window=smooth_window)
    
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


def check_reach_direction_v2(
    xyz: np.ndarray, 
    rel_time_ms: np.ndarray,
    check_window_ms: float = 200.0,
    approach_required: float = 0.1,
    target_pos: Optional[np.ndarray] = None,
) -> Tuple[bool, float, str, int]:
    """
    Check if the hand is moving TOWARD the target using the primary movement axis.
    
    Returns:
        is_approaching: True if moving toward target
        net_change: Net distance change (negative = approaching)
        reason: Description
        primary_axis: Which axis (0=x, 1=y, 2=z) had most movement
    """
    # Find primary movement axis (most change in first 200ms)
    mask_analysis = (rel_time_ms >= 0) & (rel_time_ms <= check_window_ms)
    mask_analysis &= np.all(np.isfinite(xyz), axis=1)
    
    if mask_analysis.sum() < 5:
        return True, 0.0, 'insufficient_data', 0
    
    # Calculate range for each axis
    axis_ranges = []
    for dim in range(3):
        dim_data = xyz[mask_analysis, dim]
        axis_ranges.append(np.ptp(dim_data))
    
    primary_axis = np.argmax(axis_ranges)
    axis_names = ['x', 'y', 'z']
    
    # Get start and end values on primary axis
    mask_start = (rel_time_ms >= -50) & (rel_time_ms <= 30)
    mask_start &= np.isfinite(xyz[:, primary_axis])
    
    mask_end = (rel_time_ms >= check_window_ms - 30) & (rel_time_ms <= check_window_ms + 30)
    mask_end &= np.isfinite(xyz[:, primary_axis])
    
    if not mask_start.any() or not mask_end.any():
        return True, 0.0, 'insufficient_data', primary_axis
    
    pos_at_start = np.median(xyz[mask_start, primary_axis])
    pos_at_end = np.median(xyz[mask_end, primary_axis])
    
    # Determine expected direction based on target
    if target_pos is not None and np.isfinite(target_pos[primary_axis]):
        target_on_axis = target_pos[primary_axis]
        
        # Expected direction: toward target
        expected_direction = np.sign(target_on_axis - pos_at_start)
        actual_direction = np.sign(pos_at_end - pos_at_start)
        
        movement_magnitude = abs(pos_at_end - pos_at_start)
        movement_threshold = axis_ranges[primary_axis] * approach_required
        
        if movement_magnitude < movement_threshold:
            return True, 0.0, f'minimal_movement_{axis_names[primary_axis]}', primary_axis
        
        if expected_direction == actual_direction:
            return True, -(pos_at_end - pos_at_start) * expected_direction, \
                   f'approaching_on_{axis_names[primary_axis]}', primary_axis
        else:
            return False, (pos_at_end - pos_at_start) * expected_direction, \
                   f'moving_away_on_{axis_names[primary_axis]}', primary_axis
    
    # No target - just report the movement
    net_change = pos_at_end - pos_at_start
    return True, net_change, f'movement_on_{axis_names[primary_axis]}', primary_axis


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
    dlc_2d_root: Path = None,
    condition_name: str = None,
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
    
    # Calculate raw data quality (before any cleaning)
    pct_finger_valid_raw = np.mean(np.all(np.isfinite(finger_xyz_raw), axis=1)) * 100
    pct_target_valid_raw = np.mean(np.all(np.isfinite(target_xyz_raw), axis=1)) * 100
    
    # === Apply likelihood-based masking ===
    n_finger_masked = 0
    n_finger_cam0_low = 0
    n_finger_cam1_low = 0
    n_target_masked = 0
    
    if dlc_2d_root is not None and condition_name is not None:
        # Get likelihood mask for finger
        finger_valid_mask, n_finger_cam0_low, n_finger_cam1_low = get_likelihood_mask(
            dlc_2d_root, condition_name, finger_keypoint, 
            frame_indices, threshold=LIKELIHOOD_THRESHOLD
        )
        finger_xyz_masked = apply_likelihood_mask(finger_xyz_raw, finger_valid_mask)
        n_finger_masked = (~finger_valid_mask).sum()
        
        # Get likelihood mask for target
        target_valid_mask, _, _ = get_likelihood_mask(
            dlc_2d_root, condition_name, target_keypoint,
            frame_indices, threshold=LIKELIHOOD_THRESHOLD
        )
        target_xyz_masked = apply_likelihood_mask(target_xyz_raw, target_valid_mask)
        n_target_masked = (~target_valid_mask).sum()
        
        if debug:
            print(f"    [likelihood] Finger: {n_finger_masked}/{len(frame_indices)} masked "
                  f"(cam0_low={n_finger_cam0_low}, cam1_low={n_finger_cam1_low})")
            print(f"    [likelihood] Target: {n_target_masked}/{len(frame_indices)} masked")
    else:
        # No 2D likelihoods available - use raw data
        finger_xyz_masked = finger_xyz_raw.copy()
        target_xyz_masked = target_xyz_raw.copy()
        finger_valid_mask = np.all(np.isfinite(finger_xyz_raw), axis=1)
    
    finger_xyz_clean, outlier_mask = clean_finger_signal_v2(
        finger_xyz_masked, fps,
        rel_time_ms=rel_time_ms,
        max_interp_gap=MAX_INTERP_GAP_FRAMES,
        max_speed=100.0,
    )

    
    # Calculate post-cleaning data quality
    pct_finger_valid_clean = np.mean(np.all(np.isfinite(finger_xyz_clean), axis=1)) * 100
    
    # Get target position (use masked target data)
    target_pos = get_target_position_for_reach(target_xyz_masked, method='median')
    
    if not np.all(np.isfinite(target_pos)):
        return {
            'status': 'no_target',
            'error_msg': 'Target not visible in window',
            'drop_reason': 'target_not_visible',
            'endpoint_error': np.nan,
            'pct_finger_valid_raw': pct_finger_valid_raw,
            'pct_finger_valid_clean': pct_finger_valid_clean,
            'pct_target_valid_raw': pct_target_valid_raw,
            'n_finger_masked_by_likelihood': n_finger_masked,
            'n_target_masked_by_likelihood': n_target_masked,
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
            'n_finger_masked_by_likelihood': n_finger_masked,
            'n_target_masked_by_likelihood': n_target_masked,
            'target_x': target_pos[0],
            'target_y': target_pos[1],
            'target_z': target_pos[2],
        }
    
    # Compute distance to target
    distance = euclidean_distance(finger_xyz_clean, target_pos)
    
    # Check reach direction
    is_approaching, net_change, direction_reason, primary_axis = check_reach_direction_v2(
        finger_xyz_clean,
        rel_time_ms,
        check_window_ms=200.0,
        approach_required=0.1,
        target_pos=target_pos,
    )
    
    if not is_approaching:
        return {
            'status': 'moving_away',
            'error_msg': f'Hand moving away: {direction_reason}',
            'drop_reason': 'direction_away',
            'endpoint_error': np.nan,
            'net_distance_change': net_change,
            'primary_movement_axis': primary_axis,
            'pct_finger_valid_raw': pct_finger_valid_raw,
            'pct_finger_valid_clean': pct_finger_valid_clean,
            'pct_target_valid_raw': pct_target_valid_raw,
            'n_finger_masked_by_likelihood': n_finger_masked,
            'n_target_masked_by_likelihood': n_target_masked,
            'target_x': target_pos[0],
            'target_y': target_pos[1],
            'target_z': target_pos[2],
        }
    
    # Compute velocity
    dt_sec = 1.0 / fps
    velocity = compute_velocity(finger_xyz_clean, dt_sec, smooth=True, smooth_window=3)
    
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
            'n_finger_masked_by_likelihood': n_finger_masked,
            'n_target_masked_by_likelihood': n_target_masked,
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
            'n_finger_masked_by_likelihood': n_finger_masked,
            'n_target_masked_by_likelihood': n_target_masked,
            'target_x': target_pos[0],
            'target_y': target_pos[1],
            'target_z': target_pos[2],
        }
    
    # Compute oscillation metrics
    oscillation_end_ms = approach_time_ms + 50
    oscillation_metrics = compute_direction_changes_v2(
        finger_xyz_clean,
        rel_time_ms,
        start_time_ms=0.0,
        end_time_ms=oscillation_end_ms,
        min_reversal_magnitude=0.3,
        min_reversal_duration_ms=20.0,
        smoothing_window=3,
    )
    
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
        
        # Data quality - likelihood-based
        'pct_finger_valid_raw': pct_finger_valid_raw,
        'pct_finger_valid_clean': pct_finger_valid_clean,
        'pct_target_valid_raw': pct_target_valid_raw,
        'n_finger_masked_by_likelihood': n_finger_masked,
        'n_finger_cam0_low': n_finger_cam0_low,
        'n_finger_cam1_low': n_finger_cam1_low,
        'n_target_masked_by_likelihood': n_target_masked,
        
        # Direction detection
        'primary_movement_axis': oscillation_metrics['primary_axis'],
        'direction_check_axis': ['x', 'y', 'z'][primary_axis],
        'net_distance_change_200ms': net_change,
        
        # Oscillation metrics
        'n_direction_changes': oscillation_metrics['n_direction_changes'],
        'total_oscillation_distance': oscillation_metrics['total_oscillation_distance'],
        'mean_oscillation_amplitude': oscillation_metrics['mean_oscillation_amplitude'],
        'oscillation_analysis_end_ms': oscillation_end_ms,
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
    dlc_2d_root: Path,
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
            dlc_2d_root=dlc_2d_root,
            condition_name=condition_name,
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
    dlc_2d_root: Path = None,
):
    """Create diagnostic plot for a single reach with position-based direction change markers."""
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
    
    if dlc_2d_root is not None and condition_name is not None:
        # Apply likelihood mask to finger
        finger_valid_mask, _, _ = get_likelihood_mask(
            dlc_2d_root, condition_name, finger_keypoint,
            frame_indices, threshold=LIKELIHOOD_THRESHOLD
        )
        finger_xyz_masked = apply_likelihood_mask(finger_xyz_raw, finger_valid_mask)
        
        # Apply likelihood mask to target
        target_valid_mask, _, _ = get_likelihood_mask(
            dlc_2d_root, condition_name, target_keypoint,
            frame_indices, threshold=LIKELIHOOD_THRESHOLD
        )
        target_xyz_masked = apply_likelihood_mask(target_xyz_raw, target_valid_mask)
    else:
        finger_xyz_masked = finger_xyz_raw.copy()
        target_xyz_masked = target_xyz_raw.copy()

    finger_xyz_clean, outlier_mask = clean_finger_signal_v2(
        finger_xyz_masked, fps, 
        rel_time_ms=rel_time_ms,
        max_interp_gap=MAX_INTERP_GAP_FRAMES,
        max_speed=100.0,
    )
    
    
    target_pos = get_target_position_for_reach(target_xyz_masked, method='median')

    
    if not np.all(np.isfinite(target_pos)):
        target_pos = np.array([np.nan, np.nan, np.nan])
    
    distance = euclidean_distance(finger_xyz_clean, target_pos)
    
    is_approaching, net_change, direction_reason, primary_axis_check = check_reach_direction_v2(
        finger_xyz_clean, rel_time_ms, 
        check_window_ms=200.0, 
        approach_required=0.1,
        target_pos=target_pos,
    )
    
    dt_sec = 1.0 / fps
    velocity = compute_velocity(finger_xyz_clean, dt_sec, smooth=True, smooth_window=3)
    
    approach_idx, min_dist, method = find_first_closest_approach(
        distance,
        rel_time_ms=rel_time_ms,
        min_approach_time_ms=MIN_APPROACH_TIME_MS,
        first_approach_threshold=0.3,
    )
    
    # Compute direction changes using position-based method
    oscillation_end_ms = rel_time_ms[approach_idx] + 50 if approach_idx >= 0 else 600.0
    oscillation_metrics = compute_direction_changes_v2(
        finger_xyz_clean,
        rel_time_ms,
        start_time_ms=0.0,
        end_time_ms=oscillation_end_ms,
        min_reversal_magnitude=0.3,
        min_reversal_duration_ms=20.0,
        smoothing_window=3,
    )
    
    status = reach_result.get('status', 'unknown')
    drop_reason = reach_result.get('drop_reason', '')
    
    fig = plt.figure(figsize=(18, 16))
    
    # 1. Distance over time
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.axvspan(rel_time_ms.min(), MIN_APPROACH_TIME_MS, alpha=0.1, color='gray')
    ax1.plot(rel_time_ms, distance, 'b-', lw=1.5, label='Distance to target')
    ax1.axvline(0, color='r', ls='--', lw=2, label='Event')
    ax1.axvline(MIN_APPROACH_TIME_MS, color='orange', ls=':', lw=2, label='Min approach')
    
    if approach_idx >= 0 and status == 'success':
        approach_time = rel_time_ms[approach_idx]
        ax1.axvline(approach_time, color='g', ls='-', lw=2, label='Closest approach')
        ax1.scatter([approach_time], [min_dist], c='g', s=100, zorder=5)
    
    ax1.set_xlabel('Time relative to event (ms)')
    ax1.set_ylabel('Distance to target')
    ax1.set_title(f'Distance to Target | Method: {method}')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3)
    
    # 2. PRIMARY AXIS POSITION with peaks/valleys marked
    ax2 = fig.add_subplot(3, 2, 2)
    
    primary_axis_idx = oscillation_metrics.get('primary_axis_index', 0)
    primary_axis_name = oscillation_metrics.get('primary_axis', 'x')
    
    if primary_axis_idx is not None:
        primary_pos = finger_xyz_clean[:, primary_axis_idx]
        ax2.plot(rel_time_ms, primary_pos, 'purple', lw=1.5, 
                label=f'{primary_axis_name.upper()} position (primary axis)')
        
        # Mark peaks and valleys
        dc_times = oscillation_metrics.get('direction_change_times_ms', [])
        dc_positions = oscillation_metrics.get('direction_change_positions', [])
        dc_types = oscillation_metrics.get('direction_change_types', [])
        
        for t, p, typ in zip(dc_times, dc_positions, dc_types):
            if typ == 'peak':
                ax2.scatter([t], [p], c='red', s=100, marker='^', zorder=5, 
                           edgecolors='black', linewidths=1)
            else:  # valley
                ax2.scatter([t], [p], c='blue', s=100, marker='v', zorder=5,
                           edgecolors='black', linewidths=1)
        
        # Add legend entries for markers
        ax2.scatter([], [], c='red', s=100, marker='^', label='Peak (max)', edgecolors='black')
        ax2.scatter([], [], c='blue', s=100, marker='v', label='Valley (min)', edgecolors='black')
    
    ax2.axvline(0, color='r', ls='--', lw=2, alpha=0.5)
    ax2.axvline(MIN_APPROACH_TIME_MS, color='orange', ls=':', lw=2, alpha=0.5)
    if approach_idx >= 0 and status == 'success':
        ax2.axvline(rel_time_ms[approach_idx], color='g', ls='-', lw=2, alpha=0.5)
    
    n_changes = oscillation_metrics.get('n_direction_changes', 0)
    total_osc_dist = oscillation_metrics.get('total_oscillation_distance', 0)
    ax2.set_xlabel('Time relative to event (ms)')
    ax2.set_ylabel(f'{primary_axis_name.upper()} Position')
    ax2.set_title(f'Primary Axis Position | {n_changes} direction changes | Total oscillation: {total_osc_dist:.2f}')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. All position axes
    ax3 = fig.add_subplot(3, 2, 3)
    colors = ['red', 'green', 'blue']
    labels = ['X', 'Y', 'Z']
    
    for dim, (color, label) in enumerate(zip(colors, labels)):
        lw = 2.5 if dim == primary_axis_idx else 1.0
        alpha = 1.0 if dim == primary_axis_idx else 0.5
        ax3.plot(rel_time_ms, finger_xyz_raw[:, dim], color=color, lw=0.5, alpha=0.2)
        ax3.plot(rel_time_ms, finger_xyz_clean[:, dim], color=color, lw=lw, alpha=alpha,
                label=f'{label}' + (' (primary)' if dim == primary_axis_idx else ''))
    
    # Mark direction changes on primary axis
    if primary_axis_idx is not None:
        for t, p, typ in zip(dc_times, dc_positions, dc_types):
            marker = '^' if typ == 'peak' else 'v'
            color = 'red' if typ == 'peak' else 'blue'
            ax3.scatter([t], [p], c=color, s=60, marker=marker, zorder=5,
                       edgecolors='black', linewidths=0.5)
    
    if outlier_mask.any():
        ax3.axvline(np.nan, color='gray', ls=':', alpha=0.5, 
                   label=f'Outliers removed (n={outlier_mask.sum()})')
    
    ax3.axvline(0, color='r', ls='--', lw=2, alpha=0.5)
    if approach_idx >= 0 and status == 'success':
        ax3.axvline(rel_time_ms[approach_idx], color='g', ls='-', lw=2, alpha=0.5)
    ax3.set_xlabel('Time relative to event (ms)')
    ax3.set_ylabel('Position')
    ax3.set_title('All Position Axes (faint=raw, solid=cleaned)')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Velocity magnitude (still useful for speed info)
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(rel_time_ms, velocity, 'purple', lw=1.5, label='Speed (3D magnitude)')
    ax4.axvline(0, color='r', ls='--', lw=2, alpha=0.5)
    ax4.axvline(MIN_APPROACH_TIME_MS, color='orange', ls=':', lw=2, alpha=0.5)
    if approach_idx >= 0 and status == 'success':
        ax4.axvline(rel_time_ms[approach_idx], color='g', ls='-', lw=2, alpha=0.5)
    
    # Mark velocity dip if detected
    dip_detected = reach_result.get('dip_detected', False)
    if dip_detected:
        dip_time = reach_result.get('dip_time_ms', np.nan)
        if np.isfinite(dip_time):
            ax4.axvline(dip_time, color='cyan', ls='--', lw=2, label='Velocity dip')
    
    ax4.set_xlabel('Time relative to event (ms)')
    ax4.set_ylabel('Speed')
    ax4.set_title('Movement Speed (3D velocity magnitude)')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. 3D Trajectory with direction change markers
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
                   c='gold', s=300, marker='*', label='Target', zorder=10,
                   edgecolors='black', linewidths=1)
    
    # Mark direction changes in 3D
    dc_global_indices = oscillation_metrics.get('direction_change_global_indices', [])
    for i, global_idx in enumerate(dc_global_indices):
        if global_idx < len(finger_xyz_clean) and np.all(np.isfinite(finger_xyz_clean[global_idx])):
            typ = dc_types[i] if i < len(dc_types) else 'peak'
            marker = '^' if typ == 'peak' else 'v'
            color = 'red' if typ == 'peak' else 'blue'
            ax5.scatter([finger_xyz_clean[global_idx, 0]], 
                       [finger_xyz_clean[global_idx, 1]],
                       [finger_xyz_clean[global_idx, 2]], 
                       c=color, s=100, marker=marker, edgecolors='black', 
                       linewidths=1, zorder=8)
    
    if approach_idx >= 0 and status == 'success' and valid_clean[approach_idx]:
        ax5.scatter([finger_xyz_clean[approach_idx, 0]], 
                   [finger_xyz_clean[approach_idx, 1]],
                   [finger_xyz_clean[approach_idx, 2]], 
                   c='green', s=200, marker='o', edgecolors='black', 
                   linewidths=2, label='Closest approach', zorder=10)
    
    if valid_clean.any():
        first_valid = np.where(valid_clean)[0][0]
        ax5.scatter([finger_xyz_clean[first_valid, 0]], 
                   [finger_xyz_clean[first_valid, 1]],
                   [finger_xyz_clean[first_valid, 2]], 
                   c='cyan', s=150, marker='s', edgecolors='black',
                   linewidths=2, label='Start', zorder=10)
    
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.set_title('3D Trajectory (▲=peak, ▼=valley)')
    ax5.legend(loc='upper left', fontsize=8)
    
    # 6. Top-down XY view
    ax6 = fig.add_subplot(3, 2, 6)
    
    if valid_clean.any():
        valid_xyz = finger_xyz_clean[valid_clean]
        valid_times = rel_time_ms[valid_clean]
        
        # Draw trajectory
        for i in range(len(valid_xyz) - 1):
            t_norm = (valid_times[i] - valid_times.min()) / (valid_times.max() - valid_times.min() + 1e-6)
            color = plt.cm.coolwarm(t_norm)
            ax6.plot([valid_xyz[i, 0], valid_xyz[i+1, 0]], 
                    [valid_xyz[i, 1], valid_xyz[i+1, 1]], 
                    color=color, lw=2, alpha=0.7)
        
        sc = ax6.scatter(finger_xyz_clean[valid_clean, 0], finger_xyz_clean[valid_clean, 1], 
                        c=valid_times, cmap='coolwarm', s=20, alpha=0.8, zorder=5)
        plt.colorbar(sc, ax=ax6, label='Time (ms)')
    
    # Mark direction changes
    for i, global_idx in enumerate(dc_global_indices):
        if global_idx < len(finger_xyz_clean) and np.all(np.isfinite(finger_xyz_clean[global_idx, :2])):
            typ = dc_types[i] if i < len(dc_types) else 'peak'
            marker = '^' if typ == 'peak' else 'v'
            color = 'red' if typ == 'peak' else 'blue'
            ax6.scatter([finger_xyz_clean[global_idx, 0]], 
                       [finger_xyz_clean[global_idx, 1]], 
                       c=color, s=100, marker=marker, edgecolors='black', 
                       linewidths=1, zorder=8)
    
    if np.all(np.isfinite(target_pos)):
        ax6.scatter([target_pos[0]], [target_pos[1]], c='gold', s=300, marker='*', 
                   label='Target', zorder=10, edgecolors='black', linewidths=1)
    
    if approach_idx >= 0 and status == 'success' and valid_clean[approach_idx]:
        ax6.scatter([finger_xyz_clean[approach_idx, 0]], 
                   [finger_xyz_clean[approach_idx, 1]], 
                   c='green', s=200, marker='o', edgecolors='black', 
                   linewidths=2, label='Closest approach', zorder=10)
    
    if valid_clean.any():
        first_valid = np.where(valid_clean)[0][0]
        ax6.scatter([finger_xyz_clean[first_valid, 0]], 
                   [finger_xyz_clean[first_valid, 1]], 
                   c='cyan', s=150, marker='s', edgecolors='black',
                   linewidths=2, label='Start', zorder=10)
    
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_title('Top-Down View (XY)')
    ax6.legend(loc='best', fontsize=8)
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    
    # Main title
    if status == 'success':
        title_color = 'green'
        straightness = reach_result.get('straightness', np.nan)
        dip_str = " | DIP" if dip_detected else ""
        osc_str = f" | {n_changes} reversals" if n_changes > 0 else ""
        status_str = f'SUCCESS - Error: {min_dist:.2f} | Time: {rel_time_ms[approach_idx]:.0f}ms | Straightness: {straightness:.3f}{dip_str}{osc_str}'
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
    
def create_approach_and_error_summary(
    df: pd.DataFrame, 
    output_dir: Path, 
    summary_br_list: Optional[List[int]] = None
):
    """Create a 2-panel figure: approach time histogram and endpoint error distribution."""
    
    if summary_br_list:
        df = df[df['br_idx'].isin(summary_br_list)].copy()
    
    success_df = df[df['status'] == 'success'].copy()
    
    if len(success_df) == 0:
        print("[warn] No successful reaches for summary plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    conditions = sorted(success_df['condition_type'].unique())
    colors_map = {'control': '#808080', 'stim': '#2E86AB', 'continuous_stim': '#E94F37'}
    
    # 1. Approach time histogram
    ax = axes[0]
    for cond in conditions:
        subset = success_df[success_df['condition_type'] == cond]
        color = colors_map.get(cond, '#2E86AB')
        ax.hist(subset['approach_time_ms'], bins=30, alpha=0.5, 
                label=f'{cond} (n={len(subset)})', color=color, edgecolor='black', linewidth=0.5)
    
    ax.axvline(0, color='r', ls='--', lw=2, label='Event time')
    ax.axvline(MIN_APPROACH_TIME_MS, color='orange', ls=':', lw=2, 
               label=f'Min approach ({MIN_APPROACH_TIME_MS:.0f}ms)')
    ax.set_xlabel('Approach Time (ms)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Time to Closest Approach', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Endpoint error distribution
    ax = axes[1]
    for cond in conditions:
        subset = success_df[success_df['condition_type'] == cond]['endpoint_error'].dropna()
        if len(subset) > 0:
            color = colors_map.get(cond, '#2E86AB')
            ax.hist(subset, bins=20, alpha=0.5, 
                    label=f'{cond} (n={len(subset)})', color=color, 
                    edgecolor='black', linewidth=0.5, density=True)
    
    ax.set_xlabel('Endpoint Error (distance)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Endpoint Error Distribution', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    if summary_br_list:
        br_str = '_BR_' + '_'.join(str(b) for b in sorted(summary_br_list))
        save_path = output_dir / f"approach_and_error_summary{br_str}.png"
    else:
        save_path = output_dir / "approach_and_error_summary.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[save] Approach & error summary: {save_path}")


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

    # Set up 2D DLC root for likelihood masking
    dlc_2d_root = Path(VIDEO_ROOT) / "DLC"
    if not dlc_2d_root.exists():
        print(f"[warn] 2D DLC root not found: {dlc_2d_root}")
        print("[warn] Likelihood-based masking will be disabled")
        dlc_2d_root = None
    else:
        print(f"[info] Using DLC_2D root for likelihood masking: {dlc_2d_root}")
    
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
                dlc_2d_root=dlc_2d_root,
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
                dlc_2d_root=dlc_2d_root,
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


    # Oscillation summary plot
    success_df = combined_df[combined_df['status'] == 'success'].copy()
    if len(success_df) > 0 and 'n_direction_changes' in success_df.columns:
        create_approach_and_error_summary(combined_df, RESULTS_ROOT, summary_br_list=args.summary_br)
        create_oscillation_summary_plot(combined_df, RESULTS_ROOT, summary_br_list=args.summary_br)
    
    # === Create publication-quality SVG figure ===
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