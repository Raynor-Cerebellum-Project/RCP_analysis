import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from scipy import stats
from typing import Optional, Tuple
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *
import concurrent.futures
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.signal import find_peaks

"""
Purpose:
  Quantifies reach kinematics (duration and peak speed) for stimulation experiments.
  Compares stimulation conditions against a baseline control using Welch's t-test.
  Generates visualization plots (Violin/Box plots with swarm overlays) and statistical summaries.

Key Features:
  - Speed-based reach endpoint detection (Peak Speed -> Drop -> Stability).
  - Optional outlier detection and removal (IQR method).
  - Combined analysis of conditions grouping by name pattern.
  - Detailed statistical annotations (significance stars).
  - Backwards compatible with old and new data formats.

Outputs:
  - Figures: Saved to `results/figures/quantify_kinematics/`
    - Summary plots (Violin/Box + Stripplot) for Duration and Peak Speed.
    - Debug traces (if enabled) showing speed profiles and detection logic.
  - Data:
    - Stats Summary CSV: `*_stats_summary.csv` (Mean, Std, Count).
    - Outliers CSV: `*_outliers.csv` (if detection enabled).

Configuration Toggles (Global Flags):
  - REMOVE_OUTLIERS (bool): Enable/Disable outlier detection.
  - ANALYZE_COMBINED (bool): Enable/Disable aggregation of similar conditions.
  - PLOT_FULL_CONDITIONS (bool): Enable/Disable plotting of individual conditions.
  - SAVE_STATS_CSV (bool): Enable/Disable saving of statistical summary CSV.
  - PLOT_DEBUG_TRACES (bool): Enable/Disable generation of debug trace plots for every trial.

Adjustable Parameters:
  - CAMERA_TO_USE: Select 'cam1' or 'cam0'.
  - MIN_REACH_DURATION_MS: Minimum duration threshold for valid reaches.
  - SELECTED_KEYPOINTS: Joint selection (e.g., 'middle_x').
"""


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
CAMERA_TO_USE = "cam1"  # 'cam1' or 'cam0'
START_OFFSET_MS = 0  # Start counting reach from this time (relative to alignment) -> base off of triggering IR sensor
# Duration = Time(Max Pos) - START_OFFSET_MS
MIN_REACH_DURATION_MS = 100.0 # Exclude <110ms duration
MAX_REACH_DURATION_MS = 800.0

# Position Constraints for detection
START_POS_MAX_THRESH = 0.2  # Start must be within +/- 0.2 of 0 (Tightened)
END_POS_MIN_THRESH = 1.5    # End must be at least +/- 1.5 away from 0

# --- Speed Exclusion Thresholds ---
# Exclude trials if Peak Speed is 0 or > 0.25 (User request)
SPEED_EXCL_MIN = 0.005  
SPEED_EXCL_MAX = 0.02

# Track only middle_x as requested - with backwards-compatible variants
# The matching is case-insensitive, so this covers: middle_x, Middle_x, middle_X, etc.
SELECTED_KEYPOINTS = ["middle_x", "middle"] 

# Plotting Configuration
FIG_OUT_DIR = OUT_BASE / "figures" / "quantify_kinematics"
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output format: 'png' or 'svg'
OUTPUT_FORMAT = 'png'  # Change to 'svg' for vector graphics

# Set to True to generate 5x5 grid of traces for every trial (for debugging alignment)
PLOT_DEBUG_TRACES = True 

# Set to True to include Peak Speed plots in quantification figures
PLOT_PEAK_SPEED = False

# Analysis Flags
REMOVE_OUTLIERS = False      # Set to False to skip outlier detection and removal
ANALYZE_COMBINED = False      # Set to False to skip combined condition analysis
PLOT_FULL_CONDITIONS = True # Set to True to plot individual conditions
SAVE_STATS_CSV = False        # Set to False to skip saving stats CSV 

plt.rcParams.update({'font.size': 14}) # Larger font for plots

# Module-level flag to only print keypoint diagnostics once per session
_KEYPOINT_DIAG_PRINTED = set()

# ---------------------------------------------------------------------
# SESSION-SPECIFIC BR FILTERS
# ---------------------------------------------------------------------
# Which BR indices to include for each session
SESSION_BR_FILTERS = {
    "NRR_RW011": [4, 5, 6, 10, 11],
    "NRR_RW012": [3, 7, 8, 9, 10, 11, 12, 18, 31],
    "NRR_RW022": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
}

# Get BR filter for current session
def get_br_filter_for_session():
    """Return list of BR indices to include for current session, or None for all."""
    for session_key, br_list in SESSION_BR_FILTERS.items():
        if session_key in PARAMS.session:
            return br_list
    return None  # No filter = include all

BR_FILTER = get_br_filter_for_session()

# ---------------------------------------------------------------------
# METADATA-BASED CONDITION LABELING
# ---------------------------------------------------------------------
# Path to metadata CSV - derive from available paths
# PERI_ROOT is typically: <session_root>/results/checkpoints/PeriStim
# We need to go up to session root

# Go up from PeriStim -> checkpoints -> results -> session_root
_session_root = PERI_ROOT.parent.parent.parent

# Extract session name for the filename (e.g., "NRR_RW011" from full session string)
_session_name = PARAMS.session  # Use full session name like "NRR_RW011"

# Build metadata path - at session root level, NOT in results
METADATA_CSV = _session_root / "Metadata" / f"{_session_name}_metadata.csv"

# Debug print to verify path
print(f"[info] Session root: {_session_root}")
print(f"[info] Looking for metadata at: {METADATA_CSV}")

if not METADATA_CSV.exists():
    # Try alternative naming patterns
    _alt_paths = [
        _session_root / "Metadata" / f"{_session_name.split('_')[-1]}_metadata.csv",  # RW011_metadata.csv
        _session_root / "Metadata" / f"{_session_name.replace('NRR_', '')}_metadata.csv",  # RW011_metadata.csv
    ]
    for _alt in _alt_paths:
        if _alt.exists():
            METADATA_CSV = _alt
            print(f"[info] Found metadata at: {METADATA_CSV}")
            break
    else:
        print(f"[warn] Metadata file not found. Tried:")
        print(f"       - {_session_root / 'Metadata' / f'{_session_name}_metadata.csv'}")
        for _alt in _alt_paths:
            print(f"       - {_alt}")



def get_condition_label_from_metadata(br_idx: int, is_control: bool = False) -> str:
    """
    Get condition label from metadata CSV.
    
    Returns labels like:
    - "Control" for control/baseline conditions (grouped)
    - "40p @ 400Hz (1-32)" for stim conditions
    
    Parameters:
    -----------
    br_idx : int
        BR file index
    is_control : bool
        Whether this is a control/baseline condition
    """
    # Control conditions are always grouped
    if is_control:
        return "Control"
    
    try:
        if not METADATA_CSV.exists():
            print(f"[warn] Metadata file not found: {METADATA_CSV}")
            return f"BR {br_idx}"
        
        meta_df = pd.read_csv(METADATA_CSV)
        
        # Convert BR_File to numeric, coercing errors to NaN
        br_file_numeric = pd.to_numeric(meta_df['BR_File'], errors='coerce')
        
        # Find the row for this BR index
        mask = br_file_numeric == float(br_idx)
        row = meta_df[mask]
        
        if len(row) == 0:
            return f"BR {br_idx}"
        
        row = row.iloc[0]
        
        # Check if this is a baseline/control condition from Notes column
        if 'Notes' in row.index:
            notes = str(row['Notes']).lower() if pd.notna(row['Notes']) else ''
            if 'baseline' in notes or 'rest' in notes:
                return "Control"
        
        # Get stim parameters
        duration_ms = None
        frequency_hz = None
        channels = None
        
        if 'Stim_Duration_ms' in meta_df.columns:
            val = row['Stim_Duration_ms']
            if pd.notna(val):
                duration_ms = float(val)
        
        if 'Stim_Frequency_Hz' in meta_df.columns:
            val = row['Stim_Frequency_Hz']
            if pd.notna(val):
                frequency_hz = float(val)
        
        if 'Channels' in meta_df.columns:
            val = row['Channels']
            if pd.notna(val):
                channels = str(val).strip()
        
        # Build label
        label_parts = []
        
        # Pulses and frequency
        if duration_ms is not None and frequency_hz is not None:
            pulse_duration_ms = 2.5
            n_pulses = int(round(duration_ms / pulse_duration_ms))
            label_parts.append(f"{n_pulses}p @ {int(frequency_hz)}Hz")
        elif frequency_hz is not None:
            label_parts.append(f"@ {int(frequency_hz)}Hz")
        elif duration_ms is not None:
            n_pulses = int(round(duration_ms / 2.5))
            label_parts.append(f"{n_pulses}p")
        
        # Channels
        if channels:
            label_parts.append(f"({channels})")
        
        if label_parts:
            return " ".join(label_parts)
        
        return f"BR {br_idx}"
        
    except Exception as e:
        print(f"[warn] Error reading metadata for BR {br_idx}: {e}")
        return f"BR {br_idx}"


def get_br_from_condition_string(cond_str: str) -> Optional[int]:
    """Extract BR index from condition string like 'Cond_6' or 'BR_006'."""
    # Try "Cond_X" format
    match = re.search(r"Cond_(\d+)", cond_str)
    if match:
        return int(match.group(1))
    
    # Try "BR_XXX" format
    match = re.search(r"BR_(\d+)", cond_str)
    if match:
        return int(match.group(1))
    
    # Try any number
    match = re.search(r"(\d+)", cond_str)
    if match:
        return int(match.group(1))
    
    return None


def is_control_condition(cond_str: str, br_idx: Optional[int] = None) -> bool:
    """
    Determine if a condition is a control/baseline condition.
    
    Checks:
    1. Condition string contains 'baseline' or 'control'
    2. Metadata Notes column contains 'baseline' or 'rest'
    """
    cond_lower = cond_str.lower()
    if 'baseline' in cond_lower or 'control' in cond_lower:
        return True
    
    # Check metadata if we have a BR index
    if br_idx is not None:
        try:
            if METADATA_CSV.exists():
                meta_df = pd.read_csv(METADATA_CSV)
                br_file_numeric = pd.to_numeric(meta_df['BR_File'], errors='coerce')
                mask = br_file_numeric == float(br_idx)
                row = meta_df[mask]
                
                if len(row) > 0:
                    row = row.iloc[0]
                    if 'Notes' in row.index:
                        notes = str(row['Notes']).lower() if pd.notna(row['Notes']) else ''
                        if 'baseline' in notes or 'rest' in notes:
                            return True
        except Exception:
            pass
    
    return False




# ---------------------------------------------------------------------
# SESSION-SPECIFIC TRIAL EXCLUSIONS
# ---------------------------------------------------------------------
# These remain for manual trial exclusions within conditions

if "NRR_RW012" in PARAMS.session:
    EXCLUDE_CONDITIONS = []
    
    EXCLUDE_TRIALS_TARGET_A = {
        # Add specific trial exclusions if needed
    }
    
    EXCLUDE_TRIALS_TARGET_B = {
    }
    
    EXCLUDE_BASELINE_TRIALS = {
        "baseline_portA_targetA": [],
        "baseline_portB_targetA": [],
        "baseline_portA_targetB": [],
        "baseline_portB_targetB": [],
    }

elif "NRR_RW011" in PARAMS.session:
    EXCLUDE_CONDITIONS = []  # Now handled by BR_FILTER
    
    EXCLUDE_TRIALS_TARGET_A = {
        6: [0, 8],
        7: [2, 11],
        10: [5, 9],
        11: [3, 6],
        # ... keep your existing exclusions
    }
    
    EXCLUDE_TRIALS_TARGET_B = {
        6: [1, 6, 7, 8, 9, 10],
        7: [0, 6, 13],
        10: [12],
        11: [7],
        # ... keep your existing exclusions
    }
    
    EXCLUDE_BASELINE_TRIALS = {
        "baseline_portA_targetA": [1, 3, 4, 6, 13, 14, 21, 23, 24, 25, 31, 32, 38],
        "baseline_portB_targetA": [1, 2, 4, 6, 7, 9, 13, 16, 18],
        "baseline_portA_targetB": [2, 6, 10, 13, 15, 16, 17, 19, 20, 21, 22, 24, 25, 30, 33, 39, 40, 42, 44, 47, 56, 57],
        "baseline_portB_targetB": [2, 3, 4, 6, 8, 10, 12, 15, 17, 19, 20, 22, 23, 24, 26, 28, 30, 31, 33, 35],
    }

elif "NRR_RW022" in PARAMS.session:
    EXCLUDE_CONDITIONS = []
    
    EXCLUDE_TRIALS_TARGET_A = {
    }
    
    EXCLUDE_TRIALS_TARGET_B = {
    }
    
    EXCLUDE_BASELINE_TRIALS = {
        "baseline_portA_targetA": [],
        "baseline_portB_targetA": [],
        "baseline_portA_targetB": [],
        "baseline_portB_targetB": [],
    }

else:
    # Default fallback
    print(f"[warn] No specific config for {PARAMS.session}, using defaults")
    EXCLUDE_CONDITIONS = []
    EXCLUDE_TRIALS_TARGET_A = {}
    EXCLUDE_TRIALS_TARGET_B = {}
    EXCLUDE_BASELINE_TRIALS = {
        "baseline_portA_targetA": [],
        "baseline_portB_targetA": [],
        "baseline_portA_targetB": [],
        "baseline_portB_targetB": [],
    }

# Legacy label maps (kept for backwards compatibility, but metadata is now primary)
CONDITION_LABELS = {"baseline": "Control"}
COMBINED_CONDITION_LABELS = {"baseline": "Control"}




# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def _get_keypoint_indices(all_kp_names: list[str], selected_substrings: list[str], verbose: bool = False) -> list[int]:
    """
    Return indices of keypoints that match any substrings.
    Uses case-insensitive matching for better compatibility across data formats.
    Also tries matching without underscores for flexibility.
    """
    indices = []
    
    if verbose and all_kp_names:
        preview = all_kp_names[:10] if len(all_kp_names) > 10 else all_kp_names
        print(f"  Available keypoints: {preview}{'...' if len(all_kp_names) > 10 else ''}")
    
    for i, kp in enumerate(all_kp_names):
        kp_lower = str(kp).lower()
        kp_no_underscore = kp_lower.replace("_", "")
        
        for sub in selected_substrings:
            sub_lower = sub.lower()
            sub_no_underscore = sub_lower.replace("_", "")
            
            # Match if:
            # 1. Direct substring match (case-insensitive), OR
            # 2. Match without underscores (e.g., "middlex" matches "middle_x")
            if sub_lower in kp_lower or sub_no_underscore in kp_no_underscore:
                indices.append(i)
                break
    
    return indices


def _get_keypoint_names_from_data(data: dict, camera: str = CAMERA_TO_USE) -> list[str]:
    """
    Extract keypoint names from data file, trying multiple possible key formats.
    Returns list of keypoint names, or empty list if not found.
    """
    # Try different possible key names (new format first, then old formats)
    possible_keys = [
        f"beh_{camera}_names",      # New format: beh_cam1_names
        f"{camera}_names",          # Old format: cam1_names
        "beh_names",                # Generic
        "keypoint_names",           # Alternative
        "kp_names",                 # Short form
    ]
    
    for key in possible_keys:
        if key in data:
            names = data[key]
            if hasattr(names, 'tolist'):
                return names.tolist()
            return list(names)
    
    return []

def calculate_reach_metrics(
    pos_segs: np.ndarray,      # (n_trials, n_kps, T)
    vel_segs: np.ndarray,      # (n_trials, n_kps, T)
    ts_state_segs: np.ndarray, # (n_trials, T_ts)
    t_axis: np.ndarray,        # (T,)
    kp_names: list[str],
    target_code: int,          # 1 for A, 2 for B
    exclude_trials: list = None,  # List of trial indices to exclude
    fs: float = 100.0,         # approx sampling rate if needed, or derive from t_axis
    return_traces: bool = False,
    source_file: str = "") -> Tuple[pd.DataFrame, list]:
    """
    Calculate duration, peak speed, path length for each trial.
    Metric: Start at START_OFFSET_MS, End at Max Position Excursion.
    """
    global _KEYPOINT_DIAG_PRINTED
    
    n_trials, n_all_kps, T = pos_segs.shape
    
    # Default to empty list if no exclusions
    if exclude_trials is None:
        exclude_trials = []
    
    # 1. Filter keypoints with better fallback logic
    kp_idx = []
    if SELECTED_KEYPOINTS:
        kp_idx = _get_keypoint_indices(kp_names, SELECTED_KEYPOINTS)
        
        if not kp_idx:
            # Print diagnostic info once per session
            session_key = PARAMS.session
            if session_key not in _KEYPOINT_DIAG_PRINTED:
                print(f"  [Keypoint Diagnostic] Looking for: {SELECTED_KEYPOINTS}")
                print(f"  [Keypoint Diagnostic] Available in data: {kp_names[:8]}{'...' if len(kp_names) > 8 else ''}")
                _KEYPOINT_DIAG_PRINTED.add(session_key)
            
            # Try to find ANY x-coordinate keypoint as fallback
            x_keypoints = _get_keypoint_indices(kp_names, ["_x", "X", "x"])
            if x_keypoints:
                print(f"[warn] No keypoints matched {SELECTED_KEYPOINTS}. Using first X-coordinate keypoint (idx={x_keypoints[0]}).")
                kp_idx = [x_keypoints[0]]
            else:
                print(f"[warn] No keypoints matched {SELECTED_KEYPOINTS} and no X-coordinates found. Using all keypoints.")
                kp_idx = list(range(n_all_kps))
    else:
        kp_idx = list(range(n_all_kps))
        
    # Speed profile (average across selected KPs)
    trial_speeds_all_kps = vel_segs[:, kp_idx, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        trial_speed = np.nanmean(trial_speeds_all_kps, axis=1)  # (n_trials, T)
    
    # IMPORTANT: Use absolute value of speed for peak detection
    # Velocity can be negative depending on reach direction, but we want magnitude
    trial_speed = np.abs(trial_speed)

    # Position profile
    metrics_list = []
    collected_traces = []
    all_debug_traces = []

    for i in range(n_trials):
        # Determine if manually excluded
        manual_exclude = (i in exclude_trials)
        if manual_exclude:
             # We still process it for debug traces, but flag it.
             manual_excl_reason = "Manual Exclusion (List)"
        else:
             manual_excl_reason = ""
            
        speed = trial_speed[i]  # (T,) - Already absolute value
        t_trial = t_axis  # (T,)
        
        # 1. Identify Start Index (Time closest to START_OFFSET_MS)
        idx_start = (np.abs(t_trial - START_OFFSET_MS)).argmin()
        t_start = t_trial[idx_start]
        
        # --- Speed-Based Logic ---
        # Track middle_x position for plotting
        sel_pos = pos_segs[i, kp_idx, :]
        trace_for_stop = sel_pos[0, :]
        
        # Find Peak Speed in 0-MAX_REACH_DURATION_MS window
        dt = t_axis[1] - t_axis[0] if len(t_axis) > 1 else 10.0
        if dt < 0.9: dt *= 1000.0  # Convert to ms if needed
        
        max_samples = int(MAX_REACH_DURATION_MS / dt)
        idx_max_dur = min(len(t_trial)-1, idx_start + max_samples)
        
        search_speed = speed[idx_start : idx_max_dur]
        if search_speed.size == 0: 
            continue
            
        idx_peak_speed = idx_start + np.argmax(search_speed)
        peak_v = speed[idx_peak_speed]
        
        # --- Speed Exclusion Check ---
        # User requested: Do NOT exclude trials, just don't plot these points.
        speed_for_plot = speed.copy()
        mask_outliers = (np.abs(speed_for_plot) <= SPEED_EXCL_MIN) | (np.abs(speed_for_plot) > SPEED_EXCL_MAX)
        
        has_speed_outlier = False
        if np.any(mask_outliers):
             speed_for_plot[mask_outliers] = np.nan
             has_speed_outlier = True
        
        # Define search params for next step
        STABLE_WIN_MS = 20
        LOW_SPEED_THRESH_RATIO = 0.05  # Tightened to 5% to find "about 0" speed
        thresh_v = peak_v * LOW_SPEED_THRESH_RATIO
        win_samples = max(2, int(STABLE_WIN_MS / dt))

        # --- Refine Start (Backward Scan) ---
        # Scan backward from Peak Velocity for speed < thresh_v and stable
        # AND check if position is "next to 0" (within threshold)
        idx_start_refined = idx_start
        for k in range(idx_peak_speed, -1, -1):
            if speed[k] < thresh_v:
                # Check position constraint: abs(pos) <= START_POS_MAX_THRESH
                if np.abs(trace_for_stop[k]) <= START_POS_MAX_THRESH:
                    # Check stability of PREVIOUS window [k-win : k]
                    w_start = max(0, k - win_samples)
                    if k > w_start:  # Ensure window has size
                        if np.mean(speed[w_start:k]) < thresh_v:
                            idx_start_refined = k
                            break
                    else:
                        # At start of trial (index 0)
                        idx_start_refined = k
                        break
        
        # Update Start
        idx_start = idx_start_refined
        t_start = t_trial[idx_start]

        # --- Refine End: Find where velocity stops decreasing and starts increasing ---
        # Use smoothed derivative to avoid noise sensitivity
        
        # Default endpoint is max duration unless found
        idx_peak = idx_max_dur
        
        # Calculate dist_trace (abs displacement from start) for compatibility
        dist_trace = np.abs(trace_for_stop - trace_for_stop[idx_start])
        
        # Search AFTER peak speed for velocity turnaround
        search_start = idx_peak_speed + 1
        search_end = idx_max_dur
        
        if search_start < search_end - 1:
            search_speed = speed[search_start:search_end]
            
            if len(search_speed) > 5:
                # Smooth the speed trace first to reduce noise
                from scipy.ndimage import uniform_filter1d
                smooth_win = max(3, int(30 / dt))  # ~30ms smoothing window
                smoothed_speed = uniform_filter1d(search_speed, size=smooth_win, mode='nearest')
                
                # Calculate derivative of smoothed speed
                speed_deriv = np.diff(smoothed_speed)
                
                # Find first zero-crossing from negative to positive
                # (where deceleration ends and acceleration begins)
                for j in range(len(speed_deriv) - 1):
                    # Check for sign change: negative/zero -> positive
                    if speed_deriv[j] <= 0 and speed_deriv[j + 1] > 0:
                        candidate = search_start + j + 1
                        
                        # Verify position constraint
                        if np.abs(trace_for_stop[candidate]) >= END_POS_MIN_THRESH:
                            idx_peak = candidate
                            break
                
                # Fallback: if no zero-crossing found, find the minimum speed point
                if idx_peak == idx_max_dur:
                    idx_min_speed = search_start + np.argmin(smoothed_speed)
                    if np.abs(trace_for_stop[idx_min_speed]) >= END_POS_MIN_THRESH:
                        idx_peak = idx_min_speed
                    
        t_peak = t_trial[idx_peak]
        
        duration_ms = t_peak - t_start
        
        if manual_exclude:
             is_excluded = True
             excl_reason = manual_excl_reason
        elif duration_ms < MIN_REACH_DURATION_MS or duration_ms > MAX_REACH_DURATION_MS:
             is_excluded = True
             excl_reason = f"Duration {duration_ms:.1f} ms (Limits {MIN_REACH_DURATION_MS}-{MAX_REACH_DURATION_MS})"
        else:
            is_excluded = False
            excl_reason = ""
            
        # Peak Speed in this window
        window_speed = speed[idx_start : idx_peak+1]
        peak_speed = np.nanmax(window_speed) if window_speed.size > 0 else np.nan
        
        # Pack metrics
        m_dict = {
            "idx_trial": i,
            "duration_ms": duration_ms,
            "peak_speed": peak_speed,
            "t_start": t_start,
            "t_peak": t_peak,
        }
        
        # Store debug info for ALL trials
        is_max_duration = (idx_peak == idx_max_dur)
        
        all_debug_traces.append({
            "idx_trial": i,
            "t": t_trial,
            "pos": trace_for_stop,  # Position trace
            "y": trace_for_stop,    # Legacy key (for plotting)
            "speed": speed,         # Absolute speed (for plotting)
            "idx_start": idx_start,
            "idx_final": idx_peak, 
            "idx_orig": idx_peak_speed,  # Use peak speed as ref
            "is_max_dur": is_max_duration,
            "idx_peak_speed": idx_peak_speed,
            "thresh_v": thresh_v,
            "win_50ms": int(0.050/dt),
            "is_excluded": is_excluded,
            "exclusion_reason": excl_reason
        })
        
        # Only add to metrics if NOT excluded
        if not is_excluded:
             metrics_list.append(m_dict)
             if return_traces:
                 # Shift time to align 0 to Entry (idx_end)
                 collected_traces.append((t_trial, dist_trace))
        
    df = pd.DataFrame(metrics_list)
    
    # Collect Outliers (Exclusions + Statistical)
    outliers = []
    
    # 1. Add Exclusions from debug traces
    for tr in all_debug_traces:
        if tr.get('is_excluded'):
            outliers.append({
                'idx_trial': tr['idx_trial'],
                'reason': tr['exclusion_reason'],
                'value': 'Duration',
                'z_score': 'N/A'
            })

    # 2. Check for Statistical Outliers in Speed and Duration (Z > 3)
    if not df.empty and len(df) > 3:
        for col in ['duration_ms', 'peak_speed']:
             vals = df[col].values
             mean_v = np.mean(vals)
             std_v = np.std(vals)
             if std_v > 0:
                 z_scores = (vals - mean_v) / std_v
                 # Identify outliers
                 outlier_idxs = np.where(np.abs(z_scores) > 3.0)[0]
                 for idx in outlier_idxs:
                     trial_num = df.iloc[idx]['idx_trial']
                     val = vals[idx]
                     z = z_scores[idx]
                     # Add to list
                     outliers.append({
                         'idx_trial': trial_num,
                         'reason': f"Statistical Outlier ({col})",
                         'value': f"{val:.2f}",
                         'z_score': f"{z:.2f}"
                     })
                     
    if return_traces:
        return df, collected_traces, all_debug_traces, outliers
    return df, [], all_debug_traces, outliers


def process_single_file(p: Path, target: str, target_code: int) -> Optional[pd.DataFrame]:
    try:
        with np.load(p, allow_pickle=True) as data:
            # 1. Determine Condition from 'overall_title' or filename
            if p.name.startswith("baseline__"):
                cond = "Baseline"
                br_idx = None
            else:
                # Try getting title from file
                if "overall_title" in data:
                    raw_title = data["overall_title"]
                    # Handle 0-d array
                    if raw_title.shape == ():
                        cond = str(raw_title.item())
                    else:
                        cond = str(raw_title)
                else:
                    # Fallback to parsing filename
                    parts = p.name.split("_")
                    if len(parts) > 1 and parts[1].isdigit():
                        cond = f"Cond_{parts[1]}"
                    else:
                        cond = "Stim"
                
                # Extract BR index
                br_idx = get_br_from_condition_string(cond)
                if br_idx is None:
                    # Try from filename
                    br_match = re.search(r"BR_(\d+)", p.name)
                    if br_match:
                        br_idx = int(br_match.group(1))
            
            # --- BR FILTER CHECK ---
            # If we have a BR filter and this is a stim condition, check if BR is in filter
            if BR_FILTER is not None and br_idx is not None:
                if br_idx not in BR_FILTER:
                    return None  # Skip this BR
            
            # Check if this is a control condition
            is_control = is_control_condition(cond, br_idx)
            
            # For baseline files without BR index, check if any control BR is in filter
            if cond == "Baseline" and BR_FILTER is not None:
                # We keep baseline if any control BR is in filter
                # (baseline files aggregate multiple BRs, so we include them)
                pass
            
            # Check old-style exclusions (for backwards compatibility)
            is_excluded = False
            for ex in EXCLUDE_CONDITIONS:
                if isinstance(ex, int):
                    match = re.search(r"(\d+)", cond)
                    if match and int(match.group(1)) == ex:
                        is_excluded = True
                        break
                elif isinstance(ex, str):
                    if ex in cond:
                        is_excluded = True
                        break
            
            if is_excluded:
                return None

            # Key names - try multiple formats for backwards compatibility
            pos_key = None
            vel_key = None
            
            # New format: beh_cam1_segs
            if f"beh_{CAMERA_TO_USE}_segs" in data:
                pos_key = f"beh_{CAMERA_TO_USE}_segs"
                vel_key = f"beh_{CAMERA_TO_USE}_vel_segs"
            # Old format: cam1_segs
            elif f"{CAMERA_TO_USE}_segs" in data:
                pos_key = f"{CAMERA_TO_USE}_segs"
                vel_key = f"{CAMERA_TO_USE}_vel_segs"
            # Very old format: just segs
            elif "segs" in data:
                pos_key = "segs"
                vel_key = "vel_segs"
                
            if pos_key is None or pos_key not in data:
                print(f"Skipping {p.name}: missing position data. Available keys: {list(data.keys())[:10]}")
                return None
                
            if vel_key not in data:
                print(f"Skipping {p.name}: missing velocity data ({vel_key})")
                return None
                
            pos_segs = data[pos_key] # (n_trials, n_kps, T)
            vel_segs = data[vel_key]
            
            # Time axis - try multiple keys
            if "beh_rel_t" in data:
                t_axis = data["beh_rel_t"]
            elif "rel_t" in data:
                t_axis = data["rel_t"]
            elif "t" in data:
                t_axis = data["t"]
            else:
                print(f"Skipping {p.name}: missing time axis")
                return None
            
            # TS Check
            if "ts_state_segs" in data:
                ts_state = data["ts_state_segs"]
            else:
                ts_state = np.zeros((pos_segs.shape[0], pos_segs.shape[2])) 

            # Get keypoint names using the helper function
            kp_names = _get_keypoint_names_from_data(data, CAMERA_TO_USE)
            
            if not kp_names:
                # Generate placeholder names based on data shape
                n_kps = pos_segs.shape[1]
                kp_names = [f"kp_{i}" for i in range(n_kps)]
                print(f"[warn] No keypoint names found in {p.name}, using placeholders: {kp_names[:5]}...")
            
            # Determine trial exclusions
            trials_to_exclude = []
            
            # Default Port
            fname_lower = p.name.lower()
            port = "N/A"
            if "_portb_" in fname_lower or "portb" in fname_lower or "_port_b_" in fname_lower or "port_b" in fname_lower:
                port = "B"
            elif "_porta_" in fname_lower or "porta" in fname_lower or "_port_a_" in fname_lower or "port_a" in fname_lower:
                port = "A"

            # Check if this is a baseline file
            if cond == "Baseline":
                # Determine target from target_code (1=A, 2=B)
                target_letter = "A" if target_code == 1 else "B"
                
                # Build exclusion key
                baseline_key = f"baseline_port{port}_target{target_letter}"
                
                print(f"  [Processing] Baseline File: {p.name} -> Config: {baseline_key}")
                
                # Get exclusions for this baseline configuration
                if baseline_key in EXCLUDE_BASELINE_TRIALS:
                    trials_to_exclude = EXCLUDE_BASELINE_TRIALS[baseline_key]
            
            # If not baseline, check condition-based exclusions
            if cond != "Baseline" and br_idx is not None:
                # Select correct exclusion dict based on Target (1=A, 2=B)
                exclude_dict = EXCLUDE_TRIALS_TARGET_A if target_code == 1 else EXCLUDE_TRIALS_TARGET_B
                
                if br_idx in exclude_dict:
                    trials_to_exclude.extend(exclude_dict[br_idx])
            
            # Retrieve metrics AND debug traces
            df, _, all_debug_traces, outliers = calculate_reach_metrics(
                pos_segs, vel_segs, ts_state, t_axis, kp_names, target_code, 
                exclude_trials=trials_to_exclude, return_traces=False,
                source_file=p.name)
            
            # Plot Debug Traces for ALL trials (if enabled)
            if PLOT_DEBUG_TRACES and all_debug_traces:
                # Batch into pages of 25
                batch_size = 25
                n_total = len(all_debug_traces)
                n_pages = int(np.ceil(n_total / batch_size))
                
                for page in range(n_pages):
                    batch_traces = all_debug_traces[page*batch_size : (page+1)*batch_size]
                    n_batch = len(batch_traces)
                    
                    cols = 5
                    rows = 5
                    
                    # Thread-safe figure creation
                    fig_db = Figure(figsize=(20, 15))
                    FigureCanvasAgg(fig_db) # Attach canvas
                    axes_db = fig_db.subplots(rows, cols, squeeze=False)
                    axes_db = axes_db.flatten()
                    
                    for idx_db, tr_info in enumerate(batch_traces):
                        ax_db = axes_db[idx_db]
                        t = tr_info['t']
                        y = tr_info['y']
                        idx_start = tr_info['idx_start']
                        idx_final = tr_info['idx_final']
                        idx_orig  = tr_info['idx_orig']
                        win_50ms  = tr_info.get('win_50ms', 0)
                        is_max_dur = tr_info.get('is_max_dur', False)
                        idx_peak_speed = tr_info.get('idx_peak_speed', idx_start)
                        
                        # Safety check for lengths
                        limit = len(t) - 1
                        idx_orig = min(limit, idx_orig)
                        idx_final = min(limit, idx_final)
                        idx_peak_speed = min(limit, idx_peak_speed)
                        
                        # Plot Trace
                        ax_db_pos = ax_db
                        
                        # 1. Plot Position (Left Axis)
                        is_excluded = tr_info.get('is_excluded', False)
                        
                        # Policy: If excluded, make explicitly noticeable
                        color_pos = 'tab:blue'
                        alpha_pos = 0.7
                        ls_pos = '-'
                        
                        if is_excluded:
                            color_pos = 'red'
                            alpha_pos = 0.4
                            ls_pos = ':'
                            
                        lns1 = ax_db_pos.plot(t, tr_info['pos'], color=color_pos, alpha=alpha_pos, linestyle=ls_pos, label='Pos (MidX)')
                        ax_db_pos.tick_params(axis='y', labelcolor=color_pos)
                        
                        # 2. Plot Speed (Right Axis)
                        ax_db_spd = ax_db_pos.twinx()
                        color_spd = 'tab:orange'
                        lns2 = ax_db_spd.plot(t, tr_info['speed'], color=color_spd, alpha=0.6, linestyle='-', label='Speed')
                        ax_db_spd.tick_params(axis='y', labelcolor=color_spd)
                        
                        # Mark Start
                        ax_db_pos.axvline(t[min(limit, idx_start)], color='g', linestyle='--', label='Start')
                        
                        # Mark Peak Speed
                        ax_db_spd.plot(t[idx_peak_speed], tr_info['speed'][idx_peak_speed], 'c^', markersize=6, label='Peak Vel')
                        
                        # Mark Threshold Line
                        if 'thresh_v' in tr_info:
                            ax_db_spd.axhline(tr_info['thresh_v'], color='gray', linestyle=':', alpha=0.5, label='End Thresh')

                        # Mark Settled Peak (Refined End)
                        ax_db_pos.plot(t[idx_final], tr_info['pos'][idx_final], 'rx', markersize=8, markeredgewidth=2, label='End')
                        
                        # Title
                        dur = t[idx_final] - t[idx_start]
                        title_color = 'red' if (is_max_dur or is_excluded) else 'black'
                        title_text = f"T{tr_info['idx_trial']} (Dur: {dur:.0f}ms)"
                        if is_excluded: title_text += " [EXCLUDED]"
                        ax_db.set_title(title_text, color=title_color, fontsize=8)
                        
                        if idx_db == 0:
                            ax_db.legend(fontsize='xx-small', loc='upper left')
                            
                    # Turn off unused axes
                    for k in range(n_batch, len(axes_db)):
                        axes_db[k].axis('off')
                        
                    fig_db.tight_layout()
                    # Sanitize condition string for filename
                    safe_cond = "".join([c if c.isalnum() else "_" for c in cond])
                    
                    # Create debug subfolder
                    debug_dir = FIG_OUT_DIR / "debug"
                    debug_dir.mkdir(exist_ok=True)
                    
                    fig_db.savefig(debug_dir / f"Debug_Traces_{target}_{safe_cond}_{p.stem}_page{page+1}.png")

            # Sanity: if no metrics found, skip
            if df.empty:
                return None
            
            # --- APPLY METADATA-BASED CONDITION LABEL ---
            # Get the proper label from metadata
            condition_label = get_condition_label_from_metadata(br_idx, is_control)
            
            # If it's a stim condition and we have a BR index, add BR to label for uniqueness
            if not is_control and br_idx is not None and condition_label != "Control":
                # Check if label already has BR info
                if f"BR {br_idx}" not in condition_label and f"BR{br_idx}" not in condition_label:
                    condition_label = f"{condition_label}\n(BR {br_idx})"
            
            df["Condition"] = condition_label
            df["BR_Index"] = br_idx  # Store BR index for later use
            df["SourceFile"] = p.stem
            
            # Add metadata to Outliers
            for o in outliers:
                o['Condition'] = condition_label
                o['BR_Index'] = br_idx
                o['Port'] = port
                o['Target'] = target
                o['TargetCode'] = target_code
                
            return df, outliers
            
    except Exception as e:
        import traceback
        print(f"Error processing {p.name}: {e}")
        traceback.print_exc()
        return None



def get_kinematics_id(filename: str) -> str:
    """
    Generate a unique ID for the behavior session to avoid duplicates.
    
    Baseline files: Combine across ports/depths for the same session+target.
        - baseline__NRR_RW011_Depth_43.0_port_A_target_A
        - baseline__NRR_RW011_Depth_43.0_port_B_target_A
        → Both get ID: "baseline__NRR_RW011_target_A"
    
    Peristim files: Each BR index is a SEPARATE condition - NO deduplication.
        - peristim__NRR_RW011_251203_141954__BR_006_target_A
        - peristim__NRR_RW011_251203_141954__BR_007_target_A
        → Different IDs (full filename preserved)
    """
    stem = filename.replace('.npz', '')
    
    # 1. Baseline: Deduplicate across ports/depths, but preserve target
    if "baseline__" in stem:
        # Extract session and target
        # Format: baseline__<Session>_Depth_<X>_port_<Y>_target_<Z>
        parts = stem.split("_target_")
        if len(parts) == 2:
            session_part = parts[0].split("_Depth")[0] if "_Depth" in parts[0] else parts[0]
            target_part = parts[1]
            return f"{session_part}_target_{target_part}"
        # Fallback: just remove depth/port info
        if "_Depth" in stem:
            return stem.split("_Depth")[0]
        return stem
        
    # 2. Peristim: NO deduplication - each file is unique
    # Different BR indices = different conditions, must be kept separate
    return stem

def process_target(target: str, target_code: int):
    # flexible search for directories
    # 1. Nested (New)
    # 2. Flat (Old) - try both case variants
    # 3. Root (PeriStim directly)
    candidate_subdirs = [
        Path("stim_reaches") / f"target_{target}",
        Path("control_reaches") / f"target_{target}",
        Path("at_rest"), 
        f"Target_{target}",
        f"target_{target}",
        Path(""), # Check root PeriStim directly
    ]
    
    found_dirs = []
    found_files_raw = []

    for sub in candidate_subdirs:
        d = PERI_ROOT / sub
        if d.exists():
            found_dirs.append(d)
            # Glob files
            files = sorted(d.glob("*.npz"))
            # For root directory, we might want to filter by target if mixed files exist
            # But the logic below filters by filename content anyway via get_kinematics_id logic?
            # Actually no, get_kinematics_id doesn't filter.
            # But duplicate IDs are handled.
            # If files are mixed in root (target A and B), we need to ensure we only pick the right ones?
            # The files are named `..._target_A.npz`. 
            # So filtering by `*target_{target}*.npz` is safer for the root case.
            
            if sub == Path(""): # Root case
                 # Be more specific for root to avoid picking up everything
                 files = sorted(d.glob(f"*target_{target}*.npz"))
                 # Also include baseline files which might use Target or target
                 files.extend(sorted(d.glob(f"*Target_{target}*.npz"))) 
                 # And standard files
                 if not files: # if strict filter found nothing, try filtering by general
                     pass 
            
            found_files_raw.extend(files)

    if not found_files_raw:
        print(f"Skipping {target}: No matching .npz files found in {PERI_ROOT} search paths.")
        print(f"  Checked subdirs: {[str(d) for d in candidate_subdirs]}")
        return pd.DataFrame()
    
    print(f"\nScanning directories for Target {target}:")
    for d in found_dirs:
        print(f"  - {d}")
    
    all_metrics = []
    all_outliers = []

    # Use the collected files
    all_files_raw = sorted(list(set(found_files_raw))) # unique paths
    
    if not all_files_raw:
        print(f"  No .npz files found.")
        return pd.DataFrame()
    
    # Deduplicate: Only process one file per Behavioral Session
    all_files = []
    processed_ids = set()
    
    for p in all_files_raw:
        kid = get_kinematics_id(p.name)
        if kid in processed_ids:
            continue
        processed_ids.add(kid)
        all_files.append(p)
        
    print(f"Found {len(all_files_raw)} total files. Processing {len(all_files)} unique sessions.")
    
    # Parallelize file processing
    # Use ThreadPoolExecutor for I/O bound tasks (loading large files over network)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_file, p, target, target_code): p for p in all_files}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if res is None: continue
                df, file_outliers = res
                
                if df is not None and not df.empty:
                    all_metrics.append(df)
                if file_outliers:
                    all_outliers.extend(file_outliers)
            except Exception as exc:
                print(f'Generated an exception: {exc}')
            
    if not all_metrics:
        return pd.DataFrame()
        
    if all_outliers:
        out_df = pd.DataFrame(all_outliers)
        # Reorder columns
        cols = ['Condition', 'Port', 'Target', 'idx_trial', 'reason', 'value', 'z_score']
        # Filter cols that exist
        cols = [c for c in cols if c in out_df.columns]
        out_df = out_df[cols]
        csv_path = PERI_ROOT / f"outliers_Target_{target}.csv"
        out_df.to_csv(csv_path, index=False)
        print(f"Saved {len(out_df)} outliers to {csv_path}")

    if not all_metrics:
        return pd.DataFrame()
        
    return pd.concat(all_metrics, ignore_index=True)


def combine_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine conditions 12-16 into one group and 17-21 into another.
    Returns a new dataframe with combined conditions.
    """
    if df.empty:
        return df
    
    df_combined = df.copy()
    
    import re
    def get_cond_num(cond_str):
        match = re.search(r"(\d+)", cond_str)
        if match:
            return int(match.group(1))
        return None
    
    # Map conditions to combined groups
    def map_condition(cond_str):
        cond_num = get_cond_num(cond_str)
        if cond_num is not None:
            if 12 <= cond_num <= 16:
                return "Combined_12-16"
            elif 17 <= cond_num <= 21:
                return "Combined_17-21"
        return cond_str
    
    df_combined["Condition"] = df_combined["Condition"].apply(map_condition)
    return df_combined

def detect_outliers(df: pd.DataFrame, target: str, suffix: str = ""):
    """
    Detect outliers in duration_ms using IQR method and save to CSV.
    """
    if df.empty:
        return
    
    outliers_list = []
    
    for condition in df["Condition"].unique():
        cond_df = df[df["Condition"] == condition]
        durations = cond_df["duration_ms"].values
        
        if len(durations) < 4:  # Need at least 4 points for IQR
            continue
        
        # Calculate IQR
        q1 = np.percentile(durations, 25)
        q3 = np.percentile(durations, 75)
        iqr = q3 - q1
        
        # Define outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find outliers with per-condition trial numbering
        trial_num = 0
        for idx, row in cond_df.iterrows():
            duration = row["duration_ms"]
            if duration < lower_bound or duration > upper_bound:
                outliers_list.append({
                    "Condition": condition,
                    "SourceFile": row["SourceFile"],
                    "Trial_in_Condition": trial_num,
                    "Global_Index": idx,
                    "Duration_ms": duration,
                    "Lower_Bound": lower_bound,
                    "Upper_Bound": upper_bound,
                    "Outlier_Type": "Low" if duration < lower_bound else "High"
                })
            trial_num += 1
    
    if outliers_list:
        outliers_df = pd.DataFrame(outliers_list)
        outliers_df.to_csv(FIG_OUT_DIR / f"{target}_outliers{suffix}.csv", index=False)
        # print(f"  Found {len(outliers_list)} outlier(s) for Target {target}{suffix}")
        
        # Remove outliers from dataframe
        outlier_indices = [o["Global_Index"] for o in outliers_list]
        df_clean = df.drop(index=outlier_indices)
        return df_clean
    
    return df



def plot_individual_points(ax, data_list, conditions):
    """Plot individual data points as hollow circles with jitter."""
    plot_data = []
    for i, c in enumerate(conditions):
        y_vals = data_list[i]
        if len(y_vals) > 0:
            for val in y_vals:
                plot_data.append({'x': i, 'y': val, 'condition': c})
    
    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        
        # Use scatter instead of stripplot for true transparency
        # Add jitter manually
        jitter_amount = 0.15
        x_jittered = plot_df['x'] + np.random.uniform(-jitter_amount, jitter_amount, len(plot_df))
        
        ax.scatter(x_jittered, plot_df['y'], 
                   facecolors='none',      # Transparent fill
                   edgecolors='black',     # Black edge
                   linewidths=1,
                   s=40,                   # Size (similar to size=4 in stripplot)
                   alpha=0.7,              # Edge transparency for overlap visibility
                   zorder=10)

def plot_significance(ax, valid_pos, valid_data, baseline_data, conditions, valid_indices):
    """Annotate plot with significance stars (T-test vs Baseline)."""
    if len(baseline_data) <= 3: return
    
    for i, idx in enumerate(valid_indices):
          c_name = conditions[idx]
          # Skip comparing baseline to itself
          c_lower = str(c_name).lower()
          if "baseline" in c_lower or "control" in c_lower:
               continue
               
          curr_data = valid_data[i]
          if len(curr_data) < 3: continue
          
          # T-Test (Welch's)
          t_stat, p_val = stats.ttest_ind(curr_data, baseline_data, equal_var=False)
          
          res_str = ""
          if p_val < 0.001: res_str = "***"
          elif p_val < 0.01: res_str = "**"
          elif p_val < 0.05: res_str = "*"
          
          if res_str:
               x = valid_pos[i]
               y_max = np.max(curr_data)
               # Place text above max
               ax.text(x, y_max, res_str, ha='center', va='bottom', fontsize=20, fontweight='bold', color='black')

def plot_stim_quantification(
    df: pd.DataFrame, 
    target: str,
    suffix: str = "",
    label_map: dict = None):
    if df.empty: return
    
    # Use provided label map or default
    if label_map is None:
        label_map = CONDITION_LABELS
    
    # Calculate Mean/Std per condition
    grp = df.groupby("Condition")[["duration_ms", "peak_speed"]]
    means = grp.mean()
    stds  = grp.std()
    counts = grp.count().iloc[:, 0]
    
    conditions = list(grp.groups.keys())
    
    # Sorting Logic - updated for new label format
    def sort_key(s):
        s_lower = s.lower()
        
        # Control always first
        if "control" in s_lower or "baseline" in s_lower:
            return (0, 0, 0, 0)
        
        # Extract pulses, frequency, and BR number
        n_pulses = 0
        freq = 0
        br_num = 999
        
        # Try to extract pulses: "40p" or "40 p"
        pulse_match = re.search(r"(\d+)\s*p", s_lower)
        if pulse_match:
            n_pulses = int(pulse_match.group(1))
        
        # Try to extract frequency: "400Hz" or "400 Hz"
        freq_match = re.search(r"@\s*(\d+)\s*hz", s_lower)
        if freq_match:
            freq = int(freq_match.group(1))
        
        # Try to extract BR number
        br_match = re.search(r"br\s*(\d+)", s_lower)
        if br_match:
            br_num = int(br_match.group(1))
        
        return (1, n_pulses, freq, br_num)
        
    conditions = sorted(conditions, key=sort_key)
    cond_map = {c: i for i, c in enumerate(conditions)}
    
    # Setup Colors - Control in gray, stim in viridis colors
    n_stim = sum(1 for c in conditions if "control" not in c.lower() and "baseline" not in c.lower())
    cmap = plt.get_cmap("viridis", max(1, n_stim))
    
    color_map = {}
    stim_idx = 0
    for c in conditions:
        c_lower = c.lower()
        if "control" in c_lower or "baseline" in c_lower:
            color_map[c] = "gray"
        else:
            color_map[c] = cmap(stim_idx)
            stim_idx += 1

    # Figure Setup
    plot_types = ['box']
    n_conds = len(conditions)
    
    for p_type in plot_types:
        # Width needs to accommodate conditions
        width = max(6, n_conds * 1.0)
        
        # --- Rows: Metrics ---
        metrics = [("duration_ms", "Reach Duration (ms)")]
        if PLOT_PEAK_SPEED:
            metrics.append(("peak_speed", "Peak Speed"))
        
        # Dynamic subplot creation based on number of metrics
        n_rows = len(metrics)
        height = 5 * n_rows  # 5 inches per row
        
        fig, axes = plt.subplots(n_rows, 1, figsize=(width, height), sharex=True)
        
        # Ensure axes is always iterable (even for single subplot)
        if n_rows == 1:
            axes = [axes]
        
        for ax, (metric, ylabel) in zip(axes, metrics):
            
            x_pos = np.arange(len(conditions))
            
            # 1. Prepare Data for Plotting
            data_list = [df[df["Condition"] == c][metric].dropna().values for c in conditions]
            
            valid_indices = [i for i, d in enumerate(data_list) if len(d) > 0]
            valid_data = [data_list[i] for i in valid_indices]
            valid_pos = x_pos[valid_indices]
            
            # --- Identify Baseline for Stats ---
            baseline_data = []
            for c in conditions:
                c_str = str(c).lower()
                if "baseline" in c_str or "control" in c_str:
                    baseline_data = df[df["Condition"] == c][metric].dropna().values
                    if len(baseline_data) > 0: break
            
            # --- Plotting Logic ---
            if p_type == 'box':
                if valid_data:
                    bp = ax.boxplot(valid_data, positions=valid_pos, widths=0.6, patch_artist=True,
                                   showfliers=False, zorder=5)
                    for idx, box in zip(valid_indices, bp['boxes']):
                        c_name = conditions[idx]
                        color = color_map.get(c_name, 'blue')
                        box.set(facecolor=color, alpha=0.6)
                    for median in bp['medians']:
                        median.set(color='black', linewidth=1.5)

            # --- Individual Points ---
            plot_individual_points(ax, data_list, conditions)
            
            # --- Statistics ---
            plot_significance(ax, valid_pos, valid_data, baseline_data, conditions, valid_indices)
            
            # Adjust Axis
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.set_xticks(x_pos)
            ax.set_xlim(-0.5, len(conditions) - 0.5)
            
            ax.margins(y=0.15)
            ax.set_ylim(bottom=0)
            
            if metric == "duration_ms":
                ax.set_ylim(0, 800)
            
            # X Labels - use condition strings directly (already formatted from metadata)
            labels = [f"{c}\n(n={counts[c]})" for c in conditions]
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            
        title_suffix = " (Combined)" if suffix == "_Combined" else ""
        plt.suptitle(f"Kinematics Summary - {target}{title_suffix} ({p_type})", fontsize=16)
        plt.tight_layout()
        
        # Save figure
        pt_str = p_type.replace("-", "").title()
        save_name = f"{target}_StimQuantification{suffix}_{pt_str}.{OUTPUT_FORMAT}"
        plt.savefig(FIG_OUT_DIR / save_name, bbox_inches='tight')
        plt.close(fig)
    
    if SAVE_STATS_CSV:
        stats_df = means.copy()
        stats_df.columns = [f"{c}_Mean" for c in stats_df.columns]
        stats_df_std = stds.copy()
        stats_df_std.columns = [f"{c}_Std" for c in stats_df_std.columns]
        full_stats = pd.concat([stats_df, stats_df_std, counts.rename("Count")], axis=1)
        full_stats.to_csv(FIG_OUT_DIR / f"{target}_stats_summary{suffix}.csv")



def run_target_analysis(target_char: str, target_code: int, display_name: str):
    """
    Run the full analysis pipeline for a single target:
    - Process kinematics
    - Detect/Remove outliers (optional)
    - Plot individual conditions (optional)
    - Plot combined conditions (optional)
    """
    global _KEYPOINT_DIAG_PRINTED
    # Reset diagnostic flag for each target to show info once per target
    # (Comment out next line if you want it truly once per session)
    # _KEYPOINT_DIAG_PRINTED.clear()
    
    print(f"Processing {display_name} (Target {target_char})...")
    df = process_target(target_char, target_code)
    
    if df.empty:
        print(f"No metrics found for {display_name}.")
        return

    # Use underscore name for file saving (outliers CSV)
    file_name = display_name.replace(" ", "_")

    # Outlier Removal
    if REMOVE_OUTLIERS:
        df_clean = detect_outliers(df, file_name)
    else:
        df_clean = df
        
    # Plot Full Conditions
    if PLOT_FULL_CONDITIONS:
         plot_stim_quantification(df_clean, display_name)
         
    # Combined Analysis
    if ANALYZE_COMBINED:
         df_combined = combine_conditions(df_clean)
         
         if REMOVE_OUTLIERS:
              df_comb_clean = detect_outliers(df_combined, file_name, suffix="_Combined")
         else:
              df_comb_clean = df_combined
              
         plot_stim_quantification(df_comb_clean, display_name, suffix="_Combined", label_map=COMBINED_CONDITION_LABELS)

def main():
    print(f"Quantifying kinematics for Session: {PARAMS.session}")
    print(f"Output Dir: {FIG_OUT_DIR}")
    
    # Target A (code 1) - Left Reach
    run_target_analysis("A", 1, "Left Reach")
        
    # Target B (code 2) - Right Reach
    run_target_analysis("B", 2, "Right Reach")

if __name__ == "__main__":
    main()