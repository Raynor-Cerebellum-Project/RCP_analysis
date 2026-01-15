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
MIN_REACH_DURATION_MS = 70.0 # Exclude <110ms duration
MAX_REACH_DURATION_MS = 600.0

# Position Constraints for detection
START_POS_MAX_THRESH = 0.1  # Start must be within +/- 0.1 of 0 (Tightened)
END_POS_MIN_THRESH = 2.0    # End must be at least +/- 2.0 away from 0

# Track only middle_x as requested
SELECTED_KEYPOINTS = ["middle_x"] 

EXCLUDE_CONDITIONS = [2, 3, 5, 8, 9, 25]  # Can be list of ints (Cond IDs) or strings (substring match)

# Custom labels for conditions
CONDITION_LABELS = {
    "baseline": "Control",
    6: "32 ch \n (1-32)",
    7: "32 ch \n (33-64)",
    9: "32 ch \n (Rest, 33-64)",
    10: "130Hz \n (1-32)",
    11: "130Hz \n (33-64)",
    12: "1-8",
    13: "5-12",
    14: "9-16",
    15: "13-20",
    16: "17-24",
    17: "21-28",
    18: "25-32",
    19: "29-36",
    20: "33-40",
    21: "37-44",
    23: "32 ch \n (1-32) Later",
    24: "32 ch \n (32-64) Later",
    27: "130Hz \n (1-32) Later",
}

# Labels for combined conditions
COMBINED_CONDITION_LABELS = { 
    "baseline": "Control",
    6: "32 ch \n (1-32)",
    7: "32 ch \n (33-64)",
    9: "32 ch \n (Rest, 33-64)",
    10: "130Hz \n (1-32)",
    11: "130Hz \n (33-64)",
    "Combined_12-16": "Chs 1-24",
    "Combined_17-21": "Chs 21-44",
    23: "32 ch \n (1-32) Later",
    24: "32 ch \n (32-64) Later",
    27: "130Hz \n (1-32) Later",
}

# Exclude specific trials within conditions (condition_id: [trial_indices])
# Trial indices are 0-indexed within each condition
EXCLUDE_TRIALS_TARGET_A = {
    6: [0, 8],
    7: [2, 11],
    10: [5, 9],
    11: [3, 6],
    12: [8],
    14: [2, 3, 5, 6, 8],
    15: [2, 4, 5, 6, 8],
    16: [9, 10],
    18: [5],
    19: [0, 4],
    20: [0, 1],
    21: [5],
    23: [8],
    24: [0, 1, 5, 8, 10],
    27: [1, 2],
}

EXCLUDE_TRIALS_TARGET_B = {
    6: [1, 6, 7, 9, 10],
    7: [0, 6, 13],
    10: [12],
    11: [7],
    13: [1],
    14: [10, 11],
    15: [0],
    19: [2, 11],
    21: [2],
    23: [5],
    24: [9],
    27: [1, 5, 9],
}

# Baseline-specific trial exclusions (organized by port and target)
# Format: "baseline_port{A/B}_target{A/B}": [trial_indices]
# User provided 1-based indices, converting to 0-based here:
EXCLUDE_BASELINE_TRIALS = {
    "baseline_portA_targetA": [1, 3, 4, 6, 13, 14, 21, 23, 24, 25, 31, 32, 38],
    "baseline_portB_targetA": [1, 2, 4, 6, 7, 9, 13, 18],
    "baseline_portA_targetB": [6, 10, 13, 15, 16, 17, 20, 21, 22, 24, 25, 30, 39, 40, 42, 44, 47, 56, 57],
    "baseline_portB_targetB": [2, 3, 4, 6, 8, 10, 12, 15, 17, 19, 20, 22, 23, 24, 26, 28, 30, 31, 33, 35],
}

# Plotting Configuration
FIG_OUT_DIR = OUT_BASE / "figures" / "quantify_kinematics"
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output format: 'png' or 'svg'
OUTPUT_FORMAT = 'png'  # Change to 'svg' for vector graphics

# Set to True to generate 5x5 grid of traces for every trial (for debugging alignment)
PLOT_DEBUG_TRACES = True 

# Analysis Flags
REMOVE_OUTLIERS = False      # Set to False to skip outlier detection and removal
ANALYZE_COMBINED = True      # Set to False to skip combined condition analysis
PLOT_FULL_CONDITIONS = False # Set to True to plot individual conditions
SAVE_STATS_CSV = False        # Set to False to skip saving stats CSV 

plt.rcParams.update({'font.size': 14}) # Larger font for plots

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def _get_keypoint_indices(all_kp_names: list[str], selected_substrings: list[str]) -> list[int]:
    """Return indices of keypoints that match any substrings."""
    indices = []
    for i, kp in enumerate(all_kp_names):
        kp_lower = str(kp).lower()
        for sub in selected_substrings:
            if sub.lower() in kp_lower:
                indices.append(i)
                break
    return indices

def calculate_reach_metrics(
    pos_segs: np.ndarray,      # (n_trials, n_kps, T)
    vel_segs: np.ndarray,      # (n_trials, n_kps, T)
    ts_state_segs: np.ndarray, # (n_trials, T_ts)
    t_axis: np.ndarray,        # (T,)
    kp_names: list[str],
    target_code: int,          # 1 for A, 2 for B
    exclude_trials: list = None,  # List of trial indices to exclude
    fs: float = 100.0,         # approx sampling rate if needed, or derive from t_axis
    return_traces: bool = False) -> Tuple[pd.DataFrame, list]:
    """
    Calculate duration, peak speed, path length for each trial.
    Metric: Start at START_OFFSET_MS, End at Max Position Excursion.
    """
    n_trials, n_all_kps, T = pos_segs.shape
    
    # Default to empty list if no exclusions
    if exclude_trials is None:
        exclude_trials = []
    
    # 1. Filter keypoints
    if SELECTED_KEYPOINTS:
        kp_idx = _get_keypoint_indices(kp_names, SELECTED_KEYPOINTS)
        if not kp_idx:
            print(f"[warn] No keypoints matched {SELECTED_KEYPOINTS}. Using all.")
            kp_idx = list(range(n_all_kps))
    else:
        kp_idx = list(range(n_all_kps))
        
    # Speed profile (average across selected KPs)
    trial_speeds_all_kps = vel_segs[:, kp_idx, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        trial_speed = np.nanmean(trial_speeds_all_kps, axis=1) # (n_trials, T)

    # Position profile (Euclidean distance from start pos? Or just average magnitude?)
    metrics_list = []
    collected_traces = []
    all_debug_traces = []

    for i in range(n_trials):
        # Skip excluded trials
        if i in exclude_trials:
            continue
            
        speed = trial_speed[i] # (T,)
        t_trial = t_axis # (T,)
        
        # 1. Identify Start Index (Time closest to START_OFFSET_MS)
        idx_start = (np.abs(t_trial - START_OFFSET_MS)).argmin()
        t_start = t_trial[idx_start]
        
        # --- New Speed-Based Logic ---
        # Track middle_x position for plotting
        sel_pos = pos_segs[i, kp_idx, :]
        trace_for_stop = sel_pos[0, :]
        
        # Find Peak Speed in 0-600ms window
        dt = t_axis[1] - t_axis[0] if len(t_axis) > 1 else 10.0
        if dt < 0.9: dt *= 1000.0 # Convert to ms if needed
        
        max_samples = int(MAX_REACH_DURATION_MS / dt)
        idx_max_dur = min(len(t_trial)-1, idx_start + max_samples)
        
        search_speed = speed[idx_start : idx_max_dur]
        if search_speed.size == 0: continue
            
        idx_peak_speed = idx_start + np.argmax(search_speed)
        
        # Define search params for next step
        STABLE_WIN_MS = 20
        LOW_SPEED_THRESH_RATIO = 0.10 # Tightened from 0.20 to force finding true stop/start
        peak_v = speed[idx_peak_speed]
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
                    if k > w_start: # Ensure window has size
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

        # --- Refine End (Forward Scan) ---
        # Default endpoint is max duration unless found
        idx_peak = idx_max_dur
        
        # Calculate dist_trace (abs displacement from start) for compatibility
        dist_trace = np.abs(trace_for_stop - trace_for_stop[idx_start])
        
        # Scan forward from Peak Velocity for drop
        # Look for speed < thresh_v and staying stable (low mean) for win_samples
        # AND check if position is "at least 2" (>= END_POS_MIN_THRESH)
        scan_limit = max(idx_peak_speed, idx_max_dur - win_samples)
        
        for k in range(idx_peak_speed, scan_limit):
            if speed[k] < thresh_v:
                # Check position constraint: abs(pos) >= END_POS_MIN_THRESH
                if np.abs(trace_for_stop[k]) >= END_POS_MIN_THRESH:
                    if np.mean(speed[k : k + win_samples]) < thresh_v:
                        idx_peak = k
                        break
                    
        t_peak = t_trial[idx_peak]
        
        duration_ms = t_peak - t_start
        
        if duration_ms < MIN_REACH_DURATION_MS or duration_ms > MAX_REACH_DURATION_MS:
            continue
            
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
        metrics_list.append(m_dict)
        
        # Store debug info for ALL trials
        is_max_duration = (idx_peak == idx_max_dur)
        
        all_debug_traces.append({
            "idx_trial": i,
            "t": t_trial,
            "pos": trace_for_stop,  # New key
            "y": trace_for_stop,    # Legacy key (for plotting)
            "speed": speed,
            "idx_start": idx_start,
            "idx_final": idx_peak, 
            "idx_orig": idx_peak_speed, # Use peak speed as ref
            "is_max_dur": is_max_duration,
            "idx_peak_speed": idx_peak_speed,
            "thresh_v": thresh_v,
            "win_50ms": int(0.050/dt)
        })
        
        if return_traces:
            # Shift time to align 0 to Entry (idx_end)
            collected_traces.append((t_trial, dist_trace))
        
    df = pd.DataFrame(metrics_list)
    if return_traces:
        return df, collected_traces, all_debug_traces
    return df, [], all_debug_traces


def process_single_file(p: Path, target: str, target_code: int) -> Optional[pd.DataFrame]:
    try:
        with np.load(p, allow_pickle=True) as data:
            # 1. Determine Condition from 'overall_title' or filename
            if p.name.startswith("baseline__"):
                cond = "Baseline"
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
            
            # Check exclusions
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

            # Key names
            pos_key = f"beh_{CAMERA_TO_USE}_segs"
            vel_key = f"beh_{CAMERA_TO_USE}_vel_segs"
            
            if pos_key not in data: # Fallback
                pos_key = f"{CAMERA_TO_USE}_segs"
                vel_key = f"{CAMERA_TO_USE}_vel_segs"
                
            if pos_key not in data or vel_key not in data:
                print(f"Skipping {p.name}: missing keys")
                return None
                
            pos_segs = data[pos_key] # (n_trials, n_kps, T)
            vel_segs = data[vel_key]
            t_axis   = data["beh_rel_t"]
            
            # TS Check
            if "ts_state_segs" in data:
                ts_state = data["ts_state_segs"]
            else:
                ts_state = np.zeros((pos_segs.shape[0], pos_segs.shape[2])) 

            names_key = f"beh_{CAMERA_TO_USE}_names"
            kp_names = data[names_key].tolist() if names_key in data else []
            
            # Determine trial exclusions
            trials_to_exclude = []
            
            # Check if this is a baseline file
            if cond == "Baseline":
                # Parse filename to determine port (A or B)
                fname_lower = p.name.lower()
                port = "A"  # Default
                
                if "_portb_" in fname_lower or "portb" in fname_lower or "_port_b_" in fname_lower or "port_b" in fname_lower:
                    port = "B"
                elif "_porta_" in fname_lower or "porta" in fname_lower or "_port_a_" in fname_lower or "port_a" in fname_lower:
                    port = "A"
                
                # Determine target from target_code (1=A, 2=B)
                target_letter = "A" if target_code == 1 else "B"
                
                # Build exclusion key
                baseline_key = f"baseline_port{port}_target{target_letter}"
                
                print(f"  [Processing] Baseline File: {p.name} -> Config: {baseline_key}")
                
                # Get exclusions for this baseline configuration
                if baseline_key in EXCLUDE_BASELINE_TRIALS:
                    trials_to_exclude = EXCLUDE_BASELINE_TRIALS[baseline_key]
            
            # If not baseline (or even if it is, but currently logic separates), check condition-based exclusions
            # Note: "Baseline" condition usually doesn't match (\d+) regex, but checking explicitly
            if cond != "Baseline":
                 match = re.search(r"(\d+)", cond)
                 if match:
                     cond_num = int(match.group(1))
                     # Select correct exclusion dict based on Target (1=A, 2=B)
                     exclude_dict = EXCLUDE_TRIALS_TARGET_A if target_code == 1 else EXCLUDE_TRIALS_TARGET_B
                     
                     if cond_num in exclude_dict:
                         trials_to_exclude.extend(exclude_dict[cond_num])
            
            # Retrieve metrics AND debug traces
            df, _, all_debug_traces = calculate_reach_metrics(
                pos_segs, vel_segs, ts_state, t_axis, kp_names, target_code, 
                exclude_trials=trials_to_exclude, return_traces=False)
            
            # Plot Debug Traces for ALL trials (if enabled)
            # Use thread-safe Figure interface (no global pyplot state)
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
                        color_pos = 'tab:blue'
                        lns1 = ax_db_pos.plot(t, tr_info['pos'], color=color_pos, alpha=0.7, label='Pos (MidX)')
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
                        title_color = 'red' if is_max_dur else 'black'
                        ax_db.set_title(f"T{tr_info['idx_trial']} (Dur: {dur:.0f}ms)", color=title_color, fontsize=8)
                        
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
                    # No explicit close needed for Figure object, but helpful
                    # del fig_db

            # Sanity: if no metrics found, skip
            if df.empty:
                return None
                
            df["Condition"] = cond
            df["SourceFile"] = p.stem
            
            # Exclude specific trials for this condition
            # (Moved logic to before calculate_reach_metrics so debug traces are also filtered)
                        
            return df
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing {p.name}: {e}")
        return None

def process_target(target: str, target_code: int):
    results_dir = PERI_ROOT / f"Target_{target}"
    if not results_dir.exists():
        print(f"Skipping {target}: {results_dir} not found.")
        return pd.DataFrame()
    
    print(f"\nScanning {results_dir}...")
    
    all_metrics = []

    # Files
    all_files = sorted(results_dir.glob("*.npz"))
    
    # Parallelize file processing
    # Use ThreadPoolExecutor for I/O bound tasks (loading large files over network)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_file, p, target, target_code): p for p in all_files}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                df = future.result()
                if df is not None and not df.empty:
                    all_metrics.append(df)
            except Exception as exc:
                print(f'Generated an exception: {exc}')
            
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
    
    # Sorting Logic
    def sort_key(s):
        s_lower = s.lower()
        if "baseline" in s_lower:
            return (0, 0)
        import re
        match = re.search(r"Cond_(\d+)", s)
        if match: return (1, int(match.group(1)))
        match_any = re.search(r"(\d+)", s)
        if match_any: return (1, int(match_any.group(1)))
        return (2, s)
        
    conditions = sorted(conditions, key=sort_key)
    cond_map = {c: i for i, c in enumerate(conditions)}
    
    # Setup Colors
    cmap = plt.get_cmap("viridis", len(conditions))
    color_map = {c: cmap(i) for i, c in enumerate(conditions)}
    for c in conditions:
        if "baseline" in c.lower():
            color_map[c] = "gray"

    # Figure Setup: 3 Rows using GridSpec
    # Row 0: Duration (Violin)
    # Row 1: Peak Speed (Violin)
    
    # Prepare Data Reuse
    plot_types = ['violin-box', 'box']
    n_conds = len(conditions)
    
    for p_type in plot_types:
        # Width needs to accommodate conditions
        width = max(5, n_conds * 0.8)
        height = 10
        
        fig, axes = plt.subplots(2, 1, figsize=(width, height), sharex=True)
        
        # --- Rows: Metrics ---
        metrics = [("duration_ms", "Reach Duration (ms)"), ("peak_speed", "Peak Speed")]
        
        for ax, (metric, ylabel) in zip(axes, metrics):
            
            x_pos = np.arange(len(conditions))
            
            # 1. Prepare Data for Plotting
            data_list = [df[df["Condition"] == c][metric].dropna().values for c in conditions]
            
            valid_indices = [i for i, d in enumerate(data_list) if len(d) > 0]
            valid_data = [data_list[i] for i in valid_indices]
            valid_pos = x_pos[valid_indices]
            
            # --- Identify Baseline for Stats ---
            baseline_data = [] # Data array
            for c in conditions:
                # Naive check for baseline/control condition
                c_str = str(c).lower()
                if "baseline" in c_str or "control" in c_str:
                     baseline_data = df[df["Condition"] == c][metric].dropna().values
                     if len(baseline_data) > 0: break
            
            # --- Plotting Logic based on Type ---
            
            if p_type == 'violin-box' or p_type == 'violin':
                # Violin plot
                if valid_data:
                    parts = ax.violinplot(valid_data, positions=valid_pos, showmeans=False, showmedians=False, showextrema=False, widths=0.7)
                    
                    if 'bodies' in parts:
                        for idx, pc in zip(valid_indices, parts['bodies']):
                            c_name = conditions[idx]
                            color = color_map.get(c_name, 'blue')
                            pc.set_facecolor(color)
                            pc.set_edgecolor('black')
                            pc.set_alpha(0.5)
                    
                    # Box plot inside (narrow)
                    bp = ax.boxplot(valid_data, positions=valid_pos, widths=0.15, patch_artist=True, 
                                   showfliers=False, zorder=5)
                    for box in bp['boxes']:
                        box.set(facecolor='white', alpha=0.9, linewidth=1)
                    for whisker in bp['whiskers']: 
                        whisker.set(linewidth=1)
                    for cap in bp['caps']:
                        cap.set(linewidth=1)
                    for median in bp['medians']:
                        median.set(color='black', linewidth=1.5)

            elif p_type == 'box':
                # Standard Box plot
                if valid_data:
                    bp = ax.boxplot(valid_data, positions=valid_pos, widths=0.6, patch_artist=True,
                                   showfliers=False, zorder=5)
                    for idx, box in zip(valid_indices, bp['boxes']):
                        c_name = conditions[idx]
                        color = color_map.get(c_name, 'blue')
                        box.set(facecolor=color, alpha=0.6)
                    for median in bp['medians']:
                        median.set(color='black', linewidth=1.5)

            # --- Individual Points (Hollow Circles, Jittered) ---
            plot_individual_points(ax, data_list, conditions)
            
            # --- Statistics (T-Test vs Baseline) ---
            plot_significance(ax, valid_pos, valid_data, baseline_data, conditions, valid_indices)
            
            # Adjust Axis
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.set_xticks(x_pos)
            ax.set_xlim(-0.5, len(conditions) - 0.5) # Ensure consistent spacing
            
            # Set Y-axis limits
            # Add margin for stars and remove hard cap
            ax.margins(y=0.15)
            ax.set_ylim(bottom=0)
            
            # X Labels with custom mapping
            import textwrap
            
            def get_label(cond_str):
                # Check if baseline
                if "baseline" in cond_str.lower():
                    return label_map.get("baseline", cond_str)
                # Check if combined condition
                if cond_str.startswith("Combined_"):
                    return label_map.get(cond_str, cond_str)
                # Extract condition number
                match = re.search(r"(\d+)", cond_str)
                if match:
                    cond_num = int(match.group(1))
                    return label_map.get(cond_num, cond_str)
                return cond_str
            
            labels = [f"{get_label(c)}\n(n={counts[c]})" for c in conditions]
            labels_wrapped = ["\n".join(textwrap.wrap(l, 20)) for l in labels]
            ax.set_xticklabels(labels_wrapped, rotation=45, ha="right", fontsize=9)
            
        title_suffix = " (Combined)" if suffix == "_Combined" else ""
        plt.suptitle(f"Kinematics Summary - {target}{title_suffix} ({p_type})", fontsize=16)
        plt.tight_layout()
        
        # Save figure with plot type in filename
        # Clean up p_type string for filename (remove hyphen if desired, or keep as is)
        pt_str = p_type.replace("-", "").title()
        save_name = f"{target}_StimQuantification{suffix}_{pt_str}.{OUTPUT_FORMAT}"
        plt.savefig(FIG_OUT_DIR / save_name, bbox_inches='tight')
        plt.close(fig)
    
    if SAVE_STATS_CSV:
        # Save statistics CSV
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

