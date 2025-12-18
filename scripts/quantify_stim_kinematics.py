import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from typing import Optional, Tuple
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
CAMERA_TO_USE = "cam1"  # 'cam1' or 'cam0'
START_OFFSET_MS = 0  # Start counting reach from this time (relative to alignment) -> base off of triggering IR sensor
# Duration = Time(Max Pos) - START_OFFSET_MS
MIN_REACH_DURATION_MS = 110 # Exclude <110ms duration
MAX_REACH_DURATION_MS = 600.0

SELECTED_KEYPOINTS = ["wrist_y"]  # Track only wrist_y (index 8 in beh_cam1_names)

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
EXCLUDE_TRIALS = {
    9: [7, 10, 11, 13],
    10: [5],
    14: [7],
    18: [5, 7],
    19: [4],
    20: [3, 5],
    21: [7],
    23: [1, 9],
    24: [5, 15],
    27: [1, 3, 6],
}

# Baseline-specific trial exclusions (organized by port and target)
# Format: "baseline_port{A/B}_target{A/B}": [trial_indices]
# User provided 1-based indices, converting to 0-based here:
EXCLUDE_BASELINE_TRIALS = {
    "baseline_portA_targetA": [1, 2, 3, 4, 6, 13, 14, 21, 24, 31],
    "baseline_portB_targetA": [6, 9],
    "baseline_portA_targetB": [6, 10, 13, 16, 20, 21, 22, 24, 31, 39, 40, 47, 56, 57],
    "baseline_portB_targetB": [6, 8, 11, 12, 14, 15, 17, 20, 22, 26, 30, 31],
}

# Plotting Configuration
FIG_OUT_DIR = OUT_BASE / "figures" / "quantify_kinematics"
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output format: 'png' or 'svg'
OUTPUT_FORMAT = 'svg'  # Change to 'svg' for vector graphics

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
        
        # Ensure we look FORWARD from start
        # 2. Identify Max Position Excursion *after* start
        # Calculate displacement from start for selected keypoints
        # (n_sel_kps, T)
        sel_pos = pos_segs[i, kp_idx, :]
        start_pos = sel_pos[:, idx_start][:, None] # (n_sel, 1)
        displacement = sel_pos - start_pos # (n_sel, T)
        
        # Metric: Combined displacement magnitude (Sum of Squares -> Distance proxy)
        dist_trace = np.sqrt(np.nansum(displacement**2, axis=0)) # (T,)
        
        # Find peak distance in the valid window [Start:]
        valid_dist = dist_trace[idx_start:]
        if valid_dist.size == 0: continue
            
        idx_peak_rel = np.argmax(valid_dist)
        idx_peak = idx_start + idx_peak_rel
        
        # --- Advanced End Point Detection ---
        # 1. Identify Y-channel (or fallback to Euclidean distance if no Y found)
        # Find index of Y channel in the selected keypoints
        sel_kp_names = [kp_names[k] for k in kp_idx]
        y_row_idx = None
        for r, name in enumerate(sel_kp_names):
            if "_y" in name.lower():
                y_row_idx = r
                break
        
        # Use Y-channel if found, else use dist_trace
        if y_row_idx is not None:
            trace_for_stop = sel_pos[y_row_idx, :]
        else:
            trace_for_stop = dist_trace
            
        # 2. Define "Assumed Stop" Window (20ms window starting at idx_peak)
        # Assuming fs=100Hz from default or infer from t_axis?
        # t_axis step
        dt = t_axis[1] - t_axis[0] if len(t_axis) > 1 else 0.01
        win_samples = int(0.02 / dt) # 20ms
        win_samples = max(1, win_samples)
        
        # Window: [idx_peak, idx_peak + win_samples] (clipped to length)
        stop_start = idx_peak
        stop_end = min(len(t_trial), idx_peak + win_samples)
        
        idx_peak_speed = idx_start # Default
        
        if stop_end > stop_start:
            # Check if this is a "Max Duration" reach (likely ~600ms or hit end of window)
            # User Rule: if detected at 600ms (or close), use last 100ms baseline method
            is_max_duration = (stop_start >= len(t_trial) - 5) or ((t_trial[stop_start] - t_start) >= (MAX_REACH_DURATION_MS - 10))
            
            if is_max_duration:
                # --- New Logic: Consecutive Window Stability Check (Local First Match) ---
                # User Request (Step 1602): 
                # "make windows 25ms... look to see if windows are of similar means and standard deviations... take the first one"
                
                # --- New Logic: Absolute Windowed Stability Search (0-600ms) ---
                # User Request (Step 1636): 
                # 1. Take absolute value of trace.
                # 2. Windows 0-600ms (100ms width, 50ms overlap).
                # 3. Discard huge STD (> 1.0?).
                # 4. Compare remaining windows: if similar (mean/std diff < 0.3), take first one.
                # 5. Find Min (A) or Max (B) within that window in ORIGINAL trace.
                
                # Parameters
                WIN_LEN_MS = 20
                STEP_MS = 10
                MAX_TIME_MS = 600
                HUGE_STD_THRESH = 0.5 # Heuristic for "Huge"
                SIM_THRESH = 0.3 # User defined
                
                # Correction for dt units (ms vs s)
                # t_axis is typically in ms (e.g. -600, -590...), so dt ~ 10.0
                if dt > 0.9: 
                     # dt is likely in ms
                     win_samples = int(WIN_LEN_MS / dt)
                     step_samples = int(STEP_MS / dt)
                     max_samples = int(MAX_TIME_MS / dt)
                else:
                     # dt is likely in seconds (e.g. 0.01)
                     win_samples = int(WIN_LEN_MS / (dt*1000.0))
                     step_samples = int(STEP_MS / (dt*1000.0))
                     max_samples = int(MAX_TIME_MS / (dt*1000.0))
                
                win_samples = max(1, win_samples)
                step_samples = max(1, step_samples)
                max_samples = max(1, max_samples)
                
                # 1. Absolute Trace for Stability Analysis
                abs_trace = np.abs(trace_for_stop)
                
                # 2. Generate and Analyze Windows
                # Scan from t=0 (idx_start) to t=600ms
                # Note: `idx_start` aligns with t=0 (stim onset)
                
                valid_windows = [] # Store (start_idx, end_idx, mean, std)
                
                start_range = idx_start
                end_range = idx_start + max_samples
                end_range = min(len(abs_trace), end_range)
                
                scan_limit = end_range - win_samples
                
                for k in range(start_range, scan_limit, step_samples):
                    w_start = k
                    w_end = k + win_samples
                    
                    if w_end > len(abs_trace):
                        break
                    
                    chunk = abs_trace[w_start : w_end]
                    curr_mean = np.mean(chunk)
                    curr_std = np.std(chunk)
                    
                # 3. Filter "Huge" Standard Deviations
                # User (Step 1681): "let's not discard the large std windows."
                # We interpret this as: We need them for the sequence check, but we won't 
                # select one as the start point? Or just keep them in the list.
                # Actually, we keep them to check continuity.
                valid_windows.append({
                    's': w_start, 'e': w_end, 
                    'm': curr_mean, 'std': curr_std
                })
                
                # 4. Compare Remaining Windows for Similarity + Future Check
                chosen_window = None
                last_similar_window = None
                
                if len(valid_windows) >= 2:
                    for i in range(len(valid_windows) - 1):
                        w1 = valid_windows[i]
                        w2 = valid_windows[i+1] # Adjacent
                        
                        # Similarity Check (Is this a stable pair?)
                        std_diff = abs(w1['std'] - w2['std'])
                        
                        if std_diff < SIM_THRESH:
                            # Found a potential stable start (W1)
                            
                            # Future Check (Look ahead limit to 159ms)
                            # "future change is only within 159ms of the window being looked at"
                            is_future_bad = False
                            
                            # Calculate 159ms in samples
                            # Check if dt is ms or s (same logic as above)
                            if dt > 0.9:
                                samples_159ms = int(159.0 / dt)
                            else:
                                samples_159ms = int(159.0 / (dt*1000.0))
                            
                            future_limit_idx = w1['s'] + samples_159ms
                            
                            # Scan future windows
                            for k in range(i + 2, len(valid_windows)):
                                wk = valid_windows[k]
                                
                                # Stop if we are beyond the 159ms window
                                if wk['s'] > future_limit_idx:
                                    break
                                
                                # Check for "Big" changes
                                # "big std" (std > HUGE)
                                future_std = wk['std']
                                
                                # Note: w1 is the reference for mean shift? Or wk-1?
                                # "windows afterwards have big std" -> absolute check
                                
                                if future_std > HUGE_STD_THRESH:
                                    is_future_bad = True
                                    break
                            
                            if is_future_bad:
                                # This stability is transient (just a pause before big move within 159ms).
                                # "move on to windows after that"
                                # Note this match as a fallback "last of comparable windows"
                                last_similar_window = w1
                                continue # Keep searching
                            else:
                                # Future is okay (or assumed okay if end of trace / past 159ms)
                                chosen_window = w1
                                break
                    
                    # If process exerts itself (no 'perfect' match found), use the last comparable window
                    if chosen_window is None and last_similar_window is not None:
                         chosen_window = last_similar_window
                    
                    # Fallback if NOTHING similar found at all?
                    if chosen_window is None and len(valid_windows) > 0:
                        chosen_window = valid_windows[0] # Very weak fallback
                elif len(valid_windows) == 1:
                     chosen_window = valid_windows[0]
                
                # 5. Refine End-Point in Chosen Window
                if chosen_window:
                    # Look in ORIGINAL trace (trace_for_stop)
                    search_slice = trace_for_stop[chosen_window['s'] : chosen_window['e']]
                    
                    if len(search_slice) > 0:
                        if target_code == 2: # Target B -> Maxima
                            peak_idx_rel = np.argmax(search_slice)
                        else: # Target A -> Minima (Default)
                            peak_idx_rel = np.argmin(search_slice)
                        
                        idx_peak = chosen_window['s'] + peak_idx_rel
                
                else:
                    # Fallback if NO valid windows (all high noise)
                    # Maybe just stick with original detection or max displacement?
                    pass # idx_peak remains whatever it was (default)

            else:
                # --- Previous Logic (Standard 20ms post-peak window) ---
                stop_segment = trace_for_stop[stop_start:stop_end]
                mean_stop = np.mean(stop_segment)
                std_stop = np.std(stop_segment)
                
                # 3. Scan backwards from idx_peak
                # Go back to start of reach
                # Find first point (going backwards) where value crosses 1 SD threshold
                # i.e., abs(val - mean) > 1 * std
                
                # We look for the TRANSITION from "Outside" to "Inside".
                # Scanning backwards: usually we are Inside (at peak/stop). We look for when we were Outside.
                # The "first point of where y values are crossing the 1 SD threshold" relative to the stop.
                
                final_idx = idx_peak # Default to original peak
                
                # Search backwards from idx_peak to idx_start
                found_crossing = False
                crossing_idx = idx_peak
                
                for k in range(idx_peak, idx_start, -1):
                    val = trace_for_stop[k]
                    if np.abs(val - mean_stop) > 1.0 * std_stop:
                        # Found a point outside 1 SD
                        crossing_idx = k
                        found_crossing = True
                        break
                
                if found_crossing:
                    # User: "as long as values after that point are within 2 standard deviations, consider that to be the final reach point"
                    # Check condition: From crossing_idx+1 to idx_peak (or end of window?), are all values within 2 SD?
                    candidate_idx = crossing_idx + 1
                    
                    # Verify stability from candidate to stop_end (or idx_peak?)
                    # "values after that point" -> implies up to the steady state?
                    check_segment = trace_for_stop[candidate_idx : stop_end]
                    
                    is_stable = np.all(np.abs(check_segment - mean_stop) <= 2.0 * std_stop)
                    
                    if is_stable:
                        idx_peak = candidate_idx # Update end point
                    
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
        
        if is_max_duration:
            # Store debug info for this max duration trial
            pass # Handled below generically now
        
        # Determine original peak index (before refinement)
        # For max dur: stop_start (approx 600ms)
        # For standard: stop_start (peak of valid_dist)
        idx_orig_peak = stop_start
        
        # Store debug info for ALL trials
        all_debug_traces.append({
            "idx_trial": i,
            "t": t_trial,
            "y": trace_for_stop,
            "idx_start": idx_start,
            "idx_final": idx_peak, # The refined peak
            "idx_orig": idx_orig_peak,
            "is_max_dur": is_max_duration,
            "idx_peak_speed": idx_peak_speed,
            "win_50ms": int(0.050/dt) # Passing parameter for plotting scaling if needed
        })
        
        if return_traces:
            # Shift time to align 0 to Entry (idx_end)
            collected_traces.append((t_trial, dist_trace))
        
    df = pd.DataFrame(metrics_list)
    if return_traces:
        return df, collected_traces, all_debug_traces
    return df, [], all_debug_traces


def process_target(target: str, target_code: int):
    results_dir = PERI_ROOT / f"Target_{target}"
    if not results_dir.exists():
        print(f"Skipping {target}: {results_dir} not found.")
        return pd.DataFrame(), {}
    
    print(f"\nScanning {results_dir}...")
    
    all_metrics = []
    aggregated_traces = {} # Condition -> list of (t, speed)

    # Files
    all_files = sorted(results_dir.glob("*.npz"))
    
    for p in all_files:
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
                        
                        # Cleanup: remove 'Target X' if present to make shorter?
                        # user wants 'freq: 400 ...'
                        # typically formats as "Cond_XX: freq..."
                        # Let's keep it as is, maybe strip file part if redundant.
                    else:
                        # Fallback to parsing filename
                        parts = p.name.split("_")
                        if len(parts) > 1 and parts[1].isdigit():
                            cond = f"Cond_{parts[1]}"
                        else:
                            cond = "Stim"
                
                # print(cond)
                # Check exclusions
                is_excluded = False
                for ex in EXCLUDE_CONDITIONS:
                    if isinstance(ex, int):
                        # Check if condition string contains this number as a distinct integer (e.g. Cond_027 matches 27)
                        # We specifically look for "Cond_X" or just numbers in the string
                        # Using regex to find the first integer in the string
                        match = re.search(r"(\d+)", cond)
                        if match and int(match.group(1)) == ex:
                            is_excluded = True
                            break
                    elif isinstance(ex, str):
                        if ex in cond:
                            is_excluded = True
                            break
                
                if is_excluded:
                    continue

                # Key names
                pos_key = f"beh_{CAMERA_TO_USE}_segs"
                vel_key = f"beh_{CAMERA_TO_USE}_vel_segs"
                
                if pos_key not in data: # Fallback
                    pos_key = f"{CAMERA_TO_USE}_segs"
                    vel_key = f"{CAMERA_TO_USE}_vel_segs"
                    
                if pos_key not in data or vel_key not in data:
                    print(f"Skipping {p.name}: missing keys")
                    continue
                    
                pos_segs = data[pos_key] # (n_trials, n_kps, T)
                vel_segs = data[vel_key]
                t_axis   = data["beh_rel_t"]
                print(f"  t_axis range: {t_axis[0]:.0f} to {t_axis[-1]:.0f} ms, N={len(t_axis)}")
                
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
                    # Baseline files typically named: baseline__port{A/B}_...
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
                    
                    print(f"  [DEBUG] Baseline File: {p.name} -> Config: {baseline_key}")
                    
                    # Get exclusions for this baseline configuration
                    if baseline_key in EXCLUDE_BASELINE_TRIALS:
                        trials_to_exclude = EXCLUDE_BASELINE_TRIALS[baseline_key]
                        print(f"    -> Excluding indices: {trials_to_exclude}")
                
                # Retrieve metrics AND debug traces
                df, _, all_debug_traces = calculate_reach_metrics(
                    pos_segs, vel_segs, ts_state, t_axis, kp_names, target_code, 
                    exclude_trials=trials_to_exclude, return_traces=False)
                
                # Plot Debug Traces for ALL trials
                if all_debug_traces:
                    # Batch into pages of 25
                    batch_size = 25
                    n_total = len(all_debug_traces)
                    n_pages = int(np.ceil(n_total / batch_size))
                    
                    for page in range(n_pages):
                        batch_traces = all_debug_traces[page*batch_size : (page+1)*batch_size]
                        n_batch = len(batch_traces)
                        
                        cols = 5
                        rows = 5
                        
                        fig_db, axes_db = plt.subplots(rows, cols, figsize=(20, 15), squeeze=False)
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
                            ax_db.plot(t, y, 'k-', alpha=0.7)
                            
                            # Mark Start
                            ax_db.axvline(t[min(limit, idx_start)], color='g', linestyle='--', label='Start')
                            
                            # Mark Peak Speed (Scan Start)
                            ax_db.plot(t[idx_peak_speed], y[idx_peak_speed], 'c^', markersize=6, label='Peak Vel')
                            
                            if is_max_dur:
                                # Mark Original Peak (likely 600ms limit)
                                ax_db.plot(t[idx_orig], y[idx_orig], 'bo', label='Orig 600ms')
                                
                                # Highlight Baseline Region (Last 50ms)
                                bl_start_idx = max(0, len(y) - win_50ms)
                                bl_start_idx = min(limit, bl_start_idx)
                                ax_db.axvspan(t[bl_start_idx], t[-1], color='gray', alpha=0.2, label='Baseline')
                            else:
                                # Standard trial
                                ax_db.plot(t[idx_orig], y[idx_orig], 'b.', alpha=0.5, label='Orig Peak')

                            # Mark Settled Peak (Refined End)
                            ax_db.plot(t[idx_final], y[idx_final], 'rx', markersize=8, markeredgewidth=2, label='End')
                            
                            # Title
                            dur = t[idx_final] - t[idx_start]
                            title_color = 'red' if is_max_dur else 'black'
                            ax_db.set_title(f"T{tr_info['idx_trial']} (Dur: {dur:.0f}ms)", color=title_color, fontsize=8)
                            
                            if idx_db == 0:
                                ax_db.legend(fontsize='xx-small', loc='upper left')
                                
                        # Turn off unused axes
                        for k in range(n_batch, len(axes_db)):
                            axes_db[k].axis('off')
                            
                        plt.tight_layout()
                        # Sanitize condition string for filename
                        safe_cond = "".join([c if c.isalnum() else "_" for c in cond])
                        
                        # Create debug subfolder
                        debug_dir = FIG_OUT_DIR / "debug"
                        debug_dir.mkdir(exist_ok=True)
                        
                        plt.savefig(debug_dir / f"Debug_Traces_{target}_{safe_cond}_{p.stem}_page{page+1}.png")
                        plt.close(fig_db)

                # Sanity: if no metrics found, skip
                if df.empty:
                    continue
                    
                df["Condition"] = cond
                df["SourceFile"] = p.stem
                
                # Exclude specific trials for this condition
                # Extract condition number
                match = re.search(r"(\d+)", cond)
                if match:
                    cond_num = int(match.group(1))
                    if cond_num in EXCLUDE_TRIALS:
                        # Get trials to exclude (0-indexed)
                        trials_to_exclude = EXCLUDE_TRIALS[cond_num]
                        # Filter out these trial indices
                        df = df[~df.index.isin(trials_to_exclude)].reset_index(drop=True)
                        if df.empty:
                            continue
                
                all_metrics.append(df)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing {p.name}: {e}")
            
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
        print(f"  Found {len(outliers_list)} outlier(s) for Target {target}{suffix}")
        
        # Remove outliers from dataframe
        outlier_indices = [o["Global_Index"] for o in outliers_list]
        df_clean = df.drop(index=outlier_indices)
        return df_clean
    
    return df



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
    # We want to generate 3 figures: violin-box, box, bar
    plot_types = ['violin-box', 'box', 'bar']
    n_conds = len(conditions)
    
    for p_type in plot_types:
        # Width needs to accommodate conditions
        width = max(10, n_conds * 1.5)
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

            elif p_type == 'bar':
                # Bar plot with error bars (std dev)
                mean_vals = [means.loc[c, metric] if c in means.index else 0 for c in conditions]
                std_vals = [stds.loc[c, metric] if c in stds.index else 0 for c in conditions]
                
                # Plot bars
                bars = ax.bar(x_pos, mean_vals, yerr=std_vals, capsize=5, alpha=0.6, 
                             color=[color_map.get(c, 'blue') for c in conditions],
                             edgecolor='black', linewidth=0.5)
            
            # --- Individual Points (Swarmplot) ---
            plot_data = []
            for i, c in enumerate(conditions):
                y_vals = data_list[i]
                if len(y_vals) > 0:
                    for val in y_vals:
                        plot_data.append({'x': i, 'y': val, 'condition': c})
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                sns.swarmplot(data=plot_df, x='x', y='y', ax=ax, 
                             color='black', size=2.5, alpha=0.7, zorder=10)
            
            # Adjust Axis
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.set_xticks(x_pos)
            ax.set_xlim(-0.5, len(conditions) - 0.5) # Ensure consistent spacing
            
            # Set Y-axis limits
            if metric == "duration_ms":
                ax.set_ylim(0, 600)  # Duration: 0-600ms range
            else:
                ax.set_ylim(bottom=0)  # Other metrics: start from zero
            
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
            ax.set_xticklabels(labels_wrapped, rotation=0, ha="center", fontsize=9)
            
        title_suffix = " (Combined)" if suffix == "_Combined" else ""
        plt.suptitle(f"Kinematics Summary - {target}{title_suffix} ({p_type})", fontsize=16)
        plt.tight_layout()
        
        # Save figure with plot type in filename
        # Clean up p_type string for filename (remove hyphen if desired, or keep as is)
        pt_str = p_type.replace("-", "").title()
        save_name = f"{target}_StimQuantification{suffix}_{pt_str}.{OUTPUT_FORMAT}"
        plt.savefig(FIG_OUT_DIR / save_name, bbox_inches='tight')
        plt.close(fig)
    
    # Save statistics CSV
    stats_df = means.copy()
    stats_df.columns = [f"{c}_Mean" for c in stats_df.columns]
    stats_df_std = stds.copy()
    stats_df_std.columns = [f"{c}_Std" for c in stats_df_std.columns]
    full_stats = pd.concat([stats_df, stats_df_std, counts.rename("Count")], axis=1)
    full_stats.to_csv(FIG_OUT_DIR / f"{target}_stats_summary{suffix}.csv")

def main():
    print(f"Quantifying kinematics for Session: {PARAMS.session}")
    print(f"Output Dir: {FIG_OUT_DIR}")
    
    # Target A (code 1) - Left Reach
    print("Processing Left Reach (Target A)...")
    df_a = process_target("A", 1)
    if not df_a.empty:
        # Detect outliers and remove them
        df_a_clean = detect_outliers(df_a, "Left_Reach")
        
        # Original figure (Cleaned)
        plot_stim_quantification(df_a_clean, "Left Reach")
        
        # Combined figure (Cleaned)
        df_a_combined = combine_conditions(df_a_clean)
        
        # Also detect/remove outliers from combined groups if any remain relative to group stats
        df_a_combined_clean = detect_outliers(df_a_combined, "Left_Reach", suffix="_Combined")
        
        plot_stim_quantification(df_a_combined_clean, "Left Reach", suffix="_Combined", label_map=COMBINED_CONDITION_LABELS)
    else:
        print("No metrics found for Left Reach.")
        
    # Target B (code 2) - Right Reach
    print("Processing Right Reach (Target B)...")
    df_b = process_target("B", 2)
    if not df_b.empty:
        # Detect outliers and remove them
        df_b_clean = detect_outliers(df_b, "Right_Reach")
        
        # Original figure (Cleaned)
        plot_stim_quantification(df_b_clean, "Right Reach")
        
        # Combined figure (Cleaned)
        df_b_combined = combine_conditions(df_b_clean)
        
        # Also detect/remove outliers from combined groups if any remain relative to group stats
        df_b_combined_clean = detect_outliers(df_b_combined, "Right_Reach", suffix="_Combined")
        
        plot_stim_quantification(df_b_combined_clean, "Right Reach", suffix="_Combined", label_map=COMBINED_CONDITION_LABELS)
    else:
        print("No metrics found for Right Reach.")

if __name__ == "__main__":
    main()

