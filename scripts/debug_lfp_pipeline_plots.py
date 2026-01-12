import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.io import loadmat

# SpikeInterface
from probeinterface import Probe
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se

# Add repo root to path if needed (though env var usually handles it)
# Assuming run from repo root or e:/NHP_...
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

# ---------- Config ----------
TARGET_FS = 1000
DEBUG_ROOT = OUT_BASE / "figures" / "debug_pipeline_colored"

# --- Constants ---
BLANK_PRE_MS = 5.0
BLANK_POST_MS = 101.0
EPOCH_PRE_MS = 500.0
EPOCH_POST_MS = 1000.0
FIT_START_OFFSET_MS = 0.0 # From end of blanking

# Colors for 16-channel groups
GROUP_COLORS = [
    '#1f77b4', # Blue (1-16)
    '#ff7f0e', # Orange (17-32)
    '#2ca02c', # Green (33-48)
    '#d62728', # Red (49-64)
    '#9467bd', # Purple (65-80)
    '#8c564b', # Brown (81-96)
    '#e377c2', # Pink (97-112)
    '#7f7f7f', # Gray (113-128)
]

# --- copy of BlankingRecording from analyze_lfp_bands.py ---
# --- copy of BlankingRecording from analyze_lfp_bands.py ---
class BlankingRecording(si.BaseRecording):
    def __init__(self, parent_recording, stim_indices, pre_ms=5.0, post_ms=20.0, mode='interp'):
        si.BaseRecording.__init__(self, parent_recording.get_sampling_frequency(), 
                               parent_recording.get_channel_ids(), 
                               parent_recording.get_dtype())
        
        # Copy properties to self (channel gains/offsets)
        parent_recording.copy_metadata(self)
        
        self._parent = parent_recording
        self._stim_indices = np.sort(stim_indices) # Ensure sorted
        self._pre_samples = int(pre_ms * self.get_sampling_frequency() / 1000.0)
        self._post_samples = int(post_ms * self.get_sampling_frequency() / 1000.0)
        self._mode = mode
        
        # Add segments
        for i in range(parent_recording.get_num_segments()):
            parent_segment = parent_recording._recording_segments[i]
            self.add_recording_segment(BlankingRecordingSegment(
                parent_segment, self._stim_indices, self._pre_samples, self._post_samples, mode=mode
            ))
            
        self._kwargs = {'parent_recording': parent_recording.to_dict(), 'stim_indices': stim_indices, 
                        'pre_ms': pre_ms, 'post_ms': post_ms, 'mode': mode}

class BlankingRecordingSegment(si.BaseRecordingSegment):
    def __init__(self, parent_segment, stim_indices, pre_samples, post_samples, mode='interp'):
        si.BaseRecordingSegment.__init__(self, **parent_segment.get_times_kwargs())
        self._parent_segment = parent_segment
        self._stim_indices = stim_indices
        self._pre = pre_samples
        self._post = post_samples
        self._mode = mode
        self._n_samples = parent_segment.get_num_samples() # Cache
        
    def get_num_samples(self):
        return self._parent_segment.get_num_samples()
        
    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self._parent_segment.get_traces(start_frame, end_frame, channel_indices)
        traces = traces.copy() # Make writable
        
        if len(self._stim_indices) == 0:
            return traces
            
        # We need to consider blanking window
        pad = self._pre + self._post
        i_start = np.searchsorted(self._stim_indices, start_frame - pad) 
        i_end = np.searchsorted(self._stim_indices, end_frame + pad)
        
        relevant = self._stim_indices[i_start:i_end]
        n_chunk = traces.shape[0]
        
        blank_duration = self._pre + self._post
        
        for t in relevant:
            # t is absolute sample of stim
            rel_t = int(t - start_frame)
            
            # Blanking Window Indices (Relative to chunk start)
            idx_L = rel_t - self._pre
            idx_R = rel_t + self._post
            
            # Determine overlap with current chunk
            fill_start = max(0, idx_L)
            fill_end = min(n_chunk, idx_R)
            
            if fill_start >= fill_end:
                 continue
                 
            # --- Logic per Mode ---
            if self._mode == 'copy_baseline':
                # Source Window (Absolute)
                # Ideally: [blank_start - duration, blank_start]
                abs_blank_start = t - self._pre
                abs_src_start = abs_blank_start - blank_duration
                abs_src_end = abs_blank_start
                
                # We need to map the 'fill' region (which is a subset of blanking window)
                # to the valid source region.
                # fill_start is relative index in chunk.
                # offset from blank_start = fill_start - idx_L
                offset = fill_start - idx_L
                length = fill_end - fill_start
                
                s_req_start = abs_src_start + offset
                s_req_end = s_req_start + length
                
                # Fetch baseline data (Source)
                # Handle edge case: src < 0 or > N
                n_total = self.get_num_samples()
                
                if s_req_start < 0:
                     # Partial fetch not easily supported, just zero out?
                     if s_req_end <= 0:
                          traces[fill_start:fill_end] = 0
                          continue
                     else:
                          # Split zero and valid
                          valid_len = s_req_end
                          traces[fill_start:fill_start + (length-valid_len)] = 0
                          try:
                              bsl = self._parent_segment.get_traces(0, s_req_end, channel_indices)
                              traces[fill_end-valid_len:fill_end] = bsl
                          except:
                              pass
                          continue

                try:
                    baseline = self._parent_segment.get_traces(s_req_start, s_req_end, channel_indices)
                    if baseline.shape[0] == length:
                         traces[fill_start:fill_end] = baseline
                except Exception:
                    pass 
                continue

            elif self._mode == 'zero':
                 traces[fill_start:fill_end] = 0.0
                 continue
            
            else:
                # Default: Linear Interpolation ('interp')
                # Start/End of blanking window (relative)
                # idx_L = rel_t - self._pre
                # idx_R = rel_t + self._post
                
                # Anchor L
                if idx_L >= 0 and idx_L < n_chunk:
                    val_L = traces[idx_L]
                else:
                    val_L = traces[0]
                    
                # Anchor R
                if idx_R >= 0 and idx_R < n_chunk:
                    val_R = traces[idx_R]
                else:
                    val_R = traces[-1]
                    
                width_total = idx_R - idx_L
                if width_total == 0: continue
                
                slope = (val_R - val_L) / width_total # Shape (N_ch,)
                x_vals = np.arange(fill_start, fill_end) - idx_L
                
                traces[fill_start:fill_end, :] = val_L[None, :] + x_vals[:, None] * slope[None, :]
            
        return traces

# --- Time Reversal Wrapper for Backward Filtering ---
class TimeReversedRecordingSegment(si.BaseRecordingSegment):
    def __init__(self, parent_segment):
        si.BaseRecordingSegment.__init__(self, **parent_segment.get_times_kwargs())
        self._parent_segment = parent_segment
        self._n_samples = parent_segment.get_num_samples()

    def get_num_samples(self):
        return self._n_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
        # Time Reversal Logic
        N = self._n_samples
        if start_frame is None: start_frame = 0
        if end_frame is None: end_frame = N
        
        orig_start = N - end_frame
        orig_end = N - start_frame
        
        traces = self._parent_segment.get_traces(orig_start, orig_end, channel_indices)
        return traces[::-1, :]

class TimeReversedRecording(si.BaseRecording):
    def __init__(self, parent_recording):
        si.BaseRecording.__init__(self, parent_recording.get_sampling_frequency(), 
                               parent_recording.get_channel_ids(), 
                               parent_recording.get_dtype())
        
        parent_recording.copy_metadata(self)
        self._parent = parent_recording
        
        for i in range(parent_recording.get_num_segments()):
            self.add_recording_segment(TimeReversedRecordingSegment(
                parent_recording._recording_segments[i]
            ))
            
        self._kwargs = {'parent_recording': parent_recording.to_dict()}

# --- Plotting Function ---
def plot_debug_step_colored(rec, step_name, stim_indices, pre_ms=50, post_ms=200, save_dir=None):
    """
    Plots a snippet of data around the first stimulus event with channel coloring.
    Channels 1-16 -> Color 1, 17-32 -> Color 2, etc.
    """
    if len(stim_indices) == 0:
        print("  [WARN] No stim indices for plot.")
        return
    
    # Use first stimulus
    stim_idx = stim_indices[0]
    fs = rec.get_sampling_frequency()
    n_ch = rec.get_num_channels()
    
    start_frame = int(stim_idx - (pre_ms * fs / 1000.0))
    end_frame = int(stim_idx + (post_ms * fs / 1000.0))
    
    # Safety Check
    if start_frame < 0: start_frame = 0
    if end_frame > rec.get_num_samples(): end_frame = rec.get_num_samples()
    
    traces = rec.get_traces(start_frame=start_frame, end_frame=end_frame, return_scaled=True)
    t_axis = (np.arange(traces.shape[0]) / fs * 1000.0) - pre_ms
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for ch_idx in range(n_ch):
        group_idx = ch_idx // 16
        color = GROUP_COLORS[group_idx % len(GROUP_COLORS)]
        
        ax.plot(t_axis, traces[:, ch_idx], color=color, alpha=0.6, linewidth=0.8)

    # Plot Mean
    ax.plot(t_axis, np.mean(traces, axis=1), color='k', linewidth=2.5, linestyle='--', label='Mean')
    
    # Custom Legend
    from matplotlib.lines import Line2D
    n_groups = (n_ch + 15) // 16
    leg_handles = [Line2D([0], [0], color=GROUP_COLORS[i % len(GROUP_COLORS)], lw=2) for i in range(n_groups)]
    leg_labels = [f"Ch {i*16+1}-{min((i+1)*16, n_ch)}" for i in range(n_groups)]
    leg_handles.append(Line2D([0], [0], color='k', linestyle='--', lw=2))
    leg_labels.append("Mean")
    
    ax.legend(leg_handles, leg_labels, loc='upper right', fontsize='small')
    
    ax.set_title(f"Debug: {step_name}\n(FS={fs}Hz, N={n_ch}ch)")
    ax.set_xlabel("Time from Stim (ms)")
    ax.set_ylabel("uV")
    ax.axvline(0, color='k', linestyle=':', alpha=0.5)
    
    ax.axvspan(-5, 120, color='gray', alpha=0.1, label='Blank Region')
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"step_{step_name}.png"
        plt.savefig(out_path)
        print(f"    [DEBUG PLOT] Saved -> {out_path}")
    plt.close(fig)

def indent_colors():
    return GROUP_COLORS

def run_debug_pipeline(sess_name):
    # ... (skipping unchanged parts) ...
    print(f"--- Running Debug Pipeline for {sess_name} ---")
    
    # 1. Locate Session
    matches = list(INTAN_ROOT.glob(f"*{sess_name}*"))
    if not matches:
        print(f"Session {sess_name} not found in {INTAN_ROOT}")
        return
    sess_path = matches[0]
    
    # ... (skipping geometry/stim setup) ...
    # Re-implementing simplified version to match context for replacement
    
    # Assume previous setup lines exist. We target the pipeline logic block.

# (Wait, I need to check line numbers. run_debug_pipeline starts at 259.
# I will just replace the "Step 2" block in run_debug_pipeline manually via MultiReplace or a new Replace call further down.
# This Replace call is for the classes.)

# Let's just add the classes first. 
# Current view showed plot_debug_step_colored at line 186.
# I will insert the classes BEFORE plot_debug_step_colored.


def indent_colors():
    return GROUP_COLORS

def run_debug_pipeline(sess_name):
    print(f"--- Running Debug Pipeline for {sess_name} ---")
    
    # 1. Locate Session
    matches = list(INTAN_ROOT.glob(f"*{sess_name}*"))
    if not matches:
        print(f"Session {sess_name} not found in {INTAN_ROOT}")
        return
    sess_path = matches[0]
    
    # 2. Geometry
    # Need to load geometry for mapping (reused from analyze_lfp_bands)
    GEOM_PATH = rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
    mat_probe = loadmat(Path(GEOM_PATH))
    intan_geom = {"x": mat_probe["xcoords"].ravel(), "y": mat_probe["ycoords"].ravel()}
    intan_probe_mapping = mat_probe["chanMap0ind"].ravel()
    
    nprw_probe = Probe(ndim=2)
    nprw_probe.set_contacts(positions=np.c_[intan_geom["x"], intan_geom["y"]], shape_params={"width": 12.0})
    nprw_probe.set_device_channel_indices(intan_probe_mapping)
    
    # 3. Load Stim Times
    NPRW_CFG = PARAMS.probes.get("NPRW")
    STIM_STREAM = NPRW_CFG.get("stim_data_stream")
    INTAN_STREAM = NPRW_CFG.get("neural_data_stream")
    
    # We assume stim is extracted already, or we extract it
    # Just load from aux_data if possible
    
    # Try loading existing fast to avoid re-extraction race condition
    # NPRW_AUX_DATA is imported from config_loading
    if 'NPRW_AUX_DATA' in globals():
         aux_root = NPRW_AUX_DATA
    else:
         # Fallback if somehow not imported
         # Reconstruct from OUT_BASE if possible, or just default
         aux_root = Path("results/data/NPRW_AUX") # Best effort
         
    # Better approach: Manually check file existence
    stim_npz_path = aux_root / f"{sess_name}_Intan_streams" / "stim_stream.npz"
    if stim_npz_path.exists():
         print(f"  Loading existing stim: {stim_npz_path}")
         try:
             stim_data = rcp.load_stim_detection(stim_npz_path)
             block_bounds = stim_data.get("block_bounds_samples")
         except Exception as e:
             print(f"  Error loading existing stim: {e}. Re-extracting.")
             stim_ext_arrays = rcp.extract_stim_npz(sess=sess_path, out_dir=aux_root, stim_stream_name=STIM_STREAM, chanmap_perm=intan_probe_mapping)
             block_bounds = stim_ext_arrays.get("block_bounds_samples")
    else:
         print(f"  Extracting stim (file not found)...")
         stim_ext_arrays = rcp.extract_stim_npz(sess=sess_path, out_dir=aux_root, stim_stream_name=STIM_STREAM, chanmap_perm=intan_probe_mapping)
         block_bounds = stim_ext_arrays.get("block_bounds_samples")
         
    stim_indices = block_bounds[:, 0] if (block_bounds is not None and block_bounds.size) else np.array([])
    
    if len(stim_indices) == 0:
        print("No stim indices found.")
        return

    # 4. Load Recording
    rec = se.read_split_intan_files(sess_path, mode="concatenate", stream_name=INTAN_STREAM, use_names_as_ids=True)
    rec = spre.unsigned_to_signed(rec)
    rec_reordered = rcp.reorder_recording_to_geometry(rec, intan_probe_mapping)
    rec_reordered = rec_reordered.set_probe(nprw_probe)
    
    # 5. Run Pipeline Steps & Plot
    SAVE_DIR = DEBUG_ROOT / sess_name
    
    # Step 0: Raw
    plot_debug_step_colored(rec_reordered, "0_Raw", stim_indices, save_dir=SAVE_DIR)
    
    # Step 1: Blanking (on RAW 30k)
    print("Applying Blanking (Copy Baseline) on RAW...")
    # Step 1: Blanking (on RAW 30k)
    print("Applying Blanking (Copy Baseline) on RAW...")
    rec_blank = BlankingRecording(rec_reordered, stim_indices, pre_ms=BLANK_PRE_MS, post_ms=BLANK_POST_MS, mode='copy_baseline')
    plot_debug_step_colored(rec_blank, "1_Blanked_30k", stim_indices, save_dir=SAVE_DIR)
    
    # Step 2: Bandpass Filter (0.5 - 500 Hz) - REVERSE FILTERING
    print("Applying Bandpass Filter (0.5 - 500 Hz) [REVERSE / ANTI-CAUSAL]...")
    
    # a. Reverse Time
    rec_rev = TimeReversedRecording(rec_blank)
    
    # b. Filter (Standard Butterworth, but applied to reversed stream)
    rec_rev_filt = spre.bandpass_filter(rec_rev, freq_min=0.5, freq_max=500.0, dtype='float32')
    
    # c. Reverse Time Back (Un-flip)
    rec_filt = TimeReversedRecording(rec_rev_filt)
    
    plot_debug_step_colored(rec_filt, "2_Filtered_30k_Rev", stim_indices, save_dir=SAVE_DIR)
    
    # Step 3: Resample
    print("Applying Resample...")
    rec_res = spre.resample(rec_filt, resample_rate=TARGET_FS)
    # Scale indices
    fs_native = rec_reordered.get_sampling_frequency()
    stim_res = (stim_indices * TARGET_FS / fs_native).astype(int)
    plot_debug_step_colored(rec_res, "3_Resampled_1k", stim_res, save_dir=SAVE_DIR)
    
    # ----------------------------------------------------
    # Additional Debug Steps: Epoch-based Cleaning & Filtering
    # ----------------------------------------------------
    
    # We need to extract an epoch around the FIRST stimulus manually to mimic the batch script
    # Window: -500ms to +1000ms
    
    # Extract MULTIPLE epochs (e.g., 20) for template averaging
    n_debug_trials = min(20, len(stim_res))
    print(f"  Extracting {n_debug_trials} epochs for debugging...")
    
    epoch_list = []
    
    # Pre-calc samples
    pre_samps = int(EPOCH_PRE_MS * TARGET_FS / 1000.0)
    post_samps = int(EPOCH_POST_MS * TARGET_FS / 1000.0)
    tot_samps = pre_samps + post_samps
    
    for i in range(n_debug_trials):
        s_idx = stim_res[i]
        start_samp = s_idx - pre_samps
        end_samp = s_idx + post_samps
        
        # Check bounds
        if start_samp < 0: continue
        if end_samp > rec_res.get_num_samples(): continue
        
        traces = rec_res.get_traces(start_frame=start_samp, end_frame=end_samp, return_scaled=True).T
        epoch_list.append(traces)
        
    if not epoch_list:
        print("  [Error] No valid epochs found.")
        return

    # Stack: (n_trials, n_ch, n_time)
    epochs_arr = np.stack(epoch_list, axis=0)
    
    # --- Step 4: Baseline Correction ---
    # Subtract pre-stim mean (-500 to -5ms)
    # pre_samps is index of 0ms (stim). Logic: :pre_samps excludes 0.
    
    print("Applying Baseline Correction...")
    epochs_bc = epochs_arr.copy()
    if pre_samps > 0:
        base = np.mean(epochs_bc[:, :, :pre_samps], axis=2, keepdims=True)
        epochs_bc -= base

    # Plot Step 4 (Snapshot of Trial 0)
    plot_epoch_snapshot(epochs_bc[0], "4_BaselineCorrected", TARGET_FS, pre_ms=EPOCH_PRE_MS, save_dir=SAVE_DIR)
        
    # --- Step 5: Exponential Removal (PER TRIAL) ---
    print("Applying Exponential Removal (Trial-by-Trial)...")
    
    fit_start_idx = pre_samps + int((BLANK_POST_MS + FIT_START_OFFSET_MS) * TARGET_FS / 1000.0)
    
    def exp_decay(t, a, tau, c):
        return a * np.exp(-t / tau) + c
        
    from scipy.optimize import curve_fit, OptimizeWarning
    import warnings
    warnings.simplefilter('ignore', OptimizeWarning)
    
    epochs_exp_clean = epochs_bc.copy()
    n_trials_debug = epochs_exp_clean.shape[0]
    n_ch = epochs_exp_clean.shape[1]
    n_time = epochs_exp_clean.shape[2]
    
    t_fit = np.arange(n_time - fit_start_idx)
    
    # Fit each trial and channel
    for i in range(n_trials_debug):
        for c in range(n_ch):
            y_seg = epochs_exp_clean[i, c, fit_start_idx:]
            
            a0 = y_seg[0] - y_seg[-1]
            tau0 = len(y_seg) / 4.0
            c0 = y_seg[-1]
            try:
                popt, _ = curve_fit(exp_decay, t_fit, y_seg, p0=[a0, tau0, c0], maxfev=1000)
                fit_curve = exp_decay(t_fit, *popt)
                epochs_exp_clean[i, c, fit_start_idx:] -= fit_curve
            except:
                pass
            
    plot_epoch_snapshot(epochs_exp_clean[0], "5_ExpRemoved_PerTrial", TARGET_FS, pre_ms=EPOCH_PRE_MS, save_dir=SAVE_DIR)
    
    # --- Step 6: Median Template Subtraction ---
    print("Applying Median Template Subtraction (Post-Exp)...")
    
    # 1. Compute Median Template (per channel, across trials) on EXP-CLEANED data
    template_median = np.median(epochs_exp_clean, axis=0) # (n_ch, n_time)
    
    # 2. Subtract Template from all trials
    epochs_clean = epochs_exp_clean - template_median[None, :, :]
    
    # Plot Step 6
    plot_epoch_snapshot(epochs_clean[0], "6_MedianTemplateSubtracted", TARGET_FS, pre_ms=EPOCH_PRE_MS, save_dir=SAVE_DIR)
    
    # --- PLOT TRANSIENT PROFILES (The Template) ---
    print("Plotting Transient Profiles (Templates)...")
    plot_transient_profiles(template_median, "6b_TransientProfiles_PostExp", TARGET_FS, pre_ms=EPOCH_PRE_MS, save_dir=SAVE_DIR)
    
    # --- Step 7: Alpha Band (5-13 Hz) ---
    print("Applying Alpha Filter (5-13 Hz)...")
    sos = signal.butter(4, [5.0, 13.0], btype='band', fs=TARGET_FS, output='sos')
    epochs_alpha = signal.sosfiltfilt(sos, epochs_clean, axis=2)
    
    # Plot Pre-Stim Focus
    plot_epoch_snapshot(epochs_alpha[0], "7a_Alpha_PreStim", TARGET_FS, pre_ms=EPOCH_PRE_MS, x_lim=(-500, -5), save_dir=SAVE_DIR)
    
    # Plot Post-Stim Focus
    plot_epoch_snapshot(epochs_alpha[0], "7b_Alpha_PostStim", TARGET_FS, pre_ms=EPOCH_PRE_MS, x_lim=(BLANK_POST_MS, 600), save_dir=SAVE_DIR)

    print("Done.")

def plot_epoch_snapshot(traces, step_name, fs, pre_ms=500.0, x_lim=None, save_dir=None):
    """
    Plot helper for epoch data (n_ch, n_time)
    """
    n_ch, n_time = traces.shape
    t_axis = (np.arange(n_time) / fs * 1000.0) - pre_ms
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Group Colors
    for ch_idx in range(n_ch):
        group_idx = ch_idx // 16
        color = GROUP_COLORS[group_idx % len(GROUP_COLORS)]
        ax.plot(t_axis, traces[ch_idx, :], color=color, alpha=0.6, linewidth=0.8)
        
    ax.plot(t_axis, np.mean(traces, axis=0), color='k', linewidth=2.5, linestyle='--', label='Mean')
    
    # Custom Legend (Reuse logic from main plotter)
    from matplotlib.lines import Line2D
    n_groups = (n_ch + 15) // 16
    leg_handles = [Line2D([0], [0], color=GROUP_COLORS[i % len(GROUP_COLORS)], lw=2) for i in range(n_groups)]
    leg_labels = [f"Ch {i*16+1}-{min((i+1)*16, n_ch)}" for i in range(n_groups)]
    leg_handles.append(Line2D([0], [0], color='k', linestyle='--', lw=2))
    leg_labels.append("Mean")
    ax.legend(leg_handles, leg_labels, loc='upper right', fontsize='small')
    
    ax.set_title(f"Debug: {step_name}")
    ax.set_xlabel("Time from Stim (ms)")
    ax.set_ylabel("uV")
    ax.axvline(0, color='k', linestyle=':', alpha=0.5)
    ax.axvspan(-5, 105, color='gray', alpha=0.1, label='Blank Region')
    
    if x_lim:
        ax.set_xlim(x_lim)
        
    if save_dir:
        out_path = Path(save_dir) / f"step_{step_name}.png"
        plt.savefig(out_path)
        print(f"    [DEBUG PLOT] Saved -> {out_path}")
    plt.close(fig)

def plot_transient_profiles(template_data, step_name, fs, pre_ms=500.0, save_dir=None):
    """
    Plots the median template traces for all channels overlaid.
    template_data: (n_ch, n_time)
    """
    n_ch, n_time = template_data.shape
    t_axis = (np.arange(n_time) / fs * 1000.0) - pre_ms
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot all channels with slight transparency
    for c in range(n_ch):
        group_idx = c // 16
        color = GROUP_COLORS[group_idx % len(GROUP_COLORS)]
        ax.plot(t_axis, template_data[c, :], color=color, alpha=0.5, linewidth=1.0)
        
    # Plot Mean of Medians (Global Average Artifact)
    ax.plot(t_axis, np.mean(template_data, axis=0), color='k', linewidth=3.0, linestyle='--', label='Mean of Templates')
    
    ax.set_title(f"Transient Profiles (Median Templates)\n(N={n_ch} channels)")
    ax.set_xlabel("Time from Stim (ms)")
    ax.set_ylabel("uV")
    ax.axvline(0, color='k', linestyle=':', alpha=0.5)
    ax.axvspan(-5, BLANK_POST_MS, color='gray', alpha=0.1, label='Blank Region')
    
    # Zoom in on post-stim
    ax.set_xlim(-50, 600)
    
    ax.legend(loc='upper right', fontsize='small')
    
    if save_dir:
        out_path = Path(save_dir) / f"{step_name}.png"
        plt.savefig(out_path)
        print(f"    [DEBUG PLOT] Saved -> {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    # Test on the session being processed
    SESSION = "NRR_RW011_251203_140253"
    run_debug_pipeline(SESSION)
