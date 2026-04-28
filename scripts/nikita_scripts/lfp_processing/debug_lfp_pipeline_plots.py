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
import warnings

"""
    Debug script to visualize LFP processing pipeline steps.
    Synced with logic in batch_scripts/analyze_lfp_bands.py.
    
    Generates step-by-step plots:
    0. Raw
    1. Blanked (interpolated artifact)
    1.5. Global CMR (Optional)
    2. Filtered (Reverse/Anti-causal)
    3. Resampled
    4. Baseline Corrected
    5. Exponential Removed
    6. Template Subtracted
    
    Usage:
        python debug_lfp_pipeline_plots.py
        (Update SESSION variable at bottom to target specific session)
"""

# ---------- Config ----------
TARGET_FS = 1000
DEBUG_ROOT = OUT_BASE / "figures" / "debug_pipeline_colored"

# Toggle for Global CMR
USE_GLOBAL_CMR = False

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
    """
    Custom SI segment that zeros out or interpolates over stir-related artifacts 
    defined by stim_indices.
    """
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
    
    traces = rec.get_traces(start_frame=start_frame, end_frame=end_frame, return_in_uV=True)
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

def run_debug_pipeline(sess_name, stim_indices_override=None, shift_ms=0.0):
    print(f"--- Running Debug Pipeline for {sess_name} ---")
    
    # 1. Locate Session
    matches = list(INTAN_ROOT.glob(f"*{sess_name}*"))
    if not matches:
        print(f"Session {sess_name} not found in {INTAN_ROOT}")
        return
    sess_path = matches[0]
    
    # 2. Geometry
    GEOM_PATH = rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
    mat_probe = loadmat(Path(GEOM_PATH))
    intan_geom = {"x": mat_probe["xcoords"].ravel(), "y": mat_probe["ycoords"].ravel()}
    intan_probe_mapping = mat_probe["chanMap0ind"].ravel()
    
    nprw_probe = Probe(ndim=2)
    nprw_probe.set_contacts(positions=np.c_[intan_geom["x"], intan_geom["y"]], shape_params={"width": 12.0})
    nprw_probe.set_device_channel_indices(intan_probe_mapping)
    
    # 3. Load Stim Times
    if stim_indices_override is not None:
        print(f"  Using provided stim indices (N={len(stim_indices_override)})")
        stim_indices = stim_indices_override
    else:
        # Standard stim loading logic
        NPRW_CFG = PARAMS.probes.get("NPRW")
        STIM_STREAM = NPRW_CFG.get("stim_data_stream")
        INTAN_STREAM = NPRW_CFG.get("neural_data_stream")
        
        # Manually check file existence
        # Fallback to reconstructing path if NPRW_AUX_DATA not available (should be from config_loading)
        aux_root = NPRW_AUX_DATA if 'NPRW_AUX_DATA' in globals() else Path("results/data/NPRW_AUX")
        
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
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 0: Raw
    plot_debug_step_colored(rec_reordered, "0_Raw", stim_indices, save_dir=SAVE_DIR)
    
    # Step 1: Blanking (on RAW 30k)
    print("Applying Blanking (Copy Baseline) on RAW...")
    rec_blank = BlankingRecording(rec_reordered, stim_indices, pre_ms=BLANK_PRE_MS, post_ms=BLANK_POST_MS, mode='copy_baseline')
    plot_debug_step_colored(rec_blank, "1_Blanked_30k", stim_indices, save_dir=SAVE_DIR)
    
    # Step 2: Local CMR (Intan)
    # Using hardcoded RADII or loaded? Loaded from config logic normally.
    NPRW_CFG = PARAMS.probes.get("NPRW")
    RADII = (NPRW_CFG.get("local_radius_inner"), NPRW_CFG.get("local_radius_outer"))
    
    print(f"Applying Local CMR (Median, radius {RADII})...")
    rec_cmr = spre.common_reference(rec_blank, reference='local', local_radius=RADII, operator='median')
    plot_debug_step_colored(rec_cmr, "2_LocalCMR_30k", stim_indices, save_dir=SAVE_DIR)
    
    # Step 3: Resample
    print("Applying Resample...")
    rec_res = spre.resample(rec_cmr, resample_rate=TARGET_FS)
    # Scale indices
    fs_native = rec_reordered.get_sampling_frequency()
    stim_res = (stim_indices * TARGET_FS / fs_native).astype(int)
    plot_debug_step_colored(rec_res, "3_Resampled_1k", stim_res, save_dir=SAVE_DIR)
    
    # ----------------------------------------------------
    # Additional Debug Steps: Epoch-based Filtering & Rejection
    # ----------------------------------------------------
    
    # Extract MULTIPLE epochs for debug
    n_debug_trials = min(20, len(stim_res))
    print(f"  Extracting {n_debug_trials} padded epochs for debugging...")
    
    pad_samps = int(1000.0 * TARGET_FS / 1000.0) # 1s padding for filtering
    pre_samps = int(EPOCH_PRE_MS * TARGET_FS / 1000.0) + pad_samps
    post_samps = int(EPOCH_POST_MS * TARGET_FS / 1000.0) + pad_samps
    
    epoch_list = []
    
    for i in range(n_debug_trials):
        s_idx = stim_res[i]
        start_samp = s_idx - pre_samps
        end_samp = s_idx + post_samps
        
        # Check bounds
        if start_samp < 0: continue
        if end_samp > rec_res.get_num_samples(): continue
        
        traces = rec_res.get_traces(start_frame=start_samp, end_frame=end_samp, return_in_uV=True).T
        epoch_list.append(traces)
        
    if not epoch_list:
        print("  [Error] No valid epochs found.")
        return

    # Stack: (n_trials, n_ch, n_time)
    epochs_pad = np.stack(epoch_list, axis=0) # Padded epochs
    print(f"  Extracted shape: {epochs_pad.shape}")
    
    # --- Step 4: Impedance Rejection ---
    # Need to load impedance file from session path
    print("Applying Impedance Rejection...")
    
    # Helper (copied from analyze_lfp_bands)
    def get_bad_channels_intan(sess_path, threshold_kohm=2000.0):
        matches = list(sess_path.glob("*impedance*.csv"))
        if not matches: return []
        imp_file = matches[0]
        bad_ch_names = []
        try:
            with open(imp_file, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                header = None
                for row in reader:
                    if row and "Channel Name" in row[0]: # Start of data
                        header = row; break
                if not header: return []
                
                imp_col = -1
                for i, h in enumerate(header):
                    if "kOhm" in h: imp_col = i; break
                if imp_col == -1: return []
                
                for row in reader:
                    if not row: continue
                    try:
                        ch_name = row[0].strip()
                        imp_val = float(row[imp_col])
                        if imp_val > threshold_kohm: bad_ch_names.append(ch_name)
                    except: continue
        except: pass
        return bad_ch_names

    bad_names = get_bad_channels_intan(sess_path, threshold_kohm=7000.0) # Matches analyze_lfp_bands default
    bad_indices = []
    
    chan_ids = rec_res.get_channel_ids()
    for name in bad_names:
        idx = np.where(chan_ids == name)[0]
        if len(idx) > 0: bad_indices.extend(idx)
        
    bad_indices = np.unique(bad_indices)
    
    epochs_rej = epochs_pad.copy()
    if len(bad_indices) > 0:
        print(f"  Masking {len(bad_indices)} bad channels")
        epochs_rej[:, bad_indices, :] = np.nan
        
    # --- Step 5: Zero-Phase Filter (Alpha Band Example) ---
    print("Applying Zero-Phase Filter (Alpha 8-12 Hz) on PADDED epochs...")
    
    # Filter function (copied)
    def filter_zero_phase(epochs, fs, low, high, order=4):
        nyq = 0.5 * fs
        low_norm = low / nyq
        high_norm = high / nyq
        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        return signal.filtfilt(b, a, epochs, axis=2)

    epochs_filt = filter_zero_phase(epochs_rej, TARGET_FS, 8, 12)
    
    # Crop Padding for Plotting
    # We want -500 to +1000 ms relative to stim
    # Padding was 1000ms. So crop 1000ms from start and end.
    
    n_time = epochs_filt.shape[2]
    # Indices relative to start of array
    # Array starts at -1500 ms (approx) relative to stim
    # We want start at -500 ms relative to stim
    # So skip 1000 ms (pad_samps)
    
    # Actually just reuse slice logic to be precise
    # Construct time axis for padded array
    t_pad = (np.arange(n_time) / TARGET_FS * 1000.0) - (EPOCH_PRE_MS + 1000.0)
    
    WIN_PLOT = (-EPOCH_PRE_MS, EPOCH_POST_MS)
    
    mask = (t_pad >= WIN_PLOT[0]) & (t_pad < WIN_PLOT[1])
    epochs_final = epochs_filt[:, :, mask]
    t_final = t_pad[mask]
    
    # Plot Snapshot (Alpha)
    plot_epoch_snapshot(epochs_final[0], "5_Alpha_8-12Hz_Clean", TARGET_FS, pre_ms=EPOCH_PRE_MS, save_dir=SAVE_DIR)

    print("Done.")
def plot_epoch_snapshot(traces, step_name, fs, pre_ms=500.0, x_lim=None, save_dir=None, title_suffix=""):
    """
    Plot helper for epoch data (n_ch, n_time)
    """
    n_ch, n_time = traces.shape
    t_axis = (np.arange(n_time) / fs * 1000.0) - pre_ms
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Group Colors
    for ch_idx in range(n_ch):
        # Skip NaNs
        if np.isnan(traces[ch_idx, :]).all():
            continue
            
        group_idx = ch_idx // 16
        color = GROUP_COLORS[group_idx % len(GROUP_COLORS)]
        ax.plot(t_axis, traces[ch_idx, :], color=color, alpha=0.6, linewidth=0.8)
        
    # Mean (ignoring NaNs)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_trace = np.nanmean(traces, axis=0) # safe mean
        
    ax.plot(t_axis, mean_trace, color='k', linewidth=2.5, linestyle='--', label='Mean')
    
    # Custom Legend (Reuse logic from main plotter)
    from matplotlib.lines import Line2D
    n_groups = (n_ch + 15) // 16
    leg_handles = [Line2D([0], [0], color=GROUP_COLORS[i % len(GROUP_COLORS)], lw=2) for i in range(n_groups)]
    leg_labels = [f"Ch {i*16+1}-{min((i+1)*16, n_ch)}" for i in range(n_groups)]
    leg_handles.append(Line2D([0], [0], color='k', linestyle='--', lw=2))
    leg_labels.append("Mean")
    ax.legend(leg_handles, leg_labels, loc='upper right', fontsize='small')
    
    ax.set_title(f"Debug: {step_name}{title_suffix}")
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

def run_baseline_debug_pipeline(npz_path):
    """
    Process baseline files by loading IR events from PeriStim baseline files
    and using them as stim_indices for the debug pipeline.
    
    Approach mirrors analyze_lfp_bands.py lines 730-766:
    1. Extract contributing sessions from aligned baseline LFP file
    2. Load IR events from corresponding PeriStim baseline file  
    3. Convert IR times to native samples
    4. Run debug pipeline on representative session with those IR events
    """
    print(f"\n--- Processing Baseline File: {npz_path.name} ---")
    
    # Load aligned baseline LFP file
    try:
        bl_lfp_data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"  [ERROR] Failed to load baseline LFP file: {e}")
        return
    
    # Extract metadata about contributing sessions
    sessions = bl_lfp_data.get('sessions', [])
    if isinstance(sessions, np.ndarray) and sessions.size > 0:
        sessions = sessions.tolist() if hasattr(sessions, 'tolist') else list(sessions)
    
    if not sessions:
        print(f"  [WARN] No session list found in baseline file. Cannot debug-plot.")
        return
    
    print(f"  Found {len(sessions)} contributing sessions: {sessions}")
    
    # Extract port and depth from filename: aligned_lfp__baseline__Depth_43_0_port_A.npz
    stem = npz_path.stem.replace("aligned_lfp__", "")
    
    # Parse port and depth
    port_str = None
    depth_str = None
    target_str = None
    
    if "port_" in stem:
        port_str = stem.split("port_")[-1].split("_")[0]  # Gets 'A' or 'B'
    
    if "Depth_" in stem:
        depth_part = stem.split("Depth_")[-1].split("_port")[0]
        depth_str = depth_part.replace("_", ".")  # Convert 43_0 to 43.0
    
    print(f"  [DEBUG] Parsed from filename: port={port_str}, depth={depth_str}")
    
    # Determine target from port (baseline files are port-specific)
    # Need to check both targets
    target_dirs = []
    if port_str:
        # Try both targets
        for target in ["A", "B"]:
            target_dirs.append((target, PERI_ROOT / f"Target_{target}"))
   
    # Load IR events from PeriStim baseline file
    ir_times_ms = None
    shift_ms = 0.0
    
    for target, target_dir in target_dirs:
        if not target_dir.exists():
            continue
        
        # Pattern: baseline__NRR_RW011_Depth_43.0_port_{port}_target_{target}.npz
        # Note: Depth uses DOTS not underscores (43.0 not 43_0)
        if depth_str and port_str:
            # Try exact pattern first
            pattern = f"baseline__*_Depth_{depth_str}_port_{port_str}_target_{target}.npz"
        else:
            pattern = f"baseline__*port_{port_str}_target_{target}.npz" if port_str else "baseline__*target_{target}.npz"
        
        print(f"  [DEBUG] Searching for pattern: {pattern} in {target_dir}")
        matches = list(target_dir.glob(pattern))
        
        if matches:
            peristim_file = matches[0]
            print(f"  Loading IR events from PeriStim file: {peristim_file.name}")
            try:
                ps_data = np.load(peristim_file, allow_pickle=True)
                if 'stim_ms' in ps_data and ps_data['stim_ms'].size > 0:
                    ir_times_ms = ps_data['stim_ms']
                    print(f"  Found {len(ir_times_ms)} IR events")
                    break
            except Exception as e:
                print(f"  [WARN] Failed to load PeriStim file: {e}")
    
    if ir_times_ms is None or len(ir_times_ms) == 0:
        print(f"  [WARN] No IR events found in PeriStim files. Cannot debug-plot.")
        return
    
    # Process first contributing session as representative
    representative_sess = sessions[0]
    print(f"  Using representative session: {representative_sess}")
    
    # Load shift from metadata if available
    shift_csv = METADATA_ROOT / "br_to_intan_shifts.csv"
    if shift_csv.exists():
        try:
            import pandas as pd
            df_shifts = pd.read_csv(shift_csv)
            # Look for matching session
            mask = df_shifts['Intan_Session'].astype(str).str.contains(representative_sess, na=False)
            if mask.any():
                shift_ms = float(df_shifts.loc[mask, 'Shift_ms'].iloc[0])
                print(f"  Using shift: {shift_ms} ms")
        except Exception as e:
            print(f"  [WARN] Failed to load shift: {e}")
    
    # Convert IR times from ms to native Intan samples (30kHz)
    # IR times are in BR time, need to add shift to get Intan time
    # Then convert ms to samples
    fs_native = 30000.0  # Intan native sampling rate
    ir_samps_native = ((ir_times_ms + shift_ms) * fs_native / 1000.0).astype(np.int64)
    
    print(f"  Converted {len(ir_samps_native)} IR times to native samples")
    
    # Run debug pipeline with IR events as stim_indices
    try:
        run_debug_pipeline(representative_sess, stim_indices_override=ir_samps_native, shift_ms=shift_ms)
    except Exception as e:
        print(f"  [ERROR] Failed to process representative session {representative_sess}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Batch run on all sessions found in checkpoints
    NPRW_LFP_CKPT_ROOT = OUT_BASE / "checkpoints" / "NPRW_LFP"
    
    npz_files = list(NPRW_LFP_CKPT_ROOT.glob("aligned_lfp__*.npz"))
    
    if not npz_files:
        print(f"No checkpoint files found in {NPRW_LFP_CKPT_ROOT}")
        sys.exit(0)
        
    print(f"Found {len(npz_files)} sessions to debug-plot.")
    
    for npz in npz_files:
        # Filename format: aligned_lfp__{session_name}.npz
        # Extract session name
        sess_name = npz.stem.replace("aligned_lfp__", "")
        
        # Handle baseline files separately
        if "baseline" in sess_name.lower():
            try:
                run_baseline_debug_pipeline(npz)
            except Exception as e:
                print(f"[ERROR] Failed to run baseline debug pipeline for {sess_name}: {e}")
            continue
        
        try:
            run_debug_pipeline(sess_name)
        except Exception as e:
            print(f"[ERROR] Failed to run debug pipeline for {sess_name}: {e}")
            continue
