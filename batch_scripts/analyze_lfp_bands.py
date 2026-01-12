import gc
import os
import csv
import numpy as np
from scipy import signal
from scipy.io import loadmat
from scipy.optimize import curve_fit
from pathlib import Path

# SpikeInterface
from probeinterface import Probe
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
import warnings
from joblib import Parallel, delayed
from scipy.optimize import curve_fit, OptimizeWarning

import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

"""
    Analyzes LFP bands (Alpha, Beta, Low/High Gamma) for NPRW (Intan) and Utah Array (Blackrock)
    
    Pipeline:
      1. Extraction: Loads stimulation events and extracts epochs (-500ms to +1000ms)
      2. Blanking: Applies 'copy_baseline' blanking to remove stim artifacts (-5ms to +101ms)
      3. Cleaning: Bandpass filter (Reverse/Anti-causal), Resample to 1kHz, Baseline Correction,
         Exponential Decay Removal, and Common Median Template Subtraction
      4. Filtering: Extracts power in specific frequency bands
      
    Associated Scripts:
      - debug_lfp_pipeline_plots.py: Visualizes the pipeline steps (Raw -> Blanked -> Filtered, etc.)
      - plot_lfp_check.py: General quality check of aligned LFP
      - plot_lfp_artifact_comparison.py: Uses 'comparison' output to validate cleaning
    
    Output:
        Aligned data .npz files containing windowed traces for each band.
        - NPRW: results/checkpoints/NPRW_LFP/aligned_lfp__{session_name}.npz
        - Utah: results/checkpoints/UA_LFP/aligned_lfp__{session_name}.npz
        
        Debug Plots (if enabled):
        - results/figures/debug_pipeline/{session_name}/step_*.png
"""

# ---------- TOGGLES ----------
# Toggle for including comparison windows (raw vs clean) for artifact check plots
# See: plot_lfp_artifact_comparison.py
RUN_COMPARISON_ANALYSIS = False

# Toggle for saving debug plots of the pipeline steps
# See: debug_lfp_pipeline_plots.py
SAVE_DEBUG_PLOTS = False

# ---------- CONSTANTS ----------
TARGET_FS = 1000
BLANK_PRE_MS = 5.0
BLANK_POST_MS = 101.0 
EPOCH_PRE_MS = 500.0
EPOCH_POST_MS = 1000.0
FIT_START_OFFSET_MS = 0.0 # From end of blanking
BAD_THRES = 700.0 # in uV - after common median reference (last step)

# ---------- CONFIG ----------
# Bands of interest
LFP_BANDS = {
    "alpha": (5, 13),
    "beta": (13, 30),
    "low_gamma": (30, 60),
    "high_gamma": (60, 120)
}

# Filter order for butterworth
FILTER_ORDER = 4

# Output Directories
NPRW_LFP_CKPT_ROOT = OUT_BASE / "checkpoints" / "NPRW_LFP"
UA_LFP_CKPT_ROOT = OUT_BASE / "checkpoints" / "UA_LFP"

NPRW_LFP_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
UA_LFP_CKPT_ROOT.mkdir(parents=True, exist_ok=True)

# NPRW Config
NPRW_CFG = PARAMS.probes.get("NPRW")
INTAN_STREAM = NPRW_CFG.get("neural_data_stream")
STIM_STREAM = NPRW_CFG.get("stim_data_stream")
AUX_STREAM = NPRW_CFG.get("aux_stream")

# Utah Config
UA_CFG = PARAMS.probes.get("UA")
XLS = rcp.ua_excel_path(REPO_ROOT, PARAMS.probes)
UA_MAP = rcp.load_UA_mapping_from_excel(XLS) if XLS else None
METADATA_CSV = METADATA_ROOT / f"{PARAMS.session}_metadata.csv"
SHIFT_CSV = METADATA_ROOT / "br_to_intan_shifts.csv"

# Global job kwargs
global_job_kwargs = dict(n_jobs=PARAMS.parallel_jobs, chunk_duration=PARAMS.chunk)
si.set_global_job_kwargs(**global_job_kwargs)


# =============================================================================
# Core Cleaning Functions
# =============================================================================

def clean_epochs(epochs, fs, pre_samps=500):
    """
    Applies per-trial cleaning to extracted epochs:
      1. Baseline correction (subtract pre-stim mean).
      2. Exponential recovery filling & subtraction (post-blanking).
      3. Median template subtraction (common artifact removal).
      
    Args:
        epochs (np.ndarray): (n_trials, n_ch, n_time)
        fs (float): Sampling rate (Hz)
        pre_samps (int): Number of pre-stim samples for baseline.
        
    Returns:
        np.ndarray: Cleaned epochs.
    """
    n_trials, n_ch, n_time = epochs.shape
    cleaned = epochs.copy()
    
    # --- 1. Baseline Correction (Pre-Stim) ---
    if pre_samps > 0:
        baseline = np.mean(cleaned[:, :, :pre_samps], axis=2, keepdims=True)
        cleaned -= baseline
        
    # --- 2. Exponential Removal (Per-Trial) ---
    # Fit window starts after blanking (+buffer)
    fit_start_ms = BLANK_POST_MS + FIT_START_OFFSET_MS
    fit_start_idx = pre_samps + int(fit_start_ms * fs / 1000)
    
    if fit_start_idx >= n_time:
        return cleaned

    t_fit = np.arange(n_time - fit_start_idx)
    
    print(f"    [Cleaning] Fitting Exponentials (Trial-by-Trial) using Parallel Processing...")
    
    # Suppress optimization warnings during parallel fit
    warnings.simplefilter('ignore', OptimizeWarning)
    warnings.simplefilter('ignore', RuntimeWarning)

    def exp_decay(t, a, tau, c):
        return a * np.exp(-t / tau) + c

    def fit_and_clean_single(y_data, t_fit):
        # Work on the post-blanking segment
        y_seg = y_data[fit_start_idx:]
        
        # Initial guesses
        a0 = y_seg[0] - y_seg[-1]
        tau0 = len(y_seg) / 4.0
        c0 = y_seg[-1]
        
        try:
            popt, _ = curve_fit(
                exp_decay, t_fit, y_seg,
                p0=[a0, tau0, c0],
                bounds=([-np.inf, 1.0, -np.inf], [np.inf, len(y_seg)*2, np.inf]),
                maxfev=500 
            )
            fit_curve = exp_decay(t_fit, *popt)
            
            # Subtract fit
            y_clean = y_data.copy()
            y_clean[fit_start_idx:] -= fit_curve
            return y_clean
        except:
            # Fallback: Return original if fit fails
            return y_data

    # Parallelize across all (trial * channel) traces
    # Flatten to list of 1D arrays
    flat_data = cleaned.reshape(-1, n_time)
    
    results_flat = Parallel(n_jobs=-int(os.cpu_count() or 1), batch_size='auto')(
        delayed(fit_and_clean_single)(trace, t_fit) for trace in flat_data
    )
    
    # Reshape back to (n_trials, n_ch, n_time)
    cleaned = np.array(results_flat).reshape(n_trials, n_ch, n_time)

    # --- 3. Median Template Subtraction ---
    print(f"    [Cleaning] Subtracting Median Template (Post-Exp)...")
    template_median = np.median(cleaned, axis=0) # (n_ch, n_time)
    cleaned -= template_median[None, :, :]
    
    return cleaned

def filter_epochs(epochs, fs, low, high, order=4):
    """
    Apply bandpass filter to array of epochs (n_trials, n_ch, n_time) along time axis.
    """
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    sos = signal.butter(order, [low_norm, high_norm], btype='band', output='sos')
    return signal.sosfiltfilt(sos, epochs, axis=2) # Time axis is 2

# =============================================================================
# Helper Functions
# =============================================================================

def get_intan_geometry():
    GEOM_PATH = rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
    mat_probe = loadmat(Path(GEOM_PATH))
    intan_geom = {}
    intan_geom["x"] = mat_probe["xcoords"].ravel()
    intan_geom["y"] = mat_probe["ycoords"].ravel()
    
    if "chanMap0ind" in mat_probe:
        intan_probe_mapping = intan_geom["device_index_0based"] = mat_probe["chanMap0ind"].ravel()
    else:
        raise ValueError("No 0-based chanmap in .mat geometry file.")
    return intan_geom, intan_probe_mapping

# --- Custom Blanking Implementation ---
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
        
        # Search range
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
                # Handle edge case: src < 0
                if s_req_start < 0:
                     # Partial fetch not easily supported, just zero out?
                     # Or fetch what we can.
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
                    pass # Safety
                continue

            elif self._mode == 'zero':
                 traces[fill_start:fill_end] = 0.0
                 continue
            
            else:
                # Default: Linear Interpolation ('interp')
                # Determine Anchor Values (Edge of blanking)
                # If idx_L is within chunk, use it. If < 0, use trace[0] (best effort)
                
                # We need one sample BEFORE the fill start effectively for slope?
                # Actually standard linear interp uses idx_L and idx_R values as anchors.
                
                # Anchor L
                if idx_L >= 0 and idx_L < n_chunk:
                    val_L = traces[idx_L]
                else:
                    # If anchor is outside, we can't easily get it without fetching.
                    # Fallback to nearest in chunk
                    val_L = traces[0]
                    
                # Anchor R
                if idx_R >= 0 and idx_R < n_chunk:
                    val_R = traces[idx_R]
                else:
                    val_R = traces[-1]
                
                width_total = idx_R - idx_L
                if width_total == 0: continue
                
                slope = (val_R - val_L) / width_total # Shape (N_ch,)
                
                # X coordinates for the fill region
                x_vals = np.arange(fill_start, fill_end) - idx_L
                
                # Vectorized fill
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
        # Time Reversal Logic:
        # Requesting [start, end) in reversed time
        # corresponds to [N - end, N - start) in original time
        # Then flip the result.
        
        N = self._n_samples
        if start_frame is None: start_frame = 0
        if end_frame is None: end_frame = N
        
        # Original indices
        orig_start = N - end_frame
        orig_end = N - start_frame
        
        # Handle negatives/bounds if caller requests out of bounds (though SI handles bounds usually)
        # Assuming valid inputs or handling by parent
        
        traces = self._parent_segment.get_traces(orig_start, orig_end, channel_indices)
        
        # Flip along time axis (axis 0)
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

# --- Debug Plotting ---
def plot_debug_step(rec, step_name, stim_indices, pre_ms=50, post_ms=200, save_dir=None):
    """
    Plots a snippet of data around the first stimulus event for all channels.
    """
    if len(stim_indices) == 0: return
    
    if not SAVE_DEBUG_PLOTS:
        return
    
    # Use first stimulus
    stim_idx = stim_indices[0]
    fs = rec.get_sampling_frequency()
    
    start_frame = int(stim_idx - (pre_ms * fs / 1000.0))
    end_frame = int(stim_idx + (post_ms * fs / 1000.0))
    
    # Safety Check
    if start_frame < 0: start_frame = 0
    if end_frame > rec.get_num_samples(): end_frame = rec.get_num_samples()
    
    traces = rec.get_traces(start_frame=start_frame, end_frame=end_frame, return_in_uV=True)
    t_axis = (np.arange(traces.shape[0]) / fs * 1000.0) - pre_ms
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot all channels semi-transparent
    ax.plot(t_axis, traces, color='k', alpha=0.1)
    # Plot Mean
    ax.plot(t_axis, np.mean(traces, axis=1), color='r', linewidth=2, label='Mean')
    
    ax.set_title(f"Debug: {step_name}\n(FS={fs}Hz, N={traces.shape[1]}ch)")
    ax.set_xlabel("Time from Stim (ms)")
    ax.axvline(0, color='g', linestyle='--')
    ax.legend(loc='upper right')
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"step_{step_name}.png")
        print(f"    [DEBUG PLOT] Saved -> {save_dir / f'step_{step_name}.png'}")
    plt.close(fig)

def build_lfp_preprocessing_pipeline(
    recording: si.BaseRecording,
    stim_indices_native: np.ndarray,
    curr_sess_name: str = "unknown") -> si.BaseRecording:
    """
    Custom pipeline with Manual Blanking Class.
    """
    # Debug Directory
    DEBUG_DIR = OUT_BASE / "figures" / "debug_pipeline" / curr_sess_name
    
    # 0. Raw
    plot_debug_step(recording, "0_Raw", stim_indices_native, save_dir=DEBUG_DIR)
    
    # 1. Blanking (Custom) - On RAW Data (30kHz)
    # Strategy: Copy Baseline (-115ms to -5ms)
    # Use globals BLANK_PRE_MS, BLANK_POST_MS
    if len(stim_indices_native) > 0:
        print(f"    [SI Pipeline] Custom Blanking (Copy Baseline) ({len(stim_indices_native)} events, -{BLANK_PRE_MS}ms to +{BLANK_POST_MS}ms) on RAW...")
        rec_blanked = BlankingRecording(
            recording,
            stim_indices_native,
            pre_ms=BLANK_PRE_MS,
            post_ms=BLANK_POST_MS,
            mode='copy_baseline'
        )
        
        # Debug Plot: Step 1 = Blanked
        plot_debug_step(rec_blanked, "1_Blanked_30k", stim_indices_native, pre_ms=50, post_ms=200, save_dir=DEBUG_DIR)
    else:
        rec_blanked = recording
        
    # 2. Bandpass Filter (0.5 - 500 Hz) - REVERSE FILTERING
    # "Pass our data end to start through the filter and then flip it back"
    print("    [SI Pipeline] Bandpass Filtering (0.5 - 500 Hz) [REVERSE / ANTI-CAUSAL]...")
    
    # a. Reverse Time
    rec_rev = TimeReversedRecording(rec_blanked)
    
    # b. Filter (Standard Butterworth, but applied to reversed stream)
    rec_rev_filt = spre.bandpass_filter(rec_rev, freq_min=0.5, freq_max=500.0, dtype='float32')
    
    # c. Reverse Time Back (Un-flip)
    rec_filtered = TimeReversedRecording(rec_rev_filt)
    
    # Debug Plot: Step 2 = Filtered (Reverse)
    plot_debug_step(rec_filtered, "2_Filtered_30k_Rev", stim_indices_native, pre_ms=50, post_ms=200, save_dir=DEBUG_DIR)
    
    # 3. Resampling (30k -> 1k)
    print(f"    [SI Pipeline] Resampling to {TARGET_FS} Hz...")
    rec_resampled = spre.resample(rec_filtered, resample_rate=TARGET_FS)
    
    # Scale stim indices for debug plotting of resampled data
    fs_native = recording.get_sampling_frequency() # Get native FS from original recording
    stim_indices_resampled = (stim_indices_native * TARGET_FS / fs_native).astype(int)
    
    # Debug Plot: Step 3 = Resampled
    plot_debug_step(rec_resampled, "3_Resampled_1k", stim_indices_resampled, pre_ms=50, post_ms=200, save_dir=DEBUG_DIR)
    
    rec_clean = rec_resampled
    
    return rec_clean

def process_nprw_session(sess_name: str, br_idx: int, shift_ms: float = 0.0):
    """
    Process one Intan session: lazy load, SI clean/filter/DS, epoch.
    """
    # Identify session folder
    matches = list(INTAN_ROOT.glob(f"*{sess_name}*"))
    valid_folders = [p for p in matches if p.is_dir()]
    if not valid_folders:
        print(f"[NPRW] Session folder not found for {sess_name}")
        return
    sess_path = valid_folders[0] # Take first match
    
    out_npz = NPRW_LFP_CKPT_ROOT / f"aligned_lfp__{sess_name}.npz"
    if out_npz.exists():
        print(f"[NPRW] Skipping {sess_name}, output exists.")
        return
        
    print(f"[NPRW] Processing {sess_name}")

    # Geometry
    intan_geom, intan_probe_mapping = get_intan_geometry()
    
    nprw_probe = Probe(ndim=2)
    nprw_probe.set_contacts(positions=np.c_[intan_geom["x"], intan_geom["y"]], shape_params={"width": 12.0})
    nprw_probe.set_device_channel_indices(intan_probe_mapping)
    
    # 1. Load Resulting Stim Times First (Need for Blanking)
    stim_npz_dir = NPRW_AUX_DATA / f"{sess_path.name}_Intan_streams"
    stim_npz_path = stim_npz_dir / "stim_stream.npz"
    
    if stim_npz_path.exists():
        print(f"    [Stim] Loading existing stim file: {stim_npz_path.name}")
        # Use existing function or load npz directly
        try:
             stim_data = rcp.load_stim_detection(stim_npz_path)
             stim = stim_data # Assign for metadata usage later
             # stim_data is dict, block_bounds_samples is key
             block_bounds = stim_data.get("block_bounds_samples")
        except Exception as e:
             print(f"    [Stim] Error loading existing file ({e}), re-extracting...")
             stim_ext_arrays = rcp.extract_stim_npz(sess=sess_path, out_dir=NPRW_AUX_DATA, stim_stream_name=STIM_STREAM, chanmap_perm=intan_probe_mapping)
             stim = stim_ext_arrays # Assign for metadata
             block_bounds = stim_ext_arrays.get("block_bounds_samples")
    else:
         print(f"    [Stim] Extracting stim stream (first run)...")
         stim_ext_arrays = rcp.extract_stim_npz(sess=sess_path, out_dir=NPRW_AUX_DATA, stim_stream_name=STIM_STREAM, chanmap_perm=intan_probe_mapping)
         stim = stim_ext_arrays # Assign for metadata
         block_bounds = stim_ext_arrays.get("block_bounds_samples")

    stim_start_samps_native = block_bounds[:, 0] if (block_bounds is not None and block_bounds.size) else np.array([])
    
    # 2. Load Raw Recording (Lazy)
    rec = se.read_split_intan_files(sess_path, mode="concatenate", stream_name=INTAN_STREAM, use_names_as_ids=True)
    rec = spre.unsigned_to_signed(rec)
    rec_reordered = rcp.reorder_recording_to_geometry(rec, intan_probe_mapping)
    rec_reordered = rec_reordered.set_probe(nprw_probe)
    
    fs_native = rec_reordered.get_sampling_frequency()
    
    # 3. Build SI Pipeline
    # Blank(-5, +20) -> HP(0.5) -> LP(500) -> DS(1000)
    rec_proc = build_lfp_preprocessing_pipeline(
        rec_reordered, 
        stim_indices_native=stim_start_samps_native,
        curr_sess_name=sess_name
    )
    
    # 4. Get Traces (Parallel Epoch Extraction)
    # Instead of loading WHOLE recording, we extract only the epochs we need.
    print("  [NPRW] Extracting epochs directly from Lazy Recording (Parallel)...")
    
    # We need stim indices in the TARGET_FS domain
    # stim_start_samps_native are in fs_native (30000)
    # The pipeline resamples to TARGET_FS (typically 1000)
    stim_indices_res = (stim_start_samps_native * TARGET_FS / fs_native).astype(int)
    
    # Define generic parallel extractor
    def extract_single_epoch(rec_obj, center_idx, pre_samps, post_samps):
        start = center_idx - pre_samps
        end = center_idx + post_samps
        
        # Check bounds
        if start < 0 or end > rec_obj.get_num_samples():
            return None # Skip
            
        # Get trace (n_samples, n_ch) -> Transpose to (n_ch, n_samples)
        # return_in_uV=True handles scaling
        return rec_obj.get_traces(start_frame=start, end_frame=end, return_in_uV=True).T

    pre_samps = int(EPOCH_PRE_MS * TARGET_FS / 1000.0)
    post_samps = int(EPOCH_POST_MS * TARGET_FS / 1000.0)
    
    segs_list = Parallel(n_jobs=-int(os.cpu_count() or 1), batch_size='auto')(
        delayed(extract_single_epoch)(rec_proc, idx, pre_samps, post_samps) 
        for idx in stim_indices_res
    )
    
    # Filter out Nones
    segs_list = [s for s in segs_list if s is not None]
    
    if not segs_list:
        print("  [Warn] No valid epochs found.")
        return

    segs_bb = np.stack(segs_list, axis=0) # (n_trials, n_ch, n_time)
    n_trials, n_ch, n_time = segs_bb.shape
    
    # Construct rel_t_epoch
    rel_t_epoch = (np.arange(n_time) / TARGET_FS * 1000.0) - EPOCH_PRE_MS
    
    print(f"  [NPRW] Extracted {n_trials} epochs. Shape: {segs_bb.shape}")

    # 5. Get RAW Comparison Epochs (Parallel)
    traces_raw_epochs = None
    if RUN_COMPARISON_ANALYSIS:
        # We re-run the pipeline but SKIP the blanking step
        print("  [NPRW] Building RAW (Unblanked) pipeline...")
        rec_proc_raw = build_lfp_preprocessing_pipeline(
            rec_reordered, 
            stim_indices_native=np.array([]), # Skip blanking
            curr_sess_name=sess_name
        )
        
        print("  [NPRW] Extracting RAW epochs (Parallel)...")
        segs_list_raw = Parallel(n_jobs=-int(os.cpu_count() or 1), batch_size='auto')(
            delayed(extract_single_epoch)(rec_proc_raw, idx, pre_samps, post_samps) 
            for idx in stim_indices_res
        )
        # Assume same validity mask
        segs_list_raw = [s for s in segs_list_raw if s is not None]
        traces_raw_epochs = np.stack(segs_list_raw, axis=0) # (n_trials, n_ch, n_time)
    
    # Stim times to ms (just for meta)
    stim_ms = stim_start_samps_native * 1000.0 / fs_native
    
    # 1. DISABLE GLOBAL CMR per user request
    print("  [INFO] Global Common Median Reference (CMR) DISABLED.")
    
    # 6. Epoching: Extract Broadband (-500 to +1000)
    # Already done above!
    WIN_EPOCH = (-EPOCH_PRE_MS, EPOCH_POST_MS)
    
    # Sub-windows for final saving
    WIN_PRE = (-EPOCH_PRE_MS, -BLANK_PRE_MS)
    WIN_POST = (BLANK_POST_MS, EPOCH_POST_MS - BLANK_PRE_MS) # 500ms post-blank
    WIN_COMP = (-50.0, 200.0)
    
    # Valid stim times
    valid_stim = stim_ms 
    
    results = {}
    
    if len(segs_bb) == 0:
        print(f"  [WARN] No valid segments found for {sess_name}. Skipping.")
        return
        
    # 7. Per-Trial Cleaning
    print(f"  Cleaning Epochs (Baseline + Exp Removal)...")
    segs_bb_clean = clean_epochs(segs_bb, TARGET_FS)
    
    # --- Step 7.5: Reject Bad Channels (Threshold > 500 uV) ---
    # Calculate peak amplitude per trial, per channel: (N_trials, N_ch)
    peaks_per_trial = np.nanmax(np.abs(segs_bb_clean), axis=2) 
    # Get median peak across trials for each channel: (N_ch,)
    median_peak_ch = np.nanmedian(peaks_per_trial, axis=0)
    
    bad_ch_mask = median_peak_ch > BAD_THRES
    bad_ch_indices = np.flatnonzero(bad_ch_mask)
    
    if len(bad_ch_indices) > 0:
         print(f"  [Rejection] removing {len(bad_ch_indices)} channels > {BAD_THRES}uV (median peak): {bad_ch_indices}")
         # Set to NaN
         segs_bb_clean[:, bad_ch_mask, :] = np.nan
    
    results['bad_channels'] = bad_ch_indices
    
    # Helper to slice extracted epoch into sub-windows
    def slice_epoch(data, t_axis, target_win):
        # find indices in t_axis
        mask = (t_axis >= target_win[0]) & (t_axis < target_win[1])
        return data[:, :, mask], t_axis[mask]
        
    # Slice Broadband Results
    results['broadband_pre'], t_pre = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_PRE)
    results['broadband_post'], t_post = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_POST)
    
    if RUN_COMPARISON_ANALYSIS:
        results['broadband_comp'], t_comp = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_COMP)
        
        # 8. Raw Comparison (No Cleaning)
        print(f"  Segmenting Raw (Unblanked)...", flush=True)
        
        # Use already extracted raw epochs
        segs_bb_raw = traces_raw_epochs
        segs_raw = segs_bb_raw
        
        if segs_raw is not None:
             results['broadband_raw_pre'], _ = slice_epoch(segs_raw, rel_t_epoch, WIN_PRE)
             results['broadband_raw_post'], _ = slice_epoch(segs_raw, rel_t_epoch, WIN_POST)
             results['broadband_raw_comp'], _ = slice_epoch(segs_raw, rel_t_epoch, WIN_COMP)
    
        # Save Timebase info
        results['rel_time_comp'] = t_comp
    
    # 9. Band Analysis (Filter Cleaned Epochs)
    for band_name, (low, high) in LFP_BANDS.items():
        print(f"  Filtering Epochs {band_name} ({low}-{high} Hz)...", flush=True)
        # Apply filter to the CLEANED broadband epochs
        filt_epochs = filter_epochs(segs_bb_clean, TARGET_FS, low, high)
        
        # Slice
        results[f'{band_name}_pre'], _ = slice_epoch(filt_epochs, rel_t_epoch, WIN_PRE)
        results[f'{band_name}_post'], _ = slice_epoch(filt_epochs, rel_t_epoch, WIN_POST)
    
    # Save
    save_dict = {
        "fs_lfp": TARGET_FS,
        "channel_geom_x": intan_geom["x"],
        "channel_geom_y": intan_geom["y"],
        "session": sess_name,
        "stim_ms": valid_stim, # Original indices relative to full recording
        "rel_time_pre": t_pre,
        "rel_time_post": t_post,
        # Metadata
        "stim_channels": stim.get('stim_channels', []),
        "stim_amplitudes": stim.get('amplitudes', []),
        "stim_notes": stim.get('notes', ""),
        **results
    }
    
    print(f"  [DEBUG] Keys to save: {list(save_dict.keys())}", flush=True)
    
    np.savez_compressed(out_npz, **save_dict)
    print(f"  Saved -> {out_npz}")
    
    # Cleanup memory
    try:
        del rec, rec_reordered, rec_proc, results
        if 'traces' in locals(): del traces
    except:
        pass
    gc.collect()


def process_utah_session(sess_name: str, br_idx: int, shift_ms: float, fs_intan: float, shift_sample: float):
    matches = [p for p in BR_ROOT.iterdir() if p.is_dir() and f"_{br_idx:03d}" in p.name]
    
    if matches:
        sess_path = matches[0]
        # It's a directory
    else:
        # Check for files (flat structure)
        # Look for .ns files
        matches_files = [p for p in BR_ROOT.iterdir() if p.is_file() and f"_{br_idx:03d}" in p.name and p.suffix in ['.ns2', '.ns5', '.ns6']]
        if matches_files:
            ns6 = [f for f in matches_files if f.suffix == '.ns6']
            ns2 = [f for f in matches_files if f.suffix == '.ns2']
            if ns6:
                sess_path = ns6[0]
            elif ns2:
                sess_path = ns2[0]
            else:
                sess_path = matches_files[0]
        else:
            print(f"[UA] No session folder or files found for BR {br_idx:03d}")
            return

    # Construct unique identifier for output filename
    # If it's a file, sess_path.name includes extension. removing it for cleanliness in filename
    sess_id = sess_path.stem if sess_path.is_file() else sess_path.name
    
    out_npz = UA_LFP_CKPT_ROOT / f"aligned_lfp__{sess_id}.npz"
    if out_npz.exists():
        print(f"[UA] Skipping {sess_id}, output exists.")
        return
        
    print(f"[UA] Processing {sess_id} (Source: {sess_path.name})")
    
    # Load Recording
    try:
        try:
            rec_raw = se.read_blackrock(str(sess_path), stream_name='nsx2', all_annotations=True)
            if rec_raw.get_num_channels() < 64: raise ValueError("Too few channels (AUX?)")
            print("  Using NS2 (1kHz)")
        except Exception:
            rec_raw = se.read_blackrock(str(sess_path), stream_name='nsx6', all_annotations=True)
            print("  Using NS6 (30kHz)")
    except Exception as e:
        print(f"[UA] Failed to load: {e}")
        return

    # Mapping
    rec_mapped, idx_rows, ua_elec, ua_nsp, ua_region, ua_region_names, ua_port = rcp.apply_ua_mapping_with_regions(
        rec_raw, UA_MAP, br_idx, METADATA_CSV
    )
    
    # Load Stim for Blanking AND Alignment (Using Intan info mapped to UA time)
    stim_npz_path, intan_session_name = rcp.stim_npz_path_from_br_idx(br_idx, METADATA_CSV, NPRW_AUX_DATA)
    if not stim_npz_path or not stim_npz_path.exists():
        print("[UA] No stim stream found (Intan). Skipping.")
        return
        
    stim = rcp.load_stim_detection(stim_npz_path)
    block_bounds = stim.get("block_bounds_samples", [])
    
    if block_bounds.size == 0:
        print("[UA] No stim events found.")
        stim_indices_native = np.array([], dtype=int)
        stim_ms_ua = np.array([])
    else:
        # Intan Samples -> Time -> Shift -> UA Time -> UA Native Samples
        # We need triggers in *native* UA samples for processing
        intan_stim_samps = block_bounds[:, 0]
        intan_stim_ms = intan_stim_samps * 1000.0 / fs_intan
        stim_ms_ua = intan_stim_ms - shift_ms
        
        fs_native = rec_mapped.get_sampling_frequency()
        stim_indices_native = np.floor(stim_ms_ua * fs_native / 1000.0).astype(int)

    # 2. Build SI Pipeline
    # Blank(-5, +20) -> HP(0.5) -> LP(500) -> DS(1000)
    rec_proc = build_lfp_preprocessing_pipeline(
        rec_mapped, 
        stim_indices_native=stim_indices_native,
        curr_sess_name=sess_path.name
    )
    
    # 3. Get Traces (Eager Load)
    print("  [UA] Loading processed traces into memory...")
    traces = rec_proc.get_traces(return_in_uV=True).T
    n_channels, n_samples = traces.shape

    # 1. DISABLE GLOBAL CMR per user request
    # traces_unref = traces_raw.copy() 
    # Use existing trace as "raw" for flow, though it is filtered/cleaned.
    traces_unref = traces.copy()
    print("  [UA] Global Common Median Reference (CMR) DISABLED.")
    # 4. Get Traces (Raw/Unblanked) for Comparison
    traces_raw = None
    if RUN_COMPARISON_ANALYSIS:
        print("  [UA] Building RAW (Unblanked) pipeline...")
        rec_proc_raw = build_lfp_preprocessing_pipeline(
            rec_mapped, 
            stim_indices_native=np.array([]), # Skip blanking
            curr_sess_name=sess_path.name
        )
        print("  [UA] Loading RAW traces...")
        traces_raw = rec_proc_raw.get_traces(return_in_uV=True).T

        # 1. DISABLE GLOBAL CMR per user request
        print("  [UA] Global Common Median Reference (CMR) DISABLED.")

    # Segment
    t_ms = np.arange(n_samples, dtype=float) * 1000.0 / TARGET_FS
    
    # --- Parameters ---
    WIN_EPOCH = (-EPOCH_PRE_MS, EPOCH_POST_MS)   # Epoch Window
    WIN_PRE   = (-EPOCH_PRE_MS, -BLANK_PRE_MS)    # Baseline Window
    WIN_POST  = (BLANK_POST_MS, EPOCH_POST_MS - BLANK_PRE_MS)   # Analysis Window (Post-Blank)
    WIN_COMP  = (-50.0, 200.0)   # Comparison Window

    results = {}
    
    print(f"  Segmenting Broadband Epochs ({WIN_EPOCH} ms)...", flush=True)
    segs_bb, _, rel_t_epoch, valid_stim = rcp.extract_peristim_segments(
        rate_hz=traces.astype(np.float32), 
        counts=None,
        t_ms=t_ms,
        stim_ms=stim_ms_ua,
        win_ms=WIN_EPOCH
    )
    
    if valid_stim is None:
        print(f"  [WARN] No valid segments found for {sess_path.name}. Skipping.")
        return

    # 7. Per-Trial Cleaning
    print(f"  Cleaning Epochs (Baseline + Exp Removal)...")
    segs_bb_clean = clean_epochs(segs_bb, TARGET_FS)
    
    def slice_epoch(data, t_axis, target_win):
        mask = (t_axis >= target_win[0]) & (t_axis < target_win[1])
        return data[:, :, mask], t_axis[mask]
        
    results['broadband_pre'], t_pre = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_PRE)
    results['broadband_post'], t_post = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_POST)
    
    results['broadband_raw_comp'] = None 
    results['broadband_unref_comp'] = None

    if RUN_COMPARISON_ANALYSIS:
        results['broadband_comp'], t_comp = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_COMP)
        
        # 8. Raw Comparison
        print(f"  Segmenting Raw (Unblanked)...", flush=True)
        
        if traces_raw is not None:
            # valid_stim contains timestamps (ms)
            segs_raw, _, _, _ = rcp.extract_peristim_segments(
                rate_hz=traces_raw.astype(np.float32), 
                counts=None,
                t_ms=t_ms,
                stim_ms=valid_stim,
                win_ms=WIN_EPOCH
            )
            if segs_raw is not None:
                 results['broadband_raw_pre'], _ = slice_epoch(segs_raw, rel_t_epoch, WIN_PRE)
                 results['broadband_raw_post'], _ = slice_epoch(segs_raw, rel_t_epoch, WIN_POST)
                 results['broadband_raw_comp'], _ = slice_epoch(segs_raw, rel_t_epoch, WIN_COMP)
                 results['broadband_unref_comp'] = results['broadband_raw_comp']

        # Save Timebase info
        results['rel_time_comp'] = t_comp
    
    # 9. Band Analysis
    for band_name, (low, high) in LFP_BANDS.items():
        print(f"  Filtering Epochs {band_name} ({low}-{high} Hz)...", flush=True)
        filt_epochs = filter_epochs(segs_bb_clean, TARGET_FS, low, high)
        results[f'{band_name}_pre'], _ = slice_epoch(filt_epochs, rel_t_epoch, WIN_PRE)
        results[f'{band_name}_post'], _ = slice_epoch(filt_epochs, rel_t_epoch, WIN_POST)
        
    results['rel_time_comp'] = t_comp
        
    # Save
    save_dict = {
        "fs_lfp": TARGET_FS,
        "ua_region": ua_region,
        "ua_region_names": ua_region_names,
        "ua_elec": ua_elec,
        "ua_nsp": ua_nsp,
        "session": sess_path.name,
        "stim_ms": valid_stim,
        "rel_time_pre": t_pre,
        "rel_time_post": t_post,
        **results
    }
    
    np.savez_compressed(out_npz, **save_dict)
    print(f"  Saved -> {out_npz}")
    
    del rec_raw, rec_mapped, traces, results
    gc.collect()


def quick_check_stim(sess_path: Path, stream_name: str, chanmap: np.ndarray) -> bool:
    """
    Check if stim file exists and has events. 
    If not, try to extract it. If result is empty, return False.
    """
    sess_name = sess_path.name
    stim_npz = NPRW_AUX_DATA / f"{sess_name}_Intan_streams" / "stim_stream.npz"
    
    # Try loading existing fast
    if stim_npz.exists():
        try:
            print(f"    [Check] Loading existing {stim_npz.name}")
            stim_data = rcp.load_stim_detection(stim_npz)
            bounds = stim_data.get("block_bounds_samples")
            if bounds is not None and bounds.size > 0:
                print(f"    [Check] Found {len(bounds)} stim blocks.")
                return True
            else:
                print("    [Check] Existing file has 0 stim blocks.")
                return False
        except Exception as e:
            print(f"    [Check] Failed to load existing: {e}")
            
    # Fallback to extraction
    try:
        print(f"    [Check] Extracting new stim file...")
        # This function tries to load existing or create new stim npz
        stim_ext = rcp.extract_stim_npz(sess_path, out_dir=NPRW_AUX_DATA, stim_stream_name=stream_name, chanmap_perm=chanmap)
        if stim_ext is None:
            return False
        
        bounds = stim_ext.get("block_bounds_samples")
        if bounds is None or bounds.size == 0:
            return False
            
        return True
    except Exception as e:
        print(f"  [Check Stim] Failed: {e}")
        return False

def main():
    if not SHIFT_CSV.exists():
        print(f"Error: {SHIFT_CSV} not found. Cannot proceed with aligned analysis.")
        return

    with SHIFT_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    print(f"Found {len(rows)} sessions in shift CSV.")
    
    # Pre-load geometry for stim checking
    try:
        intan_geom, intan_probe_mapping = get_intan_geometry()
    except Exception as e:
        print(f"[ERROR] Could not load Intan geometry: {e}")
        return
    
    for row in rows:
        session_name = row["session"]
        br_idx = int(row["br_idx"])
        shift_ms = float(row["shift_ms"])
        fs_intan = float(row.get("fs_intan", 30000.0))
        shift_sample = float(row.get("shift_sample", 0.0))
        
        # Resolve Intan Path
        matches = list(INTAN_ROOT.glob(f"*{session_name}*"))
        valid_folders = [p for p in matches if p.is_dir()]
        
        if not valid_folders:
            print(f"[WARN] Session folder not found: {session_name}. Skipping pair.")
            continue
            
        sess_path = valid_folders[0]
        
        # --- Pre-Check Stim ---
        # We check stim BEFORE running either NPRW or UA analysis
        print(f"\nChecking stim for {session_name}...")
        has_stim = quick_check_stim(sess_path, STIM_STREAM, intan_probe_mapping)
        
        if not has_stim:
            print(f"  -> No valid stim events found. Skipping both NPRW and UA for this session.")
            continue
            
        print(f"  -> Stim OK. Proceeding...")

        try:
            process_nprw_session(session_name, br_idx)
        except Exception as e:
            print(f"[ERROR] NPRW {session_name}: {e}")
            import traceback
            traceback.print_exc()

        try:
            process_utah_session(session_name, br_idx, shift_ms, fs_intan, shift_sample)
        except Exception as e:
            print(f"[ERROR] UA {br_idx}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
