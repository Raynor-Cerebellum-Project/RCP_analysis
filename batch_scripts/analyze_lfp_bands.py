import gc
import os
import csv
import re
import numpy as np
from scipy import signal
from scipy.io import loadmat
from scipy.optimize import curve_fit
from pathlib import Path
import pandas as pd

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
from RCP_analysis.python.functions.artifact_correction import IPCA_Artifact_Correction, Template

# ---------- BASELINE TRIAL EXCLUSION ----------

from spikeinterface.core import BaseRecording, BaseRecordingSegment

class IPCACorrectedRecordingSegment(BaseRecordingSegment):
    def __init__(self, parent_recording_segment, micro_map, micro_corrected):
        """
        micro_map: list of tuples (p_start, p_end) indicating where artifacts are
        micro_corrected: np.ndarray of shape (n_events, n_time, n_channels) holding the IPCA patches
        """
        BaseRecordingSegment.__init__(self, **parent_recording_segment.get_times_kwargs())
        self.parent_recording_segment = parent_recording_segment
        self.micro_map = micro_map
        self.micro_corrected = micro_corrected

    def get_num_samples(self):
        return self.parent_recording_segment.get_num_samples()

    def get_traces(self, start_frame, end_frame, channel_indices):
        # 1. Ask the parent for the clean disk-backed chunk
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        
        # We need a writeable copy to patch
        traces = traces.copy() 
        
        if self.micro_map is None or len(self.micro_map) == 0:
            return traces
            
        # 2. Find any artifacts that intersect with our current [start_frame, end_frame] chunk
        for p, (p_start, p_end) in enumerate(self.micro_map):
            # Check for overlap
            overlap_start = max(start_frame, p_start)
            overlap_end = min(end_frame, p_end)
            
            if overlap_start < overlap_end:
                # Calculate indices relative to our current trace chunk
                chunk_idx_start = overlap_start - start_frame
                chunk_idx_end = overlap_end - start_frame
                
                # Calculate indices relative to the pre-computed artifact patch
                patch_idx_start = overlap_start - p_start
                patch_idx_end = overlap_end - p_start
                
                # Fetch full patch for this event and time window: shape (time, all_channels)
                patch_full = self.micro_corrected[p, patch_idx_start:patch_idx_end, :]
                
                # Slice the micro_corrected patch (only for the requested channels)
                if channel_indices is None:
                    # All channels
                    patch = patch_full
                else:
                    # Specific channels
                    patch = patch_full[:, channel_indices]
                
                # Overwrite the disk trace with the IPCA-corrected data
                # traces is (time, channels)
                traces[chunk_idx_start:chunk_idx_end, :] = patch
                
        return traces

class IPCACorrectedRecording(BaseRecording):
    """
    A SpikeInterface BaseRecording that lazily streams disk data 
    and surgically inserts pre-computed IPCA artifact patches on the fly.
    """
    def __init__(self, parent_recording, micro_map, micro_corrected):
        BaseRecording.__init__(self, 
                               parent_recording.get_sampling_frequency(), 
                               parent_recording.channel_ids, 
                               parent_recording.get_dtype())
        self.parent_recording = parent_recording
        
        # Copy properties and annotations
        parent_recording.copy_metadata(self)
        
        # Create a matching segment
        for segment_index in range(parent_recording.get_num_segments()):
            parent_segment = parent_recording._recording_segments[segment_index]
            self.add_recording_segment(IPCACorrectedRecordingSegment(parent_segment, micro_map, micro_corrected))


# Format: "baseline_port{A,B}_target{A,B}": [list of 0-based trial indices to skip]
# Use this to remove false positive IR-triggered events (e.g., multiple triggers on same reach)
EXCLUDE_BASELINE_TRIALS = {
    # "baseline_portA_targetA": [1, 3, 4, 6, 13, 14, 21, 24],
    # "baseline_portB_targetA": [2, 6, 7, 9, 13, 18],
    # "baseline_portA_targetB": [6, 10, 13, 15, 16, 17, 20, 21, 22, 24, 25, 30, 39, 40, 42, 47, 56, 57],
    # "baseline_portB_targetB": [2, 3, 4, 6, 8, 10, 12, 15, 17, 20, 22, 23, 24, 26, 28, 30, 31, 35],
}

"""
    Analyzes LFP bands (Alpha, Beta, Low/High Gamma) for NPRW (Intan) and Utah Array (Blackrock)
    
    Pipeline:
      1. Extraction: Loads stimulation events and extracts epochs (-500ms to +1000ms) with padding.
      2. Blanking: Applies 'copy_baseline' blanking to remove stim artifacts (-5ms to +101ms).
      3. Local CMR (Intan): Subtracts common median within local radius (loaded from config). 
      4. Utah CMR: DISABLED.
      5. Filtering: Zero-phase bandpass filtering in specific frequency bands (using padded epochs).
      6. Cleaning/Rejection: 
         - No baseline correction or exponential/template subtraction.
         - Bad channels rejected based on IMPEDANCE thresholds (Intan > 7000 kOhm, Utah > 1000 kOhm).
      
    Associated Scripts:
      - debug_lfp_pipeline_plots.py: Visualizes the pipeline steps.
      - plot_lfp_check.py: General quality check of aligned LFP.
    
    Output:
        Aligned data .npz files containing windowed traces for each band.
        - NPRW: results/checkpoints/NPRW_LFP/aligned_lfp__{session_name}.npz
        - Utah: results/checkpoints/UA_LFP/aligned_lfp__{session_name}.npz
"""

# ---------- CONSTANTS ----------
TARGET_FS = 1000
BLANK_PRE_MS = 5.0
BLANK_POST_MS = 101.0 
EPOCH_PRE_MS = 1000.0
EPOCH_POST_MS = 1000.0
PAD_MS = 1000.0 # Padding for zero-phase filtering
PLOT_1F_DEBUG = True # Optional flag to save figure comparing 1/f correction

# --- IPCA Artifact Correction Settings ---
USE_IPCA_CORRECTION  = True
IPCA_RANK            = 10
IPCA_PULSE_WINDOW_MS = (-0.2, 0.3)
ARTRMV_MS_BEFORE = 5.0
ARTRMV_TAIL_MS   = 5.0

# ---------- CONFIG ----------
# Bands of interest
LFP_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 25),
    "low_gamma": (25, 60),
    "high_gamma": (60, 120)
}

# Filter order for butterworth
FILTER_ORDER = 4

# Output Directories
UA_LFP_CKPT_ROOT = OUT_BASE / "checkpoints" / "UA_LFP"
ALIGNED_CKPT_ROOT = OUT_BASE / "checkpoints" / "Aligned"
PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim"
UA_LFP_CKPT_ROOT.mkdir(parents=True, exist_ok=True)

# Stim / Intan Settings mapping
NPRW_CFG = PARAMS.probes.get("NPRW")
STIM_STREAM = NPRW_CFG.get("stim_data_stream")

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
# Metadata Helpers (for Baseline Grouping)
# =============================================================================

def _normalize_metadata(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase with underscores."""
    df = df_raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _find_baseline_groups(df_norm: pd.DataFrame) -> dict:
    """
    Return mapping (ua_port_norm, depth_mm_norm) -> row indices of baseline rows.
    Mirrors the grouping logic from extract_peri_IR_and_concat_baseline.py.
    """
    if "notes" not in df_norm.columns:
        return {}

    is_baseline = df_norm["notes"].astype(str).str.lower().eq("baseline")
    df_base = df_norm.loc[is_baseline].copy()

    if df_base.empty:
        return {}

    # UA_port
    if "ua_port" in df_base.columns:
        df_base["_ua_port_norm"] = (
            df_base["ua_port"].astype(str).str.upper().str.strip().replace({"": "UNKNOWN"})
        )
    else:
        df_base["_ua_port_norm"] = "UNKNOWN"

    # Depth_mm
    if "depth_mm" in df_base.columns:
        depth_num = pd.to_numeric(df_base["depth_mm"], errors="coerce")
        df_base["_depth_mm_norm"] = depth_num.where(
            depth_num.notna(), df_base["depth_mm"].astype(str)
        ).replace({"": "n/a"})
    else:
        df_base["_depth_mm_norm"] = "n/a"

    groups = {}
    for (port, depth), g in df_base.groupby(["_ua_port_norm", "_depth_mm_norm"], sort=True):
        key = (str(port), str(depth))
        groups[key] = g.index.to_numpy()
    return groups


# =============================================================================
# Core Cleaning Functions
# =============================================================================

def apply_1f_detrending_chunked(epochs, rel_t_epoch, win_base=(-500, -50), fs=1000.0, chunk_size=32, plot_debug=False, debug_out_dir=None):
    """
    Applies adaptive 1/f spectral subtraction in the time-domain using a baseline period.
    Processes channels in chunks to prevent OOM.

    Args:
        epochs (np.ndarray): (n_trials, n_ch, n_time)
        rel_t_epoch (np.ndarray): Time axis in ms
        win_base (tuple): Baseline window (start_ms, end_ms) for 1/f fitting
        fs (float): Sampling rate (Hz)
        chunk_size (int): Channels per chunk
        
    Returns:
        np.ndarray: 1/f detrended epochs
    """
    out = np.empty_like(epochs)
    n_trials, n_ch, n_time = epochs.shape
    
    # 1. Isolate Baseline Segment
    mask_base = (rel_t_epoch >= win_base[0]) & (rel_t_epoch < win_base[1])
    n_base = np.sum(mask_base)
    
    if n_base < 10:
        print("    [1/f Detrend] Warning: Baseline window too short. Skipping 1/f detrending.")
        return epochs.copy()
        
    print(f"    [1/f Detrend] Estimating 1/f curve from local baseline: {win_base} ms ({n_base} pts)")
        
    # Frequencies
    f_base = np.fft.rfftfreq(n_base, 1/fs)
    f_full = np.fft.rfftfreq(n_time, 1/fs)
    
    # Valid frequencies > 0 to avoid log(0)
    valid_base = f_base > 0
    f_valid_base = f_base[valid_base]
    f_valid_full = f_full[f_full > 0]
    
    # Pre-compute design matrix for log-log linear regression (FOOOF-lite)
    x_log = np.log10(f_valid_base)
    X_mat = np.vstack([np.ones_like(x_log), x_log]).T  # (F, 2)
    X_pinv = np.linalg.pinv(X_mat)                     # (2, F)
    
    # FFT magnitude scales cleanly by sqrt(N) for pink noise 
    scale_factor = np.sqrt(n_time / n_base)
    f_log_full = np.log10(f_valid_full)[np.newaxis, np.newaxis, :]
    
    for start_ch in range(0, n_ch, chunk_size):
        end_ch = min(start_ch + chunk_size, n_ch)
        chunk_epochs = epochs[:, start_ch:end_ch, :]
        
        # 2. Extract baseline chunk
        chunk_base = chunk_epochs[:, :, mask_base]
        
        # 3. Compute Baseline Magnitude
        X_base = np.fft.rfft(chunk_base, axis=2)
        M_base = np.abs(X_base)[:, :, valid_base]
        
        # Avoid log(0) in magnitudes
        M_base = np.maximum(M_base, 1e-10)
        y_log = np.log10(M_base)
        
        # 4. Fit 1/f coefficients per trial/channel: log(M) = c + alpha * log(f)
        n_tr, n_c = y_log.shape[0], y_log.shape[1]
        y_reshaped = y_log.reshape(-1, len(f_valid_base)).T  # (F, T*C)
        beta = X_pinv @ y_reshaped                           # (2, T*C)
        beta = beta.reshape(2, n_tr, n_c)
        
        c = beta[0, :, :, np.newaxis]
        alpha = beta[1, :, :, np.newaxis]
        
        # 5. Synthesize 1/f envelope for full time window
        # log(M_1f) = c + alpha * log(f_full)
        M_1f_log = c + alpha * f_log_full
        
        # Convert out of log space and apply sqrt(N) scaling to match the full FFT window
        M_1f_valid = (10 ** M_1f_log) * scale_factor
        
        # Construct full 1/f magnitude (DC = 0 to preserve global mean voltage)
        M_1f = np.zeros((n_tr, n_c, len(f_full)))
        M_1f[:, :, f_full > 0] = M_1f_valid
        
        # 6. Apply Spectral Subtraction to Full Epoch
        X_full = np.fft.rfft(chunk_epochs, axis=2)
        M_full = np.abs(X_full)
        Phase_full = np.angle(X_full)
        
        # Subtract the theoretical 1/f noise floor, floor at 0 to avoid negative amplitudes
        M_clean = np.maximum(M_full - M_1f, 0)
        
        # Reconstruct complex FFT array
        X_clean = M_clean * np.exp(1j * Phase_full)
        
        # Inverse FFT back to time domain
        chunk_clean = np.fft.irfft(X_clean, n=n_time, axis=2)
        
        if plot_debug and start_ch == 0 and debug_out_dir is not None:
            import matplotlib.pyplot as plt
            from pathlib import Path
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.plot(rel_t_epoch, chunk_epochs[0, 0, :], label='Original', alpha=0.7)
            ax.plot(rel_t_epoch, chunk_clean[0, 0, :], label='1/f Detrended', alpha=0.7)
            ax.axvspan(win_base[0], win_base[1], color='gray', alpha=0.2, label='Baseline Window')
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude")
            ax.set_title("1/f Detrending Debug (Trial 0, Ch 0)")
            ax.legend(loc="upper right")
            
            out_path = Path(debug_out_dir) / "debug_1f_detrend.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            print(f"    [1/f Detrend] Saved debug plot to {out_path}")
            
        out[:, start_ch:end_ch, :] = chunk_clean
        
    return out

def filter_zero_phase(epochs, fs, low, high, order=4, chunk_size=32):
    """
    Apply zero-phase bandpass filter using filtfilt along axis 2 (time).
    Processes in chunks along the channel axis (axis=1) to prevent 
    massive memory OOM spikes when filtering long continuous recordings.
    
    Args:
        epochs (np.ndarray): (n_trials, n_ch, n_time)
        fs (float): Sampling rate (Hz)
        low (float): Low cutoff (Hz)
        high (float): High cutoff (Hz)
        order (int): Filter order
        chunk_size (int): Number of channels to filter at once
        
    Returns:
        np.ndarray: Filtered data
    """
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    
    b, a = signal.butter(order, [low_norm, high_norm], btype='band')
    
    out = np.empty_like(epochs)
    n_ch = epochs.shape[1]
    
    # Process channels in chunks to avoid allocating gigantic padded 
    # copies of the entire matrix inside scipy.signal.filtfilt
    for start_ch in range(0, n_ch, chunk_size):
        end_ch = min(start_ch + chunk_size, n_ch)
        out[:, start_ch:end_ch, :] = signal.filtfilt(b, a, epochs[:, start_ch:end_ch, :], axis=2)
        
    return out

# =============================================================================
# Helper Functions
# =============================================================================

def apply_ipca_correction(rec_ns6, starts_ua, ends_ua):
    """
    Applies IPCA artifact removal to stimulation windows, using cross-channel templates 
    learned individually per brain region.
    """
    if not USE_IPCA_CORRECTION or not starts_ua.size:
        return rec_ns6

    fs_ua = rec_ns6.get_sampling_frequency()
    print(f"[IPCA] Starting Incremental PCA artifact correction on {starts_ua.size} blocks...")
    
    # 1. Learn precise timestamps using derivative method
    pulse_interval_ms = 1000.0 / PARAMS.stim_hz if hasattr(PARAMS, 'stim_hz') else 3.333
    mw_start = int(IPCA_PULSE_WINDOW_MS[0] / 1000.0 * fs_ua)
    mw_end   = int(IPCA_PULSE_WINDOW_MS[1] / 1000.0 * fs_ua)
    
    fw_start = int(-ARTRMV_MS_BEFORE / 1000.0 * fs_ua)
    fw_end   = int(ARTRMV_TAIL_MS / 1000.0 * fs_ua)
    
    # Identify active channels that belong to a region
    ch_ids = np.asarray(rec_ns6.get_channel_ids(), dtype=int)
    if len(ch_ids) == 0:
        return rec_ns6
        
    def get_region(e):
        if 1 <= e <= 64: return "SMA"
        if 65 <= e <= 128: return "PMd"
        if 129 <= e <= 192: return "M1i"
        if 193 <= e <= 256: return "M1s"
        return "Unknown"
        
    target_ch_regions = [get_region(e) for e in ch_ids]
    
    # Load raw
    print("[IPCA] Scanning Blackrock raw file sequentially for artifact triggers...")
    
    micro_signal_list = []
    micro_map = []  # (block_idx, p_start, p_end)
    
    n_total = rec_ns6.get_num_samples()
    interval_idx = int(pulse_interval_ms / 1000 * fs_ua)
    search_radius_idx = int((pulse_interval_ms * 0.4) / 1000 * fs_ua)
    refine_samp = max(int(0.2 / 1000.0 * fs_ua), 3)
    
    def _find_first_significant_peak(tr_local, ref_ratio=0.5):
        diff_abs = np.abs(np.diff(tr_local, prepend=tr_local[0]))
        pk_idx = np.argmax(diff_abs)
        max_d = diff_abs[pk_idx]
        if max_d < 1e-6:
            return pk_idx
        for k in range(len(diff_abs)):
            if diff_abs[k] > ref_ratio * max_d:
                return k
        return pk_idx
        
    for i, (st, en) in enumerate(zip(starts_ua, ends_ua)):
        # Coarse search
        search_margin = int(20.0 / 1000 * fs_ua)
        start_search  = st + fw_start - search_margin
        end_search    = st + fw_end + search_margin
        if start_search < 0 or end_search > n_total: continue
        
        tr_search = rec_ns6.get_traces(start_frame=start_search, end_frame=end_search, channel_ids=[rec_ns6.get_channel_ids()[0]], return_in_uV=True)[:, 0]
        tr_diff_search  = np.abs(np.diff(tr_search, prepend=tr_search[0]))
        
        predicted_centre = -fw_start + search_margin
        slop = int(5.0 / 1000 * fs_ua)
        win_lo = max(0, predicted_centre - slop)
        win_hi = min(len(tr_search), predicted_centre + slop)
        
        peak_in_win = np.argmax(tr_diff_search[win_lo:win_hi])
        anchor_idx  = win_lo + peak_in_win
        max_diff    = tr_diff_search[anchor_idx]
        
        if max_diff < 5.0: continue
        
        first_pulse_in_search = None
        for k in range(win_lo, win_hi):
            if tr_diff_search[k] > 0.4 * max_diff:
                first_pulse_in_search = k
                break
        if first_pulse_in_search is None: continue
        
        coarse_radius = max(refine_samp, 3)
        rc_lo = max(0, first_pulse_in_search - coarse_radius)
        rc_hi = min(len(tr_search), first_pulse_in_search + coarse_radius + 1)
        if rc_hi <= rc_lo: continue
        offset = _find_first_significant_peak(tr_search[rc_lo:rc_hi])
        
        coarse_centre = rc_lo + offset
        true_start = start_search + coarse_centre
        
        train_start_idx = true_start
        train_end_idx   = en + fw_end
        
        if train_end_idx <= train_start_idx: continue
        
        anchor_final = true_start
        pulse_centres = [anchor_final]
        curr = anchor_final
        while True:
            nxt = curr + interval_idx
            if nxt > train_end_idx: break
            s_lo = max(0, nxt - search_radius_idx)
            s_hi = min(n_total, nxt + search_radius_idx)
            if s_hi <= s_lo: break
            
            tr_diff = np.abs(np.diff(rec_ns6.get_traces(start_frame=s_lo, end_frame=s_hi, channel_ids=[rec_ns6.get_channel_ids()[0]], return_in_uV=True)[:, 0]))
            peak_idx = np.argmax(tr_diff)
            
            if tr_diff[peak_idx] < 0.15 * max_diff: break
            
            curr = s_lo + peak_idx
            pulse_centres.append(curr)
            
        pulse_centres = sorted(set(pulse_centres))
        
        for p_idx in pulse_centres:
            r_lo = max(0, p_idx - refine_samp)
            r_hi = min(n_total, p_idx + refine_samp + 1)
            if r_lo >= r_hi: continue
            
            true_centre = r_lo + _find_first_significant_peak(rec_ns6.get_traces(start_frame=r_lo, end_frame=r_hi, channel_ids=[rec_ns6.get_channel_ids()[0]], return_in_uV=True)[:, 0])
            p_start = true_centre + mw_start
            p_end   = true_centre + mw_end
            
            if p_start >= 0 and p_end <= n_total:
                micro_signal_list.append(rec_ns6.get_traces(start_frame=p_start, end_frame=p_end, return_in_uV=True).copy())
                micro_map.append((p_start, p_end))

    if not micro_signal_list:
        print("[IPCA] No pulses successfully extracted, skipping correction.")
        return rec_ns6
        
    micro_signal_array = np.stack(micro_signal_list) # (n_pulses, n_time, n_all_channels)
    print(f"[IPCA] Extracted {len(micro_signal_array)} artifacts. Correcting...")
    
    corrector = IPCA_Artifact_Correction(rank=IPCA_RANK)
    
    region_to_idxs = {}
    for valid_ch_idx, reg_name in enumerate(target_ch_regions):
        region_to_idxs.setdefault(reg_name, []).append(valid_ch_idx)
    micro_corrected = micro_signal_array.copy()
    n_stim, n_time, _ = micro_signal_array.shape
    
    print("Cross-channel IPCA by region:")
    for reg, idxs in region_to_idxs.items():
        if not idxs: continue
        
        reg_signal = micro_signal_array[:, :, idxs]
        pooled_signal = np.transpose(reg_signal, (0, 2, 1)).reshape(-1, n_time)
        
        reg_template = Template()
        _, reg_template = corrector.ipca_template_per_channel(pooled_signal, reg_template)
        
        print(f"  [{reg}] Shared subspace learned from {len(idxs)} channels.")
        
        for ch_idx in idxs:
            signal_ch = micro_signal_array[:, :, ch_idx].copy()
            baseline = np.mean(signal_ch[:, :3], axis=1, keepdims=True)
            centered = signal_ch - baseline
            
            artifact = centered @ reg_template.weights.T @ reg_template.weights
            corr_ch = centered - artifact + baseline
            micro_corrected[:, :, ch_idx] = corr_ch
            
    print("[IPCA] Building dynamically-correcting IPCACorrectedRecording...")
    rec_corr_mem = IPCACorrectedRecording(rec_ns6, micro_map, micro_corrected)
    return rec_corr_mem



# --- Debug Plotting ---
def build_lfp_preprocessing_pipeline(
    recording: si.BaseRecording,
    stim_indices_native: np.ndarray,
    curr_sess_name: str = "unknown",
    local_radii: tuple | None = None) -> si.BaseRecording:
    """
    Custom pipeline with purely Bandpass and Resampling operations.
    Blanking / CMR are bypassed heavily for Utah IPCA.
    """
    # 1. Bandpass (1-200 Hz) BEFORE resampling
    print(f"    [SI Pipeline] Bandpass filter (1-200 Hz)...")
    rec_filtered = spre.bandpass_filter(recording, freq_min=1.0, freq_max=200.0)
    
    # 2. Resampling (30k -> 1k)
    # SI resample includes anti-aliasing filter
    print(f"    [SI Pipeline] Resampling to {TARGET_FS} Hz...")
    rec_resampled = spre.resample(rec_filtered, resample_rate=TARGET_FS)
    
    return rec_resampled


def extract_single_epoch(rec_obj, center_idx, pre_samps, post_samps):
    """
    Extract a single epoch from a recording object.
    
    Args:
        rec_obj: SpikeInterface recording object
        center_idx: Center sample index (typically stim onset)
        pre_samps: Samples before center
        post_samps: Samples after center
    
    Returns:
        np.ndarray of shape (n_ch, n_time) or None if out of bounds
    """
    start = center_idx - pre_samps
    end = center_idx + post_samps
    
    # Check bounds
    if start < 0 or end > rec_obj.get_num_samples():
        return None
        
    # Get trace (n_samples, n_ch) -> Transpose to (n_ch, n_samples)
    return rec_obj.get_traces(start_frame=start, end_frame=end, return_in_uV=True).T


def slice_epoch(data, t_axis, target_win):
    """
    Slice epoch data to a specific time window.
    
    Args:
        data: np.ndarray of shape (n_trials, n_ch, n_time)
        t_axis: Time axis array
        target_win: Tuple of (start_ms, end_ms)
    
    Returns:
        Tuple of (sliced_data, sliced_time_axis)
    """
    mask = (t_axis >= target_win[0]) & (t_axis < target_win[1])
    return data[:, :, mask], t_axis[mask]





# =============================================================================
# Grouped Baseline LFP Processing - Utah Array
# =============================================================================

def process_baseline_group_utah(
    group_key: tuple,  # (port, depth)
    session_infos: list,  # List of dicts with session info
):
    """
    Process a group of baseline/control sessions for Utah Array by loading IR events from
    existing baseline PeriStim files and concatenating LFP epochs.
    
    Args:
        group_key: Tuple of (ua_port, depth_mm)
        session_infos: List of dicts with keys: session, br_idx, intan_idx, shift_ms, fs_intan
    """
    port, depth = group_key
    
    # Sanitize for filename
    def _sanitize(s):
        return str(s).replace(".", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
    
    depth_str = _sanitize(depth)
    port_str = _sanitize(port)
    
    out_npz = UA_LFP_CKPT_ROOT / f"aligned_lfp__baseline__Depth_{depth_str}_port_{port_str}.npz"
    
    if out_npz.exists():
        print(f"[Baseline UA] Skipping group Depth={depth}, Port={port}, output exists.")
        return
        
    print(f"\n[Baseline UA] Processing group: Depth={depth}, Port={port}")
    print(f"    Sessions: {[s['session'] for s in session_infos]}")
    
    # 1. Load IR events from baseline PeriStim file
    ir_events_ms = None
    loaded_target = None  # Track which target was loaded
    for target in ["target_A", "target_B", ""]:
        if target:
            target_dir = PERI_ROOT / "control_reaches" / target
        else:
            target_dir = PERI_ROOT / "control_reaches"

        if not target_dir.exists():
            if target:
                print(f"    [Info] Dir not found: {target_dir}")
            continue
            
        # Match actual peristim file naming convention
        # Files are like: peristim__NRR_RW012_260116_140816__BR_002_target_A.npz
        if target:
            matches = list(target_dir.glob("peristim__*.npz"))
        else:
            # Avoid matching files in subdirectories, just the control_reaches folder
            matches = [f for f in target_dir.glob("peristim__*.npz") if f.is_file()]
        
        if matches:
            # Collect IR events from ALL baseline files for this target
            all_events = []
            for baseline_file in matches:
                print(f"    Loading IR events from: {baseline_file.name}")
                try:
                    bl_data = np.load(str(baseline_file), allow_pickle=True)
                    if "event_ms" in bl_data and bl_data["event_ms"].size > 0:
                        all_events.append(bl_data["event_ms"].flatten())
                        print(f"      Found {bl_data['event_ms'].size} IR events.")
                    else:
                        print(f"      [WARN] No 'event_ms' key or empty in {baseline_file.name}")
                except Exception as e:
                    print(f"      [WARN] Failed to load: {e}")
            
            if all_events:
                ir_events_ms = np.concatenate(all_events)
                ir_events_ms = np.sort(ir_events_ms)
                loaded_target = target[-1] if target else "N"  # 'A', 'B', or 'N'
                print(f"    Total: {len(ir_events_ms)} IR events from {len(all_events)} files (Target {loaded_target}).")
                break
    
    if ir_events_ms is None or len(ir_events_ms) == 0:
        print(f"    [WARN] No IR events found for group. Skipping.")
        return
    
    # 1.5. Apply Baseline Trial Exclusions
    exclusion_key = f"baseline_port{port}_target{loaded_target}"
    exclude_indices = EXCLUDE_BASELINE_TRIALS.get(exclusion_key, [])
    if exclude_indices:
        print(f"    [Exclusion] Removing {len(exclude_indices)} trials from {exclusion_key}: {exclude_indices}")
        valid_mask = np.ones(len(ir_events_ms), dtype=bool)
        for idx in exclude_indices:
            if 0 <= idx < len(ir_events_ms):
                valid_mask[idx] = False
        ir_events_ms = ir_events_ms[valid_mask]
        print(f"    [Exclusion] Remaining trials: {len(ir_events_ms)}")
    
    # 2. Process each session in the group
    all_epochs_bb = []
    ua_ids_global = None
    
    for sess_info in session_infos:
        br_idx = sess_info["br_idx"]
        shift_ms = sess_info["shift_ms"]
        fs_intan = sess_info.get("fs_intan", 30000.0)
        
        # Find Blackrock ns6 file
        ns6_files = [p for p in BR_ROOT.iterdir() if p.is_file() and f"_{br_idx:03d}" in p.name and p.suffix == '.ns6']
        
        if not ns6_files:
            print(f"    [WARN] No .ns6 file found for BR {br_idx:03d}")
            continue
            
        sess_path = ns6_files[0]
        print(f"    Processing BR {br_idx:03d}: {sess_path.name}")
        
        try:
            # Load Blackrock recording
            rec_raw = se.read_blackrock(str(sess_path), stream_name='nsx6', all_annotations=True)
            
            # Apply UA mapping natively
            rec_mapped, ua_ids_1based = _apply_native_ua_mapping(rec_raw, UA_MAP, br_idx, METADATA_CSV)
            if rec_mapped is None:
                print(f"        [WARN] No mapped channels found for BR {br_idx}. Skipping.")
                continue
            
            # Store ID info from first session
            if ua_ids_global is None:
                ua_ids_global = ua_ids_1based
            
            fs_native = rec_mapped.get_sampling_frequency()
            rec_duration_ms = rec_mapped.get_total_duration() * 1000.0
            
            # Convert IR events (in BR-aligned ms) to session time
            # IR events are relative to BR time, shift_ms converts Intan->BR
            # So for BR time we don't need to shift, IR events are already in BR time
            sess_ir_ms_valid = ir_events_ms[(ir_events_ms >= 0) & (ir_events_ms < rec_duration_ms)]
            
            if len(sess_ir_ms_valid) == 0:
                print(f"        No valid IR events for this session. Skipping.")
                continue
                
            print(f"        Found {len(sess_ir_ms_valid)} valid IR events for this session.")
            
            # Convert to samples
            stim_start_samps = (sess_ir_ms_valid * fs_native / 1000.0).astype(np.int64)
            
            # Build preprocessing pipeline (no blanking for baseline)
            # local_radii=None (default) -> Skips CMR which is correct for Utah per user request
            rec_proc = build_lfp_preprocessing_pipeline(
                rec_mapped,
                stim_indices_native=np.array([]),  # No blanking for control
                curr_sess_name=sess_path.name
            )
            
            # Convert to resampled indices
            stim_indices_res = (stim_start_samps * TARGET_FS / fs_native).astype(np.int64)
            
            # Calculate epoch samples (PADDED)
            pad_samps = int(PAD_MS * TARGET_FS / 1000.0)
            pre_samps = int(EPOCH_PRE_MS * TARGET_FS / 1000) + pad_samps
            post_samps = int(EPOCH_POST_MS * TARGET_FS / 1000) + pad_samps
            
            n_samps_total = rec_proc.get_total_samples()
            
            # Extract epochs (sequential to avoid joblib pickling issues)
            segs_list = []
            for idx in stim_indices_res:
                if (idx - pre_samps >= 0) and (idx + post_samps <= n_samps_total):
                    seg = extract_single_epoch(rec_proc, idx, pre_samps, post_samps)
                    if seg is not None:
                        segs_list.append(seg)
            
            if len(segs_list) > 0:
                sess_epochs = np.stack(segs_list, axis=0)
                all_epochs_bb.append(sess_epochs)
                print(f"        Extracted {len(segs_list)} epochs from session.")
            else:
                print(f"        [WARN] No valid epochs extracted from session.")
                
            del rec_raw, rec_mapped, rec_proc
            gc.collect()
            
        except Exception as e:
            print(f"        [ERROR] Failed to process BR {br_idx:03d}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 3. Concatenate epochs across sessions
    if len(all_epochs_bb) == 0:
        print(f"    [WARN] No epochs extracted from any session in group. Skipping.")
        return
        
    concat_epochs = np.concatenate(all_epochs_bb, axis=0)
    print(f"    Concatenated epochs: {concat_epochs.shape} (trials, channels, time)")
    
    # 4. Clean epochs — build rel_t_epoch FIRST (needed by the detrending call)
    n_time = concat_epochs.shape[2]
    rel_t_epoch = (np.arange(n_time) / TARGET_FS * 1000.0) - (EPOCH_PRE_MS + PAD_MS)

    print(f"    [Cleaning] Applying 1/f Detrending to baseline epochs...")
    concat_epochs_clean = apply_1f_detrending_chunked(
        concat_epochs, rel_t_epoch, win_base=(-EPOCH_PRE_MS + 100.0, EPOCH_POST_MS - 100.0), fs=TARGET_FS,
        plot_debug=PLOT_1F_DEBUG, debug_out_dir=UA_LFP_CKPT_ROOT.parent.parent / "figures" / "UA_LFP" / "debug"
    )

    # --- Step 4.5: Reject Bad Channels Disabled ---
    bad_ch_indices = np.array([], dtype=int)
    
    WIN_PRE = (-EPOCH_PRE_MS, -BLANK_PRE_MS)
    WIN_POST = (BLANK_POST_MS, EPOCH_POST_MS - BLANK_PRE_MS)
    
    # Save the FULL continuous corrected window for UA
    WIN_FULL = (-EPOCH_PRE_MS, EPOCH_POST_MS)
    bb_full, t_full = slice_epoch(concat_epochs_clean, rel_t_epoch, WIN_FULL)
    bb_pre, t_pre = slice_epoch(concat_epochs_clean, rel_t_epoch, WIN_PRE)
    bb_post, t_post = slice_epoch(concat_epochs_clean, rel_t_epoch, WIN_POST)
    
    results = {
        "broadband_full": bb_full,
        "broadband_pre": bb_pre,
        "broadband_post": bb_post,
    }
    
    # 6. Band analysis
    for band_name, (low, high) in LFP_BANDS.items():
        print(f"    Filtering {band_name} ({low}-{high} Hz)...")
        filt_epochs = filter_zero_phase(concat_epochs_clean, TARGET_FS, low, high)
        results[f'{band_name}_full'], _ = slice_epoch(filt_epochs, rel_t_epoch, WIN_FULL)
        results[f'{band_name}_pre'], _ = slice_epoch(filt_epochs, rel_t_epoch, WIN_PRE)
        results[f'{band_name}_post'], _ = slice_epoch(filt_epochs, rel_t_epoch, WIN_POST)
    
    # 7. Save
    save_dict = {
        "fs_lfp": TARGET_FS,
        "group_port": port,
        "group_depth": depth,
        "sessions": [s["session"] for s in session_infos],
        "br_indices": [s["br_idx"] for s in session_infos],
        "stim_ms": ir_events_ms,
        "n_trials": concat_epochs_clean.shape[0],
        "rel_time_pre": t_pre,
        "rel_time_post": t_post,
        "stim_channels": ["Baseline_IR"],
        # UA-specific metadata
        "ua_ids_1based": ua_ids_global,
        **results
    }
    
    np.savez_compressed(out_npz, **save_dict)
    print(f"    Saved -> {out_npz}")
    
    gc.collect()


def _apply_native_ua_mapping(rec_raw, UA_MAP, br_idx, metadata_csv):
    ua_port = "A"
    if metadata_csv.exists():
        try:
            df_meta_raw = pd.read_csv(metadata_csv)
            df_meta = _normalize_metadata(df_meta_raw)
            if "br_file" in df_meta.columns:
                br_col = pd.to_numeric(df_meta["br_file"], errors="coerce")
                mask = br_col == br_idx
                if mask.any():
                    row_meta = df_meta.loc[mask].iloc[0]
                    port_val = row_meta.get("_ua_port_norm") if "_ua_port_norm" in row_meta else (
                        str(row_meta.get("ua_port", "UNKNOWN")).upper().strip() or "UNKNOWN"
                    )
                    if port_val in ("A", "B"):
                        ua_port = port_val
        except Exception as e:
            print(f"    [WARN] Failed to read UA port: {e}")
            
    raw_ch_ids = np.asarray(rec_raw.get_channel_ids(), int)
    if ua_port == "B":
        raw_ch_ids += 128
        
    id_to_row = {ch: i for i, ch in enumerate(raw_ch_ids)}
    active_elecs, active_rows = [], []
    for elec_zero_idx, nsp_id in enumerate(UA_MAP):
        if nsp_id is not None and not np.isnan(nsp_id) and nsp_id > 0 and int(nsp_id) in id_to_row:
            active_elecs.append(elec_zero_idx + 1)
            active_rows.append(id_to_row[int(nsp_id)])
            
    if len(active_rows) > 0:
        rec_mapped = rec_raw.select_channels(
            channel_ids=[rec_raw.channel_ids[r] for r in active_rows]
        ).rename_channels(
            new_channel_ids=[str(e) for e in active_elecs]
        )
        ua_ids_1based = np.array(active_elecs, dtype=int)
        return rec_mapped, ua_ids_1based
    else:
        return None, None

def process_utah_session(sess_name: str, br_idx: int, shift_ms: float, fs_intan: float, shift_sample: float):
    matches = [p for p in BR_ROOT.iterdir() if p.is_dir() and f"_{br_idx:03d}" in p.name]
    
    if matches:
        # Found a directory
        dir_path = matches[0]
        # Look for ns files inside
        ns_files = list(dir_path.glob("*.ns*"))
        ns6 = [f for f in ns_files if f.suffix == '.ns6']
        ns2 = [f for f in ns_files if f.suffix == '.ns2']
        ns5 = [f for f in ns_files if f.suffix == '.ns5']
        
        if ns6: sess_path = ns6[0]
        elif ns2: sess_path = ns2[0]
        elif ns5: sess_path = ns5[0]
        elif ns_files: sess_path = ns_files[0]
        else:
            print(f"[UA] Found directory {dir_path.name} but no .ns files inside.")
            return
            
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
        # User requested to use nsx6
        print(f"  Loading Blackrock stream: nsx6")
        rec_raw = se.read_blackrock(str(sess_path), stream_name='nsx6', all_annotations=True)
    except Exception as e:
        print(f"[UA] Failed to load nsx6: {e}")
        return

    # Mapping
    rec_mapped, ua_ids_1based = _apply_native_ua_mapping(rec_raw, UA_MAP, br_idx, METADATA_CSV)
    if rec_mapped is None:
        print("[WARN] No mapped channels found.")
        return
    
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
        
        # IPCA Artifact Correction before going into the LFP pipeline
        starts_intan = block_bounds[:, 0].astype(np.int64)
        ends_intan   = block_bounds[:, 1].astype(np.int64)
        scale = fs_native / fs_intan
        starts_ua = np.round((starts_intan - shift_sample) * scale).astype(np.int64)
        ends_ua   = np.round((ends_intan   - shift_sample) * scale).astype(np.int64)
        
        n_total = rec_mapped.get_num_samples()
        ends_ua = np.minimum(ends_ua, n_total)
        valid = (ends_ua > starts_ua) & (starts_ua >= 0) & (ends_ua <= n_total)
        starts_ua = starts_ua[valid]
        ends_ua   = ends_ua[valid]
        
        rec_mapped = apply_ipca_correction(
            rec_mapped, starts_ua, ends_ua
        )
        
        # Override stim_indices_native so that basic blanking doesn't run,
        # since IPCA handled it cleanly.
        stim_indices_native = np.array([], dtype=int)

    # 2. Build SI Pipeline
    # Blank(-5, +20) -> Local CMR (None) -> DS(1000)
    # local_radii=None -> No CMR for Utah
    rec_proc = build_lfp_preprocessing_pipeline(
        rec_mapped, 
        stim_indices_native=stim_indices_native,
        curr_sess_name=sess_path.name,
        local_radii=None 
    )
    
    # 3. Get Traces Memory-Safely (No Eager Load)
    print("  [UA] Segmenting traces lazily from SpikeInterface pipeline...")
    
    pad_samps = int(PAD_MS * TARGET_FS / 1000.0)
    pre_samps = int(EPOCH_PRE_MS * TARGET_FS / 1000) + pad_samps
    post_samps = int(EPOCH_POST_MS * TARGET_FS / 1000) + pad_samps
    n_samps_total = rec_proc.get_total_samples()
    
    # Convert native stim ms to resampled indices for segmentation
    stim_indices_res = (stim_ms_ua * TARGET_FS / 1000.0).astype(np.int64)

    # --- Parameters ---
    WIN_EPOCH = (-EPOCH_PRE_MS, EPOCH_POST_MS)   # Epoch Window
    WIN_PRE   = (-EPOCH_PRE_MS, -BLANK_PRE_MS)    # Baseline Window
    WIN_POST  = (BLANK_POST_MS, EPOCH_POST_MS - BLANK_PRE_MS)   # Analysis Window (Post-Blank)
    
    results = {}
    
    print(f"  Segmenting Broadband Epochs ({WIN_EPOCH} ms PAD={PAD_MS}ms)...", flush=True)
    segs_list_bb = []
    valid_stim_ms = []

    for idx_res, stim_time in zip(stim_indices_res, stim_ms_ua):
        if (idx_res - pre_samps >= 0) and (idx_res + post_samps <= n_samps_total):
            seg = extract_single_epoch(rec_proc, idx_res, pre_samps, post_samps)
            if seg is not None:
                segs_list_bb.append(seg)
                valid_stim_ms.append(stim_time)

    if not segs_list_bb:
        print(f"  [WARN] No valid segments found for {sess_path.name}. Skipping.")
        return

    segs_bb = np.stack(segs_list_bb, axis=0) # (n_trials, n_channels, n_time)
    valid_stim = np.array(valid_stim_ms)
    
    n_time = pre_samps + post_samps
    rel_t_epoch = (np.arange(n_time) / TARGET_FS * 1000.0) - (EPOCH_PRE_MS + PAD_MS)

    # 7. Per-Trial Cleaning
    print(f"  [Cleaning] Applying 1/f Detrending to broadband epochs...")
    segs_bb_clean = apply_1f_detrending_chunked(
        segs_bb, rel_t_epoch, win_base=(-EPOCH_PRE_MS + 100.0, EPOCH_POST_MS - 100.0), fs=TARGET_FS,
        plot_debug=PLOT_1F_DEBUG, debug_out_dir=UA_LFP_CKPT_ROOT.parent.parent / "figures" / "UA_LFP" / "debug"
    )
    
    # --- Step 7.5: Reject Bad Channels Disabled ---
    bad_ch_indices = np.array([], dtype=int)
    results['bad_channels'] = bad_ch_indices
        
    WIN_FULL = (-EPOCH_PRE_MS, EPOCH_POST_MS)
    results['broadband_full'], t_full = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_FULL)
    results['broadband_pre'], t_pre = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_PRE)
    results['broadband_post'], t_post = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_POST)
    
    # 9. Band Analysis (Filter Continuous -> Segment)
    # This avoids padding issues for segments since we have continuous data
    for band_name, (low, high) in LFP_BANDS.items():
        print(f"  [SI Pipeline] Filtering Continuous Trace {band_name} ({low}-{high} Hz)...", flush=True)
        
        # Use SpikeInterface directly for filtering memory-safely
        rec_filtered = spre.bandpass_filter(rec_proc, freq_min=low, freq_max=high)
        
        segs_list_filt = []
        for idx_res in stim_indices_res:
            if (idx_res - pre_samps >= 0) and (idx_res + post_samps <= n_samps_total):
                seg = extract_single_epoch(rec_filtered, idx_res, pre_samps, post_samps)
                if seg is not None:
                    segs_list_filt.append(seg)
        
        if len(segs_list_filt) > 0:
            segs_filt = np.stack(segs_list_filt, axis=0)
            # Apply Rejection
            if len(bad_ch_indices) > 0:
                segs_filt[:, bad_ch_indices, :] = np.nan
             
            results[f'{band_name}_full'], _ = slice_epoch(segs_filt, rel_t_epoch, WIN_FULL)
            results[f'{band_name}_pre'], _ = slice_epoch(segs_filt, rel_t_epoch, WIN_PRE)
            results[f'{band_name}_post'], _ = slice_epoch(segs_filt, rel_t_epoch, WIN_POST)

        del rec_filtered, segs_list_filt
        gc.collect()
    # Save
    save_dict = {
        "fs_lfp": TARGET_FS,
        "ua_ids_1based": ua_ids_1based,
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


def quick_check_stim(sess_path: Path, stream_name: str, intan_idx: int, br_idx: int) -> bool:
    """
    Check if stim file or aligned file exists and has events.
    """
    sess_name = sess_path.name
    
    # 0. Check Aligned File First
    aligned_path = ALIGNED_CKPT_ROOT / f"aligned__{sess_name}__Intan_{intan_idx:03d}__BR_{br_idx:03d}.npz"
    if aligned_path.exists():
         try:
             aln = np.load(aligned_path, allow_pickle=True)
             if "stim_ms" in aln and aln["stim_ms"].size > 0:
                 print(f"    [Check] Found aligned file with {aln['stim_ms'].size} events: {aligned_path.name}")
                 return True
         except:
             pass

    stim_npz = NPRW_AUX_DATA / f"{sess_name}_Intan_streams" / "stim_stream.npz"
    
    # 1. Standard Stim Check
    has_stim = False
    
    # Try loading existing fast
    if stim_npz.exists():
        try:
            print(f"    [Check] Loading existing {stim_npz.name}")
            stim_data = rcp.load_stim_detection(stim_npz)
            bounds = stim_data.get("block_bounds_samples")
            if bounds is not None and bounds.size > 0:
                print(f"    [Check] Found {len(bounds)} stim blocks.")
                has_stim = True
            else:
                print("    [Check] Existing file has 0 stim blocks.")
        except Exception as e:
            print(f"    [Check] Failed to load existing: {e}")

    # Fallback to extraction if not found or empty?? 
    # Actually, if existing file is empty, it means we already tried extracting and found nothing.
    # So we shouldn't re-extract standard stim. 
    # But if file didn't exist, we try extraction.
    
    if not has_stim and not stim_npz.exists():
        try:
            print(f"    [Check] Extracting new stim file...")
            stim_ext = rcp.extract_stim_npz(sess_path, out_dir=NPRW_AUX_DATA, stim_stream_name=stream_name, chanmap_perm=None)
            if stim_ext is not None:
                bounds = stim_ext.get("block_bounds_samples")
                if bounds is not None and bounds.size > 0:
                     has_stim = True
        except Exception as e:
            print(f"  [Check Stim] Failed: {e}")

    if has_stim:
        return True

    # -------------------------------------------------------------------------
    # [Control/IR Fallback Check]
    # If no stim, check if we have valid IR events from baseline PeriStim files
    # -------------------------------------------------------------------------
    print(f"    [Check] No stim found. Checking for baseline PeriStim files...")
    try:
        # Look for baseline files in PERI_ROOT / Target_A and Target_B
        sess_name = sess_path.name
        
        for target in ["control_reaches/target_A", "control_reaches/target_B", "control_reaches"]:
            target_dir = PERI_ROOT / target
            if not target_dir.exists():
                continue
                
            # Pattern: peristim__*{sess_name}*.npz
            pattern = f"peristim__*{sess_name}*.npz"
            matches = list(target_dir.glob(pattern))
            
            if matches:
                baseline_file = matches[0]
                print(f"    [Check] Found baseline file: {baseline_file.name}")
                try:
                    bl_data = np.load(baseline_file, allow_pickle=True)
                    if "stim_ms" in bl_data and bl_data["stim_ms"].size > 0:
                        print(f"    [Check] Baseline file has {bl_data['stim_ms'].size} IR events.")
                        return True
                except Exception as e:
                    print(f"    [Check] Failed to load baseline file: {e}")
                    
        print(f"    [Check] No baseline PeriStim files found for {sess_name}.")
                    
    except Exception as e:
        print(f"    [Check] Baseline file check failed: {e}")
            
    return False

def main():
    if not SHIFT_CSV.exists():
        print(f"Error: {SHIFT_CSV} not found. Cannot proceed with aligned analysis.")
        return

    with SHIFT_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    print(f"Found {len(rows)} sessions in shift CSV.")
    
    # =========================================================================
    # Load Metadata to Identify Baseline Sessions
    # =========================================================================
    baseline_br_idxs = set()  # BR indices that are baseline sessions
    baseline_groups = {}  # {(port, depth): [session_info, ...]}
    
    if METADATA_CSV.exists():
        print(f"\nLoading metadata from {METADATA_CSV.name}...")
        try:
            df_meta_raw = pd.read_csv(METADATA_CSV)
            df_meta = _normalize_metadata(df_meta_raw)
            
            groups = _find_baseline_groups(df_meta)
            if groups:
                print(f"Found {len(groups)} baseline groups: {list(groups.keys())}")
                
                # Build mapping BR -> (port, depth)
                if "br_file" in df_meta.columns:
                    br_col = pd.to_numeric(df_meta["br_file"], errors="coerce")
                    for (port, depth), idxs in groups.items():
                        for i in idxs:
                            br = br_col.iloc[i]
                            if pd.notna(br):
                                baseline_br_idxs.add(int(br))
                                
                # Initialize groups for session collection
                for key in groups:
                    baseline_groups[key] = []
                    
        except Exception as e:
            print(f"[WARN] Failed to load metadata for baseline detection: {e}")
    else:
        print(f"[WARN] Metadata CSV not found: {METADATA_CSV}. Baseline grouping disabled.")
    
    print(f"Identified {len(baseline_br_idxs)} baseline BR indices: {sorted(baseline_br_idxs)}")
    
    # =========================================================================
    # Per-Session Loop (Skip Baseline Sessions)
    # =========================================================================
    for row in rows:
        # The CSV has 'intan_filename' (e.g., NRR_RW012_260116_something). We want the session name.
        if "session" in row:
            session_name = row["session"]
        elif "intan_filename" in row:
            session_name = "_".join(row["intan_filename"].split("_")[:3])  # E.g. "NRR_RW012_260116"
        else:
            continue
            
        intan_idx = int(row["intan_idx"])
        br_idx = int(row["br_idx"])
        shift_ms = float(row["shift_ms"])
        fs_intan = float(row.get("fs_intan", 30000.0))
        shift_sample = float(row.get("shift_sample", 0.0))
        
        # Check if this is a baseline session
        if br_idx in baseline_br_idxs:
            print(f"\n[Baseline] Session {session_name} (BR={br_idx}) is baseline. Will process in group.")
            
            # Find which group this belongs to and add session info
            if METADATA_CSV.exists():
                try:
                    df_meta_raw = pd.read_csv(METADATA_CSV)
                    df_meta = _normalize_metadata(df_meta_raw)
                    
                    # Find the row for this br_idx
                    if "br_file" in df_meta.columns:
                        br_col = pd.to_numeric(df_meta["br_file"], errors="coerce")
                        mask = br_col == br_idx
                        if mask.any():
                            row_meta = df_meta.loc[mask].iloc[0]
                            port = row_meta.get("_ua_port_norm") if "_ua_port_norm" in row_meta else (
                                str(row_meta.get("ua_port", "UNKNOWN")).upper().strip() or "UNKNOWN"
                            )
                            depth = row_meta.get("_depth_mm_norm") if "_depth_mm_norm" in row_meta else (
                                str(row_meta.get("depth_mm", "n/a")).strip() or "n/a"
                            )
                            
                            # Normalize port/depth the same way as _find_baseline_groups
                            port = str(port).upper().strip() or "UNKNOWN"
                            try:
                                depth = float(depth)
                            except:
                                depth = str(depth)
                            depth = str(depth)
                            
                            group_key = (port, depth)
                            if group_key not in baseline_groups:
                                baseline_groups[group_key] = []
                            
                            baseline_groups[group_key].append({
                                "session": session_name,
                                "br_idx": br_idx,
                                "intan_idx": intan_idx,
                                "shift_ms": shift_ms,
                                "fs_intan": fs_intan,
                            })
                except Exception as e:
                    print(f"    [WARN] Failed to identify group for session: {e}")
            continue
        
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
        has_stim = quick_check_stim(sess_path, STIM_STREAM, intan_idx, br_idx)
        
        if not has_stim:
            print(f"  -> No valid stim events found. Skipping both NPRW and UA for this session.")
            continue
            
        print(f"  -> Stim OK. Proceeding...")

        try:
            process_utah_session(session_name, br_idx, shift_ms, fs_intan, shift_sample)
        except Exception as e:
            print(f"[ERROR] UA {br_idx}: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Process Baseline Groups
    # =========================================================================
    if baseline_groups:
        print(f"\n{'='*60}")
        print("Processing Baseline Groups")
        print(f"{'='*60}")
        
        for group_key, session_infos in baseline_groups.items():
            if not session_infos:
                continue
            
            try:
                process_baseline_group_utah(group_key, session_infos)
            except Exception as e:
                print(f"[ERROR] Baseline UA group {group_key}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()

