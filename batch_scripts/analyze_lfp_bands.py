import gc
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

import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

"""
    This script analyzes LFP bands (Alpha, Beta, Low Gamma, High Gamma).
    It processes both NPRW (Intan) and Utah Array (Blackrock) data.
    
    Output:
        Aligned data .npz files containing windowed traces for each band.
"""

# ---------- Config ----------
# Bands of interest
LFP_BANDS = {
    "alpha": (5, 13),
    "beta": (13, 30),
    "low_gamma": (30, 60),
    "high_gamma": (60, 120)
}

TARGET_FS = 1000  # Downsample to 1kHz
# Filter order for butterworth
FILTER_ORDER = 4

# Alignment Window
WIN_MS = (-500.0, 800.0)

# Output Directories
NPRW_LFP_CKPT_ROOT = OUT_BASE / "checkpoints" / "NPRW_LFP"
UA_LFP_CKPT_ROOT = OUT_BASE / "checkpoints" / "UA_LFP"
CACHE_NPRW_DS_ROOT = OUT_BASE / "checkpoints" / "NPRW_DS_Raw"
CACHE_UA_DS_ROOT = OUT_BASE / "checkpoints" / "UA_DS_Raw"

NPRW_LFP_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
UA_LFP_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_NPRW_DS_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_UA_DS_ROOT.mkdir(parents=True, exist_ok=True)

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


def remove_stimulation_artifacts(
    traces: np.ndarray,
    stim_indices: np.ndarray,
    fs: float,
    blank_pre_ms: float = 5.0,
    blank_post_ms: float = 120.0,
    template_post_ms: float = 800.0,
    fit_exponential: bool = True,
    use_robust_median: bool = True,
    outlier_threshold: float = 3.0,
    blanking_mode: str = 'interp') -> np.ndarray:
    """
    Comprehensive stimulation artifact removal pipeline.
    """
    n_channels, n_samples = traces.shape
    traces_cleaned = traces.copy()
    
    # Convert ms to samples
    blank_pre = int(blank_pre_ms * fs / 1000)
    blank_post = int(blank_post_ms * fs / 1000)
    template_post = int(template_post_ms * fs / 1000)
    
    # ---- STEP 1: Apply Blanking FIRST ----
    # Zero out or interpolate the immediate artifact region
    # This ensures the template is not contaminated by the massive transient
    traces_cleaned = apply_blanking(
        traces_cleaned, stim_indices, fs, blank_pre_ms, blank_post_ms, mode=blanking_mode
    )
    
    # ---- STEP 2: Bandpass Filter (0.5 - 300 Hz) ----
    # User requested bandpass after blanking
    print("    [Artifact Removal] Applying Bandpass Filter (0.5 - 300 Hz)...")
    sos_bp = signal.butter(4, [0.5, 300.0], btype='bandpass', fs=fs, output='sos')
    traces_cleaned = signal.sosfiltfilt(sos_bp, traces_cleaned, axis=1)

    # ---- STEP 3: Extract post-stim epochs for template ----
    # Only use POST-STIM data for template (not pre-stim)
    # Start from end of blank period to avoid immediate transient
    
    template_start_offset = blank_post  # Start after immediate artifact
    template_length = template_post - blank_post
    
    epochs_for_template = []
    valid_stim_indices = []
    
    for stim_idx in stim_indices:
        start = stim_idx + template_start_offset
        end = stim_idx + template_post
        
        if start >= 0 and end <= n_samples:
            epochs_for_template.append(traces_cleaned[:, start:end].copy())
            valid_stim_indices.append(stim_idx)
    
    if len(epochs_for_template) == 0:
        print("[WARN] No valid epochs for template computation")
        return traces_cleaned
    
    epochs_array = np.array(epochs_for_template)  # (n_trials, n_channels, n_time)
    
    # ---- STEP 4: Compute robust median template per channel ----
    if use_robust_median:
        template = compute_robust_median_template(
            epochs_array, outlier_threshold=outlier_threshold
        )
    else:
        template = np.median(epochs_array, axis=0)  # (n_channels, n_time)
    
    # ---- STEP 5: Optionally fit exponential to template ----
    if fit_exponential:
        template = fit_and_remove_exponential_from_template(template, fs)
    
    # ---- STEP 6: Subtract template from each trial ----
    for stim_idx in valid_stim_indices:
        start = stim_idx + template_start_offset
        end = stim_idx + template_post
        
        # Get pre-stim baseline for this trial
        baseline_start = max(0, stim_idx - int(100 * fs / 1000))  # 100ms pre-stim
        baseline_end = stim_idx - blank_pre
        
        if baseline_end > baseline_start:
            baseline = np.mean(traces_cleaned[:, baseline_start:baseline_end], axis=1, keepdims=True)
        else:
            baseline = np.zeros((n_channels, 1))
        
        # Subtract template (preserving baseline)
        template_baseline = template[:, -int(50 * fs / 1000):].mean(axis=1, keepdims=True)  # Last 50ms of template
        traces_cleaned[:, start:end] -= (template - template_baseline)
    
    return traces_cleaned


def compute_robust_median_template(
    epochs: np.ndarray,
    outlier_threshold: float = 3.0) -> np.ndarray:
    """
    Compute median template with outlier trial rejection per channel.
    """
    n_trials, n_channels, n_time = epochs.shape
    template = np.zeros((n_channels, n_time))
    
    for ch in range(n_channels):
        ch_epochs = epochs[:, ch, :]  # (n_trials, n_time)
        
        # Compute power/variance of each trial for this channel
        trial_power = np.var(ch_epochs, axis=1)
        
        # MAD-based outlier detection
        median_power = np.median(trial_power)
        mad = np.median(np.abs(trial_power - median_power))
        
        if mad > 0:
            outlier_mask = np.abs(trial_power - median_power) > outlier_threshold * mad * 1.4826
        else:
            outlier_mask = np.zeros(n_trials, dtype=bool)
        
        # Compute median excluding outliers
        clean_epochs = ch_epochs[~outlier_mask, :]
        
        if len(clean_epochs) > 0:
            template[ch, :] = np.median(clean_epochs, axis=0)
        else:
            template[ch, :] = np.median(ch_epochs, axis=0)
    
    return template


def fit_and_remove_exponential_from_template(
    template: np.ndarray,
    fs: float,
    fit_start_ms: float = 50.0,
    fit_end_ms: float = None) -> np.ndarray:
    """
    Fit exponential decay to template and subtract it.
    """
    n_channels, n_time = template.shape
    template_corrected = template.copy()
    
    fit_start = int(fit_start_ms * fs / 1000)
    fit_end = n_time if fit_end_ms is None else int(fit_end_ms * fs / 1000)
    
    def exp_decay(t, a, tau, c):
        return a * np.exp(-t / tau) + c
    
    t = np.arange(fit_end - fit_start)
    
    for ch in range(n_channels):
        segment = template[ch, fit_start:fit_end]
        
        # Initial parameter guess
        a0 = segment[0] - segment[-1]
        tau0 = len(segment) / 3
        c0 = segment[-1]
        
        try:
            popt, _ = curve_fit(
                exp_decay, t, segment,
                p0=[a0, tau0, c0],
                bounds=([-np.inf, 1, -np.inf], [np.inf, len(segment) * 10, np.inf]),
                maxfev=5000
            )
            
            # Subtract exponential from full template for this channel
            t_full = np.arange(n_time)
            # Use popt to generate full decay
            # Wait, the user code is slightly loose here. 
            # It fits on 't' which is fit_end-fit_start. 
            # It needs to extrapolate to t_full?
            # User code: exp_fit = exp_decay(t_full, popt[0], popt[1], 0)
            # This implicitly assumes t=0 is aligned to template[fit_start]?
            # Actually, t in curve_fit starts at 0 (scan index 0).
            # So if we use t_full, we are modeling decay from start of template window.
            # But we optimized params for the segment. 
            # If the segment starts at t=fit_start, then decay logic is shifted.
            # Let's assume standard exponential model: y(t) = A*exp(-t/tau).
            # The user code fits to `segment` vs `t`.
            # Then predicts `t_full`.
            # This is fine if we assume the decay "starts" (t=0) at the beginning of the template window (or wherever t_full=0 is). 
            # But the segment fit starts later. 
            # I will trust the user code logic for now.
            
            exp_fit = exp_decay(t_full, popt[0], popt[1], 0)  # No offset
            template_corrected[ch, :] -= exp_fit
            
        except (RuntimeError, ValueError):
            # If fit fails, just demean
            template_corrected[ch, :] -= np.mean(template[ch, -int(50 * fs / 1000):])
    
    return template_corrected


def apply_blanking(
    traces: np.ndarray,
    stim_indices: np.ndarray,
    fs: float,
    blank_pre_ms: float = 5.0,
    blank_post_ms: float = 120.0,
    taper_ms: float = 5.0,
    mode: str = 'interp') -> np.ndarray:
    """
    Apply blanking (Interpolation or Zeroing).
    mode: 'interp' (linear), 'zero', or 'copy_baseline' (copy pre-stim activity).
    """
    n_channels, n_samples = traces.shape
    traces_blanked = traces.copy()
    
    blank_pre = int(blank_pre_ms * fs / 1000)
    blank_post = int(blank_post_ms * fs / 1000)
    
    # Pre-calculate copy offset for 'copy_baseline'
    # Use window equal to blanking duration immediately preceding the blanking window.
    # Blanking window = [stim-pre, stim+post]
    # Source window = [stim-pre - duration, stim-pre]
    blank_duration_samps = blank_pre + blank_post
    
    taper_samples = int(taper_ms * fs / 1000) if mode == 'interp' else 0
    
    for stim_idx in stim_indices:
        start = stim_idx - blank_pre
        end = stim_idx + blank_post
        
        # Bounds check
        if start < 0 or end >= n_samples:
            continue
            
        if mode == 'copy_baseline':
            # Copy preceding usage
            src_start = start - blank_duration_samps
            src_end = start
            
            if src_start < 0:
                # Not enough baseline, fall back to zero or interp? 
                # Let's just zero it or repeat the available bit.
                traces_blanked[:, start:end] = 0.0
            else:
                traces_blanked[:, start:end] = traces_blanked[:, src_start:src_end]
            continue
            
        if mode == 'zero':
            # Strict zeroing
            s_safe = max(0, start)
            e_safe = min(n_samples, end)
            traces_blanked[:, s_safe:e_safe] = 0.0
            continue

        # Get anchor points (with small buffer for stability)
        pre_anchor_start = max(0, start - taper_samples)
        post_anchor_end = min(n_samples, end + taper_samples)
        
        # Use median of anchor regions for stability
        val_pre = np.median(traces_blanked[:, pre_anchor_start:start], axis=1)
        val_post = np.median(traces_blanked[:, end:post_anchor_end], axis=1)
        
        # Linear interpolation
        n_pts = end - start
        interp = np.linspace(val_pre, val_post, n_pts).T  # (n_channels, n_pts)
        
        # Apply with cosine taper at edges
        if taper_samples > 0:
            taper_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_samples)))
            taper_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, taper_samples)))
            
            # Blend at start
            blend_start = start
            blend_end = min(start + taper_samples, end)
            blend_len = blend_end - blend_start
            
            if blend_len > 0:
                original = traces_blanked[:, blend_start:blend_end]
                interpolated = interp[:, :blend_len]
                traces_blanked[:, blend_start:blend_end] = (
                    original * (1 - taper_in[:blend_len]) + 
                    interpolated * taper_in[:blend_len]
                )
            
            # Middle section (full interpolation)
            if blend_end < end - taper_samples:
                traces_blanked[:, blend_end:end-taper_samples] = interp[:, blend_len:-taper_samples]
            
            # Blend at end
            blend_start_out = max(start + taper_samples, end - taper_samples)
            if blend_start_out < end:
                out_len = end - blend_start_out
                original = traces_blanked[:, blend_start_out:end]
                interpolated = interp[:, -out_len:]
                traces_blanked[:, blend_start_out:end] = (
                    interpolated * taper_out[-out_len:] + 
                    original * (1 - taper_out[-out_len:])
                )
        else:
            traces_blanked[:, start:end] = interp
    
    return traces_blanked

def clean_epochs(epochs, fs, pre_samps=500, blank_post_ms=105.0):
    """
    Per-trial cleaning:
    1. Baseline correction (pre-stim mean).
    2. Exponential Recovery Removal (Fit & Subtract).
    """
    n_trials, n_ch, n_time = epochs.shape
    cleaned = epochs.copy()
    
    # 1. Baseline Correction (Pre-Stim)
    # Assume stim is at pre_samps.
    # Subtract mean of pre-stim window
    if pre_samps > 0:
        baseline = np.mean(cleaned[:, :, :pre_samps], axis=2, keepdims=True)
        cleaned -= baseline
        
    # 2. Exponential Removal
    # Fit window: starts after blanking (+buffer).
    # Blanking ends at 105ms. Let's start fitting at 110ms.
    fit_start_ms = blank_post_ms + 5.0 
    fit_start_idx = pre_samps + int(fit_start_ms * fs / 1000)
    
    if fit_start_idx >= n_time:
        return cleaned

    # Define Exponential Function
    def exp_decay(t, a, tau, c):
        return a * np.exp(-t / tau) + c
        
    t_fit = np.arange(n_time - fit_start_idx)
    
    print(f"    [Cleaning] Fitting Exponentials (Trial-by-Trial)...")
    
    # Optimize: Loop is slow. But robust.
    # We can try to parallelize or just live with it.
    # For 50 trials * 128 ch = 6400 fits. ~5-10s.
    from scipy.optimize import curve_fit, OptimizeWarning
    import warnings
    warnings.simplefilter('ignore', OptimizeWarning)
    warnings.simplefilter('ignore', RuntimeWarning)

    for i in range(n_trials):
        for c in range(n_ch):
            y_segment = cleaned[i, c, fit_start_idx:]
            
            # Initial Guess
            a0 = y_segment[0] - y_segment[-1]
            tau0 = len(y_segment) / 4.0
            c0 = y_segment[-1]
            
            try:
                popt, _ = curve_fit(
                    exp_decay, t_fit, y_segment,
                    p0=[a0, tau0, c0],
                    bounds=([-np.inf, 1.0, -np.inf], [np.inf, n_time*2, np.inf]),
                    maxfev=1000
                )
                
                # Subtract fit from the post-blank region
                # Note: We only subtract from fit_start onwards?
                # Or do we extrapolate back to the blanking end?
                # Usually we subtract the recovery tail.
                fit_curve = exp_decay(t_fit, *popt)
                cleaned[i, c, fit_start_idx:] -= fit_curve
                
            except Exception:
                pass # variable usage is implicit/handled by pass
                
    return cleaned

def filter_epochs(epochs, fs, low, high, order=4):
    """
    Apply bandpass filter to array of epochs (n_trials, n_ch, n_time).
    """
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    sos = signal.butter(order, [low_norm, high_norm], btype='band', output='sos')
    return signal.sosfiltfilt(sos, epochs, axis=2) # Time axis is 2

    
def apply_bandpass(traces, fs, lowcut, highcut, order=FILTER_ORDER):
    """
    Apply Butterworth bandpass filter.
    traces: (n_channels, n_samples)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return signal.sosfiltfilt(sos, traces, axis=1)

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


# --- Debug Plotting ---
def plot_debug_step(rec, step_name, stim_indices, pre_ms=50, post_ms=200, save_dir=None):
    """
    Plots a snippet of data around the first stimulus event for all channels.
    """
    if len(stim_indices) == 0: return
    
    # Use first stimulus
    stim_idx = stim_indices[0]
    fs = rec.get_sampling_frequency()
    
    start_frame = int(stim_idx - (pre_ms * fs / 1000.0))
    end_frame = int(stim_idx + (post_ms * fs / 1000.0))
    
    # Safety Check
    if start_frame < 0: start_frame = 0
    if end_frame > rec.get_num_samples(): end_frame = rec.get_num_samples()
    
    traces = rec.get_traces(start_frame=start_frame, end_frame=end_frame, return_scaled=True)
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
    blank_pre_ms: float = 5.0,
    blank_post_ms: float = 120.0,
    target_fs: float = 1000.0,
    curr_sess_name: str = "unknown") -> si.BaseRecording:
    """
    Custom pipeline with Manual Blanking Class.
    """
    # Debug Directory
    DEBUG_DIR = OUT_BASE / "figures" / "debug_pipeline" / curr_sess_name
    
    # 0. Raw
    plot_debug_step(recording, "0_Raw", stim_indices_native, save_dir=DEBUG_DIR)
    
    # 1. Blanking (Custom) - On RAW Data (30kHz)
    # Window: -5ms to +105ms
    # Strategy: Copy Baseline (-115ms to -5ms)
    if len(stim_indices_native) > 0:
        print(f"    [SI Pipeline] Custom Blanking (Copy Baseline) ({len(stim_indices_native)} events, -{blank_pre_ms}ms to +{blank_post_ms}ms) on RAW...")
        rec_blanked = BlankingRecording(
            recording,
            stim_indices_native,
            pre_ms=blank_pre_ms,
            post_ms=blank_post_ms,
            mode='copy_baseline'
        )
        
        # Debug Plot: Step 1 = Blanked
        plot_debug_step(rec_blanked, "1_Blanked_30k", stim_indices_native, pre_ms=50, post_ms=200, save_dir=DEBUG_DIR)
    else:
        rec_blanked = recording
        
    # 2. Bandpass Filter (0.5 - 500 Hz)
    print("    [SI Pipeline] Bandpass Filtering (0.5 - 500 Hz)...")
    rec_filtered = spre.bandpass_filter(rec_blanked, freq_min=0.5, freq_max=500.0, dtype='float32')
    
    # Debug Plot: Step 2 = Filtered
    plot_debug_step(rec_filtered, "2_Filtered_30k", stim_indices_native, pre_ms=50, post_ms=200, save_dir=DEBUG_DIR)
    
    # 3. Resampling (30k -> 1k)
    print(f"    [SI Pipeline] Resampling to {target_fs} Hz...")
    rec_resampled = spre.resample(rec_filtered, resample_rate=target_fs)
    
    # Scale stim indices for debug plotting of resampled data
    fs_native = recording.get_sampling_frequency() # Get native FS from original recording
    stim_indices_resampled = (stim_indices_native * target_fs / fs_native).astype(int)
    
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
    print(f"[NPRW] Processing {sess_name}")

    # Geometry
    intan_geom, intan_probe_mapping = get_intan_geometry()
    
    nprw_probe = Probe(ndim=2)
    nprw_probe.set_contacts(positions=np.c_[intan_geom["x"], intan_geom["y"]], shape_params={"width": 12.0})
    nprw_probe.set_device_channel_indices(intan_probe_mapping)
    
    # 1. Load Resulting Stim Times First (Need for Blanking)
    stim_ext_arrays = rcp.extract_stim_npz(sess=sess_path, out_dir=NPRW_AUX_DATA, stim_stream_name=STIM_STREAM, chanmap_perm=intan_probe_mapping)
    block_bounds = stim_ext_arrays.get("block_bounds_samples")
    stim_start_samps_native = block_bounds[:, 0] if block_bounds.size else np.array([])
    
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
        blank_pre_ms=5.0, 
        blank_post_ms=105.0, # Updated to 105ms
        target_fs=TARGET_FS,
        curr_sess_name=sess_name
    )
    
    # 4. Get Traces (Eager Load of Processed Data)
    print("  [NPRW] Loading processed traces into memory...")
    traces = rec_proc.get_traces(return_in_uV=True).T
    
    # 5. Get Traces (Raw/Unblanked) for Comparison
    # We re-run the pipeline but SKIP the blanking step by passing empty stim indices
    # This gives us Filtered + Resampled data without the artifact removal
    print("  [NPRW] Building RAW (Unblanked) pipeline...")
    rec_proc_raw = build_lfp_preprocessing_pipeline(
        rec_reordered, 
        stim_indices_native=np.array([]), # Skip blanking
        blank_pre_ms=5.0, 
        blank_post_ms=105.0,
        target_fs=TARGET_FS,
        curr_sess_name=sess_name
    )
    print("  [NPRW] Loading RAW traces...")
    traces_raw = rec_proc_raw.get_traces(return_in_uV=True).T
    
    n_channels, n_samples = traces.shape
    
    # Stim times to ms
    stim_ms = stim_start_samps_native * 1000.0 / fs_native
    
    # 1. DISABLE GLOBAL CMR per user request
    print("  [INFO] Global Common Median Reference (CMR) DISABLED.")
    
    # Construct Timebase
    t_ms = np.arange(n_samples, dtype=float) * 1000.0 / TARGET_FS
    
    # 6. Epoching: Extract Broadband (-500 to +1000)
    # We epoch first, then clean.
    WIN_EPOCH = (-500.0, 1000.0)
    
    # Sub-windows for final saving
    WIN_PRE = (-500.0, -5.0)
    WIN_POST = (105.0, 605.0) # 500ms post-blank
    WIN_COMP = (-50.0, 200.0)
    
    results = {}
    
    print(f"  Segmenting Broadband Epochs ({WIN_EPOCH} ms)...", flush=True)
    segs_bb, _, rel_t_epoch, valid_stim = rcp.extract_peristim_segments(
        rate_hz=traces.astype(np.float32), 
        counts=None,
        t_ms=t_ms,
        stim_ms=stim_ms,
        win_ms=WIN_EPOCH
    )
    
    if valid_stim is None:
        print(f"  [WARN] No valid segments found for {sess_name}. Skipping.")
        return
        
    # 7. Per-Trial Cleaning
    print(f"  Cleaning Epochs (Baseline + Exp Removal)...")
    segs_bb_clean = clean_epochs(segs_bb, TARGET_FS)
    
    # Helper to slice extracted epoch into sub-windows
    def slice_epoch(data, t_axis, target_win):
        # find indices in t_axis
        mask = (t_axis >= target_win[0]) & (t_axis < target_win[1])
        return data[:, :, mask], t_axis[mask]
        
    # Slice Broadband Results
    results['broadband_pre'], t_pre = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_PRE)
    results['broadband_post'], t_post = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_POST)
    results['broadband_comp'], t_comp = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_COMP)
    
    # 8. Raw Comparison (No Cleaning)
    print(f"  Segmenting Raw (Unblanked)...", flush=True)
    
    # Ensure valid_stim is integer indices
    if valid_stim is not None:
        valid_stim = np.array(valid_stim, dtype=int)
        
    segs_raw, _, _, _ = rcp.extract_peristim_segments(
        rate_hz=traces_raw.astype(np.float32), 
        counts=None,
        t_ms=t_ms,
        stim_ms=stim_ms[valid_stim], # Use valid subset
        win_ms=WIN_EPOCH
    )
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
        **results
    }
    
    print(f"  [DEBUG] Keys to save: {list(save_dict.keys())}", flush=True)
    
    np.savez_compressed(out_npz, **save_dict)
    print(f"  Saved -> {out_npz}")
    
    del rec, rec_reordered, rec_proc, traces, results
    gc.collect()


def process_utah_session(sess_name: str, br_idx: int, shift_ms: float, fs_intan: float, shift_sample: float):
    # Find BR session folder
    # Helper from utils? rcp.list_br_sessions finds folders.
    # We need to match br_idx to folder.
    # Usually folder name ends in _<03d>
    
    matches = [p for p in BR_ROOT.iterdir() if p.is_dir() and f"_{br_idx:03d}" in p.name]
    if not matches:
        # Fallback: check if it's just the folder name logic
        # Some sessions might be named differently. 
        # Try finding by date if possible, but br_idx is most reliable suffix
        print(f"[UA] No session folder found for BR {br_idx:03d}")
        return
    sess_path = matches[0]
    
    out_npz = UA_LFP_CKPT_ROOT / f"aligned_lfp__{sess_path.name}.npz"
    # if out_npz.exists():
    #     print(f"[UA] Skipping {sess_path.name}, output exists.")
    #     return
        
    print(f"[UA] Processing {sess_path.name}")
    
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
        blank_pre_ms=5.0, 
        blank_post_ms=105.0,
        target_fs=TARGET_FS,
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
    # Let's align with NPRW:
    # traces_raw in NPRW was filtered raw.
    # traces_unref was not captured in new NPRW block explicitly? Ah, I removed it?
    # In NPRW I didn't assign traces_unref.    # 4. Get Traces (Raw/Unblanked) for Comparison
    print("  [UA] Building RAW (Unblanked) pipeline...")
    rec_proc_raw = build_lfp_preprocessing_pipeline(
        rec_mapped, 
        stim_indices_native=np.array([]), # Skip blanking
        blank_pre_ms=5.0, 
        blank_post_ms=105.0,
        target_fs=TARGET_FS,
        curr_sess_name=sess_path.name
    )
    print("  [UA] Loading RAW traces...")
    traces_raw = rec_proc_raw.get_traces(return_in_uV=True).T

    # 1. DISABLE GLOBAL CMR per user request
    traces_unref = traces_raw.copy() 
    print("  [UA] Global Common Median Reference (CMR) DISABLED.")

    # Segment
    t_ms = np.arange(n_samples, dtype=float) * 1000.0 / TARGET_FS
    
    # 6. Epoching
    WIN_EPOCH = (-500.0, 1000.0)
    WIN_PRE = (-500.0, -5.0)
    WIN_POST = (105.0, 605.0)
    WIN_COMP = (-50.0, 200.0)

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
    results['broadband_comp'], t_comp = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_COMP)
    
    results['broadband_raw_comp'] = None 
    results['broadband_unref_comp'] = None
    
    # 8. Raw Comparison
    print(f"  Segmenting Raw (Unblanked)...", flush=True)
    
    if valid_stim is not None:
        valid_stim = np.array(valid_stim, dtype=int)

    segs_raw, _, _, _ = rcp.extract_peristim_segments(
        rate_hz=traces_raw.astype(np.float32), 
        counts=None,
        t_ms=t_ms,
        stim_ms=stim_ms_ua[valid_stim],
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
