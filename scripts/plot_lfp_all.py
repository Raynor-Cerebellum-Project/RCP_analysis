import numpy as np
import matplotlib.pyplot as plt
import RCP_analysis.python.functions.config_loading as cfg
import RCP_analysis.python.functions.utils as rcp
import RCP_analysis.python.functions.br_preproc as br_preproc
import RCP_analysis.python.functions.impedance_utils as imp_utils
import warnings
from pathlib import Path
from scipy import signal, stats
import pandas as pd
import csv
import scipy.io
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

# Define output directory from config
NPRW_LFP_DIR = cfg.OUT_BASE / "checkpoints" / "NPRW_LFP"
UA_LFP_DIR = cfg.OUT_BASE / "checkpoints" / "UA_LFP"

# Probe Geometry Map for CSD
PROBE_MAP_FILE = cfg.REPO_ROOT / "config" / "ImecPrimateStimRec128_BT_corrected_091525.mat"

NPRW_FIG_DIR = cfg.OUT_BASE / "figures" / "NPRW_LFP"
UA_FIG_DIR = cfg.OUT_BASE / "figures" / "UA_LFP"

NPRW_FIG_DIR.mkdir(parents=True, exist_ok=True)
UA_FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Global Plotting Configuration ---
FIGURE_CONFIG = {
    'process_original': False,          # The existing heatmap/trace plot
    'process_power_spectra': False,      # PSD (Pre vs Post)
    'process_power_vs_time': False,      # Band power vs time (all arrays overlaid)
    'process_band_stats': False,         # (Stage 2: New) Band Power Stats (Pre vs Post)
    'process_inter_array_coherence': False, # 4x4 inter-array coherence grid (Utah only)
    'process_csd': False,                # Current Source Density (CSD) Analysis
    'process_single_channel_traces': False, # Isolated multi-band trace plot for one channel 
    'process_spectrograms': True,            # Per-channel ERSP spectrograms (Utah only)
}

y_scaler = 1

PLOT_CONFIG = {
    "single_channel_target": 168, # Target channel ID to isolate for the single-channel trace figure
    
    # Opacity for individual trial traces
    "trace_alpha": 0.4,
    
    # Colormap for Heatmaps
    "heatmap_cmap": "RdBu_r",
    
    # Heatmap Scales (uV): Set value to enforce fixed scale, or None for dynamic (percentile)
    "scales": {
        'broadband': 30.0*y_scaler,
        'delta': None,
        'theta': 100.0*y_scaler,
        'alpha': 50.0*y_scaler,
        'beta': 50.0*y_scaler,
        'low_gamma': 30.0*y_scaler,
        'high_gamma': 30.0*y_scaler
    },
    
    # Trace Y-Axis Scales (uV): Set value for fixed +/- limit, or None for dynamic
    "trace_scales": {
        'broadband': 200, 
        'delta': 100,
        'theta': 100,
        'alpha': 100,
        'beta': 100,
        'low_gamma': 100,
        'high_gamma': 100
    },

    # X-Axis Limits (ms): Set tuple (min, max) or None for default
    "x_limits": {
        "pre": (-400, -5), 
        "post": (101, 600)
    },

    # Spectrogram Settings
    "spectrogram": {
        "stft_nperseg": 258,        # Window size in samples. Increased from 128 for sharper frequency bands (3.9 Hz res)
        "stft_overlap_frac": 0.98,  # Increased overlap for smoother time steps (~12ms steps)
        "window_type": "hann",      # Tapering window applied to each segment
        "freq_min": 0.5,            # Hz — lower bound for frequency axis
        "freq_max": 60,             # Hz — upper bound for frequency axis
        "log_freq": False,          # If True, plot frequency axis on log scale
        "vmin_db": -5,              # dB — colormap floor 
        "vmax_db": 5,               # dB — colormap ceiling
        "time_range": (-400, 500),  # ms — display time window
        "baseline_window": (-375, -125), # ms (start, end). Moved closer to stim for stability
        "normalization": "baseline", # 'baseline' (ERSP, dB relative to pre-stim)
        "per_trial_norm": False,    # Match paper: average then normalize
        "cmap": "RdBu_r",           # Changed from jet to RdBu_r for proper ERSP visualization (blue=suppression, red=enhancement)
        "shading": "nearest",       # 'nearest' (fast) or 'gouraud' (smooth, slow) or 'flat' (requires edge coords)
        "dpi": 150,                 # Output DPI for saved figures
    }
}
"""
SPECTROGRAM SETTINGS REFERENCE
==============================

stft_nperseg (int, samples)
    Window length for the Short-Time FFT. Controls the time-frequency tradeoff:
      64  → freq_res = 15.6 Hz, time_smear = 64 ms   (fast transients, blurry bands)
      128 → freq_res =  7.8 Hz, time_smear = 128 ms  (balanced)
      256 → freq_res =  3.9 Hz, time_smear = 256 ms  (sharp bands, blurry timing)
    Frequency resolution = fs / nperseg.  Time resolution ≈ nperseg / fs.

stft_overlap_frac (float, 0.0 – 0.99)
    Fraction of each window overlapping the next. Controls time-bin density:
      0.50 → step = nperseg/2    (fewest bins, fastest render)
      0.75 → step = nperseg/4    (smooth)
      0.90 → step = nperseg/10   (very smooth, slow render)
    Does NOT improve actual time resolution — only interpolates between windows.

window_type (str)
    Tapering function applied to each segment before FFT.
      'hann'     — good all-around, low spectral leakage (DEFAULT)
      'hamming'  — similar to Hann, slightly less side-lobe suppression
      'blackman' — excellent leakage suppression, wider main lobe
      'kaiser'   — tunable via beta parameter (not exposed here)
      'boxcar'   — no taper (rectangular), maximum leakage — avoid

freq_min / freq_max (float, Hz)
    Frequency display bounds. The STFT computes up to Nyquist (fs/2 = 500 Hz)
    but only the selected range is displayed.

log_freq (bool)
    If True, the frequency axis is displayed on a log scale, giving more visual
    space to low-frequency bands (delta, theta, alpha). Useful when nperseg is
    large enough to resolve them.

vmin_db / vmax_db (float, dB)
    Colormap saturation limits. Values outside this range clip to the edge color.
      Too wide (e.g. ±30)  → washed out, actual modulations invisible
      Too tight (e.g. ±2)  → everything saturates, lose dynamic range
      Typical range: ±5 to ±15 dB for ERSP

time_range (tuple of float, ms)
    (start_ms, end_ms) — controls which portion of the epoch is displayed.
    Data outside this window is computed but not plotted.

baseline_window (tuple, ms)
    (start_ms, end_ms) — time window used to compute the baseline power for
    minimum: -500ms
    ERSP normalization. Each channel's power in this window becomes 0 dB.
    Use None for either bound to default to epoch start or 0 ms.

normalization (str)
    Controls how power values are expressed:
      'baseline' — ERSP: dB relative to pre-stim baseline (DEFAULT). Shows modulation.
                   Use with diverging colormaps (RdBu_r) and vmin/vmax like ±5 to ±15.
      'absolute' — Raw power: 10*log10(|X|²). Shows overall spectral content.
                   Use with sequential colormaps (viridis) and vmin/vmax like -30 to 10.

per_trial_norm (bool)  [only applies when normalization='baseline']
    False (default): average power across trials, THEN normalize to baseline.
        → Standard ERSP. Cleaner but sensitive to outlier trials.
    True: normalize each trial to baseline, THEN average the dB values.
        → More robust to outlier trials. Shows consistent modulations.

cmap (str)
    Matplotlib colormap name.
      'RdBu_r'   — diverging: blue=suppression, white=0, red=enhancement (BEST for ERSP)
      'jet'      — rainbow, high contrast but perceptually non-uniform
      'viridis'  — sequential, good for absolute power but bad for ERSP
      'seismic'  — diverging alternative to RdBu_r

shading (str)
    Rendering mode for pcolormesh:
      'nearest' — fast, each cell maps to nearest data point (recommended)
      'flat'    — fast, but requires coordinate arrays one larger than data
      'gouraud' — smooth Gouraud interpolation, much slower render

dpi (int)
    Resolution of saved PNG files. 150 is good for review, 300 for publication.
"""


def get_representative_channel(data_array):
    """
    Find a channel index that has valid, non-flat data.
    data_array: (n_trials, n_ch, n_time)
    """
    if data_array is None:
        return 0
        
    n_ch = data_array.shape[1]
    
    # Try middle first
    mid = n_ch // 2
    
    # Scan outward from middle
    offsets = np.arange(n_ch)
    # Interleave 0, 1, -1, 2, -2...
    scan_order = []
    for i in range(n_ch):
        if mid + i < n_ch: scan_order.append(mid + i)
        if i > 0 and mid - i >= 0: scan_order.append(mid - i)
        
    for ch in scan_order:
        # Check if channel has data transparency
        traces = data_array[:, ch, :]
        if np.all(np.isnan(traces)):
             continue
        if np.nanstd(traces) < 1e-6: # Flat line
             continue
        return ch
        
    return mid # Fallback

def calculate_psd(data, fs):
    """
    Calculate Power Spectral Density averaged across trials.
    Returns: freq, psd_mean (n_ch, n_freq), psd_sem (n_ch, n_freq)
    data: (n_trials, n_ch, n_time)
    """
    n_trials, n_ch, n_time = data.shape
    
    # Calculate PSD for each trial/channel
    # nperseg = length of window. If too short, reduce it.
    nperseg = min(n_time, 256)
    
    f, Pxx = signal.welch(data, fs=fs, nperseg=nperseg, axis=2)
    
    # Valid Pxx: (n_trials, n_ch, n_freq)
    # Average across trials
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        psd_mean = np.nanmean(Pxx, axis=0)
        psd_sem = np.nanstd(Pxx, axis=0) / np.sqrt(n_trials)
    
    return f, psd_mean, psd_sem

def calculate_coherence(data, rep_ch_idx, fs):
    """
    Calculate Magnitude Squared Coherence between rep_ch and all other channels.
    Returns: freq, coh_mean (n_ch, n_freq) - averaged across trials
    """
    n_trials, n_ch, n_time = data.shape
    nperseg = min(n_time, 256)
    
    rep_data = data[:, rep_ch_idx, :] # (n_trials, n_time)
    
    all_coh = []
    
    # Iterate channels
    for ch in range(n_ch):
        target_data = data[:, ch, :]
        
        # Coherence per trial? Or coherence of concatenated trials?
        # Usually better to do coherence of the concatenation if stationary, 
        # or avg coherence across trials. 
        # signal.coherence handles raw signals.
        
        # We will calculate coherence per trial and average, to match trial structure
        f, Cxy = signal.coherence(rep_data, target_data, fs=fs, nperseg=nperseg, axis=1)
        # Cxy: (n_trials, n_freq)
        
        mean_cxy = np.nanmean(Cxy, axis=0)
        all_coh.append(mean_cxy)
        
    all_coh = np.stack(all_coh, axis=0) # (n_ch, n_freq)
    return f, all_coh

def calculate_itpc(data, fs):
    """
    Inter-Trial Phase Coherence.
    Returns: freq, itpc (n_ch, n_freq)
    Using Morlet Wavelets or Hilbert? 
    Given the short windows, let's use FFT-based ITPC (Phase consistency of Fourier components).
    """
    n_trials, n_ch, n_time = data.shape
    
    # FFT
    # axis=2 is time
    fft_res = np.fft.rfft(data, axis=2)
    # Frequencies
    freqs = np.fft.rfftfreq(n_time, d=1/fs)
    
    # Normalize to unit length (phase only)
    # Add eps to avoid divide by zero
    mags = np.abs(fft_res)
    mags[mags == 0] = 1e-10
    
    norm_fft = fft_res / mags
    
    # ITPC = |mean vector length across trials|
    itpc = np.abs(np.nanmean(norm_fft, axis=0)) # (n_ch, n_freq)
    
    return freqs, itpc

def calculate_tf_itpc(data, fs, nperseg=64, noverlap=50):
    """
    Calculate Time-Frequency Inter-Trial Phase Coherence using Short-Time FFT.
    Returns: f, t, itpc_matrix (n_ch, n_freq, n_time_bins)
    data: (n_trials, n_ch, n_time)
    """
    n_trials, n_ch, n_time = data.shape
    
    # Check if data is long enough for STFT
    if n_time < nperseg:
        nperseg = n_time // 2
        noverlap = nperseg // 2
        
    # We need to perform STFT on each trial and channel
    # This can be heavy, so loop carefully or vectorize if possible.
    # scipy.signal.stft can handle (..., time)
    
    # stft returns (f, t, Zxx) where Zxx is (..., n_freq, n_time_bins)
    f, t, Zxx = signal.stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=2)
    # Zxx shape: (n_trials, n_ch, n_freq, n_time_bins)
    
    # Normalize magnitude to 1 over trials (phase only)
    mags = np.abs(Zxx)
    mags[mags == 0] = 1e-10
    Z_norm = Zxx / mags
    
    # ITPC = Magnitude of mean vector across trials
    itpc_matrix = np.abs(np.nanmean(Z_norm, axis=0)) # (n_ch, n_freq, n_time)
    
    return f, t, itpc_matrix

def calculate_ersp(data_pre, data_post, fs, nperseg=128, noverlap=100):
    """
    Calculate Event-Related Spectral Perturbation (ERSP) in dB.
    Returns: f, t_post, ersp_matrix (n_ch, n_freq, n_time_bins)
    normalized to baseline (data_pre).
    """
    # 1. Calculate Baseline Power (Pre)
    # Use same STFT parameters
    f_pre, t_pre, Zxx_pre = signal.stft(data_pre, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=2)
    # Power = |Zxx|^2
    P_pre = np.abs(Zxx_pre)**2 
    # Mean Baseline Power per freq across time and trials (or just trials? usually across time too for true baseline)
    # Shape: (n_trials, n_ch, n_freq, n_time)
    # We want mean scalar per freq/ch to normalize.
    mean_baseline = np.nanmean(P_pre, axis=(0, 3)) # (n_ch, n_freq)
    # Avoid zero
    mean_baseline[mean_baseline == 0] = 1e-10
    
    # 2. Calculate Post Power
    f_post, t_post, Zxx_post = signal.stft(data_post, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=2)
    P_post = np.abs(Zxx_post)**2
    # Mean across trials for the "Evoked+Induced" response? 
    # ERSP is usually: Mean(Power_trial) / Baseline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_post_power = np.nanmean(P_post, axis=0) # (n_ch, n_freq, n_time)
    
    # 3. Normalize (dB): 10 * log10(Post / Baseline)
    # Broadcast baseline across time
    ersp_matrix = 10 * np.log10(mean_post_power / mean_baseline[:, :, None])
    
    return f_post, t_post, ersp_matrix

def compute_ersp_spectrogram(broadband_full, t_full_ms, fs, spec_cfg):
    """
    Compute per-channel ERSP spectrogram from broadband_full data.
    
    Args:
        broadband_full: (n_trials, n_ch, n_time) broadband LFP epochs
        t_full_ms: (n_time,) time axis in ms
        fs: sampling rate (Hz)
        spec_cfg: dict — see SPECTROGRAM SETTINGS REFERENCE above
    
    Returns:
        ersp: (n_ch, n_freq_crop, n_time_crop) power in dB (baseline-normalized or absolute)
        freqs: (n_freq_crop,) frequency axis
        t_bins_ms: (n_time_crop,) time axis in ms
    """
    nperseg = spec_cfg['stft_nperseg']
    noverlap = int(nperseg * spec_cfg['stft_overlap_frac'])
    window_type = spec_cfg.get('window_type', 'hann')
    freq_min = spec_cfg.get('freq_min', 0)
    freq_max = spec_cfg['freq_max']
    t_range = spec_cfg['time_range']
    per_trial_norm = spec_cfg.get('per_trial_norm', False)
    
    n_trials, n_ch, n_time = broadband_full.shape
    
    # 1. Compute STFT: scipy.signal.stft along time axis
    f, t_bins, Zxx = signal.stft(broadband_full, fs=fs, nperseg=nperseg,
                                  noverlap=noverlap, window=window_type, axis=2)
    # Zxx shape: (n_trials, n_ch, n_freq, n_time_bins)
    
    # 2. Power per trial
    P = np.abs(Zxx) ** 2  # (n_trials, n_ch, n_freq, n_time_bins)
    
    # 3. Convert STFT time bins (in seconds from start of segment) to ms
    #    t_full_ms[0] is the epoch start time (e.g. -500 ms)
    t_bins_ms = t_bins * 1000.0 + t_full_ms[0]
    
    # 4. Compute baseline mask
    bl_win = spec_cfg.get('baseline_window', (None, 0))
    bl_start = bl_win[0] if bl_win[0] is not None else t_bins_ms[0]
    bl_end = bl_win[1] if bl_win[1] is not None else 0
    baseline_mask = (t_bins_ms >= bl_start) & (t_bins_ms <= bl_end)
    if not np.any(baseline_mask):
        print(f"  [WARN] No time bins in baseline window ({bl_start}, {bl_end}). Using first 25% of bins.")
        n_bl = max(1, len(t_bins_ms) // 4)
        baseline_mask = np.zeros(len(t_bins_ms), dtype=bool)
        baseline_mask[:n_bl] = True
    
    # 5. Normalize
    normalization = spec_cfg.get('normalization', 'baseline')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        if normalization == 'absolute':
            # Absolute power in dB: 10 * log10(mean_power)
            P_mean = np.nanmean(P, axis=0)  # (n_ch, n_freq, n_time_bins)
            P_mean[P_mean == 0] = 1e-10
            ersp = 10.0 * np.log10(P_mean)
        elif per_trial_norm:
            # Normalize each trial individually, then average dB values
            # baseline per trial: (n_trials, n_ch, n_freq)
            bl_power = np.nanmean(P[:, :, :, baseline_mask], axis=3)
            bl_power[bl_power == 0] = 1e-10
            ersp_trials = 10.0 * np.log10(P / bl_power[:, :, :, None])
            ersp = np.nanmean(ersp_trials, axis=0)  # (n_ch, n_freq, n_time_bins)
        else:
            # Average power across trials, then normalize (standard ERSP)
            P_mean = np.nanmean(P, axis=0)  # (n_ch, n_freq, n_time_bins)
            baseline_power = np.nanmean(P_mean[:, :, baseline_mask], axis=2)  # (n_ch, n_freq)
            baseline_power[baseline_power == 0] = 1e-10
            ersp = 10.0 * np.log10(P_mean / baseline_power[:, :, None])
    
    # 6. Crop to requested frequency and time range
    freq_mask = (f >= freq_min) & (f <= freq_max)
    freqs = f[freq_mask]
    ersp = ersp[:, freq_mask, :]
    
    time_mask = (t_bins_ms >= t_range[0]) & (t_bins_ms <= t_range[1])
    t_bins_ms = t_bins_ms[time_mask]
    ersp = ersp[:, :, time_mask]
    
    return ersp, freqs, t_bins_ms

def calculate_band_power_stats(data_pre, data_post, fs):
    """
    Compare band power between Pre and Post using Paired T-Test.
    Returns: Dict of results per band
    """
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12), # Match analyze_lfp_bands range
        'Beta': (12, 25),
        'Low Gamma': (25, 60),
        'High Gamma': (60, 120)
    }
    
    results = {}
    
    # PSD calculation for spectral estimation
    # Calculate PSD per trial
    nperseg = min(data_pre.shape[2], 256)
    f, Pxx_pre = signal.welch(data_pre, fs=fs, nperseg=nperseg, axis=2) # (N, Ch, F)
    _, Pxx_post = signal.welch(data_post, fs=fs, nperseg=nperseg, axis=2)
    
    for band_name, (f_min, f_max) in bands.items():
        # Find freq indices
        idx_band = np.where((f >= f_min) & (f <= f_max))[0]
        
        if len(idx_band) == 0:
            results[band_name] = None
            continue
            
        # Mean Power in band (integrate or mean?) -> Mean for simplicity match units
        # Shape: (N, Ch)
        pow_pre = np.nanmean(Pxx_pre[:, :, idx_band], axis=2)
        pow_post = np.nanmean(Pxx_post[:, :, idx_band], axis=2)
        
        # Paired T-Test across trials, for each channel
        # ttest_rel returns statistic, pvalue arrays of shape (Ch,)
        t_stats, p_vals = stats.ttest_rel(pow_pre, pow_post, axis=0, nan_policy='omit')
        
        mean_pre = np.nanmean(pow_pre, axis=0)
        mean_post = np.nanmean(pow_post, axis=0)
        sem_pre = np.nanstd(pow_pre, axis=0) / np.sqrt(pow_pre.shape[0])
        sem_post = np.nanstd(pow_post, axis=0) / np.sqrt(pow_post.shape[0])
        
        results[band_name] = {
            't_stat': t_stats,
            'p_val': p_vals,
            'mean_pre': mean_pre,
            'mean_post': mean_post,
            'sem_pre': sem_pre,
            'sem_post': sem_post
        }
        
    return results

def add_band_annotations(ax):
    """Adds vertical spans for standard LFP bands."""
    bands = [
        (1, 4, 'pink', 'Delta'),
        (4, 8, 'purple', 'Theta'),
        (8, 12, 'green', 'Alpha'),
        (12, 25, 'yellow', 'Beta'),
        (25, 60, 'orange', 'Low Gamma'),
        (60, 120, 'red', 'High Gamma')
    ]
    
    for start, end, color, name in bands:
        ax.axvspan(start, end, color=color, alpha=0.1, label=name)
        
    # Get handles for legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Only keep band labels in a specific order if present
    
    return by_label

def calculate_csd(lfp_mean, y_coords):
    """
    Computes 1D Current Source Density (CSD) from LFP using interpolation 
    to handle irregular spacing/gaps.
    
    Parameters:
    lfp_mean: (n_ch, n_time) - Mean LFP voltage
    y_coords: (n_ch,) - Depth coordinate for each channel
    
    Returns:
    csd: (n_grid-2, n_time) - CSD values on uniform grid
    csd_depths: (n_grid-2,) - Depth coordinates for CSD rows
    """
    # 1. Sort and Unique channels by depth
    y_coords = y_coords.flatten()
    sort_idx = np.argsort(y_coords)
    sorted_lfp = lfp_mean[sort_idx, :]
    sorted_depths = y_coords[sort_idx]
    
    # Handle duplicates (multiple channels at same depth) -> Average them
    unique_depths, unique_indices = np.unique(sorted_depths, return_index=True)
    
    if len(unique_depths) < len(sorted_depths):
        # Average duplicate depths
        lfp_unique = []
        for d in unique_depths:
            idxs = np.where(sorted_depths == d)[0]
            lfp_unique.append(np.mean(sorted_lfp[idxs], axis=0))
        ls_lfp = np.array(lfp_unique)
        ls_depths = unique_depths
    else:
        ls_lfp = sorted_lfp
        ls_depths = sorted_depths

    # 2. Define Uniform Grid (to handle gaps)
    # Use 20 um spacing (standard for probes)
    dz = 20.0
    
    # Filter out NaNs from source data before interpolation
    # Filter only channels that are BAD (all NaNs or mostly NaNs). 
    # Do NOT filter if just a time-gap (column of NaNs) exists.
    # Check if a channel has valid data in at least 50% of timepoints?
    # Or just check if it's not ALL NaNs.
    valid_mask = np.any(np.isfinite(ls_lfp), axis=1)
    
    ls_lfp = ls_lfp[valid_mask]
    ls_depths = ls_depths[valid_mask]
    
    if len(ls_depths) < 2: return None, None # Cannot Calc CSD
    
    grid_y = np.arange(ls_depths[0], ls_depths[-1] + dz/2, dz) # Include end
    
    # 3. Interpolate LFP onto Grid
    # Linear interpolation is robust for gaps
    f_interp = interp1d(ls_depths, ls_lfp, kind='linear', axis=0, fill_value="extrapolate")
    lfp_grid = f_interp(grid_y)
    
    # 4. Spatial Smoothing on Grid
    # Smooth across the grid points (sigma=1 grid step = 20um)
    lfp_smoothed = gaussian_filter1d(lfp_grid, sigma=1.0, axis=0)
    
    # 5. Compute CSD (2nd derivative)
    # CSD ~ -d^2V / dz^2
    csd = -np.diff(lfp_smoothed, n=2, axis=0) / (dz**2)
    
    # Adjust depths (lost 2 edge points)
    csd_depths = grid_y[1:-1]
    
    return csd, csd_depths

def process_directory(lfp_dir, fig_dir, label="LFP"):
    if not lfp_dir.exists():
        print(f"Directory not found: {lfp_dir}")
        return

    # Find latest aligned file
    files = sorted(lfp_dir.glob("aligned_lfp__*.npz"))
    if not files:
        print(f"No aligned LFP npz files found in {lfp_dir}")
        return
    
    print(f"--- Processing {label} ({len(files)} files) ---")
    
    # Load Stimulation Summary
    summary_csv = cfg.METADATA_ROOT / "stimulation_summary.csv"
    df_summary = None
    if summary_csv.exists():
        try:
            df_summary = pd.read_csv(summary_csv)
            # Ensure columns are string/clean
            if 'Intan_Session' in df_summary.columns:
                df_summary['Intan_Session'] = df_summary['Intan_Session'].astype(str).str.strip()
            print(f"Loaded stimulation summary from {summary_csv}")
        except Exception as e:
            print(f"Warning: Failed to load stimulation summary: {e}")
    else:
        print(f"Warning: Stimulation summary not found at {summary_csv}")

    # Load Impedance Data to exclude bad channels
    bad_ch_map = imp_utils.get_session_impedances(cfg.SESSION_LOC)
    bad_ua_ids = bad_ch_map.get('utah', set())
    bad_nprw_ids = bad_ch_map.get('nprw', set())
    print(f"  [Impedance] Bad Utah channels: {len(bad_ua_ids)}, Bad NPRW channels: {len(bad_nprw_ids)}")
    if bad_ua_ids: print(f"    Utah excluded NSP IDs: {sorted(bad_ua_ids)}")
    if bad_nprw_ids: print(f"    NPRW excluded IDs: {sorted(bad_nprw_ids)}")

    # Load Probe Map for CSD (if enabled and labeled Intan)
    y_coords = None
    if FIGURE_CONFIG.get('process_csd', False) and "Intan" in label:
        if PROBE_MAP_FILE.exists():
            try:
                mat = scipy.io.loadmat(PROBE_MAP_FILE)
                if 'ycoords' in mat:
                     y_coords = mat['ycoords']
                     print(f"Loaded Probe Map for CSD: {PROBE_MAP_FILE.name}")
                else:
                     print("Warning: ycoords not found in probe map")
            except Exception as e:
                print(f"Error loading probe map: {e}")
        else:
             print(f"Warning: Probe map not found at {PROBE_MAP_FILE}")

    # Iterate through all files
    for npz_file in files:
        print(f"Loading {npz_file.name}...")
        
        try:
            data = np.load(npz_file, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            continue

        # Extract Info
        if 'fs_lfp' in data: fs = int(data['fs_lfp'])
        else: fs = 1000
            
        if 'rel_time_post' in data: t_ms = data['rel_time_post']
        elif 'rel_time_ms' in data: t_ms = data['rel_time_ms']
        else:
            try:
                n_p = data['broadband_post'].shape[2]
                t_ms = np.arange(n_p) * 1000.0 / fs
            except KeyError:
                 # If empty or corrupt
                 print(f"  [Skip] Missing broadband_post in {npz_file.name}")
                 continue
        
        # Session Name
        if 'session' in data: session = str(data['session'])
        else: session = npz_file.stem.replace("aligned_lfp__", "")
        
        # --- Detect Baseline Files ---
        is_baseline = "baseline" in npz_file.stem
        baseline_port = None
        baseline_depth = None
        
        if is_baseline:
            # Extract port and depth from filename or data
            if 'group_port' in data:
                baseline_port = str(data['group_port'])
            if 'group_depth' in data:
                baseline_depth = str(data['group_depth'])
            
            # Also try parsing from filename: aligned_lfp__baseline__Depth_43_0_port_A.npz
            if baseline_port is None or baseline_depth is None:
                stem = npz_file.stem
                if 'port_' in stem:
                    baseline_port = stem.split('port_')[-1].split('.')[0].split('_')[0]
                if 'Depth_' in stem:
                    depth_part = stem.split('Depth_')[-1].split('_port')[0]
                    baseline_depth = depth_part.replace('_', '.')
            
            # Count sessions if available
            n_sessions = len(data.get('sessions', [])) if 'sessions' in data else 1
            
            print(f"  [Baseline] Port={baseline_port}, Depth={baseline_depth}, Sessions={n_sessions}")
            
        if 'broadband_post' not in data:
            print(f"Error: 'broadband_post' key missing in {npz_file.name}. Skipping.")
            continue
            
        bb_data = data['broadband_post']
        n_trials, n_ch, n_time = bb_data.shape
        
        print(f"  Dimensions: {n_trials} trials, {n_ch} channels")

        # --- Channel Order ---
        sort_idx = np.arange(n_ch)
        
        # --- Metadata (Detailed: Lookup from Summary CSV) ---
        target_intan_session = session 
        
        active_str = "Unknown"
        freq_str = "Unknown"
        amp_str = ""
        found_row = None
        
        # 1. Try Lookup by BR Index (for Utah)
        if label == "Utah_Array" and df_summary is not None:
             try:
                parts = npz_file.stem.split('_')
                if parts[-1].isdigit(): b_idx = int(parts[-1])
                elif parts[-2].isdigit(): b_idx = int(parts[-2])
                else: b_idx = None
                
                if b_idx is not None and 'BR_Index' in df_summary.columns:
                     matches = df_summary[df_summary['BR_Index'] == b_idx]
                     if not matches.empty:
                         found_row = matches.iloc[0]
                         target_intan_session = str(found_row['Intan_Session'])
             except: pass
        
        # 2. Try Lookup by Session Name (NPRW or Fallback)
        if found_row is None and df_summary is not None:
             if 'Intan_Session' in df_summary.columns:
                 # Check exact match
                 matches = df_summary[df_summary['Intan_Session'] == session]
                 if not matches.empty:
                     found_row = matches.iloc[0]
                 else:
                     # Check if session is a prefix of Intan_Session (common if dates missing)
                     # or if Intan_Session is a prefix of session
                     # We iterate? No, vectorized check
                     # Use 'startswith'
                     mask = df_summary['Intan_Session'].str.startswith(session)
                     if mask.any():
                         found_row = df_summary[mask].iloc[0]

        # 3. Extract Values
        if found_row is not None:
             active_str = str(found_row.get('Stim_Channels', 'Unknown'))
             freq_str = str(found_row.get('Stim_Frequency', 'Unknown'))
             
             amp_val = found_row.get('Stim_Amplitudes', 'N/A')
             if pd.notna(amp_val) and str(amp_val).lower() != 'nan':
                 amp_str = f"| {amp_val} uA"
                 
             print(f"  [Meta] Found: Ch={active_str}, Freq={freq_str}")
        else:
             # Fallback to NPZ (Old Logic)
             print(f"  [Warn] No summary metadata found for {session}. Using NPZ fallback.")
             s_ch = data.get('stim_channels', [])
             if len(s_ch) > 0: active_str = str(s_ch)
             
             saved_amps = data.get('stim_amplitudes', [])
             if len(saved_amps) > 0:
                 u_amps = np.unique(saved_amps)
                 amp_str = f"| {u_amps} uA"

        # Title Construction
        if is_baseline:
            title_str = f"Baseline | Port {baseline_port} | Depth {baseline_depth} mm"
            session = f"Baseline_Depth_{baseline_depth}_port_{baseline_port}"  # For filename
        else:
            title_str = f"Stim: {active_str} | {freq_str} {amp_str}"
        
        # --- Pick Representative Channel ---
        rep_ch_idx = get_representative_channel(data.get('broadband_post'))
        
        # need to check if utah array or nprw
        groups = []
        is_utah = (label == "Utah_Array")
        

        if is_utah:
            import csv
            
            # 1. Determine Port (A or B) AND Intan Session from Metadata
            ua_port = 'A' # Default
            intan_session = None
            try:
                parts = npz_file.stem.split('_')
                if parts[-1].isdigit():
                    br_idx = int(parts[-1])
                elif parts[-2].isdigit():
                    br_idx = int(parts[-2])
                else:
                    br_idx = None
                    
                if br_idx is not None and cfg.METADATA_CSV.exists():
                     with open(cfg.METADATA_CSV, 'r', newline='') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row.get('BR_File') and int(row['BR_File']) == br_idx:
                                p = row.get('UA_port', '').strip().upper()
                                if p in ['A', 'B']:
                                    ua_port = p
                                intan_session = row.get('Intan_Session', '').strip()
                                break
            except Exception as e:
                print(f"  [Warn] Failed to lookup metadata for port/intan_session: {e}")

            # 2. Load Mapping from Excel
            try:
                xls_path = br_preproc.ua_excel_path(cfg.REPO_ROOT, cfg.PARAMS.probes)
                mapped_nsp = br_preproc.load_UA_mapping_from_excel(xls_path) # index=elec-1, val=nsp_id
                
                # Build Inverse Map: NSP_ID -> Elec_ID
                nsp_to_elec = {int(nsp): i+1 for i, nsp in enumerate(mapped_nsp) if nsp > 0}
                
                # 3. Map Channels to Regions
                offset = 128 if ua_port == 'B' else 0
                print(f"  [Info] Utah Array Port: {ua_port} (Offset: {offset})")
                
                sma_idxs, pmd_idxs, m1i_idxs, m1s_idxs = [], [], [], []
                
                for ch_idx in range(n_ch):
                    curr_nsp = ch_idx + 1 + offset
                    elec_id = nsp_to_elec.get(curr_nsp, -1)
                    
                    # Skip bad impedance channels
                    if curr_nsp in bad_ua_ids:
                        continue
                    
                    if elec_id > 0:
                        if elec_id <= 64: sma_idxs.append(ch_idx)
                        elif elec_id <= 128: pmd_idxs.append(ch_idx)
                        elif elec_id <= 192: m1i_idxs.append(ch_idx)
                        elif elec_id <= 256: m1s_idxs.append(ch_idx)
                
                if sma_idxs: groups.append((sma_idxs, f"SMA (n={len(sma_idxs)})"))
                if pmd_idxs: groups.append((pmd_idxs, f"PMd (n={len(pmd_idxs)})"))
                if m1i_idxs: groups.append((m1i_idxs, f"M1 Inf (n={len(m1i_idxs)})"))
                if m1s_idxs: groups.append((m1s_idxs, f"M1 Sup (n={len(m1s_idxs)})"))
                
                print(f"  [Info] Region mapping: SMA={len(sma_idxs)}, PMd={len(pmd_idxs)}, M1i={len(m1i_idxs)}, M1s={len(m1s_idxs)} channels")
                
                if not groups:
                    print(f"  [Warn] No valid channel mappings found for Port {ua_port}. Check Excel map.")
                    is_utah = False

            except Exception as e:
                print(f"  [Warn] Failed to load/apply Excel mapping: {e}. Fallback to linear blocks.")
                is_utah = False

        if not is_utah:
            n_groups_default = 8
            ch_per_group = 16
            for g in range(n_groups_default):
                start = g * ch_per_group
                end = start + ch_per_group
                if start < n_ch:
                    actual_end = min(end, n_ch)
                    grp_chs = [c for c in range(start, actual_end) if c not in bad_nprw_ids]
                    if grp_chs:
                        groups.append((grp_chs, f"Ch {start}-{actual_end-1}"))
        session = session.replace('.ns6', '')
        # --- 1. Original Plot (Heatmaps + Traces) ---
        if FIGURE_CONFIG.get('process_original', True):
            print("  Generating Original Heatmap/Trace Plots...")
            bands = ['broadband', 'delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
            
            # Pre-compute GLOBAL scales from ALL channels so every region figure shares the same limits
            global_scales = {}
            for band in bands:
                limit = PLOT_CONFIG['scales'].get(band)
                if limit is None:
                    d_pre_full = data.get(f"{band}_pre")
                    d_post_full = data.get(f"{band}_post")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m_pre_full = np.nanmean(d_pre_full, axis=0) if d_pre_full is not None else None
                        m_post_full = np.nanmean(d_post_full, axis=0) if d_post_full is not None else None
                    vals = []
                    if m_pre_full is not None: vals.append(m_pre_full.flatten())
                    if m_post_full is not None: vals.append(m_post_full.flatten())
                    if vals:
                        all_v = np.concatenate(vals)
                        valid_v = all_v[np.isfinite(all_v)]
                        if valid_v.size > 0:
                            vmin, vmax = np.percentile(valid_v, [2, 98])
                            limit = max(abs(vmin), abs(vmax))
                        else: limit = 50.0
                    else: limit = 50.0
                global_scales[band] = limit
            
            plot_groups = groups if is_utah else [(list(range(n_ch)), "AllChannels")]
            
            for grp_idxs, grp_name in plot_groups:
                safe_name = grp_name.split(' (')[0].replace(' ', '_')
                n_grp_ch = len(grp_idxs)
                if n_grp_ch == 0: continue
                
                grp_fig_dir = fig_dir / safe_name if is_utah else fig_dir
                grp_fig_dir.mkdir(parents=True, exist_ok=True)
                
                n_rows = len(bands)
                fig, axes = plt.subplots(n_rows, 3, figsize=(18, 3 * n_rows), constrained_layout=True, gridspec_kw={'width_ratios': [3, 1, 1]})
                fig.suptitle(f"{session} - {safe_name}\n{title_str}\n(N={n_trials})", fontsize=14)
                
                for i, band in enumerate(bands):
                    key_pre = f"{band}_pre"
                    key_post = f"{band}_post"
                    
                    d_pre = data.get(key_pre)
                    if d_pre is not None: d_pre = d_pre[:, grp_idxs, :]
                    
                    d_post = data.get(key_post)
                    if d_post is not None: d_post = d_post[:, grp_idxs, :]
                    
                    rep_ch_idx_local = get_representative_channel(d_post)
                    abs_ch_idx = grp_idxs[rep_ch_idx_local]
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Restrict data to x_limits for scaling
                        pre_lim = PLOT_CONFIG['x_limits']['pre']
                        post_lim = PLOT_CONFIG['x_limits']['post']
                        
                        m_pre, m_post = None, None
                        if d_pre is not None:
                            t_p = data.get('rel_time_pre')
                            if t_p is None: t_p = np.arange(d_pre.shape[2])
                            idx_pre = (t_p >= pre_lim[0]) & (t_p <= pre_lim[1]) if pre_lim else np.ones(len(t_p), dtype=bool)
                            if np.any(idx_pre):
                                m_pre = np.nanmean(d_pre[:, :, idx_pre], axis=0)
                            else:
                                m_pre = np.nanmean(d_pre, axis=0) # fallback
                                
                        if d_post is not None:
                            t_p = data.get('rel_time_post')
                            if t_p is None: t_p = t_ms
                            idx_post = (t_p >= post_lim[0]) & (t_p <= post_lim[1]) if post_lim else np.ones(len(t_p), dtype=bool)
                            if np.any(idx_post):
                                m_post = np.nanmean(d_post[:, :, idx_post], axis=0)
                            else:
                                m_post = np.nanmean(d_post, axis=0) # fallback
                    
                    limit = global_scales[band]
                    
                    cmap = PLOT_CONFIG['heatmap_cmap']
                    t_pre = data.get('rel_time_pre')
                    if t_pre is None and d_pre is not None: t_pre = np.arange(d_pre.shape[2])
                    
                    t_post = data.get('rel_time_post')
                    if t_post is None and d_post is not None: t_post = t_ms
                    
                    t_gap_start = t_pre[-1]
                    t_gap_end = t_post[0]
                    dt_est = t_pre[1] - t_pre[0] if len(t_pre) > 1 else 1.0 
                    n_gap = max(1, int(np.round((t_gap_end - t_gap_start) / dt_est)) - 1)
                    
                    ax = axes[i, 0]
                    
                    if is_utah:
                        key_full = f"{band}_full"
                        d_full = data.get(key_full)
                        if d_full is not None:
                            d_full_view = d_full[:, grp_idxs, :]
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                m_full = np.nanmean(d_full_view, axis=0)
                            
                            t_start = t_pre[0]
                            t_end = t_post[-1]
                            im = ax.imshow(m_full, aspect='auto', origin='upper', cmap=cmap, 
                                           vmin=-limit, vmax=limit, extent=[t_start, t_end, n_grp_ch - 0.5, -0.5])
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="uV")
                    else:
                        if m_pre is not None and m_post is not None:
                            gap_data = np.full((n_grp_ch, n_gap), np.nan)
                            m_comb = np.hstack([m_pre, gap_data, m_post])
                            
                            t_start = t_pre[0]
                            t_end = t_post[-1]
                            im = ax.imshow(m_comb, aspect='auto', origin='upper', cmap=cmap, vmin=-limit, vmax=limit, extent=[t_start, t_end, n_grp_ch - 0.5, -0.5]) 
                            
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="uV")
                            ax.axvspan(t_gap_start, t_gap_end, color='gray', alpha=0.3, hatch='///')

                    ax.set_ylabel(f"{band.capitalize()}\nCh Index")
                    ax.set_title(f"Combined Activity")
                    if i == n_rows - 1: ax.set_xlabel("Time (ms)")
                    else: ax.set_xticklabels([])
                    if PLOT_CONFIG['x_limits']['pre'] and PLOT_CONFIG['x_limits']['post']:
                         ax.set_xlim(PLOT_CONFIG['x_limits']['pre'][0], PLOT_CONFIG['x_limits']['post'][1])

                    # --- TRACES (Shared Y-Axis) ---
                    rep_pre = d_pre[:, rep_ch_idx_local, :] if d_pre is not None else None
                    rep_post = d_post[:, rep_ch_idx_local, :] if d_post is not None else None
                    
                    forced_ylim = PLOT_CONFIG['trace_scales'].get(band)
                    if forced_ylim is not None:
                        y_range = (-forced_ylim, forced_ylim)
                    else:
                        ylims = []
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pre_lim = PLOT_CONFIG['x_limits']['pre']
                            post_lim = PLOT_CONFIG['x_limits']['post']
                            
                            if rep_pre is not None:
                                t_p = data.get('rel_time_pre')
                                if t_p is None: t_p = np.arange(rep_pre.shape[1])
                                idx_pre = (t_p >= pre_lim[0]) & (t_p <= pre_lim[1]) if pre_lim else np.ones(len(t_p), dtype=bool)
                                valid_pre = rep_pre[:, idx_pre]
                                if np.any(np.isfinite(valid_pre)): 
                                    ylims.extend([np.nanmin(valid_pre), np.nanmax(valid_pre)])
                                    
                            if rep_post is not None:
                                t_p = data.get('rel_time_post')
                                if t_p is None: t_p = t_ms
                                idx_post = (t_p >= post_lim[0]) & (t_p <= post_lim[1]) if post_lim else np.ones(len(t_p), dtype=bool)
                                valid_post = rep_post[:, idx_post]
                                if np.any(np.isfinite(valid_post)): 
                                    ylims.extend([np.nanmin(valid_post), np.nanmax(valid_post)])
                        
                        ylims = [y for y in ylims if np.isfinite(y)]
                        if not ylims: y_range = (-100, 100)
                        else:
                             ymin, ymax = min(ylims), max(ylims)
                             y_range = (ymin - 10.0, ymax + 10.0) if ymin == ymax else (ymin - (ymax - ymin)*0.05, ymax + (ymax - ymin)*0.05)
                    
                    t_alpha = PLOT_CONFIG['trace_alpha']

                    if is_utah:
                        # Plot full unbroken trace explicitly using GridSpec
                        ax_pre = axes[i, 1]
                        ax_full = axes[i, 2]
                        
                        gs = ax_pre.get_subplotspec().get_gridspec()
                        ax_pre.remove()
                        ax_full.remove()
                        
                        ax_full = fig.add_subplot(gs[i, 1:])
                        key_full = f"{band}_full"
                        d_full = data.get(key_full)
                        if d_full is not None:
                            rep_full = d_full[:, grp_idxs[rep_ch_idx_local], :]
                            t_full_axis = np.linspace(t_pre[0], t_post[-1], rep_full.shape[1])
                            
                            ax_full.plot(t_full_axis, rep_full.T, color='gray', alpha=t_alpha, lw=0.5)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                rm = np.nanmean(rep_full, axis=0)
                            ax_full.plot(t_full_axis, rm, color='green', lw=1.5, label='Mean Corrected')
                            ax_full.axvspan(0, 100, color='grey', alpha=0.3, label='Stimulation')
                            ax_full.axvline(0, color='red', linestyle='--', alpha=0.5)
                        
                        ax_full.set_ylim(y_range)
                        ax_full.set_title(f"Continuous (Ch {abs_ch_idx})")
                        ax_full.set_ylabel("uV")
                        if PLOT_CONFIG['x_limits']['pre'] is not None and PLOT_CONFIG['x_limits']['post'] is not None:
                            ax_full.set_xlim(PLOT_CONFIG['x_limits']['pre'][0], PLOT_CONFIG['x_limits']['post'][1])
                    else:
                        ax = axes[i, 1]
                        if rep_pre is not None:
                            ax.plot(t_pre, rep_pre.T, color='gray', alpha=t_alpha, lw=0.5)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                rm = np.nanmean(rep_pre, axis=0)
                            ax.plot(t_pre, rm, color='blue', lw=1.5, label='Mean')
                        ax.set_ylim(y_range)
                        ax.set_title(f"Pre (Ch {abs_ch_idx})")
                        ax.set_ylabel("uV")
                        if PLOT_CONFIG['x_limits']['pre'] is not None: ax.set_xlim(PLOT_CONFIG['x_limits']['pre'])
                        elif t_pre is not None: ax.set_xlim(t_pre[0], t_pre[-1])
                        if i == 0: ax.legend(loc='upper right', fontsize='x-small')
                        
                        ax = axes[i, 2]
                        if rep_post is not None:
                            ax.plot(t_post, rep_post.T, color='gray', alpha=t_alpha, lw=0.5)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                rm = np.nanmean(rep_post, axis=0)
                            ax.plot(t_post, rm, color='red', lw=1.5, label='Mean')
                        ax.set_ylim(y_range)
                        ax.set_title(f"Post (Ch {abs_ch_idx})")
                        ax.set_ylabel("uV")
                        if PLOT_CONFIG['x_limits']['post'] is not None: ax.set_xlim(PLOT_CONFIG['x_limits']['post'])
                        elif t_post is not None: ax.set_xlim(t_post[0], t_post[-1])

                out_path = grp_fig_dir / f"check_lfp_heatmap_{session}.png" if not is_utah else grp_fig_dir / f"check_lfp_heatmap_{safe_name}_{session}.png"
                try:
                     plt.savefig(out_path, dpi=150)
                     print(f"  Saved plot to {out_path}")
                except Exception as e:
                     print(f"  Error saving {out_path}: {e}")
                plt.close(fig)

        # --- 1.5. Single Channel Trace Plot ---
        if FIGURE_CONFIG.get('process_single_channel_traces', False):
            target_ch = PLOT_CONFIG.get("single_channel_target", 168)
            ch_idx = None
            
            if is_utah:
                ua_ids = data.get('ua_ids_1based')
                if ua_ids is not None:
                    # Find the index where ua_ids == target_ch
                    idx_arr = np.where(ua_ids == target_ch)[0]
                    if len(idx_arr) > 0:
                        ch_idx = int(idx_arr[0])
                        print(f"  [Single Ch] Found target mapped channel {target_ch} at internal index {ch_idx}.")
                    else:
                        print(f"  [Single Ch] Warning: Channel {target_ch} not found in this dataset mapping.")
            else:
                # Direct indexing for NPRW
                if 0 <= target_ch < n_ch:
                    ch_idx = target_ch
                else:
                    print(f"  [Single Ch] Warning: Target channel {target_ch} out of bounds.")
                    
            if ch_idx is not None:
                print(f"  Generating Isolated Trace Plot for Channel {target_ch}...")
                bands = ['broadband', 'delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
                n_rows = len(bands)
                
                fig_sc, axes_sc = plt.subplots(n_rows, 1, figsize=(10, 2 * n_rows), constrained_layout=True)
                if n_rows == 1: axes_sc = [axes_sc]
                fig_sc.suptitle(f"{session} | Channel {target_ch}\n(N={n_trials} trials)", fontsize=14)
                
                for i, band in enumerate(bands):
                    d_pre = data.get(f"{band}_pre")
                    d_post = data.get(f"{band}_post")
                    
                    rep_pre = d_pre[:, ch_idx, :] if d_pre is not None else None
                    rep_post = d_post[:, ch_idx, :] if d_post is not None else None
                    
                    forced_ylim = PLOT_CONFIG['trace_scales'].get(band)
                    if forced_ylim is not None:
                        y_range = (-forced_ylim, forced_ylim)
                    else:
                        ylims = []
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pre_lim = PLOT_CONFIG['x_limits']['pre']
                            post_lim = PLOT_CONFIG['x_limits']['post']
                            
                            if rep_pre is not None:
                                t_p = data.get('rel_time_pre')
                                if t_p is None: t_p = np.arange(rep_pre.shape[1])
                                idx_pre = (t_p >= pre_lim[0]) & (t_p <= pre_lim[1]) if pre_lim else np.ones(len(t_p), dtype=bool)
                                valid_pre = rep_pre[:, idx_pre]
                                if np.any(np.isfinite(valid_pre)): 
                                    ylims.extend([np.nanmin(valid_pre), np.nanmax(valid_pre)])
                                    
                            if rep_post is not None:
                                t_p = data.get('rel_time_post')
                                if t_p is None: t_p = t_ms
                                idx_post = (t_p >= post_lim[0]) & (t_p <= post_lim[1]) if post_lim else np.ones(len(t_p), dtype=bool)
                                valid_post = rep_post[:, idx_post]
                                if np.any(np.isfinite(valid_post)):
                                    ylims.extend([np.nanmin(valid_post), np.nanmax(valid_post)])
                        
                        ylims = [y for y in ylims if np.isfinite(y)]
                        if not ylims: y_range = (-100, 100)
                        else:
                             ymin, ymax = min(ylims), max(ylims)
                             pad = 10.0 if ymin == ymax else (ymax - ymin) * 0.05
                             y_range = (ymin - pad, ymax + pad)
                             
                    ax = axes_sc[i]
                    t_alpha = PLOT_CONFIG['trace_alpha']
                    
                    t_pre_ax = data.get('rel_time_pre')
                    t_post_ax = data.get('rel_time_post')
                    
                    if is_utah:
                        d_full = data.get(f"{band}_full")
                        if d_full is not None:
                            rep_full = d_full[:, ch_idx, :]
                            t_full_axis = np.linspace(t_pre_ax[0], t_post_ax[-1], rep_full.shape[1])
                            
                            ax.plot(t_full_axis, rep_full.T, color='gray', alpha=t_alpha, lw=0.5)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                rm = np.nanmean(rep_full, axis=0)
                            ax.plot(t_full_axis, rm, color='green', lw=1.5, label='Mean Corrected')
                            ax.axvspan(0, 100, color='grey', alpha=0.3, label='Stimulation')
                            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
                            
                            if PLOT_CONFIG['x_limits']['pre'] is not None and PLOT_CONFIG['x_limits']['post'] is not None:
                                ax.set_xlim(PLOT_CONFIG['x_limits']['pre'][0], PLOT_CONFIG['x_limits']['post'][1])
                    else:
                        # Dual pre/post rendering on the same axis
                        if rep_pre is not None:
                            ax.plot(t_pre_ax, rep_pre.T, color='gray', alpha=t_alpha, lw=0.5)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                rm = np.nanmean(rep_pre, axis=0)
                            ax.plot(t_pre_ax, rm, color='blue', lw=1.5, label='Mean Pre')
                            
                        if rep_post is not None:
                            ax.plot(t_post_ax, rep_post.T, color='gray', alpha=t_alpha, lw=0.5)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                rm = np.nanmean(rep_post, axis=0)
                            ax.plot(t_post_ax, rm, color='red', lw=1.5, label='Mean Post')
                            
                        if PLOT_CONFIG['x_limits']['pre'] is not None and PLOT_CONFIG['x_limits']['post'] is not None:
                            ax.set_xlim(PLOT_CONFIG['x_limits']['pre'][0], PLOT_CONFIG['x_limits']['post'][1])
                        ax.axvspan(t_pre_ax[-1], t_post_ax[0], color='gray', alpha=0.3, hatch='///')
                        
                    ax.set_ylim(y_range)
                    ax.set_title(f"{band.capitalize()}")
                    ax.set_ylabel("uV")
                    if i == 0: ax.legend(loc='upper right', fontsize='x-small')
                    if i == n_rows - 1: ax.set_xlabel("Time (ms)")
                    else: ax.set_xticklabels([])
                    
                if is_utah:
                    comb_dir = fig_dir / "Combined_Analysis"
                    comb_dir.mkdir(parents=True, exist_ok=True)
                    out_path_sc = comb_dir / f"check_lfp_single_ch{target_ch}_{session}.png"
                else:
                    out_path_sc = fig_dir / f"check_lfp_single_ch{target_ch}_{session}.png"
                try:
                    plt.savefig(out_path_sc, dpi=150)
                    print(f"  Saved single channel plot to {out_path_sc}")
                except Exception as e:
                    print(f"  Error saving {out_path_sc}: {e}")
                plt.close(fig_sc)

        # --- Grab Broadband Data Single Time for Advanced Plots ---
        bb_pre = data.get("broadband_pre") # (N, Ch, T)
        bb_post = data.get("broadband_post")

        # Setup Dynamic Grid
        n_plots = len(groups)
        n_cols = 2
        n_rows = int(np.ceil(n_plots / n_cols))
        figsize = (8, 3 * n_rows)

        # --- 2. Power Spectra (Overlay, Grouped) ---
        if FIGURE_CONFIG.get('process_power_spectra', False):
            print("  Generating Grouped Power Spectra...")
            # Dynamic Grid
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
            fig.suptitle(f"{session} - Power Spectra (Group Mean ± SEM)\n{title_str}", fontsize=14)
            axes_flat = axes.flatten()
            
            # Pre-calculate to avoid redundant calls if possible, but simplicity first.
            if bb_pre is not None:
                f_pre, psd_pre_all, sem_pre_all = calculate_psd(bb_pre, fs) # (n_ch, n_freq)
            else: psd_pre_all = None
                
            if bb_post is not None:
                f_post, psd_post_all, sem_post_all = calculate_psd(bb_post, fs)
            else: psd_post_all = None

            # Fixed Y-Limits for PSD (Log Scale)
            y_min_psd = 1e-1
            y_max_psd = 1e2

            for i, (grp_idxs, grp_name) in enumerate(groups):
                if i >= len(axes_flat): break
                ax = axes_flat[i]
                
                # --- Pre ---
                if psd_pre_all is not None:
                    # Mean across group
                    g_psd = psd_pre_all[grp_idxs, :]
                    g_mean = np.nanmean(g_psd, axis=0) # (n_freq,)
                    ax.plot(f_pre, g_mean, color='blue', lw=2, label='Pre')

                # --- Post ---
                if psd_post_all is not None:
                    g_psd = psd_post_all[grp_idxs, :]
                    g_mean = np.nanmean(g_psd, axis=0)
                    ax.plot(f_post, g_mean, color='red', lw=2, label='Post')
                    
                add_band_annotations(ax)
                ax.set_title(grp_name)
                ax.set_xlim(0, 120)
                # Labels only on bottom/left edges of grid?
                # Simplify: Label all for clarity or just outer.
                ax.set_xlabel("Freq (Hz)")
                ax.set_ylabel("Power")
                ax.set_yscale('log')
                ax.set_ylim(bottom=y_min_psd, top=y_max_psd)
                if i == 0: ax.legend(loc='upper right', fontsize='small')
                ax.grid(True, alpha=0.3)
                
            # Hide unused axes
            for j in range(len(groups), len(axes_flat)):
                axes_flat[j].axis('off')

            if is_utah:
                comb_dir = fig_dir / "Combined_Analysis"
                comb_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(comb_dir / f"check_lfp_psd_grouped_{session}.png", dpi=150)
            else:
                plt.savefig(fig_dir / f"check_lfp_psd_grouped_{session}.png", dpi=150)
            plt.close(fig)

            # --- 2b. PSD Ratio (Pre vs Post) ---
            print("  Generating Grouped PSD Ratio...")
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
            fig.suptitle(f"{session} - PSD Ratio 10*log10(Post/Pre)\n{title_str}", fontsize=14)
            axes_flat = axes.flatten()

            # Pre-calculate Global Min/Max for Ratio
            r_min = np.inf
            r_max = -np.inf
            
            for grp_idxs, _ in groups:
                if psd_pre_all is not None and psd_post_all is not None:
                     g_pre = np.nanmean(psd_pre_all[grp_idxs, :], axis=0)
                     g_post = np.nanmean(psd_post_all[grp_idxs, :], axis=0)
                     g_pre[g_pre == 0] = 1e-10
                     ratio_db = 10 * np.log10(g_post / g_pre)
                     if np.any(np.isfinite(ratio_db)):
                         r_min = min(r_min, np.nanmin(ratio_db))
                         r_max = max(r_max, np.nanmax(ratio_db))
            
            if r_min == np.inf: r_min = -5; r_max = 5
            else:
                 # Symmetry? User didn't ask, but consistent range is good.
                 # Just use min/max with padding
                 r_range = r_max - r_min
                 r_min -= r_range * 0.1
                 r_max += r_range * 0.1

            for i, (grp_idxs, grp_name) in enumerate(groups):
                if i >= len(axes_flat): break
                ax = axes_flat[i]
                
                if psd_pre_all is not None and psd_post_all is not None:
                     # Mean Group PSDs
                     g_pre = np.nanmean(psd_pre_all[grp_idxs, :], axis=0) # (F,)
                     g_post = np.nanmean(psd_post_all[grp_idxs, :], axis=0)
                     
                     # Ratio in dB
                     # Handle zeros
                     g_pre[g_pre == 0] = 1e-10
                     ratio_db = 10 * np.log10(g_post / g_pre)
                     
                     ax.plot(f_pre, ratio_db, color='k', lw=2)
                     ax.axhline(0, color='gray', linestyle='--')
                     
                add_band_annotations(ax)
                ax.set_title(grp_name)
                ax.set_xlim(0, 120)
                ax.set_xlabel("Freq (Hz)")
                ax.set_ylabel("dB Change")
                ax.set_ylim(r_min, r_max)
                ax.grid(True, alpha=0.3)
                
            # Hide unused axes
            for j in range(len(groups), len(axes_flat)):
                axes_flat[j].axis('off')
            
            if is_utah:
                comb_dir = fig_dir / "Combined_Analysis"
                comb_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(comb_dir / f"check_lfp_psd_ratio_{session}.png", dpi=150)
            else:
                plt.savefig(fig_dir / f"check_lfp_psd_ratio_{session}.png", dpi=150)
            plt.close(fig)

        # --- 3. Power vs Time (All Arrays Overlaid per Band) ---
        if FIGURE_CONFIG.get('process_power_vs_time', False) and is_utah and len(groups) > 0:
            print("  Generating Power vs Time (all arrays overlaid)...")
            
            pvt_bands = {
                'Broadband':  'broadband',
                'Delta':      'delta',
                'Theta':      'theta',
                'Alpha':      'alpha',
                'Beta':       'beta',
                'Low Gamma':  'low_gamma',
                'High Gamma': 'high_gamma',
            }
            
            # Determine which time epoch to use (prefer full)
            if 'broadband_full' in data:
                epoch_suffix = 'full'
                t_pre_ax  = data.get('rel_time_pre')
                t_post_ax = data.get('rel_time_post')
                if t_pre_ax is not None and t_post_ax is not None:
                    pvt_t_axis = np.concatenate([t_pre_ax, t_post_ax])
                else:
                    pvt_t_axis = None
                pvt_label = "Full (Pre+Post)"
            elif 'broadband_post' in data:
                epoch_suffix = 'post'
                pvt_t_axis = data.get('rel_time_post')
                pvt_label = "Post-Stim"
            elif 'broadband_pre' in data:
                epoch_suffix = 'pre'
                pvt_t_axis = data.get('rel_time_pre')
                pvt_label = "Pre-Stim"
            else:
                pvt_label = None

            if pvt_label is not None:
                # Provide fallback time axis just in case
                if pvt_t_axis is None:
                    n_samps_bb = data[f'broadband_{epoch_suffix}'].shape[2]
                    pvt_t_axis = np.arange(n_samps_bb) * 1000.0 / fs
                    
                # Crop the time axis to x_limits here to avoid Hilbert edge effects
                valid_time_idx = np.ones(len(pvt_t_axis), dtype=bool)
                pre_lim = PLOT_CONFIG['x_limits']['pre']
                post_lim = PLOT_CONFIG['x_limits']['post']
                if pre_lim is not None and post_lim is not None:
                    valid_time_idx = (pvt_t_axis >= pre_lim[0]) & (pvt_t_axis <= post_lim[1])
                
                pvt_t_axis_cropped = pvt_t_axis[valid_time_idx]
                
                # Pre-compute per-band envelope from the saved waveforms:
                band_envelopes = {}
                for display_name, file_prefix in pvt_bands.items():
                    key = f"{file_prefix}_{epoch_suffix}"
                    if key in data:
                        # shape: (n_trials, n_ch, n_time)
                        waveforms = data[key]
                        # Crop waveforms using valid_time_idx (but we will pad it before Hilbert)
                        if waveforms.shape[2] == len(valid_time_idx):
                            waveforms = waveforms[:, :, valid_time_idx]
                            
                        # Compute instantaneous power via Hilbert envelope
                        # Apply trial-by-trial with reflection padding to avoid massive edge artifacts
                        n_pvt_trials = waveforms.shape[0]
                        env_all = np.empty_like(waveforms, dtype=np.float32)
                        
                        # Pad by 200 ms (or length of waveform if shorter) to absorb edge effects
                        pad_len = min(200, waveforms.shape[2] // 2)
                        
                        for tr_i in range(n_pvt_trials):
                            # Pad the 1D time series for each channel
                            padded_wave = np.pad(waveforms[tr_i], ((0, 0), (pad_len, pad_len)), mode='reflect')
                            analytic = signal.hilbert(padded_wave, axis=1)
                            # Crop back to original valid size and calculate power
                            valid_analytic = analytic[:, pad_len:-pad_len] if pad_len > 0 else analytic
                            env_all[tr_i] = np.abs(valid_analytic)**2
                        band_envelopes[display_name] = env_all
                    else:
                        print(f"    [PvT] {display_name}: key '{key}' not found in npz. Skipping.")
                        band_envelopes[display_name] = None
                
                # --- Plot ---
                n_pvt_bands = len(pvt_bands)
                fig_pvt, axes_pvt = plt.subplots(n_pvt_bands, 1,
                                                  figsize=(12, 2.5 * n_pvt_bands),
                                                  constrained_layout=True)
                if n_pvt_bands == 1:
                    axes_pvt = [axes_pvt]
                fig_pvt.suptitle(f"{session} - Band Power vs Time ({pvt_label})\n{title_str}", fontsize=14)

                array_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

                for band_i, (band_name, _) in enumerate(pvt_bands.items()):
                    ax = axes_pvt[band_i]
                    env = band_envelopes.get(band_name)
                    if env is None:
                        ax.set_title(f"{band_name} (skipped)")
                        continue

                    band_ylims = []
                    for grp_i, (grp_idxs, grp_name) in enumerate(groups):
                        # Safely index channels that exist
                        valid_grp_idxs = [idx for idx in grp_idxs if idx < env.shape[1]]
                        if not valid_grp_idxs:
                            continue
                            
                        grp_env = env[:, valid_grp_idxs, :]           # (n_trials, n_grp_ch, n_time)
                        grp_mean_ch = np.nanmean(grp_env, axis=1)  # avg channels -> (n_trials, n_time)
                        
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            trial_mean = np.nanmean(grp_mean_ch, axis=0)
                            trial_sem  = (np.nanstd(grp_mean_ch, axis=0)
                                          / np.sqrt(np.sum(~np.isnan(grp_mean_ch[:, 0]))))

                        color = array_colors[grp_i % len(array_colors)]
                        short_name = grp_name.split(' (')[0]
                        
                        # Fix dimension mismatch: generate a time axis that exactly matches the data length
                        # The data length might be slightly different than the concatenated pvt_t_axis due to 
                        # filtering/resampling artifacts in analyze_lfp_bands.
                        n_points = trial_mean.shape[0]
                        if pvt_t_axis_cropped is not None and len(pvt_t_axis_cropped) >= 2:
                            if n_points == len(pvt_t_axis_cropped):
                                t_plot = pvt_t_axis_cropped
                            else:
                                t_start = pvt_t_axis_cropped[0]
                                t_plot = np.arange(n_points) * (1000.0 / fs) + t_start
                        else:
                            t_plot = np.arange(n_points) * (1000.0 / fs)

                        ax.plot(t_plot, trial_mean, color=color, lw=1.2, label=short_name)
                        ax.fill_between(t_plot,
                                        trial_mean - trial_sem,
                                        trial_mean + trial_sem,
                                        color=color, alpha=0.2)
                                        
                        # Calculate ylims strictly within x_limits
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pre_lim = PLOT_CONFIG['x_limits']['pre']
                            post_lim = PLOT_CONFIG['x_limits']['post']
                            
                            valid_idx = np.ones(len(t_plot), dtype=bool)
                            if pre_lim is not None and post_lim is not None:
                                valid_idx = (t_plot >= pre_lim[0]) & (t_plot <= post_lim[1])
                                
                            if np.any(valid_idx):
                                top = (trial_mean + trial_sem)[valid_idx]
                                bot = (trial_mean - trial_sem)[valid_idx]
                                if np.any(np.isfinite(top)) and np.any(np.isfinite(bot)):
                                    band_ylims.extend([np.nanmin(bot), np.nanmax(top)])

                    # Apply targeted y-axis limit
                    if band_ylims:
                        band_ylims = [y for y in band_ylims if np.isfinite(y)]
                        if band_ylims:
                            ymin, ymax = min(band_ylims), max(band_ylims)
                            pad = (ymax - ymin) * 0.05 if ymin != ymax else 10.0
                            ax.set_ylim(max(0, ymin - pad), ymax + pad) # Power is inherently >= 0

                    ax.set_title(band_name, fontsize=10)
                    ax.set_ylabel("Power (µV²)")
                    ax.axvspan(0, 100, color='grey', alpha=0.3, label='Stimulation' if band_i == 0 else "")
                    ax.axvline(0, color='k', linestyle='--', alpha=0.5, lw=0.8)
                    if band_i == 0:
                        ax.legend(loc='upper right', fontsize='small', ncol=2)
                    if band_i == n_pvt_bands - 1:
                        ax.set_xlabel("Time (ms)")
                    ax.grid(True, alpha=0.3)
                    
                    if (PLOT_CONFIG['x_limits']['pre'] is not None
                            and PLOT_CONFIG['x_limits']['post'] is not None):
                        ax.set_xlim(PLOT_CONFIG['x_limits']['pre'][0],
                                    PLOT_CONFIG['x_limits']['post'][1])

                comb_dir = fig_dir / "Combined_Analysis"
                comb_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(comb_dir / f"check_lfp_power_vs_time_{session}.png", dpi=150)
                plt.close(fig_pvt)

        # --- 3b. Inter-Array Coherence (4x4 Grid, Utah Only) ---
        if FIGURE_CONFIG.get('process_inter_array_coherence', False) and is_utah and len(groups) >= 2:
            print("  Generating Inter-Array Coherence (4x4 grid)...")
            
            target_data = bb_post if bb_post is not None else bb_pre
            t_label = "Post-Stim" if bb_post is not None else "Pre-Stim"
            ua_ids = data.get('ua_ids_1based')
            
            if target_data is not None:
                n_grps = len(groups)
                nperseg = min(target_data.shape[2], 256)
                
                # Find representative channel for each group
                rep_ch_per_group = []
                for grp_idxs, grp_name in groups:
                    grp_data = target_data[:, grp_idxs, :]
                    local_rep = get_representative_channel(grp_data)
                    abs_rep = grp_idxs[local_rep]
                    rep_ch_per_group.append(abs_rep)
                
                # Build group short names
                grp_short_names = [gn.split(' (')[0] for _, gn in groups]
                
                fig_coh, axes_coh = plt.subplots(n_grps, n_grps, figsize=(4 * n_grps, 3.5 * n_grps),
                                                  constrained_layout=True)
                fig_coh.suptitle(f"{session} - Inter-Array Coherence ({t_label})\n{title_str}", fontsize=14)
                
                # If only 2 groups, axes_coh might be 1D; ensure 2D
                if n_grps == 1:
                    axes_coh = np.array([[axes_coh]])
                elif axes_coh.ndim == 1:
                    axes_coh = axes_coh.reshape(1, -1)
                
                for row_i in range(n_grps):
                    rep_abs_idx = rep_ch_per_group[row_i]
                    rep_sig = target_data[:, rep_abs_idx, :]  # (n_trials, n_time)
                    
                    rep_label = ""
                    if ua_ids is not None:
                        rep_label = f" (Ch {ua_ids[rep_abs_idx]})"
                    
                    for col_i in range(n_grps):
                        ax = axes_coh[row_i, col_i]
                        col_grp_idxs = groups[col_i][0]
                        
                        # Compute coherence: rep channel vs each channel in column group
                        coh_list = []
                        freqs = None
                        
                        for ch_idx in col_grp_idxs:
                            ch_sig = target_data[:, ch_idx, :]
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                f, Cxy = signal.coherence(ch_sig, rep_sig, fs=fs,
                                                          nperseg=nperseg, axis=1)
                                m_cxy = np.nanmean(Cxy, axis=0)
                            coh_list.append(m_cxy)
                            if freqs is None: freqs = f
                        
                        coh_matrix = np.stack(coh_list, axis=0)  # (n_ch_in_col_grp, n_freq)
                        
                        # Plot heatmap
                        im = ax.imshow(coh_matrix, aspect='auto', origin='upper', cmap='inferno',
                                       extent=[freqs[0], freqs[-1], len(col_grp_idxs)-0.5, -0.5],
                                       vmin=0, vmax=1)
                        
                        # Band boundary lines
                        for b_s, b_e, b_c, _ in [(4, 13, 'green', 'A'), (13, 30, 'yellow', 'B'),
                                                   (30, 60, 'orange', 'LG'), (60, 120, 'red', 'HG')]:
                            ax.axvline(b_s, color=b_c, linestyle='--', alpha=0.4)
                            ax.axvline(b_e, color=b_c, linestyle='--', alpha=0.4)
                        
                        ax.set_xlim(0, 120)
                        
                        # Diagonal highlight
                        if row_i == col_i:
                            ax.set_title(f"{grp_short_names[row_i]}{rep_label}\nvs Self", fontsize=9,
                                         fontweight='bold')
                        else:
                            ax.set_title(f"{grp_short_names[row_i]}{rep_label}\nvs {grp_short_names[col_i]}", fontsize=9)
                        
                        # Y-ticks: channel IDs from column group
                        if ua_ids is not None:
                            labels = [str(ua_ids[idx]) for idx in col_grp_idxs]
                            step = max(1, len(labels) // 6)
                            ax.set_yticks(range(0, len(col_grp_idxs), step))
                            ax.set_yticklabels(labels[::step], fontsize=7)
                        
                        # Axis labels on edges only
                        if row_i == n_grps - 1: ax.set_xlabel("Freq (Hz)", fontsize=8)
                        else: ax.set_xticklabels([])
                        if col_i == 0: ax.set_ylabel(f"Target Ch in\n{grp_short_names[col_i]}", fontsize=8)
                
                # Add a shared colorbar
                fig_coh.colorbar(im, ax=axes_coh, shrink=0.6, label="Coherence", pad=0.02)
                
                comb_dir = fig_dir / "Combined_Analysis"
                comb_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(comb_dir / f"check_lfp_inter_array_coherence_{session}.png", dpi=150)
                plt.close(fig_coh)

        # --- 4. Time-Frequency ITPC (Grouped 4x2) ---
        if FIGURE_CONFIG.get('process_phase_coherence', False):
            print("  Generating Grouped T-F ITPC...")
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
            fig.suptitle(f"{session} - T-F ITPC (Post-Stim Group Mean)\n{title_str}", fontsize=14)
            axes_flat = axes.flatten()

            # Using Post-Stim for event response
            target_data = bb_post if bb_post is not None else bb_pre
            
            if target_data is not None:
                # Calculate ALL ITPC first? (n_ch, n_freq, n_time)
                # Calculating for 128 channels is acceptable.
                # Use slightly coarser STFT to be safe on memory/speed needed? 128/120 is fine.
                f, t, itpc_all = calculate_tf_itpc(target_data, fs, nperseg=128, noverlap=120)
                
                # Time vector
                t0 = data.get('rel_time_post')[0] if bb_post is not None else -500
                t_real = t * 1000.0 + t0

                for i, (grp_idxs, grp_name) in enumerate(groups):
                    if i >= len(axes_flat): break
                    ax = axes_flat[i]
                    
                    # Mean ITPC for this group
                    g_itpc = itpc_all[grp_idxs, :, :] # (n_grp, n_f, n_t)
                    g_mean_itpc = np.nanmean(g_itpc, axis=0) # (n_f, n_t)
                    
                    # Plot
                    im = ax.pcolormesh(t_real, f, g_mean_itpc, shading='gouraud', cmap='viridis', vmin=0, vmax=0.6) 
                    # lowered vmax slightly as means tend to be lower
                    
                    if i == 1: fig.colorbar(im, ax=ax, label="Mean ITPC")
                    
                    ax.set_title(grp_name)
                    ax.set_ylim(0, 120)
                    if i >= 6: ax.set_xlabel("Time (ms)")
                    if i % 2 == 0: ax.set_ylabel("Freq (Hz)")

            out_path = fig_dir / f"check_lfp_itpc_grouped_{session}.png"
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

        # --- 5. ERSP (Event-Related Spectral Perturbation) ---
        if FIGURE_CONFIG.get('process_ersp', False):
            print("  Generating Grouped ERSP...")
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
            fig.suptitle(f"{session} - ERSP (dB vs Pre-Stim Baseline)\n{title_str}", fontsize=14)
            axes_flat = axes.flatten()

            if bb_pre is not None and bb_post is not None:
                # Calculate ERSP for all channels
                # Use slightly coarser resolution for speed/memory if needed
                f_ersp, t_ersp_post, ersp_all = calculate_ersp(bb_pre, bb_post, fs, nperseg=128, noverlap=100)
                # ersp_all: (n_ch, n_freq, n_time)
                
                # Time vector (post)
                t0_post = data.get('rel_time_post')[0] if data.get('rel_time_post') is not None else 0
                t_real_post = t_ersp_post * 1000.0 + t0_post

                for i, (grp_idxs, grp_name) in enumerate(groups):
                    if i >= len(axes_flat): break
                    ax = axes_flat[i]
                    
                    # Mean ERSP for group
                    g_ersp = ersp_all[grp_idxs, :, :]
                    g_mean_ersp = np.nanmean(g_ersp, axis=0) # (F, T)
                    
                    # Plot Heatmap (RdBu_r centered at 0)
                    # Use vmin/vmax symmetric for dB
                    v_lim = 3.0 # +/- 3 dB
                    im = ax.pcolormesh(t_real_post, f_ersp, g_mean_ersp, shading='gouraud', cmap='RdBu_r', 
                                       vmin=-v_lim, vmax=v_lim)
                    
                    if i == 1: fig.colorbar(im, ax=ax, label="dB")
                    
                    ax.set_title(grp_name)
                    ax.set_ylim(0, 120)
                    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("Freq (Hz)")
            
            # Hide unused axes
            for j in range(len(groups), len(axes_flat)):
                axes_flat[j].axis('off')

            out_path = fig_dir / f"check_lfp_ersp_grouped_{session}.png"
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

        # --- 6. Band Power Stats (Bar Charts) ---
        if FIGURE_CONFIG.get('process_band_stats', False):
            print("  Generating Band Power Stats...")
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
            fig.suptitle(f"{session} - Band Power Change (Pre vs Post)\n{title_str}", fontsize=14)
            axes_flat = axes.flatten()
            
            if bb_pre is not None and bb_post is not None:
                # Calculate Stats
                stats_res = calculate_band_power_stats(bb_pre, bb_post, fs)
                
                # Bands to plot
                band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Low Gamma', 'High Gamma']
                x_pos = np.arange(len(band_names))
                width = 0.35
                
                # Pre-calculate Global Max for Y-Axis (Log Scale)
                y_max_stat = 0
                y_min_stat = np.inf
                
                for grp_idxs, _ in groups:
                     for band in band_names:
                         res = stats_res.get(band)
                         if res:
                             # Re-calc similar to plotting loop
                             m_pre = np.nanmean(res['mean_pre'][grp_idxs])
                             m_post = np.nanmean(res['mean_post'][grp_idxs])
                             s_pre = np.nanstd(res['mean_pre'][grp_idxs]) / np.sqrt(len(grp_idxs))
                             s_post = np.nanstd(res['mean_post'][grp_idxs]) / np.sqrt(len(grp_idxs))
                             
                             top = max(m_pre + s_pre, m_post + s_post)
                             bot = min(m_pre, m_post) # Use mean for min check (se might make it negative? No power >=0)
                             
                             y_max_stat = max(y_max_stat, top)
                             if bot > 0: y_min_stat = min(y_min_stat, bot)
                
                if y_max_stat > 0:
                     y_max_stat = y_max_stat * 2.0 # Log scale needs head room
                     y_min_stat = y_min_stat * 0.5 if y_min_stat != np.inf else 1e-2
                else: 
                     y_max_stat = 1.0; y_min_stat = 1e-2

                for i, (grp_idxs, grp_name) in enumerate(groups):
                    if i >= len(axes_flat): break
                    ax = axes_flat[i]
                    
                    # Collect data for this group
                    means_pre = []
                    means_post = []
                    errs_pre = []
                    errs_post = []
                    p_values = []
                    
                    for band in band_names:
                        res = stats_res.get(band)
                        if res is None:
                            means_pre.append(0); means_post.append(0)
                            errs_pre.append(0); errs_post.append(0)
                            p_values.append(1.0)
                            continue
                        
                        # Subset for group
                        g_mean_pre_chs = res['mean_pre'][grp_idxs]
                        g_mean_post_chs = res['mean_post'][grp_idxs]
                        
                        # Grand Mean across channels
                        gm_pre = np.nanmean(g_mean_pre_chs)
                        gm_post = np.nanmean(g_mean_post_chs)
                        
                        # Error? SEM across channels (Spatial variability)
                        g_sem_pre = np.nanstd(g_mean_pre_chs) / np.sqrt(len(grp_idxs))
                        g_sem_post = np.nanstd(g_mean_post_chs) / np.sqrt(len(grp_idxs))
                        
                        means_pre.append(gm_pre)
                        means_post.append(gm_post)
                        errs_pre.append(g_sem_pre)
                        errs_post.append(g_sem_post)
                        
                        # P-value: We did T-test per channel.
                        # How many channels were significant? Or perform T-test on the channel means?
                        # Let's do a Paired T-test on the CHANNEL MEANS for the group.
                        # i.e. Does the group consistently change?
                        t_grp, p_grp = stats.ttest_rel(g_mean_pre_chs, g_mean_post_chs, nan_policy='omit')
                        p_values.append(p_grp)

                    # Plot Bars
                    rects1 = ax.bar(x_pos - width/2, means_pre, width, yerr=errs_pre, label='Pre', color='blue', alpha=0.6, capsize=5)
                    rects2 = ax.bar(x_pos + width/2, means_post, width, yerr=errs_post, label='Post', color='red', alpha=0.6, capsize=5)
                    
                    # Add Significance Stars
                    max_y = max(max(means_pre), max(means_post)) * 1.1
                    for idx, p in enumerate(p_values):
                        if p < 0.05:
                            txt = '*' if p < 0.01 else '*'
                            # Height roughly above the bar
                            h = max(means_pre[idx] + errs_pre[idx], means_post[idx] + errs_post[idx])
                            ax.text(x_pos[idx], h * 1.05, txt, ha='center', va='bottom', fontsize=12, fontweight='bold')

                    ax.set_title(grp_name)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(band_names)
                    ax.set_xticklabels(band_names)
                    ax.set_ylabel("Power (uV^2/Hz)")
                    ax.set_yscale('log')
                    if y_max_stat > 0: ax.set_ylim(bottom=y_min_stat, top=y_max_stat)
                    if i == 0: ax.legend()
                    ax.grid(axis='y', alpha=0.3)
            
            # Hide unused axes
            for j in range(len(groups), len(axes_flat)):
                axes_flat[j].axis('off')

            if is_utah:
                comb_dir = fig_dir / "Combined_Analysis"
                comb_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(comb_dir / f"check_lfp_band_stats_{session}.png", dpi=150)
            else:
                plt.savefig(fig_dir / f"check_lfp_band_stats_{session}.png", dpi=150)
            plt.close(fig)

        # --- 7. CSD Analysis (NPRW Only) ---
        if FIGURE_CONFIG.get('process_csd', False) and y_coords is not None and "Intan" in label:
            # Verify channel count matches map
            if len(y_coords) == n_ch and bb_post is not None:
                print("  Generating CSD Analysis (Pre + Post)...")
                
                # CSD needs Event-Locked Data. 
                # Combine Pre (if avail) and Post.
                
                # 1. Post
                erp_post = np.nanmean(bb_post, axis=0) # (n_ch, n_time)
                
                # Time vectors from Data Dict
                if 'rel_time_post' in data: t_post = data['rel_time_post']
                else: t_post = np.arange(erp_post.shape[1]) / fs * 1000
                
                erp_combined = erp_post
                t_combined = t_post
                
                # 2. Pre?
                if bb_pre is not None:
                    erp_pre = np.nanmean(bb_pre, axis=0)
                    if 'rel_time_pre' in data: t_pre = data['rel_time_pre']
                    else: t_pre = np.arange(erp_pre.shape[1]) / fs * 1000 - (erp_pre.shape[1]/fs*1000)

                    # Gap?
                    # t_pre usually negative, t_post positive.
                    # Create a gap of NaNs for visual separation (e.g. 1ms? or just gap in time axis)
                    # pcolormesh needs continuous X? No, but heatmaps usually have uniform grid.
                    # We can insert NaN columns to represent the artifact/blanking.
                    
                    if len(t_post) > 0 and len(t_pre) > 0:
                        gap_ms = t_post[0] - t_pre[-1]
                        if gap_ms > 0:
                            n_gap = int(gap_ms / 1000 * fs)
                            if n_gap > 0:
                                gap_data = np.full((n_ch, n_gap), np.nan)
                                # Construct gap time: careful with linspace to not overlap
                                gap_time = np.linspace(t_pre[-1], t_post[0], n_gap+2)[1:-1]
                                
                                erp_combined = np.concatenate([erp_pre, gap_data, erp_post], axis=1)
                                t_combined = np.concatenate([t_pre, gap_time, t_post])
                            else:
                                erp_combined = np.concatenate([erp_pre, erp_post], axis=1)
                                t_combined = np.concatenate([t_pre, t_post])
                        else:
                             # Overlap? Just concat
                             erp_combined = np.concatenate([erp_pre, erp_post], axis=1)
                             t_combined = np.concatenate([t_pre, t_post])
                            
                # Calculate CSD
                csd_val, csd_depths = calculate_csd(erp_combined, y_coords)
                
                if csd_val is not None:
                    # Plot CSD Heatmap
                    fig_csd, ax_csd = plt.subplots(figsize=(10, 6), constrained_layout=True)
                    fig_csd.suptitle(f"{session} - CSD (Current Source Density)\n{title_str}", fontsize=14)
                    
                    # Plot with pcolormesh
                    # Use symmetric colormap
                    vm = np.nanpercentile(np.abs(csd_val), 98)
                    if vm == 0: vm = 1e-6
                    
                    mesh = ax_csd.pcolormesh(t_combined, csd_depths, csd_val, 
                                            cmap='RdBu_r', vmin=-vm, vmax=vm, shading='auto')
                    
                    ax_csd.set_xlabel("Time (ms)")
                    ax_csd.set_ylabel("Depth (um from Tip)")
                    # ax_csd.invert_yaxis() # Deep is down? 
                    # If y=0 is Tip (Deep), and we want Tip at Bottom -> Normal Y Axis (0 at bottom).
                    # So remove invert_yaxis.
                    
                    # Mark 0
                    ax_csd.axvline(0, color='k', linestyle='--', alpha=0.5)
                    
                    cb = plt.colorbar(mesh, ax=ax_csd)
                    cb.set_label("CSD (uV/mm^2)")
                    
                    out_path = fig_dir / f"check_lfp_csd_{session}.png"
                    plt.savefig(out_path, dpi=150)
                    plt.close(fig_csd)

        # --- 7. Per-Channel ERSP Spectrograms (Utah Only) ---
        if FIGURE_CONFIG.get('process_spectrograms', False) and is_utah:
            print("  Generating Per-Channel ERSP Spectrograms...")
            
            d_bb_full = data.get('broadband_full')
            if d_bb_full is None:
                print("  [Skip] No broadband_full data for spectrograms.")
            else:
                spec_cfg = PLOT_CONFIG['spectrogram']
                
                # Reconstruct time axis for broadband_full
                # broadband_full spans WIN_FULL = (-EPOCH_PRE_MS, EPOCH_POST_MS)
                # from analyze_lfp_bands.py
                t_pre_arr = data.get('rel_time_pre')
                t_post_arr = data.get('rel_time_post')
                if t_pre_arr is not None and t_post_arr is not None:
                    t_full_start = t_pre_arr[0]
                    t_full_end = t_post_arr[-1]
                else:
                    t_full_start = -500.0
                    t_full_end = 1000.0
                
                n_time_full = d_bb_full.shape[2]
                t_full_ms = np.linspace(t_full_start, t_full_end, n_time_full)
                
                # Compute ERSP for ALL channels at once
                try:
                    ersp_all, freqs, t_bins_ms = compute_ersp_spectrogram(
                        d_bb_full, t_full_ms, fs, spec_cfg
                    )
                except Exception as e:
                    print(f"  [ERROR] ERSP computation failed: {e}")
                    ersp_all = None
                
                if ersp_all is not None:
                    spec_fig_dir = fig_dir / "Spectrograms"
                    spec_fig_dir.mkdir(parents=True, exist_ok=True)
                    
                    for grp_idxs, grp_name in groups:
                        safe_name = grp_name.split(' (')[0].replace(' ', '_')
                        n_grp_ch = len(grp_idxs)
                        if n_grp_ch == 0:
                            continue
                        
                        # Create tall single-column figure
                        subplot_h = 2.0  # inches per channel
                        fig_h = max(6, subplot_h * n_grp_ch + 1.5)
                        fig_sg, axes_sg = plt.subplots(
                            n_grp_ch, 1, figsize=(10, fig_h),
                            sharex=True, sharey=True,
                            layout='tight'
                        )
                        if n_grp_ch == 1:
                            axes_sg = [axes_sg]
                        
                        fig_sg.suptitle(
                            f"{session} — {safe_name}\n{title_str}\n"
                            f"ERSP Spectrogram (N={n_trials} trials)",
                            fontsize=13
                        )
                        
                        for ax_i, ch_idx in enumerate(grp_idxs):
                            ax = axes_sg[ax_i]
                            ersp_ch = ersp_all[ch_idx]  # (n_freq, n_time)
                            
                            shading = spec_cfg.get('shading', 'flat')
                            im = ax.pcolormesh(
                                t_bins_ms, freqs, ersp_ch,
                                cmap=spec_cfg['cmap'],
                                vmin=spec_cfg['vmin_db'],
                                vmax=spec_cfg['vmax_db'],
                                shading=shading
                            )
                            
                            ax.axvline(0, color='white', linewidth=0.8,
                                       linestyle='--', alpha=0.8)
                            
                            # Log frequency scale
                            if spec_cfg.get('log_freq', False):
                                ax.set_yscale('log')
                                ax.set_ylim(max(spec_cfg.get('freq_min', 1), 1), spec_cfg['freq_max'])
                            
                            # Channel label
                            nsp_id = ch_idx + 1 + (128 if ua_port == 'B' else 0)
                            ax.set_ylabel(f"Ch {nsp_id}\nFreq (Hz)", fontsize=7)
                            ax.tick_params(labelsize=6)
                            
                            if ax_i < n_grp_ch - 1:
                                ax.set_xlabel('')
                        
                        # Bottom axis label
                        axes_sg[-1].set_xlabel("Time (ms)", fontsize=9)
                        
                        # Shared colorbar
                        cb = fig_sg.colorbar(im, ax=axes_sg, fraction=0.02,
                                             pad=0.02, label="Power (dB)")
                        cb.ax.tick_params(labelsize=7)
                        
                        out_path = spec_fig_dir / f"spectrogram__{session}__{safe_name}.png"
                        fig_dpi = spec_cfg.get('dpi', 150)
                        plt.savefig(out_path, dpi=fig_dpi)
                        plt.close(fig_sg)
                        print(f"    Saved spectrogram -> {out_path.name}")
                    
                    del ersp_all



def plot_lfp_check():
    # 1. Process NPRW
    process_directory(NPRW_LFP_DIR, NPRW_FIG_DIR, label="NPRW_Intan")
    
    # 2. Process Utah Array
    process_directory(UA_LFP_DIR, UA_FIG_DIR, label="Utah_Array")

if __name__ == "__main__":
    plot_lfp_check()
