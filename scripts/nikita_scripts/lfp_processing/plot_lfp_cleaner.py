"""
LFP Analysis Suite for Cerebellar Stimulation Experiment
=========================================================

Analyses included:
1. Session Summary PSD (1-120 Hz)
2. Raw Waveform Raster - See actual data across trials
3. Band Dynamics with proper 95% CI (SEM × 1.96)
4. Inter-Regional Coherence - Functional connectivity between arrays
5. Phase Lag Analysis - Information flow / lead-lag between regions
6. Dose-Response Analysis - Effect vs pulse count
7. Spatial GIF Animation - Activity spread over electrode grid
8. Stim vs Control Comparison

Key fixes from previous version:
- Frequency range extended to 120 Hz
- Confidence intervals now use SEM × 1.96 (not just SEM or SD)
- Removed unreliable peak detection
- Added meaningful inter-array analyses (coherence, phase)
- Uses Morlet wavelets for time-frequency analysis (no external dependencies)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d, label
from pathlib import Path
import warnings
import imageio
import io
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore', message='.*tight_layout.*')

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

# =============================================================================
# SESSION SELECTION - EDIT THESE TO FILTER WHICH SESSIONS TO PROCESS
# =============================================================================

SESSION_FILTER = [17]
# SESSION_FILTER = [1]
SESSION_FILTER = [17, 16, 15, 1,]          # None = all, or list like [1, 2, 3]
SESSION_ID_PATTERN = None      # None = all, or regex like r'.*control.*'


# =============================================================================
# CONFIGURATION
# =============================================================================

FIGURE_CONFIG = {
    # Core visualizations
    'generate_session_summary': False,       # PSD overview (1-120 Hz)
    'generate_waveform_raster': False,       # Raw LFP traces across trials
    'generate_trial_heatmaps': False,        # Per-trial band power heatmaps

    # Spectrograms
    'generate_per_ua_spectrograms': False,   # 8x8 grid of spectrograms per array
    'per_ua_default_trial': 3, 
    'generate_all_trial_spectrograms': False,  # If True, generate for each trial (can be many files!)
    'generate_per_ua_trial_average': False,
    'generate_per_ua_array_median': True,  # Median across all trials AND channels per Utah array
    'generate_control_median_baseline_norm_array_median': True,
    

    # Inter-regional analyses
    'generate_coherence_matrix': False,      # Coherence between all array pairs
    
    # Condition comparisons
    'generate_stim_vs_control': False,       # Compare stim to control trials
    'generate_dose_response': False,         # Effect vs number of pulses
    
    # Spatial visualizations
    'generate_spatial_heatmaps': False,      # Static spatial maps at key timepoints
    'generate_spatial_gif': False,           # Animated GIF of spatial activity
}

TIME_WINDOWS = {
    # Pre-stim window for PSD comparison (ms relative to stim onset)
    'pre_start': -950,
    'pre_end': -5,
    
    # Post-stim window for PSD comparison (ms relative to stim onset)
    'post_start': 5,
    'post_end': 950,
    
    # Display limits for plots
    'display_pre': (-500, -5),
    'display_post': (5, 500),
    'display_full': (-500, 500),
}

ANALYSIS_CONFIG = {
    'use_gpu_wavelet': True,
    'gpu_backend': 'torch',
    'gpu_batch_size': 32,


    # Frequency settings
    'freq_range': (1, 120),                 # Extended to 120 Hz as requested
    'freq_range_display': (1, 120),         # For PSD plots
    
    # Baseline settings (safe window with no movement)
    'baseline_window': (-950, -650),        # ms - confirmed safe by user
    
    # Statistical settings
    'confidence_level': 0.95,               # For CI calculation
    'z_score': 1.96,                        # For 95% CI: SEM × 1.96
    
    # Band definitions
    'bands': {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 25),
        'Low Gamma': (25, 55),
        'High Gamma': (65, 115),
    },

    'notch_regions': [(55, 65), (115, 125)],
    
    # Coherence settings
    'coherence_nperseg': 256,
    'coherence_noverlap': 192,
    
    # Phase lag settings
    'phase_lag_max_ms': 100,                # Maximum lag to consider
    
    # GIF settings
    'gif_time_range': (-500, 500),
    'gif_step_ms': 25,
    'gif_fps': 10,
}

PLOT_CONFIG = {
    # Color scheme for regions
    'region_colors': {
        'SMA': '#1f77b4',   # Blue
        'PMd': '#ff7f0e',   # Orange
        'M1i': '#2ca02c',   # Green
        'M1s': '#d62728',   # Red
    },
    
    # Heatmap settings
    'heatmap_cmap': 'jet', # or RdBu_r
    'heatmap_vmin': -6.0,                   # dB
    'heatmap_vmax': 6.0,                    # dB
    
    # Trace opacity
    'trace_alpha': 0.3,
    
    # Time limits for display
    'x_limits': {
        'pre': (-400, -5),
        'post': (101, 600),
        'full': (-400, 600),
    },
    
    # Spectrogram settings
    'spectrogram': {
        # TFR parameters
        # Options:
        #   'morlet'          = fixed-cycle Morlet
        #   'morlet_adaptive' = frequency-adaptive Morlet
        'use_wavelet': True,
        'tfr_method': 'morlet_adaptive',

        # Fixed Morlet fallback/default
        'wavelet_cycles': 5,

        # Adaptive Morlet settings
        # <= adaptive_freq_low_max Hz uses adaptive_cycles_low
        # >= adaptive_freq_high_min Hz uses adaptive_cycles_high
        # Between them: cycles are linearly interpolated
        'adaptive_cycles_low': 3,
        'adaptive_cycles_high': 7,
        'adaptive_freq_low_max': 15,
        'adaptive_freq_high_min': 16,
        
        # Frequency settings
        'freq_min': 1,
        'freq_max': 120,
        'freq_step': 1,                     # Hz resolution
        
        # Display settings
        'cmap': 'jet', #viridis
        'normalize': 'zscore',              # 'zscore', 'dB', or 'raw'
        'vmin_zscore': -6,                  # z-score limits
        'vmax_zscore': 6,                   # z-score limits
        'vmin_uV2': 0,                      # For absolute power
        'vmax_uV2': None,                   # None = auto-scale to 95th percentile
        'vmin_db': -6,                      # For dB (if using relative)
        'vmax_db': 6,
    },
    
    # Spatial heatmap timepoints
    'spatial_timepoints_ms': [-500, -300, -200, -100, -50, 0, 50, 100, 200, 300, 500],
    
    # Selected band for spatial analysis
    'selected_band': 'Low Gamma',
}

# =============================================================================
# WAVELET COMPARISON / TEST MODE CONFIGURATION
# =============================================================================
# When WAVELET_TEST_CONFIG['enabled'] is True, the script runs a comparison of
# multiple TFR methods on ONE session/region instead of the full pipeline.
# This is designed to diagnose temporal smearing from stimulation transients.

WAVELET_TEST_CONFIG = {
    'enabled': False,          # Set True to run wavelet comparison instead of normal pipeline

    # --- Which region/array to test (must match a key in region_grids) ---
    # e.g. 'M1i', 'SMA', 'PMd', 'M1s', or None for first available
    'test_region': 'M1i',

    # --- Methods to compare ---
    # Each entry: dict with 'label' and 'method' (see compute_tfr_method)
    'methods': [
        # Fixed-cycle Morlet
        {'label': 'Morlet n=3',  'method': 'morlet', 'n_cycles': 3},
        {'label': 'Morlet n=5',  'method': 'morlet', 'n_cycles': 5},
        {'label': 'Morlet n=7',  'method': 'morlet', 'n_cycles': 7},
        {'label': 'Morlet n=10', 'method': 'morlet', 'n_cycles': 10},
        # Adaptive-cycle Morlet: cycles scale with frequency band
        # Low freq (<8 Hz): 3 cycles -> better time res; High freq (>30 Hz): 7 cycles
        {'label': 'Morlet adaptive', 'method': 'morlet_adaptive',
         'n_cycles_low': 3, 'n_cycles_high': 7,
         'freq_low_max': 15, 'freq_high_min': 16},
        # STFT: uniform time-frequency resolution, no wavelet smearing
        {'label': 'STFT 250ms',  'method': 'stft', 'window_ms': 250},
        {'label': 'STFT 100ms',  'method': 'stft', 'window_ms': 100},
        # Paul wavelet (m=4): better time resolution than Morlet, good for transients
        {'label': 'Paul m=4',    'method': 'paul', 'order': 4},
    ],

    # --- Normalization to apply to all methods ---
    'normalize': 'zscore',   # 'zscore', 'dB', or 'raw'
    'baseline_window': (-950, -650),  # ms, same as ANALYSIS_CONFIG default

    # --- Display ---
    'time_range_ms': (-950, 950),   # Time range to display
    'freq_range': (8, 55),          # Focus on low/mid freq where smearing is worst
    'vmin': -6,
    'vmax': 6,
    'cmap': 'jet',

    # --- Output ---
    'output_subdir': 'wavelet_comparison',  # Subfolder inside fig_dir
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _next_power_of_two(n):
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

def interpolate_nans_1d(x):
    """
    Return finite float32 trace with NaN/Inf linearly interpolated.
    Returns None if the whole trace is non-finite.
    """
    x = np.asarray(x, dtype=np.float32)
    finite_mask = np.isfinite(x)

    if not finite_mask.any():
        return None

    if finite_mask.all():
        return x.astype(np.float32)

    idx = np.arange(x.size)
    x_interp = x.copy()
    x_interp[~finite_mask] = np.interp(
        idx[~finite_mask],
        idx[finite_mask],
        x[finite_mask]
    )

    return x_interp.astype(np.float32)

def get_foi_with_notch_gaps(freq_min=1, freq_max=120, freq_step=1, notch_regions=None):
    """
    Create frequency array that skips notch filter regions entirely.
    
    Args:
        freq_min: Minimum frequency
        freq_max: Maximum frequency  
        freq_step: Frequency resolution
        notch_regions: List of (f_lo, f_hi) tuples to skip
    
    Returns:
        foi: Frequency array with gaps at notch regions
    """
    if notch_regions is None:
        notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])
    
    # Create full frequency array
    foi_full = np.arange(freq_min, freq_max + freq_step, freq_step)
    
    # Create mask for frequencies to keep
    keep_mask = np.ones(len(foi_full), dtype=bool)
    
    for f_lo, f_hi in notch_regions:
        notch_mask = (foi_full >= f_lo) & (foi_full <= f_hi)
        keep_mask[notch_mask] = False
    
    return foi_full[keep_mask]

def mask_notch_regions(f, psd, notch_regions=None):
    """
    Set PSD values in notch regions to NaN for cleaner plotting.
    
    Args:
        f: Frequency array
        psd: Power spectral density array (can be 1D or match f on last axis)
        notch_regions: List of (f_lo, f_hi) tuples to mask
    
    Returns:
        psd with notch regions set to NaN
    """
    if notch_regions is None:
        notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])
    
    psd_masked = psd.copy()
    
    for f_lo, f_hi in notch_regions:
        mask = (f >= f_lo) & (f <= f_hi)
        if psd_masked.ndim == 1:
            psd_masked[mask] = np.nan
        else:
            # Assume frequency is last axis
            psd_masked[..., mask] = np.nan
    
    return psd_masked

def compute_confidence_interval(data, axis=0, confidence=0.95):
    """
    Compute mean and confidence interval using SEM × z-score.
    
    Args:
        data: Array of values
        axis: Axis to compute statistics over
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        mean, ci_lower, ci_upper
    """
    z_score = stats.norm.ppf((1 + confidence) / 2)  # 1.96 for 95%
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = np.nanmean(data, axis=axis)
        n = np.sum(~np.isnan(data), axis=axis)
        sem = np.nanstd(data, axis=axis) / np.sqrt(np.maximum(n, 1))
        ci = sem * z_score
    
    return mean, mean - ci, mean + ci

def get_region_color(region_name):
    """Get color for a region, with fallback."""
    colors = PLOT_CONFIG['region_colors']
    # Handle various naming conventions
    for key in colors:
        if key.lower() in region_name.lower():
            return colors[key]
    return 'gray'

def add_band_shading(ax, bands=None, alpha=0.08, skip_notch=True):
    """Add subtle background shading for frequency bands, skipping notch regions."""
    if bands is None:
        bands = ANALYSIS_CONFIG['bands']
    
    notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)]) if skip_notch else []
    
    colors = ['#E91E63', '#9C27B0', '#3F51B5', '#4CAF50', '#FF9800', '#F44336']
    
    for i, (band_name, (f_lo, f_hi)) in enumerate(bands.items()):
        # Check if this band overlaps with any notch region
        for notch_lo, notch_hi in notch_regions:
            if f_lo < notch_hi and f_hi > notch_lo:
                # Band overlaps with notch - split the shading
                if f_lo < notch_lo:
                    ax.axvspan(f_lo, notch_lo, color=colors[i % len(colors)], alpha=alpha)
                if f_hi > notch_hi:
                    ax.axvspan(notch_hi, f_hi, color=colors[i % len(colors)], alpha=alpha)
                break
        else:
            # No overlap with any notch
            ax.axvspan(f_lo, f_hi, color=colors[i % len(colors)], alpha=alpha)

def add_stim_marker(ax, stim_duration_ms=100):
    """Add stimulation period marker to time-series plot."""
    ax.axvspan(0, stim_duration_ms, color='red', alpha=0.1, label='Stim')
    ax.axvline(0, color='red', ls='--', lw=1, alpha=0.7)

def zscore_normalize_spectrogram(Sxx, baseline_mask=None):
    """
    Z-score normalize a spectrogram.
    
    Args:
        Sxx: Power matrix (n_freq, n_time) or (n_trials, n_freq, n_time)
        baseline_mask: Boolean mask for baseline time points (optional)
                      If None, uses all time points for normalization
    
    Returns:
        Sxx_z: Z-scored spectrogram (same shape as input)
    """
    if Sxx is None:
        return None
    
    Sxx = Sxx.astype(np.float64)
    
    if Sxx.ndim == 2:
        # Single spectrogram (n_freq, n_time)
        if baseline_mask is not None and baseline_mask.any():
            # Use baseline period for mean/std
            baseline_data = Sxx[:, baseline_mask]
            mean_val = np.nanmean(baseline_data, axis=1, keepdims=True)
            std_val = np.nanstd(baseline_data, axis=1, keepdims=True)
        else:
            # Use full spectrogram
            mean_val = np.nanmean(Sxx, axis=1, keepdims=True)
            std_val = np.nanstd(Sxx, axis=1, keepdims=True)
        
        # Avoid division by zero
        std_val[std_val < 1e-10] = 1e-10
        
        Sxx_z = (Sxx - mean_val) / std_val
        
    elif Sxx.ndim == 3:
        # Multiple spectrograms (n_trials, n_freq, n_time)
        if baseline_mask is not None and baseline_mask.any():
            baseline_data = Sxx[:, :, baseline_mask]
            mean_val = np.nanmean(baseline_data, axis=(0, 2), keepdims=True)
            std_val = np.nanstd(baseline_data, axis=(0, 2), keepdims=True)
        else:
            mean_val = np.nanmean(Sxx, axis=(0, 2), keepdims=True)
            std_val = np.nanstd(Sxx, axis=(0, 2), keepdims=True)
        
        std_val[std_val < 1e-10] = 1e-10
        Sxx_z = (Sxx - mean_val) / std_val
    else:
        raise ValueError(f"Unexpected Sxx shape: {Sxx.shape}")
    
    return Sxx_z

def compute_array_median_raw_spectrograms(data, groups, fs=1000,
                                          region_grids=None, elec_to_idx=None):
    """
    Compute raw Morlet spectrogram median across all valid trial/channel pairs
    for each array/region.

    This is the centralized raw array-median spectrogram helper used by:
      - generate_per_ua_array_median_spectrogram(...)
      - compute_control_median_baseline_norm(...)

    Return keys are intentionally preserved:
      'Sxx'
      'f'
      't_ms'
      'n_valid_channels'
      'n_valid_trial_channel_pairs'
    """

    spec_cfg = PLOT_CONFIG.get('spectrogram', {})

    freq_min = spec_cfg.get('freq_min', 1)
    freq_max = spec_cfg.get('freq_max', 120)
    freq_step = spec_cfg.get('freq_step', 1)

    tfr_method = spec_cfg.get('tfr_method', 'morlet').lower()

    n_cycles = spec_cfg.get(
        'wavelet_cycles',
        ANALYSIS_CONFIG.get('wavelet_cycles', 7)
    )

    adaptive_cycles_low = spec_cfg.get('adaptive_cycles_low', 3)
    adaptive_cycles_high = spec_cfg.get('adaptive_cycles_high', 7)
    adaptive_freq_low_max = spec_cfg.get('adaptive_freq_low_max', 15)
    adaptive_freq_high_min = spec_cfg.get('adaptive_freq_high_min', 16)

    if (
        tfr_method == 'morlet_adaptive'
        and adaptive_freq_high_min <= adaptive_freq_low_max
    ):
        print(
            "    WARNING: adaptive_freq_high_min must be greater than "
            "adaptive_freq_low_max. Expanding transition band automatically."
        )
        adaptive_freq_high_min = adaptive_freq_low_max + 15

    foi = get_foi_with_notch_gaps(freq_min, freq_max, freq_step)

    # Adaptive Morlet uses CPU path unless/until Torch adaptive support is added.
    # Torch GPU path supports fixed-cycle Morlet and adaptive Morlet.
    use_gpu = (
        tfr_method in ('morlet', 'morlet_adaptive')
        and ANALYSIS_CONFIG.get('use_gpu_wavelet', False)
        and ANALYSIS_CONFIG.get('gpu_backend', 'torch').lower() == 'torch'
        and TORCH_AVAILABLE
        and torch is not None
        and torch.cuda.is_available()
    )

    gpu_batch_size = int(ANALYSIS_CONFIG.get('gpu_batch_size', 32))

    if use_gpu:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA"

        if tfr_method == 'morlet_adaptive':
            print(
                f"    Torch GPU adaptive Morlet enabled: {gpu_name}, "
                f"{adaptive_cycles_low}->{adaptive_cycles_high} cycles, "
                f"{adaptive_freq_low_max}-{adaptive_freq_high_min} Hz transition, "
                f"batch_size={gpu_batch_size}"
            )
        else:
            print(
                f"    Torch GPU fixed Morlet enabled: {gpu_name}, "
                f"n_cycles={n_cycles}, batch_size={gpu_batch_size}"
            )
    else:
        if tfr_method == 'morlet_adaptive':
            print(
                "    Adaptive Morlet enabled: "
                f"{adaptive_cycles_low}->{adaptive_cycles_high} cycles, "
                f"{adaptive_freq_low_max}-{adaptive_freq_high_min} Hz transition "
                "(CPU path)"
            )
        else:
            print("    Torch GPU Morlet not enabled/available; using CPU fixed Morlet.")

    region_specs_raw = {}

    if region_grids is None or elec_to_idx is None:
        print("    WARNING: region_grids or elec_to_idx missing; cannot compute array median raw spectrograms.")
        return region_specs_raw

    bb_full, t_ms = get_broadband_full(data)

    if bb_full is None:
        print("    WARNING: broadband data missing; cannot compute array median raw spectrograms.")
        return region_specs_raw

    bb_full = np.asarray(bb_full)

    if bb_full.ndim != 3:
        print(f"    WARNING: expected broadband data shape trial x channel x time, got {bb_full.shape}")
        return region_specs_raw

    n_trials, n_channels, n_time = bb_full.shape

    for region_name, grid_elec in region_grids.items():
        print(f"    Computing raw array-median spectrogram for {region_name}...")

        region_traces = []
        valid_channel_indices = []

        grid_elec_arr = np.asarray(grid_elec)

        for elec_id in grid_elec_arr.ravel():
            if elec_id is None:
                continue

            try:
                if np.isnan(elec_id):
                    continue
            except Exception:
                pass

            ch_idx = elec_to_idx.get(int(elec_id), None)

            if ch_idx is None:
                continue

            if ch_idx < 0 or ch_idx >= n_channels:
                continue

            valid_channel_indices.append(ch_idx)

            for trial_idx in range(n_trials):
                trace = bb_full[trial_idx, ch_idx, :]

                if trace is None:
                    continue

                trace = np.asarray(trace)

                if trace.size != n_time:
                    continue

                if not np.isfinite(trace).any():
                    continue

                region_traces.append(trace.astype(np.float32))

        n_valid_channels = len(set(valid_channel_indices))
        n_valid_trial_channel_pairs = len(region_traces)

        if n_valid_trial_channel_pairs == 0:
            print(f"      No valid trial/channel pairs for {region_name}; skipping.")
            continue

        traces_arr = np.stack(region_traces, axis=0).astype(np.float32)

        Sxx_region_median_raw = None
        f_out_saved = None
        t_spec_ms_saved = None

        if use_gpu:
            try:
                if tfr_method == 'morlet_adaptive':
                    Sxx_all, f_out, t_out = compute_wavelet_spectrogram_batch_torch_adaptive(
                        traces_arr,
                        fs=fs,
                        foi=foi,
                        n_cycles_low=adaptive_cycles_low,
                        n_cycles_high=adaptive_cycles_high,
                        freq_low_max=adaptive_freq_low_max,
                        freq_high_min=adaptive_freq_high_min,
                        batch_size=gpu_batch_size,
                        device="cuda",
                    )
                else:
                    Sxx_all, f_out, t_out = compute_wavelet_spectrogram_batch_torch(
                        traces_arr,
                        fs=fs,
                        foi=foi,
                        n_cycles=n_cycles,
                        batch_size=gpu_batch_size,
                        device="cuda",
                    )

                Sxx_region_median_raw = np.nanmedian(Sxx_all, axis=0)
                f_out_saved = f_out
                t_spec_ms_saved = np.asarray(t_out) * 1000.0 + float(t_ms[0])

                print(
                    f"      {region_name}: GPU {tfr_method} complete, "
                    f"{n_valid_channels} channels, "
                    f"{n_valid_trial_channel_pairs} trial/channel pairs"
                )

            except Exception as e:
                print(f"      WARNING: Torch GPU {tfr_method} failed for {region_name}: {e}")
                print(f"      Falling back to CPU {tfr_method} for this region.")
                Sxx_region_median_raw = None

                if torch is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

        if Sxx_region_median_raw is None:
            specs = []
            f_out_saved = None
            t_spec_ms_saved = None

            if tfr_method == 'morlet_adaptive':
                method_cfg = {
                    'method': 'morlet_adaptive',
                    'n_cycles_low': adaptive_cycles_low,
                    'n_cycles_high': adaptive_cycles_high,
                    'freq_low_max': adaptive_freq_low_max,
                    'freq_high_min': adaptive_freq_high_min,
                }
            else:
                method_cfg = {
                    'method': 'morlet',
                    'n_cycles': n_cycles,
                }

            for trace in region_traces:
                Sxx, f_out, t_out = compute_tfr_method(
                    trace,
                    fs,
                    method_cfg,
                    foi=foi,
                    t_ms=t_ms,
                )

                if Sxx is None:
                    continue

                specs.append(Sxx)

                if f_out_saved is None:
                    f_out_saved = f_out
                    t_spec_ms_saved = np.asarray(t_out) * 1000.0 + float(t_ms[0])

            if len(specs) == 0:
                print(f"      No valid CPU spectrograms for {region_name}; skipping.")
                continue

            stacked = np.stack(specs, axis=0)
            Sxx_region_median_raw = np.nanmedian(stacked, axis=0)

            print(
                f"      {region_name}: CPU {tfr_method} complete, "
                f"{n_valid_channels} channels, "
                f"{n_valid_trial_channel_pairs} trial/channel pairs"
            )

        region_specs_raw[region_name] = {
            'Sxx': Sxx_region_median_raw,
            'f': f_out_saved,
            't_ms': t_spec_ms_saved,
            'n_valid_channels': n_valid_channels,
            'n_valid_trial_channel_pairs': n_valid_trial_channel_pairs,
        }

    return region_specs_raw

def get_trial_count(data):
    return int(data['broadband_full'].shape[0])

def compute_control_median_baseline_norm(data, groups, fs=1000,
                                         region_grids=None, elec_to_idx=None):
    """
    Compute control-reference mean/std from the baseline of the control
    array-level median spectrogram.
    """
    region_specs_raw = compute_array_median_raw_spectrograms(
        data=data,
        groups=groups,
        fs=fs,
        region_grids=region_grids,
        elec_to_idx=elec_to_idx,
    )

    baseline_win = ANALYSIS_CONFIG['baseline_window']
    norm = {}

    for region_name, spec in region_specs_raw.items():
        Sxx_control_median_raw = spec['Sxx']
        t_spec_ms = spec['t_ms']

        bl_mask = (
            (t_spec_ms >= baseline_win[0]) &
            (t_spec_ms <= baseline_win[1])
        )

        if not bl_mask.any():
            print(f"      [Skip] No baseline samples for control norm: {region_name}")
            continue

        baseline_values = Sxx_control_median_raw[:, bl_mask]

        mean_val = np.nanmean(baseline_values, axis=1, keepdims=True)
        std_val = np.nanstd(baseline_values, axis=1, keepdims=True)
        std_val[std_val < 1e-10] = 1e-10

        norm[region_name] = {
            'mean': mean_val,
            'std': std_val,
            'foi': spec['f'],
            'baseline_window': baseline_win,
            'tfr_method': PLOT_CONFIG.get('spectrogram', {}).get('tfr_method', 'morlet'),
            'n_valid_channels': spec['n_valid_channels'],
            'n_valid_trial_channel_pairs': spec['n_valid_trial_channel_pairs'],
        }

    return norm

def compute_control_spatial_baseline(data, fs=1000):
    """
    Compute control baseline envelope for selected spatial band.
    Shape returned: (1, n_channels, 1)
    """
    bb_full, t_ms = get_broadband_full(data)

    band_name = PLOT_CONFIG.get('selected_band', 'Beta')
    f_lo, f_hi = ANALYSIS_CONFIG['bands'][band_name]
    baseline_win = ANALYSIS_CONFIG['baseline_window']

    nyq = fs / 2
    if f_hi >= nyq:
        f_hi = nyq - 1

    b, a = signal.butter(4, [f_lo / nyq, f_hi / nyq], btype='band')

    filtered = signal.filtfilt(b, a, bb_full, axis=2)
    envelope = np.abs(signal.hilbert(filtered, axis=2))

    bl_mask = (t_ms >= baseline_win[0]) & (t_ms <= baseline_win[1])

    baseline_power = np.nanmean(envelope[:, :, bl_mask], axis=(0, 2))
    baseline_power[baseline_power < 1e-10] = 1e-10

    return baseline_power[None, :, None]

def get_time_axis(data):
    """Extract time axis in ms from data dict."""
    for key in ['t_full_ms', 't_ms', 'rel_time_full']:
        if key in data:
            return np.array(data[key])
    
    # If no full time axis found, try to construct from pre+post
    if 'rel_time_pre' in data and 'rel_time_post' in data:
        t_pre = np.array(data['rel_time_pre'])
        t_post = np.array(data['rel_time_post'])
        return np.concatenate([t_pre, t_post])
    
    raise KeyError("No time axis found in data (tried: t_full_ms, t_ms, rel_time_full, rel_time_pre+rel_time_post)")

def get_broadband_full(data):
    """
    Get full broadband data array and corresponding time axis.
    
    Prefers 'broadband_full' if available, otherwise constructs from pre+post.
    
    Returns:
        (broadband_full, t_ms): Arrays of shape (n_trials, n_channels, n_times) and (n_times,)
    """
    if 'broadband_full' in data:
        t_ms = get_time_axis(data)
        return np.array(data['broadband_full']), t_ms
    
    # Construct from pre + post (legacy format)
    if 'broadband_pre' not in data or 'broadband_post' not in data:
        raise KeyError("Data must contain either 'broadband_full' or both 'broadband_pre' and 'broadband_post'")
    
    pre = np.array(data['broadband_pre'])
    post = np.array(data['broadband_post'])
    
    # Get time axes - check multiple possible key names
    t_pre = None
    t_post = None
    
    for key in ['rel_time_pre', 't_pre_ms']:
        if key in data:
            t_pre = np.array(data[key])
            break
    
    for key in ['rel_time_post', 't_post_ms']:
        if key in data:
            t_post = np.array(data[key])
            break
    
    if t_pre is None or t_post is None:
        raise KeyError("Missing time axes for reconstruction (tried: rel_time_pre/t_pre_ms and rel_time_post/t_post_ms)")
    
    # Concatenate
    broadband_full = np.concatenate([pre, post], axis=-1)
    t_ms = np.concatenate([t_pre, t_post])
    
    return broadband_full, t_ms

def slice_time_window(data_3d, t_ms, t_start, t_end):
    """
    Slice 3D data array to a specific time window.
    
    Args:
        data_3d: Array of shape (n_trials, n_channels, n_times)
        t_ms: Time axis in ms
        t_start: Start time in ms (inclusive)
        t_end: End time in ms (inclusive)
    
    Returns:
        (sliced_data, sliced_t_ms): Sliced arrays
    """
    mask = (t_ms >= t_start) & (t_ms <= t_end)
    return data_3d[..., mask], t_ms[mask]

# =============================================================================
# MORLET WAVELET SPECTROGRAM (replaces superlet)
# =============================================================================

def compute_wavelet_spectrogram(trace, fs, foi=None, baseline_window=None,
                                 t_ms=None, n_cycles=7, normalize=False):
    """
    Compute time-frequency representation using Morlet wavelets.
    
    Args:
        trace: 1D signal
        fs: Sampling rate
        foi: Frequencies of interest (should already exclude notch regions)
        baseline_window: (start_ms, end_ms) - stored for later normalization
        t_ms: Time axis in ms
        n_cycles: Number of cycles in wavelet
        normalize: If True, apply dB normalization (legacy behavior)
    
    Returns:
        Sxx: Power matrix (n_freq, n_time) in µV² (raw power, not normalized)
        foi: Frequencies
        t_sec: Time in seconds
    """
    if foi is None:
        # Use frequencies with notch gaps by default
        foi = get_foi_with_notch_gaps()
    
    # Handle NaN values
    trace_clean = trace.copy().astype(np.float64)
    nan_mask = np.isnan(trace_clean)
    if nan_mask.any():
        if nan_mask.all():
            return None, None, None
        trace_clean[nan_mask] = np.interp(
            np.flatnonzero(nan_mask),
            np.flatnonzero(~nan_mask),
            trace_clean[~nan_mask]
        )
    
    n_time = len(trace_clean)
    n_freq = len(foi)
    
    # Initialize power matrix
    Sxx = np.zeros((n_freq, n_time))
    
    # Compute wavelet transform for each frequency
    for i_freq, freq in enumerate(foi):
        if freq <= 0:
            continue
            
        # Morlet wavelet parameters
        sigma_t = n_cycles / (2 * np.pi * freq)
        
        # Create wavelet
        wavelet_duration = 4 * sigma_t
        wavelet_samples = int(wavelet_duration * fs)
        if wavelet_samples % 2 == 0:
            wavelet_samples += 1
        if wavelet_samples < 3:
            wavelet_samples = 3
        
        t_wavelet = np.arange(wavelet_samples) / fs - wavelet_duration / 2
        
        # Morlet wavelet = Gaussian * complex exponential
        gaussian = np.exp(-t_wavelet**2 / (2 * sigma_t**2))
        wavelet = gaussian * np.exp(2j * np.pi * freq * t_wavelet)
        
        # Normalize
        wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet)**2))
        
        # Convolve
        analytic = signal.fftconvolve(trace_clean, wavelet, mode='same')
        
        # Power
        Sxx[i_freq, :] = np.abs(analytic) ** 2
    
    # Time axis
    t_sec = np.arange(n_time) / fs
    
    # Legacy dB normalization (only if explicitly requested)
    if normalize and baseline_window is not None and t_ms is not None:
        t_spec_ms = t_sec * 1000 + t_ms[0]
        bl_mask = (t_spec_ms >= baseline_window[0]) & (t_spec_ms <= baseline_window[1])
        if bl_mask.sum() > 0:
            baseline_power = np.nanmean(Sxx[:, bl_mask], axis=1, keepdims=True)
            baseline_power[baseline_power < 1e-20] = 1e-20
            Sxx = 10 * np.log10(Sxx / baseline_power)
    
    return Sxx, foi, t_sec

def compute_wavelet_spectrogram_batch_torch(traces, fs, foi=None, n_cycles=7, batch_size=32, device=None):
    """
    GPU Torch Morlet spectrogram for a batch of traces.

    Parameters
    ----------
    traces : array-like
        Shape: (n_signals, n_time)
    fs : float
        Sampling rate in Hz.
    foi : array-like or None
        Frequencies of interest.
    n_cycles : float
        Morlet wavelet cycles.
    batch_size : int
        Number of traces processed per GPU batch.
    device : str or torch.device or None
        Usually "cuda".

    Returns
    -------
    Sxx_all : np.ndarray
        Shape: (n_signals, n_freqs, n_time)
    foi : np.ndarray
        Frequencies.
    t_out : np.ndarray
        Time vector in seconds, relative to trace start.
    """

    if torch is None:
        raise RuntimeError("Torch is not available.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA requested but not available.")

    traces = np.asarray(traces)

    if traces.ndim != 2:
        raise ValueError(f"traces must be 2D, got shape {traces.shape}")

    n_signals, n_time = traces.shape

    if foi is None:
        spec_cfg = PLOT_CONFIG.get('spectrogram', {})
        freq_min = spec_cfg.get('freq_min', 1)
        freq_max = spec_cfg.get('freq_max', 120)
        freq_step = spec_cfg.get('freq_step', 1)
        foi = get_foi_with_notch_gaps(freq_min, freq_max, freq_step)
    else:
        foi = np.asarray(foi, dtype=np.float32)

    n_freqs = len(foi)
    t_out = np.arange(n_time, dtype=np.float32) / float(fs)

    # Clean traces on CPU before GPU transfer.
    cleaned = []
    for i in range(n_signals):
        xi = interpolate_nans_1d(traces[i])
        if xi is None:
            xi = np.zeros(n_time, dtype=np.float32)
        cleaned.append(xi)

    traces_clean = np.stack(cleaned, axis=0).astype(np.float32)

    Sxx_all = np.empty((n_signals, n_freqs, n_time), dtype=np.float32)

    for start in range(0, n_signals, batch_size):
        stop = min(start + batch_size, n_signals)

        x_np = traces_clean[start:stop]
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)

        this_batch = stop - start
        Sxx_batch = torch.empty(
            (this_batch, n_freqs, n_time),
            dtype=torch.float32,
            device=device
        )

        for fi, freq in enumerate(foi):
            freq_float = float(freq)

            sigma_t = n_cycles / (2 * np.pi * freq_float)
            wavelet_duration = 4 * sigma_t
            wavelet_samples = int(wavelet_duration * fs)

            if wavelet_samples % 2 == 0:
                wavelet_samples += 1

            if wavelet_samples < 3:
                wavelet_samples = 3

            t_wavelet = (
                torch.arange(wavelet_samples, device=device, dtype=torch.float32)
                / float(fs)
                - wavelet_duration / 2.0
            )

            gaussian = torch.exp(-(t_wavelet ** 2) / (2 * sigma_t ** 2))

            phase = 2 * np.pi * freq_float * t_wavelet
            complex_sinusoid = torch.exp(1j * phase)

            wavelet = gaussian.to(torch.complex64) * complex_sinusoid.to(torch.complex64)

            wavelet_norm = torch.sqrt(torch.sum(torch.abs(wavelet) ** 2))
            wavelet = wavelet / wavelet_norm

            conv_len = n_time + wavelet_samples - 1
            n_fft = _next_power_of_two(conv_len)

            x_fft = torch.fft.fft(x.to(torch.complex64), n=n_fft, dim=1)
            w_fft = torch.fft.fft(wavelet, n=n_fft)

            analytic_full = torch.fft.ifft(x_fft * w_fft[None, :], n=n_fft, dim=1)
            analytic_full = analytic_full[:, :conv_len]

            same_start = (wavelet_samples - 1) // 2
            analytic_same = analytic_full[:, same_start:same_start + n_time]

            power = torch.abs(analytic_same) ** 2
            Sxx_batch[:, fi, :] = power.to(torch.float32)

        Sxx_all[start:stop] = Sxx_batch.detach().cpu().numpy()

        del x, Sxx_batch

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return Sxx_all, foi.astype(np.float32), t_out

def compute_wavelet_spectrogram_batch_torch_adaptive(
        traces, fs, foi=None,
        n_cycles_low=3, n_cycles_high=7,
        freq_low_max=15, freq_high_min=30,
        batch_size=32, device=None):
    """
    GPU Torch adaptive Morlet spectrogram for a batch of traces.

    Same output format as compute_wavelet_spectrogram_batch_torch(...), but
    n_cycles changes by frequency:

      freq <= freq_low_max   -> n_cycles_low
      freq >= freq_high_min  -> n_cycles_high
      between                -> linear interpolation
    """

    if torch is None:
        raise RuntimeError("Torch is not available.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA requested but not available.")

    traces = np.asarray(traces)

    if traces.ndim != 2:
        raise ValueError(f"traces must be 2D, got shape {traces.shape}")

    n_signals, n_time = traces.shape

    if foi is None:
        spec_cfg = PLOT_CONFIG.get('spectrogram', {})
        freq_min = spec_cfg.get('freq_min', 1)
        freq_max = spec_cfg.get('freq_max', 120)
        freq_step = spec_cfg.get('freq_step', 1)
        foi = get_foi_with_notch_gaps(freq_min, freq_max, freq_step)
    else:
        foi = np.asarray(foi, dtype=np.float32)

    n_freqs = len(foi)
    t_out = np.arange(n_time, dtype=np.float32) / float(fs)

    if freq_high_min <= freq_low_max:
        raise ValueError(
            f"freq_high_min must be > freq_low_max for adaptive Morlet; "
            f"got freq_low_max={freq_low_max}, freq_high_min={freq_high_min}"
        )

    # Precompute cycle count per frequency on CPU.
    n_cycles_by_freq = np.empty(n_freqs, dtype=np.float32)

    for fi, freq in enumerate(foi):
        freq_float = float(freq)

        if freq_float <= freq_low_max:
            nc = float(n_cycles_low)
        elif freq_float >= freq_high_min:
            nc = float(n_cycles_high)
        else:
            frac = (freq_float - float(freq_low_max)) / (
                float(freq_high_min) - float(freq_low_max)
            )
            nc = float(n_cycles_low) + frac * (
                float(n_cycles_high) - float(n_cycles_low)
            )

        n_cycles_by_freq[fi] = nc

    # Clean traces on CPU before GPU transfer.
    cleaned = []

    for i in range(n_signals):
        xi = interpolate_nans_1d(traces[i])
        if xi is None:
            xi = np.zeros(n_time, dtype=np.float32)
        cleaned.append(xi)

    traces_clean = np.stack(cleaned, axis=0).astype(np.float32)

    Sxx_all = np.empty((n_signals, n_freqs, n_time), dtype=np.float32)

    for start in range(0, n_signals, batch_size):
        stop = min(start + batch_size, n_signals)

        x_np = traces_clean[start:stop]
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)

        this_batch = stop - start
        Sxx_batch = torch.empty(
            (this_batch, n_freqs, n_time),
            dtype=torch.float32,
            device=device
        )

        for fi, freq in enumerate(foi):
            freq_float = float(freq)
            nc = float(n_cycles_by_freq[fi])

            if freq_float <= 0:
                Sxx_batch[:, fi, :] = 0.0
                continue

            sigma_t = nc / (2 * np.pi * freq_float)
            wavelet_duration = 4 * sigma_t
            wavelet_samples = int(wavelet_duration * fs)

            if wavelet_samples % 2 == 0:
                wavelet_samples += 1

            if wavelet_samples < 3:
                wavelet_samples = 3

            t_wavelet = (
                torch.arange(wavelet_samples, device=device, dtype=torch.float32)
                / float(fs)
                - wavelet_duration / 2.0
            )

            gaussian = torch.exp(-(t_wavelet ** 2) / (2 * sigma_t ** 2))

            phase = 2 * np.pi * freq_float * t_wavelet
            complex_sinusoid = torch.exp(1j * phase)

            wavelet = gaussian.to(torch.complex64) * complex_sinusoid.to(torch.complex64)

            wavelet_norm = torch.sqrt(torch.sum(torch.abs(wavelet) ** 2))
            wavelet = wavelet / wavelet_norm

            conv_len = n_time + wavelet_samples - 1
            n_fft = _next_power_of_two(conv_len)

            x_fft = torch.fft.fft(x.to(torch.complex64), n=n_fft, dim=1)
            w_fft = torch.fft.fft(wavelet, n=n_fft)

            analytic_full = torch.fft.ifft(x_fft * w_fft[None, :], n=n_fft, dim=1)
            analytic_full = analytic_full[:, :conv_len]

            same_start = (wavelet_samples - 1) // 2
            analytic_same = analytic_full[:, same_start:same_start + n_time]

            power = torch.abs(analytic_same) ** 2
            Sxx_batch[:, fi, :] = power.to(torch.float32)

        Sxx_all[start:stop] = Sxx_batch.detach().cpu().numpy()

        del x, Sxx_batch

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return Sxx_all, foi.astype(np.float32), t_out

# =============================================================================
# ALTERNATIVE TFR METHODS FOR WAVELET COMPARISON
# =============================================================================

def compute_morlet_adaptive_spectrogram(trace, fs, foi=None, t_ms=None,
                                        n_cycles_low=3, n_cycles_high=7,
                                        freq_low_max=8, freq_high_min=30):
    """
    Morlet wavelet with frequency-adaptive n_cycles.

    At low frequencies (<= freq_low_max Hz): uses n_cycles_low
      -> shorter window, better time resolution, less temporal smearing
    At high frequencies (>= freq_high_min Hz): uses n_cycles_high
      -> more cycles, better frequency resolution where it matters
    In between: linearly interpolated.

    This is the key approach to reduce stim-transient smearing at low freqs.
    """
    if foi is None:
        foi = get_foi_with_notch_gaps()

    trace_clean = interpolate_nans_1d(np.asarray(trace, dtype=np.float64))
    if trace_clean is None:
        return None, None, None

    n_time = len(trace_clean)
    n_freq = len(foi)
    Sxx = np.zeros((n_freq, n_time), dtype=np.float64)

    for i_freq, freq in enumerate(foi):
        if freq <= 0:
            continue

        # Adaptive cycle count: linear interpolation between low and high
        if freq <= freq_low_max:
            nc = float(n_cycles_low)
        elif freq >= freq_high_min:
            nc = float(n_cycles_high)
        else:
            t = (freq - freq_low_max) / (freq_high_min - freq_low_max)
            nc = n_cycles_low + t * (n_cycles_high - n_cycles_low)

        sigma_t = nc / (2 * np.pi * freq)
        wavelet_duration = 4 * sigma_t
        wavelet_samples = int(wavelet_duration * fs)
        if wavelet_samples % 2 == 0:
            wavelet_samples += 1
        wavelet_samples = max(wavelet_samples, 3)

        t_w = np.arange(wavelet_samples) / fs - wavelet_duration / 2
        gaussian = np.exp(-t_w**2 / (2 * sigma_t**2))
        wavelet = gaussian * np.exp(2j * np.pi * freq * t_w)
        wavelet /= np.sqrt(np.sum(np.abs(wavelet)**2))

        analytic = signal.fftconvolve(trace_clean, wavelet, mode='same')
        Sxx[i_freq, :] = np.abs(analytic) ** 2

    t_sec = np.arange(n_time) / fs
    return Sxx, foi, t_sec


def compute_stft_spectrogram(trace, fs, foi=None, window_ms=200, t_ms=None):
    """
    Short-Time Fourier Transform spectrogram.

    Unlike wavelets, STFT uses a FIXED window length -> uniform time AND
    frequency resolution everywhere. No temporal smearing that scales with
    frequency. Good for detecting sharp transients.

    Shorter window_ms = better time resolution but coarser freq resolution.
    """
    if foi is None:
        foi = get_foi_with_notch_gaps()

    trace_clean = interpolate_nans_1d(np.asarray(trace, dtype=np.float64))
    if trace_clean is None:
        return None, None, None

    n_time = len(trace_clean)
    window_samps = int(window_ms / 1000.0 * fs)
    if window_samps % 2 == 0:
        window_samps += 1
    window_samps = max(window_samps, 7)

    # 75% overlap for smooth time axis
    noverlap = int(window_samps * 0.75)
    nperseg = window_samps

    freqs_stft, t_stft, Zxx = signal.spectrogram(
        trace_clean,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density',
        mode='psd',
    )

    # Interpolate to our foi grid for consistent comparison
    Sxx_interp = np.zeros((len(foi), len(t_stft)), dtype=np.float64)
    for i_t in range(len(t_stft)):
        Sxx_interp[:, i_t] = np.interp(foi, freqs_stft, Zxx[:, i_t],
                                        left=np.nan, right=np.nan)

    return Sxx_interp, foi, t_stft


def compute_paul_wavelet_spectrogram(trace, fs, foi=None, order=4, t_ms=None):
    """
    Paul wavelet spectrogram.

    Paul wavelets have BETTER TIME RESOLUTION than Morlet at equivalent
    frequency resolution. They are one-sided (analytic) and particularly
    well-suited for detecting transient events and onset responses.

    The Paul wavelet of order m has:
      - Time-bandwidth product: sigma_t * sigma_f = (m + 0.5) / (4*pi)
      - Compared to Morlet: better time, slightly worse freq resolution

    Scales are chosen so that the center frequency matches each FOI.
    """
    if foi is None:
        foi = get_foi_with_notch_gaps()

    trace_clean = interpolate_nans_1d(np.asarray(trace, dtype=np.float64))
    if trace_clean is None:
        return None, None, None

    n_time = len(trace_clean)
    n_freq = len(foi)
    Sxx = np.zeros((n_freq, n_time), dtype=np.float64)

    # Paul wavelet center frequency factor: psi_hat peaks at omega = 2*(m+1)
    # => center_freq = (2*(m+1)) / (2*pi) * (1/scale) in Hz
    # => scale = (2*(m+1)) / (2*pi*freq)
    m = order
    omega_0 = 2 * (m + 1)  # angular frequency of peak

    for i_freq, freq in enumerate(foi):
        if freq <= 0:
            continue

        scale = omega_0 / (2 * np.pi * freq)  # in seconds
        scale_samps = scale * fs

        # Window half-width: 4*scale for reasonable truncation
        half_w = int(4 * scale_samps)
        half_w = max(half_w, 2)
        wlen = 2 * half_w + 1

        t_w = (np.arange(wlen) - half_w) / fs  # in seconds

        # Paul wavelet (unnormalized analytic form)
        # psi(t/s) = (2^m * i^m * m!) / sqrt(pi*(2m)!) * (1 - i*t/s)^(-(m+1))
        import math
        norm_const = (2**m * math.factorial(m)) / np.sqrt(np.pi * math.factorial(2 * m))
        eta = t_w / scale  # normalized time
        wavelet = norm_const * (1 - 1j * eta) ** (-(m + 1))

        # Normalize for unit energy
        wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet)**2))

        analytic = signal.fftconvolve(trace_clean, np.conj(wavelet[::-1]), mode='same')
        Sxx[i_freq, :] = np.abs(analytic) ** 2

    t_sec = np.arange(n_time) / fs
    return Sxx, foi, t_sec


def compute_tfr_method(trace, fs, method_cfg, foi=None, t_ms=None):
    """
    Dispatch function: compute TFR using the method specified in method_cfg.

    Returns (Sxx, foi, t_sec) where:
      Sxx : (n_freq, n_time) raw power
      foi : frequency array
      t_sec : time in seconds from trace start
    """
    method = method_cfg.get('method', 'morlet')

    if method == 'morlet':
        return compute_wavelet_spectrogram(
            trace, fs, foi=foi, t_ms=t_ms,
            n_cycles=method_cfg.get('n_cycles', 7),
            normalize=False,
        )
    elif method == 'morlet_adaptive':
        return compute_morlet_adaptive_spectrogram(
            trace, fs, foi=foi, t_ms=t_ms,
            n_cycles_low=method_cfg.get('n_cycles_low', 3),
            n_cycles_high=method_cfg.get('n_cycles_high', 7),
            freq_low_max=method_cfg.get('freq_low_max', 8),
            freq_high_min=method_cfg.get('freq_high_min', 30),
        )
    elif method == 'stft':
        return compute_stft_spectrogram(
            trace, fs, foi=foi,
            window_ms=method_cfg.get('window_ms', 200),
            t_ms=t_ms,
        )
    elif method == 'paul':
        return compute_paul_wavelet_spectrogram(
            trace, fs, foi=foi,
            order=method_cfg.get('order', 4),
            t_ms=t_ms,
        )
    else:
        raise ValueError(f"Unknown TFR method: {method!r}")


def run_wavelet_comparison(lfp_dir, fig_dir, label=""):
    """
    Run a side-by-side comparison of TFR methods on one session/region.

    Controlled by WAVELET_TEST_CONFIG. Outputs a grid figure with one column
    per method. Each method gets two rows: raw spectrogram + time-frequency
    resolution indicator (wavelet duration as a function of frequency).
    Also plots the mean LFP trace above for reference.

    This is the primary tool for diagnosing temporal smearing from the
    stimulation low-frequency transient.
    """
    cfg = WAVELET_TEST_CONFIG
    if not cfg.get('enabled', False):
        return

    lfp_dir = Path(lfp_dir)
    fig_dir = Path(fig_dir)

    print(f"\n{'='*60}")
    print(f"WAVELET COMPARISON MODE ({label})")
    print(f"{'='*60}")

    # --- Find session file ---
    files = sorted(lfp_dir.rglob("aligned_lfp__*.npz"))
    if not files:
        print(f"  [Skip] No aligned LFP npz files found in {lfp_dir}")
        return

    # Apply the same session filtering logic used in process_directory(...)
    files_filtered = []

    for npz_candidate in files:
        session_candidate_id = npz_candidate.stem.replace("aligned_lfp__", "")

        # Same SESSION_ID_PATTERN logic as process_directory(...)
        if SESSION_ID_PATTERN is not None:
            if not re.search(SESSION_ID_PATTERN, session_candidate_id, re.IGNORECASE):
                continue

        # Same SESSION_FILTER logic as process_directory(...)
        if SESSION_FILTER is not None:
            br_match = re.search(r'_(\d{3})(?:_|$|\.)', session_candidate_id)
            if br_match:
                br_idx = int(br_match.group(1))
                if br_idx not in SESSION_FILTER:
                    continue

        files_filtered.append(npz_candidate)

    if not files_filtered:
        print(
            f"  [Skip] No aligned LFP npz files matched "
            f"SESSION_FILTER={SESSION_FILTER}, SESSION_ID_PATTERN={SESSION_ID_PATTERN}"
        )
        return

    # Prefer stim/non-control files after applying the same filter.
    non_control_files = [
        f for f in files_filtered
        if 'baseline' not in f.stem.lower()
        and 'control' not in f.stem.lower()
    ]

    npz_path = non_control_files[0] if non_control_files else files_filtered[0]

    selected_session_id = npz_path.stem.replace("aligned_lfp__", "")
    selected_br_match = re.search(r'_(\d{3})(?:_|$|\.)', selected_session_id)
    selected_br_idx = int(selected_br_match.group(1)) if selected_br_match else None

    print(
        f"  Applied standard session filters: "
        f"SESSION_FILTER={SESSION_FILTER}, SESSION_ID_PATTERN={SESSION_ID_PATTERN}"
    )
    print(f"  Selected BR={selected_br_idx}: {npz_path.name}")

    session_id = npz_path.stem.replace('aligned_lfp__', '')
    print(f"  Session: {session_id}")

    try:
        data = dict(np.load(npz_path, allow_pickle=True))
    except Exception as e:
        print(f"  [Error] Failed to load {npz_path}: {e}")
        return

    if 'broadband_full' not in data:
        print(f"  [Skip] No broadband_full in {npz_path.name}")
        return

    groups, nsp_to_elec, elec_to_idx, region_grids, ua_ids = _build_groups(data, data['broadband_full'].shape[1])
    fs = int(data.get('fs_lfp', 1000))
    bb_full, t_ms = get_broadband_full(data)
    bb_full = np.asarray(bb_full)
    n_trials, n_channels, n_time = bb_full.shape
    t_ms_arr = np.asarray(t_ms)

    # --- Find region ---
    test_region = cfg.get('test_region')
    region_name = None

    if region_grids and test_region is not None:
        if test_region in region_grids:
            region_name = test_region
        else:
            print(f"  [Warn] test_region={test_region!r} not found. Available: {list(region_grids.keys())}")

    if region_name is None and region_grids:
        region_name = next(iter(region_grids))

    print(f"  Region: {region_name}")

    # --- Collect ALL traces for this region ---
    region_traces = []
    valid_channel_indices = []

    if region_grids and region_name and elec_to_idx:
        grid_elec = np.asarray(region_grids[region_name])

        for elec_id in grid_elec.ravel():
            if elec_id is None:
                continue

            try:
                if np.isnan(float(elec_id)):
                    continue
            except Exception:
                pass

            ch_idx = elec_to_idx.get(int(elec_id))

            if ch_idx is None:
                continue

            if ch_idx < 0 or ch_idx >= n_channels:
                continue

            valid_channel_indices.append(ch_idx)

            for trial_idx in range(n_trials):
                trace = bb_full[trial_idx, ch_idx, :]

                if trace is None:
                    continue

                trace = np.asarray(trace)

                if trace.size != n_time:
                    continue

                if not np.isfinite(trace).any():
                    continue

                region_traces.append(trace.astype(np.float64))

    else:
        # Fallback: use all channels from first available group
        if groups:
            grp_idxs = groups[0][0]

            for ch_idx in grp_idxs:
                if ch_idx is None:
                    continue

                if ch_idx < 0 or ch_idx >= n_channels:
                    continue

                valid_channel_indices.append(ch_idx)

                for trial_idx in range(n_trials):
                    trace = bb_full[trial_idx, ch_idx, :]

                    if trace is None:
                        continue

                    trace = np.asarray(trace)

                    if trace.size != n_time:
                        continue

                    if not np.isfinite(trace).any():
                        continue

                    region_traces.append(trace.astype(np.float64))

    if not region_traces:
        print(f"  [Skip] No valid traces for region {region_name}")
        return

    print(f"  Using all available data from {region_name}: {len(set(valid_channel_indices))} channels, {len(region_traces)} trial/channel pairs")

    # --- Build FOI ---
    f_min, f_max = cfg.get('freq_range', (1, 60))
    foi = get_foi_with_notch_gaps(f_min, f_max, 1)
    normalize_method = cfg.get('normalize', 'zscore')
    baseline_win = cfg.get('baseline_window', (-950, -650))
    time_range = cfg.get('time_range_ms', (-300, 500))
    methods = cfg.get('methods', [])

    if not methods:
        print("  [Skip] No methods defined in WAVELET_TEST_CONFIG['methods']")
        return

    print(f"  Computing {len(methods)} methods on {len(region_traces)} traces...")

    # --- Compute TFR for each method, then median across traces ---
    method_results = []  # list of (label, Sxx_median, f, t_ms_plot)

    for method_cfg in methods:
        mlabel = method_cfg.get('label', method_cfg.get('method', '?'))
        print(f"    [{mlabel}] ... ", end='', flush=True)

        specs = []
        f_out_saved = None
        t_ms_plot = None

        for trace in region_traces:
            try:
                Sxx, f_out, t_out = compute_tfr_method(trace, fs, method_cfg, foi=foi, t_ms=t_ms_arr)
            except Exception as e:
                print(f"\n      [Error] {e}")
                continue
            if Sxx is None:
                continue
            specs.append(Sxx)
            if f_out_saved is None:
                f_out_saved = np.asarray(f_out)
                t_ms_plot = np.asarray(t_out) * 1000.0 + float(t_ms_arr[0])

        if not specs:
            print(f"FAILED")
            method_results.append((mlabel, None, None, None))
            continue

        Sxx_median = np.nanmedian(np.stack(specs, axis=0), axis=0)  # (n_freq, n_time)

        # Normalize
        bl_mask = (t_ms_plot >= baseline_win[0]) & (t_ms_plot <= baseline_win[1])
        if normalize_method == 'zscore':
            Sxx_norm = zscore_normalize_spectrogram(Sxx_median, baseline_mask=bl_mask)
        elif normalize_method == 'dB':
            if bl_mask.any():
                bl_power = np.nanmean(Sxx_median[:, bl_mask], axis=1, keepdims=True)
                bl_power[bl_power < 1e-20] = 1e-20
                Sxx_norm = 10 * np.log10(Sxx_median / bl_power)
            else:
                Sxx_norm = Sxx_median.copy()
        else:
            Sxx_norm = Sxx_median.copy()

        print(f"done ({len(specs)}/{len(region_traces)} valid)")
        method_results.append((mlabel, Sxx_norm, f_out_saved, t_ms_plot))

    # Filter out failed methods
    valid_results = [(lbl, sxx, f, t) for lbl, sxx, f, t in method_results if sxx is not None]
    if not valid_results:
        print("  [Skip] All methods failed.")
        return

    # --- Compute mean LFP trace for reference ---
    mean_lfp = np.nanmean(np.stack(region_traces, axis=0), axis=0)  # (n_time,)
    # Trim to display time range
    t_display_mask = (t_ms_arr >= time_range[0]) & (t_ms_arr <= time_range[1])
    t_display = t_ms_arr[t_display_mask]
    mean_lfp_display = mean_lfp[t_display_mask]

    # --- Build resolution indicator: temporal resolution at each frequency ---
    # For Morlet: sigma_t = n_cycles / (2*pi*f) -> FWHM ~ 2.35*sigma_t
    # This shows HOW MUCH temporal smearing each method has
    def get_temporal_resolution_ms(method_cfg, freqs):
        method = method_cfg.get('method', 'morlet')
        freqs = np.asarray(freqs)
        freqs = freqs[freqs > 0]
        if method == 'morlet':
            nc = method_cfg.get('n_cycles', 7)
            sigma_t = nc / (2 * np.pi * freqs)
            return freqs, sigma_t * 2.35 * 1000  # FWHM in ms
        elif method == 'morlet_adaptive':
            nc_lo = method_cfg.get('n_cycles_low', 3)
            nc_hi = method_cfg.get('n_cycles_high', 7)
            f_lo_max = method_cfg.get('freq_low_max', 8)
            f_hi_min = method_cfg.get('freq_high_min', 30)
            nc_arr = np.where(
                freqs <= f_lo_max, nc_lo,
                np.where(freqs >= f_hi_min, nc_hi,
                         nc_lo + (freqs - f_lo_max) / (f_hi_min - f_lo_max) * (nc_hi - nc_lo))
            )
            sigma_t = nc_arr / (2 * np.pi * freqs)
            return freqs, sigma_t * 2.35 * 1000
        elif method == 'stft':
            win_ms = method_cfg.get('window_ms', 200)
            return freqs, np.full(len(freqs), win_ms)
        elif method == 'paul':
            m = method_cfg.get('order', 4)
            omega_0 = 2 * (m + 1)
            scale = omega_0 / (2 * np.pi * freqs)
            # Paul wavelet effective duration approx: 2*sqrt(m+0.5)/(2*pi) / freq * 1000
            return freqs, (2 * np.sqrt(m + 0.5) / (2 * np.pi * freqs)) * 1000
        return freqs, np.full(len(freqs), np.nan)

    # --- Plot ---
    n_methods = len(valid_results)
    vmin = cfg.get('vmin', -3)
    vmax = cfg.get('vmax', 3)
    cmap = cfg.get('cmap', 'jet')

    # Figure layout: 3 rows per method column
    # Row 0: mean LFP trace (shared, top)
    # Row 1: spectrogram
    # Row 2: temporal resolution curve
    fig_height = 3 + 4 * n_methods  # Fallback if stacking rows

    # Better: use a grid with shared top row
    fig = plt.figure(figsize=(4 * n_methods, 9), dpi=120)
    gs = fig.add_gridspec(
        3, n_methods,
        height_ratios=[1, 4, 1.5],
        hspace=0.45,
        wspace=0.35,
    )

    fig.suptitle(
        f"Wavelet Method Comparison\n"
        f"{session_id} | {region_name} | "
        f"{len(region_traces)} trial/ch pairs | {normalize_method} norm",
        fontsize=11, fontweight='bold', y=0.98,
    )

    for col_idx, (mlabel, Sxx_norm, f_out, t_ms_plot) in enumerate(valid_results):
        # --- Row 0: mean LFP reference ---
        ax_lfp = fig.add_subplot(gs[0, col_idx])
        ax_lfp.plot(t_display, mean_lfp_display, color='#333333', lw=0.8)
        ax_lfp.axvline(0, color='red', ls='--', lw=1, alpha=0.8)
        ax_lfp.axvspan(0, 100, color='red', alpha=0.08)
        ax_lfp.set_xlim(time_range)
        ax_lfp.set_title(mlabel, fontsize=9, fontweight='bold')
        ax_lfp.set_xlabel('')
        ax_lfp.set_xticklabels([])
        if col_idx == 0:
            ax_lfp.set_ylabel('LFP (µV)', fontsize=7)
        ax_lfp.tick_params(labelsize=7)
        ax_lfp.grid(True, alpha=0.2)

        # --- Row 1: Spectrogram ---
        ax_spec = fig.add_subplot(gs[1, col_idx])

        # Trim to display time range
        t_disp_mask = (t_ms_plot >= time_range[0]) & (t_ms_plot <= time_range[1])
        t_disp = t_ms_plot[t_disp_mask]
        Sxx_disp = Sxx_norm[:, t_disp_mask]

        # Trim frequency
        f_mask = (f_out >= f_min) & (f_out <= f_max)
        f_disp = f_out[f_mask]
        Sxx_disp = Sxx_disp[f_mask, :]

        if t_disp.size > 0 and f_disp.size > 0:
            im = ax_spec.imshow(
                Sxx_disp,
                aspect='auto',
                origin='lower',
                extent=[t_disp[0], t_disp[-1], f_disp[0], f_disp[-1]],
                vmin=vmin, vmax=vmax,
                cmap=cmap,
                interpolation='nearest',
            )
            plt.colorbar(im, ax=ax_spec, shrink=0.8, pad=0.02, label=normalize_method)

        ax_spec.axvline(0, color='white', ls='--', lw=1, alpha=0.9)
        ax_spec.set_xlim(time_range)
        ax_spec.set_ylim(f_min, f_max)
        ax_spec.set_xlabel('Time re. stim onset (ms)', fontsize=7)
        if col_idx == 0:
            ax_spec.set_ylabel('Frequency (Hz)', fontsize=7)
        ax_spec.tick_params(labelsize=7)

        # Mark notch regions
        notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])
        for nlo, nhi in notch_regions:
            if nlo < f_max and nhi > f_min:
                ax_spec.axhspan(max(nlo, f_min), min(nhi, f_max),
                                color='gray', alpha=0.3, zorder=5)

        # --- Row 2: Temporal resolution ---
        ax_res = fig.add_subplot(gs[2, col_idx])
        method_cfg_entry = methods[col_idx] if col_idx < len(methods) else {}
        res_freqs, res_ms = get_temporal_resolution_ms(method_cfg_entry, f_disp)
        if res_freqs is not None and len(res_ms) > 0:
            ax_res.plot(res_ms, res_freqs, color='navy', lw=1.5)
            ax_res.set_xlabel('Temporal resolution\n(FWHM, ms)', fontsize=7)
            if col_idx == 0:
                ax_res.set_ylabel('Freq (Hz)', fontsize=7)
            ax_res.set_ylim(f_min, f_max)
            ax_res.grid(True, alpha=0.2)
            ax_res.tick_params(labelsize=7)
            # Shade region > 100ms (larger than typical stim window) in red
            ax_res.axvline(100, color='red', ls=':', lw=1, alpha=0.6, label='100 ms')
            ax_res.legend(fontsize=6, loc='upper right')

    # --- Save ---
    out_dir = fig_dir / cfg.get('output_subdir', 'wavelet_comparison')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"wavelet_comparison__{session_id}__{region_name}__{normalize_method}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved wavelet comparison -> {out_path}")


# =============================================================================
# FIGURE 1: SESSION SUMMARY (Extended to 120 Hz)
# =============================================================================

def generate_session_summary(data, session_id, fig_dir, groups, fs=1000):
    """
    Generate comprehensive session summary with PSD from 1-120 Hz.
    """
    try:
        bb_full, t_ms = get_broadband_full(data)
    except KeyError as e:
        print(f"    [Skip] No broadband data for session summary: {e}")
        return None
    
    # Slice to analysis windows
    bb_pre, t_pre = slice_time_window(
        bb_full, t_ms,
        TIME_WINDOWS['pre_start'], TIME_WINDOWS['pre_end']
    )
    bb_post, t_post = slice_time_window(
        bb_full, t_ms,
        TIME_WINDOWS['post_start'], TIME_WINDOWS['post_end']
    )
    
    n_trials, n_ch, n_time = bb_post.shape
    n_regions = len(groups)
    
    # Create figure - single row of PSD plots
    fig, axes = plt.subplots(1, n_regions, figsize=(4 * n_regions, 4))
    if n_regions == 1:
        axes = [axes]
    
    fig.suptitle(f'Session Summary: {session_id}\n'
                 f'({n_trials} trials, {n_ch} channels, {n_regions} regions)',
                 fontsize=14, fontweight='bold')
    
    freq_range = ANALYSIS_CONFIG['freq_range_display']
    notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])
    nperseg = min(n_time, 512)
    
    # PSD for each region
    for col, (grp_idxs, grp_name) in enumerate(groups):
        ax = axes[col]
        short_name = grp_name.split(' (')[0]
        color = get_region_color(short_name)
        
        # Compute PSD
        region_data = bb_post[:, grp_idxs, :]
        f, Pxx = signal.welch(region_data, fs=fs, nperseg=nperseg, axis=2)
        
        # Average across channels, then compute stats across trials
        Pxx_ch_mean = np.nanmean(Pxx, axis=1)  # (trials, freq)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Pxx_dB = 10 * np.log10(Pxx_ch_mean + 1e-20)
        
        # Compute mean and 95% CI
        psd_mean, psd_ci_lo, psd_ci_hi = compute_confidence_interval(Pxx_dB, axis=0)
        
        # === MASK NOTCH REGIONS ===
        psd_mean = mask_notch_regions(f, psd_mean, notch_regions)
        psd_ci_lo = mask_notch_regions(f, psd_ci_lo, notch_regions)
        psd_ci_hi = mask_notch_regions(f, psd_ci_hi, notch_regions)
        
        # Mask to frequency range (1-120 Hz)
        f_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        f_plot = f[f_mask]
        
        # Plot (NaN values will create gaps automatically)
        ax.plot(f_plot, psd_mean[f_mask], color=color, lw=2, label='Post')
        ax.fill_between(f_plot,
                        psd_ci_lo[f_mask],
                        psd_ci_hi[f_mask],
                        color=color, alpha=0.2)
        
        # Add pre-stim if available
        if bb_pre is not None:
            pre_data = bb_pre[:, grp_idxs, :]
            if pre_data.shape[2] > 0:
                f_pre, Pxx_pre = signal.welch(pre_data, fs=fs, nperseg=min(pre_data.shape[2], 512), axis=2)
                Pxx_pre_ch_mean = np.nanmean(Pxx_pre, axis=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    Pxx_pre_dB = 10 * np.log10(Pxx_pre_ch_mean + 1e-20)
                
                # Compute mean and 95% CI for pre
                psd_pre_mean, psd_pre_ci_lo, psd_pre_ci_hi = compute_confidence_interval(Pxx_pre_dB, axis=0)
                
                # === MASK NOTCH REGIONS FOR PRE ===
                psd_pre_mean = mask_notch_regions(f_pre, psd_pre_mean, notch_regions)
                psd_pre_ci_lo = mask_notch_regions(f_pre, psd_pre_ci_lo, notch_regions)
                psd_pre_ci_hi = mask_notch_regions(f_pre, psd_pre_ci_hi, notch_regions)
                
                f_pre_mask = (f_pre >= freq_range[0]) & (f_pre <= freq_range[1])
                
                # Plot pre mean line
                ax.plot(f_pre[f_pre_mask], psd_pre_mean[f_pre_mask], 
                    color=color, lw=1.5, ls='--', alpha=0.7, label='Pre')
                
                # Add CI shading for pre (lighter/more transparent than post)
                ax.fill_between(f_pre[f_pre_mask],
                                psd_pre_ci_lo[f_pre_mask],
                                psd_pre_ci_hi[f_pre_mask],
                                color=color, alpha=0.1)
        
        add_band_shading(ax)
        
        # === MARK NOTCH REGIONS WITH SHADING ===
        for f_lo, f_hi in notch_regions:
            ax.axvspan(f_lo, f_hi, color='gray', alpha=0.15, zorder=0)
        
        # Add 60 Hz line marker
        ax.axvline(60, color='gray', ls=':', alpha=0.5, label='60 Hz')
        
        ax.set_title(f'{short_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (dB re: 1 µV²/Hz)' if col == 0 else '')
        ax.set_xlim(freq_range)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    
    # Save
    out_dir = fig_dir / "Session_Summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"summary_{session_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved session summary -> {out_path.name}")
    
    return out_path

    
# =============================================================================
# FIGURE 2: RAW WAVEFORM RASTER
# =============================================================================

def generate_waveform_raster(data, session_id, fig_dir, groups, fs=1000, 
                              n_trials_show=30):
    """
    Plot raw LFP waveforms for representative channels across trials.
    """
    try:
        bb_full, t_ms = get_broadband_full(data)
    except KeyError as e:
        print(f"    [Skip] No broadband data for waveform raster: {e}")
        return None
    
    n_trials, n_ch, n_time = bb_full.shape
    n_trials_plot = min(n_trials_show, n_trials)
    n_regions = len(groups)
    
    # Create figure
    fig, axes = plt.subplots(n_regions, 1, figsize=(14, 3 * n_regions),
                             sharex=True)
    if n_regions == 1:
        axes = [axes]
    
    fig.suptitle(f'Raw LFP Waveforms: {session_id}\n'
                 f'({n_trials_plot} trials, 1 representative channel per region)',
                 fontsize=12, fontweight='bold')
    
    for ax_idx, (grp_idxs, grp_name) in enumerate(groups):
        ax = axes[ax_idx]
        short_name = grp_name.split(' (')[0]
        color = get_region_color(short_name)
        
        # Pick middle channel from group
        ch_idx = grp_idxs[len(grp_idxs) // 2]
        
        # Get traces
        traces = bb_full[:n_trials_plot, ch_idx, :]
        
        # Normalize for display
        trace_std = np.nanstd(traces)
        if trace_std < 1e-10:
            trace_std = 1.0
        
        # Plot each trial with vertical offset
        for trial_idx in range(n_trials_plot):
            trace = traces[trial_idx, :]
            offset = trial_idx * 4 * trace_std
            ax.plot(t_ms, trace + offset, color=color, alpha=0.6, lw=0.5)
        
        # Mark stimulation period
        add_stim_marker(ax)
        
        ax.set_ylabel(f'{short_name}\n(µV + offset)', fontsize=10)
        ax.set_xlim(t_ms[0], t_ms[-1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add channel info
        ax.text(1.02, 0.5, f'Ch {ch_idx}', transform=ax.transAxes,
                fontsize=9, va='center', ha='left')
    
    axes[-1].set_xlabel('Time relative to stim onset (ms)', fontsize=11)
    
    plt.tight_layout()
    
    # Save
    out_dir = fig_dir / "Waveform_Rasters"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"waveforms_{session_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved waveform raster -> {out_path.name}")
    
    return out_path


# =============================================================================
# FIGURE 4: INTER-REGIONAL COHERENCE
# =============================================================================

def generate_coherence_matrix(data, session_id, fig_dir, groups, fs=1000):
    """
    Compute and plot coherence between all pairs of regions.
    """
    try:
        bb_full, t_ms = get_broadband_full(data)
    except KeyError as e:
        print(f"    [Skip] No data for coherence analysis: {e}")
        return None
    
    n_trials, n_ch, n_time = bb_full.shape
    n_regions = len(groups)
    region_names = [g[1].split(' (')[0] for g in groups]
    
    if n_regions < 2:
        print("    [Skip] Need at least 2 regions for coherence")
        return None
    
    # Compute mean trace per region per trial
    region_traces = {}
    for grp_idxs, grp_name in groups:
        short_name = grp_name.split(' (')[0]
        region_traces[short_name] = np.nanmean(bb_full[:, grp_idxs, :], axis=1)
    
    # Coherence settings
    nperseg = min(n_time, ANALYSIS_CONFIG['coherence_nperseg'])
    
    # Get frequency axis
    f, _ = signal.coherence(region_traces[region_names[0]][0],
                            region_traces[region_names[0]][0],
                            fs=fs, nperseg=nperseg)
    
    bands = ANALYSIS_CONFIG['bands']
    n_bands = len(bands)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    fig.suptitle(f'Inter-Regional Coherence: {session_id}\n'
                 f'(averaged across {n_trials} trials)',
                 fontsize=12, fontweight='bold')
    
    for band_idx, (band_name, (f_lo, f_hi)) in enumerate(bands.items()):
        if band_idx >= len(axes):
            break
        ax = axes[band_idx]
        
        # Coherence matrix for this band
        coh_matrix = np.zeros((n_regions, n_regions))
        coh_sem = np.zeros((n_regions, n_regions))
        
        f_mask = (f >= f_lo) & (f <= f_hi)
        
        for i, reg_i in enumerate(region_names):
            for j, reg_j in enumerate(region_names):
                if i == j:
                    coh_matrix[i, j] = 1.0
                    continue
                
                # Compute coherence for each trial
                coh_trials = []
                for trial in range(n_trials):
                    try:
                        _, Cxy = signal.coherence(
                            region_traces[reg_i][trial],
                            region_traces[reg_j][trial],
                            fs=fs, nperseg=nperseg
                        )
                        coh_band = np.nanmean(Cxy[f_mask])
                        if np.isfinite(coh_band):
                            coh_trials.append(coh_band)
                    except:
                        pass
                
                if coh_trials:
                    coh_matrix[i, j] = np.mean(coh_trials)
                    coh_sem[i, j] = np.std(coh_trials) / np.sqrt(len(coh_trials))
        
        # Plot heatmap
        im = ax.imshow(coh_matrix, cmap='hot', vmin=0, vmax=1, aspect='auto')
        
        # Add text annotations
        for i in range(n_regions):
            for j in range(n_regions):
                val = coh_matrix[i, j]
                text_color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=9, color=text_color, fontweight='bold')
        
        ax.set_xticks(range(n_regions))
        ax.set_xticklabels(region_names, fontsize=9)
        ax.set_yticks(range(n_regions))
        ax.set_yticklabels(region_names, fontsize=9)
        ax.set_title(f'{band_name} ({f_lo}-{f_hi} Hz)', fontsize=10, fontweight='bold')
    
    # Hide unused axes
    for idx in range(n_bands, len(axes)):
        axes[idx].axis('off')
    
    # Colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Coherence')
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    
    # Save
    out_dir = fig_dir / "Coherence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"coherence_{session_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved coherence matrix -> {out_path.name}")
    
    return out_path


# =============================================================================
# FIGURE 6: STIM VS CONTROL COMPARISON
# =============================================================================

def generate_stim_vs_control(data_stim, data_control, session_id, fig_dir, 
                              groups, fs=1000):
    """
    Compare stimulation and control conditions.
    """
    try:
        stim_full, t_stim = get_broadband_full(data_stim)
        ctrl_full, t_ctrl = get_broadband_full(data_control)
    except (KeyError, TypeError) as e:
        print(f"    [Skip] Missing data for stim vs control: {e}")
        return None
    
    # Slice to post-stim window for comparison
    bb_stim, _ = slice_time_window(
        stim_full, t_stim,
        TIME_WINDOWS['post_start'], TIME_WINDOWS['post_end']
    )
    bb_ctrl, _ = slice_time_window(
        ctrl_full, t_ctrl,
        TIME_WINDOWS['post_start'], TIME_WINDOWS['post_end']
    )
    
    n_regions = len(groups)
    bands = ANALYSIS_CONFIG['bands']
    
    # Create figure: PSD comparison and band power difference
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, n_regions, figure=fig)
    
    fig.suptitle(f'Stim vs Control: {session_id}\n'
                 f'(Stim: {bb_stim.shape[0]} trials, Control: {bb_ctrl.shape[0]} trials)',
                 fontsize=12, fontweight='bold')
    
    nperseg = min(bb_stim.shape[2], bb_ctrl.shape[2], 512)
    freq_range = ANALYSIS_CONFIG['freq_range_display']

    notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])
    
    for col, (grp_idxs, grp_name) in enumerate(groups):
        short_name = grp_name.split(' (')[0]
        
        # Validate indices
        valid_stim_idx = [i for i in grp_idxs if i < bb_stim.shape[1]]
        valid_ctrl_idx = [i for i in grp_idxs if i < bb_ctrl.shape[1]]
        
        if not valid_stim_idx or not valid_ctrl_idx:
            continue
        
        # Top row: PSD comparison
        ax_psd = fig.add_subplot(gs[0, col])
        
        # Stim PSD
        stim_data = bb_stim[:, valid_stim_idx, :]
        f, Pxx_stim = signal.welch(stim_data, fs=fs, nperseg=nperseg, axis=2)
        Pxx_stim_mean = np.nanmean(Pxx_stim, axis=(0, 1))
        
        # Control PSD
        ctrl_data = bb_ctrl[:, valid_ctrl_idx, :]
        _, Pxx_ctrl = signal.welch(ctrl_data, fs=fs, nperseg=nperseg, axis=2)
        Pxx_ctrl_mean = np.nanmean(Pxx_ctrl, axis=(0, 1))
        
        # Convert to dB
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psd_stim_dB = 10 * np.log10(Pxx_stim_mean + 1e-20)
            psd_ctrl_dB = 10 * np.log10(Pxx_ctrl_mean + 1e-20)

        psd_stim_dB = mask_notch_regions(f, psd_stim_dB, notch_regions)
        psd_ctrl_dB = mask_notch_regions(f, psd_ctrl_dB, notch_regions)
        
        f_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        
        ax_psd.plot(f[f_mask], psd_stim_dB[f_mask], 'r-', lw=2, label='Stim')
        ax_psd.plot(f[f_mask], psd_ctrl_dB[f_mask], 'b-', lw=2, label='Control')

        for f_lo, f_hi in notch_regions:
            ax_psd.axvspan(f_lo, f_hi, color='gray', alpha=0.15, zorder=0)
        
        add_band_shading(ax_psd)
        ax_psd.axvline(60, color='gray', ls=':', alpha=0.5)
        
        ax_psd.set_title(f'{short_name}', fontsize=11, fontweight='bold')
        ax_psd.set_xlabel('Frequency (Hz)')
        ax_psd.set_ylabel('Power (dB)' if col == 0 else '')
        ax_psd.set_xlim(freq_range)
        ax_psd.legend(loc='upper right', fontsize='small')
        ax_psd.grid(True, alpha=0.3)
        
        # Bottom row: Power difference by band
        ax_diff = fig.add_subplot(gs[1, col])
        
        band_names = list(bands.keys())
        x_pos = np.arange(len(band_names))
        
        diff_means = []
        diff_cis = []
        
        for band_name, (f_lo, f_hi) in bands.items():
            f_band_mask = (f >= f_lo) & (f <= f_hi)
            
            # Stim band power (per trial)
            stim_band = np.nanmean(Pxx_stim[:, :, f_band_mask], axis=(1, 2))
            ctrl_band = np.nanmean(Pxx_ctrl[:, :, f_band_mask], axis=(1, 2))
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stim_dB = 10 * np.log10(stim_band + 1e-20)
                ctrl_dB = 10 * np.log10(ctrl_band + 1e-20)
            
            # Difference (positive = stim > control)
            diff = np.mean(stim_dB) - np.mean(ctrl_dB)
            
            # Pooled SEM for difference
            sem_diff = np.sqrt(
                (np.std(stim_dB)**2 / len(stim_dB)) + 
                (np.std(ctrl_dB)**2 / len(ctrl_dB))
            ) * 1.96
            
            diff_means.append(diff)
            diff_cis.append(sem_diff)
        
        colors = ['red' if d > 0 else 'blue' for d in diff_means]
        ax_diff.bar(x_pos, diff_means, yerr=diff_cis, color=colors, alpha=0.7, 
                   capsize=4, edgecolor='black')
        ax_diff.axhline(0, color='black', ls='-', lw=1)
        
        ax_diff.set_xticks(x_pos)
        ax_diff.set_xticklabels(band_names, fontsize=8, rotation=45, ha='right')
        ax_diff.set_ylabel('Δ Power (dB)\n(Stim - Control)' if col == 0 else '')
        ax_diff.set_title('Power Difference', fontsize=10)
        ax_diff.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    out_dir = fig_dir / "Stim_vs_Control"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stim_vs_control_{session_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved stim vs control -> {out_path.name}")
    
    return out_path


# =============================================================================
# FIGURE 7: DOSE-RESPONSE ANALYSIS
# =============================================================================

def generate_dose_response(session_data_dict, base_session_id, fig_dir, groups, fs=1000):
    """
    Plot neural response as a function of stimulation dose (number of pulses).
    """
    if not session_data_dict or len(session_data_dict) < 2:
        print("    [Skip] Need multiple conditions for dose-response")
        return None
    
    pulse_counts = sorted(session_data_dict.keys())
    bands = ANALYSIS_CONFIG['bands']
    
    # Create figure
    n_bands = len(bands)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    fig.suptitle(f'Dose-Response Analysis: {base_session_id}\n'
                 f'Band power change vs number of stimulation pulses',
                 fontsize=12, fontweight='bold')
    
    # Get baseline (0 pulses / control) for reference
    baseline_powers = {}
    if 0 in session_data_dict:
        ctrl_data = session_data_dict[0].get('broadband_post')
        if ctrl_data is not None:
            nperseg = min(ctrl_data.shape[2], 256)
            f, Pxx_ctrl = signal.welch(ctrl_data, fs=fs, nperseg=nperseg, axis=2)
            
            for grp_idxs, grp_name in groups:
                short_name = grp_name.split(' (')[0]
                valid_idx = [i for i in grp_idxs if i < ctrl_data.shape[1]]
                if valid_idx:
                    baseline_powers[short_name] = np.nanmean(Pxx_ctrl[:, valid_idx, :], axis=(0, 1))
    
    for band_idx, (band_name, (f_lo, f_hi)) in enumerate(bands.items()):
        if band_idx >= len(axes):
            break
        ax = axes[band_idx]
        
        for grp_idxs, grp_name in groups:
            short_name = grp_name.split(' (')[0]
            color = get_region_color(short_name)
            
            power_means = []
            power_cis = []
            valid_pulses = []
            
            for n_pulses in pulse_counts:
                data = session_data_dict[n_pulses]
                
                try:
                    bb_full, t_ms = get_broadband_full(data)
                    bb_post, _ = slice_time_window(
                        bb_full, t_ms,
                        TIME_WINDOWS['post_start'], TIME_WINDOWS['post_end']
                    )
                except KeyError:
                    continue
                
                valid_idx = [i for i in grp_idxs if i < bb_post.shape[1]]
                if not valid_idx:
                    continue
                
                # Compute band power
                nperseg = min(bb_post.shape[2], 256)
                f, Pxx = signal.welch(bb_post[:, valid_idx, :], fs=fs, nperseg=nperseg, axis=2)
                
                f_mask = (f >= f_lo) & (f <= f_hi)
                band_power = np.nanmean(Pxx[:, :, f_mask], axis=(1, 2))  # (trials,)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    band_dB = 10 * np.log10(band_power + 1e-20)
                
                # Express relative to baseline if available
                if short_name in baseline_powers:
                    baseline_band = np.nanmean(baseline_powers[short_name][f_mask])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        baseline_dB = 10 * np.log10(baseline_band + 1e-20)
                    band_dB = band_dB - baseline_dB
                
                mean, ci_lo, ci_hi = compute_confidence_interval(band_dB, axis=0)
                
                power_means.append(mean)
                power_cis.append((ci_hi - ci_lo) / 2)
                valid_pulses.append(n_pulses)
            
            if valid_pulses:
                ax.errorbar(valid_pulses, power_means, yerr=power_cis,
                           marker='o', color=color, label=short_name,
                           capsize=3, lw=2, markersize=6)
        
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('Number of Pulses' if band_idx >= 3 else '')
        ax.set_ylabel('Δ Power (dB)' if band_idx % 3 == 0 else '')
        ax.set_title(f'{band_name} ({f_lo}-{f_hi} Hz)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if band_idx == 0:
            ax.legend(loc='best', fontsize='small')
        
        if valid_pulses:
            ax.set_xticks(pulse_counts)
    
    # Hide unused axes
    for idx in range(n_bands, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save
    out_dir = fig_dir / "Dose_Response"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dose_response_{base_session_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved dose-response -> {out_path.name}")
    
    return out_path


# =============================================================================
# FIGURE 8: SPATIAL GIF
# =============================================================================

def render_spatial_grid(ax, mean_band_dB, t_bin_idx, grid_elec, elec_to_idx, 
                        vmin, vmax, smoothing=True):
    """
    Renders a spatial grid onto an axis, optionally with smoothing.
    """
    grid = np.full((8, 8), np.nan)
    points = []
    values = []
    
    for r in range(8):
        for c in range(8):
            elec_id = int(grid_elec[r, c])
            idx = elec_to_idx.get(elec_id, None)
            if idx is not None and 0 <= idx < mean_band_dB.shape[0]:
                val = mean_band_dB[idx, t_bin_idx]
                grid[r, c] = val
                if not np.isnan(val):
                    points.append((r, c))
                    values.append(val)
    
    if smoothing and len(points) >= 4:
        # Diffusion-based smoothing
        filled_grid = np.copy(grid)
        mask = np.isnan(filled_grid)
        
        if len(values) > 0:
            filled_grid[mask] = np.nanmean(np.asarray(values))
        
        # Iterative Laplace solver
        for _ in range(10):
            prev_grid = np.copy(filled_grid)
            for r in range(8):
                for c in range(8):
                    if mask[r, c]:
                        neighbors = []
                        if r > 0: neighbors.append(prev_grid[r-1, c])
                        if r < 7: neighbors.append(prev_grid[r+1, c])
                        if c > 0: neighbors.append(prev_grid[r, c-1])
                        if c < 7: neighbors.append(prev_grid[r, c+1])
                        if neighbors:
                            filled_grid[r, c] = np.nanmean(np.asarray(neighbors))
        
        im = ax.imshow(filled_grid, cmap='jet', vmin=vmin, vmax=vmax,
                       origin='upper', aspect='equal', interpolation='bicubic')
    else:
        im = ax.imshow(grid, cmap='jet', vmin=vmin, vmax=vmax,
                       origin='upper', aspect='equal')
    
    return im


def generate_spatial_gif(band_dB, t_ms, ua_ids_1based, session_id, fig_dir, 
                         groups, nsp_to_elec, elec_to_idx, region_grids):
    """
    Generate animated GIF of the spatial heatmap over time.
    """
    gif_cfg = ANALYSIS_CONFIG
    t_start, t_end = gif_cfg.get('gif_time_range', (-500, 500))
    t_step = gif_cfg.get('gif_step_ms', 10)
    vmin = PLOT_CONFIG['heatmap_vmin']
    vmax = PLOT_CONFIG['heatmap_vmax']
    
    n_trials = band_dB.shape[0]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_band_dB = np.nanmean(band_dB, axis=0)  # (n_ch, n_time)
    
    # Define time points for GIF frames
    gif_time_pts = np.arange(t_start, t_end + t_step, t_step)
    
    heatmap_dir = fig_dir / "Spatial_GIF"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    
    band_name = PLOT_CONFIG.get('selected_band', 'Beta')
    
    for grp_idxs, grp_name in groups:
        region_name = grp_name.split(' (')[0]
        grid_elec = region_grids.get(region_name)
        
        if grid_elec is None or len(grp_idxs) == 0:
            continue
        
        safe_region = region_name.replace(' ', '_')
        
        # Generate both blocky and smoothed versions
        for mode_label, use_smoothing in [('blocky', False), ('smoothed', True)]:
            frames = []
            print(f"      Generating GIF ({mode_label}) for {region_name}...")
            
            fig, ax = plt.subplots(figsize=(6, 5))
            
            for t_target in gif_time_pts:
                t_bin_idx = int(np.argmin(np.abs(t_ms - t_target)))
                t_actual = float(t_ms[t_bin_idx])
                
                if t_actual < t_ms[0] or t_actual > t_ms[-1]:
                    continue
                
                ax.clear()
                im = render_spatial_grid(ax, mean_band_dB, t_bin_idx, grid_elec, 
                                        elec_to_idx, vmin, vmax, smoothing=use_smoothing)
                
                ax.set_title(f"{region_name} | {band_name} ({mode_label.capitalize()})\n"
                            f"T = {t_actual:.0f} ms", fontsize=12)
                ax.set_xticks(np.arange(8))
                ax.set_yticks(np.arange(8))
                ax.set_xticklabels(np.arange(1, 9), fontsize=8)
                ax.set_yticklabels(np.arange(1, 9), fontsize=8)
                ax.set_ylabel("Row", fontsize=10)
                ax.set_xlabel("Col", fontsize=10)
                
                # Add colorbar only once
                if not hasattr(fig, '_colorbar_added'):
                    fig.colorbar(im, ax=ax, label="dB")
                    fig._colorbar_added = True
                
                # Convert to image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=80)
                buf.seek(0)
                frames.append(imageio.v3.imread(buf))
            
            plt.close(fig)
            
            if frames:
                out_path = heatmap_dir / f"spatial_{band_name}_{session_id}_{safe_region}_{mode_label}.gif"
                imageio.mimsave(out_path, frames, fps=gif_cfg.get('gif_fps', 10))
                print(f"      Saved spatial GIF -> {out_path.name}")


def generate_spatial_heatmaps(band_dB, t_ms, ua_ids_1based, session_id, fig_dir,
                               groups, elec_to_idx, region_grids):
    """
    Generate static spatial heatmaps at key timepoints.
    """
    time_pts = PLOT_CONFIG.get('spatial_timepoints_ms', [-200, -50, 0, 50, 100, 200, 300, 500])
    vmin = PLOT_CONFIG['heatmap_vmin']
    vmax = PLOT_CONFIG['heatmap_vmax']
    band_name = PLOT_CONFIG.get('selected_band', 'Beta')
    
    n_trials = band_dB.shape[0]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_band_dB = np.nanmean(band_dB, axis=0)
    
    heatmap_dir = fig_dir / "Spatial_Heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    
    for grp_idxs, grp_name in groups:
        region_name = grp_name.split(' (')[0]
        grid_elec = region_grids.get(region_name)
        
        if grid_elec is None or len(grp_idxs) == 0:
            continue
        
        safe_region = region_name.replace(' ', '_')
        
        for mode_label, use_smoothing in [('blocky', False), ('smoothed', True)]:
            fig, axes = plt.subplots(1, len(time_pts), figsize=(3.5 * len(time_pts), 4))
            if len(time_pts) == 1:
                axes = [axes]
            
            fig.suptitle(f"{session_id} — {region_name} ({mode_label.capitalize()})\n"
                        f"{band_name} Spatial Heatmap — Mean of {n_trials} Trials (dB)",
                        fontsize=14)
            
            im = None
            for t_idx, t_target in enumerate(time_pts):
                ax = axes[t_idx]
                t_bin_idx = int(np.argmin(np.abs(t_ms - t_target)))
                t_actual = float(t_ms[t_bin_idx])
                
                im = render_spatial_grid(ax, mean_band_dB, t_bin_idx, grid_elec,
                                        elec_to_idx, vmin, vmax, smoothing=use_smoothing)
                
                ax.set_title(f"T = {t_actual:.0f} ms", fontsize=11)
                ax.set_xticks(np.arange(8))
                ax.set_yticks(np.arange(8))
                ax.set_xticklabels(np.arange(1, 9), fontsize=7)
                ax.set_yticklabels(np.arange(1, 9), fontsize=7)
                if t_idx == 0:
                    ax.set_ylabel("Row", fontsize=9)
                ax.set_xlabel("Col", fontsize=8)
            
            fig.subplots_adjust(right=0.92, top=0.85, bottom=0.15, left=0.05, wspace=0.3)
            if im is not None:
                cbar_ax = fig.add_axes([0.94, 0.25, 0.01, 0.5])
                fig.colorbar(im, cax=cbar_ax, label="dB")
            
            out_path = heatmap_dir / f"spatial_{band_name}_{session_id}_{safe_region}_{mode_label}.png"
            plt.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"      Saved spatial heatmap -> {out_path.name}")


# =============================================================================
# TRIAL HEATMAPS (Per-trial band power visualization)
# =============================================================================

def generate_trial_heatmaps(data, session_id, fig_dir, groups, fs=1000):
    """
    Generate per-trial band power heatmaps for each region.
    """
    try:
        bb_full, t_ms = get_broadband_full(data)
    except KeyError as e:
        print(f"    [Skip] No data for trial heatmaps: {e}")
        return None
    
    n_trials, n_ch, n_time = bb_full.shape
    
    band_name = PLOT_CONFIG.get('selected_band', 'Beta')
    bands = ANALYSIS_CONFIG['bands']
    
    if band_name not in bands:
        band_name = 'Beta'
    
    f_lo, f_hi = bands[band_name]
    baseline_win = ANALYSIS_CONFIG['baseline_window']
    
    # Filter to band
    nyq = fs / 2
    if f_hi >= nyq:
        f_hi = nyq - 1
    
    try:
        b, a = signal.butter(4, [f_lo/nyq, f_hi/nyq], btype='band')
    except:
        print(f"    [Skip] Could not create filter for {band_name}")
        return None
    
    vmin = PLOT_CONFIG['heatmap_vmin']
    vmax = PLOT_CONFIG['heatmap_vmax']
    
    n_regions = len(groups)
    fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 8))
    if n_regions == 1:
        axes = [axes]
    
    fig.suptitle(f'{band_name} Band Power by Trial: {session_id}\n'
                 f'({f_lo}-{f_hi} Hz, trials sorted by mean post-stim power)',
                 fontsize=12, fontweight='bold')
    
    bl_mask = (t_ms >= baseline_win[0]) & (t_ms <= baseline_win[1])
    
    for col, (grp_idxs, grp_name) in enumerate(groups):
        ax = axes[col]
        short_name = grp_name.split(' (')[0]
        
        # Average across channels
        region_data = np.nanmean(bb_full[:, grp_idxs, :], axis=1)
        
        # Filter and compute envelope
        filtered = signal.filtfilt(b, a, region_data, axis=1)
        envelope = np.abs(signal.hilbert(filtered, axis=1))
        
        # Convert to dB
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            baseline_power = np.nanmean(envelope[:, bl_mask], axis=1, keepdims=True)
            baseline_power[baseline_power < 1e-10] = 1e-10
            envelope_dB = 10 * np.log10(envelope / baseline_power)
        
        # Sort trials by mean post-stim power
        post_mask = (t_ms >= 100) & (t_ms <= 500)
        mean_post_power = np.nanmean(envelope_dB[:, post_mask], axis=1)
        sort_idx = np.argsort(mean_post_power)[::-1]
        
        envelope_sorted = envelope_dB[sort_idx, :]
        
        # Plot
        im = ax.imshow(envelope_sorted, aspect='auto', cmap='jet',
                      vmin=vmin, vmax=vmax,
                      extent=[t_ms[0], t_ms[-1], n_trials, 0])
        
        ax.axvline(0, color='white', ls='--', lw=1)
        ax.axvline(100, color='white', ls=':', lw=0.5)
        
        ax.set_title(f'{short_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Trial (sorted)' if col == 0 else '')
        
        if col == n_regions - 1:
            cbar = fig.colorbar(im, ax=ax, label='Power (dB)')
    
    plt.tight_layout()
    
    # Save
    out_dir = fig_dir / "Trial_Heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trial_heatmap_{band_name}_{session_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved trial heatmaps -> {out_path.name}")
    
    return out_path


# =============================================================================
# PER-UA SPECTROGRAM FUNCTION (using Morlet wavelets)
# =============================================================================

def generate_per_ua_spectrograms(data, session_id, fig_dir, groups, fs=1000,
                                  ua_ids_1based=None, nsp_to_elec=None, 
                                  region_grids=None, elec_to_idx=None,
                                  trial_index=0, generate_all_trials=False,
                                  generate_trial_average=False):
    """
    Generate 8x8 spectrogram grids for each Utah Array region.
    Frequencies in notch regions are skipped entirely (broken y-axis).
    
    NOW USES:
    - Z-score normalization per channel
    - Median averaging across trials (instead of mean)
    """
    try:
        bb_full, t_ms = get_broadband_full(data)
    except KeyError as e:
        print(f"    [Skip] No broadband data for per-UA spectrograms: {e}")
        return None
    
    if region_grids is None or elec_to_idx is None:
        print("    [Skip] Need electrode mapping for per-UA spectrograms")
        return None
    
    n_trials, n_ch, n_time = bb_full.shape

    # Spectrogram settings
    spec_cfg = PLOT_CONFIG.get('spectrogram', {})
    freq_min = spec_cfg.get('freq_min', 1)
    freq_max = spec_cfg.get('freq_max', 120)
    freq_step = spec_cfg.get('freq_step', 1)
    
    # Get frequencies with notch gaps
    foi = get_foi_with_notch_gaps(freq_min, freq_max, freq_step)
    
    cmap = spec_cfg.get('cmap', 'viridis')
    normalize_method = spec_cfg.get('normalize', 'zscore')  # NEW: default to zscore
    n_cycles = spec_cfg.get('wavelet_cycles', 5)
    
    # Baseline window for z-score normalization
    baseline_win = ANALYSIS_CONFIG['baseline_window']
    
    # Output directory
    out_dir = fig_dir / "per_UA_activity"
    
    # Extract condition name
    br_match = re.search(r'_(\d{3})(?:_|$|\.)', session_id)
    target_match = re.search(r'target_([AB])$', session_id, re.IGNORECASE)
    target_suffix = f"_{target_match.group(1).upper()}" if target_match else ""

    if br_match:
        br_idx = int(br_match.group(1))
        condition_name = f"condition_{br_idx:02d}{target_suffix}"
    else:
        br_idx = data.get('br_idx', None)
        if br_idx is not None:
            if isinstance(br_idx, np.ndarray):
                br_idx = int(br_idx.item())
            condition_name = f"condition_{br_idx:02d}{target_suffix}"
        else:
            condition_name = session_id

    condition_dir = out_dir / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which trials to process
    trials_to_process = []
    
    if generate_trial_average:
        trials_to_process.append(('average', None))  # Special marker for averaging
    
    if generate_all_trials:
        trials_to_process.extend([('single', t) for t in range(n_trials)])
    elif not generate_trial_average:  # Only add default trial if not doing average-only
        if trial_index >= n_trials:
            print(f"    [Warn] Requested trial {trial_index} but only {n_trials} trials exist. Using trial 0.")
            trial_index = 0
        trials_to_process.append(('single', trial_index))
    
    # If both average and single trials requested, add default trial
    if generate_trial_average and not generate_all_trials:
        if trial_index >= n_trials:
            trial_index = 0
        trials_to_process.append(('single', trial_index))
    
    for grp_idxs, grp_name in groups:
        region_name = grp_name.split(' (')[0]
        grid_elec = region_grids.get(region_name)
        
        if grid_elec is None:
            print(f"    [Skip] No grid for region {region_name}")
            continue
        
        safe_region = region_name.replace(' ', '_')
        
        for trial_type, trial_idx in trials_to_process:
            
            if trial_type == 'average':
                # === TRIAL-AVERAGED SPECTROGRAM (using MEDIAN) ===
                print(f"      Computing trial-averaged (median) spectrogram for {region_name}...")
                
                # First pass: compute spectrograms for all trials, then MEDIAN average
                all_spectrograms = {}
                
                for row in range(8):
                    for col in range(8):
                        elec_id = int(grid_elec[row, col])
                        data_idx = elec_to_idx.get(elec_id, None)
                        
                        if data_idx is None or data_idx >= n_ch:
                            continue
                        
                        # Collect spectrograms across all trials for this electrode
                        trial_spectrograms = []
                        t_spec_ms_saved = None
                        f_out_saved = None
                        
                        for t_idx in range(n_trials):
                            trace = bb_full[t_idx, data_idx, :]
                            
                            if np.all(np.isnan(trace)) or np.std(trace) < 1e-10:
                                continue
                            
                            # Compute spectrogram (raw power, no normalization yet)
                            Sxx, f_out, t_out = compute_wavelet_spectrogram(
                                trace, fs, foi=foi,
                                baseline_window=None,  # Don't normalize yet
                                t_ms=t_ms,
                                n_cycles=n_cycles,
                                normalize=False
                            )
                            
                            if Sxx is not None:
                                trial_spectrograms.append(Sxx)
                                if t_spec_ms_saved is None:
                                    t_spec_ms_saved = t_out * 1000 + t_ms[0]
                                    f_out_saved = f_out
                        
                        # MEDIAN across trials (instead of mean)
                        if trial_spectrograms:
                            stacked = np.stack(trial_spectrograms, axis=0)  # (n_trials, n_freq, n_time)
                            
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                Sxx_median = np.nanmedian(stacked, axis=0)  # CHANGED: median
                            
                            # Z-score normalize the median spectrogram
                            bl_mask = (t_spec_ms_saved >= baseline_win[0]) & (t_spec_ms_saved <= baseline_win[1])
                            Sxx_normalized = zscore_normalize_spectrogram(Sxx_median, baseline_mask=bl_mask)
                            
                            all_spectrograms[(row, col)] = {
                                'Sxx': Sxx_normalized,
                                'f': f_out_saved,
                                't_ms': t_spec_ms_saved,
                                'elec_id': elec_id,
                                'n_trials_averaged': len(trial_spectrograms),
                            }
                
                if not all_spectrograms:
                    print(f"    [Skip] No valid spectrograms for {region_name} (trial average)")
                    continue
                
                # Create figure for trial average
                _plot_spectrogram_grid(
                    all_spectrograms, grid_elec, elec_to_idx, n_ch,
                    session_id, region_name, safe_region, condition_dir, fig_dir,
                    spec_cfg, normalize_method, n_trials,
                    trial_label=f"Trial Median (n={n_trials})",
                    filename_suffix="trial_median"
                )
                
            else:
                # === SINGLE TRIAL SPECTROGRAM ===
                trial_data = bb_full[trial_idx, :, :]
                
                # First pass: compute all spectrograms
                all_spectrograms = {}
                
                for row in range(8):
                    for col in range(8):
                        elec_id = int(grid_elec[row, col])
                        data_idx = elec_to_idx.get(elec_id, None)
                        
                        if data_idx is None or data_idx >= n_ch:
                            continue
                        
                        trace = trial_data[data_idx, :]
                        
                        if np.all(np.isnan(trace)) or np.std(trace) < 1e-10:
                            continue
                        
                        # Compute spectrogram (raw power)
                        Sxx, f_out, t_out = compute_wavelet_spectrogram(
                            trace, fs, foi=foi,
                            baseline_window=None,
                            t_ms=t_ms,
                            n_cycles=n_cycles,
                            normalize=False
                        )
                        
                        if Sxx is not None:
                            t_spec_ms = t_out * 1000 + t_ms[0]
                            
                            # Z-score normalize single trial
                            bl_mask = (t_spec_ms >= baseline_win[0]) & (t_spec_ms <= baseline_win[1])
                            Sxx_normalized = zscore_normalize_spectrogram(Sxx, baseline_mask=bl_mask)
                            
                            all_spectrograms[(row, col)] = {
                                'Sxx': Sxx_normalized,
                                'f': f_out,
                                't_ms': t_spec_ms,
                                'elec_id': elec_id
                            }
                
                if not all_spectrograms:
                    print(f"    [Skip] No valid spectrograms for {region_name} trial {trial_idx}")
                    continue
                
                # Create figure for single trial
                _plot_spectrogram_grid(
                    all_spectrograms, grid_elec, elec_to_idx, n_ch,
                    session_id, region_name, safe_region, condition_dir, fig_dir,
                    spec_cfg, normalize_method, n_trials,
                    trial_label=f"Trial {trial_idx}",
                    filename_suffix=f"trial_{trial_idx}"
                )
    
    return out_dir
    
def _plot_spectrogram_grid(all_spectrograms, grid_elec, elec_to_idx, n_ch,
                           session_id, region_name, safe_region, condition_dir, fig_dir,
                           spec_cfg, normalize_method, n_trials,
                           trial_label="", filename_suffix=""):
    """
    Helper function to plot 8x8 spectrogram grid.
    
    Updated to handle z-score normalized data with appropriate color scaling.
    """
    cmap = spec_cfg.get('cmap', 'viridis')
    
    # Determine color scale based on normalization method
    if normalize_method == 'zscore':
        vmin = spec_cfg.get('vmin_zscore', -3)
        vmax = spec_cfg.get('vmax_zscore', 3)
        cbar_label = 'Normalized Power (a.u.)'  # CHANGED
        cmap = 'jet'  # Diverging colormap better for z-scores
    elif normalize_method == 'dB':
        vmin = spec_cfg.get('vmin_db', -6)
        vmax = spec_cfg.get('vmax_db', 6)
        cbar_label = 'Power (dB re: baseline)'
        cmap = 'jet'
    else:  # raw
        all_power = np.concatenate([s['Sxx'].flatten() for s in all_spectrograms.values()])
        vmin = spec_cfg.get('vmin_uV2', 0)
        vmax = spec_cfg.get('vmax_uV2', None)
        if vmax is None:
            vmax = np.nanpercentile(all_power, 95)
        cbar_label = 'Power (µV²)'
    
    # Create 8x8 figure
    fig, axes = plt.subplots(8, 8, figsize=(24, 24))
    
    # Build subtitle showing frequency gaps
    notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])
    notch_str = ', '.join([f'{lo}-{hi} Hz' for lo, hi in notch_regions])
    
    # Update title to reflect normalization method
    norm_str = "Z-scored" if normalize_method == 'zscore' else normalize_method.upper()

    tfr_method = spec_cfg.get('tfr_method', 'morlet').lower()
    if tfr_method == 'morlet_adaptive':
        tfr_label = (
            f"Adaptive Morlet "
            f"({spec_cfg.get('adaptive_cycles_low', 3)}-"
            f"{spec_cfg.get('adaptive_cycles_high', 7)} cycles, "
            f"{spec_cfg.get('adaptive_freq_low_max', 15)}-"
            f"{spec_cfg.get('adaptive_freq_high_min', 30)} Hz transition)"
        )
    else:
        tfr_label = f"Morlet Wavelets ({spec_cfg.get('wavelet_cycles', 5)} cycles)"
    
    fig.suptitle(f'{session_id} — {region_name}\n'
            f'Time-Frequency Power ({trial_label}) | {tfr_label} | {norm_str}\n'
            f'(Notch regions skipped: {notch_str})',
            fontsize=16, fontweight='bold')
    
    im_for_cbar = None
    
    for row in range(8):
        for col in range(8):
            ax = axes[row, col]
            elec_id = int(grid_elec[row, col])
            
            if (row, col) in all_spectrograms:
                spec_data = all_spectrograms[(row, col)]
                Sxx = spec_data['Sxx']
                f_spec = spec_data['f']
                t_spec_ms = spec_data['t_ms']
                
                im = ax.pcolormesh(t_spec_ms, np.arange(len(f_spec)), Sxx,
                                  cmap=cmap, vmin=vmin, vmax=vmax,
                                  shading='auto', rasterized=True)
                
                # Set y-ticks to show actual frequencies
                freq_step = spec_cfg.get('freq_step', 1)
                tick_freqs = [1, 10, 20, 30, 40, 55, 65, 80, 100, 115]
                tick_positions = []
                tick_labels = []
                for tf in tick_freqs:
                    if tf in f_spec:
                        idx = np.where(f_spec == tf)[0][0]
                        tick_positions.append(idx)
                        tick_labels.append(str(tf))
                    elif any(abs(f_spec - tf) < freq_step):
                        idx = np.argmin(abs(f_spec - tf))
                        tick_positions.append(idx)
                        tick_labels.append(str(int(f_spec[idx])))
                
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(tick_labels)
                
                # Mark stim onset
                ax.axvline(0, color='white', ls='--', lw=0.8, alpha=0.8)
                
                # Mark where notch gaps are with lines
                for notch_lo, notch_hi in notch_regions:
                    below_idx = np.where(f_spec < notch_lo)[0]
                    above_idx = np.where(f_spec > notch_hi)[0]
                    if len(below_idx) > 0 and len(above_idx) > 0:
                        gap_y = below_idx[-1] + 0.5
                        ax.axhline(gap_y, color='white', ls='-', lw=1.5, alpha=0.8)
                
                im_for_cbar = im
                
                # Title with trial count if averaged
                title_str = f'E{elec_id}'
                if 'n_trials_averaged' in spec_data:
                    title_str += f' (n={spec_data["n_trials_averaged"]})'
                ax.set_title(title_str, fontsize=8, pad=2)
                
            else:
                ax.set_facecolor('#e0e0e0')
                data_idx = elec_to_idx.get(elec_id, None)
                if data_idx is None:
                    label = 'N/A'
                    color = 'gray'
                else:
                    label = 'Bad'
                    color = 'red'
                ax.text(0.5, 0.5, f'E{elec_id}\n{label}', 
                       ha='center', va='center', fontsize=8,
                       transform=ax.transAxes, color=color)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Axis labels only on edges
            if row == 7:
                ax.set_xlabel('Time (ms)', fontsize=7)
            else:
                ax.set_xticklabels([])
            
            if col == 0:
                ax.set_ylabel('Freq (Hz)', fontsize=7)
            else:
                ax.set_yticklabels([])
            
            ax.tick_params(axis='both', labelsize=6)
    
    # Add colorbar
    if im_for_cbar is not None:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im_for_cbar, cax=cbar_ax)
        cbar.set_label(cbar_label, fontsize=12)  # Now shows "Normalized Power (a.u.)"
        cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout(rect=[0, 0, 0.92, 0.94])
    
    # Save
    out_path = condition_dir / f"{safe_region}_{filename_suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"    Saved -> {out_path.relative_to(fig_dir)}")

def generate_per_ua_array_median_spectrogram(data, session_id, fig_dir, groups, fs=1000,
                                             ua_ids_1based=None, nsp_to_elec=None,
                                             region_grids=None, elec_to_idx=None,
                                             control_median_baseline_norm=None):
    """
    Generate one 4-panel figure per session.

    Each subplot is one Utah array/region.
    For each region, this computes the median spectrogram across:
        1. all trials
        2. all valid channels/electrodes in that region

    Output shape per region:
        (n_freq, n_time)

    This is different from generate_per_ua_trial_average(), which gives one
    median-across-trials spectrogram per electrode in an 8x8 grid.
    """

    try:
        bb_full, t_ms = get_broadband_full(data)
    except KeyError as e:
        print(f"    [Skip] No broadband data for per-UA array median spectrogram: {e}")
        return None

    if region_grids is None or elec_to_idx is None:
        print("    [Skip] Need electrode mapping for per-UA array median spectrogram")
        return None

    # Spectrogram settings
    spec_cfg = PLOT_CONFIG.get('spectrogram', {})
    freq_step = spec_cfg.get('freq_step', 1)

    normalize_method = spec_cfg.get('normalize', 'zscore')
    baseline_win = ANALYSIS_CONFIG['baseline_window']

    # Output directory
    out_dir = fig_dir / "per_UA_activity"

    # Same condition naming logic as generate_per_ua_spectrograms()
    br_match = re.search(r'_(\d{3})(?:_|$|\.)', session_id)
    target_match = re.search(r'target_([AB])$', session_id, re.IGNORECASE)
    target_suffix = f"_{target_match.group(1).upper()}" if target_match else ""

    if br_match:
        br_idx = int(br_match.group(1))
        condition_name = f"condition_{br_idx:02d}{target_suffix}"
    else:
        br_idx = data.get('br_idx', None)
        if br_idx is not None:
            if isinstance(br_idx, np.ndarray):
                br_idx = int(br_idx.item())
            condition_name = f"condition_{br_idx:02d}{target_suffix}"
        else:
            condition_name = session_id

    condition_dir = out_dir / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    # Determine color scale
    cmap = spec_cfg.get('cmap', 'viridis')

    if normalize_method == 'zscore':
        vmin = spec_cfg.get('vmin_zscore', -3)
        vmax = spec_cfg.get('vmax_zscore', 3)
        cbar_label = 'Normalized Power (a.u.)'
        cmap = 'jet'
    elif normalize_method == 'dB':
        vmin = spec_cfg.get('vmin_db', -6)
        vmax = spec_cfg.get('vmax_db', 6)
        cbar_label = 'Power (dB re: baseline)'
        cmap = 'jet'
    else:
        vmin = spec_cfg.get('vmin_uV2', 0)
        vmax = spec_cfg.get('vmax_uV2', None)
        cbar_label = 'Power (µV²)'

    print("      Computing array-level median spectrograms across trials and channels...")

    region_specs_raw = compute_array_median_raw_spectrograms(
        data=data,
        groups=groups,
        fs=fs,
        region_grids=region_grids,
        elec_to_idx=elec_to_idx,
    )

    if not region_specs_raw:
        print("    [Skip] No valid raw array-level median spectrograms were computed")
        return None

    region_specs = {}

    for region_name, raw_spec in region_specs_raw.items():
        Sxx_region_median_raw = raw_spec['Sxx']
        t_spec_ms_saved = raw_spec['t_ms']

        Sxx_region_median = Sxx_region_median_raw.copy()

        # Normalize AFTER taking the median
        if normalize_method == 'zscore':
            bl_mask = (
                (t_spec_ms_saved >= baseline_win[0]) &
                (t_spec_ms_saved <= baseline_win[1])
            )

            Sxx_region_median = zscore_normalize_spectrogram(
                Sxx_region_median,
                baseline_mask=bl_mask,
            )

        elif normalize_method == 'dB':
            bl_mask = (
                (t_spec_ms_saved >= baseline_win[0]) &
                (t_spec_ms_saved <= baseline_win[1])
            )

            if bl_mask.any():
                baseline_power = np.nanmean(
                    Sxx_region_median[:, bl_mask],
                    axis=1,
                    keepdims=True,
                )
                baseline_power[baseline_power < 1e-20] = 1e-20

                Sxx_region_median = 10 * np.log10(
                    Sxx_region_median / baseline_power
                )

        region_specs[region_name] = {
            **raw_spec,
            'Sxx': Sxx_region_median,
        }

    if not region_specs:
        print("    [Skip] No valid array-level median spectrograms were computed")
        return None

    # If raw mode and vmax is None, compute global 95th percentile
    if normalize_method == 'raw' and vmax is None:
        all_power = np.concatenate([
            spec['Sxx'].ravel()
            for spec in region_specs.values()
            if spec['Sxx'] is not None
        ])
        vmax = np.nanpercentile(all_power, 95)

    # -------------------------------------------------------------------------
    # Plot 4-panel figure
    # -------------------------------------------------------------------------
    n_regions = len(region_specs)

    # Usually 4 arrays: SMA, PMd, M1i, M1s
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])
    notch_str = ', '.join([f'{lo}-{hi} Hz' for lo, hi in notch_regions])

    norm_str = "Z-scored" if normalize_method == 'zscore' else normalize_method.upper()

    tfr_method = spec_cfg.get('tfr_method', 'morlet').lower()

    if tfr_method == 'morlet_adaptive':
        adaptive_cycles_low = spec_cfg.get('adaptive_cycles_low', 3)
        adaptive_cycles_high = spec_cfg.get('adaptive_cycles_high', 7)
        adaptive_freq_low_max = spec_cfg.get('adaptive_freq_low_max', 15)
        adaptive_freq_high_min = spec_cfg.get('adaptive_freq_high_min', 30)

        if adaptive_freq_high_min <= adaptive_freq_low_max:
            adaptive_freq_high_min = adaptive_freq_low_max + 15

        tfr_label = (
            f"Adaptive Morlet "
            f"({adaptive_cycles_low}-{adaptive_cycles_high} cycles, "
            f"{adaptive_freq_low_max}-{adaptive_freq_high_min} Hz transition)"
        )
    else:
        tfr_label = f"Morlet Wavelets ({spec_cfg.get('wavelet_cycles', 5)} cycles)"

    fig.suptitle(
        f'{session_id}\n'
        f'Array-Level Median Spectrograms | Median Across All Trials and Channels\n'
        f'{tfr_label} | {norm_str} | Notch regions skipped: {notch_str}',
        fontsize=14,
        fontweight='bold'
    )

    im_for_cbar = None

    # Keep plotting order consistent with groups
    plot_idx = 0

    for grp_idxs, grp_name in groups:
        region_name = grp_name.split(' (')[0]

        if region_name not in region_specs:
            continue

        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]
        spec = region_specs[region_name]

        Sxx = spec['Sxx']
        f_spec = spec['f']
        t_spec_ms = spec['t_ms']

        im = ax.pcolormesh(
            t_spec_ms,
            np.arange(len(f_spec)),
            Sxx,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading='auto',
            rasterized=True
        )

        im_for_cbar = im

        # Mark stim onset
        ax.axvline(0, color='white', ls='--', lw=1.2, alpha=0.9)

        # Optional: mark end of stimulation if you consider 100 ms stimulation
        ax.axvline(100, color='white', ls=':', lw=1.0, alpha=0.7)

        # Mark notch gaps
        for notch_lo, notch_hi in notch_regions:
            below_idx = np.where(f_spec < notch_lo)[0]
            above_idx = np.where(f_spec > notch_hi)[0]

            if len(below_idx) > 0 and len(above_idx) > 0:
                gap_y = below_idx[-1] + 0.5
                ax.axhline(gap_y, color='white', ls='-', lw=1.5, alpha=0.8)

        # Frequency tick labels
        tick_freqs = [1, 10, 20, 30, 40, 55, 65, 80, 100, 115]
        tick_positions = []
        tick_labels = []

        for tf in tick_freqs:
            if tf in f_spec:
                idx = np.where(f_spec == tf)[0][0]
                tick_positions.append(idx)
                tick_labels.append(str(tf))
            else:
                idx = np.argmin(np.abs(f_spec - tf))
                if np.abs(f_spec[idx] - tf) <= freq_step:
                    tick_positions.append(idx)
                    tick_labels.append(str(int(f_spec[idx])))

        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

        ax.set_title(
            f'{region_name}\n'
            f'{spec["n_valid_channels"]} channels, '
            f'{spec["n_valid_trial_channel_pairs"]} trial-channel spectra',
            fontsize=11,
            fontweight='bold'
        )

        ax.set_xlabel('Time relative to event onset (ms)')
        ax.set_ylabel('Frequency (Hz)')

        ax.tick_params(axis='both', labelsize=9)

        plot_idx += 1

    # Hide unused axes if fewer than 4 regions
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    # Colorbar
    if im_for_cbar is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.18, 0.02, 0.65])
        cbar = fig.colorbar(im_for_cbar, cax=cbar_ax)
        cbar.set_label(cbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0, 0.88, 0.92])

    out_path = condition_dir / "array_median_across_trials_and_channels.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"    Saved array-level median spectrogram -> {out_path.relative_to(fig_dir)}")

    if control_median_baseline_norm is not None:
        out_path_ctrl = condition_dir / "array_median_across_trials_and_channels_CONTROL_MEDIAN_BASELINE_NORM.png"

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        fig.suptitle(
            f'{session_id}\n'
            f'Array-Level Median Spectrograms | Control Median-Baseline Normalized\n'
            f'{tfr_label} | Notch regions skipped: {notch_str}',
            fontsize=14,
            fontweight='bold'
        )

        im_for_cbar = None
        plot_idx = 0

        for grp_idxs, grp_name in groups:
            region_name = grp_name.split(' (')[0]

            if region_name not in region_specs_raw:
                continue
            if region_name not in control_median_baseline_norm:
                continue
            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]
            spec = region_specs_raw[region_name]

            Sxx_raw = spec['Sxx']
            mean_val = control_median_baseline_norm[region_name]['mean']
            std_val = control_median_baseline_norm[region_name]['std']

            Sxx_ctrl = (Sxx_raw - mean_val) / std_val

            f_spec = spec['f']
            t_spec_ms = spec['t_ms']

            im = ax.pcolormesh(
                t_spec_ms,
                np.arange(len(f_spec)),
                Sxx_ctrl,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading='auto',
                rasterized=True
            )

            im_for_cbar = im

            ax.axvline(0, color='white', ls='--', lw=1.2, alpha=0.9)
            ax.axvline(100, color='white', ls=':', lw=1.0, alpha=0.7)

            for notch_lo, notch_hi in notch_regions:
                below_idx = np.where(f_spec < notch_lo)[0]
                above_idx = np.where(f_spec > notch_hi)[0]
                if len(below_idx) > 0 and len(above_idx) > 0:
                    gap_y = below_idx[-1] + 0.5
                    ax.axhline(gap_y, color='white', ls='-', lw=1.5, alpha=0.8)

            tick_freqs = [1, 10, 20, 30, 40, 55, 65, 80, 100, 115]
            tick_positions = []
            tick_labels = []

            for tf in tick_freqs:
                if tf in f_spec:
                    idx = np.where(f_spec == tf)[0][0]
                    tick_positions.append(idx)
                    tick_labels.append(str(tf))
                else:
                    idx = np.argmin(np.abs(f_spec - tf))
                    if np.abs(f_spec[idx] - tf) <= freq_step:
                        tick_positions.append(idx)
                        tick_labels.append(str(int(f_spec[idx])))

            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)

            ax.set_title(
                f'{region_name}\n'
                f'{spec["n_valid_channels"]} channels, '
                f'{spec["n_valid_trial_channel_pairs"]} trial-channel spectra',
                fontsize=11,
                fontweight='bold'
            )

            ax.set_xlabel('Time relative to event onset (ms)')
            ax.set_ylabel('Frequency (Hz)')
            ax.tick_params(axis='both', labelsize=9)

            plot_idx += 1

        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')

        if im_for_cbar is not None:
            fig.subplots_adjust(right=0.88)
            cbar_ax = fig.add_axes([0.90, 0.18, 0.02, 0.65])
            cbar = fig.colorbar(im_for_cbar, cax=cbar_ax)
            cbar.set_label('Power vs control median baseline (z)', fontsize=12)
            cbar.ax.tick_params(labelsize=10)

        plt.tight_layout(rect=[0, 0, 0.88, 0.92])
        plt.savefig(out_path_ctrl, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"    Saved control median-baseline normalized array-level median spectrogram -> {out_path_ctrl.relative_to(fig_dir)}")

    return out_path


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_session(data, session_id, fig_dir, groups, fs=1000,
                    data_control=None, session_data_dict=None,
                    nsp_to_elec=None, elec_to_idx=None, region_grids=None,
                    ua_ids_1based=None, control_median_baseline_norm=None,
                    control_spatial_baseline=None):
    """
    Run all enabled analyses for a session.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {session_id}")
    print(f"{'='*60}")
    
    figure_paths = []
    
    # 1. Session Summary (1-120 Hz PSD)
    if FIGURE_CONFIG.get('generate_session_summary', True):
        print("  Generating session summary...")
        path = generate_session_summary(data, session_id, fig_dir, groups, fs)
        if path:
            figure_paths.append(path)
    
    # 2. Raw Waveform Raster
    if FIGURE_CONFIG.get('generate_waveform_raster', True):
        print("  Generating waveform raster...")
        path = generate_waveform_raster(data, session_id, fig_dir, groups, fs)
        if path:
            figure_paths.append(path)
    
    # 4. Trial Heatmaps
    if FIGURE_CONFIG.get('generate_trial_heatmaps', True):
        print("  Generating trial heatmaps...")
        path = generate_trial_heatmaps(data, session_id, fig_dir, groups, fs)
        if path:
            figure_paths.append(path)
    
    # 4b. Per-UA Spectrograms (8x8 grid per region)
    if FIGURE_CONFIG.get('generate_per_ua_spectrograms', True):
        if elec_to_idx is not None and region_grids is not None:
            print("  Generating per-UA spectrogram grids (Morlet wavelets)...")
            default_trial = FIGURE_CONFIG.get('per_ua_default_trial', 0)
            generate_all = FIGURE_CONFIG.get('generate_all_trial_spectrograms', False)
            generate_avg = FIGURE_CONFIG.get('generate_per_ua_trial_average', False)  # <-- ADD THIS
            path = generate_per_ua_spectrograms(
                data, session_id, fig_dir, groups, fs,
                ua_ids_1based=ua_ids_1based,
                nsp_to_elec=nsp_to_elec,
                region_grids=region_grids,
                elec_to_idx=elec_to_idx,
                trial_index=default_trial,
                generate_all_trials=generate_all,
                generate_trial_average=generate_avg,  # <-- ADD THIS
            )
            if path:
                figure_paths.append(path)
        else:
            print("  [Skip] Per-UA spectrograms require electrode mapping")


    # 4c. Per-UA array-level median spectrogram
    # Median across all trials and all channels/electrodes within each Utah array.
    if (FIGURE_CONFIG.get('generate_per_ua_array_median', False) or
            FIGURE_CONFIG.get('generate_control_median_baseline_norm_array_median', False)):
        if elec_to_idx is not None and region_grids is not None:
            print("  Generating per-UA array-level median spectrograms...")
            path = generate_per_ua_array_median_spectrogram(
                data=data,
                session_id=session_id,
                fig_dir=fig_dir,
                groups=groups,
                fs=fs,
                ua_ids_1based=ua_ids_1based,
                nsp_to_elec=nsp_to_elec,
                region_grids=region_grids,
                elec_to_idx=elec_to_idx,
                control_median_baseline_norm=control_median_baseline_norm,
            )
            if path:
                figure_paths.append(path)
        else:
            print("  [Skip] Per-UA array-level median spectrogram requires electrode mapping")
    
    
    # 5. Coherence Matrix
    if FIGURE_CONFIG.get('generate_coherence_matrix', True):
        print("  Generating coherence matrix...")
        path = generate_coherence_matrix(data, session_id, fig_dir, groups, fs)
        if path:
            figure_paths.append(path)
    
    # 7. Stim vs Control
    if FIGURE_CONFIG.get('generate_stim_vs_control', True) and data_control is not None:
        print("  Generating stim vs control comparison...")
        path = generate_stim_vs_control(data, data_control, session_id, fig_dir, groups, fs)
        if path:
            figure_paths.append(path)
    
    # 8. Dose Response
    if FIGURE_CONFIG.get('generate_dose_response', True) and session_data_dict is not None:
        print("  Generating dose-response analysis...")
        path = generate_dose_response(session_data_dict, session_id, fig_dir, groups, fs)
        if path:
            figure_paths.append(path)
    
    # 9. Spatial visualizations (require electrode mapping)
    if (FIGURE_CONFIG.get('generate_spatial_heatmaps', True) or 
        FIGURE_CONFIG.get('generate_spatial_gif', True)):
        
        if elec_to_idx is not None and region_grids is not None:
            # Need to compute band power first
            try:
                bb_full, t_ms = get_broadband_full(data)
            except KeyError as e:
                print(f"    [Skip] No data for spatial visualizations: {e}")
                bb_full = None
            
            if bb_full is not None:          
                # Compute band power
                band_name = PLOT_CONFIG.get('selected_band', 'Beta')
                bands = ANALYSIS_CONFIG['bands']
                f_lo, f_hi = bands.get(band_name, (12, 25))
                baseline_win = ANALYSIS_CONFIG['baseline_window']
                
                nyq = fs / 2
                if f_hi >= nyq:
                    f_hi = nyq - 1
                
                try:
                    b, a = signal.butter(4, [f_lo/nyq, f_hi/nyq], btype='band')
                    
                    # Filter and compute envelope for all channels
                    filtered = signal.filtfilt(b, a, bb_full, axis=2)
                    envelope = np.abs(signal.hilbert(filtered, axis=2))
                    
                    # Baseline normalize
                    bl_mask = (t_ms >= baseline_win[0]) & (t_ms <= baseline_win[1])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        baseline_power = np.nanmean(envelope[:, :, bl_mask], axis=2, keepdims=True)
                        baseline_power[baseline_power < 1e-10] = 1e-10
                        band_dB = 10 * np.log10(envelope / baseline_power)
                    
                    # Generate normal spatial heatmaps
                    if FIGURE_CONFIG.get('generate_spatial_heatmaps', True):
                        print("  Generating spatial heatmaps...")
                        generate_spatial_heatmaps(
                            band_dB,
                            t_ms,
                            ua_ids_1based,
                            session_id,
                            fig_dir,
                            groups,
                            elec_to_idx,
                            region_grids
                        )

                    # Generate CONTROL-NORM spatial heatmaps
                    if (FIGURE_CONFIG.get('generate_control_norm_spatial_heatmaps', False)
                        and control_spatial_baseline is not None):

                        if control_spatial_baseline.shape[1] == envelope.shape[1]:
                            eps = 1e-12
                            band_dB_control = 10 * np.log10(
                                np.maximum(envelope, eps) / np.maximum(control_spatial_baseline, eps)
                            )

                            print("  Generating CONTROL-NORM spatial heatmaps...")
                            generate_spatial_heatmaps(
                                band_dB_control,
                                t_ms,
                                ua_ids_1based,
                                session_id + "_CONTROL_NORM",
                                fig_dir,
                                groups,
                                elec_to_idx,
                                region_grids
                            )
                        else:
                            print("    [Skip] Control spatial baseline channel mismatch")
                    
                    # Generate spatial GIF
                    if FIGURE_CONFIG.get('generate_spatial_gif', True):
                        print("  Generating spatial GIF...")
                        generate_spatial_gif(band_dB, t_ms, ua_ids_1based, session_id,
                                            fig_dir, groups, nsp_to_elec, elec_to_idx, region_grids)
                
                except Exception as e:
                    print(f"    [Error] Spatial visualization failed: {e}")
        else:
            print("  [Skip] Spatial visualizations require electrode mapping")
    
    print(f"\n  Generated {len(figure_paths)} figure files")
    return figure_paths


# =============================================================================
# DIRECTORY PROCESSING
# =============================================================================

def process_directory(lfp_dir, fig_dir, label="LFP"):
    """Process all aligned LFP files in a directory, controls first."""
    from pathlib import Path
    
    if not lfp_dir.exists():
        print(f"Directory not found: {lfp_dir}")
        return

    files = sorted(lfp_dir.rglob("aligned_lfp__*.npz"))
    if not files:
        print(f"No aligned LFP npz files found in {lfp_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {label} ({len(files)} files found)")
    print(f"{'='*60}")
    
    # Show filter settings
    if SESSION_FILTER is not None:
        print(f"  Session filter (BR indices): {SESSION_FILTER}")
    if SESSION_ID_PATTERN is not None:
        print(f"  Session ID pattern: {SESSION_ID_PATTERN}")
    
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # PASS 1: Load and organize all sessions
    # =========================================================================
    print("\n--- Pass 1: Loading and organizing sessions ---")
    
    all_sessions = {}
    control_candidates_by_target = {}
    dose_response_cache = {}  # base_id -> {n_pulses: data}
    
    for npz_path in files:
        session_id = npz_path.stem.replace("aligned_lfp__", "")
        
        # Quick filter checks before loading
        if SESSION_ID_PATTERN is not None:
            if not re.search(SESSION_ID_PATTERN, session_id, re.IGNORECASE):
                continue
        
        if SESSION_FILTER is not None:
            br_match = re.search(r'_(\d{3})(?:_|$|\.)', session_id)
            if br_match:
                br_idx = int(br_match.group(1))
                if br_idx not in SESSION_FILTER:
                    continue
        
        print(f"  Loading {npz_path.name}...")
        
        try:
            data = dict(np.load(npz_path, allow_pickle=True))
        except Exception as e:
            print(f"    [Error] Failed to load: {e}")
            continue
        
        if 'broadband_full' not in data:
            print(f"    [Skip] No broadband_full data")
            continue
        
        # === EXTRACT TARGET (A or B) ===
        target_match = re.search(r'target_([AB])$', session_id, re.IGNORECASE)
        target = target_match.group(1).upper() if target_match else None
        
        # === DETERMINE IF CONTROL AND PULSE COUNT ===
        n_pulses = None
        is_control = False
        
        # Check metadata first
        if 'n_pulses' in data:
            n_pulses = int(data['n_pulses'])
        if 'is_control' in data:
            is_control = bool(data['is_control'])
        
        # Parse from filename if not in metadata
        if n_pulses is None:
            # Look for control/baseline indicators
            if ('control' in session_id.lower() or 
                'baseline' in session_id.lower() or
                '_0p_' in session_id or 
                '_0pulse' in session_id):
                n_pulses = 0
                is_control = True
            else:
                # Try to extract pulse count: _50p_, _100pulses, etc.
                pulse_match = re.search(r'_(\d+)p(?:ulse)?s?(?:_|$)', session_id, re.IGNORECASE)
                if pulse_match:
                    n_pulses = int(pulse_match.group(1))
                    is_control = (n_pulses == 0)
                else:
                    # Look up from aligned npz file matching this BR index to get stimulation duration
                    br_match = re.search(r'_(\d{3})(?:_|$|\.)', session_id)
                    if br_match:
                        br_idx = int(br_match.group(1))
                        try:
                            import RCP_analysis.python.functions.config_loading as cfg
                            import json
                            aligned_files = list(cfg.ALIGNED_CKPT_ROOT.rglob(f"*__BR_{br_idx:03d}.npz"))
                            if aligned_files:
                                aln_data = np.load(aligned_files[0], allow_pickle=True)
                                if 'align_meta' in aln_data:
                                    meta_raw = aln_data['align_meta'].item()
                                    meta = meta_raw if isinstance(meta_raw, dict) else json.loads(meta_raw)
                                    stim_dur = float(meta.get('recording_stim_dur', 0.0))
                                    if stim_dur > 0:
                                        n_pulses = int(round(stim_dur / 2.5))
                                        is_control = (n_pulses == 0)
                        except Exception as e:
                            print(f"    [Warn] Could not load stim duration from aligned file for BR {br_idx}: {e}")
                        
                        # Fallback to stim npz
                        if n_pulses is None:
                            try:
                                import RCP_analysis as rcp
                                import RCP_analysis.python.functions.config_loading as cfg
                                stim_npz_path, _ = rcp.stim_npz_path_from_br_idx(br_idx, cfg.METADATA_CSV, cfg.NPRW_AUX_DATA)
                                if stim_npz_path and stim_npz_path.exists():
                                    stim = rcp.load_stim_detection(stim_npz_path)
                                    block_bounds = stim.get("block_bounds_samples", [])
                                    if len(block_bounds) > 0:
                                        br2fs_intan = rcp.get_metadata_mapping(cfg.METADATA_ROOT / "br_to_intan_shifts.csv", 'br_idx', 'fs_intan')
                                        fs_intan = float(br2fs_intan.get(br_idx, 30000.0))
                                        block_durs = block_bounds[:, 1] - block_bounds[:, 0]
                                        median_dur_samples = np.median(block_durs)
                                        stim_dur_ms = median_dur_samples * 1000.0 / fs_intan
                                        n_pulses = stim_dur_ms / 2.5
                                        is_control = (n_pulses == 0)
                            except Exception as e:
                                pass
        
        # Extract base session ID (for dose-response grouping)
        base_id = re.sub(r'_\d+p(?:ulse)?s?(?:_|$)', '_', session_id, flags=re.IGNORECASE)
        base_id = re.sub(r'_control(?:_|$)', '_', base_id, flags=re.IGNORECASE)
        base_id = re.sub(r'_baseline(?:_|$)', '_', base_id, flags=re.IGNORECASE)
        base_id = base_id.rstrip('_')
        
        # Get sampling rate and channel info
        fs = int(data.get('fs_lfp', 1000))
        n_ch = data['broadband_full'].shape[1]
        
        # Build groups
        groups, nsp_to_elec, elec_to_idx, region_grids, ua_ids = _build_groups(data, n_ch)
        
        session_info = {
            'npz_path': npz_path,
            'session_id': session_id,
            'base_id': base_id,
            'target': target,  # NEW: store target
            'data': data,
            'fs': fs,
            'groups': groups,
            'nsp_to_elec': nsp_to_elec,
            'elec_to_idx': elec_to_idx,
            'region_grids': region_grids,
            'ua_ids': ua_ids,
            'n_pulses': n_pulses,
            'is_control': is_control,
        }
        
        all_sessions[session_id] = session_info
        
        # === CACHE CONTROL CANDIDATES BY TARGET ===
        if is_control or n_pulses == 0:
            target_key = target if target is not None else '_default'
            n_trials = get_trial_count(data)

            control_candidates_by_target.setdefault(target_key, []).append({
                'data': data,
                'session_id': session_id,
                'n_trials': n_trials,
                'fs': fs,
                'groups': groups,
                'elec_to_idx': elec_to_idx,
                'region_grids': region_grids,
            })

            print(f"    -> Control/Baseline for target {target_key}, trials={n_trials}")
        else:
            print(f"    -> Stim session: {n_pulses if n_pulses else '?'} pulses, target {target}")
        
        # Build dose-response dict (group by target if available)
        dose_key = f"{base_id}_{target}" if target else base_id
        if dose_key not in dose_response_cache:
            dose_response_cache[dose_key] = {}
        if n_pulses is not None:
            dose_response_cache[dose_key][n_pulses] = data
    

    # =========================================================================
    # SELECT BEST CONTROL PER TARGET
    # =========================================================================
    control_data_by_target = {}
    control_median_baseline_norm_by_target = {}
    control_spatial_baseline_by_target = {}

    for target_key, candidates in control_candidates_by_target.items():
        best = max(candidates, key=lambda x: x['n_trials'])

        control_data_by_target[target_key] = best['data']

        print(
            f"  Best control for target {target_key}: "
            f"{best['session_id']} ({best['n_trials']} trials)"
        )

        if FIGURE_CONFIG.get('generate_control_median_baseline_norm_array_median', False):
            control_median_baseline_norm_by_target[target_key] = compute_control_median_baseline_norm(
                best['data'],
                best['groups'],
                fs=best['fs'],
                region_grids=best['region_grids'],
                elec_to_idx=best['elec_to_idx'],
            )

        if FIGURE_CONFIG.get('generate_control_norm_spatial_heatmaps', False):
            control_spatial_baseline_by_target[target_key] = compute_control_spatial_baseline(
                best['data'],
                fs=best['fs'],
            )

    
    # =========================================================================
    # PASS 2: Process sessions (controls already loaded)
    # =========================================================================
    print(f"\n--- Pass 2: Generating figures for {len(all_sessions)} sessions ---")
    
    # Show what controls we found
    print(f"  Controls found by target: {list(control_data_by_target.keys())}")
    
    processed_count = 0
    
    for session_id, info in all_sessions.items():
        target = info['target']

        target_key = target if target is not None else '_default'

        control_median_baseline_norm = control_median_baseline_norm_by_target.get(target_key)
        if control_median_baseline_norm is None:
            control_median_baseline_norm = control_median_baseline_norm_by_target.get('_default')

        control_spatial_baseline = control_spatial_baseline_by_target.get(target_key)
        if control_spatial_baseline is None:
            control_spatial_baseline = control_spatial_baseline_by_target.get('_default')

        if info['is_control']:
            control_median_baseline_norm_for_session = None
        else:
            control_median_baseline_norm_for_session = control_median_baseline_norm
        
        # === GET CONTROL DATA FOR THIS SESSION'S TARGET ===
        data_control = None
        if not info['is_control']:
            # Try to get control for this target
            if target and target in control_data_by_target:
                data_control = control_data_by_target[target]
            elif '_default' in control_data_by_target:
                data_control = control_data_by_target['_default']
            
            if data_control is not None:
                print(f"  Matched control for {session_id} (target {target})")
        
        # Get dose-response dict (only if multiple conditions exist)
        dose_key = f"{info['base_id']}_{target}" if target else info['base_id']
        session_data_dict = None
        if dose_key in dose_response_cache and len(dose_response_cache[dose_key]) > 1:
            session_data_dict = dose_response_cache[dose_key]
        
        # Process this session
        process_session(
            data=info['data'],
            session_id=info['session_id'],
            fig_dir=fig_dir,
            groups=info['groups'],
            fs=info['fs'],
            data_control=data_control,
            session_data_dict=session_data_dict,
            nsp_to_elec=info['nsp_to_elec'],
            elec_to_idx=info['elec_to_idx'],
            region_grids=info['region_grids'],
            ua_ids_1based=info['ua_ids'],
            control_median_baseline_norm=control_median_baseline_norm_for_session,
            control_spatial_baseline=control_spatial_baseline,
        )
        
        processed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Done processing {label}")
    print(f"  Processed: {processed_count} sessions")
    print(f"{'='*60}")


def _build_groups(data, n_ch):
    """Helper to build channel groups and electrode mappings."""
    is_utah = 'ua_ids_1based' in data
    ua_ids = data.get('ua_ids_1based', None)
    
    nsp_to_elec = None
    elec_to_idx = None
    region_grids = None
    groups = []
    
    if is_utah:
        try:
            from RCP_analysis.python.functions.br_preproc import (
                load_electrode_mapping,
                get_region_grid,
                build_elec_to_data_idx
            )
            
            import RCP_analysis.python.functions.config_loading as cfg

            monkey = str(cfg.PARAMS.monkey)

            if monkey not in ("Nike", "Ada"):
                raise ValueError(f"Unknown PARAMS.monkey={monkey!r}; expected 'Nike' or 'Ada'")

            MAPPING_CSV = cfg.REPO_ROOT / "config" / f"electrode_port_mapping_{monkey}.csv"
            nsp_to_elec, UTAH_ELEC_GRIDS, elec_to_region = load_electrode_mapping(MAPPING_CSV)
            
            # Build groups by region
            sma_idxs, pmd_idxs, m1i_idxs, m1s_idxs = [], [], [], []
            
            for ch_idx in range(n_ch):
                nsp_id = int(ua_ids[ch_idx])
                elec_id = nsp_to_elec.get(nsp_id, -1)
                if elec_id < 1:
                    continue
                region = elec_to_region.get(elec_id)
                if region == "SMA":
                    sma_idxs.append(ch_idx)
                elif region == "PMd":
                    pmd_idxs.append(ch_idx)
                elif region in ["M1i", "M1 Inf"]:
                    m1i_idxs.append(ch_idx)
                elif region in ["M1s", "M1 Sup"]:
                    m1s_idxs.append(ch_idx)
            
            if sma_idxs: groups.append((sma_idxs, f"SMA (n={len(sma_idxs)})"))
            if pmd_idxs: groups.append((pmd_idxs, f"PMd (n={len(pmd_idxs)})"))
            if m1i_idxs: groups.append((m1i_idxs, f"M1i (n={len(m1i_idxs)})"))
            if m1s_idxs: groups.append((m1s_idxs, f"M1s (n={len(m1s_idxs)})"))
            
            # Build electrode mapping for spatial plots
            ua_port = 'A'
            elec_to_idx = build_elec_to_data_idx(ua_ids, nsp_to_elec, ua_port)
            
            # Region grids for spatial heatmaps
            region_grids = {
                'SMA': get_region_grid('SMA', UTAH_ELEC_GRIDS),
                'PMd': get_region_grid('PMd', UTAH_ELEC_GRIDS),
                'M1i': get_region_grid('M1i', UTAH_ELEC_GRIDS),
                'M1s': get_region_grid('M1s', UTAH_ELEC_GRIDS),
            }
            
        except Exception as e:
            print(f"    [Warn] Failed to load electrode mapping: {e}")
            is_utah = False
    
    if not groups:
        # Fallback: simple split into 4 groups
        ch_per_group = n_ch // 4
        groups = [
            (list(range(0, ch_per_group)), f"Group 1 (n={ch_per_group})"),
            (list(range(ch_per_group, 2*ch_per_group)), f"Group 2 (n={ch_per_group})"),
            (list(range(2*ch_per_group, 3*ch_per_group)), f"Group 3 (n={ch_per_group})"),
            (list(range(3*ch_per_group, n_ch)), f"Group 4 (n={n_ch - 3*ch_per_group})"),
        ]
    
    return groups, nsp_to_elec, elec_to_idx, region_grids, ua_ids


if __name__ == "__main__":
    import RCP_analysis.python.functions.config_loading as cfg
    
    # Define directories
    NPRW_LFP_DIR = cfg.OUT_BASE / "checkpoints" / "NPRW_LFP"
    UA_LFP_DIR = cfg.OUT_BASE / "checkpoints" / "UA_LFP"
    
    NPRW_FIG_DIR = cfg.OUT_BASE / "figures" / "NPRW_LFP"
    UA_FIG_DIR = cfg.OUT_BASE / "figures" / "UA_LFP"

    # -------------------------------------------------------------------------
    # WAVELET COMPARISON MODE
    # If WAVELET_TEST_CONFIG['enabled'] is True, run the wavelet comparison
    # instead of (or in addition to) the normal pipeline.
    # Edit WAVELET_TEST_CONFIG at the top of this file to configure.
    # -------------------------------------------------------------------------
    if WAVELET_TEST_CONFIG.get('enabled', False):
        print("\n[WAVELET TEST MODE ACTIVE]")
        run_wavelet_comparison(NPRW_LFP_DIR, NPRW_FIG_DIR, label="NPRW_Intan")
        run_wavelet_comparison(UA_LFP_DIR, UA_FIG_DIR, label="Utah_Array")
    else:
        # Normal pipeline
        process_directory(NPRW_LFP_DIR, NPRW_FIG_DIR, label="NPRW_Intan")
        process_directory(UA_LFP_DIR, UA_FIG_DIR, label="Utah_Array")