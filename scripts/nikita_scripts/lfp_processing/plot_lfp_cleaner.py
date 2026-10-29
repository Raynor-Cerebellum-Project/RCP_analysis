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
from PIL import Image
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
# Load process_only dynamically from RCP_PROCESS_ONLY env var or PARAMS
try:
    import json
    import os
    env_po = os.environ.get("RCP_PROCESS_ONLY")
    if env_po:
        SESSION_FILTER = [int(x) for x in json.loads(env_po)]
    else:
        from RCP_analysis.python.functions.config_loading import PARAMS
        _po = PARAMS.preprocessing.get("process_only") if PARAMS is not None else None
        SESSION_FILTER = [int(x) for x in _po] if _po else None
except Exception:
    SESSION_FILTER = None

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
    'generate_per_ua_spectrograms': False,  # 8x8 grid of spectrograms per array (turned off to omit individual array plots)
    'per_ua_default_trial': 3, 
    'generate_all_trial_spectrograms': False,  # If True, generate for each trial (can be many files!)
    'generate_per_ua_trial_average': False,
    'generate_per_ua_array_median': True,  # Median across all trials AND channels per Utah array (entire spectrogram)
    'generate_control_median_baseline_norm_array_median': False, # Now replaced by true Stim minus Control subtraction (ΔdB)
    'generate_stim_minus_control_array_median': True,  # 4-panel Stim minus Matched Control array median
    'generate_stim_minus_control_per_ua': False,        # 8x8 grid Stim minus Control spectrograms per array
    'generate_causal_diagnostic': False,      # Causal filterbank vs acausal Morlet comparison (0 to 50 ms)
    'generate_causal_zoomed_spectrogram': False,  # Alias for backward compatibility
    

    # Inter-regional analyses
    'generate_coherence_matrix': False,      # Coherence between all array pairs
    
    # Condition comparisons
    'generate_stim_vs_control': False,       # Compare stim to control trials
    'generate_dose_response': False,         # Effect vs number of pulses
    
    # Spatial visualizations
    'generate_spatial_heatmaps': False,      # Static spatial maps at key timepoints
    'generate_spatial_gif': True,            # Animated GIF of spatial activity
    'generate_spatial_gif_blocky': False,    # Blocky GIF turned off per user request
    'generate_spatial_gif_smoothed': True,   # Smoothed interpolated GIF
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
    
    # Band definitions (aligned with plot_lfp_rsa DEFAULT_BANDS)
    'bands': {
        'Low Frequency': (2.0, 12.0),
        'Delta': (1.0, 4.0),
        'Theta': (4.0, 8.0),
        'Alpha': (8.0, 12.0),
        'Beta': (13.0, 25.0),
        'Low Gamma': (25.0, 55.0),
        'LowGamma': (25.0, 55.0),
        'High Gamma': (70.0, 110.0),
        'HighGamma': (70.0, 110.0),
    },

    'notch_regions': [(55.0, 65.0), (115.0, 125.0)],
    
    # Coherence settings
    'coherence_nperseg': 256,
    'coherence_noverlap': 192,
    
    # Phase lag settings
    'phase_lag_max_ms': 100,                # Maximum lag to consider
    
    # Spatial analysis & GIF settings
    'spatial_method': 'multitaper',         # TFR extraction method: 'morlet', 'multitaper', or 'hilbert'
    'spatial_bands': ['Beta', 'High Gamma'], # Bands to generate spatial heatmaps and GIFs for
    'gif_time_range': (-500, 500),
    'gif_step_ms': 25,
    'gif_fps': 10,
}

PLOT_CONFIG = {
    # Color scheme for regions matching plot_lfp_rsa
    'region_colors': {
        'SMA': '#1f77b4',               # Blue
        'PMd': '#ff7f0e',               # Orange
        'M1i': '#2ca02c',               # Green
        'M1s': '#d62728',               # Red
        'M1 (upper limb)': '#2ca02c',   # Green
        'M1 (lower limb)': '#d62728',   # Red
        'Premotor (dorsal)': '#ff7f0e', # Orange
        'M1 Inf': '#2ca02c',            # Green
        'M1 Sup': '#d62728',            # Red
    },
    
    # Anatomical display titles for clean publication labels
    'region_display_names': {
        'M1s': 'M1 (lower limb)',
        'M1 Sup': 'M1 (lower limb)',
        'M1 (lower limb)': 'M1 (lower limb)',
        'M1i': 'M1 (upper limb)',
        'M1 Inf': 'M1 (upper limb)',
        'M1 (upper limb)': 'M1 (upper limb)',
        'PMd': 'Premotor (dorsal)',
        'Premotor (dorsal)': 'Premotor (dorsal)',
        'SMA': 'SMA',
    },

    # Title & Header Display Options
    'show_titles': True,                # Master switch for titles
    'show_suptitle': False,             # If False, omit top multi-line suptitle (centered color scale & clean publication look)
    'show_subplot_titles': True,        # Keep region subplot titles ('M1 (lower limb)', 'SMA', etc.)

    # Heatmap settings
    'heatmap_cmap': 'jet', # or RdBu_r
    'heatmap_vmin': -3.0,                   # dB
    'heatmap_vmax': 3.0,                    # dB
    
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
        #   'multitaper'      = sliding-window DPSS multitaper (GPU-accelerated)
        'use_wavelet': True,
        'tfr_method': 'multitaper',  # 'multitaper', 'morlet_adaptive', or 'morlet'

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

        # Multitaper settings
        # Window duration (win_ms), time-halfbandwidth (NW), number of tapers (K)
        # Optimal trade-off: win_ms=100, nw=2.0, k=2
        'multitaper_win_ms': 100,           # Window length in ms
        'multitaper_nw': 2.0,               # Time-halfbandwidth product NW
        'multitaper_k': 2,                  # Number of tapers (typically <= 2*NW - 1)
        'multitaper_step_ms': 10,           # Sliding window step in ms
        
        # Frequency settings
        'freq_min': 1,
        'freq_max': 120,
        'freq_step': 1,                     # Hz resolution
        
        # Display settings
        'cmap': 'jet', #viridis
        'normalize': 'dB',                  # 'dB', 'zscore', or 'raw'
        'vmin_zscore': -6,                  # z-score limits
        'vmax_zscore': 6,                   # z-score limits
        'vmin_uV2': 0,                      # For absolute power
        'vmax_uV2': None,                   # None = auto-scale to 95th percentile
        'vmin_db': -3.0,                    # For dB (if using relative)
        'vmax_db': 3.0,
    },
    
    # Spatial heatmap timepoints
    'spatial_timepoints_ms': [-650, -600, -550, -500, -450, -400, -350, -300, -250, -200, -175, -150, -125, -100, -75, -50, -25, 
                                0, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500],
    'spatial_static_timepoints_ms': [-200, -50, 0, 50, 100, 200, 300, 500], # For 8-panel static heatmap figures
    
    # Spatial heatmap & GIF configuration
    'spatial_bands': ['Beta', 'High Gamma'], # Bands to generate spatial plots/GIFs for
    'spatial_method': 'multitaper',             # Options: 'morlet', 'multitaper', 'hilbert'
    'selected_band': 'Beta',                # Fallback for single-band calls
}

# =============================================================================
# MULTITAPER PARAMETER SWEEP CONFIGURATION
# =============================================================================
# Set MULTITAPER_SWEEP_CONFIG['enabled'] = True to run a multi-parameter multitaper
# sweep across conditions (e.g. 11, 17, 18) generating comparison figures.
MULTITAPER_SWEEP_CONFIG = {
    'enabled': False,
    'conditions': [17, 18],     # Cond 11 (1 pulse), 17 (40 pulse), 18 (control)
    'target': 'target_A',
    'regions': ['M1i', 'PMd'],
    'normalize': 'dB',              # Default to dB to prevent high-frequency noise blowout
    'plot_difference': True,        # Include Stim minus Control difference panel
    'param_configs': [
        {'win_ms': 100, 'nw': 2.0, 'k': 3, 'label': 'Window = 100 ms | NW = 2.0 | K = 3 tapers'},
        {'win_ms': 100, 'nw': 3.0, 'k': 3, 'label': 'Window = 100 ms | NW = 3.0 | K = 3 tapers'},
        {'win_ms': 100, 'nw': 2.0, 'k': 4, 'label': 'Window = 100 ms | NW = 2.0 | K = 4 tapers'},
        {'win_ms': 100, 'nw': 3.0, 'k': 4, 'label': 'Window = 100 ms | NW = 3.0 | K = 4 tapers'},
        {'win_ms': 100, 'nw': 2.0, 'k': 2, 'label': 'Window = 100 ms | NW = 2.0 | K = 2 tapers'},
        {'win_ms': 100, 'nw': 3.0, 'k': 2, 'label': 'Window = 100 ms | NW = 3.0 | K = 2 tapers'},
    ],
    'output_subdir': 'multitaper_parameter_sweep',
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
        # Multitaper
        {'label': 'Multitaper 200ms', 'method': 'multitaper',
         'win_ms': 200, 'nw': 3.0, 'k': 5, 'step_ms': 10},
        # STFT: uniform time-frequency resolution, no wavelet smearing
        {'label': 'STFT 250ms',  'method': 'stft', 'window_ms': 250},
        {'label': 'STFT 100ms',  'method': 'stft', 'window_ms': 100},
        # Paul wavelet (m=4): better time resolution than Morlet, good for transients
        {'label': 'Paul m=4',    'method': 'paul', 'order': 4},
        # Causal SOS filterbank: forward-only, strictly zero backward precursor before t=0
        {'label': 'Causal SOS (forward-only)', 'method': 'causal_sos', 'order': 2},
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

def get_spectrogram_ticks(f_spec, tick_freqs=None, min_index_spacing=4):
    """
    Compute y-tick positions and labels for a spectrogram where notch regions
    are skipped (non-contiguous frequency space).
    
    Prevents adjacent frequency labels from colliding or overlapping
    (e.g., notch boundary collision around 55-65 Hz).
    """
    if tick_freqs is None:
        tick_freqs = [1, 10, 20, 30, 40, 50, 70, 80, 100]
    
    f_arr = np.asarray(f_spec)
    if len(f_arr) == 0:
        return [], []

    tick_positions = []
    tick_labels = []
    
    for tf in tick_freqs:
        # Check range
        if tf < np.min(f_arr) - 2.0 or tf > np.max(f_arr) + 2.0:
            continue

        idx = int(np.argmin(np.abs(f_arr - tf)))
        closest_f = float(f_arr[idx])
        if abs(closest_f - tf) > 4.0:
            continue

        lbl = str(int(round(closest_f)))
        
        # Enforce minimum index distance between consecutive ticks to prevent text overlap
        if tick_positions and (idx - tick_positions[-1]) < min_index_spacing:
            continue
        tick_positions.append(idx)
        tick_labels.append(lbl)
        
    return tick_positions, tick_labels

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
    if psd_masked.size == 0 or np.asarray(f).size == 0:
        return psd_masked

    f_1d = np.asarray(f).ravel()
    
    for f_lo, f_hi in notch_regions:
        mask = (f_1d >= f_lo) & (f_1d <= f_hi)
        if psd_masked.ndim == 1:
            if mask.size == psd_masked.size:
                psd_masked[mask] = np.nan
        else:
            # Assume frequency is last axis
            if mask.size == psd_masked.shape[-1]:
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
    """Get color for a region matching plot_lfp_rsa color palette."""
    rn = str(region_name).lower()
    if 'upper' in rn or 'm1i' in rn or 'inf' in rn:
        return '#2ca02c'  # Green
    if 'lower' in rn or 'm1s' in rn or 'sup' in rn:
        return '#d62728'  # Red
    if 'premotor' in rn or 'pmd' in rn:
        return '#ff7f0e'  # Orange
    if 'sma' in rn:
        return '#1f77b4'  # Blue
    colors = PLOT_CONFIG.get('region_colors', {})
    for key in colors:
        if key.lower() in rn:
            return colors[key]
    return '#2c3e50'

# 2x2 Layout Order matching plot_lfp_rsa:
# Top-Left: M1s (M1 lower limb), Top-Right: SMA
# Bottom-Left: M1i (M1 upper limb), Bottom-Right: PMd (Premotor dorsal)
PREFERRED_REGION_ORDER = ['M1s', 'SMA', 'M1i', 'PMd']

def normalize_region_name(r_name):
    """Normalize region name string to one of: M1s, SMA, M1i, PMd."""
    rn = str(r_name).lower()
    if 'lower' in rn or 'm1s' in rn or 'sup' in rn:
        return 'M1s'
    if 'sma' in rn:
        return 'SMA'
    if 'upper' in rn or 'm1i' in rn or 'inf' in rn:
        return 'M1i'
    if 'premotor' in rn or 'pmd' in rn:
        return 'PMd'
    return r_name

def order_groups_preferred(groups):
    """
    Sort groups list according to PREFERRED_REGION_ORDER:
    Top-Left: M1s (M1 lower limb), Top-Right: SMA
    Bottom-Left: M1i (M1 upper limb), Bottom-Right: PMd (Premotor dorsal)
    """
    def get_sort_key(grp):
        grp_name = grp[1] if isinstance(grp, (list, tuple)) and len(grp) > 1 else str(grp)
        rn = grp_name.split(' (')[0].strip()
        rn_norm = normalize_region_name(rn)
        try:
            return PREFERRED_REGION_ORDER.index(rn_norm)
        except ValueError:
            return 99
    return sorted(groups, key=get_sort_key)

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

    multitaper_win_ms = spec_cfg.get('multitaper_win_ms', 100)
    multitaper_nw = spec_cfg.get('multitaper_nw', 2.0)
    multitaper_k = spec_cfg.get('multitaper_k', 2)
    multitaper_step_ms = spec_cfg.get('multitaper_step_ms', 10)

    # Adaptive Morlet uses CPU path unless/until Torch adaptive support is added.
    # Torch GPU path supports fixed-cycle Morlet, adaptive Morlet, and Multitaper.
    use_gpu = (
        tfr_method in ('morlet', 'morlet_adaptive', 'multitaper')
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
        elif tfr_method == 'multitaper':
            half_bw = multitaper_nw / (multitaper_win_ms / 1000.0)
            print(
                f"    Torch GPU Multitaper enabled: {gpu_name}, "
                f"win={multitaper_win_ms}ms, NW={multitaper_nw}, K={multitaper_k} tapers (W=±{half_bw:.1f} Hz), "
                f"step={multitaper_step_ms}ms, batch_size={gpu_batch_size}"
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
        elif tfr_method == 'multitaper':
            half_bw = multitaper_nw / (multitaper_win_ms / 1000.0)
            print(
                f"    Multitaper enabled: win={multitaper_win_ms}ms, NW={multitaper_nw}, "
                f"K={multitaper_k} tapers (W=±{half_bw:.1f} Hz), step={multitaper_step_ms}ms (CPU path)"
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
                elif tfr_method == 'multitaper':
                    Sxx_all, f_out, t_out = compute_multitaper_spectrogram_batch_torch(
                        traces_arr,
                        fs=fs,
                        foi=foi,
                        win_ms=multitaper_win_ms,
                        nw=multitaper_nw,
                        k=multitaper_k,
                        step_ms=multitaper_step_ms,
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
            elif tfr_method == 'multitaper':
                method_cfg = {
                    'method': 'multitaper',
                    'win_ms': multitaper_win_ms,
                    'nw': multitaper_nw,
                    'k': multitaper_k,
                    'step_ms': multitaper_step_ms,
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

def compute_array_median_causal_raw_spectrograms(data, groups, fs=1000,
                                                region_grids=None, elec_to_idx=None,
                                                order=2, n_cycles=5):
    """
    Compute strictly causal forward-only filterbank spectrogram median across all
    valid trial/channel pairs for each array/region.

    Uses the exact same frequencies, trial time window, and channel pooling as
    compute_array_median_raw_spectrograms, but replaces acausal Morlet wavelets with
    strictly causal forward-only Butterworth bandpass filtering (sosfilt),
    analytic envelope extraction (hilbert), and frequency-matched causal
    recursive exponential smoothing.

    Returns dict mapping region_name -> {
        'Sxx': Sxx_region_median_raw,
        'f': foi,
        't_ms': t_spec_ms_saved,
        'n_valid_channels': n_valid_channels,
        'n_valid_trial_channel_pairs': n_valid_trial_channel_pairs,
    }
    """
    spec_cfg = PLOT_CONFIG.get('spectrogram', {})
    freq_min = spec_cfg.get('freq_min', 1)
    freq_max = spec_cfg.get('freq_max', 120)
    freq_step = spec_cfg.get('freq_step', 1)
    foi = get_foi_with_notch_gaps(freq_min, freq_max, freq_step)

    bb_full, t_ms = get_broadband_full(data)
    if bb_full is None:
        print("    WARNING: broadband data missing; cannot compute causal spectrograms.")
        return {}

    bb_full = np.asarray(bb_full)
    if bb_full.ndim != 3:
        print(f"    WARNING: expected broadband data shape trial x channel x time, got {bb_full.shape}")
        return {}

    n_trials, n_channels, n_time = bb_full.shape
    t_ms_arr = np.asarray(t_ms)

    region_specs_raw = {}

    # Build map of regions to process: region_name -> list of channel indices
    region_channel_map = {}
    if region_grids is not None and elec_to_idx is not None:
        for grp_idxs, grp_name in groups:
            r_name = grp_name.split(' (')[0]
            if r_name in region_grids:
                grid_elec = region_grids[r_name]
                ch_indices = []
                for elec_id in np.asarray(grid_elec).ravel():
                    if elec_id is None:
                        continue
                    try:
                        if np.isnan(float(elec_id)):
                            continue
                    except (ValueError, TypeError):
                        continue
                    ch_idx = elec_to_idx.get(int(elec_id), None)
                    if ch_idx is not None and 0 <= ch_idx < n_channels:
                        ch_indices.append(ch_idx)
                if ch_indices:
                    region_channel_map[r_name] = sorted(list(set(ch_indices)))

    if not region_channel_map:
        for grp_idxs, grp_name in groups:
            r_name = grp_name.split(' (')[0]
            valid_idxs = [idx for idx in grp_idxs if 0 <= idx < n_channels]
            if valid_idxs:
                region_channel_map[r_name] = sorted(list(set(valid_idxs)))

    for region_name, ch_indices in region_channel_map.items():
        print(f"    Computing causal array-median spectrogram for {region_name}...")

        region_traces = []
        for ch_idx in ch_indices:
            for trial_idx in range(n_trials):
                trace = bb_full[trial_idx, ch_idx, :]
                if trace is None or trace.size != n_time:
                    continue
                if not np.isfinite(trace).any():
                    continue
                region_traces.append(trace.astype(np.float64))

        n_valid_channels = len(ch_indices)
        n_valid_trial_channel_pairs = len(region_traces)

        if n_valid_trial_channel_pairs == 0:
            print(f"      No valid trial/channel pairs for {region_name}; skipping.")
            continue

        traces_arr = np.stack(region_traces, axis=0)

        # Interpolate NaNs across time to protect IIR/Hilbert filtering from NaN propagation
        for i_tr in range(len(traces_arr)):
            nan_m = np.isnan(traces_arr[i_tr])
            if nan_m.any():
                if nan_m.all():
                    traces_arr[i_tr] = 0.0
                else:
                    x_good = np.flatnonzero(~nan_m)
                    x_bad = np.flatnonzero(nan_m)
                    traces_arr[i_tr, x_bad] = np.interp(x_bad, x_good, traces_arr[i_tr, x_good])

        n_freq = len(foi)
        Sxx_region_median_raw = np.zeros((n_freq, n_time), dtype=np.float64)

        for i_f, f in enumerate(foi):
            bw = max(1.5, f / float(n_cycles))
            low = max(0.5, f - bw / 2.0)
            high = min(fs / 2.0 - 1.0, f + bw / 2.0)
            if low >= high:
                continue

            sos = signal.butter(order, [low, high], btype='bandpass', fs=fs, output='sos')

            # 1. Strictly causal forward-only filtering across all traces
            filtered = signal.sosfilt(sos, traces_arr, axis=-1)

            # 2. Instantaneous analytic envelope power across all traces
            analytic = signal.hilbert(filtered, axis=-1)
            envelope_power = np.abs(analytic) ** 2

            # 3. Frequency-matched causal exponential smoothing
            tau = float(n_cycles) / (2.0 * np.pi * f)
            alpha = np.exp(-1.0 / max(tau * fs, 1.0))
            b_filt = [1.0 - alpha]
            a_filt = [1.0, -alpha]
            zi = signal.lfilter_zi(b_filt, a_filt)
            zi_batch = np.outer(envelope_power[:, 0], zi)
            envelope_smooth, _ = signal.lfilter(b_filt, a_filt, envelope_power, axis=-1, zi=zi_batch)

            # 4. Median across all trial/channel traces for this frequency
            Sxx_region_median_raw[i_f, :] = np.nanmedian(envelope_smooth, axis=0)

        region_specs_raw[region_name] = {
            'Sxx': Sxx_region_median_raw,
            'f': foi,
            't_ms': t_ms_arr,
            'n_valid_channels': n_valid_channels,
            'n_valid_trial_channel_pairs': n_valid_trial_channel_pairs,
        }
        print(
            f"      {region_name}: Causal filterbank complete, "
            f"{n_valid_channels} channels, "
            f"{n_valid_trial_channel_pairs} trial/channel pairs"
        )

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
        bb = np.asarray(data['broadband_full'])
        t_ms = get_time_axis(data)

        # Make sure selected time axis matches broadband_full time dimension
        if bb.ndim == 3:
            n_time = bb.shape[-1]
            t_ms = np.asarray(t_ms)

            if t_ms.size != n_time:
                matched = False

                # Try alternate full-time keys
                for key in ['t_full_ms', 't_ms', 'rel_time_full']:
                    if key in data:
                        cand = np.asarray(data[key])
                        if cand.ndim == 1 and cand.size == n_time:
                            print(
                                f"    [Info] Using {key} as time axis because it matches "
                                f"broadband_full length {n_time}"
                            )
                            t_ms = cand
                            matched = True
                            break

                # If no saved time axis matches, reconstruct one from fs_lfp
                if not matched:
                    fs = float(data.get('fs_lfp', 1000))
                    dt_ms = 1000.0 / fs

                    old_t = np.asarray(t_ms).ravel()
                    if old_t.size:
                        start_ms = float(old_t[0])
                    else:
                        start_ms = -0.5 * n_time * dt_ms

                    print(
                        f"    [Warn] Time axis length mismatch: broadband_full has {n_time} samples, "
                        f"time axis has {old_t.size}; reconstructing from fs_lfp={fs}"
                    )

                    t_ms = start_ms + np.arange(n_time) * dt_ms

        return bb, np.asarray(t_ms)
    
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

    # Optional safety check for legacy path too
    if broadband_full.shape[-1] != t_ms.size:
        raise ValueError(
            f"Legacy broadband/time mismatch: "
            f"broadband_full.shape[-1]={broadband_full.shape[-1]}, "
            f"len(t_ms)={t_ms.size}"
        )
    
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

def get_clean_pre_post_windows(data, bb_full, t_ms):
    """
    Determine clean (pre_start, pre_end) and (post_start, post_end) time bounds in ms.
    
    For NPRW or stimulation sessions with blanking periods (which contain NaNs during
    stimulation, e.g. [-20 ms, stim_dur + 20 ms]), ensures that:
      - pre_end ends strictly before any blanking/NaNs begin.
      - post_start starts strictly after any blanking/NaNs end.
    This guarantees that slice_time_window produces completely finite arrays,
    preventing scipy.signal.welch from producing all-NaN PSD curves.
    """
    pre_start = float(TIME_WINDOWS['pre_start'])
    pre_end = float(TIME_WINDOWS['pre_end'])
    post_start = float(TIME_WINDOWS['post_start'])
    post_end = float(TIME_WINDOWS['post_end'])
    
    t_ms = np.asarray(t_ms)
    
    # Check if bb_full has NaNs
    if bb_full is not None and np.isnan(bb_full).any():
        nan_mask_t = np.isnan(bb_full).any(axis=tuple(range(bb_full.ndim - 1)))
        nan_times = t_ms[nan_mask_t]
        if len(nan_times) > 0:
            nan_min = float(nan_times.min())
            nan_max = float(nan_times.max())
            if nan_min < pre_end:
                pre_end = min(pre_end, nan_min - 1.0)
            if nan_max > post_start:
                post_start = max(post_start, nan_max + 1.0)
    elif 'stim_dur_ms' in data and not data.get('is_control', False):
        stim_dur = float(data.get('stim_dur_ms', 0.0))
        if stim_dur > 0 and 'ua_ids_1based' not in data:
            # NPRW blanking duration fallback
            pre_end = min(pre_end, -25.0)
            post_start = max(post_start, stim_dur + 25.0)

    # Bound within available time range
    if t_ms.size > 0:
        pre_start = max(pre_start, float(t_ms[0]))
        post_end = min(post_end, float(t_ms[-1]))
    
    return (pre_start, pre_end), (post_start, post_end)

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


def compute_multitaper_spectrogram_batch_torch(
    traces, fs, foi=None, win_ms=200, nw=3.0, k=5, step_ms=10, batch_size=64, device=None
):
    """
    GPU PyTorch sliding-window DPSS multitaper spectrogram for a batch of traces.

    Parameters
    ----------
    traces : array-like, shape (n_signals, n_time)
        Raw LFP traces.
    fs : float
        Sampling frequency in Hz.
    foi : array-like or None
        Frequencies of interest (Hz). If None, uses get_foi_with_notch_gaps().
    win_ms : float
        Window duration in milliseconds.
    nw : float
        Time-halfbandwidth product NW.
    k : int
        Number of DPSS tapers to average (typically <= 2*NW - 1).
    step_ms : float
        Sliding window time step in milliseconds.
    batch_size : int
        Number of traces processed per GPU batch.
    device : str or torch.device or None
        Target device ("cuda" or "cpu").

    Returns
    -------
    Sxx_all : np.ndarray, shape (n_signals, n_freqs, n_steps)
        Multitaper power spectrogram.
    foi : np.ndarray, shape (n_freqs,)
        Frequencies of interest.
    t_out : np.ndarray, shape (n_steps,)
        Time centers of windows in seconds relative to trace start.
    """
    if torch is None:
        raise RuntimeError("Torch is not available for compute_multitaper_spectrogram_batch_torch.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

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

    Nw = int(win_ms / 1000.0 * fs)
    if Nw % 2 == 0:
        Nw += 1
    Nw = max(Nw, 7)

    step = int(step_ms / 1000.0 * fs)
    step = max(step, 1)

    # Compute DPSS tapers on CPU and transfer to device
    tapers, _ = signal.windows.dpss(Nw, nw, k, return_ratios=True)
    tapers_torch = torch.tensor(np.ascontiguousarray(tapers), device=device, dtype=torch.float32)

    n_steps = (n_time - Nw) // step + 1
    window_starts = np.arange(n_steps) * step
    window_centers = window_starts + (Nw - 1) / 2.0
    t_out = window_centers / float(fs)

    n_fft = max(1024, _next_power_of_two(Nw * 2))
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0/fs)
    freq_indices = [np.argmin(np.abs(fft_freqs - f)) for f in foi]

    # Clean traces on CPU
    cleaned = []
    for i in range(n_signals):
        xi = interpolate_nans_1d(traces[i])
        if xi is None:
            xi = np.zeros(n_time, dtype=np.float32)
        cleaned.append(xi)
    traces_clean = np.stack(cleaned, axis=0).astype(np.float32)

    Sxx_all = np.empty((n_signals, len(foi), n_steps), dtype=np.float32)

    for start in range(0, n_signals, batch_size):
        stop = min(start + batch_size, n_signals)
        x_np = traces_clean[start:stop]
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)

        # Shape of windows: (batch_size, n_steps, Nw)
        windows = x.unfold(dimension=-1, size=Nw, step=step)
        # Apply tapers: (batch_size, K, n_steps, Nw)
        tapered = windows.unsqueeze(1) * tapers_torch.unsqueeze(0).unsqueeze(2)
        # FFT across time window dimension
        fft_res = torch.fft.rfft(tapered, n=n_fft, dim=-1)
        power = torch.abs(fft_res) ** 2
        spec_avg = power.mean(dim=1)  # average across K tapers -> (batch_size, n_steps, n_rfft_bins)

        spec_foi = spec_avg[:, :, freq_indices].permute(0, 2, 1)  # -> (batch_size, n_freqs, n_steps)
        Sxx_all[start:stop] = spec_foi.detach().cpu().numpy()

        del x, windows, tapered, fft_res, power, spec_avg, spec_foi
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return Sxx_all, foi.astype(np.float32), t_out.astype(np.float32)


def compute_multitaper_spectrogram(trace, fs, foi=None, win_ms=200, nw=3.0, k=5, step_ms=10, t_ms=None):
    """
    CPU sliding-window DPSS multitaper spectrogram for a single 1D trace.

    Parameters
    ----------
    trace : 1D array-like
        Raw LFP signal.
    fs : float
        Sampling frequency in Hz.
    foi : array-like or None
        Frequencies of interest.
    win_ms : float
        Window duration in milliseconds.
    nw : float
        Time-halfbandwidth product NW.
    k : int
        Number of DPSS tapers.
    step_ms : float
        Sliding step in milliseconds.
    t_ms : array-like or None
        Raw trace time axis in ms.

    Returns
    -------
    Sxx : np.ndarray, shape (n_freqs, n_steps)
    foi : np.ndarray
    t_sec : np.ndarray
    """
    if foi is None:
        foi = get_foi_with_notch_gaps()
    foi = np.asarray(foi, dtype=np.float32)

    trace_clean = interpolate_nans_1d(trace)
    if trace_clean is None:
        return None, None, None

    n_time = len(trace_clean)
    Nw = int(win_ms / 1000.0 * fs)
    if Nw % 2 == 0:
        Nw += 1
    Nw = max(Nw, 7)

    step = int(step_ms / 1000.0 * fs)
    step = max(step, 1)

    tapers, _ = signal.windows.dpss(Nw, nw, k, return_ratios=True)

    n_steps = (n_time - Nw) // step + 1
    window_starts = np.arange(n_steps) * step
    window_centers = window_starts + (Nw - 1) / 2.0
    t_sec = window_centers / float(fs)

    n_fft = max(1024, _next_power_of_two(Nw * 2))
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0/fs)
    freq_indices = [np.argmin(np.abs(fft_freqs - f)) for f in foi]

    Sxx = np.zeros((len(foi), n_steps), dtype=np.float64)

    for step_i, w_start in enumerate(window_starts):
        seg = trace_clean[w_start:w_start + Nw]
        tapered = seg[None, :] * tapers
        fft_res = np.fft.rfft(tapered, n=n_fft, axis=-1)
        power = np.abs(fft_res) ** 2
        spec_avg = power.mean(axis=0)
        Sxx[:, step_i] = spec_avg[freq_indices]

    return Sxx, foi, t_sec


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


def compute_causal_filterbank_spectrogram(trace, fs, foi=None, order=2, n_cycles=5, smooth_win_ms=None, t_ms=None):
    """
    Compute a strictly causal time-frequency power envelope using forward-only
    Butterworth bandpass filtering (scipy.signal.sosfilt) and analytic envelope extraction.
    
    Why this matches Morlet spectrogram aesthetics while preserving causality:
    - Bandwidth: Constant-Q scaling with frequency (f / n_cycles), matching Morlet frequency resolution.
    - Analytic Envelope: Computing the envelope |y(t) + i H{y(t)}|^2 eliminates carrier-wave
      intra-cycle oscillations (no zebra stripes or phase ripples), yielding a smooth scalogram like |W(t, f)|^2.
    - Strict Causality: Because sosfilt is forward-only, the filter state and analytic signal at t < 0
      depend ONLY on data before t = 0 ms. There is mathematically zero backward smearing or leakage
      from the stimulation artifact at t >= 0 into t < 0.
    
    Args:
        trace: 1D array (n_time,)
        fs: sampling rate (Hz)
        foi: array of frequencies (Hz)
        order: filter order (default 2 for sharp temporal resolution with minimal ringing)
        n_cycles: constant-Q quality factor matching Morlet cycles (default 5)
        smooth_win_ms: optional additional smoothing window in ms
        t_ms: time array in ms
    
    Returns:
        Sxx: (n_freq, n_time) causal instantaneous power envelope
        foi: frequency array
        t_sec: time array in seconds
    """
    if foi is None:
        foi = get_foi_with_notch_gaps()
    
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
    Sxx = np.zeros((n_freq, n_time), dtype=np.float64)
    
    for i, f in enumerate(foi):
        # Constant-Q bandwidth matching Morlet frequency scaling
        bw = max(1.5, f / float(n_cycles))
        low = max(0.5, f - bw / 2.0)
        high = min(fs / 2.0 - 1.0, f + bw / 2.0)
        
        if low >= high:
            continue
            
        sos = signal.butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
        # 1. Strictly causal forward-only filtering:
        filtered = signal.sosfilt(sos, trace_clean)
        # 2. Instantaneous analytic power envelope:
        analytic = signal.hilbert(filtered)
        envelope_power = np.abs(analytic) ** 2
        
        # 3. Frequency-matched causal exponential smoothing:
        # Matches the effective duration of Morlet wavelet tau(f) = n_cycles / (2*pi*f)
        # using a strictly causal 1-pole recursive exponential filter:
        # y[n] = alpha * y[n-1] + (1 - alpha) * x[n]
        if smooth_win_ms is not None and smooth_win_ms > 0:
            tau = float(smooth_win_ms) / 1000.0
        else:
            tau = float(n_cycles) / (2.0 * np.pi * f)
            
        alpha = np.exp(-1.0 / max(tau * fs, 1.0))
        zi = signal.lfilter_zi([1.0 - alpha], [1.0, -alpha]) * envelope_power[0]
        envelope_smooth, _ = signal.lfilter([1.0 - alpha], [1.0, -alpha], envelope_power, zi=zi)
            
        Sxx[i, :] = envelope_smooth
        
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
    elif method == 'multitaper':
        return compute_multitaper_spectrogram(
            trace, fs, foi=foi,
            win_ms=method_cfg.get('win_ms', 200),
            nw=method_cfg.get('nw', 3.0),
            k=method_cfg.get('k', 5),
            step_ms=method_cfg.get('step_ms', 10),
            t_ms=t_ms,
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
    elif method in ('causal_sos', 'causal_filterbank'):
        return compute_causal_filterbank_spectrogram(
            trace, fs, foi=foi,
            order=method_cfg.get('order', 2),
            smooth_win_ms=method_cfg.get('smooth_win_ms', 5.0),
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
            elif not ('baseline' in session_candidate_id.lower() or 'control' in session_candidate_id.lower()):
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


def run_multitaper_parameter_sweep(lfp_dir, fig_dir, config=None):
    """
    Run a parameter sweep of Multitaper Spectral Analysis across conditions (e.g. 11, 17, 18)
    and Utah Array regions (e.g. M1i, PMd, SMA, M1s).

    For each parameter configuration (window duration, NW, K tapers), generates a 3-panel figure:
      Panel 1: Condition 11 (1 pulse stim, 2.5 ms)
      Panel 2: Condition 17 (40 pulse stim, 100 ms)
      Panel 3: Condition 18 (Control reach, no stim)
    """
    if config is None:
        config = MULTITAPER_SWEEP_CONFIG

    target = config.get('target', 'target_A')
    conditions = config.get('conditions', [11, 17, 18])
    regions = config.get('regions', ['M1i', 'PMd'])
    param_configs = config.get('param_configs', [
        {'win_ms': 100, 'nw': 2.0, 'k': 3, 'label': 'Window = 100 ms | NW = 2.0 | K = 3 tapers'},
        {'win_ms': 150, 'nw': 2.0, 'k': 3, 'label': 'Window = 150 ms | NW = 2.0 | K = 3 tapers'},
        {'win_ms': 200, 'nw': 2.0, 'k': 3, 'label': 'Window = 200 ms | NW = 2.0 | K = 3 tapers'},
        {'win_ms': 200, 'nw': 3.0, 'k': 5, 'label': 'Window = 200 ms | NW = 3.0 | K = 5 tapers'},
        {'win_ms': 250, 'nw': 2.5, 'k': 4, 'label': 'Window = 250 ms | NW = 2.5 | K = 4 tapers'},
        {'win_ms': 300, 'nw': 3.0, 'k': 5, 'label': 'Window = 300 ms | NW = 3.0 | K = 5 tapers'},
    ])

    out_dir = fig_dir / config.get('output_subdir', 'multitaper_parameter_sweep')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MULTITAPER PARAMETER SWEEP MODE")
    print(f"  Target: {target}")
    print(f"  Conditions: {conditions}")
    print(f"  Regions: {regions}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # Locate condition files in lfp_dir
    cond_files = {}
    for f in lfp_dir.rglob("aligned_lfp__*.npz"):
        fname = f.name
        if target not in fname:
            continue
        for c in conditions:
            if c in (18, '18', 'control') and ('baseline' in fname.lower() or '_018_' in fname or '_18_' in fname):
                if 18 not in cond_files:
                    cond_files[18] = f
            elif f"_{int(c):03d}_" in fname or f"_{int(c)}_" in fname:
                cond_files[int(c)] = f

    missing = [c for c in conditions if c not in cond_files]
    if missing:
        print(f"  [Warn] Missing condition files for: {missing}; found: {list(cond_files.keys())}")
        if len(cond_files) < 2:
            print("  [Error] Need at least 2 conditions to run sweep comparison. Aborting.")
            return

    cond_data = {c: dict(np.load(p, allow_pickle=True)) for c, p in cond_files.items()}
    ref_data = next(iter(cond_data.values()))
    _, _, elec_to_idx, region_grids, _ = _build_groups(ref_data, 128)

    t_raw_ms = get_time_axis(ref_data)
    fs = int(ref_data.get('fs_lfp', 1000))
    foi = get_foi_with_notch_gaps()
    notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])
    baseline_win = ANALYSIS_CONFIG.get('baseline_window', (-950, -650))
    cmap = PLOT_CONFIG['spectrogram'].get('cmap', 'jet')
    normalize_method = config.get('normalize', 'dB')
    plot_difference = config.get('plot_difference', True)

    if normalize_method == 'dB':
        vmin = config.get('vmin_db', -6.0)
        vmax = config.get('vmax_db', 6.0)
        cbar_label = "Power (dB re baseline)"
    else:
        vmin = PLOT_CONFIG['spectrogram'].get('vmin_zscore', -3.5)
        vmax = PLOT_CONFIG['spectrogram'].get('vmax_zscore', 3.5)
        cbar_label = "Z-score Normalized Power (a.u.)"

    def extract_traces(d, r_name):
        bb = np.asarray(d['broadband_full'])
        grid = np.asarray(region_grids[r_name])
        tr_list = []
        for elec_id in grid.ravel():
            if elec_id is None or np.isnan(elec_id):
                continue
            ch_idx = elec_to_idx.get(int(elec_id), None)
            if ch_idx is None or ch_idx < 0 or ch_idx >= bb.shape[1]:
                continue
            for tr_idx in range(bb.shape[0]):
                sig = bb[tr_idx, ch_idx, :]
                if np.isfinite(sig).any():
                    clean = interpolate_nans_1d(sig)
                    if clean is not None:
                        tr_list.append(clean)
        return np.stack(tr_list, axis=0).astype(np.float32)

    active_conds = sorted(list(cond_data.keys()))

    for region_name in regions:
        if region_grids is None or region_name not in region_grids:
            continue
        print(f"\nProcessing region {region_name}...")
        traces_by_cond = {c: extract_traces(cond_data[c], region_name) for c in active_conds}

        for p_cfg in param_configs:
            w_ms = p_cfg['win_ms']
            nw = p_cfg['nw']
            k = p_cfg['k']
            lbl = p_cfg.get('label', f"Win={w_ms}ms, NW={nw}, K={k}")
            half_bw = nw / (w_ms / 1000.0)

            norm_by_cond = {}
            t_spec_ms = None
            for c in active_conds:
                if torch is not None and torch.cuda.is_available():
                    sxx, _, t_sec = compute_multitaper_spectrogram_batch_torch(
                        traces_by_cond[c], fs=fs, foi=foi, win_ms=w_ms, nw=nw, k=k, step_ms=10, device="cuda"
                    )
                else:
                    specs = [compute_multitaper_spectrogram(tr, fs=fs, foi=foi, win_ms=w_ms, nw=nw, k=k, step_ms=10)[0]
                             for tr in traces_by_cond[c]]
                    sxx = np.stack(specs, axis=0)
                    t_sec = np.arange(sxx.shape[-1]) * 0.010

                med_sxx = np.nanmedian(sxx, axis=0)
                if t_spec_ms is None:
                    t_spec_ms = t_sec * 1000.0 + float(t_raw_ms[0])
                bl_mask = (t_spec_ms >= baseline_win[0]) & (t_spec_ms <= baseline_win[1])
                if normalize_method == 'dB':
                    p_base = np.nanmean(med_sxx[:, bl_mask], axis=1, keepdims=True)
                    eps = 1e-12
                    norm_by_cond[c] = 10.0 * np.log10(np.maximum(med_sxx, eps) / np.maximum(p_base, eps))
                else:
                    norm_by_cond[c] = zscore_normalize_spectrogram(med_sxx, baseline_mask=bl_mask)

            has_diff = plot_difference and (18 in active_conds) and (17 in active_conds)
            n_cols = len(active_conds) + (1 if has_diff else 0)
            fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5.2),
                                     sharey=True, sharex=True, dpi=180)
            if n_cols == 1:
                axes = [axes]

            im_cbar = None
            for idx, c in enumerate(active_conds):
                ax = axes[idx]
                dur = 2.5 if c == 11 else (100.0 if c == 17 else 0.0)
                c_title = f"Condition {c} (1 Pulse Stim | 2.5 ms)" if c == 11 else (
                    f"Condition {c} (40 Pulse Stim | 100 ms)" if c == 17 else f"Condition {c} (Control Reach | No Stim)"
                )
                im = ax.pcolormesh(t_spec_ms, np.arange(len(foi)), norm_by_cond[c],
                                   cmap=cmap, vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
                im_cbar = im
                ax.axvline(0, color='white', ls='--', lw=1.5, alpha=0.9)
                if dur > 0:
                    ax.axvline(dur, color='white', ls=':', lw=1.2, alpha=0.7)
                    ax.axvspan(0, dur, color='#ffffff', alpha=0.15)
                for notch_lo, notch_hi in notch_regions:
                    b_idx = np.where(foi < notch_lo)[0]
                    a_idx = np.where(foi > notch_hi)[0]
                    if len(b_idx) > 0 and len(a_idx) > 0:
                        ax.axhline(b_idx[-1] + 0.5, color='white', ls='-', lw=1.2, alpha=0.8)
                tick_pos, tick_lbls = get_spectrogram_ticks(foi)
                ax.set_yticks(tick_pos)
                ax.set_yticklabels(tick_lbls)
                ax.set_title(c_title, fontsize=11, fontweight='bold', pad=8)
                ax.set_xlabel("Time from Event Onset (ms)", fontsize=10, fontweight='bold')
                ax.set_xlim(-500, 500)
                ax.grid(True, linestyle=":", alpha=0.3, color="white")

            im_diff = None
            if has_diff:
                ax_diff = axes[-1]
                diff_data = norm_by_cond[17] - norm_by_cond[18]
                vlim_diff = 6.0 if normalize_method == 'dB' else 4.0
                diff_lbl = "Excess Perturbation (dB)" if normalize_method == 'dB' else "Excess Z-score"
                im_diff = ax_diff.pcolormesh(t_spec_ms, np.arange(len(foi)), diff_data,
                                             cmap='RdBu_r', vmin=-vlim_diff, vmax=vlim_diff, shading='auto', rasterized=True)
                ax_diff.axvline(0, color='black', ls='--', lw=1.5, alpha=0.9)
                ax_diff.axvline(100.0, color='black', ls=':', lw=1.2, alpha=0.7)
                ax_diff.axvspan(0, 100.0, color='black', alpha=0.10)
                for notch_lo, notch_hi in notch_regions:
                    b_idx = np.where(foi < notch_lo)[0]
                    a_idx = np.where(foi > notch_hi)[0]
                    if len(b_idx) > 0 and len(a_idx) > 0:
                        ax_diff.axhline(b_idx[-1] + 0.5, color='black', ls='-', lw=1.2, alpha=0.8)
                ax_diff.set_title(f"STIM MINUS CONTROL (40p - Reach)\n{diff_lbl}", fontsize=11, fontweight='bold', pad=8, color='navy')
                ax_diff.set_xlabel("Time from Event Onset (ms)", fontsize=10, fontweight='bold')
                ax_diff.set_xlim(-500, 500)
                ax_diff.grid(True, linestyle=":", alpha=0.3, color="gray")

            axes[0].set_ylabel("Frequency (Hz)", fontsize=11, fontweight='bold')

            if has_diff and im_diff is not None:
                fig.subplots_adjust(top=0.78, bottom=0.15, left=0.05, right=0.93, wspace=0.14)
                cbar_ax1 = fig.add_axes([0.69, 0.82, 0.08, 0.025])
                cb1 = fig.colorbar(im_cbar, cax=cbar_ax1, orientation='horizontal')
                cb1.set_label(cbar_label, fontsize=9, fontweight='bold')
                cb1.ax.tick_params(labelsize=8)

                cbar_ax2 = fig.add_axes([0.83, 0.82, 0.09, 0.025])
                cb2 = fig.colorbar(im_diff, cax=cbar_ax2, orientation='horizontal')
                cb2.set_label("Stim - Control (dB)", fontsize=9, fontweight='bold')
                cb2.ax.tick_params(labelsize=8)
            else:
                fig.subplots_adjust(top=0.79, bottom=0.15, left=0.06, right=0.90, wspace=0.12)
                cbar_ax = fig.add_axes([0.915, 0.17, 0.015, 0.60])
                cb = fig.colorbar(im_cbar, cax=cbar_ax)
                cb.set_label(cbar_label, fontsize=11, fontweight='bold')

            fig.suptitle(
                f"Multitaper Spectrogram Exploration | Region: {region_name}\n"
                f"{lbl} | Frequency blur: plus or minus {half_bw:.1f} Hz | Smearing Bound: plus or minus {w_ms/2:.0f} ms | Artifact visible (unmasked)",
                fontsize=12, fontweight='bold', y=0.95
            )
            out_file = out_dir / f"multitaper_sweep_{region_name}_win{w_ms}ms_NW{nw}_K{k}.png"
            fig.savefig(out_file, dpi=180)
            plt.close(fig)
            print(f"  Saved: {out_file.name}")


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
    
    # Determine clean analysis windows (avoids blanking NaNs in NPRW)
    (pre_start, pre_end), (post_start, post_end) = get_clean_pre_post_windows(data, bb_full, t_ms)

    # Slice to analysis windows
    bb_pre, t_pre = slice_time_window(
        bb_full, t_ms,
        pre_start, pre_end
    )
    bb_post, t_post = slice_time_window(
        bb_full, t_ms,
        post_start, post_end
    )
    
    n_trials, n_ch = bb_full.shape[0], bb_full.shape[1]
    n_regions = len(groups)
    
    # Check if either pre or post has valid samples
    has_post = (bb_post is not None and bb_post.ndim == 3 and bb_post.shape[2] >= 8 and post_start < post_end)
    has_pre = (bb_pre is not None and bb_pre.ndim == 3 and bb_pre.shape[2] >= 8 and pre_start < pre_end)

    if not has_post and not has_pre:
        print(f"    [Skip] Insufficient time samples in both pre and post windows for session summary ({session_id})")
        return None

    # Create figure - single row of PSD plots
    fig, axes = plt.subplots(1, n_regions, figsize=(4 * n_regions, 4))
    if n_regions == 1:
        axes = [axes]
    
    fig.suptitle(f'Session Summary: {session_id}\n'
                 f'({n_trials} trials, {n_ch} channels, {n_regions} regions)',
                 fontsize=14, fontweight='bold')
    
    freq_range = ANALYSIS_CONFIG['freq_range_display']
    notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])
    n_time_post = bb_post.shape[2] if has_post else 0
    nperseg = min(n_time_post, 512) if has_post else 0
    
    # PSD for each region
    for col, (grp_idxs, grp_name) in enumerate(groups):
        ax = axes[col]
        short_name = grp_name.split(' (')[0]
        color = get_region_color(short_name)
        
        # Post-stim PSD (if post window is available and finite)
        if has_post:
            region_data = bb_post[:, grp_idxs, :].copy()

            # Guard against any sporadic NaNs
            if np.isnan(region_data).any():
                cleaned_list = []
                for tr_i in range(region_data.shape[0]):
                    ch_list = []
                    for ch_i in range(region_data.shape[1]):
                        tr = region_data[tr_i, ch_i]
                        if np.isnan(tr).any():
                            c_tr = interpolate_nans_1d(tr)
                            if c_tr is None:
                                c_tr = np.zeros(region_data.shape[2], dtype=np.float32)
                            ch_list.append(c_tr)
                        else:
                            ch_list.append(tr)
                    cleaned_list.append(np.stack(ch_list, axis=0))
                region_data = np.stack(cleaned_list, axis=0)

            f, Pxx = signal.welch(region_data, fs=fs, nperseg=nperseg, axis=2)
            f = np.asarray(f).ravel()
            
            # Average across channels, then compute stats across trials
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Pxx_ch_mean = np.nanmean(Pxx, axis=1)  # (trials, freq)
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
        else:
            # Indicate post is blanked / continuous stim
            ax.plot([], [], ' ', label='Post: N/A (continuous stim / blanked)')
        
        # Add pre-stim if available
        if has_pre:
            pre_data = bb_pre[:, grp_idxs, :].copy()
            if pre_data.shape[2] >= 8:
                if np.isnan(pre_data).any():
                    cleaned_list = []
                    for tr_i in range(pre_data.shape[0]):
                        ch_list = []
                        for ch_i in range(pre_data.shape[1]):
                            tr = pre_data[tr_i, ch_i]
                            if np.isnan(tr).any():
                                c_tr = interpolate_nans_1d(tr)
                                if c_tr is None:
                                    c_tr = np.zeros(pre_data.shape[2], dtype=np.float32)
                                ch_list.append(c_tr)
                            else:
                                ch_list.append(tr)
                        cleaned_list.append(np.stack(ch_list, axis=0))
                    pre_data = np.stack(cleaned_list, axis=0)

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
    
    # Slice to post-stim window for comparison (clean of any blanking NaNs)
    _, (post_start_stim, post_end_stim) = get_clean_pre_post_windows(data_stim, stim_full, t_stim)
    _, (post_start_ctrl, post_end_ctrl) = get_clean_pre_post_windows(data_control, ctrl_full, t_ctrl)
    post_start = max(post_start_stim, post_start_ctrl)
    post_end = min(post_end_stim, post_end_ctrl)

    if post_start >= post_end:
        print(f"    [Skip] Post-stim window has insufficient samples for stim vs control ({session_id})")
        return None

    bb_stim, _ = slice_time_window(stim_full, t_stim, post_start, post_end)
    bb_ctrl, _ = slice_time_window(ctrl_full, t_ctrl, post_start, post_end)
    
    if bb_stim.shape[2] < 8 or bb_ctrl.shape[2] < 8:
        print(f"    [Skip] Insufficient time points for stim vs control ({session_id})")
        return None
    
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
        if ctrl_data is not None and ctrl_data.ndim == 3 and ctrl_data.shape[2] >= 8:
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
                    _, (post_start, post_end) = get_clean_pre_post_windows(data, bb_full, t_ms)
                    bb_post, _ = slice_time_window(
                        bb_full, t_ms,
                        post_start, post_end
                    )
                except KeyError:
                    continue
                
                if bb_post.ndim != 3 or bb_post.shape[2] < 8 or post_start >= post_end:
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
# FIGURE 8: SPATIAL GIF & HEATMAPS
# =============================================================================

def compute_spatial_band_power(bb_full, t_ms, band_name, fs=1000, method='morlet'):
    """
    Compute time-varying band power (baseline-normalized dB) across all channels.

    Parameters
    ----------
    bb_full : np.ndarray
        Shape (n_trials, n_channels, n_times)
    t_ms : np.ndarray
        Time axis in ms, shape (n_times,)
    band_name : str
        Frequency band name (e.g. 'Beta', 'High Gamma')
    fs : float
        Sampling frequency in Hz
    method : str
        'morlet', 'multitaper', or 'hilbert'

    Returns
    -------
    band_dB : np.ndarray
        Baseline-normalized power in dB, shape (n_trials, n_channels, n_out_times)
    t_out_ms : np.ndarray
        Time axis in ms, shape (n_out_times,)
    """
    method = str(method).lower().strip()
    bands = ANALYSIS_CONFIG.get('bands', {})
    if band_name not in bands:
        raise ValueError(f"Band {band_name!r} not found in ANALYSIS_CONFIG['bands']: {list(bands.keys())}")
    
    f_lo, f_hi = bands[band_name]
    baseline_win = ANALYSIS_CONFIG.get('baseline_window', (-950, -650))
    nyq = fs / 2.0
    if f_hi >= nyq:
        f_hi = nyq - 1.0

    n_trials, n_channels, n_time = bb_full.shape

    if method == 'hilbert':
        # 4th-order Butterworth bandpass + Hilbert envelope
        b, a = signal.butter(4, [f_lo / nyq, f_hi / nyq], btype='band')
        filtered = signal.filtfilt(b, a, bb_full, axis=2)
        envelope = np.abs(signal.hilbert(filtered, axis=2))

        bl_mask = (t_ms >= baseline_win[0]) & (t_ms <= baseline_win[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            baseline_power = np.nanmean(envelope[:, :, bl_mask], axis=2, keepdims=True)
            baseline_power[baseline_power < 1e-10] = 1e-10
            band_dB = 10.0 * np.log10(np.maximum(envelope, 1e-10) / baseline_power)
        return band_dB, t_ms

    elif method in ('morlet', 'wavelet'):
        spec_cfg = PLOT_CONFIG.get('spectrogram', {})
        freq_step = spec_cfg.get('freq_step', 1)
        notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])

        foi = np.arange(f_lo, f_hi + 1, freq_step, dtype=np.float32)
        for n_lo, n_hi in notch_regions:
            foi = foi[(foi < n_lo) | (foi > n_hi)]
        if len(foi) == 0:
            foi = np.array([(f_lo + f_hi) / 2.0], dtype=np.float32)

        traces_flat = bb_full.reshape(n_trials * n_channels, n_time).astype(np.float32)

        use_gpu = (
            ANALYSIS_CONFIG.get('use_gpu_wavelet', False)
            and ANALYSIS_CONFIG.get('gpu_backend', 'torch').lower() == 'torch'
            and TORCH_AVAILABLE
            and torch is not None
            and torch.cuda.is_available()
        )
        gpu_batch_size = int(ANALYSIS_CONFIG.get('gpu_batch_size', 32))
        n_cycles = spec_cfg.get('wavelet_cycles', ANALYSIS_CONFIG.get('wavelet_cycles', 7))

        Sxx_flat = None
        if use_gpu:
            try:
                Sxx_flat, foi_out, _ = compute_wavelet_spectrogram_batch_torch(
                    traces_flat, fs=fs, foi=foi, n_cycles=n_cycles,
                    batch_size=gpu_batch_size, device="cuda"
                )
            except Exception as e:
                print(f"      [Warn] Torch GPU Morlet failed for spatial power ({e}); falling back to CPU.")
                Sxx_flat = None
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if Sxx_flat is None:
            specs = []
            for i in range(len(traces_flat)):
                sxx, _, _ = compute_wavelet_spectrogram(
                    traces_flat[i], fs=fs, foi=foi, n_cycles=n_cycles, normalize=False
                )
                if sxx is None:
                    sxx = np.zeros((len(foi), n_time), dtype=np.float32)
                specs.append(sxx)
            Sxx_flat = np.stack(specs, axis=0)

        # Sxx_flat shape: (n_trials * n_channels, len(foi), n_time)
        Sxx_4d = Sxx_flat.reshape(n_trials, n_channels, len(foi), n_time)
        band_power = np.nanmean(Sxx_4d, axis=2)  # (n_trials, n_channels, n_time)

        bl_mask = (t_ms >= baseline_win[0]) & (t_ms <= baseline_win[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            baseline_power = np.nanmean(band_power[:, :, bl_mask], axis=2, keepdims=True)
            baseline_power[baseline_power < 1e-10] = 1e-10
            band_dB = 10.0 * np.log10(np.maximum(band_power, 1e-10) / baseline_power)

        return band_dB, t_ms

    elif method == 'multitaper':
        spec_cfg = PLOT_CONFIG.get('spectrogram', {})
        freq_step = spec_cfg.get('freq_step', 1)
        notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])

        foi = np.arange(f_lo, f_hi + 1, freq_step, dtype=np.float32)
        for n_lo, n_hi in notch_regions:
            foi = foi[(foi < n_lo) | (foi > n_hi)]
        if len(foi) == 0:
            foi = np.array([(f_lo + f_hi) / 2.0], dtype=np.float32)

        traces_flat = bb_full.reshape(n_trials * n_channels, n_time).astype(np.float32)

        multitaper_win_ms = spec_cfg.get('multitaper_win_ms', 100)
        multitaper_nw = spec_cfg.get('multitaper_nw', 2.0)
        multitaper_k = spec_cfg.get('multitaper_k', 2)
        multitaper_step_ms = spec_cfg.get('multitaper_step_ms', 10)

        use_gpu = (
            ANALYSIS_CONFIG.get('use_gpu_wavelet', False)
            and ANALYSIS_CONFIG.get('gpu_backend', 'torch').lower() == 'torch'
            and TORCH_AVAILABLE
            and torch is not None
            and torch.cuda.is_available()
        )
        gpu_batch_size = int(ANALYSIS_CONFIG.get('gpu_batch_size', 32))

        Sxx_flat = None
        t_out_sec = None
        if use_gpu:
            try:
                Sxx_flat, foi_out, t_out_sec = compute_multitaper_spectrogram_batch_torch(
                    traces_flat, fs=fs, foi=foi, win_ms=multitaper_win_ms,
                    nw=multitaper_nw, k=multitaper_k, step_ms=multitaper_step_ms,
                    batch_size=gpu_batch_size, device="cuda"
                )
            except Exception as e:
                print(f"      [Warn] Torch GPU Multitaper failed for spatial power ({e}); falling back to CPU.")
                Sxx_flat = None
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if Sxx_flat is None:
            specs = []
            for i in range(len(traces_flat)):
                sxx, _, t_out_sec = compute_multitaper_spectrogram(
                    traces_flat[i], fs=fs, foi=foi, win_ms=multitaper_win_ms,
                    nw=multitaper_nw, k=multitaper_k, step_ms=multitaper_step_ms,
                    t_ms=t_ms
                )
                if sxx is None:
                    # fallback dummy
                    n_steps_approx = max(1, (n_time - int(multitaper_win_ms * fs / 1000.0)) // int(multitaper_step_ms * fs / 1000.0) + 1)
                    sxx = np.zeros((len(foi), n_steps_approx), dtype=np.float32)
                specs.append(sxx)
            Sxx_flat = np.stack(specs, axis=0)

        n_steps = Sxx_flat.shape[-1]
        if t_out_sec is None or len(t_out_sec) == 0:
            Nw = int(multitaper_win_ms / 1000.0 * fs)
            if Nw % 2 == 0: Nw += 1
            step = max(1, int(multitaper_step_ms / 1000.0 * fs))
            window_starts = np.arange(n_steps) * step
            window_centers = window_starts + (Nw - 1) / 2.0
            t_out_sec = window_centers / float(fs)

        t_spec_ms = np.asarray(t_out_sec, dtype=np.float32) * 1000.0 + float(t_ms[0])

        Sxx_4d = Sxx_flat.reshape(n_trials, n_channels, len(foi), n_steps)
        band_power = np.nanmean(Sxx_4d, axis=2)  # (n_trials, n_channels, n_steps)

        bl_mask = (t_spec_ms >= baseline_win[0]) & (t_spec_ms <= baseline_win[1])
        if not np.any(bl_mask):
            # Fallback to pre-stimulus window if default baseline_win is outside range
            bl_mask = (t_spec_ms < 0)
            if not np.any(bl_mask):
                bl_mask = np.zeros(n_steps, dtype=bool)
                bl_mask[:max(1, n_steps // 4)] = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            baseline_power = np.nanmean(band_power[:, :, bl_mask], axis=2, keepdims=True)
            baseline_power = np.nan_to_num(baseline_power, nan=1.0)
            baseline_power[baseline_power < 1e-10] = 1e-10
            band_dB = 10.0 * np.log10(np.maximum(band_power, 1e-10) / baseline_power)
            band_dB = np.nan_to_num(band_dB, nan=0.0)

        return band_dB, t_spec_ms

    else:
        raise ValueError(f"Unknown spatial band extraction method: {method!r}. Expected 'morlet', 'multitaper', or 'hilbert'.")


def render_spatial_grid(ax, mean_band_dB, t_bin_idx, grid_elec, elec_to_idx, 
                        vmin, vmax, smoothing=True):
    """
    Renders a spatial grid onto an axis, optionally with smoothing.
    """
    grid = np.full((8, 8), np.nan)
    points = []
    values = []
    cmap = PLOT_CONFIG.get('heatmap_cmap', 'jet')
    
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
        
        im = ax.imshow(filled_grid, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin='upper', aspect='equal', interpolation='bicubic')
    else:
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin='upper', aspect='equal')
    
    return im


def generate_spatial_gif(band_dB, t_ms, ua_ids_1based, session_id, fig_dir, 
                         groups, nsp_to_elec, elec_to_idx, region_grids,
                         band_name='Beta', method='morlet'):
    """
    Generate animated GIF of the spatial heatmap over time.
    Uses discrete time points from PLOT_CONFIG['spatial_timepoints_ms']
    so every single frame corresponds to a unique physiological state.
    """
    gif_cfg = ANALYSIS_CONFIG
    vmin = PLOT_CONFIG['heatmap_vmin']
    vmax = PLOT_CONFIG['heatmap_vmax']
    
    # 1. Use the curated discrete time points
    time_pts = PLOT_CONFIG.get('spatial_timepoints_ms', [
        -650, -600, -550, -500, -450, -400, -350, -300, -250, -200, -175, -150, -125, -100, -75, -50, -25, 
        0, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500
    ])
    
    n_trials = band_dB.shape[0] if band_dB.ndim == 3 else 1
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if band_dB.ndim == 3:
            mean_band_dB = np.nanmean(band_dB, axis=0)  # (n_ch, n_time)
        else:
            mean_band_dB = band_dB
    
    # Map each timepoint in time_pts to the nearest actual calculated spectral time bin in t_ms
    t_clean = np.asarray(t_ms, dtype=np.float64).ravel()
    t_bin_indices = [int(np.argmin(np.abs(t_clean - tp))) for tp in time_pts]
    print(f"    [Spatial GIF] Generating {len(time_pts)} discrete frames matching spatial_timepoints_ms ({time_pts[0]} to {time_pts[-1]} ms)")
    
    heatmap_dir = fig_dir / "Spatial_GIF"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    
    safe_band = str(band_name).replace(' ', '_')
    method_str = method.capitalize() if method else ""
    saved_paths = []
    
    for grp_idxs, grp_name in groups:
        region_name = grp_name.split(' (')[0]
        grid_elec = region_grids.get(region_name)
        
        if grid_elec is None or len(grp_idxs) == 0:
            continue
        
        safe_region = region_name.replace(' ', '_')
        
        # Determine modes to generate (blocky turned off for now per user request)
        modes = []
        if FIGURE_CONFIG.get('generate_spatial_gif_blocky', False):
            modes.append(('blocky', False))
        if FIGURE_CONFIG.get('generate_spatial_gif_smoothed', True):
            modes.append(('smoothed', True))
        if not modes:
            modes = [('smoothed', True)]
            
        for mode_label, use_smoothing in modes:
            frames = []
            print(f"      Generating GIF ({mode_label}, {method}) for {region_name} [{band_name}] ({len(time_pts)} frames)...")
            
            fig, ax = plt.subplots(figsize=(6, 5))
            try:
                fig.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.86)
                
                # Lookup clean anatomical Utah array display name matching LFP RSA style
                display_region = PLOT_CONFIG.get('region_display_names', {}).get(region_name, region_name)
                
                # Subdirectory for lossless individual PNG frames (immune to GIF palette compression / frame deduplication)
                frames_dir = heatmap_dir / "frames" / f"spatial_{safe_band}_{session_id}_{safe_region}_{mode_label}_{method}"
                frames_dir.mkdir(parents=True, exist_ok=True)
                
                for frame_idx, t_target in enumerate(time_pts):
                    ax.clear()
                    t_bin_idx = t_bin_indices[frame_idx]
                    im = render_spatial_grid(ax, mean_band_dB, t_bin_idx, grid_elec, 
                                            elec_to_idx, vmin, vmax, smoothing=use_smoothing)
                    
                    # Single prominent Utah array title matching LFP RSA naming style
                    ax.set_title(display_region, fontsize=18, fontweight='bold', pad=12, color='#0F172A')
                    ax.set_xticks(np.arange(8))
                    ax.set_yticks(np.arange(8))
                    ax.set_xticklabels(np.arange(1, 9), fontsize=8)
                    ax.set_yticklabels(np.arange(1, 9), fontsize=8)
                    ax.set_ylabel("Row", fontsize=10)
                    ax.set_xlabel("Col", fontsize=10)
                    
                    # Add colorbar only once
                    if not hasattr(fig, '_colorbar_added'):
                        cbar = fig.colorbar(im, ax=ax)
                        cbar.set_label("dB", fontsize=11, fontweight='bold')
                        cbar.ax.tick_params(labelsize=9)
                        fig._colorbar_added = True
                    
                    # Subtle invisible frame counter tag in corner to guarantee GIF encoders (Pillow)
                    # never collapse consecutive identical frames into one
                    ax.text(0, 0, f"f_{frame_idx:03d}", color='#ffffff', alpha=0.01, fontsize=1)
                    
                    # Convert to image
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=80)
                    buf.seek(0)
                    pil_frame = Image.open(buf).convert("RGBA")
                    
                    # Micro-tag extreme top-left pixel (0,0) in outer margin with frame_idx to guarantee
                    # Pillow GIF encoder NEVER merges/deduplicates consecutive identical frames
                    pil_frame.putpixel((0, 0), (frame_idx % 256, (frame_idx * 7) % 256, (frame_idx * 13) % 256, 255))
                    
                    # Save lossless numbered PNG frame for 100% synchronized Quad compositing
                    png_path = frames_dir / f"frame_{frame_idx:03d}_{int(t_target):+05d}ms.png"
                    pil_frame.save(png_path)
                    
                    frames.append(np.array(pil_frame))
            finally:
                plt.close(fig)
            
            if frames:
                out_path = heatmap_dir / f"spatial_{safe_band}_{session_id}_{safe_region}_{mode_label}_{method}.gif"
                fps = gif_cfg.get('gif_fps', 10)
                try:
                    if out_path.exists():
                        try:
                            out_path.unlink()
                        except Exception:
                            pass
                    imageio.mimsave(out_path, frames, fps=fps)
                except TypeError:
                    imageio.mimsave(out_path, frames, duration=1.0 / fps)
                print(f"      Saved spatial GIF ({len(frames)} frames) -> {out_path.name}")
                saved_paths.append(out_path)

    return saved_paths


def generate_spatial_heatmaps(band_dB, t_ms, ua_ids_1based, session_id, fig_dir,
                               groups, elec_to_idx, region_grids,
                               band_name='Beta', method='morlet'):
    """
    Generate static spatial heatmaps at key timepoints.
    """
    time_pts = PLOT_CONFIG.get('spatial_static_timepoints_ms', [-200, -50, 0, 50, 100, 200, 300, 500])
    vmin = PLOT_CONFIG['heatmap_vmin']
    vmax = PLOT_CONFIG['heatmap_vmax']
    
    n_trials = band_dB.shape[0] if band_dB.ndim == 3 else 1
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if band_dB.ndim == 3:
            mean_band_dB = np.nanmean(band_dB, axis=0)
        else:
            mean_band_dB = band_dB
    
    heatmap_dir = fig_dir / "Spatial_Heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    
    safe_band = str(band_name).replace(' ', '_')
    method_str = method.capitalize() if method else ""
    saved_paths = []
    
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
            
            sub_title = f" ({mode_label.capitalize()}, {method_str})" if method_str else f" ({mode_label.capitalize()})"
            fig.suptitle(f"{session_id} — {region_name}{sub_title}\n"
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
            
            out_path = heatmap_dir / f"spatial_{safe_band}_{session_id}_{safe_region}_{mode_label}_{method}.png"
            plt.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"      Saved spatial heatmap -> {out_path.name}")
            saved_paths.append(out_path)

    return saved_paths


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
    
    if PLOT_CONFIG.get('show_suptitle', False) and PLOT_CONFIG.get('show_titles', True):
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
                                  generate_trial_average=False,
                                  data_control=None):
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

        # === STIM MINUS CONTROL SPECTROGRAM GRID (8x8) ===
        if data_control is not None and FIGURE_CONFIG.get('generate_stim_minus_control_per_ua', True):
            try:
                bb_ctrl, t_ms_ctrl = get_broadband_full(data_control)
                n_trials_ctrl, n_ch_ctrl, _ = bb_ctrl.shape
                
                all_diff_spectrograms = {}
                for row in range(8):
                    for col in range(8):
                        elec_id = int(grid_elec[row, col])
                        data_idx = elec_to_idx.get(elec_id, None)
                        if data_idx is None or data_idx >= n_ch or data_idx >= n_ch_ctrl:
                            continue
                        
                        # Stim traces across trials
                        stim_specs = []
                        t_spec_saved = None
                        f_spec_saved = None
                        for t_idx in range(n_trials):
                            tr = bb_full[t_idx, data_idx, :]
                            if np.all(np.isnan(tr)) or np.std(tr) < 1e-10:
                                continue
                            Sxx, f_out, t_out = compute_wavelet_spectrogram(
                                tr, fs, foi=foi, baseline_window=None, t_ms=t_ms,
                                n_cycles=n_cycles, normalize=False
                            )
                            if Sxx is not None:
                                stim_specs.append(Sxx)
                                if t_spec_saved is None:
                                    t_spec_saved = t_out * 1000 + t_ms[0]
                                    f_spec_saved = f_out
                        
                        # Control traces across trials
                        ctrl_specs = []
                        for t_idx in range(n_trials_ctrl):
                            tr_c = bb_ctrl[t_idx, data_idx, :]
                            if np.all(np.isnan(tr_c)) or np.std(tr_c) < 1e-10:
                                continue
                            Sxx_c, _, _ = compute_wavelet_spectrogram(
                                tr_c, fs, foi=foi, baseline_window=None, t_ms=t_ms_ctrl,
                                n_cycles=n_cycles, normalize=False
                            )
                            if Sxx_c is not None:
                                ctrl_specs.append(Sxx_c)
                        
                        if stim_specs and ctrl_specs and t_spec_saved is not None:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                Sxx_stim_med = np.nanmedian(np.stack(stim_specs, axis=0), axis=0)
                                Sxx_ctrl_med = np.nanmedian(np.stack(ctrl_specs, axis=0), axis=0)
                            
                            bl_mask = (t_spec_saved >= baseline_win[0]) & (t_spec_saved <= baseline_win[1])
                            eps = 1e-12
                            p_base_stim = np.nanmean(Sxx_stim_med[:, bl_mask], axis=1, keepdims=True)
                            p_base_ctrl = np.nanmean(Sxx_ctrl_med[:, bl_mask], axis=1, keepdims=True)
                            
                            db_stim = 10.0 * np.log10(np.maximum(Sxx_stim_med, eps) / np.maximum(p_base_stim, eps))
                            db_ctrl = 10.0 * np.log10(np.maximum(Sxx_ctrl_med, eps) / np.maximum(p_base_ctrl, eps))
                            diff_db = db_stim - db_ctrl
                            
                            all_diff_spectrograms[(row, col)] = {
                                'Sxx': diff_db,
                                'f': f_spec_saved,
                                't_ms': t_spec_saved,
                                'elec_id': elec_id,
                                'n_trials_averaged': len(stim_specs),
                            }
                
                if all_diff_spectrograms:
                    _plot_spectrogram_grid(
                        all_diff_spectrograms, grid_elec, elec_to_idx, n_ch,
                        session_id, region_name, safe_region, condition_dir, fig_dir,
                        spec_cfg, normalize_method='dB', n_trials=n_trials,
                        trial_label=f"Stim minus Control (n_stim={n_trials}, n_ctrl={n_trials_ctrl})",
                        filename_suffix="spectrogram_grid_STIM_MINUS_CONTROL",
                        cmap_override='RdBu_r', vmin_override=-3.0, vmax_override=3.0,
                        cbar_label_override='Power (dB)'
                    )
                    print(f"      Saved Stim minus Control grid for {region_name} -> {condition_dir.name}/{safe_region}_spectrogram_grid_STIM_MINUS_CONTROL.png")
            except Exception as e:
                print(f"      [Warn] Could not compute stim minus control grid for {region_name}: {e}")
    
    return out_dir
    
def _plot_spectrogram_grid(all_spectrograms, grid_elec, elec_to_idx, n_ch,
                           session_id, region_name, safe_region, condition_dir, fig_dir,
                           spec_cfg, normalize_method, n_trials,
                           trial_label="", filename_suffix="",
                           cmap_override=None, vmin_override=None, vmax_override=None,
                           cbar_label_override=None):
    """
    Helper function to plot 8x8 spectrogram grid.
    
    Updated to handle z-score normalized data with appropriate color scaling,
    as well as custom difference (Stim minus Control) overrides.
    """
    if cmap_override is not None:
        cmap = cmap_override
        vmin = vmin_override if vmin_override is not None else -3.0
        vmax = vmax_override if vmax_override is not None else 3.0
        cbar_label = cbar_label_override or 'Power (dB)'
    elif normalize_method == 'zscore':
        vmin = spec_cfg.get('vmin_zscore', -3)
        vmax = spec_cfg.get('vmax_zscore', 3)
        cbar_label = 'Power (dB)'
        cmap = 'jet'  # Diverging colormap better for z-scores
    elif normalize_method == 'dB':
        vmin = spec_cfg.get('vmin_db', -3.0)
        vmax = spec_cfg.get('vmax_db', 3.0)
        cbar_label = 'Power (dB)'
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
    elif tfr_method == 'multitaper':
        win_ms = spec_cfg.get('multitaper_win_ms', 100)
        nw = spec_cfg.get('multitaper_nw', 2.0)
        k = spec_cfg.get('multitaper_k', 2)
        half_bw = nw / (win_ms / 1000.0)
        tfr_label = f"Multitaper (win={win_ms}ms, NW={nw}, K={k}, W=±{half_bw:.1f}Hz)"
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
                
                # Set y-ticks to show actual frequencies without overlapping notch boundaries
                tick_positions, tick_labels = get_spectrogram_ticks(f_spec)
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
                                             control_median_baseline_norm=None,
                                             data_control=None,
                                             return_raw_specs=False):
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
        return (None, None) if return_raw_specs else None

    if region_grids is None or elec_to_idx is None:
        print("    [Skip] Need electrode mapping for per-UA array median spectrogram")
        return (None, None) if return_raw_specs else None

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
        cbar_label = 'Power (dB)'
        cmap = 'jet'
    elif normalize_method == 'dB':
        vmin = spec_cfg.get('vmin_db', -3.0)
        vmax = spec_cfg.get('vmax_db', 3.0)
        cbar_label = 'Power (dB)'
        cmap = 'jet'
    else:
        vmin = spec_cfg.get('vmin_uV2', 0)
        vmax = spec_cfg.get('vmax_uV2', None)
        cbar_label = 'Power (dB)'

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
        return (None, None) if return_raw_specs else None

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
        return (None, None) if return_raw_specs else None

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

    # Usually 4 arrays: SMA, PMd, M1i, M1s (independent axes so right subplots don't clear left tick labels)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
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
    elif tfr_method == 'multitaper':
        win_ms = spec_cfg.get('multitaper_win_ms', 100)
        nw = spec_cfg.get('multitaper_nw', 2.0)
        k = spec_cfg.get('multitaper_k', 2)
        half_bw = nw / (win_ms / 1000.0)
        tfr_label = f"Multitaper (win={win_ms}ms, NW={nw}, K={k}, W=±{half_bw:.1f}Hz)"
    else:
        tfr_label = f"Morlet Wavelets ({spec_cfg.get('wavelet_cycles', 5)} cycles)"

    show_sup = PLOT_CONFIG.get('show_suptitle', False) and PLOT_CONFIG.get('show_titles', True)
    if show_sup:
        fig.suptitle(
            f'{session_id}\n'
            f'Array-Level Median Spectrograms | Median Across All Trials and Channels\n'
            f'{tfr_label} | {norm_str} | Notch regions skipped: {notch_str}',
            fontsize=14,
            fontweight='bold'
        )

    im_for_cbar = None

    # Keep plotting order: Top-Left: M1s, Top-Right: SMA, Bottom-Left: M1i, Bottom-Right: PMd
    plot_idx = 0
    ordered_groups = order_groups_preferred(groups)

    for grp_idxs, grp_name in ordered_groups:
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

        # Mark notch gaps
        for notch_lo, notch_hi in notch_regions:
            below_idx = np.where(f_spec < notch_lo)[0]
            above_idx = np.where(f_spec > notch_hi)[0]

            if len(below_idx) > 0 and len(above_idx) > 0:
                gap_y = below_idx[-1] + 0.5
                ax.axhline(gap_y, color='white', ls='-', lw=1.5, alpha=0.8)

        # Frequency tick labels (non-colliding, left column only)
        tick_positions, tick_labels = get_spectrogram_ticks(f_spec)
        ax.set_yticks(tick_positions)

        is_left_col = (plot_idx % 2 == 0)
        is_bottom_row = (plot_idx >= 2)

        if is_left_col:
            ax.set_yticklabels(tick_labels, fontsize=9)
            ax.set_ylabel('Frequency (Hz)', fontsize=10, fontweight='bold')
            ax.tick_params(axis='y', which='major', length=4, width=1.0, left=True, labelleft=True, labelsize=9)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='major', length=3, width=0.8, left=True, labelleft=False)

        if is_bottom_row:
            ax.set_xlabel('Time (ms)', fontsize=10, fontweight='bold')
            ax.tick_params(axis='x', which='major', length=4, width=1.0, bottom=True, labelbottom=True, labelsize=9)
        else:
            ax.set_xlabel('')
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='major', length=0, bottom=False, labelbottom=False)

        disp_title = PLOT_CONFIG.get('region_display_names', {}).get(region_name, region_name)
        reg_color = get_region_color(region_name)
        ax.set_title(disp_title, fontsize=12, fontweight='bold', color=reg_color, pad=8)

        plot_idx += 1

    # Hide unused axes if fewer than 4 regions
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout(rect=[0, 0, 0.88, 0.93 if show_sup else 0.98])

    # Colorbar centered and aligned with the active grid of subplots
    if im_for_cbar is not None:
        fig.subplots_adjust(right=0.88)
        active_axes = [a for a in axes[:plot_idx] if a.axison]
        if active_axes:
            plot_bottom = min(a.get_position().y0 for a in active_axes)
            plot_top = max(a.get_position().y1 for a in active_axes)
            cbar_height = plot_top - plot_bottom
            cbar_bottom = plot_bottom
        else:
            cbar_bottom = 0.18
            cbar_height = 0.65
        cbar_ax = fig.add_axes([0.90, cbar_bottom, 0.02, cbar_height])
        cbar = fig.colorbar(im_for_cbar, cax=cbar_ax)
        cbar.set_label('Power (dB)', fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

    out_path = condition_dir / "array_median_across_trials_and_channels.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"    Saved array-level median spectrogram -> {out_path.relative_to(fig_dir)}")

    # -------------------------------------------------------------------------
    # STIM MINUS MATCHED CONTROL SUBTRACTION (Excess Perturbation ΔdB)
    # Fully replaces the old control_median_baseline_norm
    # -------------------------------------------------------------------------
    ctrl_specs_raw = None
    if data_control is not None:
        print("      Computing array-level median spectrograms for matched control subtraction...")
        ctrl_specs_raw = compute_array_median_raw_spectrograms(
            data=data_control,
            groups=groups,
            fs=fs,
            region_grids=region_grids,
            elec_to_idx=elec_to_idx,
        )

    if ctrl_specs_raw is not None or control_median_baseline_norm is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        use_subtraction = (ctrl_specs_raw is not None)
        sub_title = (
            "Stim minus Matched Control Subtraction (Excess Perturbation)\n"
            f"{tfr_label} | ΔdB = dB(Stim) - dB(Control) | Notch regions skipped: {notch_str}"
            if use_subtraction else
            f"Control Median-Baseline Normalized\n{tfr_label} | Notch regions skipped: {notch_str}"
        )

        show_sup = PLOT_CONFIG.get('show_suptitle', False) and PLOT_CONFIG.get('show_titles', True)
        if show_sup:
            fig.suptitle(
                f'{session_id}\nArray-Level Median Spectrograms | {sub_title}',
                fontsize=13,
                fontweight='bold'
            )

        im_for_cbar = None
        plot_idx = 0
        ordered_groups = order_groups_preferred(groups)

        for grp_idxs, grp_name in ordered_groups:
            region_name = grp_name.split(' (')[0]

            if region_name not in region_specs_raw:
                continue
            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]
            spec = region_specs_raw[region_name]
            f_spec = spec['f']
            t_spec_ms = spec['t_ms']
            bl_mask = (t_spec_ms >= baseline_win[0]) & (t_spec_ms <= baseline_win[1])

            if use_subtraction and region_name in ctrl_specs_raw:
                spec_ctrl = ctrl_specs_raw[region_name]
                Sxx_stim = spec['Sxx']
                Sxx_ctrl = spec_ctrl['Sxx']

                eps = 1e-12
                p_base_stim = np.nanmean(Sxx_stim[:, bl_mask], axis=1, keepdims=True)
                p_base_ctrl = np.nanmean(Sxx_ctrl[:, bl_mask], axis=1, keepdims=True)

                db_stim = 10.0 * np.log10(np.maximum(Sxx_stim, eps) / np.maximum(p_base_stim, eps))
                db_ctrl = 10.0 * np.log10(np.maximum(Sxx_ctrl, eps) / np.maximum(p_base_ctrl, eps))
                diff_data = db_stim - db_ctrl

                plot_cmap = 'RdBu_r'
                plot_vmin, plot_vmax = -3.0, 3.0
                cbar_lbl = 'Power (dB)'

                im = ax.pcolormesh(
                    t_spec_ms,
                    np.arange(len(f_spec)),
                    diff_data,
                    cmap=plot_cmap,
                    vmin=plot_vmin,
                    vmax=plot_vmax,
                    shading='auto',
                    rasterized=True
                )
                im_for_cbar = im

                # Mark stim train with dark vertical lines and shading
                ax.axvline(0, color='black', ls='--', lw=1.2, alpha=0.9)
                ax.axvspan(0, 100, color='black', alpha=0.08)

                for notch_lo, notch_hi in notch_regions:
                    below_idx = np.where(f_spec < notch_lo)[0]
                    above_idx = np.where(f_spec > notch_hi)[0]
                    if len(below_idx) > 0 and len(above_idx) > 0:
                        gap_y = below_idx[-1] + 0.5
                        ax.axhline(gap_y, color='black', ls='-', lw=1.2, alpha=0.8)

            elif control_median_baseline_norm and region_name in control_median_baseline_norm:
                Sxx_raw = spec['Sxx']
                mean_val = control_median_baseline_norm[region_name]['mean']
                std_val = control_median_baseline_norm[region_name]['std']
                Sxx_ctrl = (Sxx_raw - mean_val) / std_val

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
                cbar_lbl = 'Power vs control median baseline (z)'

                ax.axvline(0, color='white', ls='--', lw=1.2, alpha=0.9)

                for notch_lo, notch_hi in notch_regions:
                    below_idx = np.where(f_spec < notch_lo)[0]
                    above_idx = np.where(f_spec > notch_hi)[0]
                    if len(below_idx) > 0 and len(above_idx) > 0:
                        gap_y = below_idx[-1] + 0.5
                        ax.axhline(gap_y, color='white', ls='-', lw=1.5, alpha=0.8)
            else:
                continue

            # Frequency tick labels (non-colliding, left column only)
            tick_positions, tick_labels = get_spectrogram_ticks(f_spec)
            ax.set_yticks(tick_positions)

            is_left_col = (plot_idx % 2 == 0)
            is_bottom_row = (plot_idx >= 2)

            if is_left_col:
                ax.set_yticklabels(tick_labels, fontsize=9)
                ax.set_ylabel('Frequency (Hz)', fontsize=10, fontweight='bold')
                ax.tick_params(axis='y', which='major', length=4, width=1.0, left=True, labelleft=True, labelsize=9)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel('')
                ax.tick_params(axis='y', which='major', length=3, width=0.8, left=True, labelleft=False)

            if is_bottom_row:
                ax.set_xlabel('Time (ms)', fontsize=10, fontweight='bold')
                ax.tick_params(axis='x', which='major', length=4, width=1.0, bottom=True, labelbottom=True, labelsize=9)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='major', length=0, bottom=False, labelbottom=False)

            disp_title = PLOT_CONFIG.get('region_display_names', {}).get(region_name, region_name)
            reg_color = get_region_color(region_name)
            ax.set_title(disp_title, fontsize=12, fontweight='bold', color=reg_color, pad=8)
            plot_idx += 1

        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout(rect=[0, 0, 0.88, 0.93 if show_sup else 0.98])

        if im_for_cbar is not None:
            fig.subplots_adjust(right=0.88)
            active_axes = [a for a in axes[:plot_idx] if a.axison]
            if active_axes:
                plot_bottom = min(a.get_position().y0 for a in active_axes)
                plot_top = max(a.get_position().y1 for a in active_axes)
                cbar_height = plot_top - plot_bottom
                cbar_bottom = plot_bottom
            else:
                cbar_bottom = 0.18
                cbar_height = 0.65
            cbar_ax = fig.add_axes([0.90, cbar_bottom, 0.02, cbar_height])
            cbar = fig.colorbar(im_for_cbar, cax=cbar_ax)
            cbar.set_label('Power (dB)', fontsize=11, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)

        # Save both files so CONTROL_MEDIAN_BASELINE_NORM is fully replaced by this subtraction
        out_path_sub = condition_dir / "array_median_across_trials_and_channels_STIM_MINUS_CONTROL.png"
        out_path_ctrl = condition_dir / "array_median_across_trials_and_channels_CONTROL_MEDIAN_BASELINE_NORM.png"
        plt.savefig(out_path_sub, dpi=150, bbox_inches='tight')
        plt.savefig(out_path_ctrl, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"    Saved Stim minus Control array-level median spectrogram -> {out_path_sub.relative_to(fig_dir)}")
        print(f"    Fully replaced CONTROL_MEDIAN_BASELINE_NORM -> {out_path_ctrl.relative_to(fig_dir)}")

    if return_raw_specs:
        return out_path, region_specs_raw
    return out_path


def generate_causal_diagnostic(data, session_id, fig_dir, groups, fs=1000,
                               region_grids=None, elec_to_idx=None,
                               zoom_range_ms=(0, 50), cached_morlet_raw=None):
    """
    Perform strictly causal time-frequency analysis matching regular array-level
    median spectrograms in frequencies, baseline normalization window, and channel/trial pooling,
    zoomed into the 0 to 50 ms post-stimulus window.

    Generates two figures:
    1. Standalone 2x2 Causal Array-Level Median Spectrogram (Zoomed 0 to 50 ms):
       Saved as 'array_median_causal_0_to_50ms.png' in per_UA_activity and Causal_Diagnostic.
    2. Causal Diagnostic Comparison Figure:
       Row 0: Acausal (Morlet Wavelet) [0, 50] ms
       Row 1: Strictly Causal (Forward-Only Filterbank) [0, 50] ms
       Saved as 'causal_diagnostic_{session_id}.png' in Causal_Diagnostic.
    """
    try:
        bb_full, t_ms = get_broadband_full(data)
    except KeyError as e:
        print(f"    [Skip] No broadband data for causal diagnostic: {e}")
        return None

    t_ms_arr = np.asarray(t_ms)
    zoom_mask = (t_ms_arr >= zoom_range_ms[0]) & (t_ms_arr <= zoom_range_ms[1])
    if not zoom_mask.any():
        print(f"    [Skip] Zoom window {zoom_range_ms} ms not within trial time array [{t_ms_arr[0]}, {t_ms_arr[-1]}]")
        return None

    # Discover condition name and directories
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

    out_dir_causal = fig_dir / "causal_diagnostic" / condition_name
    out_dir_causal.mkdir(parents=True, exist_ok=True)

    spec_cfg = PLOT_CONFIG.get('spectrogram', {})
    freq_min = spec_cfg.get('freq_min', 1)
    freq_max = spec_cfg.get('freq_max', 120)
    freq_step = spec_cfg.get('freq_step', 1)
    foi = get_foi_with_notch_gaps(freq_min, freq_max, freq_step)

    normalize_method = spec_cfg.get('normalize', 'zscore')
    baseline_win = ANALYSIS_CONFIG.get('baseline_window', (-950, -650))
    notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])
    notch_str = ', '.join([f'{lo}-{hi} Hz' for lo, hi in notch_regions])
    norm_str = "Z-scored" if normalize_method == 'zscore' else normalize_method.upper()

    cmap = spec_cfg.get('cmap', 'jet')
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

    # 1. Obtain raw Morlet spectrograms (reusing cached if passed)
    morlet_specs_raw = cached_morlet_raw
    if morlet_specs_raw is None:
        print("      Computing raw Morlet spectrograms for causal comparison...")
        morlet_specs_raw = compute_array_median_raw_spectrograms(
            data=data,
            groups=groups,
            fs=fs,
            region_grids=region_grids,
            elec_to_idx=elec_to_idx,
        )

    # 2. Compute strictly causal forward-only filterbank spectrograms
    print("      Computing strictly causal array-median spectrograms...")
    causal_specs_raw = compute_array_median_causal_raw_spectrograms(
        data=data,
        groups=groups,
        fs=fs,
        region_grids=region_grids,
        elec_to_idx=elec_to_idx,
        order=2,
        n_cycles=spec_cfg.get('wavelet_cycles', 5),
    )

    if not causal_specs_raw:
        print("    [Skip] No valid causal spectrograms were computed")
        return None

    # Normalization helper using identical baseline window and method
    def _normalize_dict(specs_raw_dict):
        normalized = {}
        for r_name, raw_spec in specs_raw_dict.items():
            Sxx_raw = raw_spec['Sxx'].copy()
            t_spec = raw_spec['t_ms']
            bl_m = (t_spec >= baseline_win[0]) & (t_spec <= baseline_win[1])
            if not bl_m.any():
                bl_m = (t_spec >= t_spec[0]) & (t_spec <= t_spec[0] + 300)

            if normalize_method == 'zscore':
                Sxx_norm = zscore_normalize_spectrogram(Sxx_raw, baseline_mask=bl_m)
            elif normalize_method == 'dB':
                bl_power = np.nanmean(Sxx_raw[:, bl_m], axis=1, keepdims=True)
                bl_power[bl_power < 1e-20] = 1e-20
                Sxx_norm = 10.0 * np.log10(np.maximum(Sxx_raw, 1e-20) / bl_power)
            else:
                Sxx_norm = Sxx_raw

            normalized[r_name] = {
                **raw_spec,
                'Sxx': Sxx_norm,
            }
        return normalized

    causal_specs_norm = _normalize_dict(causal_specs_raw)
    morlet_specs_norm = _normalize_dict(morlet_specs_raw) if morlet_specs_raw else None

    def _setup_spec_axes(ax, f_spec, is_bottom=False, is_first_col=False):
        ax.axvline(0, color='white', ls='--', lw=1.5, alpha=0.95)
        for notch_lo, notch_hi in notch_regions:
            below_idx = np.where(f_spec < notch_lo)[0]
            above_idx = np.where(f_spec > notch_hi)[0]
            if len(below_idx) > 0 and len(above_idx) > 0:
                gap_y = below_idx[-1] + 0.5
                ax.axhline(gap_y, color='white', ls='-', lw=1.5, alpha=0.8)

        tick_positions, tick_labels = get_spectrogram_ticks(f_spec)
        ax.set_yticks(tick_positions)
        if is_first_col:
            ax.set_yticklabels(tick_labels, fontsize=9)
            ax.set_ylabel('Frequency (Hz)', fontsize=10)
        else:
            ax.set_yticklabels([])

        ax.set_xlim(zoom_range_ms[0], zoom_range_ms[1])
        if is_bottom:
            ax.set_xlabel('Time relative to event onset (ms)', fontsize=10)
        ax.tick_params(axis='both', labelsize=9)

    # -------------------------------------------------------------------------
    # 1. Standalone 2x2 Array-Level Causal Spectrogram (Zoomed 0 to 50 ms)
    # -------------------------------------------------------------------------
    fig_causal, axes_causal = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes_causal = axes_causal.flatten()

    show_sup = PLOT_CONFIG.get('show_suptitle', False) and PLOT_CONFIG.get('show_titles', True)
    if show_sup:
        fig_causal.suptitle(
            f'{session_id}\n'
            f'Array-Level Causal Spectrograms | Median Across All Trials and Channels\n'
            f'Strictly Causal Filterbank (Forward-Only SOS) | {norm_str} | Zoom: [{zoom_range_ms[0]}, {zoom_range_ms[1]}] ms | Notch skipped: {notch_str}',
            fontsize=14,
            fontweight='bold'
        )

    im_cbar = None
    plot_idx = 0
    ordered_groups = order_groups_preferred(groups)

    for grp_idxs, grp_name in ordered_groups:
        region_name = grp_name.split(' (')[0]
        if region_name not in causal_specs_norm:
            continue
        if plot_idx >= len(axes_causal):
            break

        ax = axes_causal[plot_idx]
        spec = causal_specs_norm[region_name]

        Sxx = spec['Sxx']
        f_spec = spec['f']
        t_spec = spec['t_ms']

        z_mask = (t_spec >= zoom_range_ms[0]) & (t_spec <= zoom_range_ms[1])
        t_z = t_spec[z_mask]
        Sxx_z = Sxx[:, z_mask]

        im = ax.pcolormesh(
            t_z,
            np.arange(len(f_spec)),
            Sxx_z,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading='auto',
            rasterized=True
        )
        im_cbar = im

        _setup_spec_axes(ax, f_spec, is_bottom=(plot_idx >= 2), is_first_col=(plot_idx % 2 == 0))

        disp_title = PLOT_CONFIG.get('region_display_names', {}).get(region_name, region_name)
        reg_color = get_region_color(region_name)
        ax.set_title(
            f'{disp_title}\n'
            f'{spec["n_valid_channels"]} channels, '
            f'{spec["n_valid_trial_channel_pairs"]} trial-channel spectra',
            fontsize=11,
            fontweight='bold',
            color=reg_color,
        )
        plot_idx += 1

    for idx in range(plot_idx, len(axes_causal)):
        axes_causal[idx].axis('off')

    plt.tight_layout(rect=[0, 0, 0.88, 0.93 if show_sup else 0.98])

    if im_cbar is not None:
        fig_causal.subplots_adjust(right=0.88)
        active_axes = [a for a in axes_causal[:plot_idx] if a.axison]
        if active_axes:
            plot_bottom = min(a.get_position().y0 for a in active_axes)
            plot_top = max(a.get_position().y1 for a in active_axes)
            cbar_height = plot_top - plot_bottom
            cbar_bottom = plot_bottom
        else:
            cbar_bottom = 0.18
            cbar_height = 0.65
        cbar_ax = fig_causal.add_axes([0.90, cbar_bottom, 0.02, cbar_height])
        cbar = fig_causal.colorbar(im_cbar, cax=cbar_ax)
        cbar.set_label(cbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    path_causal_diag = out_dir_causal / "array_median_causal_0_to_50ms.png"
    plt.savefig(path_causal_diag, dpi=150, bbox_inches='tight')
    plt.close(fig_causal)

    print(f"    Saved causal array-level median spectrogram -> {path_causal_diag.relative_to(fig_dir)}")

    # -------------------------------------------------------------------------
    # 2. Causal Diagnostic Comparison Figure (Row 0: Morlet, Row 1: Causal)
    # -------------------------------------------------------------------------
    out_diag_path = out_dir_causal / f"causal_diagnostic_{session_id}.png"

    comparison_regions = []
    for grp_idxs, grp_name in groups:
        r_name = grp_name.split(' (')[0]
        if r_name in causal_specs_norm and (morlet_specs_norm is None or r_name in morlet_specs_norm):
            comparison_regions.append(r_name)

    comparison_regions = comparison_regions[:4]
    n_comp = len(comparison_regions)

    if n_comp > 0 and morlet_specs_norm is not None:
        fig_diag, axes_diag = plt.subplots(
            2, n_comp,
            figsize=(4.0 * n_comp, 7.5),
            sharex=True,
            sharey=True,
            squeeze=False
        )

        im_diag = None
        for col_idx, r_name in enumerate(comparison_regions):
            spec_m = morlet_specs_norm[r_name]
            spec_c = causal_specs_norm[r_name]

            f_spec = spec_c['f']
            t_spec = spec_c['t_ms']
            z_mask = (t_spec >= zoom_range_ms[0]) & (t_spec <= zoom_range_ms[1])
            t_z = t_spec[z_mask]

            # Row 0: Acausal (Morlet Wavelet)
            ax0 = axes_diag[0, col_idx]
            im0 = ax0.pcolormesh(
                t_z,
                np.arange(len(f_spec)),
                spec_m['Sxx'][:, z_mask],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading='auto',
                rasterized=True
            )
            im_diag = im0
            _setup_spec_axes(ax0, f_spec, is_bottom=False, is_first_col=(col_idx == 0))
            ax0.set_title(f"{r_name}\nAcausal (Morlet Wavelet)", fontsize=11, fontweight='bold')

            # Row 1: Strictly Causal (Forward-Only Filterbank)
            ax1 = axes_diag[1, col_idx]
            im1 = ax1.pcolormesh(
                t_z,
                np.arange(len(f_spec)),
                spec_c['Sxx'][:, z_mask],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading='auto',
                rasterized=True
            )
            _setup_spec_axes(ax1, f_spec, is_bottom=True, is_first_col=(col_idx == 0))
            ax1.set_title(f"{r_name}\nCausal (Forward-Only Filterbank)", fontsize=11, fontweight='bold')

        if im_diag is not None:
            fig_diag.subplots_adjust(right=0.89)
            cbar_ax = fig_diag.add_axes([0.91, 0.18, 0.015, 0.65])
            cbar = fig_diag.colorbar(im_diag, cax=cbar_ax)
            cbar.set_label(cbar_label, fontsize=11)
            cbar.ax.tick_params(labelsize=9)

        fig_diag.suptitle(
            f"{session_id} | Causal Diagnostic: Acausal (Morlet) vs Strictly Causal Filterbank\n"
            f"Zoomed post-stimulus onset window [{zoom_range_ms[0]}, {zoom_range_ms[1]}] ms | {norm_str} | Baseline [{baseline_win[0]}, {baseline_win[1]}] ms",
            fontsize=13,
            fontweight='bold'
        )

        plt.tight_layout(rect=[0, 0, 0.89, 0.92])
        plt.savefig(out_diag_path, dpi=180, bbox_inches='tight')
        plt.close(fig_diag)
        print(f"    Saved causal diagnostic -> {out_diag_path.relative_to(fig_dir)}")
        return out_diag_path

    return path_causal_per_ua

# Alias for backward compatibility
generate_causal_zoomed_spectrogram = generate_causal_diagnostic


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_session(data, session_id, fig_dir, groups, fs=1000,
                    data_control=None, data_rest=None,
                    session_data_dict=None,
                    nsp_to_elec=None, elec_to_idx=None, region_grids=None,
                    ua_ids_1based=None, control_median_baseline_norm=None,
                    control_spatial_baseline=None,
                    is_control=False, is_at_rest=False):
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
                data_control=data_control if not is_control else None,
            )
            if path:
                figure_paths.append(path)
        else:
            print("  [Skip] Per-UA spectrograms require electrode mapping")


    # 4c. Per-UA array-level median spectrogram
    # Median across all trials and all channels/electrodes within each Utah array.
    morlet_specs_raw = None
    if (FIGURE_CONFIG.get('generate_per_ua_array_median', False) or
            FIGURE_CONFIG.get('generate_control_median_baseline_norm_array_median', False) or
            FIGURE_CONFIG.get('generate_stim_minus_control_array_median', False)):
        if elec_to_idx is not None and region_grids is not None:
            print("  Generating per-UA array-level median spectrograms...")
            path, morlet_specs_raw = generate_per_ua_array_median_spectrogram(
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
                data_control=data_control if not is_control else None,
                return_raw_specs=True,
            )
            if path:
                figure_paths.append(path)
    
    # 4d. Causal diagnostic (0 to 50 ms)
    # Forward-only filtering diagnostic to compare causal vs acausal time-frequency representation.
    causal_diag_enabled = (FIGURE_CONFIG.get('generate_causal_diagnostic', False) or 
                          FIGURE_CONFIG.get('generate_causal_zoomed_spectrogram', False))
    if causal_diag_enabled and not is_control and region_grids is not None:
        print("  Generating causal diagnostic (0 to 50 ms)...")
        path = generate_causal_diagnostic(
            data=data,
            session_id=session_id,
            fig_dir=fig_dir,
            groups=groups,
            fs=fs,
            region_grids=region_grids,
            elec_to_idx=elec_to_idx,
            zoom_range_ms=(0, 50),
            cached_morlet_raw=morlet_specs_raw,
        )
        if path:
            figure_paths.append(path)
    
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
                # Determine bands and method for spatial visualizations
                spatial_bands = PLOT_CONFIG.get('spatial_bands') or ANALYSIS_CONFIG.get('spatial_bands')
                if spatial_bands is None:
                    single = PLOT_CONFIG.get('selected_band', 'Beta')
                    spatial_bands = [single] if isinstance(single, str) else list(single)
                elif isinstance(spatial_bands, str):
                    spatial_bands = [spatial_bands]

                spatial_method = PLOT_CONFIG.get('spatial_method') or ANALYSIS_CONFIG.get('spatial_method', 'morlet')
                spatial_method = str(spatial_method).lower().strip()

                for band_name in spatial_bands:
                    try:
                        print(f"  Computing spatial band power for {band_name} using {spatial_method.capitalize()}...")
                        band_dB, t_band_ms = compute_spatial_band_power(
                            bb_full=bb_full,
                            t_ms=t_ms,
                            band_name=band_name,
                            fs=fs,
                            method=spatial_method
                        )

                        # Generate normal spatial heatmaps
                        if FIGURE_CONFIG.get('generate_spatial_heatmaps', True):
                            print(f"  Generating spatial heatmaps ({band_name})...")
                            hm_paths = generate_spatial_heatmaps(
                                band_dB,
                                t_band_ms,
                                ua_ids_1based,
                                session_id,
                                fig_dir,
                                groups,
                                elec_to_idx,
                                region_grids,
                                band_name=band_name,
                                method=spatial_method
                            )
                            if hm_paths:
                                figure_paths.extend(hm_paths)

                        # Generate CONTROL-NORM spatial heatmaps
                        if (FIGURE_CONFIG.get('generate_control_norm_spatial_heatmaps', False)
                            and control_spatial_baseline is not None):
                            # Fallback envelope check if control baseline matches channel count
                            if control_spatial_baseline.shape[1] == band_dB.shape[1]:
                                eps = 1e-12
                                band_dB_control = 10 * np.log10(
                                    np.maximum(band_dB, eps) / np.maximum(control_spatial_baseline, eps)
                                )
                                print(f"  Generating CONTROL-NORM spatial heatmaps ({band_name})...")
                                c_paths = generate_spatial_heatmaps(
                                    band_dB_control,
                                    t_band_ms,
                                    ua_ids_1based,
                                    session_id + "_CONTROL_NORM",
                                    fig_dir,
                                    groups,
                                    elec_to_idx,
                                    region_grids,
                                    band_name=band_name,
                                    method=spatial_method
                                )
                                if c_paths:
                                    figure_paths.extend(c_paths)

                        # Generate spatial GIF
                        if FIGURE_CONFIG.get('generate_spatial_gif', True):
                            print(f"  Generating spatial GIF ({band_name})...")
                            gif_paths = generate_spatial_gif(
                                band_dB,
                                t_band_ms,
                                ua_ids_1based,
                                session_id,
                                fig_dir,
                                groups,
                                nsp_to_elec,
                                elec_to_idx,
                                region_grids,
                                band_name=band_name,
                                method=spatial_method
                            )
                            if gif_paths:
                                figure_paths.extend(gif_paths)

                    except Exception as e:
                        import traceback
                        print(f"    [Error] Spatial visualization failed for band {band_name}: {e}")
                        traceback.print_exc()
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
    at_rest_candidates_by_target = {}
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
            elif not ('baseline' in session_id.lower() or 'control' in session_id.lower()):
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
        
        # Ensure session_id carries canonical session/BR prefix for baseline/control files
        if not re.search(r'_(\d{3})(?:_|$|\.)', session_id):
            s_cand = None
            if 'sessions' in data and len(data['sessions']) > 0:
                s_cand = str(np.asarray(data['sessions']).ravel()[0])
            elif 'br_indices' in data and len(data['br_indices']) > 0:
                try:
                    b_val = int(np.asarray(data['br_indices']).ravel()[0])
                    s_cand = f"BR_{b_val:03d}"
                except Exception:
                    pass
            elif SESSION_FILTER is not None and 3 in SESSION_FILTER:
                s_cand = "BR_003"
            
            if s_cand:
                session_id = f"{s_cand}_{session_id}"
                session_id = re.sub(r'_+', '_', session_id)
        
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
        if 'continuous' in session_id.lower() or 'continuous_stim' in session_id.lower():
            is_control = False
        
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
        # Enforce SESSION_FILTER strictly: only generate figures/GIFs for requested conditions
        if SESSION_FILTER is not None:
            session_brs = [int(m) for m in re.findall(r'_(\d{3})(?:_|$|\.)', session_id)]
            
            data = info.get('data', {})
            if 'br_indices' in data:
                for b in np.asarray(data['br_indices']).ravel():
                    try:
                        session_brs.append(int(b))
                    except (ValueError, TypeError):
                        pass
            if 'sessions' in data:
                for s in np.asarray(data['sessions']).ravel():
                    m_s = re.search(r'_(\d{3})(?:_|$|\.)', str(s))
                    if m_s:
                        session_brs.append(int(m_s.group(1)))

            is_ctrl = info.get('is_control', False) or ('baseline' in session_id.lower()) or ('control' in session_id.lower())
            matches_br = any(b in SESSION_FILTER for b in session_brs)
            has_ctrl_in_filter = is_ctrl and any(x in SESSION_FILTER for x in [3, 0])

            if not (matches_br or has_ctrl_in_filter):
                continue

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
            
            # Append in layout order: Top-Left: M1s, Top-Right: SMA, Bottom-Left: M1i, Bottom-Right: PMd
            if m1s_idxs: groups.append((m1s_idxs, f"M1s (n={len(m1s_idxs)})"))
            if sma_idxs: groups.append((sma_idxs, f"SMA (n={len(sma_idxs)})"))
            if m1i_idxs: groups.append((m1i_idxs, f"M1i (n={len(m1i_idxs)})"))
            if pmd_idxs: groups.append((pmd_idxs, f"PMd (n={len(pmd_idxs)})"))
            
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
    # MULTITAPER PARAMETER SWEEP MODE
    # If MULTITAPER_SWEEP_CONFIG['enabled'] is True, run the parameter sweep
    # comparing conditions (e.g. 11, 17, 18) with 3-panel figures.
    # -------------------------------------------------------------------------
    if MULTITAPER_SWEEP_CONFIG.get('enabled', False):
        print("\n[MULTITAPER PARAMETER SWEEP MODE ACTIVE]")
        run_multitaper_parameter_sweep(UA_LFP_DIR, UA_FIG_DIR, MULTITAPER_SWEEP_CONFIG)
    # -------------------------------------------------------------------------
    # WAVELET COMPARISON MODE
    # If WAVELET_TEST_CONFIG['enabled'] is True, run the wavelet comparison
    # instead of (or in addition to) the normal pipeline.
    # Edit WAVELET_TEST_CONFIG at the top of this file to configure.
    # -------------------------------------------------------------------------
    elif WAVELET_TEST_CONFIG.get('enabled', False):
        print("\n[WAVELET TEST MODE ACTIVE]")
        run_wavelet_comparison(NPRW_LFP_DIR, NPRW_FIG_DIR, label="NPRW_Intan")
        run_wavelet_comparison(UA_LFP_DIR, UA_FIG_DIR, label="Utah_Array")
    else:
        # Normal pipeline
        process_directory(NPRW_LFP_DIR, NPRW_FIG_DIR, label="NPRW_Intan")
        process_directory(UA_LFP_DIR, UA_FIG_DIR, label="Utah_Array")