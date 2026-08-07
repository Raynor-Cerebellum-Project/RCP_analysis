"""
Configuration for plot_lfp_cleaner.py LFP analysis.

This module intentionally contains only configuration/constants and lightweight
optional dependency setup. It should not import plotting or analysis modules.
"""

import warnings

warnings.filterwarnings("ignore", message=".*tight_layout.*")


# =============================================================================
# SESSION SELECTION
# =============================================================================

SESSION_FILTER = [17, 1]
SESSION_ID_PATTERN = None


# =============================================================================
# OPTIONAL TORCH SUPPORT
# =============================================================================

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


# =============================================================================
# FIGURE CONFIGURATION
# =============================================================================

FIGURE_CONFIG = {
    "dpi": 300,
    "figsize_single": (10, 6),
    "figsize_grid": (16, 12),
    "figsize_wide": (18, 6),
    "cmap_diverging": "RdBu_r",
    "cmap_sequential": "viridis",
    "style": "seaborn-v0_8-whitegrid",
}


# =============================================================================
# TIME WINDOWS
# =============================================================================

TIME_WINDOWS = {
    "pre_stim": (-1.0, 0.0),
    "stim": (0.0, 0.1),
    "post_stim": (0.1, 1.0),
    "full": (-1.0, 1.0),
}


# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

ANALYSIS_CONFIG = {
    "baseline_window": (-0.5, -0.1),
    "stim_window": (0.0, 0.1),
    "post_stim_window": (0.1, 0.5),
    "freq_bands": {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "low_gamma": (30, 55),
        "high_gamma": (65, 120),
    },
    "notch_freqs": [60, 120, 180],
    "notch_width": 2,
}


# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

PLOT_CONFIG = {
    "generate_session_summary": False,
    "generate_waveform_raster": False,
    "generate_trial_heatmaps": False,
    "generate_per_ua_spectrograms": False,
    "generate_all_trial_spectrograms": False,
    "generate_per_ua_trial_average": False,
    "generate_per_ua_array_median": True,
    "generate_control_median_baseline_norm_array_median": True,
    "generate_self_baseline_minus_control_array_median": True,
    "generate_coherence_matrix": False,
    "generate_stim_vs_control": False,
    "generate_dose_response": False,
    "generate_spatial_heatmaps": False,
    "generate_spatial_gif": False,

    "spectrogram": {
        "tfr_method": "morlet_adaptive",

        "freq_min": 1,
        "freq_max": 120,
        "n_freqs": 80,

        "n_cycles": 5,

        "adaptive_cycles_low": 3,
        "adaptive_cycles_high": 7,
        "adaptive_freq_low_max": 15,
        "adaptive_freq_high_min": 16,

        "normalize": "zscore",
        "baseline_window": (-0.5, -0.1),

        "vmin": -3,
        "vmax": 3,

        "window_ms": 250,
        "step_ms": 10,

        "notch_freqs": [60],
        "notch_width": 2,
    },
}


# =============================================================================
# WAVELET / TFR DIAGNOSTIC CONFIGURATION
# =============================================================================

WAVELET_TEST_CONFIG = {
    "enabled": False,

    "test_region": "M1i",
    "output_subdir": "wavelet_comparison",

    "max_channels": None,
    "max_trials_per_channel": None,

    "freq_min": 1,
    "freq_max": 120,
    "n_freqs": 80,

    "normalize": "zscore",
    "baseline_window": (-0.5, -0.1),

    "methods": [
        {
            "label": "Morlet n=3",
            "method": "morlet",
            "n_cycles": 3,
            "enabled": True,
        },
        {
            "label": "Morlet n=5",
            "method": "morlet",
            "n_cycles": 5,
            "enabled": True,
        },
        {
            "label": "Morlet n=7",
            "method": "morlet",
            "n_cycles": 7,
            "enabled": True,
        },
        {
            "label": "Morlet n=9",
            "method": "morlet",
            "n_cycles": 9,
            "enabled": True,
        },
        {
            "label": "Morlet adaptive 3-7",
            "method": "morlet_adaptive",
            "adaptive_cycles_low": 3,
            "adaptive_cycles_high": 7,
            "adaptive_freq_low_max": 15,
            "adaptive_freq_high_min": 16,
            "enabled": True,
        },
        {
            "label": "Multitaper 300ms TW=3 K=5",
            "method": "multitaper",
            "time_bandwidth": 3,
            "n_tapers": 5,
            "window_ms": 300,
            "step_ms": 25,
            "enabled": True,
        },
        {
            "label": "Paul m=4",
            "method": "paul",
            "order": 4,
            "enabled": False,
        },
    ],
}