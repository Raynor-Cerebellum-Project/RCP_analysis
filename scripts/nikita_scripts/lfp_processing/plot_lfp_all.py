import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
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
from functools import lru_cache
import imageio
import io
import datetime

# Optional: progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# CONFIGURATION
# =============================================================================

FIGURE_CONFIG = {
    # Summary figures (recommended - fewer files, more comprehensive)
    'generate_session_summary': True,       # Combined PSD, ratio, power vs time, stats
    'generate_region_comparison': True,     # All regions side-by-side comparison
    'generate_spectrogram_grid': True,      # 8x8 spectrogram grid per region
    'generate_pdf_report': False,            # Multi-page PDF combining all figures
    
    # Animations
    'generate_spatial_gif': False,           # Animated spatial heatmaps
    
    # Individual detailed figures (for deep dives - off by default)
    'detailed_heatmaps': False,             # Original per-band heatmap traces
    'detailed_single_channel': False,       # Single channel multi-band traces
    'detailed_coherence': False,            # Inter-array coherence matrices
    'detailed_spectrograms': False,         # Individual tall spectrogram figures
    'detailed_band_overlays': False,        # Per-trial band power overlays
    'detailed_spatial_heatmaps': False,     # Static spatial heatmaps at key times
}

PLOT_CONFIG = {
    "single_channel_target": 168,
    "trace_alpha": 0.4,
    "heatmap_cmap": "RdBu_r",
    
    "scales": {
        'broadband': 30.0,
        'delta': None,
        'theta': 100.0,
        'alpha': 50.0,
        'beta': 50.0,
        'low_gamma': 30.0,
        'high_gamma': 30.0
    },
    
    "trace_scales": {
        'broadband': 200,
        'delta': 100,
        'theta': 100,
        'alpha': 100,
        'beta': 100,
        'low_gamma': 100,
        'high_gamma': 100
    },

    "x_limits": {
        "pre": (-400, -5),
        "post": (101, 600)
    },

    "spectrogram": {
        "stft_nperseg": 258,
        "stft_overlap_frac": 0.98,
        "window_type": "hann",
        "freq_min": 0.5,
        "freq_max": 60,
        "log_freq": False,
        "vmin_db": -6,
        "vmax_db": 6,
        "time_range": (-500, 500),
        "baseline_window": (-950, -650),
        "normalization": "baseline",
        "per_trial_norm": True,
        "cmap": "jet",
        "shading": "nearest",
        "dpi": 150,
    },

    "band_plots": {
        "selected_band": "Theta",
        "time_points_ms": [-400, -200, -50, 10, 50, 100, 150, 300],
        "overlay_alpha": 0.3,
        "heatmap_vmin": -6.0,
        "heatmap_vmax": 6.0,
        "gif_time_range": (-500, 500),
        "gif_step_ms": 10,
    },
    
    # Summary figure settings
    "summary": {
        "figsize": (24, 20),
        "dpi": 150,
    },
    
    # PDF settings
    "pdf": {
        "dpi": 150,
    }
}

BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 25),
    'Low Gamma': (25, 60),
    'High Gamma': (60, 120)
}

BAND_COLORS = {
    'Delta': '#E91E63',
    'Theta': '#9C27B0',
    'Alpha': '#4CAF50',
    'Beta': '#FFEB3B',
    'Low Gamma': '#FF9800',
    'High Gamma': '#F44336'
}

REGION_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


# =============================================================================
# SETUP / GLOBALS
# =============================================================================

from RCP_analysis.python.functions.br_preproc import (
    load_electrode_mapping,
    get_region_from_group_name as _region_from_group_name,
    get_region_grid,
    build_elec_to_data_idx
)

MAPPING_CSV = Path(__file__).parent.parent.parent / "electrode_port_mapping.csv"
nsp_to_elec_global, UTAH_ELEC_GRIDS, elec_to_region_global = load_electrode_mapping(MAPPING_CSV)

def _region_grid(region_name: str):
    return get_region_grid(region_name, UTAH_ELEC_GRIDS)

NPRW_LFP_DIR = cfg.OUT_BASE / "checkpoints" / "NPRW_LFP"
UA_LFP_DIR = cfg.OUT_BASE / "checkpoints" / "UA_LFP"
NPRW_FIG_DIR = cfg.OUT_BASE / "figures" / "NPRW_LFP"
UA_FIG_DIR = cfg.OUT_BASE / "figures" / "UA_LFP"

NPRW_FIG_DIR.mkdir(parents=True, exist_ok=True)
UA_FIG_DIR.mkdir(parents=True, exist_ok=True)

PROBE_MAP_FILE = cfg.REPO_ROOT / "config" / "ImecPrimateStimRec128_BT_corrected_091525.mat"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def add_colorbar(fig, im, ax, label="", fraction=0.046, pad=0.04, fontsize=9):
    """Consistent colorbar styling."""
    cb = fig.colorbar(im, ax=ax, fraction=fraction, pad=pad)
    cb.set_label(label, fontsize=fontsize)
    cb.ax.tick_params(labelsize=max(6, fontsize-2))
    return cb


def add_band_shading(ax, alpha=0.1):
    """Add vertical band shading to frequency plots."""
    for band_name, (f_lo, f_hi) in BANDS.items():
        color = BAND_COLORS.get(band_name, 'gray')
        ax.axvspan(f_lo, f_hi, color=color, alpha=alpha)


def get_representative_channel(data_array):
    """Find a channel index that has valid, non-flat data."""
    if data_array is None:
        return 0
    n_ch = data_array.shape[1]
    mid = n_ch // 2
    
    for offset in range(n_ch):
        for sign in [1, -1]:
            ch = mid + sign * offset
            if 0 <= ch < n_ch:
                traces = data_array[:, ch, :]
                if not np.all(np.isnan(traces)) and np.nanstd(traces) > 1e-6:
                    return ch
    return mid


@lru_cache(maxsize=32)
def get_dpss_windows(nperseg, NW=4):
    """Cache DPSS windows to avoid recomputation."""
    from scipy.signal.windows import dpss
    Kmax = 2 * NW - 1
    return tuple(dpss(nperseg, NW, Kmax))


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def calculate_psd(data, fs, use_multitaper=True):
    """
    Calculate Power Spectral Density averaged across trials.
    Returns: freq, psd_mean (n_ch, n_freq), psd_sem (n_ch, n_freq) in dB
    """
    n_trials, n_ch, n_time = data.shape
    nperseg = min(n_time, 256)
    
    if use_multitaper:
        windows = get_dpss_windows(nperseg)
        all_psd = []
        for win in windows:
            f, Pxx = signal.welch(data, fs=fs, nperseg=nperseg, window=win, axis=2)
            all_psd.append(Pxx)
        Pxx = np.mean(all_psd, axis=0)
    else:
        f, Pxx = signal.welch(data, fs=fs, nperseg=nperseg, axis=2)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Pxx_dB = 10 * np.log10(Pxx + 1e-20)
        psd_mean = np.nanmean(Pxx_dB, axis=0)
        psd_sem = np.nanstd(Pxx_dB, axis=0) / np.sqrt(n_trials)
    
    return f, psd_mean, psd_sem


def calculate_band_power_stats(data_pre, data_post, fs):
    """Compare band power between Pre and Post using Paired T-Test."""
    results = {}
    nperseg = min(data_pre.shape[2], 256)
    f, Pxx_pre = signal.welch(data_pre, fs=fs, nperseg=nperseg, axis=2)
    _, Pxx_post = signal.welch(data_post, fs=fs, nperseg=nperseg, axis=2)
    
    for band_name, (f_min, f_max) in BANDS.items():
        idx_band = np.where((f >= f_min) & (f <= f_max))[0]
        if len(idx_band) == 0:
            results[band_name] = None
            continue
            
        pow_pre = np.nanmean(Pxx_pre[:, :, idx_band], axis=2)
        pow_post = np.nanmean(Pxx_post[:, :, idx_band], axis=2)
        t_stats, p_vals = stats.ttest_rel(pow_pre, pow_post, axis=0, nan_policy='omit')
        
        results[band_name] = {
            't_stat': t_stats,
            'p_val': p_vals,
            'mean_pre': np.nanmean(pow_pre, axis=0),
            'mean_post': np.nanmean(pow_post, axis=0),
            'sem_pre': np.nanstd(pow_pre, axis=0) / np.sqrt(pow_pre.shape[0]),
            'sem_post': np.nanstd(pow_post, axis=0) / np.sqrt(pow_post.shape[0])
        }
    return results


def compute_ersp_spectrogram(broadband_full, t_full_ms, fs, spec_cfg):
    """Compute per-channel ERSP spectrogram from broadband_full data."""
    return_power_trials = spec_cfg.get('return_power_trials', False)
    nperseg = spec_cfg['stft_nperseg']
    noverlap = int(nperseg * spec_cfg['stft_overlap_frac'])
    window_type = spec_cfg.get('window_type', 'hann')
    freq_min = spec_cfg.get('freq_min', 0)
    freq_max = spec_cfg['freq_max']
    t_range = spec_cfg['time_range']
    per_trial_norm = spec_cfg.get('per_trial_norm', False)
    
    n_trials, n_ch, n_time = broadband_full.shape
    
    f, t_bins, Zxx = signal.stft(broadband_full, fs=fs, nperseg=nperseg,
                                  noverlap=noverlap, window=window_type, axis=2)
    P = np.abs(Zxx) ** 2
    t_bins_ms = t_bins * 1000.0 + t_full_ms[0]
    
    # Baseline
    bl_win = spec_cfg.get('baseline_window', (None, 0))
    bl_start = bl_win[0] if bl_win[0] is not None else t_bins_ms[0]
    bl_end = bl_win[1] if bl_win[1] is not None else 0
    baseline_mask = (t_bins_ms >= bl_start) & (t_bins_ms <= bl_end)
    
    if not np.any(baseline_mask):
        n_bl = max(1, len(t_bins_ms) // 4)
        baseline_mask = np.zeros(len(t_bins_ms), dtype=bool)
        baseline_mask[:n_bl] = True
    
    # Frequency crop
    freq_mask = (f >= freq_min) & (f <= freq_max)
    freqs = f[freq_mask]
    P = P[:, :, freq_mask, :]
    
    # Normalize
    normalization = spec_cfg.get('normalization', 'baseline')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        if normalization == 'absolute':
            P_mean = np.nanmean(P, axis=0)
            P_mean[P_mean == 0] = 1e-10
            ersp = 10.0 * np.log10(P_mean)
        elif per_trial_norm:
            bl_power = np.nanmean(P[:, :, :, baseline_mask], axis=3)
            bl_power[bl_power == 0] = 1e-10
            ersp_trials = 10.0 * np.log10(P / bl_power[:, :, :, None])
            ersp = np.nanmean(ersp_trials, axis=0)
        else:
            P_mean = np.nanmean(P, axis=0)
            baseline_power = np.nanmean(P_mean[:, :, baseline_mask], axis=2)
            baseline_power[baseline_power == 0] = 1e-10
            ersp = 10.0 * np.log10(P_mean / baseline_power[:, :, None])
    
    if return_power_trials:
        return ersp, freqs, t_bins_ms, P
    else:
        time_mask = (t_bins_ms >= t_range[0]) & (t_bins_ms <= t_range[1])
        return ersp[:, :, time_mask], freqs, t_bins_ms[time_mask]


def extract_band_power_trials(P, f, band_name, baseline_mask):
    """Extract per-trial band power over time in dB relative to baseline."""
    band_name_lower = band_name.lower()
    bands_lower = {k.lower(): v for k, v in BANDS.items()}
    
    if band_name_lower not in bands_lower:
        return None
    
    freq_min, freq_max = bands_lower[band_name_lower]
    f_mask = (f >= freq_min) & (f <= freq_max)
    
    if not np.any(f_mask):
        return None
    
    band_P = np.nanmean(P[:, :, f_mask, :], axis=2)
    
    if not np.any(baseline_mask):
        n_bl = max(1, band_P.shape[2] // 4)
        baseline_mask = np.zeros(band_P.shape[2], dtype=bool)
        baseline_mask[:n_bl] = True
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bl_power = np.nanmean(band_P[:, :, baseline_mask], axis=2, keepdims=True)
        bl_power[bl_power == 0] = 1e-10
        band_dB = 10.0 * np.log10(band_P / bl_power)
    
    return band_dB


def compute_band_power_timecourse(data, fs, t_axis, groups):
    """
    Compute band power time courses for all bands and groups.
    Returns dict: {band_name: {group_name: (mean, sem, t_plot)}}
    """
    results = {}
    
    for band_name, (f_lo, f_hi) in BANDS.items():
        # Bandpass filter
        sos = signal.butter(4, [f_lo, f_hi], btype='band', fs=fs, output='sos')
        filtered = signal.sosfiltfilt(sos, data, axis=2)
        
        # Envelope via Hilbert
        n_trials = data.shape[0]
        env_all = np.empty_like(filtered, dtype=np.float32)
        pad_len = min(200, filtered.shape[2] // 2)
        
        for tr_i in range(n_trials):
            padded = np.pad(filtered[tr_i], ((0, 0), (pad_len, pad_len)), mode='reflect')
            analytic = signal.hilbert(padded, axis=1)
            env_all[tr_i] = np.abs(analytic[:, pad_len:-pad_len if pad_len > 0 else None])**2
        
        results[band_name] = {}
        
        for grp_idxs, grp_name in groups:
            valid_idxs = [idx for idx in grp_idxs if idx < env_all.shape[1]]
            if not valid_idxs:
                continue
            
            grp_env = env_all[:, valid_idxs, :]
            grp_mean_ch = np.nanmean(grp_env, axis=1)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                trial_mean = np.nanmean(grp_mean_ch, axis=0)
                trial_sem = np.nanstd(grp_mean_ch, axis=0) / np.sqrt(n_trials)
            
            # Time axis
            n_points = trial_mean.shape[0]
            if t_axis is not None and len(t_axis) == n_points:
                t_plot = t_axis
            elif t_axis is not None and len(t_axis) >= 2:
                t_plot = np.linspace(t_axis[0], t_axis[-1], n_points)
            else:
                t_plot = np.arange(n_points) * (1000.0 / fs)
            
            short_name = grp_name.split(' (')[0]
            results[band_name][short_name] = (trial_mean, trial_sem, t_plot)
    
    return results


# =============================================================================
# PLOTTING HELPER FUNCTIONS
# =============================================================================

def render_spatial_grid(ax, mean_band_dB, t_bin_idx, grid_elec, elec_to_idx, vmin, vmax, smoothing=False):
    """Render spatial grid onto axis with optional smoothing."""
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
        filled_grid = np.copy(grid)
        mask = np.isnan(filled_grid)
        if len(values) > 0:
            filled_grid[mask] = np.nanmean(np.asarray(values))
        
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
        
        im = ax.imshow(filled_grid, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                       origin='upper', aspect='equal', interpolation='bicubic')
    else:
        im = ax.imshow(grid, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                       origin='upper', aspect='equal')
    
    return im


# =============================================================================
# SUMMARY FIGURE GENERATORS
# =============================================================================

def generate_session_summary(data, session_id, title_str, fig_dir, groups, fs, config):
    """
    Generate a comprehensive summary figure combining:
    - PSD (pre vs post) per group
    - PSD Ratio per group
    - Power vs Time (all groups overlaid)
    - Band Stats bar charts per group
    """
    bb_pre = data.get('broadband_pre')
    bb_post = data.get('broadband_post')
    
    if bb_pre is None or bb_post is None:
        print("    [Skip] Missing pre/post data for session summary")
        return None
    
    n_groups = len(groups)
    if n_groups == 0:
        return None
    
    # Calculate PSDs
    f_pre, psd_pre_all, _ = calculate_psd(bb_pre, fs)
    f_post, psd_post_all, _ = calculate_psd(bb_post, fs)
    
    # Calculate band stats
    band_stats = calculate_band_power_stats(bb_pre, bb_post, fs)
    
    # Create figure
    fig = plt.figure(figsize=(6 * n_groups, 16), constrained_layout=True)
    gs = fig.add_gridspec(4, n_groups, height_ratios=[1, 1, 1.5, 1])
    
    fig.suptitle(f"Session Summary: {session_id}\n{title_str}", fontsize=14, fontweight='bold')
    
    # Row 0: PSD per group
    for i, (grp_idxs, grp_name) in enumerate(groups):
        ax = fig.add_subplot(gs[0, i])
        short_name = grp_name.split(' (')[0]
        
        g_pre = np.nanmean(psd_pre_all[grp_idxs, :], axis=0)
        g_post = np.nanmean(psd_post_all[grp_idxs, :], axis=0)
        
        ax.plot(f_pre, g_pre, color='blue', lw=2, label='Pre')
        ax.plot(f_post, g_post, color='red', lw=2, label='Post')
        
        add_band_shading(ax)
        ax.set_title(f"{short_name}\nPSD", fontsize=10)
        ax.set_xlim(0, 120)
        ax.set_ylim(-20, 40)
        ax.set_ylabel("Power (dB)" if i == 0 else "")
        ax.set_xlabel("Freq (Hz)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize='small')
    
    # Row 1: PSD Ratio per group
    for i, (grp_idxs, grp_name) in enumerate(groups):
        ax = fig.add_subplot(gs[1, i])
        short_name = grp_name.split(' (')[0]
        
        g_pre = np.nanmean(psd_pre_all[grp_idxs, :], axis=0)
        g_post = np.nanmean(psd_post_all[grp_idxs, :], axis=0)
        g_pre[g_pre == 0] = 1e-10
        ratio_db = g_post - g_pre  # Already in dB, so subtract
        
        ax.plot(f_pre, ratio_db, color='k', lw=2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
        
        add_band_shading(ax)
        ax.set_title(f"{short_name}\nPSD Change", fontsize=10)
        ax.set_xlim(0, 120)
        ax.set_ylim(-10, 10)
        ax.set_ylabel("ΔdB" if i == 0 else "")
        ax.set_xlabel("Freq (Hz)")
        ax.grid(True, alpha=0.3)
    
    # Row 2: Power vs Time (all groups overlaid per band)
    # Use broadband_full if available
    d_full = data.get('broadband_full')
    if d_full is not None:
        t_pre = data.get('rel_time_pre')
        t_post = data.get('rel_time_post')
        if t_pre is not None and t_post is not None:
            t_axis = np.linspace(t_pre[0], t_post[-1], d_full.shape[2])
        else:
            t_axis = np.arange(d_full.shape[2]) * 1000.0 / fs - 500
        
        band_timecourses = compute_band_power_timecourse(d_full, fs, t_axis, groups)
        
        # Create sub-gridspec for bands
        gs_pvt = gs[2, :].subgridspec(len(BANDS), 1, hspace=0.3)
        
        for band_i, band_name in enumerate(BANDS.keys()):
            ax = fig.add_subplot(gs_pvt[band_i])
            
            if band_name in band_timecourses:
                for grp_i, (_, grp_name) in enumerate(groups):
                    short_name = grp_name.split(' (')[0]
                    if short_name in band_timecourses[band_name]:
                        mean, sem, t_plot = band_timecourses[band_name][short_name]
                        color = REGION_COLORS[grp_i % len(REGION_COLORS)]
                        ax.plot(t_plot, mean, color=color, lw=1.2, label=short_name)
                        ax.fill_between(t_plot, mean - sem, mean + sem, color=color, alpha=0.2)
            
            ax.axvspan(0, 100, color='grey', alpha=0.2)
            ax.axvline(0, color='k', linestyle='--', alpha=0.5, lw=0.8)
            ax.set_ylabel(band_name, fontsize=8)
            ax.grid(True, alpha=0.3)
            
            xlim = config['x_limits']
            if xlim['pre'] and xlim['post']:
                ax.set_xlim(xlim['pre'][0], xlim['post'][1])
            
            if band_i == 0:
                ax.legend(loc='upper right', fontsize='x-small', ncol=n_groups)
            if band_i == len(BANDS) - 1:
                ax.set_xlabel("Time (ms)")
            else:
                ax.set_xticklabels([])
    
    # Row 3: Band Stats bar charts
    band_names = list(BANDS.keys())
    x_pos = np.arange(len(band_names))
    width = 0.35
    
    for i, (grp_idxs, grp_name) in enumerate(groups):
        ax = fig.add_subplot(gs[3, i])
        short_name = grp_name.split(' (')[0]
        
        means_pre, means_post = [], []
        errs_pre, errs_post = [], []
        p_values = []
        
        for band in band_names:
            res = band_stats.get(band)
            if res is None:
                means_pre.append(0)
                means_post.append(0)
                errs_pre.append(0)
                errs_post.append(0)
                p_values.append(1.0)
                continue
            
            g_mean_pre = res['mean_pre'][grp_idxs]
            g_mean_post = res['mean_post'][grp_idxs]
            
            means_pre.append(np.nanmean(g_mean_pre))
            means_post.append(np.nanmean(g_mean_post))
            errs_pre.append(np.nanstd(g_mean_pre) / np.sqrt(len(grp_idxs)))
            errs_post.append(np.nanstd(g_mean_post) / np.sqrt(len(grp_idxs)))
            
            _, p_grp = stats.ttest_rel(g_mean_pre, g_mean_post, nan_policy='omit')
            p_values.append(p_grp)
        
        ax.bar(x_pos - width/2, means_pre, width, yerr=errs_pre,
               label='Pre', color='blue', alpha=0.6, capsize=3)
        ax.bar(x_pos + width/2, means_post, width, yerr=errs_post,
               label='Post', color='red', alpha=0.6, capsize=3)
        
        # Significance stars
        for idx, p in enumerate(p_values):
            if p < 0.05:
                h = max(means_pre[idx] + errs_pre[idx], means_post[idx] + errs_post[idx])
                ax.text(x_pos[idx], h * 1.05, '*', ha='center', fontsize=12, fontweight='bold')
        
        ax.set_title(f"{short_name}\nBand Power", fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([b[:3] for b in band_names], fontsize=7)
        ax.set_ylabel("Power" if i == 0 else "")
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)
        if i == 0:
            ax.legend(fontsize='x-small')
    
    # Save
    out_dir = fig_dir / "Session_Summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"summary_{session_id}.png"
    plt.savefig(out_path, dpi=config['summary']['dpi'])
    plt.close(fig)
    print(f"    Saved session summary -> {out_path.name}")
    
    return out_path


def generate_region_comparison(data, session_id, title_str, fig_dir, groups, fs, config,
                               ua_ids_1based, nsp_to_elec, ua_port):
    """
    Generate a figure comparing all regions side-by-side:
    - Row 0: Mean spectrogram per region
    - Row 1: Band power time course per region
    - Row 2: Spatial heatmap at peak response time per region
    """
    d_bb_full = data.get('broadband_full')
    if d_bb_full is None:
        print("    [Skip] No broadband_full for region comparison")
        return None
    
    n_groups = len(groups)
    if n_groups == 0:
        return None
    
    # Time axis
    t_pre = data.get('rel_time_pre')
    t_post = data.get('rel_time_post')
    if t_pre is not None and t_post is not None:
        t_full_ms = np.linspace(t_pre[0], t_post[-1], d_bb_full.shape[2])
    else:
        t_full_ms = np.linspace(-500.0, 1000.0, d_bb_full.shape[2])
    
    # Compute ERSP
    spec_cfg = config['spectrogram'].copy()
    try:
        ersp_all, freqs, t_bins_ms = compute_ersp_spectrogram(d_bb_full, t_full_ms, fs, spec_cfg)
    except Exception as e:
        print(f"    [Skip] ERSP computation failed: {e}")
        return None
    
    # Compute band power for selected band
    spec_cfg_trials = spec_cfg.copy()
    spec_cfg_trials['return_power_trials'] = True
    _, f_grid, t_grid_ms_full, P_trials = compute_ersp_spectrogram(d_bb_full, t_full_ms, fs, spec_cfg_trials)
    
    bl_win = spec_cfg.get('baseline_window', (-950, -650))
    baseline_mask = (t_grid_ms_full >= bl_win[0]) & (t_grid_ms_full <= bl_win[1])
    if not np.any(baseline_mask):
        baseline_mask[:max(1, len(t_grid_ms_full)//4)] = True
    
    band_name = config['band_plots']['selected_band']
    band_dB_trials = extract_band_power_trials(P_trials, f_grid, band_name, baseline_mask)
    
    if band_dB_trials is not None:
        mean_band_dB = np.nanmean(band_dB_trials, axis=0)
    else:
        mean_band_dB = None
    
    # Build electrode mapping
    elec_to_idx = build_elec_to_data_idx(ua_ids_1based, nsp_to_elec, ua_port) if ua_ids_1based is not None else {}
    
    # Create figure
    fig, axes = plt.subplots(3, n_groups, figsize=(5 * n_groups, 12), constrained_layout=True)
    if n_groups == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f"Region Comparison: {session_id}\n{title_str}", fontsize=14, fontweight='bold')
    
    vmin_spec = spec_cfg['vmin_db']
    vmax_spec = spec_cfg['vmax_db']
    vmin_heat = config['band_plots']['heatmap_vmin']
    vmax_heat = config['band_plots']['heatmap_vmax']
    
    for col, (grp_idxs, grp_name) in enumerate(groups):
        region_name = _region_from_group_name(grp_name)
        short_name = grp_name.split(' (')[0]
        grid_elec = _region_grid(region_name)
        
        # Row 0: Mean spectrogram
        ax = axes[0, col]
        grp_ersp = ersp_all[grp_idxs, :, :]
        mean_ersp = np.nanmean(grp_ersp, axis=0)
        
        im0 = ax.pcolormesh(t_bins_ms, freqs, mean_ersp, cmap=spec_cfg['cmap'],
                            vmin=vmin_spec, vmax=vmax_spec, shading='nearest')
        ax.axvline(0, color='white', lw=1, ls='--', alpha=0.8)
        ax.set_title(f"{short_name}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Freq (Hz)" if col == 0 else "")
        if col == n_groups - 1:
            add_colorbar(fig, im0, ax, "dB", fraction=0.05)
        
        # Row 1: Band power time course
        ax = axes[1, col]
        if mean_band_dB is not None:
            grp_band = mean_band_dB[grp_idxs, :]
            mean_trace = np.nanmean(grp_band, axis=0)
            sem_trace = np.nanstd(grp_band, axis=0) / np.sqrt(len(grp_idxs))
            
            ax.plot(t_grid_ms_full, mean_trace, color=REGION_COLORS[col % len(REGION_COLORS)], lw=2)
            ax.fill_between(t_grid_ms_full, mean_trace - sem_trace, mean_trace + sem_trace,
                           color=REGION_COLORS[col % len(REGION_COLORS)], alpha=0.3)
            ax.axvspan(0, 100, color='grey', alpha=0.2)
            ax.axvline(0, color='k', ls='--', alpha=0.5)
            ax.axhline(0, color='gray', ls=':', alpha=0.5)
        
        ax.set_ylabel(f"{band_name} (dB)" if col == 0 else "")
        ax.set_xlabel("Time (ms)")
        ax.set_xlim(spec_cfg['time_range'])
        ax.grid(True, alpha=0.3)
        
        # Row 2: Spatial heatmap at peak time
        ax = axes[2, col]
        if grid_elec is not None and mean_band_dB is not None:
            # Find peak time (max absolute response after stim)
            post_mask = t_grid_ms_full > 0
            if np.any(post_mask):
                grp_band = mean_band_dB[grp_idxs, :]
                mean_trace = np.nanmean(grp_band, axis=0)
                post_indices = np.where(post_mask)[0]
                peak_rel_idx = np.argmax(np.abs(mean_trace[post_mask]))
                peak_idx = post_indices[peak_rel_idx]
                peak_time = t_grid_ms_full[peak_idx]
                
                im2 = render_spatial_grid(ax, mean_band_dB, peak_idx, grid_elec, elec_to_idx,
                                         vmin_heat, vmax_heat, smoothing=True)
                ax.set_title(f"T={peak_time:.0f}ms", fontsize=10)
            else:
                ax.text(0.5, 0.5, "No post-stim data", ha='center', va='center', transform=ax.transAxes)
        else:
            ax.set_facecolor('#f0f0f0')
            ax.text(0.5, 0.5, "No grid", ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(range(1, 9), fontsize=7)
        ax.set_yticklabels(range(1, 9), fontsize=7)
        if col == 0:
            ax.set_ylabel("Row")
        ax.set_xlabel("Col")
    
    # Row labels
    axes[0, 0].annotate('ERSP', xy=(-0.3, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', rotation=90, va='center')
    axes[1, 0].annotate(f'{band_name}', xy=(-0.3, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', rotation=90, va='center')
    axes[2, 0].annotate('Spatial', xy=(-0.3, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', rotation=90, va='center')
    
    # Save
    out_dir = fig_dir / "Region_Comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"regions_{session_id}.png"
    plt.savefig(out_path, dpi=config['summary']['dpi'])
    plt.close(fig)
    print(f"    Saved region comparison -> {out_path.name}")
    
    return out_path


def generate_spectrogram_grid(ersp_all, freqs, t_bins_ms, ua_ids_1based, session_id,
                              region_name, grid_elec, elec_to_idx, fig_dir, spec_cfg):
    """
    Plot spectrograms in 8x8 grid matching physical electrode layout.
    """
    fig, axes = plt.subplots(8, 8, figsize=(24, 20), sharex=True, sharey=True)
    fig.suptitle(f"{session_id} — {region_name}\nERSP Spectrograms (8×8 Grid)", fontsize=14)
    
    im = None
    for r in range(8):
        for c in range(8):
            ax = axes[r, c]
            elec_id = int(grid_elec[r, c])
            idx = elec_to_idx.get(elec_id, None)
            
            if idx is not None and 0 <= idx < ersp_all.shape[0]:
                ersp_ch = ersp_all[idx]
                im = ax.pcolormesh(t_bins_ms, freqs, ersp_ch,
                                   cmap=spec_cfg['cmap'],
                                   vmin=spec_cfg['vmin_db'],
                                   vmax=spec_cfg['vmax_db'],
                                   shading='nearest')
                ax.axvline(0, color='white', lw=0.5, ls='--', alpha=0.7)
                
                nsp_id = ua_ids_1based[idx] if ua_ids_1based is not None else idx
                ax.set_title(f"E{elec_id}", fontsize=7, pad=2)
            else:
                ax.set_facecolor('#f0f0f0')
                ax.text(0.5, 0.5, f"E{elec_id}\nN/A", ha='center', va='center',
                        fontsize=6, alpha=0.3, transform=ax.transAxes)
            
            ax.tick_params(labelsize=5)
            if r < 7:
                ax.set_xticklabels([])
            if c > 0:
                ax.set_yticklabels([])
    
    # Shared labels
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (ms)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Freq (Hz)", fontsize=7)
    
    # Colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    if im:
        fig.colorbar(im, cax=cbar_ax, label="Power (dB)")
    
    # Save
    safe_region = region_name.replace(' ', '_')
    out_dir = fig_dir / "Spectrogram_Grids"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"spec_grid_{session_id}_{safe_region}.png"
    plt.savefig(out_path, dpi=spec_cfg.get('dpi', 120))
    plt.close(fig)
    print(f"    Saved spectrogram grid -> {out_path.name}")
    
    return out_path


def generate_spatial_gif(band_dB, t_ms, ua_ids_1based, session_id, fig_dir, config,
                         groups, nsp_to_elec, ua_port):
    """Generate animated GIF of spatial heatmap over time."""
    band_name = config['band_plots']['selected_band']
    gif_cfg = config['band_plots']
    t_start, t_end = gif_cfg.get('gif_time_range', (-500, 500))
    t_step = gif_cfg.get('gif_step_ms', 10)
    vmin = gif_cfg['heatmap_vmin']
    vmax = gif_cfg['heatmap_vmax']

    out_dir = fig_dir / "Spatial_Animations"
    out_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_band_dB = np.nanmean(band_dB, axis=0)

    elec_to_idx = build_elec_to_data_idx(ua_ids_1based, nsp_to_elec, ua_port)
    gif_time_pts = np.arange(t_start, t_end + t_step, t_step)
    
    paths = []
    
    for grp_idxs, grp_name in groups:
        region_name = _region_from_group_name(grp_name)
        grid_elec = _region_grid(region_name)

        if grid_elec is None or len(grp_idxs) == 0:
            continue

        safe_region = region_name.replace(' ', '_')
        frames = []
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        for t_target in gif_time_pts:
            t_bin_idx = int(np.argmin(np.abs(t_ms - t_target)))
            t_actual = float(t_ms[t_bin_idx])
            
            if t_actual < t_ms[0] or t_actual > t_ms[-1]:
                continue

            ax.clear()
            im = render_spatial_grid(ax, mean_band_dB, t_bin_idx, grid_elec, elec_to_idx,
                                     vmin, vmax, smoothing=True)
            
            ax.set_title(f"{region_name} | {band_name.upper()}\nT = {t_actual:.0f} ms", fontsize=12)
            ax.set_xticks(np.arange(8))
            ax.set_yticks(np.arange(8))
            ax.set_xticklabels(np.arange(1, 9), fontsize=8)
            ax.set_yticklabels(np.arange(1, 9), fontsize=8)
            ax.set_ylabel("Row", fontsize=10)
            ax.set_xlabel("Col", fontsize=10)
            
            if len(fig.axes) == 1:
                fig.colorbar(im, ax=ax, label="dB")

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=80)
            buf.seek(0)
            frames.append(imageio.v3.imread(buf))

        plt.close(fig)

        if frames:
            out_path = out_dir / f"spatial_{session_id}_{safe_region}.gif"
            imageio.mimsave(out_path, frames, fps=10)
            print(f"    Saved spatial GIF -> {out_path.name}")
            paths.append(out_path)
    
    return paths


# =============================================================================
# PDF REPORT GENERATOR
# =============================================================================

def generate_pdf_report(session_id, title_str, fig_dir, figure_paths, config):
    """
    Combine all generated figures into a multi-page PDF report.
    """
    pdf_dir = fig_dir / "Reports"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"report_{session_id}.pdf"
    
    # Filter valid paths
    valid_paths = [p for p in figure_paths if p is not None and p.exists()]
    
    if not valid_paths:
        print("    [Skip] No figures to include in PDF")
        return None
    
    with PdfPages(pdf_path) as pdf:
        # Title page
        fig_title = plt.figure(figsize=(11, 8.5))
        fig_title.text(0.5, 0.6, f"LFP Analysis Report", fontsize=24, ha='center', fontweight='bold')
        fig_title.text(0.5, 0.5, session_id, fontsize=18, ha='center')
        fig_title.text(0.5, 0.4, title_str, fontsize=12, ha='center', style='italic')
        fig_title.text(0.5, 0.2, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                      fontsize=10, ha='center', alpha=0.7)
        pdf.savefig(fig_title)
        plt.close(fig_title)
        
        # Add each figure
        for fig_path in valid_paths:
            if fig_path.suffix.lower() == '.gif':
                continue  # Skip GIFs in PDF
            
            try:
                img = plt.imread(fig_path)
                fig = plt.figure(figsize=(11, 8.5))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.imshow(img)
                ax.axis('off')
                pdf.savefig(fig, dpi=config['pdf']['dpi'])
                plt.close(fig)
            except Exception as e:
                print(f"    [Warn] Could not add {fig_path.name} to PDF: {e}")
        
        # Metadata
        d = pdf.infodict()
        d['Title'] = f'LFP Analysis Report: {session_id}'
        d['Author'] = 'Automated Analysis Pipeline'
        d['CreationDate'] = datetime.datetime.now()
    
    print(f"    Saved PDF report -> {pdf_path.name}")
    return pdf_path


# =============================================================================
# DETAILED FIGURE GENERATORS (Optional)
# =============================================================================

def generate_detailed_heatmaps(data, session_id, title_str, fig_dir, groups, fs, config, is_utah):
    """Generate the original heatmap/trace plots (detailed view)."""
    bands_list = ['broadband', 'delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
    
    # Pre-compute global scales
    global_scales = {}
    for band in bands_list:
        limit = config['scales'].get(band)
        if limit is None:
            d_pre = data.get(f"{band}_pre")
            d_post = data.get(f"{band}_post")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vals = []
                if d_pre is not None:
                    vals.append(np.nanmean(d_pre, axis=0).flatten())
                if d_post is not None:
                    vals.append(np.nanmean(d_post, axis=0).flatten())
                if vals:
                    all_v = np.concatenate(vals)
                    valid_v = all_v[np.isfinite(all_v)]
                    if valid_v.size > 0:
                        vmin, vmax = np.percentile(valid_v, [2, 98])
                        limit = max(abs(vmin), abs(vmax))
                    else:
                        limit = 50.0
                else:
                    limit = 50.0
        global_scales[band] = limit
    
    paths = []
    plot_groups = groups if is_utah else [(list(range(data['broadband_post'].shape[1])), "AllChannels")]
    
    for grp_idxs, grp_name in plot_groups:
        safe_name = grp_name.split(' (')[0].replace(' ', '_')
        n_grp_ch = len(grp_idxs)
        if n_grp_ch == 0:
            continue
        
        n_rows = len(bands_list)
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 3 * n_rows), constrained_layout=True,
                                  gridspec_kw={'width_ratios': [3, 1, 1]})
        fig.suptitle(f"{session_id} - {safe_name}\n{title_str}", fontsize=14)
        
        t_pre = data.get('rel_time_pre')
        t_post = data.get('rel_time_post')
        
        for i, band in enumerate(bands_list):
            d_pre = data.get(f"{band}_pre")
            d_post = data.get(f"{band}_post")
            
            if d_pre is not None:
                d_pre = d_pre[:, grp_idxs, :]
            if d_post is not None:
                d_post = d_post[:, grp_idxs, :]
            
            limit = global_scales[band]
            cmap = config['heatmap_cmap']
            
            # Heatmap
            ax = axes[i, 0]
            if is_utah:
                d_full = data.get(f"{band}_full")
                if d_full is not None:
                    d_full_view = d_full[:, grp_idxs, :]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m_full = np.nanmean(d_full_view, axis=0)
                    
                    t_start = t_pre[0] if t_pre is not None else -500
                    t_end = t_post[-1] if t_post is not None else 1000
                    im = ax.imshow(m_full, aspect='auto', origin='upper', cmap=cmap,
                                   vmin=-limit, vmax=limit,
                                   extent=[t_start, t_end, n_grp_ch - 0.5, -0.5])
                    add_colorbar(fig, im, ax, "µV")
            else:
                if d_pre is not None and d_post is not None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m_pre = np.nanmean(d_pre, axis=0)
                        m_post = np.nanmean(d_post, axis=0)
                    
                    dt = t_pre[1] - t_pre[0] if t_pre is not None and len(t_pre) > 1 else 1
                    n_gap = max(1, int((t_post[0] - t_pre[-1]) / dt) - 1) if t_pre is not None and t_post is not None else 1
                    gap_data = np.full((n_grp_ch, n_gap), np.nan)
                    m_comb = np.hstack([m_pre, gap_data, m_post])
                    
                    t_start = t_pre[0] if t_pre is not None else -500
                    t_end = t_post[-1] if t_post is not None else 1000
                    im = ax.imshow(m_comb, aspect='auto', origin='upper', cmap=cmap,
                                   vmin=-limit, vmax=limit,
                                   extent=[t_start, t_end, n_grp_ch - 0.5, -0.5])
                    add_colorbar(fig, im, ax, "µV")
                    if t_pre is not None and t_post is not None:
                        ax.axvspan(t_pre[-1], t_post[0], color='gray', alpha=0.3, hatch='///')
            
            ax.set_ylabel(f"{band.capitalize()}\nCh Index")
            ax.set_title("Combined Activity")
            if i == n_rows - 1:
                ax.set_xlabel("Time (ms)")
            
            # Traces (simplified - combine into one panel for Utah)
            rep_idx = get_representative_channel(d_post)
            
            if is_utah:
                ax_trace = axes[i, 1]
                axes[i, 2].axis('off')
                
                d_full = data.get(f"{band}_full")
                if d_full is not None:
                    rep_full = d_full[:, grp_idxs[rep_idx], :]
                    t_full = np.linspace(t_pre[0], t_post[-1], rep_full.shape[1]) if t_pre is not None and t_post is not None else np.arange(rep_full.shape[1])
                    
                    ax_trace.plot(t_full, rep_full.T, color='gray', alpha=0.3, lw=0.5)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ax_trace.plot(t_full, np.nanmean(rep_full, axis=0), color='green', lw=1.5)
                    ax_trace.axvspan(0, 100, color='grey', alpha=0.3)
                    ax_trace.axvline(0, color='red', ls='--', alpha=0.5)
                
                ax_trace.set_title(f"Ch {grp_idxs[rep_idx]}")
                ax_trace.set_ylabel("µV")
            else:
                # Pre trace
                ax = axes[i, 1]
                if d_pre is not None:
                    rep_pre = d_pre[:, rep_idx, :]
                    ax.plot(t_pre, rep_pre.T, color='gray', alpha=0.3, lw=0.5)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ax.plot(t_pre, np.nanmean(rep_pre, axis=0), color='blue', lw=1.5)
                ax.set_title("Pre")
                ax.set_ylabel("µV")
                
                # Post trace
                ax = axes[i, 2]
                if d_post is not None:
                    rep_post = d_post[:, rep_idx, :]
                    ax.plot(t_post, rep_post.T, color='gray', alpha=0.3, lw=0.5)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ax.plot(t_post, np.nanmean(rep_post, axis=0), color='red', lw=1.5)
                ax.set_title("Post")
        
        out_dir = fig_dir / "Detailed" / "Heatmaps"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"heatmap_{session_id}_{safe_name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        paths.append(out_path)
        print(f"    Saved detailed heatmap -> {out_path.name}")
    
    return paths


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_session(npz_file, fig_dir, label, df_summary, bad_ch_map, nsp_to_elec, elec_to_region):
    """Process a single session file and generate all configured figures."""
    print(f"  Processing {npz_file.name}...")
    
    try:
        data = np.load(npz_file, allow_pickle=True)
    except Exception as e:
        print(f"    [Error] Loading failed: {e}")
        return
    
    # Extract basic info
    fs = int(data.get('fs_lfp', 1000))
    session = str(data.get('session', npz_file.stem.replace("aligned_lfp__", "")))
    category = str(data.get('category', 'unknown'))
    target = str(data.get('target', 'none'))
    
    # Session ID for filenames
    target_suffix = f"_{category}"
    if target and target != 'none':
        target_suffix += f"_target_{target}"
    session_id = f"{session}{target_suffix}".replace('.ns6', '')
    
    # Check for required data
    if 'broadband_post' not in data:
        print(f"    [Skip] Missing broadband_post")
        return
    
    bb_data = data['broadband_post']
    n_trials, n_ch, n_time = bb_data.shape
    print(f"    Dimensions: {n_trials} trials, {n_ch} channels")
    
    # Build title string
    is_baseline = "baseline" in npz_file.stem
    disp_cat = category.replace("_", " ").title()
    disp_target = f" - Target {target}" if target and target != 'none' else ""
    
    # Lookup metadata
    active_str = "Unknown"
    freq_str = "Unknown"
    amp_str = ""
    
    if df_summary is not None and 'Intan_Session' in df_summary.columns:
        matches = df_summary[df_summary['Intan_Session'].str.startswith(session)]
        if not matches.empty:
            row = matches.iloc[0]
            active_str = str(row.get('Stim_Channels', 'Unknown'))
            freq_str = str(row.get('Stim_Frequency', 'Unknown'))
            amp_val = row.get('Stim_Amplitudes', 'N/A')
            if pd.notna(amp_val) and str(amp_val).lower() != 'nan':
                amp_str = f"| {amp_val} µA"
    
    if is_baseline:
        baseline_port = str(data.get('group_port', '?'))
        baseline_depth = str(data.get('group_depth', '?'))
        title_str = f"Baseline | Port {baseline_port} | Depth {baseline_depth} mm"
    else:
        title_str = f"{disp_cat}{disp_target} | {active_str} @ {freq_str} Hz {amp_str}"
    
    # Determine groups (Utah regions vs NPRW channel blocks)
    is_utah = (label == "Utah_Array")
    groups = []
    ua_port = 'A'
    ua_ids_1based = None
    
    if is_utah:
        # Determine port
        try:
            parts = npz_file.stem.split('_')
            br_idx = None
            for p in reversed(parts):
                if p.isdigit():
                    br_idx = int(p)
                    break
            
            if br_idx is not None and cfg.METADATA_CSV.exists():
                with open(cfg.METADATA_CSV, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('BR_File') and int(row['BR_File']) == br_idx:
                            p = row.get('UA_port', '').strip().upper()
                            if p in ['A', 'B']:
                                ua_port = p
                            break
        except:
            pass
        
        ua_ids_1based = data.get('ua_ids_1based')
        
        if ua_ids_1based is not None:
            sma_idxs, pmd_idxs, m1i_idxs, m1s_idxs = [], [], [], []
            
            for ch_idx in range(n_ch):
                nsp_id = int(ua_ids_1based[ch_idx])
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
            
            if sma_idxs:
                groups.append((sma_idxs, f"SMA (n={len(sma_idxs)})"))
            if pmd_idxs:
                groups.append((pmd_idxs, f"PMd (n={len(pmd_idxs)})"))
            if m1i_idxs:
                groups.append((m1i_idxs, f"M1i (n={len(m1i_idxs)})"))
            if m1s_idxs:
                groups.append((m1s_idxs, f"M1s (n={len(m1s_idxs)})"))
            
            print(f"    Regions: SMA={len(sma_idxs)}, PMd={len(pmd_idxs)}, M1i={len(m1i_idxs)}, M1s={len(m1s_idxs)}")
    
    if not groups:
        # Fallback for NPRW or Utah without mapping
        ch_per_group = 16
        for g in range(8):
            start = g * ch_per_group
            end = min(start + ch_per_group, n_ch)
            if start < n_ch:
                groups.append((list(range(start, end)), f"Ch {start}-{end-1}"))
    
    # Collect all figure paths for PDF
    figure_paths = []
    
    # Generate figures based on config
    if FIGURE_CONFIG.get('generate_session_summary', False):
        path = generate_session_summary(data, session_id, title_str, fig_dir, groups, fs, PLOT_CONFIG)
        if path:
            figure_paths.append(path)
    
    if FIGURE_CONFIG.get('generate_region_comparison', False) and is_utah:
        path = generate_region_comparison(data, session_id, title_str, fig_dir, groups, fs, PLOT_CONFIG,
                                          ua_ids_1based, nsp_to_elec, ua_port)
        if path:
            figure_paths.append(path)
    
    if FIGURE_CONFIG.get('generate_spectrogram_grid', False) and is_utah:
        d_bb_full = data.get('broadband_full')
        if d_bb_full is not None:
            t_pre = data.get('rel_time_pre')
            t_post = data.get('rel_time_post')
            if t_pre is not None and t_post is not None:
                t_full_ms = np.linspace(t_pre[0], t_post[-1], d_bb_full.shape[2])
            else:
                t_full_ms = np.linspace(-500.0, 1000.0, d_bb_full.shape[2])
            
            spec_cfg = PLOT_CONFIG['spectrogram']
            try:
                ersp_all, freqs, t_bins_ms = compute_ersp_spectrogram(d_bb_full, t_full_ms, fs, spec_cfg)
                elec_to_idx = build_elec_to_data_idx(ua_ids_1based, nsp_to_elec, ua_port)
                
                for grp_idxs, grp_name in groups:
                    region_name = _region_from_group_name(grp_name)
                    grid_elec = _region_grid(region_name)
                    
                    if grid_elec is not None:
                        path = generate_spectrogram_grid(ersp_all, freqs, t_bins_ms, ua_ids_1based,
                                                         session_id, region_name, grid_elec, elec_to_idx,
                                                         fig_dir, spec_cfg)
                        if path:
                            figure_paths.append(path)
            except Exception as e:
                print(f"    [Error] Spectrogram grid failed: {e}")
    
    if FIGURE_CONFIG.get('generate_spatial_gif', False) and is_utah:
        d_bb_full = data.get('broadband_full')
        if d_bb_full is not None and ua_ids_1based is not None:
            t_pre = data.get('rel_time_pre')
            t_post = data.get('rel_time_post')
            if t_pre is not None and t_post is not None:
                t_full_ms = np.linspace(t_pre[0], t_post[-1], d_bb_full.shape[2])
            else:
                t_full_ms = np.linspace(-500.0, 1000.0, d_bb_full.shape[2])
            
            spec_cfg_trials = PLOT_CONFIG['spectrogram'].copy()
            spec_cfg_trials['return_power_trials'] = True
            
            try:
                _, f_grid, t_grid_ms_full, P_trials = compute_ersp_spectrogram(d_bb_full, t_full_ms, fs, spec_cfg_trials)
                
                bl_win = spec_cfg_trials.get('baseline_window', (-950, -650))
                baseline_mask = (t_grid_ms_full >= bl_win[0]) & (t_grid_ms_full <= bl_win[1])
                if not np.any(baseline_mask):
                    baseline_mask[:max(1, len(t_grid_ms_full)//4)] = True
                
                band_name = PLOT_CONFIG['band_plots']['selected_band']
                band_dB_trials = extract_band_power_trials(P_trials, f_grid, band_name, baseline_mask)
                
                if band_dB_trials is not None:
                    # Crop to time range
                    t_range = spec_cfg_trials.get('time_range', (-400, 500))
                    time_mask = (t_grid_ms_full >= t_range[0]) & (t_grid_ms_full <= t_range[1])
                    band_dB_cropped = band_dB_trials[:, :, time_mask]
                    t_cropped = t_grid_ms_full[time_mask]
                    
                    gif_paths = generate_spatial_gif(band_dB_cropped, t_cropped, ua_ids_1based,
                                                     session_id, fig_dir, PLOT_CONFIG, groups,
                                                     nsp_to_elec, ua_port)
                    figure_paths.extend(gif_paths)
            except Exception as e:
                print(f"    [Error] Spatial GIF failed: {e}")
    
    if FIGURE_CONFIG.get('detailed_heatmaps', False):
        paths = generate_detailed_heatmaps(data, session_id, title_str, fig_dir, groups, fs, PLOT_CONFIG, is_utah)
        figure_paths.extend(paths)
    
    # Generate PDF report
    if FIGURE_CONFIG.get('generate_pdf_report', False) and figure_paths:
        generate_pdf_report(session_id, title_str, fig_dir, figure_paths, PLOT_CONFIG)


def process_directory(lfp_dir, fig_dir, label="LFP"):
    """Process all files in a directory."""
    if not lfp_dir.exists():
        print(f"Directory not found: {lfp_dir}")
        return

    files = sorted(lfp_dir.rglob("aligned_lfp__*.npz"))
    if not files:
        print(f"No aligned LFP npz files found in {lfp_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {label} ({len(files)} files)")
    print(f"{'='*60}")
    
    # Load summary CSV
    summary_csv = cfg.METADATA_ROOT / "stimulation_summary.csv"
    df_summary = None
    if summary_csv.exists():
        try:
            df_summary = pd.read_csv(summary_csv)
            if 'Intan_Session' in df_summary.columns:
                df_summary['Intan_Session'] = df_summary['Intan_Session'].astype(str).str.strip()
        except Exception as e:
            print(f"Warning: Failed to load stimulation summary: {e}")
    
    # Load impedance data
    bad_ch_map = imp_utils.get_session_impedances(cfg.SESSION_LOC)
    
    # Process files
    for npz_file in tqdm(files, desc=f"Processing {label}"):
        process_session(npz_file, fig_dir, label, df_summary, bad_ch_map,
                       nsp_to_elec_global, elec_to_region_global)


def plot_lfp_check():
    """Main entry point."""
    process_directory(NPRW_LFP_DIR, NPRW_FIG_DIR, label="NPRW_Intan")
    process_directory(UA_LFP_DIR, UA_FIG_DIR, label="Utah_Array")


if __name__ == "__main__":
    plot_lfp_check()