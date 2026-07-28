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
import re

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
    # NEW: Timing-focused figures (what you actually want)
    'generate_band_dynamics': True,         # All bands over time with peak detection
    'generate_timing_heatmap': True,        # Summary heatmap of latencies/amplitudes
    
    # Simplified session summary (PSD + band stats only)
    'generate_session_summary': True,
    
    # Optional detailed views
    'generate_spectrogram_grid': False,
    'generate_pdf_report': False,
    
    # Deprecated
    'generate_region_comparison': False,
    'generate_spatial_gif': False,
    'detailed_heatmaps': False,
}

PLOT_CONFIG = {
    "heatmap_cmap": "RdBu_r",
    
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
        "heatmap_vmin": -6.0,
        "heatmap_vmax": 6.0,
    },
    
    "summary": {
        "figsize": (24, 20),
        "dpi": 150,
    },
    
    "pdf": {
        "dpi": 150,
    }
}

# Frequency bands
BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 25),
    'Low Gamma': (25, 60),
}
# Only include High Gamma if data supports it
BANDS_WITH_HIGH_GAMMA = {
    **BANDS,
    'High Gamma': (60, 120)
}

BAND_COLORS = {
    'Delta': '#E91E63',
    'Theta': '#9C27B0',
    'Alpha': '#4CAF50',
    'Beta': '#FF9800',
    'Low Gamma': '#2196F3',
    'High Gamma': '#F44336'
}

REGION_COLORS = {
    'SMA': '#1f77b4',
    'PMd': '#ff7f0e', 
    'M1i': '#2ca02c',
    'M1s': '#d62728',
}
REGION_ORDER = ['SMA', 'PMd', 'M1i', 'M1s']


# =============================================================================
# SETUP / GLOBALS
# =============================================================================

from RCP_analysis.python.functions.br_preproc import (
    load_electrode_mapping,
    get_region_from_group_name as _region_from_group_name,
    get_region_grid,
    build_elec_to_data_idx
)

MAPPING_CSV = Path(__file__).parent.parent.parent / "config" / "electrode_port_mapping.csv"
nsp_to_elec_global, UTAH_ELEC_GRIDS, elec_to_region_global = load_electrode_mapping(MAPPING_CSV)

def _region_grid(region_name: str):
    return get_region_grid(region_name, UTAH_ELEC_GRIDS)

NPRW_LFP_DIR = cfg.OUT_BASE / "checkpoints" / "NPRW_LFP"
UA_LFP_DIR = cfg.OUT_BASE / "checkpoints" / "UA_LFP"
NPRW_FIG_DIR = cfg.OUT_BASE / "figures" / "NPRW_LFP"
UA_FIG_DIR = cfg.OUT_BASE / "figures" / "UA_LFP"

NPRW_FIG_DIR.mkdir(parents=True, exist_ok=True)
UA_FIG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# METADATA LOOKUP (from metadata.csv, like peri-stim plotting code)
# =============================================================================

def load_metadata_df():
    """Load the metadata CSV file."""
    if not cfg.METADATA_CSV.exists():
        print(f"[warn] Metadata CSV not found: {cfg.METADATA_CSV}")
        return None
    
    try:
        df = pd.read_csv(cfg.METADATA_CSV)
        return df
    except Exception as e:
        print(f"[warn] Failed to load metadata CSV: {e}")
        return None


def extract_br_idx_from_filename(filename: str) -> int | None:
    """
    Extract BR index from filename.
    Pattern: aligned_lfp__SESSION__BR_XXX_...
    """
    match = re.search(r"BR_(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def lookup_stim_info(br_idx: int, df_metadata: pd.DataFrame) -> dict:
    """
    Lookup stimulation info from metadata CSV based on BR_File index.
    
    Calculates number of pulses from Stim_Duration_ms assuming 2.5ms per pulse.
    
    Returns dict with keys:
        - n_pulses: int or None
        - frequency: float or None (Hz)
        - amplitude: float or None (µA)
        - active_channels: str or None
        - stim_duration: float or None (ms)
        - depth: float or None (mm)
        - stim_type: str or None
    """
    result = {
        'n_pulses': None,
        'frequency': None,
        'amplitude': None,
        'active_channels': None,
        'stim_duration': None,
        'depth': None,
        'stim_type': None,
    }
    
    if df_metadata is None or br_idx is None:
        return result
    
    # Check for BR_File column
    if 'BR_File' not in df_metadata.columns:
        print(f"    [warn] No BR_File column found in metadata")
        return result
    
    # Find matching row
    try:
        matches = df_metadata[df_metadata['BR_File'] == br_idx]
        if matches.empty:
            # Try string match
            matches = df_metadata[df_metadata['BR_File'].astype(str) == str(br_idx)]
        
        if matches.empty:
            return result
        
        row = matches.iloc[0]
        
        # Extract frequency (Stim_Frequency_Hz)
        if 'Stim_Frequency_Hz' in row.index and pd.notna(row['Stim_Frequency_Hz']):
            try:
                result['frequency'] = float(row['Stim_Frequency_Hz'])
            except (ValueError, TypeError):
                pass
        
        # Extract amplitude (Current_uA)
        if 'Current_uA' in row.index and pd.notna(row['Current_uA']):
            try:
                result['amplitude'] = float(row['Current_uA'])
            except (ValueError, TypeError):
                pass
        
        # Extract active channels (Channels)
        if 'Channels' in row.index and pd.notna(row['Channels']):
            result['active_channels'] = str(row['Channels'])
        
        # Extract stim duration and calculate pulses (Stim_Duration_ms)
        if 'Stim_Duration_ms' in row.index and pd.notna(row['Stim_Duration_ms']):
            try:
                duration_ms = float(row['Stim_Duration_ms'])
                result['stim_duration'] = duration_ms
                # Calculate number of pulses (each pulse is 2.5ms at 400Hz)
                # Or use frequency if available
                if result['frequency'] and result['frequency'] > 0:
                    # pulses = duration * frequency / 1000
                    result['n_pulses'] = int(duration_ms * result['frequency'] / 1000)
                else:
                    # Fallback: assume 2.5ms per pulse
                    result['n_pulses'] = int(duration_ms / 2.5)
            except (ValueError, TypeError):
                pass
        
        # Extract depth (Depth_mm)
        if 'Depth_mm' in row.index and pd.notna(row['Depth_mm']):
            try:
                result['depth'] = float(row['Depth_mm'])
            except (ValueError, TypeError):
                pass
        
        # Extract type (Type)
        if 'Type' in row.index and pd.notna(row['Type']):
            result['stim_type'] = str(row['Type'])
        
    except Exception as e:
        print(f"    [warn] Error looking up metadata for BR={br_idx}: {e}")
    
    return result


def build_title_string(session: str, category: str, target: str, stim_info: dict, is_baseline: bool = False) -> str:
    """
    Build a descriptive title string for plots.
    """
    if is_baseline:
        return "Baseline Recording"
    
    parts = []
    
    # Type (if available)
    if stim_info.get('stim_type'):
        parts.append(stim_info['stim_type'])
    elif category and category != 'unknown':
        cat_display = category.replace("_", " ").title()
        parts.append(cat_display)
    
    # Target
    if target and target != 'none':
        parts.append(f"Target {target}")
    
    # Build stim parameter string
    stim_parts = []
    
    if stim_info.get('active_channels'):
        stim_parts.append(f"Ch {stim_info['active_channels']}")
    
    if stim_info.get('frequency'):
        stim_parts.append(f"{stim_info['frequency']:.0f} Hz")
    
    if stim_info.get('amplitude'):
        stim_parts.append(f"{stim_info['amplitude']:.0f} µA")
    
    if stim_info.get('n_pulses'):
        stim_parts.append(f"{stim_info['n_pulses']} pulses")
    
    if stim_info.get('depth'):
        stim_parts.append(f"{stim_info['depth']:.1f} mm")
    
    # Format: "Ch X @ 400 Hz | 100 µA, 40 pulses, 2.0 mm"
    if stim_parts:
        if len(stim_parts) >= 2:
            parts.append(f"{stim_parts[0]} @ {stim_parts[1]}")
            if len(stim_parts) > 2:
                parts.append(", ".join(stim_parts[2:]))
        else:
            parts.append(stim_parts[0])
    
    if not parts:
        return "Stimulation"
    
    return " | ".join(parts)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def add_colorbar(fig, im, ax, label="", fraction=0.046, pad=0.04, fontsize=9):
    """Consistent colorbar styling."""
    cb = fig.colorbar(im, ax=ax, fraction=fraction, pad=pad)
    cb.set_label(label, fontsize=fontsize)
    cb.ax.tick_params(labelsize=max(6, fontsize-2))
    return cb


def add_stim_shading(ax, stim_start=0, stim_end=100, color='grey', alpha=0.2):
    """Add shading for stimulation period."""
    ax.axvspan(stim_start, stim_end, color=color, alpha=alpha, zorder=0)
    ax.axvline(stim_start, color='red', ls='--', alpha=0.8, lw=1.2)


@lru_cache(maxsize=32)
def get_dpss_windows(nperseg, NW=4):
    """Cache DPSS windows to avoid recomputation."""
    from scipy.signal.windows import dpss
    Kmax = 2 * NW - 1
    return tuple(dpss(nperseg, NW, Kmax))


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


def compute_ersp_spectrogram(broadband_full, t_full_ms, fs, spec_cfg):
    """
    Compute per-channel ERSP spectrogram from broadband_full data.
    
    Returns:
        ersp: (n_ch, n_freq, n_time) - baseline-normalized power in dB
        freqs: frequency axis
        t_bins_ms: time axis in ms
        P_trials: (n_trials, n_ch, n_freq, n_time) - raw power if return_power_trials=True
    """
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        if per_trial_norm:
            bl_power = np.nanmean(P[:, :, :, baseline_mask], axis=3, keepdims=True)
            bl_power[bl_power == 0] = 1e-10
            ersp_trials = 10.0 * np.log10(P / bl_power)
            ersp = np.nanmean(ersp_trials, axis=0)
        else:
            P_mean = np.nanmean(P, axis=0)
            baseline_power = np.nanmean(P_mean[:, :, baseline_mask], axis=2, keepdims=True)
            baseline_power[baseline_power == 0] = 1e-10
            ersp = 10.0 * np.log10(P_mean / baseline_power)
    
    if return_power_trials:
        return ersp, freqs, t_bins_ms, P
    else:
        time_mask = (t_bins_ms >= t_range[0]) & (t_bins_ms <= t_range[1])
        return ersp[:, :, time_mask], freqs, t_bins_ms[time_mask]


def extract_band_power_trials(P, f, band_name, baseline_mask):
    """
    Extract per-trial band power over time in dB relative to baseline.
    
    Args:
        P: (n_trials, n_ch, n_freq, n_time) raw power
        f: frequency axis
        band_name: name of band (must be in BANDS)
        baseline_mask: boolean mask for baseline time bins
    
    Returns:
        band_dB: (n_trials, n_ch, n_time) band power in dB relative to baseline
    """
    # Handle case-insensitive band lookup
    bands_lookup = {k.lower().replace(' ', ''): v for k, v in BANDS_WITH_HIGH_GAMMA.items()}
    band_key = band_name.lower().replace(' ', '')
    
    if band_key not in bands_lookup:
        return None
    
    freq_min, freq_max = bands_lookup[band_key]
    f_mask = (f >= freq_min) & (f <= freq_max)
    
    if not np.any(f_mask):
        return None
    
    # Average across frequencies in band
    band_P = np.nanmean(P[:, :, f_mask, :], axis=2)  # (n_trials, n_ch, n_time)
    
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


# =============================================================================
# NEW: BAND DYNAMICS FIGURE (All bands over time with peak detection)
# =============================================================================

def generate_band_dynamics(data, session_id, title_str, fig_dir, groups, fs, config):
    """
    Generate a figure showing all frequency bands over time for each region.
    
    This is the main figure you want - shows:
    - One row per frequency band
    - All regions overlaid with different colors
    - Peak latency marked and compared
    - Summary bar chart of peak latencies
    """
    d_bb_full = data.get('broadband_full')
    if d_bb_full is None:
        print("    [Skip] No broadband_full for band dynamics")
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
    
    # Compute ERSP with trial power returned
    spec_cfg = config['spectrogram'].copy()
    spec_cfg['return_power_trials'] = True
    
    try:
        _, f_grid, t_grid_ms, P_trials = compute_ersp_spectrogram(d_bb_full, t_full_ms, fs, spec_cfg)
    except Exception as e:
        print(f"    [Skip] ERSP computation failed: {e}")
        return None
    
    # Baseline mask
    bl_win = spec_cfg.get('baseline_window', (-950, -650))
    baseline_mask = (t_grid_ms >= bl_win[0]) & (t_grid_ms <= bl_win[1])
    if not np.any(baseline_mask):
        baseline_mask[:max(1, len(t_grid_ms)//4)] = True
    
    # Determine which bands to plot (exclude High Gamma if freq_max <= 60)
    freq_max = spec_cfg.get('freq_max', 60)
    if freq_max > 60:
        bands_to_plot = list(BANDS_WITH_HIGH_GAMMA.keys())
    else:
        bands_to_plot = list(BANDS.keys())
    
    n_bands = len(bands_to_plot)
    
    # Create figure
    fig = plt.figure(figsize=(14, 3 * n_bands + 3), constrained_layout=True)
    gs = fig.add_gridspec(n_bands + 1, 2, width_ratios=[3, 1], 
                          height_ratios=[1]*n_bands + [1.2])
    
    fig.suptitle(f"Band Power Dynamics: {session_id}\n{title_str}", 
                 fontsize=14, fontweight='bold')
    
    # Store peak info for summary
    peak_latencies = {band: {} for band in bands_to_plot}
    peak_amplitudes = {band: {} for band in bands_to_plot}
    
    # Time range for plotting
    t_range = config['spectrogram'].get('time_range', (-500, 500))
    time_mask = (t_grid_ms >= t_range[0]) & (t_grid_ms <= t_range[1])
    t_plot = t_grid_ms[time_mask]
    
    # Plot each band
    for band_i, band_name in enumerate(bands_to_plot):
        ax_main = fig.add_subplot(gs[band_i, 0])
        ax_peak = fig.add_subplot(gs[band_i, 1])
        
        # Extract band power (dB relative to baseline)
        band_dB_trials = extract_band_power_trials(P_trials, f_grid, band_name, baseline_mask)
        
        if band_dB_trials is None:
            ax_main.text(0.5, 0.5, f"No data for {band_name}", 
                        ha='center', va='center', transform=ax_main.transAxes)
            ax_main.set_ylabel(f"{band_name}\n(dB)", fontsize=10)
            continue
        
        # Crop to time range
        band_dB_cropped = band_dB_trials[:, :, time_mask]
        
        # Plot each region
        for grp_i, (grp_idxs, grp_name) in enumerate(groups):
            short_name = grp_name.split(' (')[0]
            color = REGION_COLORS.get(short_name, f'C{grp_i}')
            
            # Mean across channels in group, then across trials
            grp_data = band_dB_cropped[:, grp_idxs, :]  # (trials, channels, time)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                # Mean across channels first
                grp_ch_mean = np.nanmean(grp_data, axis=1)  # (trials, time)
                # Then mean and SEM across trials
                trace_mean = np.nanmean(grp_ch_mean, axis=0)  # (time,)
                trace_sem = np.nanstd(grp_ch_mean, axis=0) / np.sqrt(np.sum(np.isfinite(grp_ch_mean), axis=0))
            
            # Plot trace with SEM shading
            ax_main.plot(t_plot, trace_mean, color=color, lw=2, label=short_name)
            ax_main.fill_between(t_plot, trace_mean - trace_sem, trace_mean + trace_sem,
                                color=color, alpha=0.2)
            
            # Find peak in post-stim period (after stim onset)
            post_mask = t_plot > 0
            if np.any(post_mask):
                post_trace = trace_mean[post_mask]
                post_t = t_plot[post_mask]
                
                # Peak = max absolute deviation from 0
                if len(post_trace) > 0 and np.any(np.isfinite(post_trace)):
                    peak_idx = np.nanargmax(np.abs(post_trace))
                    peak_latency = post_t[peak_idx]
                    peak_amp = post_trace[peak_idx]
                    
                    peak_latencies[band_name][short_name] = peak_latency
                    peak_amplitudes[band_name][short_name] = peak_amp
                    
                    # Mark peak on trace
                    ax_main.plot(peak_latency, peak_amp, 'o', color=color, markersize=8,
                                markeredgecolor='black', markeredgewidth=1, zorder=5)
        
        # Formatting for main plot
        add_stim_shading(ax_main, stim_start=0, stim_end=100)
        ax_main.axhline(0, color='gray', ls=':', alpha=0.5)
        ax_main.set_xlim(t_range)
        ax_main.set_ylabel(f"{band_name}\n(dB)", fontsize=10)
        ax_main.grid(True, alpha=0.3)
        
        # Set reasonable y-limits
        ax_main.set_ylim(-8, 8)
        
        if band_i == 0:
            ax_main.legend(loc='upper right', fontsize='small', ncol=min(4, n_groups))
        if band_i == n_bands - 1:
            ax_main.set_xlabel("Time (ms)")
        else:
            ax_main.set_xticklabels([])
        
        # Peak latency bar chart (horizontal bars)
        regions_with_data = [grp_name.split(' (')[0] for _, grp_name in groups]
        latencies = [peak_latencies[band_name].get(r, np.nan) for r in regions_with_data]
        colors = [REGION_COLORS.get(r, f'C{i}') for i, r in enumerate(regions_with_data)]
        
        # Filter out NaN values for plotting
        valid_mask = [not np.isnan(l) for l in latencies]
        valid_regions = [r for r, v in zip(regions_with_data, valid_mask) if v]
        valid_latencies = [l for l, v in zip(latencies, valid_mask) if v]
        valid_colors = [c for c, v in zip(colors, valid_mask) if v]
        
        if valid_latencies:
            bars = ax_peak.barh(valid_regions, valid_latencies, color=valid_colors, alpha=0.7)
            ax_peak.axvline(0, color='k', ls='--', alpha=0.3)
            max_lat = max(valid_latencies) if valid_latencies else 100
            ax_peak.set_xlim(0, max(200, max_lat * 1.3))
            
            # Add latency values on bars
            for bar, lat in zip(bars, valid_latencies):
                ax_peak.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                           f'{lat:.0f}', va='center', fontsize=8)
        
        ax_peak.set_xlabel("Peak (ms)" if band_i == n_bands - 1 else "")
        ax_peak.set_title("Peak Latency" if band_i == 0 else "", fontsize=9)
        if band_i < n_bands - 1:
            ax_peak.set_xticklabels([])
    
    # Bottom summary: Peak latency comparison across all bands
    ax_summary = fig.add_subplot(gs[n_bands, :])
    
    x = np.arange(len(bands_to_plot))
    width = 0.8 / max(1, n_groups)
    
    for grp_i, (_, grp_name) in enumerate(groups):
        short_name = grp_name.split(' (')[0]
        color = REGION_COLORS.get(short_name, f'C{grp_i}')
        
        latencies = [peak_latencies[band].get(short_name, np.nan) for band in bands_to_plot]
        offset = (grp_i - n_groups/2 + 0.5) * width
        
        # Only plot non-NaN values
        valid_x = [xi for xi, l in zip(x, latencies) if not np.isnan(l)]
        valid_lat = [l for l in latencies if not np.isnan(l)]
        
        if valid_lat:
            ax_summary.bar([xi + offset for xi in valid_x], valid_lat, width, 
                          label=short_name, color=color, alpha=0.7)
    
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(bands_to_plot, fontsize=10)
    ax_summary.set_ylabel("Peak Latency (ms)")
    ax_summary.set_xlabel("Frequency Band")
    ax_summary.legend(loc='upper right', fontsize='small', ncol=min(4, n_groups))
    ax_summary.grid(axis='y', alpha=0.3)
    ax_summary.set_title("Peak Response Latency by Band and Region", fontsize=11, fontweight='bold')
    
    # Save
    out_dir = fig_dir / "Band_Dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"band_dynamics_{session_id}.png"
    plt.savefig(out_path, dpi=config['summary']['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved band dynamics -> {out_path.name}")
    
    return out_path


# =============================================================================
# NEW: TIMING HEATMAP (Summary of latencies and amplitudes)
# =============================================================================

def generate_timing_heatmap(data, session_id, title_str, fig_dir, groups, fs, config):
    """
    Generate a summary heatmap showing:
    - Peak latency across regions and bands
    - Peak amplitude across regions and bands
    
    This gives a quick overview of timing differences.
    """
    d_bb_full = data.get('broadband_full')
    if d_bb_full is None:
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
    
    spec_cfg = config['spectrogram'].copy()
    spec_cfg['return_power_trials'] = True
    
    try:
        _, f_grid, t_grid_ms, P_trials = compute_ersp_spectrogram(d_bb_full, t_full_ms, fs, spec_cfg)
    except Exception as e:
        print(f"    [Skip] Timing heatmap failed: {e}")
        return None
    
    bl_win = spec_cfg.get('baseline_window', (-950, -650))
    baseline_mask = (t_grid_ms >= bl_win[0]) & (t_grid_ms <= bl_win[1])
    if not np.any(baseline_mask):
        baseline_mask[:max(1, len(t_grid_ms)//4)] = True
    
    # Determine bands
    freq_max = spec_cfg.get('freq_max', 60)
    if freq_max > 60:
        bands_to_plot = list(BANDS_WITH_HIGH_GAMMA.keys())
    else:
        bands_to_plot = list(BANDS.keys())
    
    regions = [grp_name.split(' (')[0] for _, grp_name in groups]
    
    # Build matrices
    latency_matrix = np.full((len(bands_to_plot), len(regions)), np.nan)
    amplitude_matrix = np.full((len(bands_to_plot), len(regions)), np.nan)
    
    t_range = config['spectrogram'].get('time_range', (-500, 500))
    time_mask = (t_grid_ms >= t_range[0]) & (t_grid_ms <= t_range[1])
    t_plot = t_grid_ms[time_mask]
    post_mask = t_plot > 0
    
    for band_i, band_name in enumerate(bands_to_plot):
        band_dB_trials = extract_band_power_trials(P_trials, f_grid, band_name, baseline_mask)
        if band_dB_trials is None:
            continue
        
        band_dB_cropped = band_dB_trials[:, :, time_mask]
        
        for grp_i, (grp_idxs, grp_name) in enumerate(groups):
            grp_data = band_dB_cropped[:, grp_idxs, :]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                trace_mean = np.nanmean(np.nanmean(grp_data, axis=1), axis=0)
            
            if np.any(post_mask) and np.any(np.isfinite(trace_mean[post_mask])):
                post_trace = trace_mean[post_mask]
                post_t = t_plot[post_mask]
                
                # Find peak (max absolute)
                valid_post = np.isfinite(post_trace)
                if np.any(valid_post):
                    peak_idx = np.nanargmax(np.abs(post_trace))
                    latency_matrix[band_i, grp_i] = post_t[peak_idx]
                    amplitude_matrix[band_i, grp_i] = post_trace[peak_idx]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    fig.suptitle(f"Response Timing Summary: {session_id}\n{title_str}", 
                 fontsize=12, fontweight='bold')
    
    # Latency heatmap
    ax = axes[0]
    valid_latencies = latency_matrix[np.isfinite(latency_matrix)]
    if len(valid_latencies) > 0:
        vmax_lat = np.nanmax(latency_matrix)
        im1 = ax.imshow(latency_matrix, aspect='auto', cmap='viridis', 
                        vmin=0, vmax=vmax_lat)
        
        # Add text annotations
        for i in range(len(bands_to_plot)):
            for j in range(len(regions)):
                val = latency_matrix[i, j]
                if not np.isnan(val):
                    text_color = 'white' if val > vmax_lat/2 else 'black'
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center', 
                           fontsize=10, color=text_color, fontweight='bold')
        
        fig.colorbar(im1, ax=ax, label='Latency (ms)', shrink=0.8)
    
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, fontsize=11)
    ax.set_yticks(range(len(bands_to_plot)))
    ax.set_yticklabels(bands_to_plot, fontsize=11)
    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel("Frequency Band", fontsize=12)
    ax.set_title("Peak Latency (ms)", fontsize=12, fontweight='bold')
    
    # Amplitude heatmap
    ax = axes[1]
    valid_amps = amplitude_matrix[np.isfinite(amplitude_matrix)]
    if len(valid_amps) > 0:
        vmax_amp = np.nanmax(np.abs(amplitude_matrix))
        im2 = ax.imshow(amplitude_matrix, aspect='auto', cmap='RdBu_r', 
                        vmin=-vmax_amp, vmax=vmax_amp)
        
        # Add text annotations
        for i in range(len(bands_to_plot)):
            for j in range(len(regions)):
                val = amplitude_matrix[i, j]
                if not np.isnan(val):
                    text_color = 'white' if abs(val) > vmax_amp/2 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                           fontsize=10, color=text_color, fontweight='bold')
        
        fig.colorbar(im2, ax=ax, label='Amplitude (dB)', shrink=0.8)
    
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, fontsize=11)
    ax.set_yticks(range(len(bands_to_plot)))
    ax.set_yticklabels(bands_to_plot, fontsize=11)
    ax.set_xlabel("Region", fontsize=12)
    ax.set_title("Peak Amplitude (dB)", fontsize=12, fontweight='bold')
    
    # Save
    out_dir = fig_dir / "Timing_Summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"timing_{session_id}.png"
    plt.savefig(out_path, dpi=config['summary']['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved timing summary -> {out_path.name}")
    
    return out_path


# =============================================================================
# SIMPLIFIED SESSION SUMMARY (PSD + Band Stats only)
# =============================================================================

def generate_session_summary(data, session_id, title_str, fig_dir, groups, fs, config):
    """
    Generate a simplified session summary with:
    - Row 0: PSD (pre vs post) per region
    - Row 1: PSD Change (post - pre) per region  
    - Row 2: Band power bar chart (pre vs post) per region
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
    
    # Create figure
    fig = plt.figure(figsize=(5 * n_groups, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, n_groups, height_ratios=[1, 1, 1.2])
    
    fig.suptitle(f"Session Summary: {session_id}\n{title_str}", fontsize=14, fontweight='bold')
    
    # Determine frequency bands to use
    freq_max = config['spectrogram'].get('freq_max', 60)
    if freq_max > 60:
        bands_to_use = BANDS_WITH_HIGH_GAMMA
    else:
        bands_to_use = BANDS
    
    for col, (grp_idxs, grp_name) in enumerate(groups):
        short_name = grp_name.split(' (')[0]
        color = REGION_COLORS.get(short_name, f'C{col}')
        
        # Row 0: PSD
        ax = fig.add_subplot(gs[0, col])
        g_pre = np.nanmean(psd_pre_all[grp_idxs, :], axis=0)
        g_post = np.nanmean(psd_post_all[grp_idxs, :], axis=0)
        
        ax.plot(f_pre, g_pre, color='blue', lw=2, label='Pre')
        ax.plot(f_post, g_post, color='red', lw=2, label='Post')
        
        # Add band shading
        for band_name, (f_lo, f_hi) in bands_to_use.items():
            band_color = BAND_COLORS.get(band_name, 'gray')
            ax.axvspan(f_lo, f_hi, color=band_color, alpha=0.1)
        
        ax.set_title(f"{short_name}\nPSD", fontsize=11, fontweight='bold')
        ax.set_xlim(0, min(120, freq_max + 10))
        ax.set_ylim(-20, 40)
        ax.set_ylabel("Power (dB)" if col == 0 else "")
        ax.set_xlabel("Freq (Hz)")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(loc='upper right', fontsize='small')
        
        # Row 1: PSD Change
        ax = fig.add_subplot(gs[1, col])
        ratio_db = g_post - g_pre  # Already in dB
        
        ax.plot(f_pre, ratio_db, color='k', lw=2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax.fill_between(f_pre, 0, ratio_db, where=(ratio_db > 0), 
                       color='red', alpha=0.3, interpolate=True)
        ax.fill_between(f_pre, 0, ratio_db, where=(ratio_db < 0), 
                       color='blue', alpha=0.3, interpolate=True)
        
        # Add band shading
        for band_name, (f_lo, f_hi) in bands_to_use.items():
            band_color = BAND_COLORS.get(band_name, 'gray')
            ax.axvspan(f_lo, f_hi, color=band_color, alpha=0.1)
        
        ax.set_title(f"PSD Change (Post-Pre)", fontsize=10)
        ax.set_xlim(0, min(120, freq_max + 10))
        ax.set_ylim(-10, 10)
        ax.set_ylabel("ΔdB" if col == 0 else "")
        ax.set_xlabel("Freq (Hz)")
        ax.grid(True, alpha=0.3)
        
        # Row 2: Band power bar chart
        ax = fig.add_subplot(gs[2, col])
        
        band_names = list(bands_to_use.keys())
        x_pos = np.arange(len(band_names))
        width = 0.35
        
        # Calculate band power
        means_pre, means_post = [], []
        sems_pre, sems_post = [], []
        
        for band_name, (f_min, f_max) in bands_to_use.items():
            idx_band = np.where((f_pre >= f_min) & (f_pre <= f_max))[0]
            
            if len(idx_band) > 0:
                # Get power in this band for this group
                bp_pre = np.nanmean(psd_pre_all[grp_idxs, :][:, idx_band], axis=1)
                bp_post = np.nanmean(psd_post_all[grp_idxs, :][:, idx_band], axis=1)
                
                means_pre.append(np.nanmean(bp_pre))
                means_post.append(np.nanmean(bp_post))
                sems_pre.append(np.nanstd(bp_pre) / np.sqrt(len(grp_idxs)))
                sems_post.append(np.nanstd(bp_post) / np.sqrt(len(grp_idxs)))
            else:
                means_pre.append(0)
                means_post.append(0)
                sems_pre.append(0)
                sems_post.append(0)
        
        ax.bar(x_pos - width/2, means_pre, width, yerr=sems_pre,
               label='Pre', color='blue', alpha=0.6, capsize=3)
        ax.bar(x_pos + width/2, means_post, width, yerr=sems_post,
               label='Post', color='red', alpha=0.6, capsize=3)
        
        ax.set_title(f"Band Power", fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([b[:3] for b in band_names], fontsize=8, rotation=45)
        ax.set_ylabel("Power (dB)" if col == 0 else "")
        ax.grid(axis='y', alpha=0.3)
        if col == 0:
            ax.legend(fontsize='small')
    
    # Save
    out_dir = fig_dir / "Session_Summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"summary_{session_id}.png"
    plt.savefig(out_path, dpi=config['summary']['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved session summary -> {out_path.name}")
    
    return out_path


# =============================================================================
# SPECTROGRAM GRID (Optional - 8x8 layout)
# =============================================================================

def generate_spectrogram_grid(ersp_all, freqs, t_bins_ms, ua_ids_1based, session_id,
                              region_name, grid_elec, elec_to_idx, fig_dir, spec_cfg):
    """
    Plot spectrograms in 8x8 grid matching physical electrode layout.
    """
    if grid_elec is None:
        return None
    
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
    plt.savefig(out_path, dpi=spec_cfg.get('dpi', 120), bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved spectrogram grid -> {out_path.name}")
    
    return out_path


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_session(npz_file, fig_dir, label, df_metadata, bad_ch_map, nsp_to_elec, elec_to_region):
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
    
    # Extract BR index from filename for metadata lookup
    br_idx = extract_br_idx_from_filename(npz_file.name)
    
    # Lookup stimulation metadata
    stim_info = lookup_stim_info(br_idx, df_metadata)
    
    # Determine if baseline
    is_baseline = "baseline" in npz_file.stem.lower()
    
    # Build session ID for filenames
    if is_baseline:
        session_id = f"{session}_baseline".replace('.ns6', '')
    else:
        target_suffix = f"_{category}"
        if target and target != 'none':
            target_suffix += f"_target_{target}"
        session_id = f"{session}{target_suffix}".replace('.ns6', '')
    
    # Build title string
    title_str = build_title_string(session, category, target, stim_info, is_baseline)
    
    # Check for required data
    if 'broadband_post' not in data:
        print(f"    [Skip] Missing broadband_post")
        return
    
    bb_data = data['broadband_post']
    n_trials, n_ch, n_time = bb_data.shape
    print(f"    Dimensions: {n_trials} trials, {n_ch} channels")
    print(f"    Stim info: {stim_info['n_pulses']} pulses @ {stim_info['frequency']} Hz, {stim_info['amplitude']} µA")
    
    # Determine groups (Utah regions vs NPRW channel blocks)
    is_utah = (label == "Utah_Array")
    groups = []
    ua_port = 'A'
    ua_ids_1based = None
    
    if is_utah:
        # Determine port from metadata
        try:
            if br_idx is not None and cfg.METADATA_CSV.exists():
                with open(cfg.METADATA_CSV, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        br_val = row.get('BR_File', row.get('BR_file', ''))
                        if br_val and int(br_val) == br_idx:
                            p = row.get('UA_port', row.get('ua_port', '')).strip().upper()
                            if p in ['A', 'B']:
                                ua_port = p
                            break
        except Exception as e:
            print(f"    [warn] Could not determine UA port: {e}")
        
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
            
            # Add groups in consistent order
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
        for g in range(min(8, (n_ch + ch_per_group - 1) // ch_per_group)):
            start = g * ch_per_group
            end = min(start + ch_per_group, n_ch)
            if start < n_ch:
                groups.append((list(range(start, end)), f"Ch {start}-{end-1}"))
    
    # Collect all figure paths for PDF
    figure_paths = []
    
    # Generate figures based on config
    
    # NEW: Band Dynamics (main figure you want)
    if FIGURE_CONFIG.get('generate_band_dynamics', False):
        path = generate_band_dynamics(data, session_id, title_str, fig_dir, groups, fs, PLOT_CONFIG)
        if path:
            figure_paths.append(path)
    
    # NEW: Timing Heatmap
    if FIGURE_CONFIG.get('generate_timing_heatmap', False):
        path = generate_timing_heatmap(data, session_id, title_str, fig_dir, groups, fs, PLOT_CONFIG)
        if path:
            figure_paths.append(path)
    
    # Session Summary (simplified)
    if FIGURE_CONFIG.get('generate_session_summary', False):
        path = generate_session_summary(data, session_id, title_str, fig_dir, groups, fs, PLOT_CONFIG)
        if path:
            figure_paths.append(path)
    
    # Spectrogram Grid (optional)
    if FIGURE_CONFIG.get('generate_spectrogram_grid', False) and is_utah:
        d_bb_full = data.get('broadband_full')
        if d_bb_full is not None and ua_ids_1based is not None:
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
    
    # Generate PDF report
    if FIGURE_CONFIG.get('generate_pdf_report', False) and figure_paths:
        generate_pdf_report(session_id, title_str, fig_dir, figure_paths, PLOT_CONFIG)
    
    return figure_paths


def generate_pdf_report(session_id, title_str, fig_dir, figure_paths, config):
    """Combine all generated figures into a multi-page PDF report."""
    pdf_dir = fig_dir / "Reports"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"report_{session_id}.pdf"
    
    valid_paths = [p for p in figure_paths if p is not None and p.exists()]
    
    if not valid_paths:
        print("    [Skip] No figures to include in PDF")
        return None
    
    with PdfPages(pdf_path) as pdf:
        # Title page
        fig_title = plt.figure(figsize=(11, 8.5))
        fig_title.text(0.5, 0.6, "LFP Analysis Report", fontsize=24, ha='center', fontweight='bold')
        fig_title.text(0.5, 0.5, session_id, fontsize=18, ha='center')
        fig_title.text(0.5, 0.4, title_str, fontsize=12, ha='center', style='italic')
        fig_title.text(0.5, 0.2, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                      fontsize=10, ha='center', alpha=0.7)
        pdf.savefig(fig_title)
        plt.close(fig_title)
        
        # Add each figure
        for fig_path in valid_paths:
            if fig_path.suffix.lower() == '.gif':
                continue
            
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
        
        d = pdf.infodict()
        d['Title'] = f'LFP Analysis Report: {session_id}'
        d['Author'] = 'Automated Analysis Pipeline'
        d['CreationDate'] = datetime.datetime.now()
    
    print(f"    Saved PDF report -> {pdf_path.name}")
    return pdf_path


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
    
    # Load metadata CSV
    df_metadata = load_metadata_df()
    if df_metadata is not None:
        print(f"Loaded metadata with {len(df_metadata)} rows")
        print(f"Columns: {df_metadata.columns.tolist()}")
    
    # Load impedance data
    bad_ch_map = imp_utils.get_session_impedances(cfg.SESSION_LOC)
    
    # Process files
    for npz_file in tqdm(files, desc=f"Processing {label}"):
        process_session(npz_file, fig_dir, label, df_metadata, bad_ch_map,
                       nsp_to_elec_global, elec_to_region_global)


def plot_lfp_check():
    """Main entry point."""
    process_directory(NPRW_LFP_DIR, NPRW_FIG_DIR, label="NPRW_Intan")
    process_directory(UA_LFP_DIR, UA_FIG_DIR, label="Utah_Array")


if __name__ == "__main__":
    plot_lfp_check()