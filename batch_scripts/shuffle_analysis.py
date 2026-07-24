from pathlib import Path
import numpy as np
from typing import Tuple, Dict, Optional, List
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for faster rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from joblib import Parallel, delayed
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

PROCESS_ONLY = PARAMS.preprocessing.get("process_only")
from RCP_analysis.python.functions.impedance_utils import get_session_impedances
from itertools import groupby
from operator import itemgetter
import pandas as pd
from functools import lru_cache
import scipy.stats

# =============================================================================
# CONFIGURATION FLAGS
# =============================================================================

# ANALYSIS MODE SELECTION
# Set ONE of these to True:
RUN_PLOTTING_ANALYSIS = False      # Original raster/PSTH plotting
RUN_SHUFFLE_ANALYSIS = True       # New shuffle-based statistical analysis

# Which probe types to process
PROCESS_NPRW = True          # Set False to skip NPRW plots
PROCESS_UA = True            # Set False to skip Utah Array plots
PROCESS_DIFF_PLOTS = True    # Set False to skip difference plots (Stim - Control, etc.)

# Which plot views to generate
GENERATE_STANDARD_VIEW = True   # (-500, 500) ms window
GENERATE_ZOOM_VIEW = True       # Dynamic zoom window with 5ms bins

# Which views get difference plots
DIFF_PLOTS_STANDARD = True      # Generate diff plots for standard (-500, 500) view
DIFF_PLOTS_ZOOM = True          # Generate diff plots for zoom view

# Parallel processing settings
N_JOBS = 8                   # Number of parallel jobs (-1 for all cores)
PARALLEL_BACKEND = 'loky'    # 'loky', 'threading', or 'multiprocessing'

# Plot quality settings
DPI_OUTPUT = 100             # Lower = faster, smaller files (try 72 for drafts)
SKIP_EXISTING = False        # Skip files that already have output plots

# Debug/verbose output
VERBOSE = False               # Print debug messages

# =============================================================================
# SHUFFLE ANALYSIS SETTINGS
# =============================================================================

SHUFFLE_N_ITERATIONS = 1000      # Number of shuffle iterations
SHUFFLE_ALPHA = 0.05             # Significance threshold (before correction)
SHUFFLE_USE_FDR = True           # Use FDR correction (Benjamini-Hochberg)
                                 # If False, uses Bonferroni correction

# Region-specific analysis windows (ms relative to event at t=0)
# For UA channels based on their region
REGION_ANALYSIS_WINDOWS = {
    'PMd': (-500.0, 0.0),
    'SMA': (-500.0, 0.0),
    'M1i': (-300.0, 300.0),
    'M1s': (-300.0, 300.0),
}

# Default window for NPRW (no region info available)
NPRW_ANALYSIS_WINDOW = (-500.0, 0.0)

# Bin size for shuffle analysis (use same as standard view)
SHUFFLE_BIN_MS = 20.0  # Match the default bin size in the existing code

# =============================================================================
# COMMON STATISTICAL ANALYSIS WINDOWS
# =============================================================================

COMMON_ANALYSIS_WINDOWS = {
    "baseline": (-700.0, -500.0),
    "prep":     (-500.0, -200.0),
    "movement": (-200.0, 300.0),
    "post":     (300.0, 500.0),
    "full":     (-700.0, 500.0),
}

# NPRW stimulation-period blanking.
# NPRW data on stimulation trials are blanked during stimulation itself.
# The blank end is determined per file/condition from stimulation duration metadata.
NPRW_STIM_BLANK_START_MS = 0.0

EXCLUDE_NPRW_STIM_BLANK_FROM_STATS = True
EXCLUDE_NPRW_STIM_BLANK_FROM_PLOTS = True
# =============================================================================
# PLOT SETTINGS
# =============================================================================

# Time window for plots
WIN_PLOT_MS = (-500.0, 500.0)

# Build PLOT_VIEWS based on configuration
# Format: (window, bin_ms, suffix, generate_diff)
PLOT_VIEWS = []
if GENERATE_STANDARD_VIEW:
    PLOT_VIEWS.append(((-500.0, 500.0), None, '', DIFF_PLOTS_STANDARD))
if GENERATE_ZOOM_VIEW:
    PLOT_VIEWS.append((None, 5.0, '_zoom', DIFF_PLOTS_ZOOM))

# Raster dots style
RASTER_MARKER = 'o'
RASTER_MARKER_SIZE = 2
RASTER_COLOR = 'tab:blue'
RASTER_ALPHA = 0.8

# PSTH style
PSTH_COLOR = 'tab:cyan'
PSTH_EDGE_COLOR = 'k'
PSTH_LINEWIDTH = 0.5
PSTH_YLIM = (0, 100)
PSTH_YLIM_ZOOM = (0, 50)

# Difference plot y-limits (now in z-score units)
DIFF_YLIM = (-4, 6)  # z-score units (standard deviations from baseline)
DIFF_YLIM_STANDARD = (-4, 6)  # For standard view
DIFF_YLIM_ZOOM = (-4, 6)      # For zoom view

# Figure sizes
FIG_SIZE_RW = (48, 24)
FIG_SIZE_UA = (32, 24)

# Stimulation region shading
STIM_REGION_COLOR = 'grey'
STIM_REGION_ALPHA = 0.3

# Output directory
FIG_ROOT = OUT_BASE / "figures" / "peristim_raster"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

# Shuffle analysis output directory
SHUFFLE_OUT_ROOT = OUT_BASE / "figures" / "shuffle_analysis"
SHUFFLE_OUT_ROOT.mkdir(parents=True, exist_ok=True)

PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim"


# =============================================================================
# ELECTRODE MAPPING (Cached)
# =============================================================================

@lru_cache(maxsize=1)
def load_electrode_mapping_cached(csv_path_str: str):
    """
    Load electrode mapping from CSV (cached).
    """
    csv_path = Path(csv_path_str)
    elec_info = {}
    region_grids = {}

    if not csv_path.exists():
        print(f"  [Warn] Electrode mapping CSV not found: {csv_path}")
        return elec_info, region_grids

    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            elec_id = int(row['ElectrodeID'])
            nsp_id = int(row['NSP_ID'])  # Global NSP ID (1-256)
            region = str(row['Array']).strip()
            port_str = str(row['Port']).strip()
            r = int(row['GridRow'])
            c = int(row['GridCol'])
            
            # Determine port from string
            port = 'A' if 'A' in port_str.upper() else 'B'

            elec_info[elec_id] = {
                'nsp_id': nsp_id,      # Global NSP ID (1-256)
                'port': port,           # 'A' or 'B'
                'region': region,
                'row': r,
                'col': c,
            }

            if region not in region_grids:
                region_grids[region] = np.zeros((8, 8), dtype=int)
            region_grids[region][r, c] = elec_id
            
    except Exception as e:
        print(f"  [Error] Failed to load electrode mapping from {csv_path}: {e}")

    return elec_info, region_grids

def load_electrode_mapping(csv_path: Path):
    """Wrapper to call cached version with string path."""
    return load_electrode_mapping_cached(str(csv_path))


def get_time_axis_for_probe(data, probe: str):
    """
    Get bin centers for NPRW or UA.
    """
    rel_key = f'{probe}_rel_t'
    edge_key = f'{probe}_edges_ms'

    if rel_key in data:
        return np.asarray(data[rel_key])

    if edge_key in data:
        edges = np.asarray(data[edge_key])
        return (edges[:-1] + edges[1:]) / 2.0

    return None



def build_elec_to_data_idx(ua_ids_1based, elec_info, recording_port='A'):
    """
    Build electrode_id -> channel index mapping.
    
    For Port A: data channels 0-127 correspond to NSP 1-128
    For Port B: data channels 0-127 correspond to NSP 129-256
    
    Parameters
    ----------
    ua_ids_1based : array or None
        If provided, the actual NSP IDs in the data (1-based), or Electrode IDs.
        If None, assumes sequential mapping based on port.
    elec_info : dict
        Electrode info from CSV: {elec_id: {'nsp_id': ..., 'port': ..., 'region': ..., ...}}
    recording_port : str
        'A' or 'B'
    
    Returns
    -------
    dict
        {electrode_id: channel_index} mapping
    """
    elec_to_idx = {}
    recording_port = recording_port.upper()
    
    port_elecs = {elec_id for elec_id, info in elec_info.items() if info['port'] == recording_port}
    
    # Check if the values in ua_ids_1based are physical Electrode IDs for this port
    is_elec_ids = False
    if ua_ids_1based is not None and len(ua_ids_1based) > 0:
        valid_elecs_count = sum(1 for x in ua_ids_1based if int(x) in port_elecs)
        if valid_elecs_count > len(ua_ids_1based) * 0.5:
            is_elec_ids = True
            
    if is_elec_ids:
        for ch_idx, elec_id_raw in enumerate(ua_ids_1based):
            try:
                elec_id = int(elec_id_raw)
                elec_to_idx[elec_id] = ch_idx
            except (ValueError, TypeError):
                continue
    elif ua_ids_1based is not None and len(ua_ids_1based) > 0:
        # Build reverse lookup: NSP_ID -> electrode_id
        nsp_to_elec = {}
        for elec_id, info in elec_info.items():
            nsp_to_elec[info['nsp_id']] = elec_id
        
        # Determine NSP offset based on port
        nsp_offset = 0 if recording_port == 'A' else 128
        
        # Use the actual channel IDs from the data as NSP IDs
        for ch_idx, nsp_id_raw in enumerate(ua_ids_1based):
            try:
                nsp_id = int(nsp_id_raw)
            except (ValueError, TypeError):
                continue
            
            # If the data contains local IDs (1-128), add the port offset
            if 1 <= nsp_id <= 128:
                global_nsp_id = nsp_id + nsp_offset
            else:
                # Already global (129-256 range)
                global_nsp_id = nsp_id
            
            elec_id = nsp_to_elec.get(global_nsp_id)
            if elec_id is not None:
                elec_to_idx[elec_id] = ch_idx
    else:
        # No ua_ids_1based - assume sequential mapping
        # Build reverse lookup: NSP_ID -> electrode_id
        nsp_to_elec = {}
        for elec_id, info in elec_info.items():
            nsp_to_elec[info['nsp_id']] = elec_id
            
        nsp_offset = 0 if recording_port == 'A' else 128
        # Channel 0 -> NSP 1 (Port A) or NSP 129 (Port B)
        for ch_idx in range(128):
            global_nsp_id = ch_idx + 1 + nsp_offset
            elec_id = nsp_to_elec.get(global_nsp_id)
            if elec_id is not None:
                elec_to_idx[elec_id] = ch_idx
    
    return elec_to_idx


def get_active_regions(ua_ids_1based, elec_info, recording_port='A'):
    """
    Determine which regions have active channels in this recording.
    """
    REGION_ORDER = ["SMA", "PMd", "M1i", "M1s"]
    recording_port = recording_port.upper()
    
    if ua_ids_1based is None or len(ua_ids_1based) == 0:
        # Fallback: return all regions for this port
        active = set()
        for elec_id, info in elec_info.items():
            if info['port'] == recording_port:
                active.add(info['region'])
        return [r for r in REGION_ORDER if r in active]
    
    port_elecs = {elec_id for elec_id, info in elec_info.items() if info['port'] == recording_port}
    
    # Check if the values in ua_ids_1based are physical Electrode IDs
    is_elec_ids = False
    valid_elecs_count = sum(1 for x in ua_ids_1based if int(x) in port_elecs)
    if valid_elecs_count > len(ua_ids_1based) * 0.5:
        is_elec_ids = True
        
    active_regions = set()
    if is_elec_ids:
        for elec_id_raw in ua_ids_1based:
            try:
                elec_id = int(elec_id_raw)
            except (ValueError, TypeError):
                continue
            
            info = elec_info.get(elec_id)
            if info is not None:
                active_regions.add(info['region'])
    else:
        # Build NSP -> electrode lookup (all electrodes)
        nsp_to_elec = {info['nsp_id']: elec_id for elec_id, info in elec_info.items()}
        nsp_offset = 0 if recording_port == 'A' else 128
        
        for nsp_id_raw in ua_ids_1based:
            try:
                nsp_id = int(nsp_id_raw)
            except (ValueError, TypeError):
                continue
            
            # Try direct lookup
            elec_id = nsp_to_elec.get(nsp_id)
            
            # Try with offset if needed
            if elec_id is None and 1 <= nsp_id <= 128:
                elec_id = nsp_to_elec.get(nsp_id + nsp_offset)
            
            if elec_id is not None:
                region = elec_info[elec_id]['region']
                active_regions.add(region)
    
    return [r for r in REGION_ORDER if r in active_regions]


def get_electrodes_for_port(elec_info, region, recording_port):
    """
    Get list of electrode IDs in a region that belong to a specific port.
    
    Returns set of electrode IDs.
    """
    return {
        elec_id for elec_id, info in elec_info.items()
        if info['region'] == region and info['port'] == recording_port.upper()
    }


# Load mapping globally 
MAPPING_CSV = Path(__file__).parent.parent / "config" / "electrode_port_mapping.csv"
elec_info_global, UTAH_ELEC_GRIDS = load_electrode_mapping(MAPPING_CSV)


# =============================================================================
# STIM DURATION / NPRW BLANKING HELPERS
# =============================================================================

def _scalar_or_none(x):
    """
    Convert common scalar-like objects into float-compatible scalar.
    Returns None if empty/unusable.
    """
    if x is None:
        return None

    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return None
        val = arr.ravel()[0]
    except Exception:
        val = x

    try:
        if val is None:
            return None
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        if isinstance(val, str) and val.strip() == "":
            return None
        return float(val)
    except Exception:
        return None


def _get_npz_item(data, key):
    """
    Safely get a value from npz/dict-like data.
    """
    try:
        if key in data:
            return data[key]
    except Exception:
        pass
    return None


def _dict_like_get(obj, key):
    """
    Safe get for dict-like or structured metadata objects.
    """
    if obj is None:
        return None

    if isinstance(obj, dict):
        return obj.get(key, None)

    try:
        if hasattr(obj, "dtype") and obj.dtype.names and key in obj.dtype.names:
            return obj[key]
    except Exception:
        pass

    try:
        return getattr(obj, key)
    except Exception:
        return None


def get_stim_duration_ms(data, metadata_df=None, meta_br_idx=None):
    """
    Determine stimulation duration in ms using the same preferred logic as the
    plotting/reference path.

    Order:
      1. metadata_df row where BR_File == meta_br_idx, column Stim_Duration_ms
      2. metadata_df row by positional meta_br_idx if BR_File unavailable
      3. top-level/common npz duration keys
      4. meta['recording_stim_dur']
      5. nprw_meta['stim_dur']
    """

    # 1. metadata_df by BR_File
    if metadata_df is not None and meta_br_idx is not None:
        try:
            if "BR_File" in metadata_df.columns and "Stim_Duration_ms" in metadata_df.columns:
                rows = metadata_df[metadata_df["BR_File"] == meta_br_idx]
                if len(rows) > 0:
                    val = _scalar_or_none(rows.iloc[0]["Stim_Duration_ms"])
                    if val is not None and np.isfinite(val):
                        return val
        except Exception:
            pass

        # 2. metadata_df by positional index fallback
        try:
            if "Stim_Duration_ms" in metadata_df.columns:
                if int(meta_br_idx) in metadata_df.index:
                    val = _scalar_or_none(metadata_df.loc[int(meta_br_idx), "Stim_Duration_ms"])
                    if val is not None and np.isfinite(val):
                        return val
        except Exception:
            pass

    # 3. direct/common npz keys
    direct_keys = [
        "Stim_Duration_ms",
        "stim_duration_ms",
        "stimulation_duration_ms",
        "stim_dur_ms",
        "stimDur_ms",
        "stim_duration",
        "stim_dur",
        "pulse_train_duration_ms",
        "train_duration_ms",
    ]

    for key in direct_keys:
        val = _scalar_or_none(_get_npz_item(data, key))
        if val is not None and np.isfinite(val):
            return val

    # 4. meta['recording_stim_dur']
    meta = _get_npz_item(data, "meta")
    if meta is not None:
        try:
            # common case: object array containing dict
            if isinstance(meta, np.ndarray) and meta.dtype == object:
                meta_obj = meta.item()
            else:
                meta_obj = meta
        except Exception:
            meta_obj = meta

        val = _scalar_or_none(_dict_like_get(meta_obj, "recording_stim_dur"))
        if val is not None and np.isfinite(val):
            return val

        val = _scalar_or_none(_dict_like_get(meta_obj, "Stim_Duration_ms"))
        if val is not None and np.isfinite(val):
            return val

    # 5. nprw_meta['stim_dur']
    nprw_meta = _get_npz_item(data, "nprw_meta")
    if nprw_meta is not None:
        try:
            if isinstance(nprw_meta, np.ndarray) and nprw_meta.dtype == object:
                nprw_meta_obj = nprw_meta.item()
            else:
                nprw_meta_obj = nprw_meta
        except Exception:
            nprw_meta_obj = nprw_meta

        for key in ["stim_dur", "stim_duration", "Stim_Duration_ms", "stim_duration_ms"]:
            val = _scalar_or_none(_dict_like_get(nprw_meta_obj, key))
            if val is not None and np.isfinite(val):
                return val

    return None


def get_nprw_blank_window_ms(
    stim_data,
    metadata_df=None,
    meta_br_idx=None,
    default_blank_end_ms=None,
):
    """
    Return NPRW blanking window as:

        (0.0, stim_duration_ms)

    or None if duration cannot be determined.
    """

    stim_duration_ms = get_stim_duration_ms(
        stim_data,
        metadata_df=metadata_df,
        meta_br_idx=meta_br_idx,
    )

    if stim_duration_ms is None or not np.isfinite(stim_duration_ms):
        stim_duration_ms = default_blank_end_ms

    if stim_duration_ms is None or not np.isfinite(stim_duration_ms):
        return None

    if stim_duration_ms <= NPRW_STIM_BLANK_START_MS:
        return None

    return (float(NPRW_STIM_BLANK_START_MS), float(stim_duration_ms))




# =============================================================================
# OPTIMIZED DATA PROCESSING FUNCTIONS
# =============================================================================

def _get_raster_data_fast(peak_times_ms, events_ms, win_ms, blank_pre_ms=0.0, blank_post_ms=0.0):
    """
    Optimized raster data gathering using vectorized operations.
    """
    if peak_times_ms is None or len(peak_times_ms) == 0 or events_ms is None or len(events_ms) == 0:
        return [], []

    peak_times_ms = np.asarray(peak_times_ms, dtype=np.float64)
    events_ms = np.asarray(events_ms, dtype=np.float64)
    
    if peak_times_ms.size == 0 or events_ms.size == 0:
        return [], []
    
    peak_times_ms = np.sort(peak_times_ms)
    
    trial_indices = []
    rel_times = []
    
    # Pre-compute window bounds for all events
    t_starts = events_ms + win_ms[0]
    t_ends = events_ms + win_ms[1]
    
    # Use searchsorted for efficient spike finding
    idx_starts = np.searchsorted(peak_times_ms, t_starts)
    idx_ends = np.searchsorted(peak_times_ms, t_ends)
    
    for i, (idx_s, idx_e, t_stim) in enumerate(zip(idx_starts, idx_ends, events_ms)):
        if idx_e > idx_s:
            spikes = peak_times_ms[idx_s:idx_e]
            rel = spikes - t_stim
            
            if blank_pre_ms > 0 or blank_post_ms > 0:
                keep = (rel < -blank_pre_ms) | (rel > blank_post_ms)
                rel = rel[keep]
            
            if len(rel) > 0:
                trial_indices.extend([i] * len(rel))
                rel_times.extend(rel.tolist())
            
    return trial_indices, rel_times


def _rebin_from_peaks_fast(peaks_dict, events_ms, win_ms, bin_ms, stim_dur_ms=0.0,
                           blank_pre_ms=0.0, blank_post_ms=0.0):
    """
    Optimized re-binning using vectorized numpy operations.
    """
    # Sort channel keys once
    try:
        ch_keys = sorted(peaks_dict.keys(), key=lambda x: int(x))
    except:
        ch_keys = sorted(peaks_dict.keys())
    
    n_ch = len(ch_keys)
    events_ms = np.asarray(events_ms, dtype=np.float64).ravel()
    n_trials = events_ms.size

    if n_trials == 0 or n_ch == 0:
        edges = np.arange(win_ms[0], win_ms[1] + bin_ms * 0.5, bin_ms)
        centers = edges[:-1] + bin_ms / 2.0
        return np.zeros((max(n_trials, 1), max(n_ch, 1), len(centers)), dtype=np.float32), centers, edges

    edges = np.arange(win_ms[0], win_ms[1] + bin_ms * 0.5, bin_ms)
    centers = edges[:-1] + bin_ms / 2.0
    n_bins = centers.size

    # Use float32 for memory efficiency
    counts = np.zeros((n_trials, n_ch, n_bins), dtype=np.float32)

    # Pre-compute window bounds
    win_starts = events_ms + win_ms[0]
    win_ends = events_ms + win_ms[1]

    for ch_i, ch_id in enumerate(ch_keys):
        spk = peaks_dict.get(ch_id)
        if spk is None or len(spk) == 0:
            continue
            
        spk = np.asarray(spk, dtype=np.float64).ravel()
        if spk.size == 0:
            continue
            
        spk = np.sort(spk)
        
        # Find spikes for all trials at once using searchsorted
        idx_starts = np.searchsorted(spk, win_starts)
        idx_ends = np.searchsorted(spk, win_ends)
        
        for t_i in range(n_trials):
            if idx_ends[t_i] > idx_starts[t_i]:
                rel = spk[idx_starts[t_i]:idx_ends[t_i]] - events_ms[t_i]
                bin_idx = np.searchsorted(edges, rel, side='right') - 1
                valid = (bin_idx >= 0) & (bin_idx < n_bins)
                np.add.at(counts[t_i, ch_i], bin_idx[valid], 1.0)

    return counts, centers, edges


def compute_zscore_diff(stim_counts, baseline_counts, min_std=0.1):
    """
    Compute z-scored difference between stim and baseline.
    
    Z = (stim_mean - baseline_mean) / baseline_std
    
    Parameters
    ----------
    stim_counts : ndarray
        Shape (n_trials, n_channels, n_bins) - stimulus condition counts
    baseline_counts : ndarray
        Shape (n_trials, n_channels, n_bins) - baseline condition counts
    min_std : float
        Minimum standard deviation to avoid division by zero
    
    Returns
    -------
    z_diff : ndarray
        Shape (n_channels, n_bins) - z-scored difference
    stim_mean : ndarray
        Shape (n_channels, n_bins) - mean of stim counts
    baseline_mean : ndarray
        Shape (n_channels, n_bins) - mean of baseline counts
    baseline_std : ndarray
        Shape (n_channels, n_bins) - std of baseline counts
    """
    # Compute means across trials
    stim_mean = np.nanmean(stim_counts, axis=0)  # (n_channels, n_bins)
    baseline_mean = np.nanmean(baseline_counts, axis=0)  # (n_channels, n_bins)
    
    # Compute std across trials for baseline
    baseline_std = np.nanstd(baseline_counts, axis=0)  # (n_channels, n_bins)
    
    # Avoid division by zero
    baseline_std = np.maximum(baseline_std, min_std)
    
    # Compute z-score
    z_diff = (stim_mean - baseline_mean) / baseline_std
    
    return z_diff, stim_mean, baseline_mean, baseline_std


def plot_channel_group_fast(ax_raster, ax_psth, 
                            peak_times, events_ms, 
                            binned_counts, bin_edges, 
                            title, stim_dur_ms=0,
                            blank_pre_ms=0.0, blank_post_ms=0.0,
                            bin_centers=None,
                            n_trials_ref=0,
                            win_ms=WIN_PLOT_MS,
                            psth_ylim=PSTH_YLIM):
    """
    Optimized channel plotting with reduced draw calls.
    """
    rel_times = []
    n_trials = n_trials_ref
    
    # 1. Raster
    if events_ms is not None and len(events_ms) > 0:
        blank_post_total = stim_dur_ms + blank_post_ms if stim_dur_ms > 0 else 0.0
        blank_pre_total = blank_pre_ms if stim_dur_ms > 0 else 0.0
        
        trial_indices, rel_times = _get_raster_data_fast(
            peak_times, events_ms, win_ms,
            blank_pre_ms=blank_pre_total,
            blank_post_ms=blank_post_total
        )
        
        if len(rel_times) > 0:
            ax_raster.scatter(rel_times, trial_indices, 
                              s=RASTER_MARKER_SIZE, marker=RASTER_MARKER, 
                              c=RASTER_COLOR, alpha=RASTER_ALPHA,
                              rasterized=True)  # Rasterize for faster rendering
        n_trials = len(events_ms)

    has_data = (len(rel_times) > 0) or (binned_counts is not None and np.nansum(binned_counts) > 0)
    
    # Add grey shaded region for stimulation period
    if stim_dur_ms > 0:
        ax_raster.axvspan(0, stim_dur_ms, color=STIM_REGION_COLOR, alpha=STIM_REGION_ALPHA, zorder=0)
        ax_psth.axvspan(0, stim_dur_ms, color=STIM_REGION_COLOR, alpha=STIM_REGION_ALPHA, zorder=0)
    
    if has_data:
        ax_raster.axvline(0, color='r', linestyle='--', linewidth=1, alpha=0.8)
    
    ax_raster.set_xlim(win_ms)
    ax_raster.set_ylim(-0.5, max(n_trials - 0.5, 0.5))
    ax_raster.set_title(title, fontsize=10, pad=4)
    ax_raster.axis('off') 
    
    # 2. PSTH
    if binned_counts is not None and binned_counts.size > 0:
        psth_counts = np.nansum(binned_counts, axis=0)
        
        if bin_centers is not None and len(psth_counts) == len(bin_centers):
            width = 20.0 
            if len(bin_centers) > 1:
                dt = np.nanmedian(np.diff(bin_centers))
                if dt > 0: 
                    width = dt
            
            ax_psth.bar(bin_centers, psth_counts, width=width, align='center',
                        color=PSTH_COLOR, edgecolor=PSTH_EDGE_COLOR, linewidth=PSTH_LINEWIDTH)
                        
        elif bin_edges is not None and len(psth_counts) == len(bin_edges) - 1:
            ax_psth.bar(bin_edges[:-1], psth_counts, width=np.diff(bin_edges), align='edge',
                        color=PSTH_COLOR, edgecolor=PSTH_EDGE_COLOR, linewidth=PSTH_LINEWIDTH)

    if has_data:
        ax_psth.axvline(0, color='r', linestyle='--', linewidth=1, alpha=0.8)
    
    ax_psth.set_xlim(win_ms)
    ax_psth.set_ylim(psth_ylim)
    
    for spine in ['top', 'right']:
        ax_psth.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax_psth.spines[spine].set_linewidth(0.5)
    ax_psth.tick_params(axis='both', which='major', labelsize=6, width=0.5, length=2)


# =============================================================================
# REFERENCE DATA LOADING
# =============================================================================

def load_reference_data(all_files, metadata_df=None, for_shuffle=False):
    """
    Scans all files for 'control_reaches' and 'at_rest'.
    Now stores full trial-by-trial counts for z-score computation.
    
    Parameters
    ----------
    all_files : list
        List of all NPZ files to scan
    metadata_df : DataFrame or None
        Metadata DataFrame
    for_shuffle : bool
        If True, loads data optimized for shuffle analysis
    """
    if not PROCESS_DIFF_PLOTS and not for_shuffle:
        return {'controls': {}, 'rest': {}}
    
    control_files = [f for f in all_files if "control_reaches" in str(f).lower()]
    rest_files = [f for f in all_files if "at_rest" in str(f).lower()]
    
    references = {'controls': {}, 'rest': {}}
    
    if VERBOSE:
        print(f"  [Debug] Found {len(control_files)} control files and {len(rest_files)} at_rest files.")
    
    def get_freq_dur(data, b_idx):
        f_hz, d_ms = 0.0, 0.0
        if metadata_df is not None and b_idx >= 0:
            row = metadata_df[metadata_df['BR_File'] == b_idx]
            if not row.empty:
                try:
                    fh = row.iloc[0].get('Stim_Frequency_Hz', 0.0)
                    dh = row.iloc[0].get('Stim_Duration_ms', 0.0)
                    f_hz = float(fh) if pd.notnull(fh) and str(fh) != '-' else 0.0
                    d_ms = float(dh) if pd.notnull(dh) and str(dh) != '-' else 0.0
                except:
                    pass
        
        if f_hz == 0 or d_ms == 0:
            meta = data['meta'].item() if 'meta' in data and data['meta'].ndim == 0 else {}
            if f_hz == 0: f_hz = float(meta.get('recording_stim_freq', 0.0))
            if d_ms == 0: d_ms = float(meta.get('recording_stim_dur', 0.0))
        return f_hz, d_ms

    # 1. Controls (by Target)
    for target in ['Target_A', 'Target_B', 'Target_control']:
        target_files = [f for f in control_files if target.lower() in str(f).lower()]
        if not target_files: 
            continue
            
        max_trials = -1
        best_data = None
        for f in tqdm(target_files, desc=f"Scanning {target} Controls", leave=False, disable=not VERBOSE):
            try:
                data = np.load(f, allow_pickle=True)
                n_trials = data['UA_counts'].shape[0] if 'UA_counts' in data and len(data['UA_counts'].shape) > 0 else 0
                if n_trials > max_trials:
                    max_trials = n_trials
                    entry = {
                        'n_trials': n_trials, 
                        'NPRW_peaks': None, 'UA_peaks': None, 
                        'ctrl_events': None,
                        'file_path': f,  # Store file path for later reloading if needed
                    }
                    if 'event_ms' in data: 
                        entry['ctrl_events'] = np.asarray(data['event_ms']).flatten()
                    if PROCESS_NPRW and 'NPRW_peak_ms_dedup' in data:
                        entry['NPRW_peaks'] = data['NPRW_peak_ms_dedup'].item()
                    if PROCESS_UA and 'UA_peak_ms_dedup' in data:
                        entry['UA_peaks'] = data['UA_peak_ms_dedup'].item()
                    best_data = entry
            except: 
                continue
        if best_data: 
            references['controls'][target] = best_data

    # 2. At-Rest (by Freq/Dur)
    for f in tqdm(rest_files, desc="Scanning At-Rest Baselines", leave=False, disable=not VERBOSE):
        try:
            data = np.load(f, allow_pickle=True)
            b_idx = int(data['br_idx']) if 'br_idx' in data else -1
            freq, dur = get_freq_dur(data, b_idx)
            
            if freq == 0: 
                continue 
            
            key = (freq, dur)
            n_trials = data['UA_counts'].shape[0] if 'UA_counts' in data and len(data['UA_counts'].shape) > 0 else 0
            
            if key not in references['rest'] or n_trials > references['rest'][key]['n_trials']:
                entry = {
                    'n_trials': n_trials, 
                    'NPRW_peaks': None, 'UA_peaks': None, 
                    'ctrl_events': None,
                    'file_path': f,
                }
                if 'event_ms' in data: 
                    entry['ctrl_events'] = np.asarray(data['event_ms']).flatten()
                if PROCESS_NPRW and 'NPRW_peak_ms_dedup' in data:
                    entry['NPRW_peaks'] = data['NPRW_peak_ms_dedup'].item()
                if PROCESS_UA and 'UA_peak_ms_dedup' in data:
                    entry['UA_peaks'] = data['UA_peak_ms_dedup'].item()
                references['rest'][key] = entry
        except: 
            continue
            
    if VERBOSE:
        if references['rest']:
            print(f"  [Debug] Loaded At-Rest baselines for keys: {list(references['rest'].keys())}")
        else:
            print("  [Debug] No valid At-Rest baselines (with freq > 0) were found.")

    return references


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_output_exists(npz_path, br_idx, cond_type, probe_type, view_suffix, target_folder, base_folder):
    """Check if output file already exists (for SKIP_EXISTING mode)."""
    if probe_type == 'NPRW':
        fig_name = f"[processed]_Cond{br_idx:03d}_{cond_type}_NPRW{view_suffix}.png"
    else:
        fig_name = f"[processed]_Cond{br_idx:03d}_{cond_type}_UA_*{view_suffix}.png"
    
    save_dir = FIG_ROOT / base_folder / target_folder
    if probe_type == 'NPRW':
        return (save_dir / fig_name).exists()
    else:
        return len(list(save_dir.glob(fig_name))) > 0


def extract_metadata(data, npz_path, metadata_df):
    """Extract common metadata from data file."""
    sess = str(data['sess']) if 'sess' in data else ''
    br_idx = int(data['br_idx']) if 'br_idx' in data else -1
    overall_title = str(data['overall_title']) if 'overall_title' in data else ""
    
    # Event times
    if 'event_ms' in data:
        event_ms = np.asarray(data['event_ms']).flatten()
    else:
        event_ms = []
        if 'meta' in data:
            meta = data['meta'].item()
            if isinstance(meta, dict) and 'stim_ms' in meta:
                event_ms = meta['stim_ms']
        if len(event_ms) == 0 and 'nprw_meta' in data:
            nprw_meta = data['nprw_meta'].item()
            if isinstance(nprw_meta, dict) and 'stim_ms' in nprw_meta:
                event_ms = nprw_meta['stim_ms']
        event_ms = np.asarray(event_ms).flatten()
    
    # n_trials
    n_trials_ref = 0
    if len(event_ms) > 0:
        n_trials_ref = len(event_ms)
    elif 'NPRW_counts' in data:
        n_trials_ref = data['NPRW_counts'].shape[0]
    elif 'UA_counts' in data:
        n_trials_ref = data['UA_counts'].shape[0]
    
    # Condition type
    path_str = str(npz_path).lower()
    cond_type = "OTHER"
    if "control_reaches" in path_str:
        cond_type = "CTRL"
    elif "stim_reaches" in path_str:
        cond_type = "STIM"
    elif "at_rest" in path_str:
        cond_type = "REST"
    
    # Target folder
    target_folder = ""
    if "_target_a" in npz_path.name.lower(): 
        target_folder = "Target_A"
    elif "_target_b" in npz_path.name.lower(): 
        target_folder = "Target_B"
    elif "_target_control" in npz_path.name.lower(): 
        target_folder = "Target_control"
    
    # Base folder
    base_folder = "other"
    if "control_reaches" in path_str:
        base_folder = "control_reaches"
    elif "stim_reaches" in path_str:
        base_folder = "stim_reaches"
    elif "at_rest" in path_str:
        base_folder = "at_rest"
    
    # Stim freq/dur
    stim_dur_ms = 0.0
    stim_freq_hz = 0.0
    
    if metadata_df is not None and br_idx >= 0:
        row = metadata_df[metadata_df['BR_File'] == br_idx]
        if not row.empty:
            try:
                fh = row.iloc[0].get('Stim_Frequency_Hz', 0.0)
                dh = row.iloc[0].get('Stim_Duration_ms', 0.0)
                stim_freq_hz = float(fh) if pd.notnull(fh) and str(fh) != '-' else 0.0
                stim_dur_ms = float(dh) if pd.notnull(dh) and str(dh) != '-' else 0.0
            except:
                pass

    if stim_freq_hz == 0 or stim_dur_ms == 0:
        if 'meta' in data and data['meta'].ndim == 0:
            meta_dict = data['meta'].item()
            if isinstance(meta_dict, dict):
                if stim_dur_ms == 0: 
                    stim_dur_ms = float(meta_dict.get('recording_stim_dur', 0.0))
                if stim_freq_hz == 0: 
                    stim_freq_hz = float(meta_dict.get('recording_stim_freq', 0.0))
                
        if (stim_freq_hz == 0 or stim_dur_ms == 0) and 'nprw_meta' in data:
            nprw_meta = data['nprw_meta'].item()
            if isinstance(nprw_meta, dict):
                if stim_freq_hz == 0:
                    sf = nprw_meta.get('stim_freq', 0.0)
                    stim_freq_hz = float(np.median(sf)) if np.ndim(sf) > 0 else float(sf)
                if stim_dur_ms == 0:
                    sd = nprw_meta.get('stim_dur', 0.0)
                    stim_dur_ms = float(np.median(sd)) if np.ndim(sd) > 0 else float(sd)

    if cond_type == "CTRL":
        stim_dur_ms = 0.0
    
    # Stim channels string
    stim_chs_str = ""
    stim_list = []
    
    if 'nprw_meta' in data:
        nm = data['nprw_meta'].item()
        if isinstance(nm, dict) and 'stim_channels' in nm:
            stim_list = nm['stim_channels']
             
    if len(stim_list) == 0 and 'meta' in data:
        m = data['meta'].item()
        if isinstance(m, dict) and 'stim_channels' in m:
            stim_list = m['stim_channels']
    
    if np.size(stim_list) > 0:
        try:
            sl = np.array(stim_list).flatten()
            sl = sorted([int(x) - 1 for x in sl])
            ranges = []
            for _, g in groupby(enumerate(sl), lambda i_x: i_x[0] - i_x[1]):
                group = list(map(itemgetter(1), g))
                ranges.append(f"{group[0]}-{group[-1]}" if len(group) > 1 else f"{group[0]}")
            stim_chs_str = f" | Stim Chs: {', '.join(ranges)}"
        except:
            stim_chs_str = f" | Stim Chs: {stim_list}"
    
    return {
        'sess': sess,
        'br_idx': br_idx,
        'overall_title': overall_title,
        'event_ms': event_ms,
        'n_trials_ref': n_trials_ref,
        'cond_type': cond_type,
        'target_folder': target_folder,
        'base_folder': base_folder,
        'stim_dur_ms': stim_dur_ms,
        'stim_freq_hz': stim_freq_hz,
        'stim_chs_str': stim_chs_str,
    }



# =============================================================================
# SHUFFLE ANALYSIS FUNCTIONS
# =============================================================================

def get_bins_in_window(
    time_axis: np.ndarray,
    window: Tuple[float, float],
    probe: str = None,
    exclude_nprw_blank: bool = False,
    nprw_blank_window: Tuple[float, float] = None,
) -> np.ndarray:
    """
    Return boolean bin mask for a named/statistical analysis window.

    If probe == 'NPRW' and exclude_nprw_blank is True, bins inside the
    per-condition stimulation blank window are removed.
    """

    time_axis = np.asarray(time_axis).ravel()

    bin_mask = (time_axis >= window[0]) & (time_axis <= window[1])

    if (
        probe is not None
        and str(probe).upper() == "NPRW"
        and exclude_nprw_blank
        and nprw_blank_window is not None
    ):
        blank_start, blank_end = nprw_blank_window
        blank_mask = (time_axis >= blank_start) & (time_axis <= blank_end)
        bin_mask = bin_mask & (~blank_mask)

    return bin_mask


def compare_stim_vs_control(
    stim_psth: np.ndarray,
    ctrl_psth: np.ndarray,
    bin_mask: np.ndarray,
    test: str = 'mannwhitney'
) -> Tuple[float, float, float, float]:
    """
    Compare stim vs control PSTH bins within the analysis window.

    stim_psth and ctrl_psth should be 1D arrays: one value per time bin.
    """
    stim_psth = np.asarray(stim_psth).ravel()
    ctrl_psth = np.asarray(ctrl_psth).ravel()
    bin_mask = np.asarray(bin_mask).astype(bool).ravel()

    min_len = min(len(stim_psth), len(ctrl_psth), len(bin_mask))

    stim_psth = stim_psth[:min_len]
    ctrl_psth = ctrl_psth[:min_len]
    bin_mask = bin_mask[:min_len]

    stim_window = stim_psth[bin_mask]
    ctrl_window = ctrl_psth[bin_mask]

    if len(stim_window) == 0 or len(ctrl_window) == 0:
        return 1.0, np.nan, np.nan, np.nan

    stim_window = stim_window[np.isfinite(stim_window)]
    ctrl_window = ctrl_window[np.isfinite(ctrl_window)]

    if len(stim_window) == 0 or len(ctrl_window) == 0:
        return 1.0, np.nan, np.nan, np.nan

    stim_mean = float(np.mean(stim_window))
    ctrl_mean = float(np.mean(ctrl_window))
    diff = stim_mean - ctrl_mean

    # Edge case: both constant
    if np.nanstd(stim_window) == 0 and np.nanstd(ctrl_window) == 0:
        if stim_mean == ctrl_mean:
            return 1.0, stim_mean, ctrl_mean, diff
        else:
            return 1e-10, stim_mean, ctrl_mean, diff

    if test == 'ttest':
        _, p_value = stats.ttest_ind(
            stim_window,
            ctrl_window,
            equal_var=False,
            nan_policy='omit'
        )

    elif test == 'mannwhitney':
        try:
            _, p_value = stats.mannwhitneyu(
                stim_window,
                ctrl_window,
                alternative='two-sided'
            )
        except ValueError:
            p_value = 1.0

    else:
        raise ValueError(f"Unknown test: {test}")

    if not np.isfinite(p_value):
        p_value = 1.0

    return float(p_value), stim_mean, ctrl_mean, diff


def apply_fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Benjamini-Hochberg FDR correction.

    Returns
    -------
    significant : boolean array
    adjusted_p : adjusted p-values
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    if n == 0:
        return np.array([], dtype=bool), np.array([])

    finite_mask = np.isfinite(p_values)
    p_safe = p_values.copy()
    p_safe[~finite_mask] = 1.0

    sorted_idx = np.argsort(p_safe)
    sorted_p = p_safe[sorted_idx]

    ranks = np.arange(1, n + 1)
    critical = (ranks / n) * alpha

    below_critical = sorted_p <= critical

    significant = np.zeros(n, dtype=bool)

    if np.any(below_critical):
        max_significant_rank = np.max(np.where(below_critical)[0]) + 1
        significant[sorted_idx[:max_significant_rank]] = True

    adjusted_sorted = np.minimum(1.0, sorted_p * n / ranks)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]

    adjusted_p = np.zeros(n)
    adjusted_p[sorted_idx] = adjusted_sorted
    adjusted_p[~finite_mask] = 1.0

    return significant, adjusted_p


def apply_bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Bonferroni correction.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    if n == 0:
        return np.array([], dtype=bool), np.array([])

    p_safe = p_values.copy()
    p_safe[~np.isfinite(p_safe)] = 1.0

    adjusted_p = np.minimum(p_safe * n, 1.0)
    significant = adjusted_p < alpha

    return significant, adjusted_p


def classify_channels(significant: np.ndarray, observed_diffs: np.ndarray) -> np.ndarray:
    """
    Classify channels/electrodes as:
      +1 = significant increase
      -1 = significant decrease
       0 = no significant change
    """
    significant = np.asarray(significant).astype(bool)
    observed_diffs = np.asarray(observed_diffs, dtype=float)

    classifications = np.zeros(len(significant), dtype=int)
    classifications[significant & (observed_diffs > 0)] = 1
    classifications[significant & (observed_diffs < 0)] = -1

    return classifications


def run_channel_comparison(
    stim_trials: np.ndarray,
    ctrl_trials: np.ndarray,
    time_axis: np.ndarray,
    window: Tuple[float, float],
    test: str = "mannwhitney",
    probe: str = None,
    exclude_nprw_blank: bool = False,
    nprw_blank_window: Tuple[float, float] = None,
) -> Dict:
    """
    Compare stimulation vs control responses for one channel/electrode.

    stim_trials, ctrl_trials expected shape:
        trials x time

    Returns a dictionary containing raw p-value and effect metrics.
    """

    stim_trials = np.asarray(stim_trials)
    ctrl_trials = np.asarray(ctrl_trials)

    bin_mask = get_bins_in_window(
        time_axis=time_axis,
        window=window,
        probe=probe,
        exclude_nprw_blank=exclude_nprw_blank,
        nprw_blank_window=nprw_blank_window,
    )

    n_bins = int(np.sum(bin_mask))

    if n_bins == 0:
        return {
            "p_value": np.nan,
            "adjusted_p": np.nan,
            "stim_mean": np.nan,
            "ctrl_mean": np.nan,
            "diff": np.nan,
            "classification": 0,
            "significant": False,
            "n_bins": 0,
            "n_stim_trials": int(stim_trials.shape[0]) if stim_trials.ndim > 0 else 0,
            "n_ctrl_trials": int(ctrl_trials.shape[0]) if ctrl_trials.ndim > 0 else 0,
            "window": tuple(window),
        }

    stim_vals = np.nanmean(stim_trials[:, bin_mask], axis=1)
    ctrl_vals = np.nanmean(ctrl_trials[:, bin_mask], axis=1)

    stim_vals = stim_vals[np.isfinite(stim_vals)]
    ctrl_vals = ctrl_vals[np.isfinite(ctrl_vals)]

    if len(stim_vals) == 0 or len(ctrl_vals) == 0:
        p_value = np.nan
    else:
        try:
            if test.lower() in ["mannwhitney", "mannwhitneyu", "mw"]:
                _, p_value = scipy.stats.mannwhitneyu(
                    stim_vals,
                    ctrl_vals,
                    alternative="two-sided",
                )
            elif test.lower() in ["ttest", "t", "ttest_ind"]:
                _, p_value = scipy.stats.ttest_ind(
                    stim_vals,
                    ctrl_vals,
                    equal_var=False,
                    nan_policy="omit",
                )
            else:
                raise ValueError(f"Unknown statistical test: {test}")
        except Exception:
            p_value = np.nan

    stim_mean = float(np.nanmean(stim_vals)) if len(stim_vals) else np.nan
    ctrl_mean = float(np.nanmean(ctrl_vals)) if len(ctrl_vals) else np.nan
    diff = stim_mean - ctrl_mean if np.isfinite(stim_mean) and np.isfinite(ctrl_mean) else np.nan

    return {
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "adjusted_p": np.nan,
        "stim_mean": stim_mean,
        "ctrl_mean": ctrl_mean,
        "diff": float(diff) if np.isfinite(diff) else np.nan,
        "classification": 0,
        "significant": False,
        "n_bins": n_bins,
        "n_stim_trials": int(len(stim_vals)),
        "n_ctrl_trials": int(len(ctrl_vals)),
        "window": tuple(window),
    }


# =============================================================================
# SHUFFLE ANALYSIS MAIN FUNCTIONS
# =============================================================================

def extract_ua_port_and_ids(data, meta_br_idx=None, metadata_df=None):
    """
    Extract UA recording port and ua_ids_1based using the same logic as plotting.

    Returns
    -------
    recording_port : str
        'A' or 'B'
    ua_ids_1based : ndarray or None
    """
    ua_ids_1based = data.get('ua_ids_1based', None)

    if ua_ids_1based is not None:
        if isinstance(ua_ids_1based, np.ndarray) and ua_ids_1based.ndim == 0:
            ua_ids_1based = None
        else:
            ua_ids_1based = np.asarray(ua_ids_1based).ravel()

    recording_port = 'A'

    if 'ua_port' in data:
        port_val = data['ua_port']

        if isinstance(port_val, np.ndarray):
            port_val = port_val.item() if port_val.ndim == 0 else port_val[0]

        recording_port = str(port_val).strip().upper()

    elif metadata_df is not None and meta_br_idx is not None and meta_br_idx >= 0:
        if 'BR_File' in metadata_df.columns and 'UA_port' in metadata_df.columns:
            row = metadata_df[metadata_df['BR_File'] == meta_br_idx]

            if not row.empty:
                port_val = row.iloc[0].get('UA_port', 'A')

                if pd.notnull(port_val):
                    recording_port = str(port_val).strip().upper()

    if recording_port not in ('A', 'B'):
        recording_port = 'A'

    return recording_port, ua_ids_1based


def _get_psth_from_data(data, probe):
    """
    Return trial-wise PSTH/counts as:

        trials x units x time

    Preferred source:
      - *_counts, because this usually preserves trial structure.

    Accepted input shapes:
      - counts 3D: trials x units x time -> returned as-is
      - counts 2D: units x time -> wrapped as one pseudo-trial
      - psth   3D: trials x units x time -> returned as-is
      - psth   2D: units x time -> wrapped as one pseudo-trial
    """

    probe = str(probe).upper()

    if probe == "NPRW":
        count_keys = [
            "nprw_counts",
            "NPRW_counts",
            "nprw_trial_counts",
            "NPRW_trial_counts",
        ]
        psth_keys = [
            "nprw_psth",
            "NPRW_psth",
            "psth_nprw",
            "PSTH_NPRW",
        ]
    elif probe == "UA":
        count_keys = [
            "ua_counts",
            "UA_counts",
            "utah_counts",
            "Utah_counts",
            "ua_trial_counts",
            "UA_trial_counts",
        ]
        psth_keys = [
            "ua_psth",
            "UA_psth",
            "utah_psth",
            "Utah_psth",
            "psth_ua",
            "PSTH_UA",
        ]
    else:
        raise ValueError(f"Unknown probe: {probe}")

    arr = None

    for key in count_keys:
        if key in data:
            arr = np.asarray(data[key], dtype=float)
            break

    if arr is None:
        for key in psth_keys:
            if key in data:
                arr = np.asarray(data[key], dtype=float)
                break

    if arr is None:
        raise KeyError(f"Could not find PSTH/counts array for probe={probe}")

    if arr.ndim == 3:
        return arr

    if arr.ndim == 2:
        return arr[None, :, :]

    raise ValueError(
        f"PSTH/counts array for probe={probe} must be 2D or 3D, got shape {arr.shape}"
    )

def _align_psth_bins(stim_psth, ctrl_psth, time_axis, label=""):
    """
    Align stim/control PSTH/count arrays to the same number of time bins.

    Expected PSTH shape:

        trials x units x time

    Truncates along the final time axis.
    """

    stim_psth = np.asarray(stim_psth, dtype=float)
    ctrl_psth = np.asarray(ctrl_psth, dtype=float)
    time_axis = np.asarray(time_axis, dtype=float)

    if stim_psth.ndim != 3:
        raise ValueError(f"{label} stim_psth must be 3D trials x units x time, got {stim_psth.shape}")

    if ctrl_psth.ndim != 3:
        raise ValueError(f"{label} ctrl_psth must be 3D trials x units x time, got {ctrl_psth.shape}")

    min_bins = min(stim_psth.shape[-1], ctrl_psth.shape[-1], len(time_axis))

    if min_bins <= 0:
        raise ValueError(f"{label} no overlapping time bins")

    stim_psth = stim_psth[:, :, :min_bins]
    ctrl_psth = ctrl_psth[:, :, :min_bins]
    time_axis = time_axis[:min_bins]

    return stim_psth, ctrl_psth, time_axis


def run_analysis_nprw(
    stim_data,
    ctrl_data,
    time_axis,
    window,
    test="mannwhitney",
    exclude_nprw_blank=False,
    nprw_blank_window=None,
):
    """
    Run NPRW stim-vs-control comparison for one analysis window.

    Returns:
        dict[channel_id] = result_dict

    PSTH/counts arrays are expected as:
        trials x channels x time
    """

    stim_psth = _get_psth_from_data(stim_data, "NPRW")
    ctrl_psth = _get_psth_from_data(ctrl_data, "NPRW")

    stim_psth, ctrl_psth, time_axis_local = _align_psth_bins(
        stim_psth,
        ctrl_psth,
        time_axis,
        label="NPRW",
    )

    n_channels = min(stim_psth.shape[1], ctrl_psth.shape[1])

    results = {}

    for ch_idx in range(n_channels):
        try:
            r = run_channel_comparison(
                stim_trials=stim_psth[:, ch_idx, :],
                ctrl_trials=ctrl_psth[:, ch_idx, :],
                time_axis=time_axis_local,
                window=window,
                test=test,
                probe="NPRW",
                exclude_nprw_blank=exclude_nprw_blank,
                nprw_blank_window=nprw_blank_window,
            )

            r["channel"] = int(ch_idx)
            r["channel_id"] = int(ch_idx)
            r["probe"] = "NPRW"
            r["window"] = tuple(window)

            results[int(ch_idx)] = r

        except Exception as e:
            print(f"Warning: NPRW channel {ch_idx} failed: {e}")

    return results


def run_analysis_nprw_multiwindow(
    stim_data,
    ctrl_data,
    time_axis,
    windows,
    test="mannwhitney",
    nprw_blank_window=None,
):
    """
    Run NPRW analysis for multiple named windows.

    Returns:
        results[window_name][channel_id] = result_dict
    """

    all_results = {}

    for window_name, window in windows.items():
        print(f"    NPRW window: {window_name} {window}")

        window_results = run_analysis_nprw(
            stim_data=stim_data,
            ctrl_data=ctrl_data,
            time_axis=time_axis,
            window=tuple(window),
            test=test,
            exclude_nprw_blank=EXCLUDE_NPRW_STIM_BLANK_FROM_STATS,
            nprw_blank_window=nprw_blank_window,
        )

        for ch_id, r in window_results.items():
            r["window_name"] = window_name
            r["window"] = tuple(window)

        all_results[window_name] = window_results

    return all_results


def run_analysis_ua(
    stim_data,
    ctrl_data,
    time_axis,
    window,
    stim_elec_to_idx,
    ctrl_elec_to_idx,
    region=None,
    recording_port=None,
    test="mannwhitney",
):
    """
    Run Utah Array stim-vs-control comparison for one region/window.

    Comparison is by physical electrode_id, not raw data row index.

    PSTH/counts arrays are expected as:
        trials x electrodes/units x time

    Returns:
        dict[electrode_id] = result_dict
    """

    stim_psth = _get_psth_from_data(stim_data, "UA")
    ctrl_psth = _get_psth_from_data(ctrl_data, "UA")

    stim_psth, ctrl_psth, time_axis_local = _align_psth_bins(
        stim_psth,
        ctrl_psth,
        time_axis,
        label="UA",
    )

    results = {}

    common_electrodes = sorted(set(stim_elec_to_idx.keys()) & set(ctrl_elec_to_idx.keys()))

    for electrode_id in common_electrodes:
        stim_idx = stim_elec_to_idx[electrode_id]
        ctrl_idx = ctrl_elec_to_idx[electrode_id]

        if stim_idx is None or ctrl_idx is None:
            continue

        stim_idx = int(stim_idx)
        ctrl_idx = int(ctrl_idx)

        if stim_idx < 0 or stim_idx >= stim_psth.shape[1]:
            continue

        if ctrl_idx < 0 or ctrl_idx >= ctrl_psth.shape[1]:
            continue

        try:
            r = run_channel_comparison(
                stim_trials=stim_psth[:, stim_idx, :],
                ctrl_trials=ctrl_psth[:, ctrl_idx, :],
                time_axis=time_axis_local,
                window=window,
                test=test,
                probe="UA",
                exclude_nprw_blank=False,
                nprw_blank_window=None,
            )

            r["electrode"] = int(electrode_id)
            r["electrode_id"] = int(electrode_id)
            r["stim_data_idx"] = int(stim_idx)
            r["ctrl_data_idx"] = int(ctrl_idx)
            r["probe"] = "UA"
            r["region"] = region
            r["recording_port"] = recording_port
            r["window"] = tuple(window)

            results[int(electrode_id)] = r

        except Exception as e:
            print(f"Warning: UA electrode {electrode_id} failed: {e}")

    return results



def run_analysis_ua_multiwindow(
    stim_data,
    ctrl_data,
    time_axis,
    windows,
    stim_elec_to_idx,
    ctrl_elec_to_idx,
    plot_regions,
    recording_port=None,
    test="mannwhitney",
    bad_channels=None,
):
    """
    Run UA analysis for multiple named common windows.

    Returns:
        results[window_name][electrode_id] = result_dict

    Notes:
      - plot_regions is expected to be a list like ["S1", "M1"].
      - All active physical electrodes are compared by electrode_id.
      - Region labels are attached from UTAH_ELEC_GRIDS if possible.
    """

    if plot_regions is None:
        plot_regions = []

    if isinstance(plot_regions, dict):
        active_regions = list(plot_regions.keys())
    else:
        active_regions = list(plot_regions)

    all_results = {}

    for window_name, window in windows.items():
        print(f"    UA window: {window_name} {window}")

        merged_window_results = {}

        for region in active_regions:
            region_results = run_analysis_ua(
                stim_data=stim_data,
                ctrl_data=ctrl_data,
                time_axis=time_axis,
                window=tuple(window),
                stim_elec_to_idx=stim_elec_to_idx,
                ctrl_elec_to_idx=ctrl_elec_to_idx,
                region=region,
                recording_port=recording_port,
                test=test,
            )

            for electrode_id, r in region_results.items():
                r["window_name"] = window_name
                r["window"] = tuple(window)
                r["region"] = region

                merged_window_results[int(electrode_id)] = r

        all_results[window_name] = merged_window_results

    return all_results

# =============================================================================
# SHUFFLE ANALYSIS VISUALIZATION
# =============================================================================

def plot_shuffle_summary(all_results, save_dir):
    """
    Create summary plots for shuffle analysis results.
    
    Parameters
    ----------
    all_results : list
        List of result dictionaries from all conditions
    save_dir : Path
        Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate results by target and probe type
    summary_data = {
        'NPRW': {'increase': 0, 'decrease': 0, 'no_change': 0, 'total': 0},
    }
    for region in REGION_ANALYSIS_WINDOWS.keys():
        summary_data[f'UA_{region}'] = {'increase': 0, 'decrease': 0, 'no_change': 0, 'total': 0}
    
    # Also track by target
    by_target = {}
    
    for result in all_results:
        if result is None:
            continue
        
        target = result.get('target_folder', 'Unknown')
        if target not in by_target:
            by_target[target] = {
                'NPRW': {'increase': 0, 'decrease': 0, 'no_change': 0, 'total': 0},
            }
            for region in REGION_ANALYSIS_WINDOWS.keys():
                by_target[target][f'UA_{region}'] = {'increase': 0, 'decrease': 0, 'no_change': 0, 'total': 0}
        
        # NPRW results
        nprw_res = result.get('nprw_results')
        if nprw_res is not None:
            classifications = nprw_res['classifications']
            n_increase = np.sum(classifications == 1)
            n_decrease = np.sum(classifications == -1)
            n_no_change = np.sum(classifications == 0)
            n_total = len(classifications)
            
            summary_data['NPRW']['increase'] += n_increase
            summary_data['NPRW']['decrease'] += n_decrease
            summary_data['NPRW']['no_change'] += n_no_change
            summary_data['NPRW']['total'] += n_total
            
            by_target[target]['NPRW']['increase'] += n_increase
            by_target[target]['NPRW']['decrease'] += n_decrease
            by_target[target]['NPRW']['no_change'] += n_no_change
            by_target[target]['NPRW']['total'] += n_total
        
        # UA results
        ua_res = result.get('ua_results')
        if ua_res is not None:
            for region, reg_results in ua_res.items():
                key = f'UA_{region}'
                classifications = reg_results['classifications']
                n_increase = np.sum(classifications == 1)
                n_decrease = np.sum(classifications == -1)
                n_no_change = np.sum(classifications == 0)
                n_total = len(classifications)
                
                summary_data[key]['increase'] += n_increase
                summary_data[key]['decrease'] += n_decrease
                summary_data[key]['no_change'] += n_no_change
                summary_data[key]['total'] += n_total
                
                by_target[target][key]['increase'] += n_increase
                by_target[target][key]['decrease'] += n_decrease
                by_target[target][key]['no_change'] += n_no_change
                by_target[target][key]['total'] += n_total
    
    # Create overall summary figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Shuffle Analysis Summary: Channel Classifications', fontsize=14)
    
    probe_keys = ['NPRW', 'UA_PMd', 'UA_SMA', 'UA_M1i', 'UA_M1s']
    colors = {'increase': 'green', 'decrease': 'red', 'no_change': 'gray'}
    
    for ax_idx, (ax, key) in enumerate(zip(axes.flat[:5], probe_keys)):
        data = summary_data.get(key, {'increase': 0, 'decrease': 0, 'no_change': 0, 'total': 0})
        
        if data['total'] > 0:
            sizes = [data['increase'], data['decrease'], data['no_change']]
            labels = [
                f"Increase\n({data['increase']}, {100*data['increase']/data['total']:.1f}%)",
                f"Decrease\n({data['decrease']}, {100*data['decrease']/data['total']:.1f}%)",
                f"No Change\n({data['no_change']}, {100*data['no_change']/data['total']:.1f}%)"
            ]
            wedge_colors = [colors['increase'], colors['decrease'], colors['no_change']]
            
            # Only plot if there's data
            non_zero = [s for s in sizes if s > 0]
            if len(non_zero) > 0:
                ax.pie(sizes, labels=labels, colors=wedge_colors, autopct='', startangle=90)
        
        ax.set_title(f'{key}\n(n={data["total"]})')
    
    # Hide unused subplot
    axes.flat[-1].axis('off')
    
    plt.tight_layout()
    fig.savefig(save_dir / 'shuffle_summary_overall.png', dpi=DPI_OUTPUT, bbox_inches='tight')
    plt.close(fig)
    
    # Create per-target summary
    for target, target_data in by_target.items():
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Shuffle Analysis Summary: {target}', fontsize=14)
        
        for ax_idx, (ax, key) in enumerate(zip(axes.flat[:5], probe_keys)):
            data = target_data.get(key, {'increase': 0, 'decrease': 0, 'no_change': 0, 'total': 0})
            
            if data['total'] > 0:
                sizes = [data['increase'], data['decrease'], data['no_change']]
                labels = [
                    f"Increase\n({data['increase']})",
                    f"Decrease\n({data['decrease']})",
                    f"No Change\n({data['no_change']})"
                ]
                wedge_colors = [colors['increase'], colors['decrease'], colors['no_change']]
                
                non_zero = [s for s in sizes if s > 0]
                if len(non_zero) > 0:
                    ax.pie(sizes, labels=labels, colors=wedge_colors, autopct='', startangle=90)
            
            ax.set_title(f'{key}\n(n={data["total"]})')
        
        axes.flat[-1].axis('off')
        
        plt.tight_layout()
        fig.savefig(save_dir / f'shuffle_summary_{target}.png', dpi=DPI_OUTPUT, bbox_inches='tight')
        plt.close(fig)
    
    return summary_data, by_target


def plot_channel_detail(result, meta, save_dir):
    """
    Create detailed plots for individual condition results.
    
    Parameters
    ----------
    result : dict
        Result dictionary for one condition
    meta : dict
        Metadata for the condition
    save_dir : Path
        Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # NPRW detailed plot
    nprw_res = result.get('nprw_results')
    if nprw_res is not None and len(nprw_res['channel_ids']) > 0:
        n_ch = len(nprw_res['channel_ids'])
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # P-value histogram
        ax = axes[0]
        ax.hist(nprw_res['p_values'], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(SHUFFLE_ALPHA, color='r', linestyle='--', label=f'α={SHUFFLE_ALPHA}')
        if nprw_res['adjusted_alpha'] > 0:
            ax.axvline(nprw_res['adjusted_alpha'], color='orange', linestyle='--', 
                      label=f'Adjusted α={nprw_res["adjusted_alpha"]:.4f}')
        ax.set_xlabel('P-value')
        ax.set_ylabel('Count')
        ax.set_title('P-value Distribution')
        ax.legend()
        
        # Observed differences
        ax = axes[1]
        colors = np.array(['gray'] * n_ch)
        colors[nprw_res['classifications'] == 1] = 'green'
        colors[nprw_res['classifications'] == -1] = 'red'
        ax.bar(range(n_ch), nprw_res['observed_diffs'], color=colors, edgecolor='none')
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Channel Index')
        ax.set_ylabel('Stim - Control (Mean Spike Count)')
        ax.set_title('Observed Differences')
        
        # Classification pie chart
        ax = axes[2]
        classifications = nprw_res['classifications']
        n_inc = np.sum(classifications == 1)
        n_dec = np.sum(classifications == -1)
        n_nc = np.sum(classifications == 0)
        
        if n_ch > 0:
            sizes = [n_inc, n_dec, n_nc]
            labels = [f'Increase ({n_inc})', f'Decrease ({n_dec})', f'No Change ({n_nc})']
            ax.pie(sizes, labels=labels, colors=['green', 'red', 'gray'], autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Channel Classifications\n(n={n_ch})')
        
        fig.suptitle(f'NPRW Shuffle Analysis: BR{meta["br_idx"]} {meta["target_folder"]}', fontsize=12)
        plt.tight_layout()
        fig.savefig(save_dir / f'shuffle_detail_NPRW_BR{meta["br_idx"]:03d}.png', 
                   dpi=DPI_OUTPUT, bbox_inches='tight')
        plt.close(fig)
    
    # UA detailed plots (one per region)
    ua_res = result.get('ua_results')
    if ua_res is not None:
        for region, reg_results in ua_res.items():
            if len(reg_results['electrode_ids']) == 0:
                continue
            
            n_ch = len(reg_results['electrode_ids'])
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # P-value histogram
            ax = axes[0]
            ax.hist(reg_results['p_values'], bins=min(20, n_ch), edgecolor='black', alpha=0.7)
            ax.axvline(SHUFFLE_ALPHA, color='r', linestyle='--', label=f'α={SHUFFLE_ALPHA}')
            if reg_results['adjusted_alpha'] > 0:
                ax.axvline(reg_results['adjusted_alpha'], color='orange', linestyle='--', 
                          label=f'Adjusted α={reg_results["adjusted_alpha"]:.4f}')
            ax.set_xlabel('P-value')
            ax.set_ylabel('Count')
            ax.set_title('P-value Distribution')
            ax.legend()
            
            # Observed differences
            ax = axes[1]
            colors = np.array(['gray'] * n_ch)
            colors[reg_results['classifications'] == 1] = 'green'
            colors[reg_results['classifications'] == -1] = 'red'
            ax.bar(range(n_ch), reg_results['observed_diffs'], color=colors, edgecolor='none')
            ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Electrode Index')
            ax.set_ylabel('Stim - Control (Mean Spike Count)')
            ax.set_title(f'Observed Differences\nWindow: {reg_results["analysis_window"]}')
            
            # Classification pie chart
            ax = axes[2]
            classifications = reg_results['classifications']
            n_inc = np.sum(classifications == 1)
            n_dec = np.sum(classifications == -1)
            n_nc = np.sum(classifications == 0)
            
            if n_ch > 0:
                sizes = [n_inc, n_dec, n_nc]
                labels = [f'Increase ({n_inc})', f'Decrease ({n_dec})', f'No Change ({n_nc})']
                ax.pie(sizes, labels=labels, colors=['green', 'red', 'gray'], autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Channel Classifications\n(n={n_ch})')
            
            fig.suptitle(f'UA {region} Shuffle Analysis: BR{meta["br_idx"]} {meta["target_folder"]}', fontsize=12)
            plt.tight_layout()
            fig.savefig(save_dir / f'shuffle_detail_UA_{region}_BR{meta["br_idx"]:03d}.png', 
                       dpi=DPI_OUTPUT, bbox_inches='tight')
            plt.close(fig)


def save_shuffle_results_csv(all_results, save_dir):
    """
    Save shuffle analysis results to CSV files.
    
    Parameters
    ----------
    all_results : list
        List of result dictionaries
    save_dir : Path
        Directory to save CSV files
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # NPRW results
    nprw_rows = []
    for result in all_results:
        if result is None:
            continue
        nprw_res = result.get('nprw_results')
        if nprw_res is None:
            continue
        
        for i, ch_id in enumerate(nprw_res['channel_ids']):
            nprw_rows.append({
                'BR_Index': result['br_idx'],
                'Target': result['target_folder'],
                'Channel_ID': ch_id,
                'Observed_Diff': nprw_res['observed_diffs'][i],
                'P_Value': nprw_res['p_values'][i],
                'Significant': nprw_res['significant'][i],
                'Classification': ['No Change', 'Increase', 'Decrease'][nprw_res['classifications'][i]],
                'Stim_Mean': nprw_res['stim_means'][i],
                'Ctrl_Mean': nprw_res['ctrl_means'][i],
                'N_Stim_Trials': nprw_res['n_stim_trials'],
                'N_Ctrl_Trials': nprw_res['n_ctrl_trials'],
                'Analysis_Window': str(nprw_res['analysis_window']),
                'Correction_Method': nprw_res['correction_method'],
                'Adjusted_Alpha': nprw_res['adjusted_alpha'],
            })
    
    if nprw_rows:
        df_nprw = pd.DataFrame(nprw_rows)
        df_nprw.to_csv(save_dir / 'shuffle_results_NPRW.csv', index=False)
        print(f"  Saved NPRW results: {len(nprw_rows)} channels")
    
    # UA results
    ua_rows = []
    for result in all_results:
        if result is None:
            continue
        ua_res = result.get('ua_results')
        if ua_res is None:
            continue
        
        for region, reg_results in ua_res.items():
            for i, elec_id in enumerate(reg_results['electrode_ids']):
                ua_rows.append({
                    'BR_Index': result['br_idx'],
                    'Target': result['target_folder'],
                    'Region': region,
                    'Electrode_ID': elec_id,
                    'Channel_Index': reg_results['channel_indices'][i],
                    'Observed_Diff': reg_results['observed_diffs'][i],
                    'P_Value': reg_results['p_values'][i],
                    'Significant': reg_results['significant'][i],
                    'Classification': ['No Change', 'Increase', 'Decrease'][reg_results['classifications'][i]],
                    'Stim_Mean': reg_results['stim_means'][i],
                    'Ctrl_Mean': reg_results['ctrl_means'][i],
                    'N_Stim_Trials': reg_results['n_stim_trials'],
                    'N_Ctrl_Trials': reg_results['n_ctrl_trials'],
                    'Analysis_Window': str(reg_results['analysis_window']),
                    'Correction_Method': reg_results['correction_method'],
                    'Adjusted_Alpha': reg_results['adjusted_alpha'],
                    'Recording_Port': reg_results['recording_port'],
                })
    
    if ua_rows:
        df_ua = pd.DataFrame(ua_rows)
        df_ua.to_csv(save_dir / 'shuffle_results_UA.csv', index=False)
        print(f"  Saved UA results: {len(ua_rows)} electrodes")
    
    # Summary statistics
    summary_rows = []
    
    # NPRW summary
    nprw_sig = [r for r in nprw_rows if r['Significant']]
    if nprw_rows:
        summary_rows.append({
            'Probe': 'NPRW',
            'Region': 'All',
            'Total_Channels': len(nprw_rows),
            'Significant_Channels': len(nprw_sig),
            'Percent_Significant': 100 * len(nprw_sig) / len(nprw_rows),
            'N_Increase': sum(1 for r in nprw_rows if r['Classification'] == 'Increase'),
            'N_Decrease': sum(1 for r in nprw_rows if r['Classification'] == 'Decrease'),
            'N_No_Change': sum(1 for r in nprw_rows if r['Classification'] == 'No Change'),
        })
    
    # UA summary by region
    for region in REGION_ANALYSIS_WINDOWS.keys():
        region_rows = [r for r in ua_rows if r['Region'] == region]
        if region_rows:
            region_sig = [r for r in region_rows if r['Significant']]
            summary_rows.append({
                'Probe': 'UA',
                'Region': region,
                'Total_Channels': len(region_rows),
                'Significant_Channels': len(region_sig),
                'Percent_Significant': 100 * len(region_sig) / len(region_rows),
                'N_Increase': sum(1 for r in region_rows if r['Classification'] == 'Increase'),
                'N_Decrease': sum(1 for r in region_rows if r['Classification'] == 'Decrease'),
                'N_No_Change': sum(1 for r in region_rows if r['Classification'] == 'No Change'),
            })
    
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(save_dir / 'shuffle_summary_stats.csv', index=False)
        print(f"  Saved summary statistics")
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("SHUFFLE ANALYSIS SUMMARY")
        print("=" * 60)
        print(df_summary.to_string(index=False))
        print("=" * 60)


# =============================================================================
# NPRW PROCESSING (Original Plotting)
# =============================================================================

def process_nprw(data, meta, references, bad_nprw):
    """Process NPRW data and generate plots."""
    if 'NPRW_peak_ms_dedup' not in data or not data['NPRW_peak_ms_dedup'].item():
        return
    
    peaks_dict = data['NPRW_peak_ms_dedup'].item()
    counts_data_orig = data['NPRW_counts']
    edges_ms_orig = data['NPRW_edges_ms']
    centers_ms_orig = data['NPRW_rel_t'] if 'NPRW_rel_t' in data else None
    
    n_ch = 128
    cols = 16
    
    # Filter bad channels
    valid_nprw_keys = []
    for ch in peaks_dict.keys():
        try:
            ch_int = int(ch)
        except:
            ch_int = -1
        if ch_int not in bad_nprw:
            valid_nprw_keys.append(ch)
    
    try:
        sorted_ch_ids = sorted(valid_nprw_keys, key=lambda x: int(x))
    except:
        sorted_ch_ids = sorted(valid_nprw_keys)
        
    rows = (min(len(sorted_ch_ids), n_ch) + cols - 1) // cols
    if rows == 0: 
        rows = 1
    
    # Build mapping from channel key to index in counts_data (ONCE, outside all loops)
    try:
        all_ch_keys_sorted = sorted(peaks_dict.keys(), key=lambda x: int(x))
    except:
        all_ch_keys_sorted = sorted(peaks_dict.keys())
    ch_key_to_counts_idx = {ch_key: idx for idx, ch_key in enumerate(all_ch_keys_sorted)}

    # Loop over plot views
    for view_win_in, view_bin_ms, view_suffix, generate_diff in PLOT_VIEWS:
        if view_win_in is None:
            sd = meta['stim_dur_ms'] if meta['stim_dur_ms'] > 0 else 100.0
            view_win = (-100.0, sd + 100.0)
        else:
            view_win = view_win_in

        if view_bin_ms is not None:
            counts_data, centers_ms, edges_ms = _rebin_from_peaks_fast(
                peaks_dict, meta['event_ms'], view_win, view_bin_ms, meta['stim_dur_ms'],
                blank_pre_ms=20.0, blank_post_ms=20.0)
            cur_ylim = PSTH_YLIM_ZOOM
        else:
            counts_data = counts_data_orig
            centers_ms = centers_ms_orig
            edges_ms = edges_ms_orig
            cur_ylim = PSTH_YLIM
        
        # Main NPRW figure
        fig_rw = plt.figure(figsize=FIG_SIZE_RW)
        gs = gridspec.GridSpec(rows * 2, cols, figure=fig_rw, hspace=0.3, wspace=0.3)
        
        for i, ch in enumerate(sorted_ch_ids):
            if i >= n_ch: 
                break
            r = i // cols
            c = i % cols
            ax_raster = fig_rw.add_subplot(gs[2*r, c])
            ax_psth = fig_rw.add_subplot(gs[2*r+1, c], sharex=ax_raster)
            
            # Use the pre-built mapping to get correct index
            ch_idx = ch_key_to_counts_idx.get(ch)
            binned = counts_data[:, ch_idx, :] if ch_idx is not None and ch_idx < counts_data.shape[1] else None
            peak_times = peaks_dict.get(ch, [])
            
            plot_channel_group_fast(
                ax_raster, ax_psth, peak_times, meta['event_ms'], binned, edges_ms,
                f"Ch {ch}", stim_dur_ms=meta['stim_dur_ms'], 
                blank_pre_ms=20.0, blank_post_ms=20.0,
                bin_centers=centers_ms,
                n_trials_ref=meta['n_trials_ref'], win_ms=view_win, psth_ylim=cur_ylim
            )
            
            if 2*r+1 < (rows*2 - 2):
                ax_psth.set_xticklabels([])
            else:
                ax_psth.tick_params(axis='x', which='both', labelbottom=True)
        
        fig_rw.suptitle(f"{meta['overall_title']} | NPRW{meta['stim_chs_str']}", fontsize=20)
        if len(meta['stim_chs_str']) > 0:
            plt.subplots_adjust(top=0.90)
        
        save_dir = FIG_ROOT / meta['base_folder'] / meta['target_folder']
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_name = f"[processed]_Cond{meta['br_idx']:03d}_{meta['cond_type']}_NPRW{view_suffix}.png"
        fig_rw.savefig(save_dir / fig_name, dpi=DPI_OUTPUT, bbox_inches='tight')
        plt.close(fig_rw)

        # Difference plots (z-scored)
        if generate_diff and PROCESS_DIFF_PLOTS and meta['cond_type'] == "STIM" and references is not None:
            # Determine appropriate y-limits for this view
            diff_ylim = DIFF_YLIM_ZOOM if view_bin_ms is not None else DIFF_YLIM_STANDARD
            
            _process_nprw_diff_plots_zscore(
                peaks_dict, meta['event_ms'], counts_data, centers_ms, edges_ms, 
                view_win, view_bin_ms, view_suffix,
                sorted_ch_ids, n_ch, rows, cols, meta, references, ch_key_to_counts_idx,
                diff_ylim=diff_ylim
            )


def _process_nprw_diff_plots_zscore(peaks_dict, stim_events_ms, stim_counts, centers_ms, edges_ms, 
                                     view_win, view_bin_ms, view_suffix,
                                     sorted_ch_ids, n_ch, rows, cols, meta, references, 
                                     ch_key_to_counts_idx, diff_ylim=DIFF_YLIM):
    """Generate NPRW z-scored difference plots."""
    ctrl_entry = references['controls'].get(meta['target_folder'])
    rest_entry = references['rest'].get((meta['stim_freq_hz'], meta['stim_dur_ms']))
    
    def get_baseline_counts(entry, ref_type='NPRW'):
        """Rebin baseline data to match stim data binning."""
        if entry is None or entry.get(f'{ref_type}_peaks') is None:
            return None
        pk_dict = entry[f'{ref_type}_peaks']
        ev_ms = entry['ctrl_events'] if entry.get('ctrl_events') is not None else []
        if len(ev_ms) == 0: 
            return None
        
        # Use same binning as stim data
        bin_width = view_bin_ms if view_bin_ms else 20.0
        c_ref, _, _ = _rebin_from_peaks_fast(
            pk_dict, ev_ms, view_win, bin_width, 0.0, 0.0, 0.0
        )
        return c_ref  # Return full trial-by-trial counts for z-score

    ctrl_counts = get_baseline_counts(ctrl_entry, 'NPRW')
    rest_counts = get_baseline_counts(rest_entry, 'NPRW')
    
    diff_configs = []
    if ctrl_counts is not None:
        diff_configs.append(('stim_control_diff', 'Stim - Control (Z)', ctrl_counts))
    if rest_counts is not None:
        diff_configs.append(('stim_rest_diff', 'Stim - Rest (Z)', rest_counts))
    
    for dir_name, title_suffix, baseline_counts in diff_configs:
        # Check shape compatibility
        if stim_counts.shape[1:] != baseline_counts.shape[1:]:
            if VERBOSE:
                print(f"  [Warn] Shape mismatch: stim {stim_counts.shape} vs baseline {baseline_counts.shape}")
            continue
        
        # Compute z-scored difference
        z_diff, stim_mean, baseline_mean, baseline_std = compute_zscore_diff(
            stim_counts, baseline_counts, min_std=0.1
        )
        
        fig_diff = plt.figure(figsize=FIG_SIZE_RW)
        gs_diff = gridspec.GridSpec(rows, cols, figure=fig_diff, hspace=0.3, wspace=0.3)
        
        found_diff = False
        for i, ch in enumerate(sorted_ch_ids):
            if i >= n_ch: 
                break
            r, c = i // cols, i % cols
            ax_diff = fig_diff.add_subplot(gs_diff[r, c])
            
            # Use the mapping to get correct index
            ch_idx = ch_key_to_counts_idx.get(ch)
            
            if ch_idx is not None and ch_idx < z_diff.shape[0]:
                found_diff = True
                z_ch = z_diff[ch_idx, :]
                
                # Color based on z-score sign
                bar_colors = np.where(z_ch >= 0, 'tab:green', 'tab:red')
                
                if centers_ms is not None and len(z_ch) == len(centers_ms):
                    bw = view_bin_ms if view_bin_ms else (np.nanmedian(np.diff(centers_ms)) if len(centers_ms) > 1 else 20.0)
                    ax_diff.bar(centers_ms, z_ch, width=bw, align='center', color=bar_colors, edgecolor='none')
                elif edges_ms is not None and len(z_ch) == len(edges_ms) - 1:
                    ax_diff.bar(edges_ms[:-1], z_ch, width=np.diff(edges_ms), align='edge', color=bar_colors, edgecolor='none')
                
                # Add reference lines
                ax_diff.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
                ax_diff.axhline(2, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)  # +2 SD
                ax_diff.axhline(-2, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)  # -2 SD
                
                if np.nansum(np.abs(z_ch)) > 0:
                    ax_diff.axvline(0, color='r', linestyle='--', linewidth=1, alpha=0.8)
            else:
                ax_diff.axis('off')
            
            ax_diff.set_title(f"Ch {ch}", fontsize=10, pad=2)
            ax_diff.set_xlim(view_win)
            ax_diff.set_ylim(diff_ylim)
            ax_diff.set_ylabel('Z-score', fontsize=6) if c == 0 else None
            ax_diff.spines['top'].set_visible(False)
            ax_diff.spines['right'].set_visible(False)
            ax_diff.tick_params(axis='both', which='major', labelsize=6)
            if r < rows - 1: 
                ax_diff.set_xticklabels([])

        # Add stim region shading
        if meta['stim_dur_ms'] > 0:
            ax_diff.axvspan(0, meta['stim_dur_ms'], color=STIM_REGION_COLOR, alpha=STIM_REGION_ALPHA, zorder=0)

        if found_diff:
            fig_diff.suptitle(f"{meta['overall_title']} | NPRW {title_suffix}{meta['stim_chs_str']}", fontsize=20)
            save_diff_dir = FIG_ROOT / dir_name / meta['target_folder']
            save_diff_dir.mkdir(parents=True, exist_ok=True)
            fig_diff_name = f"[diff_zscore]_Cond{meta['br_idx']:03d}_{meta['cond_type']}_NPRW{view_suffix}.png"
            fig_diff.savefig(save_diff_dir / fig_diff_name, dpi=DPI_OUTPUT, bbox_inches='tight')
        plt.close(fig_diff)


# =============================================================================
# UTAH ARRAY PROCESSING (Original Plotting)
# =============================================================================

def process_ua(data, meta, references, bad_ua, metadata_df):
    """Process Utah Array data and generate plots."""
    has_ua = False
    if 'HAS_BR' in data:
        try:
            has_ua = bool(data['HAS_BR'])
        except:
            has_ua = False
    elif 'UA_counts' in data:
        has_ua = True

    if not has_ua or 'UA_peak_ms_dedup' not in data or not data['UA_peak_ms_dedup'].item():
        return

    peaks_dict = data['UA_peak_ms_dedup'].item()
    counts_data_orig = data['UA_counts']
    edges_ms_orig = data['UA_edges_ms']
    centers_ms_orig = data['UA_rel_t'] if 'UA_rel_t' in data else None

    # DEBUG: Print peaks_dict keys to understand the format
    if VERBOSE:
        pk_keys = list(peaks_dict.keys())[:10]
        print(f"  [Debug] peaks_dict sample keys: {pk_keys}")
        print(f"  [Debug] peaks_dict total keys: {len(peaks_dict)}")
        # Check if any keys have data
        for k in pk_keys:
            v = peaks_dict.get(k, [])
            if len(v) > 0:
                print(f"  [Debug] Key '{k}' has {len(v)} spikes")
                break

    # Load ua_ids_1based (local NSP IDs 1-128)
    ua_ids_1based = data.get('ua_ids_1based', None)
    if ua_ids_1based is not None:
        if isinstance(ua_ids_1based, np.ndarray) and ua_ids_1based.ndim == 0:
            ua_ids_1based = None
        else:
            ua_ids_1based = np.asarray(ua_ids_1based).ravel()
    
    if VERBOSE and ua_ids_1based is not None:
        print(f"  [Debug] ua_ids_1based sample: {ua_ids_1based[:10] if len(ua_ids_1based) >= 10 else ua_ids_1based}")
    
    # Get recording port
    recording_port = 'A'
    if 'ua_port' in data:
        port_val = data['ua_port']
        if isinstance(port_val, np.ndarray):
            port_val = port_val.item() if port_val.ndim == 0 else str(port_val)
        recording_port = str(port_val).strip().upper()
    elif metadata_df is not None and meta['br_idx'] >= 0:
        row = metadata_df[metadata_df['BR_File'] == meta['br_idx']]
        if not row.empty and 'UA_port' in row.columns:
            port_val = row.iloc[0].get('UA_port', 'A')
            if pd.notnull(port_val):
                recording_port = str(port_val).strip().upper()
    
    if recording_port not in ('A', 'B'):
        if VERBOSE:
            print(f"  [Warn] Invalid UA port '{recording_port}', defaulting to 'A'")
        recording_port = 'A'
    
    if VERBOSE:
        print(f"  [Debug] Recording Port: {recording_port} for BR{meta['br_idx']}")
    
    # Build electrode -> channel index mapping
    elec_to_idx = build_elec_to_data_idx(ua_ids_1based, elec_info_global, recording_port=recording_port)
    
    if VERBOSE:
        print(f"  [Debug] Electrode mapping: {len(elec_to_idx)} electrodes mapped to channels")
        if len(elec_to_idx) > 0:
            sample = list(elec_to_idx.items())[:5]
            print(f"  [Debug] Sample elec->ch_idx: {sample}")
    
    # Build a reverse mapping: channel_index -> list of possible keys in peaks_dict
    # This handles different key formats (str vs int, 0-based vs 1-based)
    def get_peak_times_for_electrode(elec_id, elec_to_idx, peaks_dict):
        """Get spike times for an electrode using various key formats."""
        ch_idx = elec_to_idx.get(elec_id)
        if ch_idx is None:
            return []
        
        # Try different key formats in peaks_dict
        possible_keys = [
            ch_idx,                # int: 0, 1, 2, ...
            str(ch_idx),           # str: "0", "1", "2", ...
            ch_idx + 1,            # 1-based int
            str(ch_idx + 1),       # 1-based str
        ]
        
        for key in possible_keys:
            spikes = peaks_dict.get(key)
            if spikes is not None and len(spikes) > 0:
                return spikes
        
        return []
    
    # Get active regions
    plot_regions = get_active_regions(ua_ids_1based, elec_info_global, recording_port=recording_port)
    
    if not plot_regions:
        plot_regions = list(UTAH_ELEC_GRIDS.keys())  # Try all regions
        if VERBOSE:
            print(f"  [Warn] No regions identified, trying all: {plot_regions}")
    elif VERBOSE:
        print(f"  [Debug] Active regions: {plot_regions}")

    # Loop over plot views
    for view_win_in, view_bin_ms, view_suffix, generate_diff in PLOT_VIEWS:
        if view_win_in is None:
            sd = meta['stim_dur_ms'] if meta['stim_dur_ms'] > 0 else 100.0
            view_win = (-100.0, sd + 100.0)
        else:
            view_win = view_win_in

        if view_bin_ms is not None:
            counts_data, centers_ms, edges_ms = _rebin_from_peaks_fast(
                peaks_dict, meta['event_ms'], view_win, view_bin_ms, meta['stim_dur_ms'],
                blank_pre_ms=0.0, blank_post_ms=0.0)
            cur_ylim = PSTH_YLIM_ZOOM
        else:
            counts_data = counts_data_orig
            centers_ms = centers_ms_orig
            edges_ms = edges_ms_orig
            cur_ylim = PSTH_YLIM

        if VERBOSE:
            print(f"  [Debug] counts_data shape: {counts_data.shape}")

        # Process each region
        for reg_target in plot_regions:
            grid_elec = UTAH_ELEC_GRIDS.get(reg_target)
            if grid_elec is None:
                if VERBOSE:
                    print(f"  [Warn] No grid found for region {reg_target}")
                continue
            
            # Get electrodes in this region that belong to the recording port
            port_electrodes = get_electrodes_for_port(elec_info_global, reg_target, recording_port)
            
            if VERBOSE:
                print(f"  [Debug] Region {reg_target}: {len(port_electrodes)} electrodes on Port {recording_port}")
            
            fig_ua = plt.figure(figsize=(32, 24))
            gs = gridspec.GridSpec(8 * 2, 8, figure=fig_ua, hspace=0.3, wspace=0.3)
            
            found_channels = False
            channels_with_spikes = 0
            
            for r in range(8):
                for c in range(8):
                    elec_id = int(grid_elec[r, c])
                    
                    ax_raster = fig_ua.add_subplot(gs[2*r, c])
                    ax_psth = fig_ua.add_subplot(gs[2*r+1, c], sharex=ax_raster)
                    
                    # Check if electrode exists and is on our recording port
                    if elec_id <= 0:
                        ax_raster.axis('off')
                        ax_psth.axis('off')
                        continue
                    
                    info = elec_info_global.get(elec_id, {})
                    elec_port = info.get('port', '')
                    
                    if elec_port != recording_port:
                        # Electrode is on the OTHER port
                        ax_raster.axis('off')
                        ax_psth.axis('off')
                        ax_raster.text(0.5, 0.5, f"E{elec_id}\nPort {elec_port}", 
                                    ha='center', va='center', 
                                    transform=ax_raster.transAxes, fontsize=8, 
                                    alpha=0.3, color='gray')
                        continue
                    
                    # Get channel index for this electrode
                    ch_idx = elec_to_idx.get(elec_id)
                    
                    if ch_idx is not None and 0 <= ch_idx < counts_data.shape[1]:
                        found_channels = True
                        binned = counts_data[:, ch_idx, :]
                        peak_times = get_peak_times_for_electrode(elec_id, elec_to_idx, peaks_dict)
                        
                        if len(peak_times) > 0:
                            channels_with_spikes += 1
                        
                        plot_channel_group_fast(
                            ax_raster, ax_psth, peak_times, meta['event_ms'], binned, edges_ms,
                            f"E{elec_id}", stim_dur_ms=0.0,
                            blank_pre_ms=0.0, blank_post_ms=0.0,
                            bin_centers=centers_ms,
                            n_trials_ref=meta['n_trials_ref'], win_ms=view_win, psth_ylim=cur_ylim
                        )
                    else:
                        ax_raster.axis('off')
                        ax_psth.axis('off')
                        ax_raster.text(0.5, 0.5, f"E{elec_id}\nNo Data", 
                                    ha='center', va='center', 
                                    transform=ax_raster.transAxes, fontsize=8, alpha=0.3)

            if VERBOSE:
                print(f"  [Debug] Region {reg_target}: {channels_with_spikes} channels have spike data")

            if found_channels:
                fig_ua.suptitle(f"{meta['overall_title']} | Utah Array - {reg_target} (Recording: Port {recording_port}){meta['stim_chs_str']}", fontsize=20)
                plt.subplots_adjust(top=0.90 if meta['stim_chs_str'] else 0.94)
                
                save_dir = FIG_ROOT / meta['base_folder'] / meta['target_folder']
                save_dir.mkdir(parents=True, exist_ok=True)
                fig_name = f"[processed]_Cond{meta['br_idx']:03d}_{meta['cond_type']}_UA_{reg_target}{view_suffix}.png"
                fig_ua.savefig(save_dir / fig_name, dpi=DPI_OUTPUT, bbox_inches='tight')
                plt.close(fig_ua)
            else:
                if VERBOSE:
                    print(f"  [Warn] No channels found for region {reg_target} on Port {recording_port}")
                plt.close(fig_ua)

        # Difference plots (z-scored)
        if generate_diff and PROCESS_DIFF_PLOTS and meta['cond_type'] == "STIM" and references is not None:
            # Determine appropriate y-limits for this view
            diff_ylim = DIFF_YLIM_ZOOM if view_bin_ms is not None else DIFF_YLIM_STANDARD
            
            _process_ua_diff_plots_zscore(
                peaks_dict, meta['event_ms'], counts_data, centers_ms, edges_ms, 
                view_win, view_bin_ms, view_suffix,
                plot_regions, elec_to_idx, recording_port, ua_ids_1based, meta, references,
                diff_ylim=diff_ylim
            )


def _process_ua_diff_plots_zscore(peaks_dict, stim_events_ms, stim_counts, centers_ms, edges_ms, 
                                   view_win, view_bin_ms, view_suffix,
                                   plot_regions, elec_to_idx, recording_port, ua_ids_1based, 
                                   meta, references, diff_ylim=DIFF_YLIM):
    """Generate UA z-scored difference plots."""
    ctrl_entry = references['controls'].get(meta['target_folder'])
    rest_entry = references['rest'].get((meta['stim_freq_hz'], meta['stim_dur_ms']))

    def get_baseline_counts_ua(entry):
        """Rebin baseline data to match stim data binning."""
        if entry is None or entry.get('UA_peaks') is None:
            return None
        pk_dict = entry['UA_peaks']
        ev_ms = entry['ctrl_events'] if entry.get('ctrl_events') is not None else []
        if len(ev_ms) == 0: 
            return None
        
        bin_width = view_bin_ms if view_bin_ms else 20.0
        c_ref, _, _ = _rebin_from_peaks_fast(
            pk_dict, ev_ms, view_win, bin_width, 0.0, 0.0, 0.0
        )
        return c_ref

    ctrl_counts = get_baseline_counts_ua(ctrl_entry)
    rest_counts = get_baseline_counts_ua(rest_entry)

    diff_configs_ua = []
    if ctrl_counts is not None:
        diff_configs_ua.append(('stim_control_diff', 'Stim - Control (Z)', ctrl_counts))
    if rest_counts is not None:
        diff_configs_ua.append(('stim_rest_diff', 'Stim - Rest (Z)', rest_counts))
    
    for dir_name, title_suffix, baseline_counts in diff_configs_ua:
        # Check shape compatibility
        if stim_counts.shape[1:] != baseline_counts.shape[1:]:
            if VERBOSE:
                print(f"  [Warn] UA Shape mismatch: stim {stim_counts.shape} vs baseline {baseline_counts.shape}")
            continue
        
        # Compute z-scored difference
        z_diff, stim_mean, baseline_mean, baseline_std = compute_zscore_diff(
            stim_counts, baseline_counts, min_std=0.1
        )
        
        for reg_target in plot_regions:
            grid_elec = UTAH_ELEC_GRIDS.get(reg_target)
            if grid_elec is None: 
                continue
            
            port_electrodes = get_electrodes_for_port(elec_info_global, reg_target, recording_port)
            
            fig_diff_ua = plt.figure(figsize=(32, 24))
            gs_diff = gridspec.GridSpec(8, 8, figure=fig_diff_ua, hspace=0.3, wspace=0.3)
            found_diff = False
            
            for r in range(8):
                for c in range(8):
                    elec_id = int(grid_elec[r, c])
                    ax_diff = fig_diff_ua.add_subplot(gs_diff[r, c])
                    
                    if elec_id in port_electrodes:
                        idx = elec_to_idx.get(elec_id)
                        
                        if idx is not None and 0 <= idx < z_diff.shape[0]:
                            found_diff = True
                            z_ch = z_diff[idx, :]
                            bar_colors = np.where(z_ch >= 0, 'tab:green', 'tab:red')
                            
                            if centers_ms is not None and len(z_ch) == len(centers_ms):
                                bw = view_bin_ms if view_bin_ms else (np.nanmedian(np.diff(centers_ms)) if len(centers_ms) > 1 else 20.0)
                                ax_diff.bar(centers_ms, z_ch, width=bw, align='center', color=bar_colors, edgecolor='none')
                            elif edges_ms is not None and len(z_ch) == len(edges_ms) - 1:
                                ax_diff.bar(edges_ms[:-1], z_ch, width=np.diff(edges_ms), align='edge', color=bar_colors, edgecolor='none')
                            
                            # Add reference lines
                            ax_diff.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
                            ax_diff.axhline(2, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
                            ax_diff.axhline(-2, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
                            
                            if np.nansum(np.abs(z_ch)) > 0:
                                ax_diff.axvline(0, color='r', linestyle='--', linewidth=1, alpha=0.8)
                            ax_diff.set_title(f"E{elec_id}", fontsize=10, pad=2)
                        else:
                            ax_diff.axis('off')
                            ax_diff.text(0.5, 0.5, f"E{elec_id}\nNo Data", 
                                         ha='center', va='center', 
                                         transform=ax_diff.transAxes, fontsize=8, alpha=0.3)
                    else:
                        ax_diff.axis('off')
                        if elec_id > 0:
                            other_port = 'B' if recording_port == 'A' else 'A'
                            ax_diff.text(0.5, 0.5, f"E{elec_id}\nPort {other_port}", 
                                         ha='center', va='center', 
                                         transform=ax_diff.transAxes, fontsize=8, 
                                         alpha=0.3, color='gray')
                    
                    ax_diff.set_xlim(view_win)
                    ax_diff.set_ylim(diff_ylim)
                    ax_diff.tick_params(axis='both', which='major', labelsize=6)
                    if c == 0:
                        ax_diff.set_ylabel('Z-score', fontsize=6)
            
            # Add stim region shading
            if meta['stim_dur_ms'] > 0:
                ax_diff.axvspan(0, meta['stim_dur_ms'], color=STIM_REGION_COLOR, alpha=STIM_REGION_ALPHA, zorder=0)

            if found_diff:
                fig_diff_ua.suptitle(f"{meta['overall_title']} | Utah {title_suffix} - {reg_target} (Port {recording_port}){meta['stim_chs_str']}", fontsize=20)
                save_diff_dir = FIG_ROOT / dir_name / meta['target_folder']
                save_diff_dir.mkdir(parents=True, exist_ok=True)
                fig_name = f"[diff_zscore]_Cond{meta['br_idx']:03d}_{meta['cond_type']}_UA_{reg_target}{view_suffix}.png"
                fig_diff_ua.savefig(save_diff_dir / fig_name, dpi=DPI_OUTPUT, bbox_inches='tight')
            plt.close(fig_diff_ua)


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_file(npz_path, references=None, metadata_df=None):
    """Process a single NPZ file for plotting analysis."""
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return

    # Load bad channels
    bad_map = get_session_impedances(SESSION_LOC, utah_exclude_low=False)
    bad_ua = bad_map.get('utah', set())
    bad_nprw = bad_map.get('nprw', set())

    # Extract metadata
    meta = extract_metadata(data, npz_path, metadata_df)
    
    # Check if we should process this file
    if PROCESS_ONLY and meta['br_idx'] not in PROCESS_ONLY:
        return
    
    if meta['n_trials_ref'] == 0:
        if VERBOSE:
            print(f"\nSkipping {npz_path.name}: No events and no trial counts found.")
        return

    if meta['cond_type'] == "OTHER":
        return

    if VERBOSE and meta['cond_type'] == "STIM":
        print(f"  [Debug] File: {npz_path.name} | Freq: {meta['stim_freq_hz']} | Dur: {meta['stim_dur_ms']}")
        if references and references['rest']:
            rest_entry = references['rest'].get((meta['stim_freq_hz'], meta['stim_dur_ms']))
            if rest_entry:
                print(f"    [OK] Found At-Rest baseline for Freq={meta['stim_freq_hz']}, Dur={meta['stim_dur_ms']}")
            else:
                print(f"    [Warn] No At-Rest baseline match. Available: {list(references['rest'].keys())}")

    # Process NPRW
    if PROCESS_NPRW:
        process_nprw(data, meta, references, bad_nprw)

    # Process UA
    if PROCESS_UA:
        process_ua(data, meta, references, bad_ua, metadata_df)


def process_file_shuffle(
    stim_npz_path: Path,
    references: Dict,
    metadata_df: Optional[pd.DataFrame],
    electrode_mapping,
    bad_channels: set,
    config: Dict,
) -> Optional[Dict]:
    """
    Process one stim file for statistical analysis.

    This version uses the same UA mapping logic as the plotting code:

        ua_ids_1based + ua_port -> build_elec_to_data_idx()

    It also builds separate stim/control UA mappings so that stim and control
    are compared by physical electrode ID, not by raw channel row.
    """
    try:
        stim_npz_path = Path(stim_npz_path)

        # ---------------------------------------------------------------------
        # Load stim data
        # ---------------------------------------------------------------------
        stim_data = dict(np.load(stim_npz_path, allow_pickle=True))

        br_idx = int(stim_data['br_idx']) if 'br_idx' in stim_data else -1

        if br_idx < 0:
            print(f"  Warning: No br_idx in {stim_npz_path.name}")
            return None

        # ---------------------------------------------------------------------
        # Metadata lookup, but do not crash if metadata_df is None
        # ---------------------------------------------------------------------
        meta = {}

        if metadata_df is not None and 'BR_File' in metadata_df.columns:
            meta_row = metadata_df[metadata_df['BR_File'] == br_idx]

            if not meta_row.empty:
                meta = meta_row.iloc[0].to_dict()
            else:
                print(f"  Warning: No metadata for BR{br_idx}")
        else:
            meta_row = pd.DataFrame()

        # ---------------------------------------------------------------------
        # Determine target from filename
        # ---------------------------------------------------------------------
        path_lower = stim_npz_path.name.lower()

        if "_target_a" in path_lower:
            target = "Target_A"
        elif "_target_b" in path_lower:
            target = "Target_B"
        elif "_target_control" in path_lower:
            target = "Target_control"
        else:
            target = "Unknown"

        # ---------------------------------------------------------------------
        # Find matched control file
        # ---------------------------------------------------------------------
        ctrl_ref = references.get('controls', {}).get(target)

        if ctrl_ref is None or ctrl_ref.get('file_path') is None:
            print(f"  Warning: No control for target {target}")
            return None

        ctrl_path = Path(ctrl_ref['file_path'])
        ctrl_data = dict(np.load(ctrl_path, allow_pickle=True))

        # ---------------------------------------------------------------------
        # Separate time axes for NPRW and UA
        # ---------------------------------------------------------------------
        nprw_time_axis = get_time_axis_for_probe(stim_data, 'NPRW')
        ua_time_axis = get_time_axis_for_probe(stim_data, 'UA')

        # ---------------------------------------------------------------------
        # Result object
        # ---------------------------------------------------------------------
        n_stim_trials = 0

        if 'event_ms' in stim_data:
            n_stim_trials = len(np.asarray(stim_data['event_ms']).ravel())
        elif 'NPRW_counts' in stim_data:
            n_stim_trials = np.asarray(stim_data['NPRW_counts']).shape[0]
        elif 'UA_counts' in stim_data:
            n_stim_trials = np.asarray(stim_data['UA_counts']).shape[0]

        n_ctrl_trials = 0

        if 'event_ms' in ctrl_data:
            n_ctrl_trials = len(np.asarray(ctrl_data['event_ms']).ravel())
        elif 'NPRW_counts' in ctrl_data:
            n_ctrl_trials = np.asarray(ctrl_data['NPRW_counts']).shape[0]
        elif 'UA_counts' in ctrl_data:
            n_ctrl_trials = np.asarray(ctrl_data['UA_counts']).shape[0]

        results = {
            'br_idx': br_idx,
            'target': target,
            'stim_file': str(stim_npz_path),
            'ctrl_file': str(ctrl_path),
            'n_stim_trials': n_stim_trials,
            'n_ctrl_trials': n_ctrl_trials,
            'NPRW': {},
            'UA': {},
        }

        # ---------------------------------------------------------------------
        # NPRW analysis
        # ---------------------------------------------------------------------
        if config.get('process_nprw', True):

            analysis_windows = config.get("analysis_windows", COMMON_ANALYSIS_WINDOWS)

            if 'NPRW_counts' not in stim_data and 'NPRW_psth' not in stim_data:
                if VERBOSE:
                    print(f"  Warning: No NPRW data in stim file {stim_npz_path.name}")

            elif 'NPRW_counts' not in ctrl_data and 'NPRW_psth' not in ctrl_data:
                if VERBOSE:
                    print(f"  Warning: No NPRW data in control file {ctrl_path.name}")

            elif nprw_time_axis is None:
                print(f"  Warning: No NPRW time axis found in {stim_npz_path.name}")

            else:
                nprw_window = config.get('nprw_window', (-500.0, 0.0))

                # --------------------------------------------------------------
                # NPRW multi-window statistics with per-condition stim blanking
                # --------------------------------------------------------------
                if PROCESS_NPRW:
                    nprw_time_axis = get_time_axis_for_probe(stim_data, "NPRW")

                    nprw_blank_window = get_nprw_blank_window_ms(
                        stim_data=stim_data,
                        metadata_df=metadata_df,
                        meta_br_idx=br_idx,
                        default_blank_end_ms=None,
                    )

                    if nprw_blank_window is not None:
                        print(
                            f"  NPRW blank window for stats: "
                            f"{nprw_blank_window[0]:.1f} to {nprw_blank_window[1]:.1f} ms"
                        )
                    else:
                        print("  NPRW blank window for stats: unavailable; no blank exclusion applied")

                    results["nprw_blank_window"] = nprw_blank_window
                    results["nprw_blank_start_ms"] = (
                        float(nprw_blank_window[0]) if nprw_blank_window is not None else np.nan
                    )
                    results["nprw_blank_end_ms"] = (
                        float(nprw_blank_window[1]) if nprw_blank_window is not None else np.nan
                    )
                    results["nprw_stim_duration_ms"] = (
                        float(nprw_blank_window[1] - nprw_blank_window[0])
                        if nprw_blank_window is not None
                        else np.nan
                    )

                    results["NPRW"] = run_analysis_nprw_multiwindow(
                        stim_data=stim_data,
                        ctrl_data=ctrl_data,
                        time_axis=nprw_time_axis,
                        windows=config.get("analysis_windows", COMMON_ANALYSIS_WINDOWS),
                        test=config.get("test", "mannwhitney"),
                        nprw_blank_window=nprw_blank_window,
                    )

        # ---------------------------------------------------------------------
        # UA analysis
        # ---------------------------------------------------------------------
        if config.get('process_ua', True):

            analysis_windows = config.get("analysis_windows", COMMON_ANALYSIS_WINDOWS)

            if 'UA_counts' not in stim_data and 'UA_psth' not in stim_data:
                if VERBOSE:
                    print(f"  Warning: No UA data in stim file {stim_npz_path.name}")

            elif 'UA_counts' not in ctrl_data and 'UA_psth' not in ctrl_data:
                if VERBOSE:
                    print(f"  Warning: No UA data in control file {ctrl_path.name}")

            elif ua_time_axis is None:
                print(f"  Warning: No UA time axis found in {stim_npz_path.name}")

            else:
                # Stim port + ids
                stim_recording_port, stim_ua_ids_1based = extract_ua_port_and_ids(
                    stim_data,
                    meta_br_idx=br_idx,
                    metadata_df=metadata_df
                )

                # Control port + ids.
                #
                # Most control files should have ua_port/ua_ids_1based inside the NPZ.
                # If they do not, this falls back to Port A and sequential mapping.
                ctrl_br_idx = int(ctrl_data['br_idx']) if 'br_idx' in ctrl_data else -1

                ctrl_recording_port, ctrl_ua_ids_1based = extract_ua_port_and_ids(
                    ctrl_data,
                    meta_br_idx=ctrl_br_idx,
                    metadata_df=metadata_df
                )

                # Build physical electrode ID -> data row index maps
                stim_elec_to_idx = build_elec_to_data_idx(
                    stim_ua_ids_1based,
                    elec_info_global,
                    recording_port=stim_recording_port
                )

                ctrl_elec_to_idx = build_elec_to_data_idx(
                    ctrl_ua_ids_1based,
                    elec_info_global,
                    recording_port=ctrl_recording_port
                )

                # Use stim file to decide which regions are active,
                # matching the plotting code behavior.
                plot_regions = get_active_regions(
                    stim_ua_ids_1based,
                    elec_info_global,
                    recording_port=stim_recording_port
                )

                if not plot_regions:
                    plot_regions = list(UTAH_ELEC_GRIDS.keys())

                if VERBOSE:
                    print(f"  [Debug] BR{br_idx} UA stim port: {stim_recording_port}")
                    print(f"  [Debug] BR{br_idx} UA ctrl port: {ctrl_recording_port}")
                    print(f"  [Debug] stim mapped electrodes: {len(stim_elec_to_idx)}")
                    print(f"  [Debug] ctrl mapped electrodes: {len(ctrl_elec_to_idx)}")
                    print(f"  [Debug] active regions: {plot_regions}")

                # --------------------------------------------------------------
                # Utah Array multi-window statistics
                # --------------------------------------------------------------
                if PROCESS_UA:
                    ua_time_axis = get_time_axis_for_probe(stim_data, "UA")

                    results["UA"] = run_analysis_ua_multiwindow(
                        stim_data=stim_data,
                        ctrl_data=ctrl_data,
                        time_axis=ua_time_axis,
                        stim_elec_to_idx=stim_elec_to_idx,
                        ctrl_elec_to_idx=ctrl_elec_to_idx,
                        plot_regions=plot_regions,
                        recording_port=stim_recording_port,
                        windows=config.get("analysis_windows", COMMON_ANALYSIS_WINDOWS),
                        bad_channels=bad_channels,
                        test=config.get("test", "mannwhitney"),
                    )

        return results

    except Exception as e:
        print(f"  Error processing {stim_npz_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# IMPROVED SHUFFLE / STATISTICAL SUMMARY PLOTS
# =============================================================================

def _get_classification_style(classifications):
    """
    Return colors and labels for classification values.

    Classification convention:
        +1 = significant increase
        -1 = significant decrease
         0 = no significant change
    """
    colors = []
    labels = []

    for c in classifications:
        if c == 1:
            colors.append("#d62728")  # red
            labels.append("Significant increase")
        elif c == -1:
            colors.append("#1f77b4")  # blue
            labels.append("Significant decrease")
        else:
            colors.append("#bdbdbd")  # gray
            labels.append("No significant change")

    return colors, labels


def _extract_stat_arrays(result_dict, id_label="Channel"):
    """
    Convert a result dictionary into clean plotting arrays.

    Works for NPRW dictionaries keyed by channel and UA dictionaries keyed by
    physical electrode ID.
    """
    ids = []
    stim_means = []
    ctrl_means = []
    diffs = []
    p_values = []
    adjusted_p = []
    classifications = []
    significant = []

    for key, row in result_dict.items():
        # Prefer explicit electrode/channel labels if available.
        if "electrode_id" in row:
            point_id = row["electrode_id"]
        elif "electrode" in row:
            point_id = row["electrode"]
        elif "channel" in row:
            point_id = row["channel"]
        else:
            point_id = key

        ids.append(point_id)
        stim_means.append(row.get("stim_mean", np.nan))
        ctrl_means.append(row.get("ctrl_mean", np.nan))
        diffs.append(row.get("diff", np.nan))
        p_values.append(row.get("p_value", np.nan))
        adjusted_p.append(row.get("adjusted_p", row.get("p_value", np.nan)))
        classifications.append(row.get("classification", 0))
        significant.append(bool(row.get("significant", False)))

    ids = np.array(ids, dtype=object)
    stim_means = np.array(stim_means, dtype=float)
    ctrl_means = np.array(ctrl_means, dtype=float)
    diffs = np.array(diffs, dtype=float)
    p_values = np.array(p_values, dtype=float)
    adjusted_p = np.array(adjusted_p, dtype=float)
    classifications = np.array(classifications, dtype=int)
    significant = np.array(significant, dtype=bool)

    valid = (
        np.isfinite(stim_means)
        & np.isfinite(ctrl_means)
        & np.isfinite(diffs)
        & np.isfinite(p_values)
        & np.isfinite(adjusted_p)
    )

    return {
        "ids": ids[valid],
        "stim_means": stim_means[valid],
        "ctrl_means": ctrl_means[valid],
        "diffs": diffs[valid],
        "p_values": p_values[valid],
        "adjusted_p": adjusted_p[valid],
        "classifications": classifications[valid],
        "significant": significant[valid],
    }


def _add_classification_legend(ax):
    """Add a consistent classification legend to a matplotlib axis."""
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="Significant increase",
            markerfacecolor="#d62728",
            markeredgecolor="k",
            markersize=8,
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="Significant decrease",
            markerfacecolor="#1f77b4",
            markeredgecolor="k",
            markersize=8,
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="No significant change",
            markerfacecolor="#bdbdbd",
            markeredgecolor="k",
            markersize=8,
        ),
    ]

    ax.legend(handles=legend_elements, fontsize=8, frameon=True)


def _label_significant_points(ax, x, y, ids, classifications, max_labels=20):
    """
    Label significant points on scatter-like plots.

    To avoid unreadable figures, labels are capped.
    """
    sig_idx = np.where(classifications != 0)[0]

    if len(sig_idx) == 0:
        return

    # If there are too many significant points, label the strongest ones by y.
    if len(sig_idx) > max_labels:
        order = np.argsort(y[sig_idx])[::-1]
        sig_idx = sig_idx[order[:max_labels]]

    for idx in sig_idx:
        ax.annotate(
            str(ids[idx]),
            xy=(x[idx], y[idx]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            alpha=0.85,
        )


def _plot_volcano(ax, ids, diffs, adjusted_p, classifications, alpha, title):
    """
    Volcano plot:
        x = effect size, stim_mean - ctrl_mean
        y = -log10(adjusted p-value)
    """
    p_plot = np.clip(adjusted_p, 1e-300, 1.0)
    neglogp = -np.log10(p_plot)

    colors, _ = _get_classification_style(classifications)

    ax.scatter(
        diffs,
        neglogp,
        c=colors,
        edgecolor="k",
        linewidth=0.4,
        s=45,
        alpha=0.9,
    )

    alpha_line = -np.log10(alpha)
    ax.axhline(
        alpha_line,
        color="k",
        linestyle="--",
        linewidth=1,
        label=f"alpha = {alpha:g}",
    )
    ax.axvline(0, color="0.5", linestyle="-", linewidth=0.8)

    _label_significant_points(
        ax=ax,
        x=diffs,
        y=neglogp,
        ids=ids,
        classifications=classifications,
        max_labels=20,
    )

    ax.set_xlabel("Effect size: stim mean - control mean")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    _add_classification_legend(ax)


def _plot_stim_vs_control(ax, ids, stim_means, ctrl_means, classifications, title):
    """
    Scatter comparing control and stimulation means.

    Points above identity line increased during stimulation.
    Points below identity line decreased during stimulation.
    """
    colors, _ = _get_classification_style(classifications)

    ax.scatter(
        ctrl_means,
        stim_means,
        c=colors,
        edgecolor="k",
        linewidth=0.4,
        s=45,
        alpha=0.9,
    )

    all_vals = np.concatenate([ctrl_means, stim_means])
    finite_vals = all_vals[np.isfinite(all_vals)]

    if len(finite_vals) > 0:
        vmin = np.min(finite_vals)
        vmax = np.max(finite_vals)
        if np.isclose(vmin, vmax):
            pad = 1.0 if vmax == 0 else abs(vmax) * 0.1
        else:
            pad = 0.05 * (vmax - vmin)
        lo = vmin - pad
        hi = vmax + pad

        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="No change")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    _label_significant_points(
        ax=ax,
        x=ctrl_means,
        y=stim_means,
        ids=ids,
        classifications=classifications,
        max_labels=20,
    )

    ax.set_xlabel("Control mean")
    ax.set_ylabel("Stim mean")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def _plot_ranked_effects(ax, ids, diffs, classifications, title, id_label="Channel"):
    """
    Ranked effect-size plot.

    Sorts all channels/electrodes by stim-control difference.
    """
    if len(diffs) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    order = np.argsort(diffs)
    sorted_diffs = diffs[order]
    sorted_ids = ids[order]
    sorted_classifications = classifications[order]

    colors, _ = _get_classification_style(sorted_classifications)

    x = np.arange(len(sorted_diffs))

    ax.bar(
        x,
        sorted_diffs,
        color=colors,
        edgecolor="k",
        linewidth=0.3,
        alpha=0.9,
    )

    ax.axhline(0, color="k", linewidth=0.8)

    # Only show all tick labels if not too crowded.
    if len(sorted_ids) <= 40:
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in sorted_ids], rotation=90, fontsize=7)
    else:
        ax.set_xticks([])

    ax.set_xlabel(f"{id_label}s ranked by effect size")
    ax.set_ylabel("Effect size: stim mean - control mean")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)


def _plot_sorted_adjusted_p(ax, ids, adjusted_p, classifications, alpha, title, id_label="Channel"):
    """
    Sorted adjusted p-value diagnostic plot.

    This is more interpretable than a histogram because it shows which ranked
    channels/electrodes cross the significance threshold.
    """
    if len(adjusted_p) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    order = np.argsort(adjusted_p)
    sorted_p = adjusted_p[order]
    sorted_ids = ids[order]
    sorted_classifications = classifications[order]

    colors, _ = _get_classification_style(sorted_classifications)

    x = np.arange(len(sorted_p))

    ax.scatter(
        x,
        sorted_p,
        c=colors,
        edgecolor="k",
        linewidth=0.4,
        s=40,
        alpha=0.9,
    )

    ax.axhline(
        alpha,
        color="k",
        linestyle="--",
        linewidth=1,
        label=f"alpha = {alpha:g}",
    )

    ax.set_yscale("log")

    # Label the significant points.
    sig_idx = np.where(sorted_classifications != 0)[0]
    max_labels = 20
    if len(sig_idx) > max_labels:
        sig_idx = sig_idx[:max_labels]

    for idx in sig_idx:
        ax.annotate(
            str(sorted_ids[idx]),
            xy=(x[idx], sorted_p[idx]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            alpha=0.85,
        )

    ax.set_xlabel(f"{id_label}s ranked by adjusted p-value")
    ax.set_ylabel("Adjusted p-value")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, which="both")


def _plot_ua_grid_stat_map(ax, result_dict, region, alpha, title):
    """
    Utah Array physical grid plot.

    Color indicates effect size:
        red = increased during stimulation
        blue = decreased during stimulation

    Significant electrodes are marked with a black star.
    """
    grid = UTAH_ELEC_GRIDS.get(region)

    if grid is None:
        ax.text(
            0.5,
            0.5,
            f"No Utah grid available for region:\n{region}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        ax.axis("off")
        return

    grid_arr = np.array(grid)

    effect_grid = np.full(grid_arr.shape, np.nan, dtype=float)
    sig_grid = np.zeros(grid_arr.shape, dtype=bool)

    # Build electrode -> row dictionary.
    elec_to_row = {}
    for key, row in result_dict.items():
        elec = row.get("electrode_id", row.get("electrode", key))
        try:
            elec = int(elec)
        except Exception:
            continue
        elec_to_row[elec] = row

    for r in range(grid_arr.shape[0]):
        for c in range(grid_arr.shape[1]):
            try:
                elec_id = int(grid_arr[r, c])
            except Exception:
                continue

            if elec_id in elec_to_row:
                row = elec_to_row[elec_id]
                diff = row.get("diff", np.nan)
                adjusted_p = row.get("adjusted_p", row.get("p_value", np.nan))
                classification = row.get("classification", 0)

                if np.isfinite(diff):
                    effect_grid[r, c] = diff

                if np.isfinite(adjusted_p) and adjusted_p < alpha and classification != 0:
                    sig_grid[r, c] = True

    finite_effects = effect_grid[np.isfinite(effect_grid)]

    if len(finite_effects) == 0:
        ax.text(
            0.5,
            0.5,
            "No plottable UA effects",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        ax.axis("off")
        return

    absmax = np.nanmax(np.abs(finite_effects))
    if absmax == 0 or not np.isfinite(absmax):
        absmax = 1.0

    im = ax.imshow(
        effect_grid,
        cmap="coolwarm",
        vmin=-absmax,
        vmax=absmax,
        interpolation="none",
    )

    # Draw electrode labels and significance markers.
    for r in range(grid_arr.shape[0]):
        for c in range(grid_arr.shape[1]):
            try:
                elec_id = int(grid_arr[r, c])
            except Exception:
                continue

            ax.text(
                c,
                r,
                str(elec_id),
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

            if sig_grid[r, c]:
                ax.plot(
                    c,
                    r,
                    marker="*",
                    markersize=13,
                    markeredgecolor="black",
                    markerfacecolor="yellow",
                    markeredgewidth=0.8,
                )

    ax.set_xticks(np.arange(grid_arr.shape[1]))
    ax.set_yticks(np.arange(grid_arr.shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)
    ax.set_xlabel("Utah grid column")
    ax.set_ylabel("Utah grid row")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Effect size: stim mean - control mean")

    ax.text(
        0.02,
        -0.08,
        "* = significant after correction",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="top",
    )


def plot_stat_summary_panel(
    result_dict,
    title,
    save_path,
    alpha=0.05,
    id_label="Channel",
    include_ua_grid=False,
    region=None,
    window=None,
    dpi=300,
):
    """
    Make an interpretable statistical summary figure.

    For NPRW:
        2x2 panel:
            1. Volcano plot
            2. Stim vs control scatter
            3. Ranked effect sizes
            4. Sorted adjusted p-values

    For UA:
        2x2 panel:
            1. Volcano plot
            2. Stim vs control scatter
            3. Ranked effect sizes
            4. Utah grid effect map
    """
    if result_dict is None or len(result_dict) == 0:
        print(f"No data available for plot: {title}")
        return

    arr = _extract_stat_arrays(result_dict, id_label=id_label)

    ids = arr["ids"]
    stim_means = arr["stim_means"]
    ctrl_means = arr["ctrl_means"]
    diffs = arr["diffs"]
    p_values = arr["p_values"]
    adjusted_p = arr["adjusted_p"]
    classifications = arr["classifications"]
    significant = arr["significant"]

    if len(ids) == 0:
        print(f"No finite data available for plot: {title}")
        return

    n_total = len(ids)
    n_inc = int(np.sum(classifications == 1))
    n_dec = int(np.sum(classifications == -1))
    n_no = int(np.sum(classifications == 0))

    if window is not None:
        subtitle = (
            f"{title}\n"
            f"Window: {window[0]} to {window[1]} ms | "
            f"N={n_total} | Increase={n_inc}, Decrease={n_dec}, No change={n_no}"
        )
    else:
        subtitle = (
            f"{title}\n"
            f"N={n_total} | Increase={n_inc}, Decrease={n_dec}, No change={n_no}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(subtitle, fontsize=14, fontweight="bold")

    _plot_volcano(
        ax=axes[0, 0],
        ids=ids,
        diffs=diffs,
        adjusted_p=adjusted_p,
        classifications=classifications,
        alpha=alpha,
        title="Volcano plot",
    )

    _plot_stim_vs_control(
        ax=axes[0, 1],
        ids=ids,
        stim_means=stim_means,
        ctrl_means=ctrl_means,
        classifications=classifications,
        title="Stim mean vs control mean",
    )

    _plot_ranked_effects(
        ax=axes[1, 0],
        ids=ids,
        diffs=diffs,
        classifications=classifications,
        title="Ranked effect sizes",
        id_label=id_label,
    )

    if include_ua_grid and region is not None:
        _plot_ua_grid_stat_map(
            ax=axes[1, 1],
            result_dict=result_dict,
            region=region,
            alpha=alpha,
            title=f"Utah grid map: {region}",
        )
    else:
        _plot_sorted_adjusted_p(
            ax=axes[1, 1],
            ids=ids,
            adjusted_p=adjusted_p,
            classifications=classifications,
            alpha=alpha,
            title="Sorted adjusted p-values",
            id_label=id_label,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved statistical summary plot: {save_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 60)
    
    if RUN_PLOTTING_ANALYSIS and RUN_SHUFFLE_ANALYSIS:
        print("WARNING: Both RUN_PLOTTING_ANALYSIS and RUN_SHUFFLE_ANALYSIS are True.")
        print("Running BOTH analyses. Set one to False to run only one.")
    elif not RUN_PLOTTING_ANALYSIS and not RUN_SHUFFLE_ANALYSIS:
        print("ERROR: Both RUN_PLOTTING_ANALYSIS and RUN_SHUFFLE_ANALYSIS are False.")
        print("Set one to True to run an analysis.")
        return
    
    if RUN_PLOTTING_ANALYSIS:
        print("PERISTIMULUS RASTER/PSTH PLOT GENERATOR")
    if RUN_SHUFFLE_ANALYSIS:
        print("SHUFFLE-BASED STATISTICAL ANALYSIS")
    
    print("=" * 60)
    print(f"Configuration:")
    print(f"  RUN_PLOTTING_ANALYSIS: {RUN_PLOTTING_ANALYSIS}")
    print(f"  RUN_SHUFFLE_ANALYSIS:  {RUN_SHUFFLE_ANALYSIS}")
    print(f"  PROCESS_NPRW:          {PROCESS_NPRW}")
    print(f"  PROCESS_UA:            {PROCESS_UA}")
    
    if RUN_PLOTTING_ANALYSIS:
        print(f"  PROCESS_DIFF_PLOTS:    {PROCESS_DIFF_PLOTS}")
        print(f"  STANDARD_VIEW:         {GENERATE_STANDARD_VIEW}")
        print(f"  ZOOM_VIEW:             {GENERATE_ZOOM_VIEW}")
    
    if RUN_SHUFFLE_ANALYSIS:
        print(f"  SHUFFLE_N_ITERATIONS:  {SHUFFLE_N_ITERATIONS}")
        print(f"  SHUFFLE_ALPHA:         {SHUFFLE_ALPHA}")
        print(f"  SHUFFLE_USE_FDR:       {SHUFFLE_USE_FDR}")
        print(f"  SHUFFLE_BIN_MS:        {SHUFFLE_BIN_MS}")
        print(f"  Region windows:")
        for region, win in REGION_ANALYSIS_WINDOWS.items():
            print(f"    {region}: {win}")
        print(f"  NPRW window: {NPRW_ANALYSIS_WINDOW}")
    
    print(f"  N_JOBS:                {N_JOBS}")
    print(f"  DPI:                   {DPI_OUTPUT}")
    print("=" * 60)
    
    # Verify electrode mapping loaded correctly
    if elec_info_global:
        print(f"Electrode mapping loaded: {len(elec_info_global)} electrodes")
        port_a_count = sum(1 for e in elec_info_global.values() if e['port'] == 'A')
        port_b_count = sum(1 for e in elec_info_global.values() if e['port'] == 'B')
        print(f"  Port A: {port_a_count} electrodes")
        print(f"  Port B: {port_b_count} electrodes")
        regions = set(e['region'] for e in elec_info_global.values())
        print(f"  Regions: {sorted(regions)}")
    else:
        print("WARNING: Electrode mapping not loaded!")
    
    print("=" * 60)
    
    # Find all peristim .npz files
    files = sorted(PERI_ROOT.rglob("peristim__*.npz"))
    
    if not files:
        print(f"No files found in {PERI_ROOT}")
        return
        
    print(f"Found {len(files)} files.")
    
    # Load Metadata CSV
    metadata_df = None
    if METADATA_CSV.exists():
        print(f"Loading metadata from {METADATA_CSV}")
        metadata_df = pd.read_csv(METADATA_CSV)

    # Load reference data (controls and at-rest baselines)
    print("Loading reference data...")
    references = load_reference_data(files, metadata_df=metadata_df, for_shuffle=RUN_SHUFFLE_ANALYSIS)
    
    print(f"  Control baselines: {list(references['controls'].keys())}")
    if references['rest']:
        print(f"  At-rest baselines: {list(references['rest'].keys())}")

    # ==========================================================================
    # RUN PLOTTING ANALYSIS
    # ==========================================================================
    if RUN_PLOTTING_ANALYSIS:
        print("\n" + "=" * 60)
        print("RUNNING PLOTTING ANALYSIS")
        print("=" * 60)
        
        print(f"\nProcessing files with {N_JOBS} workers...")
        Parallel(n_jobs=N_JOBS, backend=PARALLEL_BACKEND, verbose=10)(
            delayed(process_file)(f, references, metadata_df) 
            for f in files
        )
        
        print("\nPlotting analysis complete!")

    
    # ==========================================================================
    # RUN SHUFFLE ANALYSIS
    # ==========================================================================
    if RUN_SHUFFLE_ANALYSIS:
        print("\n" + "=" * 60)
        print("RUNNING STATISTICAL ANALYSIS")
        print("=" * 60 + "\n")

        SHUFFLE_OUT_ROOT.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------------------
        # Find stim files
        # ----------------------------------------------------------------------
        stim_files = [
            f for f in files
            if "stim_reaches" in str(f).lower()
        ]

        print(f"Found {len(stim_files)} stim files for analysis.\n")

        if not stim_files:
            print("No stim files found!")

        else:
            # ------------------------------------------------------------------
            # Config for analysis
            # ------------------------------------------------------------------
            analysis_config = {
                'process_nprw': PROCESS_NPRW,
                'process_ua': PROCESS_UA,
                'analysis_windows': COMMON_ANALYSIS_WINDOWS,
                'nprw_window': NPRW_ANALYSIS_WINDOW,
                'region_windows': REGION_ANALYSIS_WINDOWS,
                'bin_ms': SHUFFLE_BIN_MS,
                'test': 'mannwhitney',
            }

            # ------------------------------------------------------------------
            # Load bad channels/electrodes
            # ------------------------------------------------------------------
            try:
                bad_map = get_session_impedances(
                    SESSION_LOC,
                    utah_exclude_low=False
                )

                bad_ua_set = bad_map.get('utah', set())
                bad_nprw_set = bad_map.get('nprw', set())

            except Exception as e:
                print(f"  Warning: could not load bad channels: {e}")
                bad_ua_set = set()
                bad_nprw_set = set()

            # Current run_analysis_ua uses bad_channels only for UA.
            # NPRW currently does not exclude bad channels in the stats path.
            bad_channels_set = bad_ua_set

            # ------------------------------------------------------------------
            # Run per-file statistical analysis
            # ------------------------------------------------------------------
            print("Running analysis...")

            all_results = []

            # You can parallelize this later. Keep serial first because errors
            # are much easier to read.
            for stim_path in tqdm(stim_files, desc="Analysis"):
                result = process_file_shuffle(
                    stim_npz_path=stim_path,
                    references=references,
                    metadata_df=metadata_df,
                    electrode_mapping=None,
                    bad_channels=bad_channels_set,
                    config=analysis_config
                )

                if result is not None:
                    all_results.append(result)

            print(f"\nProcessed {len(all_results)} conditions.\n")

            # ------------------------------------------------------------------
            # Apply multiple-comparison correction per condition
            # ------------------------------------------------------------------
            condition_summaries = []

            for result in all_results:
                br_idx = result['br_idx']
                target = result['target']

                # ==============================================================
                # NPRW correction
                # ==============================================================
                if result.get('NPRW'):
                    nprw_results = result['NPRW']

                    channels = list(nprw_results.keys())
                    p_values = np.array([
                        nprw_results[ch].get('p_value', 1.0)
                        for ch in channels
                    ], dtype=float)

                    diffs = np.array([
                        nprw_results[ch].get('diff', np.nan)
                        for ch in channels
                    ], dtype=float)

                    p_values[~np.isfinite(p_values)] = 1.0
                    diffs[~np.isfinite(diffs)] = 0.0

                    if SHUFFLE_USE_FDR:
                        significant, adjusted_p = apply_fdr_correction(
                            p_values,
                            SHUFFLE_ALPHA
                        )
                        correction_method = 'FDR'
                    else:
                        significant, adjusted_p = apply_bonferroni_correction(
                            p_values,
                            SHUFFLE_ALPHA
                        )
                        correction_method = 'Bonferroni'

                    classifications = classify_channels(
                        significant,
                        diffs
                    )

                    n_increase = int(np.sum(classifications == 1))
                    n_decrease = int(np.sum(classifications == -1))
                    n_no_change = int(np.sum(classifications == 0))
                    n_total = int(len(classifications))
                    n_significant = n_increase + n_decrease

                    condition_summaries.append({
                        'br_idx': br_idx,
                        'target': target,
                        'probe': 'NPRW',
                        'region': 'All',
                        'recording_port': '',
                        'n_total': n_total,
                        'n_significant': n_significant,
                        'n_increase': n_increase,
                        'n_decrease': n_decrease,
                        'n_no_change': n_no_change,
                        'pct_significant': 100 * n_significant / n_total if n_total > 0 else 0,
                        'correction_method': correction_method,
                    })

                    for i, ch in enumerate(channels):
                        nprw_results[ch]['significant'] = bool(significant[i])
                        nprw_results[ch]['adjusted_p'] = float(adjusted_p[i])
                        nprw_results[ch]['classification'] = int(classifications[i])
                        nprw_results[ch]['correction_method'] = correction_method

                # ==============================================================
                # UA correction by region
                # ==============================================================
                if result.get('UA'):
                    ua_results = result['UA']

                    regions_in_data = sorted(set(
                        r.get('region', 'Unknown')
                        for r in ua_results.values()
                    ))

                    for region in regions_in_data:
                        region_electrodes = [
                            e for e, r in ua_results.items()
                            if r.get('region', 'Unknown') == region
                        ]

                        if not region_electrodes:
                            continue

                        p_values = np.array([
                            ua_results[e].get('p_value', 1.0)
                            for e in region_electrodes
                        ], dtype=float)

                        diffs = np.array([
                            ua_results[e].get('diff', np.nan)
                            for e in region_electrodes
                        ], dtype=float)

                        p_values[~np.isfinite(p_values)] = 1.0
                        diffs[~np.isfinite(diffs)] = 0.0

                        if SHUFFLE_USE_FDR:
                            significant, adjusted_p = apply_fdr_correction(
                                p_values,
                                SHUFFLE_ALPHA
                            )
                            correction_method = 'FDR'
                        else:
                            significant, adjusted_p = apply_bonferroni_correction(
                                p_values,
                                SHUFFLE_ALPHA
                            )
                            correction_method = 'Bonferroni'

                        classifications = classify_channels(
                            significant,
                            diffs
                        )

                        n_increase = int(np.sum(classifications == 1))
                        n_decrease = int(np.sum(classifications == -1))
                        n_no_change = int(np.sum(classifications == 0))
                        n_total = int(len(classifications))
                        n_significant = n_increase + n_decrease

                        recording_ports = [
                            ua_results[e].get('recording_port', '')
                            for e in region_electrodes
                        ]
                        recording_port = recording_ports[0] if recording_ports else ''

                        condition_summaries.append({
                            'br_idx': br_idx,
                            'target': target,
                            'probe': 'UA',
                            'region': region,
                            'recording_port': recording_port,
                            'n_total': n_total,
                            'n_significant': n_significant,
                            'n_increase': n_increase,
                            'n_decrease': n_decrease,
                            'n_no_change': n_no_change,
                            'pct_significant': 100 * n_significant / n_total if n_total > 0 else 0,
                            'correction_method': correction_method,
                        })

                        for i, elec_id in enumerate(region_electrodes):
                            ua_results[elec_id]['significant'] = bool(significant[i])
                            ua_results[elec_id]['adjusted_p'] = float(adjusted_p[i])
                            ua_results[elec_id]['classification'] = int(classifications[i])
                            ua_results[elec_id]['correction_method'] = correction_method

            # ------------------------------------------------------------------
            # Save raw Python object
            # ------------------------------------------------------------------
            raw_path = SHUFFLE_OUT_ROOT / "all_results_raw.npy"

            np.save(
                raw_path,
                np.asarray(all_results, dtype=object),
                allow_pickle=True
            )

            print(f"  Saved raw results: {raw_path}")

            # ------------------------------------------------------------------
            # Save per-condition summary
            # ------------------------------------------------------------------
            summary_df = pd.DataFrame(condition_summaries)
            summary_path = SHUFFLE_OUT_ROOT / "condition_summary.csv"
            summary_df.to_csv(summary_path, index=False)

            print(f"  Saved condition summary: {summary_path}")

            # ------------------------------------------------------------------
            # Save detailed long-format CSV
            # ------------------------------------------------------------------
            detailed_rows = []

            for result in all_results:
                br_idx = result['br_idx']
                target = result['target']
                stim_file = result.get('stim_file', '')
                ctrl_file = result.get('ctrl_file', '')
                n_stim_trials = result.get('n_stim_trials', np.nan)
                n_ctrl_trials = result.get('n_ctrl_trials', np.nan)

                # NPRW details
                for ch, r in result.get('NPRW', {}).items():
                    detailed_rows.append({
                        'br_idx': br_idx,
                        'target': target,
                        'probe': 'NPRW',
                        'region': 'NPRW',
                        'recording_port': '',
                        'channel': ch,
                        'channel_index': r.get('channel_index', np.nan),
                        'electrode_id': np.nan,
                        'stim_channel_index': r.get('channel_index', np.nan),
                        'ctrl_channel_index': r.get('channel_index', np.nan),
                        'p_value': r.get('p_value', np.nan),
                        'adjusted_p': r.get('adjusted_p', np.nan),
                        'stim_mean': r.get('stim_mean', np.nan),
                        'ctrl_mean': r.get('ctrl_mean', np.nan),
                        'diff': r.get('diff', np.nan),
                        'significant': r.get('significant', False),
                        'classification': r.get('classification', 0),
                        'classification_label': (
                            'Increase' if r.get('classification', 0) == 1
                            else 'Decrease' if r.get('classification', 0) == -1
                            else 'No Change'
                        ),
                        'window_start': r.get('window', (np.nan, np.nan))[0],
                        'window_end': r.get('window', (np.nan, np.nan))[1],
                        'n_bins': r.get('n_bins', np.nan),
                        'n_stim_trials': n_stim_trials,
                        'n_ctrl_trials': n_ctrl_trials,
                        'stim_file': stim_file,
                        'ctrl_file': ctrl_file,
                        'correction_method': r.get('correction_method', ''),
                    })

                # UA details
                for elec_id, r in result.get('UA', {}).items():
                    detailed_rows.append({
                        'br_idx': br_idx,
                        'target': target,
                        'probe': 'UA',
                        'region': r.get('region', ''),
                        'recording_port': r.get('recording_port', ''),
                        'channel': elec_id,
                        'channel_index': r.get('stim_channel_index', np.nan),
                        'electrode_id': elec_id,
                        'stim_channel_index': r.get('stim_channel_index', np.nan),
                        'ctrl_channel_index': r.get('ctrl_channel_index', np.nan),
                        'p_value': r.get('p_value', np.nan),
                        'adjusted_p': r.get('adjusted_p', np.nan),
                        'stim_mean': r.get('stim_mean', np.nan),
                        'ctrl_mean': r.get('ctrl_mean', np.nan),
                        'diff': r.get('diff', np.nan),
                        'significant': r.get('significant', False),
                        'classification': r.get('classification', 0),
                        'classification_label': (
                            'Increase' if r.get('classification', 0) == 1
                            else 'Decrease' if r.get('classification', 0) == -1
                            else 'No Change'
                        ),
                        'window_start': r.get('window', (np.nan, np.nan))[0],
                        'window_end': r.get('window', (np.nan, np.nan))[1],
                        'n_bins': r.get('n_bins', np.nan),
                        'n_stim_trials': n_stim_trials,
                        'n_ctrl_trials': n_ctrl_trials,
                        'stim_file': stim_file,
                        'ctrl_file': ctrl_file,
                        'correction_method': r.get('correction_method', ''),
                    })

            detailed_df = pd.DataFrame(detailed_rows)
            detailed_path = SHUFFLE_OUT_ROOT / "detailed_results.csv"
            detailed_df.to_csv(detailed_path, index=False)

            print(f"  Saved detailed results: {detailed_path}")

            # ------------------------------------------------------------------
            # Print per-condition summary
            # ------------------------------------------------------------------
            print("\n" + "=" * 60)
            print("PER-CONDITION SUMMARY")
            print("=" * 60)

            if len(summary_df) > 0:
                for (br_idx, target), group in summary_df.groupby(['br_idx', 'target']):
                    print(f"\nBR{int(br_idx):03d} ({target}):")
                    print("-" * 50)

                    for _, row in group.iterrows():
                        if row['probe'] == 'UA':
                            probe_region = f"{row['probe']} {row['region']} Port {row['recording_port']}"
                        else:
                            probe_region = row['probe']

                        print(
                            f"  {probe_region:18s}: "
                            f"{int(row['n_significant']):3d}/{int(row['n_total']):3d} significant "
                            f"({row['pct_significant']:5.1f}%) | "
                            f"+{int(row['n_increase']):2d} "
                            f"-{int(row['n_decrease']):2d}"
                        )
            else:
                print("No significant summary data generated.")

            print("=" * 60)

            # ------------------------------------------------------------------
            # Generate improved per-condition statistical summary plots
            # ------------------------------------------------------------------
            print("\nGenerating improved per-condition statistical summary plots...")

            for result in tqdm(all_results, desc="Plotting"):
                br_idx = result['br_idx']
                target = result['target']

                safe_target = str(target).replace("/", "_").replace("\\", "_").replace(" ", "_")

                cond_dir = SHUFFLE_OUT_ROOT / f"BR_{br_idx:03d}" / safe_target
                cond_dir.mkdir(parents=True, exist_ok=True)

                # ==============================================================
                # NPRW summary plot
                # ==============================================================
                nprw_results = result.get('NPRW', {})

                if nprw_results:
                    first_nprw = next(iter(nprw_results.values()))
                    nprw_window = first_nprw.get("window", NPRW_ANALYSIS_WINDOW)

                    nprw_save_path = cond_dir / "NPRW_summary.png"

                    plot_stat_summary_panel(
                        result_dict=nprw_results,
                        title=f"BR{br_idx:03d} | {target} | NPRW",
                        save_path=nprw_save_path,
                        alpha=SHUFFLE_ALPHA,
                        id_label="Channel",
                        include_ua_grid=False,
                        region=None,
                        window=nprw_window,
                        dpi=DPI_OUTPUT,
                    )

                # ==============================================================
                # UA summary plots by region
                # ==============================================================
                ua_results = result.get('UA', {})

                if ua_results:
                    regions = sorted(set(
                        row.get('region', 'Unknown')
                        for row in ua_results.values()
                    ))

                    for region in regions:
                        region_data = {
                            elec_id: row
                            for elec_id, row in ua_results.items()
                            if row.get('region', 'Unknown') == region
                        }

                        if not region_data:
                            continue

                        first_ua = next(iter(region_data.values()))
                        ua_window = first_ua.get(
                            "window",
                            REGION_ANALYSIS_WINDOWS.get(region, (-500.0, 0.0))
                        )

                        recording_port = first_ua.get("recording_port", "")

                        safe_region = (
                            str(region)
                            .replace("/", "_")
                            .replace("\\", "_")
                            .replace(" ", "_")
                        )

                        ua_save_path = cond_dir / f"UA_{safe_region}_summary.png"

                        plot_stat_summary_panel(
                            result_dict=region_data,
                            title=f"BR{br_idx:03d} | {target} | UA {region} Port {recording_port}",
                            save_path=ua_save_path,
                            alpha=SHUFFLE_ALPHA,
                            id_label="Electrode",
                            include_ua_grid=True,
                            region=region,
                            window=ua_window,
                            dpi=DPI_OUTPUT,
                        )

            print(f"\nAnalysis complete! Results saved to: {SHUFFLE_OUT_ROOT}")


if __name__ == "__main__":
    main()