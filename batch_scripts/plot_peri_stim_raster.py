from pathlib import Path
import numpy as np
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

# =============================================================================
# CONFIGURATION FLAGS
# =============================================================================

# Which probe types to process
PROCESS_NPRW = False          # Set False to skip NPRW plots
PROCESS_UA = True            # Set False to skip Utah Array plots
PROCESS_DIFF_PLOTS = True    # Set False to skip difference plots (Stim - Control, etc.)

# Which plot views to generate
GENERATE_STANDARD_VIEW = True   # (-500, 500) ms window
GENERATE_ZOOM_VIEW = True       # Dynamic zoom window with 5ms bins

# Parallel processing settings
N_JOBS = 8                   # Number of parallel jobs (-1 for all cores)
PARALLEL_BACKEND = 'loky'    # 'loky', 'threading', or 'multiprocessing'

# Plot quality settings
DPI_OUTPUT = 100             # Lower = faster, smaller files (try 72 for drafts)
SKIP_EXISTING = False        # Skip files that already have output plots

# Debug/verbose output
VERBOSE = True               # Print debug messages

# =============================================================================
# PLOT SETTINGS
# =============================================================================

# Time window for plots
WIN_PLOT_MS = (-500.0, 500.0)

# Build PLOT_VIEWS based on configuration
PLOT_VIEWS = []
if GENERATE_STANDARD_VIEW:
    PLOT_VIEWS.append(((-500.0, 500.0), None, ''))        # standard view
if GENERATE_ZOOM_VIEW:
    PLOT_VIEWS.append((None, 5.0, '_zoom'))               # zoomed view with 5 ms bins

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
DIFF_YLIM = (-5, 12)

# Figure sizes
FIG_SIZE_RW = (48, 24)
FIG_SIZE_UA = (32, 24)

# Output directory
FIG_ROOT = OUT_BASE / "figures" / "peristim_raster"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim"


# =============================================================================
# ELECTRODE MAPPING (Cached)
# =============================================================================

@lru_cache(maxsize=1)
def load_electrode_mapping_cached(csv_path_str: str):
    """
    Load electrode mapping from CSV (cached).
    
    Returns:
        elec_info: dict mapping electrode_id -> {
            'nsp_id': global NSP ID (1-256),
            'port': 'A' or 'B',
            'region': array name,
            'row': grid row,
            'col': grid col,
            'local_nsp': local NSP ID (1-128)
        }
        region_grids: {region: 8x8 array of electrode IDs}
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
            if 'A' in port_str.upper():
                port = 'A'
                local_nsp = nsp_id  # Port A: global 1-128 = local 1-128
            else:
                port = 'B'
                local_nsp = nsp_id - 128  # Port B: global 129-256 = local 1-128

            elec_info[elec_id] = {
                'nsp_id': nsp_id,
                'port': port,
                'region': region,
                'row': r,
                'col': c,
                'local_nsp': local_nsp
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


def build_elec_to_data_idx(ua_ids_1based, elec_info, recording_port='A'):
    """
    Build electrode_id -> channel index mapping for a specific recording port.
    
    Parameters
    ----------
    ua_ids_1based : array-like or None
        Local NSP IDs for each channel (1-128 as recorded).
        If None, assumes sequential 1-128.
    elec_info : dict
        Mapping from electrode ID to electrode info (from CSV).
    recording_port : str
        'A' or 'B' - which port the recording was made from.
    
    Returns
    -------
    elec_to_idx : dict
        Mapping from electrode ID to channel index in the data array.
    """
    elec_to_idx = {}
    
    # Build reverse lookup: for the recording port, map local_nsp -> electrode_id
    local_nsp_to_elec = {}
    for elec_id, info in elec_info.items():
        if info['port'] == recording_port.upper():
            local_nsp_to_elec[info['local_nsp']] = elec_id
    
    if ua_ids_1based is not None and len(ua_ids_1based) > 0:
        for ch_idx, local_nsp_id in enumerate(ua_ids_1based):
            try:
                local_nsp_id = int(local_nsp_id)
            except (ValueError, TypeError):
                continue
            
            elec_id = local_nsp_to_elec.get(local_nsp_id)
            if elec_id is not None:
                elec_to_idx[elec_id] = ch_idx
    else:
        # Fallback: assume sequential local NSP IDs 1-128
        for ch_idx in range(128):
            local_nsp_id = ch_idx + 1
            elec_id = local_nsp_to_elec.get(local_nsp_id)
            if elec_id is not None:
                elec_to_idx[elec_id] = ch_idx
    
    return elec_to_idx


def get_active_regions(ua_ids_1based, elec_info, recording_port='A'):
    """
    Determine which regions have active channels in this recording.
    
    Parameters
    ----------
    ua_ids_1based : array-like
        Local NSP IDs (1-128) for each channel.
    elec_info : dict
        Electrode info dictionary.
    recording_port : str
        'A' or 'B' for which port was recorded.
    
    Returns
    -------
    list of str
        Region names that have active channels, in REGION_ORDER.
    """
    REGION_ORDER = ["M1i", "M1s", "PMd", "SMA"]
    
    # Build reverse lookup for this port
    local_nsp_to_elec = {}
    for elec_id, info in elec_info.items():
        if info['port'] == recording_port.upper():
            local_nsp_to_elec[info['local_nsp']] = elec_id
    
    active_regs = set()
    
    if ua_ids_1based is not None:
        for local_nsp_id in ua_ids_1based:
            try:
                local_nsp_id = int(local_nsp_id)
                elec_id = local_nsp_to_elec.get(local_nsp_id)
                if elec_id is not None:
                    reg = elec_info[elec_id]['region']
                    active_regs.add(reg)
            except:
                continue
    
    return [r for r in REGION_ORDER if r in active_regs]


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
MAPPING_CSV = Path(__file__).parent.parent / "scripts" / "electrode_port_mapping.csv"
elec_info_global, UTAH_ELEC_GRIDS = load_electrode_mapping(MAPPING_CSV)


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

def load_reference_data(all_files, metadata_df=None):
    """
    Scans all files for 'control_reaches' and 'at_rest'.
    """
    if not PROCESS_DIFF_PLOTS:
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
                        'NPRW': None, 'UA': None, 
                        'NPRW_peaks': None, 'UA_peaks': None, 
                        'ctrl_events': None
                    }
                    if 'event_ms' in data: 
                        entry['ctrl_events'] = np.asarray(data['event_ms']).flatten()
                    if PROCESS_NPRW and 'NPRW_counts' in data:
                        entry['NPRW'] = np.nansum(data['NPRW_counts'], axis=0) / max(n_trials, 1)
                        if 'NPRW_edges_ms' in data: entry['NPRW_edges_ms'] = data['NPRW_edges_ms']
                        if 'NPRW_peak_ms_dedup' in data: entry['NPRW_peaks'] = data['NPRW_peak_ms_dedup'].item()
                    if PROCESS_UA and 'UA_counts' in data:
                        entry['UA'] = np.nansum(data['UA_counts'], axis=0) / max(n_trials, 1)
                        if 'UA_edges_ms' in data: entry['UA_edges_ms'] = data['UA_edges_ms']
                        if 'UA_peak_ms_dedup' in data: entry['UA_peaks'] = data['UA_peak_ms_dedup'].item()
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
                    'NPRW': None, 'UA': None, 
                    'NPRW_peaks': None, 'UA_peaks': None, 
                    'ctrl_events': None
                }
                if 'event_ms' in data: 
                    entry['ctrl_events'] = np.asarray(data['event_ms']).flatten()
                if PROCESS_NPRW and 'NPRW_counts' in data:
                    entry['NPRW'] = np.nansum(data['NPRW_counts'], axis=0) / max(n_trials, 1)
                    if 'NPRW_edges_ms' in data: entry['NPRW_edges_ms'] = data['NPRW_edges_ms']
                    if 'NPRW_peak_ms_dedup' in data: entry['NPRW_peaks'] = data['NPRW_peak_ms_dedup'].item()
                if PROCESS_UA and 'UA_counts' in data:
                    entry['UA'] = np.nansum(data['UA_counts'], axis=0) / max(n_trials, 1)
                    if 'UA_edges_ms' in data: entry['UA_edges_ms'] = data['UA_edges_ms']
                    if 'UA_peak_ms_dedup' in data: entry['UA_peaks'] = data['UA_peak_ms_dedup'].item()
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
# NPRW PROCESSING
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
    
    # Loop over plot views
    for view_win_in, view_bin_ms, view_suffix in PLOT_VIEWS:
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
            
            ch_idx = i 
            binned = counts_data[:, ch_idx, :] if ch_idx < counts_data.shape[1] else None
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

        # Difference plots
        if PROCESS_DIFF_PLOTS and meta['cond_type'] == "STIM" and references is not None:
            _process_nprw_diff_plots(
                counts_data, centers_ms, edges_ms, view_win, view_bin_ms, view_suffix,
                sorted_ch_ids, n_ch, rows, cols, meta, references
            )


def _process_nprw_diff_plots(counts_data, centers_ms, edges_ms, view_win, view_bin_ms, view_suffix,
                              sorted_ch_ids, n_ch, rows, cols, meta, references):
    """Generate NPRW difference plots."""
    ctrl_entry = references['controls'].get(meta['target_folder'])
    rest_entry = references['rest'].get((meta['stim_freq_hz'], meta['stim_dur_ms']))
    
    def get_norm_ref(entry, ref_type='NPRW'):
        if entry is None or entry.get(f'{ref_type}_peaks') is None:
            return None
        pk_dict = entry[f'{ref_type}_peaks']
        ev_ms = entry['ctrl_events'] if entry.get('ctrl_events') is not None else []
        if len(ev_ms) == 0: 
            return None
        
        c_ref, _, _ = _rebin_from_peaks_fast(
            pk_dict, ev_ms, view_win, 
            view_bin_ms if view_bin_ms else 20.0,
            0.0, 0.0, 0.0
        )
        return np.nansum(c_ref, axis=0) / max(c_ref.shape[0], 1)

    ctrl_norm = get_norm_ref(ctrl_entry, 'NPRW')
    rest_norm = get_norm_ref(rest_entry, 'NPRW')
    
    diff_configs = []
    if ctrl_norm is not None:
        diff_configs.append(('stim_control_diff', 'Stim - Control', ctrl_norm))
    if rest_norm is not None:
        diff_configs.append(('stim_rest_diff', 'Stim - Rest', rest_norm))
    if ctrl_norm is not None and rest_norm is not None:
        diff_configs.append(('stim_rest_control_diff', 'Stim - Control - Rest', ctrl_norm + rest_norm))
    
    stim_norm = np.nansum(counts_data, axis=0) / max(counts_data.shape[0], 1)
    
    for dir_name, title_suffix, baseline_norm in diff_configs:
        if stim_norm.shape != baseline_norm.shape:
            continue
            
        diff_norm = stim_norm - baseline_norm
        fig_diff = plt.figure(figsize=FIG_SIZE_RW)
        gs_diff = gridspec.GridSpec(rows, cols, figure=fig_diff, hspace=0.3, wspace=0.3)
        
        found_diff = False
        for i, ch in enumerate(sorted_ch_ids):
            if i >= n_ch: 
                break
            r, c = i // cols, i % cols
            ax_diff = fig_diff.add_subplot(gs_diff[r, c])
            
            if i < diff_norm.shape[0]:
                found_diff = True
                d_ch = diff_norm[i, :]
                bar_colors = np.where(d_ch >= 0, 'tab:green', 'tab:red')
                
                if centers_ms is not None and len(d_ch) == len(centers_ms):
                    bw = view_bin_ms if view_bin_ms else (np.nanmedian(np.diff(centers_ms)) if len(centers_ms) > 1 else 20.0)
                    ax_diff.bar(centers_ms, d_ch, width=bw, align='center', color=bar_colors, edgecolor='none')
                elif edges_ms is not None:
                    ax_diff.bar(edges_ms[:-1], d_ch, width=np.diff(edges_ms), align='edge', color=bar_colors, edgecolor='none')
                
                if np.nansum(np.abs(d_ch)) > 0:
                    ax_diff.axvline(0, color='r', linestyle='--', linewidth=1, alpha=0.8)
            else:
                ax_diff.axis('off')
            
            ax_diff.set_title(f"Ch {ch}", fontsize=10, pad=2)
            ax_diff.set_xlim(view_win)
            ax_diff.set_ylim(DIFF_YLIM)
            ax_diff.spines['top'].set_visible(False)
            ax_diff.spines['right'].set_visible(False)
            if r < rows - 1: 
                ax_diff.set_xticklabels([])

        if found_diff:
            fig_diff.suptitle(f"{meta['overall_title']} | NPRW Diff ({title_suffix}){meta['stim_chs_str']}", fontsize=20)
            save_diff_dir = FIG_ROOT / dir_name / meta['target_folder']
            save_diff_dir.mkdir(parents=True, exist_ok=True)
            fig_diff_name = f"[diff]_Cond{meta['br_idx']:03d}_{meta['cond_type']}_NPRW{view_suffix}.png"
            fig_diff.savefig(save_diff_dir / fig_diff_name, dpi=DPI_OUTPUT, bbox_inches='tight')
        plt.close(fig_diff)


# =============================================================================
# UTAH ARRAY PROCESSING
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
    def get_peak_times_for_channel(ch_idx):
        """Try multiple key formats to find spike times for a channel."""
        # Try different key formats
        possible_keys = [
            str(ch_idx),           # "0", "1", etc.
            ch_idx,                # 0, 1, etc. (int)
            str(ch_idx + 1),       # "1", "2", etc. (1-based)
            ch_idx + 1,            # 1, 2, etc. (1-based int)
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
    for view_win_in, view_bin_ms, view_suffix in PLOT_VIEWS:
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
                    
                    # Check if this electrode is on the recording port
                    if elec_id in port_electrodes:
                        idx = elec_to_idx.get(elec_id)
                        
                        if idx is not None and 0 <= idx < counts_data.shape[1]:
                            found_channels = True
                            binned = counts_data[:, idx, :]
                            
                            # Get spike times using the helper function
                            peak_times = get_peak_times_for_channel(idx)
                            
                            if len(peak_times) > 0:
                                channels_with_spikes += 1
                            
                            # Get electrode info for display
                            info = elec_info_global.get(elec_id, {})
                            local_nsp = info.get('local_nsp', idx + 1)
                            
                            plot_channel_group_fast(
                                ax_raster, ax_psth, peak_times, meta['event_ms'], binned, edges_ms,
                                f"E{elec_id} (L{local_nsp})", stim_dur_ms=0.0, 
                                blank_pre_ms=0.0, blank_post_ms=0.0, 
                                bin_centers=centers_ms,
                                n_trials_ref=meta['n_trials_ref'], win_ms=view_win, psth_ylim=cur_ylim
                            )
                        else:
                            # Electrode is on this port but not found in data
                            ax_raster.axis('off')
                            ax_psth.axis('off')
                            info = elec_info_global.get(elec_id, {})
                            local_nsp = info.get('local_nsp', '?')
                            ax_raster.text(0.5, 0.5, f"E{elec_id}\nL{local_nsp}\nNo Data", 
                                           ha='center', va='center', 
                                           transform=ax_raster.transAxes, fontsize=8, alpha=0.3)
                    else:
                        # Electrode is on the OTHER port - show as unavailable
                        ax_raster.axis('off')
                        ax_psth.axis('off')
                        if elec_id > 0:
                            other_port = 'B' if recording_port == 'A' else 'A'
                            ax_raster.text(0.5, 0.5, f"E{elec_id}\nPort {other_port}", 
                                           ha='center', va='center', 
                                           transform=ax_raster.transAxes, fontsize=8, 
                                           alpha=0.3, color='gray')
                    
                    if 2*r+1 < 14:
                        ax_psth.set_xticklabels([])

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

        # Difference plots
        if PROCESS_DIFF_PLOTS and meta['cond_type'] == "STIM" and references is not None:
            _process_ua_diff_plots(
                counts_data, centers_ms, edges_ms, view_win, view_bin_ms, view_suffix,
                plot_regions, elec_to_idx, recording_port, ua_ids_1based, meta, references
            )

def _process_ua_diff_plots(counts_data, centers_ms, edges_ms, view_win, view_bin_ms, view_suffix,
                           plot_regions, elec_to_idx, recording_port, ua_ids_1based, meta, references):
    """Generate UA difference plots."""
    ctrl_entry = references['controls'].get(meta['target_folder'])
    rest_entry = references['rest'].get((meta['stim_freq_hz'], meta['stim_dur_ms']))

    def get_norm_ref_ua(entry):
        if entry is None or entry.get('UA_peaks') is None:
            return None
        pk_dict = entry['UA_peaks']
        ev_ms = entry['ctrl_events'] if entry.get('ctrl_events') is not None else []
        if len(ev_ms) == 0: 
            return None
        
        c_ref, _, _ = _rebin_from_peaks_fast(
            pk_dict, ev_ms, view_win, 
            view_bin_ms if view_bin_ms else 20.0, 
            0.0, 0.0, 0.0
        )
        return np.nansum(c_ref, axis=0) / max(c_ref.shape[0], 1)

    ctrl_norm_ua = get_norm_ref_ua(ctrl_entry)
    rest_norm_ua = get_norm_ref_ua(rest_entry)

    diff_configs_ua = []
    if ctrl_norm_ua is not None:
        diff_configs_ua.append(('stim_control_diff', 'Stim - Control', ctrl_norm_ua))
    if rest_norm_ua is not None:
        diff_configs_ua.append(('stim_rest_diff', 'Stim - Rest', rest_norm_ua))
    if ctrl_norm_ua is not None and rest_norm_ua is not None:
        diff_configs_ua.append(('stim_rest_control_diff', 'Stim - Control - Rest', ctrl_norm_ua + rest_norm_ua))
    
    stim_norm = np.nansum(counts_data, axis=0) / max(counts_data.shape[0], 1)
    
    for dir_name, title_suffix, baseline_norm in diff_configs_ua:
        if stim_norm.shape != baseline_norm.shape:
            continue
            
        diff_norm = stim_norm - baseline_norm
        
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
                        
                        if idx is not None and 0 <= idx < diff_norm.shape[0]:
                            found_diff = True
                            d_ch = diff_norm[idx, :]
                            bar_colors = np.where(d_ch >= 0, 'tab:green', 'tab:red')
                            
                            if centers_ms is not None and len(d_ch) == len(centers_ms):
                                bw = view_bin_ms if view_bin_ms else (np.nanmedian(np.diff(centers_ms)) if len(centers_ms) > 1 else 20.0)
                                ax_diff.bar(centers_ms, d_ch, width=bw, align='center', color=bar_colors, edgecolor='none')
                            elif edges_ms is not None:
                                ax_diff.bar(edges_ms[:-1], d_ch, width=np.diff(edges_ms), align='edge', color=bar_colors, edgecolor='none')
                            
                            if np.nansum(np.abs(d_ch)) > 0:
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
                    ax_diff.set_ylim(DIFF_YLIM)
                    ax_diff.tick_params(axis='both', which='major', labelsize=6)
            
            if found_diff:
                fig_diff_ua.suptitle(f"{meta['overall_title']} | Utah Diff ({title_suffix}) - {reg_target} (Port {recording_port}){meta['stim_chs_str']}", fontsize=20)
                save_diff_dir = FIG_ROOT / dir_name / meta['target_folder']
                save_diff_dir.mkdir(parents=True, exist_ok=True)
                fig_name = f"[diff]_Cond{meta['br_idx']:03d}_{meta['cond_type']}_UA_{reg_target}{view_suffix}.png"
                fig_diff_ua.savefig(save_diff_dir / fig_name, dpi=DPI_OUTPUT, bbox_inches='tight')
            plt.close(fig_diff_ua)


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_file(npz_path, references=None, metadata_df=None):
    """Process a single NPZ file."""
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


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 60)
    print("PERISTIMULUS RASTER/PSTH PLOT GENERATOR")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  PROCESS_NPRW:       {PROCESS_NPRW}")
    print(f"  PROCESS_UA:         {PROCESS_UA}")
    print(f"  PROCESS_DIFF_PLOTS: {PROCESS_DIFF_PLOTS}")
    print(f"  STANDARD_VIEW:      {GENERATE_STANDARD_VIEW}")
    print(f"  ZOOM_VIEW:          {GENERATE_ZOOM_VIEW}")
    print(f"  N_JOBS:             {N_JOBS}")
    print(f"  DPI:                {DPI_OUTPUT}")
    print(f"  SKIP_EXISTING:      {SKIP_EXISTING}")
    print("=" * 60)
    
    # Verify electrode mapping loaded correctly
    if elec_info_global:
        print(f"Electrode mapping loaded: {len(elec_info_global)} electrodes")
        # Count electrodes per port
        port_a_count = sum(1 for e in elec_info_global.values() if e['port'] == 'A')
        port_b_count = sum(1 for e in elec_info_global.values() if e['port'] == 'B')
        print(f"  Port A: {port_a_count} electrodes")
        print(f"  Port B: {port_b_count} electrodes")
        # Show regions
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
    references = load_reference_data(files, metadata_df=metadata_df)

    # Process files in parallel
    print(f"\nProcessing files with {N_JOBS} workers...")
    Parallel(n_jobs=N_JOBS, backend=PARALLEL_BACKEND)(
        delayed(process_file)(f, references, metadata_df) 
        for f in tqdm(files, desc="Files", disable=not VERBOSE)
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()