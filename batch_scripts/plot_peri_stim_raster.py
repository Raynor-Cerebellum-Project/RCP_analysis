from pathlib import Path
import numpy as np
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
import json

# --- UTAH ELECTRODE MAPPING (Loaded from CSV) ---
def load_electrode_mapping(csv_path: Path):
    """
    Load electrode mapping from CSV.
    Returns:
        nsp_to_elec: {nsp_id: electrode_id} mapping.
        region_grids: {region: 8x8_array} of electrode IDs.
        elec_to_region: {electrode_id: region} mapping.
    """
    nsp_to_elec = {}
    region_grids = {}
    elec_to_region = {}

    if not csv_path.exists():
        print(f"  [Warn] Electrode mapping CSV not found: {csv_path}")
        return nsp_to_elec, region_grids, elec_to_region

    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            elec_id = int(row['ElectrodeID'])
            nsp_id = int(row['NSP_ID'])
            region = str(row['Array']).strip()
            r = int(row['GridRow'])
            c = int(row['GridCol'])

            nsp_to_elec[nsp_id] = elec_id
            elec_to_region[elec_id] = region

            if region not in region_grids:
                region_grids[region] = np.zeros((8, 8), dtype=int)
            region_grids[region][r, c] = elec_id
            
    except Exception as e:
        print(f"  [Error] Failed to load electrode mapping from {csv_path}: {e}")

    return nsp_to_elec, region_grids, elec_to_region

def build_elec_to_data_idx(ua_ids_1based, nsp_to_elec):
    """
    Build electrode_id -> channel index mapping.
    ua_ids_1based contains NSP IDs (1-256), we need to convert to electrode IDs.
    """
    if ua_ids_1based is None or nsp_to_elec is None:
        return {}
    
    elec_to_idx = {}
    for ch_idx, nsp_id in enumerate(ua_ids_1based):
        try:
            nsp_id = int(nsp_id)
        except:
            continue
        elec_id = nsp_to_elec.get(nsp_id, -1)
        if elec_id > 0:
            elec_to_idx[elec_id] = ch_idx
    
    return elec_to_idx

# Load mapping globally 
MAPPING_CSV = Path(__file__).parent.parent / "scripts" / "electrode_port_mapping.csv"
nsp_to_elec_global, UTAH_ELEC_GRIDS, elec_to_region_global = load_electrode_mapping(MAPPING_CSV)


# Time window for plots
WIN_PLOT_MS = (-500.0, 500.0)

# ──── Plot views: each is (win_ms, bin_ms_override, filename_suffix) ────
# bin_ms_override=None means use the pre-computed bins from the .npz
PLOT_VIEWS = [
    ((-500.0, 500.0), None,  ''),        # standard view
    (None,            5.0,   '_zoom'),   # short / zoomed view with 5 ms bins (dynamic window)
]

# Raster dots style
RASTER_MARKER = 'o'
RASTER_MARKER_SIZE = 2
RASTER_COLOR = 'tab:blue'
RASTER_ALPHA = 0.8

# PSTH style
PSTH_COLOR = 'tab:cyan'
PSTH_EDGE_COLOR = 'k'
PSTH_LINEWIDTH = 0.5
PSTH_YLIM = (0, 200)
PSTH_YLIM_ZOOM = (0, 50)    # lower Y-limit for the zoomed (5 ms bin) plots
DIFF_YLIM = (-5, 12)

# Figure sizes
FIG_SIZE_RW = (48, 24)  # Large horizontal figure for 128 channels
FIG_SIZE_UA = (32, 24)

# Output directory
FIG_ROOT = OUT_BASE / "figures" / "peristim_raster"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim"


def _get_raster_data(peak_times_ms, events_ms, win_ms, blank_pre_ms=0.0, blank_post_ms=0.0):
    """
    Gather spike times relative to each event.
    If blank_pre_ms > 0 or blank_post_ms > 0, exclude spikes in [-blank_pre_ms, blank_post_ms] relative to stim event.
    
    Returns:
        trial_indices: list of trial indices for each spike
        rel_times: list of relative spike times for each spike
    """
    trial_indices = []
    rel_times = []
    
    if peak_times_ms is None or len(peak_times_ms) == 0:
        return trial_indices, rel_times

    peak_times_ms = np.sort(peak_times_ms)
    events_ms = np.sort(events_ms)
    
    # Iterate over trials
    for i, t_stim in enumerate(events_ms):
        t_start = t_stim + win_ms[0]
        t_end = t_stim + win_ms[1]
        
        # Find spikes in window using searchsorted
        idx_start = np.searchsorted(peak_times_ms, t_start)
        idx_end = np.searchsorted(peak_times_ms, t_end)
        
        spikes = peak_times_ms[idx_start:idx_end]
        
        if len(spikes) > 0:
            rel = spikes - t_stim
            
            # Blank spikes during stim artifact
            # We blank from -blank_pre_ms to +blank_post_ms relative to 0
            if blank_pre_ms > 0 or blank_post_ms > 0:
                keep = (rel < -blank_pre_ms) | (rel > blank_post_ms)
                rel = rel[keep]
            if len(rel) > 0:
                trial_indices.extend([i] * len(rel))
                rel_times.extend(rel)
            
    return trial_indices, rel_times


def _rebin_from_peaks(peaks_dict, events_ms, win_ms, bin_ms, stim_dur_ms=0.0,
                      blank_pre_ms=0.0, blank_post_ms=0.0):
    """
    Re-bin raw peak times into custom-sized bins around events.
    Returns counts (n_trials, n_channels, n_bins), bin_centers, bin_edges.
    Bins inside the blanking region [-blank_pre_ms, stim_dur_ms + blank_post_ms] are set to NaN.
    """
    ch_keys = sorted(peaks_dict.keys(), key=lambda x: int(x))
    n_ch = len(ch_keys)
    events_ms = np.asarray(events_ms, float).ravel()
    n_trials = events_ms.size

    edges = np.arange(win_ms[0], win_ms[1] + bin_ms * 0.5, bin_ms)
    centers = edges[:-1] + bin_ms / 2.0
    n_bins = centers.size

    counts = np.zeros((n_trials, n_ch, n_bins), float)

    for ch_i, ch_id in enumerate(ch_keys):
        spk = np.asarray(peaks_dict.get(ch_id, []), float).ravel()
        if spk.size == 0:
            continue
        spk = np.sort(spk)
        for t_i, t_stim in enumerate(events_ms):
            rel = spk[(spk >= t_stim + win_ms[0]) & (spk <= t_stim + win_ms[1])] - t_stim
            if rel.size == 0:
                continue
            idx = np.searchsorted(edges, rel, side='right') - 1
            valid = (idx >= 0) & (idx < n_bins)
            np.add.at(counts[t_i, ch_i], idx[valid], 1.0)

    # Blank bins inside stim artifact + pre/post
    # if stim_dur_ms > 0:
    #     # Mask covers [-blank_pre_ms, stim_dur_ms + blank_post_ms]
    #     blank_mask = (centers >= -blank_pre_ms) & (centers <= stim_dur_ms + blank_post_ms)
    #     counts[:, :, blank_mask] = np.nan

    return counts, centers, edges


def plot_channel_group(ax_raster, ax_psth, 
                       peak_times, events_ms, 
                   binned_counts, bin_edges, 
                       title, stim_dur_ms=0,
                       blank_pre_ms=0.0, blank_post_ms=0.0,
                       bin_centers=None,
                       n_trials_ref=0,
                       win_ms=WIN_PLOT_MS,
                       psth_ylim=PSTH_YLIM):
    """
    Plot one channel's raster and PSTH.
    """
    # 1. Raster
    if events_ms is not None and len(events_ms) > 0:
        # Pass total blank duration relative to 0. 
        # If stim_dur_ms is e.g. 100, and we want 20ms post-stim blanking, blank_post_ms passed to this function is purely the *extra* post-stim blanking?
        # No, let's keep it simple: blank_post_total = stim_dur_ms + blank_post_ms
        blank_post_total = stim_dur_ms + blank_post_ms if stim_dur_ms > 0 else 0.0
        blank_pre_total = blank_pre_ms if stim_dur_ms > 0 else 0.0
        
        trial_indices, rel_times = _get_raster_data(peak_times, events_ms, WIN_PLOT_MS,
                                                         blank_pre_ms=blank_pre_total,
                                                         blank_post_ms=blank_post_total)
        
        # Raster plot
        if len(rel_times) > 0:
            ax_raster.scatter(rel_times, trial_indices, 
                              s=RASTER_MARKER_SIZE, marker=RASTER_MARKER, 
                              c=RASTER_COLOR, alpha=RASTER_ALPHA)
                              
        n_trials = len(events_ms)
    else:
        # No events -> Empty Raster
        n_trials = n_trials_ref
    
    # Stimulation shading
    # if stim_dur_ms > 0:
    #     # Shade from -blank_pre_ms to stim_dur_ms + blank_post_ms
    #     rect_start = -blank_pre_ms
    #     rect_end = stim_dur_ms + blank_post_ms
    #     ax_raster.axvspan(rect_start, rect_end, color='gray', alpha=0.3, edgecolor=None)

    # Red line at 0 (Only if data exists)
    has_data = (len(rel_times) > 0) or (binned_counts is not None and np.nansum(binned_counts) > 0)
    
    if has_data:
        ax_raster.axvline(0, color='r', linestyle='--', linewidth=1, alpha=0.8)
    
    # Formatting
    ax_raster.set_xlim(win_ms)
    ax_raster.set_ylim(-0.5, n_trials - 0.5)
    ax_raster.set_title(title, fontsize=10, pad=4)
    ax_raster.axis('off') 
    
    # 2. PSTH (Spike Counts)
    if binned_counts is not None and binned_counts.size > 0:
        psth_counts = np.nansum(binned_counts, axis=0) 
        
        if bin_centers is not None and len(psth_counts) == len(bin_centers):
            width = 20.0 
            if len(bin_centers) > 1:
                dt = np.nanmedian(np.diff(bin_centers))
                if dt > 0: width = dt
            
            ax_psth.bar(bin_centers, psth_counts, width=width, align='center',
                        color=PSTH_COLOR, edgecolor=PSTH_EDGE_COLOR, linewidth=PSTH_LINEWIDTH)
                        
        elif len(psth_counts) == len(bin_edges) - 1:
            ax_psth.bar(bin_edges[:-1], psth_counts, width=np.diff(bin_edges), align='edge',
                        color=PSTH_COLOR, edgecolor=PSTH_EDGE_COLOR, linewidth=PSTH_LINEWIDTH)

    if has_data:
        ax_psth.axvline(0, color='r', linestyle='--', linewidth=1, alpha=0.8)
    
    ax_psth.set_xlim(win_ms)
    ax_psth.set_ylim(psth_ylim) # Fixed Y-limits
    
    # Hide top/right spines
    ax_psth.spines['top'].set_visible(False)
    ax_psth.spines['right'].set_visible(False)
    ax_psth.spines['left'].set_linewidth(0.5)
    ax_psth.spines['bottom'].set_linewidth(0.5)
    ax_psth.tick_params(axis='both', which='major', labelsize=6, width=0.5, length=2)
    
def load_reference_data(all_files, metadata_df=None):
    """
    Scans all files for 'control_reaches' and 'at_rest'.
    Returns:
    {
        'controls': { 'Target_A': {data}, 'Target_B': {data} },
        'rest': { (freq, dur): {data} }
    }
    """
    control_files = [f for f in all_files if "control_reaches" in str(f).lower()]
    rest_files = [f for f in all_files if "at_rest" in str(f).lower()]
    
    references = {'controls': {}, 'rest': {}}
    
    print(f"  [Debug] Found {len(control_files)} control files and {len(rest_files)} at_rest files.")
    
    # helper to get freq/dur from CSV or NPZ
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
        
        # Fallback to NPZ meta
        if f_hz == 0 or d_ms == 0:
            meta = data['meta'].item() if 'meta' in data and data['meta'].ndim == 0 else {}
            if f_hz == 0: f_hz = float(meta.get('recording_stim_freq', 0.0))
            if d_ms == 0: d_ms = float(meta.get('recording_stim_dur', 0.0))
        return f_hz, d_ms

    # 1. Controls (by Target)
    for target in ['Target_A', 'Target_B', 'Target_control']:
        # If target_folder in STIM is "Target_A", we want comparison with "Target_A" control
        target_files = [f for f in control_files if target.lower() in str(f).lower()]
        if not target_files: continue
            
        max_trials = -1
        best_data = None
        for f in tqdm(target_files, desc=f"Scanning {target} Controls", leave=False):
            try:
                data = np.load(f, allow_pickle=True)
                n_trials = data['UA_counts'].shape[0] if 'UA_counts' in data and len(data['UA_counts'].shape) > 0 else 0
                if n_trials > max_trials:
                    max_trials = n_trials
                    entry = {'n_trials': n_trials, 'NPRW': None, 'UA': None, 'NPRW_peaks': None, 'UA_peaks': None, 'ctrl_events': None}
                    if 'event_ms' in data: entry['ctrl_events'] = np.asarray(data['event_ms']).flatten()
                    if 'NPRW_counts' in data:
                        entry['NPRW'] = np.nansum(data['NPRW_counts'], axis=0) / max(n_trials, 1)
                        if 'NPRW_edges_ms' in data: entry['NPRW_edges_ms'] = data['NPRW_edges_ms']
                        if 'NPRW_peak_ms_dedup' in data: entry['NPRW_peaks'] = data['NPRW_peak_ms_dedup'].item()
                    if 'UA_counts' in data:
                        entry['UA'] = np.nansum(data['UA_counts'], axis=0) / max(n_trials, 1)
                        if 'UA_edges_ms' in data: entry['UA_edges_ms'] = data['UA_edges_ms']
                        if 'UA_peak_ms_dedup' in data: entry['UA_peaks'] = data['UA_peak_ms_dedup'].item()
                    best_data = entry
            except: continue
        if best_data: references['controls'][target] = best_data

    # 2. At-Rest (by Freq/Dur)
    if not rest_files:
        print("  [Debug] No 'at_rest' string found in any file paths.")
    
    for f in tqdm(rest_files, desc="Scanning At-Rest Baselines", leave=False):
        try:
            data = np.load(f, allow_pickle=True)
            b_idx = int(data['br_idx']) if 'br_idx' in data else -1
            freq, dur = get_freq_dur(data, b_idx)
            
            if freq == 0: continue 
            
            key = (freq, dur)
            n_trials = data['UA_counts'].shape[0] if 'UA_counts' in data and len(data['UA_counts'].shape) > 0 else 0
            
            if key not in references['rest'] or n_trials > references['rest'][key]['n_trials']:
                entry = {'n_trials': n_trials, 'NPRW': None, 'UA': None, 'NPRW_peaks': None, 'UA_peaks': None, 'ctrl_events': None}
                if 'event_ms' in data: entry['ctrl_events'] = np.asarray(data['event_ms']).flatten()
                if 'NPRW_counts' in data:
                    entry['NPRW'] = np.nansum(data['NPRW_counts'], axis=0) / max(n_trials, 1)
                    if 'NPRW_edges_ms' in data: entry['NPRW_edges_ms'] = data['NPRW_edges_ms']
                    if 'NPRW_peak_ms_dedup' in data: entry['NPRW_peaks'] = data['NPRW_peak_ms_dedup'].item()
                if 'UA_counts' in data:
                    entry['UA'] = np.nansum(data['UA_counts'], axis=0) / max(n_trials, 1)
                    if 'UA_edges_ms' in data: entry['UA_edges_ms'] = data['UA_edges_ms']
                    if 'UA_peak_ms_dedup' in data: entry['UA_peaks'] = data['UA_peak_ms_dedup'].item()
                references['rest'][key] = entry
        except: continue
            
    if references['rest']:
        print(f"  [Debug] Loaded At-Rest baselines for keys: {list(references['rest'].keys())}")
    else:
        print("  [Debug] No valid At-Rest baselines (with freq > 0) were found.")

    return references

# -------------------------

def process_file(npz_path, references=None, metadata_df=None):
    
    
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return


    # --- Load Bad Channels ---
    # Try to find Impedances folder logic (assuming structure .../results/peri_stim_v2/...)
    bad_map = get_session_impedances(SESSION_LOC, utah_exclude_low=False)
    bad_ua = bad_map.get('utah', set())
    bad_nprw = bad_map.get('nprw', set())

    # Extract Metadata
    sess = str(data['sess'])
    br_idx = int(data['br_idx']) if 'br_idx' in data else -1

    if PROCESS_ONLY and br_idx not in PROCESS_ONLY:
        return

    overall_title = str(data['overall_title']) if 'overall_title' in data else ""
    
    # Event times (Stimulation onsets)
    if 'event_ms' in data:
        event_ms = data['event_ms']
    else:
        # Legacy fallback
        # Try to find stim_ms in meta or nprw_meta
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
    
    # If no events, we might still have counts for PSTH
    # Determine n_trials from available counts
    n_trials_ref = 0
    if len(event_ms) > 0:
        n_trials_ref = len(event_ms)
    elif 'NPRW_counts' in data:
        n_trials_ref = data['NPRW_counts'].shape[0]
    elif 'UA_counts' in data:
        n_trials_ref = data['UA_counts'].shape[0]
        
    if n_trials_ref == 0:
        print(f"\nSkipping {npz_path.name}: No events and no trial counts found.")
        return



    # Determine Type from path
    path_str = str(npz_path).lower()
    cond_type = "OTHER"
    if "control_reaches" in path_str:
        cond_type = "CTRL"
    elif "stim_reaches" in path_str:
        cond_type = "STIM"
    elif "at_rest" in path_str:
        cond_type = "REST"
    
    if cond_type == "OTHER":
        return

    target_folder = ""
    if "_target_a" in npz_path.name.lower(): target_folder = "Target_A"
    elif "_target_b" in npz_path.name.lower(): target_folder = "Target_B"
    elif "_target_control" in npz_path.name.lower(): target_folder = "Target_control"

    # Stim freq/dur
    stim_dur_ms = 0.0
    stim_freq_hz = 0.0
    
    # Try metadata CSV first
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

    # Fallback to NPZ meta (Main results or legacy)
    if stim_freq_hz == 0 or stim_dur_ms == 0:
        if 'meta' in data and data['meta'].ndim == 0:
            meta_dict = data['meta'].item()
            if isinstance(meta_dict, dict):
                if stim_dur_ms == 0: stim_dur_ms = float(meta_dict.get('recording_stim_dur', 0.0))
                if stim_freq_hz == 0: stim_freq_hz = float(meta_dict.get('recording_stim_freq', 0.0))
                
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
        stim_dur_ms = 0.0 # Force no blanking for control
            
    if cond_type == "STIM":
        print(f"  [Debug] File: {npz_path.name} | Freq: {stim_freq_hz} | Dur: {stim_dur_ms}")
        if references and references['rest']:
             rest_entry = references['rest'].get((stim_freq_hz, stim_dur_ms))
             if rest_entry:
                 print(f"    [OK] Found At-Rest baseline for Freq={stim_freq_hz}, Dur={stim_dur_ms}")
             else:
                 print(f"    [Warn] No At-Rest baseline match for Freq={stim_freq_hz}, Dur={stim_dur_ms}. Available: {list(references['rest'].keys())}")
    
    # Stim Channels (New)
    stim_chs_str = ""
    stim_list = []
    
    # Try nprw_meta first
    if 'nprw_meta' in data:
        nm = data['nprw_meta'].item()
        if isinstance(nm, dict) and 'stim_channels' in nm:
             stim_list = nm['stim_channels']
             
    # Fallback to general meta
    if len(stim_list) == 0 and 'meta' in data:
         m = data['meta'].item()
         if isinstance(m, dict) and 'stim_channels' in m:
             stim_list = m['stim_channels']
             
    
    if np.size(stim_list) > 0:
        # Format list
        try:
            # Flatten if needed
            sl = np.array(stim_list).flatten()
            
            # Convert to zero-indexed sorted list
            sl = sorted([int(x) - 1 for x in sl])
            
            # Create ranges
            ranges = []
            for _, g in groupby(enumerate(sl), lambda i_x: i_x[0] - i_x[1]):
                group = list(map(itemgetter(1), g))
                ranges.append(f"{group[0]}-{group[-1]}" if len(group) > 1 else f"{group[0]}")
                    
            stim_chs_str = f" | Stim Chs: {', '.join(ranges)}"
            
        except:
            stim_chs_str = f" | Stim Chs: {stim_list}"
    
    # ---------------------------------------------------------
    # 1. NPRW (Read-Write Probe) - 128 Channels
    # ---------------------------------------------------------
    if 'NPRW_peak_ms_dedup' in data and data['NPRW_peak_ms_dedup'].item():
        peaks_dict = data['NPRW_peak_ms_dedup'].item()
        counts_data_orig = data['NPRW_counts'] # (n_trials, n_channels, n_bins)
        edges_ms_orig = data['NPRW_edges_ms']
        centers_ms_orig = data['NPRW_rel_t'] if 'NPRW_rel_t' in data else None
        
        n_ch = 128
        cols = 16
        # Sort keys reliably and filter bad channels
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
        if rows == 0: rows = 1
        
        # Determine output directories (computed once)
        base_folder = "other"
        path_str_lower = str(npz_path).lower()
        if "control_reaches" in path_str_lower:
            base_folder = "control_reaches"
        elif "stim_reaches" in path_str_lower:
            base_folder = "stim_reaches"
        elif "at_rest" in path_str_lower:
            base_folder = "at_rest"
        target_folder = ""
        if "_target_A" in npz_path.name:
             target_folder = "Target_A"
        elif "_target_B" in npz_path.name:
             target_folder = "Target_B"
        
        # ──── Loop over plot views ────
        for view_win_in, view_bin_ms, view_suffix in PLOT_VIEWS:
            # Determine window dynamic
            if view_win_in is None:
                sd = stim_dur_ms if stim_dur_ms > 0 else 100.0
                view_win = (-100.0, sd + 100.0)
            else:
                view_win = view_win_in

            # Choose data source: pre-computed or re-binned
            if view_bin_ms is not None:
                counts_data, centers_ms, edges_ms = _rebin_from_peaks(
                    peaks_dict, event_ms, view_win, view_bin_ms, stim_dur_ms,
                    blank_pre_ms=20.0, blank_post_ms=20.0)
                cur_ylim = PSTH_YLIM_ZOOM
            else:
                counts_data = counts_data_orig
                centers_ms = centers_ms_orig
                edges_ms = edges_ms_orig
                cur_ylim = PSTH_YLIM
            
            fig_rw = plt.figure(figsize=FIG_SIZE_RW)
            gs = gridspec.GridSpec(rows * 2, cols, figure=fig_rw, hspace=0.3, wspace=0.3)
            
            for i, ch in enumerate(sorted_ch_ids):
                if i >= n_ch: break
                r = (i // cols)
                c = (i % cols)
                ax_raster = fig_rw.add_subplot(gs[2*r, c])
                ax_psth = fig_rw.add_subplot(gs[2*r+1, c], sharex=ax_raster)
                
                ch_idx = i 
                if ch_idx < counts_data.shape[1]:
                    binned = counts_data[:, ch_idx, :]
                else:
                    binned = None
                peak_times = peaks_dict.get(ch, [])
                
                plot_channel_group(ax_raster, ax_psth, peak_times, event_ms, binned, edges_ms,
                                   f"Ch {ch}", stim_dur_ms=stim_dur_ms, 
                                   blank_pre_ms=20.0, blank_post_ms=20.0, # NPRW default blanking
                                   bin_centers=centers_ms,
                                   n_trials_ref=n_trials_ref, win_ms=view_win, psth_ylim=cur_ylim)
                if 2*r+1 < (rows*2 - 2):
                    ax_psth.set_xticklabels([])
                else:
                    ax_psth.tick_params(axis='x', which='both', labelbottom=True)
            
            fig_rw.suptitle(f"{overall_title} | NPRW{stim_chs_str}", fontsize=20)
            if len(stim_chs_str) > 0:
                 plt.subplots_adjust(top=0.90)
            
            save_dir = FIG_ROOT / base_folder / target_folder
            save_dir.mkdir(parents=True, exist_ok=True)
            fig_name = f"[processed]_Cond{br_idx:03d}_{cond_type}_NPRW{view_suffix}.png"
            fig_rw.savefig(save_dir / fig_name, dpi=100, bbox_inches='tight')
            plt.close(fig_rw)

            # ---------------------------------------------------------
            # NPRW Difference Plots
            # ---------------------------------------------------------
            if cond_type == "STIM" and references is not None:
                # Find baselines
                ctrl_entry = references['controls'].get(target_folder)
                rest_entry = references['rest'].get((stim_freq_hz, stim_dur_ms))
                
                # Definitions for diffs
                # User said "subtract both at rest and control condition from stim condition" -> Stim - Control - Rest
                diff_configs = []
                stim_norm = np.nansum(counts_data, axis=0) / max(counts_data.shape[0], 1)
                
                # Function to get normalized counts for a reference entry
                def get_norm_ref(entry, ref_type='NPRW'):
                    if entry is None or entry.get(f'{ref_type}_peaks') is None:
                        return None
                    # Re-bin using the SAME window and bin size as Stim
                    pk_dict = entry[f'{ref_type}_peaks']
                    ev_ms = entry['ctrl_events'] if entry.get('ctrl_events') is not None else []
                    if len(ev_ms) == 0: return None
                    
                    c_ref, _, _ = _rebin_from_peaks(pk_dict, ev_ms, view_win, 
                                                    view_bin_ms if view_bin_ms else 20.0, # fallback to standard bin if None
                                                    0.0, 0.0, 0.0) # No blanking for baseline
                    # Normalize by trials
                    return np.nansum(c_ref, axis=0) / max(c_ref.shape[0], 1)

                ctrl_norm = get_norm_ref(ctrl_entry, 'NPRW')
                rest_norm = get_norm_ref(rest_entry, 'NPRW')

                if ctrl_norm is not None:
                    diff_configs.append(('stim_control_diff', 'Stim - Control', ctrl_norm))
                if rest_norm is not None:
                    diff_configs.append(('stim_rest_diff', 'Stim - Rest', rest_norm))
                if ctrl_norm is not None and rest_norm is not None:
                    diff_configs.append(('stim_rest_control_diff', 'Stim - Control - Rest', ctrl_norm + rest_norm))
                
                stim_norm = np.nansum(counts_data, axis=0) / max(counts_data.shape[0], 1)
                
                for dir_name, title_suffix, baseline_norm in diff_configs:
                    if stim_norm.shape == baseline_norm.shape:
                        diff_norm = stim_norm - baseline_norm
                        fig_diff = plt.figure(figsize=FIG_SIZE_RW)
                        gs_diff = gridspec.GridSpec(rows, cols, figure=fig_diff, hspace=0.3, wspace=0.3)
                        
                        found_diff = False
                        for i, ch in enumerate(sorted_ch_ids):
                            if i >= n_ch: break
                            r, c = i // cols, i % cols
                            ax_diff = fig_diff.add_subplot(gs_diff[r, c])
                            d_ch = None
                            if i < diff_norm.shape[0]:
                                found_diff = True
                                d_ch = diff_norm[i, :]
                                bar_colors = np.where(d_ch >= 0, 'tab:green', 'tab:red')
                                if centers_ms is not None and len(d_ch) == len(centers_ms):
                                    bw = view_bin_ms if view_bin_ms else np.nanmedian(np.diff(centers_ms)) if len(centers_ms)>1 else 20.0
                                    ax_diff.bar(centers_ms, d_ch, width=bw, align='center', color=bar_colors, edgecolor='none')
                                elif edges_ms is not None:
                                    ax_diff.bar(edges_ms[:-1], d_ch, width=np.diff(edges_ms), align='edge', color=bar_colors, edgecolor='none')
                            
                            if d_ch is not None and np.nansum(np.abs(d_ch)) > 0:
                                ax_diff.axvline(0, color='r', linestyle='--', linewidth=1, alpha=0.8)
                            ax_diff.set_title(f"Ch {ch}", fontsize=10, pad=2)
                            ax_diff.set_xlim(view_win)
                            ax_diff.set_ylim(DIFF_YLIM)
                            ax_diff.spines['top'].set_visible(False)
                            ax_diff.spines['right'].set_visible(False)
                            ax_diff.axis('off') if i >= diff_norm.shape[0] else None
                            if r < rows-1: ax_diff.set_xticklabels([])

                        if found_diff:
                            fig_diff.suptitle(f"{overall_title} | NPRW Diff ({title_suffix}){stim_chs_str}", fontsize=20)
                            save_diff_dir = FIG_ROOT / dir_name / target_folder
                            save_diff_dir.mkdir(parents=True, exist_ok=True)
                            fig_diff_name = f"[diff]_Cond{br_idx:03d}_{cond_type}_NPRW{view_suffix}.png"
                            fig_diff.savefig(save_diff_dir / fig_diff_name, dpi=100, bbox_inches='tight')
                        plt.close(fig_diff)


    # ---------------------------------------------------------
    # 2. Utah Arrays - Grouped by Region
    # ---------------------------------------------------------
    has_ua = False
    if 'HAS_BR' in data:
         try:
             has_ua = bool(data['HAS_BR'])
         except:
             has_ua = False
    elif 'UA_counts' in data: # Fallback inference
         has_ua = True

    if has_ua and 'UA_peak_ms_dedup' in data and data['UA_peak_ms_dedup'].item():
        peaks_dict = data['UA_peak_ms_dedup'].item()
        counts_data_orig = data['UA_counts']
        edges_ms_orig = data['UA_edges_ms']
        centers_ms_orig = data['UA_rel_t'] if 'UA_rel_t' in data else None

        # Load ua_ids_1based (NSP IDs)
        ua_ids_1based = data.get('ua_ids_1based', None)
        if ua_ids_1based is not None and ua_ids_1based.ndim == 0:
            ua_ids_1based = None
            
        # Mapping helpers
        elec_to_idx = build_elec_to_data_idx(ua_ids_1based, nsp_to_elec_global)
        
        # Identify which regions are actually present in this recording
        plot_regions = []
        REGION_ORDER = ["M1i", "M1s", "PMd", "SMA"]
        if ua_ids_1based is not None:
            active_regs = set()
            for nsp_id in ua_ids_1based:
                try:
                    elec_id = nsp_to_elec_global.get(int(nsp_id), -1)
                    if elec_id > 0:
                        reg = elec_to_region_global.get(elec_id)
                        if reg: active_regs.add(reg)
                except: continue
            plot_regions = [r for r in REGION_ORDER if r in active_regs]
        
        if not plot_regions: plot_regions = ["Unknown"]

        # Determine output directories
        base_folder = "other"
        path_str_lower = str(npz_path).lower()
        if "control_reaches" in path_str_lower: base_folder = "control_reaches"
        elif "stim_reaches" in path_str_lower: base_folder = "stim_reaches"
        elif "at_rest" in path_str_lower: base_folder = "at_rest"
        
        target_folder = ""
        if "_target_A" in npz_path.name: target_folder = "Target_A"
        elif "_target_B" in npz_path.name: target_folder = "Target_B"

        # Determine output directories (computed once)
        base_folder = "other"
        path_str_lower = str(npz_path).lower()
        if "control_reaches" in path_str_lower:
            base_folder = "control_reaches"
        elif "stim_reaches" in path_str_lower:
            base_folder = "stim_reaches"
        elif "at_rest" in path_str_lower:
            base_folder = "at_rest"
        target_folder = ""
        if "_target_A" in npz_path.name:
             target_folder = "Target_A"
        elif "_target_B" in npz_path.name:
             target_folder = "Target_B"
        
        # ──── Loop over plot views ────
        for view_win_in, view_bin_ms, view_suffix in PLOT_VIEWS:
            # Determine window dynamic
            if view_win_in is None:
                sd = stim_dur_ms if stim_dur_ms > 0 else 100.0
                view_win = (-100.0, sd + 100.0)
            else:
                view_win = view_win_in

            # Choose data source: pre-computed or re-binned
            if view_bin_ms is not None:
                counts_data, centers_ms, edges_ms = _rebin_from_peaks(
                    peaks_dict, event_ms, view_win, view_bin_ms, stim_dur_ms,
                    blank_pre_ms=0.0, blank_post_ms=0.0)
                cur_ylim = PSTH_YLIM_ZOOM
            else:
                counts_data = counts_data_orig
                centers_ms = centers_ms_orig
                edges_ms = edges_ms_orig
                cur_ylim = PSTH_YLIM

            for reg_target in plot_regions:
                grid_elec = UTAH_ELEC_GRIDS.get(reg_target)
                if grid_elec is None: continue
                
                fig_ua = plt.figure(figsize=(32, 24))
                gs = gridspec.GridSpec(8 * 2, 8, figure=fig_ua, hspace=0.3, wspace=0.3)
                
                found_channels = False
                for r in range(8):
                    for c in range(8):
                        elec_id = int(grid_elec[r, c])
                        idx = elec_to_idx.get(elec_id)
                        
                        ax_raster = fig_ua.add_subplot(gs[2*r, c])
                        ax_psth = fig_ua.add_subplot(gs[2*r+1, c], sharex=ax_raster)
                        
                        if idx is not None and 0 <= idx < counts_data.shape[1]:
                            found_channels = True
                            binned = counts_data[:, idx, :]
                            ch_key = str(idx)
                            peak_times = peaks_dict.get(ch_key, [])
                            nsp_val = int(ua_ids_1based[idx]) if ua_ids_1based is not None else -1
                            
                            plot_channel_group(ax_raster, ax_psth, peak_times, event_ms, binned, edges_ms,
                                               f"E{elec_id} (NSP{nsp_val})", stim_dur_ms=0.0, 
                                               blank_pre_ms=0.0, blank_post_ms=0.0, 
                                               bin_centers=centers_ms,
                                               n_trials_ref=n_trials_ref, win_ms=view_win, psth_ylim=cur_ylim)
                        else:
                            ax_raster.axis('off')
                            ax_psth.axis('off')
                            if elec_id > 0:
                                ax_raster.text(0.5, 0.5, f"E{elec_id}\nN/A", ha='center', va='center', 
                                               transform=ax_raster.transAxes, fontsize=8, alpha=0.3)
                        
                        if 2*r+1 < 14: ax_psth.set_xticklabels([])

                if found_channels:
                    fig_ua.suptitle(f"{overall_title} | Utah Array - {reg_target}{stim_chs_str}", fontsize=20)
                    plt.subplots_adjust(top=0.90 if stim_chs_str else 0.94)
                    
                    save_dir = FIG_ROOT / base_folder / target_folder
                    save_dir.mkdir(parents=True, exist_ok=True)
                    fig_name = f"[processed]_Cond{br_idx:03d}_{cond_type}_UA_{reg_target}{view_suffix}.png"
                    fig_ua.savefig(save_dir / fig_name, dpi=100, bbox_inches='tight')
            if cond_type == "STIM" and references is not None:
                ctrl_entry = references['controls'].get(target_folder)
                rest_entry = references['rest'].get((stim_freq_hz, stim_dur_ms))

                # Utah-specific norm ref helper (already has elec_to_idx from stim)
                def get_norm_ref_ua(entry):
                    if entry is None or entry.get('UA_peaks') is None:
                        return None
                    pk_dict = entry['UA_peaks']
                    ev_ms = entry['ctrl_events'] if entry.get('ctrl_events') is not None else []
                    if len(ev_ms) == 0: return None
                    
                    c_ref, _, _ = _rebin_from_peaks(pk_dict, ev_ms, view_win, 
                                                    view_bin_ms if view_bin_ms else 20.0, 
                                                    0.0, 0.0, 0.0)
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
                    if stim_norm.shape == baseline_norm.shape:
                        diff_norm = stim_norm - baseline_norm
                        for reg_target in plot_regions:
                            grid_elec = UTAH_ELEC_GRIDS.get(reg_target)
                            if grid_elec is None: continue
                            
                            fig_diff_ua = plt.figure(figsize=(32, 24))
                            gs_diff = gridspec.GridSpec(8, 8, figure=fig_diff_ua, hspace=0.3, wspace=0.3)
                            found_diff = False
                            for r in range(8):
                                for c in range(8):
                                    elec_id = int(grid_elec[r, c])
                                    idx = elec_to_idx.get(elec_id)
                                    ax_diff = fig_diff_ua.add_subplot(gs_diff[r, c])
                                    d_ch = None
                                    if idx is not None and 0 <= idx < diff_norm.shape[0]:
                                        found_diff = True
                                        d_ch = diff_norm[idx, :]
                                        bar_colors = np.where(d_ch >= 0, 'tab:green', 'tab:red')
                                        if centers_ms is not None and len(d_ch) == len(centers_ms):
                                            bw = view_bin_ms if view_bin_ms else np.nanmedian(np.diff(centers_ms)) if len(centers_ms)>1 else 20.0
                                            ax_diff.bar(centers_ms, d_ch, width=bw, align='center', color=bar_colors, edgecolor='none')
                                        elif edges_ms is not None:
                                            ax_diff.bar(edges_ms[:-1], d_ch, width=np.diff(edges_ms), align='edge', color=bar_colors, edgecolor='none')
                                        
                                        if d_ch is not None and np.nansum(np.abs(d_ch)) > 0:
                                            ax_diff.axvline(0, color='r', linestyle='--', linewidth=1, alpha=0.8)
                                        ax_diff.set_title(f"E{elec_id}", fontsize=10, pad=2)
                                    else:
                                        ax_diff.axis('off')
                                        if elec_id > 0:
                                            ax_diff.text(0.5, 0.5, f"E{elec_id}\nN/A", ha='center', va='center', 
                                                           transform=ax_diff.transAxes, fontsize=8, alpha=0.3)
                                    
                                    ax_diff.set_xlim(view_win); ax_diff.set_ylim(DIFF_YLIM)
                                    ax_diff.tick_params(axis='both', which='major', labelsize=6)
                                    # Show timing labels on every subplot per user request
                                    # if r < 7: ax_diff.set_xticklabels([])
                            
                            if found_diff:
                                fig_diff_ua.suptitle(f"{overall_title} | Utah Diff ({title_suffix}) - {reg_target}{stim_chs_str}", fontsize=20)
                                save_diff_dir = FIG_ROOT / dir_name / target_folder
                                save_diff_dir.mkdir(parents=True, exist_ok=True)
                                fig_name = f"[diff]_Cond{br_idx:03d}_{cond_type}_UA_{reg_target}{view_suffix}.png"
                                fig_diff_ua.savefig(save_diff_dir / fig_name, dpi=100, bbox_inches='tight')
                            plt.close(fig_diff_ua)
                        plt.close(fig_diff_ua)
        





def main():
    # Find all peristim .npz files recursively

    files = sorted(PERI_ROOT.rglob("peristim__*.npz"))
    
    if not files:
        print(f"No files found in {PERI_ROOT}")
        return
        
    print(f"Found {len(files)} files.")
    
    # Load Metadata CSV if available
    global metadata_df
    metadata_df = None
    if METADATA_CSV.exists():
        print(f"Loading metadata from {METADATA_CSV}")
        metadata_df = pd.read_csv(METADATA_CSV)

    # Load best controls and rest baselines first (serial)
    references = load_reference_data(files, metadata_df=metadata_df)

    # Run in parallel
    Parallel(n_jobs=8)(delayed(process_file)(f, references, metadata_df) for f in tqdm(files, desc="Files"))


if __name__ == "__main__":
    main()
