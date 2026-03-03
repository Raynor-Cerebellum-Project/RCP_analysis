from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from joblib import Parallel, delayed
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *
from RCP_analysis.python.functions.impedance_utils import get_session_impedances
from itertools import groupby
from operator import itemgetter


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

    # Red line at 0
    ax_raster.axvline(0, color='r', linestyle='--', linewidth=1, alpha=0.8)
    
    # Formatting
    ax_raster.set_xlim(win_ms)
    ax_raster.set_ylim(-0.5, n_trials - 0.5)
    ax_raster.set_title(title, fontsize=10, pad=4)
    ax_raster.axis('off') # Hide axis for cleaner look, or maybe just remove ticks
    
    # 2. PSTH (Spike Counts)
    # binned_counts provided is (n_trials, n_bins). Sum across trials.
    if binned_counts is not None and binned_counts.size > 0:
        psth_counts = np.nansum(binned_counts, axis=0) # Sum across trials
        
        # Apply blanking mask if stim_dur_ms > 0
        # if stim_dur_ms > 0:
        #     if bin_centers is not None:
        #         mask = (bin_centers >= -blank_pre_ms) & (bin_centers <= stim_dur_ms + blank_post_ms)
        #         if len(mask) == len(psth_counts):
        #              psth_counts[mask] = np.nan
        #     elif len(psth_counts) == len(bin_edges) - 1:
        #         # Use edges to find centers
        #         ctrs = (bin_edges[:-1] + bin_edges[1:]) / 2
        #         mask = (ctrs >= -blank_pre_ms) & (ctrs <= stim_dur_ms + blank_post_ms)
        #         psth_counts[mask] = np.nan
        
        # Plot using centers if available (Preferred)
        if bin_centers is not None and len(psth_counts) == len(bin_centers):
            # Infer width
            width = 20.0 # Default
            if len(bin_centers) > 1:
                # dt usually constant
                dt = np.nanmedian(np.diff(bin_centers))
                if dt > 0: width = dt
            
            ax_psth.bar(bin_centers, psth_counts, width=width, align='center',
                        color=PSTH_COLOR, edgecolor=PSTH_EDGE_COLOR, linewidth=PSTH_LINEWIDTH)
                        
        # Fallback to edges if centers not provided (Legacy/Backup)
        elif len(psth_counts) == len(bin_edges) - 1:
            ax_psth.bar(bin_edges[:-1], psth_counts, width=np.diff(bin_edges), align='edge',
                        color=PSTH_COLOR, edgecolor=PSTH_EDGE_COLOR, linewidth=PSTH_LINEWIDTH)
        elif len(psth_counts) == len(bin_edges) - 2:
             pass

    # Stimulation shading
    # if stim_dur_ms > 0:
    #     rect_start = -blank_pre_ms
    #     rect_end = stim_dur_ms + blank_post_ms
    #     ax_psth.axvspan(rect_start, rect_end, color='gray', alpha=0.3, edgecolor=None)

    # Red line at 0
    ax_psth.axvline(0, color='r', linestyle='--', linewidth=1, alpha=0.8)
    
    ax_psth.set_xlim(win_ms)
    ax_psth.set_ylim(psth_ylim) # Fixed Y-limits
    
    # Hide top/right spines
    ax_psth.spines['top'].set_visible(False)
    ax_psth.spines['right'].set_visible(False)
    ax_psth.spines['left'].set_linewidth(0.5)
    ax_psth.spines['bottom'].set_linewidth(0.5)
    ax_psth.tick_params(axis='both', which='major', labelsize=6, width=0.5, length=2)
    
def load_best_controls(all_files):
    """
    Scans all files for 'control_reaches'.
    For each target (A or B), finds the file with the maximum number of trials.
    Returns a dictionary:
    {
        'Target_A': {
            'NPRW': <normalized_counts_array (n_ch, n_bins)>,
            'UA': <normalized_counts_array (n_ch, n_bins)>,
            'n_trials': <int>
        },
        'Target_B': ...
    }
    """
    # Filter for control files
    control_files = [f for f in all_files if "control_reaches" in str(f).lower()]
    
    best_controls = {}
    
    # Group by target
    for target in ['Target_A', 'Target_B']:
        target_files = [f for f in control_files if f"_{target.lower()}" in f.name.lower() or f"_{target}" in f.name]
        
        if not target_files:
            continue
            
        max_trials = -1
        best_data = None
        best_file = None
        
        for f in tqdm(target_files, desc=f"Scanning {target} Controls"):
            try:
                data = np.load(f, allow_pickle=True)
                
                # Determine n_trials
                n_trials = 0
                if 'event_ms' in data:
                     n_trials = len(np.asarray(data['event_ms']).flatten())
                elif 'NPRW_counts' in data:
                     n_trials = data['NPRW_counts'].shape[0]
                elif 'UA_counts' in data:
                     n_trials = data['UA_counts'].shape[0]
                     
                if n_trials > max_trials:
                    max_trials = n_trials
                    best_file = f
                    
                    # Store normalized data
                    entry = {'n_trials': n_trials, 'NPRW': None, 'UA': None,
                             'NPRW_peaks': None, 'UA_peaks': None, 'ctrl_events': None}
                    
                    # Store event times for re-binning in zoom views
                    if 'event_ms' in data:
                        entry['ctrl_events'] = np.asarray(data['event_ms']).flatten()
                    
                    if 'NPRW_counts' in data:
                        # Sum over trials -> (n_ch, n_bins)
                        counts = np.nansum(data['NPRW_counts'], axis=0)
                        # Normalize
                        if n_trials > 0:
                            entry['NPRW'] = counts / n_trials
                            if 'NPRW_edges_ms' in data:
                                entry['NPRW_edges_ms'] = data['NPRW_edges_ms']
                        # Store raw peaks for re-binning
                        if 'NPRW_peak_ms_dedup' in data:
                            try:
                                entry['NPRW_peaks'] = data['NPRW_peak_ms_dedup'].item()
                            except:
                                pass
                            
                    if 'UA_counts' in data:
                        counts = np.nansum(data['UA_counts'], axis=0) # (n_ch, n_bins)
                        n_trials = data['UA_counts'].shape[0] if len(data['UA_counts'].shape) > 0 else 0
                        if n_trials > 0:
                            entry['UA'] = counts / n_trials
                            if 'UA_edges_ms' in data:
                                entry['UA_edges_ms'] = data['UA_edges_ms']
                        # Store raw peaks for re-binning
                        if 'UA_peak_ms_dedup' in data:
                            try:
                                entry['UA_peaks'] = data['UA_peak_ms_dedup'].item()
                            except:
                                pass
                    
                    best_data = entry
                    
            except Exception as e:
                print(f"Error reading {f}: {e}")
                continue
                
        if best_data and best_file:
            print(f"  Best control for {target}: {best_file.name} ({max_trials} trials)")
            best_controls[target] = best_data
            
    return best_controls

# -------------------------

def process_file(npz_path, best_controls=None):
    
    
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



    # Stim duration
    stim_dur_ms = 0.0
    if 'meta' in data and data['meta'].ndim == 0:
        meta_dict = data['meta'].item()
        if isinstance(meta_dict, dict):
            stim_dur_ms = float(meta_dict.get('recording_stim_dur', 0.0))
            
    
    # Determine Type from path
    path_str = str(npz_path).lower()
    if "control_reaches" in path_str:
        cond_type = "CTRL"
    elif "stim_reaches" in path_str:
        cond_type = "STIM"
    elif "at_rest" in path_str:
        cond_type = "REST"
    else:
        # Skip files that don't match expected patterns
        return
        
    # Legacy stim_dur check
    if stim_dur_ms == 0.0 and 'nprw_meta' in data:
         nprw_meta = data['nprw_meta'].item()
         if isinstance(nprw_meta, dict):
             # stim_dur might be a scalar or array
             sd = nprw_meta.get('stim_dur', 0.0)
             if np.ndim(sd) > 0:
                 stim_dur_ms = float(np.median(sd))
             else:
                 stim_dur_ms = float(sd)
    
    # If control reaches, force stim duration to 0 (no blanking/shading)
    if cond_type == "CTRL":
        stim_dur_ms = 0.0
    
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
            # Stim - Control Difference Plot (NPRW)
            # ---------------------------------------------------------
            if cond_type == "STIM" and best_controls is not None and target_folder in best_controls:
                ctrl_entry = best_controls[target_folder]
                if ctrl_entry['NPRW'] is not None:
                    ctrl_norm = ctrl_entry['NPRW'] # (n_ch, n_bins)
                    
                    stim_sum = np.nansum(counts_data, axis=0)
                    n_stim_trials = counts_data.shape[0]
                    
                    if n_stim_trials > 0:
                        stim_norm = stim_sum / n_stim_trials
                        diff_norm = None
                        
                        if view_bin_ms is not None:
                            # For re-binned views, re-bin control data with same params
                            ctrl_peaks = best_controls[target_folder].get('NPRW_peaks')
                            if ctrl_peaks is not None:
                                ctrl_counts_rb, ctrl_centers_rb, _ = _rebin_from_peaks(
                                    ctrl_peaks, best_controls[target_folder]['ctrl_events'],
                                    view_win, view_bin_ms, stim_dur_ms=0.0)
                                ctrl_norm_rb = np.nansum(ctrl_counts_rb, axis=0) / max(ctrl_counts_rb.shape[0], 1)
                                if stim_norm.shape == ctrl_norm_rb.shape:
                                    diff_norm = stim_norm - ctrl_norm_rb
                        else:
                            # Standard view: use pre-computed bins with alignment
                            if 'NPRW_edges_ms' in data and 'NPRW_edges_ms' in ctrl_entry:
                                stim_edges = data['NPRW_edges_ms']
                                ctrl_edges = ctrl_entry['NPRW_edges_ms']
                                stim_ctr = (stim_edges[:-1] + stim_edges[1:]) / 2
                                ctrl_ctr = (ctrl_edges[:-1] + ctrl_edges[1:]) / 2
                                if stim_norm.shape[1] < len(stim_ctr):
                                    stim_ctr = stim_ctr[:stim_norm.shape[1]]
                                keep_idx = []
                                for sc in stim_ctr:
                                    idx = np.argmin(np.abs(ctrl_ctr - sc))
                                    keep_idx.append(idx)
                                ctrl_subset = ctrl_norm[:, keep_idx]
                                if stim_norm.shape == ctrl_subset.shape:
                                    diff_norm = stim_norm - ctrl_subset
                            else:
                                if stim_norm.shape == ctrl_norm.shape:
                                    diff_norm = stim_norm - ctrl_norm
                        
                        if diff_norm is not None:
                            # Apply blanking mask to diff_norm
                            # if stim_dur_ms > 0:
                            #     t0, t1 = -20.0, stim_dur_ms + 20.0
                            #     if centers_ms is not None:
                            #         mask = (centers_ms >= t0) & (centers_ms <= t1)
                            #         if diff_norm.shape[1] == len(mask):
                            #             diff_norm[:, mask] = np.nan
                            #     elif edges_ms is not None:
                            #         ctrs = (edges_ms[:-1] + edges_ms[1:]) / 2
                            #         mask = (ctrs >= t0) & (ctrs <= t1)
                            #         if diff_norm.shape[1] == len(mask):
                            #             diff_norm[:, mask] = np.nan

                            fig_diff = plt.figure(figsize=FIG_SIZE_RW)
                            gs_diff = gridspec.GridSpec(rows, cols, figure=fig_diff, hspace=0.3, wspace=0.3)
                            
                            for i, ch in enumerate(sorted_ch_ids):
                                if i >= n_ch: break
                                r = (i // cols)
                                c = (i % cols)
                                ax_diff = fig_diff.add_subplot(gs_diff[r, c])
                                
                                ch_idx = i
                                if ch_idx < diff_norm.shape[0]:
                                    d_ch = diff_norm[ch_idx, :]
                                    bar_colors = np.where(d_ch >= 0, 'tab:green', 'tab:red')
                                    
                                    if centers_ms is not None and len(d_ch) == len(centers_ms):
                                        width = view_bin_ms if view_bin_ms else 20.0
                                        if view_bin_ms is None and len(centers_ms) > 1:
                                            width = np.nanmedian(np.diff(centers_ms))
                                        ax_diff.bar(centers_ms, d_ch, width=width, align='center',
                                                    color=bar_colors, edgecolor='none')
                                    elif edges_ms is not None:
                                        ax_diff.bar(edges_ms[:-1], d_ch, width=np.diff(edges_ms), align='edge',
                                                    color=bar_colors, edgecolor='none')
                                
                                # if stim_dur_ms > 0:
                                #     # NPRW blanking: 20ms pre, 20ms post
                                #     rect_start = -20.0
                                #     rect_end = stim_dur_ms + 20.0
                                #     ax_diff.axvspan(rect_start, rect_end, color='gray', alpha=0.3, edgecolor=None)
                                ax_diff.axvline(0, color='r', linestyle='--', linewidth=1, alpha=0.8)
                                ax_diff.set_title(f"Ch {ch}", fontsize=10, pad=4)
                                ax_diff.set_xlim(view_win)
                                ax_diff.set_ylim(DIFF_YLIM)
                                ax_diff.spines['top'].set_visible(False)
                                ax_diff.spines['right'].set_visible(False)
                                ax_diff.spines['left'].set_linewidth(0.5)
                                ax_diff.spines['bottom'].set_linewidth(0.5)
                                ax_diff.tick_params(axis='both', which='major', labelsize=6, width=0.5, length=2)
                                if r < rows-1:
                                    ax_diff.set_xticklabels([])
                            
                            fig_diff.suptitle(f"{overall_title} | NPRW Diff (Stim - Ctrl){stim_chs_str}", fontsize=20)
                            
                            save_diff_dir = FIG_ROOT / "stim_diff" / target_folder
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

        # Metadata for plotting
        ua_region_names = data['ua_region_names'] # array of strings
        if ua_region_names.ndim == 0: ua_region_names = np.array([])
        
        # Group channels by region
        regions = {} # Region -> list of (ch_id, linear_index)
        
        try:
            sorted_keys = sorted(peaks_dict.keys(), key=lambda x: int(x))
        except:
            sorted_keys = sorted(peaks_dict.keys())

        # Load ua_ids_1based
        ua_ids_1based = data.get('ua_ids_1based', None)
        if ua_ids_1based is not None and ua_ids_1based.ndim == 0:
            ua_ids_1based = None # Handle scalar None

        for i, ch in enumerate(sorted_keys):
            try:
                ch_idx = int(ch)
            except:
                ch_idx = -1
                
            # Determines actual ID
            real_id = ch_idx # Fallback
            if ua_ids_1based is not None and 0 <= ch_idx < len(ua_ids_1based):
                try:
                    real_id = int(ua_ids_1based[ch_idx])
                except:
                    pass
            
            # CHECK BAD CHANNELS (Real ID)
            if real_id in bad_ua:
                continue

            # Map based on Real ID
            if 1 <= real_id <= 64:
                reg = "SMA"
            elif 65 <= real_id <= 128:
                reg = "PMd"
            elif 129 <= real_id <= 192:
                reg = "M1i"
            elif 193 <= real_id <= 256:
                reg = "M1s"
            else:
                reg = "Unknown"
            
            if reg not in regions: regions[reg] = []
            regions[reg].append((ch, i, real_id))
            
        # Custom Region Order
        REGION_ORDER = ["M1i", "M1s", "PMd", "SMA", "Unknown"]

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

            for reg_target in REGION_ORDER:
                if reg_target not in regions or not regions[reg_target]:
                    continue
                
                ch_list = regions[reg_target]
                try:
                    ch_list.sort(key=lambda x: int(x[0]))
                except:
                    ch_list.sort(key=lambda x: x[0])
                region_chs = [(reg_target, item[0], item[1], item[2]) for item in ch_list]
                
                n_total = len(region_chs)
                cols = 8
                rows = (n_total + cols - 1) // cols
                if rows == 0: rows = 1

                fig_h = max(12, rows * 1.5)
                fig_ua = plt.figure(figsize=(48, fig_h))
                gs = gridspec.GridSpec(rows * 2, cols, figure=fig_ua, hspace=0.3, wspace=0.3)
                
                for i, (reg, ch_id, linear_idx, real_id) in enumerate(region_chs):
                    r = (i // cols)
                    c = (i % cols)
                    
                    ax_raster = fig_ua.add_subplot(gs[2*r, c])
                    ax_psth = fig_ua.add_subplot(gs[2*r+1, c], sharex=ax_raster)
                    
                    if linear_idx < counts_data.shape[1]:
                        binned = counts_data[:, linear_idx, :]
                    else:
                        binned = None
                    
                    peak_times = peaks_dict.get(ch_id, [])
                    
                    plot_channel_group(ax_raster, ax_psth, peak_times, event_ms, binned, edges_ms,
                                       f"{reg} - Ch {real_id}", stim_dur_ms=0.0, 
                                       blank_pre_ms=0.0, blank_post_ms=0.0, # UA default blanking
                                       bin_centers=centers_ms,
                                       n_trials_ref=n_trials_ref, win_ms=view_win, psth_ylim=cur_ylim)
                    if 2*r+1 < (rows*2 - 2):
                        ax_psth.set_xticklabels([])

                fig_ua.suptitle(f"{overall_title} | Utah Array - {reg_target}{stim_chs_str}", fontsize=20)
                if len(stim_chs_str) > 0:
                     plt.subplots_adjust(top=0.90)
                
                save_dir = FIG_ROOT / base_folder / target_folder
                save_dir.mkdir(parents=True, exist_ok=True)
                fig_name = f"[processed]_Cond{br_idx:03d}_{cond_type}_UA_{reg_target}{view_suffix}.png"
                fig_ua.savefig(save_dir / fig_name, dpi=100, bbox_inches='tight')
                plt.close(fig_ua)
            
            # ---------------------------------------------------------
            # Stim - Control Difference Plot (Utah)
            # ---------------------------------------------------------
            if cond_type == "STIM" and best_controls is not None and target_folder in best_controls:
                ctrl_entry = best_controls[target_folder]
                if ctrl_entry['UA'] is not None:
                    ctrl_norm = ctrl_entry['UA'] # (n_ch, n_bins)
                    
                    stim_sum = np.nansum(counts_data, axis=0)
                    n_stim_trials = counts_data.shape[0]
                    
                    if n_stim_trials > 0:
                        stim_norm = stim_sum / n_stim_trials
                        diff_norm = None
                        
                        if view_bin_ms is not None:
                            # For re-binned views, re-bin control data with same params
                            ctrl_peaks = best_controls[target_folder].get('UA_peaks')
                            if ctrl_peaks is not None:
                                ctrl_counts_rb, ctrl_centers_rb, _ = _rebin_from_peaks(
                                    ctrl_peaks, best_controls[target_folder]['ctrl_events'],
                                    view_win, view_bin_ms, stim_dur_ms=0.0)
                                ctrl_norm_rb = np.nansum(ctrl_counts_rb, axis=0) / max(ctrl_counts_rb.shape[0], 1)
                                if stim_norm.shape == ctrl_norm_rb.shape:
                                    diff_norm = stim_norm - ctrl_norm_rb
                        else:
                            # Standard view: use pre-computed bins with alignment
                            if 'UA_edges_ms' in data and 'UA_edges_ms' in ctrl_entry:
                                stim_edges = data['UA_edges_ms']
                                ctrl_edges = ctrl_entry['UA_edges_ms']
                                stim_ctr = (stim_edges[:-1] + stim_edges[1:]) / 2
                                ctrl_ctr = (ctrl_edges[:-1] + ctrl_edges[1:]) / 2
                                if stim_norm.shape[1] < len(stim_ctr):
                                    stim_ctr = stim_ctr[:stim_norm.shape[1]]
                                keep_idx = []
                                for sc in stim_ctr:
                                    idx = np.argmin(np.abs(ctrl_ctr - sc))
                                    keep_idx.append(idx)
                                ctrl_subset = ctrl_norm[:, keep_idx]
                                if stim_norm.shape == ctrl_subset.shape:
                                    diff_norm = stim_norm - ctrl_subset
                            else:
                                if stim_norm.shape == ctrl_norm.shape:
                                    diff_norm = stim_norm - ctrl_norm
                        
                        if diff_norm is not None:
                            # Apply blanking mask to diff_norm
                            # if stim_dur_ms > 0:
                            #     t0, t1 = -5.0, stim_dur_ms + 5.0
                            #     if centers_ms is not None:
                            #         mask = (centers_ms >= t0) & (centers_ms <= t1)
                            #         if diff_norm.shape[1] == len(mask):
                            #             diff_norm[:, mask] = np.nan
                            #     elif edges_ms is not None:
                            #         ctrs = (edges_ms[:-1] + edges_ms[1:]) / 2
                            #         mask = (ctrs >= t0) & (ctrs <= t1)
                            #         if diff_norm.shape[1] == len(mask):
                            #             diff_norm[:, mask] = np.nan

                            for reg_target in REGION_ORDER:
                                if reg_target not in regions or not regions[reg_target]:
                                    continue
                                
                                ch_list = regions[reg_target]
                                try:
                                    ch_list.sort(key=lambda x: int(x[0]))
                                except:
                                    ch_list.sort(key=lambda x: x[0])
                                region_chs = [(reg_target, item[0], item[1], item[2]) for item in ch_list]
                                
                                n_total = len(region_chs)
                                cols = 8
                                rows = (n_total + cols - 1) // cols
                                if rows == 0: rows = 1

                                fig_h = max(12, rows * 1.5)
                                fig_diff_ua = plt.figure(figsize=(48, fig_h))
                                gs_diff = gridspec.GridSpec(rows, cols, figure=fig_diff_ua, hspace=0.3, wspace=0.3)
                                
                                for i, (reg, ch_id, linear_idx, real_id) in enumerate(region_chs):
                                    r = (i // cols)
                                    c = (i % cols)
                                    ax_diff = fig_diff_ua.add_subplot(gs_diff[r, c])
                                    
                                    if linear_idx < diff_norm.shape[0]:
                                        d_ch = diff_norm[linear_idx, :]
                                        bar_colors = np.where(d_ch >= 0, 'tab:green', 'tab:red')
                                        
                                        if centers_ms is not None:
                                            width = view_bin_ms if view_bin_ms else 20.0
                                            if view_bin_ms is None and len(centers_ms) > 1:
                                                width = np.nanmedian(np.diff(centers_ms))
                                            ax_diff.bar(centers_ms, d_ch, width=width, align='center',
                                                        color=bar_colors, edgecolor='none')
                                        elif edges_ms is not None:
                                            ax_diff.bar(edges_ms[:-1], d_ch, width=np.diff(edges_ms), align='edge',
                                                        color=bar_colors, edgecolor='none')
                                    
                                    # if stim_dur_ms > 0:
                                    #     # UA default blanking: 5ms pre, 5ms post
                                    #     rect_start = -5.0
                                    #     rect_end = stim_dur_ms + 5.0
                                    #     ax_diff.axvspan(rect_start, rect_end, color='gray', alpha=0.3, edgecolor=None)
                                    ax_diff.axvline(0, color='r', linestyle='--', linewidth=1, alpha=0.8)
                                    ax_diff.set_title(f"{reg} - Ch {real_id}", fontsize=10, pad=4)
                                    ax_diff.set_xlim(view_win)
                                    ax_diff.set_ylim(DIFF_YLIM)
                                    ax_diff.spines['top'].set_visible(False)
                                    ax_diff.spines['right'].set_visible(False)
                                    ax_diff.spines['left'].set_linewidth(0.5)
                                    ax_diff.spines['bottom'].set_linewidth(0.5)
                                    ax_diff.tick_params(axis='both', which='major', labelsize=6, width=0.5, length=2)
                                    if r < rows-1:
                                        ax_diff.set_xticklabels([])
                                
                                fig_diff_ua.suptitle(f"{overall_title} | Utah Diff (Stim - Ctrl) - {reg_target}{stim_chs_str}", fontsize=20)
                                
                                save_diff_dir = FIG_ROOT / "stim_diff" / target_folder
                                save_diff_dir.mkdir(parents=True, exist_ok=True)
                                fig_name = f"[diff]_Cond{br_idx:03d}_{cond_type}_UA_{reg_target}{view_suffix}.png"
                                fig_diff_ua.savefig(save_diff_dir / fig_name, dpi=100, bbox_inches='tight')
                                plt.close(fig_diff_ua)
        





def main():
    # Find all peristim .npz files recursively

    files = sorted(PERI_ROOT.rglob("peristim__*.npz"))
    
    if not files:
        print(f"No files found in {PERI_ROOT}")
        return
        
    print(f"Found {len(files)} files.")
    
    # Load best controls first (serial)
    best_controls = load_best_controls(files)

    # Run in parallel
    # n_jobs=-1 uses all available cores. adjust as needed.
    Parallel(n_jobs=8)(delayed(process_file)(f, best_controls) for f in tqdm(files, desc="Files"))


if __name__ == "__main__":
    main()
