import numpy as np
import matplotlib.pyplot as plt
import RCP_analysis.python.functions.config_loading as cfg
import RCP_analysis.python.functions.utils as rcp
import warnings
from pathlib import Path

# Define output directory from config
NPRW_LFP_DIR = cfg.OUT_BASE / "checkpoints" / "NPRW_LFP"
UA_LFP_DIR = cfg.OUT_BASE / "checkpoints" / "UA_LFP"

NPRW_FIG_DIR = cfg.OUT_BASE / "figures" / "NPRW_LFP"
UA_FIG_DIR = cfg.OUT_BASE / "figures" / "UA_LFP"

NPRW_FIG_DIR.mkdir(parents=True, exist_ok=True)
UA_FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Global Plotting Configuration ---
PLOT_CONFIG = {
    # Opacity for individual trial traces
    "trace_alpha": 0.4,
    
    # Colormap for Heatmaps
    "heatmap_cmap": "RdBu_r",
    
    # Heatmap Scales (uV): Set value to enforce fixed scale, or None for dynamic (percentile)
    "scales": {
        'broadband': 30.0,
        'alpha': 10.0,
        'beta': 10.0,
        'low_gamma': 10.0,
        'high_gamma': 10.0
    },
    
    # Trace Y-Axis Scales (uV): Set value for fixed +/- limit, or None for dynamic
    "trace_scales": {
        'broadband': 200, 
        'alpha': 100,
        'beta': 100,
        'low_gamma': 100,
        'high_gamma': 100
    },

    # X-Axis Limits (ms): Set tuple (min, max) or None for default
    "x_limits": {
        "pre": (-500, -5), 
        "post": (101, 605)
    }
}

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
            
        if 'broadband_post' not in data:
            print(f"Error: 'broadband_post' key missing in {npz_file.name}. Skipping.")
            continue
            
        bb_data = data['broadband_post']
        n_trials, n_ch, n_time = bb_data.shape
        
        print(f"  Dimensions: {n_trials} trials, {n_ch} channels")

        # --- Channel Order ---
        sort_idx = np.arange(n_ch)
        
        # --- Metadata (Detailed) ---
        # Try to load stim_stream.npz for this session to get accurate freq/chans
        # Path: results/aux_data/NPRW/{session}_Intan_streams/stim_stream.npz
        # We need to guess the session name from the file if 'session' key is generic
        # Usually data['session'] is the intan session name.
        
        stim_npz = cfg.NPRW_AUX_DATA / f"{session}_Intan_streams" / "stim_stream.npz"
        
        active_str = "Unknown"
        freq_str = "Unknown"
        amp_str = ""
        
        if stim_npz.exists():
            try:
                stim_info = rcp.load_stim_detection(stim_npz)
                
                # Channels
                chs = stim_info.get('active_channels')
                if chs is not None and chs.size > 0:
                    chs = np.sort(chs)
                    if len(chs) > 2 and np.all(np.diff(chs) == 1):
                        active_str = f"{chs[0]}-{chs[-1]}"
                    else:
                        active_str = str(list(chs))
                else:
                    active_str = "None"
                    
                # Frequency
                trigs = stim_info.get('trigger_pairs')
                if trigs is not None and trigs.shape[0] > 1:
                    starts = trigs[:, 0]
                    diffs = np.diff(starts)
                    median_diff = np.median(diffs)
                    if median_diff > 0:
                         # Assume 30k unless we know otherwise (stim stream is usually native fs)
                         stim_fs = 30000.0 
                         freq_val = stim_fs / median_diff
                         
                         if abs(freq_val - 130) < 15: freq_str = "130 Hz"
                         elif abs(freq_val - 400) < 20: freq_str = "400 Hz"
                         else: freq_str = f"{freq_val:.1f} Hz"
                    else:
                        freq_str = "Error"
                else:
                    freq_str = "No Triggers"

            except:
                pass
        else:
            # Fallback to what's in the NPZ (likely from analysis script)
            s_ch = data.get('stim_channels', [])
            if len(s_ch) > 0: active_str = str(s_ch)
        
        # Amplitudes from metadata check (or from NPZ if saved)
        # analyze_lfp_bands saves 'stim_amplitudes'
        saved_amps = data.get('stim_amplitudes', [])
        if len(saved_amps) > 0:
             u_amps = np.unique(saved_amps)
             amp_str = f"| {u_amps} uA"

        # Title Construction
        title_str = f"Stim: {active_str} | {freq_str} {amp_str}"

        # Setup Figure
        bands = ['broadband', 'alpha', 'beta', 'low_gamma', 'high_gamma']
        n_rows = len(bands)
        
        # Combine Pre/Post into single heatmap
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 3 * n_rows), constrained_layout=True,
                                gridspec_kw={'width_ratios': [3, 1, 1]})
        fig.suptitle(f"{session}\n{title_str}\n(N={n_trials})", fontsize=14)
        
        # Pick representative channel (Robust)
        # Use broadband post as the reference for activity
        rep_ch_idx = get_representative_channel(data.get('broadband_post'))
        
        for i, band in enumerate(bands):
            # Keys
            key_pre = f"{band}_pre"
            key_post = f"{band}_post"
            
            # Prepare Data
            d_pre = data.get(key_pre) # (N, Ch, T)
            d_post = data.get(key_post)
            
            # Calc Means
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m_pre = np.nanmean(d_pre, axis=0) if d_pre is not None else None
                m_post = np.nanmean(d_post, axis=0) if d_post is not None else None
            
            # --- Scales ---
            # Use fixed scale if defined in config, else dynamic
            limit = PLOT_CONFIG['scales'].get(band)
            
            if limit is None:
                # Dynamic Shared Scale (Average Heatmap)
                vals_for_scale = []
                if m_pre is not None: vals_for_scale.append(m_pre.flatten())
                if m_post is not None: vals_for_scale.append(m_post.flatten())
                
                if len(vals_for_scale) > 0:
                    all_vals = np.concatenate(vals_for_scale)
                    valid_vals = all_vals[np.isfinite(all_vals)]
                    if valid_vals.size > 0:
                        vmin, vmax = np.percentile(valid_vals, [2, 98])
                        limit = max(abs(vmin), abs(vmax))
                    else:
                        limit = 50.0
                else:
                    limit = 50.0
            
            cmap = PLOT_CONFIG['heatmap_cmap']

            # Time
            t_pre = data.get('rel_time_pre')
            if t_pre is None and d_pre is not None: t_pre = np.arange(d_pre.shape[2])
            
            t_post = data.get('rel_time_post')
            if t_post is None and d_post is not None: t_post = t_ms
            
            # --- HEATMAPS (Combined) ---
            # Determine gap
            t_gap_start = t_pre[-1]
            t_gap_end = t_post[0]
            
            # Estimate fs from time
            if len(t_pre) > 1:
                dt_est = t_pre[1] - t_pre[0]
            else:
                dt_est = 1.0 # default
            
            n_gap = int(np.round((t_gap_end - t_gap_start) / dt_est)) - 1
            if n_gap < 1: n_gap = 1 # Force at least one pixel gap if close
            
            # Concatenate Data
            ax = axes[i, 0]
            
            if m_pre is not None and m_post is not None:
                # Gap data (NaNs)
                gap_data = np.full((n_ch, n_gap), np.nan)
                
                # Combined Matrix
                m_comb = np.hstack([m_pre[sort_idx, :], gap_data, m_post[sort_idx, :]])
                
                # Global Extent
                t_start = t_pre[0]
                t_end = t_post[-1]
                
                # Extent: [left, right, bottom, top]
                # Note: imshow origin='upper' means top is index 0. extent bottom/top should match data coordinates.
                # Usually we want y to be 0..n_ch
                
                im = ax.imshow(m_comb, aspect='auto', origin='upper', cmap=cmap, 
                               vmin=-limit, vmax=limit, extent=[t_start, t_end, n_ch - 0.5, -0.5]) 
                               # Adjusted extent to center pixels on integer indices 0..n_ch-1
                               
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="uV")
                
                # Visual Blanking Region
                ax.axvspan(t_gap_start, t_gap_end, color='gray', alpha=0.3, hatch='///')

            ax.set_ylabel(f"{band.capitalize()}\nCh Index")
            ax.set_title(f"Combined Activity")
            # Invert Y axis to have 0 at top (consistent with origin='upper')
            # But extent might handle it. origin='upper' + extent bottom>top implies flipping.
            # Let's simple rely on origin='upper' and correct tick labels logic if needed. 
            # With origin='upper', index 0 is at the top.
            
            # X-Label
            if i == n_rows - 1: 
                ax.set_xlabel("Time (ms)")
            else:
                ax.set_xticklabels([])
                
            # Set Limits if config exists (min of pre to max of post)
            if PLOT_CONFIG['x_limits']['pre'] and PLOT_CONFIG['x_limits']['post']:
                 ax.set_xlim(PLOT_CONFIG['x_limits']['pre'][0], PLOT_CONFIG['x_limits']['post'][1])

            # --- TRACES (Shared Y-Axis) ---
            rep_pre = d_pre[:, rep_ch_idx, :] if d_pre is not None else None
            rep_post = d_post[:, rep_ch_idx, :] if d_post is not None else None
            
            # Calc Y-Lim
            forced_ylim = PLOT_CONFIG['trace_scales'].get(band)
            
            if forced_ylim is not None:
                y_range = (-forced_ylim, forced_ylim)
            else:
                # Dynamic
                ylims = []
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if rep_pre is not None and np.any(np.isfinite(rep_pre)): 
                        ylims.extend([np.nanmin(rep_pre), np.nanmax(rep_pre)])
                    if rep_post is not None and np.any(np.isfinite(rep_post)):
                        ylims.extend([np.nanmin(rep_post), np.nanmax(rep_post)])
                
                # Filter NaNs/Inf just in case
                ylims = [y for y in ylims if np.isfinite(y)]
                
                if not ylims: 
                    y_range = (-100, 100)
                else:
                     ymin, ymax = min(ylims), max(ylims)
                     if ymin == ymax: pad = 10.0
                     else: pad = (ymax - ymin) * 0.05
                     y_range = (ymin - pad, ymax + pad)
            
            t_alpha = PLOT_CONFIG['trace_alpha']

            # Pre Traces
            ax = axes[i, 1]
            if rep_pre is not None:
                ax.plot(t_pre, rep_pre.T, color='gray', alpha=t_alpha, lw=0.5)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rm = np.nanmean(rep_pre, axis=0)
                ax.plot(t_pre, rm, color='blue', lw=1.5, label='Mean')
            ax.set_ylim(y_range)
            ax.set_title(f"Pre (Ch {rep_ch_idx})")
            ax.set_yticklabels([])
            
            if PLOT_CONFIG['x_limits']['pre'] is not None: ax.set_xlim(PLOT_CONFIG['x_limits']['pre'])
            elif t_pre is not None: ax.set_xlim(t_pre[0], t_pre[-1])
            
            if i == 0: ax.legend(loc='upper right', fontsize='x-small')
            
            # Post Traces
            ax = axes[i, 2]
            if rep_post is not None:
                ax.plot(t_post, rep_post.T, color='gray', alpha=t_alpha, lw=0.5)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rm = np.nanmean(rep_post, axis=0)
                ax.plot(t_post, rm, color='red', lw=1.5, label='Mean')
            ax.set_ylim(y_range)
            ax.set_title(f"Post (Ch {rep_ch_idx})")
            ax.set_yticklabels([])
            
            if PLOT_CONFIG['x_limits']['post'] is not None: ax.set_xlim(PLOT_CONFIG['x_limits']['post'])
            elif t_post is not None: ax.set_xlim(t_post[0], t_post[-1])

        out_path = fig_dir / f"check_lfp_heatmap_{session}.png"
        try:
             plt.savefig(out_path, dpi=150)
             print(f"  Saved plot to {out_path}")
        except Exception as e:
             print(f"  Error saving {out_path}: {e}")
        plt.close(fig)

def plot_lfp_check():
    # 1. Process NPRW
    process_directory(NPRW_LFP_DIR, NPRW_FIG_DIR, label="NPRW_Intan")
    
    # 2. Process Utah Array
    process_directory(UA_LFP_DIR, UA_FIG_DIR, label="Utah_Array")

if __name__ == "__main__":
    plot_lfp_check()
