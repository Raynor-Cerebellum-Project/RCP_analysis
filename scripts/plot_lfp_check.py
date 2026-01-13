import numpy as np
import matplotlib.pyplot as plt
import RCP_analysis.python.functions.config_loading as cfg
import warnings

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
        
        # --- Metadata (Basic from NPZ) ---
        stim_chs = data.get('stim_channels', [])
        stim_amps = data.get('stim_amplitudes', [])
        stim_str = ""
        # Formatting
        if len(stim_chs) > 0:
             if isinstance(stim_chs, (list, np.ndarray)) and len(stim_chs) > 2:
                  try:
                      stim_chs_arr = np.array(stim_chs)
                      if np.all(np.diff(stim_chs_arr) == 1):
                          stim_curr_str = f"{stim_chs_arr[0]}-{stim_chs_arr[-1]}"
                      else:
                          stim_curr_str = str(stim_chs)
                  except:
                      stim_curr_str = str(stim_chs)
             else:
                 stim_curr_str = str(stim_chs)
             stim_str = f" | Stim Ch: {stim_curr_str}"
        
        if len(stim_amps) > 0:
             uniq_amps = np.unique(stim_amps)
             stim_str += f" | Amp: {uniq_amps} uA"

        # Setup Figure
        bands = ['broadband', 'alpha', 'beta', 'low_gamma', 'high_gamma']
        n_rows = len(bands)
        
        # Combine Pre/Post into single heatmap
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 3 * n_rows), constrained_layout=True,
                               gridspec_kw={'width_ratios': [3, 1, 1]})
        fig.suptitle(f"LFP Check ({label}): {session}\n{stim_str}\n(N={n_trials})", fontsize=14)
        
        # Pick representative channel
        rep_ch_idx = n_ch // 2
        
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
            # Construct combined timeline and data
            # Determine gap
            t_gap_start = t_pre[-1]
            t_gap_end = t_post[0]
            # Create a gap of NaNs. Estimated sampling is based on fs.
            # We can just construct t_combined directly if we know the sample count, 
            # or just append NaNs for the gap duration.
            # Simplified approach: Use `imshow` extent, but that doesn't work for non-contiguous data easily without interpolation.
            # Better approach: Create a full matrix with NaNs in the gap.
            
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
                # t_pre start to t_post end
                # We can't strictly use extent if X isn't linear pixel-wise, but here it should be.
                # Total samples
                n_total = m_comb.shape[1]
                t_start = t_pre[0]
                t_end = t_post[-1]
                
                im = ax.imshow(m_comb, aspect='auto', origin='upper', cmap=cmap, 
                               vmin=-limit, vmax=limit, extent=[t_start, t_end, n_ch-1, 0])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="uV")
                
                # Visual Blanking Region
                ax.axvspan(t_gap_start, t_gap_end, color='gray', alpha=0.3, hatch='///')

            ax.set_ylabel(f"{band.capitalize()}\nCh Index")
            ax.set_title(f"Combined Activity")
            
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
            ax.set_title(f"Pre Traces (Ch {rep_ch_idx})")
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
            ax.set_title(f"Post Traces (Ch {rep_ch_idx})")
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
