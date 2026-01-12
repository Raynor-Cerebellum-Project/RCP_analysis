import numpy as np
import matplotlib.pyplot as plt
import RCP_analysis.python.functions.config_loading as cfg

# Define output directory from config
# Now reading from new aligned path
LFP_DIR = cfg.OUT_BASE / "checkpoints" / "NPRW_LFP"
FIG_DIR = cfg.OUT_BASE / "figures" / "NPRW_LFP"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def plot_lfp_check():
    if not LFP_DIR.exists():
        print(f"Directory not found: {LFP_DIR}")
        return

    # Find latest aligned file
    # Look for files matching the new naming pattern
    files = sorted(LFP_DIR.glob("aligned_lfp__*.npz"))
    if not files:
        print(f"No aligned LFP npz files found in {LFP_DIR}")
        return
    
   
    # Iterate through all files
    for npz_file in files:
        print(f"Loading {npz_file.name}...")
        
        try:
            data = np.load(npz_file, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            continue

        # Extract
        if 'fs_lfp' in data:
            fs = int(data['fs_lfp'])
        else:
            fs = 1000 # Default
            
        if 'rel_time_post' in data:
            t_ms = data['rel_time_post']
        elif 'rel_time_ms' in data:
            t_ms = data['rel_time_ms']
        else:
            # Reconstruct if missing
            n_p = data['broadband_post'].shape[2]
            t_ms = np.arange(n_p) * 1000.0 / fs # approximate
        
        if 'session' in data:
            session = str(data['session'])
        else:
            session = npz_file.stem
            
        # Check dimensions
        # 'broadband': (N_trials, N_ch, N_time)
        if 'broadband_post' not in data:
            print(f"Error: 'broadband_post' key missing in {npz_file.name}. Skipping.")
            continue
            
        bb_data = data['broadband_post']
        n_trials, n_ch, n_time = bb_data.shape
        
        print(f"  Dimensions: {n_trials} trials, {n_ch} channels, {n_time} timepoints")

   
        # --- Channel Order ---
        # User requested NO sorting (use index order)
        sort_idx = np.arange(n_ch)
        ylabel_str = "Channel Index"

        # Setup Figure
        # 5 rows x 2 columns
        # Col 1: Heatmap
        # Col 2: Sample Traces
        fig = plt.figure(figsize=(16, 20), constrained_layout=True)
        gs = fig.add_gridspec(5, 2, width_ratios=[3, 1])
        fig.suptitle(f"LFP Check: {session}\n(N={n_trials})", fontsize=16)
        
        bands_map = {
            "Broadband": bb_data,
            "Alpha": data['alpha'] if 'alpha' in data else None,
            "Beta": data['beta'] if 'beta' in data else None,
            "Low Gamma": data['low_gamma'] if 'low_gamma' in data else None,
            "High Gamma": data['high_gamma'] if 'high_gamma' in data else None
        }
        
        # Pick sample channels for line plots (e.g. 4 evenly spaced)
        sample_chs = np.linspace(0, n_ch-1, 5, dtype=int)
        colors_trace = plt.cm.viridis(np.linspace(0, 1, len(sample_chs)))
        fig, axes = plt.subplots(5, 2, figsize=(16, 20), constrained_layout=True, gridspec_kw={'width_ratios': [3, 1]})
        fig.suptitle(f"LFP Check: {session}\n(Post-Stim)", fontsize=16)
        
        bands = ['broadband', 'alpha', 'beta', 'low_gamma', 'high_gamma']
        
        for i, band in enumerate(bands):
            key = f"{band}_post"
            if key not in data:
                continue
                
            traces = data[key] # (n_trials, n_ch, n_time)
            
            # Mean across trials
            mean_traces = np.mean(traces, axis=0) # (n_ch, n_time)
            
            # Sort
            sorted_traces = mean_traces[sort_idx, :]
            
            # 1. Heatmap
            ax_map = axes[i, 0]
            
            # Robust scaling (2nd/98th percentile)
            vmin, vmax = np.percentile(sorted_traces, [2, 98])
            limit = max(abs(vmin), abs(vmax))
            
            im = ax_map.imshow(
                sorted_traces, 
                aspect='auto', 
                origin='upper',
                cmap='RdBu_r', 
                vmin=-limit, 
                vmax=limit,
                extent=[t_ms[0], t_ms[-1], 0, sorted_traces.shape[0]]
            )
            ax_map.set_ylabel(f"{band.capitalize()}\nChannel (Sorted)")
            if i == len(bands) - 1:
                ax_map.set_xlabel("Time (ms)")
            else:
                ax_map.set_xticklabels([])
                
            fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
            
            # 2. Mean Trace (Overlay) - Select a few representative channels
            ax_in = axes[i, 1]
            
            # Pick 4 evely spaced channels from sorted list
            n_ch = sorted_traces.shape[0]
            sel_ind = np.linspace(0, n_ch-1, 4, dtype=int)
            
            colors = plt.cm.viridis(np.linspace(0, 1, 4))
            
            for ch_i, color in zip(sel_ind, colors):
                ax_in.plot(t_ms, sorted_traces[ch_i, :], color=color, linewidth=1.0, label=f"Ch {ch_i}")
                
            ax_in.set_xlim(t_ms[0], t_ms[-1])
            if i == 0:
                ax_in.legend(loc='upper right', frameon=False, fontsize='x-small')
            
        # Save
        out_path = FIG_DIR / f"check_lfp_heatmap_{session}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved plot to {out_path}")

if __name__ == "__main__":
    plot_lfp_check()
