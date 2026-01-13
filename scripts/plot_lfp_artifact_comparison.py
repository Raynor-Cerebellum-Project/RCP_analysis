import numpy as np
import matplotlib.pyplot as plt
import RCP_analysis.python.functions.config_loading as cfg

"""
Description:
    Plots a comparison of raw LFP signals (red) vs. cleaned/filtered signals (black)
    around the stimulation time to verify artifact removal efficacy.
    Displays individual trial traces and their means.

Input:
    Aligned LFP .npz files containing 'broadband_comp' and 'broadband_raw_comp' (TOGGLE in analyze_lfp_bands)

Output:
    PNG plots saved to: <Session>/results/figures/NPRW_LFP/compare_artifact_traces_<session>.png
"""

LFP_DIR = cfg.OUT_BASE / "checkpoints" / "NPRW_LFP"
FIG_DIR = cfg.OUT_BASE / "figures" / "NPRW_LFP"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def plot_artifact_comparison():
    if not LFP_DIR.exists():
        print(f"Directory not found: {LFP_DIR}")
        return

    files = sorted(LFP_DIR.glob("aligned_lfp__*.npz"))
    if not files:
        print(f"No aligned LFP npz files found in {LFP_DIR}")
        return
    
    for npz_file in files:
        print(f"Loading {npz_file.name}...")
        try:
            data = np.load(npz_file, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            continue

        if 'broadband_comp' not in data or 'broadband_raw_comp' not in data:
            print(f"Skipping {npz_file.name}: Missing broadband_comp keys.")
            continue
            
        print(f"  Keys found: {list(data.keys())}")
        
        bb_clean = data['broadband_comp'] # (N, Ch, T)
        bb_raw = data['broadband_raw_comp'] # (N, Ch, T)
        bb_unref = data.get('broadband_unref_comp')
        
        rel_t = data['rel_time_comp']
        session = data['session']
        
        n_trials, n_ch, n_time = bb_clean.shape
        
        # --- Channel Sort ---
        plot_chs = np.linspace(0, n_ch-1, 4, dtype=int)
        
        # --- Figure ---
        fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True, constrained_layout=True)
        fig.suptitle(f"Artifact Removal Check: {session}\nBlanking: -5ms to +120ms (Shaded)", fontsize=16)
        
        for i, ch in enumerate(plot_chs):
            ax = axes[i]
            
            # Data for this channel
            clean_mean = np.nanmean(bb_clean[:, ch, :], axis=0)
            raw_mean = np.nanmean(bb_raw[:, ch, :], axis=0)
            
            # Plot Raw Unblanked (Red) - Individual Trials
            ax.plot(rel_t, bb_raw[:, ch, :].T, color='tab:red', linestyle='-', alpha=0.15, linewidth=0.5)
            # Mean
            ax.plot(rel_t, raw_mean, color='tab:red', linestyle='-', alpha=1.0, label='Raw Mean', linewidth=2.0)
            
            # Plot Cleaned (Black, dashed) - Individual Trials
            ax.plot(rel_t, bb_clean[:, ch, :].T, color='k', linestyle='-', alpha=0.15, linewidth=0.5)
            # Mean
            ax.plot(rel_t, clean_mean, color='k', linestyle='--', alpha=1.0, label='Cleaned Mean', linewidth=2.0)
            
            ax.set_title(f"Channel {ch}")
            ax.set_ylabel("uV")
            
            # Highlight Blanking Window
            ax.axvspan(-5, 120, color='gray', alpha=0.1, label='Blanking Window')
            ax.axvline(-5, color='green', linestyle=':', label='Blank Start')
            ax.axvline(120, color='green', linestyle=':', label='Blank End')
             
        # Fix Legend (Deduplicate)
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
        axes[-1].set_xlabel("Time from Stim (ms)")
        
        # Save
        out_path = FIG_DIR / f"compare_artifact_traces_{session}.png"
        plt.savefig(out_path, dpi=150)
        print(f"  Saved plot to {out_path}")
        plt.close(fig)

if __name__ == "__main__":
    plot_artifact_comparison()
