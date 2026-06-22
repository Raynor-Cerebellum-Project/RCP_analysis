#!/usr/bin/env python3
"""
Simple heart rate bar plots: Continuous Stim vs Control
Figure 1: Individual trials (longest control + all continuous stim)
Figure 2: Group comparison with statistics
Figure 3: Diagnostic showing peak detection quality
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.stats import ttest_ind
from RCP_analysis.python.functions.config_loading import PERI_ROOT, OUT_BASE, METADATA_CSV
from RCP_analysis.python.functions.utils import get_metadata_mapping

# === Parameters ===
HR_SAMPLING_RATE = 1000  # Hz (override, not 30kHz neural rate)
LOWPASS_CUTOFF_HZ = 20
MIN_PEAK_DISTANCE_MS = 300
PEAK_HEIGHT_PERCENTILE = 60
PEAK_MAX_PERCENTILE = 95
VALID_IBI_RANGE = (0.3, 1.5)  # seconds (40-200 BPM)
K_GLOBAL = 2.5

# Output directory
OUT_DIR = OUT_BASE / "figures" / "heart_rate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_heart_rate(hr_sig, fs=HR_SAMPLING_RATE, return_debug=False):
    """Compute median heart rate using adaptive artifact rejection."""
    # Lowpass filter
    nyq = fs / 2
    b, a = signal.butter(4, LOWPASS_CUTOFF_HZ / nyq, btype='low')
    hr_filt = signal.filtfilt(b, a, hr_sig)
    
    # Find ALL peaks first (just with distance constraint)
    min_distance = int(MIN_PEAK_DISTANCE_MS * fs / 1000)
    peaks, _ = signal.find_peaks(hr_filt, distance=min_distance)
    
    if len(peaks) < 3:
        if return_debug:
            return np.nan, hr_filt, np.array([]), np.array([]), None, None
        return np.nan
    
    # Get peak heights
    peak_heights = hr_filt[peaks]
    
    # Robust outlier detection using MAD (Median Absolute Deviation)
    median_height = np.median(peak_heights)
    mad = np.median(np.abs(peak_heights - median_height))
    mad_scale = 1.4826  # Scale factor to make MAD comparable to std for normal dist
    
    # Reject peaks outside median ± k*MAD (k=3 is common, like 3-sigma)
    lower_bound = median_height - K_GLOBAL * mad * mad_scale
    upper_bound = median_height + K_GLOBAL * mad * mad_scale
    
    valid_mask = (peak_heights >= lower_bound) & (peak_heights <= upper_bound)
    valid_peaks = peaks[valid_mask]
    rejected_peaks = peaks[~valid_mask]
    
    if len(valid_peaks) < 2:
        if return_debug:
            return np.nan, hr_filt, valid_peaks, rejected_peaks, lower_bound, upper_bound
        return np.nan
    
    # Compute IBIs and filter valid range
    ibis = np.diff(valid_peaks) / fs
    valid_ibis = ibis[(ibis >= VALID_IBI_RANGE[0]) & (ibis <= VALID_IBI_RANGE[1])]
    
    if len(valid_ibis) < 2:
        if return_debug:
            return np.nan, hr_filt, valid_peaks, rejected_peaks, lower_bound, upper_bound
        return np.nan
    
    median_ibi = np.median(valid_ibis)
    bpm = 60.0 / median_ibi
    
    if return_debug:
        return bpm, hr_filt, valid_peaks, rejected_peaks, lower_bound, upper_bound
    return bpm

def load_npz_files(folder_path, recursive=False):
    """Load NPZ files from folder, optionally recursive."""
    folder = Path(folder_path)
    if recursive:
        npz_files = list(folder.rglob("*.npz"))
    else:
        npz_files = list(folder.glob("*.npz"))
    return npz_files

def plot_diagnostic_figure(hr_sig_raw, hr_filt, valid_peaks, rejected_peaks,
                           height_thresh_low, height_thresh_high, bpm, 
                           title, fs=HR_SAMPLING_RATE):
    """Create diagnostic plot showing peak detection."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    time_sec = np.arange(len(hr_sig_raw)) / fs
    
    # Panel 1: Raw signal
    ax1 = axes[0]
    ax1.plot(time_sec, hr_sig_raw, 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_ylabel('Raw HR Signal')
    ax1.set_title(f'{title}\nRaw Signal')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Filtered signal with peaks
    ax2 = axes[1]
    ax2.plot(time_sec, hr_filt, 'b-', linewidth=0.5, label='Filtered')
    ax2.axhline(y=height_thresh_low, color='g', linestyle='--', alpha=0.7, 
                label=f'Lower bound (median - {K_GLOBAL}*MAD)')
    ax2.axhline(y=height_thresh_high, color='r', linestyle='--', alpha=0.7, 
                label=f'Upper bound (median + {K_GLOBAL}*MAD)')
    
    # Valid peaks (green)
    if len(valid_peaks) > 0:
        ax2.plot(time_sec[valid_peaks], hr_filt[valid_peaks], 'go', markersize=4, 
                 label=f'Valid peaks (n={len(valid_peaks)})')
    
    # Rejected peaks (red X)
    if len(rejected_peaks) > 0:
        ax2.plot(time_sec[rejected_peaks], hr_filt[rejected_peaks], 'rx', markersize=6, 
                 markeredgewidth=2, label=f'Rejected peaks (n={len(rejected_peaks)})')
    
    ax2.set_ylabel('Filtered HR Signal')
    ax2.set_title(f'Filtered Signal + Peak Detection | BPM: {bpm:.1f}' if not np.isnan(bpm) 
                  else 'Filtered Signal + Peak Detection | BPM: FAILED')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: IBIs over time
    ax3 = axes[2]
    if len(valid_peaks) > 1:
        ibis = np.diff(valid_peaks) / fs
        ibi_times = time_sec[valid_peaks[1:]]
        
        # Color valid vs invalid IBIs
        valid_mask = (ibis >= VALID_IBI_RANGE[0]) & (ibis <= VALID_IBI_RANGE[1])
        
        ax3.scatter(ibi_times[valid_mask], ibis[valid_mask], c='green', s=20, 
                    label=f'Valid IBIs (n={valid_mask.sum()})')
        ax3.scatter(ibi_times[~valid_mask], ibis[~valid_mask], c='red', s=20, 
                    label=f'Invalid IBIs (n={(~valid_mask).sum()})')
        
        # Valid range shading
        ax3.axhspan(VALID_IBI_RANGE[0], VALID_IBI_RANGE[1], alpha=0.2, color='green',
                    label=f'Valid range ({VALID_IBI_RANGE[0]}-{VALID_IBI_RANGE[1]}s)')
        
        # Median line
        if valid_mask.sum() > 0:
            median_ibi = np.median(ibis[valid_mask])
            ax3.axhline(y=median_ibi, color='blue', linestyle='-', linewidth=2,
                        label=f'Median IBI: {median_ibi:.3f}s ({60/median_ibi:.1f} BPM)')
    
    ax3.set_ylabel('IBI (seconds)')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_title('Inter-Beat Intervals')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 2)
    
    plt.tight_layout()
    return fig

def main():
    # Get stim frequency mapping from metadata
    stim_freq_map = get_metadata_mapping(METADATA_CSV, "BR_File", "Stim_Frequency_Hz")
    
    # === DEBUG: Check what files exist ===
    print(f"\n=== DEBUG: File Discovery ===")
    print(f"PERI_ROOT: {PERI_ROOT}")
    print(f"Looking for continuous_stim in: {PERI_ROOT / 'continuous_stim'}")
    print(f"Directory exists: {(PERI_ROOT / 'continuous_stim').exists()}")
    
    cont_stim_path = PERI_ROOT / "continuous_stim"
    if cont_stim_path.exists():
        all_files = list(cont_stim_path.glob("*"))
        print(f"All files in continuous_stim: {len(all_files)}")
        for f in all_files[:5]:  # Show first 5
            print(f"  {f.name}")
        npz_files = list(cont_stim_path.glob("*.npz"))
        print(f"NPZ files found: {len(npz_files)}")
    else:
        print("WARNING: continuous_stim directory does not exist!")
    
    # === Load and process data ===
    
    # Control trials (recursive for target_A, target_B subfolders)
    control_files = load_npz_files(PERI_ROOT / "control_reaches", recursive=True)
    print(f"\nControl files found: {len(control_files)}")
    
    control_data = []
    control_failed = 0
    for f in control_files:
        npz = np.load(f, allow_pickle=True)
        br_idx = int(npz["br_idx"])
        hr_sig = npz["hr_sig"]
        duration_sec = len(hr_sig) / HR_SAMPLING_RATE
        bpm = compute_heart_rate(hr_sig)
        if not np.isnan(bpm):
            control_data.append({
                'br_idx': br_idx,
                'bpm': bpm,
                'duration': duration_sec,
                'file': f,
                'hr_sig': hr_sig
            })
        else:
            control_failed += 1
    print(f"Control: {len(control_data)} passed, {control_failed} failed HR computation")
    
    # Continuous stim trials
    cont_stim_files = load_npz_files(PERI_ROOT / "continuous_stim")
    print(f"\nContinuous stim files found: {len(cont_stim_files)}")
    
    cont_stim_data = []
    cont_stim_failed = 0
    for f in cont_stim_files:
        npz = np.load(f, allow_pickle=True)
        br_idx = int(npz["br_idx"])
        hr_sig = npz["hr_sig"]
        print(f"  BR{br_idx}: hr_sig shape={hr_sig.shape}, duration={len(hr_sig)/HR_SAMPLING_RATE:.1f}s")
        bpm = compute_heart_rate(hr_sig)
        print(f"    -> BPM: {bpm}")
        stim_freq = stim_freq_map.get(br_idx, "?")
        if not np.isnan(bpm):
            cont_stim_data.append({
                'br_idx': br_idx,
                'bpm': bpm,
                'stim_freq': stim_freq,
                'file': f,
                'hr_sig': hr_sig
            })
        else:
            cont_stim_failed += 1
    print(f"Continuous stim: {len(cont_stim_data)} passed, {cont_stim_failed} failed HR computation")
    
    # Stim reaches (for group comparison)
    stim_reach_files = load_npz_files(PERI_ROOT / "stim_reaches", recursive=True)
    print(f"\nStim reach files found: {len(stim_reach_files)}")
    
    stim_reach_bpms = []
    stim_reach_failed = 0
    for f in stim_reach_files:
        npz = np.load(f, allow_pickle=True)
        hr_sig = npz["hr_sig"]
        bpm = compute_heart_rate(hr_sig)
        if not np.isnan(bpm):
            stim_reach_bpms.append(bpm)
        else:
            stim_reach_failed += 1
    print(f"Stim reaches: {len(stim_reach_bpms)} passed, {stim_reach_failed} failed HR computation")
    
    # === Safety checks before plotting ===
    if len(control_data) == 0:
        print("\nERROR: No valid control data. Cannot proceed.")
        return
    
    if len(cont_stim_data) == 0:
        print("\nWARNING: No valid continuous stim data. Skipping Figure 1 cont_stim bars.")
    
    # === Figure 1: Individual Bar Plot ===
    # Find longest control trial
    longest_control = max(control_data, key=lambda x: x['duration'])
    
    # Sort continuous stim by BR index
    cont_stim_data.sort(key=lambda x: x['br_idx'])
    
    # Build bars
    labels = [f"BR{longest_control['br_idx']}\n(Control)"]
    values = [longest_control['bpm']]
    colors = ['#2ecc71']  # green for control
    
    for d in cont_stim_data:
        labels.append(f"BR{d['br_idx']}\n{d['stim_freq']}Hz")
        values.append(d['bpm'])
        colors.append('#e74c3c')  # red for stim
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.bar(range(len(values)), values, color=colors, edgecolor='black', linewidth=1)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('Heart Rate (BPM)', fontsize=12)
    ax1.set_title('Heart Rate: Control vs Continuous Stimulation Trials', fontsize=14)
    ax1.set_ylim(0, 190)
    ax1.axhline(y=longest_control['bpm'], color='#2ecc71', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    fig1.savefig(OUT_DIR / "hr_individual_trials.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"\nSaved Figure 1 to {OUT_DIR / 'hr_individual_trials.png'}")
    
    # === Figure 2: Group Comparison ===
    control_bpms = [d['bpm'] for d in control_data]
    cont_stim_bpms = [d['bpm'] for d in cont_stim_data]
    
    # Only plot groups with data
    groups = []
    means = []
    sems = []
    ns = []
    bar_colors = []
    
    if len(control_bpms) > 0:
        groups.append('Control')
        means.append(np.mean(control_bpms))
        sems.append(np.std(control_bpms)/np.sqrt(len(control_bpms)) if len(control_bpms) > 1 else 0)
        ns.append(len(control_bpms))
        bar_colors.append('#2ecc71')
    
    if len(stim_reach_bpms) > 0:
        groups.append('Stim Reaches')
        means.append(np.mean(stim_reach_bpms))
        sems.append(np.std(stim_reach_bpms)/np.sqrt(len(stim_reach_bpms)) if len(stim_reach_bpms) > 1 else 0)
        ns.append(len(stim_reach_bpms))
        bar_colors.append('#3498db')
    
    if len(cont_stim_bpms) > 0:
        groups.append('Continuous Stim')
        means.append(np.mean(cont_stim_bpms))
        sems.append(np.std(cont_stim_bpms)/np.sqrt(len(cont_stim_bpms)) if len(cont_stim_bpms) > 1 else 0)
        ns.append(len(cont_stim_bpms))
        bar_colors.append('#e74c3c')
    
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    bars = ax2.bar(groups, means, yerr=sems, color=bar_colors, 
                   edgecolor='black', linewidth=1, capsize=5)
    
    ax2.set_ylabel('Heart Rate (BPM)', fontsize=12)
    ax2.set_title('Heart Rate by Condition', fontsize=14)
    ax2.set_ylim(0, 190)
    
    # Add n values
    for bar, n in zip(bars, ns):
        ax2.text(bar.get_x() + bar.get_width()/2, 5,
                f'n={n}', ha='center', va='bottom', fontsize=10)
    
    # Statistics (only if we have both groups to compare)
    def add_sig_bracket(ax, x1, x2, y, p):
        if np.isnan(p):
            return
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.plot([x1, x1, x2, x2], [y, y+1, y+1, y], 'k-', linewidth=1)
        ax.text((x1+x2)/2, y+1.5, f'{stars}\np={p:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Only add brackets if we have the groups
    if len(control_bpms) >= 2 and len(stim_reach_bpms) >= 2:
        _, p_stim_reach = ttest_ind(control_bpms, stim_reach_bpms)
        ctrl_idx = groups.index('Control')
        stim_idx = groups.index('Stim Reaches')
        add_sig_bracket(ax2, ctrl_idx, stim_idx, max(means) + 10, p_stim_reach)
    else:
        p_stim_reach = np.nan
    
    if len(control_bpms) >= 2 and len(cont_stim_bpms) >= 2:
        _, p_cont_stim = ttest_ind(control_bpms, cont_stim_bpms)
        ctrl_idx = groups.index('Control')
        cont_idx = groups.index('Continuous Stim')
        add_sig_bracket(ax2, ctrl_idx, cont_idx, max(means) + 18, p_cont_stim)
    else:
        p_cont_stim = np.nan
    
    plt.tight_layout()
    fig2.savefig(OUT_DIR / "hr_group_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved Figure 2 to {OUT_DIR / 'hr_group_comparison.png'}")
    
    # === Figure 3: Diagnostic plots ===
    diag_dir = OUT_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    # Diagnostic for longest control
    hr_sig = longest_control['hr_sig']
    bpm, hr_filt, valid_peaks, rejected_peaks, thresh_low, thresh_high = compute_heart_rate(hr_sig, return_debug=True)
    fig_diag = plot_diagnostic_figure(
        hr_sig, hr_filt, valid_peaks, rejected_peaks, thresh_low, thresh_high, bpm,
        title=f"Control BR{longest_control['br_idx']} (longest, {longest_control['duration']:.1f}s)"
    )
    fig_diag.savefig(diag_dir / f"diag_control_BR{longest_control['br_idx']}.png", 
                     dpi=150, bbox_inches='tight')
    plt.close(fig_diag)
    
    # Diagnostic for each continuous stim trial
    for d in cont_stim_data:
        hr_sig = d['hr_sig']
        bpm, hr_filt, valid_peaks, rejected_peaks, thresh_low, thresh_high = compute_heart_rate(hr_sig, return_debug=True)
        fig_diag = plot_diagnostic_figure(
            hr_sig, hr_filt, valid_peaks, rejected_peaks, thresh_low, thresh_high, bpm,
            title=f"Continuous Stim BR{d['br_idx']} ({d['stim_freq']}Hz)"
        )
        fig_diag.savefig(diag_dir / f"diag_contstim_BR{d['br_idx']}.png", 
                         dpi=150, bbox_inches='tight')
        plt.close(fig_diag)
    
    print(f"Saved diagnostic figures to {diag_dir}")
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Control: {np.mean(control_bpms):.1f} ± {np.std(control_bpms):.1f} BPM (n={len(control_bpms)})")
    print(f"Stim Reaches: {np.mean(stim_reach_bpms):.1f} ± {np.std(stim_reach_bpms):.1f} BPM (n={len(stim_reach_bpms)})")
    if len(cont_stim_bpms) > 0:
        print(f"Continuous Stim: {np.mean(cont_stim_bpms):.1f} ± {np.std(cont_stim_bpms):.1f} BPM (n={len(cont_stim_bpms)})")
    else:
        print(f"Continuous Stim: NO DATA (n=0)")
    print(f"\nStatistics vs Control:")
    print(f"  Stim Reaches: p={p_stim_reach:.4f}" if not np.isnan(p_stim_reach) else "  Stim Reaches: insufficient data")
    print(f"  Continuous Stim: p={p_cont_stim:.4f}" if not np.isnan(p_cont_stim) else "  Continuous Stim: insufficient data")


if __name__ == "__main__":
    main()