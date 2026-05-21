import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import spikeinterface.extractors as se
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

# Calibration sessions (one per port)
CALIBRATION_SESSIONS = {
    'A': 12,  # BR session with Port A
    'B': 22,  # BR session with Port B
}

def detect_artifact_peak_in_window(trace, search_window_samples, method='derivative'):
    """
    Find the artifact peak within a search window.
    
    Parameters
    ----------
    trace : 1D array
        Raw voltage trace for a single channel
    search_window_samples : tuple (start, end)
        Sample indices to search within
    method : str
        'derivative' or 'extremum'
    
    Returns
    -------
    peak_idx : int
        Sample index of detected artifact peak
    peak_value : float
        Value at peak (for QC)
    """
    start, end = search_window_samples
    segment = trace[start:end]
    
    if method == 'derivative':
        # Find steepest deflection
        deriv = np.abs(np.diff(segment, prepend=segment[0]))
        peak_local = np.argmax(deriv)
    else:  # extremum
        # Find largest absolute deviation from baseline
        baseline = np.median(trace[max(0, start-100):start])
        deviation = np.abs(segment - baseline)
        peak_local = np.argmax(deviation)
    
    peak_idx = start + peak_local
    peak_value = trace[peak_idx]
    
    return peak_idx, peak_value


def calibrate_channel_lags(br_idx, port):
    """
    Calibrate per-channel artifact lags for a given session.
    Constraint: artifact peak must be within 0.3-0.8ms AFTER Intan stim onset.
    
    Returns
    -------
    lag_table : dict
        {channel_id: {'lag_samples': int, 'lag_ms': float, 'confidence': float}}
    """
    print(f"\n{'='*70}")
    print(f"Calibrating channel lags for BR {br_idx:03d} (Port {port})")
    print(f"{'='*70}\n")
    
    # Load session
    sess_folders = rcp.list_br_sessions(BR_ROOT)
    sess = [s for s in sess_folders if int(s.name.split('_')[-1]) == br_idx][0]
    
    rec_ns6 = se.read_blackrock(sess, stream_name='nsx6', all_annotations=True)
    
    XLS = rcp.ua_excel_path(REPO_ROOT, PARAMS.probes)
    UA_MAP = rcp.load_UA_mapping_from_excel(XLS)
    
    rec_ns6, idx_rows, ua_elec, ua_nsp, ua_region, ua_region_names, ua_port = \
        rcp.apply_ua_mapping_with_regions(rec_ns6, UA_MAP, br_idx, METADATA_CSV)
    
    fs_ua = rec_ns6.get_sampling_frequency()
    n_total = rec_ns6.get_num_samples()
    
    # Get Intan stim timing
    SHIFT_CSV = METADATA_ROOT / "br_to_intan_shifts.csv"
    stim_npz_path, _ = rcp.stim_npz_path_from_br_idx(br_idx, METADATA_CSV, NPRW_AUX_DATA)
    
    stim = rcp.load_stim_detection(stim_npz_path)
    block_bounds = stim.get("block_bounds_samples", [])
    
    br2fs_intan = rcp.get_metadata_mapping(SHIFT_CSV, 'br_idx', 'fs_intan')
    br2shift_samp_intan = rcp.get_metadata_mapping(SHIFT_CSV, 'br_idx', 'shift_sample')
    
    fs_intan = float(br2fs_intan.get(br_idx, fs_ua))
    shift_samp_intan = float(br2shift_samp_intan.get(br_idx, 0))
    
    # Convert first Intan stim to Blackrock time (nominal, no PPM correction yet)
    PPM_CORRECTION = -13.951
    scale_corrected = (fs_ua / fs_intan) * (1.0 + PPM_CORRECTION / 1e6)
    
    starts_intan = block_bounds[:, 0].astype(np.int64)
    starts_ua = np.round((starts_intan - shift_samp_intan) * scale_corrected).astype(np.int64)
    
    # Use first stim block for calibration
    first_stim_ua = starts_ua[0]
    
    print(f"First Intan stim mapped to BR sample: {first_stim_ua}")
    print(f"Calibrating lags for {rec_ns6.get_num_channels()} channels...")
    print(f"Constraint: artifact peak must be within [0.3, 0.8] ms after Intan stim\n")
    
    # ══════════════════════════════════════════════════════════════════════
    # CONSTRAINED SEARCH WINDOW: 0.3 to 0.8ms after Intan stim
    # ══════════════════════════════════════════════════════════════════════
    search_start_ms = 0.3
    search_end_ms = 0.8
    
    search_start_samples = int(search_start_ms / 1000 * fs_ua)
    search_end_samples = int(search_end_ms / 1000 * fs_ua)
    
    lag_table = {}
    failed_channels = []
    
    for ch_idx in range(rec_ns6.get_num_channels()):
        ch_name = rec_ns6.get_channel_ids()[ch_idx]
        elec_id = int(ua_elec[ch_idx]) if ch_idx < len(ua_elec) else 0
        
        if elec_id == 0:
            continue  # Skip unmapped channels
        
        # Extract trace from Intan time + search_start to + search_end
        win_start = first_stim_ua + search_start_samples
        win_end = min(n_total, first_stim_ua + search_end_samples)
        
        if win_end <= win_start:
            failed_channels.append((ch_name, elec_id, "window_invalid"))
            continue
        
        trace = rec_ns6.get_traces(
            start_frame=win_start,
            end_frame=win_end,
            channel_ids=[ch_name],
            return_in_uV=True
        ).flatten()
        
        # Detect peak within constrained window
        search_window = (0, len(trace))
        peak_idx_local, peak_value = detect_artifact_peak_in_window(
            trace, search_window, method='derivative'
        )
        
        # Convert to global sample index
        peak_idx_global = win_start + peak_idx_local
        
        # Calculate lag relative to Intan time
        lag_samples = peak_idx_global - first_stim_ua
        lag_ms = (lag_samples / fs_ua) * 1000.0
        
        # Sanity check: lag should be in [search_start_ms, search_end_ms] by construction
        if not (search_start_ms <= lag_ms <= search_end_ms):
            failed_channels.append((ch_name, elec_id, f"lag_out_of_bounds_{lag_ms:.3f}ms"))
            continue
        
        # Confidence metric: peak magnitude relative to baseline noise
        baseline_segment_start = max(0, first_stim_ua - 200)
        baseline_segment_end = max(0, first_stim_ua - 50)
        baseline = rec_ns6.get_traces(
            start_frame=baseline_segment_start,
            end_frame=baseline_segment_end,
            channel_ids=[ch_name],
            return_in_uV=True
        ).flatten()
        
        noise_level = np.std(baseline)
        confidence = np.abs(peak_value) / noise_level if noise_level > 0 else 0.0
        
        lag_table[ch_name] = {
            'electrode': elec_id,
            'lag_samples': int(lag_samples),
            'lag_ms': float(lag_ms),
            'confidence': float(confidence)
        }
        
        print(f"  Ch {ch_name} (Elec {elec_id:3d}): lag = {lag_samples:+4d} samp ({lag_ms:+6.3f} ms), conf = {confidence:.2f}")
    
    # Report failures
    if failed_channels:
        print(f"\n[WARN] {len(failed_channels)} channels failed calibration:")
        for ch_name, elec_id, reason in failed_channels[:10]:  # Show first 10
            print(f"  {ch_name} (Elec {elec_id}): {reason}")
        if len(failed_channels) > 10:
            print(f"  ... and {len(failed_channels) - 10} more")
    
    print(f"\n✓ Successfully calibrated {len(lag_table)}/{rec_ns6.get_num_channels()} channels")
    
    return lag_table, rec_ns6, first_stim_ua

def save_lag_calibration(lag_table, port, out_path):
    """Save lag calibration to CSV."""
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['channel_id', 'electrode', 'port', 'lag_samples', 'lag_ms', 'confidence'])
        
        for ch_name, data in lag_table.items():
            writer.writerow([
                ch_name,
                data['electrode'],
                port,
                data['lag_samples'],
                data['lag_ms'],
                data['confidence']
            ])
    
    print(f"\n✓ Saved calibration: {out_path}")


def plot_lag_distribution(lag_table, port, out_path):
    """Plot histogram of lag values for QC."""
    lags_ms = [v['lag_ms'] for v in lag_table.values()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(lags_ms, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.median(lags_ms), color='r', linestyle='--', 
               label=f'Median: {np.median(lags_ms):.3f} ms')
    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('Number of channels')
    ax.set_title(f'Artifact Lag Distribution - Port {port}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    
    print(f"✓ Saved QC plot: {out_path}")


def plot_all_channels_grid(lag_table, rec_ns6, first_stim, port, out_path):
    """Plot all 128 channels in an 8×16 grid (-0.3 to 1.2ms window)."""
    
    print(f"\n[PLOT] Creating 8×16 grid of all channels...")
    
    fs_ua = rec_ns6.get_sampling_frequency()
    
    # Time window: -0.3 to 1.2ms
    pre_ms = 0.3
    post_ms = 1.2
    
    win_start = first_stim - int(pre_ms / 1000 * fs_ua)
    win_end = first_stim + int(post_ms / 1000 * fs_ua)
    n_samples = win_end - win_start
    time_axis = np.linspace(-pre_ms, post_ms, n_samples)
    
    # Create 8 rows × 16 columns grid
    fig, axes = plt.subplots(8, 16, figsize=(32, 16))
    axes = axes.flatten()
    
    # Get sorted list of channels (by electrode ID for consistent ordering)
    sorted_channels = sorted(lag_table.items(), key=lambda x: x[1]['electrode'])
    
    for i, (ch_name, data) in enumerate(sorted_channels[:128]):  # Max 128 channels
        ax = axes[i]
        
        # Extract trace
        trace = rec_ns6.get_traces(
            start_frame=win_start,
            end_frame=win_end,
            channel_ids=[ch_name],
            return_in_uV=True
        ).flatten()
        
        # Plot
        ax.plot(time_axis, trace, 'k-', lw=0.6, alpha=0.8)
        
        # Mark Intan onset and detected lag
        ax.axvline(0, color='b', linestyle='--', lw=0.8, alpha=0.5)
        ax.axvline(data['lag_ms'], color='r', linestyle='--', lw=0.8, alpha=0.7)
        ax.axvspan(0.3, 0.8, alpha=0.05, color='green')
        
        # Title with electrode and lag
        ax.set_title(f"E{data['electrode']:03d}\n{data['lag_ms']:.2f}ms", 
                     fontsize=6, pad=2)
        
        # Minimal axis decorations
        ax.set_xlim(-0.3, 1.2)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.2, linestyle=':')
        
        # Only show x-labels on bottom row
        if i < 112:  # Not in bottom row (128 - 16 = 112)
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('ms', fontsize=5)
        
        # Only show y-labels on left column
        if i % 16 != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('µV', fontsize=5)
    
    # Hide any unused subplots
    for i in range(len(sorted_channels), 128):
        axes[i].axis('off')
    
    # Overall title
    fig.suptitle(f'All Channels Grid - Port {port} | Blue: Intan onset | Red: Detected peak | Green: Search window',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved all-channels grid: {out_path}")


def main():
    out_dir = REPO_ROOT / 'config' / 'channel_lag_calibration'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for port, br_idx in CALIBRATION_SESSIONS.items():
        lag_table, rec_ns6, first_stim = calibrate_channel_lags(br_idx, port)
        
        # Save calibration
        csv_path = out_dir / f'channel_lags_port{port}_NEW.csv'
        save_lag_calibration(lag_table, port, csv_path)
        
        # QC plot - distribution
        plot_path = out_dir / f'lag_distribution_port{port}_NEW.png'
        plot_lag_distribution(lag_table, port, plot_path)
        
        # NEW: All channels in 8×16 grid
        grid_path = out_dir / f'all_channels_grid_port{port}_NEW.png'
        plot_all_channels_grid(lag_table, rec_ns6, first_stim, port, grid_path)
        
        # Debug: plot first 16 channels with detected peaks
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        search_margin = int(5.0 / 1000 * rec_ns6.get_sampling_frequency())
        time_axis = np.linspace(-5, 5, 2 * search_margin)
        
        for i, (ch_name, data) in enumerate(list(lag_table.items())[:16]):
            win_start = max(0, first_stim - search_margin)
            win_end = min(rec_ns6.get_num_samples(), first_stim + search_margin)
            
            trace = rec_ns6.get_traces(
                start_frame=win_start, end_frame=win_end,
                channel_ids=[ch_name], return_in_uV=True
            ).flatten()
            
            axes[i].plot(time_axis[:len(trace)], trace, 'k-', lw=0.8)
            axes[i].axvline(0, color='b', linestyle='--', alpha=0.5, label='Intan')
            axes[i].axvline(data['lag_ms'], color='r', linestyle='--', alpha=0.7, label='Detected')
            axes[i].set_title(f"{ch_name}\n{data['lag_ms']:+.2f} ms", fontsize=8)
            axes[i].grid(True, alpha=0.3)
            if i == 0:
                axes[i].legend(fontsize=7)
        
        fig.suptitle(f'First 16 Channels - Port {port} - BR {br_idx:03d}', fontweight='bold')
        fig.tight_layout()
        debug_path = out_dir / f'debug_first16_port{port}_NEW.png'
        fig.savefig(debug_path, dpi=150)
        plt.close()
        
        print(f"✓ Saved debug plot: {debug_path}\n")


if __name__ == "__main__":
    main()