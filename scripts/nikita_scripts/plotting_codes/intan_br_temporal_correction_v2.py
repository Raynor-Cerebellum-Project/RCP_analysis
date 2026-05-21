"""
calibrate_intan_br_timing_manual.py

Interactive manual calibration of Intan-Blackrock temporal alignment.

Workflow:
    1. Load Intan blocks and Blackrock recording
    2. Select N calibration blocks
    3. For each block:
       - Show Intan signal with detected pulses
       - Show BR signal with auto-detected pulses
       - Allow manual adjustment of BR detection window
    4. Fit time transform from manually verified blocks
    5. Save calibration and diagnostic plots

Controls:
    - Visual inspection of each block
    - Manual override of detection if needed
    - Interactive plots showing both systems
"""

import json
import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

# ---------- Config ----------
STIM_NPZ_ROOT = NPRW_AUX_DATA
OUTPUT_DIR = METADATA_ROOT / "time_calibrations"
FIG_DIR = METADATA_ROOT.parent / "figures" / "time_calibration"
SHIFT_CSV = METADATA_ROOT / "br_to_intan_shifts.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

UA_CFG = PARAMS.probes.get("UA")
PROCESS_ONLY = PARAMS.preprocessing.get("process_only")

# Calibration parameters
N_CALIBRATION_BLOCKS = 8
MIN_PULSES_PER_BLOCK = 3
SEARCH_RADIUS_MS = 50.0
REFINE_WINDOW_MS = 0.15

# Interactive mode
INTERACTIVE = True  # Set to False to auto-accept all detections


def _find_first_significant_peak(tr_local, ref_ratio=0.5):
    """Find the first significant peak in a trace."""
    diff_abs = np.abs(np.diff(tr_local, prepend=tr_local[0]))
    pk_idx = np.argmax(diff_abs)
    max_d = diff_abs[pk_idx]
    if max_d < 1e-6:
        return pk_idx
    for k in range(len(diff_abs)):
        if diff_abs[k] > ref_ratio * max_d:
            return k
    return pk_idx


def load_intan_blocks(stim_npz_path, verbose=True):
    """
    Load stimulation block times and pulse-level data from Intan.
    
    Returns
    -------
    block_starts : ndarray
        Start times of blocks in Intan samples
    block_ends : ndarray
        End times of blocks in Intan samples
    trigger_pairs : ndarray
        Individual pulse times [n_pulses, 2] (start, end)
    fs_intan : float
        Intan sampling frequency
    """
    if verbose:
        print(f"[load_intan] Loading {stim_npz_path.name}...")
    
    stim = np.load(stim_npz_path, allow_pickle=True)
    
    # Load sampling frequency
    fs_intan = 30000.0
    if 'meta' in stim:
        try:
            raw_meta = stim['meta']
            meta_str = raw_meta.item() if hasattr(raw_meta, 'item') else raw_meta
            meta_dict = json.loads(str(meta_str))
            if isinstance(meta_dict, dict):
                fs_intan = float(meta_dict.get('fs_hz', meta_dict.get('fs_intan', 30000.0)))
        except Exception as exc:
            if verbose:
                print(f"[load_intan] WARNING: Could not parse meta, using default")
    if 'fs_intan' in stim:
        fs_intan = float(stim['fs_intan'])
    
    # Load block boundaries
    if 'block_bounds_samples' not in stim:
        raise KeyError("No block_bounds_samples found in stim_stream.npz")
    
    block_bounds = stim['block_bounds_samples']
    block_starts = block_bounds[:, 0].astype(np.int64)
    block_ends = block_bounds[:, 1].astype(np.int64)
    
    # Load individual pulse times
    trigger_pairs = None
    if 'trigger_pairs' in stim:
        trigger_pairs = stim['trigger_pairs']
    
    if verbose:
        print(f"[load_intan] Loaded {len(block_starts)} blocks at {fs_intan} Hz")
        if trigger_pairs is not None:
            print(f"[load_intan] Loaded {len(trigger_pairs)} individual pulses")
    
    return block_starts, block_ends, trigger_pairs, fs_intan


def load_initial_shift(br_idx, fs_intan, verbose=True):
    """Load coarse Intan-to-BR shift from CSV."""
    if not SHIFT_CSV.exists():
        if verbose:
            print(f"[shift] WARNING: {SHIFT_CSV} not found, using shift=0")
        return 0.0
    
    br2shift = rcp.get_metadata_mapping(SHIFT_CSV, "br_idx", "shift_sample")
    raw = br2shift.get(br_idx)
    
    if raw is None or str(raw).strip() == "":
        if verbose:
            print(f"[shift] WARNING: No shift for BR {br_idx:03d}, using shift=0")
        return 0.0
    
    try:
        shift_val = float(str(raw).strip())
    except (ValueError, TypeError):
        if verbose:
            print(f"[shift] WARNING: Could not parse shift, using shift=0")
        return 0.0
    
    if verbose:
        print(f"[shift] Loaded shift = {shift_val:.0f} Intan samples ({shift_val/fs_intan:.3f} s)")
    
    return shift_val


def select_calibration_blocks(block_starts, block_ends, n_blocks=N_CALIBRATION_BLOCKS):
    """Select N blocks evenly distributed across recording."""
    n_total = len(block_starts)
    
    if n_total <= n_blocks:
        return np.arange(n_total)
    
    indices = np.linspace(0, n_total - 1, n_blocks, dtype=int)
    return indices


def extract_intan_block_pulses(trigger_pairs, block_start, block_end):
    """
    Extract individual pulse times within a specific Intan block.
    
    Returns
    -------
    pulse_times : ndarray
        Pulse onset times within the block
    """
    if trigger_pairs is None:
        return None
    
    # Find pulses within block
    pulse_starts = trigger_pairs[:, 0]
    mask = (pulse_starts >= block_start) & (pulse_starts <= block_end)
    
    block_pulses = pulse_starts[mask]
    return block_pulses


def detect_block_on_br(rec_ns6, rec_hpf, expected_start, expected_end,
                        ref_ch_name, fs_br, stim_freq_hz=400.0, verbose=False):
    """
    Detect artifact timing for a stimulation block on Blackrock.
    Same logic as UA_BR_analysis_mf.py.
    """
    n_total = rec_ns6.get_num_samples()
    
    search_margin = int(SEARCH_RADIUS_MS / 1000 * fs_br)
    start_search = max(0, expected_start - search_margin)
    end_search = min(n_total, expected_end + search_margin)
    
    if end_search <= start_search:
        return None
    
    # Extract HPF trace for detection
    tr_search = rec_hpf.get_traces(
        start_frame=start_search,
        end_frame=end_search,
        channel_ids=[ref_ch_name],
        return_in_uV=True
    )[:, 0]
    
    tr_diff_search = np.abs(np.diff(tr_search, prepend=tr_search[0]))
    
    # Find first pulse
    predicted_centre = expected_start - start_search
    slop = int(5.0 / 1000 * fs_br)
    win_lo = max(0, predicted_centre - slop)
    win_hi = min(len(tr_search), predicted_centre + slop)
    
    if win_hi <= win_lo:
        return None
    
    peak_in_win = np.argmax(tr_diff_search[win_lo:win_hi])
    anchor_idx = win_lo + peak_in_win
    max_diff = tr_diff_search[anchor_idx]
    
    if max_diff < 5.0:
        return None
    
    # Find first significant crossing
    first_pulse_in_search = None
    for k in range(win_lo, win_hi):
        if tr_diff_search[k] > 0.4 * max_diff:
            first_pulse_in_search = k
            break
    if first_pulse_in_search is None:
        first_pulse_in_search = anchor_idx
    
    # Refine using raw signal
    refine_samp = max(int(REFINE_WINDOW_MS / 1000.0 * fs_br), 3)
    r_lo = max(0, first_pulse_in_search - refine_samp)
    r_hi = min(len(tr_search), first_pulse_in_search + refine_samp + 1)
    
    tr_raw_refine = rec_ns6.get_traces(
        start_frame=start_search + r_lo,
        end_frame=start_search + r_hi,
        channel_ids=[ref_ch_name],
        return_in_uV=True
    )[:, 0]
    
    coarse_centre = _find_first_significant_peak(tr_raw_refine)
    true_first_pulse = start_search + r_lo + coarse_centre
    
    # Calculate expected number of pulses
    pulse_interval_ms = 1000.0 / stim_freq_hz
    interval_samp = int(pulse_interval_ms / 1000.0 * fs_br)
    stim_dur_ms = (expected_end - expected_start) * 1000.0 / fs_br
    num_pulses = int(np.floor(stim_dur_ms / pulse_interval_ms)) + 1
    
    # Forward-walk to find all pulses
    search_radius_samp = int((pulse_interval_ms * 0.4) / 1000.0 * fs_br)
    pulse_centres = [true_first_pulse]
    curr = true_first_pulse
    
    while len(pulse_centres) < num_pulses:
        nxt = curr + interval_samp
        
        s_lo = max(0, nxt - search_radius_samp)
        s_hi = min(n_total, nxt + search_radius_samp)
        
        if s_hi <= s_lo:
            curr = nxt
            pulse_centres.append(curr)
            continue
        
        tr_local_hpf = rec_hpf.get_traces(
            start_frame=s_lo,
            end_frame=s_hi,
            channel_ids=[ref_ch_name],
            return_in_uV=True
        )[:, 0]
        
        tr_diff = np.abs(np.diff(tr_local_hpf, prepend=tr_local_hpf[0]))
        peak_idx = np.argmax(tr_diff)
        
        if tr_diff[peak_idx] < 0.15 * max_diff:
            curr = nxt
        else:
            curr = s_lo + peak_idx
        
        pulse_centres.append(curr)
    
    # Refine each pulse
    refined_centres = []
    for p_idx in pulse_centres:
        r_lo = max(0, p_idx - refine_samp)
        r_hi = min(n_total, p_idx + refine_samp + 1)
        
        if r_lo >= r_hi:
            refined_centres.append(p_idx)
            continue
        
        tr_refine = rec_ns6.get_traces(
            start_frame=r_lo,
            end_frame=r_hi,
            channel_ids=[ref_ch_name],
            return_in_uV=True
        )[:, 0]
        
        tc = r_lo + _find_first_significant_peak(tr_refine)
        refined_centres.append(tc)
    
    pulse_centres = np.array(refined_centres, dtype=np.int64)
    
    return pulse_centres


def plot_block_comparison_interactive(block_idx, intan_pulses, br_pulses,
                                       rec_intan, rec_ns6, intan_ch_name, br_ch_name,
                                       block_start_intan, block_end_intan,
                                       expected_start_br, expected_end_br,
                                       fs_intan, fs_br, output_path):
    """
    Create interactive plot showing Intan and BR signals side-by-side.
    
    Returns
    -------
    accepted : bool
        Whether user accepts this detection (True if non-interactive)
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # ===== Top row: Intan block overview =====
    ax_intan_overview = fig.add_subplot(gs[0, :])
    
    # Extract Intan signal for full block
    margin_intan = int(10.0 / 1000 * fs_intan)  # 10ms margins
    start_intan_ext = max(0, block_start_intan - margin_intan)
    end_intan_ext = min(rec_intan.get_num_samples(), block_end_intan + margin_intan)
    
    tr_intan = rec_intan.get_traces(
        start_frame=start_intan_ext,
        end_frame=end_intan_ext,
        channel_ids=[intan_ch_name],
        return_in_uV=True
    ).flatten()
    
    time_intan = (np.arange(len(tr_intan)) + start_intan_ext - block_start_intan) / fs_intan * 1000
    
    ax_intan_overview.plot(time_intan, tr_intan, 'k-', linewidth=0.5, alpha=0.7)
    
    # Mark detected Intan pulses
    if intan_pulses is not None and len(intan_pulses) > 0:
        for pulse_t in intan_pulses:
            pulse_ms = (pulse_t - block_start_intan) / fs_intan * 1000
            ax_intan_overview.axvline(pulse_ms, color='blue', alpha=0.6, linewidth=1.5)
    
    ax_intan_overview.axvline(0, color='green', linestyle='--', alpha=0.5, label='Block start')
    ax_intan_overview.axvline((block_end_intan - block_start_intan) / fs_intan * 1000,
                               color='red', linestyle='--', alpha=0.5, label='Block end')
    
    ax_intan_overview.set_xlabel("Time from block start (ms)")
    ax_intan_overview.set_ylabel("Voltage (µV)")
    ax_intan_overview.set_title(f"Intan Block {block_idx} - Channel {intan_ch_name} - "
                                 f"{len(intan_pulses) if intan_pulses is not None else 0} pulses detected")
    ax_intan_overview.legend()
    ax_intan_overview.grid(True, alpha=0.3)
    
    # ===== Second row: BR block overview =====
    ax_br_overview = fig.add_subplot(gs[1, :])
    
    # Extract BR signal for full block
    margin_br = int(10.0 / 1000 * fs_br)
    start_br_ext = max(0, expected_start_br - margin_br)
    end_br_ext = min(rec_ns6.get_num_samples(), expected_end_br + margin_br)
    
    tr_br = rec_ns6.get_traces(
        start_frame=start_br_ext,
        end_frame=end_br_ext,
        channel_ids=[br_ch_name],
        return_in_uV=True
    ).flatten()
    
    time_br = (np.arange(len(tr_br)) + start_br_ext - expected_start_br) / fs_br * 1000
    
    ax_br_overview.plot(time_br, tr_br, 'k-', linewidth=0.5, alpha=0.7)
    
    # Mark detected BR pulses
    if br_pulses is not None and len(br_pulses) > 0:
        for pulse_t in br_pulses:
            pulse_ms = (pulse_t - expected_start_br) / fs_br * 1000
            ax_br_overview.axvline(pulse_ms, color='red', alpha=0.6, linewidth=1.5)
    
    ax_br_overview.axvline(0, color='green', linestyle='--', alpha=0.5, label='Expected start')
    ax_br_overview.axvline((expected_end_br - expected_start_br) / fs_br * 1000,
                            color='red', linestyle='--', alpha=0.5, label='Expected end')
    
    ax_br_overview.set_xlabel("Time from expected start (ms)")
    ax_br_overview.set_ylabel("Voltage (µV)")
    ax_br_overview.set_title(f"Blackrock Block {block_idx} - Channel {br_ch_name} - "
                             f"{len(br_pulses) if br_pulses is not None else 0} pulses detected")
    ax_br_overview.legend()
    ax_br_overview.grid(True, alpha=0.3)
    
    # ===== Bottom row: Individual pulse comparisons (first 3 pulses) =====
    for i in range(min(3, len(intan_pulses) if intan_pulses is not None else 0)):
        ax = fig.add_subplot(gs[2, i]) if i < 2 else fig.add_subplot(gs[2, 2])
        
        # Extract window around Intan pulse
        pulse_intan = intan_pulses[i]
        win_intan = int(5.0 / 1000 * fs_intan)
        start_i = max(0, pulse_intan - win_intan)
        end_i = min(rec_intan.get_num_samples(), pulse_intan + 2*win_intan)
        
        tr_i = rec_intan.get_traces(
            start_frame=start_i,
            end_frame=end_i,
            channel_ids=[intan_ch_name],
            return_in_uV=True
        ).flatten()
        
        time_i = (np.arange(len(tr_i)) + start_i - pulse_intan) / fs_intan * 1000
        
        ax.plot(time_i, tr_i, 'b-', linewidth=1, alpha=0.7, label='Intan')
        ax.axvline(0, color='blue', linestyle='--', alpha=0.6, linewidth=1.5)
        
        # Extract window around corresponding BR pulse (if detected)
        if br_pulses is not None and i < len(br_pulses):
            pulse_br = br_pulses[i]
            win_br = int(5.0 / 1000 * fs_br)
            start_b = max(0, pulse_br - win_br)
            end_b = min(rec_ns6.get_num_samples(), pulse_br + 2*win_br)
            
            tr_b = rec_ns6.get_traces(
                start_frame=start_b,
                end_frame=end_b,
                channel_ids=[br_ch_name],
                return_in_uV=True
            ).flatten()
            
            time_b = (np.arange(len(tr_b)) + start_b - pulse_br) / fs_br * 1000
            
            # Normalize BR signal for comparison
            tr_b_norm = tr_b / np.max(np.abs(tr_b)) * np.max(np.abs(tr_i))
            
            ax.plot(time_b, tr_b_norm, 'r-', linewidth=1, alpha=0.7, label='BR (norm)')
            ax.axvline(0, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        
        ax.set_xlabel("Time from pulse (ms)")
        ax.set_ylabel("Voltage (µV)")
        ax.set_title(f"Pulse {i+1}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2, 5)
    
    fig.suptitle(f"Block {block_idx} Comparison - Manual Verification", 
                 fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if INTERACTIVE:
        plt.show(block=True)
        # In real interactive mode, you'd add buttons or input prompts
        response = input(f"\nAccept Block {block_idx} detection? (y/n/q to quit): ").strip().lower()
        plt.close()
        
        if response == 'q':
            return None  # Signal to quit
        return response == 'y'
    else:
        plt.close()
        return True  # Auto-accept in non-interactive mode


def calibrate_from_blocks(intan_pulses_list, br_pulses_list, fs_intan, fs_br, verbose=True):
    """
    Fit time transformation using matched pulse pairs from multiple blocks.
    
    Parameters
    ----------
    intan_pulses_list : list of ndarray
        Intan pulse times for each block
    br_pulses_list : list of ndarray
        BR pulse times for each block
    fs_intan, fs_br : float
        Sampling frequencies
        
    Returns
    -------
    offset, scale, residual_std : float
        Calibration parameters
    pairs_intan, pairs_br : ndarray
        All matched pulse pairs
    """
    pairs_intan = []
    pairs_br = []
    
    for intan_pulses, br_pulses in zip(intan_pulses_list, br_pulses_list):
        if intan_pulses is None or br_pulses is None:
            continue
        
        # Pair pulses (assume same order and count)
        n_pairs = min(len(intan_pulses), len(br_pulses))
        pairs_intan.extend(intan_pulses[:n_pairs])
        pairs_br.extend(br_pulses[:n_pairs])
    
    pairs_intan = np.array(pairs_intan, dtype=np.float64)
    pairs_br = np.array(pairs_br, dtype=np.float64)
    
    if len(pairs_intan) < 10:
        raise ValueError(f"Too few pulse pairs ({len(pairs_intan)}) for calibration")
    
    if verbose:
        print(f"[calib] Fitting transform with {len(pairs_intan)} pulse pairs...")
    
    # Fit linear model
    def linear_model(t_intan, offset, scale):
        return offset + scale * t_intan
    
    scale_init = fs_br / fs_intan
    offset_init = np.median(pairs_br - scale_init * pairs_intan)
    
    popt, pcov = curve_fit(
        linear_model,
        pairs_intan,
        pairs_br,
        p0=[offset_init, scale_init]
    )
    
    offset, scale = popt
    
    # Calculate residuals
    predicted_br = linear_model(pairs_intan, offset, scale)
    residuals = pairs_br - predicted_br
    residual_std = np.std(residuals)
    residual_rms = np.sqrt(np.mean(residuals**2))
    
    drift_ppm = (scale - scale_init) * 1e6
    
    if verbose:
        print(f"[calib] ===== Calibration Results =====")
        print(f"[calib] Offset     = {offset:+.1f} samples ({offset/fs_br*1000:+.3f} ms)")
        print(f"[calib] Scale      = {scale:.12f}")
        print(f"[calib] Nominal    = {scale_init:.12f}")
        print(f"[calib] Drift      = {drift_ppm:+.3f} ppm")
        print(f"[calib] Residual std = {residual_std:.4f} samples ({residual_std/fs_br*1000:.5f} ms)")
        print(f"[calib] Residual RMS = {residual_rms:.4f} samples ({residual_rms/fs_br*1000:.5f} ms)")
    
    return offset, scale, residual_std, pairs_intan, pairs_br


def calibrate_session(br_idx, sess_folder, stim_npz_path, intan_recording_path, output_dir, fig_dir):
    """
    Interactive calibration for a single BR session.
    """
    print(f"\n{'='*60}")
    print(f"Calibrating BR session {br_idx:03d}: {sess_folder.name}")
    print(f"{'='*60}")
    
    try:
        # Load Intan data
        block_starts, block_ends, trigger_pairs, fs_intan = load_intan_blocks(stim_npz_path)
        
        # Load Intan recording for plotting
        print(f"[load] Loading Intan recording...")
        rec_intan = se.read_intan(intan_recording_path, stream_name='RHD2000 amplifier channel')
        # Select a channel for visualization (e.g., first channel)
        intan_ch_name = rec_intan.get_channel_ids()[0]
        
        # Load initial shift
        shift_intan_samples = load_initial_shift(br_idx, fs_intan)
        
        # Select calibration blocks
        selected_indices = select_calibration_blocks(block_starts, block_ends)
        print(f"[select] Using {len(selected_indices)} blocks for calibration")
        print(f"[select] Block indices: {selected_indices}")
        
        # Load Blackrock recording
        print(f"[load] Loading Blackrock NS6: {sess_folder.name}...")
        rec_ns6 = se.read_blackrock(sess_folder, stream_name='nsx6')
        fs_br = rec_ns6.get_sampling_frequency()
        
        # Apply UA mapping
        XLS = rcp.ua_excel_path(REPO_ROOT, PARAMS.probes)
        UA_MAP = rcp.load_UA_mapping_from_excel(XLS)
        rec_ns6, idx_rows, ua_elec, ua_nsp, ua_region, ua_region_names, ua_port = \
            rcp.apply_ua_mapping_with_regions(rec_ns6, UA_MAP, br_idx, METADATA_CSV)
        
        # Select reference channel
        ref_elec = 37 if ua_port == 'B' else 29
        ref_matches = np.where(ua_elec == ref_elec)[0]
        if ref_matches.size == 0:
            print(f"[ERROR] Reference electrode {ref_elec} not found!")
            return False
        
        ref_ch_idx = ref_matches[0]
        ref_ch_name = rec_ns6.get_channel_ids()[ref_ch_idx]
        print(f"[detect] Using reference electrode {ref_elec} (channel {ref_ch_name})")
        
        # Create HPF version
        print(f"[detect] Creating 300 Hz HPF recording...")
        rec_hpf = spre.highpass_filter(rec_ns6, freq_min=300.0)
        
        # Get stim frequency
        stim_freq_meta = rcp.get_metadata_mapping(METADATA_CSV, 'BR_File', 'Stim_Frequency_Hz')
        stim_freq = float(stim_freq_meta.get(br_idx, 400.0))
        if np.isnan(stim_freq):
            stim_freq = 400.0
        print(f"[detect] Stimulation frequency: {stim_freq} Hz")
        
        # Convert initial scale
        scale_init = fs_br / fs_intan
        
        # Process each block with interactive verification
        accepted_intan_pulses = []
        accepted_br_pulses = []
        
        for idx in selected_indices:
            start_i = block_starts[idx]
            end_i = block_ends[idx]
            
            # Get Intan pulses for this block
            intan_pulses = extract_intan_block_pulses(trigger_pairs, start_i, end_i)
            
            if intan_pulses is None or len(intan_pulses) < MIN_PULSES_PER_BLOCK:
                print(f"\n[SKIP] Block {idx}: Too few Intan pulses")
                continue
            
            # Convert to BR space
            expected_start = int((start_i - shift_intan_samples) * scale_init)
            expected_end = int((end_i - shift_intan_samples) * scale_init)
            
            print(f"\n[detect] Block {idx}: Intan [{start_i}, {end_i}] -> BR ~[{expected_start}, {expected_end}]")
            print(f"[detect] Intan pulses in block: {len(intan_pulses)}")
            
            # Detect on BR
            br_pulses = detect_block_on_br(
                rec_ns6, rec_hpf, expected_start, expected_end,
                ref_ch_name, fs_br, stim_freq, verbose=True
            )
            
            if br_pulses is None:
                print(f"[detect] ✗ BR detection failed")
                continue
            
            print(f"[detect] ✓ Detected {len(br_pulses)} BR pulses")
            
            # Create comparison plot
            block_fig_path = fig_dir / f"br{br_idx:03d}_block{idx:02d}_comparison.png"
            
            accepted = plot_block_comparison_interactive(
                idx, intan_pulses, br_pulses,
                rec_intan, rec_ns6, intan_ch_name, ref_ch_name,
                start_i, end_i, expected_start, expected_end,
                fs_intan, fs_br, block_fig_path
            )
            
            if accepted is None:  # User quit
                print("\n[INFO] Calibration cancelled by user")
                return False
            
            if accepted:
                print(f"[accept] ✓ Block {idx} accepted")
                accepted_intan_pulses.append(intan_pulses)
                accepted_br_pulses.append(br_pulses)
            else:
                print(f"[reject] ✗ Block {idx} rejected")
        
        # Check if we have enough blocks
        if len(accepted_intan_pulses) < 3:
            print(f"[ERROR] Too few accepted blocks ({len(accepted_intan_pulses)})")
            return False
        
        print(f"\n[calib] Proceeding with {len(accepted_intan_pulses)} accepted blocks")
        
        # Fit calibration
        offset, scale, residual_std, pairs_intan, pairs_br = calibrate_from_blocks(
            accepted_intan_pulses, accepted_br_pulses, fs_intan, fs_br
        )
        
        # Save calibration
        calib_path = output_dir / f"br{br_idx:03d}_time_calib.npz"
        np.savez_compressed(
            calib_path,
            offset=offset,
            scale=scale,
            residual_std=residual_std,
            fs_intan=fs_intan,
            fs_br=fs_br,
            n_blocks=len(accepted_intan_pulses),
            n_pairs=len(pairs_intan),
            pairs_intan=pairs_intan,
            pairs_br=pairs_br,
            ref_electrode=ref_elec,
            br_idx=br_idx,
            session_name=sess_folder.name,
            method='manual_interactive',
            n_calibration_blocks=N_CALIBRATION_BLOCKS
        )
        print(f"[save] Saved calibration: {calib_path}")
        
        print(f"[SUCCESS] Calibration complete for BR {br_idx:03d}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Calibration failed for BR {br_idx:03d}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main calibration loop."""
    print("="*60)
    print("Interactive Manual Intan-Blackrock Time Calibration")
    print("="*60)
    
    if not INTERACTIVE:
        print("[INFO] Running in NON-INTERACTIVE mode (auto-accept all)")
    
    # Load BR sessions
    BR_SESSION_FOLDERS = rcp.list_br_sessions(BR_ROOT)
    print(f"[INFO] Found {len(BR_SESSION_FOLDERS)} BR session folders")
    
    # Filter to PROCESS_ONLY if specified
    if PROCESS_ONLY:
        original_count = len(BR_SESSION_FOLDERS)
        filtered = []
        for br_idx in PROCESS_ONLY:
            match = [s for s in BR_SESSION_FOLDERS if int(s.name.split("_")[-1]) == br_idx]
            if not match:
                print(f"[WARN] BR index {br_idx} not found, skipping.")
                continue
            filtered.extend(match)
        print(f"[INFO] PROCESS_ONLY: {len(filtered)}/{original_count} sessions selected.")
        BR_SESSION_FOLDERS = filtered
    
    # Load metadata mapping
    intan2br = rcp.get_metadata_mapping(METADATA_CSV, "Intan_File", "BR_File")
    br2intan = {int(v): k for k, v in intan2br.items()}
    
    # Get Intan session info
    rate_files = sorted(NPRW_CKPT_ROOT.rglob("rates__*.npz"))
    intan_sessions = sorted({p.stem[len("rates__"):].split("__bin", 1)[0] for p in rate_files})
    _, intan_idx_to_sess = rcp.build_session_index_map(intan_sessions)
    
    # Process each session
    n_success = 0
    n_fail = 0
    
    for sess_folder in BR_SESSION_FOLDERS:
        br_idx = int(sess_folder.name.split('_')[-1])
        
        # Find Intan session
        intan_idx = br2intan.get(br_idx)
        if intan_idx is None:
            print(f"[SKIP] No Intan mapping for BR {br_idx}")
            n_fail += 1
            continue
        
        intan_session_name = intan_idx_to_sess.get(intan_idx)
        if intan_session_name is None:
            print(f"[SKIP] No Intan session name for index {intan_idx}")
            n_fail += 1
            continue
        
        # Find paths
        stim_npz_path = STIM_NPZ_ROOT / f"{intan_session_name}_Intan_streams" / "stim_stream.npz"
        if not stim_npz_path.exists():
            print(f"[SKIP] No stim_stream.npz for {intan_session_name}")
            n_fail += 1
            continue
        
        # Find Intan recording folder
        intan_recording_path = INTAN_ROOT / intan_session_name
        if not intan_recording_path.exists():
            print(f"[SKIP] Intan recording not found: {intan_recording_path}")
            n_fail += 1
            continue
        
        # Calibrate
        success = calibrate_session(br_idx, sess_folder, stim_npz_path, 
                                      intan_recording_path, OUTPUT_DIR, FIG_DIR)
        
        if success:
            n_success += 1
        else:
            n_fail += 1
    
    # Summary
    print("\n" + "="*60)
    print("CALIBRATION SUMMARY")
    print("="*60)
    print(f"Total sessions:  {n_success + n_fail}")
    print(f"  Successful:    {n_success}")
    print(f"  Failed:        {n_fail}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()