"""
calibrate_intan_br_timing.py

Solve for the true temporal transformation between Intan and Blackrock systems
using stimulus artifacts as ground truth timing markers.

Output:
    - time_calibrations/br{idx:03d}_time_calib.npz : Calibration parameters per session
    - figures/time_calibration/ : Diagnostic plots showing drift correction
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
STIM_NPZ_ROOT = NPRW_AUX_DATA  # Where Intan stim detections are stored
OUTPUT_DIR = METADATA_ROOT / "time_calibrations"
FIG_DIR = METADATA_ROOT.parent / "figures" / "time_calibration"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# UA probe config
UA_CFG = PARAMS.probes.get("UA")
TRIANGLE_SYNC_CH = int(UA_CFG.get("triangle_sync_ch", 138))

PROCESS_ONLY = PARAMS.preprocessing.get("process_only")

# Detection parameters
MIN_ARTIFACT_SEPARATION_MS = 2.0  # Minimum time between pulses
DERIVATIVE_PERCENTILE = 99.9  # Threshold for artifact detection
PAIRING_TOLERANCE_SAMPLES = 500  # Max distance (BR samples) to pair Intan/BR pulses
MIN_MATCH_PAIRS = 5             # Absolute minimum matched pairs needed for calibration


def detect_all_artifacts_br(rec_ns6, ref_ch_name, fs_br, verbose=True):
    """
    Detect ALL stimulus artifacts in Blackrock recording using derivative method.
    
    Parameters
    ----------
    rec_ns6 : SpikeInterface Recording
        Raw Blackrock recording
    ref_ch_name : str
        Channel ID to use for detection (should have strong artifacts)
    fs_br : float
        Blackrock sampling frequency
    verbose : bool
        Print progress
        
    Returns
    -------
    artifact_samples : ndarray
        Sample indices where artifacts were detected
    """
    if verbose:
        print(f"[detect_br] Detecting artifacts on channel {ref_ch_name}...")
    
    n_total = rec_ns6.get_num_samples()
    
    # High-pass filter to emphasize fast transients
    if verbose:
        print(f"[detect_br] Applying 300 Hz highpass filter...")
    rec_hpf = spre.highpass_filter(rec_ns6, freq_min=300.0)
    
    # Extract reference channel in chunks to manage memory
    chunk_size = int(60 * fs_br)  # 60 second chunks
    n_chunks = int(np.ceil(n_total / chunk_size))
    
    all_peaks = []
    
    for i in range(n_chunks):
        start_frame = i * chunk_size
        end_frame = min((i + 1) * chunk_size, n_total)
        
        if verbose and i % 10 == 0:
            print(f"[detect_br] Processing chunk {i+1}/{n_chunks} ({start_frame/fs_br:.1f}s - {end_frame/fs_br:.1f}s)")
        
        tr_chunk = rec_hpf.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_ids=[ref_ch_name],
            return_in_uV=True
        )[:, 0]
        
        # Compute absolute derivative
        tr_diff = np.abs(np.diff(tr_chunk, prepend=tr_chunk[0]))
        
        # Adaptive threshold based on this chunk
        threshold = np.percentile(tr_diff, DERIVATIVE_PERCENTILE)
        
        # Find peaks with minimum separation
        min_distance = int(MIN_ARTIFACT_SEPARATION_MS / 1000.0 * fs_br)
        peaks_local, _ = find_peaks(tr_diff, height=threshold, distance=min_distance)
        
        # Convert to global sample indices
        peaks_global = peaks_local + start_frame
        all_peaks.extend(peaks_global.tolist())
    
    artifact_samples = np.array(all_peaks, dtype=np.int64)
    
    if verbose:
        print(f"[detect_br] Found {len(artifact_samples)} artifacts")
        print(f"[detect_br] First 10: {artifact_samples[:10]}")
        print(f"[detect_br] Last 10: {artifact_samples[-10:]}")
    
    return artifact_samples


def load_intan_stimulus_times(stim_npz_path, verbose=True):
    """
    Load all stimulus pulse times detected on Intan system.
    
    Parameters
    ----------
    stim_npz_path : Path
        Path to stim_stream.npz from Intan detection
    verbose : bool
        Print info
        
    Returns
    -------
    pulse_times : ndarray
        All pulse times in Intan samples (pulse onsets)
    fs_intan : float
        Intan sampling frequency
    """
    if verbose:
        print(f"[load_intan] Loading {stim_npz_path.name}...")
    
    stim = np.load(stim_npz_path, allow_pickle=True)
    
    # --- Sampling frequency ---
    # meta is stored as json.dumps(dict) → always a JSON string
    fs_intan = 30000.0  # Default
    if 'meta' in stim:
        raw_meta = stim['meta']
        try:
            meta_str = raw_meta.item() if hasattr(raw_meta, 'item') else raw_meta
            meta_dict = json.loads(str(meta_str))
            if isinstance(meta_dict, dict):
                # extract_stim_npz stores the key as 'fs_hz'
                fs_intan = float(meta_dict.get('fs_hz',
                                  meta_dict.get('fs_intan', 30000.0)))
                if verbose:
                    print(f"[load_intan] fs_intan from meta: {fs_intan} Hz")
        except Exception as exc:
            if verbose:
                print(f"[load_intan] WARNING: Could not parse meta ({exc}), "
                      f"using default fs_intan = {fs_intan} Hz")
    # Top-level key overrides meta
    if 'fs_intan' in stim:
        fs_intan = float(stim['fs_intan'])

    # --- Pulse onset times ---
    # Prefer trigger_pairs[:, 0] – individual pulse ONSET samples (most precise).
    # block_bounds_samples[:, 0:1].flatten() gives only block START times.
    # DO NOT use block_bounds_samples.flatten() – that mixes starts and ends.
    if 'trigger_pairs' in stim:
        tp = stim['trigger_pairs']
        if tp.ndim == 2 and tp.shape[0] > 0 and tp.shape[1] >= 1:
            pulse_times = tp[:, 0].astype(np.int64)  # pulse onset samples
            if verbose:
                print(f"[load_intan] Using trigger_pairs onset times "
                      f"({len(pulse_times)} individual pulses)")
        else:
            pulse_times = np.array([], dtype=np.int64)
    elif 'pulse_times_samples' in stim:
        pulse_times = stim['pulse_times_samples'].astype(np.int64)
        if verbose:
            print(f"[load_intan] Using pulse_times_samples ({len(pulse_times)} pulses)")
    elif 'block_bounds_samples' in stim:
        block_bounds = stim['block_bounds_samples']
        # Use block START columns only – not flatten() which mixes starts + ends
        if block_bounds.ndim == 2 and block_bounds.shape[0] > 0:
            pulse_times = block_bounds[:, 0].astype(np.int64)
        else:
            pulse_times = np.array([], dtype=np.int64)
        if verbose:
            print(f"[load_intan] WARNING: No trigger_pairs found. "
                  f"Falling back to block START times ({len(pulse_times)} pulses). "
                  f"Temporal matching will be coarser.")
    else:
        raise KeyError("No pulse timing information found in stim_stream.npz")
    
    pulse_times = np.sort(pulse_times.flatten())
    
    if verbose:
        print(f"[load_intan] Loaded {len(pulse_times)} pulses at {fs_intan} Hz")
        if len(pulse_times) > 0:
            print(f"[load_intan] Duration: {pulse_times[-1]/fs_intan:.1f}s")
    
    return pulse_times, fs_intan

def pair_intan_br_pulses(intan_times, br_times, fs_intan, fs_br,
                          offset_intan_samples=0,
                          tolerance_samples=PAIRING_TOLERANCE_SAMPLES, verbose=True):
    """
    Match Intan pulse times to Blackrock pulse times.

    Converts each Intan pulse sample into the expected BR sample space using:
        t_br_expected = (t_intan - offset_intan_samples) * (fs_br / fs_intan)

    where offset_intan_samples is the Intan sample index at which the BR
    recording started (from br_to_intan_shifts.csv, 'shift_sample' column).

    Parameters
    ----------
    intan_times : ndarray
        Intan pulse onset times (samples)
    br_times : ndarray
        BR artifact times (samples)
    fs_intan, fs_br : float
        Sampling frequencies
    offset_intan_samples : float
        Intan sample index corresponding to BR t=0 (i.e. the shift_sample
        from compute_br_to_intan_shifts.py).  Default 0 (no offset).
    tolerance_samples : int
        Maximum distance in BR samples to consider a match
    verbose : bool

    Returns
    -------
    pairs_intan : ndarray
        Matched Intan times
    pairs_br : ndarray
        Matched BR times
    """
    if verbose:
        print(f"[pair] Pairing {len(intan_times)} Intan pulses with {len(br_times)} BR pulses...")
        print(f"[pair] Intan offset = {offset_intan_samples:.0f} samples "
              f"({offset_intan_samples/fs_intan:.3f} s), "
              f"tolerance = ±{tolerance_samples} BR samples")

    # Nominal sample-rate ratio (accounts for any hardware rate difference)
    scale_init = fs_br / fs_intan

    pairs_intan = []
    pairs_br    = []

    for t_i in intan_times:
        # Map Intan sample to expected BR sample
        t_br_expected = (t_i - offset_intan_samples) * scale_init

        # Skip pulses that fall before the BR recording started
        if t_br_expected < -tolerance_samples:
            continue

        # Find nearest BR artifact within tolerance
        candidates = br_times[np.abs(br_times - t_br_expected) < tolerance_samples]
        if len(candidates) > 0:
            best_match = candidates[np.argmin(np.abs(candidates - t_br_expected))]
            pairs_intan.append(t_i)
            pairs_br.append(best_match)

    pairs_intan = np.array(pairs_intan, dtype=np.float64)
    pairs_br    = np.array(pairs_br,    dtype=np.float64)

    if verbose:
        match_rate = 100.0 * len(pairs_intan) / max(len(intan_times), 1)
        print(f"[pair] Matched {len(pairs_intan)} pairs ({match_rate:.1f}% of Intan pulses)")
        if match_rate < 80.0:
            print(f"[pair] WARNING: Low match rate! Check offset / tolerance / artifact detection.")

    return pairs_intan, pairs_br


def fit_time_transform(pairs_intan, pairs_br, fs_intan, fs_br, verbose=True):
    """
    Fit linear time transformation: t_br = offset + scale * t_intan
    
    Parameters
    ----------
    pairs_intan : ndarray
        Matched Intan pulse times (samples)
    pairs_br : ndarray
        Matched BR pulse times (samples)
    fs_intan, fs_br : float
        Sampling frequencies
    verbose : bool
        
    Returns
    -------
    offset : float
        Time offset in BR samples
    scale : float
        True sampling rate ratio (BR/Intan)
    residual_std : float
        Standard deviation of fit residuals (BR samples)
    residuals : ndarray
        Per-pulse residuals (for diagnostics)
    """
    if verbose:
        print(f"[fit] Fitting linear transform with {len(pairs_intan)} pairs...")
    
    # Linear model
    def linear_model(t_intan, offset, scale):
        return offset + scale * t_intan
    
    # Initial guess
    scale_init = fs_br / fs_intan
    offset_init = np.median(pairs_br - scale_init * pairs_intan)
    
    # Fit
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
    
    # Calculate drift in parts-per-million
    drift_ppm = (scale - scale_init) * 1e6
    
    if verbose:
        print(f"[fit] ===== Calibration Results =====")
        print(f"[fit] Offset     = {offset:+.1f} samples ({offset/fs_br*1000:+.3f} ms)")
        print(f"[fit] Scale      = {scale:.12f}")
        print(f"[fit] Nominal    = {scale_init:.12f}")
        print(f"[fit] Drift      = {drift_ppm:+.3f} ppm")
        print(f"[fit] Residual std = {residual_std:.4f} samples ({residual_std/fs_br*1000:.5f} ms)")
        print(f"[fit] Residual RMS = {residual_rms:.4f} samples ({residual_rms/fs_br*1000:.5f} ms)")
        print(f"[fit] Max error  = {np.max(np.abs(residuals)):.2f} samples ({np.max(np.abs(residuals))/fs_br*1000:.4f} ms)")
    
    return offset, scale, residual_std, residuals


def plot_calibration_diagnostics(pairs_intan, pairs_br, offset, scale, residuals, 
                                  fs_intan, fs_br, br_idx, output_path):
    """
    Create diagnostic plots showing calibration quality.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Helper function
    def intan_to_br(t): return offset + scale * t
    scale_nominal = fs_br / fs_intan
    
    # Time axis in seconds
    time_sec = pairs_intan / fs_intan
    
    # ===== Plot 1: Raw alignment error (before calibration) =====
    ax1 = fig.add_subplot(gs[0, 0])
    error_before = (pairs_br - pairs_intan * scale_nominal) / fs_br * 1000
    ax1.scatter(time_sec, error_before, s=1, alpha=0.5, c='red')
    ax1.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Error (ms)")
    ax1.set_title(f"Before Calibration (σ = {np.std(error_before):.4f} ms)")
    ax1.grid(True, alpha=0.3)
    
    # ===== Plot 2: Residuals after calibration =====
    ax2 = fig.add_subplot(gs[0, 1])
    residuals_ms = residuals / fs_br * 1000
    ax2.scatter(time_sec, residuals_ms, s=1, alpha=0.5, c='blue')
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Residual (ms)")
    ax2.set_title(f"After Calibration (σ = {np.std(residuals_ms):.5f} ms)")
    ax2.grid(True, alpha=0.3)
    
    # ===== Plot 3: Histogram of residuals =====
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(residuals_ms, bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(0, color='r', linestyle='--', linewidth=1)
    ax3.set_xlabel("Residual (ms)")
    ax3.set_ylabel("Count")
    ax3.set_title(f"Residual Distribution (μ={np.mean(residuals_ms):.5f} ms)")
    ax3.grid(True, alpha=0.3)
    
    # ===== Plot 4: Cumulative drift =====
    ax4 = fig.add_subplot(gs[1, 1])
    # Show how error accumulates over time with uncorrected vs corrected
    cumulative_before = error_before
    cumulative_after = residuals_ms
    
    ax4.plot(time_sec, cumulative_before, 'r-', alpha=0.6, linewidth=1, label='Before (linear drift)')
    ax4.plot(time_sec, cumulative_after, 'b-', alpha=0.6, linewidth=1, label='After (random jitter)')
    ax4.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Error (ms)")
    ax4.set_title("Drift Over Time")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ===== Plot 5: Inter-pulse interval stability =====
    ax5 = fig.add_subplot(gs[2, 0])
    isi_intan = np.diff(pairs_intan) / fs_intan * 1000  # ms
    isi_br = np.diff(pairs_br) / fs_br * 1000
    isi_predicted = np.diff(intan_to_br(pairs_intan)) / fs_br * 1000
    
    ax5.scatter(time_sec[:-1], isi_intan, s=1, alpha=0.5, label='Intan', c='green')
    ax5.scatter(time_sec[:-1], isi_br, s=1, alpha=0.5, label='BR (raw)', c='red')
    ax5.scatter(time_sec[:-1], isi_predicted, s=1, alpha=0.5, label='BR (predicted)', c='blue')
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Inter-pulse interval (ms)")
    ax5.set_title("Pulse Timing Stability")
    ax5.legend(markerscale=5)
    ax5.grid(True, alpha=0.3)
    
    # ===== Plot 6: Residuals vs pulse number =====
    ax6 = fig.add_subplot(gs[2, 1])
    pulse_numbers = np.arange(len(residuals_ms))
    ax6.scatter(pulse_numbers, residuals_ms, s=1, alpha=0.5)
    ax6.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax6.set_xlabel("Pulse number")
    ax6.set_ylabel("Residual (ms)")
    ax6.set_title("Residuals vs Pulse Index")
    ax6.grid(True, alpha=0.3)
    
    fig.suptitle(f"Time Calibration Diagnostics: BR Session {br_idx:03d}", fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[plot] Saved diagnostic figure: {output_path}")


def _load_intan_offset(br_idx, fs_intan, verbose=True):
    """
    Read the Intan sample offset for this BR session from br_to_intan_shifts.csv.

    Returns
    -------
    offset_intan_samples : float
        Intan sample index that corresponds to BR t=0 (shift_sample column).
        Returns 0.0 if the file or entry is not found.
    """
    shifts_csv = METADATA_ROOT / "br_to_intan_shifts.csv"
    if not shifts_csv.exists():
        if verbose:
            print(f"[offset] br_to_intan_shifts.csv not found; using offset = 0")
        return 0.0

    br2shift = rcp.get_metadata_mapping(shifts_csv, "br_idx", "shift_sample")
    raw = br2shift.get(br_idx)
    if raw is None:
        if verbose:
            print(f"[offset] BR {br_idx:03d} not in shifts CSV; using offset = 0")
        return 0.0

    try:
        val = float(str(raw).strip())
    except (ValueError, TypeError):
        if verbose:
            print(f"[offset] Could not parse shift_sample for BR {br_idx:03d}; using offset = 0")
        return 0.0

    if verbose:
        print(f"[offset] Loaded Intan offset = {val:.0f} samples "
              f"({val/fs_intan:.3f} s) for BR {br_idx:03d}")
    return val


def calibrate_session(br_idx, sess_folder, stim_npz_path, output_dir, fig_dir):
    """
    Calibrate timing for a single BR session.
    
    Parameters
    ----------
    br_idx : int
        Blackrock session index
    sess_folder : Path
        Path to BR session folder containing .ns6
    stim_npz_path : Path
        Path to Intan stim_stream.npz
    output_dir : Path
        Where to save calibration .npz
    fig_dir : Path
        Where to save diagnostic plots
        
    Returns
    -------
    success : bool
        Whether calibration succeeded
    """
    print(f"\n{'='*60}")
    print(f"Calibrating BR session {br_idx:03d}: {sess_folder.name}")
    print(f"{'='*60}")
    
    try:
        # Load Intan stimulus times
        intan_pulse_times, fs_intan = load_intan_stimulus_times(stim_npz_path)

        # Load Intan-to-BR temporal offset from pre-computed shifts CSV
        offset_intan_samples = _load_intan_offset(br_idx, fs_intan)
        
        # Load Blackrock recording
        print(f"[load] Loading Blackrock NS6: {sess_folder.name}...")
        rec_ns6 = se.read_blackrock(sess_folder, stream_name='nsx6')
        fs_br = rec_ns6.get_sampling_frequency()
        
        # Apply UA mapping to get channel info
        XLS = rcp.ua_excel_path(REPO_ROOT, PARAMS.probes)
        UA_MAP = rcp.load_UA_mapping_from_excel(XLS)
        rec_ns6, idx_rows, ua_elec, ua_nsp, ua_region, ua_region_names, ua_port = \
            rcp.apply_ua_mapping_with_regions(rec_ns6, UA_MAP, br_idx, METADATA_CSV)
        
        # Select reference channel (electrode 37 or 29 depending on port)
        ref_elec = 37 if ua_port == 'B' else 29
        ref_matches = np.where(ua_elec == ref_elec)[0]
        if ref_matches.size == 0:
            print(f"[ERROR] Reference electrode {ref_elec} not found!")
            return False
        
        ref_ch_idx = ref_matches[0]
        ref_ch_name = rec_ns6.get_channel_ids()[ref_ch_idx]
        print(f"[detect_br] Using reference electrode {ref_elec} (channel {ref_ch_name})")
        
        # Detect artifacts on BR
        br_pulse_times = detect_all_artifacts_br(rec_ns6, ref_ch_name, fs_br)
        
        # Pair Intan and BR pulses, accounting for recording start offset
        pairs_intan, pairs_br = pair_intan_br_pulses(
            intan_pulse_times, br_pulse_times, fs_intan, fs_br,
            offset_intan_samples=offset_intan_samples
        )

        # Minimum pairs check — use at least MIN_MATCH_PAIRS and at least
        # 10% of the available Intan pulses (whichever is larger)
        min_required = max(MIN_MATCH_PAIRS, int(0.10 * len(intan_pulse_times)))
        if len(pairs_intan) < min_required:
            print(f"[ERROR] Too few matched pairs ({len(pairs_intan)} < {min_required}). "
                  f"Cannot calibrate. "
                  f"Hint: check that br_to_intan_shifts.csv has an entry for BR {br_idx:03d} "
                  f"or that stimulus artifacts are visible on channel {ref_ch_name}.")
            return False
        
        # Fit time transform
        offset, scale, residual_std, residuals = fit_time_transform(
            pairs_intan, pairs_br, fs_intan, fs_br
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
            n_pairs=len(pairs_intan),
            pairs_intan=pairs_intan,
            pairs_br=pairs_br,
            residuals=residuals,
            ref_electrode=ref_elec,
            br_idx=br_idx,
            session_name=sess_folder.name
        )
        print(f"[save] Saved calibration: {calib_path}")
        
        # Plot diagnostics
        fig_path = fig_dir / f"br{br_idx:03d}_time_calibration.png"
        plot_calibration_diagnostics(
            pairs_intan, pairs_br, offset, scale, residuals,
            fs_intan, fs_br, br_idx, fig_path
        )
        
        print(f"[SUCCESS] Calibration complete for BR {br_idx:03d}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Calibration failed for BR {br_idx:03d}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main calibration loop: process all BR sessions with matching Intan data.
    """
    print("="*60)
    print("Intan-Blackrock Time Calibration")
    print("="*60)
    
    # Load BR sessions using same method as UA_BR_analysis_mf.py
    BR_SESSION_FOLDERS = rcp.list_br_sessions(BR_ROOT)
    print(f"[INFO] Found {len(BR_SESSION_FOLDERS)} BR session folders")
    
    # Filter to PROCESS_ONLY BR indices if specified
    if PROCESS_ONLY:
        original_count = len(BR_SESSION_FOLDERS)
        filtered = []
        for br_idx in PROCESS_ONLY:
            match = [s for s in BR_SESSION_FOLDERS if int(s.name.split("_")[-1]) == br_idx]
            if not match:
                print(f"[WARN] PROCESS_ONLY BR index {br_idx} not found in session folders; skipping.")
                continue
            filtered.extend(match)
        print(f"[INFO] PROCESS_ONLY: {len(filtered)}/{original_count} sessions selected.")
        BR_SESSION_FOLDERS = filtered
    
    # Load metadata mapping to find corresponding Intan sessions
    intan2br = rcp.get_metadata_mapping(METADATA_CSV, "Intan_File", "BR_File")
    br2intan = {int(v): k for k, v in intan2br.items()}  # Reverse mapping
    
    # Get Intan session names
    rate_files = sorted(NPRW_CKPT_ROOT.rglob("rates__*.npz"))
    intan_sessions = sorted({p.stem[len("rates__"):].split("__bin", 1)[0] for p in rate_files})
    _, intan_idx_to_sess = rcp.build_session_index_map(intan_sessions)
    
    # Process each BR session
    n_success = 0
    n_fail = 0
    
    for sess_folder in BR_SESSION_FOLDERS:
        br_idx = int(sess_folder.name.split('_')[-1])
        
        # Find corresponding Intan session
        intan_idx = br2intan.get(br_idx)
        if intan_idx is None:
            print(f"[SKIP] No Intan mapping for BR {br_idx}")
            n_fail += 1
            continue
        
        # Find Intan session name
        intan_session_name = intan_idx_to_sess.get(intan_idx)
        if intan_session_name is None:
            print(f"[SKIP] No Intan session name for index {intan_idx}")
            n_fail += 1
            continue
        
        # Find stim_stream.npz
        stim_npz_path = STIM_NPZ_ROOT / f"{intan_session_name}_Intan_streams" / "stim_stream.npz"
        if not stim_npz_path.exists():
            print(f"[SKIP] No stim_stream.npz for {intan_session_name}")
            print(f"       Looked at: {stim_npz_path}")
            n_fail += 1
            continue
        
        # Calibrate this session
        success = calibrate_session(br_idx, sess_folder, stim_npz_path, OUTPUT_DIR, FIG_DIR)
        
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
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Figures: {FIG_DIR}")
    
    # Write summary CSV
    summary_csv = OUTPUT_DIR / "calibration_summary.csv"
    calib_files = sorted(OUTPUT_DIR.glob("br*_time_calib.npz"))
    
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["br_idx", "session_name", "offset_samples", "offset_ms", 
                        "scale", "drift_ppm", "residual_std_samples", "residual_std_ms", 
                        "n_pairs", "fs_intan", "fs_br"])
        
        for calib_file in calib_files:
            calib = np.load(calib_file)
            br_idx = int(calib['br_idx'])
            offset = float(calib['offset'])
            scale = float(calib['scale'])
            residual_std = float(calib['residual_std'])
            fs_intan = float(calib['fs_intan'])
            fs_br = float(calib['fs_br'])
            n_pairs = int(calib['n_pairs'])
            session_name = str(calib['session_name'])
            
            offset_ms = offset / fs_br * 1000
            drift_ppm = (scale - fs_br/fs_intan) * 1e6
            residual_std_ms = residual_std / fs_br * 1000
            
            writer.writerow([
                br_idx, session_name, offset, offset_ms, scale, drift_ppm,
                residual_std, residual_std_ms, n_pairs, fs_intan, fs_br
            ])
    
    print(f"Summary CSV: {summary_csv}")
    print("\nDone!")


if __name__ == "__main__":
    main()