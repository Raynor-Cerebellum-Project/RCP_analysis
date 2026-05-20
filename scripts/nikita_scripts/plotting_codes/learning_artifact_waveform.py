"""
learn_artifact_template.py

Extract artifact waveforms from a known-good session/channel and save the template.
Uses the same data loading infrastructure as debug_IPCA_artifact_correction.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from scipy.io import loadmat
from probeinterface import Probe

import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

# ============================================================
# CONFIGURATION
# ============================================================
CONDITION = 17  # Blackrock block index with 40 pulses at 400Hz
PROBE = "UA"    # "UA" or "NPRW"
ARTIFACT_CHANNEL_ELEC_ID = 37  # Center electrode for pulse detection
N_ARTIFACTS_TO_AVERAGE = 200
TEMPLATE_WINDOW_MS = (-0.5, 0.5)
OUTPUT_NAME = "artifact_template_UA_PortB"

# ============================================================
# Session-level paths (same as debug script)
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[3]  # Adjust based on script location
PARAMS = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE = SESSION_LOC / "results"
ALIGNED_CKPT = OUT_BASE / "checkpoints" / "Aligned"
INTAN_ROOT = SESSION_LOC / "Intan"
OUTPUT_PATH = REPO_ROOT / "config"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def find_aligned_file(aligned_dir: Path, br_idx: int) -> Path:
    """Locate an aligned .npz across standard checkpoint subfolders."""
    pattern = f"*__BR_{br_idx:03d}.npz"
    search_dirs = [
        aligned_dir / "stim_reaches",
        aligned_dir / "control_reaches",
        aligned_dir / "continuous_stim",
        aligned_dir / "at_rest",
        aligned_dir / "Grasp",
        aligned_dir / "IMU",
        aligned_dir,
    ]
    for d in search_dirs:
        if d.exists():
            hits = sorted(d.glob(pattern))
            if hits:
                return hits[-1]
    raise FileNotFoundError(f"No aligned npz matching {pattern} in {aligned_dir}")


def find_intan_folder(intan_root: Path, intan_filename: str) -> Path:
    """Resolve the Intan recording folder by exact or partial name match."""
    exact = intan_root / intan_filename
    if exact.exists() and exact.is_dir():
        return exact
    for query in [intan_filename, "_".join(intan_filename.split("_")[:2])]:
        matches = sorted(intan_root.glob(f"*{query}*"))
        if matches:
            return matches[-1]
    raise FileNotFoundError(f"Could not find Intan folder for {intan_filename}")


def load_probe_geometry(geom_path):
    """Load the Neuropixels channel map from the session's .mat file."""
    mat = loadmat(str(geom_path))
    idx_monitor = mat["chanMap0ind"].ravel()
    probe = Probe(ndim=2)
    probe.set_contacts(
        positions=np.c_[mat["xcoords"].ravel(), mat["ycoords"].ravel()],
        shapes="square",
        shape_params={"width": 12.0},
    )
    probe.set_device_channel_indices(idx_monitor)
    return probe, idx_monitor


def _find_first_significant_peak(signal_chunk: np.ndarray) -> int:
    """Return the index of the earliest significant deflection in a chunk."""
    centered = np.abs(signal_chunk - np.mean(signal_chunk))
    threshold = 0.4 * np.max(centered)
    first_crossing = int(np.argmax(centered > threshold))
    lo = max(0, first_crossing - 2)
    hi = min(len(centered), first_crossing + 3)
    return lo + int(np.argmax(centered[lo:hi]))


def main():
    print(f"Learning artifact template | Condition: {CONDITION} | Probe: {PROBE}")
    
    # ─────────────────────────────────────────────────────────────────
    # §1  Load alignment metadata and stim detection (same as debug script)
    # ─────────────────────────────────────────────────────────────────
    aligned_path = find_aligned_file(ALIGNED_CKPT, CONDITION)
    z = np.load(aligned_path, allow_pickle=True)
    if "align_meta" in z.files:
        meta_raw = z["align_meta"].item()
        if isinstance(meta_raw, dict):
            meta = meta_raw
        else:
            meta = json.loads(meta_raw)
    else:
        meta = {}
    
    intan_filename = meta.get("intan_filename", "")
    intan_folder = find_intan_folder(INTAN_ROOT, intan_filename)
    sess_name = intan_folder.name
    
    print(f"Session: {sess_name}")
    
    # Load stim detection
    fs_intan = 30000.0
    shift_samp_intan = 0.0
    
    if PROBE == "NPRW":
        stim_npz_path = NPRW_AUX_DATA / f"{sess_name}_Intan_streams" / "stim_stream.npz"
    else:
        stim_npz_path, _ = rcp.stim_npz_path_from_br_idx(CONDITION, METADATA_CSV, NPRW_AUX_DATA)
        shifts_csv = METADATA_ROOT / "br_to_intan_shifts.csv"
        if shifts_csv.exists():
            br2fs_intan = rcp.get_metadata_mapping(shifts_csv, "br_idx", "fs_intan")
            br2shift = rcp.get_metadata_mapping(shifts_csv, "br_idx", "shift_sample")
            fs_intan = float(br2fs_intan.get(CONDITION, 30000.0))
            shift_raw = br2shift.get(CONDITION, None)
            if shift_raw is not None and not (isinstance(shift_raw, str) and not shift_raw.strip()):
                shift_samp_intan = float(shift_raw)
    
    if not stim_npz_path or not stim_npz_path.exists():
        raise FileNotFoundError(f"stim_stream.npz not found: {stim_npz_path}")
    
    stim = rcp.load_stim_detection(stim_npz_path)
    block_bounds = stim.get("block_bounds_samples", np.empty((0, 2), dtype=np.int64))
    
    if len(block_bounds) == 0:
        raise ValueError("No stimulation blocks found.")
    
    if isinstance(block_bounds, list):
        block_bounds = np.array(block_bounds, dtype=np.int64)
    
    nprw_samps = block_bounds[:, 0].astype(np.int64)
    nprw_ends = block_bounds[:, 1].astype(np.int64)
    
    print(f"Found {len(nprw_samps)} stimulation blocks")
    
    # ─────────────────────────────────────────────────────────────────
    # §2  Load recording (same as debug script)
    # ─────────────────────────────────────────────────────────────────
    if PROBE == "NPRW":
        geom_path = rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
        stream_name = PARAMS.probes.get("NPRW").get("neural_data_stream", "amplifier_data")
        rec_raw = se.read_split_intan_files(
            intan_folder, mode="concatenate",
            stream_name=stream_name, use_names_as_ids=True,
        )
        rec_raw = spre.unsigned_to_signed(rec_raw)
        probe_obj, map_idx = load_probe_geometry(geom_path)
        rec_raw = rec_raw.set_probe(probe_obj)
        rec = rcp.reorder_recording_to_geometry(rec_raw, map_idx)
        fs = float(rec.get_sampling_frequency())
        
        # For NPRW, use channel index directly
        target_ch_id = rec.get_channel_ids()[ARTIFACT_CHANNEL_ELEC_ID]
        
    elif PROBE == "UA":
        br_sess = (z["ua_meta"].item().get("session")
                   if "ua_meta" in z.files else meta.get("br_session"))
        if not br_sess:
            raise ValueError("No UA session found in metadata")
        
        rec_raw = se.read_blackrock(
            BR_ROOT / br_sess, stream_name="nsx6", all_annotations=True,
        )
        ua_map = rcp.load_UA_mapping_from_excel(
            rcp.ua_excel_path(REPO_ROOT, PARAMS.probes)
        )
        (rec, _, ua_elec, _, ua_region, ua_region_names, ua_port) = rcp.apply_ua_mapping_with_regions(
            rec_raw, ua_map, CONDITION, METADATA_CSV,
        )
        fs = rec.get_sampling_frequency()
        
        # Find the target channel
        all_ch_ids = rec.get_channel_ids()
        matches = np.where(ua_elec == ARTIFACT_CHANNEL_ELEC_ID)[0]
        if matches.size == 0:
            raise ValueError(f"Electrode {ARTIFACT_CHANNEL_ELEC_ID} not found in mapping")
        target_ch_id = all_ch_ids[matches[0]]
        
        # Transform stim times to UA sample space
        scale = fs / fs_intan
        nprw_samps = np.round((nprw_samps - shift_samp_intan) * scale).astype(np.int64)
        nprw_ends = np.round((nprw_ends - shift_samp_intan) * scale).astype(np.int64)
    
    print(f"Recording loaded. Fs = {fs} Hz")
    print(f"Using channel: {target_ch_id}")
    
    # ─────────────────────────────────────────────────────────────────
    # §3  Get stim parameters from metadata
    # ─────────────────────────────────────────────────────────────────
    try:
        stim_freq = float(rcp.get_metadata_mapping(
            METADATA_CSV, 'BR_File', 'Stim_Frequency_Hz'
        ).get(CONDITION, 400.0))
        if np.isnan(stim_freq):
            stim_freq = 400.0
    except Exception:
        stim_freq = 400.0
    
    pulse_interval_ms = 1000.0 / stim_freq
    interval_samp = int(pulse_interval_ms / 1000 * fs)
    
    print(f"Stim frequency: {stim_freq} Hz (interval: {pulse_interval_ms:.2f} ms)")
    
    # ─────────────────────────────────────────────────────────────────
    # §4  Extract artifact waveforms using same alignment as debug script
    # ─────────────────────────────────────────────────────────────────
    mw_start = int(TEMPLATE_WINDOW_MS[0] / 1000 * fs)
    mw_end = int(TEMPLATE_WINDOW_MS[1] / 1000 * fs)
    window_size = mw_end - mw_start
    
    refine_samp = max(int(0.15 / 1000 * fs), 3)
    coarse_radius = max(refine_samp, 3)
    search_radius_samp = int((pulse_interval_ms * 0.4) / 1000 * fs)
    
    n_total = rec.get_num_samples()
    artifact_waveforms = []
    pulse_centres_global = []
    
    print(f"Extracting artifacts from {len(nprw_samps)} blocks...")
    
    for block_idx, (st, en) in enumerate(zip(nprw_samps, nprw_ends)):
        if len(artifact_waveforms) >= N_ARTIFACTS_TO_AVERAGE:
            break
        
        # Coarse search for first pulse (same logic as debug script)
        search_margin = int(20.0 / 1000 * fs)
        start_search = max(0, st - search_margin)
        end_search = min(n_total, en + search_margin)
        
        if end_search <= start_search:
            continue
        
        tr_search = rec.get_traces(
            start_frame=start_search, end_frame=end_search,
            channel_ids=[target_ch_id], return_in_uV=True
        )[:, 0]
        
        tr_diff_search = np.abs(np.diff(tr_search, prepend=tr_search[0]))
        
        predicted_centre = st - start_search
        slop = int(5.0 / 1000 * fs)
        win_lo = max(0, predicted_centre - slop)
        win_hi = min(len(tr_search), predicted_centre + slop)
        
        if win_hi <= win_lo:
            continue
        
        peak_in_win = np.argmax(tr_diff_search[win_lo:win_hi])
        anchor_idx = win_lo + peak_in_win
        max_diff = tr_diff_search[anchor_idx]
        
        if max_diff < 5.0:
            continue
        
        # Find first threshold crossing
        first_pulse_in_search = None
        for k in range(win_lo, win_hi):
            if tr_diff_search[k] > 0.4 * max_diff:
                first_pulse_in_search = k
                break
        if first_pulse_in_search is None:
            first_pulse_in_search = anchor_idx
        
        # Refine to earliest significant deflection
        r_lo = max(0, first_pulse_in_search - coarse_radius)
        r_hi = min(len(tr_search), first_pulse_in_search + coarse_radius + 1)
        coarse_centre = r_lo + _find_first_significant_peak(tr_search[r_lo:r_hi])
        
        true_first_pulse = start_search + coarse_centre
        
        # Calculate number of pulses in this block
        stim_dur_ms = (en - st) * 1000.0 / fs
        num_pulses = int(np.floor(stim_dur_ms / pulse_interval_ms)) + 1
        
        # Forward-walk to find all pulses (same as debug script)
        pulse_centres = [true_first_pulse]
        curr = true_first_pulse
        local_max_diff = max_diff
        
        while len(pulse_centres) < num_pulses:
            nxt = curr + interval_samp
            s_lo = max(0, nxt - search_radius_samp)
            s_hi = min(n_total, nxt + search_radius_samp)
            
            if s_hi <= s_lo:
                curr = nxt
                pulse_centres.append(curr)
                continue
            
            tr_local = rec.get_traces(
                start_frame=s_lo, end_frame=s_hi,
                channel_ids=[target_ch_id], return_in_uV=True
            )[:, 0]
            
            tr_diff = np.abs(np.diff(tr_local, prepend=tr_local[0]))
            peak_idx = np.argmax(tr_diff)
            
            if tr_diff[peak_idx] < 0.15 * local_max_diff:
                curr = nxt
            else:
                curr = s_lo + peak_idx
            
            pulse_centres.append(curr)
        
        # Refine each pulse and extract waveform
        for p_idx in pulse_centres:
            if len(artifact_waveforms) >= N_ARTIFACTS_TO_AVERAGE:
                break
            
            r_lo = max(0, p_idx - refine_samp)
            r_hi = min(n_total, p_idx + refine_samp + 1)
            
            if r_lo >= r_hi:
                continue
            
            tr_refine = rec.get_traces(
                start_frame=r_lo, end_frame=r_hi,
                channel_ids=[target_ch_id], return_in_uV=True
            )[:, 0]
            
            tc = r_lo + _find_first_significant_peak(tr_refine)
            
            # Extract waveform
            ext_start = tc + mw_start
            ext_end = tc + mw_end
            
            if ext_start >= 0 and ext_end <= n_total:
                waveform = rec.get_traces(
                    start_frame=ext_start, end_frame=ext_end,
                    channel_ids=[target_ch_id], return_in_uV=True
                )[:, 0]
                
                artifact_waveforms.append(waveform)
                pulse_centres_global.append(tc)
    
    print(f"Collected {len(artifact_waveforms)} artifact waveforms")
    
    if len(artifact_waveforms) < 10:
        print("Too few artifacts collected. Check your settings.")
        return
    
    # ─────────────────────────────────────────────────────────────────
    # §5  Compute template
    # ─────────────────────────────────────────────────────────────────
    artifact_array = np.stack(artifact_waveforms)
    
    # Remove baseline from each
    artifact_array = artifact_array - np.median(artifact_array, axis=1, keepdims=True)
    
    # Compute mean template
    template_raw = np.mean(artifact_array, axis=0)
    template_norm = template_raw / np.linalg.norm(template_raw)
    
    print(f"Template shape: {template_norm.shape}")
    
    # ─────────────────────────────────────────────────────────────────
    # §6  Visualization
    # ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    time_axis = np.linspace(TEMPLATE_WINDOW_MS[0], TEMPLATE_WINDOW_MS[1], window_size)
    
    # Plot individual waveforms
    ax = axes[0, 0]
    for i in range(min(50, len(artifact_waveforms))):
        ax.plot(time_axis, artifact_array[i], alpha=0.3, color='gray')
    ax.plot(time_axis, template_raw, color='red', linewidth=2, label='Mean template')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'Individual artifacts (n={len(artifact_waveforms)})')
    ax.legend()
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    
    # Plot normalized template
    ax = axes[0, 1]
    ax.plot(time_axis, template_norm, color='blue', linewidth=2)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized amplitude')
    ax.set_title('Normalized template (for matched filtering)')
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    
    # Plot variance
    ax = axes[1, 0]
    std = np.std(artifact_array, axis=0)
    ax.fill_between(time_axis, template_raw - 2*std, template_raw + 2*std, alpha=0.3, color='blue')
    ax.plot(time_axis, template_raw, color='blue', linewidth=2)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Mean ± 2 STD')
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    
    # Plot example raw trace with detected artifacts
    ax = axes[1, 1]
    if len(pulse_centres_global) > 0:
        first_pulse = pulse_centres_global[0]
        example_start = max(0, first_pulse - int(10 / 1000 * fs))
        example_end = min(n_total, first_pulse + int(30 / 1000 * fs))
        
        example_trace = rec.get_traces(
            start_frame=example_start, end_frame=example_end,
            channel_ids=[target_ch_id], return_in_uV=True
        )[:, 0]
        
        example_time = (np.arange(len(example_trace)) + example_start - first_pulse) / fs * 1000
        ax.plot(example_time, example_trace, color='k')
        
        # Mark detected pulse centres
        for pc in pulse_centres_global[:20]:
            t_ms = (pc - first_pulse) / fs * 1000
            if example_time[0] < t_ms < example_time[-1]:
                ax.axvline(t_ms, color='r', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Raw trace with DETECTED artifact times')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = OUTPUT_PATH / f"{OUTPUT_NAME}_visualization.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved visualization: {fig_path}")
    plt.show()
    
    # ─────────────────────────────────────────────────────────────────
    # §7  Save template
    # ─────────────────────────────────────────────────────────────────
    template_path = OUTPUT_PATH / f"{OUTPUT_NAME}.npz"
    
    np.savez(
        template_path,
        template_raw=template_raw,
        template_norm=template_norm,
        fs=fs,
        window_ms=TEMPLATE_WINDOW_MS,
        n_artifacts_averaged=len(artifact_waveforms),
        source_condition=CONDITION,
        source_channel=str(target_ch_id),
        stim_freq_hz=stim_freq,
    )
    print(f"Saved template: {template_path}")
    
    # Also save normalized template as .npy
    np.save(OUTPUT_PATH / f"{OUTPUT_NAME}.npy", template_norm)
    print(f"Saved normalized template: {OUTPUT_PATH / OUTPUT_NAME}.npy")


if __name__ == "__main__":
    main()