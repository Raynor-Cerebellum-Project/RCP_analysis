import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
from probeinterface import Probe
from scipy.io import loadmat
from pathlib import Path

import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *
from RCP_analysis.python.functions.artifact_correction import IPCA_Artifact_Correction

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

# Setup Paths & Params
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE = SESSION_LOC / "results"
ALIGNED_CKPT = OUT_BASE / "checkpoints" / "Aligned"
INTAN_ROOT = SESSION_LOC / "Intan"
FIG_DIR = OUT_BASE / "figures" / "blanking_debug"
FIG_DIR.mkdir(parents=True, exist_ok=True)
GEOM_PATH = rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)

def find_aligned_file(aligned_dir: Path, br_idx: int) -> Path:
    # Look for the aligned npz file across common subfolders
    pat = f"*__BR_{br_idx:03d}.npz"
    for d in [aligned_dir / "stim_reaches", aligned_dir / "control_reaches", aligned_dir / "at_rest", aligned_dir]:
        if d.exists() and list(d.glob(pat)):
            return sorted(d.glob(pat))[-1]
    raise FileNotFoundError(f"No aligned npz matching {pat} in {aligned_dir}")

def find_intan_folder(intan_root: Path, intan_filename: str) -> Path:
    # Match the intan folder name
    cand = intan_root / intan_filename
    if cand.exists() and cand.is_dir():
        return cand
        
    for query in [intan_filename, "_".join(intan_filename.split("_")[:2])]:
        matches = sorted(intan_root.glob(f"*{query}*"))
        if matches: return matches[-1]
        
    raise FileNotFoundError(f"Could not find Intan folder for {intan_filename}")

def get_probe_geometry():
    mat_probe = loadmat(Path(GEOM_PATH))
    idx_monitor = mat_probe["chanMap0ind"].ravel()
    
    p = Probe(ndim=2)
    p.set_contacts(
        positions=np.c_[mat_probe["xcoords"].ravel(), mat_probe["ycoords"].ravel()], 
        shapes="square", 
        shape_params={"width": 12.0}
    )
    p.set_device_channel_indices(idx_monitor)
    return p, idx_monitor

def run_ipca_debug(
    condition=10, 
    target_ch=10, 
    ipca_rank=3, 
    window_ms=(-10.0, 50.0), 
    trials_plot=5,
    pulse_window_ms=(-0.5, 0.5),
    apply_hpf=False,
    debug_single_pulse=True,
    stim_freq_override=None
):
    print(f"Running IPCA Debug | Cond: {condition} | Ch: {target_ch} | Rank: {ipca_rank}")

    # Load alignment metadata
    aligned_path = find_aligned_file(ALIGNED_CKPT, condition)
    z = np.load(aligned_path, allow_pickle=True)
    meta = json.loads(z["align_meta"].item()) if "align_meta" in z.files else {}
    
    stim_ms = np.asarray(z.get("stim_ms", []), float).ravel()
    if not stim_ms.size:
        print("No stim events found.")
        return

    pmeta = z["nprw_meta"].item() if "nprw_meta" in z.files else {}
    rec_start_ms = float(pmeta.get("rec_start_ms_aligned", 0.0))
    stim_local_ms = stim_ms - rec_start_ms
    stim_dur = float(meta.get("recording_stim_dur", 0.0))

    # Load and prep raw recording
    intan_folder = find_intan_folder(INTAN_ROOT, meta.get("intan_filename", ""))
    stream_name = PARAMS.probes.get("NPRW").get("neural_data_stream", "amplifier_data")
    
    rec_raw = se.read_split_intan_files(intan_folder, mode="concatenate", stream_name=stream_name, use_names_as_ids=True)
    rec_raw = spre.unsigned_to_signed(rec_raw)
    
    probe_obj, map_idx = get_probe_geometry()
    rec_raw = rec_raw.set_probe(probe_obj)
    rec_proc = rcp.reorder_recording_to_geometry(rec_raw, map_idx)
    
    if apply_hpf:
        rec_proc = spre.highpass_filter(rec_proc, freq_min=300.0)

    fs = rec_proc.get_sampling_frequency()
    
    # Calculate pulse timing
    if stim_freq_override is not None:
        stim_freq = float(stim_freq_override)
    else:
        stim_freq = float(meta.get("Freq", 300.0))
        
    print(f"-> Stimulation Frequency (for extraction): {stim_freq} Hz")
    pulse_interval_ms = 1000.0 / stim_freq
    num_pulses = int(np.floor(stim_dur / pulse_interval_ms)) + 1
    
    if num_pulses <= 0:
        print("Stimulation duration is too short.")
        return

    # Convert windows to frames
    fw_start = int(window_ms[0] / 1000 * fs)
    fw_end = int(window_ms[1] / 1000 * fs)
    full_n_time = fw_end - fw_start
    
    mw_start = int(pulse_window_ms[0] / 1000 * fs)
    mw_end = int(pulse_window_ms[1] / 1000 * fs)
    micro_n_time = mw_end - mw_start
    
    # Filter valid stims that fully fit in the recording
    rec_samples = rec_proc.get_total_samples()
    valid_stims = [s for s in stim_local_ms if 0 <= int(s/1000*fs) + fw_start and int(s/1000*fs) + fw_end < rec_samples]
    
    ch_id = rec_proc.get_channel_ids()[target_ch]
    
    # Pre-allocate arrays
    full_raw_array = np.zeros((len(valid_stims), full_n_time), dtype=np.float32)
    micro_signal_array = np.zeros((len(valid_stims) * num_pulses, micro_n_time, 1), dtype=np.float32)
    
    pulse_counter = 0
    micro_map = [] # Tracks (trial_idx, start_frame, end_frame)
    
    # Extract data
    for i, s_ms in enumerate(valid_stims):
        center = int(s_ms / 1000 * fs)
        start, end = center + fw_start, center + fw_end
        
        tr = rec_proc.get_traces(start_frame=start, end_frame=end, channel_ids=[ch_id], return_in_uV=True).squeeze()
        full_raw_array[i, :] = tr
        
        for p in range(num_pulses):
            pulse_offset_ms = p * pulse_interval_ms
            # Theoretical start relative to the beginning of the full window
            theoretical_center_ms = pulse_offset_ms - window_ms[0]
            theo_center_idx = int(theoretical_center_ms / 1000 * fs)
            
            # Real pulse could be slightly off due to hardware clock differences. 
            # Search dynamically up to 40% of the interval away so we don't accidentally grab the NEXT pulse
            search_radius_idx = int((pulse_interval_ms * 0.4) / 1000 * fs)
            search_start = max(0, theo_center_idx - search_radius_idx)
            search_end = min(full_n_time, theo_center_idx + search_radius_idx)
            
            if search_end > search_start:
                # Find the maximum absolute amplitude in the search window
                search_window = tr[search_start:search_end]
                true_peak_offset = np.argmax(np.abs(search_window))
                pulse_center_idx = search_start + true_peak_offset
            else:
                pulse_center_idx = theo_center_idx  # Fallback
            
            p_start = pulse_center_idx + mw_start
            p_end = pulse_center_idx + mw_end
            
            if p_start >= 0 and p_end <= full_n_time:
                micro_signal_array[pulse_counter, :, 0] = tr[p_start:p_end]
                micro_map.append((i, p_start, p_end))
                pulse_counter += 1

    micro_signal_array = micro_signal_array[:pulse_counter]
    # Run IPCA
    corrector = IPCA_Artifact_Correction(rank=ipca_rank)
    micro_corrected, _ = corrector.ipca_all(micro_signal_array)
    
    # ── DEBUG VIZ: Single Pulse Window ──
    if debug_single_pulse and pulse_counter > 0:
        print("Saving debug plot for first single pulse...")
        fig_debug, axes_debug = plt.subplots(1, 2, figsize=(24, 12), sharey=True)
        
        # Plot 1: Full Trial Trace (zoomed out) highlighting the micro-window
        first_trial, p_start, p_end = micro_map[0]
        full_time_axis = np.arange(fw_start, fw_end) / fs * 1000
        axes_debug[0].plot(full_time_axis, full_raw_array[first_trial], color='k', lw=1)
        axes_debug[0].axvspan(full_time_axis[p_start - fw_start], full_time_axis[p_end - fw_start - 1], color='r', alpha=0.3, label='IPCA "Learning" Window')
        axes_debug[0].set_title(f"Trial {first_trial + 1} | Full Trace")
        axes_debug[0].set_xlabel("Time (ms)")
        axes_debug[0].set_ylabel("Amplitude (uV)")
        axes_debug[0].axvline(0, color='g', ls='--')
        axes_debug[0].legend()
        
        # Prepare micro-window signals for ALL pulses
        micro_time_axis = np.linspace(pulse_window_ms[0], pulse_window_ms[1], micro_n_time)
        all_raw_micro = micro_signal_array[:, :, 0]
        all_corr_micro = micro_corrected[:, :, 0]
        all_art_micro = all_raw_micro - all_corr_micro

        mean_raw = np.mean(all_raw_micro, axis=0)
        mean_art = np.mean(all_art_micro, axis=0)
        mean_corr = np.mean(all_corr_micro, axis=0)

        # Plot 2: Zoomed-in Micro-window trace (All Pulses COMBINED)
        for i in range(pulse_counter):
            axes_debug[1].plot(micro_time_axis, all_raw_micro[i], color='k', alpha=0.08)
            axes_debug[1].plot(micro_time_axis, all_art_micro[i], color='r', alpha=0.08, ls='--')
            axes_debug[1].plot(micro_time_axis, all_corr_micro[i], color='b', alpha=0.08)

        axes_debug[1].plot(micro_time_axis, mean_raw, color='k', lw=3, label='Mean Raw Pulse')
        axes_debug[1].plot(micro_time_axis, mean_art, color='r', lw=3, ls='--', label='Mean Learned Template')
        axes_debug[1].plot(micro_time_axis, mean_corr, color='cyan', lw=3, label='Mean Cleaned Signal')
        
        axes_debug[1].set_title("All Pulses (Raw + Template + Cleaned)")
        axes_debug[1].set_xlabel("Time around pulse center (ms)")
        axes_debug[1].axvline(0, color='k', ls=':', alpha=0.5, label='Estimated Center')
        axes_debug[1].legend()

        fig_debug.suptitle(f"IPCA All Pulses Extraction Debug | Ch {target_ch}", fontsize=16)
        fig_debug.tight_layout()
        viz_outname = FIG_DIR / f"debug_IPCA_Ch{target_ch}_cond{condition}_all_pulses_viz.png"
        fig_debug.savefig(viz_outname)
        plt.close(fig_debug)
        print(f"Saved: {viz_outname}")
    # ────────────────────────────────────
    
    # Reconstruct full traces
    full_corr_array = full_raw_array.copy()
    full_artifact_array = np.zeros_like(full_raw_array)
    
    for p_idx, (trial, p_start, p_end) in enumerate(micro_map):
        raw_micro = micro_signal_array[p_idx, :, 0]
        corr_micro = micro_corrected[p_idx, :, 0]
        full_corr_array[trial, p_start:p_end] = corr_micro
        full_artifact_array[trial, p_start:p_end] = raw_micro - corr_micro

    # Plotting
    plot_idxs = np.linspace(0, len(valid_stims) - 1, min(trials_plot, len(valid_stims)), dtype=int)
    time_axis = np.arange(fw_start, fw_end) / fs * 1000
    
    fig, axes = plt.subplots(len(plot_idxs), 3, figsize=(15, 3*len(plot_idxs)), sharex=True, sharey=True)
    if len(plot_idxs) == 1: axes = np.array([axes])
    
    cols = ["Raw Signal", "IPCA Artifact Template", "Corrected Signal"]

    for r, trial_idx in enumerate(plot_idxs):
        axes[r, 0].plot(time_axis, full_raw_array[trial_idx], color='k', lw=1)
        axes[r, 1].plot(time_axis, full_artifact_array[trial_idx], color='r', lw=1)
        axes[r, 2].plot(time_axis, full_corr_array[trial_idx], color='b', lw=1)
        
        for c in range(3):
            ax = axes[r, c]
            ax.axvline(0, color='g', ls='--')
            if stim_dur > 0: ax.axvline(stim_dur, color='g', ls=':', alpha=0.5)
            if r == 0: ax.set_title(cols[c])
            if c == 0: ax.set_ylabel(f"Trial {trial_idx+1}\n(uV)")

    axes[0,0].set_xlim(window_ms)
    fig.suptitle(f"IPCA Artifact Correction (Rank {ipca_rank}) | Ch {target_ch} | Cond {condition}", y=0.98)
    fig.tight_layout(rect=[0,0,1,0.95])
    
    outname = FIG_DIR / f"debug_IPCA_Ch{target_ch}_cond{condition}_rank{ipca_rank}.png"
    fig.savefig(outname)
    print(f"Saved: {outname}")
    plt.close(fig)

if __name__ == "__main__":
    run_ipca_debug(
        condition=10, 
        target_ch=10, 
        ipca_rank=3,
        window_ms=(-10.0, 50.0),
        pulse_window_ms=(-0.5, 0.5),
        apply_hpf=False,
        debug_single_pulse=True,
        stim_freq_override=400.0
    )

