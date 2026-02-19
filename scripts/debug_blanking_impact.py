"""
Debug Blanking Impact Script

This script allows you to load raw data for a specific condition/window, apply 
the same preprocessing steps as the main pipeline (High-pass -> CAR -> Artifact Removal), 
and visualize the traces at each stage. This is useful for tuning blanking 
parameters without re-running the entire batch pipeline.

Steps:
1. Load Aligned NPZ to get Intan filename and stim times.
2. Load Raw Intan recording.
3. Apply High-pass filter.
4. Apply Common Average Reference (CAR).
5. Apply Artifact Removal (Blanking) with configurable parameters.
6. Plot traces for selected channels/trials comparing processing stages.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
from probeinterface import Probe
from scipy.io import loadmat
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

# ──────────────────────────────────────────────────
# USER SETTINGS
# ──────────────────────────────────────────────────
CONDITION   = 12             # BR index
PROBE       = "NPRW"        # "NPRW" only for now (Intan raw data)
CHANNELS    = list(range(7, 16))   # 0-indexed channel rows to plot
WINDOW_MS   = (60.0, 140.0)      # Plotting window relative to stim
TRIAL_INDEX = 0                    # Which valid trial index to plot (or 'all')
Y_LIM       = (-200, 200)

# Blanking / Preprocessing Params (Override here to test)
Remove_Artifacts = True
MS_BEFORE = 20.0             # Blanking time before stim (ms)
MS_AFTER  = 20.0             # Blanking time after stim (ms) - relative to stim END
STIM_DUR_OVERRIDE = None     # Force stim dur (ms). If None, tries to read from metadata.

# Standard filtering params
HP_FREQ = 300.0
CAR_RADIUS = (30, 150)       # Inner, Outer radius for local CAR

# ──────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parents[1]
PARAMS      = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
SESSION     = getattr(PARAMS, "session", "UNKNOWN")
OUT_BASE    = SESSION_LOC / "results"

ALIGNED_CKPT = OUT_BASE / "checkpoints" / "Aligned"
INTAN_ROOT   = SESSION_LOC / "Intan"
FIG_DIR      = OUT_BASE / "figures" / "blanking_debug"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GEOM_PATH = rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)


def find_aligned_file(aligned_dir: Path, br_idx: int) -> Path:
    """Find the aligned .npz for a given BR index."""
    pat = f"*__BR_{br_idx:03d}.npz"
    search_dirs = [
        aligned_dir / "stim_reaches",
        aligned_dir / "control_reaches",
        aligned_dir / "at_rest",
        aligned_dir,
    ]
    for d in search_dirs:
        if d.exists():
            cands = sorted(d.glob(pat))
            if cands:
                return cands[-1]
    raise FileNotFoundError(f"No aligned npz matching {pat} in {aligned_dir}")


def find_intan_folder(intan_root: Path, intan_filename: str) -> Path:
    """Find the raw Intan session folder."""
    # Match by session name (e.g. NRR_RW012) and timestamp if possible
    # intan_filename example: "NRR_RW012_260116_145123"
    
    # Direct match first
    cand = intan_root / intan_filename
    if cand.exists() and cand.is_dir():
        return cand
        
    # Search glob
    matches = sorted(intan_root.glob(f"*{intan_filename}*"))
    if matches:
        return matches[-1]
        
    # Search parts
    parts = intan_filename.split("_")
    if len(parts) >= 2:
        prefix = "_".join(parts[:2])
        matches = sorted(intan_root.glob(f"*{prefix}*"))
        # Filter for closest timestamp match if needed, for now return last
        if matches:
            return matches[-1]
            
    raise FileNotFoundError(f"Could not find Intan folder for {intan_filename} in {intan_root}")


def get_probe_geometry():
    """Load probe geometry from mat file."""
    mat_probe = loadmat(Path(GEOM_PATH))
    intan_geom = {}
    intan_geom["x"] = mat_probe["xcoords"].ravel()
    intan_geom["y"] = mat_probe["ycoords"].ravel()
    
    if "chanMap0ind" in mat_probe:
        idx_monitor = mat_probe["chanMap0ind"].ravel()
    else:
        raise ValueError("No 0-based chanmap in geometry file.")
        
    p = Probe(ndim=2)
    p.set_contacts(positions=np.c_[intan_geom["x"], intan_geom["y"]], 
                   shapes="square", shape_params={"width": 12.0})
    p.set_device_channel_indices(idx_monitor)
    return p, idx_monitor


def extract_trace_window(rec, ch_id, center_frame, win_frames):
    """Extract trace window in uV."""
    start = center_frame + win_frames[0]
    end = center_frame + win_frames[1]
    if start < 0 or end > rec.get_total_samples():
        return None
    trace = rec.get_traces(segment_index=0, start_frame=start, end_frame=end, channel_ids=[ch_id], return_in_uV=True)
    return trace.squeeze()


def main():
    print(f"── Debug Blanking Impact ──")
    print(f"Condition: {CONDITION}")
    print(f"Window: {WINDOW_MS} ms")
    
    # 1. Load Alignment Info
    aligned_path = find_aligned_file(ALIGNED_CKPT, CONDITION)
    z = np.load(aligned_path, allow_pickle=True)
    meta = json.loads(z["align_meta"].item()) if "align_meta" in z.files else {}
    intan_fn = meta.get("intan_filename", "")
    
    # Get stim times (ALIGNED frame)
    stim_ms = z.get("stim_ms", np.array([], dtype=float))
    stim_ms = np.asarray(stim_ms, float).ravel()
    
    if stim_ms.size == 0:
        print("No stim events found.")
        return

    # Get Rec Start Alignment
    if PROBE == "NPRW":
        pmeta = z["nprw_meta"].item() if "nprw_meta" in z.files else {}
        rec_start_ms_aligned = float(pmeta.get("rec_start_ms_aligned", 0.0))
        # Stim Duration
        rec_stim_dur = float(meta.get("recording_stim_dur", 0.0))
    else:
        print("UA not supported for raw Intan logic debug yet.")
        return

    # Calc local stim times
    stim_local_ms = stim_ms - rec_start_ms_aligned
    print(f"Loaded {len(stim_local_ms)} stim events.")
    
    # Stim Duration logic
    stim_dur = STIM_DUR_OVERRIDE if STIM_DUR_OVERRIDE is not None else rec_stim_dur
    print(f"Stim Duration: {stim_dur:.1f} ms")

    # 2. Load Raw Recording
    intan_folder = find_intan_folder(INTAN_ROOT, intan_fn)
    print(f"Loading Raw Intan: {intan_folder}")
    
    # Stream name from config or default to amplifier
    stream_name = PARAMS.probes.get("NPRW").get("neural_data_stream", "amplifier_data")
    
    rec_raw = se.read_split_intan_files(intan_folder, mode="concatenate", stream_name=stream_name, use_names_as_ids=True)
    rec_raw = spre.unsigned_to_signed(rec_raw)
    
    # Attach Probe
    probe_obj, map_idx = get_probe_geometry()
    rec_raw = rec_raw.set_probe(probe_obj)
    
    # Reorder to geometry
    rec_reordered = rcp.reorder_recording_to_geometry(rec_raw, map_idx)
    fs = rec_reordered.get_sampling_frequency()
    
    # 3. Processing Stages
    
    # A) Highlight: High-pass
    rec_hp = spre.highpass_filter(rec_reordered, freq_min=HP_FREQ)
    
    # B) CAR (Common Average Reference)
    rec_car = spre.common_reference(rec_hp, reference="local", operator="median", 
                                    local_radius=CAR_RADIUS)
                                    
    # C) Artifact Removal (Blanking)
    if Remove_Artifacts:
        # Convert stim times to samples for blanking triggers
        # Filter valid events first
        valid_stim = []
        rec_dur_samples = rec_raw.get_total_samples()
        
        # Calculate samples
        ms_before_samp = int(MS_BEFORE / 1000 * fs)
        # Blanking length needs to cover stim_dur + ms_after
        # remove_artifacts expects 'ms_after' to be total duration AFTER trigger to resume?
        # Actually si.remove_artifacts(ms_after) defines the window END relative to trigger.
        # trigger is at 0. stim ends at stim_dur. we want to blank until stim_dur + MS_AFTER.
        total_ms_after = stim_dur + MS_AFTER
        
        triggers = []
        for s in stim_local_ms:
            samp = int(s / 1000 * fs)
            if 0 <= samp < rec_dur_samples:
                triggers.append(samp)
                
        # To avoid massive overhead of blanking *everything*, usually we just blank the viewer window? 
        # But SI applies it lazily, so we can define it for all triggers.
        
        rec_blanked = spre.remove_artifacts(
            rec_car,
            list_triggers=triggers,
            ms_before=MS_BEFORE,
            ms_after=total_ms_after,
            mode="zeros"
        )
    else:
        rec_blanked = rec_car

    # 4. Plotting
    
    # Filter stims for window
    valid_plot_stims = []
    rec_dur_ms = rec_dur_samples / fs * 1000
    for s in stim_local_ms:
        if (s + WINDOW_MS[0] >= 0) and (s + WINDOW_MS[1] <= rec_dur_ms):
            valid_plot_stims.append(s)
            
    if not valid_plot_stims:
        print("No stims in window.")
        return
        
    # Select Trial
    if isinstance(TRIAL_INDEX, str) and TRIAL_INDEX == 'all':
        trials_to_plot = valid_plot_stims
    else:
        idx = TRIAL_INDEX if TRIAL_INDEX < len(valid_plot_stims) else 0
        trials_to_plot = [valid_plot_stims[idx]]

    win_frames = (int(WINDOW_MS[0]/1000*fs), int(WINDOW_MS[1]/1000*fs))
    time_axis = np.arange(win_frames[0], win_frames[1]) / fs * 1000
    
    n_rows = len(CHANNELS)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 3*n_rows), sharex=True, sharey=True)
    if n_rows == 1: axes = np.array([axes])

    # Columns: Raw (Actually just HP), CAR, Blanked
    cols = ["High-Pass Only", "HP + CAR", "HP + CAR + Blanked"]
    
    print(f"Plotting {len(trials_to_plot)} trials...")
    
    import spikeinterface.widgets as sw

    for r, ch_idx in enumerate(CHANNELS):
        ch_id = rec_hp.get_channel_ids()[ch_idx]
        
        for t_idx, stim_time in enumerate(trials_to_plot):
            center_samp = int(stim_time / 1000 * fs)
            
            # 1. HP Only
            tr_hp = extract_trace_window(rec_hp, ch_id, center_samp, win_frames)
            if tr_hp is not None:
                axes[r, 0].plot(time_axis, tr_hp, color='k', alpha=0.5, lw=0.8)
                
            # 2. CAR
            tr_car = extract_trace_window(rec_car, ch_id, center_samp, win_frames)
            if tr_car is not None:
                axes[r, 1].plot(time_axis, tr_car, color='b', alpha=0.5, lw=0.8)
                
            # 3. Blanked
            tr_bl = extract_trace_window(rec_blanked, ch_id, center_samp, win_frames)
            if tr_bl is not None:
                axes[r, 2].plot(time_axis, tr_bl, color='r', alpha=0.5, lw=0.8)
    
        # Look and Feel
        for c in range(3):
            ax = axes[r, c]
            ax.axvline(0, color='g', ls='--')
            ax.axvline(stim_dur, color='g', ls=':', alpha=0.5)
            
            # Show blanking window on non-blanked plots for reference
            if c < 2 and Remove_Artifacts:
                 rect_start = -MS_BEFORE
                 rect_end = stim_dur + MS_AFTER # Wait, MS_AFTER in SI is relative to TRIGGER?
                 # In SI remove_artifacts: ms_after is end time relative to trigger
                 # OK in my call above I passed total_ms_after = stim_dur + MS_AFTER
                 # So yes, axvspan should be (-MS_BEFORE, stim_dur + MS_AFTER)
                 
                 # Correction: MS_AFTER above user constant is "Time after stim".
                 # So total blank end is stim_dur + MS_AFTER.
                 ax.axvspan(-MS_BEFORE, stim_dur + MS_AFTER, color='y', alpha=0.1)
            
            if r == 0:
                ax.set_title(cols[c])
            if c == 0:
                ax.set_ylabel(f"Ch {ch_idx}\n(uV)")
                
    axes[0, 0].set_xlim(*WINDOW_MS)
    if Y_LIM:
        axes[0, 0].set_ylim(*Y_LIM)
        
    fig.suptitle(f"Blanking Debug: Cond {CONDITION}, StimDur {stim_dur}ms\n"
                 f"Blanking: -{MS_BEFORE}ms ... +{MS_AFTER}ms (post-stim)", y=0.98)
    fig.tight_layout(rect=[0,0,1,0.95])
    
    outname = FIG_DIR / f"debug_cond{CONDITION}_win{WINDOW_MS}_blank-{MS_BEFORE}-{MS_AFTER}.png"
    fig.savefig(outname)
    print(f"Saved: {outname}")
    plt.close(fig)

if __name__ == "__main__":
    main()
