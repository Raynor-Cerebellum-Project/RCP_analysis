"""
Debug Blanking Impact Script

This script allows you to load raw data for a specific condition/window, apply 
the same preprocessing steps as the main pipeline (High-pass, CMR, Artifact Removal), 
and visualize the traces at each stage. This is useful for tuning blanking 
parameters without re-running the entire batch pipeline.

Steps are configurable to run in any order (e.g., Blanking -> HPF -> CMR).
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
PROBE       = "UA"        # "NPRW" or "UA"
# CHANNELS         = list(range(7, 16))   # 0-indexed channel rows to plot
# CHANNELS         = [7, 8, 9, 10, 11, 32, 33, 50, 80]   # intan
CHANNELS         = [178, 182, 230, 70, 72, 124, 14, 25, 38]   # blackrock - 2M1i, 1M1s, 3PMd, 3SMA
WINDOW_MS        = (-200.0, 200.0)      # Full extraction window for proper filtering
PLOT_WINDOW_MS   = (70.0, 130.0)       # Zoomed-in window for the figure
TRIAL_INDEX      = 0                    # Which valid trial index to plot (or 'all')
Y_LIM       = (-200, 200)

# Processing Order (1, 2, 3) - Determines the sequence of pipeline steps
ORDER_BLANK = 2
ORDER_HPF   = 1
ORDER_CMR   = 3

# Blanking / Preprocessing Params (Override here to test)
Remove_Artifacts = True
MS_BEFORE = 5.0             # Blanking time before stim (ms)
MS_AFTER  = 5.0             # Blanking time after stim (ms) - relative to stim END
BLANK_MODE = "linear"         # "zeros" or "linear"
STIM_DUR_OVERRIDE = None     # Force stim dur (ms). If None, tries to read from metadata.

# Standard filtering params
HP_FREQ = 300.0
CMR_RADIUS = (30, 150)       # Inner, Outer radius for local CMR

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
    cand = intan_root / intan_filename
    if cand.exists() and cand.is_dir():
        return cand
        
    matches = sorted(intan_root.glob(f"*{intan_filename}*"))
    if matches:
        return matches[-1]
        
    parts = intan_filename.split("_")
    if len(parts) >= 2:
        prefix = "_".join(parts[:2])
        matches = sorted(intan_root.glob(f"*{prefix}*"))
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


def extract_trace_window(rec, ch_ids, center_frame, win_frames):
    """Extract trace window in uV."""
    start = center_frame + win_frames[0]
    end = center_frame + win_frames[1]
    if start < 0 or end > rec.get_total_samples():
        return None
    traces = rec.get_traces(segment_index=0, start_frame=start, end_frame=end, channel_ids=ch_ids, return_in_uV=True)
    return traces


def main():
    print(f"-- Debug Blanking Impact --")
    print(f"Condition: {CONDITION}")
    print(f"Extraction Window: {WINDOW_MS} ms")
    print(f"Plot Window: {PLOT_WINDOW_MS} ms")
    print(f"Processing Order: Blank({ORDER_BLANK}), HPF({ORDER_HPF}), CMR({ORDER_CMR})")
    
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
        rec_stim_dur = float(meta.get("recording_stim_dur", 0.0))
    elif PROBE == "UA":
        pmeta = z["ua_meta"].item() if "ua_meta" in z.files else {}
        rec_start_ms_aligned = 0.0  # Stim times align directly to UA start (0)
        rec_stim_dur = float(meta.get("recording_stim_dur", 0.0))
    else:
        print(f"Probe {PROBE} not recognized.")
        return

    # Calc local stim times
    stim_local_ms = stim_ms - rec_start_ms_aligned
    print(f"Loaded {len(stim_local_ms)} stim events.")
    
    # Stim Duration logic
    stim_dur = STIM_DUR_OVERRIDE if STIM_DUR_OVERRIDE is not None else rec_stim_dur
    print(f"Stim Duration: {stim_dur:.1f} ms")

    # 2. Load Raw Recording
    if PROBE == "NPRW":
        intan_folder = find_intan_folder(INTAN_ROOT, intan_fn)
        print(f"Loading Raw Intan: {intan_folder}")
        
        stream_name = PARAMS.probes.get("NPRW").get("neural_data_stream", "amplifier_data")
        rec_raw = se.read_split_intan_files(intan_folder, mode="concatenate", stream_name=stream_name, use_names_as_ids=True)
        rec_raw = spre.unsigned_to_signed(rec_raw)
        
        # Attach Probe & Reorder
        probe_obj, map_idx = get_probe_geometry()
        rec_raw = rec_raw.set_probe(probe_obj)
        rec_reordered = rcp.reorder_recording_to_geometry(rec_raw, map_idx)
        fs = rec_reordered.get_sampling_frequency()
        
    elif PROBE == "UA":
        br_sess_name = pmeta.get("session")
        if not br_sess_name:
            print("No UA session found in metadata.")
            return
            
        br_folder = BR_ROOT / br_sess_name
        print(f"Loading Raw UA (Blackrock): {br_folder}")
        
        rec_raw = se.read_blackrock(br_folder, stream_name='nsx6', all_annotations=True)
        
        XLS = rcp.ua_excel_path(REPO_ROOT, PARAMS.probes)
        UA_MAP = rcp.load_UA_mapping_from_excel(XLS) if XLS else None
        if UA_MAP is None:
            print("UA mapping required for mapping on NS6.")
            return
            
        rec_reordered, idx_rows, ua_elec, ua_nsp, ua_region, ua_region_names, ua_port = rcp.apply_ua_mapping_with_regions(
            rec_raw, UA_MAP, CONDITION, METADATA_CSV
        )
        fs = rec_reordered.get_sampling_frequency()
    
    # 3. Processing Stages
    
    # Pre-calculate triggers for blanking
    rec_dur_samples = rec_raw.get_total_samples()
    total_ms_after = stim_dur + MS_AFTER
    triggers = []
    for s in stim_local_ms:
        samp = int(s / 1000 * fs)
        if 0 <= samp < rec_dur_samples:
            triggers.append(samp)
            
    # Define steps specific to each probe's pipeline
    if PROBE == "NPRW":
        steps = [
            (ORDER_HPF, "HPF"),
            (ORDER_CMR, "CMR"),
            (ORDER_BLANK, "Blank")
        ]
    elif PROBE == "UA":
        steps = [
            (ORDER_BLANK, "Blank"),
            (ORDER_HPF, "HPF") 
        ]
    steps.sort(key=lambda x: x[0])  # Sort ascending by user-defined order
    
    recs = []
    cols = []
    
    rec_current = rec_reordered
    current_title = "Raw"
    
    for order, step_name in steps:
        if step_name == "HPF":
            rec_current = spre.highpass_filter(rec_current, freq_min=HP_FREQ)
            current_title += f"\n+ HPF"
        elif step_name == "CMR":
            try:
                rec_current = spre.common_reference(rec_current, reference="local", operator="median", local_radius=CMR_RADIUS)
                current_title += f"\n+ CMR"
            except Exception as e:
                print(f"Local CMR failed ({e}), falling back to global CMR.")
                rec_current = spre.common_reference(rec_current, reference="global", operator="median")
                current_title += f"\n+ CMR (Global)"
        elif step_name == "Blank":
            if Remove_Artifacts:
                rec_current = spre.remove_artifacts(
                    rec_current,
                    list_triggers=triggers,
                    ms_before=MS_BEFORE,
                    ms_after=total_ms_after,
                    mode=BLANK_MODE
                )
                current_title += f"\n+ Blank"
            else:
                current_title += f"\n+ Blank (Skip)"
                
        recs.append(rec_current)
        cols.append(current_title)

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
    n_cols = len(recs)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 3*n_rows), sharex=True, sharey=True)
    if n_rows == 1: axes = np.array([axes])

    print(f"Plotting {len(trials_to_plot)} trials across {n_cols} stages...")
    
    # Map user CHANNELS to recording row indices and labels
    plot_channels = [] # list of (label, row_idx)
    if PROBE == "UA":
        for ch_id in CHANNELS:
            matches = np.where(ua_elec == ch_id)[0]
            if matches.size > 0:
                plot_channels.append((str(ch_id), matches[0]))
            else:
                print(f"Warning: UA Channel {ch_id} not mapped in this recording.")
    else:
        for ch_id in CHANNELS:
            plot_channels.append((str(ch_id), ch_id))
            
    if not plot_channels:
        print("No valid channels to plot.")
        return

    # Map colors to stages: stage 1 -> black, stage 2 -> blue, stage 3 -> red
    stage_colors = ['k', 'b', 'r']
    ch_ids = [recs[-1].get_channel_ids()[row_idx] for _, row_idx in plot_channels]

    for t_idx, stim_time in enumerate(trials_to_plot):
        center_samp = int(stim_time / 1000 * fs)
        
        for c, rec_stage in enumerate(recs):
            traces = extract_trace_window(rec_stage, ch_ids, center_samp, win_frames)
            if traces is not None:
                for r, (ch_label, _) in enumerate(plot_channels):
                    color = stage_colors[c % len(stage_colors)]
                    axes[r, c].plot(time_axis, traces[:, r], color=color, alpha=0.5, lw=0.8)

    # Look and Feel
    for r, (ch_label, _) in enumerate(plot_channels):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axvline(0, color='g', ls='--')
            if stim_dur > 0:
                ax.axvline(stim_dur, color='g', ls=':', alpha=0.5)
            
            # Show scheduled blanking window on all plots for reference
            if Remove_Artifacts:
                 ax.axvspan(-MS_BEFORE, stim_dur + MS_AFTER, color='y', alpha=0.1)
            
            if r == 0:
                # Add title for this column
                ax.set_title(cols[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"Ch {ch_label}\n(uV)")
                
    axes[0, 0].set_xlim(*PLOT_WINDOW_MS)
    if Y_LIM:
        axes[0, 0].set_ylim(*Y_LIM)
        
    order_str = "-".join([step[1] for step in steps])
    fig.suptitle(f"Processing Order: {order_str} | Cond {CONDITION}, StimDur {stim_dur}ms\n"
                 f"Blanking: -{MS_BEFORE}ms ... +{MS_AFTER}ms (post-stim)", y=0.98)
    fig.tight_layout(rect=[0,0,1,0.92])
    
    outname = FIG_DIR / f"debug_cond{CONDITION}_win{int(PLOT_WINDOW_MS[0])}-{int(PLOT_WINDOW_MS[1])}_{order_str}.png"
    fig.savefig(outname)
    print(f"Saved: {outname}")
    plt.close(fig)

if __name__ == "__main__":
    main()
