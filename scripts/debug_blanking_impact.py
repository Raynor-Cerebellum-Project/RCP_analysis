"""
Debug Blanking Impact Script

This script allows you to load raw data for a specific condition/window, apply 
the same preprocessing steps as the main pipeline (High-pass, CMR, Artifact Removal), 
and visualize the traces at each stage. This is useful for tuning blanking 
parameters without re-running the entire batch pipeline.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from probeinterface import Probe
from scipy.io import loadmat

import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

################################################
CONDITION = 12                   # BR index
PROBE = "UA"                     # "NPRW" or "UA"
CHANNELS = [178, 182, 230, 70, 72, 124, 14, 25, 38]  # Blackrock array map
WINDOW_MS = (-200.0, 200.0)      # Full extraction window for proper filtering
PLOT_WINDOW_MS = (70.0, 130.0)   # Zoomed-in window for the actual figure
TRIAL_INDEX = 0                  # Trial index to plot (or 'all')
Y_LIM = (-200, 200)

# Processing pipeline order
ORDER_BLANK = 2
ORDER_HPF = 1
ORDER_CMR = 3

# Blanking / Preprocessing Params
REMOVE_ARTIFACTS = True
MS_BEFORE = 5.0                  # Blanking buffer before stim
MS_AFTER = 5.0                   # Blanking buffer after stim ends
BLANK_MODE = "linear"            # "zeros" or "linear"
STIM_DUR_OVERRIDE = None         # Force stim duration if metadata is incorrect

HP_FREQ = 300.0
CMR_RADIUS = (30, 150)           # Inner, Outer radius for local CMR


# Project context and paths
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE = SESSION_LOC / "results"
ALIGNED_CKPT = OUT_BASE / "checkpoints" / "Aligned"
INTAN_ROOT = SESSION_LOC / "Intan"

FIG_DIR = OUT_BASE / "figures" / "blanking_debug"
FIG_DIR.mkdir(parents=True, exist_ok=True)
GEOM_PATH = rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)


def find_aligned_file(br_idx: int) -> Path:
    pat = f"*__BR_{br_idx:03d}.npz"
    for d in [ALIGNED_CKPT / "stim_reaches", ALIGNED_CKPT / "control_reaches", ALIGNED_CKPT / "at_rest", ALIGNED_CKPT]:
        if d.exists() and list(d.glob(pat)):
            return sorted(d.glob(pat))[-1]
    raise FileNotFoundError(f"Missing aligned npz matching {pat} in {ALIGNED_CKPT}")


def get_nprw_probe_geometry():
    mat = loadmat(Path(GEOM_PATH))
    p = Probe(ndim=2)
    p.set_contacts(
        positions=np.c_[mat["xcoords"].ravel(), mat["ycoords"].ravel()], 
        shapes="square", 
        shape_params={"width": 12.0}
    )
    p.set_device_channel_indices(mat["chanMap0ind"].ravel())
    return p, mat["chanMap0ind"].ravel()


def load_raw_recording(meta, intan_fn):
    if PROBE == "NPRW":
        folder = next(INTAN_ROOT.glob(f"*{intan_fn}*"), None)
        if not folder: raise FileNotFoundError(f"Could not find Intan folder for {intan_fn}")
            
        stream = PARAMS.probes.get("NPRW").get("neural_data_stream", "amplifier_data")
        rec = spre.unsigned_to_signed(se.read_split_intan_files(folder, mode="concatenate", stream_name=stream, use_names_as_ids=True))
        
        probe_obj, map_idx = get_nprw_probe_geometry()
        rec = rcp.reorder_recording_to_geometry(rec.set_probe(probe_obj), map_idx)
        
        plot_channels = [(str(ch), ch) for ch in CHANNELS]
        return rec, plot_channels
        
    elif PROBE == "UA":
        br_sess = meta.get("session")
        if not br_sess: raise ValueError("No UA session found in metadata")
        
        rec_raw = se.read_blackrock(BR_ROOT / br_sess, stream_name='nsx6', all_annotations=True)
        ua_map = rcp.load_UA_mapping_from_excel(rcp.ua_excel_path(REPO_ROOT, PARAMS.probes))
        rec, _, ua_elec, _, _, _, _ = rcp.apply_ua_mapping_with_regions(rec_raw, ua_map, CONDITION, METADATA_CSV)
        
        plot_channels = []
        for ch in CHANNELS:
            matches = np.where(ua_elec == ch)[0]
            if matches.size > 0:
                plot_channels.append((str(ch), matches[0]))
            else:
                print(f"Warning: UA Channel {ch} not mapped in this recording.")
        return rec, plot_channels


def run_processing_pipeline(base_rec, triggers, stim_dur_ms):
    steps = [(ORDER_HPF, "HPF"), (ORDER_CMR, "CMR"), (ORDER_BLANK, "Blank")] if PROBE == "NPRW" else [(ORDER_BLANK, "Blank"), (ORDER_HPF, "HPF")]
    steps.sort()
    
    stages_rec = [base_rec]
    stages_titles = ["Raw"]
    rec_current = base_rec
    
    for _, step in steps:
        title = stages_titles[-1]
        if step == "HPF":
            rec_current = spre.highpass_filter(rec_current, freq_min=HP_FREQ)
            title += "\n+ HPF"
        elif step == "CMR":
            try:
                rec_current = spre.common_reference(rec_current, reference="local", operator="median", local_radius=CMR_RADIUS)
                title += "\n+ CMR"
            except Exception as e:
                rec_current = spre.common_reference(rec_current, reference="global", operator="median")
                title += "\n+ CMR (Global)"
        elif step == "Blank":
            if REMOVE_ARTIFACTS:
                rec_current = spre.remove_artifacts(
                    rec_current, 
                    list_triggers=triggers, 
                    ms_before=MS_BEFORE, 
                    ms_after=stim_dur_ms + MS_AFTER, 
                    mode=BLANK_MODE
                )
                title += "\n+ Blank"
            else:
                title += "\n+ Blank (Skip)"
                
        stages_rec.append(rec_current)
        stages_titles.append(title)
        
    return stages_rec[1:], stages_titles[1:]  # Don't plot untouched raw as its own column usually, or keep it depending on UX. Actually, user expects the sequence.
    
    
def generate_debug_plot(recs, titles, plot_channels, stim_local_ms, fs, stim_dur_ms):
    # Determine valid extraction frames
    frames = (int(WINDOW_MS[0] / 1000 * fs), int(WINDOW_MS[1] / 1000 * fs))
    time_ax = np.arange(frames[0], frames[1]) / fs * 1000

    # Filter trials fully within recording
    valid_stims = [s for s in stim_local_ms if (s + WINDOW_MS[0] >= 0) and (int((s + WINDOW_MS[1])/1000*fs) <= recs[0].get_total_samples())]
    if not valid_stims:
        return print("No events fit in the extraction window.")

    trials = valid_stims if TRIAL_INDEX == 'all' else [valid_stims[min(TRIAL_INDEX, len(valid_stims)-1)]]

    n_rows, n_cols = len(plot_channels), len(recs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows), sharex=True, sharey=True)
    if n_rows == 1: axes = np.array([axes])

    stage_colors = ['k', 'b', 'r', 'g']
    extract_pts = lambda r, c_samp: r.get_traces(start_frame=c_samp + frames[0], end_frame=c_samp + frames[1], channel_ids=[r.get_channel_ids()[idx] for _, idx in plot_channels], return_in_uV=True)

    for stim_t in trials:
        center = int(stim_t / 1000 * fs)
        for stage_idx, rec_stage in enumerate(recs):
            traces = extract_pts(rec_stage, center)
            if traces is not None:
                for row_idx, _ in enumerate(plot_channels):
                    axes[row_idx, stage_idx].plot(time_ax, traces[:, row_idx], color=stage_colors[stage_idx % len(stage_colors)], alpha=0.5, lw=0.8)

    for r, (ch_label, _) in enumerate(plot_channels):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axvline(0, color='g', ls='--')
            if stim_dur_ms > 0: ax.axvline(stim_dur_ms, color='g', ls=':', alpha=0.5)
            if REMOVE_ARTIFACTS: ax.axvspan(-MS_BEFORE, stim_dur_ms + MS_AFTER, color='y', alpha=0.1)
                
            if r == 0: ax.set_title(titles[c], fontsize=10)
            if c == 0: ax.set_ylabel(f"Ch {ch_label}\n(uV)")
            
    axes[0, 0].set_xlim(*PLOT_WINDOW_MS)
    if Y_LIM: axes[0, 0].set_ylim(*Y_LIM)
    
    order_flags = [t.split("\n+ ")[-1].replace(" (Skip)", "") for t in titles]
    fig.suptitle(f"Processing Pipeline: {' -> '.join(order_flags)} | Cond {CONDITION}\nBlanking: -{MS_BEFORE}ms to +{MS_AFTER}ms (post-stim)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    
    fname = FIG_DIR / f"debug_cond{CONDITION}_win{int(PLOT_WINDOW_MS[0])}-{int(PLOT_WINDOW_MS[1])}_{'-'.join(order_flags)}.png"
    fig.savefig(fname)
    plt.close(fig)
    print(f"Plot saved successfully to:\n{fname}")


def run_pipeline():
    print(f"Debugger Active | Condition: {CONDITION} | Probe: {PROBE}")
    
    # 1. Load condition alignment array
    z = np.load(find_aligned_file(CONDITION), allow_pickle=True)
    meta = json.loads(z["align_meta"].item()) if "align_meta" in z.files else {}
    
    if PROBE == "UA":
        pmeta = z["ua_meta"].item() if "ua_meta" in z.files else {}
        rec_start = 0.0
    else:
        pmeta = z["nprw_meta"].item() if "nprw_meta" in z.files else {}
        rec_start = float(pmeta.get("rec_start_ms_aligned", 0.0))
    
    # 2. Extract stimulus timeline
    stim_ms = np.asarray(z.get("stim_ms", []), float).ravel()
    if not stim_ms.size: return print("No stim events found.")
    
    # Align to local recording start
    stim_local_ms = stim_ms - rec_start
    stim_dur_ms = STIM_DUR_OVERRIDE if STIM_DUR_OVERRIDE is not None else float(meta.get("recording_stim_dur", 0.0))
    
    # 3. Initialize mapping and recording block
    rec_raw, plot_channels = load_raw_recording(pmeta, meta.get("intan_filename", ""))
    if not plot_channels: return print("No valid channels selected for plotting.")
    
    # 4. Filter triggers safely against out-of-bounds
    fs = rec_raw.get_sampling_frequency()
    triggers = [int(s / 1000 * fs) for s in stim_local_ms if 0 <= int(s / 1000 * fs) < rec_raw.get_total_samples()]
    
    # 5. Process and Plot
    recs, titles = run_processing_pipeline(rec_raw, triggers, stim_dur_ms)
    generate_debug_plot(recs, titles, plot_channels, stim_local_ms, fs, stim_dur_ms)


if __name__ == "__main__":
    run_pipeline()
