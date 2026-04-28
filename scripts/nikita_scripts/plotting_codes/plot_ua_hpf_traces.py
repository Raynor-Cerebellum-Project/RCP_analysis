"""
Plot raw high-pass-filtered (HPF) traces for selected channels around
stimulation events for a given condition (BR index), grouping by UA Arrays.

Usage:
    python scripts/plot_ua_hpf_traces.py

Edit the USER SETTINGS section below to choose condition, channels, and window.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import pandas as pd
import spikeinterface as si
import RCP_analysis as rcp

matplotlib.use("Agg")  # non-interactive backend
matplotlib.rcParams["svg.fonttype"] = "none"

# ──────────────────────────────────────────────────
# USER SETTINGS
# ──────────────────────────────────────────────────
CONDITION   = 56             # BR index  (= condition number)
PROBE       = "UA"           # Exclusively analyzing Utah Array channels
CHANNELS    = list(range(1, 256))   # Electrode IDs to plot (1–256)
WINDOW_MS   = (-200.0, 200.0)      # (start, end) relative to stim onset
Y_LIM       = (-100, 100)         # µV range; set to None for auto-scale
TRIALS_TO_PLOT = [5] #[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]                  # which trials to plot: e.g. [0, 1] or 'all'
PLOT_WAVEFORM = True          # If True, plot spike waveforms on the right
# ──────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parents[1]
PARAMS      = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
SESSION     = getattr(PARAMS, "session", "UNKNOWN")
OUT_BASE    = SESSION_LOC / "results"

ALIGNED_CKPT = OUT_BASE / "checkpoints" / "Aligned"
CKPT_DIR     = OUT_BASE / "checkpoints" / PROBE

FIG_DIR = OUT_BASE / "figures" / "hpf_traces"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def find_aligned_file(aligned_dir: Path, br_idx: int) -> Path:
    """Find the aligned .npz for a given BR index across all subdirectories."""
    pat = f"*__BR_{br_idx:03d}.npz"
    # Search in stim_reaches, control_reaches, at_rest, and top-level
    search_dirs = [
        aligned_dir / "stim_reaches",
        aligned_dir / "control_reaches",
        aligned_dir / "at_rest",
        aligned_dir / "continuous_stim",
        aligned_dir,
    ]
    for d in search_dirs:
        if d.exists():
            cands = sorted(d.glob(pat))
            if cands:
                return cands[-1]
    raise FileNotFoundError(f"No aligned npz matching {pat} in {aligned_dir}")


def find_pp_folder_for_intan(ckpt_dir: Path, intan_filename: str, probe: str, condition: int) -> Path:
    """
    Find the preprocessed SpikeInterface folder that matches a specific
    Intan session filename or Blackrock condition index.
    """
    all_folders = sorted(ckpt_dir.glob("pp_*"))
    
    if probe == "UA":
        # For Blackrock, the condition ID (br_idx) is in the folder name (e.g. pp__NRR_RW022_015__NS6)
        target = f"_{condition:03d}__NS6"
        for f in all_folders:
            if target in f.name:
                return f
                
    # Fallback to matching Intan filename substring (NPRW uses this)
    for f in all_folders:
        if intan_filename in f.name:
            return f
            
    # Legacy Fallback: try without the timestamp prefix
    parts = intan_filename.split("_")
    if len(parts) >= 2:
        session_prefix = "_".join(parts[:2])
        matches = [f for f in all_folders if session_prefix in f.name]
        if matches:
            return matches[-1]
            
    if all_folders:
        print(f"[warn] Could not match folder for condition {condition}; using last available folder: {all_folders[-1].name}")
        return all_folders[-1]
    raise FileNotFoundError(f"No preprocessed folders in {ckpt_dir}")


def extract_traces(rec, ch_row: int, center_ms: float, win_ms: tuple, fs: float):
    """Extract one trace window (in µV) for a channel around a stim event."""
    ch_id = rec.get_channel_ids()[ch_row]
    s0    = int(round(center_ms / 1000.0 * fs))
    i0    = int(s0 + round(win_ms[0] / 1000.0 * fs))
    i1    = int(s0 + round(win_ms[1] / 1000.0 * fs))
    if i0 < 0 or i1 > rec.get_num_frames() or i1 <= i0:
        return None, None
    y = rec.get_traces(start_frame=i0, end_frame=i1,
                       channel_ids=[ch_id], return_in_uV=True).squeeze()
    t = (np.arange(i0, i1) - s0) / fs * 1000.0
    return t, y


def main():
    # ── Load aligned file for this condition ──
    aligned_path = find_aligned_file(ALIGNED_CKPT, CONDITION)
    z = np.load(aligned_path, allow_pickle=True)
    print(f"[info] Aligned file: {aligned_path.name}")

    # ── Parse metadata ──
    meta = json.loads(z["align_meta"].item()) if "align_meta" in z.files else {}
    intan_filename = meta.get("intan_filename", "")
    shift_ms = float(meta.get("shift_ms", 0.0))

    # stim_ms are in the ALIGNED time base (Intan ms - shift_ms).
    stim_ms = z.get("stim_ms", np.array([], dtype=float))
    stim_ms = np.asarray(stim_ms, float).ravel()
    if stim_ms.size == 0:
        print("[error] No stimulation events found in aligned file.")
        return

    # ── Get the recording's start time in the aligned frame ──
    # The SI recording starts at sample 0.
    if PROBE == "NPRW":
        probe_meta = z["nprw_meta"].item() if "nprw_meta" in z.files else {}
        rec_start_ms = float(probe_meta.get("rec_start_ms_aligned", 0.0))
        pk_obj = z.get("nprw_peak_ms_dedup", z.get("nprw_peak_ms", None))
        peak_ms_dict = pk_obj.reshape(-1)[0] if (isinstance(pk_obj, np.ndarray) and pk_obj.dtype == object) else pk_obj
        if peak_ms_dict is None: peak_ms_dict = {}
        target_ch_regions = []
        region_names = []
        target_ch_elecs = []
    else:  # UA
        # UA SpikeInterface recording (NS6) starts at Blackrock Local Time 0.0
        rec_start_ms = 0.0
        pk_obj = z.get("ua_peak_ms_dedup", z.get("ua_peak_ms", None))
        peak_ms_dict = pk_obj.reshape(-1)[0] if (isinstance(pk_obj, np.ndarray) and pk_obj.dtype == object) else pk_obj
        if peak_ms_dict is None: peak_ms_dict = {}
        target_ch_regions = z.get("ua_region", [])
        region_names = z.get("ua_region_names", [])
        target_ch_elecs = z.get("ua_elec", [])
        if isinstance(target_ch_elecs, np.ndarray) and target_ch_elecs.ndim == 0:
            target_ch_elecs = target_ch_elecs.item()  # Handle occasional object scalar
        target_ch_elecs = np.asarray(target_ch_elecs, int).ravel()
        ua_idx_rows = z.get("ua_idx_rows", [])
        if isinstance(ua_idx_rows, np.ndarray) and ua_idx_rows.ndim == 0:
            ua_idx_rows = ua_idx_rows.item()
        ua_idx_rows = np.asarray(ua_idx_rows, int).ravel()

    # Convert stim times from aligned time base to recording-local time base
    stim_local_ms = stim_ms - rec_start_ms
    print(f"[info] Aligned stim_ms range: {stim_ms.min():.1f} – {stim_ms.max():.1f} ms")
    print(f"[info] shift_ms (anchor offset) assumed: {rec_start_ms:.1f} ms")
    print(f"[info] Local stim_ms range: {stim_local_ms.min():.1f} – {stim_local_ms.max():.1f} ms")

    # ── Resolve Stimulation Duration from Hardware Bounds ──
    from RCP_analysis.python.functions.config_loading import METADATA_CSV, NPRW_AUX_DATA, METADATA_ROOT
    stim_npz_path, _ = rcp.stim_npz_path_from_br_idx(CONDITION, METADATA_CSV, NPRW_AUX_DATA)
    fs_nprw = 30_000.0
    shifts_csv = OUT_BASE.parent / "Metadata" / "br_to_intan_shifts.csv"
    if shifts_csv.exists():
        br2fs_intan = rcp.get_metadata_mapping(shifts_csv, "br_idx", "fs_intan")
        fs_nprw = float(br2fs_intan.get(CONDITION, 30_000.0))
        
        # Override shift_ms dynamically if processing UA (Blackrock is anchor)
        if PROBE == "UA":
            br2shift_samp = rcp.get_metadata_mapping(shifts_csv, "br_idx", "shift_sample")
            if CONDITION in br2shift_samp and br2shift_samp[CONDITION]:
                shift_ms = float(br2shift_samp[CONDITION]) / fs_nprw * 1000.0
                rec_start_ms = 0.0
                stim_local_ms = stim_ms - rec_start_ms

    stim_dur = 0.0
    if stim_npz_path and stim_npz_path.exists():
        stim = rcp.load_stim_detection(stim_npz_path)
        block_bounds = stim.get("block_bounds_samples", np.empty((0, 2), dtype=np.int64))
        if len(block_bounds) > 0:
            if isinstance(block_bounds, list):
                block_bounds = np.array(block_bounds, dtype=np.int64)
            nprw_samps = block_bounds[:, 0].astype(np.int64)
            nprw_ends  = block_bounds[:, 1].astype(np.int64)
            stim_dur   = float((nprw_ends - nprw_samps).max() * 1000.0 / fs_nprw)
    print(f"[info] Extracted stim_dur: {stim_dur:.1f} ms")

    # ── Load SpikeInterface recording for the correct Intan session ──
    pp_folder = find_pp_folder_for_intan(CKPT_DIR, intan_filename, PROBE, CONDITION)
    rec = si.load(str(pp_folder))
    fs  = float(rec.get_sampling_frequency())
    n_ch = rec.get_num_channels()
    rec_dur_ms = rec.get_num_frames() / fs * 1000.0
    print(f"[info] Recording: {pp_folder.name}  |  fs={fs:.0f} Hz  |  {n_ch} ch  |  {rec_dur_ms:.1f} ms")
    print(f"[info] Stim events: {stim_ms.size}  |  condition {CONDITION}")

    # Resolve channel metadata (Electrode ID and Original Index)
    # This map allows us to find the correct spikes in the .npz for any recording channel
    ridx_to_meta = {}
    ch_ids = rec.get_channel_ids()
    
    # Path to robust mapping CSV
    mapping_csv = REPO_ROOT / "scripts" / "electrode_port_mapping.csv"
    mapping_df = pd.read_csv(mapping_csv) if mapping_csv.exists() else None
    
    for ridx, cid in enumerate(ch_ids):
        # 1. Try to parse Electrode ID from name (e.g. UAe005 -> 5)
        m = re.search(r'UAe(\d+)', str(cid))
        eid = int(m.group(1)) if m else -1
        
        # 2. Fallback to indexing target_ch_elecs if parsing fails
        if eid < 0 and ridx < len(target_ch_elecs):
            eid = int(target_ch_elecs[ridx])
            
        # 3. Find where this Electrode is in the .npz mapping to align spikes
        # peak_ms_dict is usually keyed by the index into the 'peaks' array saved in the .npz
        # which corresponds to the index in target_ch_elecs
        oid = ridx
        if eid > 0:
            matches = np.where(target_ch_elecs == eid)[0]
            if matches.size > 0:
                oid = int(matches[0])
                
        ridx_to_meta[ridx] = {"eid": eid, "oid": oid, "cid": cid}

    # Resolve specifically requested channels for plotting
    resolved_channels = []
    if PROBE == "UA" and target_ch_elecs.size > 0:
        print(f"[info] Resolving Electrode IDs to recording indices...")
        for eid_req in CHANNELS:
            # Find which RIDX in the current recording has this Electrode ID
            # We check the metadata we just built from the channel IDs
            found = False
            for ridx, meta in ridx_to_meta.items():
                if meta["eid"] == eid_req:
                    resolved_channels.append(ridx)
                    found = True
                    break
            if not found:
                # Fallback to direct mapping if parsing didn't find it
                matches = np.where(target_ch_elecs == eid_req)[0]
                if matches.size > 0:
                    resolved_channels.append(int(matches[0]))
    else:
        # Fallback for NPRW or if no electrode mapping
        resolved_channels = [c for c in CHANNELS if 0 <= c < n_ch]

    if not resolved_channels:
        print(f"[error] No valid channels found for {CHANNELS} (recording has {n_ch} channels).")
        return



    # ── Filter valid stim events that fit within the recording ──
    valid = []
    for s in stim_local_ms:
        if (s + WINDOW_MS[0]) >= 0 and (s + WINDOW_MS[1]) <= rec_dur_ms:
            valid.append(s)
    valid = np.array(valid)
    if valid.size == 0:
        print("[error] No stim events fit within the recording for this window.")
        return
    print(f"[info] Valid stim events for window: {valid.size} / {stim_ms.size}")

    # ── Build figure ──
    if isinstance(TRIALS_TO_PLOT, str) and TRIALS_TO_PLOT.lower() == "all":
        plot_idxs = np.arange(len(valid))
    else:
        plot_idxs = np.array([i for i in TRIALS_TO_PLOT if i < len(valid)])

    # Group channels by region
    region_to_idxs = {}
    if PROBE == "UA" and len(target_ch_regions) > 0:
        for ch_idx in resolved_channels:
            if ch_idx < len(target_ch_regions):
                reg_idx = int(target_ch_regions[ch_idx])
                reg_name = region_names[reg_idx] if reg_idx >= 0 else "Unknown"
                region_to_idxs.setdefault(reg_name, []).append(ch_idx)
    else:
        region_to_idxs["All Channels"] = resolved_channels

    for trial_i in plot_idxs:
        center_time = valid[trial_i]
        
        for reg_name, reg_channels in region_to_idxs.items():
            if not reg_channels: continue
            
            n_rows = len(reg_channels)
            PLOT_WAVEFORM = globals().get("PLOT_WAVEFORM", False)
            if PLOT_WAVEFORM:
                fig = plt.figure(figsize=(15, 2.0 * n_rows))
                gs = matplotlib.gridspec.GridSpec(n_rows, 2, width_ratios=[4, 1])
                axes_trace = []
                axes_wv = []
                for r in range(n_rows):
                    ax_t = fig.add_subplot(gs[r, 0], sharex=axes_trace[0] if r>0 else None, sharey=axes_trace[0] if r>0 else None)
                    ax_w = fig.add_subplot(gs[r, 1], sharex=axes_wv[0] if r>0 else None, sharey=ax_t)
                    axes_trace.append(ax_t)
                    axes_wv.append(ax_w)
            else:
                fig, axes_raw = plt.subplots(n_rows, 1, figsize=(12, 2.0 * n_rows),
                                         sharex=True, sharey=True)
                axes_trace = np.atleast_1d(axes_raw)
                axes_wv = [None] * n_rows
                
            for r, ch in enumerate(reg_channels):
                ax = axes_trace[r]
                ax_wv = axes_wv[r]
                t, y = extract_traces(rec, ch, center_time, WINDOW_MS, fs)
                
                meta = ridx_to_meta.get(ch, {"eid": ch, "oid": ch})
                eid = meta["eid"]
                oid = meta["oid"]
                
                if t is not None:
                    ax.plot(t, y, lw=0.7, color="k")
                
                # Label with Electrode ID
                ch_label = f"Elec {eid}" if PROBE == "UA" and eid > 0 else f"Ch {ch}"
                ax.set_ylabel(f"{ch_label}\n(µV)", fontsize=8)

                # Overlay spikes - IMPORTANT: lookup by Alignment Index (oid) 
                # This ensures we get spikes for the physical electrode shown in the trace
                ch_peaks_abs_aligned = np.asarray(peak_ms_dict.get(oid, []), float).ravel()
                
                # Fallback to keying by Electrode ID directly (for some pipeline versions)
                if ch_peaks_abs_aligned.size == 0 and PROBE == "UA" and eid > 0:
                    ch_peaks_abs_aligned = np.asarray(peak_ms_dict.get(eid, []), float).ravel()
                
                stim_ms_trial = center_time + rec_start_ms
                pk_rel = ch_peaks_abs_aligned - stim_ms_trial


                
                # filter to window
                pk_rel_win = pk_rel[(pk_rel >= WINDOW_MS[0]) & (pk_rel <= WINDOW_MS[1])]
                
                # Get trace y-value at these relative times
                if pk_rel_win.size > 0 and t is not None:
                    peak_idx = np.searchsorted(t, pk_rel_win)
                    valid_idx = (peak_idx >= 0) & (peak_idx < len(t))
                    peak_idx = peak_idx[valid_idx]
                    pk_rel_clean = pk_rel_win[valid_idx]
                    ax.plot(pk_rel_clean, y[peak_idx], "ro", markersize=3, alpha=0.8)
                    
                    if ax_wv is not None:
                        wf_pre = int(1.0 * fs / 1000.0)
                        wf_post = int(2.0 * fs / 1000.0)
                        t_wv = np.arange(-wf_pre, wf_post) * 1000.0 / fs
                        for p_idx in peak_idx:
                            i0 = p_idx - wf_pre
                            i1 = p_idx + wf_post
                            if i0 >= 0 and i1 < len(y):
                                ax_wv.plot(t_wv, y[i0:i1], color="k", lw=0.5, alpha=0.5)
                        ax_wv.set_xlim(-1, 1)
                        ax_wv.axvline(0, ls="--", color="r", alpha=0.5)
                        if r == n_rows - 1:
                            ax_wv.set_xlabel("Time (ms)")

                ax.axvline(0.0, ls="--", lw=0.8, color="green", label="Stim onset")
                if stim_dur > 0:
                    ax.axvline(stim_dur, ls="--", lw=0.8, color="green", label="Stim end")
                    ax.axvspan(0.0, stim_dur, color="gray", alpha=0.3, zorder=0)
                
                if Y_LIM is not None:
                    ax.set_ylim(*Y_LIM)


            axes_trace[-1].set_xlabel("Time relative to stimulation (ms)")
            axes_trace[0].set_xlim(*WINDOW_MS)

            fig.suptitle(
                f"{SESSION}  •  Condition {CONDITION} (BR_{CONDITION:03d})  •  "
                f"Region: {reg_name}  •  Trial {trial_i + 1}",
                fontsize=11, y=0.99 if n_rows < 10 else 1.0 - (0.5 / n_rows),
            )
            fig.tight_layout(rect=[0, 0, 1, 0.98 if n_rows < 10 else 1.0 - (1.0 / n_rows)])

            reg_clean = reg_name.replace(' ', '_').replace('/', '_')
            out_path = (
                FIG_DIR
                / f"{SESSION}__cond{CONDITION:03d}__{PROBE}__Region_{reg_clean}"
                  f"__trial{trial_i + 1}__win{int(WINDOW_MS[0])}-{int(WINDOW_MS[1])}ms.png"
            )
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
