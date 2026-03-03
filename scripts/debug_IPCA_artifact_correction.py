"""
debug_IPCA_artifact_correction.py
=================================

Diagnostic tool for PCA-based stimulation artifact correction on Utah Array
and Neuropixels recordings. Extracts per-pulse micro-windows from raw traces,
learns artifact templates via Incremental PCA, and produces visualisations
of raw, template, and corrected signals.

Architecture
------------
1. Stimulus Discovery – Read hardware stim blocks from Intan auxiliary
   traces and (optionally) filter to behaviourally valid Target A trials.
2. Pulse Alignment – For each trial, locate every individual pulse within
   the stim train by derivative-walking from the first pulse.  Each pulse
   centre is then refined to the *earliest* significant deflection in the
   mean-subtracted signal (immune to baseline drift and biphasic ambiguity).
3. PCA Template Learning – All aligned micro-windows are stacked and
   passed through IPCA_Artifact_Correction.ipca_all() to learn a
   low-rank artifact template shared across pulses and trials.
4. Reconstruction & Visualisation – Corrected micro-windows are spliced
   back into full trial traces for side-by-side comparison plots.

Note
----
The PCA artifact correction block (§3 above) is designed as a drop-in
candidate for integration into the production
UA_BR_analysis_threshold.py pipeline.  Look for the banner:

    # ══════════════════════════════════════════════════════════════════
    # §3  PCA ARTIFACT CORRECTION — portable to UA_BR_analysis_threshold
    # ══════════════════════════════════════════════════════════════════
"""

# ─────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import spikeinterface as si
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


# ─────────────────────────────────────────────────────────────────────
# Session-level paths (resolved once at import time)
# ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
PARAMS      = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE    = SESSION_LOC / "results"
ALIGNED_CKPT = OUT_BASE / "checkpoints" / "Aligned"
INTAN_ROOT  = SESSION_LOC / "Intan"
FIG_DIR     = OUT_BASE / "figures" / "blanking_debug"
FIG_DIR.mkdir(parents=True, exist_ok=True)
GEOM_PATH   = rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)


# ═════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════

def find_aligned_file(aligned_dir: Path, br_idx: int) -> Path:
    """Locate an aligned ``.npz`` across standard checkpoint subfolders."""
    pattern = f"*__BR_{br_idx:03d}.npz"
    search_dirs = [
        aligned_dir / "stim_reaches",
        aligned_dir / "control_reaches",
        aligned_dir / "at_rest",
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


def load_probe_geometry():
    """Load the Neuropixels channel map from the session's .mat file."""
    mat = loadmat(str(GEOM_PATH))
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
    """Return the index of the *earliest* significant deflection in a chunk.

    The chunk is mean-subtracted so that baseline drift does not influence
    the result.  A 40 % threshold relative to the local maximum is used to
    reject noise, and the peak of the first supra-threshold region is returned.

    This function is critical for consistent alignment of biphasic artifacts:
    it always locks onto the same (first) phase regardless of drift.
    """
    centered = np.abs(signal_chunk - np.mean(signal_chunk))
    threshold = 0.4 * np.max(centered)
    first_crossing = int(np.argmax(centered > threshold))
    # Refine to the local peak within ±2 samples of the crossing
    lo = max(0, first_crossing - 2)
    hi = min(len(centered), first_crossing + 3)
    return lo + int(np.argmax(centered[lo:hi]))


# ═════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════

def run_ipca_debug(
    condition: int = 10,
    probe: str = "NPRW",
    target_ch = 10,
    ipca_rank: int = 3,
    window_ms: tuple = (-10.0, 50.0),
    trials_plot: int = 5,
    pulse_window_ms: tuple = (-0.5, 0.5),
    apply_hpf: bool = False,
    debug_single_pulse: bool = True,
    use_behavior_filter: bool = True,
    ua_refine_search_ms: float = 0.15,
    reference_ch: int = None,
):
    """Run the full IPCA artifact-correction debug pipeline.

    Parameters
    ----------
    condition : int
        Blackrock block index to analyse.
    probe : str
        "NPRW" (Neuropixels) or "UA" (Utah Array).
    target_ch : int or str
        Channel number to analyse (1-based electrode ID for UA), or
        "all" to process every mapped channel with cross-channel IPCA.
    ipca_rank : int
        Number of principal components used to model the artifact.
    window_ms : tuple of float
        (pre, post) extraction window around each stim block (ms).
    trials_plot : int
        Maximum number of trials to include in the summary figure.
    pulse_window_ms : tuple of float
        (pre, post) micro-window around each detected pulse (ms).
    apply_hpf : bool
        If True, apply a 300 Hz high-pass filter before extraction.
    debug_single_pulse : bool
        If True, save a diagnostic overlay of all micro-windows.
    use_behavior_filter : bool
        If True, restrict to behaviourally valid Target A trials.
    ua_refine_search_ms : float
        Search radius (ms) for pulse centre refinement on the UA trace.
    reference_ch : int or None
        When target_ch="all", which channel to use for pulse detection.
        Defaults to the first mapped channel.

    Returns
    -------
    dict
        All extracted arrays and metadata for downstream analysis.
    """
    all_channels_mode = isinstance(target_ch, str) and target_ch.lower() == "all"
    ch_label = "ALL" if all_channels_mode else str(target_ch)
    print(f"Running IPCA Debug | Cond: {condition} | Probe: {probe} "
          f"| Ch: {ch_label} | Rank: {ipca_rank}")

    # ─────────────────────────────────────────────────────────────────
    # §1  STIMULUS DISCOVERY
    # ─────────────────────────────────────────────────────────────────

    # 1a. Load alignment metadata
    aligned_path = find_aligned_file(ALIGNED_CKPT, condition)
    z = np.load(aligned_path, allow_pickle=True)
    meta = json.loads(z["align_meta"].item()) if "align_meta" in z.files else {}

    # 1b. Locate Intan recording folder
    intan_filename = meta.get("intan_filename", "")
    intan_folder = find_intan_folder(INTAN_ROOT, intan_filename)
    sess_name = intan_folder.name

    # 1c. Load stim block boundaries from Intan hardware detection
    fs_nprw = 30_000.0
    shift_samp_intan = 0.0

    if probe == "NPRW":
        stim_npz_path = NPRW_AUX_DATA / f"{sess_name}_Intan_streams" / "stim_stream.npz"
    else:
        stim_npz_path, _ = rcp.stim_npz_path_from_br_idx(
            condition, METADATA_CSV, NPRW_AUX_DATA
        )
        shifts_csv = METADATA_ROOT / "br_to_intan_shifts.csv"
        if not shifts_csv.exists():
            raise FileNotFoundError("br_to_intan_shifts.csv missing")
        br2fs_intan = rcp.get_metadata_mapping(shifts_csv, "br_idx", "fs_intan")
        br2shift    = rcp.get_metadata_mapping(shifts_csv, "br_idx", "shift_sample")
        fs_nprw = float(br2fs_intan.get(condition, 30_000.0))
        shift_raw = br2shift.get(condition, None)
        if shift_raw is None or (isinstance(shift_raw, str) and not shift_raw.strip()):
            raise ValueError(
                f"Missing anchor_sample for br_idx={condition} in {shifts_csv}."
            )
        shift_samp_intan = float(shift_raw)

    if not stim_npz_path or not stim_npz_path.exists():
        raise FileNotFoundError(
            f"stim_stream.npz not found for Intan session: {sess_name} "
            f"(looked at: {stim_npz_path})"
        )
    stim = rcp.load_stim_detection(stim_npz_path)
    block_bounds = stim.get("block_bounds_samples", np.empty((0, 2), dtype=np.int64))
    if len(block_bounds) == 0:
        raise ValueError("No stimulation blocks found in Intan auxiliary traces.")
    if isinstance(block_bounds, list):
        block_bounds = np.array(block_bounds, dtype=np.int64)

    nprw_samps = block_bounds[:, 0].astype(np.int64)
    nprw_ends  = block_bounds[:, 1].astype(np.int64)
    stim_dur   = float((nprw_ends - nprw_samps).max() * 1000.0 / fs_nprw)
    print(f"Extracted {len(nprw_samps)} raw hardware stim blocks. "
          f"Max duration: {stim_dur:.2f} ms")

    # 1d. (Optional) Filter to behaviourally valid Target A trials
    if use_behavior_filter:
        try:
            peri_dir = (OUT_BASE / "checkpoints" / "PeriStim"
                        / "stim_reaches" / "Target_A")
            cands = list(peri_dir.glob(f"*BR_{condition:03d}*.npz"))
            if cands:
                pz = np.load(cands[0], allow_pickle=True)
                valid_br_ms = pz["event_ms"]
                if "stim_ms" in z.files:
                    br_ms_all = z["stim_ms"]
                    keep = np.zeros(len(nprw_samps), dtype=bool)
                    for vms in valid_br_ms:
                        idx = np.argmin(np.abs(br_ms_all - vms))
                        if np.abs(br_ms_all[idx] - vms) < 2.0:
                            keep[idx] = True
                    nprw_samps = nprw_samps[keep]
                    if len(nprw_ends) == len(keep):
                        nprw_ends = nprw_ends[keep]
                    print(f"-> Filtered to {keep.sum()} behaviourally valid "
                          f"Target A trials.")
                else:
                    print("Warning: 'stim_ms' not in aligned NPZ; "
                          "skipping behaviour filter.")
        except Exception as exc:
            print(f"Warning: behaviour filter failed ({exc})")

    # ─────────────────────────────────────────────────────────────────
    # §2  RAW RECORDING & PULSE ALIGNMENT
    # ─────────────────────────────────────────────────────────────────

    # 2a. Open raw recording and resolve channel
    if probe == "NPRW":
        stim_local_ms = nprw_samps * 1000.0 / fs_nprw
        stream_name = PARAMS.probes.get("NPRW").get(
            "neural_data_stream", "amplifier_data"
        )
        rec_raw = se.read_split_intan_files(
            intan_folder, mode="concatenate",
            stream_name=stream_name, use_names_as_ids=True,
        )
        rec_raw = spre.unsigned_to_signed(rec_raw)
        probe_obj, map_idx = load_probe_geometry()
        rec_raw = rec_raw.set_probe(probe_obj)
        rec_proc = rcp.reorder_recording_to_geometry(rec_raw, map_idx)
        fs = float(rec_proc.get_sampling_frequency())

    elif probe == "UA":
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
        (rec_proc, _, ua_elec, _, ua_region, ua_region_names, _) = rcp.apply_ua_mapping_with_regions(
            rec_raw, ua_map, condition, METADATA_CSV,
        )
        fs = rec_proc.get_sampling_frequency()

        # Map Intan block starts → Blackrock sample indices
        scale = fs / fs_nprw
        starts_ua = np.round((nprw_samps - shift_samp_intan) * scale).astype(np.int64)
        ends_ua   = np.round((nprw_ends  - shift_samp_intan) * scale).astype(np.int64)
        n_total   = rec_proc.get_num_samples()
        ends_ua   = np.minimum(ends_ua, n_total)
        valid     = (ends_ua > starts_ua) & (starts_ua >= 0) & (ends_ua <= n_total)
        starts_ua = starts_ua[valid]
        stim_local_ms = starts_ua * 1000.0 / fs
    else:
        print(f"Probe '{probe}' not recognised.")
        return

    if not stim_local_ms.size:
        print("No stim events found.")
        return

    if apply_hpf:
        rec_proc = spre.highpass_filter(rec_proc, freq_min=300.0)

    # 2b. Timing constants
    stim_freq        = float(meta.get("Freq", 300.0))
    pulse_interval_ms = 1000.0 / stim_freq
    num_pulses       = int(np.floor(stim_dur / pulse_interval_ms)) + 1
    print(f"-> Stim frequency: {stim_freq} Hz")

    if num_pulses <= 0:
        print("Stim duration too short for any pulse extraction.")
        return

    # 2c. Frame indices for windows
    fw_start     = int(window_ms[0] / 1000 * fs)
    fw_end       = int(window_ms[1] / 1000 * fs)
    full_n_time  = fw_end - fw_start
    mw_start     = int(pulse_window_ms[0] / 1000 * fs)
    mw_end       = int(pulse_window_ms[1] / 1000 * fs)
    micro_n_time = mw_end - mw_start

    rec_samples  = rec_proc.get_total_samples()
    valid_stims  = [
        s for s in stim_local_ms
        if 0 <= int(s / 1000 * fs) + fw_start
        and int(s / 1000 * fs) + fw_end < rec_samples
    ]

    # Resolve channel identifiers
    all_ch_ids = rec_proc.get_channel_ids()

    if all_channels_mode:
        # All-channel mode: extract every mapped channel
        if probe == "UA":
            # Only keep channels that were successfully mapped (ua_elec > 0)
            valid_idx = np.where(ua_elec > 0)[0]
            target_ch_ids = [all_ch_ids[i] for i in valid_idx]
            target_ch_elecs = [int(ua_elec[i]) for i in valid_idx]
            target_ch_regions = [int(ua_region[i]) for i in valid_idx]
        else:
            target_ch_ids = list(all_ch_ids)
            target_ch_elecs = list(range(len(all_ch_ids)))
            target_ch_regions = []
        # Reference channel for pulse detection
        if reference_ch is not None and probe == "UA":
            ref_matches = np.where(ua_elec == reference_ch)[0]
            if ref_matches.size == 0:
                print(f"Warning: reference ch {reference_ch} not mapped; using first channel.")
                ref_ch_id = target_ch_ids[0]
            else:
                ref_ch_id = all_ch_ids[ref_matches[0]]
        else:
            ref_ch_id = target_ch_ids[0]
        n_channels = len(target_ch_ids)
        print(f"-> All-channel mode: {n_channels} channels, "
              f"pulse detection on ref ch {reference_ch or 'first'}")
    else:
        # Single-channel mode (original behaviour)
        if probe == "UA":
            matches = np.where(ua_elec == target_ch)[0]
            if matches.size == 0:
                print(f"Warning: UA ch {target_ch} not mapped in this recording.")
                return
            ref_ch_id = all_ch_ids[matches[0]]
        else:
            ref_ch_id = all_ch_ids[target_ch]
        target_ch_ids = [ref_ch_id]
        target_ch_elecs = [target_ch]
        n_channels = 1

    # 2d. Pre-allocate extraction buffers
    full_raw_array    = np.zeros((len(valid_stims), full_n_time), dtype=np.float32)
    micro_signal_array = np.zeros(
        (len(valid_stims) * (num_pulses + 10), micro_n_time, n_channels),
        dtype=np.float32,
    )
    pulse_counter    = 0
    micro_map        = []          # (trial_idx, start_frame, end_frame)
    interval_idx     = int(pulse_interval_ms / 1000 * fs)
    search_radius_idx = int((pulse_interval_ms * 0.4) / 1000 * fs)
    train_start_idx  = max(0,  int((-5.0 - window_ms[0]) / 1000 * fs))
    train_end_idx    = min(full_n_time, int((stim_dur + 5.0 - window_ms[0]) / 1000 * fs))
    refine_samp      = (int(round(ua_refine_search_ms / 1000.0 * fs))
                        if ua_refine_search_ms > 0 else 3)
    coarse_radius    = max(refine_samp, 3)

    drift_logs = []

    # 2e. Per-trial extraction loop
    for i, s_ms in enumerate(valid_stims):
        center = int(s_ms / 1000 * fs)

        # ── Coarse search: find the first pulse in a wider window ────
        search_margin = int(20.0 / 1000 * fs)
        start_search  = center + fw_start - search_margin
        end_search    = center + fw_end + search_margin
        if start_search < 0 or end_search > rec_samples:
            continue

        tr_search = rec_proc.get_traces(
            start_frame=start_search, end_frame=end_search,
            channel_ids=[ref_ch_id], return_in_uV=True,
        ).squeeze()

        tr_diff_search  = np.abs(np.diff(tr_search, prepend=tr_search[0]))
        predicted_centre = -fw_start + search_margin
        slop = int(5.0 / 1000 * fs)
        win_lo = max(0, predicted_centre - slop)
        win_hi = min(len(tr_search), predicted_centre + slop)

        peak_in_win = np.argmax(tr_diff_search[win_lo:win_hi])
        anchor_idx  = win_lo + peak_in_win
        max_diff    = tr_diff_search[anchor_idx]

        if max_diff < 5.0:                       # no credible artifact
            continue

        # Scan left→right for the first threshold-crossing derivative
        first_pulse_in_search = None
        for k in range(win_lo, win_hi):
            if tr_diff_search[k] > 0.4 * max_diff:
                first_pulse_in_search = k
                break
        if first_pulse_in_search is None:
            first_pulse_in_search = anchor_idx

        # Refine coarse centre to the earliest significant deflection
        r_lo = max(0, first_pulse_in_search - coarse_radius)
        r_hi = min(len(tr_search), first_pulse_in_search + coarse_radius + 1)
        coarse_centre = r_lo + _find_first_significant_peak(tr_search[r_lo:r_hi])

        # ── Shift the extraction window so pulse 1 ≈ time 0 ─────────
        true_start = start_search + coarse_centre
        final_start = true_start + fw_start
        final_end   = true_start + fw_end
        if final_start < 0 or final_end > rec_samples:
            continue

        # Extract reference channel for pulse detection + full trace plot
        tr = rec_proc.get_traces(
            start_frame=final_start, end_frame=final_end,
            channel_ids=[ref_ch_id], return_in_uV=True,
        ).squeeze()
        full_raw_array[i, :] = tr

        # If all-channels mode, also extract all channels for this window
        if all_channels_mode:
            tr_all = rec_proc.get_traces(
                start_frame=final_start, end_frame=final_end,
                channel_ids=target_ch_ids, return_in_uV=True,
            )  # shape (n_time, n_channels)
            
            # Save the full trace of all channels for trial 0 to use in the geographic plot later
            if i == 0:
                full_raw_array_all_t0 = tr_all.copy()

        if train_end_idx <= train_start_idx:
            continue

        # ── Forward-walk to identify all pulses in the train ─────────
        tr_diff          = np.abs(np.diff(tr, prepend=tr[0]))
        anchor_final     = -fw_start
        local_max_diff  = np.max(
            tr_diff[max(0, anchor_final - refine_samp)
                    :min(full_n_time, anchor_final + refine_samp + 1)]
        )

        if i == 0 or i == len(valid_stims) - 1:
            shift_ms = (true_start - center) / fs * 1000.0
            drift_logs.append(f"Trial {i}: window shifted {shift_ms:+.2f} ms")

        pulse_centres = [anchor_final]
        curr = anchor_final
        while True:
            nxt = curr + interval_idx
            if nxt > train_end_idx:
                break
            s_lo = max(0, nxt - search_radius_idx)
            s_hi = min(full_n_time, nxt + search_radius_idx)
            if s_hi <= s_lo:
                break
            curr = s_lo + np.argmax(tr_diff[s_lo:s_hi])
            if tr_diff[curr] < 0.15 * local_max_diff:
                break
            pulse_centres.append(curr)

        pulse_centres = sorted(set(pulse_centres))

        # ── Refine each pulse centre to its first significant peak ───
        for p_idx in pulse_centres:
            r_lo = max(0, p_idx - refine_samp)
            r_hi = min(full_n_time, p_idx + refine_samp + 1)
            if r_lo >= r_hi:
                continue
            true_centre = r_lo + _find_first_significant_peak(tr[r_lo:r_hi])

            p_start = true_centre + mw_start
            p_end   = true_centre + mw_end
            if p_start >= 0 and p_end <= full_n_time:
                if all_channels_mode:
                    micro_signal_array[pulse_counter, :, :] = tr_all[p_start:p_end, :]
                else:
                    micro_signal_array[pulse_counter, :, 0] = tr[p_start:p_end]
                micro_map.append((i, p_start, p_end))
                pulse_counter += 1

    micro_signal_array = micro_signal_array[:pulse_counter]

    if drift_logs:
        print("\n--- Drift Logs ---")
        for entry in drift_logs:
            print(f"  {entry}")
        print("------------------\n")

    # ══════════════════════════════════════════════════════════════════
    # §3  PCA ARTIFACT CORRECTION — portable to UA_BR_analysis_threshold
    # ══════════════════════════════════════════════════════════════════
    #
    # INPUT:  micro_signal_array  — (n_pulses, n_time, n_channels)
    #
    # OUTPUT: micro_corrected     — (n_pulses, n_time, n_channels)
    #
    # When n_channels == 1: standard per-channel batch PCA.
    # When n_channels  > 1: cross-channel IPCA — a single IncrementalPCA
    #   model is shared across channels via sequential partial_fit.
    #   Each channel is then corrected using the shared subspace.
    # ──────────────────────────────────────────────────────────────────

    if n_channels == 1:
        # Single-channel batch PCA (original behaviour)
        corrector = IPCA_Artifact_Correction(rank=ipca_rank)
        micro_corrected, _ = corrector.ipca_all(micro_signal_array)
    else:
        # Cross-channel IPCA using existing artifact_correction.py methods,
        # learned separately for each brain region (SMA, PMd, M1i, M1s).
        from RCP_analysis.python.functions.artifact_correction import Template
        corrector = IPCA_Artifact_Correction(rank=ipca_rank)
        micro_corrected = np.zeros_like(micro_signal_array)
        
        # Group channel indices by region
        region_to_idxs = {}
        if probe == "UA":
            for ch_idx, reg_idx in enumerate(target_ch_regions):
                reg_name = ua_region_names[reg_idx] if reg_idx >= 0 else "Unknown"
                region_to_idxs.setdefault(reg_name, []).append(ch_idx)
        else:
            region_to_idxs["All Channels"] = list(range(n_channels))
            
        n_stim, n_time, _ = micro_signal_array.shape
        
        print("Cross-channel IPCA by region:")
        for reg, idxs in region_to_idxs.items():
            if not idxs: continue
            
            # Pool all pulses from channels within this region
            reg_signal = micro_signal_array[:, :, idxs]  # (n_stim, n_time, n_reg_ch)
            pooled_signal = np.transpose(reg_signal, (0, 2, 1)).reshape(-1, n_time)
            
            # Learn the shared template for this specific region
            reg_template = Template()
            _, reg_template = corrector.ipca_template_per_channel(pooled_signal, reg_template)
            
            print(f"  [{reg}] Shared subspace learned from {len(idxs)} channels.")
            
            # Apply this region's template to each channel in the region
            for ch_idx in idxs:
                signal_ch = micro_signal_array[:, :, ch_idx].copy()
                baseline = np.mean(signal_ch[:, :3], axis=1, keepdims=True)
                centered = signal_ch - baseline
                
                corrected_centered = corrector.apply_template(centered, reg_template)
                micro_corrected[:, :, ch_idx] = corrected_centered + baseline

    # ══════════════════════════════════════════════════════════════════
    # End of PCA artifact correction block
    # ══════════════════════════════════════════════════════════════════

    # ─────────────────────────────────────────────────────────────────
    # §4  DEBUG VISUALISATION — micro-window overlay
    # ─────────────────────────────────────────────────────────────────
    if debug_single_pulse and pulse_counter > 0:
        print("Saving debug overlay plot...")
        fig_dbg, ax_dbg = plt.subplots(1, 2, figsize=(24, 12), sharey=True)

        # Left panel: full trial trace with IPCA windows highlighted
        first_trial   = micro_map[0][0]
        full_time_ms  = np.arange(fw_start, fw_end) / fs * 1000
        ax_dbg[0].plot(full_time_ms, full_raw_array[first_trial], color="k", lw=1)

        label_added = False
        for t, ps, pe in micro_map:
            if t == first_trial:
                lbl = 'IPCA "Learning" Window' if not label_added else None
                ax_dbg[0].axvspan(
                    full_time_ms[ps], full_time_ms[pe - 1],
                    color="r", alpha=0.3, label=lbl,
                )
                label_added = True

        ax_dbg[0].set_title(f"Trial {first_trial + 1} | Full Trace")
        ax_dbg[0].set_xlabel("Time (ms)")
        ax_dbg[0].set_ylabel("Amplitude (µV)")
        ax_dbg[0].axvline(0, color="g", ls="--")
        ax_dbg[0].legend()

        # Right panel: all micro-windows for the first trial
        trial_idxs  = [j for j, (t, _, _) in enumerate(micro_map)
                       if t == first_trial]
        micro_t_ms  = np.arange(mw_start, mw_end) / fs * 1000.0
        # Use reference channel (index 0) for plotting
        plot_ch = 0
        raw_micro   = micro_signal_array[trial_idxs, :, plot_ch]
        corr_micro  = micro_corrected[trial_idxs, :, plot_ch]
        art_micro   = raw_micro - corr_micro

        for j in range(len(trial_idxs)):
            ax_dbg[1].plot(micro_t_ms, raw_micro[j],  color="k", alpha=0.3)
            ax_dbg[1].plot(micro_t_ms, art_micro[j],  color="r", alpha=0.3, ls="--")
            ax_dbg[1].plot(micro_t_ms, corr_micro[j], color="b", alpha=0.3)

        ax_dbg[1].plot(micro_t_ms, raw_micro.mean(0),  color="k",    lw=3, label="Mean Raw Pulse")
        ax_dbg[1].plot(micro_t_ms, art_micro.mean(0),  color="r",    lw=3, ls="--", label="Mean Learned Template")
        ax_dbg[1].plot(micro_t_ms, corr_micro.mean(0), color="cyan", lw=3, label="Mean Cleaned Signal")

        ax_dbg[1].set_title("All Pulses (Raw + Template + Cleaned)")
        ax_dbg[1].set_xlabel("Time around pulse centre (ms)")
        ax_dbg[1].axvline(0, color="k", ls=":", alpha=0.5, label="Estimated Centre")
        ax_dbg[1].legend()

        fig_dbg.suptitle(
            f"IPCA All Pulses Extraction Debug | Ch {ch_label}", fontsize=16,
        )
        fig_dbg.tight_layout()
        viz_path = FIG_DIR / f"debug_IPCA_Ch{ch_label}_cond{condition}_all_pulses_viz.png"
        fig_dbg.savefig(viz_path)
        plt.close(fig_dbg)
        print(f"Saved: {viz_path}")

    # ─────────────────────────────────────────────────────────────────
    # §5  FULL-TRACE RECONSTRUCTION & SUMMARY FIGURE
    # ─────────────────────────────────────────────────────────────────
    full_corr_array     = full_raw_array.copy()
    full_artifact_array = np.zeros_like(full_raw_array)

    for p_idx, (trial, ps, pe) in enumerate(micro_map):
        full_corr_array[trial, ps:pe]     = micro_corrected[p_idx, :, 0]
        full_artifact_array[trial, ps:pe] = (micro_signal_array[p_idx, :, 0]
                                             - micro_corrected[p_idx, :, 0])

    plot_idxs = np.linspace(
        0, len(valid_stims) - 1, min(trials_plot, len(valid_stims)), dtype=int,
    )
    time_axis = np.arange(fw_start, fw_end) / fs * 1000
    col_labels = ["Raw Signal", "IPCA Artifact Template", "Corrected Signal"]

    fig, axes = plt.subplots(
        len(plot_idxs), 3,
        figsize=(15, 3 * len(plot_idxs)),
        sharex=True, sharey=True,
    )
    if len(plot_idxs) == 1:
        axes = np.array([axes])

    for r, ti in enumerate(plot_idxs):
        axes[r, 0].plot(time_axis, full_raw_array[ti],      color="k", lw=1)
        axes[r, 1].plot(time_axis, full_artifact_array[ti],  color="r", lw=1)
        axes[r, 2].plot(time_axis, full_corr_array[ti],      color="b", lw=1)
        for c in range(3):
            ax = axes[r, c]
            ax.axvline(0, color="g", ls="--")
            if stim_dur > 0:
                ax.axvline(stim_dur, color="g", ls=":", alpha=0.5)
            if r == 0:
                ax.set_title(col_labels[c])
            if c == 0:
                ax.set_ylabel(f"Trial {ti + 1}\n(µV)")

    axes[0, 0].set_xlim(window_ms)
    fig.suptitle(
        f"IPCA Artifact Correction (rank {ipca_rank}) "
        f"| Ch {ch_label} | Cond {condition}",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = FIG_DIR / f"debug_IPCA_Ch{ch_label}_cond{condition}_rank{ipca_rank}.png"
    fig.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────
    # §5B CROSS-CHANNEL SUMMARY FIGURE (ALL-CHANNELS MODE ONLY)
    # ─────────────────────────────────────────────────────────────────
    if all_channels_mode and n_channels > 1:
        print("Saving multi-channel summary plots per region...")
        
        # Group channel indices by region again for plotting
        region_to_idxs = {}
        if probe == "UA":
            for ch_idx, reg_idx in enumerate(target_ch_regions):
                reg_name = ua_region_names[reg_idx] if reg_idx >= 0 else "Unknown"
                region_to_idxs.setdefault(reg_name, []).append(ch_idx)
        else:
            region_to_idxs["All Channels"] = list(range(n_channels))
            
        trial_to_plot = 0
        col_labels = ["Raw Signal", "IPCA Artifact Template", "Corrected Signal", "Filtered Corrected"]
        
        from scipy.signal import butter, sosfiltfilt
        sos = butter(4, 300.0, btype='high', fs=fs, output='sos')
        
        for reg, sample_ch_idxs in region_to_idxs.items():
            if not sample_ch_idxs:
                continue
                
            n_sample_channels = len(sample_ch_idxs)
            print(f"  Generating plot for region {reg} with {n_sample_channels} channels...")
            
            fig_multi, axes_multi = plt.subplots(
                n_sample_channels, 4,
                figsize=(18, 2.5 * n_sample_channels),
                sharex=True, sharey=False,
            )
            if n_sample_channels == 1:
                axes_multi = np.array([axes_multi])
                
            # Manually share Y axes across all rows for the first 3 columns (matching previous sharey=True behaviour)
            ref_ax = axes_multi[0, 0]
            for r in range(n_sample_channels):
                for c in range(3):
                    if r == 0 and c == 0: continue
                    axes_multi[r, c].sharey(ref_ax)
                
            for r, ch_idx in enumerate(sample_ch_idxs):
                ch_name = f"Ch {target_ch_elecs[ch_idx]}" if probe == "UA" else f"Ch {ch_idx}"
                
                # Reconstruct the full trace for this specific channel for Trial 0
                chan_full_raw = full_raw_array_all_t0[:, ch_idx].copy()
                chan_full_corr = chan_full_raw.copy()
                chan_full_art = np.zeros(full_n_time, dtype=np.float32)
                
                for p_idx, (m_trial, ps, pe) in enumerate(micro_map):
                    if m_trial == trial_to_plot:
                        raw_mc = micro_signal_array[p_idx, :, ch_idx]
                        corr_mc = micro_corrected[p_idx, :, ch_idx]
                        art_mc = raw_mc - corr_mc
                        
                        chan_full_corr[ps:pe] = corr_mc
                        chan_full_art[ps:pe] = art_mc
                
                # Apply high-pass filter to the corrected signal
                chan_full_filt = sosfiltfilt(sos, chan_full_corr)
                
                axes_multi[r, 0].plot(time_axis, chan_full_raw,      color="k", lw=1)
                axes_multi[r, 1].plot(time_axis, chan_full_art,  color="r", lw=1)
                axes_multi[r, 2].plot(time_axis, chan_full_corr,      color="b", lw=1)
                axes_multi[r, 3].plot(time_axis, chan_full_filt,      color="purple", lw=1)
                
                # Enforce fixed y-limits for the filtered column
                axes_multi[r, 3].set_ylim(-80, 80)
                
                for c in range(4):
                    ax = axes_multi[r, c]
                    ax.axvline(0, color="g", ls="--")
                    if stim_dur > 0:
                        ax.axvline(stim_dur, color="g", ls=":", alpha=0.5)
                    if r == 0:
                        ax.set_title(col_labels[c])
                    if c == 0:
                        ax.set_ylabel(f"{ch_name}\n(µV)")
    
            axes_multi[0, 0].set_xlim(window_ms)
            fig_multi.suptitle(
                f"Region: {reg} | IPCA Correction (rank {ipca_rank}) | Cond {condition} | Trial 1",
                y=0.99 if n_sample_channels < 10 else 1.0 - (0.5 / n_sample_channels),
            )
            fig_multi.tight_layout(rect=[0, 0, 1, 0.98 if n_sample_channels < 10 else 1.0 - (1.0 / n_sample_channels)])
    
            multi_out_path = FIG_DIR / f"debug_IPCA_Region_{reg}_cond{condition}_rank{ipca_rank}.png"
            fig_multi.savefig(multi_out_path)
            print(f"Saved: {multi_out_path}")
            plt.close(fig_multi)

    # ─────────────────────────────────────────────────────────────────
    # §6  RETURN DATA FOR NOTEBOOK / INTERACTIVE USE
    # ─────────────────────────────────────────────────────────────────
    return {
        "full_raw_array":      full_raw_array,
        "full_corr_array":     full_corr_array,
        "full_artifact_array": full_artifact_array,
        "micro_signal_array":  micro_signal_array,
        "micro_corrected":     micro_corrected,
        "micro_map":           micro_map,
        "time_axis":           time_axis,
        "mw_start":            mw_start,
        "mw_end":              mw_end,
        "fw_start":            fw_start,
        "fw_end":              fw_end,
        "fs":                  fs,
        "window_ms":           window_ms,
        "stim_dur":            stim_dur,
    }

if __name__ == "__main__":
    run_ipca_debug(
        condition=10, 
        probe="NPRW",
        target_ch=10, # or "all"
        ipca_rank=10,
        window_ms=(-20.0, 40.0),
        pulse_window_ms=(-0.4, 0.5), # utah - -0.2 to 0.3ms
        apply_hpf=False,
        debug_single_pulse=True,
        use_behavior_filter=False,
        reference_ch=124,
    )
