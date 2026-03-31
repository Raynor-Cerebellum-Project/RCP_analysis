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
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
from probeinterface import Probe
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt
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


def _exp_model(t: np.ndarray, A: float, tau: float) -> np.ndarray:
    """Single-term exponential decay model: A * exp(-t / tau)."""
    return A * np.exp(-t / tau)


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
    exp_subtract: bool = True,
    exp_fit_guard_ms: float = 1.0,
    exp_n_tau_fits: int = 5,
    exp_tau_bounds_ms: tuple = (0.1, 30.0),
    exp_tau_seed_ms: float = 5.0,
    plot_rank_variance: bool = False,
    plot_learning_rate_sweep: bool = False,
    override_freq: float = None,
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
    plot_rank_variance : bool
        If True and running in single-channel mode, plots the Cumulative 
        Explained Variance vs IPCA Rank to help find the optimal elbow.
    plot_learning_rate_sweep : bool
        If True and running in single-channel mode, sweeps learning rates 
        (0.1 to 1.0) and plots the residual temporal variance/MSE.

    Returns
    -------
    dict
        All extracted arrays and metadata for downstream analysis.
    """
    all_channels_mode = isinstance(target_ch, str) and target_ch.lower() == "all"
    ch_label = "ALL" if all_channels_mode else str(target_ch)
    print(f"Running IPCA Debug | Cond: {condition} | Probe: {probe} | Ch: {ch_label} | Rank: {ipca_rank}")

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
    if override_freq is not None:
        stim_freq = float(override_freq)
    else:
        stim_freq = float(meta.get("Freq", 400.0))
        
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
            target_ch_id = all_ch_ids[matches[0]]
            
            # Use reference channel for pulse detection if provided
            if reference_ch is not None:
                ref_matches = np.where(ua_elec == reference_ch)[0]
                if ref_matches.size == 0:
                    print(f"Warning: reference ch {reference_ch} not mapped; using target channel.")
                    ref_ch_id = target_ch_id
                else:
                    ref_ch_id = all_ch_ids[ref_matches[0]]
            else:
                ref_ch_id = target_ch_id
        else:
            target_ch_id = all_ch_ids[target_ch]
            ref_ch_id = target_ch_id if reference_ch is None else all_ch_ids[reference_ch]
            
        target_ch_ids = [target_ch_id]
        target_ch_elecs = [target_ch]
        n_channels = 1
        print(f"-> Single-channel mode: Target Ch {target_ch}, Pulse detection on ref ch {reference_ch or target_ch}")

    # 2d. Pre-allocate extraction buffers
    full_raw_array    = np.zeros((len(valid_stims), full_n_time), dtype=np.float32)
    full_orig_array   = np.zeros((len(valid_stims), full_n_time), dtype=np.float32)  # true raw, never overwritten
    micro_signal_array = np.zeros(
        (len(valid_stims) * (num_pulses + 10), micro_n_time, n_channels),
        dtype=np.float32,
    )
    pulse_counter     = 0
    micro_map         = []          # (trial_idx, start_frame, end_frame)
    interval_idx      = int(pulse_interval_ms / 1000 * fs)
    search_radius_idx = int((pulse_interval_ms * 0.4) / 1000 * fs)
    train_start_idx   = max(0,  int((-5.0 - window_ms[0]) / 1000 * fs))
    train_end_idx     = min(full_n_time, int((stim_dur + 5.0 - window_ms[0]) / 1000 * fs))
    refine_samp       = (int(round(ua_refine_search_ms / 1000.0 * fs))
                         if ua_refine_search_ms > 0 else 3)
    coarse_radius     = max(refine_samp, 3)

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

        # Extract reference channel for pulse detection ONLY
        tr = rec_proc.get_traces(
            start_frame=final_start, end_frame=final_end,
            channel_ids=[ref_ch_id], return_in_uV=True,
        ).squeeze()

        # Extract actual target channels for processing and final plot
        tr_all = rec_proc.get_traces(
            start_frame=final_start, end_frame=final_end,
            channel_ids=target_ch_ids, return_in_uV=True,
        )  # shape (n_time, n_channels)
        
        # Store target channel 0 as the full raw array for visualisation
        full_raw_array[i, :]  = tr_all[:, 0]
        full_orig_array[i, :] = tr_all[:, 0]

        # Save trial 0 raw all-channels for the geographic plot later.
        if all_channels_mode and i == 0:
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
            if len(pulse_centres) >= num_pulses:
                break
                
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

        # ── Pass 1: Refine each pulse centre to its first significant peak ───
        true_centres = []   # refined pulse sample indices within full trial
        for p_idx in pulse_centres:
            r_lo = max(0, p_idx - refine_samp)
            r_hi = min(full_n_time, p_idx + refine_samp + 1)
            if r_lo >= r_hi:
                continue
            tc = r_lo + _find_first_significant_peak(tr[r_lo:r_hi])
            true_centres.append(tc)


        # Per-pulse exponential decay subtraction.
        # Fitting window for pulse k: [tc_k + mw_end, tc_{k+1} + mw_start],
        # i.e. the gap between adjacent IPCA windows. Last pulse uses the same
        # width (one stim interval). Fit is done on a DC-zeroed segment so that
        # the large electrode baseline offset does not bias the tau estimate.
        tr_exp_corrected = tr.copy()
        tr_all_exp_corrected = tr_all.copy()

        if exp_subtract and probe == "NPRW" and len(true_centres) > 0:
            tau_lo      = float(exp_tau_bounds_ms[0]) / 1000.0
            tau_hi      = float(exp_tau_bounds_ms[1]) / 1000.0
            tau_seed    = float(exp_tau_seed_ms) / 1000.0
            guard_samps = max(1, int(exp_fit_guard_ms / 1000.0 * fs))

            # Re-fit (A, τ) at exp_n_tau_fits evenly spaced checkpoints across
            # the pulse train; use fast linear A-fit between checkpoints.
            n_pulses      = len(true_centres)
            n_fits        = max(1, min(exp_n_tau_fits, n_pulses))
            refit_interval = max(1, n_pulses // n_fits)
            tau_fitted: float | None = None

            # We build a continuous baseline for the entire trial to subtract, 
            # so that no DC jumps are introduced at the IPCA micro-window boundaries.
            baseline_single = np.zeros(full_n_time, dtype=float)
            valid_mask      = np.zeros(full_n_time, dtype=bool)
            baseline_all = np.zeros((full_n_time, n_channels), dtype=float)

            for k, tc0 in enumerate(true_centres):
                fit_start = tc0 + mw_end - guard_samps
                if k + 1 < len(true_centres):
                    fit_end = true_centres[k + 1] + mw_start + guard_samps
                else:
                    fit_end = min(tc0 + interval_idx + mw_start + guard_samps, full_n_time)

                if fit_end <= fit_start + 3:
                    continue

                t_fit      = np.arange(fit_end - fit_start, dtype=float) / fs
                ref_seg    = tr_exp_corrected[fit_start:fit_end].astype(float)
                ref_zeroed = ref_seg - ref_seg[-1]

                if tau_fitted is None or k % refit_interval == 0:
                    # Checkpoint: full two-parameter fit to refresh τ.
                    try:
                        (A_ref, tau_fitted), _ = curve_fit(
                            _exp_model, t_fit, ref_zeroed,
                            p0=[float(ref_zeroed[0]), tau_fitted if tau_fitted else tau_seed],
                            bounds=([-np.inf, tau_lo], [np.inf, tau_hi]),
                            maxfev=500,
                        )
                    except Exception:
                        if tau_fitted is None:
                            tau_fitted = tau_seed
                        A_ref = float(ref_zeroed[0])
                else:
                    # Between checkpoints: linear A-fit only (τ fixed).
                    exp_vec = np.exp(-t_fit / tau_fitted)
                    A_ref   = float(np.dot(ref_zeroed, exp_vec) / (np.dot(exp_vec, exp_vec) + 1e-12))

                exp_vec   = np.exp(-t_fit / tau_fitted)
                sub_curve = A_ref * exp_vec
                
                baseline_single[fit_start:fit_end] = sub_curve
                valid_mask[fit_start:fit_end] = True

                denom        = np.dot(exp_vec, exp_vec) + 1e-12
                block        = tr_all_exp_corrected[fit_start:fit_end].astype(float)
                block_zeroed = block - block[-1]               # endpoint-zero each channel
                A_all        = (block_zeroed.T @ exp_vec) / denom   # (n_ch,)
                sub_all      = np.outer(exp_vec, A_all)             # (n_samp, n_ch)
                baseline_all[fit_start:fit_end, :] = sub_all

            # Interpolate the baseline across the gaps (the IPCA windows) 
            # to guarantee perfect continuity, then subtract.
            x_all = np.arange(full_n_time)
            valid_idx = np.where(valid_mask)[0]
            if len(valid_idx) > 0:
                baseline_single = np.interp(x_all, valid_idx, baseline_single[valid_idx])
                tr_exp_corrected -= baseline_single.astype(tr_exp_corrected.dtype)
                
                for ch in range(n_channels):
                    baseline_all[:, ch] = np.interp(x_all, valid_idx, baseline_all[valid_idx, ch])
                tr_all_exp_corrected -= baseline_all.astype(tr_all_exp_corrected.dtype)

        # Extract IPCA micro-windows from the exp-corrected trace.
        for tc in true_centres:
            p_start = tc + mw_start
            p_end   = tc + mw_end
            if p_start >= 0 and p_end <= full_n_time:
                micro_signal_array[pulse_counter, :, :] = tr_all_exp_corrected[p_start:p_end, :]
                micro_map.append((i, p_start, p_end))
                pulse_counter += 1

        # Store corrected full-trial trace for visualisation (single channel target trace)
        full_raw_array[i, :] = tr_all_exp_corrected[:, 0]


    micro_signal_array = micro_signal_array[:pulse_counter]

    if pulse_counter == 0:
        print(f"Warning: No valid IPCA micro-windows found! (pulse_counter=0 on channel {target_ch})")
        return

    if drift_logs:
        print("\n--- Drift Logs ---")
        for entry in drift_logs:
            print(f"  {entry}")
        print("------------------\n")

    # --- §3  PCA artifact correction (portable to UA_BR_analysis_threshold) ---
    print("Running Pass 3: IPCA training and application...")
    from RCP_analysis.python.functions.artifact_correction import Template

    corrector = IPCA_Artifact_Correction(rank=ipca_rank)
    micro_corrected = np.zeros_like(micro_signal_array)

    if n_channels == 1:
        sig      = micro_signal_array[:, :, 0]
        baseline = np.mean(sig[:, :3], axis=1, keepdims=True)
        centered = sig - baseline
        # Train on reversed order: last trial first, trial 0 last. By the time
        # the template encounters the first trial's first pulse, it is already
        # well-established from all subsequent trials.
        tpl = Template()
        _, tpl = corrector.ipca_template_per_channel(centered[::-1], tpl)
        micro_corrected[:, :, 0] = corrector.apply_template(centered, tpl) + baseline

        # --- DIAGNOSTIC PLOTS (Single Channel Only) ---
        if plot_rank_variance or plot_learning_rate_sweep:
            from sklearn.decomposition import IncrementalPCA
            import os
            
            # Use raw mean-centered pulses for diagnostics
            X_diag = centered.copy()
            n_samples_diag = X_diag.shape[0]

            if plot_rank_variance:
                print("  [Diag] Computing Cumulative Explained Variance vs Rank...")
                max_rank_test = min(X_diag.shape[1], 30) # test up to 30 components
                ipca_var = IncrementalPCA(n_components=max_rank_test)
                ipca_var.fit(X_diag)
                
                cum_var = np.cumsum(ipca_var.explained_variance_ratio_)
                
                fig_var, ax_var = plt.subplots(figsize=(8, 6))
                ax_var.plot(range(1, max_rank_test + 1), cum_var, marker='o', linestyle='-', color='b')
                ax_var.axhline(0.90, color='r', linestyle='--', label='90% Variance')
                ax_var.axhline(0.95, color='g', linestyle='--', label='95% Variance')
                ax_var.set_xlabel('IPCA Rank (Number of Components)')
                ax_var.set_ylabel('Cumulative Explained Variance Ratio')
                ax_var.set_title(f'IPCA Rank Selection | Ch {ch_label}')
                ax_var.set_ylim([0, 1.05])
                ax_var.set_xticks(range(1, max_rank_test + 1, max(1, max_rank_test//10)))
                ax_var.grid(True, alpha=0.3)
                ax_var.legend()
                
                out_path_var = FIG_DIR / f"debug_IPCA_Ch{ch_label}_cond{condition}_variance_vs_rank.png"
                fig_var.savefig(out_path_var, dpi=150)
                plt.close(fig_var)
                print(f"  [Diag] Saved Rank Variance plot -> {out_path_var.name}")

            if plot_learning_rate_sweep:
                print("  [Diag] Sweeping Learning Rates (0.1 to 1.0)...")
                lr_rates = np.arange(0.1, 1.1, 0.1)
                
                # We need trial-level structure for online tracking. 
                # micro_map maps pulse -> trial.
                trial_indices = [item[0] for item in micro_map]
                unique_trials = sorted(list(set(trial_indices)))
                
                mse_results = []
                target_rank = corrector.rank
                
                for lr in lr_rates:
                    # Fresh template for this learning rate
                    test_tpl = Template()
                    test_corrector = IPCA_Artifact_Correction(rank=target_rank)
                    total_mse = 0.0
                    total_pulses = 0
                    
                    pulse_buffer = []
                    
                    # Simulate streaming trial by trial
                    for tr in unique_trials:
                        # Pulses for this trial
                        idx_tr = [i for i, t in enumerate(trial_indices) if t == tr]
                        if not idx_tr: continue
                        
                        chunk = X_diag[idx_tr]
                        
                        # Apply OLD template to measure residual BEFORE updating
                        # (To mimic real online performance where we only have past info)
                        if test_tpl.weights is not None:
                            # Use rank=test_tpl.weights.shape[0] just in case
                            temp_corrector = IPCA_Artifact_Correction(rank=target_rank)
                            corrected_chunk = temp_corrector.apply_template(chunk, test_tpl)
                            total_mse += np.sum(corrected_chunk**2)
                            total_pulses += len(chunk)
                            
                        # Buffer pulses until we have enough to satisfy IncrementalPCA rank math
                        pulse_buffer.append(chunk)
                        pooled_chunk = np.vstack(pulse_buffer)
                        
                        if len(pooled_chunk) >= target_rank:
                            fresh_corrector = IPCA_Artifact_Correction(rank=target_rank)
                            _, test_tpl = fresh_corrector.ipca_template_per_channel(pooled_chunk, test_tpl, learning_rate=lr)
                            pulse_buffer = [] # Clear buffer after learning
                        
                    # Calculate mean MSE per pulse
                    mean_mse = total_mse / max(1, total_pulses)
                    mse_results.append(mean_mse)
                    
                fig_lr, ax_lr = plt.subplots(figsize=(8, 6))
                ax_lr.plot(lr_rates, mse_results, marker='s', linestyle='-', color='m')
                ax_lr.set_xlabel('Learning Rate (Alpha)')
                ax_lr.set_ylabel('Sequential Residual MSE (Online Error)')
                ax_lr.set_title(f'IPCA Learning Rate Optimization | Ch {ch_label}')
                ax_lr.grid(True, alpha=0.3)
                
                # Mark minimum
                best_idx = np.argmin(mse_results)
                ax_lr.plot(lr_rates[best_idx], mse_results[best_idx], marker='*', color='gold', markersize=15, 
                           label=f'Best LR = {lr_rates[best_idx]:.1f}')
                ax_lr.legend()
                
                out_path_lr = FIG_DIR / f"debug_IPCA_Ch{ch_label}_cond{condition}_learning_rate_sweep.png"
                fig_lr.savefig(out_path_lr, dpi=150)
                plt.close(fig_lr)
                print(f"  [Diag] Saved LR Sweep plot -> {out_path_lr.name}")
    else:
        # Group channels by brain region; learn one shared subspace per region.
        region_to_idxs: dict = {}
        if probe == "UA":
            for ch_idx, reg_idx in enumerate(target_ch_regions):
                reg_name = ua_region_names[reg_idx] if reg_idx >= 0 else "Unknown"
                region_to_idxs.setdefault(reg_name, []).append(ch_idx)
        else:
            region_to_idxs["All Channels"] = list(range(n_channels))

        for reg, idxs in region_to_idxs.items():
            if not idxs:
                continue
            n_time = micro_signal_array.shape[1]
            # Pool all pulses reversed (last trial first): template is well-formed
            # before encountering trial 0's first pulse.
            reg_signal = micro_signal_array[::-1, :, idxs]
            pooled_sig = np.transpose(reg_signal, (0, 2, 1)).reshape(-1, n_time)
            reg_template = Template()
            _, reg_template = corrector.ipca_template_per_channel(pooled_sig, reg_template)
            print(f"  [{reg}] template from {len(idxs)} ch.")


            for ch_idx in idxs:
                sig      = micro_signal_array[:, :, ch_idx].copy()
                baseline = np.mean(sig[:, :3], axis=1, keepdims=True)
                micro_corrected[:, :, ch_idx] = (
                    corrector.apply_template(sig - baseline, reg_template) + baseline
                )

    # --- §5  Full-trace reconstruction and summary figure ---
    # Columns: Raw | (After Exp Subtraction) | IPCA template | Corrected | BPF 300-10 kHz

    def _bandpass(sig, lo, hi, fs_hz):
        sos = butter(4, [lo / (fs_hz / 2), hi / (fs_hz / 2)], btype='band', output='sos')
        return sosfiltfilt(sos, sig.astype(float)).astype(np.float32)

    full_corr_array     = full_raw_array.copy()   # exp-corrected, will be further updated by IPCA
    full_artifact_array = np.zeros_like(full_raw_array)

    for p_idx, (trial, ps, pe) in enumerate(micro_map):
        full_corr_array[trial, ps:pe]     = micro_corrected[p_idx, :, 0]
        full_artifact_array[trial, ps:pe] = (micro_signal_array[p_idx, :, 0]
                                             - micro_corrected[p_idx, :, 0])



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

    # --- §5  Full-trace reconstruction and summary figure ---
    # Columns: Raw | (After Exp Subtraction) | After IPCA | Final Corrected | BPF 300-10 kHz
    # For NPRW: Final = IPCA + (optional Exp subtraction).  For UA: Final = IPCA only.

    def _bandpass(sig, lo, hi, fs_hz):
        sos = butter(4, [lo / (fs_hz / 2), hi / (fs_hz / 2)], btype='band', output='sos')
        return sosfiltfilt(sos, sig.astype(float)).astype(np.float32)

    plot_idxs = np.linspace(
        0, len(valid_stims) - 1, min(trials_plot, len(valid_stims)), dtype=int,
    )
    time_axis = np.arange(fw_start, fw_end) / fs * 1000
    if exp_subtract:
        col_labels = ["Raw Signal", "After Exp Subtraction",
                      "IPCA Artifact Template", "Corrected Signal",
                      "BPF 300–10kHz (Corrected)"]
        row_colors = ["k", "darkorange", "r", "b", "purple"]
    else:
        col_labels = ["Raw Signal",
                      "IPCA Artifact Template", "Corrected Signal",
                      "BPF 300–10kHz (Corrected)"]
        row_colors = ["k", "r", "b", "purple"]
    n_cols = len(col_labels)

    fig, axes = plt.subplots(
        len(plot_idxs), n_cols,
        figsize=(5 * n_cols, 3 * len(plot_idxs)),
        sharex=True,
    )
    if len(plot_idxs) == 1:
        axes = np.array([axes])

    for r, ti in enumerate(plot_idxs):
        bpf_sig = _bandpass(full_corr_array[ti], 300.0, 10000.0, fs)
        if exp_subtract:
            row_data = [
                full_orig_array[ti],       # col 0: true raw
                full_raw_array[ti],        # col 1: after exp subtraction
                full_artifact_array[ti],   # col 2: IPCA template
                full_corr_array[ti],       # col 3: fully corrected
                bpf_sig,                   # col 4: BPF of corrected
            ]
        else:
            row_data = [
                full_orig_array[ti],       # col 0: true raw
                full_artifact_array[ti],   # col 1: IPCA template
                full_corr_array[ti],       # col 2: fully corrected
                bpf_sig,                   # col 3: BPF of corrected
            ]
        for c, (sig, col) in enumerate(zip(row_data, row_colors)):
            ax = axes[r, c]
            ax.plot(time_axis, sig, color=col, lw=0.7)
            ax.axvline(0, color="g", ls="--", lw=0.8)
            if stim_dur > 0:
                ax.axvline(stim_dur, color="g", ls=":", alpha=0.5, lw=0.8)
            if r == 0:
                ax.set_title(col_labels[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(f"Trial {ti + 1}\n(µV)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)

    axes[-1, 0].set_xlim(window_ms)
    for c in range(n_cols):
        axes[-1, c].set_xlabel("Time (ms)", fontsize=8)
    fig.suptitle(
        f"IPCA Artifact Correction (rank {ipca_rank}) "
        f"| Ch {ch_label} | Cond {condition}",
        y=0.99, fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = FIG_DIR / f"debug_IPCA_Ch{ch_label}_cond{condition}_rank{ipca_rank}.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")
    plt.close(fig)

    # --- §5b  Per-region multi-channel figure (all-channels mode only) ---
    if all_channels_mode and n_channels > 1:
        print("Saving multi-channel summary plots per region...")
        region_to_idxs: dict = {}
        if probe == "UA":
            for ch_idx, reg_idx in enumerate(target_ch_regions):
                reg_name = ua_region_names[reg_idx] if reg_idx >= 0 else "Unknown"
                region_to_idxs.setdefault(reg_name, []).append(ch_idx)
        else:
            region_to_idxs["All Channels"] = list(range(n_channels))
            
        trial_to_plot = 0
        col_labels_multi = ["Raw Signal", "IPCA Template", "Corrected", "HPF Corrected"]
        sos_hpf = butter(4, 300.0, btype='high', fs=fs, output='sos')
        
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

            # Share Y axis across the first 3 columns (raw / template / corrected).
            ref_ax = axes_multi[0, 0]
            for r in range(n_sample_channels):
                for c in range(1, 3):
                    axes_multi[r, c].sharey(ref_ax)

            for r, ch_idx in enumerate(sample_ch_idxs):
                ch_name = f"Ch {target_ch_elecs[ch_idx]}" if probe == "UA" else f"Ch {ch_idx}"

                chan_full_raw  = full_raw_array_all_t0[:, ch_idx].copy()
                chan_full_corr = chan_full_raw.copy()
                chan_full_art  = np.zeros(full_n_time, dtype=np.float32)

                for p_idx, (m_trial, ps, pe) in enumerate(micro_map):
                    if m_trial == trial_to_plot:
                        raw_mc  = micro_signal_array[p_idx, :, ch_idx]
                        corr_mc = micro_corrected[p_idx, :, ch_idx]
                        chan_full_corr[ps:pe] = corr_mc
                        chan_full_art[ps:pe]  = raw_mc - corr_mc

                chan_full_filt = sosfiltfilt(sos_hpf, chan_full_corr)

                axes_multi[r, 0].plot(time_axis, chan_full_raw,  color="k",      lw=1)
                axes_multi[r, 1].plot(time_axis, chan_full_art,  color="r",      lw=1)
                axes_multi[r, 2].plot(time_axis, chan_full_corr, color="b",      lw=1)
                axes_multi[r, 3].plot(time_axis, chan_full_filt, color="purple", lw=1)
                axes_multi[r, 3].set_ylim(-80, 80)

                for c in range(4):
                    ax = axes_multi[r, c]
                    ax.axvline(0, color="g", ls="--")
                    if stim_dur > 0:
                        ax.axvline(stim_dur, color="g", ls=":", alpha=0.5)
                    if r == 0:
                        ax.set_title(col_labels_multi[c])
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

    # --- §6  Return data for notebook / interactive use ---
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

    # target_ch = 'all' or int

    # run_ipca_debug(
    #     condition=10,
    #     probe="UA",
    #     trials_plot=5,
    #     target_ch=168,
    #     ipca_rank=10,
    #     window_ms=(-20.0, 40.0),
    #     pulse_window_ms=(-0.3, 0.4),
    #     apply_hpf=False,
    #     debug_single_pulse=True,
    #     use_behavior_filter=False,
    #     reference_ch=124,
    #     plot_rank_variance=True,
    #     plot_learning_rate_sweep=True,
    #     exp_subtract=False,
    #     exp_fit_guard_ms=0.1,
    #     exp_n_tau_fits=10,
    #     exp_tau_bounds_ms=(0.1, 20.0),
    #     exp_tau_seed_ms=5.0,
    # )

    run_ipca_debug(
        condition=10,
        probe="NPRW",
        trials_plot=5,
        target_ch=60,
        ipca_rank=20,
        window_ms=(-20.0, 40.0),
        pulse_window_ms=(-0.3, 0.4),
        apply_hpf=False,
        debug_single_pulse=True,
        use_behavior_filter=False,
        reference_ch=60,
        plot_rank_variance=True,
        plot_learning_rate_sweep=True,
        exp_subtract=False,
        exp_fit_guard_ms=0.1,
        exp_n_tau_fits=10,
        exp_tau_bounds_ms=(0.1, 20.0),
        exp_tau_seed_ms=5.0,
    )
