from __future__ import annotations
from pathlib import Path
from typing import Optional
import gc
import numpy as np

# SpikeInterface
import spikeinterface as si
import spikeinterface.preprocessing as spre
from spikeinterface import sorters, create_sorting_analyzer
from spikeinterface.core import concatenate_recordings
from spikeinterface.exporters import export_to_phy
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from scipy.ndimage import gaussian_filter1d

# Project config
from RCP_analysis import load_experiment_params, resolve_data_root

# BR/UA helpers
from RCP_analysis import (
    list_br_sessions, ua_excel_path,
    load_ns6_spikes, load_UA_mapping_from_excel, apply_ua_mapping_properties,
    build_blackrock_bundle, save_bundle_npz,
)


# INTAN HELPERS HERE

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
OUT_BASE = (REPO_ROOT / "results").resolve()
OUT_BASE.mkdir(parents=True, exist_ok=True)
global_job_kwargs = dict(n_jobs=PARAMS.parallel_jobs, chunk_duration=PARAMS.chunk)
si.set_global_job_kwargs(**global_job_kwargs)

# === FIRING RATE PARAMETERS ===
BIN_MS   = 1.0    # bin width for counts
SIGMA_MS = 20.0   # Gaussian kernel std in ms (tunable here!)
THRESH   = 4.5    # threshold in SD
PEAK_SIGN = "neg" # detect negative-going spikes
# ----------------------------------------

def threshold_mua_rates(
    recording,
    detect_threshold=THRESH,
    peak_sign=PEAK_SIGN,
    bin_ms=BIN_MS,
    sigma_ms=SIGMA_MS,
    n_jobs=4,
):
    """
    Threshold-crossing MUA → binned + smoothed firing rates.

    Returns
    -------
    rate_hz : np.ndarray
        (n_channels, n_bins) Gaussian-smoothed firing rate in Hz.
    t_ms : np.ndarray
        (n_bins,) bin-center times in ms.
    counts : np.ndarray
        (n_channels, n_bins) raw spike counts per bin.
    """
    fs = recording.get_sampling_frequency()
    n_ch = recording.get_num_channels()
    n_seg = recording.get_num_segments()
    bin_samps = int(round(bin_ms * 1e-3 * fs))
    if bin_samps < 1:
        raise ValueError("bin_ms too small for sampling rate")

    # 1) Detect peaks
    noise_levels = si.get_noise_levels(recording)
    peaks = detect_peaks(
        recording,
        method="by_channel_torch",
        detect_threshold=detect_threshold,
        peak_sign=peak_sign,
        noise_levels=noise_levels,
        n_jobs=n_jobs,
    )

    # --- robust field picking across SI versions ---
    def _pick_field(peaks, candidates):
        for f in candidates:
            if f in peaks.dtype.names:
                return f
        raise KeyError(f"Expected one of {candidates}, got {peaks.dtype.names}")

    ch_field   = _pick_field(peaks, ("channel_ind", "channel_index", "channel_id", "channel"))
    samp_field = _pick_field(peaks, ("sample_ind", "sample_index", "sample"))
    seg_field  = _pick_field(peaks, ("segment_ind", "segment_index", "segment"))

    # 2) Bin counts per segment
    counts_all, t_all = [], []
    sigma_bins = max(1e-9, sigma_ms / bin_ms)  # ms → bins
    bin_offset = 0

    for seg in range(n_seg):
        n_samps = recording.get_num_frames(seg)
        seg_bins = int(np.ceil(n_samps / bin_samps))
        counts = np.zeros((n_ch, seg_bins), dtype=np.int32)

        # select peaks for this segment
        if seg_field is not None and seg_field in peaks.dtype.names:
            seg_peaks = peaks[peaks[seg_field] == seg]
        else:
            seg_peaks = peaks if n_seg == 1 else None
            if seg_peaks is None:
                raise RuntimeError("No segment field in peaks for multi-segment recording.")

        if seg_peaks.size > 0:
            ch_idx = seg_peaks[ch_field].astype(np.int64)
            samp   = seg_peaks[samp_field].astype(np.int64)
            bins   = np.clip(samp // bin_samps, 0, seg_bins - 1)
            np.add.at(counts, (ch_idx, bins), 1)

        counts_all.append(counts)
        t_ms = (np.arange(seg_bins) + 0.5 + bin_offset) * bin_ms
        t_all.append(t_ms)
        bin_offset += seg_bins

    counts_cat = np.concatenate(counts_all, axis=1)
    t_cat_ms   = np.concatenate(t_all)

    # 3) Smooth → Hz
    counts_smooth = gaussian_filter1d(counts_cat.astype(float), sigma=sigma_bins, axis=1, mode="nearest")
    rate_hz = counts_smooth * (1000.0 / bin_ms)

    return rate_hz, t_cat_ms, counts_cat

def main(use_br: bool = True, use_intan: bool = False, limit_sessions: Optional[int] = None):
    data_root = resolve_data_root(PARAMS)
    session_folders = list_br_sessions(data_root, PARAMS.blackrock_rel)
    if limit_sessions:
        session_folders = session_folders[:limit_sessions]
    print("Found session folders:", len(session_folders))

    xls = ua_excel_path(REPO_ROOT, PARAMS.probes)
    ua_map = load_UA_mapping_from_excel(xls) if xls else None
    if ua_map is None:
        raise RuntimeError("UA mapping required for mapping on NS6.")

    bundles_out = OUT_BASE / "bundles"
    checkpoint_out = OUT_BASE / "checkpoint"
    checkpoint_out.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    # --- per session: save non-spike bundle + preproc ns6
    for sess in session_folders:
        print(f"=== Session: {sess.name} ===")

        bundle = build_blackrock_bundle(sess)
        save_bundle_npz(sess.name, bundle, bundles_out)

        rec_ns6 = load_ns6_spikes(sess)
        apply_ua_mapping_properties(rec_ns6, ua_map["mapped_nsp"])  # metadata only

        rec_hp  = spre.highpass_filter(rec_ns6, freq_min=float(PARAMS.highpass_hz))
        rec_ref = spre.common_reference(rec_hp, reference="global", operator="median")

        try:
            _ = rec_ref.get_channel_locations()
        except Exception:
            n_ch = rec_ref.get_num_channels()
            locs = np.column_stack([np.arange(n_ch, dtype=float), np.zeros(n_ch, dtype=float)])
            rec_ref.set_channel_locations(locs)
    
        out_geom = checkpoint_out / f"pp_global__{sess.name}__NS6"
        out_geom.mkdir(parents=True, exist_ok=True)
        rec_ref.save(folder=out_geom, overwrite=True)
        print(f"[{sess.name}] (ns6) saved preprocessed -> {out_geom}")

        
        rate_hz, t_ms, counts = threshold_mua_rates(
            rec_ref,
            detect_threshold=THRESH,
            peak_sign=PEAK_SIGN,
            bin_ms=BIN_MS,
            sigma_ms=SIGMA_MS,
            n_jobs=PARAMS.parallel_jobs,
        )

        out_npz = checkpoint_out / f"rates__{sess.name}__bin{int(BIN_MS)}ms_sigma{int(SIGMA_MS)}ms.npz"
        np.savez_compressed(
            out_npz,
            rate_hz=rate_hz.astype(np.float32),
            t_ms=t_ms.astype(np.float32),
            counts=counts.astype(np.uint16),
            meta=dict(
                detect_threshold=THRESH,
                peak_sign=PEAK_SIGN,
                bin_ms=BIN_MS,
                sigma_ms=SIGMA_MS,
                fs=float(rec_ref.get_sampling_frequency()),
                n_channels=int(rec_ref.get_num_channels()),
                session=str(sess.name),
            ),
        )
        print(f"[{sess.name}] saved rate matrix -> {out_npz}")

        # cleanup to keep memory stable on long batches
        del bundle, rec_ns6, rec_hp, rec_ref, rate_hz, t_ms, counts
        gc.collect()

        saved_paths.append(out_geom)



    # if not saved_paths:
    #     raise RuntimeError("No sessions processed; nothing to concatenate.")
    
    # # --- concat + sorting (per-channel, no geometry)
    # print("Concatenating preprocessed sessions...")
    # recs_for_concat = []
    # for p in saved_paths:
    #     try:
    #         r = si.load(p)
    #     except Exception:
    #         r = si.load_extractor(p / "si_folder.json")
    #     recs_for_concat.append(r)

    # rec_concat = concatenate_recordings(recs_for_concat)
    # gc.collect()

    # sorting_ms5 = sorters.run_sorter(
    #     "mountainsort5",
    #     recording=rec_concat,
    #     folder=str(OUT_BASE / "mountainsort5"),
    #     remove_existing_folder=True,
    #     verbose=True,
    #     scheme="2",
    #     # scheme1_detect_channel_radius=1,
    #     detect_threshold=6,
    #     npca_per_channel=3,
    #     filter=False, whiten=True,
    #     delete_temporary_recording=True, progress_bar=True,
    # )

    # sa_folder = OUT_BASE / "sorting_ms5_analyzer"
    # phy_folder = OUT_BASE / "phy_ms5"

    # sa = create_sorting_analyzer(
    #     sorting=sorting_ms5,
    #     recording=rec_concat,
    #     folder=sa_folder,
    #     overwrite=True,
    #     sparse=False,
    # )
    # sa.compute("random_spikes", method="uniform", max_spikes_per_unit=1000, seed=0)
    # sa.compute("waveforms", ms_before=1.0, ms_after=2.0, progress_bar=True)
    # sa.compute("templates")
    # sa.compute("principal_components", n_components=3, mode="by_channel_global", progress_bar=True)
    # sa.compute("spike_amplitudes")

    # export_to_phy(sa, output_folder=phy_folder, copy_binary=True, remove_if_exists=True)
    
    # print(f"Phy export ready: {phy_folder}")

if __name__ == "__main__":
    main(use_intan=False, limit_sessions=None)
