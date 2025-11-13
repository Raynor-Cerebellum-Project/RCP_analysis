from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import gc
import numpy as np

# SpikeInterface
import spikeinterface as si
import spikeinterface.preprocessing as spre
from sklearn.decomposition import PCA

import RCP_analysis as rcp

""" 
    This script preprocesses the Intan data.
    Input:
        .rhs files from Intan
    Output:
        Checkpoint after preprocessing
        Checkpoint after thresholding and calculating MUA peak locations and firing rate
"""

# ---------- Config ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
DATA_ROOT = rcp.resolve_data_root(PARAMS)
OUT_BASE  = rcp.resolve_output_root(PARAMS); OUT_BASE.mkdir(parents=True, exist_ok=True)
INTAN_ROOT = rcp.resolve_intan_root(PARAMS); INTAN_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_CSV  = (DATA_ROOT / PARAMS.metadata_rel).resolve(); METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
GEOM_PATH = rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT)

# Intan streams
INTAN_STREAM = getattr(PARAMS, "neural_data_stream", "RHS2000 amplifier channel") # Neural data
STIM_STREAM = getattr(PARAMS, "stim_data_stream", "Stim channel") # Stim channel
# Auxiliary channels are grouped and saved

# Local reference params
probe_cfg = (PARAMS.probes or {}).get("NPRW", {})
inner = float(probe_cfg.get("local_radius_inner", 30.0))
outer = float(probe_cfg.get("local_radius_outer", 150.0))
RADII: Tuple[float, float] = (inner, outer)

# Artifact correction parameters
params = rcp.PCAArtifactParams(
    # drift removal
    rolling_median_ms=15.0,
    gaussian_sigma_ms=5.0,
    gaussian_len_ms=31.0,

    # pulse-aligned window: start = start-13, end = end+13+15
    pre_samples=13,
    post_pad_samples=40,

    # PCA/template
    center_snippets=True,
    first_pulse_special=True,
    exclude_first_n_for_pca=1,

    # subtraction
    scale_amplitude=True,

    # interp ramp
    interp_ramp=True,
    ramp_tail_ms=1.0,
    ramp_fraction=1.0,
)

RATES = PARAMS.intan_rate_est or {}
BIN_MS     = float(RATES.get("bin_ms", 1.0))
SIGMA_MS   = float(RATES.get("sigma_ms", 25.0))
THRESH     = float(RATES.get("detect_threshold", 4.5))
PEAK_SIGN  = str(RATES.get("peak_sign", "neg"))
        
global_job_kwargs = dict(n_jobs=PARAMS.parallel_jobs, chunk_duration=PARAMS.chunk)
si.set_global_job_kwargs(**global_job_kwargs)

def main(limit_sessions: Optional[int] = None):
    # set limit_session to run only a few conditions
    # limit_sessions = [14, 15, 16, 17]
    # 1) Load geometry & mapping
    geom = rcp.load_stim_geometry(GEOM_PATH)
    perm = rcp.get_chanmap_perm_from_geom(geom)
    probe = rcp.make_identity_probe_from_geom(geom, radius_um=5.0) # This radius is for visualization of the channel contacts

    # 2) Find sessions and load data from each Intan folder
    sess_folders = rcp.list_intan_sessions(INTAN_ROOT)
    # limit_sessions logic needed
    if limit_sessions:
        if isinstance(limit_sessions, int):
            # 1-based: 11 -> the 11th folder
            idx = limit_sessions - 1 if limit_sessions > 0 else limit_sessions
            sess_folders = [sess_folders[idx]]
        elif isinstance(limit_sessions, (list, tuple)):
            # 1-based indices: e.g., (2, 5, 11)
            idxs = [(i - 1 if i > 0 else i) for i in limit_sessions]
            sess_folders = [sess_folders[i] for i in idxs]
        else:
            pass
    print(f"Found Intan sessions: {len(sess_folders)}")

    checkpoint_out = OUT_BASE / "checkpoints" / "NPRW" # Save location for checkpoints
    checkpoint_out.mkdir(parents=True, exist_ok=True)
    
    for sess in sess_folders:
        # 3) Extract stim sessions and aux channels
        print(f"[RUN] session {sess.name}")
        bundles_root = OUT_BASE / "bundles" / "NPRW"
        bundles_root.mkdir(parents=True, exist_ok=True)

        # stim streams
        rcp.extract_stim_npz(
            sess_folder=sess,
            out_root=bundles_root,
            stim_stream_name=STIM_STREAM,
            chanmap_perm=perm,
        )

        # aux streams (sync channels etc.)
        rcp.extract_aux_streams_npz(
            sess_folder=sess,
            out_root=bundles_root,
            include_streams=("USB board ADC input channel",),
            chunk_s=60.0,
            chanmap_perm=perm,
        )
        
        stim_npz_path = bundles_root / f"{sess.name}_Intan_bundle" / "stim_stream.npz"

        # Load Intan neural stream
        rec = rcp.read_intan_recording(sess, stream_name=INTAN_STREAM)
        rec = rcp.reorder_recording_to_geometry(rec, perm)
        rec = rec.set_probe(probe, in_place=False)
        
        # Local CMR
        rec_hp = spre.highpass_filter(rec, freq_min=float(PARAMS.highpass_hz))
        rec_ref = spre.common_reference(rec_hp, reference="local", operator="median", local_radius=(float(RADII[0]), float(RADII[1])))
        
        # block_bounds_samples: shape (B, 2) in absolute samples
        stim = rcp.load_stim_detection(stim_npz_path)
        block_bounds = np.asarray(stim.get("block_bounds_samples", []), dtype=int)

        rec_artif_removed = rec_ref  # fallback

        if block_bounds.size:
            fs = float(rec_ref.get_sampling_frequency())
            n_total = rec_ref.get_num_frames() if hasattr(rec_ref, "get_num_frames") else rec_ref.get_num_samples()

            starts_samp = block_bounds[:, 0].astype(int)
            ends_samp   = block_bounds[:, 1].astype(int)

            valid = (ends_samp > starts_samp) & (starts_samp >= 0) & (starts_samp < n_total)
            starts_samp = starts_samp[valid]
            ends_samp   = ends_samp[valid]

            if starts_samp.size:
                ms_before_each = 20.0
                tail_ms = 20.0
                dur_ms    = (ends_samp - starts_samp) * 1000.0 / fs
                ms_after  = float(dur_ms.max() + tail_ms)

                rec_artif_removed = si.preprocessing.remove_artifacts(
                    rec_ref,
                    list_triggers=[starts_samp.tolist()],
                    ms_before=ms_before_each,
                    ms_after=ms_after,
                    mode="zeros",
                )
            else:
                print("[WARN] all block spans invalid or empty; skipping artifact removal.")
        else:
            print("[WARN] no block spans found; skipping artifact removal.")
        
        # Save preprocessed session
        out_dir = checkpoint_out / f"pp_local_{int(RADII[0])}_{int(RADII[1])}__interp_{sess.name}"
        rcp.save_recording(rec_artif_removed, out_dir)
        print(f"[{sess.name}] saved interpolated -> {out_dir}")

        del rec, rec_ref
        gc.collect()
        
        rate_hz, t_cat_ms, counts_cat, peaks, peak_t_ms = rcp.threshold_mua_rates(
            rec_artif_removed,
            detect_threshold=THRESH,
            peak_sign=PEAK_SIGN,
            bin_ms=BIN_MS,
            sigma_ms=SIGMA_MS,
            n_jobs=PARAMS.parallel_jobs,
        )
        
        # rate_hz is (n_channels, n_bins) → transpose to (n_bins, n_channels)
        X = rate_hz.T

        # PCA for visualization
        pca = PCA(n_components=5, random_state=0)
        pcs = pca.fit_transform(X)            # shape: (n_bins, 5)
        explained_var = pca.explained_variance_ratio_  # shape: (5,)

        # Transpose back to (n_components, n_bins) for consistency
        pcs_T = pcs.T.astype(np.float32)
        
        out_npz = checkpoint_out / f"rates__{sess.name}__bin{int(BIN_MS)}ms_sigma{int(SIGMA_MS)}ms.npz"
        # infer common field names across SI versions
        names = peaks.dtype.names
        samp_f = "sample_index" if "sample_index" in names else ("sample_ind" if "sample_ind" in names else None)
        chan_f = "channel_index" if "channel_index" in names else ("channel_ind" if "channel_ind" in names else None)
        amp_f  = "amplitude" if "amplitude" in names else None

        # convenience arrays (optional)
        peak_sample = peaks[samp_f].astype(np.int64)   if samp_f else None
        peak_ch     = peaks[chan_f].astype(np.int16)   if chan_f else None
        peak_amp    = peaks[amp_f].astype(np.float32)  if amp_f  else None

        save = dict(
            rate_hz=rate_hz.astype(np.float32),
            t_ms=t_cat_ms.astype(np.float32),
            counts=counts_cat.astype(np.uint32),
            peaks=peaks,
            peak_t_ms=peak_t_ms.astype(np.float32),
            pcs=pcs_T.astype(np.float32),
            explained_var=explained_var.astype(np.float32),
            meta=dict(
                detect_threshold=THRESH,
                peak_sign=PEAK_SIGN,
                bin_ms=BIN_MS,
                sigma_ms=SIGMA_MS,
                fs=float(rec_artif_removed.get_sampling_frequency()),
                n_channels=int(rec_artif_removed.get_num_channels()),
                session=str(sess.name),
                samp_field=samp_f, chan_field=chan_f, amp_field=amp_f,
            ),
        )

        if peak_sample is not None: save["peak_sample"] = peak_sample
        if peak_ch is not None:     save["peak_ch"] = peak_ch
        if peak_amp is not None:    save["peak_amp"] = peak_amp

        np.savez_compressed(out_npz, **save)
        print(f"[{sess.name}] saved rate matrix + PCA -> {out_npz}")

        # cleanup to keep memory stable on long batches
        del rec_artif_removed, rate_hz, t_cat_ms, counts_cat, peaks, peak_t_ms
        gc.collect()

if __name__ == "__main__":
    main(limit_sessions=None)
