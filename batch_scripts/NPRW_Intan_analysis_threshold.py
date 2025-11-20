from pathlib import Path
from typing import Optional
import gc
import numpy as np
from scipy.io import loadmat

# SpikeInterface
from probeinterface import Probe
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
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
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE  = SESSION_LOC / "results"; OUT_BASE.mkdir(parents=True, exist_ok=True)
INTAN_ROOT = SESSION_LOC / "Intan"; INTAN_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_LOC  = SESSION_LOC / "Metadata"; METADATA_LOC.parent.mkdir(parents=True, exist_ok=True)
GEOM_PATH = rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT)
NPRW_CKPT_ROOT = OUT_BASE / "checkpoints" / "NPRW"
NPRW_BUNDLES   = OUT_BASE / "bundles" / "NPRW"; NPRW_BUNDLES.mkdir(parents=True, exist_ok=True)

# Intan streams
NPRW_CFG = PARAMS.probes.get("NPRW")
INTAN_STREAM = NPRW_CFG.get("neural_data_stream")
STIM_STREAM = NPRW_CFG.get("stim_data_stream") # "Stim channel"
AUX_STREAM = NPRW_CFG.get("aux_stream")

# Local reference params, both floats
RADII = (PARAMS.probes.get("NPRW").get("local_radius_inner"), PARAMS.probes.get("NPRW").get("local_radius_outer"))
RATES = PARAMS.intan_rate_est
BIN_MS     = RATES.get("bin_ms")
SIGMA_MS   = RATES.get("sigma_ms")
THRESH     = RATES.get("detect_threshold")
PEAK_SIGN  = RATES.get("peak_sign")

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
        
global_job_kwargs = dict(n_jobs=PARAMS.parallel_jobs, chunk_duration=PARAMS.chunk)
si.set_global_job_kwargs(**global_job_kwargs)

def main(limit_sessions: Optional[int] = None):
    # set limit_session to run only a few conditions
    # limit_sessions = [3]
    
    # 1) Load geometry & mapping
    mat_probe = loadmat(Path(GEOM_PATH))
    intan_geom = {}
    intan_geom["x"] = mat_probe["xcoords"].ravel()
    intan_geom["y"] = mat_probe["ycoords"].ravel()
    assert intan_geom["x"].size == intan_geom["y"].size, "x/y must have same length"
    if "chanMap0ind" in mat_probe: # 0-based device mapping if present
        intan_probe_mapping = intan_geom["device_index_0based"] = mat_probe["chanMap0ind"].ravel()
    else:
        raise ValueError("No 0-based chanmap in .mat geometry file.")
    if intan_probe_mapping.size != intan_geom["x"].size:
        raise ValueError("device_index_0based length != #contacts")
    
    # Build ProbeInterface Probe
    nprw_probe = Probe(ndim=2)
    nprw_probe.set_contacts(positions=np.c_[intan_geom["x"], intan_geom["y"]], shapes="square", shape_params={"width": 12.0})
    nprw_probe.set_device_channel_indices(intan_probe_mapping )# Apply mapping
    
    # 2) Find sessions and load data from each Intan folder
    sess_folders = rcp.list_intan_sessions(INTAN_ROOT)
    # limit_sessions logic needed
    if limit_sessions:
        if isinstance(limit_sessions, int):
            idx = limit_sessions - 1 if limit_sessions > 0 else limit_sessions
            sess_folders = [sess_folders[idx]]
        elif isinstance(limit_sessions, (list, tuple)):
            # 1-based indices: e.g., (2, 5, 11)
            idxs = [(i - 1 if i > 0 else i) for i in limit_sessions]
            sess_folders = [sess_folders[i] for i in idxs]
        else:
            pass
    print(f"Found Intan sessions: {len(sess_folders)}")
    
    for sess in sess_folders:
        # 3) Extract stim sessions and aux channels
        print(f"[RUN] session {sess.name}")

        # stim streams TODO anyway to leverage that this is sparse?
        _, stim_ext_arrays = rcp.extract_stim_npz(sess_folder=sess, out_root=NPRW_BUNDLES, stim_stream_name=STIM_STREAM, chanmap_perm=intan_probe_mapping)
        # stim_ext_arrays = rcp.load_stim_detection(NPRW_BUNDLES / f"{sess.name}_Intan_bundle" / "stim_stream.npz") - skip to speed up when debugging
        
        # aux streams (sync channels etc.)
        rcp.extract_aux_streams_npz(sess_folder=sess, out_root=NPRW_BUNDLES, aux_streams=AUX_STREAM)

        # Load Intan neural stream and reorder
        rec = se.read_split_intan_files(sess, mode="concatenate", stream_name=INTAN_STREAM, use_names_as_ids=True)
        rec = spre.unsigned_to_signed(rec) # Convert UInt16 to int16
        rec_reordered = rcp.reorder_recording_to_geometry(rec, intan_probe_mapping)
        rec_reordered = rec_reordered.set_probe(nprw_probe)
        
        # Local CMR
        rec_hp = spre.highpass_filter(rec_reordered, freq_min=float(PARAMS.highpass_hz))
        rec_ref = spre.common_reference(rec_hp, reference="local", operator="median", local_radius=(RADII[0], RADII[1]))
        
        # block_bounds_samples: shape (# stim blocks, 2) in absolute samples
        block_bounds = stim_ext_arrays.get("block_bounds_samples")

        rec_artif_removed = rec_ref  # fallback

        fs = rec_reordered.get_sampling_frequency()
        n_total = rec_reordered.get_num_samples()
        
        if block_bounds.size:
            starts_samp = block_bounds[:, 0]
            ends_samp   = block_bounds[:, 1]

            valid = (ends_samp > starts_samp) & (starts_samp >= 0) & (starts_samp < n_total)
            # TODO if this is already checked in extract stim, isn't it redundant?
            starts_samp = starts_samp[valid]
            ends_samp   = ends_samp[valid]

            if starts_samp.size:
                ms_before_each = 20.0
                tail_ms = 20.0
                dur_ms    = (ends_samp - starts_samp) * 1000.0 / fs
                ms_after  = dur_ms.max() + tail_ms

                rec_artif_removed = si.preprocessing.remove_artifacts(
                    rec_ref,
                    list_triggers=starts_samp.tolist(),
                    ms_before=ms_before_each,
                    ms_after=ms_after,
                    mode="zeros",
                )
            else:
                print("[WARN] all block spans invalid or empty; skipping artifact removal.")
        else:
            print("[WARN] no block spans found; skipping artifact removal.")
        
        # Save preprocessed session
        out_dir = NPRW_CKPT_ROOT / f"pp_local_{int(RADII[0])}_{int(RADII[1])}__interp_{sess.name}"
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
        
        # transpose to (n_bins, n_channels)
        X = rate_hz.T

        # PCA for visualization
        pca = PCA(n_components=5, random_state=0)
        pcs = pca.fit_transform(X)            # shape: (n_bins, 5)
        explained_var = pca.explained_variance_ratio_  # shape: (5,)

        # Transpose back to (n_components, n_bins) for consistency
        pcs_T = pcs.T
        
        out_npz = NPRW_CKPT_ROOT / f"rates__{sess.name}__bin{int(BIN_MS)}ms_sigma{int(SIGMA_MS)}ms.npz"

        save = dict(
            rate_hz=rate_hz,
            t_ms=t_cat_ms,
            counts=counts_cat,
            peaks=peaks,
            peak_t_ms=peak_t_ms,
            pcs=pcs_T,
            explained_var=explained_var,
            meta=dict(
                detect_threshold=THRESH,
                peak_sign=PEAK_SIGN,
                bin_ms=BIN_MS,
                sigma_ms=SIGMA_MS,
                fs=fs,
                n_channels=rec_artif_removed.get_num_channels(),
                session=str(sess.name),
            ))

        # TODO is this necessary?
        peak_sample = peaks["sample_index"]
        peak_ch     = peaks["channel_index"]
        peak_amp    = peaks["amplitude"]
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
