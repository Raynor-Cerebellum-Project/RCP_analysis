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
NPRW_AUX_DATA   = OUT_BASE / "aux_data" / "NPRW"; NPRW_AUX_DATA.mkdir(parents=True, exist_ok=True)

# Intan streams
NPRW_CFG = PARAMS.probes.get("NPRW")
INTAN_STREAM = NPRW_CFG.get("neural_data_stream")
STIM_STREAM = NPRW_CFG.get("stim_data_stream") # "Stim channel"
AUX_STREAM = NPRW_CFG.get("aux_stream")


ARTRMV_MS_BEFORE = 20.0
ARTCORR_TAIL_MS = 20.0

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

def main():
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
    nprw_probe.set_device_channel_indices(intan_probe_mapping)# Apply mapping
    
    # 2) Find sessions and load data from each Intan folder
    sess_folders = rcp.list_intan_sessions(INTAN_ROOT)
    print(f"Found Intan sessions: {len(sess_folders)}")
    
    for sess in sess_folders[:]: # Can tweak here to isolate sessions
        # 3) Extract stim sessions and aux channels
        print(f"[RUN] session {sess.name}")

        # stim streams TODO anyway to leverage that this is sparse?
        _, stim_ext_arrays = rcp.extract_stim_npz(sess=sess, out_dir=NPRW_AUX_DATA, stim_stream_name=STIM_STREAM, chanmap_perm=intan_probe_mapping)
        # stim_ext_arrays = rcp.load_stim_detection(NPRW_AUX_DATA / f"{sess.name}_Intan_streams" / "stim_stream.npz") - skip to speed up when debugging
        
        # aux streams (sync channels etc.)
        rcp.extract_intan_aux_streams_npz(sess=sess, out_dir=NPRW_AUX_DATA, aux_streams=AUX_STREAM)

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
        blank_windows = None

        rec_artif_removed = rec_ref  # fallback

        fs_nprw = rec_reordered.get_sampling_frequency()
        n_total = rec_reordered.get_num_samples()
        
        if block_bounds.size:
            starts_samp = block_bounds[:, 0]
            ends_samp   = block_bounds[:, 1]

            valid = (ends_samp > starts_samp) & (starts_samp >= 0) & (starts_samp < n_total)
            # TODO if this is already checked in extract stim, isn't it redundant?
            starts_samp = starts_samp[valid]
            ends_samp   = ends_samp[valid]

            if starts_samp.size:
                dur_ms    = (ends_samp - starts_samp) * 1000.0 / fs_nprw
                ms_after  = float(dur_ms.max() + ARTCORR_TAIL_MS)
                
                rec_artif_removed = si.preprocessing.remove_artifacts(
                    rec_ref,
                    list_triggers=starts_samp.tolist(),
                    ms_before=ARTRMV_MS_BEFORE,
                    ms_after=ms_after,
                    mode="zeros",
                )
                
                pad_before_samp = int(round(ARTRMV_MS_BEFORE * fs_nprw / 1000.0))
                pad_after_samp  = int(round(ARTCORR_TAIL_MS  * fs_nprw / 1000.0))

                starts_exp = np.clip(starts_samp - pad_before_samp, 0, None)
                ends_exp   = np.clip(ends_samp   + pad_after_samp,  0, n_total)

                blank_windows = {0: np.column_stack([starts_exp, ends_exp])}  # seg 0
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
            blank_windows_samples=blank_windows,
        )
        
        X = rate_hz.T  # (n_bins, n_channels)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        n_bins, n_ch = X.shape
        n_comp = min(5, n_bins, n_ch)

        total_var = np.var(X, axis=0).sum()
        if n_comp >= 1 and total_var > 0.0:
            pca = PCA(n_components=n_comp, random_state=0)
            pcs = pca.fit_transform(X)  # (n_bins, n_comp)
            explained_var = np.nan_to_num(
                pca.explained_variance_ratio_, nan=0.0
            ).astype(np.float32)
            pcs_T = pcs.T.astype(np.float32)  # (n_comp, n_bins)
        else:
            pcs_T = np.empty((0, n_bins), dtype=np.float32)
            explained_var = np.empty((0,), dtype=np.float32)
        
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
                fs=fs_nprw,
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
    main()
