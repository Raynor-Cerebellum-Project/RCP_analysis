import gc
from scipy.io import loadmat

# SpikeInterface
from probeinterface import Probe
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

"""
    This script preprocesses the Intan data.
    Input:
        .rhs files from Intan
    Output:
        Checkpoint after preprocessing
        Checkpoint after thresholding and calculating MUA peak locations and firing rate
"""

# ---------- Config ----------
# Base paths from config_loading
GEOM_PATH = rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)

# Intan streams
NPRW_CFG = PARAMS.probes.get("NPRW")
INTAN_STREAM = NPRW_CFG.get("neural_data_stream")
STIM_STREAM = NPRW_CFG.get("stim_data_stream") # "Stim channel"
AUX_STREAM = NPRW_CFG.get("aux_stream")
IR_STREAM = NPRW_CFG.get("ir_stream")
PROCESS_ONLY = PARAMS.preprocessing.get("process_only")  # list of BR indices to process; empty = all, need to map from BR indices to Intan indices using METADATA

# Local reference params, both floats
RADII = (PARAMS.probes.get("NPRW").get("local_radius_inner"), PARAMS.probes.get("NPRW").get("local_radius_outer"))
RATES = PARAMS.NPRW_rate_est
BIN_MS     = RATES.get("bin_ms")
SIGMA_MS   = RATES.get("sigma_ms")
THRESH     = RATES.get("detect_threshold")
PEAK_SIGN  = RATES.get("peak_sign")
ARTRMV_MS_BEFORE = float(RATES.get("remove_ms_before", 20.0))
ARTRMV_TAIL_MS   = float(RATES.get("remove_tail_ms_after", 20.0))


# Artifact correction parameters
# params = rcp.PCAArtifactParams(
#     # drift removal
#     rolling_median_ms=15.0,
#     gaussian_sigma_ms=5.0,
#     gaussian_len_ms=31.0,

#     # pulse-aligned window: start = start-13, end = end+13+15
#     pre_samples=13,
#     post_pad_samples=40,

#     # PCA/template
#     center_snippets=True,
#     first_pulse_special=True,
#     exclude_first_n_for_pca=1,

#     # subtraction
#     scale_amplitude=True,

#     # interp ramp
#     interp_ramp=True,
#     ramp_tail_ms=1.0,
#     ramp_fraction=1.0,
# )
        
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
    nprw_probe.set_device_channel_indices(intan_probe_mapping) # Apply mapping
    
    # 2) Find sessions and load data from each Intan folder
    sess_folders = rcp.list_intan_sessions(INTAN_ROOT)
    print(f"Found Intan sessions: {len(sess_folders)}")
    
    for sess in sess_folders[:]: # Can tweak here to isolate sessions
        # 3) Extract stim sessions and aux channels
        print(f"[RUN] session {sess.name}")

        out_dir = NPRW_CKPT_ROOT / f"pp_local_{int(RADII[0])}_{int(RADII[1])}__interp_{sess.name}"
        out_npz = NPRW_CKPT_ROOT / f"rates__{sess.name}__bin{int(BIN_MS)}ms_sigma{int(SIGMA_MS)}ms.npz"
        if out_dir.exists() and out_npz.exists():
            print(f"[SKIP] Both outputs already exist for {sess.name}")
            continue

        # aux streams (sync channels etc.)
        rcp.extract_intan_aux_streams_npz(sess=sess, out_dir=NPRW_AUX_DATA, aux_streams=AUX_STREAM)
        
        stim_ext_arrays = rcp.extract_stim_npz(sess=sess, out_dir=NPRW_AUX_DATA, stim_stream_name=STIM_STREAM, chanmap_perm=intan_probe_mapping)
        # stim_ext_arrays = rcp.load_stim_detection(NPRW_AUX_DATA / f"{sess.name}_Intan_streams" / "stim_stream.npz") - skip to speed up when debugging

        # 4) Load Intan IR stream
        rec_ir = se.read_split_intan_files(
            sess,
            mode="concatenate",
            stream_name=IR_STREAM,
            use_names_as_ids=True,
        )
        rec_ir = spre.unsigned_to_signed(rec_ir)  # UInt16 -> int16
        
        fs_ir = float(rec_ir.sampling_frequency)
        sig = np.asarray(rec_ir.get_traces()).squeeze()
        if sig is None or sig.size == 0:
            print(f"[IR] Intan={sess}: empty IR signal.")

        ir_idx = rcp.detect_IR_crossings(sig, fs_ir, refractory_sec=0.0005)
        if ir_idx.size == 0:
            print(f"[IR] Intan={sess}: no IR crossings.")

        # Convert to ms
        ir_ms = ir_idx / fs_ir * 1000.0
        
        del rec_ir, sig
        gc.collect()
        
        # Load Intan neural stream and reorder
        rec = se.read_split_intan_files(sess, mode="concatenate", stream_name=INTAN_STREAM, use_names_as_ids=True)
        rec = spre.unsigned_to_signed(rec) # Convert UInt16 to int16
        
        rec = rec.set_probe(nprw_probe)
        rec_reordered = rcp.reorder_recording_to_geometry(rec, intan_probe_mapping)
        
        # Local CMR
        rec_hp = spre.highpass_filter(rec_reordered, freq_min=float(PARAMS.highpass_hz))
        rec_ref = spre.common_reference(rec_hp, reference="local", operator="median", local_radius=(RADII[0], RADII[1]))
        
        # block_bounds_samples: shape (# stim blocks, 2) in absolute samples
        block_bounds = stim_ext_arrays.get("block_bounds_samples")
        stim_channels = stim_ext_arrays['active_channels']

        rec_artif_removed = rec_ref  # fallback
        fs_nprw = rec_reordered.get_sampling_frequency()
        n_total = rec_reordered.get_num_samples()
        
        dur_ms = None
        if block_bounds is not None and block_bounds.size:
            starts_samp = block_bounds[:, 0]
            ends_samp   = block_bounds[:, 1]

            valid = (ends_samp > starts_samp) & (starts_samp >= 0) & (starts_samp < n_total)
            # TODO if this is already checked in extract stim, isn't it redundant?
            starts_samp = starts_samp[valid]
            ends_samp   = ends_samp[valid]

            if starts_samp.size:
                dur_ms    = (ends_samp - starts_samp) * 1000.0 / fs_nprw
                ms_after  = float(dur_ms.max() + ARTRMV_TAIL_MS)
                
                rec_artif_removed = si.preprocessing.remove_artifacts(
                    rec_ref,
                    list_triggers=starts_samp.tolist(),
                    ms_before=ARTRMV_MS_BEFORE,
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

        del rec_reordered, rec_hp, rec_ref
        gc.collect()
        
        n_seg = rec_artif_removed.get_num_segments()
        noise_levels = si.get_noise_levels(rec_artif_removed, method="mad", return_in_uV=False) # They didn't write return_in_uV in their documentation
        
        print(f"[INFO] Segments of recording: {n_seg}, Average noise level: {np.nanmean(noise_levels)}")
        try:
            # Force n_jobs=1 for PyTorch backend to prevent GPU memory fragmentation / OOM
            peaks = detect_peaks(
                rec_artif_removed,
                method="by_channel_torch",
                detect_threshold=THRESH,
                peak_sign=PEAK_SIGN,
                noise_levels=noise_levels,
                n_jobs=1,
            )
        except Exception as exc:
            print(f"[WARN] Torch peak detection failed ({exc}). Falling back to CPU locally_exclusive...")
            peaks = detect_peaks(
                rec_artif_removed,
                method="locally_exclusive",
                detect_threshold=THRESH,
                peak_sign=PEAK_SIGN,
                noise_levels=noise_levels,
                n_jobs=PARAMS.parallel_jobs,
            )
        
        out_npz = NPRW_CKPT_ROOT / f"rates__{sess.name}__bin{int(BIN_MS)}ms_sigma{int(SIGMA_MS)}ms.npz"

        save = dict(
            peaks=peaks,
            noise_levels=noise_levels,
            ir_idx=ir_idx,
            ir_ms=ir_ms,
            meta=dict(
                detect_threshold=THRESH,
                peak_sign=PEAK_SIGN,
                bin_ms=BIN_MS,
                sigma_ms=SIGMA_MS,
                fs=fs_nprw,
                stim_channels=stim_channels,
                stim_dur = dur_ms if dur_ms is not None else 0.0,
                n_channels=rec_artif_removed.get_num_channels(),
                session=str(sess.name),
                n_samples = rec.get_total_samples(),
                n_segs = rec.get_num_segments(),
                rec_dur = rec.get_total_duration(),
                rec_start_ms = rec.get_start_time()*1000,
                rec_end_ms = rec.get_end_time()*1000,
            ))
        
        np.savez_compressed(out_npz, **save)
        print(f"[{sess.name}] saved peaks -> {out_npz}")

        # cleanup to keep memory stable on long batches
        del rec, rec_artif_removed, peaks
        gc.collect()

if __name__ == "__main__":
    main()
