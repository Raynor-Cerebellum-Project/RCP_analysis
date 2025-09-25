from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import gc
import numpy as np

# SpikeInterface
import spikeinterface as si
from sklearn.decomposition import PCA

# Project config
from RCP_analysis import load_experiment_params, resolve_data_root

# Intan helpers
from RCP_analysis import (
    # PP
    read_intan_recording,
    local_cm_reference,
    save_recording,
    list_intan_sessions,
    extract_and_save_other_streams_npz,
    # Probes
    load_stim_geometry,
    get_chanmap_perm_from_geom,
    make_identity_probe_from_geom,
    reorder_recording_to_geometry,
    # Stim
    extract_and_save_stim_npz,
    PCAArtifactParams,
    threshold_mua_rates, load_stim_detection,
)

# Package API
from RCP_analysis import (
    resolve_output_root, resolve_probe_geom_path, plot_all_quads_for_session, 
)

# ---- PLOT-ONLY ENTRYPOINT ----------------------------------------------------
def plot_selected_sessions(
    indices=(0,),            # e.g. (3,) or (4, 5)
    pre_s: float = 0.30,
    post_s: float = 0.30,
    chunk_s: float = 60.0,
    preproc_root: Path = Path("/home/bryan/mnt/cullen/Current Project Databases - NHP/2025 Cerebellum prosthesis/Nike/NRR_RW001/results/checkpoints"),
    template_samples_before: float | None = None,
    template_samples_after: float | None = None,
):
    """
    Plot 4×4 panels + probe for selected sessions using the artifact-corrected checkpoints.
    Also ensures a stim bundle exists (creates if missing).
    """
    # geometry / perm
    geom = load_stim_geometry(GEOM_PATH)
    perm = get_chanmap_perm_from_geom(geom)

    figs_dir = OUT_BASE / "figures" / "NPRW"
    figs_dir.mkdir(parents=True, exist_ok=True)
    bundles_root = OUT_BASE / "bundles" / "NPRW"
    bundles_root.mkdir(parents=True, exist_ok=True)

    sess_folders = list_intan_sessions(INTAN_ROOT)
    if not sess_folders:
        print("No Intan sessions found.")
        return

    # normalize indices to an iterable of ints
    if isinstance(indices, int):
        indices = [indices]
    else:
        indices = list(indices)

    for i in indices:
        if i < 0 or i >= len(sess_folders):
            print(f"[WARN] index {i} out of range (0..{len(sess_folders)-1}). Skipping.")
            continue

        sess = sess_folders[i]
        print(f"--- Plotting session #{i}: {sess.name} ---")

        # ensure stim bundle exists
        stim_npz_path = bundles_root / f"{sess.name}_Intan_bundle" / "stim_stream.npz"
        if not stim_npz_path.exists():
            stim_npz_path = extract_and_save_stim_npz(
                sess_folder=sess,
                out_root=bundles_root,
                stim_stream_name=STIM_STREAM,
                chunk_s=chunk_s,
                chanmap_perm=perm,
            )

        # plot from artifact-corrected checkpoints
        try:
            plot_all_quads_for_session(
                sess_folder=sess,
                geom_path=GEOM_PATH,
                neural_stream=INTAN_STREAM,
                stim_stream=STIM_STREAM,
                out_dir=figs_dir,
                stim_npz_path=stim_npz_path,
                pre_s=pre_s,
                post_s=post_s,
                preproc_root=preproc_root,
                template_samples_before=template_samples_before,
                template_samples_after=template_samples_after,
            )
        except Exception as e:
            print(f"[WARN] plotting failed for {sess.name}: {e}")
        print(f"[Plot] Plotted session #{i}: {sess.name}")

# ==============================
# Config
# ==============================
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
OUT_BASE = resolve_output_root(PARAMS)
OUT_BASE.mkdir(parents=True, exist_ok=True)
DATA_ROOT = resolve_data_root(PARAMS)

# --- Intan root ---
INTAN_ROOT = (
    Path(PARAMS.intan_root).resolve()
    if getattr(PARAMS, "intan_root", None) and str(PARAMS.intan_root).startswith("/")
    else (DATA_ROOT / PARAMS.intan_root_rel).resolve()
    if getattr(PARAMS, "intan_root_rel", None)
    else None
)
if INTAN_ROOT is None:
    raise ValueError("No Intan root specified. Set 'intan_root_rel' or 'intan_root' in params.yaml.")

# --- Geometry path ---
GEOM_PATH = (
    Path(PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None) and str(PARAMS.geom_mat_rel).startswith("/")
    else (REPO_ROOT / PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None)
    else resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
)

# === Intan stream name ===
INTAN_STREAM = getattr(PARAMS, "neural_data_stream", "RHS2000 amplifier channel")
STIM_STREAM = getattr(PARAMS, "stim_data_stream", "Stim channel")

# === Local reference radii (µm) ===
probe_cfg = (PARAMS.probes or {}).get("NPRW", {})
inner = float(probe_cfg.get("local_radius_inner", 30.0))
outer = float(probe_cfg.get("local_radius_outer", 150.0))
RADII: Tuple[float, float] = (inner, outer)

# Artifact correction parameters
params = PCAArtifactParams(
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

# ==============================
# Pipeline
# ==============================
def main(use_br: bool = False, use_intan: bool = True, limit_sessions: Optional[int] = None):
    # plot_selected_sessions(indices=(2), pre_s=0.1, post_s=0.2, 
    #         template_samples_before = params.pre_samples,
    #         template_samples_after = params.post_pad_samples
    # )
    # 1) Load geometry & mapping
    geom = load_stim_geometry(GEOM_PATH)
    perm = get_chanmap_perm_from_geom(geom)
    probe = make_identity_probe_from_geom(geom, radius_um=5.0) # Radius is for visualization of the channel contacts

    # 2) Find sessions & load each Intan folder
    sess_folders = list_intan_sessions(INTAN_ROOT)
    if limit_sessions:
        sess_folders = sess_folders[:limit_sessions]
    print(f"Found Intan sessions: {len(sess_folders)}")

    preproc_paths: list[Path] = []
    checkpoint_out = OUT_BASE / "checkpoints" / "NPRW"
    checkpoint_out.mkdir(parents=True, exist_ok=True)
    
    # 3) Extract stim sessions and aux channels, preprocess and save
    for sess in sess_folders:
        print(f"[RUN] session {sess.name}")
        bundles_root = OUT_BASE / "bundles" / "NPRW"
        bundles_root.mkdir(parents=True, exist_ok=True)

        # extract stim streams
        extract_and_save_stim_npz(
            sess_folder=sess,
            out_root=bundles_root,
            stim_stream_name=STIM_STREAM,
            chanmap_perm=perm,
        )

        # extract aux streams
        extract_and_save_other_streams_npz(
            sess_folder=sess,
            out_root=bundles_root,
            include_streams=("USB board ADC input channel",),
            chunk_s=60.0,
            chanmap_perm=perm,
        )
        
        stim_npz_path = bundles_root / f"{sess.name}_Intan_bundle" / "stim_stream.npz"

        # Load Intan neural stream
        rec = read_intan_recording(sess, stream_name=INTAN_STREAM)
        rec = reorder_recording_to_geometry(rec, perm)
        rec = rec.set_probe(probe, in_place=False)
        
        # Local CMR
        rec_ref = local_cm_reference(
            rec,
            freq_min=float(PARAMS.highpass_hz),
            inner_outer_radius_um=RADII
        )
        # block_bounds_samples: shape (B, 2) in absolute samples
        stim = load_stim_detection(stim_npz_path)
        block_bounds = np.asarray(stim.get("block_bounds_samples", []), dtype=int)

        rec_interp = rec_ref  # fallback

        if block_bounds.size:
            fs = float(rec_ref.get_sampling_frequency())
            n_total = rec_ref.get_num_frames() if hasattr(rec_ref, "get_num_frames") else rec_ref.get_num_samples()

            starts_samp = block_bounds[:, 0].astype(int)
            ends_samp   = block_bounds[:, 1].astype(int)

            valid = (ends_samp > starts_samp) & (starts_samp >= 0) & (starts_samp < n_total)
            starts_samp = starts_samp[valid]
            ends_samp   = ends_samp[valid]

            if starts_samp.size:
                dur_ms = (ends_samp - starts_samp) * 1000.0 / fs
                tail_ms = 5.0
                ms_before_each = 5.0

                rec_interp = si.preprocessing.remove_artifacts(
                    rec_ref,
                    list_triggers=[starts_samp.tolist()],
                    ms_before=ms_before_each,
                    ms_after=float(dur_ms.max() + tail_ms),  # one size
                    mode="zeros",
                )
            else:
                print("[WARN] all block spans invalid or empty; skipping artifact removal.")
        else:
            print("[WARN] no block spans found; skipping artifact removal.")

        figs_dir_during = OUT_BASE / "figures" / "NPRW" / "interp"
        figs_dir_after = OUT_BASE / "figures" / "NPRW" / "after_interp"
        preproc_root = OUT_BASE / "checkpoints"
        
        # Save preprocessed session (artifact-corrected + referenced)
        out_dir = checkpoint_out / f"pp_local_{int(RADII[0])}_{int(RADII[1])}__interp_{sess.name}"
        save_recording(rec_interp, out_dir)
        print(f"[{sess.name}] saved interpolated -> {out_dir}")

        del rec, rec_ref
        gc.collect()

        plot_all_quads_for_session(
            sess_folder=sess,
            geom_path=GEOM_PATH,
            neural_stream=INTAN_STREAM,
            stim_stream=STIM_STREAM,
            out_dir=figs_dir_during,
            stim_npz_path=stim_npz_path,
            pre_s=0.1,
            post_s=0.2,
            preproc_root=preproc_root,
            template_samples_before=params.pre_samples,
            template_samples_after=params.post_pad_samples
        )
        plot_all_quads_for_session(
            sess_folder=sess,
            geom_path=GEOM_PATH,
            neural_stream=INTAN_STREAM,
            stim_stream=STIM_STREAM,
            out_dir=figs_dir_after,
            stim_npz_path=stim_npz_path,
            pre_s=-0.9,
            post_s=0.4,
            preproc_root=preproc_root,
            template_samples_before=params.pre_samples,
            template_samples_after=params.post_pad_samples
        )
        
        rate_hz, t_ms, counts = threshold_mua_rates(
            rec_interp,
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
        np.savez_compressed(
            out_npz,
            rate_hz=rate_hz.astype(np.float32),
            t_ms=t_ms.astype(np.float32),
            counts=counts.astype(np.uint16),
            pcs=pcs_T,                                  # (5, n_bins)
            explained_var=explained_var.astype(np.float32),
            meta=dict(
                detect_threshold=THRESH,
                peak_sign=PEAK_SIGN,
                bin_ms=BIN_MS,
                sigma_ms=SIGMA_MS,
                fs=float(rec_interp.get_sampling_frequency()),
                n_channels=int(rec_interp.get_num_channels()),
                session=str(sess.name),
            ),
        )
        print(f"[{sess.name}] saved rate matrix + PCA -> {out_npz}")

        # cleanup to keep memory stable on long batches
        del rec_interp, rate_hz, t_ms, counts
        gc.collect()
        # TODO make plots firing rate

if __name__ == "__main__":
    main(limit_sessions=None)
