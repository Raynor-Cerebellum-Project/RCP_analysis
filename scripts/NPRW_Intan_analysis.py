from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import gc

# SpikeInterface
import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.preprocessing as spre
from spikeinterface.core import concatenate_recordings
from spikeinterface.exporters import export_to_phy

# Project config
from RCP_analysis import load_experiment_params, resolve_data_root

# Intan helpers
from RCP_analysis import (
    load_stim_geometry,
    make_probe_from_geom,
    read_intan_recording,
    local_cm_reference,
    save_recording,
    list_intan_sessions,
    extract_and_save_stim_npz,
    extract_and_save_other_streams_npz,
    get_chanmap_perm_from_geom,
    make_identity_probe_from_geom,
    reorder_recording_to_geometry,
)

# Package API
from RCP_analysis import (
    load_experiment_params, resolve_data_root, resolve_output_root,
    resolve_probe_geom_path, resolve_session_intan_dir, plot_all_quads_for_session, 
)

from RCP_analysis import (
    correct_recording_with_stim_npz, TemplateParams
)

# ---- PLOT-ONLY ENTRYPOINT ----------------------------------------------------
def plot_selected_sessions(
    indices=(4, 5),
    pre_s: float = 0.30,
    post_s: float = 0.30,
    chunk_s: float = 60.0,
):
    """
    Plot 4×4 panels + probe for selected Intan sessions (0-based indices).
    Uses existing stim bundles if present; creates them if missing.
    Does NOT run any preprocessing or Kilosort.
    """
    # geometry / perm (for stim bundle reordering, same as your pipeline)
    geom = load_stim_geometry(GEOM_PATH)
    perm = get_chanmap_perm_from_geom(geom)

    # where to write figures + find/create stim bundles
    figs_dir = OUT_BASE / "figures" / "NPRW"
    figs_dir.mkdir(parents=True, exist_ok=True)
    bundles_root = OUT_BASE / "bundles" / "NPRW"
    bundles_root.mkdir(parents=True, exist_ok=True)

    # session list
    sess_folders = list_intan_sessions(INTAN_ROOT)
    if not sess_folders:
        print("No Intan sessions found.")
        return

    for i in indices:
        if i < 0 or i >= len(sess_folders):
            print(f"[WARN] index {i} out of range (0..{len(sess_folders)-1}). Skipping.")
            continue

        sess = sess_folders[i]
        print(f"--- Plotting session #{i}: {sess.name} ---")

        # expected stim bundle path
        stim_npz_path = bundles_root / f"{sess.name}_Intan_bundle" / "stim_stream.npz"
        if not stim_npz_path.exists():
            # create stim bundle only (reordered to geometry so plotting aligns)
            stim_npz_path = extract_and_save_stim_npz(
                sess_folder=sess,
                out_root=bundles_root,
                stim_stream_name=STIM_STREAM,
                chunk_s=chunk_s,
                chanmap_perm=perm,
            )

        # make the figures (uses raw Intan + geometry; no dependency on preproc)
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
            )
        except Exception as e:
            print(f"[WARN] plotting failed for {sess.name}: {e}")


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
params = TemplateParams(
    fs=30000.0,   # or 30000.0
    buffer=25,
    template_leeway=15,
    stim_neural_delay=13,
    med_filt_range=25,
    gauss_filt_range=25,
    pca_k=3,
)

global_job_kwargs = dict(n_jobs=PARAMS.parallel_jobs, chunk_duration=PARAMS.chunk)
si.set_global_job_kwargs(**global_job_kwargs)

# ==============================
# Pipeline
# ==============================
def main(use_br: bool = False, use_intan: bool = True, limit_sessions: Optional[int] = None):
    
    # plot_selected_sessions(indices=(4, 5), pre_s=0.30, post_s=0.30)
    
    # 1) Load geometry & mapping
    geom = load_stim_geometry(GEOM_PATH)
    perm = get_chanmap_perm_from_geom(geom)
    probe = make_identity_probe_from_geom(geom, radius_um=5.0)

    # 2) Find sessions & load each Intan folder
    sess_folders = list_intan_sessions(INTAN_ROOT)
    if limit_sessions:
        sess_folders = sess_folders[:limit_sessions]
    print(f"Found Intan sessions: {len(sess_folders)}")

    preproc_paths: list[Path] = []
    checkpoint_out = OUT_BASE / "checkpoints" / "NPRW"
    checkpoint_out.mkdir(parents=True, exist_ok=True)
    
    # 3) Preprocess, extract stim sessions and aux channels, save preprocessed Intan
    for sess in sess_folders:
        print(f"=== Session: {sess.name} ===")
        
        bundles_root = OUT_BASE / "bundles" / "NPRW"
        bundles_root.mkdir(parents=True, exist_ok=True)

        # extract stim streams
        extract_and_save_stim_npz(
            sess_folder=sess,
            out_root=bundles_root,
            stim_stream_name=STIM_STREAM,
            chunk_s=60.0,
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
        
        # Load Intan
        rec = read_intan_recording(sess, stream_name=INTAN_STREAM)
        rec = reorder_recording_to_geometry(rec, perm)
        rec = rec.set_probe(probe, in_place=False)

        # 3) Local CMR (inner/outer radius)
        rec_ref = local_cm_reference(rec, freq_min=float(PARAMS.highpass_hz), inner_outer_radius_um=RADII)

        # Save preprocessed session
        out_dir = checkpoint_out / f"pp_local_{int(RADII[0])}_{int(RADII[1])}__{sess.name}"
        save_recording(rec_ref, out_dir)
        print(f"[{sess.name}] saved preprocessed -> {out_dir}")

        del rec
        stim_npz_path = bundles_root / f"{sess.name}_Intan_bundle" / "stim_stream.npz"

        # 4) Artifact correction via PCA
        rec_corr = correct_recording_with_stim_npz(
            recording=rec_ref,
            stim_npz_path=stim_npz_path,
            params=params,
            mode="pca",
            channels=None,
            n_jobs=int(PARAMS.parallel_jobs),
        )

        # Save the artifact-corrected recording as the checkpoint for step 5
        corr_dir = checkpoint_out / f"pp_local_{int(RADII[0])}_{int(RADII[1])}__AC_{sess.name}"
        save_recording(rec_corr, corr_dir)
        print(f"[ArtCorr] saved artifact-corrected -> {corr_dir}")
        del rec_ref, rec_corr
        gc.collect()

        preproc_paths.append(corr_dir)

    if not preproc_paths:
        raise RuntimeError("No Intan sessions processed; nothing to concatenate.")
    
    # 5) Concatenate
    print("Concatenating preprocessed sessions...")
    recs = []
    for p in preproc_paths:
        try:
            r = si.load(p)
        except Exception:
            r = si.load_extractor(p / "si_folder.json")
        recs.append(r)
    rec_concat = concatenate_recordings(recs)
    gc.collect()

    # 6) KS4
    ks4_out = OUT_BASE / "kilosort4"
    sorting_ks4 = ss.run_sorter(
        "kilosort4",
        recording=rec_concat,
        folder=str(ks4_out),
        remove_existing_folder=True,
        verbose=True,
    ) # TODO: Check if geometry is used

    # 7) Export to Phy
    sa_folder = OUT_BASE / "sorting_ks4_analyzer"
    phy_folder = OUT_BASE / "phy_ks4"

    sa = si.create_sorting_analyzer(
        sorting=sorting_ks4,
        recording=rec_concat,
        folder=sa_folder,
        overwrite=True,
        sparse=True,
    )

    # Phy needs at least waveforms/templates/PCs; random_spikes makes it deterministic
    sa.compute("random_spikes", method="uniform", max_spikes_per_unit=1000, seed=0)
    sa.compute("waveforms", ms_before=1.0, ms_after=2.0,
               n_jobs=int(PARAMS.parallel_jobs), chunk_duration=str(PARAMS.chunk), progress_bar=True)
    sa.compute("templates")
    sa.compute("principal_components", n_components=5, mode="by_channel_local",
               n_jobs=int(PARAMS.parallel_jobs), chunk_duration=str(PARAMS.chunk), progress_bar=True)
    sa.compute("spike_amplitudes")

    export_to_phy(sa, output_folder=phy_folder, copy_binary=True, remove_if_exists=True)
    print(f"Phy export ready: {phy_folder}")

    # SLAy
    #TODO Separate by condition
    #TODO Firing rate (FR) estimation using Gaussian filter?
    #TODO Align with BR using two sync pulses (one from BR side)

if __name__ == "__main__":
    main(limit_sessions=None)
