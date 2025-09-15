from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import gc
import numpy as np

# SpikeInterface
import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.preprocessing as spre
from spikeinterface.core import concatenate_recordings
from spikeinterface.exporters import export_to_phy

# Project config
from RCP_analysis import load_experiment_params, resolve_data_root

# Intan helpers
from RCP_analysis.python.functions.intan_preproc import (
    load_stim_geometry,
    make_probe_from_geom,
    read_intan_recording,
    local_cm_reference,
    save_recording,
    list_intan_sessions,
)

# ==============================
# Config
# ==============================
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
OUT_BASE = (REPO_ROOT / "results_intan").resolve()
OUT_BASE.mkdir(parents=True, exist_ok=True)

# Intan locations (edit these two)
INTAN_ROOT = Path(PARAMS.sessions.get("intan_root", "/path/to/Intan")).resolve()
GEOM_PATH  = Path(PARAMS.sessions.get("intan_geom_path", "/path/to/geometry.mat")).resolve()

# Stream name for Intan (RHS2000 is the usual)
INTAN_STREAM = PARAMS.sessions.get("intan_stream_name", "RHS2000 amplifier channel")

# Local reference annulus [inner, outer] in µm
ANNULUS: Tuple[float, float] = (30.0, 150.0)

# ==============================
# Pipeline
# ==============================
def main(limit_sessions: Optional[int] = None):
    # 1) Load geometry & mapping
    geom = load_stim_geometry(GEOM_PATH)
    probe = make_probe_from_geom(geom, radius_um=5.0)

    # 2) Find sessions & load each Intan folder
    sess_folders = list_intan_sessions(INTAN_ROOT)
    if limit_sessions:
        sess_folders = sess_folders[:limit_sessions]
    print(f"Found Intan sessions: {len(sess_folders)}")

    preproc_paths: list[Path] = []
    checkpoint_out = OUT_BASE / "checkpoint"
    checkpoint_out.mkdir(parents=True, exist_ok=True)

    for sess in sess_folders:
        print(f"=== Session: {sess.name} ===")

        # Load Intan
        rec = read_intan_recording(sess, stream_name=INTAN_STREAM)
        rec = rec.set_probe(probe, in_place=False)

        # 3) Local CMR (inner/outer radius)
        rec_ref = local_cm_reference(rec, freq_min=float(PARAMS.highpass_hz), inner_outer_radius_um=ANNULUS)

        # Ensure channel locations exist (Kilosort4 requires geometry)
        try:
            _ = rec_ref.get_channel_locations()
        except Exception:
            n_ch = rec_ref.get_num_channels()
            locs = np.column_stack([np.arange(n_ch, dtype=float), np.zeros(n_ch, dtype=float)])
            rec_ref.set_channel_locations(locs)

        # Persist preprocessed session
        out_dir = checkpoint_out / f"pp_local_{int(ANNULUS[0])}_{int(ANNULUS[1])}__{sess.name}"
        save_recording(rec_ref, out_dir)
        print(f"[{sess.name}] saved preprocessed -> {out_dir}")

        del rec, rec_ref
        gc.collect()

        preproc_paths.append(out_dir)

    if not preproc_paths:
        raise RuntimeError("No Intan sessions processed; nothing to concatenate.")

    # 4) Concatenate
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

    # 5) Kilosort 4
    ks4_out = OUT_BASE / "kilosort4"
    sorting_ks4 = ss.run_sorter(
        "kilosort4",
        recording=rec_concat,
        folder=str(ks4_out),
        remove_existing_folder=True,
        verbose=True,
    )

    # 6) Export to Phy
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
    sa.compute("waveforms", ms_before=1.0, ms_after=2.0, max_spikes_per_unit=1000,
               n_jobs=int(PARAMS.parallel_jobs), chunk_duration=str(PARAMS.chunk), progress_bar=True)
    sa.compute("templates")
    sa.compute("principal_components", n_components=5, mode="by_channel_local",
               n_jobs=int(PARAMS.parallel_jobs), chunk_duration=str(PARAMS.chunk), progress_bar=True)
    sa.compute("spike_amplitudes")

    export_to_phy(sa, output_folder=phy_folder, copy_binary=True, remove_if_exists=True)
    print(f"Phy export ready: {phy_folder}")


if __name__ == "__main__":
    main(limit_sessions=None)
