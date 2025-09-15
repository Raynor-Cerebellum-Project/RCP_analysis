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

# Project config
from RCP_analysis import load_experiment_params, resolve_data_root

# BR/UA helpers
from RCP_analysis import (
    list_sessions, ua_excel_path,
    load_ns6_spikes, load_UA_mapping_from_excel, apply_ua_mapping_properties,
    build_blackrock_bundle, save_bundle_npz,
)

# INTAN HELPERS HERE

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
OUT_BASE = (REPO_ROOT / "results").resolve()
OUT_BASE.mkdir(parents=True, exist_ok=True)

def main(use_intan: bool = False, limit_sessions: Optional[int] = None):
    data_root = resolve_data_root(PARAMS)
    session_folders = list_sessions(data_root, PARAMS.blackrock_rel, use_intan=use_intan)
    if limit_sessions:
        session_folders = session_folders[:limit_sessions]
    print("Found session folders:", len(session_folders))

    xls = ua_excel_path(REPO_ROOT, PARAMS.probes)
    ua_map = load_UA_mapping_from_excel(xls) if xls else None
    if ua_map is None:
        raise RuntimeError("UA mapping required for mapping on NS6.")

    bundles_out = OUT_BASE / "bundles"
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
    
        out_geom = OUT_BASE / f"pp_global__{sess.name}__NS6"
        out_geom.mkdir(parents=True, exist_ok=True)
        rec_ref.save(folder=out_geom, overwrite=True)
        print(f"[{sess.name}] (ns6) saved preprocessed -> {out_geom}")

        del bundle, rec_ns6, rec_hp, rec_ref
        gc.collect()

        saved_paths.append(out_geom)

    if not saved_paths:
        raise RuntimeError("No sessions processed; nothing to concatenate.")

    # --- concat + sorting (per-channel, no geometry)
    print("Concatenating preprocessed sessions...")
    recs_for_concat = []
    for p in saved_paths:
        try:
            r = si.load(p)
        except Exception:
            r = si.load_extractor(p / "si_folder.json")
        recs_for_concat.append(r)

    rec_concat = concatenate_recordings(recs_for_concat)
    gc.collect()

    sorting_ms5 = sorters.run_sorter(
        "mountainsort5",
        rec_concat,
        str(OUT_BASE / "mountainsort5"),
        remove_existing_folder=True,
        verbose=False,
        
        n_jobs=int(PARAMS.parallel_jobs),
        chunk_duration=str(PARAMS.chunk),
        pool_engine="process",
        max_threads_per_worker=int(PARAMS.threads_per_worker),

        # per-channel sorting (no neighbors)
        scheme1_detect_channel_radius=0,
        scheme2_phase1_detect_channel_radius=0,
        scheme2_detect_channel_radius=0,
        snippet_mask_radius=0,
    )

    sa_folder = OUT_BASE / "sorting_ms5_analyzer"
    phy_folder = OUT_BASE / "phy_ms5"

    sa = create_sorting_analyzer(
        sorting=sorting_ms5,
        recording=rec_concat,
        folder=sa_folder,
        overwrite=True,
        sparse=False,
    )
    sa.compute("waveforms", ms_before=1.0, ms_after=2.0, max_spikes_per_unit=1000,
               n_jobs=int(PARAMS.parallel_jobs), chunk_duration=str(PARAMS.chunk), progress_bar=True)
    sa.compute("templates")
    sa.compute("principal_components", n_components=5, mode="by_channel_global",
               n_jobs=int(PARAMS.parallel_jobs), chunk_duration=str(PARAMS.chunk), progress_bar=True)
    sa.compute("spike_amplitudes")

    export_to_phy(sa, output_folder=phy_folder, copy_binary=True, remove_if_exists=True)
    print(f"Phy export ready: {phy_folder}")

if __name__ == "__main__":
    main(use_intan=False, limit_sessions=None)
