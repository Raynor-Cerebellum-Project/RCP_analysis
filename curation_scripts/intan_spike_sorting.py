#!/usr/bin/env python3
"""
- Loads just baseline recording
- Runs KS4
- Phy export
"""

from __future__ import annotations
from pathlib import Path
import sys
import gc

import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.exporters as sie

import RCP_analysis as rcp

# ---------- CONFIG / PATHS ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
OUT_BASE  = rcp.resolve_output_root(PARAMS); OUT_BASE.mkdir(parents=True, exist_ok=True)

# Match the same radii used when creating the pp_local_* preprocessed folders
probe_cfg = (PARAMS.probes or {}).get("NPRW", {})
inner = float(probe_cfg.get("local_radius_inner", 30.0))
outer = float(probe_cfg.get("local_radius_outer", 150.0))

CHECKPOINT_NPRW = OUT_BASE / "checkpoints" / "NPRW"
PATTERN = f"pp_local_{int(inner)}_{int(outer)}__interp_*"

# Parallel job settings (reuse your PARAMS)
global_job_kwargs = dict(n_jobs=getattr(PARAMS, "parallel_jobs", 1),
                         chunk_duration=getattr(PARAMS, "chunk", "2s"))
si.set_global_job_kwargs(**global_job_kwargs)

# ---------- SORTER CHOICE & PARAMS ----------
# Try Kilosort4 first, else fall back to Mountainsort5
AVAILABLE = ss.get_installed_sorters()
if "kilosort4" in AVAILABLE:
    SORTER = "kilosort4"
    sorter_params = {
        "detect_threshold": 5.0,
        "freq_min": 300.0,
        "freq_max": 6000.0,
        "phase_shift": False,
        "filter": False,
        "curation": False,
    }
elif "mountainsort5" in AVAILABLE:
    SORTER = "mountainsort5"
    sorter_params = {
        "detect_threshold": 5.0,
        "freq_min": 300.0,
        "freq_max": 6000.0,
        "phase_shift": False,
        "filter": False,
        "curation": False,
    }
else:
    print("[ERROR] Neither 'mountainsort5' nor 'tridesclous' is installed. Install one and re-run.")
    sys.exit(1)

print(f"[INFO] Using sorter: {SORTER} with params: {sorter_params}")

# ---------- MAIN ----------
def main():
    pp_dirs = sorted(CHECKPOINT_NPRW.glob(PATTERN))
    if not pp_dirs:
        print(f"[WARN] No preprocessed folders found at: {CHECKPOINT_NPRW}/{PATTERN}")
        return

    for pp_dir in pp_dirs:
        sess_name = pp_dir.name.replace(f"pp_local_{int(inner)}_{int(outer)}__interp_", "")
        print(f"\n[RUN] Session: {sess_name}")
        print(f"      Loading recording from: {pp_dir}")

        # Load the preprocessed recording saved by rcp.save_recording(...)
        try:
            rec = si.load_extractor(pp_dir)
        except Exception as e:
            print(f"[SKIP] Could not load extractor at {pp_dir}: {e}")
            continue

        # Output folders
        out_root   = pp_dir.parent / f"sorting_{SORTER}__{sess_name}"
        phy_folder = out_root / "phy"
        si_folder  = out_root / "sorting.si"
        out_root.mkdir(parents=True, exist_ok=True)

        # Run sorter
        try:
            sorting = ss.run_sorter(
                sorter_name=SORTER,
                recording=rec,
                output_folder=out_root / "sorter_work",
                remove_existing_folder=True,
                verbose=True,
                **sorter_params,
            )
        except Exception as e:
            print(f"[FAIL] Sorting failed for {sess_name}: {e}")
            del rec
            gc.collect()
            continue

        # Save the Sorting extractor (reloadable)
        try:
            si.save_extractor(sorting, si_folder, format="binary_folder", overwrite=True)
            print(f"[OK] Saved Sorting -> {si_folder}")
        except Exception as e:
            print(f"[WARN] Could not save Sorting extractor: {e}")

        # Export to Phy for curation (computes PCs & amplitudes)
        try:
            sie.export_to_phy(
                recording=rec,
                sorting=sorting,
                output_folder=phy_folder,
                compute_pc_features=True,
                compute_amplitudes=True,
                copy_binary=True,           # keep this self-contained
                remove_if_exists=True,
                sparse=True,                # speeds PC computation
            )
            print(f"[OK] Exported Phy project -> {phy_folder}")
        except Exception as e:
            print(f"[WARN] Phy export failed for {sess_name}: {e}")

        # Cleanup
        del rec, sorting
        gc.collect()

if __name__ == "__main__":
    main()







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
    
    # TODO can use something similar to cilantro postprocessing in custom_metrics.py to filter out bad waveforms
    
    sa.compute("templates")
    sa.compute("principal_components", n_components=5, mode="by_channel_local",
               n_jobs=int(PARAMS.parallel_jobs), chunk_duration=str(PARAMS.chunk), progress_bar=True)
    sa.compute("spike_amplitudes")

    export_to_phy(sa, output_folder=phy_folder, copy_binary=True, remove_if_exists=True)
    print(f"Phy export ready: {phy_folder}")