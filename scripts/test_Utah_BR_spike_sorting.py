from pathlib import Path
import spikeinterface as si
import spikeinterface.preprocessing as spre

from functions.utils import (
    read_blackrock_recording,
    read_intan_recording,
    load_stim_geometry,
    load_sample_index_from_cal,
)

# ==============================
# Config (Blackrock)
# ==============================
DATA_DIR = Path("/home/bryan/mnt/cullen/Current Project Databases - NHP/2025 Cerebellum prosthesis")
GEOM_MAT = DATA_DIR / "BL_RW_003_Session_1/ImecPrimateStimRec128_042421.mat"
blackrock_root = DATA_DIR / "Nike/Nike_Utah_pilot/Blackrock"
SESSION_FOLDERS = sorted([p for p in blackrock_root.iterdir() if p.is_dir()])

# Each subfolder under this root should contain one session's Blackrock files
blackrock_root = DATA_DIR / "Nike/Nike_Utah_pilot/Blackrock"
SESSION_FOLDERS = sorted([p for p in blackrock_root.iterdir() if p.is_dir()])
print("Found session folders:", len(SESSION_FOLDERS))

# Example: if you have a mapping for Neuropixels (device→geometry), pass it; for Utah, set None.
device_to_geometry_index_0based = None  # or np.asarray(NEUROPIXEL_INDEX_1BASED) - 1

for sess in SESSION_FOLDERS:
    # Choose the loader you need:
    rec = read_blackrock_recording(sess)                 # Blackrock path
    # rec = read_intan_recording(sess, stream_name="RHS2000 amplifier channel")  # Intan path

    # Set probe elsewhere (your probes module), then preprocess
    # rec = rec.set_probe(probe, in_place=False)
    rec_hpf   = spre.highpass_filter(rec, freq_min=300)
    rec_local = spre.common_reference(rec_hpf, reference="local", operator="median", local_radius=(60, 150))

    # Optional sanity check if your *Cal.mat provides an index vector
    idx0 = _load_idx_maybe(folder, key_candidates=("Blackrock_idx", "Intan_idx"))
    if idx0 is not None:
        n_rec = int(rec.get_num_frames())
        if idx0.size != n_rec:
            print(f"[{folder.name}] WARN: idx length {idx0.size} != recording frames {n_rec} (skipping strict check).")


    # preprocess per session
    rec_hpf   = spre.highpass_filter(rec, freq_min=300)
    rec_local = spre.common_reference(rec_hpf, reference="local", operator="median", local_radius=(60, 150))

    # ---- APPLY THE MATLAB PERMUTATION BEFORE SAVING (match neuropixel_index) ----
    rec_local_perm = permuted_geometry_view(rec_local)
    # save only the permuted, preprocessed data
    out_geom = OUT_BASE / f"pp_local_30_150__{folder.name}_GEOM"

    # free upstream objects ASAP
    rec_local_perm.save(folder=out_geom, overwrite=True)
    del rec_hpf, rec, rec_local, rec_local_perm
    gc.collect()

    # ---- reload the SAME permuted view (no extra reordering) ----
    try:
        rec_perm = si.load(out_geom)
    except Exception:
        rec_perm = si.load_extractor(out_geom / "si_folder.json")

    # quicklooks in the MATLAB order
    fs = float(rec_perm.get_sampling_frequency())
    stim_num = STIM_NUMS.get(folder.name, DEFAULT_STIM_NUM)
    
    # quick diagnostic of neighbors used for local referencing (matches local_radius=(30,150))
    if idx == 1:
        neighbor_map = neighbors_in_radius(rec_perm, r_min_um=30.0, r_max_um=150.0, same_shank=True, return_ids=True)
        # Save to CSV for later inspection
        csv_out = OUT_BASE / "neighbors" / f"neighbors_{folder.name}_r30-150.csv"
        save_neighbors_csv(neighbor_map, csv_out)

    
    quicklook_stim_grid_all(
        rec_sess=rec_perm,
        sess_folder=folder,
        out_dir=OUT_BASE / "quicklooks",
        fs=fs,
        stim_num=stim_num,
        nrows=4, ncols=4,
        pre_s=WIN_PRE_S, post_s=WIN_POST_S,
        bar_ms=STIM_BAR_MS,
        dig_line=DIG_LINE,
        stride=16,
    )
    print(f"[{folder.name}] saved permuted session -> {out_geom}")

    # cleanup (only what still exists)
    del rec_perm
    gc.collect()

    # concat the permuted saves
    saved_paths.append(out_geom)
        
# concatenate all preprocessed sessions from disk (lazy)
print("Concatenating preprocessed sessions...")
recs_for_concat = []
for p in saved_paths:
    try:
        r = si.load(p)
    except Exception:
        r = si.load_extractor(p / "si_folder.json")
    recs_for_concat.append(r)
rec_concat = concatenate_recordings(recs_for_concat)
gc.collect()  # optional

# Run MountainSort5 (tweak params as you like)
sorting_ms5, analyzer_ms5 = run_sorter_and_export(
    "mountainsort5",
    rec_concat,
    OUT_BASE,
    sorter_params={
        "n_jobs": PARALLEL_JOBS,
        "chunk_duration": CHUNK,
        "pool_engine": "process",          # process pool (good isolation)
        "max_threads_per_worker": THREADS_PER_WORKER,
        # optional MS5 knobs if you need: "detect_threshold": 5, "detect_sign": "neg"
    },
)

# If desired, free the individual handles now (concat keeps references)
del recs_for_concat, saved_paths
gc.collect()

# Run KiloSort4 (if GPU available)
# sorting_ks4, analyzer_ks4 = run_sorter_and_export("kilosort4", rec_concat, OUT_BASE)

print("Done. Open with:")
print(f"  cd '{OUT_BASE}/phy_output_mountainsort5' && phy template-gui params.py")
print(f"  cd '{OUT_BASE}/phy_output_kilosort4' && phy template-gui params.py")
