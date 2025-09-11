# scripts/test_Utah_BR...
from __future__ import annotations
from pathlib import Path
import gc

import spikeinterface as si
import spikeinterface.preprocessing as spre
from spikeinterface.core import concatenate_recordings

from RCP_analysis import (
    read_blackrock_recording,
    read_intan_recording,        # optional
    load_sample_index_from_cal,
    neighbors_in_radius,
    quicklook_stim_grid_all,
    permuted_geometry_view,
    save_neighbors_csv,
    load_stim_geometry,
)

# ==============================
# Paths (repo-relative + host auto-switch)
# ==============================
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
OUT_BASE   = (REPO_ROOT / "results").resolve()
OUT_BASE.mkdir(parents=True, exist_ok=True)

def set_paths() -> Path:
    """Return the correct data root for this host."""
    mac_root  = Path("/Volumes/cullenlab_server/Current Project Databases - NHP/2025 Cerebellum prosthesis")
    linux_root= Path("/mnt/cullen/Current Project Databases - NHP/2025 Cerebellum prosthesis")
    if mac_root.exists():
        return mac_root
    if linux_root.exists():
        return linux_root
    # Fallback: repo-local data dir so script doesn’t crash
    return (REPO_ROOT / "data").resolve()

DATA_ROOT = set_paths()
blackrock_root = DATA_ROOT / "Nike/Nike_reaching_002/Blackrock"
SESSION_FOLDERS = sorted([p for p in blackrock_root.iterdir() if p.is_dir()])
print("Found session folders:", len(SESSION_FOLDERS))

# Geometry / config (adjust to your helper’s signature)
GEOM_MAT = CONFIG_DIR / "ImecPrimateStimRec128_042421.mat"
stim_geom = None
if GEOM_MAT.exists():
    try:
        stim_geom = load_stim_geometry(GEOM_MAT)  # if your function returns an index/mapping
    except Exception as e:
        print(f"[WARN] Could not load geometry from {GEOM_MAT}: {e}")

# ---- knobs ----
PARALLEL_JOBS = 2
THREADS_PER_WORKER = 1
CHUNK = "0.5s"

WIN_PRE_S  = 0.050
WIN_POST_S = 0.250
STIM_BAR_MS= 1.0
DIG_LINE   = None
DEFAULT_STIM_NUM = 1
STIM_NUMS = {}  # e.g., {"BL_RW_003_Session_1": 2}


# ==============================
# Main
# ==============================
saved_paths = []

for sess in SESSION_FOLDERS:
    print(f"=== Session: {sess.name} ===")

    # Choose the loader you need (keep both ready)
    rec = read_blackrock_recording(sess)
    # rec = read_intan_recording(sess, stream_name="RHS2000 amplifier channel")

    # Preprocess
    rec_hpf   = spre.highpass_filter(rec, freq_min=300)
    rec_local = spre.common_reference(rec_hpf, reference="local", operator="median", local_radius=(60, 150))

    # Optional: strict index length check if *Cal.mat provides sample index
    idx0 = load_sample_index_from_cal(sess)  # returns None if no cal file found
    if idx0 is not None:
        n_rec = int(rec.get_num_frames())
        if idx0.size != n_rec:
            print(f"[{sess.name}] WARN: idx length {idx0.size} != recording frames {n_rec} (skipping strict check).")

    # Optional channel permutation (only if you truly need to match MATLAB order)
    rec_local_perm = permuted_geometry_view(rec_local)

    # Save preprocessed, permuted data for this session
    out_geom = OUT_BASE / f"pp_local_60_150__{sess.name}_GEOM"
    out_geom.mkdir(parents=True, exist_ok=True)

    rec_local_perm.save(folder=out_geom, overwrite=True)
    print(f"[{sess.name}] saved permuted session -> {out_geom}")

    # Free upstream objects ASAP
    del rec, rec_hpf, rec_local, rec_local_perm
    gc.collect()

    # Reload saved extractor
    try:
        rec_perm = si.load(out_geom)
    except Exception:
        rec_perm = si.load_extractor(out_geom / "si_folder.json")

    # Quick diagnostics / figures
    fs = float(rec_perm.get_sampling_frequency())
    stim_num = STIM_NUMS.get(sess.name, DEFAULT_STIM_NUM)

    # one-time neighbor map
    if sess == SESSION_FOLDERS[0]:
        neighbor_map = neighbors_in_radius(rec_perm, r_min_um=60.0, r_max_um=150.0, same_shank=True, return_ids=True)
        csv_out = OUT_BASE / "neighbors" / f"neighbors_{sess.name}_r60-150.csv"
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        save_neighbors_csv(neighbor_map, csv_out)

    quicklook_dir = OUT_BASE / "quicklooks"
    quicklook_dir.mkdir(parents=True, exist_ok=True)

    quicklook_stim_grid_all(
        rec_sess=rec_perm,
        sess_folder=sess,
        out_dir=quicklook_dir,
        fs=fs,
        stim_num=stim_num,
        nrows=4, ncols=4,
        pre_s=WIN_PRE_S, post_s=WIN_POST_S,
        bar_ms=STIM_BAR_MS,
        dig_line=DIG_LINE,
        stride=16,
    )

    saved_paths.append(out_geom)

    del rec_perm
    gc.collect()

# Concatenate all sessions
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

# ====== Run sorters ======
# MountainSort5
sorting_ms5 = si.run_sorter(
    sorter_name="mountainsort5",
    recording=rec_concat,
    output_folder=OUT_BASE / "mountainsort5",
    verbose=True,
    remove_existing_folder=True,
    docker_image=None,
    **{
        "n_jobs": PARALLEL_JOBS,
        "chunk_duration": CHUNK,
        "pool_engine": "process",
        "max_threads_per_worker": THREADS_PER_WORKER,
    },
)

print("Done. You can export/open in phy using your existing helpers if desired.")
