from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import gc
import re
import numpy as np

# SpikeInterface
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
from spikeinterface import sorters
from spikeinterface.core import concatenate_recordings
from spikeinterface import create_sorting_analyzer
from spikeinterface.exporters import export_to_phy


# Project imports
from RCP_analysis import (
    load_UA_mapping_from_excel,
    apply_ua_geometry_to_recording,
)
from RCP_analysis.python.functions.params_loading import (
    load_experiment_params,
    resolve_data_root,
)

# ----------------------
# Config / params
# ----------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_experiment_params(
    yaml_path=REPO_ROOT / "config" / "params.yaml",
    repo_root=REPO_ROOT,
)
OUT_BASE = (REPO_ROOT / "results").resolve()
OUT_BASE.mkdir(parents=True, exist_ok=True)

# ----------------------
# Session discovery
# ----------------------
def _list_sessions(use_intan: bool = False) -> list[Path]:
    data_root = resolve_data_root(PARAMS)
    root = data_root if use_intan else data_root / PARAMS.blackrock_rel
    if not root.exists():
        raise FileNotFoundError(f"Session root not found: {root}")

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if subdirs:
        return sorted(subdirs)

    files = [p for p in root.iterdir() if p.is_file()]
    nsx_nev = [p for p in files if p.suffix.lower().startswith(".ns") or p.suffix.lower() == ".nev"]
    if not nsx_nev:
        raise FileNotFoundError(f"No sessions found under {root}.")
    bases = sorted({p.stem for p in nsx_nev})
    return [root / b for b in bases]

def _ua_excel_path() -> Optional[Path]:
    ua_cfg = (PARAMS.probes or {}).get("UA", {})
    rel = ua_cfg.get("mapping_excel_rel") or ua_cfg.get("mapping_mat_rel")
    if not rel:
        return None
    p = (REPO_ROOT / rel) if not str(rel).startswith("/") else Path(rel)
    return p if p.exists() else None

# ----------------------
# BR loaders & helpers
# ----------------------
def load_nsx(sess: Path, ext: str):
    sess = Path(sess)
    root = sess if sess.is_dir() else sess.parent
    base = sess.name if not sess.is_dir() else sorted({p.stem for p in root.glob("*.ns*")})[0]
    f = root / f"{base}.{ext}"
    if not f.exists():
        raise FileNotFoundError(f"Missing {ext} for base {base} in {root}")
    print(f"[LOAD] {f.name}")
    return se.read_blackrock(str(f), all_annotations=True)

def load_ns6_spikes(sess: Path):
    return load_nsx(sess, "ns6")   # 30 kHz, channels 1..128

def load_ns5_aux(sess: Path):
    return load_nsx(sess, "ns5")   # 30 kHz, (134, 138) here

def load_ns2_lfp(sess: Path):
    return load_nsx(sess, "ns2")   # 1 kHz, 129..133,135..137,139..144

def _ensure_int_ids(rec) -> np.ndarray:
    ids = rec.get_channel_ids()
    out = []
    for cid in ids:
        try:
            out.append(int(cid))
        except Exception:
            m = re.search(r"(\d+)", str(cid))
            out.append(int(m.group(1)) if m else None)
    return np.array(out, dtype=int)

def _pick_cols_by_ids(rec, wanted_ids) -> Tuple[list[int], list[int]]:
    ids = _ensure_int_ids(rec)
    id2col = {ch: i for i, ch in enumerate(ids)}
    cols = [id2col[c] for c in wanted_ids if c in id2col]
    missing = [c for c in wanted_ids if c not in id2col]
    return cols, missing

# --- 256×T from ns6 using UA map ---
def ns6_to_elec_matrix_256(rec_ns6, ua_map: dict, fill_value=np.nan, max_seconds: float | None = None):
    mapped_nsp = ua_map["mapped_nsp"]  # length 256, value = NSP ch id (1..128) or 0
    if mapped_nsp.size != 256:
        raise ValueError(f"Expected 256-entry mapping; got {mapped_nsp.size}")

    fs = float(rec_ns6.get_sampling_frequency())
    n_frames = rec_ns6.get_num_frames(0)
    end_frame = n_frames if max_seconds is None else min(n_frames, int(max_seconds * fs))

    ids_int = _ensure_int_ids(rec_ns6)  # e.g., [1..128]
    nsp_to_col = {ch: idx for idx, ch in enumerate(ids_int)}

    traces = rec_ns6.get_traces(segment_index=0, start_frame=0, end_frame=end_frame).astype(np.float32)  # (T, n_ch)

    M = np.full((256, traces.shape[0]), fill_value, dtype=np.float32)  # (256, T)
    for elec_i, nsp_ch in enumerate(mapped_nsp):
        if nsp_ch <= 0:
            continue
        col = nsp_to_col.get(int(nsp_ch))
        if col is not None:
            M[elec_i, :] = traces[:, col]
    return M, fs

# ----------------------
# Bundle build + save
# ----------------------
def build_blackrock_bundle(sess: Path, ua_map: dict) -> dict:
    """
    bundle['neural_data']: {'data': (256,T) float32, 'fs': float, 'mapped_nsp': np.ndarray}
    bundle['aux']:         {'fs': float|None, 'camera_sync': 1D|None, 'intan_sync': 1D|None}
    bundle['lfp_ns2']:     {'fs': float|None, 'channels': {'ch129':1D, ...}}
    """
    bundle: Dict[str, Any] = {}

    # ns6 -> 256×T neural matrix
    rec_ns6 = load_ns6_spikes(sess)
    M256, fs_ns6 = ns6_to_elec_matrix_256(rec_ns6, ua_map, fill_value=np.nan, max_seconds=None)
    bundle["neural_data"] = {"data": M256, "fs": fs_ns6, "mapped_nsp": ua_map["mapped_nsp"]}

    # ns5 -> camera_sync (134), intan_sync (138)
    camera_sync = None
    intan_sync = None
    fs_ns5 = None
    try:
        rec_ns5 = load_ns5_aux(sess)
        fs_ns5 = float(rec_ns5.get_sampling_frequency())
        cols, missing = _pick_cols_by_ids(rec_ns5, [134, 138])
        if missing:
            print(f"[WARN] ns5 missing channels: {missing}")
        if cols:
            tr = rec_ns5.get_traces(segment_index=0, start_frame=0, end_frame=rec_ns5.get_num_frames(0)).astype(np.float32)
            ids = _ensure_int_ids(rec_ns5) # Can remove
            for col in cols:
                ch_id = int(ids[col])
                if ch_id == 134:
                    camera_sync = tr[:, col]
                elif ch_id == 138:
                    intan_sync = tr[:, col]
    except FileNotFoundError:
        print("[WARN] ns5 not found; aux syncs unavailable.")

    bundle["aux"] = {"fs": fs_ns5, "camera_sync": camera_sync, "intan_sync": intan_sync}

    # ns2 -> individual channels ch129, ch130, ...
    lfp = {"fs": None, "channels": {}}
    try:
        rec_ns2 = load_ns2_lfp(sess)
        fs_ns2 = float(rec_ns2.get_sampling_frequency())
        lfp["fs"] = fs_ns2
        wanted = [129,130,131,132,133,135,136,137,139,140,141,142,143,144]
        cols, missing = _pick_cols_by_ids(rec_ns2, wanted)
        if missing:
            print(f"[WARN] ns2 missing channels: {missing}")
        if cols:
            tr = rec_ns2.get_traces(segment_index=0, start_frame=0, end_frame=rec_ns2.get_num_frames(0)).astype(np.float32)
            ids = _ensure_int_ids(rec_ns2)
            for col in cols:
                ch_id = int(ids[col])
                lfp["channels"][f"ch{ch_id}"] = tr[:, col]
    except FileNotFoundError:
        print("[WARN] ns2 not found; no LFP channels.")

    bundle["lfp_ns2"] = lfp
    return bundle

def save_bundle_npz(sess_name: str, bundle: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{sess_name}_bundle.npz"

    # Flatten dict into arrays + small metadata
    np.savez_compressed(
        npz_path,
        neural_data=bundle["neural_data"]["data"],
        neural_fs=np.array(bundle["neural_data"]["fs"], dtype=np.float64),
        mapped_nsp=bundle["neural_data"]["mapped_nsp"],
        aux_fs=np.array(bundle["aux"]["fs"] if bundle["aux"]["fs"] is not None else np.nan, dtype=np.float64),
        camera_sync=bundle["aux"]["camera_sync"] if bundle["aux"]["camera_sync"] is not None else np.array([]),
        intan_sync=bundle["aux"]["intan_sync"] if bundle["aux"]["intan_sync"] is not None else np.array([]),
        lfp_fs=np.array(bundle["lfp_ns2"]["fs"] if bundle["lfp_ns2"]["fs"] is not None else np.nan, dtype=np.float64),
        **{k: v for k, v in bundle["lfp_ns2"]["channels"].items()},
    )
    print(f"[SAVED] {npz_path}")

# ----------------------
# Main
# ----------------------
def main(use_intan: bool = False, limit_sessions: Optional[int] = None):
    session_folders = _list_sessions(use_intan=use_intan)
    if limit_sessions is not None:
        session_folders = session_folders[:limit_sessions]
    print("Found session folders:", len(session_folders))

    ua_excel = _ua_excel_path()
    ua_map = load_UA_mapping_from_excel(ua_excel) if ua_excel else None
    if ua_map is None:
        raise RuntimeError("UA mapping required to build neural_data (ns6→256×T).")
    saved_paths: list[Path] = []
    bundles_out = OUT_BASE / "bundles"
    for sess in session_folders:
        print(f"=== Session: {sess.name} ===")
        bundle = build_blackrock_bundle(sess, ua_map)
        # === add per-session preprocessing for spike sorting (NS6 only) ===
        rec_ns6_for_sort = load_ns6_spikes(sess)

        # Stamp UA geometry so local/global ref & Phy have locations
        try:
            apply_ua_geometry_to_recording(rec_ns6_for_sort, ua_map["mapped_nsp"], pitch_mm=0.4)
        except Exception as e:
            print(f"[WARN] ({sess.name}) could not set UA geometry on ns6: {e}")

        # Highpass
        rec_hp = spre.highpass_filter(rec_ns6_for_sort, freq_min=float(PARAMS.highpass_hz))

        # Global median reference (you asked for global explicitly)
        rec_ref = spre.common_reference(rec_hp, reference="global", operator="median")

        # Persist a stable extractor for later concat/sorting
        out_geom = OUT_BASE / f"pp_global__{sess.name}__NS6"
        out_geom.mkdir(parents=True, exist_ok=True)
        rec_ref.save(folder=out_geom, overwrite=True)
        print(f"[{sess.name}] (ns6) saved preprocessed -> {out_geom}")

        # Free temps
        del rec_ns6_for_sort, rec_hp, rec_ref
        gc.collect()

        # Keep track for concatenation/sorting
        saved_paths.append(out_geom)
        save_bundle_npz(sess.name, bundle, bundles_out)
        del bundle
        gc.collect()
    if not saved_paths:
        raise RuntimeError("No sessions processed; nothing to concatenate.")

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

    # --- run sorting ---
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
    )

    sa_folder = OUT_BASE / "sorting_ms5_analyzer"
    phy_folder = OUT_BASE / "phy_ms5"

    # Create sorting analyzer object
    sa = create_sorting_analyzer(
        sorting=sorting_ms5,
        recording=rec_concat,
        folder=sa_folder,
        overwrite=True,
        sparse=True,              # True if you want sparse channel neighborhoods
    )

    # Compute phy features
    #    (tweak these to your preference; these are common defaults)
    sa.compute(
        "waveforms",
        ms_before=1.0, # 1ms before spike
        ms_after=2.0,  # 2ms after spike
        max_spikes_per_unit=1000,
        n_jobs=int(PARAMS.parallel_jobs),
        chunk_duration=str(PARAMS.chunk),
        progress_bar=True,
    )
    sa.compute("templates")
    sa.compute(
        "principal_components",
        n_components=5,            # typical for Phy
        mode="by_channel_local",   # common choice; can be "by_channel_global" as well
        n_jobs=int(PARAMS.parallel_jobs),
        chunk_duration=str(PARAMS.chunk),
        progress_bar=True,
    )
    sa.compute("spike_amplitudes")

    # 3) export to Phy (works directly from SortingAnalyzer)
    export_to_phy(
        sa,
        output_folder=phy_folder,
        compute_pc_features=False,
        compute_amplitudes=False,
        copy_binary=True,
        remove_if_exists=True,
    )

    print(f"Phy export ready: {phy_folder}")


if __name__ == "__main__":
    main(use_intan=False, limit_sessions=None)
