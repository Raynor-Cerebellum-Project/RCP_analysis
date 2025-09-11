# scripts/test_Utah_BR.py
from __future__ import annotations
from pathlib import Path
import gc
from typing import Optional
import spikeinterface as si
import spikeinterface.preprocessing as spre
from spikeinterface.core import concatenate_recordings

from RCP_analysis import (
    read_blackrock_recording,
    read_intan_recording,
    quicklook_stim_grid_all,
    load_ua_geom_from_excel,
    align_geom_index_to_recording,
)
from RCP_analysis.python.functions.params_loading import (
    load_experiment_params,
    resolve_data_root,
)

# ---------------- Module-level constants ----------------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_experiment_params(
    yaml_path=REPO_ROOT / "config" / "params.yaml",
    repo_root=REPO_ROOT,
)

OUT_BASE = (REPO_ROOT / "results").resolve()
OUT_BASE.mkdir(parents=True, exist_ok=True)

HPF_HZ = float(PARAMS.highpass_hz)
PARALLEL_JOBS = int(PARAMS.parallel_jobs)
THREADS_PER_WORKER = int(PARAMS.threads_per_worker)
CHUNK = str(PARAMS.chunk)

WIN_PRE_S = float(PARAMS.win_pre_s)
WIN_POST_S = float(PARAMS.win_post_s)
STIM_BAR_MS = float(PARAMS.stim_bar_ms)
DIG_LINE = None if not PARAMS.dig_line else PARAMS.dig_line

DEFAULT_STIM_NUM = int(PARAMS.default_stim_num)
STIM_NUMS = getattr(PARAMS, "stim_nums", {}) or {}


def _list_sessions(use_intan: bool = False) -> list[Path]:
    data_root = resolve_data_root(PARAMS)
    root = data_root if use_intan else data_root / PARAMS.blackrock_rel
    if not root.exists():
        raise FileNotFoundError(f"Session root not found: {root}")

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if subdirs:
        return sorted(subdirs)

    has_nsx = any(p.suffix.lower().startswith(".ns") for p in root.iterdir())
    has_nev = any(p.suffix.lower() == ".nev" for p in root.iterdir())
    if has_nsx or has_nev:
        return [root]

    raise FileNotFoundError(
        f"No sessions found under {root}.\n"
        "Expected subfolders OR .ns*/.nev files in this directory."
    )


def _probe_radii_for_session(sess_name: str) -> tuple[int, int]:
    sess_cfg = (PARAMS.sessions or {}).get(sess_name, {})
    probe_tag = sess_cfg.get("probe") or (next(iter(PARAMS.probes.keys())) if PARAMS.probes else None)
    if not probe_tag:
        raise ValueError("No probes defined in YAML.")

    probe_cfg = PARAMS.probes.get(probe_tag)
    if not probe_cfg:
        raise ValueError(f"Probe '{probe_tag}' not found in YAML 'probes'.")

    try:
        return int(probe_cfg["local_radius_inner"]), int(probe_cfg["local_radius_outer"])
    except KeyError as e:
        raise KeyError(f"Missing key in probes['{probe_tag}']: {e}")


def _ua_excel_path() -> Optional[Path]:
    """Prefer per-probe Excel geometry from YAML if present."""
    ua_cfg = (PARAMS.probes or {}).get("UA", {})
    rel = ua_cfg.get("geom_excel_rel")
    if not rel:
        return None
    p = (REPO_ROOT / rel) if not str(rel).startswith("/") else Path(rel)
    return p if p.exists() else None


def main(
    use_intan: bool = False,
    limit_sessions: Optional[int] = None,
    overwrite_saved: bool = True,
):
    session_folders = _list_sessions(use_intan=use_intan)
    if limit_sessions is not None:
        session_folders = session_folders[:limit_sessions]
    print("Found session folders:", len(session_folders))

    # --- Load UA geometry Excel once (if configured) ---
    ua_excel = _ua_excel_path()
    ua_map = None
    if not use_intan and ua_excel is not None:
        ua_map = load_ua_geom_from_excel(ua_excel)
    elif not use_intan and ua_excel is None:
        print("[WARN] UA Excel geometry not configured/found; proceeding without geometry mapping.")

    saved_paths: list[Path] = []

    for sess in session_folders:
        print(f"=== Session: {sess.name} ===")

        rec = read_intan_recording(sess, stream_name="RHS2000 amplifier channel") if use_intan \
              else read_blackrock_recording(sess)

        lr_inner, lr_outer = _probe_radii_for_session(sess.name)

        rec_hpf = spre.highpass_filter(rec, freq_min=HPF_HZ)
        rec_local = spre.common_reference(
            rec_hpf, reference="local", operator="median", local_radius=(lr_inner, lr_outer)
        )

        out_geom = OUT_BASE / f"pp_local_{lr_inner}_{lr_outer}__{sess.name}_GEOM"
        out_geom.mkdir(parents=True, exist_ok=True)
        rec_local.save(folder=out_geom, overwrite=overwrite_saved)
        print(f"[{sess.name}] saved permuted session -> {out_geom}")

        del rec, rec_hpf, rec_local
        gc.collect()

        try:
            rec_perm = si.load(out_geom)
        except Exception:
            rec_perm = si.load_extractor(out_geom / "si_folder.json")

        fs = float(rec_perm.get_sampling_frequency())
        stim_num = STIM_NUMS.get(sess.name, DEFAULT_STIM_NUM)

        quicklook_dir = OUT_BASE / "quicklooks"
        quicklook_dir.mkdir(parents=True, exist_ok=True)

        # If we have UA geometry, align to recording row order (available for downstream use)
        geom_idx_rows = None
        if ua_map is not None:
            try:
                geom_idx_rows = align_geom_index_to_recording(rec_perm, ua_map["geom_corr_ind_nsp"])
            except Exception as e:
                print(f"[WARN] Could not align UA geometry to recording: {e}")

        # If your quicklook accepts geometry/indices, pass them here (left as-is if not needed)
        quicklook_stim_grid_all(
            rec_sess=rec_perm,
            sess_folder=sess,
            out_dir=quicklook_dir,
            fs=fs,
            stim_num=stim_num,
            nrows=int(PARAMS.quicklook_rows),
            ncols=int(PARAMS.quicklook_cols),
            pre_s=WIN_PRE_S,
            post_s=WIN_POST_S,
            bar_ms=STIM_BAR_MS,
            dig_line=DIG_LINE,
            stride=int(PARAMS.stride),
            # geom_idx_rows=geom_idx_rows,  # uncomment if your function supports it
        )

        saved_paths.append(out_geom)
        del rec_perm
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
    return sorting_ms5


if __name__ == "__main__":
    main(use_intan=False, limit_sessions=None, overwrite_saved=True)
