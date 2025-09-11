# scripts/test_Utah_BR.py
from __future__ import annotations
from pathlib import Path
import gc
from typing import List, Optional
import spikeinterface as si
import spikeinterface.preprocessing as spre
from spikeinterface.core import concatenate_recordings

from RCP_analysis import (
    read_blackrock_recording,
    read_intan_recording,
    quicklook_stim_grid_all,
    load_br_geometry,
    load_session_geometry,
)
from RCP_analysis.python.functions.params_loading import (
    load_experiment_params,
    resolve_data_root,
)

# ---------------- Module-level constants ----------------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_experiment_params(  # YAML, not TOML
    yaml_path=REPO_ROOT / "config" / "params.yaml",
    repo_root=REPO_ROOT,
)

OUT_BASE = (REPO_ROOT / "results").resolve()
OUT_BASE.mkdir(parents=True, exist_ok=True)

# Short aliases from params
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
    if use_intan:
        root = data_root  # TODO: add an intan_rel in YAML when you have it
    else:
        root = data_root / PARAMS.blackrock_rel

    if not root.exists():
        raise FileNotFoundError(f"Session root not found: {root}")

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if subdirs:
        return sorted(subdirs)

    # No subdirs; check for Blackrock files sitting in this directory
    has_nsx = any(p.suffix.lower().startswith(".ns") for p in root.iterdir())
    has_nev = any(p.suffix.lower() == ".nev" for p in root.iterdir())
    if has_nsx or has_nev:
        return [root]  # treat the folder as a single session

    raise FileNotFoundError(
        f"No sessions found under {root}.\n"
        "Expected subfolders OR .ns*/.nev files in this directory."
    )


def _probe_radii_for_session(sess_name: str) -> tuple[int, int]:
    """
    Pull (inner, outer) radii straight from YAML:
      - If session mapped in PARAMS.sessions with `probe`, use that probe's radii.
      - Else fall back to the first probe defined in PARAMS.probes.
    No extra resolver; just declarative YAML.
    """
    sess_cfg = (PARAMS.sessions or {}).get(sess_name, {})
    probe_tag = sess_cfg.get("probe")

    if not probe_tag:
        # fallback to the first probe defined
        if not PARAMS.probes:
            raise ValueError("No probes defined in YAML.")
        probe_tag = next(iter(PARAMS.probes.keys()))

    probe_cfg = PARAMS.probes.get(probe_tag)
    if not probe_cfg:
        raise ValueError(f"Probe '{probe_tag}' not found in YAML 'probes'.")

    try:
        lr_in = int(probe_cfg["local_radius_inner"])
        lr_out = int(probe_cfg["local_radius_outer"])
    except KeyError as e:
        raise KeyError(f"Missing key in probes['{probe_tag}']: {e}")

    return lr_in, lr_out


def main(
    use_intan: bool = False,
    limit_sessions: Optional[int] = None,
    overwrite_saved: bool = True,
):
    """
    Run the Utah BR preprocessing + concatenation + mountainsort5 pipeline.
    """
    # List sessions at run-time (not import-time)
    session_folders = _list_sessions(use_intan=use_intan)
    if limit_sessions is not None:
        session_folders = session_folders[:limit_sessions]

    print("Found session folders:", len(session_folders))

    saved_paths: list[Path] = []

    for sess in session_folders:
        print(f"=== Session: {sess.name} ===")

        # Choose the loader you need
        if use_intan:
            rec = read_intan_recording(sess, stream_name="RHS2000 amplifier channel")
        else:
            rec = read_blackrock_recording(sess)

        # Per-session probe radii from YAML
        lr_inner, lr_outer = _probe_radii_for_session(sess.name)

        # Preprocess
        rec_hpf = spre.highpass_filter(rec, freq_min=HPF_HZ)
        rec_local = spre.common_reference(
            rec_hpf,
            reference="local",
            operator="median",
            local_radius=(lr_inner, lr_outer),
        )

        # Save preprocessed, permuted data for this session
        out_geom = OUT_BASE / f"pp_local_{lr_inner}_{lr_outer}__{sess.name}_GEOM"
        out_geom.mkdir(parents=True, exist_ok=True)

        rec_local.save(folder=out_geom, overwrite=overwrite_saved)
        print(f"[{sess.name}] saved permuted session -> {out_geom}")

        # Free upstream objects ASAP
        del rec, rec_hpf, rec_local
        gc.collect()

        # Reload saved extractor
        try:
            rec_perm = si.load(out_geom)
        except Exception:
            rec_perm = si.load_extractor(out_geom / "si_folder.json")

        # Quick diagnostics / figures
        fs = float(rec_perm.get_sampling_frequency())
        stim_num = STIM_NUMS.get(sess.name, DEFAULT_STIM_NUM)

        quicklook_dir = OUT_BASE / "quicklooks"
        quicklook_dir.mkdir(parents=True, exist_ok=True)

        # Per-session UA geometry (if available)
        ua_geom = load_session_geometry(
            sess_name=sess.name, params=PARAMS, repo_root=REPO_ROOT
        )
        geom_idx = ua_geom.get("geom_corr_ind") if ua_geom else None
        xy = ua_geom.get("xy") if ua_geom else None

        # If your quicklook accepts geometry, pass it (uncomment the arg below if supported)
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
    # Tweak args here for one-off runs
    main(use_intan=False, limit_sessions=None, overwrite_saved=True)
