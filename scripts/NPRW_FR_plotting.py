#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
from RCP_analysis import load_experiment_params, resolve_output_root, load_stim_ms_from_stimstream, resolve_probe_geom_path, detect_stim_channels_from_npz, run_one_Intan_FR_heatmap

if __name__ == "__main__":
    # ==============================
    # Config
    # ==============================
    REPO_ROOT = Path(__file__).resolve().parents[1]
    PARAMS = load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
    OUT_BASE = resolve_output_root(PARAMS)
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # --- Geometry path ---
    GEOM_PATH = (
        Path(PARAMS.geom_mat_rel).resolve()
        if getattr(PARAMS, "geom_mat_rel", None) and str(PARAMS.geom_mat_rel).startswith("/")
        else (REPO_ROOT / PARAMS.geom_mat_rel).resolve()
        if getattr(PARAMS, "geom_mat_rel", None)
        else resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
    )

    ckpt_root    = OUT_BASE / "checkpoints" / "NPRW"
    bundles_root = OUT_BASE / "bundles" / "NPRW"
    figs_root    = OUT_BASE / "figures" / "peri_stim" / "NPRW"

    if not ckpt_root.exists():
        raise SystemExit(f"[error] checkpoints dir not found: {ckpt_root}")

    # Helper: session name from rates filename (between 'rates__' and '__bin')
    def session_from_rates_path(p: Path) -> str:
        stem = p.stem  # e.g., rates__NRR_RW001_250915_160649__bin1ms_sigma25ms
        if not stem.startswith("rates__"):
            raise ValueError(f"Unexpected rates filename: {p.name}")
        body = stem[len("rates__"):]
        sess = body.split("__bin", 1)[0]
        return sess

    # Find all rates files (recursively)
    rate_files = sorted(ckpt_root.glob("rates__*.npz"))
    if not rate_files:
        raise SystemExit(f"[error] No rates__*.npz found under {ckpt_root}")

    n_ok, n_skip = 0, 0
    print(f"[info] Found {len(rate_files)} Intan rates files. Processing...")

    for i, rates_npz in enumerate(rate_files, 1):
        try:
            # Subfolder under checkpoints (e.g., NPRW) to mirror fig and bundle layout
            try:
                rel = rates_npz.relative_to(ckpt_root)
                subproj = rel.parts[0] if len(rel.parts) > 1 else ""  # "" if directly under checkpoints
            except Exception:
                subproj = ""

            session = session_from_rates_path(rates_npz)
            stim_npz = bundles_root / f"{session}_Intan_bundle" / "stim_stream.npz"

            if not stim_npz.exists():
                print(f"[warn] Missing stim stream for session {session}: {stim_npz}  -> skipping")
                n_skip += 1
                continue

            # Load stim triggers (ms) and run
            stim_ms = load_stim_ms_from_stimstream(stim_npz)
            try:
                stim_idx = detect_stim_channels_from_npz(stim_npz, eps=1e-12, min_edges=1)
            except Exception:
                stim_idx = None
            out_dir = figs_root / subproj
            out_dir.mkdir(parents=True, exist_ok=True)

            # For the very first file, keep the single-channel debug overlay; otherwise off
            debug_ch = 0 if i == 1 else None

            print(f"[info] ({i}/{len(rate_files)}) session={session}  trials={stim_ms.size}")
            run_one_Intan_FR_heatmap(
                npz_path=rates_npz,
                out_dir=out_dir,
                win_ms=(-800.0, 1200.0),
                baseline_first_ms=100.0,
                min_trials=1,
                save_npz=True,
                stim_ms=stim_ms,
                debug_channel=debug_ch,
                intan_file=i,
                geom_path=GEOM_PATH,
                stim_idx=stim_idx,
            )
            n_ok += 1

        except Exception as e:
            print(f"[error] Failed on {rates_npz}: {e}")
            n_skip += 1

    print(f"[done] processed={n_ok}, skipped={n_skip}, figs at: {figs_root}")

