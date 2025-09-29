#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
from RCP_analysis import load_experiment_params, resolve_output_root, load_stim_ms_from_stimstream, resolve_probe_geom_path, detect_stim_channels_from_npz, run_one_Intan_FR_heatmap, plot_single_channel_trial_quad_raw, load_stim_detection, _find_interp_dir_for_session, _load_cached_recording


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

            
            if i == 3:
                # ---- Window declarations (single source of truth) ----
                raw_window_ms = (90.0, 150.0)       # for per-channel raw trial quads
                heatmap_debug_window_ms = (0.0, 250.0)

                # load AC recording for this session
                ac_dir = _find_interp_dir_for_session(OUT_BASE / "checkpoints", session)
                rec = _load_cached_recording(ac_dir)
                try:
                    fs = float(rec.get_sampling_frequency())
                except Exception:
                    fs = float(rec.get_sampling_frequency(0))

                # get stim start sample indices
                stim = load_stim_detection(stim_npz)
                tp = np.asarray(stim.get("trigger_pairs", []))
                if tp.ndim != 2 or tp.shape[1] != 2 or tp.size == 0:
                    raise RuntimeError(f"No trigger_pairs in {stim_npz}")
                stim_start_samples = tp[:, 0].astype(np.int64)

                # Make per-channel files for channels 0..15 (i.e., 1–16 one-based)
                debug_dir = (figs_root / "debug_trials" / session)
                debug_dir.mkdir(parents=True, exist_ok=True)

                # Load peaks for overlays (robust to both "new" and "legacy" saves)
                peak_ch = None
                peak_t  = None
                with np.load(rates_npz, allow_pickle=True) as z:
                    if "peak_ch" in z and "peak_t_ms" in z:
                        peak_ch = z["peak_ch"].astype(int)
                        peak_t  = z["peak_t_ms"].astype(float)
                    elif "peaks" in z:
                        P = z["peaks"]
                        if getattr(P, "dtype", None) is not None and P.dtype.names:
                            if "channel_index" in P.dtype.names and "t_ms" in P.dtype.names:
                                peak_ch = P["channel_index"].astype(int)
                                peak_t  = P["t_ms"].astype(float)
                            elif "channel_index" in P.dtype.names and "sample_index" in P.dtype.names and "meta" in z:
                                fs_hz = float(z["meta"].item().get("fs", 1.0))
                                peak_ch = P["channel_index"].astype(int)
                                peak_t  = (P["sample_index"].astype(float) / fs_hz) * 1000.0
                        else:
                            # Nx2 fallback: columns [ch, t_ms]
                            arr = np.asarray(P)
                            if arr.ndim == 2 and arr.shape[1] >= 2:
                                peak_ch = arr[:, 0].astype(int)
                                peak_t  = arr[:, 1].astype(float)

                # Per-channel plots (0..15)
                for ch in range(16):
                    out_png = debug_dir / f"{session}__ch{ch:03d}__raw_trials_quad_{int(raw_window_ms[0])}-{int(raw_window_ms[1])}ms.png"

                    # Filter overlays to this channel (optional but keeps it light)
                    if peak_ch is not None and peak_t is not None:
                        mask = (peak_ch == ch)
                        peak_t_ch = peak_t[mask]
                        peak_ch_arg = np.array([ch], dtype=int)
                        peak_t_arg  = peak_t_ch
                    else:
                        peak_ch_arg = None
                        peak_t_arg  = None

                    plot_single_channel_trial_quad_raw(
                        rec, fs, stim_start_samples, ch, out_png,
                        window_ms=raw_window_ms, n_show=4,
                        title_prefix="Stim-aligned raw voltage",
                        peak_ch=peak_ch_arg, peak_t_ms=peak_t_arg,  # overlay spikes
                    )

            print(f"[info] ({i}/{len(rate_files)}) session={session}  trials={stim_ms.size}")

            run_one_Intan_FR_heatmap(
                npz_path=rates_npz,
                out_dir=out_dir,
                win_ms=(-800.0, 1200.0),
                baseline_first_ms=100.0,
                min_trials=1,
                save_npz=True,
                stim_ms=stim_ms,
                intan_file=i,
                geom_path=GEOM_PATH,
                stim_idx=stim_idx,
                debug_chans=(list(range(0, 16)) if i == 3 else None),   # channels 1–16 → 0–15
                debug_window_ms=(heatmap_debug_window_ms if i == 3 else None),
            )

            n_ok += 1

        except Exception as e:
            print(f"[error] Failed on {rates_npz}: {e}")
            n_skip += 1

    print(f"[done] processed={n_ok}, skipped={n_skip}, figs at: {figs_root}")

