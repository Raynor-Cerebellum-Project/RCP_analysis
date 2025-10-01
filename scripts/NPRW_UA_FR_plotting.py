#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- project helpers (same entrypoints you already use) ----
from RCP_analysis import (
    load_experiment_params,
    resolve_output_root,
    resolve_probe_geom_path,
    # reuse your NPRW logic for peri-stim:
    extract_peristim_segments,
    baseline_zero_each_trial,
    average_across_trials,
    plot_channel_heatmap,            # we’ll use this for the top (Intan)
    build_probe_and_locs_from_geom,  # probe for Intan only (Utah geom not provided)
)

def _load_combined_npz(p: Path):
    d = np.load(p, allow_pickle=True)
    # Intan
    i_rate = d["intan_rate_hz"]
    i_t    = d["intan_t_ms_aligned"] if "intan_t_ms_aligned" in d.files else d["intan_t_ms"]
    # UA
    u_rate = d["ua_rate_hz"]
    u_t    = d["ua_t_ms_aligned"] if "ua_t_ms_aligned" in d.files else d["ua_t_ms"]
    # stim (absolute Intan ms); we’ll align it below
    stim_ms = d["stim_ms"] if "stim_ms" in d.files else np.array([], dtype=float)
    # alignment meta (JSON)
    meta = json.loads(d["align_meta"].item()) if "align_meta" in d.files else {}
    return i_rate, i_t.astype(float), u_rate, u_t.astype(float), stim_ms.astype(float), meta

def _aligned_stim_ms(stim_ms_abs: np.ndarray, meta: dict) -> np.ndarray:
    """
    Convert absolute Intan stim times (ms) into the aligned timebase used in the
    combined file:
      intan_t_ms_aligned = intan_t_ms - t0_intan_ms
    where t0_intan_ms = anchor_sample * 1000 / fs_intan
    """
    if stim_ms_abs.size == 0:
        return stim_ms_abs
    if "anchor_ms" in meta:
        return stim_ms_abs - float(meta["anchor_ms"])
    # fallback if some files only had anchor_sample
    if "anchor_sample" in meta:
        fs_intan = float(meta.get("fs_intan", 30000.0))
        return stim_ms_abs - (float(meta["anchor_sample"]) * 1000.0 / fs_intan)
    return stim_ms_abs  # nothing to do


def _stacked_heatmaps_two_t(intan_avg, ua_avg, t_intan, t_ua, out_png, title_top, title_bot,
                            cmap="jet", vmin=None, vmax=None):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False, constrained_layout=True)

    im0 = axes[0].imshow(intan_avg, aspect="auto", cmap=cmap, origin="lower",
                         extent=[t_intan[0], t_intan[-1], 0, intan_avg.shape[0]],
                         vmin=vmin, vmax=vmax)
    axes[0].axvline(0.0, color="k", alpha=0.8, linewidth=1.2)
    axes[0].axvspan(0.0, 100.0, color="gray", alpha=0.2)
    axes[0].set_title(title_top); axes[0].set_ylabel("Intan ch")
    fig.colorbar(im0, ax=axes[0]).set_label("Δ FR (Hz)")

    im1 = axes[1].imshow(ua_avg, aspect="auto", cmap=cmap, origin="lower",
                         extent=[t_ua[0], t_ua[-1], 0, ua_avg.shape[0]],
                         vmin=vmin, vmax=vmax)
    axes[1].axvline(0.0, color="k", alpha=0.8, linewidth=1.2)
    axes[1].axvspan(0.0, 100.0, color="gray", alpha=0.2)
    axes[1].set_title(title_bot); axes[1].set_xlabel("Time (ms) rel. stim"); axes[1].set_ylabel("UA ch")
    fig.colorbar(im1, ax=axes[1]).set_label("Δ FR (Hz)")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close(fig)


def _ua_title_from_meta(meta: dict) -> str:
    """
    Build Utah array title from metadata.
    """
    if "br_idx" in meta and meta["br_idx"] is not None:
        return f"NRR_RW_001_{int(meta['br_idx']):03d}"
    return "Utah/BR"

if __name__ == "__main__":
    # ---------- CONFIG ----------
    REPO_ROOT = Path(__file__).resolve().parents[1]
    PARAMS    = load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
    OUT_BASE  = resolve_output_root(PARAMS)

    ALIGNED_ROOT = OUT_BASE / "checkpoints" / "Aligned"
    FIG_ROOT     = OUT_BASE / "figures" / "peri_stim" / "Aligned"
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    # Peri-stim settings — identical to your NPRW run_one_Intan_FR_heatmap defaults
    WIN_MS            = (-800.0, 1200.0)
    BASELINE_FIRST_MS = 100.0
    MIN_TRIALS        = 1

    # Intan probe (optional, only used if you later want a probe panel like NPRW)
    try:
        GEOM_PATH = (
            Path(PARAMS.geom_mat_rel).resolve()
            if getattr(PARAMS, "geom_mat_rel", None) and str(PARAMS.geom_mat_rel).startswith("/")
            else (REPO_ROOT / PARAMS.geom_mat_rel).resolve()
            if getattr(PARAMS, "geom_mat_rel", None)
            else resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
        )
        probe, locs = build_probe_and_locs_from_geom(GEOM_PATH)
    except Exception:
        probe, locs = None, None

    files = sorted(ALIGNED_ROOT.glob("aligned__*.npz"))
    if not files:
        raise SystemExit(f"[error] No combined aligned NPZs found at {ALIGNED_ROOT}")

    for file in files:
        try:
            NPRW_rate, NPRW_t, UA_rate, UA_t, stim_ms_abs, meta = _load_combined_npz(file)
            sess = meta.get("session", file.stem)

            # stim times aligned to Intan aligned timebase
            stim_ms = _aligned_stim_ms(stim_ms_abs, meta)

            # --- NPRW/Intan ---
            NPRW_segments, rel_time_ms_i = extract_peristim_segments(
                rate_hz=NPRW_rate, t_ms=NPRW_t, stim_ms=stim_ms,
                win_ms=WIN_MS, min_trials=MIN_TRIALS
            )
            NPRW_zeroed = baseline_zero_each_trial(
                NPRW_segments, rel_time_ms_i, baseline_first_ms=BASELINE_FIRST_MS
            )
            NPRW_avg = average_across_trials(NPRW_zeroed)

            # ---------- UA: same logic but keep its own rel_time_ms ----------
            UA_segments, ua_rel_time_ms = extract_peristim_segments(
                rate_hz=UA_rate, t_ms=UA_t, stim_ms=stim_ms,
                win_ms=WIN_MS, min_trials=MIN_TRIALS
            )
            UA_zeroed = baseline_zero_each_trial(
                UA_segments, ua_rel_time_ms, baseline_first_ms=BASELINE_FIRST_MS
            )
            UA_avg = average_across_trials(UA_zeroed)

            # ---------- produce stacked heatmap figure ----------
            out_png = FIG_ROOT / f"{sess}__Intan_vs_UA__peri_stim_heatmaps.png"
            title_top = f"{sess} — NPRW/Intan (n={NPRW_segments.shape[0]} trials)"
            title_bot = f"{_ua_title_from_meta(meta)} (n={UA_segments.shape[0]} trials)"

            # sanity checks (optional but helpful)
            assert NPRW_avg.shape[1] == rel_time_ms_i.size, "Intan avg vs time mismatch"
            assert UA_avg.shape[1]   == ua_rel_time_ms.size, "UA avg vs time mismatch"

            _stacked_heatmaps_two_t(
                NPRW_avg, UA_avg, rel_time_ms_i, ua_rel_time_ms,
                out_png, title_top, title_bot, cmap="jet", vmin=-500, vmax=1000
            )

            # ---------- (optional) also save separate NPRW-style heatmap for Intan ----------
            # If you still want the NPRW single-panel with probe, reuse your helper:
            # # NOTE: identical logic; here we just repurpose plot_channel_heatmap directly.
            # try:
            #     single_out = FIG_ROOT / f"{sess}__Intan_only__peri_stim_heatmap.png"
            #     plot_channel_heatmap(
            #         i_avg, rel_time_ms, single_out,
            #         n_trials=i_segments.shape[0],
            #         vmin=-500, vmax=1000,
            #         intan_file=meta.get("intan_idx", None),
            #         locs=locs, probe=probe, probe_ratio=1.2,
            #         stim_idx=None,  # or pass your detected sites if desired
            #         padding=15.0,
            #     )
            # except Exception as e:
            #     print(f"[warn] Intan single-panel plot skipped: {e}")

        except Exception as e:
            print(f"[error] Failed on {file.name}: {e}")
            continue
