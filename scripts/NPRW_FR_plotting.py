#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from RCP_analysis import load_experiment_params, resolve_output_root, load_stim_detection

def load_rate_npz(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    rate_hz = d["rate_hz"]           # (n_ch, n_bins_total)
    t_ms    = d["t_ms"]              # (n_bins_total,)
    meta    = d.get("meta", None)
    return rate_hz, t_ms, (meta.item() if hasattr(meta, "item") else meta)

def load_stim_ms_from_stimstream(stim_npz_path: Path) -> np.ndarray:
    """
    Load stimulation *block onset times* in ms from stim_stream.npz.
    Uses 'block_bounds_samples[:,0]' (Intan sample indices) and converts
    to milliseconds using meta['fs_hz'].
    """
    with np.load(stim_npz_path, allow_pickle=False) as z:
        if "block_bounds_samples" not in z or "meta" not in z:
            raise KeyError(f"{stim_npz_path} missing 'block_bounds_samples' or 'meta'.")

        blocks = z["block_bounds_samples"].astype(np.int64)  # (n_blocks, 2)
        if blocks.size == 0:
            raise ValueError(f"No block boundaries found in {stim_npz_path}.")

        # Parse meta
        meta_json = z["meta"].item() if hasattr(z["meta"], "item") else z["meta"]
        meta = json.loads(meta_json)
        fs_hz = float(meta.get("fs_hz"))
        if not np.isfinite(fs_hz):
            raise ValueError("meta['fs_hz'] missing or non-finite in stim_stream.npz")

        # Use block starts only
        onset_samples = blocks[:, 0]
        t_ms = onset_samples.astype(float) * (1000.0 / fs_hz)
        return np.unique(np.sort(t_ms))


def extract_peristim_segments(
    rate_hz: np.ndarray,
    t_ms: np.ndarray,
    stim_ms: np.ndarray,
    win_ms=(-800.0, 1200.0),
    min_trials: int = 1,
):
    """
    Return segments shape (n_trials, n_ch, n_twin) and rel_time_ms shape (n_twin,).
    Skips triggers whose window falls outside t_ms.
    """
    t_min, t_max = float(t_ms[0]), float(t_ms[-1])
    seg_len_ms = win_ms[1] - win_ms[0]

    # Relative timebase for a *perfectly* aligned segment (used only for plotting/logic)
    # We will slice on t_ms for each trial, so segment lengths are equal if t_ms is uniform.
    # Assume uniform binning for rates:
    dt = float(np.median(np.diff(t_ms)))  # ms per bin
    n_twin = int(round(seg_len_ms / dt))
    rel_time_ms = np.arange(n_twin) * dt + win_ms[0]

    segments = []
    kept = 0
    for s in np.asarray(stim_ms, dtype=float):
        start_ms = s + win_ms[0]
        end_ms   = s + win_ms[1]
        if start_ms < t_min or end_ms > t_max:
            continue  # skip partial windows

        # slice indices on t_ms
        i0 = int(np.searchsorted(t_ms, start_ms, side="left"))
        i1 = int(np.searchsorted(t_ms, end_ms,   side="left"))
        seg = rate_hz[:, i0:i1]  # (n_ch, n_twin)
        # Safety: ensure equal length (can happen if boundary falls between bins)
        if seg.shape[1] != n_twin:
            continue

        segments.append(seg)
        kept += 1

    if kept < min_trials:
        raise RuntimeError(f"Only {kept} peri-stim segments available (min_trials={min_trials}).")

    segments = np.stack(segments, axis=0)  # (n_trials, n_ch, n_twin)
    return segments, rel_time_ms

def baseline_zero_each_trial(
    segments: np.ndarray,
    rel_time_ms: np.ndarray,
    baseline_first_ms: float = 100.0,
):
    """
    For each trial & channel, subtract the mean over the first `baseline_first_ms`
    of the segment (i.e., from window start to start+baseline_first_ms).
    segments: (n_trials, n_ch, n_twin)
    """
    # mask for first 100 ms of the segment window (e.g., [-800, -700))
    t0 = rel_time_ms[0]
    mask = (rel_time_ms >= t0) & (rel_time_ms < t0 + baseline_first_ms)
    if mask.sum() < 1:
        raise ValueError("Baseline window has 0 bins — check your timebase and dt.")
    base = segments[:, :, mask].mean(axis=2, keepdims=True)  # (n_trials, n_ch, 1)
    return segments - base

def average_across_trials(zeroed_segments: np.ndarray):
    """
    Average across trials per channel.
    Input: (n_trials, n_ch, n_twin) -> returns (n_ch, n_twin)
    """
    return zeroed_segments.mean(axis=0)


def plot_channel_heatmap(
    avg_change: np.ndarray,           # (n_ch, n_twin)
    rel_time_ms: np.ndarray,          # (n_twin,)
    out_png: Path,
    n_trials: int,
    title: str = "Peri-stim avg Δ rate (baseline=first 100 ms)",
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
):
    plt.figure(figsize=(12, 6))
    plt.imshow(
        avg_change,
        aspect="auto",
        cmap=cmap,
        extent=[rel_time_ms[0], rel_time_ms[-1], avg_change.shape[0], 0],  # <-- ms
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label="Δ Firing rate (Hz)")
    plt.xlabel("Time (ms) rel. stim")   # <-- ms axis label
    plt.ylabel("Channel index")
    plt.title(f"{title} (n={n_trials} trials)")  # <-- include trial count
    plt.axvline(0.0, color="k", alpha=0.8, linewidth=1.2)
    plt.axvspan(0.0, 100.0, color="gray", alpha=0.2, zorder=2)  # shaded 0–100 ms region
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved heatmap -> {out_png}")

def run_one_file(
    npz_path: Path,
    out_dir: Path,
    win_ms=(-800.0, 1200.0),
    baseline_first_ms=100.0,
    min_trials=1,
    save_npz=True,
    stim_ms: np.ndarray | None = None,
    debug_channel: int | None = None,    # <-- NEW
):
    rate_hz, t_ms, _ = load_rate_npz(npz_path)
    stim_ms = np.asarray(stim_ms, dtype=float).ravel()

    segments, rel_time_ms = extract_peristim_segments(
        rate_hz=rate_hz, t_ms=t_ms, stim_ms=stim_ms, win_ms=win_ms, min_trials=min_trials
    )

    zeroed = baseline_zero_each_trial(
        segments=segments, rel_time_ms=rel_time_ms, baseline_first_ms=baseline_first_ms
    )

    # -------- DEBUG: single-channel quick look --------
    if debug_channel is not None:
        # print some shapes/numbers to stdout
        dt = float(np.median(np.diff(t_ms)))
        print(f"[debug] segments shape={segments.shape} (trials, ch, timebins); dt={dt} ms")
        print(f"[debug] rel_time_ms: {rel_time_ms[0]} .. {rel_time_ms[-1]} (n={rel_time_ms.size})")
        out_base = (Path(out_dir) / npz_path.stem)
        plot_debug_channel_traces(zeroed, rel_time_ms, debug_channel, out_base)
    # ---------------------------------------------------

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = npz_path.stem
    avg_change = average_across_trials(zeroed)
    n_trials = segments.shape[0]   # number of kept trials

    out_png = out_dir / f"{stem}__peri_stim_heatmap.png"
    plot_channel_heatmap(avg_change, rel_time_ms, out_png,
                        n_trials=n_trials,
                        vmin=-500, vmax=1000)
    if save_npz:
        out_npz = out_dir / f"{stem}__peri_stim_avg.npz"
        np.savez_compressed(
            out_npz,
            avg_change=avg_change.astype(np.float32),
            rel_time_ms=rel_time_ms.astype(np.float32),
            meta=dict(
                source_file=str(npz_path),
                win_ms=win_ms,
                baseline_first_ms=baseline_first_ms,
                n_trials=int(segments.shape[0]),
            ),
        )
        print(f"Saved averaged peri-stim data -> {out_npz}")

def plot_debug_channel_traces(
    zeroed_segments: np.ndarray,   # (n_trials, n_ch, n_twin)
    rel_time_ms: np.ndarray,       # (n_twin,)
    ch: int,
    out_png_base: Path,
):
    n_trials, n_ch, n_twin = zeroed_segments.shape
    assert 0 <= ch < n_ch, f"debug channel {ch} out of range [0, {n_ch-1}]"

    y = zeroed_segments[:, ch, :]  # (n_trials, n_twin)

    # 1) overlay all trials + mean (x-axis in ms)
    plt.figure(figsize=(10, 5))
    for i in range(n_trials):
        plt.plot(rel_time_ms, y[i], alpha=0.25, linewidth=0.8)
    plt.plot(rel_time_ms, y.mean(axis=0), linewidth=2.0, label="mean")
    plt.axvline(0.0, color="k", alpha=0.6, linewidth=1.0)
    plt.xlabel("Time (ms) rel. stim")
    plt.ylabel("Δ Firing rate (Hz)")
    plt.title(f"Channel {ch}: peri-stim trials (n={n_trials})")
    plt.legend(loc="upper right")
    plt.tight_layout()
    out_png = out_png_base.with_name(out_png_base.stem + f"__ch{ch:03d}_overlay.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[debug] Saved overlay for ch {ch} -> {out_png}")

    # 2) trial-by-time image (x-axis in ms)
    plt.figure(figsize=(10, 5))
    plt.imshow(
        y, aspect="auto", cmap="hot",
        extent=[rel_time_ms[0], rel_time_ms[-1], n_trials, 0],
        origin="upper"
    )
    plt.colorbar(label="Δ Firing rate (Hz)")
    plt.axvline(0.0, color="k", alpha=0.6, linewidth=1.0)
    plt.xlabel("Time (ms) rel. stim")
    plt.ylabel("Trial index")
    plt.title(f"Channel {ch}: trials × time")
    plt.tight_layout()
    out_png2 = out_png_base.with_name(out_png_base.stem + f"__ch{ch:03d}_trials.png")
    plt.savefig(out_png2, dpi=150)
    plt.close()
    print(f"[debug] Saved trials×time for ch {ch} -> {out_png2}")


if __name__ == "__main__":
    # ==============================
    # Config
    # ==============================
    REPO_ROOT = Path(__file__).resolve().parents[1]
    PARAMS = load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
    OUT_BASE = resolve_output_root(PARAMS)
    OUT_BASE.mkdir(parents=True, exist_ok=True)

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
    rate_files = sorted(ckpt_root.rglob("rates__*.npz"))
    if not rate_files:
        raise SystemExit(f"[error] No rates__*.npz found under {ckpt_root}")

    n_ok, n_skip = 0, 0
    print(f"[info] Found {len(rate_files)} rates files. Processing...")

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
            out_dir = figs_root / subproj
            out_dir.mkdir(parents=True, exist_ok=True)

            # For the very first file, keep the single-channel debug overlay; otherwise off
            debug_ch = 0 if i == 1 else None

            print(f"[info] ({i}/{len(rate_files)}) session={session}  trials={stim_ms.size}")
            run_one_file(
                npz_path=rates_npz,
                out_dir=out_dir,
                win_ms=(-800.0, 1200.0),
                baseline_first_ms=100.0,
                min_trials=1,
                save_npz=True,
                stim_ms=stim_ms,
                debug_channel=debug_ch,
            )
            n_ok += 1

        except Exception as e:
            print(f"[error] Failed on {rates_npz}: {e}")
            n_skip += 1

    print(f"[done] processed={n_ok}, skipped={n_skip}, figs at: {figs_root}")

