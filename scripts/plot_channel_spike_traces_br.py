from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import spikeinterface as si
import RCP_analysis as rcp
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'

# ---- CONFIG ----
BR_IDX = 4  # choose which BR file you want

# Optional alignment tweak
ADJUST_SAMPLES = 3  # TODO: remove after triangle alignment

WINDOW_MS = (100.0, 500.0) # (-100.0, 250.0) (100.0, 200.0) (160.0, 180.0)
CHANNELS_TO_SHOW = list(range(96, 128))
N_TRIALS_TO_SHOW = 4

# ---- Resolving paths ----
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
OUT_BASE  = rcp.resolve_output_root(PARAMS)
OUT_BASE.mkdir(parents=True, exist_ok=True)

ALIGNED_ROOT = OUT_BASE / "checkpoints" / "Aligned"
PATH_UA_SI = OUT_BASE / "checkpoints" / "UA" / f"pp_global__NRR_RW_001_{BR_IDX:03d}__NS6"
OUT_DIR = OUT_BASE / "figures" / "debug_quads_aligned" / "UA" / f"BR_{BR_IDX:03d}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Helpers ----
def _find_aligned_file(aligned_dir: Path, br_idx: int) -> Path:
    pat = f"*__BR_{br_idx:03d}.npz"
    cands = sorted(aligned_dir.glob(pat))
    if not cands:
        raise FileNotFoundError(f"No aligned npz found in {aligned_dir} matching {pat}")
    if len(cands) > 1:
        print(f"[warn] multiple matches found, using {cands[-1].name}")
    return cands[-1]

def _safe_peaks_UA(z):
    """
    Returns (peak_ch, peak_t_ms_aligned) where:
      - peak_ch: 1D int array (channel indices)
      - peak_t_ms_aligned: 1D float array (times in ms, in the same time base as UA)
    Prefers 'ua_peaks_t_ms_aligned' if present.
    """
    peak_ch = None
    t_ms = None

    # Prefer explicitly aligned ms if present 
    if "ua_peaks_t_ms_aligned" in z and z["ua_peaks_t_ms_aligned"].size:
        t_ms = z["ua_peaks_t_ms_aligned"].astype(float)

    # Try to locate channel indices
    if "ua_peaks" in z and z["ua_peaks"].size:
        P = z["ua_peaks"]
        # If structured
        if getattr(P, "dtype", None) is not None and P.dtype.names:
            if "channel_index" in P.dtype.names:
                peak_ch = P["channel_index"].astype(int)
            elif "ch" in P.dtype.names:
                peak_ch = P["ch"].astype(int)
        else:
            # assume it's already the channel indices
            peak_ch = np.asarray(P).astype(int)

    # Fallbacks:
    if peak_ch is None and "peak_ch" in z:
        peak_ch = z["peak_ch"].astype(int)
    if t_ms is None and "ua_peaks_t_ms" in z and z["ua_peaks_t_ms"].size:
        t_ms = z["ua_peaks_t_ms"].astype(float)

    return peak_ch, t_ms

def _overlay_peaks(ax, t_ms, y, s0_ms, peak_ch, peak_t_ms, ch_pos, adjust_ms=0.0):
    if peak_ch is None or peak_t_ms is None: 
        return
    mask = (peak_ch == int(ch_pos))
    if not np.any(mask): 
        return
    tt = peak_t_ms[mask] + float(adjust_ms)
    in_win = (tt >= s0_ms + t_ms[0]) & (tt <= s0_ms + t_ms[-1])
    if not np.any(in_win): 
        return
    x = tt[in_win] - s0_ms
    dt = np.median(np.diff(t_ms)) if len(t_ms) > 1 else 1.0
    idx = np.clip(np.round((x - t_ms[0]) / dt).astype(int), 0, len(t_ms)-1)
    ax.scatter(x, y[idx], s=12, c="red", marker="x", alpha=0.9, zorder=3)

def _valid_centers_ms(stim_ms_aligned: np.ndarray, fs_ua: float, rec_len_samples: int, win_ms) -> np.ndarray:
    """Keep only centers whose [w0, w1] window fits inside the UA recording."""
    if stim_ms_aligned is None or stim_ms_aligned.size == 0:
        return np.array([], dtype=float)
    w0, w1 = float(win_ms[0]), float(win_ms[1])
    w0s, w1s = w0/1000.0, w1/1000.0
    s = np.round(stim_ms_aligned/1000.0 * fs_ua).astype(np.int64)
    i0 = s + np.round(w0s*fs_ua).astype(int)
    i1 = s + np.round(w1s*fs_ua).astype(int)
    ok = (i0 >= 0) & (i1 <= rec_len_samples) & (i1 > i0)
    return stim_ms_aligned[ok]

def _extract_trials(rec, ch_pos, centers_ms, win_ms, fs, n_show):
    w0, w1 = win_ms
    t_list, y_list = [], []
    ch_id = rec.get_channel_ids()[ch_pos]
    kept = 0
    for s0_ms in centers_ms:
        s0 = int(round((s0_ms / 1000.0) * fs))
        i0 = int(s0 + round((w0/1000.0)*fs))
        i1 = int(s0 + round((w1/1000.0)*fs))
        if i0 < 0 or i1 > rec.get_num_frames() or i1 <= i0:
            continue
        y = rec.get_traces(start_frame=i0, end_frame=i1, channel_ids=[ch_id], return_in_uV=True).squeeze()
        t = (np.arange(i0, i1) - s0) / fs * 1000.0
        t_list.append(t); y_list.append(y)
        kept += 1
        if kept >= n_show:
            break
    return t_list, y_list, kept

# ---- Main ----
def main():
    # SpikeInterface recording (for raw traces)
    rec = si.load(str(PATH_UA_SI))
    fs_ua = float(rec.get_sampling_frequency())
    rec_len = rec.get_num_frames()
    ua_duration_ms = rec_len / fs_ua * 1000.0
    adjust_ms = (ADJUST_SAMPLES / fs_ua) * 1000.0

    # Find and load aligned npz by suffix (for the same BR index)
    PATH_ALIGNED_NPZ = _find_aligned_file(ALIGNED_ROOT, BR_IDX)
    z = np.load(PATH_ALIGNED_NPZ, allow_pickle=True)


    # --- pick whether CHANNELS_TO_SHOW are 'row' indices or 'elec' IDs (1-based) ---
    UA_CHANNEL_MODE = "row"  # change to "elec" if CHANNELS_TO_SHOW are electrode IDs

    # Try to load UA row-index mapping from the aligned file (or fall back to identity)
    ua_row_index_from_electrode = None
    if "ua_row_index_from_electrode" in z.files:
        ua_row_index_from_electrode = z["ua_row_index_from_electrode"].astype(int)  # shape (max_elec,) 1-based→row
    elif "ua_row_to_elec" in z.files:
        # invert ua_row_to_elec (1-based elec id per row) -> build row index from electrode
        ua_row_to_elec = z["ua_row_to_elec"].astype(int).ravel()
        max_e = int(ua_row_to_elec.max()) if ua_row_to_elec.size else 0
        inv = -np.ones(max(256, max_e)+1, dtype=int)  # 1..256
        for r, e in enumerate(ua_row_to_elec):
            if e > 0 and e < inv.size:
                inv[e] = r
        ua_row_index_from_electrode = inv  # 1-based elec -> row index (or -1 if missing)

    # Translate CHANNELS_TO_SHOW -> positional rows for SpikeInterface
    if UA_CHANNEL_MODE == "elec":
        if ua_row_index_from_electrode is None:
            raise RuntimeError("Need UA electrode→row mapping (ua_row_index_from_electrode or ua_row_to_elec).")
        CHANNEL_ROWS = []
        for e in CHANNELS_TO_SHOW:
            r = int(ua_row_index_from_electrode[e]) if e < len(ua_row_index_from_electrode) else -1
            if r < 0:
                print(f"[warn] electrode {e} has no mapped row; skipping.")
            else:
                CHANNEL_ROWS.append(r)
        CHANNEL_ROWS = list(dict.fromkeys(CHANNEL_ROWS))  # dedupe, keep order
    else:
        CHANNEL_ROWS = CHANNELS_TO_SHOW  # already positional row indices

    print(f"[info] plotting UA rows: {CHANNEL_ROWS}")

    # Stim & anchor
    stim_ms_abs = z.get("stim_ms", np.array([], dtype=float))
    if stim_ms_abs is None:
        stim_ms_abs = np.array([], dtype=float)
    try:
        align_meta = json.loads(z["align_meta"].item()) if hasattr(z["align_meta"], "item") else json.loads(z["align_meta"])
    except Exception:
        align_meta = {}
    anchor_ms = float(align_meta.get("anchor_ms", 0.0))

    # Stim centers aligned to UA time base (your convention)
    stim_ms_aligned_raw = stim_ms_abs.astype(float) - anchor_ms

    # Filter to centers whose window fits entirely inside UA
    valid_centers = _valid_centers_ms(stim_ms_aligned_raw, fs_ua, rec_len, WINDOW_MS)

    # Peaks
    peak_ch, peak_t_ms = _safe_peaks_UA(z)

    # ---- Diagnostics ----
    print(f"[info] UA fs = {fs_ua:.2f} Hz, frames = {rec_len}, duration ≈ {ua_duration_ms/1000.0:.2f} s")
    print(f"[info] stim centers (raw-aligned) count = {stim_ms_aligned_raw.size}")
    if stim_ms_aligned_raw.size:
        print(f"       stim_ms_aligned range: {stim_ms_aligned_raw.min():.1f} .. {stim_ms_aligned_raw.max():.1f} ms")
    print(f"[info] valid centers for window {WINDOW_MS} ms: {valid_centers.size}")

    if valid_centers.size == 0:
        print("[warn] No valid trials after window check. Nothing to plot.")
        return

    # Plot
    for ch in CHANNELS_TO_SHOW:
        t_list, y_list, kept = _extract_trials(rec, ch, valid_centers, WINDOW_MS, fs_ua, N_TRIALS_TO_SHOW)
        if kept == 0:
            print(f"[warn] ch {ch}: 0 trials fit window (after filtering). Skipping plot.")
            continue

        fig, axes = plt.subplots(2, 2, figsize=(10,6), sharex=True, sharey=True)
        axes = axes.ravel()
        for k, ax in enumerate(axes):
            if k < len(t_list):
                t, y = t_list[k], y_list[k]
                ax.plot(t, y, lw=0.9)
                ax.axvline(0.0, ls="--", lw=0.8)
                s0_ms = valid_centers[k] if k < valid_centers.size else valid_centers[-1]
                _overlay_peaks(ax, t, y, s0_ms, peak_ch, peak_t_ms, ch_pos=ch, adjust_ms=adjust_ms)
                ax.set_title(f"Trial {k+1}", fontsize=9)
                
                # --- enforce xlim if requested ---
                if WINDOW_MS[0] > 0:
                    ax.set_xlim(WINDOW_MS[0], WINDOW_MS[1])
            else:
                ax.axis("off")
        for ax in axes[-2:]:
            ax.set_xlabel("Time (ms) rel. stim")
        for ax in axes:
            ax.set_ylim(-100, 100)

        axes[0].set_ylabel("µV"); axes[2].set_ylabel("µV")
        fig.suptitle(f"{PATH_UA_SI.name} • UA ch {ch} • {int(WINDOW_MS[0])}–{int(WINDOW_MS[1])} ms", y=0.98)
        out_svg = OUT_DIR / f"UA_ch{ch:03d}__quad.svg"
        fig.tight_layout(); fig.savefig(out_svg, dpi=300, bbox_inches="tight"); plt.close(fig)
        print(f"[saved] {out_svg}")

    print("[done] all quads saved →", OUT_DIR)

if __name__ == "__main__":
    main()
