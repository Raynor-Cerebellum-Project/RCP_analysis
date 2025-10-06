from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt
import spikeinterface as si
import RCP_analysis as rcp
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'

# ---- CONFIG ----
INTAN_FOLDER = "pp_local_30_150__interp_NRR_RW001_250915_144530"

# Optional alignment tweak
ADJUST_SAMPLES = 3  # TODO: remove after triangle alignment

WINDOW_MS = (100.0, 200.0) # (-100.0, 250.0) (100.0, 200.0) (160.0, 180.0)
CHANNELS_TO_SHOW = [1, 2, 3, 4, 5, 6, 7, 8]
N_TRIALS_TO_SHOW = 4

# ---- Resolving paths ----
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
OUT_BASE  = rcp.resolve_output_root(PARAMS)
OUT_BASE.mkdir(parents=True, exist_ok=True)

ALIGNED_DIR = OUT_BASE / "checkpoints" / "Aligned"
PATH_NPRW_SI = OUT_BASE / "checkpoints" / "NPRW/" / INTAN_FOLDER
OUT_DIR = OUT_BASE / "figures" / "debug_quads_aligned" / "NPRW" / INTAN_FOLDER
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _find_aligned_for_session(aligned_dir: Path, session_key: str) -> Path:
    """
    Find aligned file by Intan session key, e.g. 'NRR_RW001_250915_144113'.
    Accept any Intan/BR indices; prefer the newest (sorted).
    """
    pat = f"aligned__{session_key}__Intan_*__BR_*.npz"
    cands = sorted(aligned_dir.glob(pat))
    if not cands:
        raise FileNotFoundError(f"No aligned npz in {aligned_dir} matching {pat}")
    return cands[-1]

def _session_key_from_nprw_path(p: Path) -> str:
    """
    Extract the Intan session key from an NPRW SI folder name like:
      'pp_local_30_150__interp_NRR_RW001_250915_144113'
    Fallback: search for the first token that looks like NRR_*_YYYYMM_HHMMSS.
    """
    name = p.name
    if "__interp_" in name:
        cand = name.split("__interp_", 1)[1]
        # strip any suffixes that aren’t part of the session
        # (e.g., if there is a trailing '__something')
        cand = cand.split("__", 1)[0]
        return cand
    # Fallback: regex
    m = re.search(r"(NRR[_A-Z0-9]*_\d{6}_\d{6})", name)
    if not m:
        raise ValueError(f"Could not parse Intan session key from folder name: {name}")
    return m.group(1)

def _safe_peaks_intan(z, anchor_ms: float, rec) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Returns (peak_pos, peak_t_ms_aligned)

    - peak_pos: int positions 0..n-1 matching rec.get_channel_ids() order
    - peak_t_ms_aligned: times in ms, aligned (stim and traces use this base)
    """
    peak_pos = None
    t_ms = None

    # --- times ---
    if "intan_peaks_t_ms_aligned" in z and z["intan_peaks_t_ms_aligned"].size:
        t_ms = z["intan_peaks_t_ms_aligned"].astype(float) + float(anchor_ms)
    elif "intan_peaks_t_ms" in z and z["intan_peaks_t_ms"].size:
        # align on the fly
        t_ms = z["intan_peaks_t_ms"].astype(float)

    # --- channels ---
    chan_positions = None
    if "intan_peaks" in z and z["intan_peaks"].size:
        P = z["intan_peaks"]
        if getattr(P, "dtype", None) is not None and P.dtype.names:
            names = set(P.dtype.names)
            # Most common: position fields
            if "channel_index" in names:
                chan_positions = P["channel_index"].astype(int)
            elif "ch" in names:
                chan_positions = P["ch"].astype(int)
            else:
                # If we only have channel *IDs*, map them to positions
                id_field = "channel_id" if "channel_id" in names else ("channel" if "channel" in names else None)
                if id_field is not None:
                    ids = np.asarray(P[id_field]).astype(int)
                    rec_ids = np.asarray(rec.get_channel_ids())
                    id2pos = {int(cid): i for i, cid in enumerate(rec_ids)}
                    # map unknowns to -1 and drop later
                    chan_positions = np.array([id2pos.get(int(x), -1) for x in ids], dtype=int)

    # Fallbacks from your older files
    if chan_positions is None and "peak_ch" in z and z["peak_ch"].size:
        chan_positions = np.asarray(z["peak_ch"]).astype(int)

    if chan_positions is None:
        return None, t_ms

    # Drop peaks whose channel_id wasn't found in rec (mapped to -1)
    ok = chan_positions >= 0
    if t_ms is not None and t_ms.shape[0] == ok.shape[0]:
        t_ms = t_ms[ok]
    peak_pos = chan_positions[ok]

    return peak_pos, t_ms

def _overlay_peaks(ax, t_ms, y, s0_ms, peak_ch, peak_t_ms, ch_pos, adjust_ms=0.0):
    if peak_ch is None or peak_t_ms is None:
        return
    mask = (peak_ch == int(ch_pos))
    if not np.any(mask):
        return

    # apply user-specified shift (in ms)
    tt = peak_t_ms[mask] + float(adjust_ms)

    in_win = (tt >= s0_ms + t_ms[0]) & (tt <= s0_ms + t_ms[-1])
    if not np.any(in_win):
        return
    x = tt[in_win] - s0_ms
    dt = np.median(np.diff(t_ms)) if len(t_ms) > 1 else 1.0
    idx = np.clip(np.round((x - t_ms[0]) / dt).astype(int), 0, len(t_ms)-1)
    ax.scatter(x, y[idx], s=12, c="red", marker="x", alpha=0.9, zorder=3)

def _valid_centers_ms(stim_ms_aligned: np.ndarray, fs_intan: float, rec_len_samples: int, win_ms) -> np.ndarray:
    """Keep only centers whose [w0, w1] window fits inside the NPRW recording."""
    if stim_ms_aligned is None or stim_ms_aligned.size == 0:
        return np.array([], dtype=float)
    w0, w1 = float(win_ms[0]), float(win_ms[1])
    w0s, w1s = w0/1000.0, w1/1000.0
    s = np.round(stim_ms_aligned/1000.0 * fs_intan).astype(np.int64)
    i0 = s + np.round(w0s*fs_intan).astype(int)
    i1 = s + np.round(w1s*fs_intan).astype(int)
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
    rec = si.load(str(PATH_NPRW_SI))
    fs_intan = float(rec.get_sampling_frequency())
    rec_len = rec.get_num_frames()
    intan_duration_ms = rec_len / fs_intan * 1000.0
    
    adjust_ms = (ADJUST_SAMPLES / fs_intan) * 1000.0

    # Find and load aligned npz by suffix (for the same BR index)
    session_key = _session_key_from_nprw_path(PATH_NPRW_SI)
    PATH_ALIGNED_NPZ = _find_aligned_for_session(ALIGNED_DIR, session_key)
    z = np.load(PATH_ALIGNED_NPZ, allow_pickle=True)

    # Stim & anchor
    stim_ms_abs = z.get("stim_ms", np.array([], dtype=float))
    if stim_ms_abs is None:
        stim_ms_abs = np.array([], dtype=float)
    try:
        align_meta = json.loads(z["align_meta"].item()) if hasattr(z["align_meta"], "item") else json.loads(z["align_meta"])
    except Exception:
        align_meta = {}
    anchor_ms = float(align_meta.get("anchor_ms", 0.0))

    # Stim centers aligned to NPRW time base (your convention)
    stim_ms_aligned_raw = stim_ms_abs.astype(float)# - anchor_ms

    # Filter to centers whose window fits entirely inside NPRW
    valid_centers = _valid_centers_ms(stim_ms_aligned_raw, fs_intan, rec_len, WINDOW_MS)

    # Peaks
    peak_ch, peak_t_ms = _safe_peaks_intan(z, anchor_ms, rec)

    # ---- Diagnostics ----
    print(f"[info] NPRW fs = {fs_intan:.2f} Hz, frames = {rec_len}, duration ≈ {intan_duration_ms/1000.0:.2f} s")
    print(f"[info] stim centers (raw-aligned) count = {stim_ms_aligned_raw.size}")
    if stim_ms_aligned_raw.size:
        print(f"       stim_ms_aligned range: {stim_ms_aligned_raw.min():.1f} .. {stim_ms_aligned_raw.max():.1f} ms")
    print(f"[info] valid centers for window {WINDOW_MS} ms: {valid_centers.size}")

    if valid_centers.size == 0:
        print("[warn] No valid trials after window check. Nothing to plot.")
        return

    # Plot
    for ch in CHANNELS_TO_SHOW:
        t_list, y_list, kept = _extract_trials(rec, ch, valid_centers, WINDOW_MS, fs_intan, N_TRIALS_TO_SHOW)
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
        axes[0].set_ylabel("µV"); axes[2].set_ylabel("µV")
        fig.suptitle(f"{PATH_NPRW_SI.name} • NPRW ch {ch} • {int(WINDOW_MS[0])}–{int(WINDOW_MS[1])} ms", y=0.98)
        out_svg = OUT_DIR / f"NPRW_ch{ch:03d}__quad.svg"
        fig.tight_layout(); fig.savefig(out_svg, dpi=300, bbox_inches="tight"); plt.close(fig)
        print(f"[saved] {out_svg}")

    print("[done] all quads saved →", OUT_DIR)

if __name__ == "__main__":
    main()
