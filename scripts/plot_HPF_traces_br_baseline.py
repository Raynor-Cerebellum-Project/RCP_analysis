from __future__ import annotations
from pathlib import Path
import csv, re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import spikeinterface as si
import RCP_analysis as rcp
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'

# ---- CONFIG ----
BR_IDX = 2
# TRIAL_INDICES = [0, 3, 4, 7, 8, 11, 13, 15, 16]  # one folder per trial
TRIAL_INDICES = [0, 1, 2, 3, 4, 5, 6]  # one folder per trial
ADJUST_SAMPLES = 3
WINDOW_MS = (3.0, 6.0)
CHANNELS_TO_SHOW = list(range(0, 128))           # will be chunked into groups of 6
IR_STREAM = "USB board digital input channel"
YLIM_UV = (-50, 50)                               # tighten or set to None for autoscale

# ---- Resolving paths ----
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
OUT_BASE  = rcp.resolve_output_root(PARAMS); OUT_BASE.mkdir(parents=True, exist_ok=True)
DATA_ROOT = rcp.resolve_data_root(PARAMS)
INTAN_ROOT = rcp.resolve_intan_root(PARAMS)

SESSION = getattr(PARAMS, "session", None)
# --- metadata + shifts CSV (for Intan session & anchor_ms) ---
metadata_rel = getattr(PARAMS, "metadata_rel", None) or ""
METADATA_CSV = (DATA_ROOT / metadata_rel).resolve()
SHIFTS_CSV   = METADATA_CSV.parent / "br_to_intan_shifts.csv"

ALIGNED_ROOT = OUT_BASE / "checkpoints" / "Aligned"
PATH_UA_SI   = OUT_BASE / "checkpoints" / "UA" / f"pp_global__{SESSION}_{BR_IDX:03d}__NS6"
# Build short tags
amp_tag = ("auto" if YLIM_UV is None else (f"pm_{abs(YLIM_UV[1]):g}uV" if YLIM_UV[0] == -YLIM_UV[1] else f"{YLIM_UV[0]:g}to{YLIM_UV[1]:g}uV"))
win_tag = f"{WINDOW_MS[0]:g}to{WINDOW_MS[1]:g}ms"

# Path with adaptive tag
OUT_DIR_BASE = (OUT_BASE / "figures" / "debug_8ch_aligned_ir_baseline_hpf" / "UA" / f"{SESSION}__BR_{BR_IDX:03d}" / f"{amp_tag}_{win_tag}")
OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

# ---- Helpers ----
def _find_aligned_file(aligned_dir: Path, br_idx: int) -> Path | None:
    pat = f"*__BR_{br_idx:03d}.npz"
    cands = sorted(aligned_dir.glob(pat))
    return cands[-1] if cands else None

def _safe_peaks_UA(z):
    peak_ch = None; t_ms = None
    if "ua_peaks_t_ms_aligned" in z and z["ua_peaks_t_ms_aligned"].size:
        t_ms = z["ua_peaks_t_ms_aligned"].astype(float)
    if "ua_peaks" in z and z["ua_peaks"].size:
        P = z["ua_peaks"]
        if getattr(P, "dtype", None) is not None and P.dtype.names:
            if "channel_index" in P.dtype.names: peak_ch = P["channel_index"].astype(int)
            elif "ch" in P.dtype.names:          peak_ch = P["ch"].astype(int)
        else:
            peak_ch = np.asarray(P).astype(int)
    if peak_ch is None and "peak_ch" in z: peak_ch = z["peak_ch"].astype(int)
    if t_ms is None and "ua_peaks_t_ms" in z and z["ua_peaks_t_ms"].size:
        t_ms = z["ua_peaks_t_ms"].astype(float)
    return peak_ch, t_ms

def _overlay_peaks(ax, t_ms, y, s0_ms, peak_ch, peak_t_ms, ch_pos, adjust_ms=0.0):
    if peak_ch is None or peak_t_ms is None: return
    mask = (peak_ch == int(ch_pos))
    if not np.any(mask): return
    tt = peak_t_ms[mask] + float(adjust_ms)
    in_win = (tt >= s0_ms + t_ms[0]) & (tt <= s0_ms + t_ms[-1])
    if not np.any(in_win): return
    x = tt[in_win] - s0_ms
    dt = np.median(np.diff(t_ms)) if len(t_ms) > 1 else 1.0
    idx = np.clip(np.round((x - t_ms[0]) / dt).astype(int), 0, len(t_ms)-1)
    ax.scatter(x, y[idx], s=12, c="red", marker="x", alpha=0.9, zorder=3)

def _valid_centers_ms(centers_ms: np.ndarray, fs_ua: float, rec_len_samples: int, win_ms) -> np.ndarray:
    if centers_ms is None or centers_ms.size == 0: return np.array([], dtype=float)
    w0, w1 = float(win_ms[0]), float(win_ms[1])
    s = np.round(centers_ms/1000.0 * fs_ua).astype(np.int64)
    i0 = s + np.round((w0/1000.0)*fs_ua).astype(int)
    i1 = s + np.round((w1/1000.0)*fs_ua).astype(int)
    ok = (i0 >= 0) & (i1 <= rec_len_samples) & (i1 > i0)
    return centers_ms[ok]

def _extract_trials(rec, ch_pos, centers_ms, win_ms, fs):
    w0, w1 = win_ms
    t_list, y_list = [], []
    ch_id = rec.get_channel_ids()[ch_pos]
    for s0_ms in centers_ms:
        s0 = int(round((s0_ms / 1000.0) * fs))
        i0 = int(s0 + round((w0/1000.0)*fs))
        i1 = int(s0 + round((w1/1000.0)*fs))
        if i0 < 0 or i1 > rec.get_num_frames() or i1 <= i0:
            continue
        y = rec.get_traces(start_frame=i0, end_frame=i1, channel_ids=[ch_id], return_in_uV=True).squeeze()
        t = (np.arange(i0, i1) - s0) / fs * 1000.0
        t_list.append(t); y_list.append(y)
    return t_list, y_list

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# --- IR helpers ---
def _extract_signal_fs(obj):
    sig = None; fs = None
    if hasattr(obj, "get_traces") and hasattr(obj, "get_sampling_frequency"):
        try: fs = float(obj.get_sampling_frequency())
        except Exception: fs = None
        try: sig = np.asarray(obj.get_traces(), dtype=float)
        except Exception: sig = None
    if sig is None and isinstance(obj, (tuple, list)) and len(obj) >= 1:
        try:
            sig = np.asarray(obj[0], dtype=float)
            if len(obj) >= 2: fs = float(obj[1])
        except Exception: sig = None
    if sig is None and hasattr(obj, "__array__"):
        try: sig = np.asarray(obj, dtype=float)
        except Exception: sig = None
    if sig is None: return None, None
    if sig.ndim == 1: sig = sig[:, None]
    elif sig.ndim == 2 and sig.shape[0] < sig.shape[1] and sig.shape[0] <= 16: sig = sig.T
    else: sig = sig.reshape(sig.shape[0], -1)
    return sig.astype(np.float32, copy=False), fs

def _choose_ch_with_signal(sig: np.ndarray) -> int:
    if sig.ndim != 2 or sig.shape[1] == 0: return 0
    stds = [np.nanstd(sig[:, j]) for j in range(sig.shape[1])]
    return int(np.nanargmax(stds)) if len(stds) else 0

def _falling_edges_from_analog(x: np.ndarray, fs: float | None, refractory_sec: float = 0.0005) -> np.ndarray:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0: return np.array([], dtype=np.int64)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    thr = 0.5 * (lo + hi)
    b = (x > thr).astype(np.int8)
    db = np.diff(b, prepend=b[0])
    edges = np.flatnonzero(db == -1)
    if edges.size == 0: return edges.astype(np.int64)
    refr = max(1, int(round((refractory_sec if fs and fs > 0 else 0.0) * (fs or 1.0))))
    keep = []
    last = -10**12
    for i in edges:
        if i - last >= refr:
            keep.append(i); last = i
    return np.asarray(keep, dtype=np.int64)

def _load_br_to_intan_map_full(shifts_csv: Path) -> dict[int, dict]:
    if not shifts_csv.exists():
        raise SystemExit(f"[error] shifts CSV not found: {shifts_csv}")
    with shifts_csv.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            raise SystemExit(f"[error] {shifts_csv.name} has no header")
        cols = { re.sub(r'[^a-z0-9]+','', c.lower()): c for c in rdr.fieldnames }
        def col(*cands):
            for cand in cands:
                k = re.sub(r'[^a-z0-9]+','', cand.lower())
                if k in cols: return cols[k]
            return None
        c_br   = col("br_idx","br","brfile")
        c_sess = col("session","intan_session","intan")
        c_anchor = col("anchor_ms","anchoroffsetms","offset_ms","br2intan_anchor_ms")
        out: dict[int, dict] = {}
        for row in rdr:
            try: br = int(float(str(row[c_br]).strip()))
            except Exception: continue
            sess = str(row.get(c_sess, "")).strip()
            if not sess: continue
            try: anchor_ms = float(str(row.get(c_anchor, "0")).strip()) if c_anchor else 0.0
            except Exception: anchor_ms = 0.0
            out[br] = {"session": sess, "anchor_ms": anchor_ms}
    return out

# ---- Main ----
def main():
    # UA recording (for raw traces)
    rec = si.load(str(PATH_UA_SI))
    fs_ua = float(rec.get_sampling_frequency())
    rec_len = rec.get_num_frames()
    adjust_ms = (ADJUST_SAMPLES / fs_ua) * 1000.0

    # IR events from Intan (align by IR onset)
    shifts_full = _load_br_to_intan_map_full(SHIFTS_CSV)
    sess_entry = shifts_full.get(BR_IDX)
    if not sess_entry or not sess_entry.get("session"):
        raise SystemExit(f"[error] No Intan session mapping for BR {BR_IDX} in {SHIFTS_CSV}")
    intan_session = sess_entry["session"]
    anchor_ms = float(sess_entry.get("anchor_ms", 0.0))

    try:
        rec_ir = rcp.read_intan_recording(INTAN_ROOT / intan_session, stream_name=IR_STREAM)
    except Exception as e:
        raise SystemExit(f"[error] read_intan_recording failed for Intan session {intan_session}: {e}")

    sig, fs_ir = _extract_signal_fs(rec_ir)
    if sig is None or sig.size == 0:
        raise SystemExit(f"[error] No IR signal for Intan session {intan_session}.")

    ir_ch = _choose_ch_with_signal(sig)
    evt_idx = _falling_edges_from_analog(sig[:, ir_ch], fs_ir, refractory_sec=0.0005)
    if evt_idx.size == 0:
        raise SystemExit(f"[error] {intan_session}: no IR falling edges detected on ch {ir_ch}")

    evt_sec = evt_idx / float(fs_ir if fs_ir and fs_ir > 0 else 1.0)
    centers_ms_raw = (evt_sec * 1000.0) - anchor_ms
    centers_ms_all = _valid_centers_ms(centers_ms_raw, fs_ua, rec_len, WINDOW_MS)

    # Optional: peaks from aligned NPZ
    peak_ch = peak_t_ms = None
    aligned_npz = _find_aligned_file(ALIGNED_ROOT, BR_IDX)
    if aligned_npz is not None:
        z = np.load(aligned_npz, allow_pickle=True)
        peak_ch, peak_t_ms = _safe_peaks_UA(z)

    CHANNEL_ROWS = CHANNELS_TO_SHOW

    print(f"[info] {SESSION} / BR {BR_IDX:03d} / Intan session={intan_session} / anchor_ms={anchor_ms:g}")
    print(f"[info] UA fs={fs_ua:.2f} Hz, frames={rec_len}")
    print(f"[info] IR events detected: {evt_idx.size}, valid windows in {WINDOW_MS} ms: {centers_ms_all.size}")

    if centers_ms_all.size == 0:
        print("[warn] No valid trials after window check. Nothing to plot.")
        return

    # ---- Iterate over requested TRIAL_INDICES; each in its own folder ----
    for trial_idx in TRIAL_INDICES:
        out_dir = OUT_DIR_BASE / f"trial_{trial_idx:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if trial_idx < 0 or trial_idx >= centers_ms_all.size:
            print(f"[skip] trial index {trial_idx} out of range (N={centers_ms_all.size}).")
            continue

        # Stacked view: 6 channels per figure (6 rows x 1 column)
        fig_idx = 0
        for ch_group in _chunks(CHANNEL_ROWS, 6):
            fig_idx += 1
            fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)  # taller for detail
            # If only one axis returned (in case len(ch_group)<2), wrap in list
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
            axes = np.asarray(axes).ravel()

            for ax, ch in zip(axes, ch_group):
                t_list, y_list = _extract_trials(rec, ch, centers_ms_all, WINDOW_MS, fs_ua)
                if len(t_list) == 0 or trial_idx >= len(t_list):
                    ax.axis("off")
                    continue

                t, y = t_list[trial_idx], y_list[trial_idx]
                ax.plot(t, y, lw=1.1)
                ax.axvline(0.0, ls="--", lw=0.9)
                ax.grid(True, alpha=0.3, linestyle=":")
                s0_ms = centers_ms_all[trial_idx]
                _overlay_peaks(ax, t, y, s0_ms, peak_ch, peak_t_ms, ch_pos=ch, adjust_ms=adjust_ms)
                ax.set_ylabel(f"ch {ch}\nµV", rotation=0, labelpad=25, va="center")
                ax.set_xlim(WINDOW_MS[0], WINDOW_MS[1])
                if YLIM_UV is not None:
                    ax.set_ylim(*YLIM_UV)

            # turn off any leftover axes if channels < multiple of 6
            for k in range(len(ch_group), len(axes)):
                axes[k].axis("off")

            axes[-1].set_xlabel("Time (ms) rel. IR onset")

            fig.suptitle(
                f"{SESSION} / BR {BR_IDX:03d} / {PATH_UA_SI.name} / IR-aligned {int(WINDOW_MS[0])}–{int(WINDOW_MS[1])} ms / trial {trial_idx} / group {fig_idx}",
                y=0.995
            )

            fig.tight_layout(rect=[0, 0.02, 1, 0.97])
            out_png = out_dir / (
                f"{SESSION}__BR_{BR_IDX:03d}__trial_{trial_idx:03d}"
                f"__UA_rows_{ch_group[0]:03d}-{ch_group[-1]:03d}__IR__win_{int(WINDOW_MS[0])}-{int(WINDOW_MS[1])}ms.png"
            )
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {out_png}")

    print("[done] IR-aligned stacked figures per trial →", OUT_DIR_BASE)

if __name__ == "__main__":
    main()
