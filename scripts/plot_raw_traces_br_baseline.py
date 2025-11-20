from __future__ import annotations
from pathlib import Path
import csv, re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import RCP_analysis as rcp
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'

# ---- CONFIG ----
BR_IDX = 3
# TRIAL_INDICES = [0, 3, 4, 7, 8, 11, 13, 15, 16]  # one folder per trial
TRIAL_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # one folder per trial
ADJUST_SAMPLES = 3
WINDOW_MS = (125.0, 325.0)
CHANNELS_TO_SHOW = list(range(0, 128))           # will be chunked into groups of 6
IR_STREAM = "USB board digital input channel"
YLIM_UV = None                               # tighten or set to None for autoscale

# ---- Resolving paths ----
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE  = SESSION_LOC / "results"; OUT_BASE.mkdir(parents=True, exist_ok=True)
INTAN_ROOT = SESSION_LOC / "Intan"; INTAN_ROOT.mkdir(parents=True, exist_ok=True)
BR_ROOT = SESSION_LOC / "Blackrock"; BR_ROOT.mkdir(parents=True, exist_ok=True)

# --- metadata + shifts CSV (for Intan session & anchor_ms) ---
METADATA_ROOT = SESSION_LOC / "Metadata"; METADATA_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_CSV  = METADATA_ROOT / f"{Path(PARAMS.session)}_metadata.csv"
SHIFTS_CSV   = METADATA_ROOT / "br_to_intan_shifts.csv"
PATH_UA_SI   = OUT_BASE / "checkpoints" / "UA" / f"pp_global__{Path(PARAMS.session)}_{BR_IDX:03d}__NS6"
ALIGNED_CKPT_ROOT = OUT_BASE / "checkpoints" / "Aligned"

# Build short tags
amp_tag = ("auto" if YLIM_UV is None else (f"pm_{abs(YLIM_UV[1]):g}uV" if YLIM_UV[0] == -YLIM_UV[1] else f"{YLIM_UV[0]:g}to{YLIM_UV[1]:g}uV"))
win_tag = f"{WINDOW_MS[0]:g}to{WINDOW_MS[1]:g}ms"

# Path with adaptive tag
OUT_DIR_BASE = (OUT_BASE / "figures" / "debug_8ch_aligned_ir_baseline_raw" / "UA" / f"{SESSION}__BR_{BR_IDX:03d}" / f"{amp_tag}_{win_tag}")
OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

# --- add near other imports/config ---
XLS = rcp.ua_excel_path(REPO_ROOT, getattr(PARAMS, "probes", {}))
UA_MAP = rcp.load_UA_mapping_from_excel(XLS) if XLS and Path(XLS).exists() else None
if UA_MAP is None:
    raise SystemExit("[error] UA mapping required to align raw NS6 channel order.")

# ---- Impedance file(s) ----
IMP_BASE = OUT_BASE.parents[0]
IMP_FILES = {
    "A": IMP_BASE / "Impedances" / "Utah_imp_Bank_A_start",
    "B": IMP_BASE / "Impedances" / "Utah_imp_Bank_B_start",
}

def _group_tag_from_elecs(rec, ch_group):
    elecs = []
    for ch in ch_group:
        cid = rec.get_channel_ids()[ch]   # e.g. 'UAe005_NSP001'
        e, _ = parse_elec_nsp_from_id(cid)
        if e is not None:
            elecs.append(e)
    if elecs:
        return f"UAelec_{min(elecs):03d}-{max(elecs):03d}"
    # fallback if nothing parsed
    return f"UAidx_{ch_group[0]:03d}-{ch_group[-1]:03d}"


_id_pat = re.compile(r"UAe(\d{1,3})_NSP(\d{3})", re.IGNORECASE)

def parse_elec_nsp_from_id(x) -> tuple[int|None, int|None]:
    """
    Accepts things like 'UAe005_NSP001' and returns (elec=5, nsp=1).
    Returns (None, None) if not parseable.
    """
    s = str(x)
    m = _id_pat.fullmatch(s)
    if m:
        elec = int(m.group(1))
        nsp  = int(m.group(2))
        return elec, nsp
    # fallbacks: 'UAe005', '005', etc.
    m2 = re.search(r"UAe(\d{1,3})", s, re.IGNORECASE)
    if m2:
        return int(m2.group(1)), None
    m3 = re.search(r"\d{1,3}$", s)
    return (int(m3.group(0)), None) if m3 else (None, None)

# find the BR session folder under the Blackrock tree
def _find_br_session_dir(
    br_root: Path,
    session: str | None,
    br_idx: int,
) -> Path:
    """
    Find the BR session folder under br_root corresponding to BR index br_idx.

    If `session` is provided, prefer folders whose name starts with that
    session string and ends with '_{br_idx:03d}'.
    """
    br_folders = rcp.list_br_sessions(br_root)

    if session:
        # Prefer: session prefix + _NNN
        cands = [
            p for p in br_folders
            if p.name.startswith(session) and re.search(rf"_{br_idx:03d}$", p.name)
        ]
        if not cands:
            # Fallback: any folder ending with _NNN
            cands = [
                p for p in br_folders
                if re.search(rf"_{br_idx:03d}$", p.name)
            ]
    else:
        cands = [
            p for p in br_folders
            if re.search(rf"_{br_idx:03d}$", p.name)
        ]

    if not cands:
        raise SystemExit(f"[error] could not locate BR session folder for BR {br_idx:03d}")

    # Latest / lexicographically last candidate
    return sorted(cands)[-1]


# ---- Helpers ----
# Impedance parsing
_imp_pat_elecnum = re.compile(
    r"\belec\s*\d+\s*-\s*(\d{1,3})\s+([0-9]+(?:\.[0-9]+)?)\s*(k?ohms?|kΩ|ohms?|Ω)\b",
    flags=re.IGNORECASE,
)

def _unit_to_kohm(val: float, unit: str) -> float:
    u = (unit or "").lower()
    if "k" in u:
        return float(val)
    return float(val) / 1000.0  # Ω → kΩ

def _read_text_loose(p: Path) -> str:
    if not p.exists():
        raise FileNotFoundError(str(p))
    b = p.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1", "mac_roman"):
        try:
            return b.decode(enc)
        except Exception:
            pass
    # keep printable bytes
    filtered = bytes(ch for ch in b if 9 <= ch <= 126 or ch in (10, 13))
    return filtered.decode("latin-1", errors="ignore")

def load_impedances_from_textedit_dump(path_like: str | Path) -> dict[int, float]:
    """
    Parse the 'AutoImpedance' dump you pasted (lines like 'elec1-5   201 kOhm').
    Returns { Elec#: impedance_kΩ }.
    """
    txt = _read_text_loose(Path(path_like))
    out: dict[int, float] = {}
    for m in _imp_pat_elecnum.finditer(txt):
        elec_str, val_str, unit = m.groups()
        try:
            elec = int(elec_str)
            val = float(val_str)
        except Exception:
            continue
        out[elec] = _unit_to_kohm(val, unit)
    return out

def _fmt_impedance_kohm(z: float | None) -> str:
    if z is None or not np.isfinite(z):
        return "—"
    if z >= 1000:
        return f"{z:.0f} kΩ (open)"
    return f"{z:.0f} kΩ" if z >= 100 else f"{z:.1f} kΩ"

def _imp_color(z: float | None) -> str:
    """
    Colors to match quick-view:
      - >= 1000 kΩ   → red
      - 500–<1000 kΩ → tab:orange
      - < 500 kΩ     → tab:blue
      - unknown/NaN  → gray
    """
    if z is None or not np.isfinite(z):
        return "gray"
    if z >= 1000:
        return "red"
    if z >= 500:
        return "tab:orange"
    return "tab:blue"

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
    # UA recording (RAW NS6 traces, not CMR)
    sess_dir = _find_br_session_dir(BR_ROOT, SESSION_LOC, BR_IDX)
    rec_raw = se.read_blackrock(sess_dir, stream_name = 'nsx6', all_annotations=True)

    # Keep channel indexing consistent with UA mapping (no properties, just IDs/ordering)
    # Returns (recording_with_renamed_ids, idx_rows) → we only need the recording here
    rec, _ = rcp.apply_ua_mapping_by_renaming(rec_raw, UA_MAP["mapped_nsp"], BR_IDX, METADATA_CSV)
    # Reorder to true Elec# order (UAe###), not NSP order
    cids = list(rec.get_channel_ids())  # e.g., 'UAe005_NSP001'
    elec_sorted_idx = []
    unparsed_idx = []

    for i, cid in enumerate(cids):
        elec_id, _ = parse_elec_nsp_from_id(cid)
        if elec_id is not None:
            elec_sorted_idx.append((i, elec_id))
        else:
            unparsed_idx.append(i)

    # sort by Elec# ascending; append any unparsed at the end to avoid dropping channels
    elec_sorted_idx.sort(key=lambda t: t[1])
    CHANNELS_TO_SHOW = [i for i, _ in elec_sorted_idx] + unparsed_idx

    # Determine UA port for this BR block (A/B) if available in the UA mapping/meta
    ua_port = None
    for k in ("br_to_port", "br_port", "br2port", "port_by_br"):
        if UA_MAP and k in UA_MAP and isinstance(UA_MAP[k], dict):
            ua_port = UA_MAP[k].get(int(BR_IDX)) or UA_MAP[k].get(str(BR_IDX))
            if ua_port:
                ua_port = str(ua_port).strip().upper()
                break

    # Fallback: if not present, try meta on the aligned NPZ later (optional), else default to 'A'
    if not ua_port:
        ua_port = "A"

    # --- Load impedances for the detected port (or both, if desired) ---
    imp_by_elec = {}

    def _try_load_imp(p: Path, tag: str):
        if not p:
            return
        if not p.exists():
            print(f"[warn] Impedance file not found for port {tag}: {p}")
            return
        try:
            d = load_impedances_from_textedit_dump(p)
            imp_by_elec.update(d)  # merge
            print(f"[info] Parsed {len(d)} impedances from {p.name} (port {tag}).")
        except Exception as e:
            print(f"[warn] could not parse impedances from {p} (port {tag}): {e}")

    # Prefer the port actually used in this BR block
    _imp_path = IMP_FILES.get(ua_port)
    if _imp_path is not None:
        _try_load_imp(_imp_path, ua_port)
    else:
        print(f"[warn] No impedance path configured for ua_port={ua_port!r}.")

    # Optional: also load the *other* port if you want everything available.
    # Comment these two lines out if you strictly want one port.
    other_port = "B" if ua_port == "A" else "A"
    if IMP_FILES.get(other_port):
        _try_load_imp(IMP_FILES[other_port], other_port)


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
        rec_ir = se.read_split_intan_files(INTAN_ROOT / intan_session, mode="concatenate", stream_name=IR_STREAM, use_names_as_ids=True)
        rec_ir = spre.unsigned_to_signed(rec_ir) # Convert UInt16 to int16
    except Exception as e:
        raise SystemExit(f"[warn] Reading IR stream failed: Intan={intan_session}: {e}")

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
    aligned_npz = _find_aligned_file(ALIGNED_CKPT_ROOT, BR_IDX)
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
                cid = rec.get_channel_ids()[ch]        # e.g., 'UAe005_NSP001'
                elec_id, nsp_ch = parse_elec_nsp_from_id(cid)
                imp = imp_by_elec.get(elec_id) if elec_id is not None else None
                col = _imp_color(imp)

                # plot trace with impedance-based color
                ax.plot(t, y, lw=1.1, color=col)
                ax.axvline(0.0, ls="--", lw=0.9, color="k")
                ax.grid(True, alpha=0.3, linestyle=":")
                ax.spines[['right', 'top']].set_visible(False)

                # overlay peaks (keep default red)
                s0_ms = centers_ms_all[trial_idx]
                _overlay_peaks(ax, t, y, s0_ms, peak_ch, peak_t_ms, ch_pos=ch, adjust_ms=adjust_ms)

                # label with impedance, colored to match trace
                if elec_id is not None:
                    nsp_txt = f" · NSP {nsp_ch:03d}" if nsp_ch is not None else ""
                    ax.set_ylabel(
                        f"elec {elec_id}{nsp_txt}\n({_fmt_impedance_kohm(imp)})",
                        rotation=0, labelpad=25, va="center", color=col
                    )
                    ax.yaxis.set_label_coords(-0.10, 0.5)
                else:
                    ax.set_ylabel("elec ?\n(—)", rotation=0, labelpad=25, va="center", color="gray")

                ax.set_xlim(WINDOW_MS[0], WINDOW_MS[1])
                if YLIM_UV is not None:
                    ax.set_ylim(*YLIM_UV)

            # turn off any leftover axes if channels < multiple of 6
            for k in range(len(ch_group), len(axes)):
                axes[k].axis("off")

            axes[-1].set_xlabel("Time (ms) rel. IR onset")

            # in the figure title:
            fig.suptitle(
                f"{SESSION} / BR {BR_IDX:03d} / RAW NS6 / IR-aligned {int(WINDOW_MS[0])}–{int(WINDOW_MS[1])} ms / "
                f"trial {trial_idx} / group {fig_idx}\n"
                "imp: ≥1000 kΩ = red • 500–<1000 kΩ = orange • <500 kΩ = blue",
                y=0.995
            )

            fig.tight_layout(rect=[0, 0.02, 1, 0.97])
            fig.subplots_adjust(left=0.20)  # extra left margin so y-labels aren't clipped

            group_tag = _group_tag_from_elecs(rec, ch_group)
            out_png = out_dir / (
                f"{SESSION}__BR_{BR_IDX:03d}__trial_{trial_idx:03d}"
                f"__{group_tag}__IR__win_{int(WINDOW_MS[0])}-{int(WINDOW_MS[1])}ms.png"
            )
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {out_png}")

    print("[done] IR-aligned stacked figures per trial →", OUT_DIR_BASE)

if __name__ == "__main__":
    main()
