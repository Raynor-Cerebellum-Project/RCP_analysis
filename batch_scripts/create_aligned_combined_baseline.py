from pathlib import Path
import re, sys, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from scipy.signal import filtfilt
from scipy.signal.windows import gaussian as _scipy_gaussian
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
import RCP_analysis as rcp

# ---------- CONFIG / roots ----------
REPO_ROOT  = Path(__file__).resolve().parents[1]
PARAMS     = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE  = SESSION_LOC / "results"; OUT_BASE.mkdir(parents=True, exist_ok=True)
INTAN_ROOT = SESSION_LOC / "Intan"; INTAN_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_ROOT = SESSION_LOC / "Metadata"; METADATA_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_CSV  = METADATA_ROOT / f"{Path(PARAMS.session)}_metadata.csv"
SHIFTS_CSV    = METADATA_ROOT / "br_to_intan_shifts.csv"

# ---------- checkpoints / aux_data ----------
NPRW_CKPT_ROOT  = OUT_BASE / "checkpoints" / "NPRW"
UA_CKPT_ROOT    = OUT_BASE / "checkpoints" / "UA"
BEHV_CKPT_ROOT  = OUT_BASE / "checkpoints" / "Behavior"
BASELINE_ROOT   = OUT_BASE / "checkpoints" / "Behavior" / "baseline_concat"
ALIGNED_OUT     = BASELINE_ROOT / "rates_from_curated"
ALIGNED_OUT.mkdir(parents=True, exist_ok=True)

NPRW_AUX_DATA    = OUT_BASE / "aux_data" / "NPRW"

# ---------- analysis constants ----------
IR_STREAM        = "USB board digital input channel"  # stream to preview
DEFAULT_BR_FS    = 30000.0  # Hz; override if you know the true ns5 rate
WIN_MS           = (-600.0, 600.0)
BASELINE_FIRST_MS = 150.0

# ---------- keypoints order (from YAML, with fallback) ----------
KEYPOINTS_ORDER = tuple((PARAMS.kinematics).get("keypoints"))

CURATED_EVENTS_A = []
CURATED_EVENTS_B = []
# CURATED_EVENTS_B = [0, 3, 4, 7, 8, 11, 13, *range(15, 20), 23, 24, 30, 33, 37, 44, 53, 55]

# ---------- helpers ----------
def _load_br_to_intan_shifts(shifts_csv: Path) -> dict[int, dict]:
    """
    Return {BR_idx: {"session": <str>, "anchor_ms": <float or 0.0>}}.
    Accepts header variations.
    """
    if not shifts_csv.exists():
        raise SystemExit(f"[error] shifts CSV not found: {shifts_csv}")
    with shifts_csv.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            raise SystemExit(f"[error] {shifts_csv.name} has no header")
        # normalize header names -> original
        cols = { re.sub(r'[^a-z0-9]+','', c.lower()): c for c in rdr.fieldnames }

        def col(*cands):
            for cand in cands:
                k = re.sub(r'[^a-z0-9]+','', cand.lower())
                if k in cols: return cols[k]
            return None

        c_br   = col("br_idx")
        c_sess = col("session")
        c_shift = col("anchor_ms")

        if not c_br or not c_sess:
            raise SystemExit(f"[error] {shifts_csv.name} missing BR and/or Session columns (have: {rdr.fieldnames})")

        out: dict[int, dict] = {}
        for row in rdr:
            try:
                br = int(float(str(row[c_br]).strip()))
            except Exception:
                continue
            sess = str(row.get(c_sess, "")).strip()
            if not sess:
                continue
            try:
                shift_ms = float(str(row.get(c_shift, "0")).strip()) if c_shift else 0.0
            except Exception:
                shift_ms = 0.0
            out[br] = {"session": sess, "anchor_ms": shift_ms}
    return out

def _read_csv_robust(path: Path) -> pd.DataFrame:
    encodings = ("utf-8", "utf-8-sig")
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not read CSV {path} (last error: {last_err})")

def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def _detect_IR_crossings(x: np.ndarray, fs: float | None, refractory_sec: float = 0.0005) -> np.ndarray:
    """
    Binarize at mid-point and return sample indices of 1→0 edges.
    Debounce via refractory (default 0.5 ms).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([], dtype=np.int64)

    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    thr = 0.5 * (lo + hi)
    b = (x > thr).astype(np.int8)

    # edges where b[t-1]=1 and b[t]=0
    db = np.diff(b, prepend=b[0])
    edges = np.flatnonzero(db == -1)

    if edges.size == 0:
        return edges.astype(np.int64)

    if fs and fs > 0:
        refr = max(1, int(round(refractory_sec * fs)))
    else:
        refr = 1

    keep = []
    last = -10**12
    for i in edges:
        if i - last >= refr:
            keep.append(i)
            last = i
    return np.asarray(keep, dtype=np.int64)

def _resolve_curated_indices_for_port(port: str, total_n: int):
    """
    Return np.array of curated indices for this port, or None if not specified.
    Curation is optional; empty lists disable curation.
    """
    p = str(port).upper().strip()
    if p == "A":
        src = CURATED_EVENTS_A
    elif p == "B":
        src = CURATED_EVENTS_B
    else:
        return None

    keep = np.array(list(src), dtype=int)
    if keep.size == 0:
        return None
    # de-dup, sort, and clip to valid range
    keep = np.unique(keep[(keep >= 0) & (keep < int(total_n))])
    return keep if keep.size else None

# --- helpers for xy matrix (behavior)

def _find_aligned_cam_csv(trial: str, cam: int = 0) -> Path | None:
    """
    Prefer combined '<trial>_both_cams_aligned.csv', else per-cam '<trial>_Cam-{cam}_aligned.csv'.
    """
    p_both = BEHV_CKPT_ROOT / f"{trial}_both_cams_aligned.csv"
    if p_both.exists():
        return p_both
    p_cam = BEHV_CKPT_ROOT / f"{trial}_Cam-{cam}_aligned.csv"
    return p_cam if p_cam.exists() else None

def _extract_xy_matrix(aligned_csv: Path, cam: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load an aligned behavior CSV and return:
      - t_cam_sec : (T,) BR-aligned time in seconds
      - XY        : (T, 2*K) stacked [x0, y0, x1, y1, ...]
      - names     : list of column names used for each 2-column pair
    """
    # ---- local helpers ----------------------------------------------------
    def flatten_cols(cols) -> list[str]:
        out = []
        for c in cols:
            if isinstance(c, tuple):
                parts = [str(p) for p in c if p is not None and str(p) != "nan"]
                out.append("_".join(parts))
            else:
                out.append(str(c))
        return out

    def norm_name(s: str) -> str:
        s = str(s).lower()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        return re.sub(r'_+', '_', s).strip('_')

    def discover_cam_groups(df_cols, cam: int):
        """
        Return dict base -> {'x': col, 'y': col, 'lik': col or None} for the camera.
        Matches ANY 'cam{cam}_<...>_{x|y|likelihood}' columns (normalized),
        no assumptions about the <...> tokens.
        """
        cam_tag = f"cam{cam}_"
        groups = {}
        orig = [str(c) for c in df_cols]
        norm = [norm_name(c) for c in orig]

        for n, o in zip(norm, orig):
            if not n.startswith(cam_tag):
                continue
            if n.endswith('_likelihood'):
                base = n[len(cam_tag):-len('_likelihood')]
                g = groups.setdefault(base, {'x': None, 'y': None, 'lik': None})
                g['lik'] = o
            elif n.endswith('_x'):
                base = n[len(cam_tag):-len('_x')]
                g = groups.setdefault(base, {'x': None, 'y': None, 'lik': None})
                g['x'] = o
            elif n.endswith('_y'):
                base = n[len(cam_tag):-len('_y')]
                g = groups.setdefault(base, {'x': None, 'y': None, 'lik': None})
                g['y'] = o

        # keep only groups that have both x & y
        groups = {k: v for k, v in groups.items() if v['x'] is not None and v['y'] is not None}
        return groups

    def order_groups_for_output(groups: dict, keypoints_order: tuple[str, ...]):
        """
        Try to map discovered bases to the requested order (fuzzy).
        If no clear match, fall back to a sensible default order.
        Returns a list of (base, cols_dict).
        """
        bases = list(groups.keys())

        # quick fuzzy scorer: count overlapping tokens
        def score(base, kp):
            btok = set(norm_name(base).split('_'))
            ktok = set(norm_name(kp).split('_'))
            # small helpers to catch common variants
            if 'fingertip' in btok:
                ktok.add('tip')
            if 'tip' in btok:
                ktok.add('fingertip')
            return len(btok & ktok)

        picked = []
        used = set()
        for kp in keypoints_order:
            best = None
            best_s = 0
            for b in bases:
                if b in used:
                    continue
                s = score(b, kp)
                if s > best_s:
                    best_s, best = s, b
            if best is not None and best_s > 0:
                picked.append((best, groups[best]))
                used.add(best)

        # add any leftovers in a deterministic way
        prefer = ('wrist', 'mcp', 'fingertip', 'target', 'start')
        leftovers = [b for b in bases if b not in used]
        leftovers.sort(key=lambda b: (0 if any(p in b for p in prefer) else 1, b))
        picked.extend((b, groups[b]) for b in leftovers)
        return picked
    # -----------------------------------------------------------------------

    # read with MultiIndex tolerance, then normalize columns
    try:
        df = pd.read_csv(aligned_csv, header=[0, 1])
        flat = flatten_cols(df.columns)
        if sum(c.lower().startswith('unnamed') for c in flat) > len(flat) // 2:
            df = pd.read_csv(aligned_csv)
            df.columns = flatten_cols(df.columns)
        else:
            df.columns = flat
    except Exception:
        df = pd.read_csv(aligned_csv)
        df.columns = flatten_cols(df.columns)

    cols = list(df.columns)

    cam_prefix = f"cam{cam}"
    time_cands = ["ns5_sample", f"{cam_prefix}_ns5_sample"]
    time_col = next((c for c in cols if c in time_cands), None)
    if time_col is None:
        raise RuntimeError(f"{aligned_csv.name} has no ns5_sample column (tried {time_cands}).")

    t_cam_sec = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float) / DEFAULT_BR_FS

    # discover groups for this camera
    groups = discover_cam_groups(df.columns, cam=cam)
    if not groups:
        # keep behavior as before: return all-NaN if nothing found
        K = len(KEYPOINTS_ORDER) * 2
        XY = np.full((len(df), K), np.nan, float)
        names = [f"{kp}_{ax}(MISSING)" for kp in KEYPOINTS_ORDER for ax in ('x', 'y')]
        return t_cam_sec, XY, names

    # choose an output order (try to align to KEYPOINTS_ORDER, then leftovers)
    ordered = order_groups_for_output(groups, KEYPOINTS_ORDER)

    # build XY (T x 2*len(ordered)) with names
    XY_cols = []
    names = []
    for base, d in ordered:
        xvals = pd.to_numeric(df[d['x']], errors="coerce").to_numpy()
        yvals = pd.to_numeric(df[d['y']], errors="coerce").to_numpy()
        XY_cols.extend([xvals, yvals])
        names.extend([d['x'], d['y']])   # original column strings

    XY = np.vstack(XY_cols).T  # shape (T, 2*K)

    finite_pct = float(np.isfinite(XY).mean()) if XY.size else 1.0
    print(f"[kin] {aligned_csv.name} cam={cam} T={XY.shape[0]} K={XY.shape[1]//2} finite%={finite_pct:.3f}")

    return t_cam_sec, XY, names

def _remove_teleports_and_spikes(XY: np.ndarray, max_z: float = 6.0,
                                 jump_px: float | None = None,
                                 dilate: int = 1,
                                 interp_max_gap: int = 6) -> np.ndarray:
    """
    Detect frames where velocity (frame-to-frame) is implausibly large and set to NaN,
    then lightly inpaint small gaps.
    - max_z: robust z-score threshold on speed (per keypoint)
    - jump_px: absolute per-step jump threshold (pixels); if None, derive from IQR
    - dilate: also drop +/- dilate frames around each detected outlier
    - interp_max_gap: linearly fill NaN runs up to this length
    """
    if XY.size == 0:
        return XY
    out = XY.astype(float, copy=True)
    T, D = out.shape
    for col0 in range(0, D, 2):  # pair (x,y)
        x = out[:, col0]
        y = out[:, col0+1]
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        spd = np.sqrt(dx*dx + dy*dy)

        # robust z-score
        med = np.nanmedian(spd)
        mad = np.nanmedian(np.abs(spd - med)) + 1e-12
        z = 0.6745 * (spd - med) / mad

        # absolute jump threshold
        if jump_px is None:
            q75 = np.nanpercentile(spd, 75.0)
            q25 = np.nanpercentile(spd, 25.0)
            iqr = max(q75 - q25, 1.0)
            jthr = q75 + 6.0*iqr  # pretty conservative
        else:
            jthr = float(jump_px)

        bad = (np.abs(z) > max_z) | (spd > jthr)
        if bad.any() and dilate > 0:
            bad_idx = np.flatnonzero(bad)
            for i in bad_idx:
                lo = max(0, i - dilate)
                hi = min(T, i + dilate + 1)
                bad[lo:hi] = True

        # zero out those samples
        x[bad] = np.nan; y[bad] = np.nan
        out[:, col0] = x; out[:, col0+1] = y

        # small-gap interpolation
        if interp_max_gap > 0:
            # find NaN runs
            nanmask = ~np.isfinite(x)
            if nanmask.any():
                starts = np.where(nanmask & ~np.r_[False, nanmask[:-1]])[0]
                ends   = np.where(nanmask & ~np.r_[nanmask[1:], False])[0]
                for s,e in zip(starts, ends):
                    if (e - s + 1) <= interp_max_gap and s > 0 and e < T-1:
                        xi = np.interp(np.arange(s, e+1), [s-1, e+1], [x[s-1], x[e+1]])
                        yi = np.interp(np.arange(s, e+1), [s-1, e+1], [y[s-1], y[e+1]])
                        out[s:e+1, col0]   = xi
                        out[s:e+1, col0+1] = yi
    return out

def _make_gaussian_kernel(sigma_frames: float) -> np.ndarray:
    """
    Return a normalized odd-length 1D Gaussian kernel for 'sigma_frames' (in frames).
    """
    if sigma_frames <= 0:
        return np.array([1.0], dtype=float)
    # window length ~ 6σ and forced odd
    win_len = int(max(5, round(6.0 * sigma_frames)))
    if win_len % 2 == 0:
        win_len += 1

    if _scipy_gaussian is not None:
        g = _scipy_gaussian(win_len, std=sigma_frames)
    else:
        # local fallback
        n = np.arange(win_len) - (win_len - 1) / 2.0
        g = np.exp(-0.5 * (n / float(sigma_frames))**2)

    g = g.astype(float, copy=False)
    g_sum = g.sum()
    if not np.isfinite(g_sum) or g_sum == 0:
        return np.array([1.0], dtype=float)
    return g / g_sum

def _gaussian_filtfilt(XY: np.ndarray, sigma_frames: float = 2.0) -> np.ndarray:
    """
    Zero-phase Gaussian smoothing (per column) using filtfilt with a FIR Gaussian kernel.
    sigma_frames is in frames (corrected-frame domain).
    """
    if XY.size == 0 or sigma_frames <= 0:
        return XY
    out = XY.copy()
    g = _make_gaussian_kernel(sigma_frames)
    # filtfilt default padlen is 3*(max(len(a), len(b)) - 1) => 3*(len(g)-1) for FIR
    base_pad = 3 * (len(g) - 1)

    for j in range(out.shape[1]):
        col = out[:, j]
        finite = np.isfinite(col)
        if finite.sum() < len(g) + 2:
            # too short / too many NaNs to filter robustly
            continue
        # simple inpaint of NaNs for stability
        if (~finite).any():
            idx = np.arange(col.size)
            col = np.interp(idx, idx[finite], col[finite])

        padlen = min(base_pad, len(col) - 1)
        if padlen < 1:
            continue
        out[:, j] = filtfilt(g, [1.0], col, padlen=padlen)

    return out

def build_peri_event_windows(trial: str,
                             ir_crossing_sec: np.ndarray,
                             cam:int=0,
                             pre_ms: float = 750.0,
                             post_ms: float = 750.0,
                             sigma_frames: float = 2.0,
                             offset_sec: float = 0.0,
                             shift_ms: float = 0.0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load aligned cam CSV, clean, smooth, and return windows aligned to IR events.
    Returns (windows[N,K,T], rel_t_ms[T], labels[K]).

    offset_sec: any additional constant BR–Intan shift you may know (in seconds).
    shift_ms:  per-row shift (ms) from br_to_intan_shifts; this is SUBTRACTED.
    """
    csvp = _find_aligned_cam_csv(trial, cam=cam)
    if not csvp:
        raise FileNotFoundError(f"No aligned CSV for {trial} Cam-{cam}")
    t_cam_sec, XY, labels = _extract_xy_matrix(csvp, cam=cam)

    # clean/smooth
    XY = _remove_teleports_and_spikes(XY, max_z=6.0, jump_px=None, dilate=1, interp_max_gap=6)
    XY = _gaussian_filtfilt(XY, sigma_frames=sigma_frames)

    # windowing
    dt = np.nanmedian(np.diff(t_cam_sec[np.isfinite(t_cam_sec)]))
    if not np.isfinite(dt) or dt <= 0:
        raise RuntimeError("Could not infer frame dt from ns5_sample timing.")
    pre = pre_ms/1000.0; post = post_ms/1000.0
    T = int(round((pre+post)/dt)) + 1
    rel_t = (np.arange(T) * dt - pre) * 1000.0  # ms
    K = XY.shape[1]
    N = len(ir_crossing_sec)
    out = np.full((N, K, T), np.nan, float)

    shift_sec = float(shift_ms) / 1000.0
    for i, t0 in enumerate(ir_crossing_sec):
        # Align NPRW event time to BR frame time:
        t_center = t0 + offset_sec - shift_sec
        t0s = t_center - pre
        t1s = t_center + post
        i0 = np.searchsorted(t_cam_sec, t0s, side="left")
        i1 = np.searchsorted(t_cam_sec, t1s, side="right")
        idx = np.clip(np.round(np.linspace(i0, i1-1, T)).astype(int), 0, len(t_cam_sec)-1)
        out[i, :, :] = XY[idx, :].T
    return out, rel_t, labels

# NPRW/UA helpers
def find_nprw_rates_for_session(nprw_ckpt_root: Path, session: str) -> Path | None:
    cands = sorted(nprw_ckpt_root.glob(f"rates__{session}__*.npz"))
    return cands[0] if cands else None

def find_ua_rates_by_index(ua_ckpt_root: Path, br_idx: int) -> Path | None:
    cands = sorted(ua_ckpt_root.glob(f"rates__*_{int(br_idx):03d}__*.npz"))
    if not cands:
        return None
    pref = [p for p in cands if re.search(r"__sigma?25ms", p.stem)]
    return pref[0] if pref else cands[-1]

def _window_rate_around_events(rate: np.ndarray, t_vec_ms: np.ndarray,
                               event_ms: np.ndarray, pre_ms: float, post_ms: float):
    """
    rate: (n_ch, T_full) sampled at times t_vec_ms (in ms, increasing)
    event_ms: centers in the same timebase (ms)
    Returns (windows[N, n_ch, T], rel_t_ms[T])
    """
    if t_vec_ms.size < 2:
        return np.zeros((0, rate.shape[0], 0), np.float32), np.zeros((0,), np.float32)

    dt = float(np.median(np.diff(t_vec_ms)))
    T = int(round((pre_ms + post_ms) / dt)) + 1
    rel_t = (np.arange(T) * dt - pre_ms).astype(np.float32)

    out = np.full((len(event_ms), rate.shape[0], T), np.nan, np.float32)
    for i, c in enumerate(event_ms.astype(float)):
        t0 = c - pre_ms; t1 = c + post_ms
        i0 = int(np.searchsorted(t_vec_ms, t0, side="left"))
        i1 = int(np.searchsorted(t_vec_ms, t1, side="right"))
        if i0 >= t_vec_ms.size or i1 <= 0:
            continue
        idx = np.clip(np.round(np.linspace(max(i0,0), max(i1-1,0), T)).astype(int), 0, t_vec_ms.size-1)
        out[i] = rate[:, idx]
    return out, rel_t

def main():
    if not METADATA_CSV.exists():
        print(f"[error] Metadata CSV not found: {METADATA_CSV}", file=sys.stderr)
        sys.exit(2)

    # Build once: Video_File idx → trial (Cam-0 preference for inventory)
    shifts_full = _load_br_to_intan_shifts(SHIFTS_CSV)
    df = _read_csv_robust(METADATA_CSV)
    
    col_notes = "Notes"
    col_port  = "UA_port"
    col_depth = "Depth_mm"
    col_br    = "BR_File"
    col_video = "Video_File"

    if col_notes is None:
        print("[error] No 'Notes' column in metadata.", file=sys.stderr)
        sys.exit(3)

    is_baseline = df[col_notes].astype(str).str.lower().eq("baseline")
    baseline_rows = df.loc[is_baseline].copy()
    print(f"[info] total rows: {len(df)}")
    print(f"[info] baseline rows: {len(baseline_rows)}\n")
    if baseline_rows.empty:
        return

    baseline_rows["_ua_port_norm"] = (
        baseline_rows[col_port].str.upper().replace({"": "UNKNOWN"})
        if col_port else "UNKNOWN"
    )
    if col_depth:
        depth_num = pd.to_numeric(baseline_rows[col_depth], errors="coerce")
        baseline_rows["_depth_mm_norm"] = depth_num.where(depth_num.notna(), baseline_rows[col_depth].astype(str)).replace({"": "n/a"})
    else:
        baseline_rows["_depth_mm_norm"] = "n/a"

    groups = list(baseline_rows.groupby(["_ua_port_norm", "_depth_mm_norm"], sort=True))

    for (port, depth), port_depth_group in groups:
        depth_label = str(depth)
        print(f"=== UA_port: {port} | Depth_mm: {depth_label}  (n={len(port_depth_group)}) ===")

        # Collectors across rows in this UA_port×Depth group
        ir_crossing_sec_all: list[np.ndarray] = []          # per-row arrays of ir crossing centers (sec, Intan)
        trial_per_event: list[str] = []            # one entry per event
        shift_per_event_ms: list[float] = []      # one entry per event
        session_blocks: list[dict] = []            # for later curated-rate extraction

        # per-camera kinematic windows (to be concatenated after the loop)
        win_all_cam0: list[np.ndarray] = []
        win_all_cam1: list[np.ndarray] = []
        rel_t_ms_cam0 = rel_t_ms_cam1 = None
        labels_cam0 = labels_cam1 = None

        # running global event index for concat bookkeeping
        global_start = 0

        for _, row in port_depth_group.iterrows():
            # processing matching files and loading shifts
            br_str = row.get(col_br, "") if col_br else ""
            vid_str = row.get(col_video, "") if col_video else ""
            br_idx = int(br_str) if br_str else None
            video_idx = int(vid_str) if vid_str else None
            cond_br_idx = f"{PARAMS.session}_{int(video_idx):03d}" if video_idx is not None else None
            sess_entry = shifts_full.get(br_idx) if br_idx is not None else None
            intan_session = (sess_entry).get('session')
            shift_ms = float((sess_entry).get("anchor_ms", 0.0))

            print(f"[Processing baseline] [{port} | Depth={depth_label}]: BR={cond_br_idx or 'n/a'}  Intan={intan_session or 'n/a'}  "
                  f"BR={br_str or 'n/a'}  Video={vid_str or 'n/a'}  shift_ms={shift_ms:g}")

            if not intan_session or not cond_br_idx:
                continue

            # load IR edges
            try:
                rec_ir = se.read_split_intan_files(INTAN_ROOT / intan_session, mode="concatenate", stream_name=IR_STREAM, use_names_as_ids=True)
                rec_ir = spre.unsigned_to_signed(rec_ir) # Convert UInt16 to int16
            except Exception as e:
                print(f"[warn] Reading IR stream failed: Intan={intan_session}: {e}")
                continue
            
            fs = rec_ir.sampling_frequency
            sig = np.asarray(rec_ir.get_traces()).squeeze()
            if sig is None or sig.size == 0:
                print(f"[warn] No signal for Intan={intan_session}.")
                continue
            ir_crossing_idx = _detect_IR_crossings(sig, fs, refractory_sec=0.0005)
            if ir_crossing_idx.size == 0:
                print(f"[IR] {cond_br_idx}: no IR crossings in  in {intan_session}")
                continue

            ir_crossing_sec = ir_crossing_idx / fs if (fs and fs > 0) else np.full(ir_crossing_idx.size, np.nan)
            print(f"[IR] {cond_br_idx}: {ir_crossing_idx.size} IR crossings in {intan_session}")

            # Kinematics windows — Cam-0 (required)
            try:
                kin0, rel0, lab0 = build_peri_event_windows(
                    cond_br_idx, ir_crossing_sec, cam=0,
                    pre_ms=-WIN_MS[0], post_ms=WIN_MS[1],
                    sigma_frames=2.0, offset_sec=0.0,
                    shift_ms=shift_ms
                )
            except Exception as e:
                print(f"[warn] kinematics failed for Trial={cond_br_idx}: {e}")
                continue

            # Kinematics windows — Cam-1 (optional)
            kin1 = rel1 = lab1 = None
            try:
                kin1, rel1, lab1 = build_peri_event_windows(
                    cond_br_idx, ir_crossing_sec, cam=1,
                    pre_ms=-WIN_MS[0], post_ms=WIN_MS[1],
                    sigma_frames=2.0, offset_sec=0.0,
                    shift_ms=shift_ms
                )
            except Exception as e:
                print(f"[warn] kinematics Cam-1 failed for Trial={cond_br_idx}: {e}")

            # append per-row chunks
            win_all_cam0.append(kin0); rel_t_ms_cam0 = rel0; labels_cam0 = lab0
            if isinstance(kin1, np.ndarray):
                win_all_cam1.append(kin1); rel_t_ms_cam1 = rel1; labels_cam1 = lab1

            # provenance per event
            ir_crossing_sec_all.append(ir_crossing_sec)
            trial_per_event.extend([cond_br_idx] * len(ir_crossing_sec))
            shift_per_event_ms.extend([shift_ms] * len(ir_crossing_sec))

            # concat index bookkeeping for later curated-rate extraction
            n_this = len(ir_crossing_sec)
            session_blocks.append({
                "intan_session": intan_session,
                "br_idx": br_idx,
                "shift_ms": shift_ms,
                "ir_crossing_sec": ir_crossing_sec,            # Intan seconds (pre-shift)
                "concat_start": global_start,  # inclusive
                "concat_end": global_start + n_this,  # exclusive
            })
            global_start += n_this

        # ===== after iterating all rows in this group, concatenate & save =====
        if not win_all_cam0:
            print(f"[group] UA={port} Depth={depth_label}: no usable events.")
            continue

        ir_crossing_cat = np.concatenate(ir_crossing_sec_all, axis=0)                  # (N,)
        cam0_cat = np.concatenate(win_all_cam0, axis=0)                # (N, K0, T)
        cam1_cat = np.concatenate(win_all_cam1, axis=0) if win_all_cam1 else None  # (N, K1, T) or None
        keep = _resolve_curated_indices_for_port(port, cam0_cat.shape[0]) if cam0_cat is not None else None

        def _save_cam_npz(suffix: str, win_cat: np.ndarray, rel_t_ms: np.ndarray, labels: list[str]):
            if win_cat is None:
                return None
            fname_base = f"baseline_UA_{_sanitize(str(port))}_Depth{_sanitize(str(depth_label))}{suffix}"
            if keep is None:
                fname = f"{fname_base}.npz"
                windows_to_save = win_cat
                events_sec_to_save = ir_crossing_cat
                trials_to_save = np.array(trial_per_event, dtype=object)
                shifts_to_save = np.array(shift_per_event_ms, dtype=float)
                curated_flag = "no"
                print(f"[curate] UA={port}{suffix}: saving ALL {win_cat.shape[0]} events.")
            else:
                fname = f"{fname_base}_curated.npz"
                windows_to_save = win_cat[keep, :, :]
                events_sec_to_save = ir_crossing_cat[keep]
                trials_to_save = np.array(trial_per_event, dtype=object)[keep]
                shifts_to_save = np.array(shift_per_event_ms, dtype=float)[keep]
                curated_flag = "yes"
                print(f"[curate] UA={port}{suffix}: keeping {keep.size}/{win_cat.shape[0]} events → {fname}")

            out_npz = BASELINE_ROOT / fname
            np.savez_compressed(
                out_npz,
                windows=windows_to_save,
                t_ms=np.asarray(rel_t_ms, float) if rel_t_ms is not None else np.array([], float),
                labels=np.array(labels, dtype=object) if labels is not None else np.array([], dtype=object),
                events_sec=events_sec_to_save,
                trials=trials_to_save,             # now 1:1 with events
                shift_ms=shifts_to_save,         # now 1:1 with events
                ua_port=str(port),
                depth_mm=str(depth_label),
            )
            print(f"[save] wrote {out_npz}")

            # manifest row
            manifest = BASELINE_ROOT / "manifest.csv"
            write_header = not manifest.exists()
            with manifest.open("a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["file","ua_port","depth_mm","n_events","n_traces","T","curated"])
                w.writerow([
                    str(out_npz), str(port), str(depth_label),
                    int(windows_to_save.shape[0]), int(windows_to_save.shape[1]), int(windows_to_save.shape[2]),
                    curated_flag
                ])
            return out_npz

        out_cam0 = _save_cam_npz("_cam0", cam0_cat, rel_t_ms_cam0, labels_cam0)
        out_cam1 = _save_cam_npz("_cam1", cam1_cat, rel_t_ms_cam1, labels_cam1)

        # ===== NPRW/UA rate windows (always produce ALL; add CURATED if keep exists) =====
        print(f"[rates] UA={port}: extracting NPRW/UA rate windows (ALL"
              f"{' + CURATED' if (keep is not None and keep.size) else ''})...")

        # Accumulators per selection label
        acc = {
            "ALL": {
                "nprw_chunks": [], "ua_chunks": [], "ir_crossing_sec": [],
                "sess": [], "br": [], "gidx": [], "lidx": [],
                "rel_t_nprw_ref": None,  # NPRW peri-event time axis
                "rel_t_ua_ref": None,  # UA peri-event time axis
            }
        }
        if keep is not None and keep.size:
            acc["CURATED"] = {
                "nprw_chunks": [], "ua_chunks": [], "ir_crossing_sec": [],
                "sess": [], "br": [], "gidx": [], "lidx": [],
                "rel_t_nprw_ref": None,
                "rel_t_ua_ref": None,
            }

        # helper for optional resampling across *sessions* (never UA→NPRW)
        def _resample_win(win: np.ndarray, t_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
            """
            win: (N, C, T_src), t_src/t_tgt: (T,)
            Resample along time to t_tgt.
            """
            if win.size == 0 or t_src.size == 0 or t_tgt.size == 0:
                return win
            n_ev, n_ch, _T = win.shape
            out = np.empty((n_ev, n_ch, t_tgt.size), dtype=np.float32)
            for e in range(n_ev):
                for ch in range(n_ch):
                    out[e, ch] = np.interp(t_tgt, t_src, win[e, ch].astype(np.float32, copy=False))
            return out

        for blk in session_blocks:
            g0, g1 = int(blk["concat_start"]), int(blk["concat_end"])
            n_this = g1 - g0
            if n_this <= 0:
                continue

            # Build selection sets for this block
            sel = {"ALL": np.arange(n_this, dtype=int)}
            if "CURATED" in acc:
                sel_global = keep[(keep >= g0) & (keep < g1)]
                if sel_global.size:
                    sel["CURATED"] = (sel_global - g0).astype(int)

            if not sel:
                sel = {"ALL": np.arange(n_this, dtype=int)}

            nprw_rates_npz = find_nprw_rates_for_session(NPRW_CKPT_ROOT, blk["intan_session"])
            ua_rates_npz    = find_ua_rates_by_index(UA_CKPT_ROOT, int(blk["br_idx"]) if blk["br_idx"] is not None else -1)
            if nprw_rates_npz is None or ua_rates_npz is None:
                print(f"[rates] skip session={blk['intan_session']}: missing rates "
                      f"(NPRW={bool(nprw_rates_npz)}, UA={bool(ua_rates_npz)})")
                continue

            nprw_rate, nprw_t_ms, *_ = rcp.load_rate_npz(nprw_rates_npz)
            ua_rate, ua_t_ms, *_ = rcp.load_rate_npz(ua_rates_npz)

            # Align Intan to BR frame using anchor; UA already in BR ms
            nprw_t_ms_aligned = nprw_t_ms - float(blk["shift_ms"])
            ua_t_ms_aligned = ua_t_ms

            # dt diagnostics (for logging/meta only)
            dt_nprw = float(np.median(np.diff(nprw_t_ms_aligned))) if nprw_t_ms_aligned.size > 1 else np.nan
            dt_ua = float(np.median(np.diff(ua_t_ms_aligned))) if ua_t_ms_aligned.size > 1 else np.nan
            st = {"dt_nprw_ms": dt_nprw, "dt_ua_ms": dt_ua}
            print(f"[rates] session={blk['intan_session']}: dt_nprw≈{dt_nprw:.3f} ms, dt_ua≈{dt_ua:.3f} ms (native bins)")

            for tag, sel_local in sel.items():
                if sel_local.size == 0:
                    continue

                # Event centers in BR ms (same for Intan+UA; different dt, same origin)
                ir_crossing_ms_aligned = (blk["ir_crossing_sec"][sel_local].astype(float) * 1000.0) - float(blk["shift_ms"])

                # Build peri-event windows using native timebases
                nprw_win, rel_t_nprw = _window_rate_around_events(
                    nprw_rate, nprw_t_ms_aligned, ir_crossing_ms_aligned, -WIN_MS[0], WIN_MS[1]
                )
                ua_win, rel_t_ua = _window_rate_around_events(
                    ua_rate, ua_t_ms_aligned, ir_crossing_ms_aligned, -WIN_MS[0], WIN_MS[1]
                )

                # Ensure per-tag consistent rel_t across sessions separately for NPRW/UA
                if acc[tag]["rel_t_nprw_ref"] is None:
                    acc[tag]["rel_t_nprw_ref"] = rel_t_nprw.copy()
                elif not np.allclose(rel_t_nprw, acc[tag]["rel_t_nprw_ref"], atol=1e-6):
                    nprw_win = _resample_win(nprw_win, rel_t_nprw, acc[tag]["rel_t_nprw_ref"])
                    rel_t_nprw = acc[tag]["rel_t_nprw_ref"]

                if acc[tag]["rel_t_ua_ref"] is None:
                    acc[tag]["rel_t_ua_ref"] = rel_t_ua.copy()
                elif not np.allclose(rel_t_ua, acc[tag]["rel_t_ua_ref"], atol=1e-6):
                    ua_win = _resample_win(ua_win, rel_t_ua, acc[tag]["rel_t_ua_ref"])
                    rel_t_ua = acc[tag]["rel_t_ua_ref"]

                # Accumulate
                acc[tag]["nprw_chunks"].append(nprw_win.astype(np.float32, copy=False))
                acc[tag]["ua_chunks"].append(ua_win.astype(np.float32, copy=False))
                acc[tag]["ir_crossing_sec"].append(ir_crossing_ms_aligned.astype(np.float32, copy=False))
                acc[tag]["sess"].append(
                    np.array([blk["intan_session"]] * nprw_win.shape[0], dtype=object)
                )
                acc[tag]["br"].append(
                    np.array([int(blk["br_idx"]) if blk["br_idx"] is not None else -1] * nprw_win.shape[0],
                             dtype=np.int32)
                )

                # For indices, store global and local (relative to concat)
                if tag == "ALL":
                    sel_global_for_tag = np.arange(g0, g1, dtype=int)
                else:
                    sel_global_for_tag = (sel_local + g0).astype(int)

                acc[tag]["gidx"].append(sel_global_for_tag.astype(np.int32, copy=False))
                acc[tag]["lidx"].append(sel_local.astype(np.int32, copy=False))

                # Per-session save (native dt for each side)
                out_rates = ALIGNED_OUT / (
                    f"rates_from_curated__UA_{_sanitize(str(port))}"
                    f"__Depth{_sanitize(str(depth_label))}"
                    f"__{blk['intan_session']}__BR_{int(blk['br_idx']) if blk['br_idx'] is not None else -1:03d}"
                    f"__{tag}.npz"
                )
                np.savez_compressed(
                    out_rates,
                    nprw_rate_win=nprw_win.astype(np.float32),
                    ua_rate_win=ua_win.astype(np.float32),
                    nprw_t_rel_ms=rel_t_nprw.astype(np.float32),
                    ua_t_rel_ms=rel_t_ua.astype(np.float32),
                    event_center_ms=ir_crossing_ms_aligned.astype(np.float32),
                    port=str(port),
                    depth_mm=str(depth_label),
                    nprw_session=str(blk["intan_session"]),
                    br_idx=int(blk["br_idx"]) if blk["br_idx"] is not None else -1,
                    dt_nprw_ms=float(st.get("dt_nprw_ms", np.nan)),
                    dt_ua_ms=float(st.get("dt_ua_ms", np.nan)),
                    n_events_kept=int(sel_local.size),
                    curated_global_indices=sel_global_for_tag.astype(int),
                    curated_local_indices=sel_local.astype(int),
                    nprw_rates=str(nprw_rates_npz),
                    ua_rates=str(ua_rates_npz),
                    align_note=(
                        "Intan ms aligned to BR by subtracting anchor_ms; "
                        "UA kept on BR ms; NPRW/UA peri-event windows use native binning."
                    ),
                )
                print(f"[rates] wrote {out_rates}")

        # Concatenate across sessions and save per selection
        for tag, A in acc.items():
            if not A["nprw_chunks"]:
                continue
            nprw_all = np.concatenate(A["nprw_chunks"], axis=0)
            ua_all = np.concatenate(A["ua_chunks"], axis=0)
            ir_crossing_all = np.concatenate(A["ir_crossing_sec"], axis=0)
            sess_all = np.concatenate(A["sess"], axis=0)
            br_all   = np.concatenate(A["br"],   axis=0)
            gidx_all = np.concatenate(A["gidx"], axis=0)
            lidx_all = np.concatenate(A["lidx"], axis=0)

            out_cat = ALIGNED_OUT / (
                f"rates_from_curated__UA_{_sanitize(str(port))}"
                f"__Depth{_sanitize(str(depth_label))}__{tag}.npz"
            )
            np.savez_compressed(
                out_cat,
                nprw_rate_win=nprw_all.astype(np.float32),
                ua_rate_win=ua_all.astype(np.float32),
                nprw_t_rel_ms=A["rel_t_nprw_ref"].astype(np.float32),
                ua_t_rel_ms=A["rel_t_ua_ref"].astype(np.float32),
                event_center_ms=ir_crossing_all.astype(np.float32),
                event_session=sess_all,
                event_br_idx=br_all.astype(np.int32),
                curated_global_idx=gidx_all.astype(np.int32),
                curated_local_idx=lidx_all.astype(np.int32),
                port=str(port),
                depth_mm=str(depth_label),
                n_events_total=int(nprw_all.shape[0]),
                align_note=(
                    "Peri-event rate windows saved separately for NPRW and UA with native binning; "
                    "timebases are already in a common BR-aligned origin."
                ),
            )
            print(f"[rates][{tag}] wrote {out_cat}  "
                  f"(N={nprw_all.shape[0]}, NPRW_ch={nprw_all.shape[1]}, UA_ch={ua_all.shape[1]}, "
                  f"T_NPRW={nprw_all.shape[2]}, T_UA={ua_all.shape[2]})")

        # Concatenate across sessions and save per selection
        for tag, A in acc.items():
            if not A["nprw_chunks"]:
                continue
            nprw_all = np.concatenate(A["nprw_chunks"], axis=0)
            ua_all = np.concatenate(A["ua_chunks"], axis=0)
            ir_crossing_all = np.concatenate(A["ir_crossing_sec"], axis=0)
            sess_all = np.concatenate(A["sess"], axis=0)
            br_all   = np.concatenate(A["br"],   axis=0)
            gidx_all = np.concatenate(A["gidx"], axis=0)
            lidx_all = np.concatenate(A["lidx"], axis=0)

            out_cat = ALIGNED_OUT / (
                f"rates_from_curated__UA_{_sanitize(str(port))}"
                f"__Depth{_sanitize(str(depth_label))}__{tag}.npz"
            )
            np.savez_compressed(
                out_cat,
                nprw_rate_win=nprw_all.astype(np.float32),
                ua_rate_win=ua_all.astype(np.float32),
                nprw_t_rel_ms=A["rel_t_nprw_ref"].astype(np.float32),
                ua_t_rel_ms=A["rel_t_ua_ref"].astype(np.float32),
                event_center_ms=ir_crossing_all.astype(np.float32),
                event_session=sess_all,
                event_br_idx=br_all.astype(np.int32),
                curated_global_idx=gidx_all.astype(np.int32),
                curated_local_idx=lidx_all.astype(np.int32),
                port=str(port),
                depth_mm=str(depth_label),
                n_events_total=int(nprw_all.shape[0]),
                align_note=(
                    "Peri-event rate windows saved separately for NPRW and UA with native binning; "
                    "timebases share a BR-aligned origin."
                ),
            )
            print(f"[rates][{tag}] wrote {out_cat}  "
                f"(N={nprw_all.shape[0]}, NPRW={nprw_all.shape[1]}, UA_ch={ua_all.shape[1]}, "
                f"T_NPRW={nprw_all.shape[2]}, T_UA={ua_all.shape[2]})")

if __name__ == "__main__":
    main()
