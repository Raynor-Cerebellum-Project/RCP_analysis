from pathlib import Path
from typing import Tuple, Optional
from scipy.signal import filtfilt
from types import SimpleNamespace
import numpy as np
import re, json
from probeinterface import Probe
from scipy.io import loadmat

import RCP_analysis as rcp

import matplotlib
matplotlib.use("Agg")                 # set FIRST
from matplotlib import pyplot as plt  # then import pyplot
matplotlib.rcParams['svg.fonttype'] = 'none'

# ---------- CONFIG ----------
WIN_MS            = (-600.0, 600.0)
NORMALIZE_FIRST_MS = 150.0
MIN_TRIALS        = 1

VMIN_NPRW_BASELINE, VMAX_NPRW_BASELINE = -25, 100
VMIN_UA_BASELINE, VMAX_UA_BASELINE = -25, 100
VMIN_SMA_BASELINE, VMAX_SMA_BASELINE = -10, 35

VMIN_NPRW, VMAX_NPRW = -25, 200
VMIN_UA, VMAX_UA = -50, 150
COLORMAP = "jet"

ARTRMV_MS_BEFORE = 5.0
ARTCORR_TAIL_MS   = 5.0

# --- Velocity setting ---
VEL_THRESH = 5.0          # absolute velocity threshold (a.u./ms) for despiking
VEL_MAX_GAP = 5           # interpolate NaN runs up to this many samples
# --- Visualization settings ---
GAUSS_SMOOTH_MS = 0   # 0 → disable smoothing

# ---- FIGURE LAYOUT KNOBS ----
BEH_RATIO = 0.6               # height ratio for behavior rows (position/velocity); adjust as needed
CH_RATIO_PER_ROW = 0.015      # height ratio per neural channel row (heatmaps)
MIN_HEATMAP_RATIO = 0.6       # minimum ratio so tiny arrays don't vanish
UA_COMPACT_FACTOR = 0.95   # < 1.0 shrinks UA panel heights
NPRW_SCALE      = 0.6   # < 1.0 shrinks Intan height (e.g., 0.6 = 60% of previous)
GAP_BEH_NPRW    = 0.25  # height "ratio" for a spacer row between behavior and Intan
FIG_WIDTH_IN         = 16.0        # ← overall width (inches)
HEIGHT_PER_RATIO_IN  = 4.0         # ← height per unit of `ratios` sum

# ---------- params / roots ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)

SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE  = SESSION_LOC / "results"; OUT_BASE.mkdir(parents=True, exist_ok=True)

# ---------- Impedance loading ----------
UA_IMP_MAX_KOHM = 1000.0        # threshold for excluding UA rows; adjust as needed
EXCLUDE_UA_HIGH_Z = True       # set False to disable masking quickly

# Where the files live; mirror the other script:
IMP_BASE = OUT_BASE.parents[0]  # same as your script (one level above OUT_BASE)
IMP_FILES = {
    "A": IMP_BASE / "Impedances" / "Utah_imp_Port_A",
    "B": IMP_BASE / "Impedances" / "Utah_imp_Port_B",
}

# ---------- figures ----------
FIG_ROOT   = OUT_BASE / "figures"; FIG_ROOT.mkdir(parents=True, exist_ok=True)
FIG = SimpleNamespace(peri_posvel_median=FIG_ROOT / "median_fr_plots", peri_posvel_var=FIG_ROOT / "var_fr_plots")
FIG.peri_posvel_median.mkdir(parents=True, exist_ok=True)
FIG.peri_posvel_var.mkdir(parents=True, exist_ok=True)

# ---------- checkpoints / inputs ----------
ALIGNED_ROOT = OUT_BASE / "checkpoints" / "Aligned"
BEHAV_ROOT   = OUT_BASE / "checkpoints" / "Behavior" / "baseline_concat"
NPRW_AUX_DATA   = OUT_BASE / "aux_data" / "NPRW"
METADATA_ROOT = SESSION_LOC / "Metadata"; METADATA_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_CSV  = METADATA_ROOT / f"{Path(PARAMS.session)}_metadata.csv"
PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim"; PERI_ROOT.mkdir(parents=True, exist_ok=True)

# UA mapping (only load if path exists and probes provided)
XLS = rcp.ua_excel_path(REPO_ROOT, getattr(PARAMS, "probes", {}))
UA_MAP = rcp.load_UA_mapping_from_excel(XLS) if XLS and Path(XLS).exists() else None

GEOM_PATH = (
    Path(PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None) and str(PARAMS.geom_mat_rel).startswith("/")
    else (REPO_ROOT / PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None)
    else rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
)

# ---------- kinematics ----------
# Tuple order for keypoints (robust to missing config)
KEYPOINTS_ORDER = tuple(PARAMS.kinematics.get("keypoints"))
RATES_DIR = BEHAV_ROOT / "rates_from_curated"

def _pick_rates_all_path(rates_dir: Path, ua_port_choice: str = "A") -> tuple[Path, str]:
    """
    Prefer, in order:
      1) CURATED, port-specific, session-aggregated (no __BR_) 
      2) ALL,     port-specific, session-aggregated (no __BR_)
      5) last ALL of anything (per-BR, etc.)
    """
    def _no_br(lst):
        # keep only files that do NOT have __BR_ in the name
        return [p for p in lst if "__BR_" not in p.name]
    
    cur = sorted(rates_dir.glob(f"rates_from_curated__UA_{ua_port_choice}__Depth*__CURATED.npz"))
    cur = _no_br(cur)
    if cur:
        return cur[-1], "CURATED"
    
    all_port = sorted(rates_dir.glob(f"rates_from_curated__UA_{ua_port_choice}__Depth*__ALL.npz"))
    all_port = _no_br(all_port)
    if all_port:
        return all_port[-1], "ALL"
    return rates_dir / "rates_from_curated__UA_X__DepthY__ALL.npz", "ALL"

RATES_ALL_PATH, _rates_selection_tag = _pick_rates_all_path(RATES_DIR, "A")

_imp_pat_elecnum = re.compile(
    r"\belec\s*\d+\s*-\s*(\d{1,3})\s+([0-9]+(?:\.[0-9]+)?)\s*(k?ohms?|kΩ|ohms?|Ω)\b",
    flags=re.IGNORECASE,
)

def _order_rows_by_region_then_peak(
    ua_mat: np.ndarray,
    t_vec: np.ndarray,
    ids_1based: Optional[np.ndarray],
    *,
    region_order=(0, 1, 2, 3),   # SMA, DPM, M1 inf, M1 sup
    pre_only: bool = True,       # search t<0 (or <=0 if include_zero=True)
    include_zero: bool = True,
    peak_mode: str = "max",      # "max" | "min" | "abs"
    earliest_at_top: bool = True # <<< NEW: place earliest rows visually at top
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Sort rows within each region by earliest *time* of peak in the selected window.
    Tie-breaker: larger peak magnitude (by peak_mode).
    Rows with no finite values go to the bottom of their region.
    Unknown-region rows go after known regions, sorted the same way.

    If earliest_at_top=True, each region's block is reversed so that with
    imshow(origin="lower") the earliest peaks appear at the top.
    """
    if ua_mat is None or ua_mat.size == 0 or ids_1based is None:
        return ua_mat, ids_1based

    ua  = np.asarray(ua_mat, float)
    ids = np.asarray(ids_1based).astype(float)  # may contain NaN
    t   = np.asarray(t_vec, float)
    n_rows = ua.shape[0]

    # time mask
    if pre_only:
        mask_t = (t <= 0.0) if include_zero else (t < 0.0)
    else:
        mask_t = np.isfinite(t)
    if not np.any(mask_t):
        mask_t = np.isfinite(t)

    time_ix = np.where(mask_t)[0]

    # region codes
    regs = np.full(n_rows, 1_000_000, int)
    for r in range(n_rows):
        eid = ids[r]
        if np.isfinite(eid):
            regs[r] = _ua_region_code_from_elec(int(eid))

    # choose peak fn
    if peak_mode == "min":
        val_fn = np.nanmin
        arg_fn = np.nanargmin
        mag_fn = lambda x: -x  # more negative = “larger magnitude” for tie-break
    elif peak_mode == "abs":
        val_fn = lambda x: np.nanmax(np.abs(x))
        def arg_fn(x):
            ax = np.abs(x)
            return int(np.nanargmax(ax))
        mag_fn = lambda x: np.abs(x)
    else:  # "max"
        val_fn = np.nanmax
        arg_fn = np.nanargmax
        mag_fn = lambda x: x

    # features per row (global time for stability)
    peak_time = np.full(n_rows, np.inf, float)   # earlier is better
    peak_mag  = np.full(n_rows, -np.inf, float)  # larger is better (per mode)

    for r in range(n_rows):
        sub = ua[r, :][mask_t]
        if np.isfinite(sub).any():
            try:
                pv = val_fn(sub)
                peak_mag[r] = float(pv) if np.isfinite(pv) else -np.inf
                local_ix = arg_fn(sub)
                global_ix = time_ix[int(local_ix)]
                peak_time[r] = float(t[global_ix])
            except Exception:
                pass  # leave defaults

    def sort_block(ix: np.ndarray) -> np.ndarray:
        if ix.size == 0:
            return ix
        # primary: earlier time (ascending); secondary: larger magnitude (descending)
        ordered = ix[np.lexsort((-peak_mag[ix], peak_time[ix]))]
        # Put earliest at *top* when drawn with origin="lower"
        return ordered[::-1] if earliest_at_top else ordered

    order_chunks = []
    seen = np.zeros(n_rows, bool)

    # known regions in requested order
    for reg in region_order:
        idxs = np.where(regs == reg)[0]
        if idxs.size:
            ii = sort_block(idxs)
            order_chunks.append(ii)
            seen[ii] = True

    # unknown regions last (apply same top/bottom rule)
    rest = np.where(~seen)[0]
    if rest.size:
        order_chunks.append(sort_block(rest))

    order = np.concatenate(order_chunks) if order_chunks else np.arange(n_rows)
    return ua[order, :], (ids[order] if ids_1based is not None else ids_1based)

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
    filtered = bytes(ch for ch in b if 9 <= ch <= 126 or ch in (10, 13))
    return filtered.decode("latin-1", errors="ignore")

def _ylabel_horizontal(ax, text: str, pad: float = 8.0):
    # Horizontal y-label on the left; right-aligned to sit close to the axis.
    t = ax.set_ylabel(text, rotation=0, labelpad=pad)
    t.set_horizontalalignment("right")
    t.set_verticalalignment("center")
    return t

def load_impedances_from_textedit_dump(path_like: str | Path) -> dict[int, float]:
    """
    Parse lines like 'elec1-5   201 kOhm' from a loose text dump.
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

def _apply_relative_offsets_by_ref(
    lines: Optional[np.ndarray],
    labels: list[str],
    keypoints=KEYPOINTS_ORDER,
    ref_kp: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    For position lines (K,T), add a constant offset per trace so that:
      reference_kp_{x,y} → offset 0,
      other kp_{x}       → + (median(kp_x) - median(reference_kp_x)),
      other kp_{y}       → + (median(kp_y) - median(reference_kp_y)).
    If ref_kp is None, the first keypoint from `keypoints` that appears in labels is used.
    """
    if lines is None or not isinstance(lines, np.ndarray) or lines.ndim != 2 or lines.size == 0:
        return lines
    if not labels:
        return lines

    L = lines.copy()
    lab_lc = [str(s).lower() for s in labels]

    def find_idx(sought: str) -> Optional[int]:
        soughtrl = sought.lower()
        for i, s in enumerate(lab_lc):
            if s == soughtrl or s.endswith(soughtrl):
                return i
        return None

    # choose reference keypoint
    if ref_kp is None:
        for kp in keypoints:
            if find_idx(f"{kp}_x") is not None or find_idx(f"{kp}_y") is not None:
                ref_kp = kp
                break
    if ref_kp is None:
        return L  # nothing matched

    # reference medians
    irx = find_idx(f"{ref_kp}_x")
    iry = find_idx(f"{ref_kp}_y")
    med_rx = np.nanmedian(L[irx]) if irx is not None else None
    med_ry = np.nanmedian(L[iry]) if iry is not None else None

    # shift others relative to reference (per axis)
    for kp in keypoints:
        if kp == ref_kp:
            continue
        ix = find_idx(f"{kp}_x")
        if ix is not None and med_rx is not None:
            med_kx = np.nanmedian(L[ix])
            L[ix, :] = L[ix, :] + (med_kx - med_rx)
        iy = find_idx(f"{kp}_y")
        if iy is not None and med_ry is not None:
            med_ky = np.nanmedian(L[iy])
            L[iy, :] = L[iy, :] + (med_ky - med_ry)

    return L

def _gaussian_fir_coeffs(sigma_ms: float, dt_ms: float, truncate: float = 4.0) -> np.ndarray:
    """
    Build a discrete Gaussian FIR kernel (normalized) with std = sigma_ms,
    truncated at +/- truncate*sigma, length odd.
    """
    if not np.isfinite(sigma_ms) or sigma_ms <= 0 or not np.isfinite(dt_ms) or dt_ms <= 0:
        return np.array([1.0], float)
    sigma_samp = float(sigma_ms) / float(dt_ms)
    # ensure at least ~1 sample of width to avoid degeneracy
    sigma_samp = max(sigma_samp, 0.5)
    half = max(1, int(np.ceil(truncate * sigma_samp)))
    k = np.arange(-half, half + 1, dtype=float)
    g = np.exp(-0.5 * (k / sigma_samp) ** 2)
    g /= g.sum()
    return g

def _filtfilt_gaussian_nanaware_1d(y: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    NaN-aware zero-phase Gaussian smoothing using filtfilt with FIR 'b' (a=[1]).
    """
    y = np.asarray(y, float)
    if y.size == 0:
        return y
    if b.ndim != 1 or b.size == 0:
        return y
    # filtfilt needs minimum length; for FIR, ~3*len(b) is a safe guard
    min_len = max(9, 3 * b.size)

    # If too short, just return as-is (no smoothing)
    if y.size < min_len:
        return y.copy()

    mask = ~np.isfinite(y)
    if mask.all():
        return y.copy()

    filled = y.copy()
    filled[mask] = np.nanmedian(filled)

    # Zero-phase filtering with FIR kernel
    out = filtfilt(b, [1.0], filled, method="gust")
    out[mask] = np.nan
    return out

def _smooth_lines_gaussian_filtfilt(lines: np.ndarray, rel_t_ms: np.ndarray, sigma_ms: float) -> np.ndarray:
    """
    Apply filtfilt-based Gaussian smoothing along time to (K,T) or (T,) arrays.
    """
    if lines is None or np.size(lines) == 0 or sigma_ms <= 0.0:
        return lines
    L = np.asarray(lines, float)
    one_d = (L.ndim == 1)
    if one_d:
        L = L[None, :]

    dt_ms = float(np.nanmedian(np.diff(rel_t_ms))) if rel_t_ms.size > 1 else 1.0
    b = _gaussian_fir_coeffs(sigma_ms, dt_ms)

    out = L.copy()
    for k in range(out.shape[0]):
        out[k, :] = _filtfilt_gaussian_nanaware_1d(out[k, :], b)
    return out[0] if one_d else out

def _ua_region_code_from_elec(e: int) -> int:
    """Return 0..3 for the 4 UA regions (SMA, DPM, M1 inf, M1 sup). Unknown -> large code."""
    e = int(e)
    if   1  <= e <=  64:  return 0  # SMA
    elif 65 <= e <= 128:  return 1  # Dorsal premotor
    elif 129<= e <= 192:  return 2  # M1 inferior
    elif 193<= e <= 256:  return 3  # M1 superior
    return 1_000_000

def _simple_beh_labels(names: list[str],
                       keypoints: Tuple[str, ...] = KEYPOINTS_ORDER) -> list[str]:
    """
    Map raw DLC-style names like
      'DLC_Resnet50_..._wrist_x'  -> 'wrist_x'
      'cam0_elbow_y'              -> 'elbow_y'
    If a keypoint/axis can’t be found, fall back to the original name.
    """
    out: list[str] = []
    kps_lc = tuple(kp.lower() for kp in keypoints)
    for n in names:
        s = str(n).lower()
        kp = next((kp for kp in kps_lc if kp in s), None)
        axis = "x" if ("_x" in s or s.endswith("x")) else (
                "y" if ("_y" in s or s.endswith("y")) else None)
        if kp and axis:
            out.append(f"{kp}_{axis}")
        else:
            out.append(str(n))
    return out

def _baseline_zero_trials(wins: np.ndarray, rel_t_ms: np.ndarray, normalize_first_ms: float) -> np.ndarray:
    """
    wins: (N_events, K_traces, T) — baseline-zero per event/trace using
          [rel_t_ms[0], normalize_first_ms].
    """
    if wins.size == 0:
        return wins
    bl_mask = (rel_t_ms >= rel_t_ms[0]) & (rel_t_ms <= float(normalize_first_ms))
    if not np.any(bl_mask):
        cut = max(1, int(round(0.1 * rel_t_ms.size)))  # fallback: first 10%
        bl_mask = np.zeros_like(rel_t_ms, dtype=bool); bl_mask[:cut] = True
    wins = wins.astype(float, copy=True)
    bl = np.nanmedian(wins[:, :, bl_mask], axis=2, keepdims=True)  # (N,K,1)
    return wins - bl

def _find_baseline_cam_npz(root: Path, port: str, depth: str, cam: int) -> Path | None:
    """
    Prefer curated → non-curated. Accept legacy naming without _camX as last fallback.
    """
    port = str(port).upper().strip()
    depth = str(depth).strip()
    # most specific first
    cands = [
        root / f"baseline_UA_{port}_Depth{depth}_cam{cam}_curated.npz",
        root / f"baseline_UA_{port}_Depth{depth}_cam{cam}.npz",
        root / f"baseline_UA_{port}_Depth{depth}.npz",  # legacy single-cam
    ]
    for p in cands:
        if p.exists():
            return p
    # last resort: newest file for that port/depth & cam
    globs = list(root.glob(f"baseline_UA_{port}_Depth{depth}*cam{cam}*.npz"))
    return max(globs, key=lambda p: p.stat().st_mtime) if globs else None

def _load_baseline_cam(npz_path):
    if not npz_path or not Path(npz_path).exists():
        return None
    with np.load(npz_path, allow_pickle=True) as z:
        wins = z["windows"].astype(float) if "windows" in z.files else np.zeros((0,0,0), float)
        t_ms = z["t_ms"].astype(float) if "t_ms" in z.files else np.arange(0.0)
        labels = [str(x) for x in (z["labels"].tolist() if "labels" in z.files else [f"trace_{i}" for i in range(wins.shape[1])])]
    if not wins.size or not t_ms.size:
        return None
    wins0 = _baseline_zero_trials(wins, t_ms, NORMALIZE_FIRST_MS)
    wins0_filt, v_wins, _ = rcp.butter_lowpass_pos_and_vel_3d(wins0, t_ms, cutoff_hz=10.0, order=3)
    pos_med = np.nanmedian(wins0_filt, axis=0)
    vel_med = np.nanmedian(v_wins, axis=0)
    labels_s = _simple_beh_labels(labels, KEYPOINTS_ORDER)
    return dict(pos=pos_med, vel=vel_med, t=t_ms, labels=labels_s)

def main_baselines():
    """
    Baseline figure styled like stacked_heatmaps_plus_behv (no probe):
      [Kinematics median lines] + [Intan heatmap] + [UA heatmap]
    """
    # ---- load ALL (concatenated) curated rate windows ----
    if not RATES_ALL_PATH.exists():
        print(f"[baseline] ALL rates file not found: {RATES_ALL_PATH}")
        return

    with np.load(RATES_ALL_PATH, allow_pickle=True) as z:
        nprw_win = z["nprw_rate_win"].astype(float) if "nprw_rate_win" in z.files else np.zeros((0,0,0), float)
        ua_win    = z["ua_rate_win"].astype(float)    if "ua_rate_win"    in z.files else np.zeros((0,0,0), float)
        nprw_t     = z["nprw_t_rel_ms"].astype(float)  if "nprw_t_rel_ms"  in z.files else np.arange(0.0)
        ua_t     = z["ua_t_rel_ms"].astype(float)  if "ua_t_rel_ms"  in z.files else np.arange(0.0)
        # keep for later lookups
        all_dir   = RATES_ALL_PATH.parent
        port_lbl  = str(z.get("port", "B"))
        depth_lbl = str(z.get("depth_mm", "43"))
    KIN0 = _find_baseline_cam_npz(BEHAV_ROOT, port_lbl, depth_lbl, cam=0)
    KIN1 = _find_baseline_cam_npz(BEHAV_ROOT, port_lbl, depth_lbl, cam=1)

    cam0 = _load_baseline_cam(KIN0) if KIN0 else None
    cam1 = _load_baseline_cam(KIN1) if KIN1 else None

    # legacy single-file fallback if neither cam file exists
    if not cam0 and not cam1:
        legacy = _find_baseline_cam_npz(BEHAV_ROOT, port_lbl, depth_lbl, cam=0)  # will also return legacy path
        if legacy and legacy.name.endswith(".npz") and "_cam" not in legacy.stem:
            solo = _load_baseline_cam(legacy)
            cam0 = solo
        
    n_ev_i = nprw_win.shape[0] if nprw_win.ndim == 3 else 0
    n_ev_u = ua_win.shape[0]    if ua_win.ndim    == 3 else 0
    if n_ev_i == 0 and n_ev_u == 0:
        print("[baseline] No events in ALL rates file — nothing to plot.")
        return

    # ---- baseline-zero per event/channel; then median across events → (n_ch, T) ----
    nprw_med = None; ua_med = None
    # ---- baseline-zero per event/channel; then median across events ----
    if n_ev_i:
        nprw_zero = rcp.baseline_zero_each_trial(nprw_win, nprw_t, normalize_first_ms=NORMALIZE_FIRST_MS)
        nprw_med = np.nanmedian(nprw_zero, axis=0)
    if n_ev_u:
        ua_zero = rcp.baseline_zero_each_trial(ua_win,    ua_t, normalize_first_ms=NORMALIZE_FIRST_MS)
        ua_med = np.nanmedian(ua_zero, axis=0)
        
    nprw_var = None; ua_var = None
    if n_ev_i:
        # you already have i_zero above
        nprw_var = np.nanvar(nprw_zero, axis=0)
    if n_ev_u:
        ua_var = np.nanvar(ua_zero, axis=0)
            
    # ---- load curated kinematics windows (median lines) ----
    beh_labels: list[str] = []

    # ---- UA region sorting (optional) ----
    ua_plot = ua_med
    ids_plot = None
    ua_groups_baseline = []
    
    if ua_med is not None:
        # --- UA row→electrode IDs (simple: newest __ALL.npz for this port/depth) ---
        ua_ids_1based = None
        try:
            pat_base = f"rates_from_curated__UA_{port_lbl}__Depth{depth_lbl}__*__ALL.npz"
            cands = sorted(all_dir.glob(pat_base))
            if cands:
                chosen = cands[-1]  # newest
                with np.load(chosen, allow_pickle=True) as zc:
                    # try direct arrays first
                    for k in ("ua_elec"):
                        if k in zc.files:
                            ua_ids_1based = np.asarray(zc[k], dtype=int).ravel()
                            break
                    # fall back: follow pointer to per-session UA file (if present)
                    if ua_ids_1based is None and "ua_rates" in zc.files:
                        ua_rates_path = Path(str(zc["ua_rates"]))
                        if ua_rates_path.exists():
                            with np.load(ua_rates_path, allow_pickle=True) as zr:
                                if "meta" in zr.files:
                                    meta_u = zr["meta"].item()
                                    for k in ("ua_elec"):
                                        if k in meta_u:
                                            ua_ids_1based = np.asarray(meta_u[k], dtype=int).ravel()
                                            break
                                if ua_ids_1based is None:
                                    for k in ("ua_elec"):
                                        if k in zr.files:
                                            ua_ids_1based = np.asarray(zr[k], dtype=int).ravel()
                                            break
            else:
                print(f"[baseline][warn] No UA curated ALL files found for port={port_lbl}, depth={depth_lbl}")
        except Exception as e:
            print(f"[baseline][warn] UA mapping discovery failed: {e}")

        # sanity check to avoid misalignment
        if ua_ids_1based is not None and ua_ids_1based.size != ua_med.shape[0]:
            print(f"[baseline][warn] ua_ids_1based len={ua_ids_1based.size} != UA rows={ua_med.shape[0]} — ignoring")
            ua_ids_1based = None

        # prepare for plotting/sorting
        ua_plot = ua_med
        ids_plot = ua_ids_1based
        if ua_ids_1based is not None:
            ids = np.asarray(ua_ids_1based, int)
            valid = ids > 0
            regs = np.array([_ua_region_code_from_elec(int(e)) for e in ids], int)
            order_valid = np.lexsort((ids[valid], regs[valid]))
            order = np.r_[np.where(valid)[0][order_valid], np.where(~valid)[0]]
            ua_plot = ua_med[order, :]
            ids_plot = ids[order]
            
    # ---- Impedance-based exclusion for BASELINE UA panel ----
    if EXCLUDE_UA_HIGH_Z and ua_plot is not None and ids_plot is not None:
        # load impedances the same way as your IR-aligned script
        def _try_load_imp(p: Path, tag: str):
            if not p:
                return
            if not p.exists():
                print(f"[warn] UA impedance file not found for port {tag}: {p}")
                return
            try:
                d = load_impedances_from_textedit_dump(p)
                imp_by_elec.update(d)
                print(f"[info] Parsed {len(d)} UA impedances from {p.name} (port {tag}).")
            except Exception as e:
                print(f"[warn] could not parse UA impedances from {p} (port {tag}): {e}")

        imp_by_elec: dict[int, float] = {}
        # Use the port from the rates file label; also try the other side
        _port = (port_lbl or "A").strip().upper()
        _try_load_imp(IMP_FILES.get(_port), _port)
        _other = "B" if _port == "A" else "A"
        _try_load_imp(IMP_FILES.get(_other), _other)

        # build keep mask: drop only rows whose impedance is known AND > threshold
        try:
            ids_arr = np.asarray(ids_plot)
            keep = np.ones(ua_plot.shape[0], dtype=bool)
            any_imp = False
            for r, eid in enumerate(ids_arr):
                if eid is None or not np.isfinite(eid):
                    continue
                z = imp_by_elec.get(int(eid))
                if z is not None and np.isfinite(z):
                    any_imp = True
                    if z > UA_IMP_MAX_KOHM:
                        keep[r] = False
            if any_imp:
                if keep.any():
                    ua_plot = ua_plot[keep, :]
                    ids_plot = ids_arr[keep]
                    print(f"[info] Baseline UA: kept {keep.sum()}/{keep.size} rows (≤ {UA_IMP_MAX_KOHM:g} kΩ).")

                    try:
                        ua_plot, ids_plot = _order_rows_by_region_then_peak(
                            ua_plot, ua_t, ids_plot,
                            pre_only=True, include_zero=True
                        )
                    except Exception as e:
                        print(f"[warn] Baseline UA: resort failed after mask: {e}")
                else:
                    print("[warn] Baseline UA: impedance mask would remove all rows; skipping mask.")
        except Exception as e:
            print(f"[warn] Baseline UA impedance mask failed: {e}")
        # ---- Split UA rows into region groups for baseline plotting ----
        if ua_plot is not None:
            ids_arr = np.asarray(ids_plot) if ids_plot is not None else np.full(ua_plot.shape[0], np.nan)
            regs = np.array(
                [_ua_region_code_from_elec(int(e)) if np.isfinite(e) else 1_000_000 for e in ids_arr],
                dtype=int
            )

            def _append_group(mask: np.ndarray, label: str):
                if mask.size and np.count_nonzero(mask):
                    ua_groups_baseline.append(
                        {"mat": ua_plot[mask, :], "ids": (ids_arr[mask] if ids_plot is not None else None), "label": label}
                    )

            # M1 group = inferior(2) + superior(3), PMd=1, SMA=0
            _append_group((regs == 2) | (regs == 3), "M1i+M1s")
            _append_group((regs == 1), "PMd")
            _append_group((regs == 0), "SMA")

    # If we haven't built groups yet, make a single-panel default:
    if ua_plot is not None and not ua_groups_baseline:
        ua_groups_baseline = [{"mat": ua_plot, "ids": ids_plot, "label": "UA (all)"}]

    # ---- Build UA groups for VARIANCE (robustly align to baseline rows) ----
    ua_groups_baseline_var = []
    if ua_var is not None:
        ua_plot_var  = ua_var
        ids_plot_var = None

        # If we have baseline-kept ids (ids_plot) and also know the original
        # per-row ids matching ua_var (ua_ids_1based), use them to align rows.
        try:
            # ids_plot: ids AFTER baseline sorting + impedance mask
            # ua_ids_1based: original ids BEFORE masking; must match ua_var rows
            if (ids_plot is not None and ua_ids_1based is not None and
                ua_var.shape[0] == np.asarray(ua_ids_1based).size):

                base_ids = np.asarray(ua_ids_1based, dtype=int).ravel()
                # map electrode id -> original row index in ua_var
                idx_map = {int(e): i for i, e in enumerate(base_ids) if e > 0 and np.isfinite(e)}
                # build row index list following ids_plot order; drop any unknowns
                row_idx = [idx_map.get(int(e), None) for e in np.asarray(ids_plot).ravel()
                        if e is not None and np.isfinite(e)]
                row_idx = [i for i in row_idx if i is not None]

                if row_idx and len(row_idx) == len(ids_plot):
                    ua_plot_var  = ua_var[row_idx, :]
                    ids_plot_var = np.asarray(ids_plot, dtype=int).ravel()
                else:
                    print("[warn] UA VAR: could not perfectly align to baseline-kept ids; using UA(all).")
            else:
                print("[warn] UA VAR: missing ids for alignment; using UA(all).")
        except Exception as e:
            print(f"[warn] UA VAR: alignment failed ({e}); using UA(all).")

        # If we successfully aligned, we can split by region; else show all
        if ids_plot_var is not None and ua_plot_var.shape[0] == ids_plot_var.size:
            ids_arr = ids_plot_var
            regs = np.array([_ua_region_code_from_elec(int(e)) if np.isfinite(e) else 1_000_000
                            for e in ids_arr], dtype=int)

            def _append_group_var(mask, label):
                if mask.size and np.count_nonzero(mask):
                    ua_groups_baseline_var.append({
                        "mat": ua_plot_var[mask, :],
                        "ids": ids_arr[mask],
                        "label": label
                    })

            _append_group_var((regs == 2) | (regs == 3), "M1i+M1s")
            _append_group_var((regs == 1), "PMd")
            _append_group_var((regs == 0), "SMA")
            if not ua_groups_baseline_var:
                ua_groups_baseline_var = [{"mat": ua_plot_var, "ids": ids_arr, "label": "UA (all)"}]
        else:
            ua_groups_baseline_var = [{"mat": ua_plot_var, "ids": None, "label": "UA (all)"}]

    # ---- helper that renders one combined figure (behavior + Intan + UA) ----
    def _render_and_save_baseline(
        beh_time0, beh_pos0, beh_vel0, *,
        title_beh0,
        out_dir,
        beh_time1=None, beh_pos1=None, beh_vel1=None,
        title_beh1=None,
        beh_labels: list[str] = None,
        ua_groups: list[dict] = None,
        nprw_mat=None,
        nprw_title:str="",
        nprw_cb_label:str="Δ FR (Hz)",
        ua_vrange_override:dict|None=None,  # optional per-group vrange
        nprw_vmin: float | None = None,
        nprw_vmax: float | None = None,
        suffix:str=""                   # filename suffix
    ):
        import matplotlib.gridspec as gridspec
        beh_labels = beh_labels or []
        ua_groups = ua_groups or []

        have_pos0 = isinstance(beh_pos0, np.ndarray) and beh_pos0.ndim == 2 and beh_pos0.size > 0
        have_vel0 = isinstance(beh_vel0, np.ndarray) and beh_vel0.ndim == 2 and beh_vel0.size > 0
        have_pos1 = isinstance(beh_pos1, np.ndarray) and beh_pos1.ndim == 2 and beh_pos1.size > 0
        have_vel1 = isinstance(beh_vel1, np.ndarray) and beh_vel1.ndim == 2 and beh_vel1.size > 0

        # ---- kinematics row order: cam0_pos, cam1_pos, cam0_vel, cam1_vel ----
        ratios = []
        row_kinds = []
        def _append_beh(kind):
            ratios.append(BEH_RATIO); row_kinds.append(("beh", kind))

        if have_pos0: _append_beh("cam0_pos")
        if have_pos1: _append_beh("cam1_pos")
        if have_vel0: _append_beh("cam0_vel")
        if have_vel1: _append_beh("cam1_vel")

        # Spacer before Intan if any behavior rows exist
        has_nprw = (nprw_mat is not None)
        if has_nprw and any(k[0] == "beh" for k in row_kinds):
            ratios.append(GAP_BEH_NPRW); row_kinds.append(("gap", None))

        # Intan
        if has_nprw:
            nprw_ratio = max(MIN_HEATMAP_RATIO, CH_RATIO_PER_ROW * nprw_mat.shape[0]) * float(NPRW_SCALE)
            ratios.append(nprw_ratio); row_kinds.append(("nprw", None))

        # UA groups
        if ua_groups:
            for group in ua_groups:
                mat_group = group["mat"]
                label_group = group.get("label", "")
                rows = (mat_group.shape[0] if mat_group is not None else 0)
                base = max(MIN_HEATMAP_RATIO, CH_RATIO_PER_ROW * rows)
                scale = 0.5 if (("pmd" in label_group.lower()) or ("sma" in label_group.lower())) else 1.0
                ratios.append(base * scale * UA_COMPACT_FACTOR)
                row_kinds.append(("ua", label_group))

        nrows = len(ratios)
        if nrows == 0:
            print("[baseline] nothing to plot.")
            return

        fig = plt.figure(
            figsize=(FIG_WIDTH_IN, HEIGHT_PER_RATIO_IN * sum(ratios)),
            constrained_layout=False  # or True if you prefer auto spacing
        )

        gs = gridspec.GridSpec(
            nrows=nrows, ncols=2, figure=fig,
            width_ratios=[1.0, 0.02], height_ratios=ratios, hspace=0.18, wspace=0.05
        )

        # Helper with a legend toggle
        def _plot_lines(ax, t, lines, labels, title, ylabel, show_legend: bool):
            if not (isinstance(lines, np.ndarray) and lines.ndim == 2 and lines.size > 0):
                ax.axis("off"); return
            K = lines.shape[0]
            for i in range(K):
                lab = labels[i] if i < len(labels) else f"trace_{i+1}"
                ax.plot(t, lines[i], lw=1.2, alpha=0.95, label=lab)
            ax.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2, ls="--")
            _ylabel_horizontal(ax, ylabel)
            if title:  # only set when non-empty
                ax.set_title(title)
            if show_legend and K:
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                        frameon=False, fontsize=9, ncols=1, borderaxespad=0.0)
            ax.grid(alpha=0.15, linestyle=":")


        row = 0
        ua_axes = []
        events_label = f"{_rates_selection_tag.lower()} events"

        # We'll count behavior panels as we render; show legend only on the 3rd one.
        beh_plot_count = 0
        total_beh_rows = sum(1 for k in row_kinds if k[0] == "beh")

        for kind, subkind in row_kinds:
            if kind == "beh":
                beh_plot_count += 1
                show_legend = (beh_plot_count == 3)  # legend only on the third kinematics row

                # Determine camera/time and which data to plot
                if subkind in ("cam0_pos", "cam0_vel"):
                    cam_label = "Cam-0"
                    t_use     = beh_time0
                else:
                    cam_label = "Cam-1"
                    t_use     = beh_time1

                if subkind.endswith("_pos"):
                    metric_label = "\n Position Δ (a.u.)"
                    y_label      = f"{cam_label} {metric_label}"
                    lines_use    = beh_pos0 if subkind == "cam0_pos" else beh_pos1
                else:
                    metric_label = "\n Velocity (a.u./ms)"
                    y_label      = f"{cam_label} {metric_label}"
                    lines_use    = beh_vel0 if subkind == "cam0_vel" else beh_vel1

                # Only the FIRST kinematics row gets a title; others blank
                if beh_plot_count == 1:
                    if subkind.startswith("cam0"):
                        base_title = (title_beh0 or "Cam-0 Kinematics")
                    else:
                        base_title = (title_beh1 or "Cam-1 Kinematics")
                    title_txt = base_title  # no “— Position/Velocity” suffix; y-axis already says that
                else:
                    title_txt = ""

                ax = fig.add_subplot(gs[row, 0])
                _plot_lines(ax, t_use, lines_use, beh_labels, title_txt, y_label, show_legend)

                # Hide x tick labels unless this is the LAST behavior row
                if beh_plot_count < total_beh_rows:
                    ax.set_xlabel(""); ax.tick_params(axis="x", labelbottom=False)
                ax._is_time_axis = True
                row += 1
                continue

            if kind == "gap":
                ax_gap = fig.add_subplot(gs[row, 0])
                ax_gap.axis("off")
                row += 1
                continue

            if kind == "nprw":
                # NPRW row
                ax_nprw     = fig.add_subplot(gs[row, 0])
                ax_nprw_cax = fig.add_subplot(gs[row, 1])

                vmin_nprw = VMIN_NPRW_BASELINE if nprw_vmin is None else float(nprw_vmin)
                vmax_nprw = VMAX_NPRW_BASELINE if nprw_vmax is None else float(nprw_vmax)
                im0 = ax_nprw.imshow(
                    nprw_mat, aspect="auto", cmap=COLORMAP, origin="lower",
                    extent=[nprw_t[0], nprw_t[-1], 0, nprw_mat.shape[0]],
                    vmin=vmin_nprw, vmax=vmax_nprw
                )
                
                ax_nprw.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2, ls="--")
                title_txt = (nprw_title or
                        f"No stim. Neural Activity (median Δ across {n_ev_i} {events_label})")
                ax_nprw.set_title(title_txt)
                ax_nprw.set_xlabel(""); ax_nprw.tick_params(axis="x", labelbottom=False)
                _ylabel_horizontal(ax_nprw, f"IP {nprw_mat.shape[0]} chs")

                ax_nprw_cax.set_xticks([]); ax_nprw_cax.set_yticks([])
                ax_nprw._is_time_axis = True

                # create the colorbar in the dedicated cax
                cb0 = fig.colorbar(im0, cax=ax_nprw_cax)
                # cb0 = fig.colorbar(im0, cax=ax_i_cax, ticks = ticker.MultipleLocator(20))
                cb0.solids.set_edgecolor('face')
                cb0.ax.tick_params(width=1.2, labelsize=9)
                cb0.set_label(nprw_cb_label)
        
                row += 1
                continue

            if kind == "ua":
                label_group = subkind
                group = ua_groups[len(ua_axes)]
                mat_group = group["mat"]
                ax_ua     = fig.add_subplot(gs[row, 0])
                ax_ua_cax = fig.add_subplot(gs[row, 1])
                # pick vrange per group
                vmin_ua_group, vmax_ua_group = VMIN_UA_BASELINE, VMAX_UA_BASELINE
                if ua_vrange_override and label_group in ua_vrange_override:
                    vmin_ua_group, vmax_ua_group = ua_vrange_override[label_group]

                im1 = ax_ua.imshow(
                    mat_group, aspect="auto", cmap=COLORMAP, origin="lower",
                    extent=[ua_t[0], ua_t[-1], 0, mat_group.shape[0]],
                    vmin=vmin_ua_group, vmax=vmax_ua_group
                )

                ax_ua.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2, ls="--")
                _ylabel_horizontal(ax_ua, f"{label_group} {mat_group.shape[0]} chs")
                ax_ua.set_yticks([]); ax_ua.tick_params(left=False, labelleft=False)
                ax_ua._is_time_axis = True
                
                if (len(ua_axes) == len(ua_groups) - 1):
                    ax_ua.set_xlabel("Time (ms) rel. stim")
                else:
                    ax_ua.set_xlabel(""); ax_ua.tick_params(axis="x", labelbottom=False)

                cb1 = fig.colorbar(im1, cax=ax_ua_cax)
                try: cb1.solids.set_edgecolor('face')
                except: pass
                cb1.ax.tick_params(width=1.2, labelsize=9)
                cb1.outline.set_linewidth(1.0)
                # cb1.set_label("Δ FR (Hz)")

                ua_axes.append(ax_ua)
                row += 1
                continue

        # Sync x-lims
        for ax in [a for a in fig.axes if isinstance(a, plt.Axes)]:
            if getattr(ax, "_is_time_axis", False):
                ax.set_xlim(*WIN_MS)
        fig.suptitle("Baseline", fontsize=13, fontweight="bold", y=0.995, va="top")
        out_path = out_dir / f"Baseline_UA_{port_lbl}__Depth{depth_lbl}__{_rates_selection_tag}{suffix}.png"
        fig.subplots_adjust(top=0.96)           # pull axes up toward the top
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        print(f"[baseline] Wrote {out_path}")

    # Shared labels (for legends)
    if cam0 and cam0["labels"]:
        beh_labels = cam0["labels"]
    elif cam1 and cam1["labels"]:
        beh_labels = cam1["labels"]
    else:
        beh_labels = []

    # POSITION + VELOCITY (overlayed) — SINGLE FIGURE
    pos0 = _smooth_lines_gaussian_filtfilt(cam0["pos"], cam0["t"], GAUSS_SMOOTH_MS) if cam0 else None
    pos1 = _smooth_lines_gaussian_filtfilt(cam1["pos"], cam1["t"], GAUSS_SMOOTH_MS) if cam1 else None
    if pos0 is not None:
        pos0 = _apply_relative_offsets_by_ref(pos0, beh_labels, KEYPOINTS_ORDER)
    if pos1 is not None:
        pos1 = _apply_relative_offsets_by_ref(pos1, beh_labels, KEYPOINTS_ORDER)

    vel0 = _smooth_lines_gaussian_filtfilt(cam0["vel"], cam0["t"], GAUSS_SMOOTH_MS) if cam0 else None
    vel1 = _smooth_lines_gaussian_filtfilt(cam1["vel"], cam1["t"], GAUSS_SMOOTH_MS) if cam1 else None

    if (pos0 is not None) or (pos1 is not None):
        _render_and_save_baseline(
            cam0["t"] if cam0 else (cam1["t"] if cam1 else None),
            pos0, vel0,
            title_beh0=(f"Kinematics / Referenced to first {int(NORMALIZE_FIRST_MS)} ms"
                        + (f" • Gaussian (filtfilt) σ={GAUSS_SMOOTH_MS:g} ms" if GAUSS_SMOOTH_MS > 0 else "")) if cam0 else "",
            out_dir=FIG.peri_posvel_median,
            beh_time1=(cam1["t"] if cam1 else None),
            beh_pos1=pos1, beh_vel1=vel1,
            title_beh1=(f""),
            nprw_mat=nprw_med,
            ua_vrange_override={"M1i+M1s": (VMIN_UA_BASELINE, VMAX_UA_BASELINE), "PMd": (VMIN_UA_BASELINE, VMAX_UA_BASELINE), "SMA": (VMIN_SMA_BASELINE, VMAX_SMA_BASELINE)},
            nprw_vmin=VMIN_NPRW_BASELINE, nprw_vmax=VMAX_NPRW_BASELINE,
            beh_labels=beh_labels,
            ua_groups=ua_groups_baseline
        )
    else:
        _render_and_save_baseline(
            None, None, None, title_beh0="",
            out_dir=FIG.peri_posvel_median,
            beh_labels=[], ua_groups=ua_groups_baseline
    )
    if (pos0 is not None) or (pos1 is not None) or (nprw_var is not None) or (ua_groups_baseline_var):
        _render_and_save_baseline(
            cam0["t"] if cam0 else (cam1["t"] if cam1 else None),
            pos0, vel0,
            title_beh0=(f"Kinematics / Referenced to first {int(NORMALIZE_FIRST_MS)} ms"
                        + (f" • Gaussian (filtfilt) σ={GAUSS_SMOOTH_MS:g} ms" if GAUSS_SMOOTH_MS > 0 else "")) if cam0 else "",
            out_dir=FIG.peri_posvel_var,
            beh_time1=(cam1["t"] if cam1 else None),
            beh_pos1=pos1, beh_vel1=vel1,
            title_beh1="",
            beh_labels=beh_labels,
            ua_groups=ua_groups_baseline_var,
            nprw_mat=nprw_var,
            nprw_title=f"No stim. Neural Activity VAR (across {n_ev_i} {_rates_selection_tag.lower()} events)",
            nprw_cb_label="Var(Δ FR) (Hz²)",
            ua_vrange_override={"M1i+M1s": (0.0, 10000.0), "PMd": (0.0, 10000.0), "SMA": (0.0, 5000.0)},
            nprw_vmin=0.0, nprw_vmax=20000.0,
            suffix="__VAR",
        )

def _strip_cam_prefix(n: str) -> str:
    # Turn "cam0_wrist_x" -> "wrist_x", "cam1_middle_finger_tip_y" -> "middle_finger_tip_y"
    n = str(n)
    if n.startswith("cam0_"):
        return n[len("cam0_"):]
    if n.startswith("cam1_"):
        return n[len("cam1_"):]
    return n

def main():
    # --- geometry / probe as you already have ---
    mat_probe = loadmat(Path(GEOM_PATH))
    nprw_geom = {}
    nprw_geom["x"] = mat_probe["xcoords"].ravel()
    nprw_geom["y"] = mat_probe["ycoords"].ravel()
    assert nprw_geom["x"].size == nprw_geom["y"].size
    if "chanMap0ind" in mat_probe:
        nprw_probe_mapping = nprw_geom["device_index_0based"] = mat_probe["chanMap0ind"].ravel()
    else:
        raise ValueError("No 0-based chanmap in .mat geometry file.")
    if nprw_probe_mapping.size != nprw_geom["x"].size:
        raise ValueError("device_index_0based length != #contacts")

    nprw_probe = Probe(ndim=2)
    nprw_probe.set_contacts(
        positions=np.c_[nprw_geom["x"], nprw_geom["y"]],
        shapes="square", shape_params={"width": 12.0}
    )
    nprw_probe.set_device_channel_indices(nprw_probe_mapping)
    locs = nprw_probe.contact_positions.astype(float)

    files = sorted(ALIGNED_ROOT.glob("aligned__*.npz"))
    if not files:
        raise SystemExit(f"[error] No combined aligned NPZs found at {ALIGNED_ROOT}")

    for k, file in enumerate(files):
        # ---- load combined npz (for NPRW/UA + stim) ----
        aligned_npz = np.load(file, allow_pickle=True)
        stim_ms_abs = aligned_npz["stim_ms"]
        # alignment meta (JSON)
        meta = json.loads(aligned_npz["align_meta"].item()) if "align_meta" in aligned_npz.files else {}
        
        sess   = meta.get("session", file.stem)
        br_idx = int(meta.get("br_idx", -1))

        peribase = f"peristim__{sess}__BR_{br_idx:03d}"
        peri_stim_npz_loc  = PERI_ROOT / f"{peribase}.npz"
        if not peri_stim_npz_loc.exists():
            print(f"[warn] missing peri-stim npz for {sess}, BR {br_idx:03d}: {peri_stim_npz_loc.name}")
            continue

        with np.load(peri_stim_npz_loc, allow_pickle=True) as peri_stim_npz:
            beh_rel_t          = peri_stim_npz["beh_rel_t"]
            beh_rel_t_v        = peri_stim_npz["beh_rel_t_v"]
            cam0_pos_med       = peri_stim_npz["cam0_pos_med"]
            cam1_pos_med       = peri_stim_npz["cam1_pos_med"]
            cam0_vel_med       = peri_stim_npz["cam0_vel_med"]
            cam1_vel_med       = peri_stim_npz["cam1_vel_med"]
            cam0_names         = peri_stim_npz["cam0_names"].astype(str).tolist()
            cam1_names         = peri_stim_npz["cam1_names"].astype(str).tolist()
            NPRW_med           = peri_stim_npz["NPRW_med"]
            UA_med             = peri_stim_npz["UA_med"]
            NPRW_var           = peri_stim_npz["NPRW_var"]
            UA_var             = peri_stim_npz["UA_var"]
            rel_time_ms_i      = peri_stim_npz["nprw_rel_t"]
            ua_rel_time_ms     = peri_stim_npz["ua_rel_t"]
            nprw_rel_time_ms_var = peri_stim_npz["nprw_rel_t_var"]
            ua_rel_time_ms_var   = peri_stim_npz["ua_rel_t_var"]
            ua_ids_1based      = peri_stim_npz["ua_ids_1based"]
            overall_title      = str(peri_stim_npz["overall_title"].item() if peri_stim_npz["overall_title"].shape == () else peri_stim_npz["overall_title"])
            n_nprw             = int(peri_stim_npz["n_nprw"])
            n_nprw_var         = int(peri_stim_npz["n_nprw_var"])

        # build behavior labels like before
        if len(cam0_names) > 0:
            beh_labels = _simple_beh_labels([_strip_cam_prefix(n) for n in cam0_names], KEYPOINTS_ORDER)
        elif len(cam1_names) > 0:
            beh_labels = _simple_beh_labels([_strip_cam_prefix(n) for n in cam1_names], KEYPOINTS_ORDER)
        else:
            beh_labels = []

        beh_time_for_both = beh_rel_t

        title_kinematics = (
            f"Kinematics / Referenced to first {int(NORMALIZE_FIRST_MS)} ms"
        )

        # stim site indices (unchanged)
        stim_npz = NPRW_AUX_DATA / f"{sess}_Intan_streams" / "stim_stream.npz"
        stim_locs = None
        if stim_npz.exists():
            try:
                stim_locs = rcp.detect_stim_channels_from_npz(stim_npz, eps=1e-12, min_edges=1)
            except Exception as e:
                print(f"[warn] stim-site detection failed for {sess}, Condition {br_idx}: {e}")

        # ---- median figure ----
        out_dir_median = FIG.peri_posvel_median / f"Cond_{br_idx}__peri_stim__posvel.png"
        title_NA = f"Neural Activity (median Δ across {n_nprw} events)"

        rcp.stacked_heatmaps_plus_behv(
            NPRW_med, UA_med,
            rel_time_ms_i if NPRW_med.size else None,
            ua_rel_time_ms if UA_med.size else None,
            out_dir_median, title_kinematics, title_NA,
            cmap=COLORMAP,
            vmin_nprw=VMIN_NPRW, vmax_nprw=VMAX_NPRW,
            probe=nprw_probe, probe_locs=locs, stim_idx=stim_locs,
            probe_title="NPRW probe (stim sites highlighted)",
            ua_ids_1based=ua_ids_1based,
            ua_sort="region_then_elec",
            beh_rel_time=beh_time_for_both,
            beh_cam0_lines=cam0_pos_med,
            beh_cam1_lines=cam1_pos_med,
            beh_cam0_vel_lines=cam0_vel_med,
            beh_cam1_vel_lines=cam1_vel_med,
            beh_labels=beh_labels,
            title_cam1="",
            title_cam0_vel="",
            title_cam1_vel="",
            sess=sess,
            overall_title=overall_title,
        )

        # ---- variance figure ----
        out_dir_var = FIG.peri_posvel_var / f"Cond_{br_idx}__peri_stim__posvel_VAR.png"
        title_NA_var = f"Neural Activity VAR (variance of Δ across {n_nprw_var} events)"

        rcp.stacked_heatmaps_plus_behv(
            NPRW_var, UA_var,
            nprw_rel_time_ms_var if NPRW_var.size else None,
            ua_rel_time_ms_var   if UA_var.size else None,
            out_dir_var, title_kinematics, title_NA_var,
            cmap=COLORMAP,
            vmin_nprw=0.0, vmax_nprw=40000.0,
            vmin_ua={"M1i+M1s": 0.0, "PMd": 0.0, "SMA": 0.0},
            vmax_ua={"M1i+M1s": 10000.0, "PMd": 10000.0, "SMA": 10000.0},
            probe=nprw_probe, probe_locs=locs, stim_idx=stim_locs,
            probe_title="NPRW probe (stim sites highlighted)",
            ua_ids_1based=ua_ids_1based,
            ua_sort="none",
            beh_rel_time=beh_time_for_both,
            beh_cam0_lines=cam0_pos_med,
            beh_cam1_lines=cam1_pos_med,
            beh_cam0_vel_lines=cam0_vel_med,
            beh_cam1_vel_lines=cam1_vel_med,
            beh_labels=beh_labels,
            title_cam1="",
            title_cam0_vel="",
            title_cam1_vel="",
            sess=sess,
            overall_title=overall_title,
        )


if __name__ == "__main__":
    # plot the aligned baseline traces
    # main_baselines()
    
    # plot stim trials
    main()