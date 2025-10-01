import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Optional SciPy for .mat + peak finding
try:
    from scipy.io import loadmat
    from scipy.signal import find_peaks
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ------------------------------ small utilities ------------------------------

def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-12)

def _load_meta(npz) -> dict:
    if "meta" not in npz.files:
        return {}
    m = npz["meta"]
    if isinstance(m, (bytes, bytearray)):
        s = m.decode("utf-8", "ignore")
    elif isinstance(m, np.ndarray) and m.ndim == 0:
        m = m.item()
        if isinstance(m, (bytes, bytearray)):
            s = m.decode("utf-8", "ignore")
        elif isinstance(m, str):
            s = m
        elif isinstance(m, dict):
            return m
        else:
            return {}
    elif isinstance(m, str):
        s = m
    elif isinstance(m, dict):
        return m
    else:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}

def _ensure_channels_first(arr2d: np.ndarray) -> np.ndarray:
    """Return (channels, samples)."""
    n0, n1 = arr2d.shape
    if n0 <= 512 and n1 > n0:
        return arr2d.T
    if n1 <= 512 and n0 > n1:
        return arr2d
    return arr2d.T if n0 <= n1 else arr2d

# ----------------------- data loading (Intan & UA bundles) --------------------

def load_intan_adc_2ch(npz_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (adc_triangle, adc_lock, fs) from Intan ADC bundle (.npz).
    Expects either chunk_* arrays with shape (samples, >=2 channels) or
    a single 2D array with >=2 channels.
    """
    z = np.load(npz_path, allow_pickle=True, mmap_mode="r")
    fs = float(_load_meta(z).get("fs_hz", 30000.0))

    chunk_keys = sorted([k for k in z.files if k.startswith("chunk_")])
    ch0_parts, ch1_parts = [], []
    if chunk_keys:
        for k in chunk_keys:
            a = z[k]
            if a.ndim == 1:
                ch0_parts.append(a.astype(np.float64))
            elif a.ndim == 2:
                if a.shape[1] < 2:
                    raise RuntimeError(f"Chunk '{k}' has < 2 channels.")
                ch0_parts.append(a[:, 0].astype(np.float64))
                ch1_parts.append(a[:, 1].astype(np.float64))
        if not ch0_parts or not ch1_parts:
            raise RuntimeError("Missing ch0 or ch1 in ADC chunks.")
        adc_triangle = np.concatenate(ch0_parts)
        adc_lock     = np.concatenate(ch1_parts)
        return adc_triangle, adc_lock, fs

    # Fallback: a single 2D array
    for k in z.files:
        a = z[k]
        if hasattr(a, "ndim") and a.ndim == 2:
            chxT = _ensure_channels_first(a)  # (ch, T)
            assert chxT.shape[0] >= 2, "ADC bundle must have at least 2 channels"
            return chxT[0].astype(np.float64), chxT[1].astype(np.float64), fs

    raise RuntimeError("No chunk_* arrays or 2D array found in ADC NPZ.")

def load_ua_intan_sync(npz_path: Path) -> tuple[np.ndarray, float]:
    """Returns (ua_sync, fs) from UA bundle (key: 'intan_sync', rate from aux_fs/meta)."""
    u = np.load(npz_path, allow_pickle=True, mmap_mode="r")
    if "intan_sync" not in u.files:
        raise KeyError("UA bundle is missing key 'intan_sync'.")
    ua_sync = np.asarray(u["intan_sync"], float).ravel()
    fs = 30000.0
    if "aux_fs" in u.files:
        aux = u["aux_fs"]
        fs = float(aux.item() if isinstance(aux, np.ndarray) else aux)
    else:
        fs = float(_load_meta(u).get("fs_hz", fs))
    return ua_sync, fs

def load_template(template_mat_path: Path) -> np.ndarray:
    if not _HAS_SCIPY:
        raise ImportError("scipy is required to load template.mat (install scipy).")
    m = loadmat(str(template_mat_path))
    if "template" not in m:
        raise KeyError("'template' not found in template.mat")
    return np.asarray(m["template"], float).squeeze()

# ---------------------------- core computations ------------------------------

def xcorr_normalized(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full xcorr of z-scored x vs y; returns (corr, lags)."""
    xz, yz = _z(x), _z(y)
    c = np.correlate(xz, yz, mode="full")
    c /= (c.max() if c.size else 1.0)
    lags = np.arange(-(y.size - 1), x.size)
    return c, lags

def peak_lags(corr: np.ndarray, lags: np.ndarray, height: float = 0.95) -> np.ndarray:
    if _HAS_SCIPY:
        idx, _ = find_peaks(corr, height=height)
        return lags[idx].astype(int)
    mid = np.arange(1, corr.size - 1)
    mask = (corr[mid-1] < corr[mid]) & (corr[mid] >= corr[mid+1]) & (corr[mid] >= height)
    return lags[mid[mask]].astype(int)

def best_shift_around(a: np.ndarray, b: np.ndarray, center: int, search: int = 500, win: int = 30000) -> tuple[int, float]:
    """
    Find integer shift n (a[t] vs b[t+n]) maximizing correlation
    in a ±win window centered at 'center'. Search n in [-search, +search].
    """
    nbest, rbest = 0, -np.inf
    i0 = max(0, center - win)
    i1 = min(min(len(a), len(b)), center + win + 1)
    aw = a[i0:i1]
    for n in range(-search, search + 1):
        j0, j1 = i0 + n, i1 + n
        if j0 < 0 or j1 > len(b):
            continue
        bw = b[j0:j1]
        r = float(np.dot(_z(aw), _z(bw)) / len(aw))
        if r > rbest:
            rbest, nbest = r, n
    return nbest, rbest

# ------------------------------ plotting helpers -----------------------------

def save_plot_lock_template_overlay(lock, template, center, fs, pad, out_path):
    """
    Plot z-scored lock with a z-scored template overlaid, centered so that
    the template starts at sample 'center' in 'lock'. The template is clipped
    to the plotting window to avoid broadcasting errors.
    """
    # Window on lock: [i0, i1)
    i0 = max(0, center - pad)
    i1 = min(len(lock), center + pad)
    lock_seg = lock[i0:i1]
    t = (np.arange(i0, i1) - center) / fs

    # Prepare overlay buffer (same length as lock window)
    templ_overlay = np.full_like(lock_seg, np.nan, dtype=float)

    # Template nominal placement: [templ_start, templ_end)
    templ_start = center
    templ_end = center + len(template)

    # Compute actual overlap with the lock window
    ov0 = max(i0, templ_start)    # first overlapping sample (absolute index)
    ov1 = min(i1, templ_end)      # first non-overlapping sample (absolute index)

    if ov1 > ov0:
        # Convert to window-relative indices
        w0 = ov0 - i0
        w1 = ov1 - i0

        # Convert to template-relative indices
        t0 = ov0 - templ_start
        t1 = t0 + (w1 - w0)

        # Assign only the overlapping chunk
        templ_overlay[w0:w1] = _z(template[t0:t1])

    # Plot
    plt.figure(figsize=(12, 3.6))
    plt.plot(t, _z(lock_seg), label="ADC ch1 lock (z)", linewidth=1)
    plt.plot(t, templ_overlay, label="template (z)", linewidth=1)
    plt.axvline(0, linestyle="--", color="k", linewidth=1)
    plt.xlabel("Time from first loc (s)")
    plt.ylabel("z-score")
    plt.title(f"Lock vs template overlay (±{pad} samples)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_plot_shift_curve(a, b, center, fs, search, win, out_path):
    i0 = max(0, center - win)
    i1 = min(min(len(a), len(b)), center + win + 1)
    aw = a[i0:i1]
    shifts, rs = [], []
    for n in range(-search, search + 1):
        j0, j1 = i0 + n, i1 + n
        if j0 < 0 or j1 > len(b):
            continue
        bw = b[j0:j1]
        r = float(np.dot(_z(aw), _z(bw)) / len(aw))
        shifts.append(n); rs.append(r)
    shifts = np.array(shifts); rs = np.array(rs)
    nbest = shifts[np.argmax(rs)]

    plt.figure(figsize=(12, 3.2))
    plt.plot(shifts, rs, linewidth=1)
    plt.axvline(nbest, color="r", linestyle="--", linewidth=1, label=f"best n={int(nbest)} (~{nbest/fs:.4f}s)")
    plt.xlabel("Shift n (samples)  [a[t] vs b[t+n]]")
    plt.ylabel("corr")
    plt.title("Triangle vs intan_sync: correlation by shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()

def save_plot_triangle_overlay(adc_triangle, ua_sync, center, fs, n_best, pad, out_path):
    # window on ADC triangle
    i0 = max(0, center - pad)
    i1 = min(len(adc_triangle), center + pad)
    # apply shift to UA and crop to overlap
    j0 = max(0, i0 + n_best)
    j1 = min(len(ua_sync), i1 + n_best)
    span = min(i1 - i0, j1 - j0)
    if span <= 0:
        return False
    t = (np.arange(span) + i0 - center) / fs
    x = _z(adc_triangle[i0:i0+span])
    y = _z(ua_sync[j0:j0+span])

    plt.figure(figsize=(12, 3.6))
    # plt.plot(t, x, label="Intan ADC ch0 triangle (z)", linewidth=1)
    plt.plot(t, y, label=f"UA intan_sync (z), n={n_best}", linewidth=1)

    plt.axvline(0, linestyle="--", color="k", linewidth=1)
    plt.xlabel("Time from first loc (s)")
    plt.ylabel("z-score")
    plt.title(f"Triangle vs intan_sync overlay (±{pad} samples)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return True

# ----------------------------- main pipeline --------------------------------
def _range_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    rng = x.max() - x.min()
    return (x - x.mean()) / (rng + 1e-12)
    
def best_shift_rms_range_on_intan(a: np.ndarray, b: np.ndarray, center: int,
                                  search: int = 500, win: int = 30000) -> tuple[int, float]:
    """
    MATLAB-style: for shifts n in [-search, search], compare
    UA window (b) to range-normalized Intan window (a), both cut
    from a ±win window centered at `center`. Return (n_best, rms_min).
    
    a := Intan triangle
    b := UA intan_sync (raw)
    """
    i0 = max(0, center - win)
    i1 = min(min(len(a), len(b)), center + win + 1)

    aw = a[i0:i1]  # Intan window
    L = aw.size
    if L < 3:
        return 0, float("nan")

    best_n, best_err = 0, float("inf")
    for n in range(-search, search + 1):
        j0, j1 = i0 + n, i1 + n
        if j0 < 0 or j1 > len(b):
            continue
        # Intan slice (range-normalized) vs UA slice (raw)
        tri = _range_norm(aw)
        ua  = b[j0:j1].astype(float)
        err = float(np.sqrt(np.mean((ua - tri) ** 2)))  # RMS
        if err < best_err:
            best_err, best_n = err, n
    return best_n, best_err

def save_plot_shift_curve_err(a, b, center, fs, search, win, out_path):
    i0 = max(0, center - win)
    i1 = min(min(len(a), len(b)), center + win + 1)
    aw = a[i0:i1]
    shifts, errs = [], []
    for n in range(-search, search + 1):
        j0, j1 = i0 + n, i1 + n
        if j0 < 0 or j1 > len(b):
            continue
        tri = _range_norm(aw)
        ua  = b[j0:j1].astype(float)
        err = float(np.sqrt(np.mean((ua - tri) ** 2)))
        shifts.append(n); errs.append(err)
    shifts = np.array(shifts); errs = np.array(errs)
    nbest = int(shifts[np.argmin(errs)])

    plt.figure(figsize=(12, 3.2))
    plt.plot(shifts, errs, linewidth=1)
    plt.axvline(nbest, color="r", linestyle="--", linewidth=1,
                label=f"best n={nbest} (~{nbest/fs:.4f}s)")
    plt.xlabel("Shift n (samples)  [Intan[t] vs UA[t+n]]")
    plt.ylabel("RMS error (UA − range-norm Intan)")
    plt.title("Triangle vs UA sync: RMS error by shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
def save_plot_triangle_overlay_matlab(adc_triangle, ua_sync, center, fs, n_best, pad, out_path):
    i0 = max(0, center - pad)
    i1 = min(len(adc_triangle), center + pad)

    j0 = i0 + n_best
    j1 = i1 + n_best
    if j0 < 0 or j1 > len(ua_sync):
        return False

    tri = _range_norm(adc_triangle[i0:i1])
    ua  = ua_sync[j0:j1].astype(float)

    t = (np.arange(i1 - i0) + i0 - center) / fs
    plt.figure(figsize=(12, 3.6))
    plt.plot(t, tri, label="Intan triangle (range-norm)", linewidth=1)
    plt.plot(t, ua,  label=f"UA intan_sync (raw), n={n_best}", linewidth=1, alpha=0.9)
    plt.axvline(0, linestyle="--", color="k", linewidth=1)
    plt.xlabel("Time from first loc (s)")
    plt.ylabel("arb. units")
    plt.title(f"Triangle vs UA overlay (MATLAB-style)  ±{pad} samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return True

def align_and_save_plot(
    adc_npz_path: Path,
    ua_npz_path: Path,
    template_mat_path: Path,
    out_dir: Path,
    search_samples: int = 500,
    window_half_width: int = 30000,
    align_to_ua: bool = False,   # NEW: default off
) -> tuple[np.ndarray, np.ndarray, int | None, dict]:
    """
    Returns:
      session_trigger (2 x T), locs (np.ndarray), best_shift (int or None),
      plot_paths (dict of Path)

    Behavior:
      - Always performs template matching on the lock channel to get `locs`.
      - If `align_to_ua` is False (default), it STOPs here (best_shift=None).
      - If `align_to_ua` is True, it proceeds to UA alignment/plots.
    """
    plot_paths = {}

    # 1) Intan ADC → session_trigger
    adc_triangle, adc_lock, fs_adc = load_intan_adc_2ch(adc_npz_path)
    session_trigger = np.vstack([adc_triangle, adc_lock])  # [0]=triangle, [1]=lock

    # 2) Template matching on the lock channel (find block starts)
    template = load_template(template_mat_path)
    corr, lags = xcorr_normalized(adc_lock, template)
    locs = peak_lags(corr, lags, height=0.95)
    print(f"[locs] peaks ≥0.95: {locs.size} | first 10: {locs[:10].tolist() if locs.size else []}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # Save: lock vs template overlay around first loc
    if locs.size:
        plot_paths["lock_template_overlay"] = out_dir / f"lock_template_overlay__{adc_npz_path.stem}.png"
        save_plot_lock_template_overlay(adc_lock, template, center=int(locs[0]),
                                        fs=fs_adc, pad=window_half_width,
                                        out_path=plot_paths["lock_template_overlay"])
    else:
        print("No peaks found in template matching; returning without UA alignment.")
        return session_trigger, locs, None, plot_paths

    # ---- Early exit if we only want template matching ----
    if not align_to_ua:
        best_shift = None
        return session_trigger, locs, best_shift, plot_paths

    # 3) (Optional) Align triangles: Intan (adc_triangle) vs UA (ua_sync), deprecated for now, fix later after getting triangle sync waves
    ua_sync, fs_ua = load_ua_intan_sync(ua_npz_path)
    if int(round(fs_adc)) != int(round(fs_ua)):
        print(f"WARNING: sampling-rate mismatch (ADC={fs_adc:g} Hz, UA={fs_ua:g} Hz). Proceeding.")

    center = int(locs[0])
    best_shift, err = best_shift_rms_range_on_intan(
        adc_triangle, ua_sync, center,
        search=search_samples, win=window_half_width
    )
    print(f"[align] best shift near first loc: n={best_shift} (RMS={err:.4g})")

    # Save: triangle vs intan_sync overlay
    plot_paths["triangle_overlay"] = out_dir / (
        f"align_{adc_npz_path.stem}__vs__{ua_npz_path.stem}__firstloc_pm{window_half_width}.png"
    )
    ok = save_plot_triangle_overlay_matlab(
        adc_triangle, ua_sync, center, fs_adc,
        n_best=best_shift, pad=window_half_width,
        out_path=plot_paths["triangle_overlay"]
    )
    if not ok:
        del plot_paths["triangle_overlay"]
        print("Shifted window has no overlap; triangle overlay not saved.")

    return session_trigger, locs, best_shift, plot_paths

# ------------------------------- example usage -------------------------------
if __name__ == "__main__":
    from RCP_analysis import load_experiment_params, resolve_output_root
    REPO_ROOT = Path(__file__).resolve().parents[1]
    PARAMS = load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
    OUT_ROOT = resolve_output_root(PARAMS)

    ADC_NPZ  = OUT_ROOT / "bundles" / "NPRW" / "NRR_RW001_250915_150532_Intan_bundle" / "USB_board_ADC_input_channel.npz"
    UA_NPZ   = OUT_ROOT / "bundles" / "UA" / "NRR_RW_001_006_UA_bundle.npz"
    TEMPLATE = REPO_ROOT / "config" / "template.mat"
    PLOTS    = OUT_ROOT / "plots"

    session_trigger, locs, best_shift, paths = align_and_save_plot(
        adc_npz_path=ADC_NPZ,
        ua_npz_path=UA_NPZ,
        template_mat_path=TEMPLATE,
        out_dir=PLOTS,
        search_samples=500,
        window_half_width=150000,
        align_to_ua=False,   # <-- skip UA/triangle alignment
    )

    for k, v in paths.items():
        print(f"  {k}: {v}")
