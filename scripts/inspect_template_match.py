import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import find_peaks
from RCP_analysis import load_experiment_params, resolve_output_root

# ---- paths ----
# template.mat you specified:
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
OUT_BASE = resolve_output_root(PARAMS)
OUT_BASE.mkdir(parents=True, exist_ok=True)

UA_bundles_root = OUT_BASE / "bundles" / "UA" / "NRR_RW_001_006_UA_bundle.npz"
INTAN_NPZ = OUT_BASE / "bundles" / "NPRW" / "NRR_RW001_250915_150532_Intan_bundle" / "USB_board_ADC_input_channel.npz"
TEMPLATE_MAT = REPO_ROOT / "config" / "template.mat"

def _load_meta(npz):
    if "meta" not in npz.files: return {}
    m = npz["meta"]
    if isinstance(m, np.ndarray) and m.ndim == 0: m = m.item()
    if isinstance(m, (bytes, bytearray)): m = m.decode("utf-8", "ignore")
    if isinstance(m, str):
        import json
        try: m = json.loads(m)
        except Exception: m = {}
    return m if isinstance(m, dict) else {}

def _concat_intan_channel(npz_path: Path, ch: int):
    z = np.load(npz_path, allow_pickle=True, mmap_mode="r")
    meta = _load_meta(z)
    fs = float(meta.get("fs_hz", 30000.0))
    keys = sorted([k for k in z.files if k.startswith("chunk_")])
    if not keys:
        raise RuntimeError("No 'chunk_####' arrays found in NPZ.")
    parts = []
    for k in keys:
        a = z[k]
        if a.ndim == 1:
            if ch != 0: raise IndexError("Chunk is 1D; only channel 0 exists.")
            parts.append(a.astype(np.float64))
        else:
            if ch >= a.shape[1]: raise IndexError(f"Requested ch {ch}, chunk {k} has {a.shape[1]} channels.")
            parts.append(a[:, ch].astype(np.float64))
    return np.concatenate(parts), fs

def _normalized_xcorr(lock: np.ndarray, key: np.ndarray):
    x = (lock - lock.mean()) / (lock.std() + 1e-12)
    k = (key  - key.mean())  / (key.std()  + 1e-12)
    c = np.correlate(x, k, mode="full")
    c = c / (c.max() if c.size else 1.0)
    lags = np.arange(-(k.size - 1), x.size)
    return c, lags

# ---- load signals ----
lock, fs_intan = _concat_intan_channel(INTAN_NPZ, ch=1)  # use channel 1
key = np.asarray(loadmat(str(TEMPLATE_MAT))["template"]).squeeze().astype(np.float64)

# ---- xcorr + peaks ----
C, LAGS = _normalized_xcorr(lock, key)
peak_idx, _ = find_peaks(C, height=0.95)
locs = LAGS[peak_idx].astype(int)

print(f"Found {locs.size} peaks >= 0.95")
if locs.size == 0:
    raise SystemExit("No peaks >= 0.95; consider lowering the threshold.")

first_loc = int(locs[0])
print(f"first loc (samples): {first_loc}  |  (seconds): {first_loc/fs_intan:.6f}")

# ---- plot ±30,000 samples around first loc, overlay template ----
pad = 30000
N = lock.size
win_i0 = max(0, first_loc - pad)
win_i1 = min(N, first_loc + pad)
t = (np.arange(win_i0, win_i1) - first_loc) / fs_intan

lock_win = lock[win_i0:win_i1]
lock_win_z = (lock_win - lock_win.mean()) / (lock_win.std() + 1e-12)

templ_len = key.size

# --- robust overlap between template [first_loc, first_loc+templ_len) and window [win_i0, win_i1) ---
overlap_start_global = max(win_i0, first_loc)
overlap_end_global   = min(win_i1, first_loc + templ_len)

templ_overlay = np.full(lock_win.shape, np.nan, dtype=float)

if overlap_end_global > overlap_start_global:
    # indices in the window
    rel_start = overlap_start_global - win_i0
    rel_end   = overlap_end_global   - win_i0

    # corresponding indices in the template
    t0 = overlap_start_global - first_loc
    t1 = overlap_end_global   - first_loc

    key_seg = key[t0:t1]
    key_seg_z = (key_seg - key_seg.mean()) / (key_seg.std() + 1e-12)

    # lengths match by construction
    templ_overlay[rel_start:rel_end] = key_seg_z
else:
    print("Warning: no overlap between template placement and window (this can happen near boundaries).")

plt.figure(figsize=(12,4))
plt.plot(t, lock_win_z, label="Intan ch1 (z)", linewidth=1)
plt.plot(t, templ_overlay, label="template (z, aligned at loc)", linewidth=1)
plt.axvline(0.0, color='k', linestyle='--', linewidth=1, label="first loc")
plt.xlabel("Time from first loc [s]")
plt.ylabel("z-scored amplitude")
plt.title("Lock (Intan ch1) with template overlay around first loc (±30k samples)")
plt.legend(loc="best")
plt.tight_layout()

out_dir = INTAN_NPZ.parent / "plots"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "lock_ch1_template_overlay_firstloc_pm30000.png"
plt.savefig(out_path, dpi=160, bbox_inches="tight")
plt.close()

print("Saved plot ->", out_path)
