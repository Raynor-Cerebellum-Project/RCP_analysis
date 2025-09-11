from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import spikeinterface as si
import spikeinterface.extractors as se
import time, threading

# ---------------- user knobs ----------------
DATA_DIR = Path(
    "/home/bryan/mnt/cullen/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/data"
)
TARGET_STREAM = "RHS2000 amplifier channel"
OUT_BASE = DATA_DIR / "BL_RW_003_Session_1/Sorting Results/combined_all"
SESSION_ROOT = DATA_DIR / "BL_RW_003_Session_1/Intan"

TRIAL_IDX = 0  # which Intan subfolder (0-based)
STIM_NUM = 1  # 1 = first stim, 2 = second, ...
DIG_LINE = 0  # digital input line index (0-based)
WIN_PRE_S = 0.8
WIN_POST_S = 1.2
STIM_BAR_MS = 100
N_CHANNELS = 5
# -------------------------------------------

SESSION_FOLDERS = sorted([p for p in SESSION_ROOT.iterdir() if p.is_dir()])


# ---- instrumentation helpers ----
def timed(label):
    def wrapper(func):
        def inner(*args, **kwargs):
            t0 = time.perf_counter()
            print(f"[{label}] start", flush=True)
            out = func(*args, **kwargs)
            dt = time.perf_counter() - t0
            print(f"[{label}] done in {dt:.2f}s", flush=True)
            return out

        return inner

    return wrapper


class Heartbeat:
    def __init__(self, msg="working", interval=5):
        self.msg = msg
        self.interval = interval
        self._stop = False
        self._thread = threading.Thread(target=self._beat, daemon=True)

    def _beat(self):
        while not self._stop:
            print(f"[hb] {self.msg} @ {time.strftime('%H:%M:%S')}", flush=True)
            time.sleep(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop = True


# ---- helpers ----
def rising_edges(x, thresh=None, min_gap_samp=1):
    x = np.asarray(x).ravel()
    if x.size < 3:
        return np.array([], dtype=np.int64)
    if thresh is None:
        lo, hi = np.percentile(x, [5, 95])
        thresh = lo + 0.5 * (hi - lo)
    on = (x[:-1] <= thresh) & (x[1:] > thresh)
    idx = np.flatnonzero(on) + 1
    if idx.size == 0:
        return idx
    keep = [idx[0]]
    for r in idx[1:]:
        if r - keep[-1] >= min_gap_samp:
            keep.append(r)
    return np.asarray(keep, dtype=np.int64)


def fit_affine(s, d):
    A = np.c_[s.astype(float), np.ones_like(s, float)]
    a, b = np.linalg.lstsq(A, d.astype(float), rcond=None)[0]
    return float(a), float(b)


def session_offsets(recs):
    lens = [int(r.get_num_frames()) for r in recs]
    starts = np.cumsum([0] + lens[:-1])
    return lens, starts


# ---- load preprocessed global recording ----
@timed("load preprocessed")
def load_preproc():
    preproc = OUT_BASE / "preprocessed_global"
    try:
        return si.load(preproc)  # new API
    except Exception:
        return si.load_extractor(preproc / "si_folder.json")  # older API


rec_pp = load_preproc()
fs = float(rec_pp.get_sampling_frequency())
T_total = int(rec_pp.get_num_frames())


# ---- get analog segment offsets (for absolute indexing) ----
@timed("load analog recordings")
def load_ana():
    return [
        se.read_split_intan_files(
            folder_path=f,
            mode="concatenate",
            stream_name=TARGET_STREAM,
            use_names_as_ids=True,
        )
        for f in SESSION_FOLDERS
    ]


ana_recs = load_ana()
lens, starts = session_offsets(ana_recs)
seg0 = int(starts[TRIAL_IDX])
segN = int(lens[TRIAL_IDX])


# ---- digital line + stim onsets for THIS trial ----
@timed("load digital")
def load_dig():
    return se.read_split_intan_files(
        folder_path=SESSION_FOLDERS[TRIAL_IDX],
        mode="concatenate",
        stream_name="Digital Input Port",
        use_names_as_ids=True,
    )


rec_dig = load_dig()
assert abs(fs - float(rec_dig.get_sampling_frequency())) < 1e-6
dig_id = rec_dig.get_channel_ids()[DIG_LINE]
dig = rec_dig.get_traces(channel_ids=[dig_id]).ravel().astype(float)
dig = (dig - dig.min()) / (dig.ptp() if dig.ptp() > 0 else 1.0)  # normalize 0..1

min_gap = max(1, int(0.001 * fs))  # ~1 ms debounce
ed_pos = rising_edges(dig, min_gap_samp=min_gap)
ed_neg = rising_edges(1.0 - dig, min_gap_samp=min_gap)
dig_edges = ed_neg if ed_neg.size > ed_pos.size else ed_pos

if dig_edges.size < STIM_NUM:
    raise RuntimeError(f"Only {dig_edges.size} stims found; need STIM_NUM={STIM_NUM}")
stim_idx_local = int(dig_edges[STIM_NUM - 1])
stim_idx_abs = seg0 + stim_idx_local


# ---- stim_data.mat alignment to this trial ----
@timed("load stim_data.mat")
def load_stim():
    M = loadmat(SESSION_FOLDERS[TRIAL_IDX] / "stim_data.mat")
    return M["Stim_data"] if "Stim_data" in M else M["stim_data"]


Stim_data = load_stim()
Stim_data = np.asarray(Stim_data)
act = np.flatnonzero(np.any(Stim_data != 0, axis=1))
if act.size == 0:
    raise RuntimeError("No active stimulation channel in stim_data.mat")
stim_src = Stim_data[int(act[0]), :].astype(float).ravel()

stim_edges = rising_edges(stim_src, min_gap_samp=1)
aligned_ok = stim_edges.size >= 3 and dig_edges.size >= 3

if aligned_ok:
    K = min(stim_edges.size, dig_edges.size)
    a, b = fit_affine(stim_edges[:K], dig_edges[:K])  # d ≈ a*s + b

    def stim_on_window(abs_start, abs_end):
        n_local = np.arange(abs_start - seg0, abs_end - seg0, dtype=float)
        s_hat = (n_local - b) / (a if abs(a) > 1e-12 else 1.0)
        return np.interp(
            s_hat, np.arange(stim_src.size, dtype=float), stim_src, left=0.0, right=0.0
        )
else:

    def stim_on_window(abs_start, abs_end):  # naive length match
        n_local = np.arange(abs_start - seg0, abs_end - seg0, dtype=float)
        s_hat = n_local * (stim_src.size / max(1, segN))
        return np.interp(
            s_hat, np.arange(stim_src.size, dtype=float), stim_src, left=0.0, right=0.0
        )


# ---- slice window and plot ----
s0 = max(0, stim_idx_abs - int(WIN_PRE_S * fs))
s1 = min(T_total, stim_idx_abs + int(WIN_POST_S * fs))
t_rel = (np.arange(s1 - s0) + s0 - stim_idx_abs) / fs

chan_ids = list(rec_pp.get_channel_ids())[:N_CHANNELS]

hb = Heartbeat("get_traces", interval=5)
hb.start()
t0 = time.perf_counter()
tr = rec_pp.get_traces(start_frame=s0, end_frame=s1, channel_ids=chan_ids)
print(f"[get_traces] done in {time.perf_counter() - t0:.2f}s", flush=True)
hb.stop()

stim_win = stim_on_window(s0, s1)

# plotting
t0 = time.perf_counter()
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)

ax = fig.add_subplot(gs[0, 0])
offset = 0.0
for i, ch in enumerate(chan_ids):
    y = tr[:, i]
    mad = np.median(np.abs(y - np.median(y))) + 1e-9
    ax.plot(t_rel, y + offset, lw=0.8, label=f"ch {ch}")
    offset += 6 * mad

bar = STIM_BAR_MS / 1000.0
ax.axvspan(0.0, bar, alpha=0.25, label=f"stim {STIM_NUM} ({STIM_BAR_MS} ms)")
for e in dig_edges:
    e_abs = seg0 + int(e)
    if s0 <= e_abs < s1 and e_abs != stim_idx_abs:
        x0 = (e_abs - stim_idx_abs) / fs
        ax.axvspan(x0, x0 + bar, alpha=0.12)

ax.set_xlim(t_rel[0], t_rel[-1] if t_rel.size else (-WIN_PRE_S, WIN_POST_S))
ax.set_xlabel("Time (s, rel. to stim)")
ax.set_ylabel("Channels (offset)")
ax.legend(loc="upper right", fontsize=8)
ax.set_title(
    f"rec_pp around stim #{STIM_NUM} (trial {TRIAL_IDX}) — "
    f"{WIN_PRE_S:.1f}s pre / {WIN_POST_S:.1f}s post"
)

ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
ax2.plot(t_rel, stim_win, lw=0.9)
ax2.axvspan(0.0, bar, alpha=0.25)
ax2.set_xlabel("Time (s, rel. to stim)")
ax2.set_ylabel("stim")
ax2.grid(alpha=0.3)

plt.tight_layout()
print(f"[plot render] done in {time.perf_counter() - t0:.2f}s", flush=True)

print(f"[matplotlib] backend={plt.get_backend()} (blocking show)")
plt.show(block=True)
print("[matplotlib] window closed")
