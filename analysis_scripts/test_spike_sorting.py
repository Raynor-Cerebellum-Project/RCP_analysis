import os  # <-- add
# Limit BLAS/OpenMP threads so multiple Python workers don't oversubscribe CPUs
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

from pathlib import Path
import numpy as np
from scipy.io import loadmat
import h5py
import gc  # for explicit cleanup

# Save-only backend so it works in tmux/SSH/headless
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
from scipy.spatial import cKDTree

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.exporters as sexp
#import spikeinterface.postprocessing as spost
from spikeinterface.core import concatenate_recordings
from probeinterface import Probe
from probeinterface.plotting import plot_probe


# ----- parallel tuning -----
PARALLEL_JOBS = 2          # try 2–4; raise only if RAM is comfy
CHUNK = "0.5s"            # smaller chunk = lower per-worker RAM (0.25–0.5s good)
THREADS_PER_WORKER = 1     # keep 1 to avoid nested threading

# ==============================
# Config
# ==============================
DATA_DIR = Path("/home/bryan/mnt/cullen/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/data")
GEOM_MAT = DATA_DIR / "BL_RW_003_Session_1/ImecPrimateStimRec128_042421.mat"
TARGET_STREAM = "RHS2000 amplifier channel"

OUT_BASE = DATA_DIR / "BL_RW_003_Session_1/Sorting Results/combined_all"
OUT_BASE.mkdir(parents=True, exist_ok=True)

intan_root = DATA_DIR / "BL_RW_003_Session_1/Intan"
SESSION_FOLDERS = sorted([p for p in intan_root.iterdir() if p.is_dir()])
print("Found session folders:", len(SESSION_FOLDERS))

# ↓ Lower concurrency/chunk to reduce peak RAM (you can raise later once stable)
si.set_global_job_kwargs(n_jobs=PARALLEL_JOBS, chunk_duration=CHUNK, progress_bar=True)

# ---- quicklook knobs ----
DEFAULT_STIM_NUM = 1
DIG_LINE         = None
WIN_PRE_S        = 0.015   # 15 ms
WIN_POST_S       = 0.115
STIM_BAR_MS      = 20

# Optional per-session override (name -> stim_num)
STIM_NUMS = {
    # "BL_closed_loop_STIM_003_018": 3,
}

# ==============================
# Helpers
# ==============================
def ensure_signed(rec):
    if rec.get_dtype().kind == "u":
        rec = spre.unsigned_to_signed(rec)
        rec = spre.astype(rec, dtype="int16")
    return rec

def load_stim_device(sess_folder: Path) -> np.ndarray | None:
    """Load stim_data.mat as a 2D array (rows = device order, cols = time)."""
    mat = _load_stim_matrix(sess_folder / "stim_data.mat")    
    if mat is None:
        return None
    if mat.ndim == 1:
        # Single-channel vector -> treat as (1, T) device row
        return mat[np.newaxis, :]
    if mat.shape[0] != NCH:
        print(f"[stim] Unexpected #rows {mat.shape[0]} (expected {NCH}). Continuing anyway.")
    return np.asarray(mat, float)

def _load_intan_idx(sess_folder: Path):
    """
    Load Data.Intan_idx from the first '*Cal.mat' in the session folder.
    Returns a 0-based int array or None if not found.
    """
    cals = sorted(sess_folder.glob("*Cal.mat"))
    if not cals:
        return None

    M = loadmat(cals[0], squeeze_me=True, struct_as_record=False)
    Data = M.get("Data", None)
    if Data is None:
        return None

    # Robust access across MATLAB struct flavors
    idx = None
    try:
        # object-like (fields accessible as attributes)
        idx = getattr(Data, "Intan_idx", None)
    except Exception:
        pass
    if idx is None:
        # record-array / void dtype with named fields
        try:
            if isinstance(Data, np.void) and "Intan_idx" in Data.dtype.names:
                idx = Data["Intan_idx"]
        except Exception:
            pass
    if idx is None:
        return None

    idx = np.asarray(idx).ravel().astype(np.int64)
    # MATLAB -> Python indexing
    return idx - 1

def load_stim_geometry(sess_folder: Path) -> np.ndarray | None:
    """
    Return stim matrix with rows in GEOMETRY (MATLAB) order.
    The file is in DEVICE order; use G2D to reorder.
    """
    mat_dev = load_stim_device(sess_folder)
    if mat_dev is None:
        return None
    if mat_dev.shape[0] >= geom_corr_ind.size:
        # ✅ device (rows) -> geometry rows
        return mat_dev[geom_corr_ind, :]
    return mat_dev

def get_stimulated_channel_ids_from_geom(sess_folder: Path, rec_geom) -> set[str]:
    """
    Detect active stim rows from a GEOMETRY-ordered stim matrix,
    return the corresponding channel IDs from rec_geom (also geometry-ordered).
    """
    stim_geom = load_stim_geometry(sess_folder)
    if stim_geom is None or stim_geom.ndim != 2:
        return set()

    # any nonzero at any time -> stimulated row
    active_rows = np.flatnonzero(np.any(stim_geom != 0, axis=1)).astype(int)
    if active_rows.size == 0:
        return set()

    ids_geom = list(rec_geom.get_channel_ids())  # geometry order
    return {ids_geom[i] for i in active_rows if 0 <= i < len(ids_geom)}

def _load_stim_matrix(path: Path) -> np.ndarray | None:
    """
    Return Stim_data/stim_data as a 2D float array or None.
    Supports both classic MAT and v7.3 (HDF5) files.
    """
    if not path.exists():
        return None

    # Try classic MAT first
    try:
        M = loadmat(path, squeeze_me=False, struct_as_record=False)
        key = "Stim_data" if "Stim_data" in M else ("stim_data" if "stim_data" in M else None)
        if key is None:
            return None
        arr = np.asarray(M[key])
        # ensure 2D float
        arr = np.array(arr, dtype=float)
        # normalize orientation: time should be the longer axis
        if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
            arr = arr.T
        return arr
    except NotImplementedError:
        # v7.3 fallback via h5py
        pass

    # HDF5 (MAT v7.3)
    try:
        with h5py.File(path, "r") as f:
            # typical top-level dataset names
            for key in ("Stim_data", "stim_data"):
                if key in f:
                    ds = f[key]
                    arr = np.array(ds, dtype=float)          # load to numpy
                    # normalize orientation: time should be the longer axis
                    if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                        arr = arr.T
                    return arr
            # if stored under a group, try a shallow scan
            for key in f.keys():
                obj = f[key]
                if isinstance(obj, h5py.Dataset):
                    continue
                # look for dataset children named Stim_data/stim_data
                for cand in ("Stim_data", "stim_data"):
                    if cand in obj:
                        arr = np.array(obj[cand], dtype=float)
                        if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                            arr = arr.T
                        return arr
    except Exception as e:
        print(f"[stim] HDF5 read failed for {path.name}: {e}")

    return None

def _probe_sidebar(ax, rec_sess, sel_indices, stim_ids, title="Probe (top→bottom)"):
    probe = rec_sess.get_probe()
    ids   = list(rec_sess.get_channel_ids())
    pos   = probe.contact_positions  # (n_contacts, 2)

    # Colors: default light gray; stim sites red
    colors = ["0.8"] * pos.shape[0]
    id2idx = {cid: i for i, cid in enumerate(ids)}
    for sid in (stim_ids or []):
        if sid in id2idx:
            colors[id2idx[sid]] = "red"

    # Plot probe (use only supported args across versions)
    try:
        plot_probe(probe, ax=ax, contacts_colors=colors)
    except TypeError:
        plot_probe(probe, ax=ax)  # very old versions: no colors

    # --- Blue box around viewed contacts (ALWAYS run) ---
    # Try to infer contact radius; fall back to 6 µm
    r = 6.0
    shpp = getattr(probe, "contact_shape_params", None)
    try:
        if isinstance(shpp, dict) and "radius" in shpp:
            r = float(shpp["radius"])
        elif isinstance(shpp, (list, tuple)) and shpp and isinstance(shpp[0], dict) and "radius" in shpp[0]:
            r = float(shpp[0]["radius"])
        elif isinstance(shpp, np.ndarray):
            if getattr(shpp, "dtype", None) is not None and shpp.dtype.names and ("radius" in shpp.dtype.names):
                val = shpp["radius"][0]
                r = float(val) if np.ndim(val) == 0 else float(val[0])
            else:
                first = shpp[0]
                if isinstance(first, dict) and "radius" in first:
                    r = float(first["radius"])
    except Exception:
        pass

    # Normalize sel_indices safely (handles None/int/array/list)
    if sel_indices is None:
        sel_arr = np.array([], dtype=int)
    else:
        sel_arr = np.atleast_1d(sel_indices).astype(int)

    # Draw ONE big box around all viewed contacts
    if sel_arr.size:
        xs = np.asarray(pos[sel_arr, 0], float)
        ys = np.asarray(pos[sel_arr, 1], float)
        pad = 2.2 * r  # a little extra margin beyond the contact radius
        rect = patches.Rectangle(
            (xs.min() - pad, ys.min() - pad),
            (xs.max() - xs.min()) + 2 * pad,
            (ys.max() - ys.min()) + 2 * pad,
            linewidth=1.6, edgecolor="tab:blue", facecolor="none", zorder=5
        )
        ax.add_patch(rect)

    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")

def run_sorter_and_export(name: str, recording, base_folder: Path, sorter_params=None):
    sorter_params = sorter_params or {}
    sort_out = base_folder / f"{name}_output"

    sorting = ss.run_sorter(
        name,
        recording,
        folder=sort_out,
        remove_existing_folder=True,
        delete_output_folder=False,
        verbose=False,
        **sorter_params,
    )

    an_dir = base_folder / f"analyzer_{name}"
    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="binary_folder",
        folder=an_dir,
        sparse=True,
        overwrite=True,
    )
    analyzer.compute(
        ["random_spikes", "waveforms", "templates", "noise_levels",
         "spike_amplitudes", "principal_components"],
        extension_params={"random_spikes": {"max_spikes_per_unit": 500}},
        n_jobs=PARALLEL_JOBS,           # <-- was 1
        chunk_duration=CHUNK,           # <-- was "0.25s" hardcoded
        progress_bar=True,
        save=True,
    )

    phy_dir = base_folder / f"phy_output_{name}"
    sexp.export_to_phy(analyzer, output_folder=phy_dir, remove_if_exists=True)
    
    return sorting, analyzer

def extract_triggers_and_repeats(stim_geom: np.ndarray, buffer_samples: int = 20):
    """
    Trigger extraction on geometry-ordered stim matrix.
    Returns:
      trigs : (N, 2) int array [begin_sample, end_sample] in stim sample index
      repeat_boundaries : 1D int array of group boundary indices (0-based, with final N)
      STIM_CHANS : 1D int array of active row indices (geometry order)
    """
    # === Detect stim channels ===
    STIM_CHANS = np.flatnonzero(np.any(stim_geom != 0, axis=1))
    if STIM_CHANS.size == 0:
        return np.empty((0, 2), dtype=int), np.array([0], dtype=int), STIM_CHANS

    # === Trigger detection on FIRST active row ===
    TRIGDAT = np.asarray(stim_geom[STIM_CHANS[0], :], float).ravel()
    d = np.diff(TRIGDAT)

    trigs1 = np.flatnonzero(d < 0)   # falling
    trigs2 = np.flatnonzero(d > 0)   # rising

    # Next zero after each falling edge
    rz = []
    for idx in trigs1:
        tail = TRIGDAT[idx+1:]
        zrel = np.flatnonzero(np.abs(tail) <= 1e-9)
        if zrel.size:
            rz.append(idx + 1 + int(zrel[0]))
    trigs_rz = np.asarray(rz, int)

    # Begin/end pairing
    trigs_beg = trigs1 if trigs2.size <= trigs1.size else trigs2
    trigs_beg = trigs_beg[::2]    # 1:2:end
    trigs_end = trigs_rz[1::2]    # 2:2:end

    n = int(min(trigs_beg.size, trigs_end.size))
    if n == 0:
        return np.empty((0, 2), int), np.array([0], int), STIM_CHANS

    trigs = np.c_[trigs_beg[:n], trigs_end[:n]]

    # === Repeat boundaries ===
    if n >= 2:
        diffs = np.diff(trigs[:, 0])
        repeat_gap_threshold = 2 * (2 * buffer_samples + 1)
        cuts = np.flatnonzero(diffs > repeat_gap_threshold) + 1
        repeat_boundaries = np.r_[0, cuts, n].astype(int)
    else:
        repeat_boundaries = np.array([0, n], int)

    return trigs, repeat_boundaries, STIM_CHANS

def quicklook_stim_grid(
    rec_sess, sess_folder: Path, out_dir: Path, fs: float,
    stim_num: int, nrows: int = 4, ncols: int = 4,
    pre_s: float = 0.015, post_s: float = 0.020, bar_ms: float = 20,
    dig_line=None, pick="top", sel_indices=None, name_suffix="",
    show_probe=True
):
    out_dir.mkdir(parents=True, exist_ok=True)
    sess_name = sess_folder.name
    bar = float(bar_ms) / 1000.0

    # ---- load stim_data (geometry-ordered) ----
    stim_geom_full = load_stim_geometry(sess_folder)
    if stim_geom_full is None:
        print(f"[{sess_name}] stim_data missing; skipping.")
        return

    # ---- OPTIONAL sanity check against Data.Intan_idx (if present) ----
    intan_idx_0 = _load_intan_idx(sess_folder)
    if intan_idx_0 is not None:
        n_rec  = int(rec_sess.get_num_frames())
        n_idx  = int(intan_idx_0.size)
        tstim  = int(stim_geom_full.shape[1])
        # 1) same number of indices as Intan frames
        assert n_idx == n_rec, (
            f"[{sess_name}] ERROR: Data.Intan_idx has {n_idx} samples "
            f"but Intan recording has {n_rec} frames."
        )
        # 2) all indices in-bounds for stim_data
        max_idx = int(np.max(intan_idx_0))
        min_idx = int(np.min(intan_idx_0))
        assert 0 <= min_idx and max_idx < tstim, (
            f"[{sess_name}] ERROR: Data.Intan_idx range [{min_idx}, {max_idx}] "
            f"exceeds stim_data length {tstim}."
        )

    # ---- choose first active stim row (as in your MATLAB) ----
    active_rows = np.flatnonzero(np.any(stim_geom_full != 0, axis=1))
    if active_rows.size == 0:
        print(f"[{sess_name}] no active stim rows; skipping.")
        return
    chosen_row = int(active_rows[0])
    stim_trace = stim_geom_full[chosen_row, :]

    # ---- onsets from stim_data only (first falling edge / 0->nonzero) ----
    onsets = _pulse_onsets_from_stim_row(stim_trace, eps=1e-12, min_gap=1)
    if onsets.size < stim_num:
        print(f"[{sess_name}] only {onsets.size} pulses in stim_data; need {stim_num}. Skipping.")
        return

    # pick the requested pulse as time-zero
    stim_idx = int(onsets[stim_num - 1])

    # ---- slice window centered on chosen stim ----
    s0 = max(0,                stim_idx - int(pre_s  * fs))
    s1 = min(int(rec_sess.get_num_frames()), stim_idx + int(post_s * fs))
    n  = s1 - s0
    if n <= 1:
        print(f"[{sess_name}] WARN: empty window; skipping.")
        return
    t_rel = (np.arange(n) + s0 - stim_idx) / fs

    # onsets in this window, relative to t=0
    ts = ((onsets[(onsets >= s0) & (onsets < s1)] - stim_idx) / fs).astype(float)
    ts.sort()

    # ----- GEOMETRY-ORDERED CHANNEL SELECTION -----
    ids = list(rec_sess.get_channel_ids())
    pos = rec_sess.get_probe().contact_positions  # (nch, 2): [:,1] = y
    order = np.lexsort((pos[:, 0], pos[:, 1]))    # y then x (ascending)

    k = nrows * ncols
    if sel_indices is None:
        if pick == "bottom":
            sel = order[-k:][::-1]
        elif pick == "center":
            y = pos[:, 1]
            near = np.argsort(np.abs(y - np.median(y)))[:k]
            sel = near[np.lexsort((pos[near, 0], y[near]))]
        else:
            sel = order[:k]  # 'top'
    else:
        sel = np.array(sel_indices, dtype=int)
        if sel.size != k:
            print(f"[{sess_name}] sel_indices has size {sel.size}, expected {k}. Skipping.")
            return

    ch_ids = [ids[i] for i in sel]
    traces = rec_sess.get_traces(start_frame=s0, end_frame=s1, channel_ids=ch_ids)

    stim_set_total = get_stimulated_channel_ids_from_geom(sess_folder, rec_sess)

    # ---------- figure layout: grid + optional probe sidebar ----------
    if show_probe:
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1], wspace=0.15)
        gs_grid = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs[0, 0], wspace=0.05, hspace=0.15)
        axes = np.array([[plt.subplot(gs_grid[r, c]) for c in range(ncols)] for r in range(nrows)])
        ax_probe = plt.subplot(gs[0, 1])
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10), sharex=True)
        ax_probe = None

    axes = np.atleast_2d(axes)

    # ----- plot the nrows×ncols traces -----
    for k_idx, cid in enumerate(ch_ids):
        r, c = divmod(k_idx, ncols)
        r = nrows - 1 - r   # flip vertical indexing
        ax = axes[r, c]
        ych = traces[:, k_idx]
        med = np.median(ych)
        mad = np.median(np.abs(ych - med)) + 1e-9

        ax.plot(t_rel, ych, lw=0.8)
        ax.axhline(med, lw=0.3, alpha=0.3)

        # dashed lines + shaded spans at stim onsets (from stim_data only)
        if ts.size:
            for t in ts:
                ax.axvline(t, color="tab:blue", linestyle="--", lw=0.6, alpha=0.6)

            gap_thresh = max(2.0 * bar, 0.010)
            diffs = np.diff(ts)
            cut = np.where(diffs > gap_thresh)[0]
            starts = np.r_[0, cut + 1]
            ends   = np.r_[cut, ts.size - 1]
            for a_i, b_i in zip(starts, ends):
                ax.axvspan(ts[a_i], ts[b_i] + bar, facecolor="tab:blue", alpha=0.10, zorder=-1)

        stim_tag = " (Stim)" if cid in stim_set_total else ""
        ax.set_title(f"{cid}{stim_tag}", fontsize=9)
        ax.set_ylim(med - 6 * mad, med + 6 * mad)
        if r < nrows - 1: ax.tick_params(labelbottom=False)
        if c > 0:         ax.tick_params(labelleft=False)

    axes[-1, 0].set_xlabel(f"Time relative to stim #{stim_num} (s)")
    axes[-1, 0].set_ylabel("µV")

    fig.suptitle(
        f"{sess_name} — {nrows}×{ncols} channels by geometry "
        f"(pre {pre_s:.3f}s / post {post_s:.3f}s){name_suffix}",
        y=0.995
    )

    if show_probe and ax_probe is not None:
        _probe_sidebar(ax_probe, rec_sess, sel_indices=sel, stim_ids=stim_set_total,
                       title="Probe (stim=red, viewed=blue box)")

    fig.tight_layout()
    suffix = f"_{name_suffix.strip('_')}" if name_suffix else ""
    out_png = out_dir / f"quicklook_grid_{sess_name}_stim{stim_num}_{nrows}x{ncols}{suffix}.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[{sess_name}] grid quicklook -> {out_png}")

def _pulse_onsets_from_stim_row(stim_row, eps=1e-12, min_gap=1):
    """
    Return indices where the stimulation goes from ~0 to non-zero (pulse onsets).
    eps: zero threshold. min_gap: debounce in samples (>=1).
    """
    x = np.asarray(stim_row, float).ravel()
    is_on = np.abs(x) > eps
    idx = np.flatnonzero((~is_on[:-1]) & (is_on[1:])) + 1
    if idx.size and min_gap > 1:
        keep = [idx[0]]
        for j in idx[1:]:
            if j - keep[-1] >= min_gap:
                keep.append(j)
        idx = np.asarray(keep, dtype=int)
    return idx

def quicklook_stim_grid_all(
    rec_sess, sess_folder: Path, out_dir: Path, fs: float,
    stim_num: int, nrows: int = 4, ncols: int = 4,
    pre_s: float = 0.015, post_s: float = 0.020, bar_ms: float = 20,
    dig_line=None, stride: int | None = None, show_probe=True
):
    """Generate multiple 4x4 panels down the probe (top -> bottom).
    stride: how many channels to move between panels (default = nrows*ncols for no overlap).
            Use smaller (e.g., 8) for 50% overlap.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = list(rec_sess.get_channel_ids())
    pos = rec_sess.get_probe().contact_positions
    order = np.lexsort((pos[:, 0], pos[:, 1]))  # y then x
    group = nrows * ncols
    if stride is None:
        stride = group  # no overlap by default

    n = len(ids)
    panel_idx = 0
    for start in range(0, max(1, n - group + 1), stride):
        sel = order[start:start + group]
        if sel.size < group:
            break
        name_suffix = f"_p{panel_idx:02d}_ch{start:03d}-{start+group-1:03d}"
        quicklook_stim_grid(
            rec_sess=rec_sess, sess_folder=sess_folder, out_dir=out_dir, fs=fs,
            stim_num=stim_num, nrows=nrows, ncols=ncols,
            pre_s=pre_s, post_s=post_s, bar_ms=bar_ms, dig_line=dig_line,
            pick="top", sel_indices=sel, name_suffix=name_suffix,
            show_probe=show_probe,
        )
        panel_idx += 1

def count_neighbors_in_radius(rec, r_min_um=0.0, r_max_um=200.0, same_shank=True):
    """
    Return an array 'counts' such that counts[i] = # of channels whose center is
    at distance in [r_min_um, r_max_um] from channel i.

    If same_shank=True and the probe has shank IDs, neighbors are restricted to the same shank.
    """
    probe = rec.get_probe()
    pos = np.asarray(probe.contact_positions, float)  # (nch, 2)
    nch = pos.shape[0]

    # optional shank restriction
    if same_shank and hasattr(probe, "shank_ids") and probe.shank_ids is not None:
        shank_ids = np.asarray(probe.shank_ids)
    else:
        shank_ids = np.zeros(nch, dtype=int)

    counts = np.zeros(nch, dtype=int)
    for sh in np.unique(shank_ids):
        idx = np.where(shank_ids == sh)[0]
        if idx.size == 0: 
            continue
        tree = cKDTree(pos[idx])
        # query neighbors within outer radius
        neighs_max = tree.query_ball_point(pos[idx], r_max_um + 1e-9)
        if r_min_um > 0:
            # remove those within inner radius
            neighs_min = tree.query_ball_point(pos[idx], r_min_um - 1e-9)
        for k, nb in enumerate(neighs_max):
            # exclude self
            nb = [j for j in nb if j != k]
            if r_min_um > 0:
                inner = set(neighs_min[k])
                nb = [j for j in nb if j not in inner]
            counts[idx[k]] = len(nb)
    return counts

def make_probe_with_np_permutation(geom_mat_path: Path, index_list_1based) -> Probe:
    mat = loadmat(geom_mat_path)
    x = mat["xcoords"].astype(float).ravel()
    y = mat["ycoords"].astype(float).ravel()
    pr = Probe(ndim=2)
    pr.set_contacts(positions=np.c_[x, y], shapes="circle", shape_params={"radius": 5})

    # DIRECT mapping: for contact c (geometry order), device channel = index_list[c] - 1
    dev_idx = np.asarray(index_list_1based, dtype=int).ravel() - 1
    if dev_idx.size != x.size:
        raise ValueError("Permutation length != #contacts in geometry file.")
    if np.min(dev_idx) < 0 or np.max(dev_idx) >= dev_idx.size:
        raise ValueError("Permutation contains out-of-range device indices.")

    pr.set_device_channel_indices(dev_idx)
    print("Probe remapped using new config")
    return pr

def permuted_geometry_view(rec):
    """
    Slice 'rec' into the exact geometry order used in MATLAB:
      fwrite(... amplifier_data_copy(neuropixel_index, :), 'int16')
    Here, NEUROPIXEL_INDEX_1BASED encodes geometry->device (1-based).
    """
    ids_device = list(rec.get_channel_ids())     # current device order
    geom_ids_by_perm = [ids_device[i] for i in geom_corr_ind]  # G2D is 0-based geometry->device
    if hasattr(rec, "channel_slice"):
        return rec.channel_slice(channel_ids=geom_ids_by_perm)
    from spikeinterface.core import ChannelSliceRecording
    return ChannelSliceRecording(rec, channel_ids=geom_ids_by_perm)

import csv

def neighbors_in_radius(rec, r_min_um=0.0, r_max_um=150.0, same_shank=True, return_ids=True):
    """
    Build a mapping: center channel -> list of neighbors whose centers are within [r_min_um, r_max_um].
    If same_shank=True, neighbors are limited to the same shank (when available).
    If return_ids=True, returns channel IDs; otherwise returns global indices.
    """
    probe = rec.get_probe()
    pos = np.asarray(probe.contact_positions, float)  # (nch, 2)
    ids = list(rec.get_channel_ids())
    nch = pos.shape[0]

    # shank restriction (if available)
    if same_shank and hasattr(probe, "shank_ids") and probe.shank_ids is not None:
        shank_ids = np.asarray(probe.shank_ids)
    else:
        shank_ids = np.zeros(nch, dtype=int)

    # Prepare result as lists per center index
    neigh_indices = [[] for _ in range(nch)]

    for sh in np.unique(shank_ids):
        idx = np.where(shank_ids == sh)[0]
        if idx.size == 0:
            continue
        tree = cKDTree(pos[idx])
        # within outer radius
        neighs_max = tree.query_ball_point(pos[idx], r_max_um + 1e-9)
        # within inner radius (to exclude)
        if r_min_um > 0:
            neighs_min = tree.query_ball_point(pos[idx], r_min_um - 1e-9)
        else:
            neighs_min = [set() for _ in idx]

        for k_local, nb_local in enumerate(neighs_max):
            # convert local indices -> global indices
            center_global = int(idx[k_local])
            nb_global = [int(idx[j]) for j in nb_local if j != k_local]  # exclude self
            if r_min_um > 0:
                inner = set(int(idx[j]) for j in neighs_min[k_local])
                nb_global = [j for j in nb_global if j not in inner]
            neigh_indices[center_global] = nb_global

    if return_ids:
        return {ids[i]: [ids[j] for j in neigh_indices[i]] for i in range(nch)}
    else:
        return {i: neigh_indices[i] for i in range(nch)}

def save_neighbors_csv(neighbor_map, csv_path: Path):
    """
    Save rows: center_id, neighbor_id (one row per neighbor).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["center_channel", "neighbor_channel"])
        for center, neighs in neighbor_map.items():
            if not neighs:
                w.writerow([center, ""])  # no neighbors in annulus
            else:
                for nb in neighs:
                    w.writerow([center, nb])


# ==============================
# Main pipeline
# ==============================

# 1-based Neuropixels reindex list from MATLAB (paste your exact list here)
NEUROPIXEL_INDEX_1BASED = [
    18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 29, 17, 2, 32, 1, 30,
    31, 39, 3, 36, 38, 28, 35, 37,
    4, 34, 16, 33, 15, 14, 13, 12,
    11, 10, 9, 8, 7, 6, 5, 63,
    59, 56, 64, 58, 55, 40, 57, 54,
    41, 60, 53, 43, 61, 52, 44, 62,
    51, 42, 47, 50, 45, 48, 49, 46,
    65, 96, 69, 66, 95, 68, 67, 94,
    70, 83, 93, 72, 84, 92, 71, 85,
    91, 73, 88, 90, 81, 87, 89, 82,
    86, 108, 107, 106, 105, 104, 103, 102,
    101, 100, 99, 98, 80, 97, 79, 109,
    76, 78, 117, 75, 77, 110, 74, 114,
    115, 112, 113, 111, 128, 116, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 127,
]
# geometry -> device (0-based)
geom_corr_ind = np.asarray(NEUROPIXEL_INDEX_1BASED, dtype=int) - 1
assert geom_corr_ind.ndim == 1
NCH = geom_corr_ind.size

probe = make_probe_with_np_permutation(GEOM_MAT, NEUROPIXEL_INDEX_1BASED)

saved_paths = []  # keep only paths, not big recording objects

for idx, folder in enumerate(SESSION_FOLDERS):
    if idx < 2:
        print(f"[sess {idx+1}/{len(SESSION_FOLDERS)}] {folder.name}: skipping first two by request")
        continue
    print(f"[sess {idx+1}/{len(SESSION_FOLDERS)}] {folder.name}: load analog")

    rec = se.read_split_intan_files(
        folder, mode="concatenate", stream_name=TARGET_STREAM, use_names_as_ids=True
    )
    rec = ensure_signed(rec).set_probe(probe, in_place=False)

    # preprocess per session
    rec_hpf   = spre.highpass_filter(rec, freq_min=300)
    rec_local = spre.common_reference(rec_hpf, reference="local", operator="median", local_radius=(60, 150))

    # ---- APPLY THE MATLAB PERMUTATION BEFORE SAVING (match neuropixel_index) ----
    rec_local_perm = permuted_geometry_view(rec_local)
    # save only the permuted, preprocessed data
    out_geom = OUT_BASE / f"pp_local_30_150__{folder.name}_GEOM"

    # free upstream objects ASAP
    rec_local_perm.save(folder=out_geom, overwrite=True)
    del rec_hpf, rec, rec_local, rec_local_perm
    gc.collect()

    # ---- reload the SAME permuted view (no extra reordering) ----
    try:
        rec_perm = si.load(out_geom)
    except Exception:
        rec_perm = si.load_extractor(out_geom / "si_folder.json")

    # quicklooks in the MATLAB order
    fs = float(rec_perm.get_sampling_frequency())
    stim_num = STIM_NUMS.get(folder.name, DEFAULT_STIM_NUM)
    
    # quick diagnostic of neighbors used for local referencing (matches local_radius=(30,150))
    if idx == 1:
        neighbor_map = neighbors_in_radius(rec_perm, r_min_um=30.0, r_max_um=150.0, same_shank=True, return_ids=True)
        # Save to CSV for later inspection
        csv_out = OUT_BASE / "neighbors" / f"neighbors_{folder.name}_r30-150.csv"
        save_neighbors_csv(neighbor_map, csv_out)

    
    quicklook_stim_grid_all(
        rec_sess=rec_perm,
        sess_folder=folder,
        out_dir=OUT_BASE / "quicklooks",
        fs=fs,
        stim_num=stim_num,
        nrows=4, ncols=4,
        pre_s=WIN_PRE_S, post_s=WIN_POST_S,
        bar_ms=STIM_BAR_MS,
        dig_line=DIG_LINE,
        stride=16,
    )
    print(f"[{folder.name}] saved permuted session -> {out_geom}")

    # cleanup (only what still exists)
    del rec_perm
    gc.collect()

    # concat the permuted saves
    saved_paths.append(out_geom)
        
# concatenate all preprocessed sessions from disk (lazy)
print("Concatenating preprocessed sessions...")
recs_for_concat = []
for p in saved_paths:
    try:
        r = si.load(p)
    except Exception:
        r = si.load_extractor(p / "si_folder.json")
    recs_for_concat.append(r)
rec_concat = concatenate_recordings(recs_for_concat)
gc.collect()  # optional

# Run MountainSort5 (tweak params as you like)
sorting_ms5, analyzer_ms5 = run_sorter_and_export(
    "mountainsort5",
    rec_concat,
    OUT_BASE,
    sorter_params={
        "n_jobs": PARALLEL_JOBS,
        "chunk_duration": CHUNK,
        "pool_engine": "process",          # process pool (good isolation)
        "max_threads_per_worker": THREADS_PER_WORKER,
        # optional MS5 knobs if you need: "detect_threshold": 5, "detect_sign": "neg"
    },
)

# If desired, free the individual handles now (concat keeps references)
del recs_for_concat, saved_paths
gc.collect()

# Run KiloSort4 (if GPU available)
# sorting_ks4, analyzer_ks4 = run_sorter_and_export("kilosort4", rec_concat, OUT_BASE)

print("Done. Open with:")
print(f"  cd '{OUT_BASE}/phy_output_mountainsort5' && phy template-gui params.py")
print(f"  cd '{OUT_BASE}/phy_output_kilosort4' && phy template-gui params.py")
