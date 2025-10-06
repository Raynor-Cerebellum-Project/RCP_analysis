from pathlib import Path
import numpy as np
from ..functions.intan_preproc import load_stim_geometry, make_identity_probe_from_geom

def load_rate_npz(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    rate_hz = d["rate_hz"]           # (n_ch, n_bins_total)
    t_ms    = d["t_ms"]              # (n_bins_total,)
    meta    = d.get("meta", None)
    return rate_hz, t_ms, (meta.item() if hasattr(meta, "item") else meta)

def median_across_trials(zeroed_segments: np.ndarray):
    """
    Find median across trials per channel.
    Input: (n_trials, n_ch, n_twin) -> returns (n_ch, n_twin)
    """
    return np.median(zeroed_segments, axis=0)

# ---- alignment utils ----
def extract_peristim_segments(
    rate_hz: np.ndarray,
    t_ms: np.ndarray,
    stim_ms: np.ndarray,
    win_ms=(-800.0, 1200.0),
    min_trials: int = 1,
):
    """
    Return segments shape (n_trials, n_ch, n_twin) and rel_time_ms shape (n_twin,).
    Skips triggers whose window falls outside t_ms.
    """
    t_min, t_max = float(t_ms[0]), float(t_ms[-1])
    seg_len_ms = win_ms[1] - win_ms[0]

    # Relative timebase for a *perfectly* aligned segment (used only for plotting/logic)
    # We will slice on t_ms for each trial, so segment lengths are equal if t_ms is uniform.
    # Assume uniform binning for rates:
    dt = float(np.median(np.diff(t_ms)))  # ms per bin
    n_twin = int(round(seg_len_ms / dt))
    rel_time_ms = np.arange(n_twin) * dt + win_ms[0]

    segments = []
    kept = 0
    for s in np.asarray(stim_ms, dtype=float):
        start_ms = s + win_ms[0]
        end_ms   = s + win_ms[1]
        if start_ms < t_min or end_ms > t_max:
            continue  # skip partial windows

        # slice indices on t_ms
        i0 = int(np.searchsorted(t_ms, start_ms, side="left"))
        i1 = int(np.searchsorted(t_ms, end_ms,   side="left"))
        seg = rate_hz[:, i0:i1]  # (n_ch, n_twin)
        # Safety: ensure equal length (can happen if boundary falls between bins)
        if seg.shape[1] != n_twin:
            continue

        segments.append(seg)
        kept += 1

    if kept < min_trials:
        raise RuntimeError(f"Only {kept} peri-stim segments available (min_trials={min_trials}).")

    segments = np.stack(segments, axis=0)  # (n_trials, n_ch, n_twin)
    return segments, rel_time_ms

def detect_stim_channels_from_npz(
    stim_npz_path: Path,
    eps: float = 1e-12,
    min_edges: int = 1,
) -> np.ndarray:
    """
    Return GEOMETRY-ORDERED indices of stimulated channels by counting 0→nonzero
    rising edges over the full 'stim_traces' array stored in the NPZ.

    Note: extract_and_save_stim_npz() sets meta['order'] to 'geometry' when
    it can reorder using chanmap_perm, so indices should already be in geometry order.
    """
    stim_npz_path = Path(stim_npz_path)
    if not stim_npz_path.exists():
        return np.array([], dtype=int)

    with np.load(stim_npz_path, allow_pickle=False) as z:
        if "stim_traces" not in z:
            return np.array([], dtype=int)

        X = z["stim_traces"]  # (n_channels, n_samples)
        if X.ndim != 2 or X.shape[1] < 2:
            return np.array([], dtype=int)

        # per-channel thresholds: midpoint between 5th & 95th percentiles
        p5  = np.nanpercentile(X, 5, axis=1)
        p95 = np.nanpercentile(X, 95, axis=1)
        thr = 0.5 * (p5 + p95)

        above = (X > (thr[:, None] + eps))
        rising = above[:, 1:] & (~above[:, :-1])    # (ch, frames-1)
        counts = rising.sum(axis=1)                 # per-channel counts

        active = np.where(counts >= int(min_edges))[0].astype(int)

        # If meta says order=='geometry', active are geometry indices already.
        # If order=='device' we don't have a perm in this NPZ; return as-is.
        # (extract_and_save_stim_npz normally sets geometry when it can.)
        return np.unique(active)
 
def build_probe_and_locs_from_geom(geom_path: Path, radius_um: float = 5.0):
    """Load your saved geometry -> ProbeInterface Probe + (n_ch,2) locs."""
    geom = load_stim_geometry(geom_path)                  # your project format
    probe = make_identity_probe_from_geom(geom, radius_um=radius_um)  # ProbeInterface Probe
    locs  = probe.contact_positions.astype(float)         # (n_ch, 2)
    return probe, locs
