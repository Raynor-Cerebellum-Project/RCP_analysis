#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from RCP_analysis import load_experiment_params, resolve_output_root, load_stim_detection
import csv
import re
# ---------- TTL helpers for Intan USB-ADC and UA bundles ----------

def _npz_meta_dict(arr):
    if arr is None:
        return {}
    try:
        obj = arr.item() if hasattr(arr, "item") else arr
    except Exception:
        obj = arr
    if isinstance(obj, (bytes, str)):
        try:
            return json.loads(obj)
        except Exception:
            return {}
    return obj if isinstance(obj, dict) else {}

def _find_matrix_2d(z):
    # Try common keys, else any 2D array
    for k in ("adc", "board_adc_data", "data", "traces", "signal", "values", "digital_in"):
        if k in z.files and getattr(z[k], "ndim", 0) == 2:
            return z[k], k
    for k in z.files:
        a = z[k]
        if getattr(a, "ndim", 0) == 2:
            return a, k
    return None, None

def _find_names(z):
    for k in ("chan_names", "channel_names", "names", "labels", "channels"):
        if k in z.files:
            try:
                return [str(x) for x in z[k].tolist()]
            except Exception:
                pass
    return None

def _pick_fs_hz(meta, z):
    for k in ("fs_hz","fs","sampling_rate_hz","sampling_rate","sample_rate_hz","sample_rate"):
        if k in meta:
            try: return float(meta[k])
            except Exception: pass
    if "t" in z.files:
        t = np.asarray(z["t"], dtype=float)
        dt = np.median(np.diff(t))
        if dt > 0:
            return 1.0/dt
    return None

def _detect_rising_onsets(x, fs_hz, min_gap_ms=5.0, auto_threshold=True, threshold=None):
    x = np.asarray(x, dtype=float)
    if auto_threshold or threshold is None:
        p5, p95 = np.nanpercentile(x, [5, 95])
        threshold = 0.5*(p5+p95)
    hi = x > float(threshold)
    d  = np.diff(hi.astype(np.int8), prepend=0)
    idx = np.flatnonzero(d == +1)
    if idx.size == 0:
        return idx
    min_gap = int(round(min_gap_ms*fs_hz/1000.0))
    keep = [idx[0]]
    for k in idx[1:]:
        if (k - keep[-1]) >= min_gap:
            keep.append(k)
    return np.asarray(keep, dtype=np.int64)

def _group_into_blocks(onsets_samp, fs_hz, gap_ms=200.0):
    if onsets_samp.size == 0:
        return onsets_samp
    gap = int(round(gap_ms*fs_hz/1000.0))
    starts = [onsets_samp[0]]
    for a,b in zip(onsets_samp[:-1], onsets_samp[1:]):
        if (b - a) > gap:
            starts.append(b)
    return np.asarray(starts, dtype=np.int64)

def load_stim_ms_from_intan_usb_adc(
    usb_adc_npz: Path,
    channel_name_hint: str | None = "stim",   # use substring match if names exist
    channel_index: int | None = None,         # set directly if you know it
    group_to_blocks: bool = True,
    block_gap_ms: float = 200.0,
) -> np.ndarray:
    """
    From .../NPRW/<SESSION>_Intan_bundle/USB_board_ADC_input_channel.npz
    -> rising-edge TTL onsets (ms). If group_to_blocks=True, returns block starts.
    """
    with np.load(usb_adc_npz, allow_pickle=True) as z:
        meta = _npz_meta_dict(z["meta"]) if "meta" in z.files else {}

        # Prefer the analog USB ADC matrix if present
        if "board_adc_data" in z.files and getattr(z["board_adc_data"], "ndim", 0) == 2:
            mat, key = z["board_adc_data"], "board_adc_data"
        else:
            mat, key = _find_matrix_2d(z)
        if mat is None:
            raise KeyError(f"No 2D matrix in {usb_adc_npz}; keys={z.files}")
        if mat.shape[0] < mat.shape[1]:
            adc = mat
        else:
            adc = mat.T
        names = _find_names(z) or [f"ADC{i}" for i in range(adc.shape[0])]
        fs_hz = _pick_fs_hz(meta, z)
        if not (fs_hz and np.isfinite(fs_hz)):
            raise ValueError(f"Sampling rate not found in {usb_adc_npz}")

        # choose channel
        if channel_index is not None:
            ch = int(channel_index)
        elif channel_name_hint is not None:
            low = [s.lower() for s in names]
            hit = [i for i,s in enumerate(low) if channel_name_hint.lower() in s]
            ch  = hit[0] if hit else int(np.argmax([np.percentile(adc[i],95)-np.percentile(adc[i],5) for i in range(adc.shape[0])]))
        else:
            ch  = int(np.argmax([np.percentile(adc[i],95)-np.percentile(adc[i],5) for i in range(adc.shape[0])]))

        onsets = _detect_rising_onsets(adc[ch], fs_hz)
        if group_to_blocks:
            onsets = _group_into_blocks(onsets, fs_hz, gap_ms=block_gap_ms)
        stim_ms = onsets.astype(float)*(1000.0/fs_hz)
        return np.unique(np.sort(stim_ms))

def load_stim_ms_from_ua_bundle(
    ua_npz: Path,
    channel_name_hint: str | None = "ttl",
    channel_index: int | None = None,
    group_to_blocks: bool = True,
    block_gap_ms: float = 200.0,
) -> np.ndarray:
    """
    From .../UA/NRR_RW_001_<IDX>_UA_bundle.npz -> rising-edge TTL onsets (ms).
    """
    with np.load(ua_npz, allow_pickle=True) as z:
        meta = _npz_meta_dict(z["meta"]) if "meta" in z.files else {}

        # Prefer digital TTLs if present
        if "digital_in" in z.files and getattr(z["digital_in"], "ndim", 0) == 2:
            mat, key = z["digital_in"], "digital_in"
        else:
            mat, key = _find_matrix_2d(z)
        if mat is None:
            raise KeyError(f"No 2D matrix in {ua_npz}; keys={z.files}")

        # shape -> (n_chan, n_samp)
        adc = mat if mat.shape[0] < mat.shape[1] else mat.T
        names = _find_names(z) or [f"CH{i}" for i in range(adc.shape[0])]
        fs_hz = _pick_fs_hz(meta, z)
        if not (fs_hz and np.isfinite(fs_hz)):
            raise ValueError(f"Sampling rate not found in {ua_npz}")

        if channel_index is not None:
            ch = int(channel_index)
        elif channel_name_hint is not None:
            low = [s.lower() for s in names]
            hit = [i for i,s in enumerate(low) if channel_name_hint.lower() in s or "stim" in s]
            ch  = hit[0] if hit else int(np.argmax([np.percentile(adc[i],95)-np.percentile(adc[i],5) for i in range(adc.shape[0])]))
        else:
            ch  = int(np.argmax([np.percentile(adc[i],95)-np.percentile(adc[i],5) for i in range(adc.shape[0])]))

        onsets = _detect_rising_onsets(adc[ch], fs_hz)
        if group_to_blocks:
            onsets = _group_into_blocks(onsets, fs_hz, gap_ms=block_gap_ms)
        stim_ms = onsets.astype(float)*(1000.0/fs_hz)
        return np.unique(np.sort(stim_ms))

def read_intan_to_br_map(csv_path: Path) -> dict[int, int]:
    """
    Read NRR_RW001_metadata.csv and return {Intan_File -> BR_File},
    robust to Excel encodings and header variations.
    """
    import io, codecs, csv, re

    def norm(s: str) -> str:
        # lower + remove non-alnum (spaces/underscores/hyphens vanish)
        return re.sub(r"[^a-z0-9]", "", s.lower())

    # --- robust decode the file ---
    raw = csv_path.read_bytes()
    try_order = []
    if raw.startswith(codecs.BOM_UTF8):
        try_order = ["utf-8-sig"]
    elif raw.startswith(codecs.BOM_UTF16_LE):
        try_order = ["utf-16-le"]
    elif raw.startswith(codecs.BOM_UTF16_BE):
        try_order = ["utf-16-be"]
    try_order += ["utf-8", "cp1252", "latin1", "utf-16"]
    text = None
    for enc in try_order:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("latin1", errors="replace")

    rdr = csv.DictReader(io.StringIO(text))
    fieldnames = [f for f in (rdr.fieldnames or []) if f is not None]
    fieldmap = {norm(k): k for k in fieldnames}

    # Accept common aliases for each logical field
    aliases = {
        "br_file":     ["br_file", "brfile", "br", "brindex", "brfileindex"],
        "intan_file":  ["intan_file", "intanfile", "intan", "intanindex", "intanfileindex"],
    }

    resolved: dict[str, str] = {}
    for logical, cands in aliases.items():
        for c in cands:
            if c in fieldmap:
                resolved[logical] = fieldmap[c]
                break
        if logical not in resolved:
            raise KeyError(
                f"CSV missing required column like '{logical}' "
                f"(have headers: {rdr.fieldnames})"
            )

    mapping: dict[int, int] = {}
    for row in rdr:
        try:
            intan_idx = int(str(row[resolved["intan_file"]]).strip())
            br_idx    = int(str(row[resolved["br_file"]]).strip())
        except Exception:
            continue
        mapping[intan_idx] = br_idx

    if not mapping:
        raise ValueError(f"No rows parsed from {csv_path}")

    print(f"[UA-map] Header mapping: Intan_File='{resolved['intan_file']}', BR_File='{resolved['br_file']}'")
    return mapping

def parse_intan_session_dtkey(session: str) -> int:
    """
    Convert 'NRR_RW001_YYMMDD_HHMMSS' -> sortable integer key YYMMDDHHMMSS.
    Fallback to lexical sort if pattern not found.
    """
    m = re.search(r"(\d{6})_(\d{6})$", session)
    if not m:
        return int("9"*12)  # push unknowns to end
    return int(m.group(1) + m.group(2))

def build_session_index_map(intan_sessions: list[str]) -> dict[str, int]:
    """
    Given unique NPRW session names, return {session -> Intan_File index (1-based)}
    using timestamp sort so index aligns with acquisition order used in your CSV.
    """
    ordered = sorted(intan_sessions, key=parse_intan_session_dtkey)
    return {sess: i+1 for i, sess in enumerate(ordered)}

def find_ua_rates_by_index(ua_root: Path, br_idx: int) -> Path | None:
    """
    Find the UA rates file for BR index (e.g., 7 -> 'rates__NRR_RW_001_007__*.npz').
    Prefer sigma25ms if multiple; else pick highest sigma.
    """
    patt = f"rates__NRR_RW_001_{br_idx:03d}__*.npz"
    candidates = sorted(ua_root.glob(patt))
    if not candidates:
        return None
    pref = [p for p in candidates if "__sigma25ms" in p.stem]
    if pref:
        return pref[0]
    def sigma_from_name(p: Path) -> int:
        m = re.search(r"sigma(\d+)ms", p.stem)
        return int(m.group(1)) if m else -1
    candidates.sort(key=sigma_from_name, reverse=True)
    return candidates[0]

def inspect_npz_fields(npz_path: Path) -> None:
    """Print a concise summary of arrays and a few meta fields (if present)."""
    with np.load(npz_path, allow_pickle=True) as z:
        print(f"[UA] Inspecting {npz_path.name}:")
        for key in z.files:
            val = z[key]
            if isinstance(val, np.ndarray):
                print(f"   - {key}: shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"   - {key}: type={type(val)}")
        if "meta" in z.files:
            meta = z["meta"]
            try:
                meta_obj = meta.item() if hasattr(meta, "item") else meta
            except Exception:
                meta_obj = meta
            if isinstance(meta_obj, (bytes, str)):
                try:
                    import json as _json
                    meta_obj = _json.loads(meta_obj)
                except Exception:
                    pass
            if isinstance(meta_obj, dict):
                keys = ["bin_ms", "sigma_ms", "fs", "fs_hz", "n_channels", "session"]
                sel = {k: meta_obj.get(k) for k in keys if k in meta_obj}
                print(f"   - meta (selected): {sel if sel else '(dict)'}")
            else:
                print("   - meta: (unparsed)")

def load_rate_npz(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    rate_hz = d["rate_hz"]           # (n_ch, n_bins_total)
    t_ms    = d["t_ms"]              # (n_bins_total,)
    meta    = d.get("meta", None)
    return rate_hz, t_ms, (meta.item() if hasattr(meta, "item") else meta)

def load_stim_ms_from_stimstream(stim_npz_path: Path) -> np.ndarray:
    """
    Load stimulation *block onset times* in ms from stim_stream.npz.
    Uses 'block_bounds_samples[:,0]' (Intan sample indices) and converts
    to milliseconds using meta['fs_hz'].
    """
    with np.load(stim_npz_path, allow_pickle=False) as z:
        if "block_bounds_samples" not in z or "meta" not in z:
            raise KeyError(f"{stim_npz_path} missing 'block_bounds_samples' or 'meta'.")

        blocks = z["block_bounds_samples"].astype(np.int64)  # (n_blocks, 2)
        if blocks.size == 0:
            raise ValueError(f"No block boundaries found in {stim_npz_path}.")

        # Parse meta
        meta_json = z["meta"].item() if hasattr(z["meta"], "item") else z["meta"]
        meta = json.loads(meta_json)
        fs_hz = float(meta.get("fs_hz"))
        if not np.isfinite(fs_hz):
            raise ValueError("meta['fs_hz'] missing or non-finite in stim_stream.npz")

        # Use block starts only
        onset_samples = blocks[:, 0]
        t_ms = onset_samples.astype(float) * (1000.0 / fs_hz)
        return np.unique(np.sort(t_ms))

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

def baseline_zero_each_trial(
    segments: np.ndarray,
    rel_time_ms: np.ndarray,
    baseline_first_ms: float = 100.0,
):
    """
    For each trial & channel, subtract the mean over the first `baseline_first_ms`
    of the segment (i.e., from window start to start+baseline_first_ms).
    segments: (n_trials, n_ch, n_twin)
    """
    # mask for first 100 ms of the segment window (e.g., [-800, -700))
    t0 = rel_time_ms[0]
    mask = (rel_time_ms >= t0) & (rel_time_ms < t0 + baseline_first_ms)
    if mask.sum() < 1:
        raise ValueError("Baseline window has 0 bins — check your timebase and dt.")
    base = segments[:, :, mask].mean(axis=2, keepdims=True)  # (n_trials, n_ch, 1)
    return segments - base

def average_across_trials(zeroed_segments: np.ndarray):
    """
    Average across trials per channel.
    Input: (n_trials, n_ch, n_twin) -> returns (n_ch, n_twin)
    """
    return zeroed_segments.mean(axis=0)

def plot_channel_heatmap(
    avg_change: np.ndarray,           # (n_ch, n_twin)
    rel_time_ms: np.ndarray,          # (n_twin,)
    out_png: Path,
    n_trials: int,
    title: str = "Peri-stim avg Δ rate (baseline=first 100 ms)",
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
):
    plt.figure(figsize=(12, 6))
    plt.imshow(
        avg_change,
        aspect="auto",
        cmap=cmap,
        extent=[rel_time_ms[0], rel_time_ms[-1], avg_change.shape[0], 0],  # <-- ms
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label="Δ Firing rate (Hz)")
    plt.xlabel("Time (ms) rel. stim")   # <-- ms axis label
    plt.ylabel("Channel index")
    plt.title(f"{title} (n={n_trials} trials)")  # <-- include trial count
    plt.axvline(0.0, color="k", alpha=0.8, linewidth=1.2)
    plt.axvspan(0.0, 100.0, color="gray", alpha=0.2, zorder=2)  # shaded 0–100 ms region
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved heatmap -> {out_png}")

def run_one_file(
    npz_path: Path,
    out_dir: Path,
    win_ms=(-800.0, 1200.0),
    baseline_first_ms=100.0,
    min_trials=1,
    save_npz=True,
    stim_ms: np.ndarray | None = None,
    debug_channel: int | None = None,    # <-- NEW
):
    rate_hz, t_ms, _ = load_rate_npz(npz_path)
    stim_ms = np.asarray(stim_ms, dtype=float).ravel()

    segments, rel_time_ms = extract_peristim_segments(
        rate_hz=rate_hz, t_ms=t_ms, stim_ms=stim_ms, win_ms=win_ms, min_trials=min_trials
    )

    zeroed = baseline_zero_each_trial(
        segments=segments, rel_time_ms=rel_time_ms, baseline_first_ms=baseline_first_ms
    )

    # -------- DEBUG: single-channel quick look --------
    if debug_channel is not None:
        # print some shapes/numbers to stdout
        dt = float(np.median(np.diff(t_ms)))
        print(f"[debug] segments shape={segments.shape} (trials, ch, timebins); dt={dt} ms")
        print(f"[debug] rel_time_ms: {rel_time_ms[0]} .. {rel_time_ms[-1]} (n={rel_time_ms.size})")
        out_base = (Path(out_dir) / npz_path.stem)
        plot_debug_channel_traces(zeroed, rel_time_ms, debug_channel, out_base)
    # ---------------------------------------------------

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = npz_path.stem
    avg_change = average_across_trials(zeroed)
    n_trials = segments.shape[0]   # number of kept trials

    out_png = out_dir / f"{stem}__peri_stim_heatmap.png"
    plot_channel_heatmap(avg_change, rel_time_ms, out_png,
                        n_trials=n_trials,
                        vmin=-500, vmax=1000)
    if save_npz:
        out_npz = out_dir / f"{stem}__peri_stim_avg.npz"
        np.savez_compressed(
            out_npz,
            avg_change=avg_change.astype(np.float32),
            rel_time_ms=rel_time_ms.astype(np.float32),
            meta=dict(
                source_file=str(npz_path),
                win_ms=win_ms,
                baseline_first_ms=baseline_first_ms,
                n_trials=int(segments.shape[0]),
            ),
        )
        print(f"Saved averaged peri-stim data -> {out_npz}")

def plot_debug_channel_traces(
    zeroed_segments: np.ndarray,   # (n_trials, n_ch, n_twin)
    rel_time_ms: np.ndarray,       # (n_twin,)
    ch: int,
    out_png_base: Path,
):
    n_trials, n_ch, n_twin = zeroed_segments.shape
    assert 0 <= ch < n_ch, f"debug channel {ch} out of range [0, {n_ch-1}]"

    y = zeroed_segments[:, ch, :]  # (n_trials, n_twin)

    # 1) overlay all trials + mean (x-axis in ms)
    plt.figure(figsize=(10, 5))
    for i in range(n_trials):
        plt.plot(rel_time_ms, y[i], alpha=0.25, linewidth=0.8)
    plt.plot(rel_time_ms, y.mean(axis=0), linewidth=2.0, label="mean")
    plt.axvline(0.0, color="k", alpha=0.6, linewidth=1.0)
    plt.xlabel("Time (ms) rel. stim")
    plt.ylabel("Δ Firing rate (Hz)")
    plt.title(f"Channel {ch}: peri-stim trials (n={n_trials})")
    plt.legend(loc="upper right")
    plt.tight_layout()
    out_png = out_png_base.with_name(out_png_base.stem + f"__ch{ch:03d}_overlay.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[debug] Saved overlay for ch {ch} -> {out_png}")

    # 2) trial-by-time image (x-axis in ms)
    plt.figure(figsize=(10, 5))
    plt.imshow(
        y, aspect="auto", cmap="hot",
        extent=[rel_time_ms[0], rel_time_ms[-1], n_trials, 0],
        origin="upper"
    )
    plt.colorbar(label="Δ Firing rate (Hz)")
    plt.axvline(0.0, color="k", alpha=0.6, linewidth=1.0)
    plt.xlabel("Time (ms) rel. stim")
    plt.ylabel("Trial index")
    plt.title(f"Channel {ch}: trials × time")
    plt.tight_layout()
    out_png2 = out_png_base.with_name(out_png_base.stem + f"__ch{ch:03d}_trials.png")
    plt.savefig(out_png2, dpi=150)
    plt.close()
    print(f"[debug] Saved trials×time for ch {ch} -> {out_png2}")


if __name__ == "__main__":
    # ==============================
    # Config & roots
    # ==============================
    REPO_ROOT = Path(__file__).resolve().parents[1]
    PARAMS = load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
    OUT_BASE = resolve_output_root(PARAMS)
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    nprw_ckpt_root   = OUT_BASE / "checkpoints" / "NPRW"
    ua_ckpt_root     = OUT_BASE / "checkpoints" / "UA"
    nprw_bundles     = OUT_BASE / "bundles" / "NPRW"
    ua_bundles       = OUT_BASE / "bundles" / "UA"
    figs_nprw_root   = OUT_BASE / "figures" / "peri_stim" / "NPRW"
    figs_ua_root     = OUT_BASE / "figures" / "peri_stim" / "UA"
    figs_nprw_root.mkdir(parents=True, exist_ok=True)
    figs_ua_root.mkdir(parents=True, exist_ok=True)

    # Metadata CSV: .../Nike/NRR_RW001/Metadata/NRR_RW001_metadata.csv
    metadata_csv = OUT_BASE.parent / "Metadata" / "NRR_RW001_metadata.csv"
    if metadata_csv.exists():
        try:
            intan_to_br = read_intan_to_br_map(metadata_csv)
            print(f"[UA-map] Loaded Intan->BR map ({len(intan_to_br)} rows).")
        except Exception as e:
            print(f"[UA-map][warn] Could not read mapping: {e}")
            intan_to_br = None
    else:
        print(f"[UA-map][warn] Not found: {metadata_csv}")
        intan_to_br = None

    # Build NPRW session -> sequential Intan index (1-based), ordered by timestamp
    def session_from_rates_path(p: Path) -> str:
        stem = p.stem  # rates__NRR_RW001_250915_160649__bin1ms_sigma25ms
        body = stem[len("rates__"):]
        return body.split("__bin", 1)[0]

    all_sessions = sorted({
        session_from_rates_path(p) for p in nprw_ckpt_root.rglob("rates__*.npz")
    })
    session_to_intan_idx = build_session_index_map(all_sessions)

    # Gather NPRW rate files
    rate_files = sorted(nprw_ckpt_root.rglob("rates__*.npz"))
    if not rate_files:
        raise SystemExit(f"[error] No rates__*.npz under {nprw_ckpt_root}")

    n_ok, n_skip = 0, 0
    print(f"[info] Found {len(rate_files)} NPRW rate files. Processing...")

    for i, rates_npz in enumerate(rate_files, 1):
        try:
            session = session_from_rates_path(rates_npz)

            # -------- INTAN: stim onsets from USB ADC bundle --------
            usb_adc_npz = nprw_bundles / f"{session}_Intan_bundle" / "USB_board_ADC_input_channel.npz"
            if not usb_adc_npz.exists():
                print(f"[warn] No USB ADC bundle for {session}: {usb_adc_npz} -> skipping INTAN plot")
            else:
                stim_ms_intan = load_stim_ms_from_intan_usb_adc(
                    usb_adc_npz,
                    channel_name_hint="stim",   # tweak if your name differs
                    group_to_blocks=True,
                    block_gap_ms=200.0,
                )
                print(f"[NPRW] {session}: {stim_ms_intan.size} stim onsets (Intan USB-ADC)")
                run_one_file(
                    npz_path=rates_npz,
                    out_dir=figs_nprw_root,
                    win_ms=(-800.0, 1200.0),
                    baseline_first_ms=100.0,
                    min_trials=1,
                    save_npz=True,
                    stim_ms=stim_ms_intan,
                    debug_channel=(0 if i == 1 else None),
                )

            # -------- UA: map Intan->BR, load UA rates + UA bundle TTLs, plot --------
            if intan_to_br is None:
                print(f"[UA-map][skip] No mapping loaded; skipping UA for {session}")
            else:
                intan_idx = session_to_intan_idx.get(session)
                if intan_idx is None:
                    print(f"[UA-map][warn] No Intan index for {session}")
                else:
                    br_idx = intan_to_br.get(intan_idx)
                    if br_idx is None:
                        print(f"[UA-map][warn] Intan_File {intan_idx} not mapped to a BR_File")
                    else:
                        # UA rates file (choose sigma25ms if multiple)
                        ua_rates = None
                        patt = f"rates__NRR_RW_001_{br_idx:03d}__*.npz"
                        cands = sorted(ua_ckpt_root.glob(patt))
                        if cands:
                            pref = [p for p in cands if "__sigma25ms" in p.stem]
                            ua_rates = (pref[0] if pref else cands[-1])
                        if ua_rates is None:
                            print(f"[UA-map][warn] No UA rates for BR_File {br_idx:03d} in {ua_ckpt_root}")
                        else:
                            ua_bundle = ua_bundles / f"NRR_RW_001_{br_idx:03d}_UA_bundle.npz"
                            if not ua_bundle.exists():
                                print(f"[UA][warn] UA bundle not found: {ua_bundle} — skipping UA peri-stim")
                            else:
                                stim_ms_ua = load_stim_ms_from_ua_bundle(
                                    ua_bundle,
                                    channel_name_hint="ttl",   # or "stim" if that matches your names
                                    group_to_blocks=True,
                                    block_gap_ms=200.0,
                                )
                                print(f"[UA] BR_File {br_idx:03d}: {stim_ms_ua.size} stim onsets (UA bundle)")
                                run_one_file(
                                    npz_path=ua_rates,
                                    out_dir=figs_ua_root,
                                    win_ms=(-800.0, 1200.0),
                                    baseline_first_ms=100.0,
                                    min_trials=1,
                                    save_npz=True,
                                    stim_ms=stim_ms_ua,
                                    debug_channel=None,
                                )

            n_ok += 1

        except Exception as e:
            print(f"[error] Failed on {rates_npz}: {e}")
            n_skip += 1

    print(f"[done] processed={n_ok}, skipped={n_skip}, figs at: {figs_nprw_root} and {figs_ua_root}")
