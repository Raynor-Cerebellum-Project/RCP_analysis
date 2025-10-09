from __future__ import annotations
from pathlib import Path
import re
from typing import Dict
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import RCP_analysis as rcp

# ---------- CONFIG ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
DATA_ROOT = rcp.resolve_data_root(PARAMS)
OUT_BASE  = rcp.resolve_output_root(PARAMS); OUT_BASE.mkdir(parents=True, exist_ok=True)

BR_ROOT    = (DATA_ROOT / PARAMS.blackrock_rel); BR_ROOT.mkdir(parents=True, exist_ok=True)
VIDEO_ROOT = (DATA_ROOT / PARAMS.video_rel);     VIDEO_ROOT.mkdir(parents=True, exist_ok=True)

BEHV_BUNDLES   = OUT_BASE / "bundles" / "behavior"
BEHV_CKPT_ROOT = OUT_BASE / "checkpoints" / "behavior"
BEHV_CKPT_ROOT.mkdir(parents=True, exist_ok=True)

# -------------------------- OCR helpers --------------------------

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find a column (case/underscore-insensitive) among candidates."""
    def norm(s: str) -> str:
        return re.sub(r'[^a-z0-9]+', '', s.lower())
    lookup = {norm(str(c)): c for c in df.columns}
    for name in candidates:
        key = norm(name)
        if key in lookup:
            return lookup[key]
    raise KeyError(f"Missing any of columns {candidates}; found {list(df.columns)[:6]} ...")


def load_ocr_map(ocr_csv: Path) -> pd.DataFrame:
    """
    Read an OCR CSV and return a DataFrame with columns:
      AVI_framenum, OCR_framenum, CORRECTED_framenum (numeric floats; may contain NaNs)
    Asserts last OCR_framenum == last CORRECTED_framenum (after rounding to ints).
    """
    df = pd.read_csv(ocr_csv)
    col_avi = _find_col(df, ["AVI_framenum","avi_framenum","avi_frame","avi"])
    col_ocr = _find_col(df, ["OCR_framenum","ocr_framenum","ocr_frame","ocr"])
    col_cor = _find_col(df, ["CORRECTED_framenum","corrected_framenum","corrected_frame","corrected"])

    out = df[[col_avi, col_ocr, col_cor]].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")  # keep as float with NaNs

    # assert on LAST row that has both OCR & CORRECTED, rounding to nearest int
    last = out.dropna(subset=[col_ocr, col_cor]).tail(1)
    if last.empty:
        raise ValueError(f"{ocr_csv} has no valid OCR rows.")
    last_ocr = int(np.rint(float(last[col_ocr].iloc[0])))
    last_cor = int(np.rint(float(last[col_cor].iloc[0])))
    if last_ocr != last_cor:
        raise AssertionError(
            f"{ocr_csv.name}: last OCR_framenum ({last_ocr}) != CORRECTED_framenum ({last_cor})"
        )

    return out.rename(columns={
        col_avi: "AVI_framenum",
        col_ocr: "OCR_framenum",
        col_cor: "CORRECTED_framenum",
    })


# -------------------------- DLC helpers --------------------------
def _flatten_cols(cols) -> list[str]:
    """Flatten DLC MultiIndex columns to strings, dropping 'nan' parts."""
    flat = []
    for col in cols:
        if isinstance(col, tuple):
            parts = [str(p) for p in col if p is not None and str(p) != "nan"]
            name = "_".join(parts).strip()
        else:
            name = str(col).strip()
        flat.append(name)
    return flat


def _first_column_is_avi(series: pd.Series) -> bool:
    """Heuristic: first column is AVI frame index if it's (nearly) integer and monotonic by 1."""
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().mean() > 0.05:
        return False
    diffs = np.diff(s.values.astype(float))
    frac_ok = np.mean(np.isclose(diffs, 1.0, atol=1e-6))
    return frac_ok > 0.98 and (s.iloc[0] in (0, 1))


def load_dlc(dlc_csv: Path) -> pd.DataFrame:
    """
    Load DLC CSV (3 header rows). If the first column is 'frame' (in any header
    level), use it as the AVI_framenum index; otherwise use 0..N-1.
    Columns are flattened and cast to numeric where possible.
    """
    df = pd.read_csv(dlc_csv, header=[0, 1, 2])

    # Detect a 'frame' first column (any header level says 'frame')
    first = df.columns[0]
    levels = [first] if not isinstance(first, tuple) else list(first)
    levels_lc = [str(x).strip().lower() for x in levels]

    if any(x == "frame" for x in levels_lc):
        frame_col = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        df = df.drop(columns=[df.columns[0]])
        idx = pd.Index(np.rint(frame_col.values).astype("Int64"), name="AVI_framenum")
    else:
        # Drop DLC “scorer/index” style column if present
        first0 = str(levels[0]).lower()
        if first0 in ("scorer", "index", ""):
            df = df.drop(columns=[df.columns[0]])
        idx = pd.RangeIndex(len(df), name="AVI_framenum")

    # Flatten multi-index columns
    flat_cols = []
    for col in df.columns.values:
        if isinstance(col, tuple):
            parts = [str(x) for x in col if str(x) != "nan"]
        else:
            parts = [str(col)]
        flat_cols.append("_".join(parts).strip())
    df.columns = flat_cols

    # Numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.index = idx
    return df


def expand_dlc_to_corrected(dlc_df: pd.DataFrame, ocr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map DLC rows (indexed by AVI_framenum) into a DataFrame indexed by
    CORRECTED_framenum. Dropped frames become NaNs.
    """
    # Round to nearest int when constructing mapping; NaNs are dropped
    map_df = ocr_df.dropna(subset=["AVI_framenum", "CORRECTED_framenum"]).copy()
    map_df["AVI_framenum"] = np.rint(map_df["AVI_framenum"].to_numpy()).astype(int)
    map_df["CORRECTED_framenum"] = np.rint(map_df["CORRECTED_framenum"].to_numpy()).astype(int)

    avi_to_corr = dict(zip(map_df["AVI_framenum"].values, map_df["CORRECTED_framenum"].values))

    max_corr = int(np.rint(ocr_df["CORRECTED_framenum"].dropna().max()))
    out = pd.DataFrame(
        index=pd.RangeIndex(0, max_corr + 1, name="CORRECTED_framenum"),
        columns=dlc_df.columns,
        dtype=float,
    )

    # place DLC rows at corrected positions
    # dlc_df.index is AVI_framenum (nullable int); coerce to ints where present
    for avi, row in dlc_df.iterrows():
        if pd.isna(avi):
            continue
        avi_int = int(np.rint(float(avi)))
        corr = avi_to_corr.get(avi_int)
        if corr is not None:
            out.loc[corr] = row.values

    return out

# -------------------------- BR helpers --------------------------

def detect_rising_edges(sig: np.ndarray) -> np.ndarray:
    """Fast hysteresis threshold rising-edge detector."""
    if sig.size == 0:
        return np.array([], dtype=np.int64)
    ds = sig[:: max(1, sig.size // 20000)]
    lo = 0.5 * (ds.min() + ds.max())
    hi = 0.5 * (lo + ds.max())
    edges, state = [], False
    for i, v in enumerate(sig):
        if not state and v >= hi:
            edges.append(i); state = True
        elif state and v <= lo:
            state = False
    return np.asarray(edges, dtype=np.int64)


def load_br_sync(ns_path: Path, chan_id: int):
    rec = se.read_blackrock(str(ns_path), all_annotations=True)
    ch_ids = list(rec.get_channel_ids())
    if str(chan_id) in map(str, ch_ids):
        channel_ids = [str(chan_id)]
    elif chan_id in ch_ids:
        channel_ids = [chan_id]
    else:
        raise ValueError(f"Sync channel {chan_id} not in {ns_path.name}. Available: {ch_ids}")
    fs = float(rec.get_sampling_frequency())
    sig = rec.get_traces(channel_ids=channel_ids).astype(float).squeeze()
    return sig, fs


def corrected_to_time_ns5(n_corrected: int, ns_path: Path, sync_chan: int):
    """Return (ns5_samples, t_sec, fs) arrays of length n_corrected, using rising edges."""
    sig, fs = load_br_sync(ns_path, sync_chan)
    edges = detect_rising_edges(sig)
    if edges.size < n_corrected:
        raise RuntimeError(f"NS5 rising edges ({edges.size}) fewer than corrected frames ({n_corrected}).")
    samples = edges[:n_corrected].astype(np.int64)
    t_sec = samples / float(fs)
    return samples, t_sec, fs


# -------------------------- Discovery & main --------------------------
# Matches: NRR_RW003_001_Cam-0DLC*.csv  → trial="NRR_RW003_001", cam=0
_DLC_RE = re.compile(
    r'^(?P<trial>NRR_[A-Za-z]+[0-9]{3}_[0-9]{3})_Cam-(?P<cam>[01])DLC.*\.csv$',
    re.IGNORECASE
)

def _trial_cam_from_dlc_name(p: Path):
    m = _DLC_RE.match(p.name)
    if not m:
        return None, None
    return m.group("trial"), int(m.group("cam"))

def _find_per_trial_inputs(video_root: Path) -> Dict[str, dict]:
    """
    Returns:
      { trial: { 'ocr': {0: Path, 1: Path}, 'dlc': {0: Path, 1: Path} } }
    Keeps the newest file by mtime if duplicates exist.
    """
    per: Dict[str, dict] = {}

    # ---- OCR files (assumes names like NRR_RW003_001_Cam-0_ocr.csv) ----
    for p in video_root.rglob("*_ocr.csv"):
        name = p.name
        # trial is the part before "_Cam-<d>_ocr"
        try:
            base, tail = name.split("_Cam-", 1)
            cam = int(tail.split("_", 1)[0])  # '0_ocr.csv' → 0
            trial = base
        except Exception:
            continue
        d = per.setdefault(trial, {"ocr": {}, "dlc": {}})
        d["ocr"][cam] = p

    # ---- DLC files (always start with NRR_*_Cam-<d>DLC...) ----
    for p in video_root.rglob("NRR_*_Cam-[01]DLC*.csv"):
        trial, cam = _trial_cam_from_dlc_name(p)
        if trial is None:
            continue
        d = per.setdefault(trial, {"ocr": {}, "dlc": {}})
        prev = d["dlc"].get(cam)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            d["dlc"][cam] = p

    # Keep only trials that have both cams for both OCR and DLCk
    return {
        trial: d for trial, d in per.items()
        if set(d["ocr"].keys()) >= {0, 1} and set(d["dlc"].keys()) >= {0, 1}
    }

def _find_ns5_for_trial(br_root: Path, trial: str) -> Path | None:
    # Try .ns5 first, then any NSx
    hits = list(br_root.rglob("*.ns5"))
    hits += list(br_root.rglob("*.ns6"))
    trial_low = trial.lower()
    matches = [p for p in hits if trial_low in p.stem.lower()]
    if not matches:
        return None
    # deterministic choice
    return sorted(matches)[0]

def main():
    print(f"[scan] VIDEO_ROOT={VIDEO_ROOT}")
    print(f"[scan] BR_ROOT={BR_ROOT}")
    per_trial = _find_per_trial_inputs(VIDEO_ROOT)
    if not per_trial:
        print("[warn] No complete trials with {Cam-0, Cam-1} × {OCR, DLC} found.")
        return

    print(f"[info] Found {len(per_trial)} trials.")

    for trial, d in sorted(per_trial.items()):
        print(f"\n=== Trial: {trial} ===")
        cam_files = d['dlc'].keys()

        # Load OCR and DLC
        ocr0 = load_ocr_map(d['ocr'][0]); ocr1 = load_ocr_map(d['ocr'][1])
        dlc0 = load_dlc(d['dlc'][0]);     dlc1 = load_dlc(d['dlc'][1])

        # Expand to corrected index
        e0 = expand_dlc_to_corrected(dlc0, ocr0)
        e1 = expand_dlc_to_corrected(dlc1, ocr1)

        # Optional: attach NS5 time if a file matches trial
        ns_path = _find_ns5_for_trial(BR_ROOT, trial)
        if ns_path is not None:
            try:
                n_corr = max(len(e0.index), len(e1.index))
                ns5_samples, _, _ = corrected_to_time_ns5(n_corr, ns_path,
                                          sync_chan=int(getattr(PARAMS, "camera_sync_ch", 134)))
                e0 = e0.reindex(range(n_corr)); e1 = e1.reindex(range(n_corr))
                e0.insert(0, "ns5_sample", ns5_samples);   e1.insert(0, "ns5_sample", ns5_samples)
                print(f"[map] Attached NS5 time from {ns_path.name}")
            except Exception as e:
                print(f"[warn] Could not attach NS5 time for {trial}: {e}")

        # Save combined CSV (side-by-side)
        e0_cam = pd.concat({"cam0": e0}, axis=1)
        e1_cam = pd.concat({"cam1": e1}, axis=1)
        both = pd.concat([e0_cam, e1_cam], axis=1)
        both_csv = BEHV_CKPT_ROOT / f"{trial}_both_cams_aligned.csv"
        both.to_csv(both_csv, index_label="CORRECTED_framenum")
        print(f"[write] {both_csv}")

if __name__ == "__main__":
    main()
