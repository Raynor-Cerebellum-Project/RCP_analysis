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

METADATA_CSV  = (DATA_ROOT / PARAMS.metadata_rel).resolve(); METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)

# -------------------------- OCR helpers --------------------------

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

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

def _load_video_to_br_map(meta_csv: Path) -> dict[int, int]:
    """
    Returns { video_idx -> br_idx } from METADATA_CSV.
    Tolerates common encodings and small header-name variations.
    """
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1", "mac_roman")
    last_err = None
    for enc in encodings:
        try:
            import csv
            with meta_csv.open("r", newline="", encoding=enc, errors="strict") as f:
                rdr = csv.DictReader(f)
                cols = { _norm(c): c for c in (rdr.fieldnames or []) }

                def _col(*cands: str) -> str:
                    for c in cands:
                        k = _norm(c)
                        if k in cols:
                            return cols[k]
                    raise KeyError(f"Missing any of columns {cands}; found {list(cols.values())}")

                col_video = _col("video_file", "videofile", "video")
                col_br    = _col("br_file", "brfile", "br")

                out: dict[int, int] = {}
                for row in rdr:
                    try:
                        v_raw = str(row[col_video]).strip()
                        b_raw = str(row[col_br]).strip()
                        if not v_raw or not b_raw:
                            continue
                        v = int(float(v_raw))
                        b = int(float(b_raw))
                        out[v] = b  # last one wins if duplicated
                    except Exception:
                        continue
                if out and enc != "utf-8":
                    print(f"[info] Read {meta_csv.name} with encoding={enc}.")
                if not out:
                    print(f"[warn] No Video_File→BR_File pairs found in {meta_csv.name}.")
                return out
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not decode {meta_csv} with tried encodings; last error: {last_err}")


# Extract the video index (second 3-digit block), e.g. 'NRR_RW001_002' -> 2
_TRIAL_VIDEO_IDX_RE = re.compile(
    r'NRR_[A-Za-z]+(?:\d{3})_(\d{3})(?:_[0-9]{4}_[0-9]{4}_[0-9]{6})?(?:\b|_)',
    re.IGNORECASE
)

def _extract_video_idx_from_trial(trial: str) -> int | None:
    """
    Returns the Video_File index from names like:
      NRR_RW001_002
      NRR_RW001_002_2025_0915_141059
    -> 2  (i.e., the second 3-digit block)
    """
    m = _TRIAL_VIDEO_IDX_RE.search(trial)
    if m:
        return int(m.group(1))

    # Fallback: pick the last 3-digit token (dates are 4/4/6 digits, so they won't collide)
    nums = re.findall(r'(?<!\d)(\d{3})(?!\d)', trial)
    return int(nums[-1]) if nums else None

def _find_ns5_by_br_index(br_root: Path, br_idx: int) -> Path | None:
    """Return the best-matching .ns5 for a given BR index; never returns ns6."""
    hits = list(br_root.rglob("*.ns5"))
    if not hits:
        print(f"[warn] No .ns5 files found under {br_root}")
        return None
    token = f"{br_idx:03d}"

    def _score(p: Path) -> tuple[int, float]:
        # strong token match + newest mtime
        strong = int(bool(re.search(rf"(?:^|[^0-9]){token}(?:[^0-9]|$)", p.stem)))
        return (strong, p.stat().st_mtime)

    hits.sort(key=_score, reverse=True)
    best = hits[0]
    if token not in best.stem:
        print(f"[warn] No obvious .ns5 name match for BR {token}; using {best.name}")
    return best

def load_ocr_map(ocr_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(ocr_csv)
    col_avi = _find_col(df, ["AVI_framenum","avi_framenum","avi_frame","avi"])
    col_ocr = _find_col(df, ["OCR_framenum","ocr_framenum","ocr_frame","ocr"])
    col_cor = _find_col(df, ["CORRECTED_framenum","corrected_framenum","corrected_frame","corrected"])

    out = df[[col_avi, col_ocr, col_cor]].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    last = out.dropna(subset=[col_ocr, col_cor]).tail(1)
    if last.empty:
        raise ValueError(f"{ocr_csv} has no valid OCR rows.")
    try:
        last_ocr = int(np.rint(float(last[col_ocr].iloc[0])))
        last_cor = int(np.rint(float(last[col_cor].iloc[0])))
        if last_ocr != last_cor:
            print(f"[warn] {ocr_csv.name}: last OCR_framenum ({last_ocr}) != CORRECTED_framenum ({last_cor}); continuing.")
    except Exception:
        pass

    return out.rename(columns={col_avi:"AVI_framenum", col_ocr:"OCR_framenum", col_cor:"CORRECTED_framenum"})

# -------------------------- DLC helpers --------------------------
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
    ch_ids_str = list(map(str, ch_ids))
    target = str(chan_id)
    if target not in ch_ids_str:
        raise ValueError(
            f"Sync channel {chan_id} not in {ns_path.name}. "
            f"Available: {ch_ids_str}"
        )
    fs = float(rec.get_sampling_frequency())
    sig = rec.get_traces(channel_ids=[target]).astype(float).squeeze()
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
# Accept OCR like:
#   NRR_RW003_001_Cam-0_ocr.csv
#   NRR_RW001_002_2025_0915_141059_Cam-0_ocr.csv
_TRIAL_CAM_OCR_RE = re.compile(
    r'^(?P<trial>NRR_[A-Za-z]+[0-9]{3}_[0-9]{3})(?:_[0-9]{4}_[0-9]{4}_[0-9]{6})?_Cam-(?P<cam>[01])(?:_|-)?ocr\.csv$',
    re.IGNORECASE
)
def _trial_cam_from_ocr_name(p: Path):
    m = _TRIAL_CAM_OCR_RE.match(p.name)
    if not m:
        return None, None
    return m.group("trial"), int(m.group("cam"))

# Accept DLC like:
#   NRR_RW003_001_Cam-0DLC_Resnet50_....csv
#   NRR_RW001_002_2025_0915_141059_Cam-0DLC_Resnet50_....csv
_TRIAL_CAM_RE = re.compile(
    r'^(?P<trial>NRR_[A-Za-z]+[0-9]{3}_[0-9]{3})(?:_[0-9]{4}_[0-9]{4}_[0-9]{6})?_Cam-(?P<cam>[01])DLC',
    re.IGNORECASE
)
def _trial_cam_from_dlc_name(p: Path):
    m = _TRIAL_CAM_RE.match(p.name)
    if not m:
        return None, None
    return m.group("trial"), int(m.group("cam"))

def _find_per_trial_inputs(video_root: Path) -> Dict[str, dict]:
    per: Dict[str, dict] = {}

    # OCR files
    for p in video_root.rglob("*_ocr.csv"):
        trial, cam = _trial_cam_from_ocr_name(p)
        if trial is None:
            continue
        d = per.setdefault(trial, {"ocr": {}, "dlc": {}})
        prev = d["ocr"].get(cam)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            d["ocr"][cam] = p

    # DLC files
    for p in video_root.rglob("NRR_*_Cam-[01]DLC*.csv"):
        trial, cam = _trial_cam_from_dlc_name(p)
        if trial is None:
            continue
        d = per.setdefault(trial, {"ocr": {}, "dlc": {}})
        prev = d["dlc"].get(cam)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            d["dlc"][cam] = p

    # Keep only trials that have BOTH cams for BOTH OCR & DLC
    return {
        t: d for t, d in per.items()
        if set(d["ocr"].keys()) >= {0, 1} and set(d["dlc"].keys()) >= {0, 1}
    }

def _find_ns5_for_trial(br_root: Path, trial: str) -> Path | None:
    """Find a .ns5 whose stem contains the trial string; never returns ns6."""
    hits = list(br_root.rglob("*.ns5"))
    trial_low = trial.lower()
    matches = [p for p in hits if trial_low in p.stem.lower()]
    if not matches:
        print(f"[warn] No .ns5 matched trial '{trial}' in {br_root}")
        return None
    return sorted(matches)[0]


def main():
    print(f"[scan] VIDEO_ROOT={VIDEO_ROOT}")
    print(f"[scan] BR_ROOT={BR_ROOT}")
    per_trial = _find_per_trial_inputs(VIDEO_ROOT)
    if not per_trial:
        print("[warn] No complete trials with {Cam-0, Cam-1} × {OCR, DLC} found.")
        return

    print(f"[info] Found {len(per_trial)} trials.")

    # load once
    try:
        VIDEO_TO_BR = _load_video_to_br_map(METADATA_CSV)
    except Exception as e:
        print(f"[warn] Could not load metadata map ({METADATA_CSV.name}): {e}")
        VIDEO_TO_BR = {}
        
    for trial, d in sorted(per_trial.items()):
        print(f"\n=== Trial: {trial} ===")
        cam_files = d['dlc'].keys()

        # Load OCR and DLC
        ocr0 = load_ocr_map(d['ocr'][0]); ocr1 = load_ocr_map(d['ocr'][1])
        dlc0 = load_dlc(d['dlc'][0]);     dlc1 = load_dlc(d['dlc'][1])

        # Expand to corrected index
        e0 = expand_dlc_to_corrected(dlc0, ocr0)
        e1 = expand_dlc_to_corrected(dlc1, ocr1)

        # per-trial
        vid_idx = _extract_video_idx_from_trial(trial)
        ns_path = None
        if vid_idx is not None and VIDEO_TO_BR:
            br_idx = VIDEO_TO_BR.get(vid_idx)
            if br_idx is not None:
                print(f"[pair] Video_File={vid_idx:03d} → BR_File={br_idx:03d} (from {METADATA_CSV.name})")
                ns_path = _find_ns5_by_br_index(BR_ROOT, br_idx)
                if ns_path is None:
                    print(f"[warn] BR_File {br_idx:03d} not found by index; fallback to trial-name search.")
        if ns_path is None:
            ns_path = _find_ns5_for_trial(BR_ROOT, trial)  # your existing fallback

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
