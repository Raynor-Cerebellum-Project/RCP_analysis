from __future__ import annotations
from pathlib import Path
import re
from typing import Dict
import argparse
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
def load_dlc(dlc_csv: Path) -> pd.DataFrame:
    """
    Load DLC CSV (3 header rows). If the first column is 'frame' (in any header
    level), use it as the AVI_framenum index; otherwise use 0..N-1.
    Columns are flattened and cast to numeric where possible.
    """
    df = pd.read_csv(dlc_csv, header=[0, 1, 2])

    first = df.columns[0]
    levels = [first] if not isinstance(first, tuple) else list(first)
    levels_lc = [str(x).strip().lower() for x in levels]

    if any(x == "frame" for x in levels_lc):
        frame_col = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        df = df.drop(columns=[df.columns[0]])
        idx = pd.Index(np.rint(frame_col.values).astype("Int64"), name="AVI_framenum")
    else:
        first0 = str(levels[0]).lower()
        if first0 in ("scorer", "index", ""):
            df = df.drop(columns=[df.columns[0]])
        idx = pd.RangeIndex(len(df), name="AVI_framenum")

    flat_cols = []
    for col in df.columns.values:
        if isinstance(col, tuple):
            parts = [str(x) for x in col if str(x) != "nan"]
        else:
            parts = [str(col)]
        flat_cols.append("_".join(parts).strip())
    df.columns = flat_cols

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.index = idx
    return df


def expand_dlc_to_corrected(dlc_df: pd.DataFrame, ocr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map DLC rows (indexed by AVI_framenum) into a DataFrame indexed by
    CORRECTED_framenum. Dropped frames become NaNs.
    """
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


# -------------------------- Discovery --------------------------
_DLC_RE = re.compile(
    r'^(?P<trial>NRR_[A-Za-z]+[0-9]{3}_[0-9]{3})_Cam-(?P<cam>[01])DLC.*\.csv$',
    re.IGNORECASE
)

def _trial_cam_from_dlc_name(p: Path):
    m = _DLC_RE.match(p.name)
    if not m:
        return None, None
    return m.group("trial"), int(m.group("cam"))

def _find_per_trial_inputs_one_or_both(video_root: Path) -> Dict[str, dict]:
    """
    Returns:
      { trial: { 'ocr': {cam: Path}, 'dlc': {cam: Path} } }
    Keeps newest file by mtime if duplicates exist.
    Unlike the 2-cam version, this does NOT require both cams to exist.
    """
    per: Dict[str, dict] = {}

    # OCR files: e.g., NRR_RW003_001_Cam-0_ocr.csv
    for p in video_root.rglob("*_ocr.csv"):
        name = p.name
        try:
            base, tail = name.split("_Cam-", 1)
            cam = int(tail.split("_", 1)[0])
            trial = base
        except Exception:
            continue
        d = per.setdefault(trial, {"ocr": {}, "dlc": {}})
        prev = d["ocr"].get(cam)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            d["ocr"][cam] = p

    # DLC files: e.g., NRR_*_Cam-0DLC*.csv
    for p in video_root.rglob("NRR_*_Cam-[01]DLC*.csv"):
        trial, cam = _trial_cam_from_dlc_name(p)
        if trial is None:
            continue
        d = per.setdefault(trial, {"ocr": {}, "dlc": {}})
        prev = d["dlc"].get(cam)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            d["dlc"][cam] = p

    return per

def _find_ns5_for_trial(br_root: Path, trial: str) -> Path | None:
    hits = list(br_root.rglob("*.ns5"))
    hits += list(br_root.rglob("*.ns6"))
    trial_low = trial.lower()
    matches = [p for p in hits if trial_low in p.stem.lower()]
    if not matches:
        return None
    return sorted(matches)[0]

# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser(description="Align a single camera's DLC to OCR (and optionally BR time).")
    ap.add_argument("--cam", type=int, choices=(0, 1), default=int(getattr(PARAMS, "video_cam", 0)),
                    help="Which camera to process (0 or 1). Default comes from PARAMS.video_cam or 0.")
    args = ap.parse_args()
    CAM = int(args.cam)

    print(f"[scan] VIDEO_ROOT={VIDEO_ROOT}")
    print(f"[scan] BR_ROOT={BR_ROOT}")
    per_trial = _find_per_trial_inputs_one_or_both(VIDEO_ROOT)
    if not per_trial:
        print("[warn] No OCR/DLC files found.")
        return

    # Keep only trials that have the requested cam for BOTH OCR and DLC
    valid = {
        trial: d for trial, d in per_trial.items()
        if (CAM in d.get("ocr", {})) and (CAM in d.get("dlc", {}))
    }

    if not valid:
        print(f"[warn] No trials with both OCR and DLC for Cam-{CAM}.")
        return

    print(f"[info] Found {len(valid)} trials with Cam-{CAM}.")

    for trial, d in sorted(valid.items()):
        print(f"\n=== Trial: {trial} (Cam-{CAM}) ===")

        # Load OCR + DLC for the selected cam
        ocr = load_ocr_map(d["ocr"][CAM])
        dlc = load_dlc(d["dlc"][CAM])

        # Expand to corrected index
        aligned = expand_dlc_to_corrected(dlc, ocr)

        # Optional: attach NS5 time if a file matches trial
        ns_path = _find_ns5_for_trial(BR_ROOT, trial)
        if ns_path is not None:
            try:
                n_corr = len(aligned.index)
                ns5_samples, _, _ = corrected_to_time_ns5(
                    n_corr, ns_path, sync_chan=int(getattr(PARAMS, "camera_sync_ch", 134))
                )
                aligned = aligned.reindex(range(n_corr))
                aligned.insert(0, "ns5_sample", ns5_samples)
                print(f"[map] Attached NS5 time from {ns_path.name}")
            except Exception as e:
                print(f"[warn] Could not attach NS5 time for {trial}: {e}")
        else:
            print(f"[warn] No BR file found for {trial}; skipping ns5_sample attachment.")

        # Save CSV for the single camera
        out_csv = BEHV_CKPT_ROOT / f"{trial}_Cam-{CAM}_aligned.csv"
        aligned.to_csv(out_csv, index_label="CORRECTED_framenum")
        print(f"[write] {out_csv}")

if __name__ == "__main__":
    main()
