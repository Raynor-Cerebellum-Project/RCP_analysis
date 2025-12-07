from pathlib import Path
import random
import numpy as np
import spikeinterface.extractors as se
import RCP_analysis as rcp

"""
Extract Utah (Blackrock) array traces from all .ns6 files under the Blackrock folder
and save them as compressed `.npz` files containing:
  - `utah_traces`: (n_channels, n_samples) float32
  - `channel_ids`: list of channel id strings
  - `fs_hz`: sampling frequency
  - `dtype`: data dtype
  - `meta_path`: original ns6 path

Usage: run from repo root with the same environment used for other scripts.
"""


REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
BR_ROOT = SESSION_LOC / "Blackrock"
OUT_DIR = SESSION_LOC / "results" / "utah_traces"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_ns6_random_segment(ns_path: Path, out_dir: Path, n_ch: int = 3, seg_sec: float = 1.0, max_tries: int = 30):
    """
    Read a .ns6/.ns5 file and extract `n_ch` random channels with a segment of
    approximately `seg_sec` seconds chosen randomly. Try up to `max_tries` to
    find a non-flat segment; otherwise accept the last attempt.
    """
    try:
        rec = se.read_blackrock(ns_path)
    except Exception as e:
        print(f"[skip] failed to open {ns_path}: {e}")
        return False

    try:
        fs = rec.get_sampling_frequency()
    except Exception:
        fs = float('nan')

    try:
        n_frames = rec.get_num_frames()
    except Exception:
        print(f"[error] could not determine number of frames for {ns_path}")
        return False

    channel_ids = list(rec.get_channel_ids())
    if len(channel_ids) < n_ch:
        print(f"[warn] file {ns_path} has only {len(channel_ids)} channels, requested {n_ch}; will use available channels.")
        n_ch = len(channel_ids)

    seg_len = max(1, int(round(seg_sec * (fs if fs == fs else 30000.0))))
    max_start = max(0, n_frames - seg_len)

    chosen_segment = None
    chosen_chs = None
    last_err = None
    for attempt in range(max_tries):
        start = random.randint(0, max_start) if max_start > 0 else 0
        chs = random.sample(channel_ids, k=n_ch)
        try:
            # read only requested channels and time window
            seg = rec.get_traces(start_frame=start, end_frame=min(start + seg_len, n_frames), channel_ids=chs, return_in_uV=True)
            seg = np.asarray(seg, dtype=np.float32).T  # (n_channels, n_samples)
        except Exception as e:
            last_err = e
            seg = None

        if seg is None:
            continue

        # measure activity: median std across channels
        stds = np.std(seg, axis=1)
        med_std = float(np.median(stds))
        if med_std > 0.0:
            chosen_segment = seg
            chosen_chs = chs
            chosen_start = start
            break

    if chosen_segment is None:
        if last_err is not None:
            print(f"[warn] all attempts failed to read segment for {ns_path}: last error: {last_err}")
        else:
            print(f"[warn] could not find non-flat 1s segment for {ns_path}; saving final attempt (may be flat)")
        try:
            start = 0
            chs = channel_ids[:n_ch]
            seg = rec.get_traces(start_frame=start, end_frame=min(start + seg_len, n_frames), channel_ids=chs, return_in_uV=True)
            seg = np.asarray(seg, dtype=np.float32).T
            chosen_segment = seg
            chosen_chs = chs
            chosen_start = start
        except Exception as e:
            print(f"[error] final attempt failed for {ns_path}: {e}")
            return False

    out_npz = out_dir / f"{ns_path.stem}__utah_random_ns6.npz"
    np.savez_compressed(
        out_npz,
        utah_traces=chosen_segment,  # (n_channels, n_samples)
        channel_ids=np.array(chosen_chs, dtype=object),
        fs_hz=float(fs),
        start_sample=int(chosen_start),
        duration_samples=int(chosen_segment.shape[1]),
        meta_path=str(ns_path),
    )
    print(f"[saved] {out_npz}  (channels={chosen_segment.shape[0]}, samples={chosen_segment.shape[1]})")
    return True


def main():
    if not BR_ROOT.exists():
        raise SystemExit(f"[error] Blackrock folder not found: {BR_ROOT}")

    ns6_files = sorted(BR_ROOT.rglob("*.ns6"))
    if ns6_files:
        files = ns6_files
        print(f"Found {len(files)} .ns6 files; extracting random 1s segments to {OUT_DIR}")
    else:
        ns5_files = sorted(BR_ROOT.rglob("*.ns5"))
        if not ns5_files:
            print(f"[info] No .ns6 or .ns5 files found under {BR_ROOT}")
            return
        files = ns5_files
        print(f"Found {len(files)} .ns5 files (no .ns6 present); extracting random 1s segments to {OUT_DIR}")

    for ns in files:
        save_ns6_random_segment(ns, OUT_DIR, n_ch=3, seg_sec=1.0, max_tries=30)


if __name__ == "__main__":
    main()
