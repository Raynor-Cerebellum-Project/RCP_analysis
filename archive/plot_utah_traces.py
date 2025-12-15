from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import RCP_analysis as rcp

"""
Plot Utah traces saved by `extract_utah_traces.py`.

For each `*__utah_traces.npz` in the session `results/utah_traces` folder,
the script creates a stacked-trace PNG in `results/utah_traces/figs/`.

Usage:
    python .\batch_scripts\plot_utah_traces.py

Options inside the script can be changed to limit time window, channel subset,
or maximum points for plotting (it will downsample for display if necessary).
"""


REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
UTAH_DIR = SESSION_LOC / "results" / "utah_traces"
FIG_DIR = UTAH_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def downsample_trace(trace: np.ndarray, max_points: int) -> np.ndarray:
    n = trace.size
    if n <= max_points:
        return trace
    factor = int(np.ceil(n / float(max_points)))
    # simple decimation by reshaping (trim to multiple)
    trim = (n // factor) * factor
    t = trace[:trim].reshape(-1, factor)
    return t.mean(axis=1)


def plot_stacked(traces: np.ndarray, fs: float, out_path: Path, title: str = None,
                 channel_ids=None, t_start_s=0.0, t_dur_s=None, max_points=3000):
    # traces: (n_channels, n_samples)
    n_ch, n_samp = traces.shape

    start = int(round(t_start_s * fs))
    if t_dur_s is None:
        end = n_samp
    else:
        end = int(round((t_start_s + t_dur_s) * fs))
    end = min(end, n_samp)

    seg = traces[:, start:end]
    seg = np.asarray(seg, dtype=float)

    # Downsample by selecting uniform indices if there are too many points
    n_points = seg.shape[1]
    if n_points > max_points:
        idx = np.linspace(0, n_points - 1, num=max_points, dtype=int)
        seg_ds = seg[:, idx]
        t = idx.astype(float) / fs + t_start_s
    else:
        seg_ds = seg
        t = np.arange(seg.shape[1], dtype=float) / fs + t_start_s

    # vertical offsets so traces don't overlap
    med = np.median(np.abs(seg_ds))
    if med == 0:
        med = 1.0
    offset = med * 4.0

    fig_h = max(6.0, 0.12 * n_ch)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    for i in range(n_ch):
        ax.plot(t, seg_ds[i, :] + i * offset, color="k", linewidth=0.6)

    ax.set_xlabel("Time (s)")
    ax.set_yticks(np.arange(0, n_ch * offset, offset))
    if channel_ids is not None and len(channel_ids) == n_ch:
        # show every Nth label to avoid crowding
        nth = max(1, n_ch // 40)
        labels = [channel_ids[i] if (i % nth == 0) else "" for i in range(n_ch)]
        ax.set_yticklabels(labels)
    else:
        ax.set_yticklabels([str(i) for i in range(n_ch)])

    ax.set_title(title or out_path.stem)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-offset, n_ch * offset)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    if not UTAH_DIR.exists():
        raise SystemExit(f"[error] Utah traces folder not found: {UTAH_DIR}")

    # Accept both full-trace dumps and the random-segment npz created by extractor
    pattern1 = sorted(UTAH_DIR.rglob("*__utah_traces.npz"))
    pattern2 = sorted(UTAH_DIR.rglob("*__utah_random_ns6.npz"))
    files = sorted({*pattern1, *pattern2})
    if not files:
        print(f"[info] No utah trace npz files found under {UTAH_DIR}")
        return

    print(f"Found {len(files)} utah trace files; plotting...")

    for f in files:
        try:
            with np.load(f, allow_pickle=True) as d:
                # `utah_traces` is expected to be (n_channels, n_samples)
                traces = np.asarray(d["utah_traces"])
                fs = float(d.get("fs_hz", np.nan))
                ch_ids = d.get("channel_ids", None)
                if ch_ids is not None:
                    # handle arrays or pickled objects
                    try:
                        ch_ids = [str(x) for x in (ch_ids.tolist() if hasattr(ch_ids, "tolist") else list(ch_ids))]
                    except Exception:
                        ch_ids = None
        except Exception as e:
            print(f"[skip] failed to load {f}: {e}")
            continue

        out_png = FIG_DIR / f"{f.stem}__stacked.png"
        title = f.stem.replace("__utah_traces", "").replace("__utah_random_ns6", "")
        # default: plot first 10 seconds or entire file if shorter
        try:
            n_samples = traces.shape[1]
        except Exception:
            print(f"[error] unexpected traces shape in {f}; skipping")
            continue
        t_dur_s = min(10.0, n_samples / max(1.0, fs)) if not np.isnan(fs) else 10.0
        try:
            plot_stacked(traces, fs if not np.isnan(fs) else 30000.0, out_png, title=title,
                         channel_ids=ch_ids, t_start_s=0.0, t_dur_s=t_dur_s, max_points=3000)
            print(f"[saved] {out_png}")
        except Exception as e:
            print(f"[error] plotting {f}: {e}")


if __name__ == "__main__":
    main()
