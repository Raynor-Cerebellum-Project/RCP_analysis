"""
Plot raw high-pass-filtered (HPF) traces for selected channels around
stimulation events for a given condition (BR index).

Usage:
    python scripts/plot_hpf_traces.py

Edit the USER SETTINGS section below to choose condition, channels, and window.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import spikeinterface as si
import RCP_analysis as rcp

matplotlib.use("Agg")  # non-interactive backend
matplotlib.rcParams["svg.fonttype"] = "none"

# ──────────────────────────────────────────────────
# USER SETTINGS
# ──────────────────────────────────────────────────
CONDITION   = 10             # BR index  (= condition number)
PROBE       = "UA"        # "NPRW" or "UA"
CHANNELS    = list(range(1, 128))   # 0-indexed channel rows to plot (7–15)
WINDOW_MS   = (-50.0, 200.0)      # (start, end) relative to stim onset
Y_LIM       = (-50, 50)         # µV range; set to None for auto-scale
TRIAL_INDEX = 6#'all'                     # which trial (stim event) to plot; 'all' for overlay
# ──────────────────────────────────────────────────

REPO_ROOT   = Path().resolve().parents[1]
PARAMS      = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
SESSION     = getattr(PARAMS, "session", "UNKNOWN")
OUT_BASE    = SESSION_LOC / "results"

ALIGNED_CKPT = OUT_BASE / "checkpoints" / "Aligned"
CKPT_DIR     = OUT_BASE / "checkpoints" / PROBE

FIG_DIR = OUT_BASE / "figures" / "hpf_traces"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def find_aligned_file(aligned_dir: Path, br_idx: int) -> Path:
    """Find the aligned .npz for a given BR index across all subdirectories."""
    pat = f"*__BR_{br_idx:03d}.npz"
    # Search in stim_reaches, control_reaches, at_rest, and top-level
    search_dirs = [
        aligned_dir / "stim_reaches",
        aligned_dir / "control_reaches",
        aligned_dir / "at_rest",
        aligned_dir,
    ]
    for d in search_dirs:
        if d.exists():
            cands = sorted(d.glob(pat))
            if cands:
                return cands[-1]
    raise FileNotFoundError(f"No aligned npz matching {pat} in {aligned_dir}")


def find_pp_folder_for_intan(ckpt_dir: Path, intan_filename: str, probe: str) -> Path:
    """
    Find the preprocessed SpikeInterface folder that matches a specific
    Intan session filename.
    """
    # Try to match by Intan filename substring
    all_folders = sorted(ckpt_dir.glob("pp_*"))
    for f in all_folders:
        if intan_filename in f.name:
            return f
    # Fallback: try without the timestamp prefix (session name only)
    # e.g. intan_filename = "NRR_RW012_260116_145123" but folder might
    # have a different timestamp portion
    parts = intan_filename.split("_")
    if len(parts) >= 2:
        session_prefix = "_".join(parts[:2])  # e.g. "NRR_RW012"
        for f in all_folders:
            if session_prefix in f.name:
                # If multiple, prefer later ones
                pass
        # Return last match for session prefix
        matches = [f for f in all_folders if session_prefix in f.name]
        if matches:
            return matches[-1]
    if all_folders:
        print(f"[warn] Could not match Intan session '{intan_filename}'; "
              f"using last available folder: {all_folders[-1].name}")
        return all_folders[-1]
    raise FileNotFoundError(f"No preprocessed folders in {ckpt_dir}")


def extract_traces(rec, ch_row: int, center_ms: float, win_ms: tuple, fs: float):
    """Extract one trace window (in µV) for a channel around a stim event."""
    ch_id = rec.get_channel_ids()[ch_row]
    s0    = int(round(center_ms / 1000.0 * fs))
    i0    = int(s0 + round(win_ms[0] / 1000.0 * fs))
    i1    = int(s0 + round(win_ms[1] / 1000.0 * fs))
    if i0 < 0 or i1 > rec.get_num_frames() or i1 <= i0:
        return None, None
    y = rec.get_traces(start_frame=i0, end_frame=i1,
                       channel_ids=[ch_id], return_in_uV=True).squeeze()
    t = (np.arange(i0, i1) - s0) / fs * 1000.0
    return t, y


def main():
    # ── Load aligned file for this condition ──
    aligned_path = find_aligned_file(ALIGNED_CKPT, CONDITION)
    z = np.load(aligned_path, allow_pickle=True)
    print(f"[info] Aligned file: {aligned_path.name}")

    # ── Parse metadata ──
    meta = json.loads(z["align_meta"].item()) if "align_meta" in z.files else {}
    intan_filename = meta.get("intan_filename", "")
    shift_ms = float(meta.get("shift_ms", 0.0))

    # stim_ms are in the ALIGNED time base (Intan ms - shift_ms).
    stim_ms = z.get("stim_ms", np.array([], dtype=float))
    stim_ms = np.asarray(stim_ms, float).ravel()
    if stim_ms.size == 0:
        print("[error] No stimulation events found in aligned file.")
        return

    # ── Get the NPRW recording's start time in the aligned frame ──
    # The SI recording starts at sample 0, which corresponds to
    # nprw_meta["rec_start_ms_aligned"] in the aligned time base.
    # To convert aligned stim_ms to the recording's local time:
    #   local_ms = stim_ms - rec_start_ms_aligned
    if PROBE == "NPRW":
        probe_meta = z["nprw_meta"].item() if "nprw_meta" in z.files else {}
        rec_start_ms = float(probe_meta.get("rec_start_ms_aligned", 0.0))
    else:  # UA
        probe_meta = z["ua_meta"].item() if "ua_meta" in z.files else {}
        rec_start_ms = float(probe_meta.get("rec_start_ms", 0.0))

    # Convert stim times from aligned time base to recording-local time base
    stim_local_ms = stim_ms - rec_start_ms
    print(f"[info] Aligned stim_ms range: {stim_ms.min():.1f} – {stim_ms.max():.1f} ms")
    print(f"[info] Recording start (aligned): {rec_start_ms:.1f} ms")
    print(f"[info] Local stim_ms range: {stim_local_ms.min():.1f} – {stim_local_ms.max():.1f} ms")

    # ── Load SpikeInterface recording for the correct Intan session ──
    pp_folder = find_pp_folder_for_intan(CKPT_DIR, intan_filename, PROBE)
    rec = si.load(str(pp_folder))
    fs  = float(rec.get_sampling_frequency())
    n_ch = rec.get_num_channels()
    rec_dur_ms = rec.get_num_frames() / fs * 1000.0
    print(f"[info] Recording: {pp_folder.name}  |  fs={fs:.0f} Hz  |  {n_ch} ch  |  {rec_dur_ms:.1f} ms")
    print(f"[info] Stim events: {stim_ms.size}  |  condition {CONDITION}")

    # Validate channel indices
    channels = [c for c in CHANNELS if 0 <= c < n_ch]
    if not channels:
        print(f"[error] No valid channels in {CHANNELS} (recording has {n_ch}).")
        return

    # ── Filter valid stim events that fit within the recording ──
    valid = []
    for s in stim_local_ms:
        if (s + WINDOW_MS[0]) >= 0 and (s + WINDOW_MS[1]) <= rec_dur_ms:
            valid.append(s)
    valid = np.array(valid)
    if valid.size == 0:
        print("[error] No stim events fit within the recording for this window.")
        return
    print(f"[info] Valid stim events for window: {valid.size} / {stim_ms.size}")

    # ── Build figure ──
    n_rows = len(channels)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.0 * n_rows),
                             sharex=True, sharey=True)
    if n_rows == 1:
        axes = [axes]

    plot_all = isinstance(TRIAL_INDEX, str) and TRIAL_INDEX.lower() == "all"

    for ax, ch in zip(axes, channels):
        if plot_all:
            for trial_i, center in enumerate(valid):
                t, y = extract_traces(rec, ch, center, WINDOW_MS, fs)
                if t is not None:
                    ax.plot(t, y, lw=0.5, alpha=0.4, color="steelblue")
        else:
            idx = TRIAL_INDEX if TRIAL_INDEX < len(valid) else 0
            t, y = extract_traces(rec, ch, valid[idx], WINDOW_MS, fs)
            if t is not None:
                ax.plot(t, y, lw=0.7, color="steelblue")

        ax.axvline(0.0, ls="--", lw=0.8, color="red", label="Stim onset")
        ax.set_ylabel(f"Ch {ch}\n(µV)", fontsize=8)
        if Y_LIM is not None:
            ax.set_ylim(*Y_LIM)

    axes[-1].set_xlabel("Time relative to stimulation (ms)")
    axes[0].set_xlim(*WINDOW_MS)

    trial_str = "all trials" if plot_all else f"trial {TRIAL_INDEX}"
    fig.suptitle(
        f"{SESSION}  •  Condition {CONDITION} (BR_{CONDITION:03d})  •  "
        f"{PROBE} channels {channels[0]}–{channels[-1]}  •  {trial_str}",
        fontsize=11, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = (
        FIG_DIR
        / f"{SESSION}__cond{CONDITION:03d}__{PROBE}__ch{channels[0]}-{channels[-1]}"
          f"__win{int(WINDOW_MS[0])}-{int(WINDOW_MS[1])}ms.png"
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
