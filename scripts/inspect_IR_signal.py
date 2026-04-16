"""
Quick diagnostic: plot the raw IR digital signal for selected Intan sessions
to understand why detect_IR_crossings returns 100K+ events for continuous stim.

Usage:
    python scripts/inspect_IR_signal.py                # all continuous stim sessions
    python scripts/inspect_IR_signal.py --all           # all sessions
    python scripts/inspect_IR_signal.py --idx 2 3 11    # specific Intan indices
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

# ---------- CONFIG ----------
NPRW_CFG = PARAMS.probes.get("NPRW")
IR_STREAM = NPRW_CFG.get("ir_stream")
# How many seconds max to load (to avoid slow reads on long sessions)
MAX_LOAD_SEC = 120.0

# ---------- HELPERS ----------

def get_continuous_stim_indices() -> list[int]:
    """Return Intan indices marked as continuous stim in metadata."""
    idx2trigger = rcp.get_metadata_mapping(METADATA_CSV, "Intan_File", "Movement_Trigger")
    idx2notes = rcp.get_metadata_mapping(METADATA_CSV, "Intan_File", "Notes")
    continuous = []
    for idx, trigger in idx2trigger.items():
        note_norm = str(idx2notes.get(idx, "")).strip().lower()
        if trigger.strip().lower() == "sig gen" and note_norm != "rest":
            continuous.append(idx)
    return sorted(continuous)

def load_ir_signal(sess_path: Path, max_sec: float = MAX_LOAD_SEC):
    """Load IR stream, handle multi-channel, and optionally truncate."""
    rec_ir = se.read_split_intan_files(
        sess_path,
        mode="concatenate",
        stream_name=IR_STREAM,
        use_names_as_ids=True,
    )
    rec_ir = spre.unsigned_to_signed(rec_ir)
    
    fs_ir = float(rec_ir.sampling_frequency)
    n_total = rec_ir.get_num_samples()
    n_load = min(n_total, int(max_sec * fs_ir))
    
    traces = rec_ir.get_traces(start_frame=0, end_frame=n_load)
    # traces shape: (n_samples, n_channels)
    
    ch_ids = list(rec_ir.get_channel_ids())
    print(f"    IR stream channels: {ch_ids}, shape: {traces.shape}, fs: {fs_ir} Hz")
    
    return traces, fs_ir, n_total, ch_ids

def plot_ir_channel(ax, sig, fs, ch_name, intan_idx, trigger, notes, n_total):
    """Plot one channel of the IR signal with crossing detection."""
    sig = sig.astype(float)
    ir_idx = rcp.detect_IR_crossings(sig, fs, refractory_sec=0.0005)
    ir_idx_long = rcp.detect_IR_crossings(sig, fs, refractory_sec=0.200)
    
    t_sec = np.arange(sig.size) / fs
    total_dur = n_total / fs
    loaded_dur = sig.size / fs
    
    # Downsample for plotting if too many points
    if sig.size > 200_000:
        step = sig.size // 200_000
        plot_t = t_sec[::step]
        plot_sig = sig[::step]
    else:
        plot_t = t_sec
        plot_sig = sig
    
    ax.plot(plot_t, plot_sig, linewidth=0.3, alpha=0.7, color='steelblue', label='IR signal')
    
    # Threshold line
    lo, hi = float(np.min(sig)), float(np.max(sig))
    thr = 0.5 * (lo + hi)
    ax.axhline(thr, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label=f'thr={thr:.0f}')
    
    # Mark crossings (200ms refr) — these should be actual reaches
    if ir_idx_long.size > 0:
        valid = ir_idx_long[ir_idx_long < sig.size]
        ax.plot(valid / fs, sig[valid], 'g^', markersize=8, alpha=0.8,
                label=f'200ms refr: {ir_idx_long.size}')
    
    # Mark crossings (0.5ms refr) — if manageable
    if 0 < ir_idx.size <= 200:
        valid = ir_idx[ir_idx < sig.size]
        ax.plot(valid / fs, sig[valid], 'rv', markersize=4, alpha=0.5,
                label=f'0.5ms refr: {ir_idx.size}')
    
    ax.set_title(
        f"Intan {intan_idx:03d} ch={ch_name} | trigger={trigger} | notes={notes}\n"
        f"loaded={loaded_dur:.1f}s / total={total_dur:.1f}s | "
        f"crossings: 0.5ms={ir_idx.size:,}  200ms={ir_idx_long.size}",
        fontsize=9
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("IR (digital)")
    ax.legend(loc='upper right', fontsize=7)
    
    # Print ISI stats
    if ir_idx.size > 1:
        isi_ms = np.diff(ir_idx / fs * 1000.0)
        print(f"    ch={ch_name}: crossings(0.5ms)={ir_idx.size:,}, crossings(200ms)={ir_idx_long.size}")
        print(f"    ISI stats (ms): min={isi_ms.min():.3f}, median={np.median(isi_ms):.3f}, max={isi_ms.max():.3f}")
        if ir_idx.size > 5:
            print(f"    First 5 ISIs (ms): {isi_ms[:5].tolist()}")
    else:
        print(f"    ch={ch_name}: crossings(0.5ms)={ir_idx.size}, crossings(200ms)={ir_idx_long.size}")

# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser(description="Inspect IR signal for selected sessions")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--idx", nargs="+", type=int, help="Specific Intan indices to inspect")
    group.add_argument("--all", action="store_true", help="Inspect all sessions")
    group.add_argument("--continuous", action="store_true", default=True, 
                       help="Only continuous stim sessions (default)")
    parser.add_argument("--max-sec", type=float, default=MAX_LOAD_SEC,
                       help=f"Max seconds to load per session (default: {MAX_LOAD_SEC})")
    args = parser.parse_args()
    
    sess_folders = rcp.list_intan_sessions(INTAN_ROOT)
    sess_names = [s.name for s in sess_folders]
    sess2idx, idx2sess = rcp.build_session_index_map(sess_names)
    
    idx2trigger = rcp.get_metadata_mapping(METADATA_CSV, "Intan_File", "Movement_Trigger")
    idx2notes = rcp.get_metadata_mapping(METADATA_CSV, "Intan_File", "Notes")
    
    # Determine which indices to process
    if args.idx:
        intan_idxs = args.idx
    elif args.all:
        intan_idxs = sorted(idx2sess.keys())
    else:
        # Default: continuous stim only
        intan_idxs = get_continuous_stim_indices()
        if not intan_idxs:
            print("[warn] No continuous stim sessions found in metadata. Use --idx or --all.")
            return
        print(f"[info] Found {len(intan_idxs)} continuous stim session(s): {intan_idxs}")
    
    # Also add one normal session for comparison if doing continuous-only
    if not args.idx and not args.all and len(intan_idxs) < len(idx2sess):
        # Find first IR-triggered session as reference
        for idx in sorted(idx2sess.keys()):
            if idx not in intan_idxs and idx2trigger.get(idx, "").strip().lower() == "ir":
                intan_idxs.append(idx)
                print(f"[info] Adding Intan {idx:03d} (IR-triggered) for comparison")
                break
    
    # Process
    plot_rows = []
    for intan_idx in intan_idxs:
        sess_name = idx2sess.get(intan_idx)
        if sess_name is None:
            print(f"[skip] Intan {intan_idx:03d}: no session found")
            continue
        
        trigger = idx2trigger.get(intan_idx, "?")
        notes = idx2notes.get(intan_idx, "?")
        sess_path = INTAN_ROOT / sess_name
        
        if not sess_path.exists():
            print(f"[skip] Intan {intan_idx:03d}: path not found: {sess_path}")
            continue
        
        print(f"\n[load] Intan {intan_idx:03d} ({sess_name}) | trigger={trigger} | notes={notes}")
        
        try:
            traces, fs, n_total, ch_ids = load_ir_signal(sess_path, max_sec=args.max_sec)
            # One row per channel
            for ch_i, ch_name in enumerate(ch_ids):
                sig = traces[:, ch_i] if traces.ndim == 2 else traces
                plot_rows.append((sig, fs, ch_name, intan_idx, trigger, notes, n_total))
        except Exception as e:
            print(f"  [error] {e}")
            plot_rows.append((None, None, "?", intan_idx, trigger, notes, 0))
    
    if not plot_rows:
        print("[done] Nothing to plot.")
        return
    
    # Plot
    fig, axes = plt.subplots(len(plot_rows), 1,
                             figsize=(18, 3.5 * len(plot_rows)),
                             squeeze=False)
    
    for row_i, (sig, fs, ch_name, intan_idx, trigger, notes, n_total) in enumerate(plot_rows):
        ax = axes[row_i, 0]
        if sig is None:
            ax.set_title(f"Intan {intan_idx:03d}: failed to load")
            continue
        plot_ir_channel(ax, sig, fs, ch_name, intan_idx, trigger, notes, n_total)
    
    plt.tight_layout()
    
    out_path = OUT_BASE / "figures" / "debug_IR_signal.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\n[saved] {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
