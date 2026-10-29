"""
plot_ipca_artifact_correction.py
================================

Poster-quality multi-panel visualization showing neural data before and after
Incremental Principal Component Analysis (IPCA) stimulation artifact correction,
alongside precisely aligned action potential waveforms.

Figure Layout & Architecture:
-----------------------------
For N selected channels (e.g., M1 superior electrodes 190, 210, 225, or a single
channel for compact poster layouts), each channel is displayed in its own row:
  - Column 1 (Left):   Trace from -100 ms to +200 ms BEFORE IPCA correction (Raw HPF),
                       displaying the raw stimulation artifact bursts and train dynamics.
  - Column 2 (Center): Trace from -100 ms to +200 ms AFTER IPCA correction,
                       demonstrating effective artifact suppression and preserved neural spikes.
  - Column 3 (Right):  Spike waveforms from -1.0 ms to +1.5 ms, extracted across the full
                       recording scope (like in plot_ua_hpf_traces.py), centered precisely at
                       their negative peak, with individual waveforms in light gray and a dark
                       boldly outlined median action potential.

Aesthetic Standards:
--------------------
  - Curated, publication-grade color palette adhering to project standards (e.g. M1s crimson,
    M1i green, PMd orange, SMA blue).
  - High-resolution vector and raster export (300 DPI PNG and clean SVG).
  - Clean spines, elegant typography, shaded stimulation duration windows, and statistical badges.

Usage:
------
    # Run with default settings (Channels 190, 210, 225, Condition 17, Trial 5)
    python analysis_scripts/plot_ipca_artifact_correction.py

    # Single-channel compact view for poster space constraints:
    python analysis_scripts/plot_ipca_artifact_correction.py --channels 190

    # Custom condition, trial, and window:
    python analysis_scripts/plot_ipca_artifact_correction.py --condition 17 --channels 190 210 225 --trial 5 --win-start -100 --win-end 200
"""

from __future__ import annotations

import os
import re
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/batch execution
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.edgecolor"] = "#333333"
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["xtick.direction"] = "out"
matplotlib.rcParams["ytick.direction"] = "out"
matplotlib.rcParams["xtick.major.size"] = 4
matplotlib.rcParams["ytick.major.size"] = 4

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe

# Add repository root and analysis_scripts to sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se

import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *


# ─────────────────────────────────────────────────────────────────────────────
# USER SETTINGS (Defaults for quick interactive configuration)
# ─────────────────────────────────────────────────────────────────────────────
# Condition / BR index (condition number)
CONDITION: int = 12

# Selected channels / Electrode IDs to plot (1–256).
# Configured for single channel 225 (can also be a list: e.g. [190, 210, 225])
CHANNELS: List[int] = [225]

# Which trial index to visualize (0-based index among valid trials in condition)
TRIAL_TO_PLOT: int = 5

# Probe type ('UA' for Utah Array)
PROBE: str = "UA"

# Time window relative to event onset (t = 0.0 ms) for Left and Center panels
WINDOW_MS: Tuple[float, float] = (-100.0, 200.0)

# Waveform window parameters for Right panel (ms relative to negative peak)
WAVEFORM_PRE_MS: float = 1.0     # Time before negative peak (ms)
WAVEFORM_POST_MS: float = 1.5    # Time after negative peak (ms)
PEAK_SEARCH_RADIUS_MS: float = 0.35 # Search window around detected peak for true trough

# Only count spikes within legal time windows of true trials from extract_peri_stim
ONLY_LEGAL_PERISTIM_SPIKES: bool = True
LEGAL_WINDOW_PAD_MS: Tuple[float, float] = (-800.0, 600.0)

# Maximum number of individual waveforms to overlay in gray (avoids solid gray blob)
MAX_WAVEFORMS_OVERLAY: int = 350

# Y-axis limits (µV).
Y_LIM_TRACE: Optional[Tuple[float, float]] = (-75.0, 75.0)
Y_LIM_WAVEFORM: Optional[Tuple[float, float]] = (-75.0, 45.0)

# Synchronize Y-axis limits between Before and After IPCA columns
SHARE_Y_BEFORE_AFTER: bool = True

# Overlay detected spike points on the After IPCA trace
OVERLAY_SPIKES_ON_TRACE: bool = True

# Grid display on panels
SHOW_GRIDS: bool = False

# Publication plot quality (DPI)
OUTPUT_DPI: int = 300
SAVE_SVG: bool = True
# ─────────────────────────────────────────────────────────────────────────────


# Anatomical colors adhering to project standard
REGION_COLORS: Dict[str, str] = {
    "M1s": "#d62728",               # Crimson / Red (Primary Motor Superficial / Lower Limb)
    "M1 (lower limb)": "#d62728",
    "M1 Sup": "#d62728",
    "M1 superior": "#d62728",
    "M1i": "#2ca02c",               # Forest Green (Primary Motor Deep / Upper Limb)
    "M1 (upper limb)": "#2ca02c",
    "M1 Inf": "#2ca02c",
    "M1 inferior": "#2ca02c",
    "PMd": "#ff7f0e",               # Amber / Orange (Dorsal Premotor)
    "Premotor (dorsal)": "#ff7f0e",
    "Dorsal premotor": "#ff7f0e",
    "SMA": "#1f77b4",               # Cobalt Blue (Supplementary Motor Area)
}

REGION_DISPLAY_NAMES: Dict[str, str] = {
    "M1s": "M1 (lower limb)",
    "M1 Sup": "M1 (lower limb)",
    "M1 superior": "M1 superior",
    "M1i": "M1 (upper limb)",
    "M1 Inf": "M1 (upper limb)",
    "M1 inferior": "M1 inferior",
    "PMd": "Premotor (dorsal)",
    "Dorsal premotor": "PMd",
    "SMA": "SMA",
}


def get_region_color(region_name: str) -> str:
    """Return project-standard color for a cortical array region."""
    rn = str(region_name).lower()
    if "upper" in rn or "m1i" in rn or "inf" in rn:
        return "#2ca02c"
    if "lower" in rn or "m1s" in rn or "sup" in rn:
        return "#d62728"
    if "premotor" in rn or "pmd" in rn:
        return "#ff7f0e"
    if "sma" in rn:
        return "#1f77b4"
    return REGION_COLORS.get(region_name, "#2c3e50")


def get_region_display_name(region_name: str) -> str:
    """Return clean anatomical display title for an array region."""
    return REGION_DISPLAY_NAMES.get(region_name, region_name)


def find_aligned_file(aligned_dir: Path, br_idx: int) -> Path:
    """Locate the aligned .npz checkpoint for a given condition index."""
    pattern = f"*__BR_{br_idx:03d}.npz"
    search_dirs = [
        aligned_dir / "stim_reaches",
        aligned_dir / "control_reaches",
        aligned_dir / "continuous_stim",
        aligned_dir / "at_rest",
        aligned_dir,
    ]
    for d in search_dirs:
        if d.exists():
            candidates = sorted(d.glob(pattern))
            if candidates:
                return candidates[-1]
    raise FileNotFoundError(f"No aligned npz matching {pattern} in {aligned_dir}")


def find_pp_folder_for_condition(ckpt_dir: Path, condition: int, intan_filename: str = "") -> Optional[Path]:
    """Find preprocessed SpikeInterface folder matching a condition index."""
    if not ckpt_dir.exists():
        return None
    all_folders = sorted(ckpt_dir.glob("pp_*"))
    target = f"_{condition:03d}__NS6"
    for f in all_folders:
        if target in f.name:
            return f
    if intan_filename:
        for f in all_folders:
            if intan_filename in f.name:
                return f
    return None


def get_legal_trial_windows(
    session_loc: Path, 
    condition: int, 
    z_aligned: Any, 
    win_pad_ms: Tuple[float, float] = (-800.0, 600.0)
) -> List[Tuple[float, float]]:
    """
    Retrieve legal time windows [event_ms + win_pad_ms[0], event_ms + win_pad_ms[1]]
    for true trials that passed gating in extract_peri_stim.
    """
    peri_base = session_loc / "results" / "checkpoints" / "PeriStim"
    valid_events = []
    found_peri = False

    if peri_base.exists():
        patterns = [f"*BR_{condition:03d}*.npz", f"*BR_{condition}*.npz"]
        cands = []
        for pat in patterns:
            cands.extend(list(peri_base.rglob(pat)))

        seen = set()
        unique_cands = []
        for c in cands:
            if c not in seen:
                seen.add(c)
                unique_cands.append(c)

        for c in unique_cands:
            try:
                pz = np.load(c, allow_pickle=True)
                if "win" in pz.files:
                    w = pz["win"]
                    if hasattr(w, "__len__") and len(w) == 2:
                        win_pad_ms = (float(w[0]), float(w[1]))
                if "event_ms" in pz.files:
                    evs = np.asarray(pz["event_ms"], float).ravel()
                    if evs.size > 0:
                        valid_events.extend(evs.tolist())
                        found_peri = True
                elif "stim_ms" in pz.files:
                    evs = np.asarray(pz["stim_ms"], float).ravel()
                    if evs.size > 0:
                        valid_events.extend(evs.tolist())
                        found_peri = True
            except Exception:
                continue

    if found_peri and valid_events:
        valid_events = sorted(list(set(valid_events)))
        print(f"[info] Filtered to {len(valid_events)} legal trials from PeriStim for BR_{condition:03d} (win={win_pad_ms} ms)")
    else:
        # Fallback to aligned.npz stim_ms / event_ms
        if "stim_ms" in z_aligned.files:
            valid_events = np.asarray(z_aligned["stim_ms"], float).ravel().tolist()
        elif "event_ms" in z_aligned.files:
            valid_events = np.asarray(z_aligned["event_ms"], float).ravel().tolist()
        print(f"[info] PeriStim checkpoint not found; using {len(valid_events)} trials from Aligned NPZ")

    legal_intervals = []
    for ev in valid_events:
        t0 = ev + win_pad_ms[0]
        t1 = ev + win_pad_ms[1]
        legal_intervals.append((t0, t1))

    return legal_intervals


def extract_trace_window(
    rec, 
    channel_id: Any, 
    center_ms: float, 
    win_ms: Tuple[float, float], 
    fs: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract one trace window (in µV) for a channel centered around a stimulation event.
    
    Returns
    -------
    t : time axis relative to event center in ms
    y : signal trace in µV
    """
    s0 = int(round(center_ms / 1000.0 * fs))
    i0 = int(s0 + round(win_ms[0] / 1000.0 * fs))
    i1 = int(s0 + round(win_ms[1] / 1000.0 * fs))
    n_frames = rec.get_num_frames()

    if i0 < 0 or i1 > n_frames or i1 <= i0:
        return None, None

    try:
        y = rec.get_traces(
            start_frame=i0, 
            end_frame=i1, 
            channel_ids=[channel_id], 
            return_in_uV=True
        ).squeeze()
        t = (np.arange(i0, i1) - s0) / fs * 1000.0
        return t, y
    except Exception as e:
        print(f"[warn] Failed to extract trace for channel {channel_id}: {e}")
        return None, None


def extract_aligned_spike_waveforms(
    rec,
    channel_id: Any,
    spike_times_ms: np.ndarray,
    fs: float,
    pre_ms: float = 1.0,
    post_ms: float = 1.5,
    search_radius_ms: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract spike waveforms centered exactly at their negative peak (trough).
    
    Parameters
    ----------
    rec : SpikeInterface recording
    channel_id : channel ID to extract
    spike_times_ms : 1D array of spike times in recording-local ms
    fs : sampling frequency in Hz
    pre_ms : duration before peak in ms (e.g. 1.0)
    post_ms : duration after peak in ms (e.g. 1.5)
    search_radius_ms : window around spike time to search for true minimum
    
    Returns
    -------
    t_wf : time axis in ms from -pre_ms to +post_ms
    waveforms : (n_spikes, n_samples) array in µV
    """
    if spike_times_ms.size == 0:
        n_samples = int(round((pre_ms + post_ms) / 1000.0 * fs))
        return np.linspace(-pre_ms, post_ms, n_samples), np.empty((0, n_samples), dtype=np.float32)

    pre_samp = int(round(pre_ms / 1000.0 * fs))
    post_samp = int(round(post_ms / 1000.0 * fs))
    n_samples = pre_samp + post_samp
    search_samp = max(2, int(round(search_radius_ms / 1000.0 * fs)))
    n_frames = rec.get_num_frames()

    waveforms = []
    for s_ms in spike_times_ms:
        center_frame = int(round(s_ms / 1000.0 * fs))
        s_lo = center_frame - search_samp
        s_hi = center_frame + search_samp + 1

        if s_lo < 0 or s_hi > n_frames:
            continue

        try:
            # Search for local negative peak (minimum voltage trough)
            search_chunk = rec.get_traces(
                start_frame=s_lo,
                end_frame=s_hi,
                channel_ids=[channel_id],
                return_in_uV=True
            ).squeeze()

            if search_chunk.size == 0:
                continue

            trough_idx = np.argmin(search_chunk)
            refined_center = s_lo + trough_idx

            w_lo = refined_center - pre_samp
            w_hi = refined_center + post_samp

            if w_lo < 0 or w_hi > n_frames:
                continue

            wf = rec.get_traces(
                start_frame=w_lo,
                end_frame=w_hi,
                channel_ids=[channel_id],
                return_in_uV=True
            ).squeeze()

            if wf.shape[0] == n_samples:
                waveforms.append(wf)
        except Exception:
            continue

    t_wf = (np.arange(n_samples) - pre_samp) / fs * 1000.0
    if len(waveforms) == 0:
        return t_wf, np.empty((0, n_samples), dtype=np.float32)

    return t_wf, np.asarray(waveforms, dtype=np.float32)


def resolve_channel_metadata(
    rec, 
    target_ch_elecs: np.ndarray, 
    target_ch_regions: np.ndarray, 
    region_names: Sequence[str],
    monkey: str,
    repo_root: Path
) -> Dict[int, Dict[str, Any]]:
    """
    Build unified mapping between Recording Index (ridx), Electrode ID (eid),
    SpikeInterface Channel ID (cid), and Anatomical Region.
    """
    ch_ids = rec.get_channel_ids()
    ridx_to_meta: Dict[int, Dict[str, Any]] = {}

    # Load mapping CSV if available for monkey
    mapping_csv = repo_root / "config" / f"electrode_port_mapping_{monkey}.csv"
    mapping_df = pd.read_csv(mapping_csv) if mapping_csv.exists() else None

    for ridx, cid in enumerate(ch_ids):
        # 1. Try to parse Electrode ID directly from channel name (e.g. UAe190_NSP... -> 190)
        m = re.search(r'UAe(\d+)', str(cid))
        eid = int(m.group(1)) if m else -1

        # 2. Fallback to target_ch_elecs array
        if eid < 0 and ridx < len(target_ch_elecs):
            eid = int(target_ch_elecs[ridx])

        # 3. Alignment index (oid) for peak_ms_dict lookup
        oid = ridx
        if eid > 0 and target_ch_elecs.size > 0:
            matches = np.where(target_ch_elecs == eid)[0]
            if matches.size > 0:
                oid = int(matches[0])

        # 4. Resolve anatomical region
        reg_name = "Unknown"
        if ridx < len(target_ch_regions):
            reg_idx = int(target_ch_regions[ridx])
            if 0 <= reg_idx < len(region_names):
                reg_name = region_names[reg_idx]

        # Region fallback from electrode number if needed
        if reg_name in ("Unknown", "") and eid > 0:
            if monkey.lower() == "nike":
                if eid <= 64:   reg_name = "SMA"
                elif eid <= 128: reg_name = "Dorsal premotor"
                elif eid <= 192: reg_name = "M1 inferior"
                elif eid <= 256: reg_name = "M1 superior"
            elif monkey.lower() == "ada":
                if eid <= 64:   reg_name = "M1 superior"
                elif eid <= 128: reg_name = "M1 inferior"
                elif eid <= 192: reg_name = "SMA"
                elif eid <= 256: reg_name = "Dorsal premotor"

        reg_color = get_region_color(reg_name)
        reg_disp = get_region_display_name(reg_name)

        ridx_to_meta[ridx] = {
            "ridx": ridx,
            "eid": eid,
            "oid": oid,
            "cid": cid,
            "region": reg_name,
            "region_display": reg_disp,
            "color": reg_color,
        }

    return ridx_to_meta


def load_raw_ns6_recording(
    session_loc: Path,
    condition: int,
    br_session_name: str,
    repo_root: Path,
    params
) -> Optional[Any]:
    """
    Load raw Blackrock NS6 recording and apply electrode mapping and highpass filter.
    """
    br_root = session_loc / "Blackrock"
    if not br_root.exists():
        print(f"[warn] Blackrock directory not found at {br_root}")
        return None

    # Search for Blackrock session folder or .ns6 file
    candidates = []
    if br_session_name:
        candidates.extend(sorted(br_root.glob(f"*{br_session_name}*")))
    candidates.extend(sorted(br_root.glob(f"*_{condition:03d}*")))

    target_path = None
    for c in candidates:
        if c.is_dir() or c.suffix.lower() in (".ns6", ".nev"):
            target_path = c
            break

    if not target_path or not target_path.exists():
        print(f"[warn] Could not locate raw Blackrock files for BR_{condition:03d} in {br_root}")
        return None

    print(f"[info] Loading raw Blackrock recording: {target_path.name}")
    try:
        rec_raw = se.read_blackrock(target_path, stream_name="nsx6", all_annotations=True)
        ua_map_path = rcp.ua_excel_path(repo_root, params.probes)
        ua_map = rcp.load_UA_mapping_from_excel(ua_map_path)
        
        metadata_csv = Path(METADATA_CSV) if "METADATA_CSV" in globals() and METADATA_CSV else (session_loc / "Metadata" / f"{params.session}_metadata.csv")
        (rec_proc, _, ua_elec, _, ua_region, ua_region_names, ua_port) = rcp.apply_ua_mapping_with_regions(
            rec_raw, ua_map, condition, metadata_csv, monkey=str(params.monkey)
        )
        # Apply 300 Hz highpass filter (standard action potential band)
        rec_hp = spre.highpass_filter(rec_proc, freq_min=float(getattr(params, "highpass_hz", 300.0)))
        return rec_hp
    except Exception as e:
        print(f"[warn] Failed to load/filter raw Blackrock recording: {e}")
        return None


def plot_ipca_correction_figure(
    rec_before,
    rec_after,
    channels_to_plot: List[int],
    ridx_to_meta: Dict[int, Dict[str, Any]],
    peak_ms_dict: Dict[Any, Any],
    target_ch_elecs: np.ndarray,
    event_time_local: float,
    stim_dur: float,
    fs: float,
    session_title: str,
    condition: int,
    trial_idx: int,
    out_path: Path,
    window_ms: Tuple[float, float] = (-100.0, 200.0),
    wf_pre_ms: float = 1.0,
    wf_post_ms: float = 1.5,
    y_lim_trace: Optional[Tuple[float, float]] = (-75.0, 75.0),
    y_lim_waveform: Optional[Tuple[float, float]] = (-75.0, 45.0),
    share_y_before_after: bool = True,
    overlay_spikes_on_trace: bool = True,
    max_waveforms_overlay: int = 350,
    legal_intervals: Optional[List[Tuple[float, float]]] = None,
    output_dpi: int = 300,
    save_svg: bool = True,
):
    """
    Generate the 3-column publication-quality comparison figure.
    """
    n_rows = len(channels_to_plot)
    if n_rows == 0:
        print("[error] No channels specified to plot.")
        return

    # Proportional sizing: width accommodates 3 columns, height adapts to row count
    fig_w = 16.5
    fig_h = max(3.8, 3.2 * n_rows + 0.8) if n_rows > 1 else 3.8
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=output_dpi)

    # GridSpec with custom width ratios: wide for time traces, compact for waveforms
    gs = gridspec.GridSpec(
        n_rows, 3,
        figure=fig,
        width_ratios=[1.25, 1.25, 0.95],
        wspace=0.22,
        hspace=0.38,
        top=0.84 if n_rows > 1 else 0.80,
        bottom=0.14 if n_rows > 1 else 0.20,
        left=0.08,
        right=0.96
    )

    axes_before = []
    axes_after = []
    axes_wv = []

    # Iterate through channels (each channel gets its own row)
    for r, target_eid in enumerate(channels_to_plot):
        # Resolve recording metadata for this electrode
        matched_meta = None
        for ridx, meta in ridx_to_meta.items():
            if meta["eid"] == target_eid:
                matched_meta = meta
                break

        if not matched_meta:
            # Fallback search by electrode in target_ch_elecs
            if target_ch_elecs.size > 0:
                m = np.where(target_ch_elecs == target_eid)[0]
                if m.size > 0:
                    ridx_cand = int(m[0])
                    matched_meta = ridx_to_meta.get(ridx_cand, {
                        "ridx": ridx_cand, "eid": target_eid, "oid": ridx_cand,
                        "cid": f"UAe{target_eid}", "region": "M1s",
                        "region_display": "M1 superior", "color": "#d62728"
                    })

        if not matched_meta:
            print(f"[warn] Electrode {target_eid} could not be resolved in recording metadata.")
            continue

        ridx = matched_meta["ridx"]
        cid = matched_meta["cid"]
        oid = matched_meta["oid"]
        eid = matched_meta["eid"]
        reg_color = matched_meta["color"]

        # Create subplots for this row
        ax_b = fig.add_subplot(gs[r, 0], sharex=axes_before[0] if r > 0 else None)
        ax_a = fig.add_subplot(gs[r, 1], sharex=ax_b, sharey=ax_b if share_y_before_after else None)
        ax_w = fig.add_subplot(gs[r, 2], sharex=axes_wv[0] if r > 0 else None)

        axes_before.append(ax_b)
        axes_after.append(ax_a)
        axes_wv.append(ax_w)

        # ─────────────────────────────────────────────────────────────
        # 1. Left Column: Before IPCA Correction (Raw HPF)
        # ─────────────────────────────────────────────────────────────
        t_b, y_b = (None, None)
        if rec_before is not None:
            t_b, y_b = extract_trace_window(rec_before, cid, event_time_local, window_ms, fs)

        if t_b is not None and y_b is not None:
            ax_b.plot(t_b, y_b, color="#34495e", lw=0.75, alpha=0.92, label="Raw HPF")
        else:
            ax_b.text(0.5, 0.5, "Raw recording\nunavailable", ha="center", va="center",
                      transform=ax_b.transAxes, fontsize=10, color="gray", fontstyle="italic")

        # ─────────────────────────────────────────────────────────────
        # 2. Center Column: After IPCA Correction
        # ─────────────────────────────────────────────────────────────
        t_a, y_a = (None, None)
        if rec_after is not None:
            t_a, y_a = extract_trace_window(rec_after, cid, event_time_local, window_ms, fs)
        elif t_b is not None and y_b is not None:
            # On-the-fly IPCA correction fallback if preprocessed checkpoint folder was not generated yet
            try:
                from RCP_analysis.python.functions.artifact_correction import IPCA_Artifact_Correction, Template
                # Stim pulse interval ~2.5 ms (400 Hz)
                pulse_interval_samp = int(round(2.5 / 1000.0 * fs))
                mw_pre = int(round(0.4 / 1000.0 * fs))
                mw_post = int(round(0.4 / 1000.0 * fs))

                s0_idx = int(round(-window_ms[0] / 1000.0 * fs))
                s1_idx = int(round((-window_ms[0] + stim_dur) / 1000.0 * fs))
                s1_idx = min(len(y_b), max(s0_idx + 10, s1_idx))

                pulse_centers = []
                curr = s0_idx
                while curr < s1_idx:
                    w_lo = max(0, curr - mw_pre)
                    w_hi = min(len(y_b), curr + mw_post)
                    if w_hi > w_lo:
                        local_p = w_lo + np.argmax(np.abs(y_b[w_lo:w_hi]))
                        pulse_centers.append(local_p)
                    curr += pulse_interval_samp

                pulses = []
                valid_centers = []
                for pc in pulse_centers:
                    p0 = pc - mw_pre
                    p1 = pc + mw_post
                    if p0 >= 0 and p1 <= len(y_b):
                        pulses.append(y_b[p0:p1])
                        valid_centers.append((p0, p1))

                if len(pulses) >= 3:
                    pulses_arr = np.array(pulses, dtype=np.float32)
                    corrector = IPCA_Artifact_Correction(rank=min(3, len(pulses)))
                    tpl = Template()
                    _, tpl = corrector.ipca_template_per_channel(pulses_arr, tpl)
                    pulses_corr = corrector.apply_template(pulses_arr, tpl)

                    y_a = y_b.copy()
                    for idx_p, (p0, p1) in enumerate(valid_centers):
                        y_a[p0:p1] = pulses_corr[idx_p]
                    t_a = t_b.copy()
                else:
                    t_a, y_a = t_b, y_b
            except Exception as e:
                print(f"[warn] On-the-fly IPCA fallback failed: {e}")
                t_a, y_a = (None, None)

        if t_a is not None and y_a is not None:
            ax_a.plot(t_a, y_a, color=reg_color, lw=0.85, alpha=0.95, label="IPCA Cleaned")
        else:
            ax_a.text(0.5, 0.5, "Cleaned trace\nunavailable", ha="center", va="center",
                      transform=ax_a.transAxes, fontsize=10, color="gray", fontstyle="italic")

        # Stimulation Window Shading & Reference Lines on Before & After Traces
        for ax_t, is_after in [(ax_b, False), (ax_a, True)]:
            if stim_dur > 0:
                ax_t.axvspan(0.0, stim_dur, color="#e74c3c", alpha=0.14, zorder=0, label="Stim window")
                ax_t.axvline(0.0, color="#c0392b", linestyle="--", lw=1.1, alpha=0.85)
                # Dotted line at stim_dur is omitted as requested
            else:
                ax_t.axvline(0.0, color="#2980b9", linestyle="--", lw=1.1, alpha=0.85, label="Stim onset")

            ax_t.axhline(0.0, color="#7f8c8d", linestyle=":", lw=0.6, alpha=0.45)
            ax_t.set_xlim(*window_ms)
            if y_lim_trace is not None:
                ax_t.set_ylim(*y_lim_trace)

            # Clean spines and no grids
            ax_t.spines["top"].set_visible(False)
            ax_t.spines["right"].set_visible(False)
            ax_t.grid(False)

        # Overlay detected spikes on After IPCA trace
        ch_peaks_abs = np.asarray(peak_ms_dict.get(oid, []), float).ravel()
        if ch_peaks_abs.size == 0 and eid > 0:
            ch_peaks_abs = np.asarray(peak_ms_dict.get(eid, []), float).ravel()

        if overlay_spikes_on_trace and ch_peaks_abs.size > 0 and t_a is not None and y_a is not None:
            pk_rel = ch_peaks_abs - event_time_local
            pk_rel_win = pk_rel[(pk_rel >= window_ms[0]) & (pk_rel <= window_ms[1])]
            if pk_rel_win.size > 0:
                p_idx = np.searchsorted(t_a, pk_rel_win)
                valid_p = (p_idx >= 0) & (p_idx < len(t_a))
                p_idx = p_idx[valid_p]
                pk_clean = pk_rel_win[valid_p]
                ax_a.plot(pk_clean, y_a[p_idx], "o", color="#c0392b", markersize=3.2,
                          alpha=0.9, zorder=5, label="Detected Spikes")

        # Row Badge / Y-axis Label strictly on Left Column: "Electrode 225 \n (µV)"
        ax_b.set_ylabel(f"Electrode {eid}\n(µV)", fontsize=11, fontweight="bold",
                        color="#2c3e50", labelpad=10)
        ax_a.set_ylabel("")
        ax_w.set_ylabel("")

        # ─────────────────────────────────────────────────────────────
        # 3. Right Column: Spike Waveform Plot (-1.0 ms to +1.5 ms)
        # ─────────────────────────────────────────────────────────────
        rec_for_wf = rec_after if rec_after is not None else rec_before
        all_waveforms = np.empty((0, int((wf_pre_ms + wf_post_ms) / 1000.0 * fs)))
        t_wf = np.linspace(-wf_pre_ms, wf_post_ms, max(1, all_waveforms.shape[1]))

        # Filter spikes strictly to legal time windows of true trials from extract_peri_stim
        filtered_peaks = ch_peaks_abs.copy()
        if legal_intervals and len(legal_intervals) > 0 and filtered_peaks.size > 0:
            in_legal = np.zeros(len(filtered_peaks), dtype=bool)
            for (w0, w1) in legal_intervals:
                in_legal |= (filtered_peaks >= w0) & (filtered_peaks <= w1)
            filtered_peaks = filtered_peaks[in_legal]
            print(f"[info] Elec {eid}: filtered from {len(ch_peaks_abs)} to {len(filtered_peaks)} spikes within legal trial windows")

        if rec_for_wf is not None and filtered_peaks.size > 0:
            rec_dur_ms = rec_for_wf.get_num_frames() / fs * 1000.0
            valid_pks = filtered_peaks[(filtered_peaks >= wf_pre_ms) & (filtered_peaks <= rec_dur_ms - wf_post_ms)]
            if valid_pks.size > 0:
                t_wf, all_waveforms = extract_aligned_spike_waveforms(
                    rec=rec_for_wf,
                    channel_id=cid,
                    spike_times_ms=valid_pks,
                    fs=fs,
                    pre_ms=wf_pre_ms,
                    post_ms=wf_post_ms,
                    search_radius_ms=PEAK_SEARCH_RADIUS_MS,
                )

        n_spikes = all_waveforms.shape[0]

        if n_spikes > 0:
            # Subsample waveforms for individual gray overlay to keep plot crisp and uncluttered
            if n_spikes > max_waveforms_overlay:
                sub_idx = np.random.choice(n_spikes, max_waveforms_overlay, replace=False)
                wf_plot = all_waveforms[sub_idx]
            else:
                wf_plot = all_waveforms

            # Individual waveforms in light gray
            for wf in wf_plot:
                ax_w.plot(t_wf, wf, color="#95a5a6", alpha=0.18, lw=0.55, zorder=2)

            # Compute robust median spike waveform across all collected legal spikes
            median_wf = np.median(all_waveforms, axis=0)
            p25 = np.percentile(all_waveforms, 25, axis=0)
            p75 = np.percentile(all_waveforms, 75, axis=0)

            # Shaded IQR consistency band
            ax_w.fill_between(t_wf, p25, p75, color=reg_color, alpha=0.14, zorder=4)

            # Dark outline for median spike waveform using matplotlib patheffects
            ax_w.plot(
                t_wf,
                median_wf,
                color=reg_color,
                lw=2.5,
                zorder=6,
                label="Median",
                path_effects=[
                    pe.Stroke(linewidth=4.2, foreground="#111111"),
                    pe.Normal()
                ]
            )

            # Horizontal baseline at 0 µV (vertical dashed line at 0 is omitted as requested)
            ax_w.axhline(0.0, color="#718096", linestyle=":", lw=0.8, alpha=0.5, zorder=1)
            # N and P-T metrics badge is omitted as requested
        else:
            ax_w.text(0.5, 0.5, "No spikes detected", ha="center", va="center",
                      transform=ax_w.transAxes, fontsize=10, color="gray", fontstyle="italic")

        ax_w.set_xlim(-wf_pre_ms, wf_post_ms)
        if y_lim_waveform is not None:
            ax_w.set_ylim(*y_lim_waveform)
        ax_w.spines["top"].set_visible(False)
        ax_w.spines["right"].set_visible(False)
        ax_w.grid(False)
        ax_w.set_ylabel("")

        # Bottom row displays X-axis labels strictly as "Time (ms)"
        if r == n_rows - 1:
            ax_b.set_xlabel("Time (ms)", fontsize=10.5, fontweight="medium")
            ax_a.set_xlabel("Time (ms)", fontsize=10.5, fontweight="medium")
            ax_w.set_xlabel("Time (ms)", fontsize=10.5, fontweight="medium")
        else:
            plt.setp(ax_b.get_xticklabels(), visible=False)
            plt.setp(ax_a.get_xticklabels(), visible=False)
            plt.setp(ax_w.get_xticklabels(), visible=False)

    # Column Super-Headers on Row 0
    if axes_before:
        axes_before[0].set_title(
            "Pre-Artifact Removal",
            fontsize=12.5, fontweight="bold", pad=12, color="#2c3e50"
        )
    if axes_after:
        axes_after[0].set_title(
            "Post-Artifact Removal",
            fontsize=12.5, fontweight="bold", pad=12, color="#2c3e50"
        )
    if axes_wv:
        axes_wv[0].set_title(
            "Spike Waveforms",
            fontsize=12.5, fontweight="bold", pad=12, color="#2c3e50"
        )

    # Master Figure Title
    if n_rows > 1:
        ch_str = ", ".join(f"Elec {c}" for c in channels_to_plot)
        fig.suptitle(
            f"{session_title}  •  Condition {condition:03d} (Stim)  •  Trial {trial_idx + 1}  •  {ch_str}\n"
            f"Incremental PCA (IPCA) Artifact Correction & Spike Waveforms",
            fontsize=13, fontweight="bold", y=0.98
        )
    else:
        fig.suptitle(
            f"{session_title}  •  Condition {condition:03d}  •  Trial {trial_idx + 1}",
            fontsize=11.5, fontweight="medium", color="#555555", y=0.97
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=output_dpi, bbox_inches="tight")
    if save_svg:
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")
    if save_svg:
        print(f"[saved] {out_path.with_suffix('.svg')}")


def process_single_condition(
    cond: int,
    channels: List[int],
    trial_i: int,
    win_ms: Tuple[float, float],
    wf_pre: float,
    wf_post: float,
    y_trace: Optional[Tuple[float, float]],
    y_wf: Optional[Tuple[float, float]],
    dpi: int,
    save_svg: bool,
    repo_root: Path,
    params: Any,
    session_loc: Path,
    session_name: str,
    monkey: str,
    aligned_ckpt: Path,
    ckpt_dir: Path,
    fig_dir: Path,
):
    """Process and render artifact correction figure for a single condition."""
    print("\n" + "=" * 75)
    print(f"IPCA ARTIFACT CORRECTION & SPIKE WAVEFORM PLOTTER")
    print(f"Condition: {cond} | Channels: {channels} | Trial: {trial_i} | Win: {win_ms} ms")
    print("=" * 75)

    # ── Load Aligned File ──
    try:
        aligned_path = find_aligned_file(aligned_ckpt, cond)
    except FileNotFoundError as e:
        print(f"[warn] Aligned file not found for BR_{cond:03d}: {e}. Skipping.")
        return

    print(f"[info] Aligned file: {aligned_path.name}")
    z = np.load(aligned_path, allow_pickle=True)

    align_meta = {}
    if "align_meta" in z.files:
        raw_m = z["align_meta"].item() if z["align_meta"].ndim == 0 else z["align_meta"]
        align_meta = raw_m if isinstance(raw_m, dict) else json.loads(raw_m)
    intan_filename = align_meta.get("intan_filename", "")

    # Event stimulation times
    stim_ms = z.get("stim_ms", np.array([], dtype=float))
    stim_ms = np.asarray(stim_ms, float).ravel()
    if stim_ms.size == 0:
        print(f"[info] Condition {cond} has no stimulation events (e.g. control reaches); skipping IPCA artifact plot.")
        return

    # Blackrock SpikeInterface recording starts at local time 0.0
    rec_start_ms = 0.0
    event_local_ms = stim_ms - rec_start_ms

    # Extract Spike Peaks
    pk_obj = z.get("ua_peak_ms_dedup", z.get("ua_peak_ms", None))
    peak_ms_dict = pk_obj.reshape(-1)[0] if (isinstance(pk_obj, np.ndarray) and pk_obj.dtype == object) else pk_obj
    if peak_ms_dict is None:
        peak_ms_dict = {}

    target_ch_regions = z.get("ua_region", [])
    region_names = z.get("ua_region_names", ["SMA", "Dorsal premotor", "M1 inferior", "M1 superior"])
    target_ch_elecs = z.get("ua_elec", [])
    if isinstance(target_ch_elecs, np.ndarray) and target_ch_elecs.ndim == 0:
        target_ch_elecs = target_ch_elecs.item()
    target_ch_elecs = np.asarray(target_ch_elecs, int).ravel()

    # ── Retrieve Legal Trial Windows from PeriStim Checkpoints ──
    legal_intervals = None
    if ONLY_LEGAL_PERISTIM_SPIKES:
        legal_intervals = get_legal_trial_windows(session_loc, cond, z, LEGAL_WINDOW_PAD_MS)

    # ── Extract Hardware Stim Duration ──
    from RCP_analysis.python.functions.config_loading import METADATA_CSV, NPRW_AUX_DATA
    stim_npz_path, _ = rcp.stim_npz_path_from_br_idx(cond, METADATA_CSV, NPRW_AUX_DATA)
    fs_nprw = 30_000.0
    stim_dur = 0.0
    if stim_npz_path and stim_npz_path.exists():
        stim = rcp.load_stim_detection(stim_npz_path)
        block_bounds = stim.get("block_bounds_samples", np.empty((0, 2), dtype=np.int64))
        if len(block_bounds) > 0:
            if isinstance(block_bounds, list):
                block_bounds = np.array(block_bounds, dtype=np.int64)
            nprw_samps = block_bounds[:, 0].astype(np.int64)
            nprw_ends = block_bounds[:, 1].astype(np.int64)
            stim_dur = float((nprw_ends - nprw_samps).max() * 1000.0 / fs_nprw)
    print(f"[info] Stimulation pulse train duration: {stim_dur:.1f} ms")

    # ── Load Recordings ──
    # 1. Cleaned Recording (After IPCA)
    pp_folder = find_pp_folder_for_condition(ckpt_dir, cond, intan_filename)
    rec_after = None
    if pp_folder and pp_folder.exists():
        print(f"[info] Loading preprocessed (IPCA cleaned) recording: {pp_folder.name}")
        rec_after = si.load(str(pp_folder))
    else:
        print(f"[warn] No preprocessed pp_* folder found in {ckpt_dir} for condition {cond}.")

    # 2. Raw Recording (Before IPCA)
    ua_meta = z["ua_meta"].item() if "ua_meta" in z.files else {}
    br_session_name = ua_meta.get("session", "")
    rec_before = load_raw_ns6_recording(session_loc, cond, br_session_name, repo_root, params)

    if rec_after is None and rec_before is None:
        print("[error] Neither preprocessed (after) nor raw (before) recording could be loaded.")
        return

    # Establish sampling rate and duration
    reference_rec = rec_after if rec_after is not None else rec_before
    fs = float(reference_rec.get_sampling_frequency())
    rec_dur_ms = reference_rec.get_num_frames() / fs * 1000.0

    # ── Channel Metadata Mapping ──
    ridx_to_meta = resolve_channel_metadata(
        rec=reference_rec,
        target_ch_elecs=target_ch_elecs,
        target_ch_regions=target_ch_regions,
        region_names=region_names,
        monkey=monkey,
        repo_root=repo_root
    )

    # ── Validate & Filter Event Window ──
    valid_events = []
    for s in event_local_ms:
        if (s + win_ms[0]) >= 0 and (s + win_ms[1]) <= rec_dur_ms:
            valid_events.append(s)
    valid_events = np.asarray(valid_events, float)

    if valid_events.size == 0:
        print(f"[error] No stimulation events fit within recording boundaries for window {win_ms} ms.")
        return

    if trial_i >= len(valid_events):
        print(f"[warn] Requested trial index {trial_i} exceeds valid events count ({len(valid_events)}). Defaulting to trial 0.")
        trial_i = 0

    chosen_event_time = valid_events[trial_i]
    print(f"[info] Selected Trial {trial_i + 1} (onset at {chosen_event_time:.1f} ms in recording)")

    # ── Build Output Filename ──
    ch_tag = "_".join(f"ch{c}" for c in channels)
    out_filename = (
        f"{session_name}__cond{cond:03d}__trial{trial_i + 1}__{ch_tag}__"
        f"win{int(win_ms[0])}to{int(win_ms[1])}ms__IPCA_Artifact_Correction.png"
    )
    out_path = fig_dir / out_filename

    # ── Generate Poster Plot ──
    print("[info] Rendering publication comparison figure...")
    plot_ipca_correction_figure(
        rec_before=rec_before,
        rec_after=rec_after,
        channels_to_plot=channels,
        ridx_to_meta=ridx_to_meta,
        peak_ms_dict=peak_ms_dict,
        target_ch_elecs=target_ch_elecs,
        event_time_local=chosen_event_time,
        stim_dur=stim_dur,
        fs=fs,
        session_title=session_name,
        condition=cond,
        trial_idx=trial_i,
        out_path=out_path,
        window_ms=win_ms,
        wf_pre_ms=wf_pre,
        wf_post_ms=wf_post,
        y_lim_trace=y_trace,
        y_lim_waveform=y_wf,
        share_y_before_after=SHARE_Y_BEFORE_AFTER,
        overlay_spikes_on_trace=OVERLAY_SPIKES_ON_TRACE,
        max_waveforms_overlay=MAX_WAVEFORMS_OVERLAY,
        legal_intervals=legal_intervals,
        output_dpi=dpi,
        save_svg=save_svg,
    )
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot poster-quality before & after IPCA artifact correction and spike waveforms."
    )
    parser.add_argument("--condition", "-c", type=int, default=CONDITION,
                        help=f"BR condition index (default: {CONDITION})")
    parser.add_argument("--channels", "--ch", type=int, nargs="+", default=CHANNELS,
                        help=f"Electrode IDs to plot (default: {CHANNELS})")
    parser.add_argument("--trial", "-t", type=int, default=TRIAL_TO_PLOT,
                        help=f"0-based trial index to plot (default: {TRIAL_TO_PLOT})")
    parser.add_argument("--win-start", type=float, default=WINDOW_MS[0],
                        help=f"Window start relative to stim in ms (default: {WINDOW_MS[0]})")
    parser.add_argument("--win-end", type=float, default=WINDOW_MS[1],
                        help=f"Window end relative to stim in ms (default: {WINDOW_MS[1]})")
    parser.add_argument("--wf-pre", type=float, default=WAVEFORM_PRE_MS,
                        help=f"Waveform pre-peak duration in ms (default: {WAVEFORM_PRE_MS})")
    parser.add_argument("--wf-post", type=float, default=WAVEFORM_POST_MS,
                        help=f"Waveform post-peak duration in ms (default: {WAVEFORM_POST_MS})")
    parser.add_argument("--y-trace-min", type=float, default=Y_LIM_TRACE[0] if Y_LIM_TRACE else None)
    parser.add_argument("--y-trace-max", type=float, default=Y_LIM_TRACE[1] if Y_LIM_TRACE else None)
    parser.add_argument("--y-wf-min", type=float, default=Y_LIM_WAVEFORM[0] if Y_LIM_WAVEFORM else None)
    parser.add_argument("--y-wf-max", type=float, default=Y_LIM_WAVEFORM[1] if Y_LIM_WAVEFORM else None)
    parser.add_argument("--dpi", type=int, default=OUTPUT_DPI,
                        help=f"Output DPI (default: {OUTPUT_DPI})")
    parser.add_argument("--no-svg", action="store_true",
                        help="Disable SVG export")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Optional custom output directory")

    args = parser.parse_args()

    cond = args.condition
    channels = args.channels
    trial_i = args.trial
    win_ms = (args.win_start, args.win_end)
    wf_pre = args.wf_pre
    wf_post = args.wf_post
    dpi = args.dpi
    save_svg = not args.no_svg

    y_trace = (args.y_trace_min, args.y_trace_max) if args.y_trace_min is not None and args.y_trace_max is not None else Y_LIM_TRACE
    y_wf = (args.y_wf_min, args.y_wf_max) if args.y_wf_min is not None and args.y_wf_max is not None else Y_LIM_WAVEFORM

    # ── Resolve Experiment Paths & Session Settings ──
    repo_root = _REPO_ROOT
    params = rcp.load_experiment_params(repo_root / "config" / "params.yaml", repo_root=repo_root)
    session_loc = (Path(params.data_root) / Path(params.location)).resolve()
    session_name = getattr(params, "session", "UNKNOWN")
    monkey = str(getattr(params, "monkey", "Nike"))

    out_base = session_loc / "results"
    aligned_ckpt = out_base / "checkpoints" / "Aligned"
    ckpt_dir = out_base / "checkpoints" / PROBE

    fig_dir = Path(args.out_dir) if args.out_dir else (out_base / "figures" / "ipca_artifact_correction")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Determine condition(s) to process
    conditions_to_run = [cond]
    env_proc = os.environ.get("RCP_PROCESS_ONLY")
    has_cli_cond = any(a.startswith(("-c", "--condition")) for a in sys.argv[1:])
    if not has_cli_cond and env_proc:
        try:
            parsed = json.loads(env_proc)
            if isinstance(parsed, list) and len(parsed) > 0:
                conditions_to_run = [int(c) for c in parsed]
        except Exception:
            pass
    elif not has_cli_cond:
        proc_only = getattr(params, "preprocessing", {}).get("process_only") if hasattr(params, "preprocessing") else None
        if proc_only and isinstance(proc_only, list) and len(proc_only) > 0:
            conditions_to_run = [int(c) for c in proc_only]

    for c in conditions_to_run:
        process_single_condition(
            cond=c,
            channels=channels,
            trial_i=trial_i,
            win_ms=win_ms,
            wf_pre=wf_pre,
            wf_post=wf_post,
            y_trace=y_trace,
            y_wf=y_wf,
            dpi=dpi,
            save_svg=save_svg,
            repo_root=repo_root,
            params=params,
            session_loc=session_loc,
            session_name=session_name,
            monkey=monkey,
            aligned_ckpt=aligned_ckpt,
            ckpt_dir=ckpt_dir,
            fig_dir=fig_dir,
        )


if __name__ == "__main__":
    main()
